"""Runtime-owned lifecycle, snapshots, isolation, and weight merging."""

from __future__ import annotations

from copy import copy, deepcopy
from typing import TYPE_CHECKING, Callable, TypeVar

import numpy as np

from continual_agent.agent.drives import ArrayDrive, DriveAggregator, HomeostasisDrive
from continual_agent.agent.plugins import (
    HomeostasisPlugin,
    MetricsPlugin,
    NetworkContext,
    PlasticityPlugin,
)
from continual_agent.agent.population_homeostasis import PopulationHomeostasis
from continual_agent.agent.runtime_metrics import RuntimeMetrics
from continual_agent.agent.session import (
    ConflictPolicy,
    ResponseSession,
    SessionExecutionSnapshot,
    SessionSnapshot,
    SessionState,
)
from continual_agent.cognition.readout import EventReadout

if TYPE_CHECKING:
    from continual_agent.agent.spiking_runtime import SpikingRuntime

T = TypeVar("T")


class RuntimeSession:
    """Apply session policy to a runtime without owning network mechanics."""

    def __init__(self, runtime: SpikingRuntime) -> None:
        self.runtime = runtime

    @property
    def response(self) -> ResponseSession:
        return self.runtime.response_session

    def snapshot(
        self, affect: object | None = None, working_memory: object | None = None
    ) -> SessionSnapshot:
        state = self.runtime.network.state_snapshot()
        return SessionSnapshot(
            np.asarray(state["voltage"]),
            np.asarray(state["refractory"]),
            np.asarray(state["pending_current"]),
            affect=affect,
            working_memory=working_memory,
            plasticity_eligibility=self.runtime.plasticity.eligibility,
        )

    def start(self, *, affect: object | None = None, working_memory: object | None = None) -> None:
        session = self.response
        if session.state in (
            SessionState.COMPLETE,
            SessionState.EOS_RECEIVED,
            SessionState.EXHAUSTED,
        ):
            session.reset_for_next_response()
        elif session.state is SessionState.RESPONDING:
            session.abort()
        policy = session.policy
        if policy.reset_neuron_state and policy.reset_synaptic_activity:
            self.runtime.network.reset_state()
        elif policy.reset_neuron_state:
            self.runtime.network.reset_neuron_state()
        elif policy.reset_synaptic_activity:
            self.runtime.network.reset_synaptic_activity()
        if not policy.persist_plasticity_eligibility:
            self.runtime.plasticity.reset_traces()
        if policy.reset_readout:
            self.runtime.output_readout.reset()
        session.begin(self.snapshot(affect, working_memory))

    def finish(
        self, *, exhausted: bool, affect: object | None = None, working_memory: object | None = None
    ) -> None:
        snapshot = self.snapshot(affect, working_memory)
        if exhausted:
            self.response.exhaust(snapshot)
        else:
            self.response.receive_eos()
            self.response.complete(snapshot)

    def execute_isolated(
        self,
        operation: Callable[[SpikingRuntime], T],
        *,
        conflict_policy: ConflictPolicy,
        merge: bool,
        extension_hook: Callable[[SpikingRuntime, SpikingRuntime], None] | None = None,
    ) -> tuple[T, SessionExecutionSnapshot]:
        execution = SessionExecutionSnapshot(self.snapshot(), self.runtime.network.synapses.weight)
        isolated = self.clone(extension_hook)
        result = operation(isolated)
        changed = np.flatnonzero(
            ~np.isclose(isolated.network.synapses.weight, execution.weights, rtol=0.0, atol=0.0)
        )
        if changed.size:
            execution.record_weight_update(changed, isolated.network.synapses.weight[changed])
        if merge:
            execution.merge_into(self.runtime.network.synapses.weight, conflict_policy)
        return result, execution

    def clone(
        self, extension_hook: Callable[[SpikingRuntime, SpikingRuntime], None] | None = None
    ) -> SpikingRuntime:
        source = self.runtime
        isolated = copy(source)
        isolated.layout = source.layout
        isolated.network = source.network.copy()
        isolated.plasticity = deepcopy(source.plasticity)
        isolated.plasticity.synapses = isolated.network.synapses
        isolated.output_readout = EventReadout(
            source.layout,
            activation_threshold=source.output_readout.activation_threshold,
            release_threshold=source.output_readout.release_threshold,
            arbitration=deepcopy(source.output_readout.arbitration),
        )
        isolated.response_session = deepcopy(source.response_session)
        isolated.network.synapses.weight[:] = source.network.synapses.weight
        isolated.edge_enabled = source.edge_enabled.copy()
        isolated.background_drive = deepcopy(source.background_drive)
        isolated.external_drive = ArrayDrive(source.network.neurons.count)
        isolated.drives = DriveAggregator(source.network.neurons.count)
        isolated.drives.add(isolated.external_drive)
        isolated.drives.add(isolated.background_drive)
        isolated.homeostasis = PopulationHomeostasis(
            source.layout,
            enabled=source.homeostasis.enabled,
            target_rate=source.homeostasis.target_rate,
            strength=source.homeostasis.strength,
            update_interval=source.homeostasis.update_interval,
            max_current=source.homeostasis.max_current,
            populations=source.homeostasis.populations,
        )
        isolated.homeostasis.drive = source.homeostasis.drive.copy()
        isolated.homeostasis._ticks = source.homeostasis._ticks
        isolated.homeostasis._spikes = source.homeostasis._spikes.copy()
        isolated.drives.add(HomeostasisDrive(isolated.homeostasis))
        isolated.metrics = RuntimeMetrics(source.layout, source.metrics.saturation_rate)
        isolated.metrics.ticks = source.metrics.ticks
        isolated.metrics.output_events = source.metrics.output_events
        isolated.metrics.spikes = source.metrics.spikes.copy()
        isolated.metrics.voltage_sum = source.metrics.voltage_sum.copy()
        isolated.metrics.voltage_square_sum = source.metrics.voltage_square_sum.copy()
        isolated.metrics.voltage_minimum = source.metrics.voltage_minimum.copy()
        isolated.metrics.voltage_maximum = source.metrics.voltage_maximum.copy()
        isolated.plugins = [
            MetricsPlugin(isolated.metrics),
            HomeostasisPlugin(isolated.homeostasis),
            PlasticityPlugin(isolated.plasticity),
        ]
        isolated.metrics_plugin = isolated.plugins[0]
        isolated._context = NetworkContext(
            isolated.network.tick,
            isolated.network.neurons.voltage,
            isolated.network.neurons.voltage,
        )
        isolated.apply_ablation_mask()
        if extension_hook is not None:
            extension_hook(source, isolated)
        isolated.session = RuntimeSession(isolated)
        isolated.trainer = None
        return isolated
