"""Runtime-owned lifecycle, snapshots, isolation, and weight merging."""

from __future__ import annotations

from copy import copy, deepcopy
from typing import TYPE_CHECKING, Callable, TypeVar, cast

import numpy as np

from continual_agent.agent.drives import ArrayDrive
from continual_agent.agent.input_runner import InputRunner
from continual_agent.agent.plugins import (
    HomeostasisPlugin,
    MetricsPlugin,
    NetworkContext,
    NetworkPlugin,
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
            pending_current=np.asarray(state["pending_current"]),
            plasticity_pre_trace=self.runtime.plasticity.pre_trace,
            plasticity_post_trace=self.runtime.plasticity.post_trace,
            affect=(
                getattr(affect, "snapshot")()
                if affect is not None and hasattr(affect, "snapshot")
                else None
            ),
            working_memory=(
                getattr(working_memory, "snapshot")()
                if working_memory is not None and hasattr(working_memory, "snapshot")
                else None
            ),
            plasticity_eligibility=self.runtime.plasticity.eligibility,
        )

    def reproducibility_snapshot(self) -> dict[str, object]:
        return {
            "network": self.runtime.network.state_snapshot(),
            "weights": self.runtime.network.synapses.weight.copy(),
            "eligibility": self.runtime.plasticity.eligibility.copy(),
            "pre_trace": self.runtime.plasticity.pre_trace.copy(),
            "post_trace": self.runtime.plasticity.post_trace.copy(),
            "background_drive": self.runtime.background_drive.state_snapshot(),
            "response_session": deepcopy(self.runtime.response_session),
        }

    def restore_reproducibility_snapshot(self, snapshot: dict[str, object]) -> None:
        network = snapshot["network"]
        assert isinstance(network, dict)
        self.runtime.network.restore_state(network)
        self.runtime.network.synapses.weight[:] = np.asarray(snapshot["weights"])
        self.runtime.plasticity.eligibility[:] = np.asarray(snapshot["eligibility"])
        self.runtime.plasticity.pre_trace[:] = np.asarray(
            snapshot.get("pre_trace", np.zeros_like(self.runtime.plasticity.pre_trace))
        )
        self.runtime.plasticity.post_trace[:] = np.asarray(
            snapshot.get("post_trace", np.zeros_like(self.runtime.plasticity.post_trace))
        )
        drive = snapshot["background_drive"]
        assert isinstance(drive, dict)
        self.runtime.background_drive.restore_state(drive)
        response_session = snapshot.get("response_session")
        if response_session is not None:
            self.runtime.response_session = cast(ResponseSession, deepcopy(response_session))
        self.runtime.apply_ablation_mask()

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
        if policy.reset_membrane:
            self.runtime.network.neurons.voltage.fill(
                self.runtime.network.neurons.resting_potential
            )
        if policy.reset_refractory:
            self.runtime.network.neurons.refractory.fill(0)
        if policy.reset_pending_current or policy.reset_recurrent_activity:
            self.runtime.network.reset_synaptic_activity()
        if policy.reset_eligibility:
            self.runtime.plasticity.reset_traces()
        if policy.reset_background_rng:
            self.runtime.background_drive.reset_rng()
        if policy.reset_affect and affect is not None and hasattr(affect, "reset"):
            getattr(affect, "reset")()
        if (
            policy.reset_working_memory
            and working_memory is not None
            and hasattr(working_memory, "reset")
        ):
            getattr(working_memory, "reset")()
        if policy.reset_readout:
            self.runtime.output_readout.reset()
        if policy.reset_membrane or policy.reset_refractory or policy.reset_pending_current:
            self.runtime.network.tick = 0
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
            arbitration=deepcopy(source.output_readout.arbitration),
        )
        isolated.response_session = deepcopy(source.response_session)
        isolated.network.synapses.weight[:] = source.network.synapses.weight
        isolated.edge_enabled = source.edge_enabled.copy()
        isolated.background_drive = deepcopy(source.background_drive)
        isolated.external_drive = ArrayDrive(source.network.neurons.count)
        isolated.homeostasis = PopulationHomeostasis(
            source.layout,
            source.homeostasis.config,
        )
        isolated.homeostasis.drive = source.homeostasis.drive.copy()
        isolated.homeostasis._ticks = source.homeostasis._ticks
        isolated.homeostasis._spikes = source.homeostasis._spikes.copy()
        isolated.metrics = RuntimeMetrics(source.layout, source.metrics.saturation_rate)
        isolated.metrics.ticks = source.metrics.ticks
        isolated.metrics.output_events = source.metrics.output_events
        isolated.metrics.spikes = source.metrics.spikes.copy()
        isolated.metrics.voltage_sum = source.metrics.voltage_sum.copy()
        isolated.metrics.voltage_square_sum = source.metrics.voltage_square_sum.copy()
        isolated.metrics.voltage_minimum = source.metrics.voltage_minimum.copy()
        isolated.metrics.voltage_maximum = source.metrics.voltage_maximum.copy()
        copied_plugins: list[NetworkPlugin] = []
        for plugin in source.plugins:
            if isinstance(plugin, MetricsPlugin):
                copied_plugins.append(MetricsPlugin(isolated.metrics))
            elif isinstance(plugin, HomeostasisPlugin):
                copied_plugins.append(HomeostasisPlugin(isolated.homeostasis))
            elif isinstance(plugin, PlasticityPlugin):
                copied_plugins.append(PlasticityPlugin(isolated.plasticity))
            else:
                try:
                    copied_plugins.append(deepcopy(plugin))
                except Exception as exc:
                    raise TypeError(f"plugin {type(plugin).__name__} cannot be isolated") from exc
        isolated.plugins = copied_plugins
        isolated.metrics_plugin = next(
            (plugin for plugin in copied_plugins if isinstance(plugin, MetricsPlugin)),
            MetricsPlugin(isolated.metrics),
        )
        if not any(isinstance(plugin, MetricsPlugin) for plugin in copied_plugins):
            isolated.plugins.insert(0, isolated.metrics_plugin)
        isolated._context = NetworkContext(
            isolated.network.tick,
            isolated.network.neurons.voltage,
            isolated.network.neurons.voltage,
        )
        isolated.apply_ablation_mask()
        if extension_hook is not None:
            extension_hook(source, isolated)
        isolated.session = RuntimeSession(isolated)
        isolated.trainer = InputRunner(isolated)
        return isolated
