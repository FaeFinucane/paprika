"""Construction of the concrete spiking network and its runtime services."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from continual_agent.agent.drives import (
    ArrayDrive,
    BackgroundDrive,
    DriveAggregator,
    HomeostasisDrive,
)
from continual_agent.agent.plugins import (
    HomeostasisPlugin,
    MetricsPlugin,
    NetworkContext,
    NetworkPlugin,
    PlasticityPlugin,
)
from continual_agent.agent.population_homeostasis import PopulationHomeostasis
from continual_agent.agent.runtime_metrics import RuntimeMetrics
from continual_agent.agent.session import BOUNDARY_CHANNEL_COUNT, ResponseSession, SessionPolicy
from continual_agent.cognition.readout import EventReadout
from continual_agent.plasticity.stdp import RewardModulatedSTDP
from continual_agent.simulation.core import NetworkCore
from continual_agent.simulation.neurons import LIFNeurons
from continual_agent.simulation.population_layout import Population, PopulationLayout
from continual_agent.simulation.synapses import SparseSynapses
from continual_agent.simulation.weight_initialization import (
    WeightInitializationConfig,
    WeightInitializer,
)


@dataclass
class NetworkBundle:
    input_features: int
    output_tokens: tuple[str, ...]
    neurons_per_token: int
    layout: PopulationLayout
    network: NetworkCore
    output_readout: EventReadout
    edge_enabled: np.ndarray
    plasticity: RewardModulatedSTDP
    response_session: ResponseSession
    metrics: RuntimeMetrics
    homeostasis: PopulationHomeostasis
    external_drive: ArrayDrive
    background_drive: BackgroundDrive
    drives: DriveAggregator
    plugins: list[NetworkPlugin]
    metrics_plugin: MetricsPlugin
    context: NetworkContext
    hidden_feature_groups: tuple[np.ndarray, ...]
    input_hidden_edge_indices: np.ndarray
    affect_edge_indices: np.ndarray
    affect_action_edge_indices: np.ndarray
    direct_input_output_edge_indices: np.ndarray
    hidden_output_edge_indices: np.ndarray
    token_input_edge_indices: np.ndarray
    hidden_recurrent_edge_indices: np.ndarray


@dataclass
class NetworkConfig:
    """Build layout, projections, state, drives, and scheduled plugins."""

    input_features: int
    hidden_neurons: int
    output_tokens: tuple[str, ...]
    neurons_per_token: int
    action_names: tuple[str, ...] = ()
    neurons_per_action: int = 1
    affect_names: tuple[str, ...] = ()
    neurons_per_affect: int = 1
    connection_probability: float = 0.08
    seed: int = 0
    weight_initialization: WeightInitializationConfig | None = None
    learning_rate: float = 0.08
    background_rate: float = 0.0
    background_current: float = 0.05
    session_policy: SessionPolicy = field(default_factory=SessionPolicy)
    homeostasis_enabled: bool = False
    homeostasis_target_rate: float = 0.1
    homeostasis_strength: float = 0.01
    homeostasis_update_interval: int = 100
    homeostasis_max_current: float = 0.25
    homeostasis_populations: tuple[Population, ...] = (Population.HIDDEN,)

    def __post_init__(self) -> None:
        integer_fields = (
            "input_features",
            "hidden_neurons",
            "neurons_per_token",
            "neurons_per_action",
            "neurons_per_affect",
            "homeostasis_update_interval",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.input_features <= BOUNDARY_CHANNEL_COUNT:
            raise ValueError("input_features must leave room for boundary channels")
        if not self.output_tokens:
            raise ValueError("output_tokens must not be empty")
        numeric_fields = (
            "connection_probability",
            "learning_rate",
            "background_rate",
            "background_current",
            "homeostasis_target_rate",
            "homeostasis_strength",
            "homeostasis_max_current",
        )
        for name in numeric_fields:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.connection_probability <= 1:
            raise ValueError("connection_probability must be in [0, 1]")
        if not 0 <= self.background_rate <= 1 or self.background_current < 0:
            raise ValueError("background rate must be in [0, 1] and current must be non-negative")
        if (
            self.learning_rate < 0
            or self.homeostasis_target_rate < 0
            or self.homeostasis_strength < 0
            or self.homeostasis_max_current < 0
        ):
            raise ValueError("rate, strength, and current values must be non-negative")

    def build(self) -> NetworkBundle:
        input_features = self.input_features
        hidden_neurons = self.hidden_neurons
        output_tokens = self.output_tokens
        neurons_per_token = self.neurons_per_token
        action_names = self.action_names
        neurons_per_action = self.neurons_per_action
        affect_names = self.affect_names
        neurons_per_affect = self.neurons_per_affect
        affect_start = input_features + hidden_neurons
        action_start = affect_start + len(affect_names) * neurons_per_affect
        char_start = action_start + len(action_names) * neurons_per_action
        total = char_start + len(output_tokens) * neurons_per_token
        layout = PopulationLayout(
            input_count=input_features,
            hidden_count=hidden_neurons,
            affect_count=action_start - affect_start,
            action_count=char_start - action_start,
            char_count=total - char_start,
            affect_subgroups={
                n: slice(
                    affect_start + i * neurons_per_affect,
                    affect_start + (i + 1) * neurons_per_affect,
                )
                for i, n in enumerate(affect_names)
            },
            action_subgroups={
                n: slice(
                    action_start + i * neurons_per_action,
                    action_start + (i + 1) * neurons_per_action,
                )
                for i, n in enumerate(action_names)
            },
            char_subgroups={
                n: slice(
                    char_start + i * neurons_per_token, char_start + (i + 1) * neurons_per_token
                )
                for i, n in enumerate(output_tokens)
            },
        )
        neurons = LIFNeurons(total, dt=1.0, tau_membrane=5.0, threshold=1.0, refractory_ticks=2)
        initializer = WeightInitializer(self.seed, self.weight_initialization)
        c = initializer.config
        synapses = initializer.random_synapses(total, self.connection_probability)
        # Character neurons are terminal readout populations.  Filter the
        # random graph before appending projections so this invariant covers
        # accidental base-random contacts as well as explicit pathways.
        valid = synapses.source < char_start
        synapses = SparseSynapses(
            synapses.source[valid],
            synapses.target[valid],
            synapses.weight[valid],
            total,
        )
        initial_edges = synapses.weight.size
        hidden_groups = tuple(
            np.asarray(g, dtype=np.int64)
            for g in np.array_split(np.arange(input_features, affect_start), input_features)
        )
        for source, group in enumerate(hidden_groups):
            # Use three unit-weight contacts instead of one out-of-range
            # weight. This preserves the intended bootstrap drive while every
            # individual synapse obeys the global bound.
            initializer.bootstrap_input_hidden(
                synapses, np.array([source]), group, f"input_hidden_bootstrap_{source}"
            )
        input_hidden = np.arange(initial_edges, synapses.weight.size)
        for index, seed in enumerate(c.population_projections):
            if seed.source is Population.OUTPUT_CHAR:
                raise ValueError("OUTPUT_CHAR cannot be a projection source")
            source_bounds = layout.slice(seed.source)
            target_bounds = layout.slice(seed.target)
            sources = np.arange(source_bounds.start, source_bounds.stop)
            targets = np.arange(target_bounds.start, target_bounds.stop)
            for contact in range(seed.contacts):
                initializer.population_projection(
                    synapses,
                    seed,
                    sources,
                    targets,
                    f"population_projection_{index}_{contact}",
                )
        initializer.projection(
            synapses,
            "input_action",
            np.arange(input_features),
            np.arange(action_start, char_start),
            c.input_action_mean,
            c.input_action_spread,
        )
        affect_edges = synapses.weight.size
        initializer.projection(
            synapses,
            "input_affect",
            np.arange(input_features),
            np.arange(affect_start, action_start),
            c.input_affect_mean,
            c.input_affect_spread,
        )
        affect_indices = (
            np.arange(affect_edges, synapses.weight.size)
            .reshape(input_features, len(affect_names), neurons_per_affect)
            .transpose(1, 0, 2)
        )
        affect_action_edges = synapses.weight.size
        initializer.projection(
            synapses,
            "affect_action",
            np.arange(affect_start, action_start),
            np.arange(action_start, char_start),
            c.affect_action_mean,
            c.affect_action_spread,
        )
        affect_action_indices = np.arange(affect_action_edges, synapses.weight.size)
        token_edges = synapses.weight.size
        initializer.projection(
            synapses,
            "recurrent_output",
            np.arange(char_start),
            np.arange(char_start, total),
            c.recurrent_output_mean,
            c.recurrent_output_spread,
        )
        projection = self._projection_indices
        direct = (
            token_edges + projection(input_features, len(output_tokens), neurons_per_token).ravel()
        )
        hidden = (
            token_edges
            + projection(input_features + hidden_neurons, len(output_tokens), neurons_per_token)[
                input_features : input_features + hidden_neurons
            ].ravel()
        )
        synapses.weight[direct] = np.clip(
            initializer.rng("direct_output").normal(
                c.direct_output_mean, c.direct_output_spread, direct.size
            ),
            -1.0,
            1.0,
        )
        synapses.weight[hidden] = np.clip(
            initializer.rng("hidden_output").normal(
                c.hidden_output_mean, c.hidden_output_spread, hidden.size
            ),
            -1.0,
            1.0,
        )
        network = NetworkCore(neurons, synapses)
        metrics = RuntimeMetrics(layout)
        homeostasis = PopulationHomeostasis(
            layout,
            enabled=self.homeostasis_enabled,
            target_rate=self.homeostasis_target_rate,
            strength=self.homeostasis_strength,
            update_interval=self.homeostasis_update_interval,
            max_current=self.homeostasis_max_current,
            populations=self.homeostasis_populations,
        )
        external = ArrayDrive(total)
        drives = DriveAggregator(total)
        drives.add(external)
        background = BackgroundDrive(
            total,
            self.background_rate,
            self.background_current,
            initializer.seed_for("background"),
        )
        drives.add(background)
        drives.add(HomeostasisDrive(homeostasis))
        plasticity = RewardModulatedSTDP(synapses, learning_rate=self.learning_rate)
        metrics_plugin = MetricsPlugin(metrics)
        plugins: list[NetworkPlugin] = [
            metrics_plugin,
            HomeostasisPlugin(homeostasis),
            PlasticityPlugin(plasticity),
        ]
        hidden_recurrent = np.flatnonzero(
            (synapses.source >= input_features)
            & (synapses.source < affect_start)
            & (synapses.target >= input_features)
            & (synapses.target < affect_start)
        )
        return NetworkBundle(
            input_features,
            output_tokens,
            neurons_per_token,
            layout,
            network,
            EventReadout(layout),
            np.ones(synapses.weight.size, dtype=bool),
            plasticity,
            ResponseSession(policy=self.session_policy),
            metrics,
            homeostasis,
            external,
            background,
            drives,
            plugins,
            metrics_plugin,
            NetworkContext(network.tick, neurons.voltage, neurons.voltage),
            hidden_groups,
            input_hidden,
            affect_indices,
            affect_action_indices,
            direct,
            hidden,
            direct.reshape(input_features, len(output_tokens), neurons_per_token),
            hidden_recurrent,
        )

    @staticmethod
    def _projection_indices(
        source_count: int, token_count: int, neurons_per_token: int
    ) -> np.ndarray:
        size = token_count * neurons_per_token
        return np.array(
            [
                np.arange(i * size, (i + 1) * size).reshape(token_count, neurons_per_token)
                for i in range(source_count)
            ]
        )
