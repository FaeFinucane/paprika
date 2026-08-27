"""Construction of the concrete spiking network and its runtime services."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from continual_agent.agent.drives import ArrayDrive, BackgroundDrive
from continual_agent.agent.plugins import (
    HomeostasisPlugin,
    MetricsPlugin,
    NetworkContext,
    NetworkPlugin,
    PlasticityPlugin,
)
from continual_agent.agent.population_homeostasis import HomeostasisConfig, PopulationHomeostasis
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
    homeostasis: HomeostasisConfig = field(default_factory=HomeostasisConfig)

    def __post_init__(self) -> None:
        integer_fields = (
            "input_features",
            "hidden_neurons",
            "neurons_per_token",
            "neurons_per_action",
            "neurons_per_affect",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.output_tokens, tuple):
            raise ValueError("output_tokens must be a tuple")
        for name in ("action_names", "affect_names"):
            if not isinstance(getattr(self, name), tuple):
                raise ValueError(f"{name} must be a tuple")
        if not isinstance(self.session_policy, SessionPolicy):
            raise ValueError("session_policy must be a SessionPolicy")
        if self.weight_initialization is not None and not isinstance(
            self.weight_initialization, WeightInitializationConfig
        ):
            raise ValueError("weight_initialization must be a WeightInitializationConfig or None")
        if self.input_features <= BOUNDARY_CHANNEL_COUNT:
            raise ValueError("input_features must leave room for boundary channels")
        if not self.output_tokens:
            raise ValueError("output_tokens must not be empty")
        if any(not isinstance(token, str) or not token for token in self.output_tokens):
            raise ValueError("output_tokens must contain non-empty strings")
        if "<EOS>" not in self.output_tokens:
            raise ValueError("output_tokens must contain <EOS>")
        for name, values in (
            ("action_names", self.action_names),
            ("affect_names", self.affect_names),
        ):
            if len(set(values)) != len(values) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(f"{name} must contain unique non-empty strings")
        if len(set(self.output_tokens)) != len(self.output_tokens):
            raise ValueError("output_tokens must be unique")
        if not isinstance(self.seed, (int, np.integer)) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        numeric_fields = (
            "connection_probability",
            "learning_rate",
            "background_rate",
            "background_current",
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
        if self.learning_rate < 0:
            raise ValueError("learning_rate must be non-negative")
        if not isinstance(self.homeostasis, HomeostasisConfig):
            raise ValueError("homeostasis must be a HomeostasisConfig")

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
        explicit_pairs = np.concatenate(
            (
                np.stack(
                    np.meshgrid(
                        np.arange(input_features),
                        np.arange(input_features, affect_start),
                        indexing="ij",
                    ),
                    axis=-1,
                ).reshape(-1, 2),
                np.stack(
                    np.meshgrid(
                        np.arange(input_features),
                        np.arange(affect_start, action_start),
                        indexing="ij",
                    ),
                    axis=-1,
                ).reshape(-1, 2),
                np.stack(
                    np.meshgrid(
                        np.arange(input_features),
                        np.arange(action_start, char_start),
                        indexing="ij",
                    ),
                    axis=-1,
                ).reshape(-1, 2),
                np.stack(
                    np.meshgrid(
                        np.arange(input_features), np.arange(char_start, total), indexing="ij"
                    ),
                    axis=-1,
                ).reshape(-1, 2),
                np.stack(
                    np.meshgrid(
                        np.arange(input_features, affect_start),
                        np.arange(char_start, total),
                        indexing="ij",
                    ),
                    axis=-1,
                ).reshape(-1, 2),
                np.stack(
                    np.meshgrid(
                        np.arange(affect_start, action_start),
                        np.arange(action_start, char_start),
                        indexing="ij",
                    ),
                    axis=-1,
                ).reshape(-1, 2),
            )
        )
        explicit = set(map(tuple, explicit_pairs.tolist()))
        valid = np.array(
            [
                (int(source), int(target)) not in explicit and target < char_start
                for source, target in zip(synapses.source, synapses.target)
            ],
            dtype=bool,
        )
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

        def projection(
            name: str,
            sources: np.ndarray,
            targets: np.ndarray,
            mean: float,
            spread: float,
        ) -> np.ndarray:
            return initializer.projection(synapses, name, sources, targets, mean, spread)

        projection(
            "input_action",
            np.arange(input_features),
            np.arange(action_start, char_start),
            c.input_action_mean,
            c.input_action_spread,
        )
        affect_edges = projection(
            "input_affect",
            np.arange(input_features),
            np.arange(affect_start, action_start),
            c.input_affect_mean,
            c.input_affect_spread,
        )
        affect_indices = affect_edges.reshape(
            input_features, len(affect_names), neurons_per_affect
        ).transpose(1, 0, 2)
        affect_action_edges = synapses.weight.size
        projection(
            "affect_action",
            np.arange(affect_start, action_start),
            np.arange(action_start, char_start),
            c.affect_action_mean,
            c.affect_action_spread,
        )
        affect_action_indices = np.arange(affect_action_edges, synapses.weight.size)
        projection(
            "recurrent_output",
            np.arange(char_start, total),
            np.arange(char_start, total),
            c.recurrent_output_mean,
            c.recurrent_output_spread,
        )
        direct = projection(
            "direct_output",
            np.arange(input_features),
            np.arange(char_start, total),
            c.direct_output_mean,
            c.direct_output_spread,
        )
        hidden = projection(
            "hidden_output",
            np.arange(input_features, affect_start),
            np.arange(char_start, total),
            c.hidden_output_mean,
            c.hidden_output_spread,
        )
        network = NetworkCore(neurons, synapses)
        metrics = RuntimeMetrics(layout)
        homeostasis = PopulationHomeostasis(
            layout,
            self.homeostasis,
        )
        external = ArrayDrive(total)
        background = BackgroundDrive(
            total,
            self.background_rate,
            self.background_current,
            initializer.seed_for("background"),
        )
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
