"""Construction of the concrete spiking network and its runtime services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

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
    recurrent_event_edge_indices: np.ndarray


class NetworkFactory:
    """Build layout, projections, state, drives, and scheduled plugins."""

    def __init__(self, **options: object) -> None:
        self.options = options

    def build(self) -> NetworkBundle:
        o = self.options
        input_features = cast(int, o["input_features"])
        hidden_neurons = cast(int, o["hidden_neurons"])
        output_tokens = cast(tuple[str, ...], o["output_tokens"])
        neurons_per_token = cast(int, o["neurons_per_token"])
        action_names = cast(tuple[str, ...], o["action_names"])
        neurons_per_action = cast(int, o["neurons_per_action"])
        affect_names = cast(tuple[str, ...], o["affect_names"])
        neurons_per_affect = cast(int, o["neurons_per_affect"])
        if input_features <= BOUNDARY_CHANNEL_COUNT:
            raise ValueError("input_features must leave room for boundary channels")
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
        seed = cast(int, o["seed"])
        rng = np.random.default_rng(seed)
        synapses = SparseSynapses.random(
            total,
            cast(float, o["connection_probability"]),
            rng,
            excitatory_weight=0.18,
            inhibitory_weight=-0.12,
            inhibitory_fraction=0.15,
        )
        initial_edges = synapses.weight.size
        hidden_groups = tuple(
            np.asarray(g, dtype=np.int64)
            for g in np.array_split(np.arange(input_features, affect_start), input_features)
        )
        for source, group in enumerate(hidden_groups):
            self._projection(synapses, rng, np.array([source]), group, 3.0, 0.0)
        input_hidden = np.arange(initial_edges, synapses.weight.size)
        self._projection(
            synapses, rng, np.arange(input_features), np.arange(action_start, char_start), 0.9, 0.08
        )
        affect_edges = synapses.weight.size
        self._projection(
            synapses,
            rng,
            np.arange(input_features),
            np.arange(affect_start, action_start),
            0.0,
            0.02,
        )
        affect_indices = (
            np.arange(affect_edges, synapses.weight.size)
            .reshape(input_features, len(affect_names), neurons_per_affect)
            .transpose(1, 0, 2)
        )
        affect_action_edges = synapses.weight.size
        self._projection(
            synapses,
            rng,
            np.arange(affect_start, action_start),
            np.arange(action_start, char_start),
            0.0,
            0.02,
        )
        affect_action_indices = np.arange(affect_action_edges, synapses.weight.size)
        token_edges = synapses.weight.size
        self._projection(
            synapses, rng, np.arange(char_start), np.arange(char_start, total), 0.0, 0.025
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
        network = NetworkCore(neurons, synapses)
        metrics = RuntimeMetrics(layout)
        homeostasis = PopulationHomeostasis(
            layout,
            enabled=cast(bool, o["homeostasis_enabled"]),
            target_rate=cast(float, o["homeostasis_target_rate"]),
            strength=cast(float, o["homeostasis_strength"]),
            update_interval=cast(int, o["homeostasis_update_interval"]),
            max_current=cast(float, o["homeostasis_max_current"]),
            populations=cast(tuple[Population, ...], o["homeostasis_populations"]),
        )
        external = ArrayDrive(total)
        drives = DriveAggregator(total)
        drives.add(external)
        background = BackgroundDrive(
            total, cast(float, o["background_rate"]), cast(float, o["background_current"]), seed + 1
        )
        drives.add(background)
        drives.add(HomeostasisDrive(homeostasis))
        plasticity = RewardModulatedSTDP(synapses, learning_rate=cast(float, o["learning_rate"]))
        plugins: list[NetworkPlugin] = [
            MetricsPlugin(metrics),
            HomeostasisPlugin(homeostasis),
            PlasticityPlugin(plasticity),
        ]
        return NetworkBundle(
            input_features,
            output_tokens,
            neurons_per_token,
            layout,
            network,
            EventReadout(layout),
            np.ones(synapses.weight.size, dtype=bool),
            plasticity,
            ResponseSession(policy=cast(SessionPolicy, o["session_policy"])),
            metrics,
            homeostasis,
            external,
            background,
            drives,
            plugins,
            cast(MetricsPlugin, plugins[0]),
            NetworkContext(network.tick, neurons.voltage, neurons.voltage),
            hidden_groups,
            input_hidden,
            affect_indices,
            affect_action_indices,
            direct,
            hidden,
            direct.reshape(input_features, len(output_tokens), neurons_per_token),
            np.flatnonzero((synapses.source >= input_features) & (synapses.target >= char_start)),
        )

    @staticmethod
    def _projection(
        synapses: SparseSynapses,
        rng: np.random.Generator,
        sources: np.ndarray,
        targets: np.ndarray,
        mean: float,
        spread: float,
    ) -> None:
        source = np.repeat(sources, targets.size)
        synapses.add_edges(
            source, np.tile(targets, sources.size), rng.normal(mean, spread, source.size)
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
