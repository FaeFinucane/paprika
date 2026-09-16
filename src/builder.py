"""Construction of networks and their session-local interaction plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from .interaction.drives import HomeostaticDrive, TonicDrive
from .interaction.plasticity.homeostasis import SynapticScaling
from .interaction.plugins import Hook
from .interaction.rates import PopulationRate
from .network.connectivity import (
    ConnectionSpec,
    LearningPolicy,
    StrengthSpec,
    TopologySpec,
    build_synapses,
)
from .network.population import (
    FeaturePopulationSpec,
    NeuronPopulationSpec,
    PopulationLayout,
    PopulationSpec,
)
from .network.snn import SNN
from .session import Session


@dataclass(frozen=True, slots=True)
class TonicDriveSpec:
    """Declarative specification for a population tonic current."""

    current: float
    heterogeneity: float = 0.0


@dataclass(frozen=True, slots=True)
class HomeostaticDriveSpec:
    """Declarative specification for a population homeostatic current."""

    target_rate: float
    learning_rate: float = 0.0001
    rate_decay: float = 0.999
    minimum_current: float = -0.1
    maximum_current: float = 0.1
    initial_current: float = 0.0
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class PopulationRateSpec:
    """Declarative low-pass population-rate observer."""

    decay: float = 0.95


@dataclass(frozen=True, slots=True)
class SynapticScalingSpec:
    """Declarative slow scaling of explicitly scalable excitatory inputs."""

    target_rate: float
    learning_rate: float = 0.00001
    rate_decay: float = 0.9999


PluginSpec: TypeAlias = (
    TonicDriveSpec | HomeostaticDriveSpec | PopulationRateSpec | SynapticScalingSpec
)


@dataclass(frozen=True, slots=True)
class PopulationHandle:
    """Named reference returned while constructing a population."""

    name: str


class NetworkBuilder:
    """Mutable network declaration that compiles directly into a Session.

    Population and connection declarations remain lightweight until compile
    time. Plugin specifications are also declarations: runtime hooks are
    instantiated only after the SNN layout exists, using the compile seed.
    """

    def __init__(self) -> None:
        self._populations: list[PopulationSpec] = []
        self._connections: list[ConnectionSpec] = []
        self._plugins: list[tuple[str, PluginSpec]] = []

    @property
    def populations(self) -> tuple[PopulationSpec, ...]:
        return tuple(self._populations)

    @property
    def connections(self) -> tuple[ConnectionSpec, ...]:
        return tuple(self._connections)

    def add_population(
        self,
        name: str | NeuronPopulationSpec,
        neuron_count: int | None = None,
        *,
        output: str = "excitatory",
        dopamine_response: str = "neutral",
        plugins: tuple[PluginSpec, ...] = (),
    ) -> PopulationHandle:
        if isinstance(name, NeuronPopulationSpec):
            if neuron_count is not None or output != "excitatory" or dopamine_response != "neutral":
                raise TypeError("population spec cannot be combined with population options")
            spec = name
        else:
            if neuron_count is None:
                raise TypeError("neuron_count is required")
            spec = NeuronPopulationSpec(
                name,
                neuron_count,
                output=output,  # type: ignore[arg-type]
                dopamine_response=dopamine_response,  # type: ignore[arg-type]
            )
        self._add_population_spec(spec, plugins)
        return PopulationHandle(spec.name)

    def add_feature_population(
        self,
        name: str | FeaturePopulationSpec,
        features: tuple[str, ...] = (),
        *,
        width: int | None = None,
        plugins: tuple[PluginSpec, ...] = (),
    ) -> PopulationHandle:
        if isinstance(name, FeaturePopulationSpec):
            if features or width is not None:
                raise TypeError("population spec cannot be combined with population options")
            spec = name
        else:
            if not features or width is None:
                raise TypeError("features and width are required")
            spec = FeaturePopulationSpec(name, features, width)
        self._add_population_spec(spec, plugins)
        return PopulationHandle(spec.name)

    def _add_population_spec(self, spec: PopulationSpec, plugins: tuple[PluginSpec, ...]) -> None:
        if any(existing.name == spec.name for existing in self._populations):
            raise ValueError(f"population names must be unique: {spec.name!r}")
        self._populations.append(spec)
        for plugin in plugins:
            self.attach(spec.name, plugin)

    def attach(self, population: str | PopulationHandle, plugin: PluginSpec) -> None:
        """Attach a declarative population-local plugin."""
        name = population.name if isinstance(population, PopulationHandle) else population
        if not isinstance(
            plugin,
            (TonicDriveSpec, HomeostaticDriveSpec, PopulationRateSpec, SynapticScalingSpec),
        ):
            raise TypeError("unsupported plugin specification")
        if not any(spec.name == name for spec in self._populations):
            raise KeyError(f"unknown population {name!r}")
        self._plugins.append((name, plugin))

    def connect(
        self,
        source: str | ConnectionSpec,
        target: str | None = None,
        topology: TopologySpec | None = None,
        strength: StrengthSpec | None = None,
        learning: LearningPolicy = "fixed",
        scalable: bool = False,
        name: str | None = None,
    ) -> ConnectionSpec:
        """Declare a projection, accepting either a ConnectionSpec or fields."""
        if isinstance(source, ConnectionSpec):
            if any(value is not None for value in (target, topology, strength)):
                raise TypeError("connection spec cannot be combined with connection options")
            spec = source
        else:
            if target is None or topology is None or strength is None:
                raise TypeError("target, topology, and strength are required")
            spec = ConnectionSpec(source, target, topology, strength, learning, scalable, name)
        self._connections.append(spec)
        return spec

    def compile(self, seed: int = 0) -> Session:
        """Compile declarations and return a ready-to-run session."""
        if not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
        root_rng = np.random.default_rng(int(seed))
        layout = PopulationLayout.build(self._populations)
        snn = SNN.build(layout, build_synapses(layout, self._connections, root_rng))
        populations = {population.spec.name: population for population in layout.populations}
        hooks: list[Hook] = []
        for index, (name, plugin) in enumerate(self._plugins):
            population = populations[name]
            plugin_rng = np.random.default_rng(root_rng.integers(0, 2**63, dtype=np.int64) + index)
            if isinstance(plugin, TonicDriveSpec):
                hooks.append(
                    TonicDrive(population, plugin.current, plugin.heterogeneity, plugin_rng)
                )
            elif isinstance(plugin, HomeostaticDriveSpec):
                hooks.append(
                    HomeostaticDrive(
                        population,
                        plugin.target_rate,
                        plugin.learning_rate,
                        plugin.rate_decay,
                        plugin.minimum_current,
                        plugin.maximum_current,
                        plugin.initial_current,
                        plugin.enabled,
                    )
                )
            elif isinstance(plugin, PopulationRateSpec):
                hooks.append(PopulationRate(population, plugin.decay))
            elif isinstance(plugin, SynapticScalingSpec):
                hooks.append(
                    SynapticScaling(
                        snn,
                        population,
                        plugin.target_rate,
                        plugin.learning_rate,
                        plugin.rate_decay,
                    )
                )
        return Session.build(snn, hooks)
