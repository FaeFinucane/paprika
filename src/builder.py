"""Construction of networks and their session-local interaction plugins."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

import numpy as np

from .interaction.plugins import Hook
from .network.connectivity import (
    ConnectionSpec,
    LearningPolicy,
    StrengthSpec,
    TopologySpec,
    build_synapses,
)
from .network.population import (
    DopamineResponse,
    FeaturePopulationSpec,
    NeuronPopulationSpec,
    OutputKind,
    Population,
    PopulationLayout,
    PopulationSpec,
)
from .network.snn import SNN
from .session import Session


class PluginSpec(Protocol):
    """A fully bound declaration that builds session hooks after compilation."""

    def build_hooks(self, session: Session, rng: np.random.Generator) -> tuple[Hook, ...]: ...


class PopulationPluginSpec(Protocol):
    """Population-local settings that create hooks for one compiled population."""

    def build_population_hooks(
        self,
        session: Session,
        population: Population[Any],
        rng: np.random.Generator,
    ) -> tuple[Hook, ...]: ...


@dataclass(frozen=True, slots=True)
class _BoundPopulationPluginSpec(PluginSpec):
    """Generic binding of population-local settings to a declared population."""

    population_name: str
    plugin: PopulationPluginSpec

    def build_hooks(self, session: Session, rng: np.random.Generator) -> tuple[Hook, ...]:
        population = session.snn.layout.population(self.population_name)
        return self.plugin.build_population_hooks(session, population, rng)


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
        self._plugins: list[PluginSpec] = []

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
        output: OutputKind = "excitatory",
        dopamine_response: DopamineResponse = "neutral",
        plugins: tuple[PopulationPluginSpec, ...] = (),
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
                output=output,
                dopamine_response=dopamine_response,
            )
        self._add_population_spec(spec, plugins)
        return PopulationHandle(spec.name)

    def add_feature_population(
        self,
        name: str | FeaturePopulationSpec,
        features: tuple[str, ...] = (),
        *,
        width: int | None = None,
        plugins: tuple[PopulationPluginSpec, ...] = (),
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

    def _add_population_spec(
        self, spec: PopulationSpec, plugins: tuple[PopulationPluginSpec, ...]
    ) -> None:
        if any(existing.name == spec.name for existing in self._populations):
            raise ValueError(f"population names must be unique: {spec.name!r}")
        self._populations.append(spec)
        for plugin in plugins:
            self.attach(spec.name, plugin)

    def attach(self, population: str | PopulationHandle, plugin: PopulationPluginSpec) -> None:
        """Bind population-local plugin settings to a declared population."""
        name = population.name if isinstance(population, PopulationHandle) else population
        if not any(spec.name == name for spec in self._populations):
            raise KeyError(f"unknown population {name!r}")
        if not callable(getattr(plugin, "build_population_hooks", None)):
            raise TypeError("population plugin must define build_population_hooks")
        self._plugins.append(_BoundPopulationPluginSpec(name, plugin))

    def add_plugin(self, plugin: PluginSpec) -> None:
        """Declare a system-level plugin whose hooks are built at compilation."""
        if not callable(getattr(plugin, "build_hooks", None)):
            raise TypeError("plugin must define build_hooks(session, rng)")
        self._plugins.append(plugin)

    def connect(
        self,
        source: str | PopulationHandle | ConnectionSpec,
        target: str | PopulationHandle | None = None,
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
            source_name = source.name if isinstance(source, PopulationHandle) else source
            target_name = target.name if isinstance(target, PopulationHandle) else target
            spec = ConnectionSpec(
                source_name,
                target_name,
                topology,
                replace(strength),
                learning,
                scalable,
                name,
            )
        self._connections.append(spec)
        return spec

    def connection(
        self, source: str | PopulationHandle, target: str | PopulationHandle
    ) -> ConnectionSpec:
        """Return the unique declared projection between two populations."""
        source_name = source.name if isinstance(source, PopulationHandle) else source
        target_name = target.name if isinstance(target, PopulationHandle) else target
        matches = [
            connection
            for connection in self._connections
            if connection.source == source_name and connection.target == target_name
        ]
        if len(matches) != 1:
            raise KeyError(
                f"expected one connection from {source_name!r} to {target_name!r}, found {len(matches)}"
            )
        return matches[0]

    def compile(self, seed: int = 0) -> Session:
        """Compile declarations and return a ready-to-run session."""
        if not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
        root_rng = np.random.default_rng(int(seed))
        layout = PopulationLayout.build(self._populations)
        snn = SNN.build(layout, build_synapses(layout, self._connections, root_rng))
        session = Session(snn)
        for index, plugin in enumerate(self._plugins):
            plugin_rng = np.random.default_rng(root_rng.integers(0, 2**63, dtype=np.int64) + index)
            session.add(*plugin.build_hooks(session, plugin_rng))
        return session
