"""Reusable VTA dopamine populations and their intrinsic modulation pathway."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..builder import NetworkBuilder, PluginSpec
from ..interaction.drives import HomeostaticDriveSpec, TonicDriveSpec
from ..interaction.plasticity import DopamineSTDP
from ..interaction.plugins import Hook
from ..interaction.rates import DopamineReadout, PopulationRate, PopulationRateSpec
from ..network.connectivity import FanInSpec, StrengthSpec
from ..network.population import Population
from ..session import Session

VTA_DOPAMINE_LEARNING_RATE = 0.0001


@dataclass(frozen=True, slots=True)
class VtaDopamineSpec:
    """Intrinsic VTA population dimensions and modulation parameters."""

    name: str = "VTA"
    inhibitory_size: int = 16
    dopamine_size: int = 32
    inhibitory_fanin: int = 8
    inhibitory_strength: float = 0.22
    dopamine_tonic_drive: float = 0.42
    dopamine_tonic_heterogeneity: float = 0.06
    dopamine_target_rate: float = 0.25
    dopamine_rate_decay: float = 0.8
    dopamine_baseline: float = 0.25
    dopamine_deadzone: float = 0.04
    learning_rate: float = VTA_DOPAMINE_LEARNING_RATE


@dataclass(frozen=True, slots=True)
class VtaDopamineHandle:
    """Ports of the reusable VTA dopamine motif.

    Parent circuit assemblies choose every incoming pathway.  In particular,
    direct prediction, temporal prediction, and sensory outcome projections do
    not belong to this component.
    """

    inhibitory: str
    dopamine: str

    @property
    def inhibitory_input(self) -> str:
        return self.inhibitory

    @property
    def dopamine_input(self) -> str:
        return self.dopamine


@dataclass
class VtaDopamineSystem:
    """Compiled VTA populations, readout, and one session-managed DA rule."""

    inhibitory: Population
    dopamine_population: Population
    dopamine_rate: PopulationRate
    dopamine: DopamineReadout
    stdp: DopamineSTDP


@dataclass(frozen=True, slots=True)
class _DopamineSystemPluginSpec(PluginSpec):
    dopamine_name: str
    baseline: float
    deadzone: float
    learning_rate: float

    def build_hooks(self, session: Session, _rng: np.random.Generator) -> tuple[Hook, ...]:
        population = session.snn.layout.population(self.dopamine_name)
        rate = next(
            observer
            for observer in session.observers
            if isinstance(observer, PopulationRate) and observer.population == population
        )
        dopamine = DopamineReadout(rate, baseline=self.baseline, deadzone=self.deadzone)
        return dopamine, DopamineSTDP(session.snn, dopamine, learning_rate=self.learning_rate)


def add_vta_dopamine(
    builder: NetworkBuilder, spec: VtaDopamineSpec = VtaDopamineSpec()
) -> VtaDopamineHandle:
    """Declare VTA-DA/VTA-INHIB and their intrinsic learnable inhibition.

    The only projection owned here is ``VTA_INHIB -> VTA_DA``.  All projections
    from sensory, state, or temporal circuits are declared by the parent that
    composes this motif.
    """
    if (
        min(spec.inhibitory_size, spec.dopamine_size, spec.inhibitory_fanin) <= 0
        or spec.inhibitory_fanin > spec.inhibitory_size
        or not 0 <= spec.inhibitory_strength <= 0.8
        or not 0 <= spec.dopamine_target_rate <= 1
        or not 0 <= spec.dopamine_rate_decay < 1
        or not 0 <= spec.dopamine_baseline <= 1
        or not 0 <= spec.dopamine_deadzone < 1
        or spec.learning_rate < 0
    ):
        raise ValueError("invalid VTA dopamine configuration")

    inhibitory = f"{spec.name}_INHIB"
    dopamine = f"{spec.name}_DA"
    builder.add_population(
        inhibitory,
        spec.inhibitory_size,
        output="inhibitory",
        dopamine_response="aligned",
        plugins=(HomeostaticDriveSpec(0.03),),
    )
    builder.add_population(
        dopamine,
        spec.dopamine_size,
        output="modulatory",
        dopamine_response="aligned",
        plugins=(
            TonicDriveSpec(
                spec.dopamine_tonic_drive, heterogeneity=spec.dopamine_tonic_heterogeneity
            ),
            HomeostaticDriveSpec(spec.dopamine_target_rate, learning_rate=0.00001),
            PopulationRateSpec(decay=spec.dopamine_rate_decay),
        ),
    )
    builder.connect(
        inhibitory,
        dopamine,
        FanInSpec(spec.inhibitory_fanin),
        StrengthSpec(spec.inhibitory_strength, 0.02, maximum=0.8),
        "dopamine_stdp",
        False,
        f"{inhibitory}_to_{dopamine}",
    )
    builder.add_plugin(
        _DopamineSystemPluginSpec(
            dopamine,
            spec.dopamine_baseline,
            spec.dopamine_deadzone,
            spec.learning_rate,
        )
    )
    return VtaDopamineHandle(inhibitory, dopamine)


def compiled_vta_dopamine(session: Session, handle: VtaDopamineHandle) -> VtaDopamineSystem:
    """Return the runtime VTA system automatically installed during compilation."""
    inhibitory = session.snn.layout.population(handle.inhibitory)
    dopamine_population = session.snn.layout.population(handle.dopamine)
    rate = next(
        observer
        for observer in session.observers
        if isinstance(observer, PopulationRate) and observer.population == dopamine_population
    )
    dopamine = next(
        observer
        for observer in session.observers
        if isinstance(observer, DopamineReadout) and observer.source is rate
    )
    stdp = next(
        adaptation
        for adaptation in session.adaptations
        if isinstance(adaptation, DopamineSTDP) and adaptation.dopamine is dopamine
    )
    return VtaDopamineSystem(inhibitory, dopamine_population, rate, dopamine, stdp)


__all__ = [
    "VTA_DOPAMINE_LEARNING_RATE",
    "VtaDopamineHandle",
    "VtaDopamineSpec",
    "VtaDopamineSystem",
    "add_vta_dopamine",
    "compiled_vta_dopamine",
]
