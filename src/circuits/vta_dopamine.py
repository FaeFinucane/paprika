"""Cue, inferred-state, temporal, and VTA dopamine circuit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..builder import (
    HomeostaticDriveSpec,
    NetworkBuilder,
    PopulationHandle,
    PopulationRateSpec,
    TonicDriveSpec,
)
from ..interaction.plasticity import DopamineSTDP
from ..interaction.plugins import Hook
from ..interaction.rates import DopamineReadout, PopulationRate, RatePatternInput, UnipolarRateInput
from ..network.connectivity import FanInSpec, StrengthSpec
from ..network.population import Population
from ..session import Session
from .asymmetric_recurrent import (
    AsymmetricRecurrentHandle,
    AsymmetricRecurrentSpec,
    add_asymmetric_recurrent_circuit,
)
from .attractor import AttractorHandle, AttractorSpec, add_attractor


@dataclass
class DopamineCircuit:
    """A continuously running VTA circuit with no scalar value-comparator path."""

    session: Session
    populations: dict[str, Population[Any]]
    cue_input: RatePatternInput
    outcome_positive: UnipolarRateInput
    outcome_negative: UnipolarRateInput
    inferred_state: AttractorHandle
    temporal: AsymmetricRecurrentHandle
    dopamine_rate: PopulationRate
    dopamine: DopamineReadout
    stdp: DopamineSTDP

    def present_cue(self, pattern: np.ndarray, duration: int | None = None) -> None:
        """Present a bounded unipolar cue without clearing ongoing network state."""
        self.cue_input.write(pattern, duration)

    def deliver_outcome(self, value: float, duration: int | None = None) -> None:
        """Route one signed outcome through its fixed VTA-DA sensory pathway."""
        if not np.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError("outcome must be finite and in [-1, 1]")
        if value >= 0:
            self.outcome_positive.write(float(value), duration)
        else:
            self.outcome_negative.write(float(-value), duration)

    def tick(self):
        return self.session.tick()


@dataclass(frozen=True, slots=True)
class VtaDopamineCircuitDeclaration:
    """Uncompiled VTA circuit declared into a mutable network builder."""

    builder: NetworkBuilder
    cue: PopulationHandle
    outcome_positive: PopulationHandle
    outcome_negative: PopulationHandle
    inferred_state: AttractorHandle
    temporal: AsymmetricRecurrentHandle
    vta_inhibitory: PopulationHandle
    vta_dopamine: PopulationHandle

    def build(self, seed: int = 0, *, enable_dopamine_learning: bool = True) -> DopamineCircuit:
        """Compile this declaration and attach its session-local interaction hooks."""
        return _compile_dopamine_circuit(self, seed, enable_dopamine_learning)


def add_vta_dopamine_circuit(
    builder: NetworkBuilder,
    *,
    attractor: AttractorSpec = AttractorSpec(),
    temporal: AsymmetricRecurrentSpec = AsymmetricRecurrentSpec(),
) -> VtaDopamineCircuitDeclaration:
    """Declare the VTA dopamine circuit into ``builder`` without compiling it."""
    cue = builder.add_feature_population("CUE", ("CUE",), width=16)
    outcome_positive = builder.add_population("OUTCOME_POSITIVE", 16)
    outcome_negative = builder.add_population("OUTCOME_NEGATIVE", 16, output="inhibitory")
    inferred_state = add_attractor(builder, attractor, cue=cue)
    timer = add_asymmetric_recurrent_circuit(builder, temporal, state=inferred_state)
    vta_inhibitory = builder.add_population(
        "VTA_INHIB",
        16,
        output="inhibitory",
        dopamine_response="aligned",
        plugins=(HomeostaticDriveSpec(0.03),),
    )
    vta_dopamine = builder.add_population(
        "VTA_DA",
        32,
        output="modulatory",
        dopamine_response="aligned",
        plugins=(
            TonicDriveSpec(0.42, heterogeneity=0.06),
            HomeostaticDriveSpec(0.25, learning_rate=0.00001),
            PopulationRateSpec(decay=0.8),
        ),
    )

    direct_state_to_dopamine = StrengthSpec(0.12, 0.02, maximum=0.8)
    temporal_to_inhibitory = StrengthSpec(0.10, 0.02, maximum=0.8)
    outcome_strength = StrengthSpec(0.30, 0.02, maximum=0.8)
    inhibitory = StrengthSpec(0.22, 0.02, maximum=0.8)
    state_fanin = min(8, attractor.excitatory_size)
    temporal_fanin = min(8, temporal.excitatory_size)
    vta_inhibitory_fanin = 8
    outcome_fanin = 8
    builder.connect(
        inferred_state.excitatory,
        vta_dopamine,
        FanInSpec(state_fanin),
        direct_state_to_dopamine,
        "dopamine_stdp",
        True,
        "INFERRED_STATE_to_VTA_DA",
    )
    builder.connect(
        timer.excitatory,
        vta_inhibitory,
        FanInSpec(temporal_fanin),
        temporal_to_inhibitory,
        "dopamine_stdp",
        True,
        f"{timer.excitatory}_to_VTA_INHIB",
    )
    builder.connect(
        vta_inhibitory,
        vta_dopamine,
        FanInSpec(vta_inhibitory_fanin),
        inhibitory,
        "dopamine_stdp",
        False,
        "VTA_INHIB_to_VTA_DA",
    )
    builder.connect(
        outcome_positive,
        vta_dopamine,
        FanInSpec(outcome_fanin),
        outcome_strength,
        "fixed",
        False,
        "OUTCOME_POSITIVE_to_VTA_DA",
    )
    builder.connect(
        outcome_negative,
        vta_dopamine,
        FanInSpec(outcome_fanin),
        outcome_strength,
        "fixed",
        False,
        "OUTCOME_NEGATIVE_to_VTA_DA",
    )

    return VtaDopamineCircuitDeclaration(
        builder,
        cue,
        outcome_positive,
        outcome_negative,
        inferred_state,
        timer,
        vta_inhibitory,
        vta_dopamine,
    )


def build_dopamine_circuit(
    seed: int = 0, *, enable_dopamine_learning: bool = True
) -> DopamineCircuit:
    """Build the default VTA dopamine circuit in one call."""
    return add_vta_dopamine_circuit(NetworkBuilder()).build(
        seed, enable_dopamine_learning=enable_dopamine_learning
    )


def _compile_dopamine_circuit(
    declaration: VtaDopamineCircuitDeclaration,
    seed: int,
    enable_dopamine_learning: bool,
) -> DopamineCircuit:
    session = declaration.builder.compile(seed)
    populations = {
        population.spec.name: population for population in session.snn.layout.populations
    }
    dopamine_rate = next(
        observer
        for observer in session.observers
        if isinstance(observer, PopulationRate)
        and observer.population == populations[declaration.vta_dopamine.name]
    )
    dopamine = DopamineReadout(dopamine_rate, baseline=0.25, deadzone=0.04)
    stdp = DopamineSTDP(session.snn, dopamine, learning_rate=0.002)
    cue_input = RatePatternInput(
        populations[declaration.cue.name], np.random.default_rng([seed, 1])
    )
    outcome_positive = UnipolarRateInput(
        populations[declaration.outcome_positive.name], np.random.default_rng([seed, 2])
    )
    outcome_negative = UnipolarRateInput(
        populations[declaration.outcome_negative.name], np.random.default_rng([seed, 3])
    )
    hooks: list[Hook] = [cue_input, outcome_positive, outcome_negative, dopamine]
    if enable_dopamine_learning:
        hooks.append(stdp)
    session.add(*hooks)
    return DopamineCircuit(
        session,
        populations,
        cue_input,
        outcome_positive,
        outcome_negative,
        declaration.inferred_state,
        declaration.temporal,
        dopamine_rate,
        dopamine,
        stdp,
    )


__all__ = [
    "DopamineCircuit",
    "VtaDopamineCircuitDeclaration",
    "add_vta_dopamine_circuit",
    "build_dopamine_circuit",
]
