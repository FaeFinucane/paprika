"""Cue, inferred-state, temporal, and VTA dopamine circuit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..builder import (
    HomeostaticDriveSpec,
    NetworkBuilder,
    PopulationRateSpec,
    TonicDriveSpec,
)
from ..interaction.plasticity import DopamineSTDP, InhibitoryHomeostasis
from ..interaction.rates import DopamineReadout, PopulationRate, RatePatternInput, UnipolarRateInput
from ..network.connectivity import FanInSpec, StrengthSpec
from ..network.population import Population
from ..session import Session
from .components import (
    AttractorHandle,
    AttractorSpec,
    TemporalSequenceHandle,
    TemporalSequenceSpec,
    add_attractor,
    add_temporal_sequence,
)


@dataclass
class DopamineCircuit:
    """A continuously running VTA circuit with no scalar value-comparator path."""

    session: Session
    populations: dict[str, Population[Any]]
    cue_input: RatePatternInput
    outcome_positive: UnipolarRateInput
    outcome_negative: UnipolarRateInput
    inferred_state: AttractorHandle
    temporal: TemporalSequenceHandle
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


def build_dopamine_circuit(
    seed: int = 0,
    *,
    cue_width: int = 16,
    outcome_size: int = 16,
    dopamine_size: int = 32,
    vta_inhibitory_size: int = 16,
    attractor: AttractorSpec = AttractorSpec(),
    temporal: TemporalSequenceSpec = TemporalSequenceSpec(),
    enable_dopamine_learning: bool = True,
) -> DopamineCircuit:
    """Build a VTA circuit from reusable state-attractor and temporal helpers."""
    if min(cue_width, outcome_size, dopamine_size, vta_inhibitory_size) <= 0:
        raise ValueError("population sizes must be positive")
    builder = NetworkBuilder()
    cue = builder.add_feature_population("CUE", ("CUE",), width=cue_width)
    builder.add_population("OUTCOME_POSITIVE", outcome_size)
    builder.add_population("OUTCOME_NEGATIVE", outcome_size, output="inhibitory")
    inferred_state = add_attractor(builder, attractor, cue=cue)
    temporal_handle = add_temporal_sequence(builder, temporal, state=inferred_state)
    builder.add_population(
        "VTA_INHIB",
        vta_inhibitory_size,
        output="inhibitory",
        dopamine_response="aligned",
        plugins=(HomeostaticDriveSpec(0.03),),
    )
    builder.add_population(
        "VTA_DA",
        dopamine_size,
        output="modulatory",
        dopamine_response="aligned",
        plugins=(
            TonicDriveSpec(0.42, heterogeneity=0.06),
            HomeostaticDriveSpec(0.25, learning_rate=0.00001),
            PopulationRateSpec(decay=0.9),
        ),
    )

    learned = StrengthSpec(0.12, 0.02, maximum=0.8)
    temporal_to_inhibitory = StrengthSpec(0.30, 0.02, maximum=0.8)
    outcome_strength = StrengthSpec(0.30, 0.02, maximum=0.8)
    inhibitory = StrengthSpec(0.22, 0.02, maximum=0.8)
    state_fanin = min(8, attractor.excitatory_size)
    temporal_fanin = min(4, temporal.stage_size)
    vta_inhibitory_fanin = min(8, vta_inhibitory_size)
    outcome_fanin = min(8, outcome_size)
    builder.connect(
        inferred_state.excitatory,
        "VTA_DA",
        FanInSpec(state_fanin),
        learned,
        "dopamine_stdp",
        True,
        "INFERRED_STATE_to_VTA_DA",
    )
    for stage in temporal_handle.stages:
        builder.connect(
            stage,
            "VTA_INHIB",
            FanInSpec(temporal_fanin),
            temporal_to_inhibitory,
            "dopamine_stdp",
            True,
            f"{stage}_to_VTA_INHIB",
        )
    builder.connect(
        "VTA_INHIB",
        "VTA_DA",
        FanInSpec(vta_inhibitory_fanin),
        inhibitory,
        "inhibitory_homeostatic",
        False,
        "VTA_INHIB_to_VTA_DA",
    )
    builder.connect(
        "OUTCOME_POSITIVE",
        "VTA_DA",
        FanInSpec(outcome_fanin),
        outcome_strength,
        "fixed",
        False,
        "OUTCOME_POSITIVE_to_VTA_DA",
    )
    builder.connect(
        "OUTCOME_NEGATIVE",
        "VTA_DA",
        FanInSpec(outcome_fanin),
        outcome_strength,
        "fixed",
        False,
        "OUTCOME_NEGATIVE_to_VTA_DA",
    )

    session = builder.compile(seed)
    populations = {
        population.spec.name: population for population in session.snn.layout.populations
    }
    dopamine_rate = next(
        observer
        for observer in session.observers
        if isinstance(observer, PopulationRate) and observer.population == populations["VTA_DA"]
    )
    dopamine = DopamineReadout(dopamine_rate, baseline=0.25, deadzone=0.04)
    stdp = DopamineSTDP(session.snn, dopamine, learning_rate=0.002)
    cue_input = RatePatternInput(populations["CUE"], np.random.default_rng([seed, 1]))
    outcome_positive = UnipolarRateInput(
        populations["OUTCOME_POSITIVE"], np.random.default_rng([seed, 2])
    )
    outcome_negative = UnipolarRateInput(
        populations["OUTCOME_NEGATIVE"], np.random.default_rng([seed, 3])
    )
    session.drive_sources.extend((cue_input, outcome_positive, outcome_negative))
    session.observers.append(dopamine)
    if enable_dopamine_learning:
        session.adaptations.append(stdp)
    session.adaptations.append(InhibitoryHomeostasis(session.snn, target_rate=0.04))
    return DopamineCircuit(
        session,
        populations,
        cue_input,
        outcome_positive,
        outcome_negative,
        inferred_state,
        temporal_handle,
        dopamine_rate,
        dopamine,
        stdp,
    )


__all__ = ["DopamineCircuit", "build_dopamine_circuit"]
