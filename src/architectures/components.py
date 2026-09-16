"""Reusable population assemblies for dopamine-circuit architectures."""

from __future__ import annotations

from dataclasses import dataclass

from ..builder import (
    HomeostaticDriveSpec,
    NetworkBuilder,
    PopulationHandle,
    SynapticScalingSpec,
)
from ..network.connectivity import FanInSpec, FanOutSpec, StrengthSpec


@dataclass(frozen=True, slots=True)
class AttractorSpec:
    """Parameters for a sparse excitatory/inhibitory inferred-state assembly."""

    name: str = "INFERRED_STATE"
    excitatory_size: int = 48
    inhibitory_size: int = 12
    cue_fanin: float = 6
    recurrent_fanout: float = 5
    inhibition_fanout: float = 6
    target_rate: float = 0.04


@dataclass(frozen=True, slots=True)
class AttractorHandle:
    excitatory: str
    inhibitory: str


@dataclass(frozen=True, slots=True)
class TemporalSequenceSpec:
    """Parameters for an explicit, locally regulated temporal sequence."""

    name: str = "TEMPORAL"
    stages: int = 8
    stage_size: int = 12
    inhibitory_size: int = 6
    state_fanin: float = 4
    forward_fanin: float = 4
    recurrent_fanout: float = 3
    target_rate: float = 0.03


@dataclass(frozen=True, slots=True)
class TemporalSequenceHandle:
    stages: tuple[str, ...]
    inhibitory: tuple[str, ...]


def add_attractor(
    builder: NetworkBuilder,
    spec: AttractorSpec,
    *,
    cue: str | PopulationHandle,
) -> AttractorHandle:
    """Declare a cue-driven inferred-state E/I attractor."""
    if min(spec.excitatory_size, spec.inhibitory_size) <= 0 or not 0 <= spec.target_rate <= 1:
        raise ValueError("invalid attractor dimensions or target rate")
    cue_name = cue.name if isinstance(cue, PopulationHandle) else cue
    excitatory = f"{spec.name}_E"
    inhibitory = f"{spec.name}_I"
    builder.add_population(
        excitatory,
        spec.excitatory_size,
        dopamine_response="aligned",
        plugins=(
            HomeostaticDriveSpec(spec.target_rate),
            SynapticScalingSpec(spec.target_rate),
        ),
    )
    builder.add_population(
        inhibitory,
        spec.inhibitory_size,
        output="inhibitory",
        dopamine_response="aligned",
        plugins=(HomeostaticDriveSpec(spec.target_rate),),
    )
    cue_strength = StrengthSpec(0.25, 0.02, maximum=0.8)
    recurrent_strength = StrengthSpec(0.12, 0.02, maximum=0.8)
    inhibitory_strength = StrengthSpec(0.22, 0.02, maximum=0.8)
    builder.connect(
        cue_name,
        excitatory,
        FanInSpec(spec.cue_fanin),
        cue_strength,
        "dopamine_stdp",
        True,
        f"{cue_name}_to_{excitatory}",
    )
    builder.connect(
        excitatory,
        excitatory,
        FanOutSpec(spec.recurrent_fanout),
        recurrent_strength,
        "dopamine_stdp",
        True,
        f"{excitatory}_recurrent",
    )
    builder.connect(
        excitatory,
        inhibitory,
        FanOutSpec(spec.inhibition_fanout),
        recurrent_strength,
        "dopamine_stdp",
        True,
        f"{excitatory}_to_{inhibitory}",
    )
    builder.connect(
        inhibitory,
        excitatory,
        FanOutSpec(spec.inhibition_fanout),
        inhibitory_strength,
        "inhibitory_homeostatic",
        False,
        f"{inhibitory}_to_{excitatory}",
    )
    return AttractorHandle(excitatory, inhibitory)


def add_temporal_sequence(
    builder: NetworkBuilder,
    spec: TemporalSequenceSpec,
    *,
    state: AttractorHandle,
) -> TemporalSequenceHandle:
    """Declare an explicit sequence of locally regulated temporal assemblies."""
    if (
        spec.stages <= 0
        or min(spec.stage_size, spec.inhibitory_size) <= 0
        or not 0 <= spec.target_rate <= 1
    ):
        raise ValueError("invalid temporal sequence dimensions or target rate")
    stages = tuple(f"{spec.name}_{index}" for index in range(spec.stages))
    inhibitory = tuple(f"{spec.name}_INHIB_{index}" for index in range(spec.stages))
    learned = StrengthSpec(0.12, 0.02, maximum=0.8)
    state_drive = StrengthSpec(0.30, 0.02, maximum=0.8)
    forward_drive = StrengthSpec(0.22, 0.02, maximum=0.8)
    inhibitory_strength = StrengthSpec(0.22, 0.02, maximum=0.8)
    for stage, stabilizer in zip(stages, inhibitory, strict=True):
        builder.add_population(
            stage,
            spec.stage_size,
            dopamine_response="aligned",
            plugins=(
                HomeostaticDriveSpec(spec.target_rate),
                SynapticScalingSpec(spec.target_rate),
            ),
        )
        builder.add_population(
            stabilizer,
            spec.inhibitory_size,
            output="inhibitory",
            dopamine_response="aligned",
            plugins=(HomeostaticDriveSpec(spec.target_rate),),
        )
        builder.connect(
            stage,
            stage,
            FanOutSpec(spec.recurrent_fanout),
            learned,
            "dopamine_stdp",
            True,
            f"{stage}_recurrent",
        )
        builder.connect(
            stage,
            stabilizer,
            FanOutSpec(spec.recurrent_fanout),
            learned,
            "dopamine_stdp",
            True,
            f"{stage}_to_{stabilizer}",
        )
        builder.connect(
            stabilizer,
            stage,
            FanOutSpec(spec.recurrent_fanout),
            inhibitory_strength,
            "inhibitory_homeostatic",
            False,
            f"{stabilizer}_to_{stage}",
        )
    builder.connect(
        state.excitatory,
        stages[0],
        FanInSpec(spec.state_fanin),
        state_drive,
        "dopamine_stdp",
        True,
        f"{state.excitatory}_to_{stages[0]}",
    )
    for source, target in zip(stages[:-1], stages[1:], strict=True):
        builder.connect(
            source,
            target,
            FanInSpec(spec.forward_fanin),
            forward_drive,
            "dopamine_stdp",
            True,
            f"{source}_to_{target}",
        )
    return TemporalSequenceHandle(stages, inhibitory)


__all__ = [
    "AttractorHandle",
    "AttractorSpec",
    "TemporalSequenceHandle",
    "TemporalSequenceSpec",
    "add_attractor",
    "add_temporal_sequence",
]
