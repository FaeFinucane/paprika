"""Cue-driven inferred-state attractor circuit."""

from __future__ import annotations

from dataclasses import dataclass

from ..builder import HomeostaticDriveSpec, NetworkBuilder, PopulationHandle, SynapticScalingSpec
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


__all__ = ["AttractorHandle", "AttractorSpec", "add_attractor"]
