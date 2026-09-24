"""Cue-driven inferred-state attractor circuit."""

from __future__ import annotations

from dataclasses import dataclass

from ..builder import NetworkBuilder
from ..interaction.drives import HomeostaticDriveSpec
from ..interaction.plasticity.homeostasis import SynapticScalingSpec
from ..network.connectivity import FanOutSpec, StrengthSpec


@dataclass(frozen=True, slots=True)
class AttractorSpec:
    """Parameters for a sparse excitatory/inhibitory inferred-state assembly."""

    name: str = "INFERRED_STATE"
    excitatory_size: int = 48
    inhibitory_size: int = 12
    recurrent_fanout: float = 5
    inhibition_fanout: float = 6
    target_rate: float = 0.04


@dataclass(frozen=True, slots=True)
class AttractorHandle:
    """Ports of an inferred-state attractor.

    External circuits may drive ``input`` and read ``output``.  They are both
    the excitatory assembly: the parent composition decides every projection
    crossing the circuit boundary.
    """

    excitatory: str
    inhibitory: str

    @property
    def input(self) -> str:
        return self.excitatory

    @property
    def output(self) -> str:
        return self.excitatory


def add_attractor(
    builder: NetworkBuilder,
    spec: AttractorSpec,
) -> AttractorHandle:
    """Declare an inferred-state E/I attractor with explicit external ports."""
    if min(spec.excitatory_size, spec.inhibitory_size) <= 0 or not 0 <= spec.target_rate <= 1:
        raise ValueError("invalid attractor dimensions or target rate")
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
    recurrent_strength = StrengthSpec(0.12, 0.02, maximum=0.8)
    inhibitory_strength = StrengthSpec(0.22, 0.02, maximum=0.8)
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
