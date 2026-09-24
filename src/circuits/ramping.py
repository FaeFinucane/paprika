"""Input-triggered E/I ramps with dopamine-trainable termination."""

from __future__ import annotations

from dataclasses import dataclass

from ..builder import NetworkBuilder
from ..interaction.drives import HomeostaticDriveSpec
from ..interaction.plasticity.homeostasis import SynapticScalingSpec
from ..network.connectivity import FanOutSpec, StrengthSpec


@dataclass(frozen=True, slots=True)
class RampingCircuitSpec:
    """Dimensions for an input-triggered recurrent E/I ramp.

    The circuit uses the simulator's shared neuron dynamics. Its temporal
    profile comes from recurrent excitation recruiting feedback inhibition,
    rather than from cell-specific time constants or an explicit tick chain.
    """

    name: str = "RAMP"
    excitatory_size: int = 32
    inhibitory_size: int = 12
    recurrent_fanout: float = 8
    inhibitory_fanout: float = 8
    feedback_fanout: float = 16
    target_rate: float = 0.03


@dataclass(frozen=True, slots=True)
class RampingCircuitHandle:
    """Named populations exposed by a recurrent ramp circuit."""

    excitatory: str
    inhibitory: str

    @property
    def input(self) -> str:
        return self.excitatory

    @property
    def output(self) -> str:
        return self.excitatory


def add_ramping_circuit(
    builder: NetworkBuilder,
    spec: RampingCircuitSpec,
) -> RampingCircuitHandle:
    """Declare a bounded E/I ramp whose termination is dopamine-trainable.

    Parent-provided input and recurrent excitation recruit ``RAMP_E``. Its causal
    projection to ``RAMP_INHIB`` is the circuit's sole dopamine-STDP pathway,
    while the inhibitory return projection ends the ramp. Therefore an early
    negative dopamine deviation weakens premature termination, whereas a
    positive deviation strengthens it. Keeping the sustaining scaffold fixed
    makes reward timing adjust termination rather than erase the input response.
    """
    if (
        min(spec.excitatory_size, spec.inhibitory_size) <= 0
        or not 0 < spec.recurrent_fanout < spec.excitatory_size
        or not 0 < spec.inhibitory_fanout <= spec.inhibitory_size
        or not 0 < spec.feedback_fanout <= spec.excitatory_size
        or not 0 <= spec.target_rate <= 1
    ):
        raise ValueError("invalid ramping-circuit configuration")

    excitatory = f"{spec.name}_E"
    inhibitory = f"{spec.name}_INHIB"
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

    recurrent_strength = StrengthSpec(0.36, 0.02, maximum=0.8)
    termination_strength = StrengthSpec(0.20, 0.02, maximum=0.8)
    feedback_strength = StrengthSpec(0.15, 0.02, maximum=0.8)
    builder.connect(
        excitatory,
        excitatory,
        FanOutSpec(spec.recurrent_fanout),
        recurrent_strength,
        "fixed",
        False,
        f"{excitatory}_recurrent",
    )
    builder.connect(
        excitatory,
        inhibitory,
        FanOutSpec(spec.inhibitory_fanout),
        termination_strength,
        "dopamine_stdp",
        True,
        f"{excitatory}_to_{inhibitory}",
    )
    builder.connect(
        inhibitory,
        excitatory,
        FanOutSpec(spec.feedback_fanout),
        feedback_strength,
        "inhibitory_homeostatic",
        False,
        f"{inhibitory}_to_{excitatory}",
    )
    return RampingCircuitHandle(excitatory, inhibitory)


__all__ = ["RampingCircuitHandle", "RampingCircuitSpec", "add_ramping_circuit"]
