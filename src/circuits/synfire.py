"""Explicit E/I temporal-basis chains for bounded interval timing."""

from __future__ import annotations

from dataclasses import dataclass

from ..builder import NetworkBuilder
from ..interaction.drives import HomeostaticDriveSpec
from ..interaction.plasticity.homeostasis import SynapticScalingSpec
from ..network.connectivity import FanInSpec, FanOutSpec, StrengthSpec


@dataclass(frozen=True, slots=True)
class SynfireChainSpec:
    """A structural feed-forward temporal basis with local E/I regulation."""

    name: str = "TEMPORAL"
    stages: int = 8
    stage_size: int = 12
    inhibitory_size: int = 6
    forward_fanin: float = 4
    recurrent_fanout: float = 3
    target_rate: float = 0.03


@dataclass(frozen=True, slots=True)
class SynfireChainHandle:
    """Explicit input and output ports of a temporal-basis chain."""

    stages: tuple[str, ...]
    inhibitory: tuple[str, ...]

    @property
    def input(self) -> str:
        return self.stages[0]

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.stages


def add_synfire_chain(builder: NetworkBuilder, spec: SynfireChainSpec) -> SynfireChainHandle:
    """Declare a temporal basis with local E/I regulation and explicit ports.

    Forward links form a stable causal scaffold.  The parent composition
    chooses what starts the chain and where individual stage outputs project;
    local E/I regulation remains available to ordinary dopamine-STDP.
    """
    if (
        spec.stages <= 0
        or min(spec.stage_size, spec.inhibitory_size) <= 0
        or not 0 < spec.forward_fanin <= spec.stage_size
        or not 0 < spec.recurrent_fanout <= min(spec.stage_size - 1, spec.inhibitory_size)
        or not 0 <= spec.target_rate <= 1
    ):
        raise ValueError("invalid synfire-chain configuration")

    stages = tuple(f"{spec.name}_{index}" for index in range(spec.stages))
    inhibitory = tuple(f"{spec.name}_INHIB_{index}" for index in range(spec.stages))
    learned = StrengthSpec(0.12, 0.02, maximum=0.8)
    forward_drive = StrengthSpec(0.50, 0.02, maximum=0.8)
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
    for source, target in zip(stages[:-1], stages[1:], strict=True):
        builder.connect(
            source,
            target,
            FanInSpec(spec.forward_fanin),
            forward_drive,
            "fixed",
            False,
            f"{source}_to_{target}",
        )
    return SynfireChainHandle(stages, inhibitory)


__all__ = ["SynfireChainHandle", "SynfireChainSpec", "add_synfire_chain"]
