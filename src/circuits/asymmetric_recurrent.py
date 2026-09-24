"""Sparse asymmetric recurrent E/I temporal circuit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..builder import NetworkBuilder
from ..interaction.drives import HomeostaticDriveSpec
from ..interaction.plasticity.homeostasis import SynapticScalingSpec
from ..network.connectivity import FanOutSpec, StrengthSpec
from ..network.population import Population


@dataclass(frozen=True)
class ForwardBiasedFanOutSpec:
    """Sparse recurrent fan-out through a cyclic, forward-biased ordering."""

    expected: int
    forward_span: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.expected, bool)
            or isinstance(self.forward_span, bool)
            or not isinstance(self.expected, (int, np.integer))
            or not isinstance(self.forward_span, (int, np.integer))
            or self.expected <= 0
            or self.forward_span <= 0
            or self.expected > self.forward_span
        ):
            raise ValueError("invalid forward-biased fan-out")

    def build_edges(
        self, source: Population[Any], target: Population[Any], rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        if source.spec.name != target.spec.name or source.count != target.count:
            raise ValueError("forward-biased fan-out requires one recurrent population")
        if self.forward_span * 2 >= source.count:
            raise ValueError("forward span must exclude reciprocal recurrent pairs")

        source_ids = np.arange(source.bounds.start, source.bounds.stop)
        offsets = np.arange(1, self.forward_span + 1)
        local_targets = np.concatenate(
            [rng.choice(offsets, self.expected, replace=False) for _ in source_ids]
        )
        src = np.repeat(source_ids, self.expected)
        dst = source.bounds.start + ((src - source.bounds.start + local_targets) % source.count)
        return src, dst


@dataclass(frozen=True)
class TargetWindowFanOutSpec:
    """Random fan-out restricted to a contiguous target-population window."""

    expected: float
    target_start: int
    target_count: int

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.expected)
            or self.expected < 0
            or self.target_start < 0
            or self.target_count <= 0
            or self.expected > self.target_count
        ):
            raise ValueError("invalid target-window fan-out")

    def build_edges(
        self, source: Population[Any], target: Population[Any], rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.target_start + self.target_count > target.count:
            raise ValueError("target window exceeds target population")
        source_ids = np.arange(source.bounds.start, source.bounds.stop)
        target_ids = np.arange(
            target.bounds.start + self.target_start,
            target.bounds.start + self.target_start + self.target_count,
        )
        src = np.repeat(source_ids, target_ids.size)
        dst = np.tile(target_ids, source_ids.size)
        keep = rng.random(src.size) < self.expected / target_ids.size
        if source.spec.name == target.spec.name:
            keep &= src != dst
        return src[keep], dst[keep]


@dataclass(frozen=True, slots=True)
class AsymmetricRecurrentSpec:
    """Parameters for a cue-evoked asymmetric recurrent E/I circuit."""

    name: str = "TEMPORAL"
    excitatory_size: int = 48
    inhibitory_size: int = 12
    recurrent_fanout: int = 4
    forward_span: int = 8
    excitatory_to_inhibitory_fanout: float = 8
    inhibitory_to_excitatory_fanout: float = 24
    recurrent_strength: float = 0.62
    inhibitory_strength: float = 0.02
    target_rate: float = 0.03


@dataclass(frozen=True, slots=True)
class AsymmetricRecurrentHandle:
    excitatory: str
    inhibitory: str

    @property
    def input(self) -> str:
        return self.excitatory

    @property
    def output(self) -> str:
        return self.excitatory


def add_asymmetric_recurrent_circuit(
    builder: NetworkBuilder,
    spec: AsymmetricRecurrentSpec,
) -> AsymmetricRecurrentHandle:
    """Declare a sparse asymmetric recurrent E/I temporal circuit."""
    if (
        min(spec.excitatory_size, spec.inhibitory_size) <= 0
        or not 0 < spec.recurrent_fanout <= spec.forward_span
        or not 0 < spec.forward_span * 2 < spec.excitatory_size
        or not 0 < spec.excitatory_to_inhibitory_fanout <= spec.inhibitory_size
        or not 0 < spec.inhibitory_to_excitatory_fanout <= spec.excitatory_size
        or min(
            spec.recurrent_strength,
            spec.inhibitory_strength,
        )
        < 0
        or not 0 <= spec.target_rate <= 1
    ):
        raise ValueError("invalid asymmetric recurrent circuit configuration")
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
    builder.connect(
        excitatory,
        excitatory,
        ForwardBiasedFanOutSpec(spec.recurrent_fanout, spec.forward_span),
        StrengthSpec(spec.recurrent_strength, 0.02, maximum=0.8),
        "dopamine_stdp",
        True,
        f"{excitatory}_recurrent",
    )
    builder.connect(
        excitatory,
        inhibitory,
        FanOutSpec(spec.excitatory_to_inhibitory_fanout),
        StrengthSpec(spec.recurrent_strength, 0.02, maximum=0.8),
        "dopamine_stdp",
        True,
        f"{excitatory}_to_{inhibitory}",
    )
    builder.connect(
        inhibitory,
        excitatory,
        FanOutSpec(spec.inhibitory_to_excitatory_fanout),
        StrengthSpec(spec.inhibitory_strength, 0.02, maximum=0.8),
        "inhibitory_homeostatic",
        False,
        f"{inhibitory}_to_{excitatory}",
    )
    return AsymmetricRecurrentHandle(excitatory, inhibitory)


__all__ = [
    "AsymmetricRecurrentHandle",
    "AsymmetricRecurrentSpec",
    "ForwardBiasedFanOutSpec",
    "TargetWindowFanOutSpec",
    "add_asymmetric_recurrent_circuit",
]
