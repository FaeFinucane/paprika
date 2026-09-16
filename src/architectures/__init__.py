"""Reusable network architecture builders."""

from .components import AttractorSpec, TemporalSequenceSpec
from .dopamine import DopamineCircuit, build_dopamine_circuit

__all__ = [
    "AttractorSpec",
    "DopamineCircuit",
    "TemporalSequenceSpec",
    "build_dopamine_circuit",
]
