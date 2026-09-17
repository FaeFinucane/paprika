"""Specialised, reusable neural-circuit assemblies."""

from .asymmetric_recurrent import AsymmetricRecurrentSpec
from .attractor import AttractorSpec
from .vta_dopamine import DopamineCircuit, build_dopamine_circuit

__all__ = [
    "AsymmetricRecurrentSpec",
    "AttractorSpec",
    "DopamineCircuit",
    "build_dopamine_circuit",
]
