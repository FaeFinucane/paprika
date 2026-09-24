"""Specialised, reusable neural-circuit assemblies."""

from .asymmetric_recurrent import AsymmetricRecurrentSpec
from .attractor import AttractorSpec
from .ramping import RampingCircuitSpec
from .synfire import SynfireChainSpec
from .vta_dopamine import VtaDopamineSpec, add_vta_dopamine

__all__ = [
    "AsymmetricRecurrentSpec",
    "AttractorSpec",
    "RampingCircuitSpec",
    "SynfireChainSpec",
    "VtaDopamineSpec",
    "add_vta_dopamine",
]
