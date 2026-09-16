"""Read-only inspection of compiled network state and activity."""

from .currents import incoming_projection_currents
from .health import NetworkHealth, inspect_network
from .plasticity import ProjectionEligibility, inspect_dopamine_eligibility
from .trace import NetworkTrace, TraceSample

__all__ = [
    "NetworkHealth",
    "NetworkTrace",
    "ProjectionEligibility",
    "TraceSample",
    "incoming_projection_currents",
    "inspect_dopamine_eligibility",
    "inspect_network",
]
