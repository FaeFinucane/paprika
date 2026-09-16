"""Read-only summaries of dopamine-modulated plasticity state."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..interaction.plasticity import DopamineSTDP


@dataclass(frozen=True)
class ProjectionEligibility:
    """Eligibility and current proposed update for one named projection."""

    projection: str
    synapse_count: int
    active_fraction: float
    mean_eligibility: float
    mean_absolute_eligibility: float
    mean_proposed_delta: float
    mean_absolute_proposed_delta: float


def inspect_dopamine_eligibility(
    rule: DopamineSTDP, projections: Sequence[str]
) -> tuple[ProjectionEligibility, ...]:
    """Summarise selected projections without changing network or rule state.

    The proposed delta uses the rule's current dopamine value and therefore
    answers the practical question: if this tick committed learning, which
    pathways would be changed, and by how much?
    """
    proposed = rule.propose(rule.snn).strength_delta
    if proposed is None:
        proposed = np.zeros_like(rule.eligibility)
    synapses = rule.snn.synapses
    summaries: list[ProjectionEligibility] = []
    for projection in projections:
        mask = synapses.projection_mask(projection) & rule.plastic
        eligibility = rule.eligibility[mask]
        delta = proposed[mask]
        if eligibility.size == 0:
            raise ValueError(f"projection {projection!r} has no dopamine-plastic synapses")
        summaries.append(
            ProjectionEligibility(
                projection=projection,
                synapse_count=int(eligibility.size),
                active_fraction=float(np.mean(np.abs(eligibility) > np.finfo(float).eps)),
                mean_eligibility=float(np.mean(eligibility)),
                mean_absolute_eligibility=float(np.mean(np.abs(eligibility))),
                mean_proposed_delta=float(np.mean(delta)),
                mean_absolute_proposed_delta=float(np.mean(np.abs(delta))),
            )
        )
    return tuple(summaries)
