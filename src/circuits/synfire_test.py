"""Contract for the explicit synfire timing circuit."""

import pytest

from src.builder import NetworkBuilder
from src.network.connectivity import FanInSpec, StrengthSpec

from .attractor import AttractorSpec, add_attractor
from .synfire import SynfireChainHandle, SynfireChainSpec, add_synfire_chain

pytestmark = pytest.mark.unit


def test_synfire_chain_exposes_each_successive_stage():
    builder = NetworkBuilder()
    cue = builder.add_feature_population("CUE", ("CUE",), width=16)
    state = add_attractor(builder, AttractorSpec())
    builder.connect(
        cue,
        state.input,
        FanInSpec(6),
        StrengthSpec(0.25, 0.02, maximum=0.8),
        "dopamine_stdp",
        True,
    )
    chain = add_synfire_chain(builder, SynfireChainSpec(stages=3))
    builder.connect(
        state.output,
        chain.input,
        FanInSpec(4),
        StrengthSpec(0.30, 0.02, maximum=0.8),
        "fixed",
        False,
    )

    assert isinstance(chain, SynfireChainHandle)
    assert len(chain.stages) == 3
    entry = builder.connection(state.excitatory, chain.stages[0])
    assert entry.learning == "fixed"
    assert not entry.scalable
    for source, target in zip(chain.stages[:-1], chain.stages[1:], strict=True):
        projection = builder.connection(source, target)
        assert projection.projection_name == f"{source}_to_{target}"
        assert projection.learning == "fixed"
        assert not projection.scalable
