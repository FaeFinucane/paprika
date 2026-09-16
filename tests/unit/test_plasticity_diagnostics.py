import pytest
from src.builder import NetworkBuilder
from src.diagnostics import inspect_dopamine_eligibility
from src.interaction.plasticity import DopamineSTDP
from src.network.connectivity import FanOutSpec, StrengthSpec


class _Signal:
    value = 0.5


def test_eligibility_summary_reports_current_proposed_delta() -> None:
    builder = NetworkBuilder()
    builder.add_population("SOURCE", 1)
    builder.add_population("TARGET", 1, dopamine_response="aligned")
    builder.connect("SOURCE", "TARGET", FanOutSpec(1), StrengthSpec(0.2), "dopamine_stdp")
    snn = builder.compile(1).snn
    rule = DopamineSTDP(snn, _Signal(), learning_rate=0.1)
    rule.eligibility[:] = 0.4

    summary = inspect_dopamine_eligibility(rule, ("SOURCE_to_TARGET",))[0]

    assert summary.synapse_count == 1
    assert summary.active_fraction == 1.0
    assert summary.mean_eligibility == 0.4
    assert summary.mean_absolute_eligibility == 0.4
    assert summary.mean_proposed_delta == pytest.approx(0.02)
    assert summary.mean_absolute_proposed_delta == pytest.approx(0.02)
