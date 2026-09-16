import numpy as np
import pytest
from src.diagnostics import inspect_dopamine_eligibility
from src.interaction.plasticity import DopamineSTDP
from src.network.connectivity import ConnectionSpec, FanOutSpec, StrengthSpec
from src.network.definition import NetworkDefinition
from src.network.population import NeuronPopulationSpec


class _Signal:
    value = 0.5


def test_eligibility_summary_reports_current_proposed_delta() -> None:
    definition = NetworkDefinition(
        (
            NeuronPopulationSpec("SOURCE", 1),
            NeuronPopulationSpec("TARGET", 1, dopamine_response="aligned"),
        ),
        (ConnectionSpec("SOURCE", "TARGET", FanOutSpec(1), StrengthSpec(0.2), "dopamine_stdp"),),
    )
    snn = definition.compile(np.random.default_rng(1))
    rule = DopamineSTDP(snn, _Signal(), learning_rate=0.1)
    rule.eligibility[:] = 0.4

    summary = inspect_dopamine_eligibility(rule, ("SOURCE_to_TARGET",))[0]

    assert summary.synapse_count == 1
    assert summary.active_fraction == 1.0
    assert summary.mean_eligibility == 0.4
    assert summary.mean_absolute_eligibility == 0.4
    assert summary.mean_proposed_delta == pytest.approx(0.02)
    assert summary.mean_absolute_proposed_delta == pytest.approx(0.02)
