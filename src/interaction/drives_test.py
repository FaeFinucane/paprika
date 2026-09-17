"""Contract for homeostatic drive dynamics."""

import numpy as np
import pytest

from src.builder import NetworkBuilder
from src.interaction.drives import HomeostaticDrive
from src.network.snn import Spikes

pytestmark = pytest.mark.unit


def test_homeostatic_drive_moves_toward_its_target_and_respects_bounds():
    builder = NetworkBuilder()
    builder.add_population("POPULATION", 4)
    snn = builder.compile(1).snn
    population = snn.layout.population("POPULATION")
    homeostasis = HomeostaticDrive(
        population,
        target_rate=0.2,
        learning_rate=0.5,
        rate_decay=0.0,
        minimum_current=-0.1,
        maximum_current=0.1,
    )

    silent = Spikes(np.zeros(4, dtype=bool), tick=1, _fingerprint=snn.layout.fingerprint)
    assert np.allclose(homeostasis.produce().drives[population], 0.0)
    homeostasis.observe(silent)
    homeostasis.propose(snn)
    homeostasis.commit_state()
    assert homeostasis.current == 0.1

    active = Spikes(np.ones(4, dtype=bool), tick=2, _fingerprint=snn.layout.fingerprint)
    homeostasis.observe(active)
    homeostasis.propose(snn)
    homeostasis.commit_state()
    assert homeostasis.current == -0.1
