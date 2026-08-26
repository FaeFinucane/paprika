import json

import numpy as np

from continual_agent.agent.conversation_agent import ConversationAgent
from continual_agent.cognition.affect import AffectiveEvent, AffectiveState
from continual_agent.cognition.affect_circuit import AffectiveCircuit
from continual_agent.environment.scenarios import default_scenarios
from continual_agent.simulation.population_layout import Population, PopulationLayout
from continual_agent.simulation.synapses import SparseSynapses


def test_success_and_threat_move_affect_in_expected_directions() -> None:
    state = AffectiveState()
    initial = state.as_dict()

    state.observe(AffectiveEvent(reward_prediction_error=1.0))
    assert state.valence > initial["valence"]
    assert state.competence > initial["competence"]

    state.observe(AffectiveEvent(threat=1.0, urgency=1.0))
    assert state.threat > 0.1
    assert state.arousal > 0.2


def test_affect_stays_bounded_and_returns_toward_baseline() -> None:
    state = AffectiveState()
    for _ in range(100):
        state.observe(
            AffectiveEvent(
                reward_prediction_error=1.0,
                novelty=1.0,
                learning_progress=1.0,
                threat=1.0,
                urgency=1.0,
            )
        )
    assert -1.0 <= state.valence <= 1.0
    assert all(0.0 <= value <= 1.0 for value in state.as_dict().values())

    state.advance(200)
    assert state.threat < 0.2
    assert state.arousal < 0.3


def test_agent_exposes_latest_debug_snapshot() -> None:
    agent = ConversationAgent()
    agent.respond(default_scenarios()[0].messages[0])
    snapshot = agent.debug_snapshot()

    assert snapshot is not None
    assert snapshot["act"] in {action.value for action in agent.actions}
    assert "affect" in snapshot
    assert "control" in snapshot
    assert isinstance(snapshot["affect"], dict)
    assert "uncertainty" in snapshot["affect"]
    assert "working_memory" in snapshot
    json.dumps(snapshot)
    assert "neural_affect" in snapshot


def test_affective_spiking_circuit_aligns_to_targets() -> None:
    circuit = AffectiveCircuit(neurons_per_signal=4)
    input_count = 8
    affect_names = circuit.signal_names
    affect_start = input_count + 48
    layout = PopulationLayout(
        input_count=input_count,
        affect_count=circuit.neuron_count,
        affect_subgroups={
            name: slice(affect_start + index * 4, affect_start + (index + 1) * 4)
            for index, name in enumerate(affect_names)
        },
    )
    affect = layout.slice(Population.AFFECT)
    affect_count = circuit.neuron_count
    source = np.repeat(np.arange(input_count), affect_count)
    target_indices = np.tile(np.arange(affect.start, affect.stop), input_count)
    synapses = SparseSynapses(
        source=source,
        target=target_indices,
        weight=np.zeros(source.size),
        neuron_count=layout.total_count,
    )
    edge_indices = circuit.projection_indices(layout)
    features = np.zeros(8)
    features[2] = 1.0
    before = circuit.projection_prediction(synapses, edge_indices, features)
    target_state = AffectiveState(threat=1.0, arousal=1.0)

    for _ in range(80):
        circuit.align(synapses, edge_indices, features, target_state)

    after = circuit.projection_prediction(synapses, edge_indices, features)
    assert after["threat"] > before["threat"]
    assert after["arousal"] > before["arousal"]
    frame = np.zeros(layout.total_count, dtype=bool)
    frame[affect] = True
    assert set(circuit.decode([frame], layout)) == {
        "valence",
        "arousal",
        "uncertainty",
        "curiosity",
        "threat",
        "competence",
        "social_affiliation",
    }
