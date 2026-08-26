import continual_agent
import continual_agent.simulation as simulation


def test_network_core_is_the_canonical_public_network_type() -> None:
    assert continual_agent.__all__ == ["NetworkCore"]
    assert simulation.__all__ == [
        "LIFNeurons",
        "NetworkCore",
        "SparseSynapses",
        "SynapticActivityState",
    ]
    assert not hasattr(continual_agent, "SpikingNetwork")
    assert not hasattr(simulation, "SpikingNetwork")
