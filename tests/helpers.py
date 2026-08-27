from typing import Any

from continual_agent.agent.network_config import NetworkConfig
from continual_agent.simulation.population_layout import PopulationLayout


def network_config(
    input_features: int = 48,
    hidden_neurons: int = 48,
    output_tokens: tuple[str, ...] | None = None,
    neurons_per_token: int = 3,
    neurons_per_action: int = 4,
    neurons_per_affect: int = 4,
    language_alphabet: tuple[str, ...] | None = None,
    **settings: Any,
) -> NetworkConfig:
    if language_alphabet is not None:
        output_tokens = ("<EOS>",) + language_alphabet
    if output_tokens is None:
        output_tokens = ("<EOS>", "m", "a", "b", " ", "d", "n", "o", "i", ".", "?", "!")
    layout = PopulationLayout.from_dimensions(
        input_count=input_features,
        hidden_count=hidden_neurons,
        output_tokens=output_tokens,
        neurons_per_token=neurons_per_token,
        neurons_per_action=neurons_per_action,
        neurons_per_affect=neurons_per_affect,
    )
    return NetworkConfig(layout=layout, **settings)
