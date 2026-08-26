"""Configuration for the conversational task adapter."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    input_features: int = 48
    neurons_per_action: int = 4
    neurons_per_affect: int = 4
    hidden_neurons: int = 48
    max_thinking_ticks: int = 18
    connection_probability: float = 0.08
    seed: int = 0
    persistent_working_memory: bool = True
    neurons_per_token: int = 3
    max_response_tokens: int = 48
    max_response_ticks: int = 144
    presentation_speed: int = 1
    language_alphabet: tuple[str, ...] = ("m", "a", "b", " ", "d", "n", "o", "i", ".", "?", "!")
