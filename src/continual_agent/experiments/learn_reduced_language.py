"""Train and inspect the reduced symbol curriculum."""

from continual_agent.agent.conversation_agent import AgentConfig, ConversationAgent
from continual_agent.language.curriculum import (
    reduced_language_curriculum,
    train_reduced_curriculum,
)


def main() -> None:
    alphabet = tuple("mab dn oi.?!")
    agent = ConversationAgent(AgentConfig(language_alphabet=alphabet))
    train_reduced_curriculum(agent, repetitions=30)
    for stage in reduced_language_curriculum():
        for example in stage.examples:
            result = agent.generate_response(agent.actions[0], max_tokens=len(example))
            print(f"{stage.name:18} {example!r} -> {result.text!r}")


if __name__ == "__main__":
    main()
