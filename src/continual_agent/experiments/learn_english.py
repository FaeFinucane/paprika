"""Replay a tiny English corpus through the online character learner."""

from continual_agent.language.decoder import CharacterDecoder


CORPUS = (
    "hello there",
    "how are you",
    "i am learning",
    "the cat is here",
    "the dog is here",
    "thank you",
    "i do not know",
)


def main() -> None:
    decoder = CharacterDecoder(context_size=3, seed=1)
    for _ in range(100):
        for sentence in CORPUS:
            decoder.observe(sentence)

    print(f"held-out loss: {decoder.loss('hello there'):.3f}")
    for prompt in ("hello ", "the ", "i ", "thank "):
        print(f"{prompt!r} -> {decoder.generate(prompt, max_tokens=24).text!r}")


if __name__ == "__main__":
    main()
