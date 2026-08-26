from continual_agent.language.decoder import CharacterDecoder


def test_decoder_learns_next_character_transitions() -> None:
    decoder = CharacterDecoder(context_size=2, seed=2)
    before = decoder.loss("hello")
    for _ in range(20):
        decoder.observe("hello")
    after = decoder.loss("hello")

    assert after < before
    assert decoder.generate("he", max_tokens=4).text.startswith("llo")


def test_decoder_stops_with_eos_after_observed_phrase() -> None:
    decoder = CharacterDecoder(context_size=2)
    for _ in range(20):
        decoder.observe("ok")

    result = decoder.generate("ok", max_tokens=4)

    assert result.stopped_on_eos
    assert result.text == ""
