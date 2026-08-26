"""Small recurrent character-level language output."""

from .decoder import CharacterDecoder
from .spiking_decoder import GeneratedResponse, SpikingCharacterDecoder
from .curriculum import SequenceStage, reduced_language_curriculum, train_reduced_curriculum

__all__ = [
    "CharacterDecoder",
    "GeneratedResponse",
    "SequenceStage",
    "SpikingCharacterDecoder",
    "reduced_language_curriculum",
    "train_reduced_curriculum",
]
