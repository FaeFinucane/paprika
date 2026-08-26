"""Spiking character-level language output."""

from .spiking_decoder import GeneratedResponse, SpikingCharacterDecoder
from .curriculum import SequenceStage, reduced_language_curriculum, train_reduced_curriculum

__all__ = [
    "GeneratedResponse",
    "SequenceStage",
    "SpikingCharacterDecoder",
    "reduced_language_curriculum",
    "train_reduced_curriculum",
]
