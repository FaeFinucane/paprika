from dataclasses import dataclass, field
from typing import Protocol

from ..channels import NumericChannel


class RewardSignal(Protocol):
    """Anything that can report a reward-prediction-error each tick - RPE
    reads it from the neural REWARD population; other implementations (e.g.
    an external/ground-truth stand-in) may compute it differently."""

    def calculate_rpe(self) -> float: ...


@dataclass
class RPE:
    # reward_channel (R) is forced with ground truth; predictor_channel (V)
    # never is, and is trained purely via this same rpe broadcast.
    reward_channel: NumericChannel
    predictor_channel: NumericChannel

    # Discounts the bootstrap target (V(t+1)), not the value being corrected
    # - damps the self-referential feedback loop that comes from V training
    # on its own predictions, on top of the usual short-horizon discounting.
    discount: float = 0.95

    prev_prediction: float = field(init=False, default=0.0)

    def calculate_rpe(self) -> float:
        value = self.predictor_channel.value
        rpe = self.reward_channel.value + self.discount * value - self.prev_prediction
        self.prev_prediction = value
        return rpe
