from dataclasses import dataclass, field
from typing import Protocol

from ...network.snn import Spikes
from ..channels import NumericChannel
from ..plugins import Observer


class RewardSignal(Protocol):
    """Gives the 'reward prediction error' value"""

    value: float


@dataclass
class RPE(Observer):
    # reward_channel (R) is forced with ground truth; predictor_channel (V)
    # never is, and is trained purely via this same rpe broadcast.
    reward_channel: NumericChannel
    predictor_channel: NumericChannel

    # Discount of future prediction
    discount: float = 0.95

    value: float = field(init=False, default=0.0)
    prev_prediction: float = field(init=False, default=0.0)

    def observe(self, spikes: Spikes):
        prediction = self.predictor_channel.value
        self.value = self.reward_channel.value + self.discount * prediction - self.prev_prediction
        self.prev_prediction = prediction
