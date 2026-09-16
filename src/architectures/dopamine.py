"""Reference-rate implementation of the dopamine TD circuit.

The temporal comparator is deliberately isolated behind ``transition``.  Its
previous-value state is the reference oracle; the rest of the circuit is an
ordinary rate-coded SNN and can be retained when that oracle is replaced by a
transition-gated neural memory circuit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable

import numpy as np

from ..interaction.drives import HomeostaticDrive, TonicDrive
from ..interaction.plasticity import DopamineSTDP, InhibitoryHomeostasis
from ..interaction.plasticity.homeostasis import SynapticScaling
from ..interaction.plugins import Drives, DriveSource, Hook
from ..interaction.rates import DopamineReadout, PopulationRate, RatePatternInput, UnipolarRateInput
from ..interaction.td import TDComparison, TemporalDifferenceComparator
from ..network.connectivity import (
    ConnectionSpec,
    FanInSpec,
    FanOutSpec,
    StrengthSpec,
)
from ..network.definition import NetworkDefinition
from ..network.population import (
    FeaturePopulationSpec,
    NeuronPopulationSpec,
    Population,
)
from ..session import Session


@dataclass
class TDPulse(DriveSource):
    """Reference implementation of only the previous-value TD term."""

    population: Population[Any]
    gain: float = 1.0
    duration: int = 10
    _value: float = 0.0
    _remaining: int = 0

    def emit(self, value: float, duration: int | None = None) -> None:
        if not np.isfinite(value):
            raise ValueError("TD value must be finite")
        ticks = self.duration if duration is None else duration
        if ticks <= 0:
            raise ValueError("duration must be positive")
        self._value = float(np.clip(value, -1.0, 1.0))
        self._remaining = ticks

    def produce(self) -> Drives:
        if self._remaining <= 0:
            return Drives()
        self._remaining -= 1
        # A signed external current is appropriate only at the explicit
        # reference-comparator boundary. Ordinary representation stays
        # unipolar; this is replaced by neural temporal memory in Phase 7.
        return Drives({self.population: np.full(self.population.count, self.gain * self._value)})


@dataclass
class DopamineCircuit:
    session: Session
    populations: dict[str, Population[Any]]
    reward_positive: UnipolarRateInput
    reward_negative: UnipolarRateInput
    state_input: RatePatternInput
    value_positive_rate: PopulationRate
    value_negative_rate: PopulationRate
    dopamine_rate: PopulationRate
    dopamine: DopamineReadout
    stdp: DopamineSTDP
    comparator: TemporalDifferenceComparator
    td_pulse: TDPulse
    gamma: float
    reward_rate_scale: float

    def deliver_reward(self, value: float, duration: int | None = None) -> None:
        """Route signed environment reward into exactly one unipolar input."""
        if not np.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError("reward must be finite and in [-1, 1]")
        duration = self.td_pulse.duration if duration is None else duration
        encoded_rate = abs(value) * self.reward_rate_scale
        if value >= 0:
            self.reward_positive.write(encoded_rate, duration)
        else:
            self.reward_negative.write(encoded_rate, duration)

    def deliver_state(self, pattern: np.ndarray, duration: int | None = None) -> None:
        """Drive an arbitrary unipolar rate pattern over ``STATE``."""
        self.state_input.write(pattern, duration)

    def transition(
        self,
        reward: float,
        *,
        transition_id: Hashable | None = None,
        duration: int | None = None,
    ) -> TDComparison:
        """Evaluate exactly one semantic state transition and pulse dopamine."""
        if not np.isfinite(reward) or not -1.0 <= reward <= 1.0:
            raise ValueError("reward must be finite and in [-1, 1]")
        previous_term = -self.comparator.previous_positive + self.comparator.previous_negative
        comparison = self.comparator.evaluate_transition(
            max(reward, 0.0),
            max(-reward, 0.0),
            self.value_positive_rate.rate,
            self.value_negative_rate.rate,
            self.gamma,
            transition_id=transition_id,
        )
        self.deliver_reward(reward, duration)
        # Reward and current value are carried by fixed neural paths. The
        # reference pulse supplies only -V_positive(previous) +
        # V_negative(previous), until the transition-gated neural memory
        # circuit has demonstrated equivalent behavior.
        self.td_pulse.emit(previous_term, duration)
        return comparison

    def tick(self):
        return self.session.tick()

    def reset_episode(self) -> None:
        self.comparator.reset()
        self.value_positive_rate.reset()
        self.value_negative_rate.reset()
        self.dopamine.reset()
        self.stdp.reset()


def build_dopamine_circuit(
    seed: int = 0,
    *,
    gamma: float = 0.95,
    hidden_size: int = 48,
    value_size: int = 16,
    dopamine_size: int = 32,
    value_signal_span: float = 0.17,
    enable_dopamine_learning: bool = True,
) -> DopamineCircuit:
    """Build the smallest complete clean-break dopamine architecture."""
    if (
        not 0 <= gamma <= 1
        or min(hidden_size, dopamine_size) <= 0
        or value_size < 3
        or not 0.0 < value_signal_span <= 1.0
    ):
        raise ValueError("invalid circuit dimensions or discount")
    hidden_inhibitory = max(4, hidden_size // 4)
    population_specs: list[NeuronPopulationSpec | FeaturePopulationSpec] = [
        FeaturePopulationSpec("STATE", ("STATE",), 16),
        NeuronPopulationSpec("HIDDEN_ALIGNED", hidden_size, dopamine_response="aligned"),
        NeuronPopulationSpec("HIDDEN_OPPOSED", hidden_size, dopamine_response="opposed"),
        NeuronPopulationSpec("HIDDEN_NEUTRAL", hidden_size),
        NeuronPopulationSpec("HIDDEN_INHIBITORY", hidden_inhibitory, output="inhibitory"),
        NeuronPopulationSpec("POSITIVE_VALUE", value_size, dopamine_response="aligned"),
        NeuronPopulationSpec(
            "NEGATIVE_VALUE", value_size, output="inhibitory", dopamine_response="opposed"
        ),
        NeuronPopulationSpec("REWARD_POSITIVE", value_size),
        NeuronPopulationSpec("REWARD_NEGATIVE", value_size, output="inhibitory"),
        NeuronPopulationSpec("DOPAMINE", dopamine_size, output="modulatory"),
    ]
    definition = NetworkDefinition(tuple(population_specs), ())
    # One provisional sparse rate convention is shared by value and reward.
    # The reward population mirrors value_size and maps external reward into
    # this signal span rather than treating a scalar reward as a raw rate.
    value_baseline_rate = 0.03
    # Preserve the former full-scale dopamine-current budget while encoding a
    # reward in the same sparse span as value. The resulting dense sampling is
    # deliberately measured by the comparator viability test below.
    comparator_current_budget = 6 * 0.32
    comparator_fanin = value_size - 2
    comparator_strength = comparator_current_budget / (comparator_fanin * value_signal_span)

    state_to_hidden = StrengthSpec(0.25, 0.02, maximum=0.8)
    learned = StrengthSpec(0.12, 0.02, maximum=0.8)
    fixed = StrengthSpec(0.30, 0.02, maximum=0.8)
    comparator = StrengthSpec(comparator_strength, 0.02, maximum=0.8)
    discounted = StrengthSpec(comparator_strength * gamma, 0.02, maximum=0.8)
    inhibitory = StrengthSpec(0.22, 0.02, maximum=0.8)
    specs: list[ConnectionSpec] = []
    hidden_exc = ("HIDDEN_ALIGNED", "HIDDEN_OPPOSED", "HIDDEN_NEUTRAL")
    for target in hidden_exc:
        specs.append(
            ConnectionSpec(
                "STATE",
                target,
                FanOutSpec(6, target_neurons=12),
                state_to_hidden,
                "dopamine_stdp",
                True,
            )
        )
    for source in hidden_exc:
        for target in hidden_exc:
            specs.append(
                ConnectionSpec(source, target, FanOutSpec(5), learned, "dopamine_stdp", True)
            )
        specs.append(ConnectionSpec(source, "HIDDEN_INHIBITORY", FanOutSpec(5), fixed))
        specs.append(
            ConnectionSpec(source, "POSITIVE_VALUE", FanOutSpec(4), learned, "dopamine_stdp", True)
        )
        specs.append(
            ConnectionSpec(source, "NEGATIVE_VALUE", FanOutSpec(4), learned, "dopamine_stdp", True)
        )
    for target in hidden_exc:
        specs.append(
            ConnectionSpec(
                "HIDDEN_INHIBITORY",
                target,
                FanOutSpec(6),
                inhibitory,
                "inhibitory_homeostatic",
            )
        )
    # Fixed comparator paths use source transmitter sign directly.
    specs.extend(
        (
            ConnectionSpec("REWARD_POSITIVE", "DOPAMINE", FanInSpec(comparator_fanin), comparator),
            ConnectionSpec("REWARD_NEGATIVE", "DOPAMINE", FanInSpec(comparator_fanin), comparator),
            ConnectionSpec("POSITIVE_VALUE", "DOPAMINE", FanInSpec(comparator_fanin), discounted),
            ConnectionSpec("NEGATIVE_VALUE", "DOPAMINE", FanInSpec(comparator_fanin), discounted),
        )
    )
    topology_rng = np.random.default_rng(seed)
    state_rng = np.random.default_rng([seed, 1])
    reward_positive_rng = np.random.default_rng([seed, 2])
    reward_negative_rng = np.random.default_rng([seed, 3])
    tonic_rng = np.random.default_rng([seed, 4])
    definition = NetworkDefinition(definition.populations, tuple(specs))
    snn = definition.compile(topology_rng)
    populations = {population.spec.name: population for population in snn.layout.populations}
    state_input = RatePatternInput(populations["STATE"], state_rng)
    reward_positive = UnipolarRateInput(populations["REWARD_POSITIVE"], reward_positive_rng)
    reward_negative = UnipolarRateInput(populations["REWARD_NEGATIVE"], reward_negative_rng)
    positive_rate = PopulationRate(populations["POSITIVE_VALUE"], decay=0.9)
    negative_rate = PopulationRate(populations["NEGATIVE_VALUE"], decay=0.9)
    dopamine_rate = PopulationRate(populations["DOPAMINE"], decay=0.9)
    dopamine = DopamineReadout(dopamine_rate, baseline=0.25, deadzone=0.04)
    td_pulse = TDPulse(populations["DOPAMINE"], gain=comparator_fanin * comparator_strength)
    # DopamineReadout already rejects tonic baseline variation. Applying a
    # second STDP deadzone would discard the small phasic signal that remains.
    stdp = DopamineSTDP(snn, dopamine, learning_rate=0.002)
    components: list[Hook] = [
        state_input,
        reward_positive,
        reward_negative,
        TonicDrive(populations["DOPAMINE"], current=0.42, heterogeneity=0.06, rng=tonic_rng),
        td_pulse,
        positive_rate,
        negative_rate,
        dopamine,
    ]
    if enable_dopamine_learning:
        components.append(stdp)
    # Slow regulators are explicit current sources. They do not alter neuron
    # parameters or hide a source of activity in the network runtime.
    homeostasis = [
        HomeostaticDrive(populations["HIDDEN_ALIGNED"], 0.04),
        HomeostaticDrive(populations["HIDDEN_OPPOSED"], 0.04),
        HomeostaticDrive(populations["HIDDEN_NEUTRAL"], 0.04),
        HomeostaticDrive(populations["POSITIVE_VALUE"], value_baseline_rate),
        HomeostaticDrive(populations["NEGATIVE_VALUE"], value_baseline_rate),
        HomeostaticDrive(populations["DOPAMINE"], 0.25, learning_rate=0.00001, initial_current=0.0),
        SynapticScaling(snn, populations["HIDDEN_ALIGNED"], 0.04),
        SynapticScaling(snn, populations["HIDDEN_OPPOSED"], 0.04),
        SynapticScaling(snn, populations["HIDDEN_NEUTRAL"], 0.04),
        InhibitoryHomeostasis(snn, target_rate=0.04),
    ]
    components.extend(homeostasis)
    session = Session.build(snn, components)
    return DopamineCircuit(
        session,
        populations,
        reward_positive,
        reward_negative,
        state_input,
        positive_rate,
        negative_rate,
        dopamine_rate,
        dopamine,
        stdp,
        TemporalDifferenceComparator(),
        td_pulse,
        gamma,
        value_signal_span,
    )
