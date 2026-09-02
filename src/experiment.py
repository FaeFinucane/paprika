"""Small, reproducible reward-modulated copying experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .conversation import Conversation, OutputEvent
from .evaluation import EvaluationReport, Evaluator
from .interaction.stdp import STDP
from .interaction.channels import FeatureInChannel, FeatureOutChannel, PopulationDecoder, PopulationEncoder
from .network.population import FeaturePopulationSpec, NeuronPopulationSpec, Population, PopulationLayout
from .network.connectivity import BernoulliTopologySpec, BimodalWeightSpec, ConnectionSpec, Connectivity
from .network.snn import SNN
from .reward import RewardPolicy

@dataclass
class MinimalExperiment:
    snn: SNN
    input_channel: FeatureInChannel
    output_channel: FeatureOutChannel
    learning: STDP
    evaluator: Evaluator
    reward_policy: RewardPolicy
    rng: np.random.Generator
    trials: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])

    def run_trial(self, symbol: str, *, training: bool = True, trial_id: str | None = None):
        if symbol not in {"A", "B"}:
            raise ValueError("symbol must be A or B")
        # The minimal task is a sequence of independent turns.  Keep learned
        # weights, but clear transient neural state so a prior timeout cannot
        # decide the next actor response.
        # self.snn.neurons.voltage.fill(0)
        # self.snn.neurons.refractory.fill(0)
        # self.snn.pending_current.fill(0)
        identity = trial_id or f"trial-{len(self.trials)}"
        conversation = Conversation()
        turn = conversation.begin(self.snn.tick)
        conversation.input(symbol)

        self.input_channel.write(symbol)

        traces_before = self.learning.eligibility.copy()

        # TODO: Merge drives together

        for _ in range(2):
            spikes = self.snn.step()
            if training:
                self.learning.observe(spikes)


        observations: list[str] = []
        emitted: list[str] = []
        # Why 4?
        for _ in range(4):
            spikes = self.snn.step()
            if training:
                self.learning.observe(spikes)
            self.output_channel.observe(spikes)

            if self.output_channel.has_feature_changed and self.output_channel.current:
                observations.append(self.output_channel.current.feature)

            if not emitted:
                # EOS without a payload is a malformed response.  Keep it out
                # of the first actor arbitration, rather than turning a tie in
                # the detector's feature names into a hidden policy.
                active = observation.active and observation.feature in {"A", "B"}
            else:
                active = observation.active and observation.feature == "EOS"
            if active:
                winner = observation
                emitted.append(winner.feature)
                conversation.output(OutputEvent(winner.feature, winner.timestamp, winner.evidence))
                if winner.feature == "EOS":
                    break

        status = "eos" if emitted and emitted[-1] == "EOS" else "timeout"
        turn = conversation.complete(status, self.snn.tick)
        expected = (symbol, "EOS")
        report: EvaluationReport = self.evaluator.evaluate(turn, expected, identity)
        reward = self.reward_policy.reward(report)
        next_value = 0.0

        if training:
            rpe = 1.0 # TODO:
            self.learning.update(self.snn, rpe)
        else:
            rpe = 0.0

        diagnostics: dict[str, Any] = {
            "identity": identity,
            "input": symbol,
            "output": tuple(emitted),
            "status": status,
            "valid": report.score == 1.0,
            "accuracy": report.score,
            "eos": status == "eos",
            "response_length": len(emitted),
            "reward": reward,
            "current_value": current_value,
            "next_value": next_value,
            "rpe": rpe,
            # TODO: Not sure if this is worth reporting on
            "eligibility_before": (float(traces_before.min()), float(traces_before.max())),
            "eligibility_after": (
                float(self.learning.eligibility.min()),
                float(self.learning.eligibility.max()),
            ),
            "weight_distribution": (
                float(self.snn.synapses.weight.min()),
                float(self.snn.synapses.weight.max()),
            ),
        }
        self.trials.append(diagnostics)
        return diagnostics


def build_minimal_experiment(seed: int = 0) -> MinimalExperiment:
    layout = PopulationLayout.build(
        [
            FeaturePopulationSpec("INPUT_CHARS", ("A", "B", "?"), 4),
            FeaturePopulationSpec("OUTPUT_CHARS", ("A", "B", "EOS"), 4),
            NeuronPopulationSpec("HIDDEN", 64),
            NeuronPopulationSpec("CRITIC", 16),
            NeuronPopulationSpec("REWARD", 16),
        ]
    )
    specs: list[ConnectionSpec] = []
    for source, target in (
        ("INPUT_CHARS", "HIDDEN"),
        ("HIDDEN", "OUTPUT_CHARS"),
        ("HIDDEN", "CRITIC"),
        ("HIDDEN", "REWARD"),
        ("OUTPUT_CHARS", "HIDDEN"),
    ):
        target_count = layout.population(target).count
        fan_out = min(16 if source == "INPUT_CHARS" else 8, target_count)
        specs.append(
            ConnectionSpec(
                source,
                target,
                BernoulliTopologySpec(fan_out),
                BimodalWeightSpec(0.6, -0.1, 0.15, 0.04, 0.02),
            )
        )
    connectivity = Connectivity.build(layout, specs, seed)
    snn = SNN.build(layout, connectivity)
    input_pop = layout.population("INPUT_CHARS")
    output_pop = layout.population("OUTPUT_CHARS")
    # TODO: Ideally we can extract these with better typing later.
    return MinimalExperiment(
        snn,
        FeatureInChannel(input_pop, PopulationEncoder(4.0)),
        FeatureOutChannel(output_pop, PopulationDecoder(3.5)),
        STDP(snn),
        Evaluator(),
        RewardPolicy(),
        np.random.default_rng(seed),
    )


def run_trial(
    experiment: MinimalExperiment,
    symbol: str,
    *,
    training: bool = True,
    trial_id: str | None = None,
):
    return experiment.run_trial(symbol, training=training, trial_id=trial_id)


def train(experiment: MinimalExperiment, trials: int = 200):
    if trials < 1:
        raise ValueError("trials must be positive")
    for index in range(trials):
        run_trial(
            experiment, "A" if index % 2 == 0 else "B", training=True, trial_id=f"train-{index}"
        )
    return experiment


if __name__ == "__main__":
    baseline = build_minimal_experiment(0)
    before = [
        run_trial(baseline, symbol, training=False, trial_id=f"baseline-{symbol}")
        for symbol in "AB"
    ]
    trained = train(build_minimal_experiment(0), 200)
    after = trained.trials[-20:]
    before_score = sum(item["valid"] for item in before) / len(before)
    after_score = sum(item["valid"] for item in after) / len(after)
    print(f"baseline={before_score:.2f} trained_tail={after_score:.2f}")
