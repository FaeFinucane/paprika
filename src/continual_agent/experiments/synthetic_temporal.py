"""Fresh-agent paired experiments for structural and reward-modulated learning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np

from continual_agent.agent.session import InputSignal
from continual_agent.agent.spiking_runtime import SpikingRuntime
from continual_agent.cognition.readout import OutputEvent
from continual_agent.evaluation.event_stream import (
    EventStreamConfig,
    EventStreamReport,
    evaluate_event_stream,
)
from continual_agent.simulation.population_layout import Population


class CopyCondition(str, Enum):
    IMMEDIATE = "immediate-copy"
    DELAYED = "delayed-copy"


class AgentKind(str, Enum):
    SUPERVISED = "supervised-structural"
    REWARD_MODULATED_STDP = "reward-modulated-stdp"


class TrainingMode(str, Enum):
    UNTRAINED = "untrained"
    NO_LEARNING = "no-learning"
    SUPERVISED = "supervised"
    REWARD_MODULATED_STDP = "reward-modulated-stdp"


class Control(str, Enum):
    TRAINED = "trained"
    UNTRAINED = "untrained"
    NO_LEARNING = "no-learning"
    SHUFFLED_TARGET = "shuffled-target"
    RECURRENT_ABLATION = "recurrent-ablation"
    DIRECT_INPUT_OUTPUT_ABLATION = "direct-input-to-output-ablation"


@dataclass(frozen=True)
class TemporalExperimentConfig:
    alphabet: tuple[str, ...] = ("A", "B")
    sequences: tuple[tuple[str, ...], ...] = (("A",), ("B",))
    evaluation_sequences: tuple[tuple[str, ...], ...] | None = None
    training_trials: int = 12
    ticks_per_frame: int = 3
    patience_window: int = 24
    response_ticks: int = 64
    hidden_neurons: int = 12
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.alphabet or len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("alphabet must contain distinct symbols")
        all_sequences = (*self.sequences, *(self.evaluation_sequences or ()))
        if any(symbol not in self.alphabet for sequence in all_sequences for symbol in sequence):
            raise ValueError("sequences must use the configured alphabet")
        if self.training_trials < 0 or self.ticks_per_frame <= 0 or self.response_ticks <= 0:
            raise ValueError("trial and timing values must be positive")


@dataclass(frozen=True)
class TemporalTrialResult:
    condition: CopyCondition
    agent: AgentKind
    control: Control
    training_mode: TrainingMode
    target: tuple[str, ...]
    observed: tuple[OutputEvent, ...]
    report: EventStreamReport
    hidden_activity: float
    weight_change: float


@dataclass
class TemporalExperimentResult:
    trials: list[TemporalTrialResult]

    @property
    def by_condition(self) -> dict[tuple[CopyCondition, Control], list[TemporalTrialResult]]:
        return {
            key: [trial for trial in self.trials if (trial.condition, trial.control) == key]
            for key in {(trial.condition, trial.control) for trial in self.trials}
        }

    def summary(self) -> dict[tuple[str, str, str], dict[str, float]]:
        result: dict[tuple[str, str, str], dict[str, float]] = {}
        keys = {(x.condition, x.agent, x.control) for x in self.trials}
        for condition, agent, control in keys:
            trials = [
                x
                for x in self.trials
                if (x.condition, x.agent, x.control) == (condition, agent, control)
            ]
            result[(condition.value, agent.value, control.value)] = {
                "event_precision": float(np.mean([x.report.event_precision for x in trials])),
                "event_recall": float(np.mean([x.report.event_recall for x in trials])),
                "eos_accuracy": float(np.mean([x.report.eos_accuracy for x in trials])),
                "mean_latency": float(np.mean([x for t in trials for x in t.report.latencies]))
                if any(t.report.latencies for t in trials)
                else 0.0,
                "hidden_activity": float(np.mean([x.hidden_activity for x in trials])),
                "weight_change": float(np.mean([x.weight_change for x in trials])),
            }
        return result


def _frame(config: TemporalExperimentConfig, index: int, value: float = 5.0) -> np.ndarray:
    frame = np.zeros(len(config.alphabet) + 2)
    frame[index] = value
    return frame


def _stream(
    config: TemporalExperimentConfig, sequence: tuple[str, ...]
) -> tuple[list[InputSignal | np.ndarray], list[tuple[np.ndarray, str]]]:
    events: list[InputSignal | np.ndarray] = [InputSignal.INPUT_BEGIN]
    labelled: list[tuple[np.ndarray, str]] = []
    for symbol in sequence:
        frame = _frame(config, 2 + config.alphabet.index(symbol))
        events.append(frame)
        labelled.append((frame, symbol))
        events.extend(np.zeros_like(frame) for _ in range(config.ticks_per_frame))
    end = _frame(config, 1)
    events.append(end)
    labelled.append((end, "<EOS>"))
    events.append(InputSignal.INPUT_END)
    return events, labelled


def _make_runtime(config: TemporalExperimentConfig, seed: int) -> SpikingRuntime:
    agent = SpikingRuntime(
        input_features=len(config.alphabet) + 2,
        hidden_neurons=config.hidden_neurons,
        output_tokens=("<EOS>", *config.alphabet),
        neurons_per_token=2,
        seed=seed,
    )
    agent.output_readout.activation_threshold = 0.25
    agent.output_readout.release_threshold = 0.2
    agent.network.neurons.tau_membrane = 3.0
    agent.network.neurons.refractory_ticks = 1
    output = agent.layout.slice(Population.OUTPUT_CHAR)
    incoming = np.flatnonzero(
        (agent.network.synapses.target >= output.start)
        & (agent.network.synapses.target < output.stop)
    )
    preserved = np.concatenate(
        (agent.direct_input_output_edge_indices, agent.hidden_output_edge_indices)
    )
    agent.ablate_edges(incoming[~np.isin(incoming, preserved)])
    return agent


def _train(
    agent: SpikingRuntime,
    labelled: list[tuple[np.ndarray, str]],
    kind: AgentKind,
    mode: TrainingMode,
    trials: int,
) -> None:
    if mode in (TrainingMode.UNTRAINED, TrainingMode.NO_LEARNING):
        return
    for _ in range(trials):
        events: list[InputSignal | tuple[np.ndarray, str]] = [
            InputSignal.INPUT_BEGIN,
            *labelled,
            InputSignal.INPUT_END,
        ]
        if kind is AgentKind.SUPERVISED:
            agent.train_input_events(events)
        else:
            agent.train_reward_modulated_events(events)
            if mode is TrainingMode.REWARD_MODULATED_STDP:
                agent.plasticity.reinforce(1.0)
            agent.plasticity.reset_traces()
        agent.network.reset_state()


def run_temporal_experiment(
    config: TemporalExperimentConfig | None = None,
    *,
    conditions: Iterable[CopyCondition] = tuple(CopyCondition),
    controls: Iterable[Control] = (
        Control.TRAINED,
        Control.UNTRAINED,
        Control.NO_LEARNING,
        Control.SHUFFLED_TARGET,
        Control.RECURRENT_ABLATION,
        Control.DIRECT_INPUT_OUTPUT_ABLATION,
    ),
    agents: Iterable[AgentKind] = (AgentKind.SUPERVISED,),
) -> TemporalExperimentResult:
    """Run fresh, independent runtime cells; no sequence is retained by the host."""
    config = config or TemporalExperimentConfig()
    evaluation_sequences = config.evaluation_sequences or config.sequences
    trials: list[TemporalTrialResult] = []
    for condition in conditions:
        for kind in agents:
            for control in controls:
                for trial_index, sequence in enumerate(evaluation_sequences):
                    seed = (
                        config.seed
                        + trial_index
                        + 1000 * list(CopyCondition).index(condition)
                        + 10000 * list(Control).index(control)
                        + 100000 * list(AgentKind).index(kind)
                    )
                    agent = _make_runtime(config, seed)
                    if control is Control.RECURRENT_ABLATION:
                        agent.ablate_edges(agent.hidden_output_edge_indices)
                    if control is Control.DIRECT_INPUT_OUTPUT_ABLATION:
                        agent.ablate_edges(agent.direct_input_output_edge_indices)
                    events, labelled = _stream(config, sequence)
                    target = (*sequence, "<EOS>")
                    before = agent.network.synapses.weight.copy()
                    mode = (
                        TrainingMode.UNTRAINED
                        if control is Control.UNTRAINED
                        else TrainingMode.NO_LEARNING
                        if control is Control.NO_LEARNING
                        else TrainingMode.REWARD_MODULATED_STDP
                        if kind is AgentKind.REWARD_MODULATED_STDP
                        else TrainingMode.SUPERVISED
                    )
                    if control not in (Control.UNTRAINED, Control.NO_LEARNING):
                        training_target = (
                            labelled
                            if control is not Control.SHUFFLED_TARGET
                            else [
                                (frame, label)
                                for (frame, _), label in zip(labelled, reversed(target))
                            ]
                        )
                        _train(agent, training_target, kind, mode, config.training_trials)
                    observed = agent.run_input_events(
                        events,
                        response_ticks=config.response_ticks,
                        observe_during_input=condition is CopyCondition.IMMEDIATE,
                    )
                    hidden = agent.layout.slice(Population.HIDDEN)
                    report = evaluate_event_stream(
                        target,
                        observed,
                        config=EventStreamConfig(patience_window=config.patience_window),
                        end_time=config.response_ticks,
                        withheld_prefix=0,
                    )
                    trials.append(
                        TemporalTrialResult(
                            condition,
                            kind,
                            control,
                            mode,
                            target,
                            observed,
                            report,
                            float(np.mean(np.abs(agent.network.neurons.voltage[hidden]))),
                            float(np.abs(agent.network.synapses.weight - before).sum()),
                        )
                    )
    return TemporalExperimentResult(trials)


def run_paired_temporal_experiment(
    config: TemporalExperimentConfig | None = None,
) -> TemporalExperimentResult:
    """Run the supervised and reward-modulated agents over identical fresh trials."""
    return run_temporal_experiment(config, agents=tuple(AgentKind))


def main() -> None:
    for key, values in run_paired_temporal_experiment().summary().items():
        print(
            f"{' / '.join(key)}: recall={values['event_recall']:.2f} eos={values['eos_accuracy']:.2f}"
        )


if __name__ == "__main__":
    main()
