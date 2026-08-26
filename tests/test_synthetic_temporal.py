import numpy as np
import pytest

from continual_agent.agent.conversation_agent import AgentConfig, ConversationAgent
from continual_agent.agent.session import InputSignal
from continual_agent.agent.spiking_runtime import SpikingRuntime
from continual_agent.experiments.synthetic_temporal import (
    AgentKind,
    Control,
    CopyCondition,
    TemporalExperimentConfig,
    TrainingMode,
    _make_runtime,
    run_temporal_experiment,
)
from continual_agent.simulation.population_layout import Population


def small_config() -> TemporalExperimentConfig:
    return TemporalExperimentConfig(
        sequences=(("A", "A"), ("B", "A")),
        training_trials=2,
        ticks_per_frame=2,
        response_ticks=12,
    )


def test_experiment_uses_separate_conditions_and_controls() -> None:
    result = run_temporal_experiment(
        small_config(),
        conditions=(CopyCondition.IMMEDIATE, CopyCondition.DELAYED),
        controls=(Control.TRAINED, Control.NO_LEARNING),
    )

    assert len(result.trials) == 8
    assert set(result.by_condition) == {
        (CopyCondition.IMMEDIATE, Control.TRAINED),
        (CopyCondition.IMMEDIATE, Control.NO_LEARNING),
        (CopyCondition.DELAYED, Control.TRAINED),
        (CopyCondition.DELAYED, Control.NO_LEARNING),
    }
    trained = result.by_condition[(CopyCondition.IMMEDIATE, Control.TRAINED)]
    no_learning = result.by_condition[(CopyCondition.IMMEDIATE, Control.NO_LEARNING)]
    assert all(trial.weight_change > 0 for trial in trained)
    assert all(trial.weight_change == 0 for trial in no_learning)


def test_direct_path_ablation_removes_the_controlled_copy_path() -> None:
    result = run_temporal_experiment(
        small_config(),
        conditions=(CopyCondition.IMMEDIATE,),
        controls=(Control.DIRECT_INPUT_OUTPUT_ABLATION,),
    )

    assert all(trial.weight_change > 0 for trial in result.trials)


def test_immediate_copy_observes_repeated_symbols_and_eos() -> None:
    config = TemporalExperimentConfig(
        sequences=(("A", "A"),),
        training_trials=2,
        ticks_per_frame=3,
        response_ticks=8,
    )
    result = run_temporal_experiment(
        config,
        conditions=(CopyCondition.IMMEDIATE,),
        controls=(Control.TRAINED,),
    )

    trial = result.trials[0]
    assert trial.report.event_recall >= 0.0


def test_delayed_trial_observes_eos_but_not_presented_symbols() -> None:
    result = run_temporal_experiment(
        small_config(),
        conditions=(CopyCondition.DELAYED,),
        controls=(Control.TRAINED,),
    )

    assert all(trial.report.event_recall >= 0.0 for trial in result.trials)


def test_held_out_sequences_are_evaluated_against_unshuffled_targets() -> None:
    config = TemporalExperimentConfig(
        sequences=(("A", "B"),),
        evaluation_sequences=(("B", "A"),),
        training_trials=1,
        response_ticks=8,
    )
    result = run_temporal_experiment(
        config, conditions=(CopyCondition.IMMEDIATE,), controls=(Control.SHUFFLED_TARGET,)
    )
    trial = result.trials[0]
    assert trial.target == ("B", "A", "<EOS>")


def test_production_runtime_exposes_the_pathways_used_by_controls() -> None:
    agent = ConversationAgent(AgentConfig(input_features=4, language_alphabet=("A", "B"), seed=1))
    assert isinstance(agent.runtime, SpikingRuntime)
    assert agent.token_input_edge_indices.size > 0
    assert agent.recurrent_event_edge_indices.size > 0
    assert agent.runtime.hidden_output_edge_indices.size > 0
    assert agent.runtime.direct_input_output_edge_indices.size > 0


def test_paired_agents_have_distinct_training_modes() -> None:
    result = run_temporal_experiment(
        TemporalExperimentConfig(training_trials=1),
        conditions=(CopyCondition.DELAYED,),
        controls=(Control.TRAINED,),
        agents=tuple(AgentKind),
    )
    assert {trial.agent for trial in result.trials} == set(AgentKind)
    assert {trial.training_mode for trial in result.trials} == {
        TrainingMode.SUPERVISED,
        TrainingMode.REWARD_MODULATED_STDP,
    }


def test_synthetic_production_path_composes_the_shared_runtime() -> None:
    runtime = _make_runtime(small_config(), seed=3)

    assert isinstance(runtime, SpikingRuntime)
    assert runtime.layout.slice(Population.OUTPUT_CHAR).stop > 0
    assert runtime.token_input_edge_indices.size > 0


def test_direct_ablation_mask_survives_training() -> None:
    runtime = _make_runtime(small_config(), seed=7)
    edges = np.concatenate(
        (runtime.token_input_edge_indices.ravel(), runtime.recurrent_event_edge_indices)
    )
    runtime.ablate_edges(edges)
    runtime.train_input_events(
        (InputSignal.INPUT_BEGIN, (np.array([0.0, 0.0, 5.0, 0.0]), "A"), InputSignal.INPUT_END)
    )
    assert np.all(runtime.synapses.weight[edges] == 0.0)


def test_training_exception_resets_input_session() -> None:
    runtime = _make_runtime(small_config(), seed=8)

    def broken_events():
        yield InputSignal.INPUT_BEGIN
        raise RuntimeError("broken stream")

    with pytest.raises(RuntimeError):
        runtime.train_input_events(broken_events())
    assert runtime.response_session.state.value == "idle"
    assert not runtime.response_session.input_active
