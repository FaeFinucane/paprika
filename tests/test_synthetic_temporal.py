from unittest.mock import Mock, patch

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
    _stream,
    main,
    run_temporal_experiment,
)
from continual_agent.plasticity.stdp import RewardModulatedSTDP
from continual_agent.simulation.population_layout import Population
from continual_agent.simulation.synapses import SparseSynapses


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


def test_summary_has_stable_condition_agent_control_order() -> None:
    result = run_temporal_experiment(
        small_config(),
        conditions=(CopyCondition.DELAYED, CopyCondition.IMMEDIATE),
        controls=(Control.NO_LEARNING, Control.TRAINED),
        agents=(AgentKind.REWARD_MODULATED_STDP, AgentKind.SUPERVISED),
    )

    assert list(result.summary()) == [
        ("immediate-copy", "supervised-structural", "trained"),
        ("immediate-copy", "supervised-structural", "no-learning"),
        ("immediate-copy", "reward-modulated-stdp", "trained"),
        ("immediate-copy", "reward-modulated-stdp", "no-learning"),
        ("delayed-copy", "supervised-structural", "trained"),
        ("delayed-copy", "supervised-structural", "no-learning"),
        ("delayed-copy", "reward-modulated-stdp", "trained"),
        ("delayed-copy", "reward-modulated-stdp", "no-learning"),
    ]


def test_main_default_is_compact_and_verbose_includes_all_rows(
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary = {
        (condition.value, agent.value, control.value): {
            "event_recall": 0.5,
            "eos_accuracy": 1.0,
        }
        for condition in CopyCondition
        for agent in AgentKind
        for control in Control
    }
    result = Mock()
    result.summary.return_value = summary
    with patch(
        "continual_agent.experiments.synthetic_temporal.run_paired_temporal_experiment",
        return_value=result,
    ):
        main([])
    default_output = capsys.readouterr().out
    assert "Trained comparison" in default_output
    assert "Immediate" in default_output and "Delayed" in default_output
    assert default_output.count("/ supervised structural /") == 0
    assert len(default_output.splitlines()) == 10

    with patch(
        "continual_agent.experiments.synthetic_temporal.run_paired_temporal_experiment",
        return_value=result,
    ):
        main(["--verbose"])
    verbose_output = capsys.readouterr().out
    assert "all aggregates" in verbose_output
    assert verbose_output.count("/ Supervised /") == 12


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


def test_supervised_immediate_trains_eos_from_input_end() -> None:
    config = TemporalExperimentConfig(
        sequences=(("A",),), training_trials=12, ticks_per_frame=3, response_ticks=64
    )
    trial = run_temporal_experiment(
        config,
        conditions=(CopyCondition.IMMEDIATE,),
        controls=(Control.TRAINED,),
    ).trials[0]

    assert [event.name for event in trial.observed] == ["A", "<EOS>"]
    assert trial.report.event_recall == 1.0
    assert trial.report.eos_accuracy
    assert trial.weight_change > 0.0


def test_delayed_trial_observes_eos_but_not_presented_symbols() -> None:
    result = run_temporal_experiment(
        small_config(),
        conditions=(CopyCondition.DELAYED,),
        controls=(Control.TRAINED,),
    )

    assert all(trial.report.event_recall >= 0.0 for trial in result.trials)


def test_supervised_delayed_starts_response_at_input_end() -> None:
    config = TemporalExperimentConfig(
        sequences=(("A",),), training_trials=12, ticks_per_frame=3, response_ticks=64
    )
    trial = run_temporal_experiment(
        config,
        conditions=(CopyCondition.DELAYED,),
        controls=(Control.TRAINED,),
    ).trials[0]

    assert [event.name for event in trial.observed] == ["A", "<EOS>"]
    assert trial.report.event_recall == 1.0
    assert trial.report.eos_accuracy
    assert trial.weight_change > 0.0


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
    assert agent.runtime.token_input_edge_indices.size > 0
    assert agent.runtime.recurrent_event_edge_indices.size > 0
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
    assert np.all(runtime.network.synapses.weight[edges] == 0.0)


def test_training_exception_resets_input_session() -> None:
    runtime = _make_runtime(small_config(), seed=8)

    def broken_events():
        yield InputSignal.INPUT_BEGIN
        raise RuntimeError("broken stream")

    with pytest.raises(RuntimeError):
        runtime.train_input_events(broken_events())
    assert runtime.response_session.state.value == "idle"
    assert not runtime.response_session.input_active


def test_synthetic_input_has_no_eos_or_teacher_frame() -> None:
    events, labelled = _stream(small_config(), ("A",))

    assert any(event is InputSignal.INPUT_END for event in events)
    assert all(not isinstance(event, tuple) for event in events)
    assert [label for _, label in labelled] == ["A"]


def test_supervised_eos_alignment_does_not_receive_a_fabricated_boundary_frame() -> None:
    config = small_config()
    runtime = _make_runtime(config, seed=12)
    original = runtime._apply_supervised_target
    wrapped = Mock(wraps=original)

    _, labelled = _stream(config, ("A",))
    with patch.object(runtime, "_apply_supervised_target", wrapped):
        runtime.train_input_events([InputSignal.INPUT_BEGIN, *labelled, InputSignal.INPUT_END])
    seen = [call.args[0] for call in wrapped.call_args_list]
    assert seen
    assert all(not frame[0] and not frame[1] for frame in seen)


def test_reward_training_uses_actual_output_for_reward() -> None:
    config = TemporalExperimentConfig(
        sequences=(("A",),), training_trials=1, response_ticks=4, ticks_per_frame=1
    )
    result = run_temporal_experiment(
        config,
        conditions=(CopyCondition.DELAYED,),
        controls=(Control.TRAINED,),
        agents=(AgentKind.REWARD_MODULATED_STDP,),
    )
    trial = result.trials[0]
    assert trial.training_mode is TrainingMode.REWARD_MODULATED_STDP
    assert np.isfinite(trial.reward)
    assert trial.eligibility_change > 0.0


def test_synaptic_weights_use_one_global_bound() -> None:
    synapses = SparseSynapses(np.array([0]), np.array([1]), np.array([3.0]), neuron_count=2)
    plasticity = RewardModulatedSTDP(synapses, weight_limit=1.0)

    plasticity.reinforce(0.0)

    assert synapses.weight[0] == 1.0


def test_delayed_reward_events_are_timestamped_after_input() -> None:
    config = TemporalExperimentConfig(ticks_per_frame=2, response_ticks=4)
    runtime = _make_runtime(config, seed=11)
    events, _ = _stream(config, ("A",))

    observed = runtime.run_input_events(
        events, response_ticks=config.response_ticks, observe_during_input=False
    )

    input_ticks = sum(isinstance(event, np.ndarray) for event in events) + 2
    assert runtime.output_readout.timestamp == input_ticks + config.response_ticks - 1
    assert all(event.timestamp >= input_ticks - 1 for event in observed)
