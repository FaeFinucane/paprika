# Training

Training is implemented as cooperating services around one recurrent spiking
network. The supported workload is synthetic temporal copy.

`synthetic_temporal._stream()` emits numeric feature frames bracketed by
`InputSignal.INPUT_BEGIN` and `InputSignal.INPUT_END`. `InputRunner` validates
the boundaries and advances the production runtime for every frame. No
semantic text encoding or fabricated teacher frame is supplied.

`SpikingRuntime.train_input_events()` performs supervised alignment on labelled
raw frames and aligns EOS from genuine boundary-response activity.
`train_delayed_copy_baseline()` additionally strengthens existing same-group
hidden recurrence for delayed retention. Updates remain bounded and the
persistent ablation mask is reapplied after updates.

The reward agent runs the raw stream through `run_input_events()`.
`evaluate_event_stream()` accounts for actual timestamped output events using a
stage-specific `RewardSchedule`. `RewardLedger` commits event rewards and
`RewardModulatedSTDP` applies reinforcement to eligible synapses.

Every condition constructs an independent runtime. Controls distinguish
trained, untrained, no-learning, shuffled-target, recurrent-ablation, and
direct input/output-ablation behavior. Trial results retain event precision and
recall, EOS accuracy, latency, firing/event rates, eligibility, weight changes,
and pathway-specific diagnostics.

`RuntimeSession` and `SessionExecutionSnapshot` provide reusable lifecycle and
state-isolation infrastructure for the runtime.
