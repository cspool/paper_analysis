# Optimization Value Questions

Use these questions as a Topic-neutral expert lens for deciding whether an
Anchor identifies a real optimization opportunity and whether a Direction is
technically distinct and ready for experimental handoff. They are internal
review prompts, not an output questionnaire, scoring system, or source of
evidence.

Select only the few questions that can change the current verdict. Ignore
irrelevant questions. Answers must come from the bound Work Result, its cited
sources, allowed verification, and the Workflow Goal—not from this Ref.

## 1. Optimization opportunity and performance baseline

First classify the basis of the gap. A value copied from a paper, note, or
reference implementation is a source-reported gap and supports a candidate
opportunity. A measurement produced by an integrated EXP Goal is an observed
gap for that exact environment. Do not silently promote the first into the
second. Ask whether a smallest discriminating experiment can determine whether
the source-reported opportunity exists under an accessible target regime.

- What is the actual execution decomposition, data/control dependency, and
  critical path under the stated workload?
- Which values vary at runtime—shape, sequence length, batch, request mix,
  sparsity, topology, resource occupancy—and what decision cost follows?
- Is the reported or experimentally observed limit compute, memory capacity or bandwidth, communication,
  synchronization, queueing, launch overhead, serialization, or a combination?
- Do stages have complementary idle resources that could overlap, or do they
  compete for the same bottleneck and therefore invalidate the opportunity?
- Does the proposed observation include framework, dispatch, packing,
  transfer, synchronization, compilation, or control overhead that a narrower
  kernel/model metric would hide?
- Under which model, workload, precision, hardware, concurrency, cache state,
  and SLO regime does the symptom appear or disappear?
- Is there credible remaining headroom relative to an algorithmic, physical,
  architectural, or service limit, or is the baseline already close to it?
- Is the claimed limit fundamental, or can it be moved by changing mapping,
  granularity, scheduling, representation, or the layer interface?

## 2. Closest method baseline and Direction difference

- What existing method is closest in objective, modified object, mechanism,
  granularity, and operating regime—not merely closest by name?
- What is the strongest simple policy or implementation that can explain the
  proposed gain without the Direction's added information or complexity?
- Does that simple baseline retain its complete legal parameter domain and a
  fair calibration/search opportunity, or has the comparison made it weak by
  construction?
- Which design choices does it make about static versus dynamic work,
  scheduling granularity, synchronization, switching, placement, caching, or
  isolation?
- What overhead or failure boundary does that method introduce, and is it
  already sufficient under the current Anchor conditions?
- Does the Direction add a genuinely different mechanism, boundary, regime,
  interface, or hardware assumption, or only rename/recombine an existing
  solution without a testable difference?
- Can the primary change be independently toggled and attributed? If several
  parts are claimed together, are they technically inseparable?
- In which bounded event class should the added information or mechanism change
  an observable action, state transition, kernel path, or execution route?
- Can a calibrated simpler baseline produce the same relevant action trace? If
  so, is only implementation overhead left to compare rather than the claimed
  decision mechanism?
- Is the variant dominated by a simpler strategy under the same correctness,
  quality, throughput, and resource guards?
- Into which known method does the Direction degenerate under limiting
  conditions, and does that reveal duplication or a useful generalization?
- Is an existing framework or open implementation already the correct method
  baseline, and what must be frozen to compare against it fairly?
- Are expected benefits and degradation cases consistent with the proposed
  difference rather than generic advantages of the method family?
- Does the experiment order stop after correctness, calibration, trigger, and
  behavioral-equivalence checks when the unique lever is absent, instead of
  spending resources on a nondiscriminating sweep?

## 3. Reference experiment and environment reuse

Apply this group only after the Direction itself remains valid.

- What measurement granularity separates the claimed mechanism from unrelated
  end-to-end effects while preserving the Goal metric and guardrails?
- Which baseline implementation, benchmark, workload, trace, profiler, or
  evaluation harness can be reused rather than rebuilt?
- Is a simulator sufficient for the conclusion? Which decisive architecture
  mechanisms, timing effects, contention, or software overheads does its model
  omit, approximate, or cover inaccurately?
- Which claims require real hardware, and which can be answered by trace
  replay, profiling, microbenchmark, ablation, or sensitivity analysis?
- What model/task/hardware/precision/configuration coverage does the reference
  environment provide, and where would extrapolation become unsafe?
- Which code modules, interfaces, configuration files, datasets, metrics, and
  reproducibility artifacts are directly reusable?
- What minimal extension, port, instrumentation, or recalibration is needed to
  represent the current Direction or a new architecture?
- If no reusable environment exists after a bounded search, is that absence
  recorded clearly enough for an EXP Goal to build from the closest baseline?

## Converting answers into review output

- An unanswered question becomes a finding only when it can change admission,
  the principal comparison, baseline reproducibility, causal attribution, a
  material guardrail, or the experiment handoff.
- Use one bounded `queryGap` per resolution channel (`experiment`, `idea`,
  `knowledge`, or `human`). Do not encode several channels in one dimension.
- When a simple empirical discriminator can change whether the principal
  optimization opportunity is admitted, pair one `BLOCKING` finding with one
  `experiment` query gap. The requested observation should test the gap, not
  prematurely implement the full Direction.
- Do not copy these questions into `summary`, require the Worker to answer all
  of them, or add value scores and new JSON fields.
- A mature implementation environment cannot rescue an invalid or duplicate
  Direction. First establish opportunity and method difference; then assess
  experiment reuse.
