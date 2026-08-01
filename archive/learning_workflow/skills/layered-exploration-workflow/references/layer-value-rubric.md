# Layer and Value Rubric

## L1-L6 coarse expansion

Each layer is a coordinate for modifiable objects, not a requirement that every Direction span the full stack.

| Layer | Scope | Coarse questions |
|---|---|---|
| L1 Algorithm/Pipeline | graph, workload decomposition, algorithmic dynamicity | What subcomputations, phases, shapes, routing patterns, or approximations create a bottleneck or independent work? |
| L2 Serving/Runtime | requests, batches, stages, queues, placement, resource scheduling | What runtime object can be regrouped, overlapped, prioritized, partitioned, cached, or migrated? |
| L3 Compiler | IR, dependency analysis, fusion, pass pipeline, multiversion, codegen | What compiler representation or transformation can expose or specialize the opportunity? |
| L4 Kernel | tile/warp/instruction pipeline, sync, memory movement, multi-kernel execution | What kernel object, launch boundary, tile, layout, or synchronization can change? |
| L5 Architecture | execution units, scheduler, memory hierarchy, NoC, hardware primitives | Which physical resource enables or limits the opportunity, and is the limit hard or software-manageable? |
| L6 Chip/System | chiplet, PIM, wafer-scale, packaging, die-to-die/interconnect | Which chip-level mapping, topology, capacity, or data-movement boundary changes the result? |

## Three value axes

Run a coarse discovery task for each Layer × axis.

### exploration

Look first for:

- precise workload phase and request/shape regime;
- measured or testable bottleneck;
- independent work or pipeline boundary;
- runtime dynamicity and its distribution;
- underused orthogonal resource;
- additional overhead that could be hidden;
- concrete modifiable object;
- falsifiable metric and expected regime.

High-signal evidence identifies conditions and likely counterexamples. A paper's headline speedup alone is not sufficient.

### implementation_reuse

Look for:

- repository, module, class, pass, kernel, config, trace, simulator, profiler, dataset, or benchmark;
- production framework integration;
- code path that implements current practice or a strong baseline;
- measurement tooling and observable granularity;
- license/platform/dependency constraints;
- smallest reusable unit and adaptation cost.

An implementation asset can be valuable even if its original method is not a new exploration direction.

### method_reference

Look for:

- mechanism and design choice;
- substitute/complement/dependency relation;
- applicable conditions and degradation boundary;
- quantitative comparison;
- transferable abstraction;
- negative result or architecture limit.

Method reputation or venue is not a value criterion.

## Baseline is cross-cutting

Every task also asks:

- What is current practice in this layer?
- What is the strongest fair comparison?
- What tool/evaluation baseline exposes the effect?
- What reusable implementation can instantiate either side?

Do not discard a baseline because it lacks exploration novelty. Baseline facts create entries and remain available to all later Directions under the Anchor.

## Coarse screening

Use coarse labels only to allocate later effort:

```text
candidate: concrete signal plus at least one valid claim
uncertain: plausible signal with a named evidence gap
baseline_candidate: valid comparison/tool/implementation regardless of novelty
low_signal: no concrete modifiable object or no relevant scope
invalid: citation or scope validation failed
```

`low_signal` is not a final rejection. It can still contribute a GlobalEntity or valid baseline.

## Deep review questions

### Scenario and acceleration opportunity

- Are independent subcomputations or boundary dependencies demonstrated?
- Which values are known only at runtime, and what is their distribution?
- Which resource is the bottleneck?
- Are overlapped resources genuinely orthogonal?
- Which overhead is introduced, and can it be hidden?
- What exact regime makes the hypothesis fail?

### Baseline and fairness

- Does each baseline run the same workload, precision, backend allocation, and metric definition?
- Is current practice represented, not just a weak strawman?
- Can a single-layer baseline separate local gain?
- Can a combined baseline separate true cross-layer synergy?
- Are code and tool baselines versioned and runnable?

### Implementation

- What object is modified?
- Which file/module/pass/kernel/runtime decision is the likely entry point?
- What is compile-time versus runtime?
- What synchronization and switching overhead is added?
- What resources contend?
- Which implementation can be reused with minimal new code?

### Cross-layer validity

- What exact data/control/resource interface connects the entries?
- Is directionality correct?
- Is the edge direct evidence or inference?
- Are selected entries complementary, or are they substitutes accidentally combined?
- Is an unselected entry actually a required prerequisite?
- Can the synergy be isolated by ablation?

### Experiment and tool

- What metric can falsify the hypothesis?
- What measurement granularity is required?
- Which concurrency/contention factors are visible or missing?
- What is simulator/profiler error?
- What workloads cover the predicted regime and degradation boundary?
- What must be fixed for a fair comparison?

## Decision policy

Use the following order:

1. If evidence or graph integrity is invalid, `rejected`.
2. If the object is a valid baseline/tool/reference but not an exploration candidate, `baseline_reference`.
3. If exploration is promising but a decisive claim/edge/baseline is missing, `needs_evidence`.
4. If the hypothesis is falsifiable, baseline is fair, and implementation/measurement path is adequate, `experiment_candidate`.

Do not calculate a weighted average that lets method quality compensate for low exploration value.

