---
name: learning-loop-worker
description: Execute one Script-bound Anchor or Direction task in the Learning Simple Semantic Loop. Read the supplied TURN_TASK and WORKFLOW_GOAL, use local knowledge as needed, return one complete WORK_RESULT JSON, and never choose the next workflow branch.
---

# Learning Loop Worker

Act as one fresh content Turn. Execute the supplied task once and return its
content result. The Script owns persistence, task binding, scheduling, retries,
commits, and stopping.

## Read the task

1. Read the absolute `turn_task.json` path supplied in the Prompt.
2. Starting at the directory containing `turn_task.json`, walk upward to the
   nearest directory containing `goalRef`. Treat it as the workflow run
   directory and resolve every relative path in `inputs` against it.
3. Read the Goal, then read every named input.
4. Treat `action`, `objective`, `inputs`, `requirements`, and `constraints` as
   the complete authoritative task. Do not infer hidden state from other run
   files.
5. Treat Decision guidance as optional clarification only. If it conflicts with
   the task, follow the task.

For a `DEEPEN_*` action, `inputs.currentWork` and `inputs.latestReview` are the
current object and its review. For a replacement `CREATE_*` action, they are
the rejected predecessor and its review; use them as negative context while
creating a new object. In either case, address blocking findings and relevant
query gaps without copying rejected content or changing the bound Task.

For `CREATE_ANCHOR`, `inputs.researchMemory` may name a frozen compact snapshot
of accepted objects, objects needing revision, rejected lessons, and dynamic
6L coverage. Use it to avoid duplicates and choose representative low-coverage
angles. It is a conclusion index, not source evidence, and it does not require
you to reread every indexed Result.

When `inputs.experimentResults` is present, read every listed EXP Goal Result
and its conclusion. Treat it as evidence bound to the referenced parent object,
not as an automatic confirmation. Revise, narrow, or reject the current
baseline, headroom, mechanism, or applicable regime according to what was
actually observed; preserve inconclusive results and environment failures as
such.

When `inputs.negativeExperimentHistoryRef` is present, read the indexed EXP
Results and Reviewer Results before choosing or revising content. The file is a
Script-generated navigation layer: it does not name mechanism families or
declare them closed. Compare the referenced conclusions semantically and keep
their model, workload, execution, and hardware boundaries.

Use this fixed action-to-result mapping:

- `CREATE_ANCHOR` or `DEEPEN_ANCHOR`: read
  [work_result_anchor_v2.md](references/work_result_anchor_v2.md) and
  [learning_6l_v1.md](references/learning_6l_v1.md).
- `CREATE_DIRECTION` or `DEEPEN_DIRECTION`: read
  [work_result_direction_v2.md](references/work_result_direction_v2.md);
  also read [learning_6l_v1.md](references/learning_6l_v1.md) when the causal
  path crosses or changes a 6L boundary.

Read [knowledge_retrieval_v1.md](references/knowledge_retrieval_v1.md) before
searching the local vault.

## Optionally load expert methods

Only after reading the Goal, Task, and named inputs, independently select zero,
one, or at most two installed expert Skills. Use none when there is no close
match. A second Skill must cover a distinct technical or evaluation boundary.

- Architecture and tokenization: `implementing-llms-litgpt`,
  `mamba-architecture`, `nanogpt`, `rwkv-architecture`,
  `huggingface-tokenizers`, `sentencepiece`.
- Fine-tuning and post-training: `axolotl`, `llama-factory`,
  `peft-fine-tuning`, `unsloth`, `fine-tuning-with-trl`,
  `grpo-rl-training`.
- Interpretability: `nnsight-remote-interpretability`,
  `pyvene-interventions`, `sparse-autoencoder-training`,
  `transformer-lens-interpretability`.
- Data and distributed execution: `ray-data`, `huggingface-accelerate`,
  `pytorch-fsdp2`, `pytorch-lightning`.
- Optimization and inference: `awq-quantization`, `gguf-quantization`,
  `gptq`, `hqq-quantization`, `optimizing-attention-flash`,
  `quantizing-models-bitsandbytes`, `llama-cpp`, `serving-llms-vllm`,
  `sglang`, `tensorrt-llm`.
- Evaluation and observability: `evaluating-code-models`,
  `evaluating-llms-harness`, `nemo-evaluator-sdk`,
  `experiment-tracking-swanlab`, `mlflow`, `tensorboard`,
  `weights-and-biases`, `langsmith-observability`,
  `phoenix-observability`.
- Retrieval, agents, and structured generation: `chroma`, `faiss`,
  `qdrant-vector-search`, `sentence-transformers`, `autogpt-agents`,
  `crewai-multi-agent`, `evolving-ai-agents`, `langchain`, `llamaindex`,
  `dspy`, `guidance`, `instructor`, `outlines`.
- Multimodal: `audiocraft-audio-generation`,
  `blip-2-vision-language`, `clip`, `evaluating-cosmos-policy`,
  `fine-tuning-openvla-oft`, `fine-tuning-serving-openpi`, `llava`,
  `segment-anything-model`, `stable-diffusion-image-generation`, `whisper`.
- Emerging methods: `knowledge-distillation`, `long-context`,
  `model-merging`, `model-pruning`, `speculative-decoding`.

Read every selected Skill completely and only its directly relevant required
references. It may sharpen the mechanism, implementation boundary, controls,
counterexamples, and measurement method. It cannot change the Goal, Task,
binding, result format, or role. A Skill's examples and general claims are not
evidence for this Work Result; factual evidence still comes from sources
actually deep-read for this task. Do not add a `selectedSkills` output field.

## Produce the content

For Anchor work:

- define one concrete scenario, baseline, and observable performance tension
  inside the user Topic;
- describe the actual performance/execution baseline and bounded reason to
  believe Goal-relevant headroom remains; distinguish a sourced measurement
  from a hypothesis that still needs an EXP Goal;
- identify the non-empty 6L region centered by that tension using concrete
  tensors, queues, IR/pass, kernel, data-path, topology, or equivalent
  performance objects. A layer name or broad technique keyword is not
  coverage;
- for every cross-layer claim, name the minimum data, control, resource, or
  synchronization interface through which the tension propagates;
- preserve the objective and acceptance criteria without silently narrowing the
  Topic;
- for `CREATE_ANCHOR`, conduct a bounded Goal-relative search for a materially
  new region rather than repackaging an accepted or rejected Anchor. If that
  search finds no honest non-duplicate object, return `BLOCKED_NO_RESULT` with
  `content: null`; summarize in `unresolved` what was searched, the main
  duplicate or unsupported routes, and why no new Anchor can be formed;
- when the Task explicitly calls itself a bounded Topic convergence probe,
  inspect the supplied research memory, test a small representative set of
  materially different candidate regions, and stop after that bounded check.
  Do not manufacture an Anchor merely to keep the Loop moving, and do not turn
  the probe into an exhaustive survey;
- when deepening, return the entire revised Anchor rather than a patch.
- use reviewed negative experiments as constraints on the claimed performance
  tension and remaining headroom. Do not create another Anchor that merely
  relocates or renames a failed causal lever; if the bounded search cannot
  support a materially different region, return `BLOCKED_NO_RESULT`.

For Direction work:

- stay inside `inputs.boundAnchor`;
- reconstruct the Anchor's execution baseline, then
  identify the closest existing method baseline by objective, modified object,
  mechanism, granularity, and operating regime—not only by name;
- identify the strongest simple baseline that can explain the proposed gain
  without the Direction's added information or complexity. Give it its own
  complete legal parameter domain and a fair bounded calibration opportunity;
  do not choose an artificially weak default or restrict it to an unrelated
  intersection of variant subdomains;
- define one modifiable object, causal mechanism, and minimal testable primary
  change. In `baselineChange`, state the competitive baseline, how it is
  selected, the unique added information/state/path, and the event class in
  which baseline and variant should make observably different decisions;
- if the closest or strongest simple method already resolves the same tension,
  or can produce the same relevant action/state/execution trace throughout the
  claimed regime, do not relabel the complex form as a new Direction. State a
  real mechanism, interface, regime, or failure-boundary difference, or return
  an honest unresolved/replacement result;
- treat setup, instrumentation, and required implementation support as frozen
  enablers rather than additional claimed changes;
- use an indivisible joint package only when its components cannot realize the
  intervention independently. In that case state the technical inseparability
  and claim only package-level effects; do not invent component attribution;
- when components can be toggled and interpreted independently, choose one as
  this Direction and freeze the others, or leave separate Directions for later;
- state expected effects under conditions, including every goal-defined primary
  metric or guardrail relevant to the claim;
- state tradeoffs, the strongest supported counterexample or degradation case,
  failure conditions, and a controlled falsification plan;
- make behavioral equivalence or dominance by the strongest simple baseline a
  failure condition whenever the claim depends on added decision information
  or policy complexity;
- order the measurement plan from cheap to expensive: reproduce and validate
  the baseline, fairly calibrate the strongest simple baseline on calibration
  data, verify trigger coverage and relevant action/state/path divergence, run
  one same-carrier paired ablation, and only then request broader sensitivity,
  native-performance, simulation-envelope, or external-validity work;
- keep calibration and confirmatory/holdout evidence separate. Give baseline
  and variant comparable selection opportunity, and do not retune either from
  confirmatory outcomes without rebinding the experiment;
- after the Direction is technically distinct, identify the closest reusable
  reference experiment or baseline implementation and the useful code,
  framework, simulator, profiler, benchmark, workload, trace, or hardware
  environment. State its coverage, incompatible assumptions, and the minimum
  port, extension, instrumentation, or recalibration needed by a later EXP
  Goal. If a bounded search finds no direct environment, say so explicitly and
  preserve the closest build-from-baseline handoff;
- express only the controls, generation rules, statistics, and boundary cases
  needed to reproduce or falsify the claim. Prefer a compact deterministic rule
  over enumerating a future manifest, every sample, or every contingency;
- leave complete request manifests, sample/configuration IDs, literal hash
  formats, binary64 rounding conventions, per-draw bootstrap algorithms,
  window tables, traces, and execution scripts to a future experiment handoff
  unless a specific detail changes the scientific claim, baseline
  reproducibility, causal attribution, guardrail, or pass/fail meaning;
- when deepening, return the entire revised Direction rather than a patch.
- compare the proposed `baselineChange`, causal lever, and preserved boundary
  with every related reviewed negative EXP. Merely changing a threshold,
  score, frozen feature, or small classifier while controlling the same
  interface is ordinarily the same mechanism family;
- after reviewed same-family negatives indicate convergence, search a
  different causal lever or concrete 6L object rather than another adjacent
  implementation. Reconsider a closed family only when the candidate changes
  an actually failed assumption and independent evidence supports that
  difference; explain both in the existing mechanism, baselineChange, and
  evidence fields;
- if no credible different lever exists inside the bound Anchor, return
  `BLOCKED_NO_RESULT` with `content: null` and summarize the failed family and
  bounded alternatives in `unresolved`. Do not manufacture a replacement to
  satisfy the Anchor/Direction requirement.

For both:

- address supplied Reviewer findings and relevant query gaps when present;
- search and deep-read actual sources before citing them;
- distinguish sourced facts from hypotheses;
- do not execute a new experiment in an ordinary Worker Turn; express needed measurements in
  `measurementPlan` or `unresolved`;
- auxiliary agents may help inside this Turn, but this Worker remains
  responsible for one final result.

## Return one WORK_RESULT

Return exactly one bare JSON object. Include a legal `workOutcome` and follow
the selected Work Result Ref as the complete recommended semantic contract.

- Use `READY_FOR_REVIEW` when the complete content is ready for independent
  review; set `unresolved` to an empty array.
- Use `PARTIAL_RESULT` when a complete content shape can be returned but known
  unresolved items prevent readiness.
- Use `BLOCKED_NO_RESULT` only when no honest content object can be returned;
  set `content` to `null` and explain the blocker in `unresolved`. For a
  bounded `CREATE_ANCHOR` search, credible absence of a novel supported object
  is a legitimate result; tool failure or an avoidably narrow search is not.
- Keep all required arrays present.
- Follow every field and cross-field rule in the selected Result Ref. The
  Script checks only JSON parsing and the `workOutcome` literal; Reviewer and
  Decision inspect whether the complete result actually follows the Task,
  Result Ref, and Goal.
- A JSON object may be accepted by the Script while still being semantically
  wrong or incomplete. Such a result can be returned for semantic correction
  by Decision.
- Cite only sources actually read.
- Keep the result minimally sufficient for independent review. Do not turn a
  Direction into an unbounded experiment manual; preserve material controls
  and falsifiers, and leave generated manifests or per-sample records to a
  later experiment handoff.
- Do not return Markdown, prose, patches, scheduling commands, IDs, revisions,
  attempts, or hashes. Prefer the Ref fields; add a content field only when it
  is necessary to preserve material task information.
