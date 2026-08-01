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
- identify the non-empty 6L region centered by that tension;
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

For Direction work:

- stay inside `inputs.boundAnchor`;
- define a modifiable object, causal mechanism, and one minimal testable
  primary change from the bound baseline;
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
- reproduce and validate the baseline before testing the primary change;
- express only the controls, generation rules, statistics, and boundary cases
  needed to reproduce or falsify the claim. Prefer a compact deterministic rule
  over enumerating a future manifest, every sample, or every contingency;
- leave complete request manifests, sample/configuration IDs, literal hash
  formats, binary64 rounding conventions, per-draw bootstrap algorithms,
  window tables, traces, and execution scripts to a future experiment handoff
  unless a specific detail changes the scientific claim, baseline
  reproducibility, causal attribution, guardrail, or pass/fail meaning;
- when deepening, return the entire revised Direction rather than a patch.

For both:

- address supplied Reviewer findings and relevant query gaps when present;
- search and deep-read actual sources before citing them;
- distinguish sourced facts from hypotheses;
- do not execute a new experiment; express needed measurements in
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
