import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  CanonicalStore,
  createInitialState,
} from "../canonical_store.ts";
import {
  buildExpertReview,
  mergeCuratorDelta,
  validateDirectionProposal,
} from "../domain_validators.ts";
import { runDirectionStage } from "../direction_stage.ts";
import {
  findFirstJsonValue,
  parseProtocol,
  semanticPayloadUnchanged,
} from "../protocol_parser.ts";
import { validateProtocolTransition } from "../protocol_state_machine.ts";
import { renderRun } from "../renderer.ts";
import {
  anchorId,
  normalizeAnchorSignature,
  stableId,
} from "../stable_ids.ts";
import {
  validateClaimCandidate,
} from "../source_validator.ts";
import type {
  AgentHandle,
  EvidenceClaim,
  JsonValue,
  ReviewQuestionAnswer,
} from "../types.ts";
import { validateRunState } from "../validators.ts";
import type { RuntimeManager } from "../runtime_manager.ts";
import { testConfig } from "./test_helpers.ts";

test("stable IDs normalize semantically equivalent Anchor signatures", () => {
  const first = normalizeAnchorSignature({
    workload: "  VLM_Inference ",
    phase: "Decode",
    regime: "Batch 1",
    backend: "H100",
    bottleneck: "HBM bandwidth",
    primaryBaselineExecutionPath: "Sequential_Dispatch",
    targetMetrics: ["Latency", "Throughput", "latency"],
  });
  const second = normalizeAnchorSignature({
    workload: "vlm inference",
    phase: "decode",
    regime: "batch-1",
    backend: "h100",
    bottleneck: "hbm bandwidth",
    primaryBaselineExecutionPath: "sequential dispatch",
    targetMetrics: ["throughput", "latency"],
  });
  assert.equal(anchorId(first), anchorId(second));
});

test("protocol parser enforces one marker and role transition", () => {
  const text = `___ANCHOR_ROUND_PLAN_START___
round: 1
action: plan_round
task_count: 1
___SEMANTIC_PAYLOAD_START___
[{"focus":"decode overlap","layer":"L2","value_axis":"exploration","avoid":[]}]
___SEMANTIC_PAYLOAD_END___
___ANCHOR_ROUND_PLAN_END___

[LOOP: §EVAL_ROUND | await=ROUND_RESULT | round=1]`;
  const parsed = parseProtocol("anchor_stage_controller", text);
  validateProtocolTransition("anchor_stage_controller", parsed);
  assert.equal(parsed.marker, "ANCHOR_ROUND_PLAN");
  assert.equal((parsed.payload as JsonValue[]).length, 1);
  assert.throws(
    () => parseProtocol("anchor_stage_controller", `${text}\n${text}`),
    /exactly one main marker/,
  );
});

test("protocol repair comparison detects semantic mutation", () => {
  const raw = "prose before\n{\"a\":[1,2]}";
  const payload = findFirstJsonValue(raw);
  assert.deepEqual(payload, { a: [1, 2] });
  assert.equal(semanticPayloadUnchanged(payload!, { a: [1, 2] }), true);
  assert.equal(semanticPayloadUnchanged(payload!, { a: [2, 1] }), false);
});

test("source validator locates an exact quote and corrects wrong lines", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "workflow-source-"));
  const noteDir = path.join(root, "paper_secs");
  fs.mkdirSync(noteDir, { recursive: true });
  fs.writeFileSync(
    path.join(noteDir, "note.md"),
    ["# Title", "", "Baseline dispatch is sequential.", "Overlap cuts the idle tail."].join("\n"),
    "utf8",
  );
  const result = validateClaimCandidate({
    statement: "The baseline dispatch path is sequential.",
    claim_type: "baseline",
    evidence_kind: "direct",
    source_path: "paper_secs/note.md",
    line_start: 99,
    line_end: 99,
    quote: "Baseline dispatch is sequential.",
    applicable_scope: "test",
    confidence: "high",
  }, "AE-1", root, ["paper_secs"]);
  assert.equal(result.rejection, null);
  assert.equal(result.claim?.lineStart, 3);
});

function buildGraphState() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "workflow-domain-"));
  const noteDir = path.join(root, "paper_secs");
  fs.mkdirSync(noteDir, { recursive: true });
  fs.writeFileSync(
    path.join(noteDir, "evidence.md"),
    [
      "VLM decode at batch one is memory bound.",
      "The baseline launches visual and text kernels sequentially.",
      "A runtime bucket can select a specialized fused kernel.",
      "Nsight Systems records launch gaps.",
    ].join("\n"),
    "utf8",
  );
  const config = testConfig(root);
  const state = createInitialState("test", config);
  const candidates = [
    ["VLM decode at batch one is memory bound.", "scenario", 1],
    ["The baseline launches visual and text kernels sequentially.", "baseline", 2],
    ["A runtime bucket can select a specialized fused kernel.", "opportunity", 3],
    ["Nsight Systems records launch gaps.", "metric", 4],
  ];
  for (const [statement, claimType, line] of candidates) {
    const validated = validateClaimCandidate({
      statement,
      claim_type: claimType,
      evidence_kind: "direct",
      source_path: "paper_secs/evidence.md",
      line_start: line,
      line_end: line,
      quote: statement,
      applicable_scope: "VLM decode batch one H100",
      confidence: "high",
    }, "AE-1", root, ["paper_secs"]);
    state.claims.push(validated.claim!);
  }
  const [scenario, baselineClaim, opportunity, metric] = state.claims.map((claim) => claim.claimId);
  const delta = {
    entities: [{
      kind: "implementation",
      name: "runtime bucket",
      description: "dispatch implementation",
      evidence_refs: [opportunity],
    }],
    anchors: [{
      local_id: "a1",
      title: "VLM batch-one decode launch gap",
      scenario: "VLM decode at batch one on H100",
      signature: {
        workload: "VLM inference",
        phase: "decode",
        regime: "batch one",
        backend: "H100",
        bottleneck: "memory and launch gap",
        primary_baseline_execution_path: "sequential visual and text kernel launch",
        target_metrics: ["latency", "throughput"],
      },
      evidence_refs: [scenario, baselineClaim],
      baselines: [{
        local_id: "b0",
        kind: "current_practice",
        name: "sequential launch",
        execution_path: "visual kernel then text kernel",
        implementation: "existing runtime",
        comparison_scope: "same H100 and batch one",
        evidence_refs: [baselineClaim],
        exploration_value: "low",
      }],
      entries: [{
        local_id: "e1",
        entity_name: "runtime bucket",
        layer: "L2",
        role: "opportunity",
        claim: "Bucket the runtime request regime.",
        modifiable_object: "runtime dispatch bucket",
        applicable_baselines: ["b0"],
        preconditions: ["stable batch-one regime"],
        expected_effect: "reduce dispatch gap",
        evidence_refs: [opportunity],
        confidence: "middle",
      }, {
        local_id: "e2",
        layer: "L4",
        role: "implementation",
        claim: "Select a specialized fused kernel.",
        modifiable_object: "kernel variant",
        applicable_baselines: ["b0"],
        preconditions: ["runtime bucket"],
        expected_effect: "lower latency",
        evidence_refs: [opportunity],
        confidence: "middle",
      }, {
        local_id: "e3",
        layer: "L4",
        role: "evaluation",
        claim: "Measure launch gaps with Nsight Systems.",
        modifiable_object: "measurement setup",
        applicable_baselines: ["b0"],
        preconditions: [],
        expected_effect: "measure launch gap",
        evidence_refs: [metric],
        confidence: "high",
      }],
      edges: [{
        from_entry: "e1",
        to_entry: "e2",
        relation: "controls",
        interface: "runtime bucket selects kernel variant",
        compatibility: "conditional",
        condition: "dispatch overhead is smaller than saved launch gap",
        evidence_refs: [opportunity],
        confidence: "middle",
      }],
      gaps: ["strong comparison baseline is not yet evidenced"],
    }],
    dispositions: [],
  };
  const merged = mergeCuratorDelta(state, delta as unknown as JsonValue, 1);
  return { root, state, merged };
}

test("curator merge keeps multiple same-layer entries and low-exploration baseline", () => {
  const { state, merged } = buildGraphState();
  assert.equal(merged.newAcceptedAnchorIds.length, 1);
  assert.equal(state.entries.filter((entry) => entry.layer === "L4").length, 2);
  assert.equal(state.baselines[0].explorationValue, "low");
  assert.equal(state.baselines[0].status, "active");
});

test("Direction validator accepts a connected compatible subgraph and rejects conflicts", () => {
  const { state } = buildGraphState();
  const anchor = state.anchors[0];
  const opportunity = state.entries.find((entry) => entry.role === "opportunity")!;
  const implementation = state.entries.find((entry) => entry.role === "implementation")!;
  const edge = state.edges[0];
  const valid = validateDirectionProposal(state, anchor.anchorId, {
    selected_entry_ids: [opportunity.entryId, implementation.entryId],
    selected_edge_ids: [edge.edgeId],
    baseline_ids: [state.baselines[0].baselineId],
    hypothesis: "Runtime bucketing selects a fused kernel and lowers p50 latency without reducing throughput.",
    ablation_plan: ["baseline", "bucket only", "bucket plus fused kernel"],
    implementation_plan: ["add one dispatch bucket"],
  }, 1);
  assert.equal(valid.errors.length, 0);
  assert.ok(valid.direction);
  state.edges.push({
    ...edge,
    edgeId: stableId("X", ["conflict"], 16),
    relation: "substitutes",
    compatibility: "conflict",
  });
  const invalid = validateDirectionProposal(state, anchor.anchorId, {
    selected_entry_ids: [opportunity.entryId, implementation.entryId],
    selected_edge_ids: [edge.edgeId],
    baseline_ids: [state.baselines[0].baselineId],
    hypothesis: "invalid conflict",
  }, 2);
  assert.match(invalid.errors.join("\n"), /conflict/);
});

test("Direction cap gives the persistent Planner a terminal completion turn", async () => {
  const { root, state } = buildGraphState();
  const anchor = state.anchors.find((item) => item.status === "accepted")!;
  const entry = state.entries.find((item) => item.anchorId === anchor.anchorId)!;
  const baseline = state.baselines.find((item) => item.anchorId === anchor.anchorId)!;
  state.stage1.status = "complete";
  state.stage1.acceptedAnchorIds = [anchor.anchorId];
  state.stage1.stopReason = "target_reached";
  state.stage1.anchorSpaceVersion = "test-anchor-space";

  const runDir = path.join(root, "planner-finalization");
  const store = new CanonicalStore(runDir);
  store.initialize(state);
  const handle: AgentHandle = {
    role: "direction_planner",
    scopeId: anchor.anchorId,
    threadId: "planner-thread",
    skillPath: "learning-direction-planner/SKILL.md",
    skillHash: "test",
    persistent: true,
    firstTurn: true,
    turnCount: 0,
  };
  const prompts: string[] = [];
  const proposal = `___DIRECTION_PROPOSAL_START___
anchor_id: ${anchor.anchorId}
proposal_index: 1
___SEMANTIC_PAYLOAD_START___
${JSON.stringify({
    selected_entry_ids: [entry.entryId],
    selected_edge_ids: [],
    baseline_ids: [baseline.baselineId],
    hypothesis: "The selected entry lowers latency without reducing throughput under a fixed workload.",
    ablation_plan: ["baseline", "selected entry"],
    implementation_plan: ["apply the selected modification"],
  })}
___SEMANTIC_PAYLOAD_END___
___DIRECTION_PROPOSAL_END___

[LOOP: §EVAL_DIRECTION | await=DIRECTION_COMMIT_RESULT | anchor_id=${anchor.anchorId}]`;
  const completion = `___DIRECTION_PLANNING_COMPLETE_START___
anchor_id: ${anchor.anchorId}
reason: budget_exhausted
___DIRECTION_PLANNING_COMPLETE_END___

[LOOP: §TERMINATED | done]`;
  const runtime = {
    persistentAgent: async () => handle,
    runProtocolTurn: async (_handle: AgentHandle, prompt: string) => {
      prompts.push(prompt);
      const protocol = parseProtocol(
        "direction_planner",
        prompts.length === 1 ? proposal : completion,
      );
      validateProtocolTransition("direction_planner", protocol);
      return {
        handle,
        result: {} as never,
        protocol,
        repaired: false,
      };
    },
  } as unknown as RuntimeManager;

  await runDirectionStage(state, store, runtime, "direction-plan");

  assert.equal(prompts.length, 2);
  assert.match(prompts[1], /只输出 DIRECTION_PLANNING_COMPLETE/);
  assert.match(prompts[1], /SCRIPT_STOP_REASON: direction_cap_reached/);
  assert.equal(state.stage2.anchorPlanning[anchor.anchorId].status, "complete");
  assert.equal(state.stage2.anchorPlanning[anchor.anchorId].reason, "direction_cap_reached");
  assert.equal(state.stage2.anchorPlanning[anchor.anchorId].turns, 2);
});

test("complete review requires all six dimensions and final render is deterministic", () => {
  const { root, state } = buildGraphState();
  const anchor = state.anchors[0];
  const entry = state.entries.find((item) => item.role === "opportunity")!;
  const proposal = validateDirectionProposal(state, anchor.anchorId, {
    selected_entry_ids: [entry.entryId],
    selected_edge_ids: [],
    baseline_ids: [state.baselines[0].baselineId],
    hypothesis: "Runtime bucketing reduces latency while preserving throughput.",
    ablation_plan: ["baseline", "bucket"],
    implementation_plan: ["add dispatch bucket"],
  }, 1);
  state.directions.push(proposal.direction!);
  const claimId = state.claims[0].claimId;
  const dimensions = [
    "scenario_opportunity",
    "baseline_fairness",
    "entry_validity",
    "cross_layer_validity",
    "implementation_reuse",
    "experiment_measurement",
  ] as const;
  const qas: ReviewQuestionAnswer[] = dimensions.map((dimension, index) => ({
    round: index + 1,
    questionId: `Q-${index + 1}`,
    dimension,
    question: `Check ${dimension}`,
    answer: "Bounded evidence answer.",
    conclusion: "partial",
    evidenceRefs: [claimId],
    gaps: [],
  }));
  const incomplete = buildExpertReview(proposal.direction!.directionId, {
    exploration_value: "middle",
  }, qas.slice(0, 5), []);
  assert.match(incomplete.errors.join("\n"), /missing review dimensions/);
  const complete = buildExpertReview(proposal.direction!.directionId, {
    exploration_value: "middle",
    implementation_reuse: "high",
    method_reference: "middle",
    baseline_quality: "fair",
    cross_layer_validity: "valid",
    experiment_readiness: "ready",
    decision: "experiment_candidate",
    rationale: "A falsifiable canary is available.",
    minimum_implementation_plan: ["add bucket"],
    baseline_ablation_matrix: ["baseline vs bucket"],
    metrics_tools: ["latency and throughput"],
    failure_stop_conditions: ["throughput regresses"],
    selected_refs: [],
    alternative_refs: [],
    gaps: [],
  } as unknown as JsonValue, qas, []);
  assert.ok(complete.review);
  state.reviews.push(complete.review!);
  state.provider = {
    codexCliVersion: "test",
    appServerSchemaHash: "test",
    schemaHashAlgorithm: "canonical-json-v1",
    generatedAt: new Date(0).toISOString(),
    supportedMethods: [],
    modelVerified: true,
    supportedEfforts: ["low", "medium", "high", "xhigh"],
    obsidianConfigured: true,
  };
  state.stage1.status = "complete";
  state.stage1.round = 1;
  state.stage1.stopReason = "target_reached";
  state.stage1.acceptedAnchorIds = [anchor.anchorId];
  state.stage1.anchorSpaceVersion = "version";
  state.stage2.status = "complete";
  state.stage2.anchorPlanning[anchor.anchorId] = { status: "complete", reason: "done", turns: 1 };
  state.stage2.directionReview[proposal.direction!.directionId] = { status: "complete", reason: null, rounds: 7 };
  state.status = "complete";
  const report = validateRunState(state);
  assert.deepEqual(report.errors, []);
  state.validation = { ...report, checkedAt: new Date(0).toISOString() };
  const runDir = path.join(root, "render");
  fs.mkdirSync(path.join(runDir, "reviews"), { recursive: true });
  const first = renderRun(state, runDir);
  const second = renderRun(state, runDir);
  assert.equal(first, second);
  assert.match(first, /Global Layer Catalog/);
  assert.match(first, /L1–L6 Intervention Map/);
  assert.match(first, /Baseline \/ Reference Registry/);
});

test("CanonicalStore recovers the last complete checkpoint", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "workflow-store-"));
  const config = testConfig(root);
  const runDir = path.join(root, "run");
  const store = new CanonicalStore(runDir);
  const state = createInitialState("recover", config);
  store.initialize(state);
  state.stage1.round = 1;
  store.save(state, "round_one");
  fs.writeFileSync(store.statePath, "{\"truncated\":", "utf8");
  const recovered = store.load();
  assert.equal(recovered.stage1.round, 1);
});
