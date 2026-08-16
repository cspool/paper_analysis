import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { test } from "node:test";
import {
  buildContentPrompt,
  buildDecisionPrompt,
} from "../refactor/prompts.ts";
import {
  REFERENCE_TEMPLATE_SCHEMAS,
  validateCoreOutputForAction,
  validateReferenceTemplateForAction,
} from "../refactor/schemas.ts";
import { selectProtocolAgentMessage } from "../refactor/runtime.ts";
import {
  LOOP_DECISIONS,
  REVIEW_VERDICTS,
  TASK_ACTIONS,
  TURN_STATES,
  WORK_OUTCOMES,
} from "../refactor/types.ts";

test("04-contract literals are the only exported control vocabulary", () => {
  assert.deepEqual([...TASK_ACTIONS], [
    "CREATE_ANCHOR",
    "DEEPEN_ANCHOR",
    "CREATE_DIRECTION",
    "DEEPEN_DIRECTION",
    "REVIEW_ANCHOR",
    "REVIEW_DIRECTION",
  ]);
  assert.deepEqual([...LOOP_DECISIONS], [
    "RUN_WORKER",
    "RUN_REVIEWER",
    "RUN_EXP_GOAL",
    "FINISH_WORKFLOW",
    "RETRY_WORKER",
    "RETRY_REVIEWER",
  ]);
  assert.deepEqual([...WORK_OUTCOMES], [
    "READY_FOR_REVIEW",
    "PARTIAL_RESULT",
    "BLOCKED_NO_RESULT",
  ]);
  assert.deepEqual([...REVIEW_VERDICTS], ["PASS", "REVISE", "REJECT"]);
  assert.deepEqual([...TURN_STATES], [
    "RUNNING",
    "INVALID_OUTPUT",
    "PENDING_DECISION",
    "COMMITTED",
    "SUPERSEDED_BY_RETRY",
    "RUNTIME_FAILED",
  ]);
  assert.deepEqual(Object.keys(REFERENCE_TEMPLATE_SCHEMAS).sort(), [
    "review-result-v2",
    "work-result-anchor-v2",
    "work-result-direction-v2",
  ]);
});

test("Worker and Reviewer Prompt carries one T01 path and raw guidance only", () => {
  const prompt = buildContentPrompt({
    skillName: "learning-loop-reviewer",
    taskPath: "/run/tasks/binding/turn_task.json",
    guidance: "从反例角度审阅",
  });
  assert.equal(
    prompt,
    [
      "使用 $learning-loop-reviewer",
      "",
      "本次任务：/run/tasks/binding/turn_task.json",
      "Decision guidance：从反例角度审阅",
      "",
      "按照 Skill 指定的 Result Ref 输出一个 JSON 对象。",
      "",
    ].join("\n"),
  );
  assert.doesNotMatch(
    prompt,
    /state\.json|objects\/index|round|revision|current_conclusions/,
  );
});

test("Decision Prompt exposes one D01 path and allowed literals", () => {
  const prompt = buildDecisionPrompt({
    contextPath: "/run/contexts/c/decision_context.json",
    allowed: ["RUN_WORKER", "RETRY_REVIEWER"],
  });
  assert.match(
    prompt,
    /本次决策上下文：\/run\/contexts\/c\/decision_context\.json/,
  );
  assert.match(prompt, /\[ALLOWED_DECISIONS\]\n- RUN_WORKER\n- RETRY_REVIEWER/);
  assert.match(prompt, /decision = <一个允许的字面量>/);
  assert.match(prompt, /Script 只保存并转发，不解释/);
  assert.doesNotMatch(
    prompt,
    /state\.json|objects\/index|TaskBinding|event|budget/,
  );
});

test("published schemas are Ref templates rather than online gates", () => {
  const review = REFERENCE_TEMPLATE_SCHEMAS["review-result-v2"] as {
    properties: Record<string, unknown>;
    required: string[];
  };
  assert.ok(review.properties.reviewVerdict);
  assert.equal(Object.hasOwn(review.properties, "verdict"), false);
  assert.ok(review.required.includes("reviewVerdict"));

  const nonStandard = {
    workOutcome: "READY_FOR_REVIEW",
    novelContent: { useful: true },
  };
  assert.deepEqual(
    validateCoreOutputForAction("CREATE_ANCHOR", nonStandard),
    {
      control: {
        role: "WORKER",
        workOutcome: "READY_FOR_REVIEW",
      },
      errors: [],
    },
  );
  const advisories = validateReferenceTemplateForAction(
    "CREATE_ANCHOR",
    nonStandard,
  );
  assert.ok(advisories.length > 0);
  assert.match(
    advisories.map((item) => `${item.path} ${item.message}`).join("\n"),
    /reference-template property is missing|outside the reference template/,
  );
});

test("Codex Turn transport selects the explicit final_answer phase", () => {
  const progress =
    '{"workOutcome":"PARTIAL_RESULT","content":null,"evidence":[],"unresolved":["working"]}';
  const final =
    '{"workOutcome":"READY_FOR_REVIEW","content":{"name":"final"},"evidence":[],"unresolved":[]}';
  assert.deepEqual(
    selectProtocolAgentMessage(
      {
        items: [
          { type: "agentMessage", text: final, phase: "final_answer" },
          { type: "mcpToolCall", text: "ignored" },
          { type: "agentMessage", text: progress, phase: "commentary" },
        ],
      },
      [],
    ),
    { text: final, error: null },
  );
  assert.deepEqual(
    selectProtocolAgentMessage(null, [{ text: final, phase: null }]),
    { text: final, error: null },
  );
  assert.match(
    selectProtocolAgentMessage(null, [
      { text: progress, phase: null },
      { text: final, phase: null },
    ]).error!,
    /phase-unknown.*ambiguous/,
  );
  assert.match(
    selectProtocolAgentMessage(
      {
        items: [
          { type: "agentMessage", text: progress, phase: "commentary" },
        ],
      },
      [],
    ).error!,
    /without a final_answer/,
  );
  assert.match(
    selectProtocolAgentMessage(null, [
      { text: progress, phase: "final_answer" },
      { text: final, phase: "final_answer" },
    ]).error!,
    /2 final_answer.*ambiguous/,
  );
});

test("v8 Skills keep semantic review and EXP handoff outside core content fields", () => {
  const projectRoot = resolve(import.meta.dirname, "../../..");
  const decision = readFileSync(
    resolve(projectRoot, ".codex/skills/learning-loop-decision/SKILL.md"),
    "utf8",
  );
  const worker = readFileSync(
    resolve(projectRoot, ".codex/skills/learning-loop-worker/SKILL.md"),
    "utf8",
  );
  const reviewer = readFileSync(
    resolve(projectRoot, ".codex/skills/learning-loop-reviewer/SKILL.md"),
    "utf8",
  );
  const valueQuestions = readFileSync(
    resolve(
      projectRoot,
      ".codex/skills/learning-loop-reviewer/references/optimization_value_questions_v1.md",
    ),
    "utf8",
  );
  assert.match(decision, /openQueryGaps=\[\].*insufficient/s);
  assert.match(decision, /BLOCKED_NO_RESULT/);
  assert.match(decision, /One recent credible negative probe/);
  assert.match(decision, /Do not search or deep-read papers/);
  assert.match(worker, /inputs\.researchMemory/);
  assert.match(worker, /bounded Topic convergence probe/);
  assert.match(worker, /future experiment handoff/);
  assert.match(worker, /closest existing method baseline/);
  assert.match(worker, /inputs\.experimentResults/);
  assert.match(worker, /inputs\.negativeExperimentHistoryRef/);
  assert.match(worker, /baselineChange.*causal lever.*preserved boundary/s);
  assert.match(reviewer, /inputs\.previousReview/);
  assert.match(reviewer, /pass\/fail result, baseline reproducibility/);
  assert.match(reviewer, /optimization_value_questions_v1\.md/);
  assert.match(reviewer, /negativeExperimentHistoryRef/);
  assert.match(reviewer, /After two credible same-family negatives/);
  assert.match(valueQuestions, /Optimization opportunity and performance baseline/);
  assert.match(valueQuestions, /Closest method baseline and Direction difference/);
  assert.match(valueQuestions, /Reference experiment and environment reuse/);
  assert.match(decision, /RUN_EXP_GOAL/);
  assert.match(decision, /second credible negative/);
  assert.match(decision, /POST_EXP_REVIEWER/);
  assert.doesNotMatch(
    JSON.stringify(REFERENCE_TEMPLATE_SCHEMAS),
    /quietExpansion|globalGap|saturation|memoryUpdate|trajectoryUpdate|negativeLesson|familyId/,
  );
});
