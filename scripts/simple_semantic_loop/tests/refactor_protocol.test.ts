import assert from "node:assert/strict";
import { test } from "node:test";
import {
  parseDecisionProtocol,
  parseStrictJsonObject,
} from "../refactor/protocol.ts";
import {
  validateCoreOutputForAction,
  validateReferenceTemplateForAction,
} from "../refactor/schemas.ts";

test("Decision parser extracts one explicit allowed 04-contract literal", () => {
  const parsed = parseDecisionProtocol(
    [
      "我已阅读完整上下文。",
      "- Decision : run_worker",
      "Guidance = 增加 Direction",
      "这行非控制文本保留在原始输出中。",
    ].join("\n"),
    ["RUN_WORKER", "RUN_REVIEWER"],
  );
  assert.deepEqual(parsed.result, {
    decision: "RUN_WORKER",
    guidance: "增加 Direction",
  });
  assert.deepEqual(parsed.errors, []);

  const ambiguous = parseDecisionProtocol(
    "decision = RUN_WORKER\ndecision = RUN_REVIEWER",
    ["RUN_WORKER", "RUN_REVIEWER"],
  );
  assert.equal(ambiguous.result, null);
  assert.match(ambiguous.errors[0]!.message, /exactly one/);

  const proseOnly = parseDecisionProtocol(
    "I considered RUN_WORKER but might choose RUN_REVIEWER.",
    ["RUN_WORKER", "RUN_REVIEWER"],
  );
  assert.equal(proseOnly.result, null);

  const legacy = parseDecisionProtocol(
    "decision = WORKER",
    ["RUN_WORKER"],
  );
  assert.equal(legacy.result, null);
});

test("Decision guidance is optional opaque text", () => {
  const missing = parseDecisionProtocol(
    "decision = RETRY_WORKER",
    ["RETRY_WORKER"],
  );
  assert.deepEqual(missing.result, {
    decision: "RETRY_WORKER",
    guidance: null,
  });

  const noRef = parseDecisionProtocol(
    [
      "decision = RETRY_REVIEWER",
      "guidance = verdict 与正文矛盾，请重新审阅。",
      "guidance = 不改变同一 review target。",
    ].join("\n"),
    ["RETRY_REVIEWER"],
  );
  assert.deepEqual(noRef.result, {
    decision: "RETRY_REVIEWER",
    guidance: "verdict 与正文矛盾，请重新审阅。\n不改变同一 review target。",
  });
});

test("WORK_RESULT parser fails only JSON transport and core-control errors", () => {
  assert.equal(
    parseStrictJsonObject('```json\n{"workOutcome":"READY_FOR_REVIEW"}\n```')
      .parsed,
    null,
  );
  assert.equal(parseStrictJsonObject('{"a":1} trailing').parsed, null);

  assert.deepEqual(
    validateCoreOutputForAction(
      "CREATE_ANCHOR",
      {
        workOutcome: "READY_FOR_REVIEW",
        nextAgent: "REVIEWER",
      },
    ),
    {
      control: {
        role: "WORKER",
        workOutcome: "READY_FOR_REVIEW",
      },
      errors: [],
    },
  );
  assert.ok(
    validateReferenceTemplateForAction(
      "CREATE_ANCHOR",
      {
        workOutcome: "READY_FOR_REVIEW",
        nextAgent: "REVIEWER",
      },
    ).length > 0,
  );

  const missingCore = validateCoreOutputForAction(
    "CREATE_ANCHOR",
    { content: {} },
  );
  assert.equal(missingCore.control, null);
  assert.equal(missingCore.errors[0]!.check, "CORE_CONTROL");
  assert.equal(missingCore.errors[0]!.path, "/workOutcome");

  const unknownCore = validateCoreOutputForAction(
    "REVIEW_ANCHOR",
    { reviewVerdict: "ACCEPTED" },
  );
  assert.equal(unknownCore.control, null);
  assert.match(unknownCore.errors[0]!.message, /PASS, REVISE, REJECT/);
});

test("Reviewer cross-field inconsistency remains semantic, not Script-invalid", () => {
  const passWithBlocking = {
    reviewVerdict: "PASS",
    summary: "错误 PASS",
    findings: [
      {
        severity: "BLOCKING",
        issue: "缺少 baseline",
        basis: "content.baseline 为空泛描述",
        expected: "补充执行路径",
      },
    ],
    queryGaps: [],
  };
  assert.deepEqual(
    validateCoreOutputForAction("REVIEW_ANCHOR", passWithBlocking),
    {
      control: {
        role: "REVIEWER",
        reviewVerdict: "PASS",
      },
      errors: [],
    },
  );
  assert.deepEqual(
    validateReferenceTemplateForAction("REVIEW_ANCHOR", passWithBlocking),
    [],
    "template lint does not pretend to understand verdict semantics",
  );

  const nonStandardFindings = {
    reviewVerdict: "REVISE",
    findings: "需要补查证据",
  };
  assert.equal(
    validateCoreOutputForAction("REVIEW_DIRECTION", nonStandardFindings)
      .errors.length,
    0,
  );
  assert.ok(
    validateReferenceTemplateForAction(
      "REVIEW_DIRECTION",
      nonStandardFindings,
    ).length > 0,
  );
});
