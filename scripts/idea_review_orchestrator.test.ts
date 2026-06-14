import assert from "node:assert/strict";
import {
  buildAnchor,
  buildAAIsolatedRecoveryPrompt,
  buildSessionArgs,
  buildProtocolRepairPrompt,
  canAutoResumeInterruptedAA,
  canAutoResumeInterruptedQA,
  formatAAInitPayload,
  formatAAOutput,
  formatQAQuestionPayload,
  isCompleteProtocolResponse,
  parseAAInitMarker,
  parseAAMarker,
  parseFinalJudgmentFromRawJsonl,
  parseLastQAQuestionFromRawJsonl,
  parseQAMarker,
  describeAAProtocolError,
  describeLoopSemantics,
  reviewPathForTitle,
  selectSessionOutput,
  shouldUseIsolatedAARecovery,
  summarizeApiRetry,
} from "./idea_review_orchestrator";

assert.equal(summarizeApiRetry(null), "no API retry telemetry");
assert.equal(
  summarizeApiRetry({
    attempt: 10,
    max_retries: 10,
    retry_delay_ms: 38110.4,
    error_status: null,
    error: "unknown",
  }),
  "last API retry=10/10, next delay=38110ms, status=null, error=unknown",
);

const resumedArgs = buildSessionArgs("11111111-1111-4111-8111-111111111111", "qa", 5, true);
assert.deepEqual(
  resumedArgs.slice(resumedArgs.indexOf("--resume"), resumedArgs.indexOf("--resume") + 2),
  ["--resume", "11111111-1111-4111-8111-111111111111"],
);
assert.equal(resumedArgs.includes("--session-id"), false);
assert.deepEqual(
  resumedArgs.slice(resumedArgs.indexOf("--max-budget-usd"), resumedArgs.indexOf("--max-budget-usd") + 2),
  ["--max-budget-usd", "5"],
);
const noToolOverrideArgs = buildSessionArgs("aa", "aa", 5, false, "NoToolA,NoToolB", "");
assert.deepEqual(
  noToolOverrideArgs.slice(
    noToolOverrideArgs.indexOf("--disallowedTools"),
    noToolOverrideArgs.indexOf("--disallowedTools") + 2,
  ),
  ["--disallowedTools", "NoToolA,NoToolB"],
);
assert.deepEqual(
  noToolOverrideArgs.slice(noToolOverrideArgs.indexOf("--tools"), noToolOverrideArgs.indexOf("--tools") + 2),
  ["--tools", ""],
);
assert.equal(shouldUseIsolatedAARecovery("", {
  subtype: "success",
  is_error: false,
  stop_reason: "tool_use",
}), true);
assert.equal(shouldUseIsolatedAARecovery("", {
  subtype: "success",
  is_error: false,
  stop_reason: null,
}), true);
assert.equal(shouldUseIsolatedAARecovery("visible answer", {
  subtype: "success",
  is_error: false,
  stop_reason: "tool_use",
}), false);
assert.equal(shouldUseIsolatedAARecovery("", {
  subtype: "success",
  is_error: false,
  stop_reason: "end_turn",
}), false);
assert.equal(canAutoResumeInterruptedQA({
  protocol_version: 8,
  idea_note_path: "_auto_stub.md",
  idea_note_title: "HATB (mPLUG-Owl3)",
  qa_session_id: "qa",
  aa_session_id: "aa",
  round: 2,
  qa_history: [{ round: 1, question: "q1", answer: "a1" }, { round: 2, question: "q2", answer: "a2" }],
  qa_loaded_references: ["动态(调度/并发)的背景/需求"],
  qa_next_entry: "[LOOP: §DIM_EVAL | await=AA_OUTPUT | dimension=背景与需求 | round=2]",
  aa_next_entry: "[LOOP: §ANSWER | await=QA_QUESTION | completed_round=2]",
  final_judgment: null,
  started_at: 0,
  updated_at: 0,
}), true);
assert.equal(canAutoResumeInterruptedAA({
  protocol_version: 8,
  idea_note_path: "_auto_stub.md",
  idea_note_title: "HATB (mPLUG-Owl3)",
  qa_session_id: "qa",
  aa_session_id: "aa",
  round: 6,
  qa_history: [
    { round: 1, question: "q1", answer: "a1" },
    { round: 2, question: "q2", answer: "a2" },
    { round: 3, question: "q3", answer: "a3" },
    { round: 4, question: "q4", answer: "a4" },
    { round: 5, question: "q5", answer: "a5" },
  ],
  qa_loaded_references: [],
  qa_next_entry: "[LOOP: §DIM_EVAL | await=AA_OUTPUT | dimension=实验证据 | round=6]",
  aa_next_entry: "[LOOP: §ANSWER | await=QA_QUESTION | completed_round=5]",
  final_judgment: null,
  started_at: 0,
  updated_at: 0,
}), true);

const numericQALoopSemantics = describeLoopSemantics(
  "Question Agent",
  "[LOOP: §7 | await=AA_OUTPUT | dimension=架构影响 | round=5]",
  "AA_OUTPUT",
);
assert.match(numericQALoopSemantics, /§7 \/ §DIM_EVAL — 评估回答/);
assert.match(numericQALoopSemantics, /已经收到 Answer Agent 的 `AA_OUTPUT` 回答/);
assert.match(numericQALoopSemantics, /当前维度为「架构影响」/);
assert.match(numericQALoopSemantics, /当前轮次为 5/);
assert.match(numericQALoopSemantics, /评估 Answer Agent 回答/);

const legacyAALoopSemantics = describeLoopSemantics(
  "Answer Agent",
  "[LOOP: §ANSWER | await=QA_QUESTION | completed_round=4]",
  "QA_QUESTION",
);
assert.match(legacyAALoopSemantics, /§2 \/ §ANSWER — 接收问题并回答/);
assert.match(legacyAALoopSemantics, /已经收到 Question Agent 的 `QA_QUESTION` 问题/);
assert.match(legacyAALoopSemantics, /此前已完成 round 4/);
assert.match(legacyAALoopSemantics, /提取当前问题/);

const anchoredAAInput = buildAnchor(
  "Answer Agent",
  "[LOOP: §2 | await=QA_QUESTION | completed_round=4]",
  "QA_QUESTION",
  "runtime guard",
);
assert.doesNotMatch(anchoredAAInput, /\[LOOP:/);
assert.match(anchoredAAInput, /本次执行语义/);
assert.match(anchoredAAInput, /§2 \/ §ANSWER — 接收问题并回答/);
assert.match(anchoredAAInput, /已经收到 Question Agent 的 `QA_QUESTION` 问题/);
assert.match(anchoredAAInput, /runtime guard/);
assert.match(anchoredAAInput, /── 协议载荷 ──/);

const aaInit = formatAAInitPayload({
  paper_title: "Kitsune: Enabling Dataflow Execution on GPUs",
  paper_subdir: "paper_secs/secs_multimodal_kernel/Kitsune Enabling Dataflow Execution on GPUs/",
});
assert.deepEqual(parseAAInitMarker(aaInit), {
  paper_title: "Kitsune: Enabling Dataflow Execution on GPUs",
  paper_subdir: "paper_secs/secs_multimodal_kernel/Kitsune Enabling Dataflow Execution on GPUs/",
});
assert.equal(parseAAInitMarker(formatAAInitPayload({
  paper_title: "<论文主文件 H1 中的真实完整标题>",
  paper_subdir: "<vault-relative paper_secs 子目录路径>",
})), null);
assert.match(
  reviewPathForTitle("Kitsune: Enabling Dataflow Execution on GPUs"),
  /Kitsune: Enabling Dataflow Execution on GPUs_review\.md$/,
);
const thinkingOnlyInit = `${aaInit}
[LOOP: §ANSWER | await=QA_QUESTION | loaded_paths=7]`;
assert.equal(parseAAInitMarker(selectSessionOutput("", [thinkingOnlyInit], "aa"))?.paper_title,
  "Kitsune: Enabling Dataflow Execution on GPUs");

const recoverableQuestion = `___QA_QUESTION___
{
  "round": 5,
  "question_level": 2,
  "question_category": "影响并发的架构/机制"
}
___QA_QUESTION_TEXT___
请补充架构限制的量化边界。
___QA_QUESTION_TEXT_END___

\`\`\`
[LOOP: §DIM_EVAL | await=AA_OUTPUT | dimension=影响并发的架构/机制 | round=5]
\`\`\``;

const questionMarker = parseQAMarker(recoverableQuestion);
assert.equal(questionMarker?.type, "question");
assert.equal(questionMarker?.type === "question" ? questionMarker.q.round : null, 5);
assert.match(
  questionMarker?.type === "question" ? formatQAQuestionPayload(questionMarker.q) : "",
  /___QA_QUESTION_END___/,
);

const truncatedQuestion = recoverableQuestion.replace("___QA_QUESTION_TEXT_END___", "");
assert.equal(parseQAMarker(truncatedQuestion), null);
const recoveredRound5Question = parseLastQAQuestionFromRawJsonl([
  JSON.stringify({ type: "assistant", message: { content: [{ type: "text", text: recoverableQuestion }] } }),
  JSON.stringify({ type: "result", result: "" }),
].join("\n"), 5);
assert.equal(recoveredRound5Question?.round, 5);
assert.match(recoveredRound5Question?.question || "", /架构限制/);

const thinkingOnlyReferenceRequest = `**§DIM_EVAL**
内部评估和 review_material 省略。

___QA_REFERENCE_REQUEST___
{ "round": 6, "question_category": "影响并发的架构/机制" }
___QA_REFERENCE_REQUEST_END___

[LOOP: §DIM_ASK | await=QA_REFERENCE | dimension=影响并发的架构/机制 | round=6]`;
const recoveredReferenceRequest = selectSessionOutput("", [thinkingOnlyReferenceRequest], "qa");
assert.equal(parseQAMarker(recoveredReferenceRequest)?.type, "reference_request");
assert.match(recoveredReferenceRequest, /\[LOOP: §DIM_ASK \| await=QA_REFERENCE/);
assert.doesNotMatch(recoveredReferenceRequest, /内部评估/);
assert.equal(selectSessionOutput(recoverableQuestion, [thinkingOnlyReferenceRequest], "qa"), recoverableQuestion);

const stoppedAtInternalGoto = `**§DIM_EVAL 执行 — 评估回答**

判定：review_ready。
→ GOTO §DIM_NEXT

**§DIM_NEXT 执行**
下一维度：并发方法的应用和实现。
→ GOTO §DIM_REF`;
assert.equal(isCompleteProtocolResponse(stoppedAtInternalGoto, "qa"), false);
assert.match(buildProtocolRepairPrompt("qa"), /从刚才尚未完成的线性步骤继续执行/);
assert.match(buildProtocolRepairPrompt("qa"), /___QA_REFERENCE_REQUEST___/);
const aaRepairPrompt = buildProtocolRepairPrompt("aa", recoverableQuestion.replace('"round": 5', '"round": 6'));
assert.match(aaRepairPrompt, /禁止调用任何工具/);
assert.match(aaRepairPrompt, /\{ "round": 6 \}/);
assert.match(aaRepairPrompt, /\[LOOP: §2 \| await=QA_QUESTION \| completed_round=6\]/);
assert.equal(isCompleteProtocolResponse(thinkingOnlyReferenceRequest, "qa"), true);
const isolatedRecoveryPrompt = buildAAIsolatedRecoveryPrompt({
  protocol_version: 8,
  idea_note_path: "_auto_stub.md",
  idea_note_title: "HATB (mPLUG-Owl3)",
  paper_title: "MPLUG-Owl3: Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models",
  paper_subdir: "paper_secs/secs_video_image/mPLUG-Owl3/",
  qa_session_id: "qa",
  aa_session_id: "aa",
  round: 6,
  qa_history: [{ round: 5, question: "已有问题", answer: "已有回答" }],
  qa_loaded_references: [],
  qa_next_entry: "[LOOP: §DIM_EVAL | await=AA_OUTPUT | round=6]",
  aa_next_entry: "[LOOP: §ANSWER | await=QA_QUESTION | completed_round=5]",
  final_judgment: null,
  started_at: 0,
  updated_at: 0,
}, {
  round: 6,
  question: "当前问题",
  question_level: 2,
  question_category: "影响并发的架构/机制",
});
assert.match(isolatedRecoveryPrompt, /禁止调用任何工具/);
assert.match(isolatedRecoveryPrompt, /已有回答/);
assert.match(isolatedRecoveryPrompt, /当前问题/);
assert.match(isolatedRecoveryPrompt, /recovery=no_tools/);
assert.match(isolatedRecoveryPrompt, /本次执行语义/);
assert.match(isolatedRecoveryPrompt, /§2 \/ §ANSWER — 接收问题并回答/);

const judgmentWithMarkdownBraces = `___JUDGMENT_COMPLETE___
{
  "relevance": "high",
  "reference_value": "high",
  "depth_value": "high"
}

**复现指南**：使用 batch={1,2,4,8,16}，并保留 JSON 示例 {"mode":"AOT"}。
___JUDGMENT_COMPLETE_END___

[LOOP: §TERMINATED | done]`;
const judgmentMarker = parseQAMarker(judgmentWithMarkdownBraces);
assert.equal(judgmentMarker?.type, "judgment");
assert.equal(judgmentMarker?.type === "judgment" ? judgmentMarker.j.relevance : null, "high");
assert.match(judgmentMarker?.type === "judgment" ? judgmentMarker.j._body : "", /batch=\{1,2,4,8,16\}/);
assert.equal(
  parseQAMarker(selectSessionOutput(judgmentWithMarkdownBraces, [], "qa"))?.type,
  "judgment",
);
const recoveredLoggedJudgment = parseFinalJudgmentFromRawJsonl([
  JSON.stringify({
    type: "assistant",
    message: { content: [{ type: "text", text: judgmentWithMarkdownBraces }] },
  }),
  JSON.stringify({ type: "result", result: judgmentWithMarkdownBraces }),
].join("\n"));
assert.equal(recoveredLoggedJudgment?.relevance, "high");
assert.match(recoveredLoggedJudgment?._body || "", /batch=\{1,2,4,8,16\}/);

const legacyAnswer = `___AA_OUTPUT_START___
{
  "round": 5,
  "sources": [],
  "information_gaps": []
}
___AA_ANSWER_START___
这是回答正文。
___AA_ANSWER_END___`;

const legacyAnswerMarker = parseAAMarker(legacyAnswer);
assert.equal(legacyAnswerMarker?.round, 5);

const roundTripAnswer = formatAAOutput({
  round: 6,
  answer: "这是带有 `\"quoted\"` 文本的回答。",
  sources: ["paper_secs/Paper with \"quotes\".md"],
  information_gaps: [
    "论文只假设 \"无性能退化\"，未提供实测。",
    "描述为 \"modest change\"，但没有 cycle 数。",
  ],
});
const answerMarker = parseAAMarker(roundTripAnswer);
assert.deepEqual(answerMarker, {
  round: 6,
  answer: "这是带有 `\"quoted\"` 文本的回答。",
  sources: ["paper_secs/Paper with \"quotes\".md"],
  information_gaps: [
    "论文只假设 \"无性能退化\"，未提供实测。",
    "描述为 \"modest change\"，但没有 cycle 数。",
  ],
});
assert.match(roundTripAnswer, /___AA_SOURCES_START___/);
assert.match(roundTripAnswer, /___AA_GAPS_START___/);

const thinkingOnlyAnswer = `${recoverableQuestion}
${roundTripAnswer}
[LOOP: §ANSWER | await=QA_QUESTION | completed_round=6]`;
const recoveredAnswer = selectSessionOutput("", [thinkingOnlyAnswer], "aa");
assert.equal(parseAAMarker(recoveredAnswer)?.round, 6);
assert.doesNotMatch(recoveredAnswer, /___QA_QUESTION___/);
assert.equal(
  parseAAMarker(selectSessionOutput("", [`${aaInit}\n${thinkingOnlyAnswer}`], "aa"))?.round,
  6,
);

const invalidLegacyAnswer = legacyAnswer.replace(
  '"information_gaps": []',
  '"information_gaps": ["论文只假设 "无性能退化""]',
);
assert.equal(parseAAMarker(invalidLegacyAnswer), null);
assert.match(describeAAProtocolError(invalidLegacyAnswer), /invalid legacy JSON metadata/);

const truncatedAnswer = roundTripAnswer.replace("___AA_ANSWER_END___", "");
assert.equal(parseAAMarker(truncatedAnswer), null);

console.log("idea_review_orchestrator protocol parser tests passed");
