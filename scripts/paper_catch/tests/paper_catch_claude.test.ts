import assert from "node:assert/strict";
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { PaperCatchController } from "../controller.ts";
import { extractClaudeStructuredOutput } from "../codex_filter.ts";
import { PaperCatchStore } from "../store.ts";
import type { PaperCandidate, PaperCatchRun, SourceSnapshot } from "../types.ts";

const here = dirname(fileURLToPath(import.meta.url));
const schemaPath = resolve(here, "..", "batch_result.schema.json");

test("extractClaudeStructuredOutput prefers structured_output and reports failed turns", () => {
  const ok = extractClaudeStructuredOutput(JSON.stringify({
    type: "result",
    subtype: "success",
    is_error: false,
    result: "```json\n{\"fallback\":true}\n```",
    structured_output: { schemaVersion: "paper-catch-batch-result-v1" },
  }));
  assert.equal(ok.error, null);
  assert.deepEqual(ok.value, { schemaVersion: "paper-catch-batch-result-v1" });

  const fenced = extractClaudeStructuredOutput(JSON.stringify({
    type: "result",
    subtype: "success",
    is_error: false,
    result: "```json\n{\"fallback\":true}\n```",
  }));
  assert.equal(fenced.error, null);
  assert.deepEqual(fenced.value, { fallback: true });

  const failed = extractClaudeStructuredOutput(JSON.stringify({
    type: "result",
    subtype: "error_max_turns",
    is_error: true,
    result: "ran out of turns",
  }));
  assert.match(failed.error ?? "", /error_max_turns/);
  assert.match(extractClaudeStructuredOutput("not json").error ?? "", /not a JSON envelope/);
});

test("controller runs batches through a fake Claude CLI with structured output", async () => {
  const root = mkdtempSync(resolve(tmpdir(), "paper-catch-claude-"));
  const output = resolve(root, "paper_catch");
  const project = resolve(root, "project");
  mkdirSync(project, { recursive: true });
  mkdirSync(output, { recursive: true });
  writeFileSync(resolve(output, "PAPER_ENTRY_TEMPLATE.md"), "### template\n");
  const argsLog = resolve(root, "claude-args.json");
  const fake = resolve(root, "fake-claude.mjs");
  writeFileSync(fake, `#!/usr/bin/env node
import fs from 'node:fs';
const args=process.argv.slice(2);
fs.writeFileSync(${JSON.stringify(argsLog)},JSON.stringify(args));
const prompt=fs.readFileSync(0,'utf8');
const taskPath=/本批冻结任务：([^\\n]+)/.exec(prompt)[1].trim();
const task=JSON.parse(fs.readFileSync(taskPath,'utf8'));
const schema=JSON.parse(args[args.indexOf('--json-schema')+1]);
if(schema.$schema) throw new Error('$schema must be stripped');
const selected=task.candidates.map((candidate)=>({
  candidateId:candidate.candidateId,
  layers:['SYSTEM'],
  priority:'MEDIUM',
  relevanceReason:'agent-driven kernel tuning',
  performanceProblem:'kernel latency',
  conciseContribution:'LLM agent rewrites kernels',
  openSource:'UNKNOWN',
  codeUrl:null,
  newnessReason:'new Git update'
}));
const structured={schemaVersion:'paper-catch-batch-result-v1',runId:task.runId,batchId:task.batchId,batchSummary:'fake claude batch',selected,rejected:[]};
console.log(JSON.stringify({type:'result',subtype:'success',is_error:false,result:JSON.stringify(structured),structured_output:structured}));
`);
  chmodSync(fake, 0o755);

  const store = new PaperCatchStore(output);
  store.initialize();
  const runId = "20260915_010203";
  const runRoot = `.runs/${runId}`;
  const candidates: PaperCandidate[] = ["paper-1", "paper-2", "paper-3"].map((id, index) => ({
    candidateId: id,
    title: `Agentic Kernel ${index + 1}`,
    normalizedTitle: `agentic kernel ${index + 1}`,
    paperUrl: `https://arxiv.org/abs/2609.0000${index + 1}`,
    codeUrls: [],
    urls: [],
    latestCommittedAt: "2026-09-01T00:00:00Z",
    sourceRefs: [{
      sourceId: "src",
      sourceUrl: "https://github.com/o/r#llm-for-systems",
      repositoryUrl: "https://github.com/o/r",
      fragment: "llm-for-systems",
      filePath: "README.md",
      line: index + 1,
      addedLine: "- x",
      context: "",
      commitSha: "abcdef1234567890",
      committedAt: "2026-09-01T00:00:00Z",
      commitSubject: "add",
    }],
  }));
  const snapshots = [{
    sourceId: "src",
    url: "https://github.com/o/r#llm-for-systems",
    cloneUrl: "https://github.com/o/r.git",
    fragment: "llm-for-systems",
    baselineHead: "1111111111",
    baselineMode: "INITIAL_LOOKBACK",
    currentHead: "2222222222",
    checkedAt: "2026-09-15T00:00:00Z",
    changed: true,
    commits: [],
    changedMarkdownFiles: ["README.md"],
    stats: { markdownFilesChanged: 1, linesAdded: 3, linesDeleted: 0, candidateLines: 3 },
    warnings: [],
  }] as unknown as SourceSnapshot[];
  store.writeImmutableJson(`${runRoot}/inputs/config.json`, {
    schemaVersion: "paper-catch-config-v1",
    configPath: "fixture",
    configHash: "hash",
    urls: [],
    interest: "agent-driven performance optimization",
  });
  store.writeImmutableText(`${runRoot}/inputs/PAPER_ENTRY_TEMPLATE.md`, "### template\n");
  store.writeImmutableText(`${runRoot}/inputs/batch_result.schema.json`, readFileSync(schemaPath, "utf8"));
  store.writeImmutableJson(`${runRoot}/candidates.json`, candidates);
  store.writeImmutableJson(`${runRoot}/sources/source_snapshots.json`, snapshots);
  const run: PaperCatchRun = {
    schemaVersion: "paper-catch-run-v1",
    runId,
    mode: "FULL",
    createdAt: "2026-09-15T01:02:03Z",
    updatedAt: "2026-09-15T01:02:03Z",
    status: "BATCHING",
    phase: "CANDIDATES_FROZEN",
    projectRoot: project,
    outputDir: output,
    configRef: `${runRoot}/inputs/config.json`,
    configHash: "hash",
    interest: "agent-driven performance optimization",
    previousReportRef: null,
    previousManifestRef: null,
    baselineTimestamp: "2026-09-08T00:00:00Z",
    lookbackDays: 7,
    batchSize: 2,
    provider: "claude",
    model: null,
    useWebSearch: true,
    sourceSnapshotsRef: `${runRoot}/sources/source_snapshots.json`,
    candidatesRef: `${runRoot}/candidates.json`,
    batches: [],
    reportRef: null,
    manifestRef: null,
    error: null,
  };
  store.writeRun(run);
  store.setActiveRun(runId);

  const outcome = await new PaperCatchController(store, {
    projectRoot: project,
    configPath: "unused",
    outputDir: output,
    batchSize: 2,
    lookbackDays: 7,
    provider: "claude",
    model: null,
    codexBin: "codex-should-not-run",
    claudeBin: fake,
    useWebSearch: true,
    maxAttemptsPerInvocation: 1,
    codexTimeoutMs: 30_000,
    scanOnly: false,
  }).run();

  assert.equal(outcome.status, "COMPLETED");
  assert.equal(outcome.totalBatches, 2);
  assert.equal(outcome.completedBatches, 2);
  const args = JSON.parse(readFileSync(argsLog, "utf8")) as string[];
  assert.equal(args[0], "-p");
  assert.equal(args[args.indexOf("--model") + 1], "claude-sonnet-5");
  assert.equal(args[args.indexOf("--output-format") + 1], "json");
  assert.equal(args[args.indexOf("--tools") + 1], "Read,WebSearch,WebFetch");
  assert.ok(args.includes("--strict-mcp-config"));
  const report = store.readText(`${runId}.md`);
  assert.match(report, /Agentic Kernel 1/);
  assert.match(report, /Agentic Kernel 3/);
  const attempt = store.readJson<{ valid: boolean; refs: { output: string } }>(
    `${runRoot}/batches/batch-001/attempt-01/attempt.json`,
  );
  assert.equal(attempt.valid, true);
  assert.equal(store.readJson<{ batchSummary: string }>(attempt.refs.output).batchSummary, "fake claude batch");
});
