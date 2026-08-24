import assert from "node:assert/strict";
import { chmodSync, mkdtempSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { parseCatchConfig, sourceSpecFromUrl } from "../config.ts";
import { PaperCatchController } from "../controller.ts";
import { validateBatchResult } from "../codex_filter.ts";
import {
  collectSourceUpdate,
  extractPaperCandidateFromLine,
  findGithubSectionRange,
  mergePaperCandidates,
  normalizeTitle,
} from "../git_sources.ts";
import { PaperCatchStore } from "../store.ts";
import type {
  BatchTask,
  PaperCandidate,
  PaperCatchRun,
  SourceSnapshot,
} from "../types.ts";

const here = dirname(fileURLToPath(import.meta.url));
const schemaPath = resolve(here, "../batch_result.schema.json");

test("parses URL lines and the free-form Chinese interest section", () => {
  const root = temporary("config");
  const path = resolve(root, "Catch_Paper_Urls.md");
  writeFileSync(path, [
    "https://github.com/owner/repo",
    "- https://github.com/owner/other#serving",
    "",
    "感兴趣主题：推理性能加速，不要 training-only。",
    "优先有开源代码。",
  ].join("\n"));
  const config = parseCatchConfig(path);
  assert.deepEqual(config.urls, [
    "https://github.com/owner/repo",
    "https://github.com/owner/other#serving",
  ]);
  assert.match(config.interest, /不要 training-only/);
  assert.match(config.interest, /开源代码/);
  assert.equal(config.configHash.length, 64);
});

test("normalizes GitHub source URLs while preserving fragment identity", () => {
  const source = sourceSpecFromUrl("https://github.com/Owner/Repo#Multi-Modal-Serving-Systems");
  assert.equal(source.cloneUrl, "https://github.com/Owner/Repo.git");
  assert.equal(source.fragment, "Multi-Modal-Serving-Systems");
  assert.match(source.sourceId, /^owner-repo-/);
});

test("extracts linked and venue-prefixed paper titles but rejects descriptions", () => {
  const linked = extractPaperCandidateFromLine(
    "- [arxiv'26] [FastServe: Efficient LLM Inference](https://arxiv.org/abs/2601.00001) [[Code](https://github.com/x/fastserve)]",
  );
  assert.equal(linked?.title, "FastServe: Efficient LLM Inference");
  assert.equal(linked?.paperUrl, "https://arxiv.org/abs/2601.00001");
  assert.deepEqual(linked?.codeUrls, ["https://github.com/x/fastserve"]);

  const workshop = extractPaperCandidateFromLine(
    "- [SAA'25] Useful Agentic AI: A Systems Outlook",
  );
  assert.equal(workshop?.title, "Useful Agentic AI: A Systems Outlook");
  assert.equal(
    extractPaperCandidateFromLine("    - Improves throughput with a new scheduler."),
    null,
  );
});

test("finds a GitHub-style anchored section and stops at its sibling", () => {
  const lines = [
    "# Papers",
    "## Training",
    "- A",
    "## Multi-Modal Serving Systems",
    "- B",
    "### Runtime",
    "- C",
    "## Hardware",
    "- D",
  ];
  assert.deepEqual(findGithubSectionRange(lines, "multi-modal-serving-systems"), [5, 7]);
});

test("merges duplicate candidates across sources by normalized title", () => {
  const first = candidate("paper-a", "Fast Serve: LLM Inference", "2026-08-20T00:00:00Z");
  const second = candidate("paper-b", "Fast-Serve LLM Inference", "2026-08-21T00:00:00Z");
  second.normalizedTitle = normalizeTitle(second.title);
  first.normalizedTitle = normalizeTitle(first.title);
  const merged = mergePaperCandidates([[first], [second]]);
  assert.equal(merged.length, 1);
  assert.equal(merged[0]?.sourceRefs.length, 2);
  assert.equal(merged[0]?.latestCommittedAt, "2026-08-21T00:00:00Z");
});

test("Git baseline title subtraction ignores a list reorganization and keeps only a new paper", () => {
  const root = temporary("git-reorganization");
  const repository = resolve(root, "source");
  const cache = resolve(root, "cache");
  mkdirSync(repository, { recursive: true });
  git(repository, ["init", "-b", "main"]);
  git(repository, ["config", "user.email", "paper-catch@example.test"]);
  git(repository, ["config", "user.name", "Paper Catch Test"]);
  writeFileSync(resolve(repository, "README.md"), [
    "# Papers",
    "- [AlphaServe: Fast LLM Inference Runtime](https://arxiv.org/abs/2601.00001)",
    "- [BetaCache: Efficient KV Cache Management](https://arxiv.org/abs/2601.00002)",
  ].join("\n"));
  git(repository, ["add", "README.md"]);
  git(repository, ["commit", "-m", "initial list"], "2026-07-01T00:00:00Z");
  writeFileSync(resolve(repository, "README.md"), [
    "# Papers",
    "## Runtime Systems",
    "- [BetaCache: Efficient KV Cache Management](https://arxiv.org/abs/2601.00002)",
    "- [AlphaServe: Fast LLM Inference Runtime](https://arxiv.org/abs/2601.00001)",
    "- [GammaEngine: Low-Latency Model Serving](https://arxiv.org/abs/2608.00003)",
  ].join("\n"));
  git(repository, ["add", "README.md"]);
  git(repository, ["commit", "-m", "reorganize and add gamma"], "2026-08-23T00:00:00Z");

  const collected = collectSourceUpdate({
    spec: {
      sourceId: "fixture-source",
      url: "https://github.com/example/fixture",
      cloneUrl: repository,
      fragment: null,
      displayName: "example/fixture",
    },
    cacheRoot: cache,
    baselineTimestamp: "2026-08-01T00:00:00Z",
    previousManifest: null,
    checkedAt: "2026-08-24T00:00:00Z",
  });
  assert.deepEqual(collected.candidates.map((item) => item.title), [
    "GammaEngine: Low-Latency Model Serving",
  ]);
});

test("batch validation requires every frozen candidate exactly once", () => {
  const task = batchTask([candidate("paper-a", "A Fast Inference Paper"), candidate("paper-b", "A Training Paper")]);
  const valid = {
    schemaVersion: "paper-catch-batch-result-v1",
    runId: task.runId,
    batchId: task.batchId,
    batchSummary: "one selected, one rejected",
    selected: [{
      candidateId: "paper-a",
      layers: ["SYSTEM"],
      priority: "HIGH",
      relevanceReason: "serving acceleration",
      performanceProblem: "decode latency",
      conciseContribution: "changes scheduling",
      openSource: "UNKNOWN",
      codeUrl: null,
      newnessReason: "new commit",
    }],
    rejected: [{ candidateId: "paper-b", reason: "training-only" }],
  };
  assert.deepEqual(validateBatchResult(valid, task), []);
  assert.match(
    validateBatchResult({ ...valid, rejected: [] }, task).join("\n"),
    /not classified: paper-b/,
  );
});

test("store lock rejects a second live owner and can be reacquired after release", () => {
  const output = temporary("lock");
  const first = new PaperCatchStore(output);
  const second = new PaperCatchStore(output);
  first.acquireLock();
  assert.throws(() => second.acquireLock(), /already running/);
  first.releaseLock();
  second.acquireLock();
  second.releaseLock();
});

test("controller executes multiple fresh fake-Codex batches then publishes one report", async () => {
  const root = temporary("controller");
  const output = resolve(root, "paper_catch");
  const project = resolve(root, "project");
  mkdirSync(project, { recursive: true });
  mkdirSync(output, { recursive: true });
  writeFileSync(resolve(output, "PAPER_ENTRY_TEMPLATE.md"), "### [论文标题](论文链接)\n");
  const fake = resolve(root, "fake-codex.mjs");
  writeFileSync(fake, `#!/usr/bin/env node
import fs from 'node:fs';
const args=process.argv.slice(2);
const output=args[args.indexOf('--output-last-message')+1];
const prompt=args.at(-1);
const taskPath=/本批冻结任务：([^\\n]+)/.exec(prompt)[1].trim();
const task=JSON.parse(fs.readFileSync(taskPath,'utf8'));
const selected=task.candidates.map((candidate)=>({
  candidateId:candidate.candidateId,
  layers:['SYSTEM'],
  priority:'HIGH',
  relevanceReason:'matches inference acceleration',
  performanceProblem:'runtime latency',
  conciseContribution:'changes the execution path',
  openSource:candidate.codeUrls.length?'YES':'UNKNOWN',
  codeUrl:candidate.codeUrls[0]??null,
  newnessReason:'new Git update'
}));
const result={schemaVersion:'paper-catch-batch-result-v1',runId:task.runId,batchId:task.batchId,batchSummary:'fake batch',selected,rejected:[]};
fs.writeFileSync(output,JSON.stringify(result));
console.log(JSON.stringify({type:'thread.started',thread_id:'fake-'+task.batchId}));
console.log(JSON.stringify({type:'turn.completed'}));
`);
  chmodSync(fake, 0o755);

  const store = new PaperCatchStore(output);
  store.initialize();
  const runId = "20260824_010203";
  const runRoot = `.runs/${runId}`;
  const candidates = [
    candidate("paper-1", "Fast Inference One", "2026-08-21T00:00:00Z"),
    candidate("paper-2", "Fast Inference Two", "2026-08-22T00:00:00Z"),
    candidate("paper-3", "Fast Inference Three", "2026-08-23T00:00:00Z"),
  ];
  const snapshots: SourceSnapshot[] = [sourceSnapshot()];
  store.writeImmutableJson(`${runRoot}/inputs/config.json`, {
    schemaVersion: "paper-catch-config-v1",
    configPath: "fixture",
    configHash: "hash",
    urls: [],
    interest: "inference acceleration",
  });
  store.writeImmutableText(`${runRoot}/inputs/PAPER_ENTRY_TEMPLATE.md`, "### template\n");
  store.writeImmutableText(`${runRoot}/inputs/batch_result.schema.json`, readFileSync(schemaPath, "utf8"));
  store.writeImmutableJson(`${runRoot}/candidates.json`, candidates);
  store.writeImmutableJson(`${runRoot}/sources/source_snapshots.json`, snapshots);
  const run: PaperCatchRun = {
    schemaVersion: "paper-catch-run-v1",
    runId,
    mode: "FULL",
    createdAt: "2026-08-24T01:02:03Z",
    updatedAt: "2026-08-24T01:02:03Z",
    status: "BATCHING",
    phase: "CANDIDATES_FROZEN",
    projectRoot: project,
    outputDir: output,
    configRef: `${runRoot}/inputs/config.json`,
    configHash: "hash",
    interest: "inference acceleration",
    previousReportRef: null,
    previousManifestRef: null,
    baselineTimestamp: "2026-08-17T00:00:00Z",
    lookbackDays: 7,
    batchSize: 2,
    model: null,
    useWebSearch: false,
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
    model: null,
    codexBin: fake,
    useWebSearch: false,
    maxAttemptsPerInvocation: 1,
    codexTimeoutMs: 30_000,
    scanOnly: false,
  }).run();
  assert.equal(outcome.status, "COMPLETED");
  assert.equal(outcome.totalBatches, 2);
  assert.equal(outcome.completedBatches, 2);
  assert.equal(store.activeRunId(), null);
  const report = store.readText(`${runId}.md`);
  assert.match(report, /Fast Inference One/);
  assert.match(report, /Fast Inference Three/);
  assert.match(report, /共 2 个固定 batch/);
});

test("an invalid batch pauses and the next invocation resumes only that batch", async () => {
  const root = temporary("resume");
  const output = resolve(root, "paper_catch");
  const project = resolve(root, "project");
  mkdirSync(project, { recursive: true });
  mkdirSync(output, { recursive: true });
  writeFileSync(resolve(output, "PAPER_ENTRY_TEMPLATE.md"), "### template\n");
  const mode = resolve(root, "mode.txt");
  writeFileSync(mode, "invalid");
  const fake = resolve(root, "fake-codex-resume.mjs");
  writeFileSync(fake, `#!/usr/bin/env node
import fs from 'node:fs';
const args=process.argv.slice(2);const output=args[args.indexOf('--output-last-message')+1];
const prompt=args.at(-1);const taskPath=/本批冻结任务：([^\\n]+)/.exec(prompt)[1].trim();
const task=JSON.parse(fs.readFileSync(taskPath,'utf8'));const valid=fs.readFileSync(${JSON.stringify(mode)},'utf8').trim()==='valid';
const selected=valid?task.candidates.map((c)=>({candidateId:c.candidateId,layers:['SYSTEM'],priority:'HIGH',relevanceReason:'relevant',performanceProblem:'latency',conciseContribution:'runtime change',openSource:'UNKNOWN',codeUrl:null,newnessReason:'new'})):[];
const result={schemaVersion:'paper-catch-batch-result-v1',runId:task.runId,batchId:task.batchId,batchSummary:'resume fixture',selected,rejected:[]};
fs.writeFileSync(output,JSON.stringify(result));console.log(JSON.stringify({type:'turn.completed'}));
`);
  chmodSync(fake, 0o755);
  const store = new PaperCatchStore(output);
  store.initialize();
  const runId = "20260824_020304";
  const runRoot = `.runs/${runId}`;
  store.writeImmutableJson(`${runRoot}/inputs/config.json`, { interest: "inference" });
  store.writeImmutableText(`${runRoot}/inputs/PAPER_ENTRY_TEMPLATE.md`, "### template\n");
  store.writeImmutableText(`${runRoot}/inputs/batch_result.schema.json`, readFileSync(schemaPath, "utf8"));
  store.writeImmutableJson(`${runRoot}/candidates.json`, [candidate("paper-r", "Resume Inference Paper")]);
  store.writeImmutableJson(`${runRoot}/sources/source_snapshots.json`, [sourceSnapshot()]);
  store.writeRun(fixtureRun(runId, output, project, runRoot, "paper-r"));
  store.setActiveRun(runId);
  const config = {
    projectRoot: project,
    configPath: "unused",
    outputDir: output,
    batchSize: 1,
    lookbackDays: 7,
    model: null,
    codexBin: fake,
    useWebSearch: false,
    maxAttemptsPerInvocation: 1,
    codexTimeoutMs: 30_000,
    scanOnly: false,
  };
  const first = await new PaperCatchController(store, config).run();
  assert.equal(first.status, "PAUSED");
  assert.equal(first.completedBatches, 0);
  writeFileSync(mode, "valid");
  const second = await new PaperCatchController(store, config).run();
  assert.equal(second.status, "COMPLETED");
  assert.equal(second.completedBatches, 1);
  assert.equal(store.readRun(runId).batches[0]?.attempts, 2);
});

function temporary(label: string): string {
  return mkdtempSync(resolve(tmpdir(), `paper-catch-${label}-`));
}

function git(cwd: string, args: string[], date?: string): void {
  const result = spawnSync("git", args, {
    cwd,
    encoding: "utf8",
    env: {
      ...process.env,
      ...(date ? { GIT_AUTHOR_DATE: date, GIT_COMMITTER_DATE: date } : {}),
    },
  });
  assert.equal(result.status, 0, result.stderr);
}

function candidate(
  id: string,
  title: string,
  committedAt = "2026-08-20T00:00:00Z",
): PaperCandidate {
  return {
    candidateId: id,
    title,
    normalizedTitle: normalizeTitle(title),
    paperUrl: "https://arxiv.org/abs/2601.00001",
    codeUrls: [],
    urls: ["https://arxiv.org/abs/2601.00001"],
    latestCommittedAt: committedAt,
    sourceRefs: [{
      sourceId: "source-1",
      sourceUrl: "https://github.com/example/papers",
      repositoryUrl: "https://github.com/example/papers",
      fragment: null,
      filePath: "README.md",
      line: 10,
      addedLine: title,
      context: title,
      commitSha: "0123456789abcdef",
      committedAt,
      commitSubject: "add paper",
      urls: ["https://arxiv.org/abs/2601.00001"],
    }],
  };
}

function batchTask(candidates: PaperCandidate[]): BatchTask {
  return {
    schemaVersion: "paper-catch-batch-task-v1",
    runId: "run-1",
    batchId: "batch-001",
    batchIndex: 1,
    batchTotal: 1,
    createdAt: "2026-08-24T00:00:00Z",
    interest: "inference",
    entryTemplateRef: "/tmp/template.md",
    policy: {
      excludeTrainingOnly: true,
      prioritizeRecent: true,
      prioritizeOpenSource: true,
      allowedLayers: ["ALGORITHM", "FRAMEWORK", "SYSTEM", "HARDWARE"],
    },
    candidates,
  };
}

function sourceSnapshot(): SourceSnapshot {
  return {
    sourceId: "source-1",
    url: "https://github.com/example/papers",
    cloneUrl: "https://github.com/example/papers.git",
    fragment: null,
    baselineHead: "aaaaaaaa",
    baselineMode: "INITIAL_LOOKBACK",
    currentHead: "bbbbbbbb",
    checkedAt: "2026-08-24T00:00:00Z",
    changed: true,
    commits: [{ sha: "bbbbbbbb", committedAt: "2026-08-23T00:00:00Z", subject: "add papers" }],
    changedMarkdownFiles: ["README.md"],
    stats: { markdownFilesChanged: 1, linesAdded: 3, linesDeleted: 0, candidateLines: 3 },
    warnings: [],
  };
}

function fixtureRun(
  runId: string,
  output: string,
  project: string,
  runRoot: string,
  _candidateId: string,
): PaperCatchRun {
  return {
    schemaVersion: "paper-catch-run-v1",
    runId,
    mode: "FULL",
    createdAt: "2026-08-24T02:03:04Z",
    updatedAt: "2026-08-24T02:03:04Z",
    status: "BATCHING",
    phase: "CANDIDATES_FROZEN",
    projectRoot: project,
    outputDir: output,
    configRef: `${runRoot}/inputs/config.json`,
    configHash: "hash",
    interest: "inference acceleration",
    previousReportRef: null,
    previousManifestRef: null,
    baselineTimestamp: "2026-08-17T00:00:00Z",
    lookbackDays: 7,
    batchSize: 1,
    model: null,
    useWebSearch: false,
    sourceSnapshotsRef: `${runRoot}/sources/source_snapshots.json`,
    candidatesRef: `${runRoot}/candidates.json`,
    batches: [],
    reportRef: null,
    manifestRef: null,
    error: null,
  };
}
