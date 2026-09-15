import assert from "node:assert/strict";
import { test } from "node:test";
import { mergeReports, type LoadedReport } from "../../paper_catch_merge.ts";
import type {
  AggregatePaper,
  PaperCandidate,
  PaperCatchRun,
  ReportManifest,
  SourceSnapshot,
} from "../types.ts";

function candidate(title: string, committedAt: string): PaperCandidate {
  const normalizedTitle = title.toLowerCase();
  return {
    candidateId: `cand-${normalizedTitle.replace(/\s+/g, "-")}`,
    title,
    normalizedTitle,
    paperUrl: `https://arxiv.org/abs/${normalizedTitle.replace(/\s+/g, "")}`,
    codeUrls: [],
    urls: [],
    latestCommittedAt: committedAt,
    sourceRefs: [{
      sourceId: "src",
      sourceUrl: "https://github.com/o/r#s",
      repositoryUrl: "https://github.com/o/r",
      fragment: "s",
      filePath: "README.md",
      line: 1,
      addedLine: `- [${title}](x)`,
      context: "",
      commitSha: "abcdef1234567890",
      committedAt,
      commitSubject: "add",
    }],
  };
}

function selected(paper: PaperCandidate, priority: "HIGH" | "MEDIUM" | "LOW", reason: string): AggregatePaper {
  return {
    candidate: paper,
    batchId: "batch-001",
    decision: {
      candidateId: paper.candidateId,
      layers: ["SYSTEM"],
      priority,
      relevanceReason: reason,
      performanceProblem: "p",
      conciseContribution: "c",
      openSource: "UNKNOWN",
      codeUrl: null,
      newnessReason: "n",
    },
  };
}

function report(
  runId: string,
  baselineTimestamp: string,
  candidates: PaperCandidate[],
  picks: AggregatePaper[],
): LoadedReport {
  const run = {
    schemaVersion: "paper-catch-run-v1",
    runId,
    mode: "FULL",
    createdAt: "2026-09-15T00:00:00.000Z",
    updatedAt: "2026-09-15T00:00:00.000Z",
    status: "COMPLETED",
    phase: "REPORT_PUBLISHED",
    projectRoot: "/p",
    outputDir: `/p/${runId}`,
    configRef: "inputs/config.md",
    configHash: "hash",
    interest: "interest",
    previousReportRef: null,
    previousManifestRef: null,
    baselineTimestamp,
    lookbackDays: 7,
    batchSize: 20,
    model: null,
    useWebSearch: true,
    sourceSnapshotsRef: "sources.json",
    candidatesRef: "candidates.json",
    batches: [{
      batchId: `${runId}-batch`,
      batchIndex: 1,
      taskRef: "task.json",
      status: "COMPLETED",
      resultRef: "result.json",
      attempts: 1,
      lastError: null,
    }],
    reportRef: `${runId}.md`,
    manifestRef: null,
    error: null,
  } as unknown as PaperCatchRun;
  const sources = [{
    sourceId: "src",
    url: "https://github.com/o/r#s",
    cloneUrl: "https://github.com/o/r.git",
    fragment: "s",
    baselineHead: "1111111111",
    baselineMode: "INITIAL_LOOKBACK",
    currentHead: "2222222222",
    checkedAt: "2026-09-15T00:00:00.000Z",
    changed: true,
    commits: [],
    changedMarkdownFiles: ["README.md"],
    stats: { markdownFilesChanged: 1, linesAdded: 1, linesDeleted: 0, candidateLines: candidates.length },
    warnings: [],
  }] as unknown as SourceSnapshot[];
  return {
    dir: `/p/${runId}`,
    manifest: { reportRef: `${runId}.md`, runId } as ReportManifest,
    run,
    aggregate: {
      schemaVersion: "paper-catch-aggregate-v1",
      runId,
      generatedAt: "2026-09-15T00:00:00.000Z",
      candidateCount: candidates.length,
      selectedCount: picks.length,
      rejectedCount: candidates.length - picks.length,
      selected: picks,
      batchResultRefs: ["result.json"],
    },
    sources,
    candidates,
  };
}

test("mergeReports dedupes by normalized title, keeps the primary decision, and takes the earliest baseline", () => {
  const shared = candidate("AccelOpt Kernel Optimization", "2026-05-07T00:00:00Z");
  const onlyPrimary = candidate("Vortex Sparse Attention", "2026-09-01T00:00:00Z");
  const onlyBackfill = candidate("ARGUS GPU Optimization", "2026-06-21T00:00:00Z");
  const rejected = candidate("Unrelated Paper", "2026-06-21T00:00:00Z");

  const primary = report(
    "20260915_120000",
    "2026-08-24T20:29:48+08:00",
    [shared, onlyPrimary],
    [selected(shared, "LOW", "primary view"), selected(onlyPrimary, "MEDIUM", "x")],
  );
  const backfill = report(
    "20260915_110000",
    "2023-12-31T16:00:00.000Z",
    [shared, onlyBackfill, rejected],
    [selected(shared, "HIGH", "backfill view"), selected(onlyBackfill, "HIGH", "y")],
  );

  const { markdown, aggregate } = mergeReports([primary, backfill]);

  assert.equal(aggregate.runId, "20260915_120000_merged");
  assert.equal(aggregate.candidateCount, 4);
  assert.equal(aggregate.selectedCount, 3);
  assert.equal(aggregate.rejectedCount, 1);
  const sharedEntry = aggregate.selected.find((paper) => paper.candidate === shared);
  assert.equal(sharedEntry?.decision.relevanceReason, "primary view");
  // HIGH priority first, then MEDIUM, then LOW.
  assert.deepEqual(
    aggregate.selected.map((paper) => paper.decision.priority),
    ["HIGH", "MEDIUM", "LOW"],
  );
  assert.match(markdown, /增量起点\*\*：2023-12-31T16:00:00.000Z/);
  assert.match(markdown, /## 合并审计/);
  assert.match(markdown, /合并前入选 4 条，按标题去重后 3 条/);
  assert.equal((markdown.match(/^### /gm) ?? []).length, 3);
  assert.match(markdown, /run 20260915_120000/);
  assert.match(markdown, /run 20260915_110000/);
});
