#!/usr/bin/env node
// Merge the published reports of several paper-catch output directories (for
// example the regular paper_catch/ run plus a one-off backfill produced by
// scripts/paper_catch_backfill.ts) into a single Markdown report. Selected papers
// are deduplicated by normalized title; the first directory listed wins on
// conflicts. The merged file is written next to the first directory's reports
// with a `_merged` suffix so store.latestReport() never mistakes it for a run.
import { existsSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { fileURLToPath } from "node:url";
import { compareAggregatePapers, renderReport } from "./paper_catch/controller.ts";
import { PaperCatchStore } from "./paper_catch/store.ts";
import type {
  AggregatePaper,
  AggregateResult,
  PaperCandidate,
  PaperCatchRun,
  ReportManifest,
  SourceSnapshot,
} from "./paper_catch/types.ts";

export interface LoadedReport {
  dir: string;
  manifest: ReportManifest;
  run: PaperCatchRun;
  aggregate: AggregateResult;
  sources: SourceSnapshot[];
  candidates: PaperCandidate[];
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    main();
  } catch (error) {
    process.stderr.write(`ERROR: ${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  }
}

function main(): void {
  const { positionals, values } = parseArgs({
    allowPositionals: true,
    options: {
      out: { type: "string" },
      force: { type: "boolean", default: false },
    },
  });
  if (positionals.length < 2) {
    usage();
    process.exitCode = 2;
    return;
  }
  const reports = positionals.map(loadReport);
  const { markdown, aggregate } = mergeReports(reports);
  const outPath = resolve(values.out ?? resolve(reports[0]!.dir, `${aggregate.runId}.md`));
  if (existsSync(outPath) && !values.force) {
    throw new Error(`refusing to overwrite ${outPath}; pass --force or --out`);
  }
  writeFileSync(outPath, markdown, "utf8");
  process.stdout.write(`${JSON.stringify({
    out: outPath,
    reports: reports.map((report) => ({ dir: report.dir, reportRef: report.manifest.reportRef })),
    candidateCount: aggregate.candidateCount,
    selectedCount: aggregate.selectedCount,
  }, null, 2)}\n`);
}

export function mergeReports(reports: LoadedReport[]): { markdown: string; aggregate: AggregateResult } {
  const primary = reports[0]!;

  const selectedByTitle = new Map<string, AggregatePaper>();
  const candidateTitles = new Set<string>();
  for (const report of reports) {
    for (const candidate of report.candidates) candidateTitles.add(candidate.normalizedTitle);
    for (const paper of report.aggregate.selected) {
      if (!selectedByTitle.has(paper.candidate.normalizedTitle)) {
        selectedByTitle.set(paper.candidate.normalizedTitle, paper);
      }
    }
  }
  const selected = [...selectedByTitle.values()].sort(compareAggregatePapers);

  const runId = `${primary.run.runId}_merged`;
  const earliest = reports.map((report) => report.run.baselineTimestamp)
    .sort((a, b) => Date.parse(a) - Date.parse(b))[0]!;
  const run: PaperCatchRun = {
    ...primary.run,
    runId,
    baselineTimestamp: earliest,
    batches: reports.flatMap((report) => report.run.batches),
  };
  const aggregate: AggregateResult = {
    schemaVersion: "paper-catch-aggregate-v1",
    runId,
    generatedAt: new Date().toISOString(),
    candidateCount: candidateTitles.size,
    selectedCount: selected.length,
    rejectedCount: candidateTitles.size - selected.length,
    selected,
    batchResultRefs: reports.flatMap((report) => report.aggregate.batchResultRefs),
  };
  const sources = reports.flatMap((report) =>
    report.sources.map((source) => ({
      ...source,
      warnings: [`run ${report.run.runId}`, ...source.warnings],
    }))
  );

  const audit = [
    "## 合并审计",
    "",
    "| 输出目录 | run | 增量起点 | 候选 | 入选 | 排除 |",
    "|---|---|---|---:|---:|---:|",
    ...reports.map((report) =>
      `| \`${report.dir}\` | \`${report.manifest.reportRef}\` | ${report.run.baselineTimestamp} | ${report.aggregate.candidateCount} | ${report.aggregate.selectedCount} | ${report.aggregate.rejectedCount} |`
    ),
    "",
    `合并前入选 ${reports.reduce((sum, report) => sum + report.aggregate.selectedCount, 0)} 条，按标题去重后 ${selected.length} 条；重复标题以先列出的目录为准。`,
    "",
  ];
  return { markdown: `${renderReport(run, aggregate, sources)}${audit.join("\n")}\n`, aggregate };
}

function loadReport(argument: string): LoadedReport {
  const separator = argument.lastIndexOf(":");
  const hasRunId = separator > 0 && /^\d{8}_\d{6}$/.test(argument.slice(separator + 1));
  const dir = resolve(hasRunId ? argument.slice(0, separator) : argument);
  const store = new PaperCatchStore(dir);
  const manifest = hasRunId
    ? store.readJson<ReportManifest>(`.state/reports/${argument.slice(separator + 1)}.json`)
    : store.readLatestManifest();
  if (!manifest) throw new Error(`no published report found in ${dir}`);
  return {
    dir,
    manifest,
    run: store.readRun(manifest.runId),
    aggregate: store.readJson<AggregateResult>(manifest.aggregateRef),
    sources: store.readJson<SourceSnapshot[]>(manifest.sourceSnapshotsRef),
    candidates: store.readJson<PaperCandidate[]>(manifest.candidatesRef),
  };
}

function usage(): void {
  process.stderr.write(`Usage:
  node scripts/paper_catch_merge.ts [--out PATH] [--force] DIR[:RUN_ID] DIR[:RUN_ID] ...

Each DIR is a paper-catch output directory; its latest published report is used
unless :RUN_ID (YYYYMMDD_HHMMSS) pins one. The first DIR is primary: its interest
and config are kept, and duplicate titles keep its decision.

Options:
  --out PATH    Default: <first DIR>/<primary run id>_merged.md
  --force       Overwrite an existing --out file
`);
}
