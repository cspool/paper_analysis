import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseCatchConfig, sourceSpecFromUrl } from "./config.ts";
import {
  invokeCodexBatch,
  readAndValidateBatchResult,
} from "./codex_filter.ts";
import {
  collectSourceUpdate,
  mergePaperCandidates,
} from "./git_sources.ts";
import { PaperCatchStore, timestampId } from "./store.ts";
import type {
  AggregatePaper,
  AggregateResult,
  BatchRecord,
  BatchResult,
  BatchTask,
  CatchConfig,
  ControllerOptions,
  PaperCandidate,
  PaperCatchRun,
  ReportManifest,
  SourceSnapshot,
} from "./types.ts";

export interface ControllerOutcome {
  status: "COMPLETED" | "PAUSED" | "SCANNED";
  runId: string;
  candidateCount: number;
  completedBatches: number;
  totalBatches: number;
  reportRef: string | null;
  error: string | null;
}

export class PaperCatchController {
  readonly store: PaperCatchStore;
  readonly options: ControllerOptions;
  private invocationAttemptCount = new Map<string, number>();

  constructor(store: PaperCatchStore, options: ControllerOptions) {
    this.store = store;
    this.options = options;
  }

  async run(): Promise<ControllerOutcome> {
    let run = this.loadOrCreateRun();
    try {
      if (!run.candidatesRef || !run.sourceSnapshotsRef) {
        run = this.updateRun(run, { status: "FETCHING", phase: "GIT_INCREMENT_SCAN", error: null });
        run = this.fetchAndFreeze(run);
      }
      const candidates = this.store.readJson<PaperCandidate[]>(run.candidatesRef);
      if (run.batches.length === 0 && candidates.length > 0) {
        run = this.createBatches(run, candidates);
      }
      if (this.options.scanOnly) {
        run = this.updateRun(run, { status: "SCANNED", phase: "SCAN_COMPLETE", error: null });
        this.store.clearActiveRun(run.runId);
        return this.outcome(run, candidates.length);
      }
      if (candidates.length === 0) {
        run = this.updateRun(run, { status: "AGGREGATING", phase: "NO_CANDIDATES", error: null });
        run = this.aggregateAndPublish(run, candidates);
        return this.outcome(run, 0);
      }

      run = this.updateRun(run, { status: "FILTERING", phase: "CODEX_BATCH_LOOP", error: null });
      for (const batch of run.batches) {
        if (batch.status === "COMPLETED") continue;
        const result = await this.executeBatch(run, batch);
        run = this.store.readRun(run.runId);
        if (!result) {
          return this.outcome(run, candidates.length);
        }
      }
      run = this.updateRun(run, { status: "AGGREGATING", phase: "DETERMINISTIC_AGGREGATION", error: null });
      run = this.aggregateAndPublish(run, candidates);
      return this.outcome(run, candidates.length);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      run = this.updateRun(run, { status: "PAUSED", phase: "ERROR", error: message });
      this.event(run.runId, "RUN_PAUSED", [`.runs/${run.runId}/run.json`], { error: message });
      return this.outcome(run, run.candidatesRef && this.store.exists(run.candidatesRef)
        ? this.store.readJson<PaperCandidate[]>(run.candidatesRef).length
        : 0);
    }
  }

  private loadOrCreateRun(): PaperCatchRun {
    const active = this.store.activeRunId();
    if (active && this.store.exists(`.runs/${active}/run.json`)) {
      const run = this.store.readRun(active);
      if (run.status !== "COMPLETED" && run.status !== "SCANNED") {
        const existingMode = run.mode ?? "FULL";
        const requestedMode = this.options.scanOnly ? "SCAN" : "FULL";
        if (existingMode !== requestedMode) {
          throw new Error(
            `unfinished ${existingMode} run ${run.runId} must be resumed before starting ${requestedMode}`,
          );
        }
        this.event(run.runId, "RUN_RESUMED", [`.runs/${run.runId}/run.json`]);
        return run;
      }
    }
    const config = parseCatchConfig(this.options.configPath);
    const latest = this.store.latestReport();
    const previousManifest = this.store.readLatestManifest();
    const createdAt = new Date().toISOString();
    const baselineTimestamp = latest?.timestamp ?? new Date(
      Date.now() - this.options.lookbackDays * 86_400_000,
    ).toISOString();
    const runId = this.nextRunId();
    const runRoot = `.runs/${runId}`;
    const configRef = `${runRoot}/inputs/config.json`;
    const configTextRef = `${runRoot}/inputs/Catch_Paper_Urls.md`;
    const templateRef = `${runRoot}/inputs/PAPER_ENTRY_TEMPLATE.md`;
    const schemaRef = `${runRoot}/inputs/batch_result.schema.json`;
    const templateSource = resolve(this.options.outputDir, "PAPER_ENTRY_TEMPLATE.md");
    const schemaSource = resolve(dirname(new URL(import.meta.url).pathname), "batch_result.schema.json");
    this.store.writeImmutableJson(configRef, config);
    this.store.writeImmutableText(configTextRef, readFileSync(config.configPath, "utf8"));
    this.store.writeImmutableText(templateRef, readFileSync(templateSource, "utf8"));
    this.store.writeImmutableText(schemaRef, readFileSync(schemaSource, "utf8"));

    const run: PaperCatchRun = {
      schemaVersion: "paper-catch-run-v1",
      runId,
      mode: this.options.scanOnly ? "SCAN" : "FULL",
      createdAt,
      updatedAt: createdAt,
      status: "INITIALIZING",
      phase: "CREATED",
      projectRoot: this.options.projectRoot,
      outputDir: this.options.outputDir,
      configRef,
      configHash: config.configHash,
      interest: config.interest,
      previousReportRef: latest?.reportRef ?? null,
      previousManifestRef: latest?.manifestRef ?? null,
      baselineTimestamp,
      lookbackDays: this.options.lookbackDays,
      batchSize: this.options.batchSize,
      model: this.options.model,
      useWebSearch: this.options.useWebSearch,
      sourceSnapshotsRef: null,
      candidatesRef: null,
      batches: [],
      reportRef: null,
      manifestRef: null,
      error: null,
    };
    this.store.writeRun(run);
    this.store.setActiveRun(runId);
    this.event(runId, "RUN_CREATED", [configRef, configTextRef, templateRef, schemaRef]);
    if (previousManifest) {
      this.event(runId, "PREVIOUS_REPORT_BOUND", [run.previousManifestRef!]);
    }
    return run;
  }

  private fetchAndFreeze(run: PaperCatchRun): PaperCatchRun {
    const config = this.store.readJson<CatchConfig>(run.configRef);
    const previousManifest = run.previousManifestRef
      ? this.store.readJson<ReportManifest>(run.previousManifestRef)
      : null;
    const snapshots: SourceSnapshot[] = [];
    const groups: PaperCandidate[][] = [];
    const checkedAt = new Date().toISOString();
    for (const rawUrl of config.urls) {
      const spec = sourceSpecFromUrl(rawUrl);
      process.stderr.write(`[paper-catch] git ${spec.displayName}\n`);
      const collected = collectSourceUpdate({
        spec,
        cacheRoot: this.store.absolute(".state/cache"),
        baselineTimestamp: run.baselineTimestamp,
        previousManifest,
        checkedAt,
      });
      snapshots.push(collected.snapshot);
      groups.push(collected.candidates);
      this.event(run.runId, "SOURCE_SCANNED", [], {
        sourceId: spec.sourceId,
        commits: collected.snapshot.commits.length,
        candidates: collected.candidates.length,
      });
    }
    const candidates = mergePaperCandidates(groups);
    const root = `.runs/${run.runId}`;
    const snapshotsRef = `${root}/sources/source_snapshots.json`;
    const candidatesRef = `${root}/candidates.json`;
    this.store.writeImmutableJson(snapshotsRef, snapshots);
    this.store.writeImmutableJson(candidatesRef, candidates);
    this.event(run.runId, "GIT_SNAPSHOT_FROZEN", [snapshotsRef, candidatesRef], {
      sources: snapshots.length,
      candidates: candidates.length,
    });
    return this.updateRun(run, {
      status: "BATCHING",
      phase: "CANDIDATES_FROZEN",
      sourceSnapshotsRef: snapshotsRef,
      candidatesRef,
      error: null,
    });
  }

  private createBatches(run: PaperCatchRun, candidates: PaperCandidate[]): PaperCatchRun {
    const total = Math.ceil(candidates.length / run.batchSize);
    const batches: BatchRecord[] = [];
    const templateRef = this.store.absolute(`.runs/${run.runId}/inputs/PAPER_ENTRY_TEMPLATE.md`);
    for (let index = 0; index < total; index += 1) {
      const batchId = `batch-${String(index + 1).padStart(3, "0")}`;
      const taskRef = `.runs/${run.runId}/batches/${batchId}/task.json`;
      const task: BatchTask = {
        schemaVersion: "paper-catch-batch-task-v1",
        runId: run.runId,
        batchId,
        batchIndex: index + 1,
        batchTotal: total,
        createdAt: new Date().toISOString(),
        interest: run.interest,
        entryTemplateRef: templateRef,
        policy: {
          excludeTrainingOnly: true,
          prioritizeRecent: true,
          prioritizeOpenSource: true,
          allowedLayers: ["ALGORITHM", "FRAMEWORK", "SYSTEM", "HARDWARE"],
        },
        candidates: candidates.slice(index * run.batchSize, (index + 1) * run.batchSize),
      };
      this.store.writeImmutableJson(taskRef, task);
      batches.push({
        batchId,
        batchIndex: index + 1,
        taskRef,
        status: "PENDING",
        resultRef: null,
        attempts: 0,
        lastError: null,
      });
    }
    this.event(run.runId, "BATCH_PLAN_FROZEN", batches.map((batch) => batch.taskRef), {
      batchSize: run.batchSize,
      batches: total,
    });
    return this.updateRun(run, { status: "FILTERING", phase: "BATCH_PLAN_FROZEN", batches });
  }

  private async executeBatch(run: PaperCatchRun, batchInput: BatchRecord): Promise<BatchResult | null> {
    let batch = run.batches.find((entry) => entry.batchId === batchInput.batchId)!;
    const task = this.store.readJson<BatchTask>(batch.taskRef);
    const schemaPath = this.store.absolute(`.runs/${run.runId}/inputs/batch_result.schema.json`);
    const previousCorrectionRef = batch.attempts > 0
      ? `.runs/${run.runId}/batches/${batch.batchId}/attempt-${String(batch.attempts).padStart(2, "0")}/validation_error.json`
      : null;
    let correctionPath = previousCorrectionRef && this.store.exists(previousCorrectionRef)
      ? this.store.absolute(previousCorrectionRef)
      : null;
    let attemptsThisInvocation = this.invocationAttemptCount.get(batch.batchId) ?? 0;

    while (attemptsThisInvocation < this.options.maxAttemptsPerInvocation) {
      batch = { ...batch, status: "RUNNING", attempts: batch.attempts + 1, lastError: null };
      run = this.replaceBatch(run, batch);
      const attemptId = `attempt-${String(batch.attempts).padStart(2, "0")}`;
      const attemptRef = `.runs/${run.runId}/batches/${batch.batchId}/${attemptId}`;
      const startedAt = new Date().toISOString();
      process.stderr.write(
        `[paper-catch] Codex ${batch.batchId} (${batch.batchIndex}/${run.batches.length}) ${attemptId}\n`,
      );
      this.event(run.runId, "BATCH_ATTEMPT_STARTED", [batch.taskRef], {
        batchId: batch.batchId,
        attempt: batch.attempts,
      });
      const outcome = await invokeCodexBatch({
        codexBin: this.options.codexBin,
        projectRoot: this.options.projectRoot,
        taskPath: this.store.absolute(batch.taskRef),
        schemaPath,
        attemptDir: this.store.absolute(attemptRef),
        model: run.model,
        useWebSearch: run.useWebSearch,
        timeoutMs: this.options.codexTimeoutMs,
        correctionPath,
      });
      const validation = outcome.exitCode === 0 && !outcome.timedOut
        ? readAndValidateBatchResult(outcome.outputPath, task)
        : {
          result: null,
          errors: [outcome.error ?? `Codex exit=${outcome.exitCode} signal=${outcome.signal} timedOut=${outcome.timedOut}`],
        };
      const endedAt = new Date().toISOString();
      this.store.writeJson(`${attemptRef}/attempt.json`, {
        batchId: batch.batchId,
        attempt: batch.attempts,
        startedAt,
        endedAt,
        exitCode: outcome.exitCode,
        signal: outcome.signal,
        timedOut: outcome.timedOut,
        valid: validation.errors.length === 0,
        errors: validation.errors,
        refs: {
          prompt: this.store.relative(outcome.promptPath),
          providerRaw: this.store.relative(outcome.providerRawPath),
          stderr: this.store.relative(outcome.stderrPath),
          output: this.store.relative(outcome.outputPath),
        },
      });
      attemptsThisInvocation += 1;
      this.invocationAttemptCount.set(batch.batchId, attemptsThisInvocation);
      if (validation.result) {
        const resultRef = `.runs/${run.runId}/batches/${batch.batchId}/result.json`;
        this.store.writeImmutableJson(resultRef, validation.result);
        batch = {
          ...batch,
          status: "COMPLETED",
          resultRef,
          lastError: null,
        };
        run = this.replaceBatch(run, batch);
        this.event(run.runId, "BATCH_COMPLETED", [batch.taskRef, resultRef], {
          batchId: batch.batchId,
          selected: validation.result.selected.length,
          rejected: validation.result.rejected.length,
        });
        return validation.result;
      }
      const correctionRef = `${attemptRef}/validation_error.json`;
      this.store.writeJson(correctionRef, { errors: validation.errors });
      correctionPath = this.store.absolute(correctionRef);
      batch = {
        ...batch,
        status: "PENDING",
        lastError: validation.errors.join("; "),
      };
      run = this.replaceBatch(run, batch);
      this.event(run.runId, "BATCH_ATTEMPT_INVALID", [correctionRef], {
        batchId: batch.batchId,
        attempt: batch.attempts,
      });
    }

    batch = { ...batch, status: "PAUSED" };
    run = this.replaceBatch(run, batch);
    run = this.updateRun(run, {
      status: "PAUSED",
      phase: "CODEX_BATCH_RETRY_REQUIRED",
      error: `batch ${batch.batchId} needs another manual invocation: ${batch.lastError}`,
    });
    this.event(run.runId, "RUN_PAUSED_FOR_BATCH", [batch.taskRef], { batchId: batch.batchId });
    return null;
  }

  private aggregateAndPublish(run: PaperCatchRun, candidates: PaperCandidate[]): PaperCatchRun {
    const selected: AggregatePaper[] = [];
    let rejectedCount = 0;
    const batchResultRefs: string[] = [];
    for (const batch of run.batches) {
      if (!batch.resultRef) throw new Error(`batch has no committed result: ${batch.batchId}`);
      const result = this.store.readJson<BatchResult>(batch.resultRef);
      batchResultRefs.push(batch.resultRef);
      rejectedCount += result.rejected.length;
      const byId = new Map(candidates.map((candidate) => [candidate.candidateId, candidate]));
      for (const decision of result.selected) {
        const candidate = byId.get(decision.candidateId);
        if (!candidate) throw new Error(`aggregate cannot resolve ${decision.candidateId}`);
        selected.push({ candidate, decision, batchId: batch.batchId });
      }
    }
    selected.sort(compareAggregatePapers);
    const aggregate: AggregateResult = {
      schemaVersion: "paper-catch-aggregate-v1",
      runId: run.runId,
      generatedAt: new Date().toISOString(),
      candidateCount: candidates.length,
      selectedCount: selected.length,
      rejectedCount,
      selected,
      batchResultRefs,
    };
    const aggregateRef = `.runs/${run.runId}/aggregate.json`;
    const sourceSnapshots = this.store.readJson<SourceSnapshot[]>(run.sourceSnapshotsRef!);
    const draftRef = `.runs/${run.runId}/final_report.md`;
    const reportRef = `${run.runId}.md`;
    const report = renderReport(run, aggregate, sourceSnapshots);
    this.store.writeImmutableJson(aggregateRef, aggregate);
    this.store.writeImmutableText(draftRef, report);
    this.store.writeImmutableText(reportRef, report);
    const manifestRef = `.state/reports/${run.runId}.json`;
    const manifest: ReportManifest = {
      schemaVersion: "paper-catch-report-manifest-v1",
      reportId: run.runId,
      runId: run.runId,
      reportRef,
      generatedAt: aggregate.generatedAt,
      baselineTimestamp: run.baselineTimestamp,
      configHash: run.configHash,
      sourceHeads: Object.fromEntries(sourceSnapshots.map((source) => [
        source.sourceId,
        { url: source.url, head: source.currentHead, checkedAt: source.checkedAt },
      ])),
      sourceSnapshotsRef: run.sourceSnapshotsRef!,
      candidatesRef: run.candidatesRef!,
      aggregateRef,
      batchResultRefs,
    };
    this.store.writeImmutableJson(manifestRef, manifest);
    run = this.updateRun(run, {
      status: "COMPLETED",
      phase: "REPORT_PUBLISHED",
      reportRef,
      manifestRef,
      error: null,
    });
    this.store.clearActiveRun(run.runId);
    this.store.writeJson(".state/last_run.json", {
      runId: run.runId,
      status: run.status,
      reportRef,
      completedAt: new Date().toISOString(),
    });
    this.event(run.runId, "REPORT_PUBLISHED", [reportRef, manifestRef, aggregateRef]);
    return run;
  }

  private replaceBatch(run: PaperCatchRun, batch: BatchRecord): PaperCatchRun {
    const batches = run.batches.map((entry) => entry.batchId === batch.batchId ? batch : entry);
    return this.updateRun(run, { batches });
  }

  private updateRun(run: PaperCatchRun, patch: Partial<PaperCatchRun>): PaperCatchRun {
    const next = { ...run, ...patch, updatedAt: new Date().toISOString() };
    this.store.writeRun(next);
    return next;
  }

  private event(runId: string, type: string, refs: string[] = [], detail?: unknown): void {
    this.store.appendJsonLine(`.runs/${runId}/events.jsonl`, {
      time: new Date().toISOString(),
      type,
      refs,
      ...(detail === undefined ? {} : { detail }),
    });
  }

  private outcome(run: PaperCatchRun, candidateCount: number): ControllerOutcome {
    return {
      status: run.status === "COMPLETED" ? "COMPLETED" : run.status === "SCANNED" ? "SCANNED" : "PAUSED",
      runId: run.runId,
      candidateCount,
      completedBatches: run.batches.filter((batch) => batch.status === "COMPLETED").length,
      totalBatches: run.batches.length,
      reportRef: run.reportRef,
      error: run.error,
    };
  }

  private nextRunId(): string {
    let date = new Date();
    for (let attempt = 0; attempt < 120; attempt += 1) {
      const id = timestampId(date);
      if (!this.store.exists(`.runs/${id}`) && !this.store.exists(`${id}.md`)) return id;
      date = new Date(date.getTime() + 1000);
    }
    throw new Error("cannot allocate a unique timestamped run id");
  }
}

export function renderReport(
  run: PaperCatchRun,
  aggregate: AggregateResult,
  sources: SourceSnapshot[],
): string {
  const lines = [
    `# Paper Catch ${run.runId}`,
    "",
    `- **增量起点**：${run.baselineTimestamp}`,
    `- **生成时间**：${aggregate.generatedAt}`,
    `- **监控来源**：${sources.length}`,
    `- **Git 新标题候选**：${aggregate.candidateCount}`,
    `- **语义筛选入选**：${aggregate.selectedCount}`,
    `- **排除**：${aggregate.rejectedCount}`,
    `- **兴趣主题**：${oneLine(run.interest)}`,
    "",
    "## 筛选结果",
    "",
  ];
  if (aggregate.selected.length === 0) {
    lines.push("本次增量中没有符合兴趣主题的新论文。", "");
  } else {
    for (const paper of aggregate.selected) lines.push(...renderEntry(paper));
  }
  lines.push(
    "## Git 更新审计",
    "",
    "| 来源 | baseline → current | commits | Markdown 变更 | + / - | 候选 | 警告 |",
    "|---|---|---:|---:|---:|---:|---|",
  );
  for (const source of sources) {
    lines.push(
      `| [${escapeTable(source.url)}](${source.url}) | \`${source.baselineHead.slice(0, 8)}\` → \`${source.currentHead.slice(0, 8)}\` | ${source.commits.length} | ${source.stats.markdownFilesChanged} | ${source.stats.linesAdded} / ${source.stats.linesDeleted} | ${source.stats.candidateLines} | ${escapeTable(source.warnings.join("; ") || "无")} |`,
    );
  }
  lines.push(
    "",
    "## 批次审计",
    "",
    `共 ${run.batches.length} 个固定 batch；全部 batch Result 通过候选覆盖和字段合同校验后才生成本报告。`,
    "",
  );
  return `${lines.join("\n")}\n`;
}

function renderEntry(paper: AggregatePaper): string[] {
  const { candidate, decision } = paper;
  const paperUrl = candidate.paperUrl ?? candidate.urls.find((url) => !isCodeUrl(url)) ??
    candidate.sourceRefs[0]?.sourceUrl ?? "";
  const title = escapeLinkText(candidate.title);
  const heading = paperUrl ? `### [${title}](${paperUrl})` : `### ${title}`;
  const latest = candidate.sourceRefs[0];
  const codeUrl = decision.codeUrl ?? candidate.codeUrls[0] ?? null;
  const openSource = codeUrl
    ? `[${decision.openSource === "NO" ? "候选代码链接" : "代码"}](${codeUrl})`
    : decision.openSource === "YES"
    ? "已确认开源（未返回链接）"
    : decision.openSource === "NO"
    ? "未开源"
    : "未知";
  const sourceLinks = unique(candidate.sourceRefs.map((source) => source.sourceUrl))
    .map((url) => `[${escapeLinkText(shortSource(url))}](${url})`).join("、");
  return [
    heading,
    "",
    `- **更新证据**：${latest ? `${latest.committedAt}，\`${latest.commitSha.slice(0, 8)}\`，${escapeMarkdown(latest.commitSubject)}` : candidate.latestCommittedAt}`,
    `- **相关层次**：${decision.layers.map((layer) => LAYER_LABEL[layer]).join(" / ")}`,
    `- **性能问题**：${escapeMarkdown(decision.performanceProblem)}`,
    `- **方法与贡献**：${escapeMarkdown(decision.conciseContribution)}`,
    `- **兴趣匹配**：${escapeMarkdown(decision.relevanceReason)}`,
    `- **开源情况**：${openSource}`,
    `- **优先级**：${PRIORITY_LABEL[decision.priority]}；${escapeMarkdown(decision.newnessReason)}`,
    `- **更新来源**：${sourceLinks}`,
    "",
  ];
}

function compareAggregatePapers(a: AggregatePaper, b: AggregatePaper): number {
  const priority = { HIGH: 0, MEDIUM: 1, LOW: 2 };
  const aCode = a.decision.codeUrl || a.candidate.codeUrls.length > 0 ? 0 : 1;
  const bCode = b.decision.codeUrl || b.candidate.codeUrls.length > 0 ? 0 : 1;
  return priority[a.decision.priority] - priority[b.decision.priority] ||
    aCode - bCode ||
    b.candidate.latestCommittedAt.localeCompare(a.candidate.latestCommittedAt) ||
    a.candidate.normalizedTitle.localeCompare(b.candidate.normalizedTitle);
}

function oneLine(value: string): string {
  return escapeMarkdown(value.replace(/\s+/g, " ").trim());
}

function escapeMarkdown(value: string): string {
  return value.replace(/\r?\n/g, " ").replace(/\|/g, "\\|").trim();
}

function escapeTable(value: string): string {
  return value.replace(/\|/g, "\\|").replace(/\r?\n/g, " ");
}

function escapeLinkText(value: string): string {
  return value.replace(/([\[\]])/g, "\\$1");
}

function shortSource(value: string): string {
  try {
    const url = new URL(value);
    return `${url.hostname}${url.pathname.replace(/\/$/, "")}${url.hash}`;
  } catch {
    return value;
  }
}

function isCodeUrl(value: string): boolean {
  try {
    const host = new URL(value).hostname;
    return host === "github.com" || host === "gitlab.com" || host === "huggingface.co";
  } catch {
    return false;
  }
}

function unique<T>(values: T[]): T[] {
  return [...new Set(values)];
}

const LAYER_LABEL = {
  ALGORITHM: "算法层",
  FRAMEWORK: "框架层",
  SYSTEM: "系统层",
  HARDWARE: "硬件层",
};
const PRIORITY_LABEL = { HIGH: "高", MEDIUM: "中", LOW: "低" };
