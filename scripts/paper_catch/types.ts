export type FilterProvider = "codex" | "claude";

/** Per-backend knobs; every field has a default so the CLI can expose each as a switch. */
export interface ProviderSettings {
  claude: {
    /** `--effort`; null keeps the CLI default. */
    effort: string | null;
    /** `--max-budget-usd`; null means unlimited. */
    maxBudgetUsd: number | null;
    /** `--tools`/`--allowedTools` override; null derives from useWebSearch. */
    tools: string | null;
    /** `--setting-sources`; default "user" so project hooks/settings stay out of batch sessions. */
    settingSources: string;
    /** Extra raw arguments appended before stdin prompt handoff. */
    extraArgs: string[];
  };
  codex: {
    /** `model_reasoning_effort`; default "high". */
    reasoningEffort: string;
    /** Extra raw arguments inserted before `exec`. */
    extraArgs: string[];
  };
}

export type RunStatus =
  | "INITIALIZING"
  | "FETCHING"
  | "BATCHING"
  | "FILTERING"
  | "AGGREGATING"
  | "COMPLETED"
  | "PAUSED"
  | "SCANNED";

export type RelevanceLayer =
  | "ALGORITHM"
  | "FRAMEWORK"
  | "SYSTEM"
  | "HARDWARE";

export type Priority = "HIGH" | "MEDIUM" | "LOW";
export type OpenSourceState = "YES" | "NO" | "UNKNOWN";

export interface CatchConfig {
  schemaVersion: "paper-catch-config-v1";
  configPath: string;
  configHash: string;
  urls: string[];
  interest: string;
}

export interface SourceSpec {
  sourceId: string;
  url: string;
  cloneUrl: string;
  fragment: string | null;
  displayName: string;
}

export interface CommitSummary {
  sha: string;
  committedAt: string;
  subject: string;
}

export interface SourceStats {
  markdownFilesChanged: number;
  linesAdded: number;
  linesDeleted: number;
  candidateLines: number;
}

export interface SourceSnapshot {
  sourceId: string;
  url: string;
  cloneUrl: string;
  fragment: string | null;
  baselineHead: string;
  baselineMode: "PREVIOUS_REPORT_HEAD" | "REPORT_TIMESTAMP" | "INITIAL_LOOKBACK";
  currentHead: string;
  checkedAt: string;
  changed: boolean;
  commits: CommitSummary[];
  changedMarkdownFiles: string[];
  stats: SourceStats;
  warnings: string[];
}

export interface CandidateSourceRef {
  sourceId: string;
  sourceUrl: string;
  repositoryUrl: string;
  fragment: string | null;
  filePath: string;
  line: number;
  addedLine: string;
  context: string;
  commitSha: string;
  committedAt: string;
  commitSubject: string;
  urls: string[];
}

export interface PaperCandidate {
  candidateId: string;
  title: string;
  normalizedTitle: string;
  paperUrl: string | null;
  codeUrls: string[];
  urls: string[];
  latestCommittedAt: string;
  sourceRefs: CandidateSourceRef[];
}

export interface BatchTask {
  schemaVersion: "paper-catch-batch-task-v1";
  runId: string;
  batchId: string;
  batchIndex: number;
  batchTotal: number;
  createdAt: string;
  interest: string;
  entryTemplateRef: string;
  policy: {
    excludeTrainingOnly: true;
    prioritizeRecent: true;
    prioritizeOpenSource: true;
    allowedLayers: RelevanceLayer[];
  };
  candidates: PaperCandidate[];
}

export interface SelectedPaperDecision {
  candidateId: string;
  layers: RelevanceLayer[];
  priority: Priority;
  relevanceReason: string;
  performanceProblem: string;
  conciseContribution: string;
  openSource: OpenSourceState;
  codeUrl: string | null;
  newnessReason: string;
}

export interface RejectedPaperDecision {
  candidateId: string;
  reason: string;
}

export interface BatchResult {
  schemaVersion: "paper-catch-batch-result-v1";
  runId: string;
  batchId: string;
  batchSummary: string;
  selected: SelectedPaperDecision[];
  rejected: RejectedPaperDecision[];
}

export interface BatchRecord {
  batchId: string;
  batchIndex: number;
  taskRef: string;
  status: "PENDING" | "RUNNING" | "COMPLETED" | "PAUSED";
  resultRef: string | null;
  attempts: number;
  lastError: string | null;
}

export interface PaperCatchRun {
  schemaVersion: "paper-catch-run-v1";
  runId: string;
  mode: "FULL" | "SCAN";
  createdAt: string;
  updatedAt: string;
  status: RunStatus;
  phase: string;
  projectRoot: string;
  outputDir: string;
  configRef: string;
  configHash: string;
  interest: string;
  previousReportRef: string | null;
  previousManifestRef: string | null;
  baselineTimestamp: string;
  lookbackDays: number;
  batchSize: number;
  /** Absent on runs created before the Claude CLI provider existed (Codex). */
  provider?: FilterProvider;
  model: string | null;
  useWebSearch: boolean;
  sourceSnapshotsRef: string | null;
  candidatesRef: string | null;
  batches: BatchRecord[];
  reportRef: string | null;
  manifestRef: string | null;
  error: string | null;
}

export interface ReportManifest {
  schemaVersion: "paper-catch-report-manifest-v1";
  reportId: string;
  runId: string;
  reportRef: string;
  generatedAt: string;
  baselineTimestamp: string;
  configHash: string;
  sourceHeads: Record<string, {
    url: string;
    head: string;
    checkedAt: string;
  }>;
  sourceSnapshotsRef: string;
  candidatesRef: string;
  aggregateRef: string;
  batchResultRefs: string[];
}

export interface AggregatePaper {
  candidate: PaperCandidate;
  decision: SelectedPaperDecision;
  batchId: string;
}

export interface AggregateResult {
  schemaVersion: "paper-catch-aggregate-v1";
  runId: string;
  generatedAt: string;
  candidateCount: number;
  selectedCount: number;
  rejectedCount: number;
  selected: AggregatePaper[];
  batchResultRefs: string[];
}

export interface ControllerOptions {
  projectRoot: string;
  configPath: string;
  outputDir: string;
  batchSize: number;
  lookbackDays: number;
  provider: FilterProvider;
  model: string | null;
  codexBin: string;
  claudeBin: string;
  providerSettings: ProviderSettings;
  useWebSearch: boolean;
  maxAttemptsPerInvocation: number;
  codexTimeoutMs: number;
  scanOnly: boolean;
}
