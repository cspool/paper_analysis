export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };

export type Layer = "L1" | "L2" | "L3" | "L4" | "L5" | "L6";
export type ValueAxis = "exploration" | "implementation" | "method" | "baseline";
export type EvidenceKind = "direct" | "inferred";
export type Confidence = "low" | "middle" | "high";
export type Effort = "low" | "medium" | "high" | "xhigh";

export type Role =
  | "anchor_stage_controller"
  | "anchor_evidence_worker"
  | "anchor_curator_worker"
  | "direction_planner"
  | "direction_reviewer"
  | "review_evidence_worker";

export type ReviewDimension =
  | "scenario_opportunity"
  | "baseline_fairness"
  | "entry_validity"
  | "cross_layer_validity"
  | "implementation_reuse"
  | "experiment_measurement";

export interface RoleProfile {
  role: Role;
  skillName: string;
  skillPath: string;
  effort: Effort;
  persistent: boolean;
  knowledgeAccess: "none" | "obsidian_readonly";
  allowedMainMarkers: string[];
}

export interface RunConfig {
  protocolVersion: number;
  topic: string;
  constraints: string[];
  model: "gpt-5.6-sol";
  roleEffort: Record<Role, Effort>;
  protocolRepairEffort: "low";
  maxAnchors: number;
  noNewAnchorStop: number;
  maxStage1Rounds: number;
  maxStage1Tasks: number;
  evidenceTasksPerRound: number;
  anchorEvidenceConcurrency: number;
  curatorConcurrency: number;
  maxDirectionsPerAnchor: number;
  maxPlannerTurnsPerAnchor: number;
  maxReviewRoundsPerDirection: number;
  directionConcurrency: number;
  maxTotalTurns: number;
  maxPersistentTurns: number;
  turnTimeoutMs: number;
  requestTimeoutMs: number;
  startupTimeoutMs: number;
  vaultRoot: string;
  evidenceRoots: string[];
  projectRoot: string;
  skillRoot: string;
  sourceConfigPath: string;
  sourceAuthPath: string;
  createdAt: string;
}

export interface TokenUsage {
  inputTokens: number;
  cachedInputTokens: number;
  outputTokens: number;
  reasoningOutputTokens: number;
  totalTokens: number;
}

export interface EvidenceClaim {
  claimId: string;
  statement: string;
  claimType: string;
  evidenceKind: EvidenceKind;
  sourcePath: string;
  lineStart: number;
  lineEnd: number;
  quote: string;
  applicableScope: string;
  confidence: Confidence;
  taskId: string;
  createdAt: string;
}

export interface RejectedClaim {
  candidate: JsonValue;
  taskId: string;
  reasons: string[];
  rejectedAt: string;
}

export interface GlobalEntity {
  entityId: string;
  kind: "method" | "implementation" | "tool" | "software" | "hardware" | "other";
  name: string;
  description: string;
  evidenceRefs: string[];
}

export type BaselineKind =
  | "current_practice"
  | "strong_comparison"
  | "tool_evaluation"
  | "reusable_implementation";

export interface Baseline {
  baselineId: string;
  anchorId: string | null;
  kind: BaselineKind;
  name: string;
  executionPath: string;
  implementation: string;
  comparisonScope: string;
  evidenceRefs: string[];
  explorationValue: "low" | "middle" | "high" | "unknown";
  status: "active" | "reference_only";
}

export interface AnchorSignature {
  workload: string;
  phase: string;
  regime: string;
  backend: string;
  bottleneck: string;
  primaryBaselineExecutionPath: string;
  targetMetrics: string[];
}

export interface Anchor {
  anchorId: string;
  title: string;
  scenario: string;
  signature: AnchorSignature;
  evidenceRefs: string[];
  baselineIds: string[];
  entryIds: string[];
  edgeIds: string[];
  gaps: string[];
  status: "accepted" | "rejected_cap" | "invalid";
  firstSeenRound: number;
  lastUpdatedRound: number;
  priority: number[];
}

export type EntryRole =
  | "baseline_behavior"
  | "opportunity"
  | "method"
  | "implementation"
  | "constraint"
  | "evaluation";

export interface LayerEntry {
  entryId: string;
  entityId: string | null;
  anchorId: string;
  layer: Layer;
  role: EntryRole;
  claim: string;
  modifiableObject: string;
  applicableBaselineIds: string[];
  preconditions: string[];
  expectedEffect: string;
  evidenceRefs: string[];
  confidence: Confidence;
  status: "candidate" | "active" | "rejected";
}

export type EdgeRelation =
  | "controls"
  | "depends_on"
  | "enables"
  | "complements"
  | "conflicts"
  | "substitutes"
  | "incompatible";

export interface CrossLayerEdge {
  edgeId: string;
  anchorId: string;
  fromEntryId: string;
  toEntryId: string;
  relation: EdgeRelation;
  interface: string;
  compatibility: "compatible" | "conditional" | "conflict";
  condition: string;
  evidenceRefs: string[];
  confidence: Confidence;
}

export interface Direction {
  directionId: string;
  anchorId: string;
  selectedEntryIds: string[];
  selectedEdgeIds: string[];
  baselineIds: string[];
  hypothesis: string;
  ablationPlan: string[];
  metrics: string[];
  implementationPlan: string[];
  evidenceRefs: string[];
  status: "accepted" | "rejected";
  rejectionReasons: string[];
  proposalIndex: number;
}

export interface ReviewQuestionAnswer {
  round: number;
  questionId: string;
  dimension: ReviewDimension;
  question: string;
  answer: string;
  conclusion: "supported" | "contradicted" | "partial" | "unknown" | "not_applicable";
  evidenceRefs: string[];
  gaps: string[];
}

export interface ExpertReview {
  directionId: string;
  status: "complete" | "pending";
  pendingReason: string | null;
  explorationValue: "low" | "middle" | "high" | "unknown";
  implementationReuse: "low" | "middle" | "high" | "unknown";
  methodReference: "low" | "middle" | "high" | "unknown";
  baselineQuality: "invalid" | "weak" | "fair" | "strong" | "unknown";
  crossLayerValidity: "invalid" | "weak" | "conditional" | "valid" | "unknown";
  experimentReadiness: "not_ready" | "partial" | "ready" | "unknown";
  decision: "rejected" | "baseline_reference" | "needs_evidence" | "experiment_candidate";
  rationale: string;
  minimumImplementationPlan: string[];
  baselineAblationMatrix: string[];
  metricsTools: string[];
  failureStopConditions: string[];
  selectedRefs: string[];
  alternativeRefs: string[];
  gaps: string[];
  questionAnswers: ReviewQuestionAnswer[];
  referenceKeysUsed: string[];
}

export interface SessionRecord {
  role: Role;
  scopeId: string;
  threadId: string;
  lastTurnId: string | null;
  model: string;
  effort: Effort;
  skillPath: string;
  skillHash: string;
  lastLoop: string | null;
  lastNormalizedOutputPath: string | null;
  status: "not_started" | "initializing" | "waiting_input" | "running" | "yielded" | "terminated" | "contaminated";
  turnCount: number;
  cumulativeUsage: TokenUsage;
}

export interface TaskRecord {
  taskId: string;
  role: Role;
  scopeId: string;
  status:
    | "pending"
    | "dispatched"
    | "response_received"
    | "protocol_valid"
    | "domain_valid"
    | "committed"
    | "failed_retriable"
    | "failed_terminal"
    | "security_invalid";
  attempts: number;
  inputPath: string | null;
  outputPath: string | null;
  error: string | null;
}

export interface ProviderManifest {
  codexCliVersion: string;
  appServerSchemaHash: string;
  schemaHashAlgorithm: "canonical-json-v1";
  generatedAt: string;
  supportedMethods: string[];
  modelVerified: boolean;
  supportedEfforts: string[];
  obsidianConfigured: boolean;
}

export interface RunState {
  schemaVersion: 1;
  runId: string;
  status: "initialized" | "running" | "complete" | "failed";
  config: RunConfig;
  provider: ProviderManifest | null;
  stage1: {
    status: "pending" | "running" | "complete";
    round: number;
    taskCount: number;
    consecutiveRoundsWithoutNewAnchor: number;
    acceptedAnchorIds: string[];
    stopReason: string | null;
    anchorSpaceVersion: string | null;
    controllerLastOutput: string | null;
  };
  stage2: {
    status: "pending" | "planning" | "reviewing" | "complete";
    anchorPlanning: Record<string, { status: "pending" | "running" | "complete" | "pending_budget"; reason: string | null; turns: number }>;
    directionReview: Record<string, { status: "pending" | "running" | "complete"; reason: string | null; rounds: number }>;
  };
  claims: EvidenceClaim[];
  rejectedClaims: RejectedClaim[];
  entities: GlobalEntity[];
  baselines: Baseline[];
  anchors: Anchor[];
  entries: LayerEntry[];
  edges: CrossLayerEdge[];
  directions: Direction[];
  reviews: ExpertReview[];
  sessions: Record<string, SessionRecord>;
  tasks: Record<string, TaskRecord>;
  usage: {
    turns: number;
    total: TokenUsage;
    byRole: Record<string, TokenUsage>;
  };
  validation: {
    ok: boolean;
    errors: string[];
    warnings: string[];
    checkedAt: string | null;
  };
  createdAt: string;
  updatedAt: string;
}

export interface NormalizedEvent {
  method: string;
  threadId: string | null;
  turnId: string | null;
  itemType: string | null;
  server: string | null;
  tool: string | null;
  raw: JsonValue;
}

export interface AgentHandle {
  role: Role;
  scopeId: string;
  threadId: string;
  skillPath: string;
  skillHash: string;
  persistent: boolean;
  firstTurn: boolean;
  turnCount: number;
}

export interface TurnResult {
  threadId: string;
  turnId: string;
  status: "completed" | "interrupted" | "failed";
  text: string;
  usage: TokenUsage | null;
  observedEvents: NormalizedEvent[];
  rawLogPath: string;
  securityViolations: string[];
  compacted: boolean;
  error: string | null;
}

export interface ParsedProtocol {
  role: Role;
  marker: string;
  control: Record<string, string>;
  payload: JsonValue | null;
  textBlocks: Record<string, string>;
  loop: string | null;
  terminated: boolean;
  rawText: string;
}

export interface ValidationReport {
  ok: boolean;
  errors: string[];
  warnings: string[];
}

export const ZERO_USAGE: TokenUsage = {
  inputTokens: 0,
  cachedInputTokens: 0,
  outputTokens: 0,
  reasoningOutputTokens: 0,
  totalTokens: 0,
};
