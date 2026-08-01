# Simple Semantic Loop 共享契约规范

## 1. 规范定位

本规范是四类 Turn Agent 和 Scheduler 共用的唯一协议来源。它不作为独立实施项目，也不实现 Agent 行为或调度循环，而是冻结：

- canonical 领域对象；
- workflow 控制对象；
- Turn 输入输出 Envelope；
- WorkflowTrigger、DecisionAction 和 action/payload 关系；
- StageContract、GateDefinition 和 plan patch；
- JSON Schema、TypeScript 类型、validator 和 fixture；
- revision、hash、CAS 和 No Experiment invariant。

需求入口：[实现计划入口](../learning_workflow_simple_semantic_loop_implementation_plan.md)。

## 2. 交付目录

第一版实现应提供：

```text
scripts/simple_semantic_loop/
├── contracts/
│   ├── domain.ts
│   ├── control.ts
│   ├── messages.ts
│   ├── registries.ts
│   ├── canonical_json.ts
│   └── index.ts
├── schemas/
│   ├── topic_frame.schema.json
│   ├── anchor.schema.json
│   ├── direction.schema.json
│   ├── search_need.schema.json
│   ├── evidence_task.schema.json
│   ├── evidence_packet.schema.json
│   ├── direction_review_task.schema.json
│   ├── review_delta.schema.json
│   ├── closure_review_task.schema.json
│   ├── closure_review.schema.json
│   ├── workflow_turn_task.schema.json
│   └── workflow_decision_proposal.schema.json
├── validators/
│   ├── envelope_validator.ts
│   ├── domain_validator.ts
│   ├── workflow_proposal_validator.ts
│   ├── evidence_packet_validator.ts
│   ├── review_delta_validator.ts
│   └── closure_review_validator.ts
└── tests/
    ├── contracts.test.ts
    ├── fixtures/
    │   ├── valid/
    │   └── invalid/
    └── schema_manifest.test.ts
```

Skill 内已有 schema 可以作为迁移输入，但运行时权威 schema 必须由本目录统一发布；Skill 通过 manifest 引用相同内容，不保留分叉副本。

## 3. 公共身份、引用和哈希

```ts
export interface ObjectRef {
  objectType: string;
  objectId: string;
  revision: number;
}

export interface ArtifactRef {
  artifactId: string;
  kind: string;
  relativePath: string;
  sha256: string;
  sizeBytes: number;
  trustClass:
    | "canonical"
    | "validated_result"
    | "user_input"
    | "untrusted_log";
}

export interface RubricBinding {
  rubricId: string;
  version: string;
  sha256: string;
}

export interface TurnIdentity {
  protocolVersion: 1;
  messageType: string;
  workflowId: string;
  runId: string;
  taskId: string;
  attemptId: string;
  stageId: string;
  stageContractHash: string;
}

export interface StateBinding {
  snapshotVersion: number;
  canonicalRevision: number;
  eventCursor: number;
  workflowPlanRevision: number;
}

export interface PayloadTurnEnvelope<T> extends TurnIdentity {
  stateBinding: StateBinding;
  inputHash: string;
  payload: T;
}
```

普通 Worker/Evaluator Envelope 使用 `inputHash` 绑定完整输入；Workflow 控制消息使用语义更明确的 `decisionInputHash`。这些 hash 不属于状态版本本身。

Role 和 reasoning effort 也是共享 registry：

```ts
export type RegisteredRole =
  | "workflow_decision"
  | "evidence_reader"
  | "direction_reviewer"
  | "closure_reviewer";

export type ReasoningEffort = "high" | "max";

export const ROLE_REASONING_EFFORT: Readonly<
  Record<RegisteredRole, ReasoningEffort>
> = {
  workflow_decision: "max",
  evidence_reader: "high",
  direction_reviewer: "high",
  closure_reviewer: "high",
};

export type RegisteredTurnInputMessageType =
  | "WORKFLOW_TURN_TASK"
  | "EVIDENCE_READER_TASK"
  | "DIRECTION_REVIEW_TASK"
  | "CLOSURE_REVIEW_TASK";

export type RegisteredTurnOutputMessageType =
  | "WORKFLOW_DECISION_PROPOSAL"
  | "EVIDENCE_PACKET"
  | "REVIEW_DELTA"
  | "CLOSURE_REVIEW";

export const ROLE_MESSAGE_TYPES = {
  workflow_decision: {
    input: "WORKFLOW_TURN_TASK",
    output: "WORKFLOW_DECISION_PROPOSAL",
  },
  evidence_reader: {
    input: "EVIDENCE_READER_TASK",
    output: "EVIDENCE_PACKET",
  },
  direction_reviewer: {
    input: "DIRECTION_REVIEW_TASK",
    output: "REVIEW_DELTA",
  },
  closure_reviewer: {
    input: "CLOSURE_REVIEW_TASK",
    output: "CLOSURE_REVIEW",
  },
} as const satisfies Readonly<
  Record<
    RegisteredRole,
    {
      input: RegisteredTurnInputMessageType;
      output: RegisteredTurnOutputMessageType;
    }
  >
>;
```

`ROLE_REASONING_EFFORT` 是 workflow 的逻辑 effort 常量；`ROLE_MESSAGE_TYPES` 是唯一 role/input/output 映射。run config、CLI、StageContractDraft 和 Agent proposal 都不能覆盖。Attempt 必须同时记录逻辑 effort 和 provider wire value。Runtime capability manifest 必须显式声明 `max` 到该 provider 最高 effort 的映射；缺少映射时在 `doctor` 或 dispatch 前失败，不能静默降为 `high`。

规则：

1. 路径必须是 run 根目录或已批准 vault 根目录下的规范化相对路径。
2. Artifact 内容、大小或 trust class 变化时必须创建新 manifest entry。
3. 所有 hash 使用 canonical JSON UTF-8 bytes 的 SHA-256。
4. `untrusted_log` 只能作为数据读取，不能成为 action、Gate 或权限指令。
5. Response 必须回显 dispatch 时的 task、attempt、stage、contract 和 state binding。
6. 每个 Agent Skill 必须在 terminal output 前自检 expected schema、identity/binding、role/message 映射和唯一顶层 JSON。
7. raw response 的规范化不产生新的业务消息：只允许移除 BOM、统一换行、移除包裹完整响应的单一 Markdown fence，以及在整段响应中提取唯一一个 JSON value。
8. 规范化后仍无效时，Scheduler 只能依据原 task 和 validator report 创建同角色 fresh attempt；不能改写字段或转换成另一 Agent 角色。

## 4. Canonical 领域对象

### 4.1 TopicFrame

```ts
export type Layer = "L1" | "L2" | "L3" | "L4" | "L5" | "L6";

export interface TopicFrame {
  topicId: string;
  revision: number;
  userTopic: string;
  objective: string;
  workloads: string[];
  phases: string[];
  regimes: string[];
  stackScope: string[];
  layerScope: Layer[];
  targetMetrics: string[];
  invariants: string[];
  exclusions: string[];
  seedTerms: string[];
  synonymGroups: string[][];
  unresolvedScopeQuestions: string[];
  scopeAudit: {
    initialFingerprint: string;
    currentFingerprint: string;
    changes: Array<{
      changeId: string;
      field: string;
      changeType: "narrow" | "broaden" | "clarify";
      userAuthorized: boolean;
      reason: string;
    }>;
  };
}
```

`objectiveHash` 和 acceptance criteria 不存放在可由 Agent 修改的 TopicFrame 字段中，而由 WorkflowPlan 单独冻结。

### 4.2 Anchor

```ts
export interface Anchor {
  anchorId: string;
  topicId: string;
  revision: number;
  scenario: {
    workload: string;
    phase: string;
    regime: string;
    stack: string[];
  };
  baseline: {
    name: string;
    executionPath: string[];
    configuration: string[];
    comparisonScope: string[];
  };
  performanceTension: {
    symptom: string;
    suspectedMechanism: string;
    bottleneckResources: string[];
    targetMetrics: string[];
  };
  constraints: string[];
  evidenceRefs: string[];
  openNeedIds: string[];
  directionIds: string[];
  status: "candidate" | "active" | "saturated" | "rejected";
  statusReason: string;
  saturationReason: string | null;
}
```

Anchor 的领域 invariant 是“场景边界 × baseline 执行路径 × 性能张力”，不能只用一个方法名表示。

### 4.3 Direction

```ts
export interface ModificationAtom {
  atomId: string;
  layer: Layer;
  object: string;
  fromState: string;
  toState: string;
  role: "primary" | "enabler" | "alternative" | "constraint";
  conditions: string[];
  evidenceRefs: string[];
}

export interface Direction {
  directionId: string;
  anchorId: string;
  revision: number;
  title: string;
  hypothesis: string;
  changes: ModificationAtom[];
  causalLinks: Array<{
    from: string;
    to: string;
    relation: "causes" | "enables" | "controls" | "requires" | "conflicts";
    condition: string;
    evidenceRefs: string[];
    directness: "direct" | "inferred" | "unknown";
  }>;
  comparison: {
    baseline: string;
    controlledVariables: string[];
    ablations: string[];
  };
  expectedEffects: Array<{
    metric: string;
    expectedDirection: "increase" | "decrease" | "maintain";
    rationale: string;
  }>;
  implementation: {
    targetComponents: string[];
    knownEntryPoints: string[];
    unresolvedInterfaces: string[];
  };
  falsifiers: string[];
  degradationConditions: string[];
  supportingEvidenceRefs: string[];
  contradictingEvidenceRefs: string[];
  inferredClaims: string[];
  unresolvedNeedIds: string[];
  status:
    | "seed"
    | "exploring"
    | "testable"
    | "experiment_required"
    | "rejected";
  statusReason: string;
  experimentHandoffId: string | null;
}
```

`testable` 只表示实验候选定义完整，不表示已被本 workflow 实验验证。

## 5. 查询、证据和审阅对象

### 5.1 SearchNeed

```ts
export type SearchIntent =
  | "discover_anchor"
  | "define_baseline"
  | "find_modification"
  | "explain_mechanism"
  | "find_implementation"
  | "design_measurement"
  | "challenge_direction"
  | "verify_primary_source";

export type KnowledgeDimension =
  | "idea"
  | "knowledge"
  | "experiment"
  | "human"
  | "paper";

export interface SearchNeed {
  needId: string;
  revision: number;
  owner: {
    topicId: string;
    anchorId: string | null;
    directionId: string | null;
  };
  intent: SearchIntent;
  question: string;
  rationale: string;
  successCriteria: string[];
  primaryDimension: KnowledgeDimension;
  auxiliaryDimension: KnowledgeDimension | null;
  targetDimensions: KnowledgeDimension[];
  queryVariants: string[];
  technicalObjects: string[];
  knownTerms: string[];
  synonymGroups: string[][];
  scenarioTerms: string[];
  performanceRelations: string[];
  evidenceIntentTerms: string[];
  excludedSourceUnits: string[];
  previousAttemptIds: string[];
  critical: boolean;
  answerability:
    | "knowledge_base"
    | "experiment_only"
    | "unknown"
    | "not_applicable";
  status:
    | "pending"
    | "answered"
    | "no_delta"
    | "closed"
    | "experiment_only";
}
```

一个 SearchNeed 只有一个主要问题、一个主要维度和至多一个辅助维度。`targetDimensions` 必须严格等于 primary 加非空 auxiliary 的有序去重集合。

### 5.2 EvidencePacket

```ts
export interface EvidenceFinding {
  evidenceId: string;
  claimKey: string;
  claim: string;
  evidenceRole:
    | "scenario"
    | "baseline"
    | "mechanism"
    | "modification"
    | "implementation"
    | "measurement"
    | "constraint"
    | "counterexample"
    | "human_prior";
  directness: "direct" | "inferred";
  attribution: "source_report" | "workflow_inference";
  applicableConditions: string[];
  comparisonBaseline: string | null;
  sourcePath: string;
  sourceUnitId: string;
  sourceFamily: string;
  heading: string;
  quoteOrExactContext: string;
}

export interface EvidencePacket {
  packetId: string;
  needId: string;
  status: "complete";
  searches: unknown[];
  hitsConsidered: unknown[];
  contextsRead: unknown[];
  findings: EvidenceFinding[];
  contradictions: unknown[];
  unanswered: unknown[];
  conclusion: "answered" | "partial" | "not_found";
  conclusionRationale: string;
}
```

完整 query/read provenance 由 `evidence_packet.schema.json` 定义；TypeScript 中禁止用 `unknown[]` 作为最终实现，SC-2 必须替换为 schema 生成的具体类型。

### 5.3 SemanticDelta、ReviewDelta 和 Handoff

```ts
export interface SemanticDelta {
  deltaId: string;
  basisResultRefs: ObjectRef[];
  expectedTargetRevision: number;
  target:
    | { type: "topic"; id: string }
    | { type: "anchor"; id: string }
    | { type: "direction"; id: string };
  action:
    | "create"
    | "revise"
    | "add_evidence"
    | "add_contradiction"
    | "no_semantic_delta"
    | "reject";
  changedFields: string[];
  rationale: string;
  proposedObject: TopicFrame | Anchor | Direction | null;
}

export interface ExperimentHandoff {
  handoffId: string;
  directionId: string;
  tag: "EXPERIMENT_REQUIRED";
  reason: string;
  requiredArtifact:
    | "trace"
    | "prototype"
    | "benchmark"
    | "equivalence_test"
    | "code_audit"
    | "hardware_measurement";
  hypothesisToTest: string;
  suggestedEntryPoints: string[];
  controlledVariables: string[];
  metrics: string[];
  acceptanceCriteria: string[];
  failureStopConditions: string[];
  executionAuthorized: false;
}

export type DirectionReviewPurpose =
  | "initial"
  | "after_evidence"
  | "terminal_check"
  | "adversarial_recheck";

export interface DirectionDuplicateComparison {
  baselineScopeEquivalent: boolean;
  primaryChangeEquivalent: boolean;
  causalTargetEquivalent: boolean;
  materialDifference: string | null;
}

export interface ReviewDelta {
  reviewId: string;
  directionId: string;
  directionRevision: number;
  supportedParts: string[];
  evidenceRefsUsed: string[];
  weakestCausalLink: string | null;
  baselineProblem: string | null;
  implementationProblem: string | null;
  measurementProblem: string | null;
  strongestCounterexample: string | null;
  counterexampleResolution: string | null;
  nextQuestion: string | null;
  nextQuestionAnswerableFromKnowledgeBase: boolean;
  decision:
    | "continue_search"
    | "testable"
    | "experiment_required"
    | "rejected";
  rationale: string;
  duplicateOfDirectionRef: ObjectRef | null;
  duplicateComparison: DirectionDuplicateComparison | null;
  rejectionCategory:
    | "duplicate"
    | "out_of_scope"
    | "causal_contradiction"
    | "unfair_comparison"
    | "no_performance_mechanism"
    | "invalid_evidence"
    | "other"
    | null;
  readinessChecks: {
    inTopicAndAnchorScope: boolean;
    baselineFair: boolean;
    minimumChangeSetExplicit: boolean;
    causalChainFalsifiable: boolean;
    implementationPathBounded: boolean;
    measurementPlanComplete: boolean;
    falsifiersPresent: boolean;
    criticalCounterexampleResolved: boolean;
    evidenceTraceable: boolean;
    knowledgeAnswerableCriticalGapRemaining: boolean;
    newExperimentRequired: boolean;
  };
  experimentHandoff: ExperimentHandoff | null;
}
```

### 5.4 StopCandidate 与 ClosureReview

```ts
export interface StopCandidate {
  stopCandidateId: string;
  stopProofId: string;
  runId: string;
  topicId: string;
  canonicalRevision: number;
  reason: string;
}

export interface StopProofClaims {
  topicScopePreserved: boolean;
  noKnowledgeAnswerableCriticalNeed: boolean;
  allAnchorsClosed: boolean;
  allDirectionsTerminal: boolean;
  lastTopicExpansionNoDelta: boolean;
  noUnconsumedOrUncommittedWork: boolean;
  criticalContradictionsReviewed: boolean;
  experimentHandoffsComplete: boolean;
  runtimeEligibleForCompletion: boolean;
  finalOutputTraceable: boolean;
}

export interface StopProof {
  proofId: string;
  stopCandidateId: string;
  canonicalRevision: number;
  topicFrameRevision: number;
  anchorRefs: ObjectRef[];
  directionRefs: ObjectRef[];
  openNeedIds: string[];
  pendingTaskIds: string[];
  inFlightTaskIds: string[];
  pendingOutputRetryTaskIds: string[];
  unconsumedResultIds: string[];
  uncommittedDeltaIds: string[];
  unresolvedValidationFailureIds: string[];
  failedTaskIds: string[];
  unreviewedCriticalContradictionIds: string[];
  experimentHandoffIds: string[];
  lastTopicExpansionNeedId: string | null;
  outputCoverageProjectionId: string;
  claims: StopProofClaims;
}

export interface StopCandidateBundle {
  candidate: StopCandidate;
  proof: StopProof;
}

export interface ClosureChecks {
  stopProofRevisionCurrent: boolean;
  stopProofMatchesCanonical: boolean;
  mechanicalPreflightPassed: boolean;
  topicScopePreserved: boolean;
  noKnowledgeAnswerableCriticalNeed: boolean;
  allAnchorsClosed: boolean;
  allDirectionsTerminal: boolean;
  lastTopicExpansionNoDelta: boolean;
  noUnconsumedOrUncommittedWork: boolean;
  criticalContradictionsReviewed: boolean;
  experimentHandoffsComplete: boolean;
  runtimeEligibleForCompletion: boolean;
  finalOutputTraceable: boolean;
}

export type ClosureFindingType =
  | "knowledge_gap"
  | "state_inconsistency"
  | "incomplete_handoff"
  | "runtime_pause";

export type ClosureRecoveryAction =
  | "REOPEN_FRONTIER"
  | "REPAIR_STATE"
  | "COMPLETE_HANDOFF"
  | "RESUME_RUNTIME";

export type ClosureFindingCode =
  | "stale_stop_proof_revision"
  | "stop_proof_canonical_mismatch"
  | "mechanical_preflight_failed"
  | "topic_scope_silently_narrowed"
  | "knowledge_answerable_open_need"
  | "anchor_not_closed"
  | "anchor_missing_saturation_reason"
  | "anchor_missing_status_reason"
  | "direction_nonterminal"
  | "direction_missing_terminal_reason"
  | "last_topic_expansion_missing"
  | "last_topic_expansion_not_quiet"
  | "pending_task"
  | "in_flight_task"
  | "pending_output_retry"
  | "unconsumed_result"
  | "uncommitted_delta"
  | "unresolved_validation_failure"
  | "failed_task"
  | "unreviewed_critical_contradiction"
  | "experiment_handoff_missing"
  | "experiment_handoff_invalid"
  | "runtime_budget_exhausted"
  | "runtime_failed_or_paused"
  | "final_output_missing_field"
  | "final_output_untraceable";

export interface ClosureScopeRef {
  objectType: string;
  objectId: string;
  revision: number | null;
}

export interface ClosureBlockingFinding {
  findingId: string;
  check: keyof ClosureChecks;
  type: ClosureFindingType;
  code: ClosureFindingCode;
  summary: string;
  objectRefs: ClosureScopeRef[];
  reopenScope: ClosureScopeRef;
  recoveryAction: ClosureRecoveryAction;
}

export interface ClosureBasis {
  check: keyof ClosureChecks;
  statement: string;
  objectRefs: ClosureScopeRef[];
}

export type FinalizationRequirement =
  | "canonical_revision_unchanged"
  | "full_validator_passed"
  | "final_output_rendered"
  | "final_output_coverage_validated"
  | "atomic_completed_commit";

export interface ClosureReview {
  reviewId: string;
  stopCandidateId: string;
  canonicalRevision: number;
  status: "complete";
  decision: "accept" | "reject";
  verifiedClosureBasis: ClosureBasis[];
  closureChecks: ClosureChecks;
  blockingFindings: ClosureBlockingFinding[];
  reopenScopes: ClosureScopeRef[];
  allowsFinalization: boolean;
  finalizationRequirements: FinalizationRequirement[];
  rationale: string;
}
```

`StopProof.failedTaskIds` 只收录尚未被成功 retry、supersede 或显式 reconcile 的 unresolved failed task；历史上已解决的失败只留在 event log，不能永久阻塞闭包，也不能从审计中删除。

`closure_review.schema.json` 必须由这些类型生成或通过双向一致性测试；finding `type`、`code` 和 `recoveryAction` 的关系由注册 validator 固定。

## 6. Workflow 控制对象

### 6.1 WorkflowState

```ts
export type WorkflowLifecycle =
  | "initialized"
  | "running"
  | "waiting_turn"
  | "waiting_user"
  | "waiting_external"
  | "closure_preflight"
  | "waiting_closure_review"
  | "finalizing"
  | "paused_budget"
  | "paused_operator"
  | "failed_retriable"
  | "failed_terminal"
  | "blocked_semantic"
  | "blocked_external"
  | "completed"
  | "cancelled";

export interface WorkflowState {
  workflowId: string;
  runId: string;
  snapshotVersion: number;
  canonicalRevision: number;
  eventCursor: number;
  workflowPlanRevision: number;
  lifecycle: WorkflowLifecycle;
  currentStageId: string | null;
  activeFocusRef: ObjectRef | null;
  runnableStageIds: string[];
  pendingTaskIds: string[];
  inFlightTaskIds: string[];
  committedUnconsumedResultIds: string[];
  pendingProposalIds: string[];
  retryCounters: Record<string, number>;
  noProgressCounters: Record<string, number>;
  budgetState: unknown;
  pauseOrBlockReason: string | null;
}
```

SC-3 必须把 `budgetState` 替换为具体 schema。

### 6.2 Plan、Stage 和 Gate

固定 Stage 类型：

```ts
export type RegisteredStageType =
  | "SCRIPT_APPLY_TOPIC_FRAME"
  | "SCRIPT_APPLY_SEMANTIC_DELTA"
  | "WORKFLOW_DECISION"
  | "EVIDENCE_READ"
  | "DIRECTION_REVIEW"
  | "CLOSURE_REVIEW"
  | "RENDER_FINAL";

export type StageExecutionKind =
  | "SCRIPT_TRANSITION"
  | "DECISION_TURN"
  | "WORKER_TURN"
  | "EVALUATOR_TURN";
```

固定 Stage registry：

| Stage type | executionKind | role | expected Agent output | 创建权限 |
|---|---|---|---|---|
| `SCRIPT_APPLY_TOPIC_FRAME` | `SCRIPT_TRANSITION` | null | null | Workflow proposal 的 `RUN_STAGE` |
| `SCRIPT_APPLY_SEMANTIC_DELTA` | `SCRIPT_TRANSITION` | null | null | Workflow proposal 的 `RUN_STAGE` |
| `WORKFLOW_DECISION` | `DECISION_TURN` | `workflow_decision` | `WORKFLOW_DECISION_PROPOSAL` | Controller trigger engine only |
| `EVIDENCE_READ` | `WORKER_TURN` | `evidence_reader` | `EVIDENCE_PACKET` | Workflow proposal 的 `RUN_STAGE` |
| `DIRECTION_REVIEW` | `EVALUATOR_TURN` | `direction_reviewer` | `REVIEW_DELTA` | Workflow proposal 的 `REQUEST_EVALUATION` |
| `CLOSURE_REVIEW` | `EVALUATOR_TURN` | `closure_reviewer` | `CLOSURE_REVIEW` | Controller closure path only |
| `RENDER_FINAL` | `SCRIPT_TRANSITION` | null | null | Controller finalization only |

`REPLAN` 新增 Stage 也受同一创建权限约束。Workflow Agent 不能创建 `WORKFLOW_DECISION`、`CLOSURE_REVIEW`、`RENDER_FINAL`，不能用 `RUN_STAGE` 绕过 evaluator 路径，也不能把任一 Stage 绑定到表外 role/output。

```ts
export interface WorkflowPlan {
  workflowId: string;
  revision: number;
  objectiveHash: string;
  acceptanceCriteriaHash: string;
  stageNodes: StageNode[];
  dependencies: StageDependency[];
  planStatus: "active" | "superseded" | "closed";
}

export interface WorkflowPlanPatch {
  expectedPlanRevision: number;
  operations: Array<
    | { op: "add_stage"; stage: StageNodeDraft }
    | { op: "supersede_stage"; stageId: string; reason: string }
    | { op: "add_dependency"; dependency: StageDependency }
    | { op: "remove_dependency"; dependencyId: string; reason: string }
  >;
  objectiveHash: string;
  acceptanceCriteriaHash: string;
  rationale: string;
}
```

Patch 不支持任意 JSON Patch，不允许直接修改已冻结或已执行 Stage。

```ts
export interface EvidenceReadBudget {
  maxLogicalQueries: number;
  maxSearchToolCalls: number;
  maxHitsConsidered: number;
  maxSelectedSources: number;
  maxContextsRead: number;
}

export interface TurnBudget {
  timeoutMs: number;
  maxInputTokens: number;
  maxOutputTokens: number;
  maxToolCalls: number;
  evidenceRead: EvidenceReadBudget | null;
}

export interface StageContractDraft {
  proposalLocalStageKey: string;
  stageType: RegisteredStageType;
  objective: string;
  scope: ObjectRef[];
  executionKind: StageExecutionKind;
  role: RegisteredRole | null;
  requiredInputs: ArtifactRef[];
  expectedOutputMessageType: RegisteredTurnOutputMessageType | null;
  requestedTools: string[];
  requestedPaths: string[];
  prohibitedActions: string[];
  budget: TurnBudget;
}

export interface GateDefinitionDraft {
  proposalLocalStageKey: string;
  mechanicalChecks: MechanicalGateCheck[];
  semanticEvaluation: {
    required: boolean;
    evaluatorRole: RegisteredRole | null;
    rubricId: string | null;
    inputProjection: string[];
    expectedOutputMessageType: RegisteredTurnOutputMessageType | null;
  };
}
```

`TurnBudget` 是冻结的上限，不是 Agent 必须耗尽的配额。Controller 必须在接受 Stage 草案和 dispatch task 时执行同一组 role-specific invariant：

1. 所有数值均为正整数；只有零工具角色的 `maxToolCalls` 必须为 literal `0`。
2. `workflow_decision`、`direction_reviewer`、`closure_reviewer` 以及 `role = null` 的 script transition 必须满足 `maxToolCalls = 0`、`evidenceRead = null`。
3. `evidence_reader` 必须满足 `maxToolCalls > 0` 且 `evidenceRead != null`。
4. Evidence 的 `maxLogicalQueries` 在 `1..3`；Q1、Q2、Q3 是全 task 的 logical-query 序号。
5. `maxSearchToolCalls >= maxLogicalQueries` 且 `maxSearchToolCalls <= maxToolCalls`；opaque-cursor pagination 增加 search tool-call 计数，但不增加 logical-query 计数。
6. `maxHitsConsidered` 在 `1..50`，`maxSelectedSources <= maxHitsConsidered`，`maxContextsRead >= maxSelectedSources`。
7. task 中的 budget 必须与 frozen StageContract 中的 budget canonical-equal；Agent proposal、prompt 文本和 output 都不能放大它。
8. `timeoutMs`、token 和 tool 上限还必须小于等于 run config 与 permission envelope 的相应上限。运行时按实际 provider/tool event 计数，达到任一上限即停止该 attempt。

Domain validator 负责数值关系，runtime admission 负责实际消耗；二者都 fail closed。

Controller 接受草案后生成带 ID、revision、定义时 snapshot 和 hash 的冻结 `StageContract`、`GateDefinition`。

受限 Gate DSL：

```ts
type GateValueType =
  | "string" | "number" | "boolean" | "null"
  | "string_array" | "object";

type GateActual =
  | {source: "result"; pointer: string; valueType: GateValueType}
  | {source: "task"; pointer: string; valueType: GateValueType}
  | {source: "canonical"; objectRef: ObjectRef; pointer: string;
     valueType: GateValueType}
  | {source: "runtime"; fact: RuntimeGateFact; valueType: GateValueType}
  | {source: "validator"; fact: ValidatorGateFact;
     pointer: string | null; valueType: GateValueType}
  | {source: "artifact"; artifactId: string; fact: "exists" | "sha256";
     valueType: GateValueType};

type MechanicalGatePredicate = "equals" | "contains_fields";

interface MechanicalGateCheck {
  checkId: string;
  predicate: MechanicalGatePredicate;
  actual: GateActual;
  expected: string | number | boolean | null | string[];
}
```

Gate 中不允许任意 shell、代码、网络调用、动态正则或 Agent 生成表达式。
Workflow Agent 只提出 Stage 特定 criteria；它不提供完整权威 Gate：

1. `result` 只允许 output schema 中稳定的 `/payload` 根对象或
   `/payload/...` pointer；根对象只配合 `contains_fields`；
2. `task` 只允许 input schema 中稳定的 `/payload/...` domain pointer，禁止
   correction、Skill、schema、permission、identity 和 termination metadata；
3. `canonical` 当前只允许 Stage scope 内 exact ObjectRef 的数值 `/revision`；
4. `artifact` 必须来自 frozen `requiredInputs`；
5. runtime/validator fact 来自关闭 registry，并按固定类型解析；
6. object 只能用 `contains_fields`，禁止 whole-object equality；
7. whole-result validator facts 的 pointer 必须为 null；只有
   `references_resolve` / `source_context_present` 使用 schema-valid result
   pointer；
8. Agent criteria 最多 24 个，`checkId` 唯一且不能使用 `controller.` 前缀；
9. inline `semanticEvaluation` 必须关闭；语义判断通过单独 evaluator Stage。

Controller 在 dispatch 前编译草案，拒绝不解析、类型错误、权限越界或与强制
条件矛盾的 criteria，并注入不可覆盖的 schema、message binding、registered
domain validator、tool/path、budget、No Experiment、duplicate commit、
Evidence provenance 和 required artifact integrity 检查。有效 Gate 额外冻结
`stageContractHash`、Agent criteria hash、compiler policy version 和 evaluator
version 后整体哈希。

Gate engine 是 total/fail-closed 函数：operand 缺失、pointer 不解析、类型不符、
artifact byte/hash 不符或 evaluator version 不符都形成 failed check，而不是
抛异常或默认为通过。一个已通过 pre-Gate output validator 的结果若 Gate 失败，
只形成 workflow failure event；不得改写同一输出绕过 Gate。

同任务输出纠错使用以下独立输入对象：

```ts
interface TurnCorrectionFeedback {
  previousAttemptId: string;
  previousOutputSha256: string;
  validationReportId: string;
  validationReportSha256: string;
  failureClass:
    | "STRUCTURE_INVALID"
    | "BINDING_INVALID"
    | "SEMANTIC_INVALID";
  errors: Array<{
    code: string;
    jsonPointer: string | null;
    message: string;
    requiredRule: string;
    validExamples: string[];
  }>;
  retryInstruction: string;
}
```

错误包最多 32 条，字段长度受 schema 限制，且不包含上一份 raw response。
`message` 说明实际错误，`requiredRule` 是 Controller 权威规则，
`validExamples` 只包含 Controller 固定生成的合法形式，不复制 Agent 文本。
Controller 原子保存 raw artifact、ValidationReport、attempt failure 和 task
retry 状态；新 attempt 使用同一 logical task/Stage/Gate/role/Skill/effort，
只更新 attempt identity、input hash 和 `correctionFeedback`。状态变化、
Gate failure、安全违规或预算违规不走这条纠错路径。

Workflow permission 的 `suppliedResultRefs` 可包含已消费和未消费的
committed result，供 proposal 形成可审计 basis；`resultIndex`
保留两者的显式分类。只有 `SemanticDelta` 可以消费结果，且其
`basisResultRefs` 必须全部来自 `committedUnconsumedResultRefs`。

## 7. Workflow Turn 契约

```ts
export type WorkflowTrigger =
  | "INITIALIZE_TOPIC"
  | "COMMITTED_RESULT_REQUIRES_INTEGRATION"
  | "FRONTIER_SELECTION_REQUIRED"
  | "MULTIPLE_NON_EQUIVALENT_STAGES_RUNNABLE"
  | "GATE_FAILED_WITHOUT_RECOVERY_RULE"
  | "PLAN_EXHAUSTED_OBJECTIVE_OPEN"
  | "EVIDENCE_CONTRADICTION"
  | "NO_PROGRESS_THRESHOLD_REACHED"
  | "CLOSURE_REJECTED"
  | "NO_RUNNABLE_STAGE"
  | "USER_DECISION_REQUIRED";

export type WorkflowDecisionAction =
  | "RUN_STAGE"
  | "RETRY_STAGE"
  | "REPLAN"
  | "REQUEST_EVALUATION"
  | "ASK_USER"
  | "REPORT_BLOCKED"
  | "PROPOSE_PAUSE"
  | "PROPOSE_COMPLETE";
```

`WorkflowTurnTask` 必须包含：

- TurnIdentity；
- trigger；
- immutable objective 和 acceptance criteria；
- StateBinding；
- 当前 lifecycle、focus 和最小充分 domain projection；
- task/result index；
- approved artifacts、trigger report 和短 event tail；
- Workflow Skill name、version、SHA-256；
- action、stage type、role、tool、path 和 budget permission envelope。
- 对完整规范化输入计算的 `decisionInputHash`。

`WorkflowDecisionProposal` 必须回显全部绑定，并包含：

```ts
export type DomainProposal =
  | { kind: "topic_frame"; value: TopicFrame }
  | { kind: "search_need"; value: SearchNeed }
  | { kind: "semantic_delta"; value: SemanticDelta }
  | { kind: "direction_review_request"; value: ObjectRef }
  | { kind: "stop_candidate"; value: StopCandidateBundle };

export interface WorkflowDecisionProposal extends TurnIdentity {
  messageType: "WORKFLOW_DECISION_PROPOSAL";
  expectedState: StateBinding;
  decisionInputHash: string;
  proposalId: string;
  decision: WorkflowDecisionAction;
  reason: string;
  assumptions: string[];
  proposedStageContract: StageContractDraft | null;
  proposedGateDefinition: GateDefinitionDraft | null;
  proposedPlanPatch: WorkflowPlanPatch | null;
  targetStageId: string | null;
  domainProposal: DomainProposal | null;
  basisArtifactRefs: ArtifactRef[];
  basisResultRefs: ObjectRef[];
  requestedUserInput: unknown | null;
  blockedReport: unknown | null;
  pauseProposal: unknown | null;
}
```

SC-3 必须为最后三个 `unknown` 生成具体 schema。

Action/payload 矩阵：

| Action | 必需 | 必须为空 |
|---|---|---|
| `RUN_STAGE` | StageContractDraft、GateDefinitionDraft | targetStageId、blocked、pause、user input |
| `RETRY_STAGE` | targetStageId | 新合同、新 Gate、domain proposal |
| `REPLAN` | WorkflowPlanPatch | blocked、pause、user input |
| `REQUEST_EVALUATION` | evaluator StageContractDraft、GateDefinitionDraft | domain result |
| `ASK_USER` | requestedUserInput | Stage、Gate、blocked、pause |
| `REPORT_BLOCKED` | blockedReport | Stage、Gate、user input、pause |
| `PROPOSE_PAUSE` | pauseProposal | Stage、Gate、user input、blocked |
| `PROPOSE_COMPLETE` | StopCandidateBundle | Stage、Gate、plan patch、blocked、pause |

协议不定义 `STOP`、`DONE`、`GOTO`、`resumePoint` 或 Agent checkpoint。

## 8. 版本、草案和 canonical commit

1. Agent 输出中的新对象首先是 proposal-local draft。
2. Controller 预分配或重新映射 canonical ID。
3. 现有对象修改必须带 `expectedTargetRevision`。
4. Controller 在同一事务中校验、分配 revision、写事件并更新 projection。
5. stale draft 被拒绝，不能静默 rebase。
6. StopCandidate 和 ClosureReview 绑定完整 canonical revision。
7. Skill hash 或输入 artifact hash 变化会使未提交 proposal 失效。

## 9. Validator 分层

每条消息依次通过：

```text
JSON parse
→ JSON Schema
→ Turn identity / state binding
→ role and message-type registry
→ artifact hash / path / trust class
→ action-payload matrix
→ domain invariant
→ No Experiment invariant
→ stage-specific Gate
```

Validator 返回结构化报告：

```ts
interface ValidationReport {
  validatorVersion: string;
  valid: boolean;
  errors: Array<{
    code: string;
    jsonPointer: string | null;
    message: string;
  }>;
  checkedArtifactHashes: string[];
  checkedObjectRefs: ObjectRef[];
}
```

不得从自然语言猜测缺失 action、状态、ID 或完成含义。

## 10. No Experiment Invariants

以下字符串或等价 action 不能进入 tool registry、StageContract 或 task queue：

- shell/command execution；
- build/compile；
- benchmark/profile；
- GPU/cluster/remote job；
- target code write；
- experiment launch。

`ExperimentHandoff.executionAuthorized` 必须为 literal `false`。现有 `experiment_notes/` 只作为历史证据来源。

## 11. 规范实现检查项

以下 `SC-*` 是六份 implementation plan 共用的依赖门和验收分组，不构成独立的第七份实施计划。具体代码由 Scheduler 与对应 Agent 工作包按依赖关系交付。

### SC-1：Canonical JSON 与公共身份

交付：

- canonical JSON serializer；
- TurnIdentity、StateBinding、ArtifactRef；
- hash fixtures；
- path normalization。

验收：同一对象跨 key 顺序产生相同 hash；路径越界失败。

### SC-2：领域对象和 Evidence schema

交付：

- TopicFrame、Anchor、Direction、SearchNeed；
- Evidence task/packet；
- source/query/pagination provenance 具体类型；
- domain validators。

验收：没有来源上下文的 Evidence、越界 Direction 和空 Anchor 不能通过。

### SC-3：控制对象和 Workflow proposal

交付：

- WorkflowState、Plan/Patch、Stage/Gate；
- Workflow task/proposal；
- UserQuestion、BlockedReport、PauseProposal、BudgetState 具体类型；
- action/payload validator；
- role-specific TurnBudget validator，以及 task/StageContract budget equality validator。

验收：stale、未知 action、互斥字段、权限扩大、角色与工具预算不一致以及 task 放大冻结预算均失败。

### SC-4：Reviewer schema

交付：

- DirectionReviewTask/ReviewDelta；
- Direction readiness/decision/rejection-category matrix，以及 sibling Direction semantic-dedup projection/comparison；
- ClosureReviewTask/ClosureReview；
- thirteen closure checks 和 finding type/code/recoveryAction matrix。

验收：每个 response 均绑定唯一 source task 和当前 revision。

### SC-5：Schema manifest

生成：

```json
{
  "protocolVersion": 1,
  "schemas": {
    "WORKFLOW_TURN_TASK": {"path": "...", "sha256": "..."}
  }
}
```

Scheduler dispatch 和 Skill package 都记录 manifest hash。

### SC-6：Fixture 与兼容测试

覆盖：

- 每种合法消息；
- 每个 action；
- stale state；
- 非法 ID/revision/hash；
- raw log prompt injection；
- 未知字段和未知 Stage type；
- ExperimentHandoff 越权；
- schema version 不兼容；
- 任一角色 effort 与固定 registry 不一致或被请求降级；
- 四类 Agent 输出的合法/非法唯一 JSON、无语义规范化和同角色 retry binding；
- canonical role/message mapping，旧 `EVIDENCE_TASK` 被拒绝；
- Stage/executionKind/role/output/creation-authority registry 不一致或 Workflow self-scheduling 被拒绝；
- invalid task 在 dispatch 前失败且不进入 output retry；
- zero-tool role 的非零 tool/evidence budget、Evidence logical-query/search-call 混淆、超过 50 hits 和 task/StageContract budget 不一致均被拒绝；
- Direction 十一项 readiness/decision matrix；
- Direction `duplicate` 缺 sibling ref、category/check 不一致或 fabricated duplicate ref 被拒绝；
- Closure 十三项 checks、finding mapping 和 legacy Session/protocol-repair 字段拒绝；
- StopProof claims 只接受固定十项 projection；旧 `allAnchorsSaturated` 和未知 claim 被拒绝。

## 12. 完成标准

1. 所有计划引用同一个 protocol version 和 schema manifest。
2. 所有 `unknown` 占位已被具体类型替换。
3. TypeScript 类型与 JSON Schema 由同一来源生成或有双向一致性测试。
4. 四个 Turn 类型各只有一个允许输入和一个允许输出消息族。
5. action/payload、role/message 和 Stage type registry 都 fail closed。
6. canonical JSON、hash、revision 和 CAS 有固定测试向量。
7. No Experiment invariant 不能被 Agent proposal 覆盖。
8. shared contract 测试全部通过后，Scheduler 和 Agent 工作包才可开始集成。
9. role/effort registry 固定为 Workflow `max`、其余三个 Turn `high`，并有不可覆盖测试。
10. 契约中不存在格式修复或异常恢复 Agent、Stage、消息或 effort 项。
