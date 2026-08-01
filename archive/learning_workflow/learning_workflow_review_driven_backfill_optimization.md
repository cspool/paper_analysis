# Learning Workflow：Review-driven Backfill Loop 优化讨论稿

> 归档状态：旧版 Backfill Loop 讨论稿，仅供设计追溯。

> 文档状态：设计讨论稿，尚未实现  
> 创建日期：2026-07-28  
> 历史实现：`archive/learning_workflow/scripts/codex_learning_workflow.ts`
> 及 `archive/learning_workflow/scripts/codex_learning_workflow/`  
> 核心目标：把当前单向的 Stage 1 → Stage 2 流程优化为可恢复、可审计、成本有界的定向证据回填闭环

## 0. 背景与关联文档

本讨论稿建立在以下已有设计和实现之上：

- [learning_workflow_optimization_discussion.md](learning_workflow_optimization_discussion.md)：定义 Topic、Anchor、L1–L6 Entry、Edge、Direction 等业务对象。
- [learning_workflow_agent_orchestration_design.md](learning_workflow_agent_orchestration_design.md)：定义角色拆分、持久/短生命周期线程和脚本编排边界。
- [learning_workflow_codex_implementation_plan.md](learning_workflow_codex_implementation_plan.md)：定义 Codex App Server 新实现、协议、状态、校验和最终验收要求。
- [learning_workflow_reusable_knowledge_extraction.md](learning_workflow_reusable_knowledge_extraction.md)：总结可复用的专家审阅和证据组织原则。
- [learning_workflow_source_provenance.md](learning_workflow_source_provenance.md)：定义来源、引用、quote 和 provenance 要求。

当前实现采用：

```text
Stage 1：Anchor Explore
    ↓
冻结 anchor_space_version
    ↓
Stage 2：全部 Anchor 的 Direction Planning
    ↓
Stage 2：全部 Direction 的 Review
    ↓
Validation + Render
```

该结构保证了阶段边界清晰、历史不可静默改写，但 Stage 2 发现的结构缺口不能反向补充 Anchor graph。结果是 Reviewer 能指出问题，却无法驱动 canonical state 修复。

本讨论稿提出的不是无限循环，也不是让 Agent 自主重启前序阶段，而是：

> 由脚本把 Review 发现的缺口分类为结构化 Ticket，在固定版本和预算下定向回填，只重规划、重审受影响对象；无法通过资料补齐的问题明确交给实验。

## 1. 当前运行暴露出的主要问题

审计样本：

- Run：`learning_outputs_codex/multimodal_acceleration_acceptance_20260728`
- Topic：多模态推理加速相关探索
- Stage 1 Evidence Task：2
- accepted Anchor：10
- accepted Direction：10
- Review Evidence Task：60
- 累计 Codex App Server thread：85
- 累计 turn：149
- Review 决策：9 个 `needs_evidence`，1 个 `rejected`

证据分布：

- accepted EvidenceClaim：196
- Stage 1 产生：29
- Stage 2 产生：167
- 来自 ModServe 的 Claim：141，占 71.9%
- 其中 94 条来自 ModServe 的同一个 characterization 章节
- 不同 Review 的 Claim-ID 两两 Jaccard 平均值约 1.4%
- 不同 Review 的 source-path 两两 Jaccard 平均值约 32%，最高约 80%

这说明：

1. Stage 2 确实生成了大量新 Claim，并非简单复用相同 Claim ID。
2. 但新 Claim 高度集中于少量相同来源，证据数量增长没有转化为同等程度的来源多样性。
3. 同一论文或章节被切分成大量不同 statement，导致多个 Review 在语义和缺口上仍然相似。
4. Stage 1 只用两个宽任务形成十个 Anchor，Anchor 准入速度快于证据覆盖速度。
5. Stage 2 的 Evidence Worker 只能补充 Review ledger，不能新增或修复 canonical Entry、Edge、Baseline。
6. Reviewer 发现 graph inconsistency 后，当前 decision schema 仍容易把结果归入宽泛的 `needs_evidence`。

典型问题包括：

- Direction 假设使用了 canonical graph 中不存在的方法或实现对象；
- selected Entry 只描述 baseline behavior，却被 Planner 当作新的 intervention；
- 缺少资源控制接口、跨层张量契约或强 baseline；
- 需要目标 trace、最小原型、数值等价或性能测量，但系统仍继续搜索文献；
- rationale 与结构化 decision 不一致，例如 rationale 已判断“必须拒绝”，decision 仍为 `needs_evidence`。

## 2. 优化目标

### 2.1 功能目标

优化后的 workflow 应满足：

1. Review 的每个关键缺口都能被脚本识别和路由。
2. 可通过本地资料补齐的缺口进入 Evidence Backfill。
3. 缺少 Entry、Edge、Baseline 的问题进入 Anchor Graph Backfill。
4. 必须依靠代码、trace 或实验解决的问题进入 Experiment Ticket，不再无限检索资料。
5. 致命 graph inconsistency 进入拒绝或重新规划，不继续堆积普通 Claim。
6. Backfill 后只重新规划、重新审阅受影响对象。
7. 每次 canonical 变化都有版本、delta、来源和影响范围。
8. Loop 有明确的 round、task、turn、token 和 no-new-evidence 终止条件。

### 2.2 成本目标

1. 不再默认对每个 Direction 固定发起六个 Evidence Task。
2. 同一事实需求可跨 Direction 去重和共享。
3. Reviewer 已能从 canonical claims 回答的维度不再检索。
4. 相同来源不能通过拆分大量 Claim 人为提高证据覆盖度。
5. Backfill 不触发全量 Stage 1 或全量 Stage 2 重跑。

### 2.3 审计目标

1. Stage 1 原始 Anchor Space 不被静默覆盖。
2. 新增对象通过 append-only Backfill Delta 进入新版本。
3. 旧 Direction 被修订时保留 `supersedes` 关系。
4. 每个 Review 能追溯到使用的 Anchor Space 版本。
5. 每个终态能解释为什么停止以及下一步是资料补查、重规划、实验还是拒绝。

## 3. 非目标

本优化不追求：

- 让 Stage 1 与 Stage 2 无限互相调用；
- 让 Reviewer 直接修改 Anchor graph；
- 让 Evidence Worker 做最终价值裁决；
- 让 Planner 自己管理线程、创建 Worker 或决定全局预算；
- 用更多 Claim 数量代替来源独立性；
- 通过继续搜索文献解决只能由代码、trace 或实验回答的问题；
- 在 Backfill 时原地重写已经发布的历史对象。

## 4. 目标流程

```text
Topic
  ↓
Stage 1 Broad Discovery
  ↓
Candidate Anchor Admission
  ↓
Anchor Space v1
  ↓
┌──────────────────── Backfill Epoch N ────────────────────┐
│                                                          │
│  Dirty Anchor Planning                                   │
│      ↓                                                   │
│  Dirty Direction Canonical-only Review                   │
│      ↓                                                   │
│  Structured Gap Tickets                                  │
│      ↓                                                   │
│  Deterministic Gap Router                                │
│      ├─ source_gap ─────────→ Evidence Backfill           │
│      ├─ graph_gap ──────────→ Evidence + Curator Delta    │
│      ├─ experiment_gap ─────→ Experiment Ticket           │
│      └─ fatal_inconsistency → Reject / Replan             │
│                                  ↓                       │
│                          Backfill Merge Barrier           │
│                                  ↓                       │
│                          Anchor Space vN+1                │
│                                  ↓                       │
│                          Impact Analysis                  │
└──────────────────────────────────┬───────────────────────┘
                                   │
                    only dirty Anchor/Direction continue
                                   ↓
                    Terminal Reviews + Experiment Backlog
                                   ↓
                         Validation + Render
```

### 4.1 Epoch Barrier

为了保持并发下的确定性，每个 Backfill Epoch 使用版本屏障：

1. Epoch 开始时冻结 `anchor_space_version`。
2. 同一 Epoch 内所有 Planner、Reviewer、Evidence Worker 读取同一版本。
3. Worker 不直接修改 canonical graph，只返回 delta candidate。
4. 所有 delta 按稳定 ID 排序，在单线程 merge barrier 中验证、去重和提交。
5. 提交后生成新版本，并计算 dirty impact set。

这样可以避免：

- 一个 Planner 读取 v1，另一个同时读取被部分修改的 v1.5；
- 并发 Curator 按完成顺序产生非确定性结果；
- Reviewer 的结论引用已经被后台改写的 Entry。

## 5. Gap Ticket 数据模型

当前 `review.gaps: string[]` 应继续保留供人阅读，但脚本调度必须依赖结构化 `GapTicket`。

建议类型：

```ts
type GapKind =
  | "source_gap"
  | "graph_gap"
  | "experiment_gap"
  | "fatal_inconsistency";

type MissingObject =
  | "claim"
  | "baseline"
  | "entry"
  | "edge"
  | "implementation"
  | "measurement"
  | "trace"
  | "prototype";

interface GapTicket {
  gapId: string;
  anchorId: string;
  directionId: string;
  reviewRound: number;
  dimension: ReviewDimension;

  kind: GapKind;
  missingObject: MissingObject;
  question: string;
  rationale: string;
  successCriteria: string[];

  targetLayers: Layer[];
  targetEntryIds: string[];
  targetEdgeIds: string[];
  targetBaselineIds: string[];

  consumedClaimIds: string[];
  consumedSourcePaths: string[];
  excludedSourceFamilies: string[];

  priority: "high" | "medium" | "low";
  status:
    | "pending"
    | "dispatched"
    | "resolved"
    | "no_new_evidence"
    | "experiment_required"
    | "terminal";

  resolutionRefs: string[];
  resolutionDeltaId: string | null;
  terminalReason: string | null;
}
```

### 5.1 Stable Gap ID

建议：

```text
gap_id = stable_hash(
  direction_id
  + dimension
  + kind
  + missing_object
  + normalized_question
)
```

同一问题在后续 Review 中再次出现时应更新原 Ticket，而不是创建语义重复 Ticket。

### 5.2 Gap Ticket 的最小有效性要求

每个 Ticket 必须：

- 只绑定一个主要 Review dimension；
- 指定缺少的对象；
- 给出可判定的 success criteria；
- 说明已有 Claim 为什么不足；
- 列出已消费来源，避免无意识重复；
- 能被脚本确定性路由；
- 不以“继续研究”“寻找更多资料”等宽泛表述作为问题。

无有效 Ticket 的 `needs_evidence` 不能成为合法终态。

## 6. 确定性 Gap Router

Router 由脚本实现，不由 Agent 自主决定线程和全局阶段。

| 条件 | Gap kind | 动作 |
|---|---|---|
| 缺少直接来源或反例，但 canonical 对象已存在 | `source_gap` | 定向 Evidence Backfill |
| Hypothesis 使用了不存在的 Entry、Edge、Baseline 或实现对象 | `graph_gap` | Evidence → Curator → Graph Delta |
| 需要目标 trace、原型、数值等价、性能数据 | `experiment_gap` | 生成 Experiment Ticket |
| Entry 与 hypothesis 不一致、选择冲突对象、比较不可定义 | `fatal_inconsistency` | Reject 或 Replan |

### 6.1 Router 优先级

```text
fatal_inconsistency
  > graph_gap
  > experiment_gap
  > source_gap
```

原因：

- Graph 已无效时，继续补普通来源没有意义。
- Canonical 修改对象不存在时，应先修 Graph。
- 必须实验的问题不应继续消耗文献检索预算。
- 只有对象和验证路径都成立时，才补普通 source gap。

### 6.2 决策与下一步解耦

为了兼容现有输出，可保留：

```ts
decision:
  | "rejected"
  | "baseline_reference"
  | "needs_evidence"
  | "experiment_candidate";
```

新增：

```ts
nextAction:
  | "none"
  | "source_backfill"
  | "graph_backfill"
  | "experiment"
  | "replan";
```

例如：

```text
decision = needs_evidence
next_action = graph_backfill
```

比单独一个宽泛的 `needs_evidence` 更适合调度和阅读。

## 7. Stage 1 优化：Anchor Admission Gate

### 7.1 当前问题

当前 Curator 可以从一个较大的 Claim packet 中一次形成大量 accepted Anchor。这样虽然提高召回率，但会让证据覆盖不足的 Anchor 过早进入高成本 Stage 2。

### 7.2 Anchor 状态扩展

建议：

```text
candidate
→ accepted
→ enriched
→ superseded

candidate
↘ rejected
↘ rejected_cap
```

### 7.3 accepted Anchor 最低要求

Anchor 进入 Direction Planning 前至少需要：

1. 明确的 scenario 和 signature；
2. 至少一个 `current_practice` baseline；
3. 至少一个真正可修改的 Entry，而不只是 baseline behavior；
4. hypothesis 所需的主要修改对象已出现在 Entry graph；
5. 明确 target metrics；
6. baseline comparison scope 非空；
7. 至少两个独立 source family，或显式标记为 `single_source_candidate`；
8. 关键 Entry 有直接 Claim，或被明确标为 inferred opportunity；
9. 不存在尚未解决的致命 Anchor signature 冲突。

### 7.4 Curator Batch 限制

建议每个 Curator batch：

- 最多直接接纳 3–5 个新 Anchor；
- 其余作为 candidate；
- 优先 enrichment 现有 Anchor 的 layer、baseline 和 implementation gap；
- 不以 Anchor 数量作为唯一进度指标；
- 同时记录 `coverage_delta` 和 `independent_source_delta`。

## 8. Stage 2 Planning 优化

### 8.1 Readiness Gate

Planner 只处理：

- `accepted` 或 `enriched` Anchor；
- 具有至少一个 modifiable Entry；
- 具有可验证 baseline；
- 没有 high-priority unresolved graph gap。

否则脚本直接创建 Anchor-level Gap Ticket，不启动 Planner。

### 8.2 Planner 输出

Planner 仍然每轮只允许：

```text
DIRECTION_PROPOSAL
或
DIRECTION_PLANNING_COMPLETE
```

不让 Planner 创建 Worker 或控制 Backfill。

但 Planner input 应增加：

```text
anchor_space_version
backfill_delta_since_last_turn
superseded_direction_ids
resolved_gap_ids
unresolved_high_priority_gap_ids
```

### 8.3 Direction 修订

Backfill 导致 selected subgraph 或 hypothesis 改变时，不原地改写旧 Direction：

```ts
interface Direction {
  // existing fields...
  basedOnAnchorSpaceVersion: string;
  supersedesDirectionId: string | null;
  supersededByDirectionId: string | null;
  revision: number;
}
```

旧 Direction 保留：

```text
status = superseded
```

新 Direction 获得新稳定 ID，并重新 Review。

## 9. Stage 2 Review 优化

### 9.1 Pass A：Canonical-only Audit

Reviewer 首先只使用：

- Direction ExperimentBundle；
- 当前 Anchor graph；
- canonical claims；
- baseline registry；
- 已有 Review ledger；
- 当前版本 delta。

一次性判断六个维度：

```text
scenario_opportunity
baseline_fairness
entry_validity
cross_layer_validity
implementation_reuse
experiment_measurement
```

输出：

- 已充分支持的维度；
- 需要 source backfill 的维度；
- 存在 graph gap 的维度；
- 必须实验验证的维度；
- fatal inconsistency；
- 结构化 Gap Tickets。

这一阶段不自动为每个维度创建 Evidence Worker。

### 9.2 Pass B：按需 Evidence

只有 Router 接纳的 `source_gap` 或 `graph_gap` Ticket 才创建 Evidence Task。

Review Evidence Worker 仍然：

- 一次回答一个原子事实需求；
- 只使用 canonical claims 和 Obsidian 只读检索；
- 不做最终 decision；
- 不创建 Direction；
- 不修改 canonical graph。

### 9.3 局部重审

Backfill 后：

- 只新增 baseline Claim：重审 `baseline_fairness`；
- 只新增实现接口：重审 `implementation_reuse`；
- 只新增测量来源：重审 `experiment_measurement`；
- 新增 Entry/Edge：重新规划 Anchor，并完整审查新 Direction；
- Direction hypothesis 或 selected subgraph 未变化：保留未受影响 QA。

### 9.4 决策一致性规则

脚本必须验证：

```text
fatal_inconsistency 非空
OR cross_layer_validity = invalid
OR selected Entry 不支持 hypothesis 中的修改对象
→ decision 必须为 rejected 或 next_action=replan
```

```text
experiment_readiness = ready
AND baseline_quality >= fair
AND 无 high-priority Gap Ticket
→ 不允许继续 needs_evidence
```

```text
decision = needs_evidence
→ 至少存在一个非 terminal Gap Ticket
→ next_action 不能为空
```

```text
decision = experiment_candidate
→ 不允许存在 source_gap、graph_gap 或 fatal_inconsistency
```

## 10. Evidence Need 去重与来源多样性

### 10.1 EvidenceNeedKey

不同 Ticket 如果询问同一个事实，应共享一次 Evidence Task。

建议：

```text
evidence_need_key = stable_hash(
  normalized_question
  + missing_object
  + target_backend
  + target_version
  + source_scope
)
```

一个 Evidence Need 可有多个 consumer：

```ts
interface SharedEvidenceNeed {
  evidenceNeedId: string;
  consumerGapIds: string[];
  question: string;
  successCriteria: string[];
  excludedSourcePaths: string[];
  status: "pending" | "committed" | "no_new_evidence";
  resultClaimIds: string[];
}
```

### 10.2 Source Family

来源独立性不应只按文件路径计算。建议 source family 至少区分：

- 同一论文；
- 同一代码仓库；
- 同一官方文档；
- 同一实验笔记的上游论文；
- 独立实现；
- 独立 benchmark 或 trace。

同一论文被拆成多个 Markdown section，仍应主要计为同一个 source family。

### 10.3 防止 Claim 膨胀

新增报告字段：

```text
claim_count
unique_source_path_count
independent_source_family_count
direct_claim_count
inferred_claim_count
contradicting_claim_count
```

规则：

- 同一来源的多个 Claim 可以覆盖不同事实，但不能被当作多份独立验证；
- Backfill 优先检索未消费 source family；
- 再次使用旧来源时必须说明补充了哪个新事实；
- 如果只得到同义 Claim，结果为 `no_new_evidence`；
- Claim stable ID 和语义去重在 merge barrier 执行。

## 11. Graph Backfill

### 11.1 Backfill Evidence Task

`graph_gap` 应被转换成 Anchor 级原子任务，例如：

```json
{
  "gap_id": "G-...",
  "anchor_id": "A-...",
  "direction_id": "D-...",
  "missing_object": "entry",
  "target_layers": ["L2"],
  "question": "是否存在可独立控制视觉编码资源的具体接口，控制量是 TP、实例数、GPU、MIG、MPS 还是 SM quota？",
  "success_criteria": [
    "给出具体控制对象",
    "给出配置/API/代码入口",
    "说明改变该对象同时改变哪些混杂变量"
  ],
  "exclude": [
    "只报告端到端收益而不说明控制接口的来源"
  ]
}
```

### 11.2 Curator Enrichment Mode

Curator 增加：

```text
mode = new_anchor | enrichment
```

`enrichment` 模式下：

- 只能修改指定 Anchor；
- 不得创建无关 Anchor；
- 必须引用 Gap Ticket；
- 新增 Entry/Edge/Baseline 必须满足现有 domain validator；
- 不得删除旧对象；
- 可以提出 `supersedes`；
- 输出最小 `AnchorBackfillDelta`。

### 11.3 Backfill Delta

```ts
interface AnchorBackfillDelta {
  deltaId: string;
  baseAnchorSpaceVersion: string;
  targetAnchorId: string;
  sourceGapIds: string[];

  addedClaimIds: string[];
  addedBaselineIds: string[];
  addedEntryIds: string[];
  addedEdgeIds: string[];

  supersededObjectIds: string[];
  resolvedGapIds: string[];
  unresolvedGapIds: string[];
}
```

## 12. Experiment Ticket

### 12.1 为什么必须单独建模

以下问题通常不能通过继续搜索本地资料解决：

- 目标 vLLM/CroAttn 实际 kernel trace；
- 特定版本的 tensor layout 和 mask 兼容性；
- CUDA Graph 是否真实 capture/replay；
- 最小补丁是否可编译和运行；
- 数值等价；
- launch count、HBM bytes、GPU 时间、TTFT、TPOT、吞吐；
- 资源切换的实际同步和退化边界。

如果这些问题继续进入 Evidence Worker，会造成：

- 重复搜索相同论文；
- 产生更多“现有来源未给出”的 Claim；
- Review gap 越来越长；
- 仍无法升级到 experiment candidate。

### 12.2 数据模型

```ts
interface ExperimentTicket {
  experimentTicketId: string;
  anchorId: string;
  directionId: string;
  sourceGapIds: string[];

  requiredArtifact:
    | "trace"
    | "prototype"
    | "benchmark"
    | "equivalence_test"
    | "code_audit";

  hypothesis: string;
  targetImplementation: string[];
  commandsOrEntryPoints: string[];
  controlledVariables: string[];
  metrics: string[];
  acceptanceCriteria: string[];
  failureStopConditions: string[];

  status: "proposed" | "ready" | "running" | "complete" | "blocked";
  resultRefs: string[];
}
```

### 12.3 Review 终态

一个 Direction 可以：

```text
decision = needs_evidence
next_action = experiment
```

此时 Review 对资料阶段是 terminal，不再继续 Backfill Loop；后续由实验工作流消费 Experiment Ticket。

## 13. Canonical 版本和影响分析

### 13.1 Anchor Space Version

```ts
interface AnchorSpaceVersionRecord {
  versionId: string;
  parentVersionId: string | null;
  createdAt: string;
  reason: "stage1_complete" | "backfill_merge";
  deltaIds: string[];
  canonicalDigest: string;
}
```

历史：

```text
AS-v1：Stage 1 完成
AS-v2：应用 BF-001、BF-002
AS-v3：应用 BF-003
```

### 13.2 Impact Set

```ts
interface ImpactSet {
  deltaId: string;
  dirtyAnchorIds: string[];
  dirtyDirectionIds: string[];
  dirtyReviewDimensions: Record<string, ReviewDimension[]>;
  requiresReplanAnchorIds: string[];
  requiresFullReviewDirectionIds: string[];
}
```

确定性影响规则示例：

- 新 Claim 被现有 Baseline 引用：该 Direction 的 `baseline_fairness` dirty；
- 新 Claim 被现有 Entry 引用：对应 `entry_validity` dirty；
- 新增 Edge：该 Anchor 需要 replan；
- 新增 modifiable Entry：该 Anchor 需要 replan；
- Direction selected subgraph 改变：新 Direction 需要 full review；
- 只补充 experiment measurement source：只重审 `experiment_measurement`。

## 14. 调度算法草案

```ts
async function runLearningWorkflow(state: RunState): Promise<void> {
  await runAnchorStage(state);
  freezeAnchorSpaceVersion(state, "stage1_complete");

  for (
    let epoch = state.backfill.epoch;
    epoch <= state.config.maxBackfillRounds;
    epoch += 1
  ) {
    const snapshotVersion = state.anchorSpace.currentVersionId;

    await planDirtyAnchors(state, snapshotVersion);
    await canonicalAuditDirtyDirections(state, snapshotVersion);

    const tickets = validateAndRouteGapTickets(state);
    emitExperimentTickets(tickets.experiment);
    terminalizeFatalDirections(tickets.fatal);

    const sharedNeeds = deduplicateEvidenceNeeds([
      ...tickets.source,
      ...tickets.graph,
    ]);

    if (sharedNeeds.length === 0) {
      break;
    }

    const evidenceResults = await runBackfillEvidence(sharedNeeds);
    const deltaCandidates = await curateGraphBackfills(
      tickets.graph,
      evidenceResults,
    );

    const mergeResult = mergeBackfillAtBarrier(
      state,
      snapshotVersion,
      evidenceResults,
      deltaCandidates,
    );

    if (!mergeResult.hasCanonicalDelta) {
      state.backfill.noNewEvidenceRounds += 1;
    } else {
      state.backfill.noNewEvidenceRounds = 0;
      state.anchorSpace.currentVersionId = mergeResult.newVersionId;
      applyImpactSet(state, mergeResult.impactSet);
    }

    checkpoint(state, `backfill_epoch_${epoch}`);

    if (
      state.backfill.noNewEvidenceRounds
      >= state.config.noNewBackfillStop
    ) {
      break;
    }
  }

  terminalizeUnresolvedTickets(state);
  validateAndRender(state);
}
```

## 15. 并发策略

### 15.1 可并发阶段

- 同一 snapshot 上不同 Anchor 的 Planner；
- 同一 snapshot 上不同 Direction 的 canonical-only Reviewer；
- 去重后的独立 Evidence Need；
- 只读验证任务。

### 15.2 必须串行阶段

- canonical Claim merge；
- Curator delta merge；
- Anchor Space version commit；
- Impact Set 计算和 dirty 状态更新；
- final validation 和 render。

### 15.3 建议默认配置

```text
direction_concurrency = 2
backfill_evidence_concurrency = 3
curator_concurrency = 1

max_backfill_rounds = 2
max_backfill_tasks_per_round = 8
max_backfill_tasks_per_direction = 2
max_replans_per_anchor = 1
no_new_backfill_stop = 2
min_independent_source_families = 2
```

这些值应通过小规模 canary 调整，不应直接视为最终生产参数。

## 16. 线程生命周期

角色生命周期保持：

| 角色 | 生命周期 |
|---|---|
| Anchor Controller | 每个 Stage 1 一个 persistent thread |
| Anchor Evidence Worker | 每个原子 discovery/backfill task 一个 ephemeral thread |
| Anchor Curator | 每个 delta batch 一个 ephemeral thread |
| Direction Planner | 每个 Anchor/version-aware planning scope 一个 persistent thread |
| Direction Reviewer | 每个 Direction/revision 一个 persistent thread |
| Review Evidence Worker | 每个去重后的原子 evidence need 一个 ephemeral thread |

新增要求：

- Persistent Planner 达到 Direction cap 后必须收到终止 turn；
- 被 superseded 的 Planner/Reviewer session 必须 terminal；
- Backfill 不能复用已 contaminated 或版本不匹配的 session；
- session record 增加 `anchor_space_version` 或 `direction_revision`；
- Loop 完成后不允许存在无 pending reason 的 `yielded` session；
- Evidence Need 被多个 Direction 消费时仍只创建一个 ephemeral thread。

## 17. Skill 和协议调整

### 17.1 Direction Reviewer

Completion payload 增加：

```json
{
  "decision": "needs_evidence",
  "next_action": "graph_backfill",
  "gap_tickets": []
}
```

Reviewer：

- 提出和分类 Gap；
- 不创建 Worker；
- 不直接检索；
- 不修改 graph；
- 不决定全局 Loop 是否继续。

### 17.2 Anchor Evidence Worker

支持两种输入：

```text
mode = discovery
mode = backfill
```

Backfill 模式必须包含：

- Gap ID；
- 指定 Anchor/Direction；
- success criteria；
- 已消费和排除来源；
- 不得扩展到无关 Topic frontier。

### 17.3 Anchor Curator

支持：

```text
mode = new_anchor
mode = enrichment
```

Enrichment 模式必须只返回最小 delta。

### 17.4 Direction Planner

获得：

- exact Anchor Space version；
- 上一版 Direction；
- Backfill Delta；
- resolved/unresolved Gap；
- script commit result。

Planner 可以：

```text
keep
revise
complete
```

协议层仍映射为 proposal 或 completion，`keep/revise` 由 payload/revision 字段表达。

## 18. Validation Invariants

新增 deterministic validator：

### 18.1 Gap 完整性

- `needs_evidence` 必须有至少一个有效 Gap Ticket；
- 每个 Gap Ticket 必须有 `kind`、`missingObject` 和 success criteria；
- terminal Gap 必须有 terminal reason；
- resolved Gap 必须引用 Claim、Delta 或 Experiment Ticket。

### 18.2 决策一致性

- fatal inconsistency 不得对应 experiment candidate；
- invalid cross-layer graph 不得对应 experiment candidate；
- rationale 判定必须拒绝时，decision 不得仍为普通 needs_evidence；
- experiment candidate 不得存在 high-priority source/graph gap；
- experiment-required Direction 必须有 Experiment Ticket。

### 18.3 版本一致性

- Planner、Direction、Review 必须记录使用的 Anchor Space version；
- Delta 的 base version 必须等于 merge barrier snapshot；
- 旧 Direction 修订后必须保留 supersedes 链；
- 同一版本的 canonical digest 必须确定性一致。

### 18.4 生命周期

- complete planning 的 persistent Planner session 必须 terminal；
- complete Review 的 Reviewer session 必须 terminal；
- pending 状态必须有 pending reason；
- ephemeral task 必须 committed、failed_terminal 或 no_new_evidence；
- 不允许已 terminal session 被静默 resume。

### 18.5 来源多样性

- 输出必须报告 source path 和 source family 数量；
- 同一论文的不同 section 不计为独立 source family；
- source diversity 不足必须成为显式 warning 或 Gap；
- no-new-evidence 必须记录搜索范围和已排除来源。

## 19. 输出结构优化

建议新增：

```text
<work-dir>/
├── executive_summary.md
├── final.md
├── state.json
├── validation.json
├── backfill/
│   ├── tickets.json
│   ├── shared_evidence_needs.json
│   ├── epoch_01_delta.json
│   ├── epoch_01_impact.json
│   └── epoch_02_delta.json
├── versions/
│   ├── anchor_space_v1.json
│   ├── anchor_space_v2.json
│   └── index.json
├── experiments/
│   ├── tickets.json
│   └── <experiment-ticket-id>.md
├── directions/
├── reviews/
└── evidence/
```

### 19.1 executive_summary.md

只提供决策入口：

| Direction | Decision | Readiness | 首要阻塞 | Next action | Version |
|---|---|---|---|---|---|
| D-... | needs_evidence | partial | 缺目标 trace | experiment | AS-v2 |
| D-... | needs_evidence | not_ready | 缺方法 Entry | graph_backfill | AS-v1 |
| D-... | rejected | not_ready | graph inconsistency | none | AS-v1 |

### 19.2 final.md

继续作为完整 canonical registry，但：

- 每个 Direction 只显示最多三个最高优先级 unresolved Gap；
- 完整 Gap 链接到 Ticket；
- 显示 `next_action`；
- 显示 Anchor Space version 和 Direction revision；
- 显示 source-family count，而不只显示 Claim ID；
- 显示 Backfill history；
- Experiment Ticket 单独链接；
- 不在全局 Unresolved gaps 中重复打印所有维度的同义缺口。

### 19.3 Review Markdown

推荐顺序：

```text
Decision / Next action
Rationale
Top blockers
Gap Tickets
Minimum implementation plan
Baseline and ablation
Metrics and tools
Failure conditions
Review ledger
Evidence trace
Backfill history
```

## 20. 与当前代码的建议映射

### `types.ts`

新增：

- `GapTicket`
- `SharedEvidenceNeed`
- `AnchorBackfillDelta`
- `AnchorSpaceVersionRecord`
- `ImpactSet`
- `ExperimentTicket`
- Review `nextAction`
- Direction version/revision/supersedes 字段
- Backfill canonical state

### `direction_stage.ts`

将当前：

```text
plan all anchors
→ review all directions
→ complete
```

重构为 epoch controller：

```text
plan dirty anchors
→ canonical-only review dirty directions
→ route tickets
→ backfill
→ impact
→ repeat
```

### `anchor_stage.ts`

抽取可复用的：

- Evidence task dispatch；
- Claim validation；
- Curator batch；
- deterministic merge。

Discovery 和 Backfill 共用底层执行器，但使用不同输入约束。

### 新增 `backfill_stage.ts`

负责：

- Gap routing；
- Evidence Need 去重；
- Backfill task dispatch；
- Curator enrichment；
- merge barrier；
- version commit；
- impact analysis；
- no-new-evidence 终止。

### `domain_validators.ts`

新增：

- Gap Ticket validator；
- Anchor Admission Gate；
- decision consistency；
- Backfill Delta validator；
- supersedes validator；
- source-family independence validator。

### `validators.ts`

新增：

- Backfill terminal invariant；
- version integrity；
- dirty object terminal state；
- Experiment Ticket reachability；
- persistent session terminal check；
- final decision/Gap consistency。

### `renderer.ts`

新增：

- executive summary；
- next action；
- source diversity；
- Backfill history；
- Experiment Ticket；
- Top blocker 渲染；
- Gap 全量内容外置。

### Skills

修改六个 workflow skill 的协议说明，但继续保持：

- 一 Agent 一 Skill；
- Evidence role 只读知识库；
- blind role 无工具；
- 不允许子代理；
- 不允许 Agent 自主管理线程；
- semantic merge 和最终状态由脚本裁决。

## 21. 分阶段实施计划

### Phase A：结构化 Gap 与一致性校验

实现：

- `GapTicket`
- Review `nextAction`
- Gap parser/validator
- decision consistency validator
- renderer 显示 Top Gap 和 next action

暂不执行 Backfill。

验收：

- 100% `needs_evidence` Review 有可路由 Ticket；
- fatal inconsistency 不再被普通 needs_evidence 掩盖；
- rationale 与 decision 冲突会 validation fail。

### Phase B：Canonical-only Review 与按需 Evidence

实现：

- Reviewer 首轮 canonical-only audit；
- 只为有效 Ticket 创建 Evidence Task；
- 局部维度 QA ledger；
- Evidence Need 去重。

验收：

- 不再固定每 Direction 六个 Evidence Worker；
- 相同 Evidence Need 跨 Direction 只执行一次；
- 已有 canonical evidence 足够的维度不重新检索。

### Phase C：Source Backfill

实现：

- consumed/excluded source；
- source-family 统计；
- no-new-evidence；
- resolved Ticket；
- 只重审 dirty dimension。

验收：

- 同一来源的同义 Claim 不构成新进展；
- 连续 no-new-evidence 能终止；
- Review 可从 needs_evidence 进入更明确终态。

### Phase D：Graph Backfill 与版本化

实现：

- Curator enrichment mode；
- AnchorBackfillDelta；
- Anchor Space version；
- merge barrier；
- Impact Set；
- selective replan。

验收：

- Stage 2 能补充缺失 Entry/Edge/Baseline；
- v1 历史保持不变；
- 只重规划受影响 Anchor；
- 旧 Direction 保留 supersedes 链。

### Phase E：Experiment Ticket

实现：

- experiment gap router；
- Experiment Ticket；
- 输出实验 backlog；
- 资料阶段 terminal 状态。

验收：

- trace/prototype/benchmark 问题不再反复搜索文献；
- 每个 experiment-required Direction 有可执行验收条件；
- Experiment Ticket 能被后续实验工作流消费。

### Phase F：输出和成本优化

实现：

- executive summary；
- Top blocker；
- source diversity；
- Backfill history；
- token/turn/elapsed 细粒度统计；
- per-epoch budget。

验收：

- 人可以先读 summary 再按 Direction 下钻；
- final 不再被重复 Gap 淹没；
- 能比较优化前后线程、turn、token、来源多样性和决策质量。

## 22. 测试矩阵

### 确定性单元测试

- Gap stable ID；
- Gap schema validation；
- Gap Router 优先级；
- decision consistency；
- Evidence Need 去重；
- source-family 归并；
- no-new-evidence；
- Anchor Admission Gate；
- Backfill Delta merge；
- version digest；
- Impact Set；
- selective replan；
- Direction supersedes；
- Experiment Ticket reachability；
- Top blocker 排序；
- deterministic render。

### 状态与恢复测试

- Backfill Evidence 完成前进程退出；
- Curator delta 生成后、merge 前退出；
- merge barrier 中断；
- version commit 后、impact save 前退出；
- persistent Planner/Reviewer resume；
- skill hash 或 model/effort 变化后轮换；
- event replay 恢复最新完整版本；
- truncated event 忽略；
- 重放不会重复应用同一 Delta。

### 并发测试

- 多 Ticket 共享 Evidence Need；
- 两个 Evidence Task 同时返回重复 Claim；
- 两个 Curator delta 同时修改同一 Anchor；
- merge 顺序不影响最终 digest；
- dirty set 不遗漏消费者；
- 达到全局预算时 active task 能安全收尾。

### 安全测试

- blind Reviewer 尝试直接检索；
- Planner 尝试创建 Agent；
- Evidence Worker 尝试写文件；
- repair turn 使用工具；
- Backfill task 越出指定 Anchor/Direction scope；
- Curator enrichment 创建无关 Anchor。

### Canary 对比

在相同 Topic、相同知识库 snapshot 下比较当前 flow 和优化 flow：

- accepted Anchor 数量；
- candidate/accepted 比例；
- Direction 数量；
- Review decision 分布；
- `needs_evidence` 的 next-action 完整率；
- Claim 数量；
- unique source path；
- independent source family；
- 重复 Evidence Need；
- Evidence Worker thread 数；
- 总 turns/tokens/elapsed；
- Backfill canonical delta 数；
- experiment ticket 数；
- 人工判断的一致性；
- 是否出现 decision/rationale 矛盾。

不以“experiment candidate 越多越好”为验收目标。更重要的是：

- 错误候选被正确拒绝；
- 可补结构问题被定向修复；
- 必须实验的问题被准确交接；
- 成本与证据多样性可解释。

## 23. 最终验收标准

优化版可以宣称 Review-driven Backfill Loop 生效，至少需要：

1. Stage 1 原始版本和所有 Backfill 版本均可恢复。
2. 每个 `needs_evidence` 有结构化 Gap Ticket 和 next action。
3. `source_gap`、`graph_gap`、`experiment_gap`、`fatal_inconsistency` 能确定性路由。
4. 同一 Evidence Need 可被多个 Direction 共享。
5. Backfill 能新增 canonical Claim、Baseline、Entry 或 Edge。
6. Graph 修改后只重规划受影响 Anchor。
7. 未受影响的 Review QA 不被重复执行。
8. Direction 修订保留 supersedes 链。
9. 必须实验的问题生成 Experiment Ticket 并终止资料 Backfill。
10. 连续 no-new-evidence 或预算耗尽后 Loop 确定性停止。
11. `experiment_candidate` 不含 high-priority source/graph/fatal Gap。
12. decision/rationale 不一致会 validation fail。
13. complete Planner 和 Reviewer persistent session 均为 terminal。
14. final 输出包含 next action、Top blocker、版本和 Backfill history。
15. event log 能重建最新完整 canonical 版本和 Backfill 状态。

## 24. 待讨论问题

后续讨论需要明确：

1. accepted Anchor 是否强制要求两个独立 source family，还是允许高价值单来源 candidate 进入 Stage 2？
2. 每个 Curator batch 最多直接接纳多少 Anchor？
3. `max_backfill_rounds` 默认是 1、2 还是按 Anchor readiness 动态决定？
4. 一个 `graph_gap` 是否允许复活已经 rejected 的 Direction，还是必须产生新 Direction revision？
5. `fatal_inconsistency` 的 deterministic 规则应严格到什么程度？
6. Reviewer canonical-only audit 是一次覆盖六维，还是仍保持一次一维但不检索？
7. Shared Evidence Need 可以服务多少 Direction，如何防止 applicable scope 被错误扩展？
8. source family 如何识别同一论文的不同 Markdown section、二手笔记和上游原文？
9. Experiment Ticket 是本 workflow 的最终产物，还是直接接入已有实验工作流？
10. Backfill 后的 Anchor Space version 是全局版本还是 per-Anchor 版本？
11. Direction revision 是否保留原 ID 加 revision，还是内容变化后生成全新稳定 ID？
12. executive summary 的排序由哪些字段决定，是否允许 Agent 排名，还是完全由脚本评分？
13. 完整 Gap ledger 是否保留在 Review Markdown，还是只保留 JSON 并渲染 Top blocker？
14. 如何为 Backfill 单独设置 token budget，避免 Review Loop 消耗超过 Stage 1？
15. 对当前历史运行是否只用于对比，还是提供一次迁移工具把自然语言 gap 转换为初始 Ticket？

## 25. 建议结论

当前 flow 的核心优点是阶段边界清晰、canonical state 可审计；主要缺点是 Stage 2 只能发现问题，不能修复 Anchor graph，也不能区分“继续找资料”和“必须做实验”。

建议保留：

- Stage 1 broad discovery；
- Anchor Space snapshot；
- per-Anchor Planner；
- per-Direction Reviewer；
- ephemeral Evidence Worker；
- 脚本管理生命周期和 canonical merge；
- validate-before-render。

在此基础上增加：

```text
Structured Gap Ticket
+ Deterministic Gap Router
+ Canonical-only Review
+ On-demand Shared Evidence
+ Versioned Graph Backfill
+ Selective Replan/Re-review
+ Experiment Ticket
+ Bounded Termination
```

目标不是让流程运行得更久，而是让每一次额外 Evidence Task 都对应一个明确缺口、一个可验证成功条件和一个确定性的后续状态。
