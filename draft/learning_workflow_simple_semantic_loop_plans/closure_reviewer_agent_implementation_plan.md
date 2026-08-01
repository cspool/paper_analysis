# Closure Reviewer Turn Agent 实现计划

## 1. 计划定位

本文实现 Topic 级语义闭包的独立 Evaluator。

- 角色：`closure_reviewer`
- Skill：`learning-semantic-loop-closure-reviewer`
- 生命周期：一个通过机械 preflight 的 StopCandidate 对应一个 fresh Turn
- Reasoning effort：固定 `high`
- 工具：零工具
- 输出：一个 `CLOSURE_REVIEW`
- 状态写入和完成提交：禁止

依赖：

- [共享契约规范](shared_contracts.md)
- [Workflow Turn Agent 实现计划](workflow_turn_agent_implementation_plan.md)
- [Scheduler Script 实现计划](scheduler_script_implementation_plan.md)

现有 `.codex/skills/learning-semantic-loop-closure-reviewer/` 只作为 closure checks 和 adversarial fixture 的迁移输入，不能直接注册到新 Controller：它仍含 provider Session、pending protocol-repair 和旧 `allAnchorsSaturated` 契约。

## 2. 唯一完成路径中的位置

```text
Workflow Turn:
  PROPOSE_COMPLETE + StopCandidateBundle
→ Controller:
  schema/revision/task/mechanical preflight
→ fresh Closure Reviewer Turn:
  accept | reject
→ reject:
  Controller commit review
  + trigger CLOSURE_REJECTED Workflow Turn
→ accept:
  Controller recheck unchanged revision
  + full validators
  + deterministic render
  + output coverage
  + atomic completed commit
```

Closure Reviewer 的 `accept` 只是 finalization permission，不是 `completed`。

## 3. 职责

Reviewer 只回答：

> 当前 canonical projection 和 StopProof 是否足以支持整个 Topic 的语义闭包？

它检查：

1. StopCandidate/StopProof 与 canonical revision；
2. Scheduler mechanical preflight；
3. Topic scope integrity；
4. 最后一次 Topic `discover_anchor`；
5. Anchor saturation/rejection；
6. Direction terminal state；
7. critical SearchNeed；
8. pending/in-flight/unconsumed/uncommitted work；
9. critical contradiction；
10. ExperimentHandoff；
11. budget、protocol、runtime 伪完成；
12. final output coverage projection。

它不判断 Direction 价值，不生成 SearchNeed，不继续探索，不修复状态。

## 4. 输入契约

输入为：

```ts
PayloadTurnEnvelope<ClosureReviewTask>
```

Envelope 必须使用 `messageType = "CLOSURE_REVIEW_TASK"`，并包含当前 TurnIdentity、StateBinding、`inputHash` 和 `stageContractHash`。Controller 构造的 dispatch packet 还必须固定 Skill name/version/hash、schema manifest hash、frozen StageContract/GateDefinition、zero-tool permission envelope、expected output schema 和本 Turn 终止条件。其 task budget 必须与 frozen StageContract canonical-equal，并固定 `maxToolCalls = 0`、`evidenceRead = null`。

Task 必须包含：

- StopCandidate 和 StopProof；
- current canonical revision；
- TopicFrame 和 scope audit；
- 全部 Anchor closure projection；
- 全部 Direction terminal projection；
- Need、task、result、delta、output-attempt 和 validation-failure indexes；
- recent semantic/no-delta records；
- final Topic expansion record；
- contradiction index；
- ExperimentHandoff index；
- Controller mechanical preflight report；
- fixed closure rubric ID/version/hash；
- budget、pause、failure 和 runtime eligibility；
- output coverage projection；
- `freshTurn = true`、`providerHistoryIncluded = false` 和 `canonicalOnly = true` assertions。

Controller 只有在 task schema、candidate/proof revision、registered closure rubric hash、mechanical preflight 和上述 independence assertions 全部通过后才能 dispatch。Closure rubric 由 Controller registry 固定，Workflow proposal 不能替换。Reviewer 在 Turn 开始时重新绑定；若认为事实缺失或不一致，不允许从记忆补全、形成 accept/reject 或自行写 attempt 状态，而应拒绝产生业务结果。Controller 重新运行权威 task validator：只有 validator 也失败时才记录 `input_contract_invalid`，且不重试同一无效 task；若 task 仍有效，该拒绝作为无效产出进入普通同角色 fresh retry。协议不为此增加错误消息族。

## 5. 审阅算法

```text
BIND candidate/proof to canonical revision
RECHECK mechanical closure facts
CHECK Topic scope and final expansion
CHECK every Anchor
CHECK every Direction
CHECK critical Needs and contradictions
CHECK experiment isolation
CHECK false-completion causes
CHECK output traceability
SELECT accept or reject
SELF-CHECK expected schema, bindings, check/decision consistency and one-JSON rule
EMIT CLOSURE_REVIEW
TERMINATE
```

## 6. Closure checks

输出固定十三项：

| Check | 为 true 的精确定义 |
|---|---|
| `stopProofRevisionCurrent` | candidate、proof、task 和当前 canonical revision 完全一致 |
| `stopProofMatchesCanonical` | proof 中全部对象、Need、task/result/delta、contradiction 和 handoff index 与 supplied canonical projection 一致 |
| `mechanicalPreflightPassed` | preflight 绑定同一 candidate/revision，全部注册检查和 aggregate `passed` 均为 true |
| `topicScopePreserved` | scope fingerprint 一致，或每个 scope change 都有显式用户授权 |
| `noKnowledgeAnswerableCriticalNeed` | 不存在 pending critical Need whose answerability 为 `knowledge_base` 或 `unknown` |
| `allAnchorsClosed` | 每个 Anchor 均为 `saturated` 或 `rejected`；前者有 saturation reason，后者有 status reason |
| `allDirectionsTerminal` | 每个 Direction 均为 `testable`、`experiment_required` 或 `rejected`，且有非空 status reason |
| `lastTopicExpansionNoDelta` | 存在 completed Topic-owned `discover_anchor`，outcome 为 `no_new_anchor_no_critical_delta`，有 committed no-delta record 且无 SemanticDelta |
| `noUnconsumedOrUncommittedWork` | 无 pending/in-flight task、pending output retry、unconsumed result、uncommitted delta、unresolved validation failure 或尚未 retry/supersede/reconcile 的 failed task |
| `criticalContradictionsReviewed` | 每个 critical contradiction 均已有 committed review/disposition |
| `experimentHandoffsComplete` | 每个 `experiment_required` Direction 恰有一个完整且 `executionAuthorized = false` 的 handoff，其他状态没有 handoff |
| `runtimeEligibleForCompletion` | lifecycle 为 `closure_preflight`，budget 未耗尽，无 pause/block/failure，fresh Turn 与 canonical-only assertions 成立 |
| `finalOutputTraceable` | 七个 coverage field 全部存在且只引用 supplied canonical refs |

所有项必须从 supplied projection 判定。

七个 coverage field 固定为 `topic_scope`、`anchor_summaries`、`direction_statuses`、`evidence_provenance`、`contradictions_and_limits`、`experiment_handoffs` 和 `unresolved_questions`。

## 7. Accept 与 Reject

### 7.1 Accept

只有十三项全部为 true 才能 accept：

- `blockingFindings = []`；
- `reopenScopes = []`；
- `allowsFinalization = true`；
- `verifiedClosureBasis` 对十三项 check 各提供至少一个可解析依据；
- finalization requirements 固定为：
  - `canonical_revision_unchanged`
  - `full_validator_passed`
  - `final_output_rendered`
  - `final_output_coverage_validated`
  - `atomic_completed_commit`

### 7.2 Reject

任一项 false：

- `allowsFinalization = false`；
- 至少一个 blocking finding；
- `verifiedClosureBasis` 只能覆盖值为 true 的 check；
- 每个 finding 指向具体 canonical object/scope；
- 不生成 SearchNeed；
- 不修改 StopProof。

Finding 类型和后续处理映射：

| Type | 处理类别 | 后续处理 |
|---|---|---|
| `knowledge_gap` | `REOPEN_FRONTIER` | Controller 触发新的 Workflow Turn |
| `state_inconsistency` | `REPAIR_STATE` | 唯一机械修正由 Controller reconcile；否则触发 Workflow Turn |
| `incomplete_handoff` | `COMPLETE_HANDOFF` | Workflow Turn 重开 Direction 边界 |
| `runtime_pause` | `RESUME_RUNTIME` | Controller 保存非完成状态 |

`REPAIR_STATE` 只是 ClosureReview schema 中的 machine-readable recoveryAction，不是 Agent、Stage 或脚本可调度角色。Reviewer 只分类；Controller 决定 deterministic reconcile 还是构造 Workflow trigger。

Shared `ClosureFindingCode` registry 的映射固定为：

- revision/proof/preflight、pending/in-flight/output-retry、unconsumed/uncommitted、validation/failed-task 和 final-output consistency codes → `state_inconsistency` / `REPAIR_STATE`；
- Topic scope、open Need、Anchor/Direction closure、last expansion 和 unreviewed contradiction codes → `knowledge_gap` / `REOPEN_FRONTIER`；
- handoff missing/invalid codes → `incomplete_handoff` / `COMPLETE_HANDOFF`；
- budget exhausted 或 run paused/failed codes → `runtime_pause` / `RESUME_RUNTIME`。

Validator 使用以下精确 registry；不允许仅按字符串前缀猜测：

| Check | Finding code | Type / recoveryAction |
|---|---|---|
| `stopProofRevisionCurrent` | `stale_stop_proof_revision` | `state_inconsistency` / `REPAIR_STATE` |
| `stopProofMatchesCanonical` | `stop_proof_canonical_mismatch` | `state_inconsistency` / `REPAIR_STATE` |
| `mechanicalPreflightPassed` | `mechanical_preflight_failed` | `state_inconsistency` / `REPAIR_STATE` |
| `topicScopePreserved` | `topic_scope_silently_narrowed` | `knowledge_gap` / `REOPEN_FRONTIER` |
| `noKnowledgeAnswerableCriticalNeed` | `knowledge_answerable_open_need` | `knowledge_gap` / `REOPEN_FRONTIER` |
| `allAnchorsClosed` | `anchor_not_closed`, `anchor_missing_saturation_reason`, `anchor_missing_status_reason` | `knowledge_gap` / `REOPEN_FRONTIER` |
| `allDirectionsTerminal` | `direction_nonterminal`, `direction_missing_terminal_reason` | `knowledge_gap` / `REOPEN_FRONTIER` |
| `lastTopicExpansionNoDelta` | `last_topic_expansion_missing`, `last_topic_expansion_not_quiet` | `knowledge_gap` / `REOPEN_FRONTIER` |
| `noUnconsumedOrUncommittedWork` | `pending_task`, `in_flight_task`, `pending_output_retry`, `unconsumed_result`, `uncommitted_delta`, `unresolved_validation_failure`, `failed_task` | `state_inconsistency` / `REPAIR_STATE` |
| `criticalContradictionsReviewed` | `unreviewed_critical_contradiction` | `knowledge_gap` / `REOPEN_FRONTIER` |
| `experimentHandoffsComplete` | `experiment_handoff_missing`, `experiment_handoff_invalid` | `incomplete_handoff` / `COMPLETE_HANDOFF` |
| `runtimeEligibleForCompletion` | `runtime_budget_exhausted`, `runtime_failed_or_paused` | `runtime_pause` / `RESUME_RUNTIME` |
| `finalOutputTraceable` | `final_output_missing_field`, `final_output_untraceable` | `state_inconsistency` / `REPAIR_STATE` |

旧 StopCandidate 在 reject 后作废；恢复后必须形成新候选和新 Closure Turn。

## 8. 输出和 Controller Gate

只输出：

```text
PayloadTurnEnvelope<ClosureReview>
```

Payload 必须包含 `decision`、`verifiedClosureBasis`、完整十三项 `closureChecks`、`blockingFindings`、去重后的 `reopenScopes`、`allowsFinalization`、`finalizationRequirements` 和 `rationale`。

`ClosureReview.status` 固定为 `complete`，只表示 Reviewer Turn 已终止；即使 decision 为 `accept`，也不表示 run 已进入 `completed`。

发出响应前，Reviewer 必须对 expected schema、TurnIdentity、StateBinding、`inputHash`、candidate/proof revision、十三项检查、accept/reject 一致性和唯一顶层 JSON 做一次内部自检。

Controller Gate 检查：

- Turn/contract/state binding；
- candidate/proof/revision；
- fresh-turn/canonical-only assertions；
- thirteen checks 与 supplied facts 一致；
- accept iff all checks true；
- verified basis 只覆盖 true checks，accept 时覆盖全部十三项；
- reject findings 覆盖所有 false checks；对 Need、Anchor、Direction、task/result/delta、validation failure、contradiction、handoff 和 coverage 等可枚举 index，还必须逐个覆盖每个 supplied blocking object；
- finding 的 `(check, code, objectRefs)` 不得遗漏、伪造或重复；机械可重算部分必须与 Controller issue projection 集合相等；
- `reopenScopes` 恰为 findings 中 scope 的去重集合；
- object refs 可解析；
- 后续处理映射合法；
- no tools/events；
- Reviewer 没有写 completion action。

Gate 通过后：

- reject：提交 review 后触发 `CLOSURE_REJECTED`；
- accept：进入 finalization，不能直接写 completed。

## 9. 权限配置

```json
{
  "role": "closure_reviewer",
  "lifecycle": "fresh_turn",
  "reasoningEffort": "high",
  "tools": [],
  "filesystem": "none",
  "network": false,
  "delegation": false,
  "goals": false,
  "stateWrite": false,
  "experimentExecution": false,
  "allowedInputMessageTypes": [
    "CLOSURE_REVIEW_TASK"
  ],
  "allowedOutputMessageTypes": [
    "CLOSURE_REVIEW"
  ]
}
```

不得复用 Workflow Turn、Direction Reviewer 或失败 Closure Turn 的 provider history。

## 10. 实现文件

```text
.codex/skills/learning-semantic-loop-closure-reviewer/
├── SKILL.md
├── references/
│   ├── role_profile.json
│   ├── scheduler_contract.md
│   └── schema_manifest.json
└── scripts/
    └── validate_closure_review.py

scripts/simple_semantic_loop/
├── prompt_templates/closure_reviewer.ts
├── role_profiles/closure_reviewer.json
└── tests/fixtures/closure_reviewer/
```

Skill 内 schema 必须删除并引用 shared manifest，或由 shared schema 自动生成并接受 hash 一致性测试。Python helper 与 TypeScript runtime validator 必须共享十三项检查、finding code/action matrix 和测试向量。

## 11. 实现工作包

### CR-1：契约迁移

- 接入 TurnIdentity、StateBinding、StageContract hash；
- 对齐 shared StopCandidate/Proof/ClosureReview；
- 接入固定 closure rubric ID/version/hash；
- 将旧误导性 check/code `allAnchorsSaturated`/`anchor_not_saturated` 单点迁移为 `allAnchorsClosed`/`anchor_not_closed`，把 StopProof claims 收窄为固定十项 projection，并同步 schema、validator 和 fixture；
- 将旧 `freshSession`/`requiresFreshSession`/session health 字段迁移为 `freshTurn`、provider-history isolation 和 canonical-only assertions；
- 将旧 pending protocol-repair 字段迁移为 pending output retry / validation failure，并保持 `REPAIR_STATE` 仅作为 finding action enum；
- 将 finding code `pending_protocol_repair` 拆为 `pending_output_retry` 和 `unresolved_validation_failure`；删除 `runtime_session_lost`、`runtime_protocol_failure`，分别由 `failed_task`、`unresolved_validation_failure` 或 `runtime_failed_or_paused` 表达；
- 移除旧持久 Controller 和 provider Session 语义。

验收：任何旧 checkpoint、provider Session 状态或 protocol-repair role/code 字段不进入 task/result；迁移后 schema/fixture 不双重接受旧字段。

### CR-2：Closure projection builder contract

- 定义 Reviewer 所需最小全局 projection；
- 定义 scope、Need、task/result/delta、output attempt、validation failure、contradiction、handoff 和 coverage indexes；
- 绑定 artifact hashes。

验收：Reviewer 无工具也能完成全部十三项检查。

### CR-3：Decision validator

- accept iff all checks true；
- reject finding coverage；
- reopen scope；
- 十三项检查的独立重算；
- finding code/type/recoveryAction 映射。

验收：缺 blocker、伪 accept 或 fabricated ref fail closed。

### CR-4：伪完成测试

覆盖：

- budget exhaustion；
- no runnable stage；
- output correction retry 或 validation failure pending；
- failed Turn；
- legacy Session/protocol-repair field；
- 非零 tool/evidence budget 或 task/StageContract budget 不一致；
- `freshTurn = false` 或 provider history included；
- stale candidate；
- unconsumed result；
- rejected Anchor with reason（合法闭合）与 rejected Anchor missing reason（阻塞）；
- nonterminal Direction；
- missing final expansion；
- incomplete handoff；
- unreviewed contradiction；
- 七类 coverage 缺失或引用 fabricated ref。

验收：所有情形 reject 或由 preflight 阻止 dispatch。

### CR-5：Accept/finalization 集成

- valid accept；
- canonical revision 在 accept 后变化；
- render failure；
- coverage failure；
- duplicate finalization。

验收：只有 unchanged revision 的一次原子 transaction 可提交 completed。

## 12. 完成标准

1. 每个候选由新的 zero-tool Turn 审阅。
2. Reviewer 只依据 canonical projection 和 StopProof。
3. 十三项 closure checks 均有 fixture。
4. budget、failure、empty plan 和 context loss 不会被接受为完成。
5. Reject 能准确区分 knowledge、state、handoff 和 runtime 问题。
6. Reviewer 不生成 SearchNeed、不修复对象、不调度 Agent。
7. Accept 仍必须经过 Controller finalization。
8. `REQ-23`–`REQ-25` 的闭包证据可由最终 Workflow test 验证。
9. 每次 attempt 使用固定 `high` effort，不能被 task 或 run config 覆盖。
10. Agent 在输出前完成契约自检，且不调用或依赖格式修复辅助 Agent。
11. `allAnchorsClosed` 对 saturated/rejected 两种合法闭合状态的语义有固定测试。
12. Skill/schema/Python/TypeScript validator 对十三项检查和 finding mapping 完全一致。
13. provider Session 健康、丢失或恢复不再是 closure 事实；只验证 fresh Turn 和权威外部状态。
