# Direction Reviewer Turn Agent 实现计划

## 1. 计划定位

本文实现针对一个 committed Direction revision 的独立 Semantic Evaluator。

- 角色：`direction_reviewer`
- Skill：`learning-semantic-loop-direction-reviewer`
- 生命周期：一个 `DIRECTION_REVIEW_TASK` 对应一个 fresh Turn
- Reasoning effort：固定 `high`
- 工具：零工具
- 输出：一个 `REVIEW_DELTA`
- 状态写入、证据检索、调度和实验：禁止

依赖：

- [共享契约规范](shared_contracts.md)
- [Workflow Turn Agent 实现计划](workflow_turn_agent_implementation_plan.md)
- [Scheduler Script 实现计划](scheduler_script_implementation_plan.md)

现有 `.codex/skills/learning-semantic-loop-direction-reviewer/` 只作为审阅顺序、decision matrix 和 validator fixture 的迁移输入，不能直接注册到新 Controller：它尚未接入 TurnIdentity、StateBinding、`inputHash` 和 shared schema manifest。

## 2. 在 Workflow 中的位置

```text
Workflow Turn 提出 DirectionReviewRequest
  + DIRECTION_REVIEW Stage/Gate 草案
→ Controller 冻结 Direction revision、输入 projection 和 rubric
→ Controller 启动 fresh Direction Reviewer Turn
→ Reviewer 输出 REVIEW_DELTA 并退出
→ Controller 执行 frozen Gate 并提交 result
→ Controller 触发 COMMITTED_RESULT_REQUIRES_INTEGRATION
→ 新 Workflow Turn 决定 SearchNeed、SemanticDelta 或终态
```

Reviewer 的决定不是 canonical commit。它不把结果直接交给 Workflow Agent；Controller 先验证并提交。

## 3. 职责

Reviewer 按固定顺序检查：

1. Direction 是否仍在 Topic 和 Anchor 范围内；
2. baseline、comparison scope 和 controlled variables 是否公平；
3. ModificationAtom 是否表达实际 from/to change；
4. 最小修改集合是否与 enabler 区分；
5. 性能因果链是否完整、可证伪；
6. 跨层接口和资源冲突是否明确；
7. 实现入口是否有边界；
8. 指标、消融、falsifier 和 degradation condition 是否足够；
9. 支持 Evidence、反证和 inference 是否可追溯；
10. 下一关键缺口能否由知识库回答；
11. 选择一个审阅决定。

它不判断 Topic 是否完成，也不比较 Direction 的研究价值。

## 4. 输入契约

输入为：

```ts
PayloadTurnEnvelope<DirectionReviewTask>
```

Envelope 必须使用 `messageType = "DIRECTION_REVIEW_TASK"`，并包含当前 TurnIdentity、StateBinding、`inputHash` 和 `stageContractHash`。Controller 构造的 dispatch packet 还必须固定 Skill name/version/hash、schema manifest hash、frozen StageContract/GateDefinition、zero-tool permission envelope、expected output schema 和本 Turn 终止条件。其 task budget 必须与 frozen StageContract canonical-equal，并固定 `maxToolCalls = 0`、`evidenceRead = null`。

Task 必须包含：

- TopicFrame 相关投影；
- 一个完整 Anchor revision；
- 一个完整 Direction revision；
- 同一 Topic/Anchor 下其他 committed sibling Direction 的最小 dedup projection（ID/revision、baseline/comparison scope、primary ModificationAtom 和 causal target）；
- Direction 实际引用的 committed Evidence findings；
- contradicting Evidence；
- unresolved SearchNeeds；
- known counterexamples/degradation conditions；
- prior review 的 compact index；
- frozen review purpose（仅 `initial`、`after_evidence`、`terminal_check`、`adversarial_recheck`）和完整 registered `RubricBinding`（ID/version/SHA-256）；
- input artifact hashes；
- schema manifest hash。

以下条件必须由 Controller 在 dispatch 前 fail closed：

- Direction/Anchor/Topic ID 不一致；
- revision stale；
- 引用未提交 Evidence；
- task 要求搜索新证据；
- rubric ID 未注册、rubric version/hash 不匹配，或 rubric 在 Turn 启动后改变；
- 输入包含未批准 raw log 作为事实。

Reviewer 在 Turn 开始时再次断言这些绑定。若认为输入契约错误，不得选择四种业务决定、提交 ReviewDelta 或自行写 attempt 状态；它应拒绝产生业务结果。Controller 重新运行权威 task validator：只有 validator 也失败时才记录 `input_contract_invalid`，且同一无效 task 不做格式重试；若 task 仍有效，Reviewer 的拒绝作为无效产出进入普通同角色 fresh retry。协议不为此增加错误消息族。

Rubric 必须来自 Controller 的只读 registry，并与本 Skill 的固定审阅顺序兼容。Workflow proposal 只能选择已注册 rubric ID，不能把任意自然语言 rubric 注入 Reviewer。

Sibling dedup projection 由 Controller 从 canonical Direction 构造并绑定 hash，不由 Agent 自报。固定 rubric 只在以下三项均成立时允许语义 `duplicate`：baseline/comparison scope 等价；primary from/to change 没有实质差异；causal target 与性能假设等价。仅措辞、Evidence 数量或 enabler 细节不同不构成新 Direction；layer、修改对象、适用条件、受控变量、目标 metric 或因果方向有实质差异时不得判为 duplicate。

## 5. 审阅算法

```text
VALIDATE task identity and frozen revision
CHECK Topic/Anchor scope
CHECK baseline fairness
CHECK minimum change set
TRACE causal chain and weakest link
CHECK implementation boundary
CHECK measurement and falsifiability
CHECK contradiction and degradation
CLASSIFY strongest unresolved issue
SELECT exactly one decision
SELF-CHECK expected schema, bindings, decision fields and one-JSON rule
EMIT REVIEW_DELTA
TERMINATE
```

最强问题选择顺序：

1. 越界或对象内部无效；
2. 不公平 baseline 或致命反证；
3. 因果链断裂；
4. 实现接口缺失；
5. 测量或 falsifier 缺失；
6. 未解决 counterexample；
7. 确认性细节。

`continue_search` 只能提出一个主要 next question。

## 6. 四种决定

每个 ReviewDelta 都必须包含以下十一项 `readinessChecks`，不得省略或增加字段：

1. `inTopicAndAnchorScope`
2. `baselineFair`
3. `minimumChangeSetExplicit`
4. `causalChainFalsifiable`
5. `implementationPathBounded`
6. `measurementPlanComplete`
7. `falsifiersPresent`
8. `criticalCounterexampleResolved`
9. `evidenceTraceable`
10. `knowledgeAnswerableCriticalGapRemaining`
11. `newExperimentRequired`

同时必须输出 `weakestCausalLink`、`baselineProblem`、`implementationProblem`、`measurementProblem`、`strongestCounterexample`、`counterexampleResolution`、`duplicateOfDirectionRef`、`duplicateComparison`、`rejectionCategory`、`nextQuestion`、`nextQuestionAnswerableFromKnowledgeBase` 和 `experimentHandoff`。所有 Evidence ID 和对象引用只能来自 task allowlist。

跨 decision 的字段耦合固定为：

- `baselineFair = false` iff `baselineProblem` 非空；
- `implementationPathBounded = false` iff `implementationProblem` 非空；
- `measurementPlanComplete = false` 或 `falsifiersPresent = false` iff `measurementProblem` 非空；
- `causalChainFalsifiable = false` 时 `weakestCausalLink` 必须非空；为 true 时允许记录非阻塞最弱环节或 null；
- `criticalCounterexampleResolved = false` 时 `strongestCounterexample` 必须非空且 `counterexampleResolution = null`；
- supplied counterexample 被标记 resolved 时，counterexample 与 resolution 必须成对出现，并在 `evidenceRefsUsed` 中引用 supplied counterevidence；没有 supplied counterexample 时 `criticalCounterexampleResolved = true` 且二者都为 null；
- `evidenceTraceable = true` 时 `evidenceRefsUsed` 非空且均来自 task；`testable`/`experiment_required` 还要求 `supportedParts` 非空。

### 6.1 `continue_search`

使用条件：

- Direction 仍值得保留；
- 存在一个 critical gap；
- 该 gap 可由本地知识库回答。

输出要求：

- exactly one `nextQuestion`；
- `nextQuestionAnswerableFromKnowledgeBase = true`；
- `knowledgeAnswerableCriticalGapRemaining = true`；
- `newExperimentRequired = false`；
- `inTopicAndAnchorScope = true`；
- `duplicateOfDirectionRef = null`、`duplicateComparison = null`、`rejectionCategory = null` 且无 ExperimentHandoff；
- 至少一个 core readiness check 为 false，或 weakest/problem/counterexample 字段至少一个非空，以证明存在具体 critical gap。

### 6.2 `testable`

必须同时满足：

- scope 合法；
- baseline 公平；
- minimum change set 明确；
- causal chain 可证伪；
- implementation path 有边界；
- measurement/ablation 完整；
- falsifiers 存在；
- critical counterexample 已解决；
- Evidence 可追溯；
- 无 knowledge-answerable critical gap。

九个 core readiness checks 必须全部为 true，两个 gap checks 均为 false，`nextQuestionAnswerableFromKnowledgeBase = false`；`nextQuestion`、`duplicateOfDirectionRef`、`duplicateComparison`、`rejectionCategory`、ExperimentHandoff、`baselineProblem`、`implementationProblem` 和 `measurementProblem` 必须为 null。`weakestCausalLink` 可以记录非阻塞的最弱环节；存在 supplied counterexample 时，`strongestCounterexample` 和 `counterexampleResolution` 必须成对出现，resolution 必须引用相应 counterevidence；不存在时二者必须为 null。

不得称为 experimentally validated。

### 6.3 `experiment_required`

使用条件：

- Direction 语义上足够完整；
- 知识库已无可回答的 critical gap；
- 剩余问题必须依赖新 trace、prototype、benchmark、equivalence test、code audit 或 hardware measurement。

`inTopicAndAnchorScope`、`baselineFair`、`minimumChangeSetExplicit`、`causalChainFalsifiable`、`falsifiersPresent` 和 `evidenceTraceable` 必须为 true；`knowledgeAnswerableCriticalGapRemaining = false`，`newExperimentRequired = true`，`nextQuestionAnswerableFromKnowledgeBase = false`。`nextQuestion`、`duplicateOfDirectionRef`、`duplicateComparison` 和 `rejectionCategory` 必须为 null。实现、测量或 counterexample readiness 可以因所需新 artifact 而保持 false，但 rationale 和 handoff 必须精确对应该缺口。

必须包含完整 ExperimentHandoff，且：

```text
executionAuthorized = false
```

### 6.4 `rejected`

适用于：

- duplicate；
- 越出 Topic/Anchor；
- fatal contradiction；
- 无法公平比较；
- 没有性能机制；
- Evidence 无效。

必须有以下一个 rejection category：`duplicate`、`out_of_scope`、`causal_contradiction`、`unfair_comparison`、`no_performance_mechanism`、`invalid_evidence` 或 `other`。`nextQuestion = null`、`nextQuestionAnswerableFromKnowledgeBase = false`、两个 gap checks 均为 false，并且 ExperimentHandoff 为 null。

Category 与结构化事实的映射固定为：

| Category | 必需条件 |
|---|---|
| `duplicate` | `duplicateOfDirectionRef` 非空并引用 supplied sibling；`duplicateComparison` 非空，三个 equivalence flag 全为 true 且 `materialDifference = null` |
| `out_of_scope` | 两个 duplicate 字段均为 null，且 `inTopicAndAnchorScope = false` |
| `causal_contradiction` | 两个 duplicate 字段均为 null，存在 supplied contradicting Evidence，且 `causalChainFalsifiable = false` 或 `criticalCounterexampleResolved = false` |
| `unfair_comparison` | 两个 duplicate 字段均为 null，且 `baselineFair = false` |
| `no_performance_mechanism` | 两个 duplicate 字段均为 null，且 `causalChainFalsifiable = false` |
| `invalid_evidence` | 两个 duplicate 字段均为 null，且 `evidenceTraceable = false` |
| `other` | 两个 duplicate 字段均为 null，至少一个 core readiness check 为 false，且 rationale 说明其他类别为何不适用 |

除 `duplicate` 外，不能在九个 core readiness checks 全为 true 时 reject。两个 duplicate 字段在非 `duplicate` 决定和 category 中必须同时为 null。

不得通过请求实验来“救活”已拒绝 Direction。

## 7. 输出和 Gate

只输出：

```text
PayloadTurnEnvelope<ReviewDelta>
```

发出响应前，Reviewer 必须对 expected schema、TurnIdentity、StateBinding、`inputHash`、Direction revision、role/message、十一项 readiness checks、decision/field 一致性和唯一顶层 JSON 做一次内部自检。

Controller Gate 至少检查：

- Turn/contract/state binding；
- Direction ID/revision；
- Evidence refs 属于输入 allowlist；
- `duplicateOfDirectionRef` 只引用 supplied sibling Direction；duplicate comparison 的三项等价判断和 material-difference 字段满足固定 rubric/schema；
- Reviewer 声称为 true 的 readiness check 不与 task 中可机械推出的事实冲突；
- problem/counterexample/evidence 字段与 readiness checks 满足跨 decision 耦合；
- decision 与 readiness checks 一致；
- `continue_search` 恰有一个问题；
- `testable` 所有硬条件成立；
- `experiment_required` handoff 完整且不可执行；
- `rejected` 有合法 category；
- rejection category、duplicate binding 与 readiness checks 满足固定矩阵；
- counterexample marked resolved 时有 resolution 且引用其 counterevidence；
- tool events 为空；
- 没有 Topic closure 或调度 action。

Gate 通过后只提交 ReviewDelta。Direction 的实际状态变化由下一次 Workflow Turn 提出、Controller 提交。

## 8. 权限配置

```json
{
  "role": "direction_reviewer",
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
    "DIRECTION_REVIEW_TASK"
  ],
  "allowedOutputMessageTypes": [
    "REVIEW_DELTA"
  ]
}
```

不得复用 Workflow Turn、Evidence Reader 或先前 Reviewer 的 provider history。

## 9. 实现文件

```text
.codex/skills/learning-semantic-loop-direction-reviewer/
├── SKILL.md
├── references/
│   ├── role_profile.json
│   └── schema_manifest.json
└── scripts/
    └── validate_review_delta.py

scripts/simple_semantic_loop/
├── prompt_templates/direction_reviewer.ts
├── role_profiles/direction_reviewer.json
└── tests/fixtures/direction_reviewer/
```

Skill 内 schema 必须删除并引用 shared manifest，或由 shared schema 自动生成并接受 hash 一致性测试。Python helper 与 TypeScript runtime validator 共享同一 decision matrix 和测试向量，不能分别维护两套 readiness 规则。

## 10. 实现工作包

### DR-1：契约迁移

- 接入 TurnIdentity、StateBinding、StageContract hash；
- 对齐 shared DirectionReviewTask/ReviewDelta，并把十一项 readiness checks、counterexampleResolution、duplicate binding/comparison 和 rejectionCategory 纳入唯一 schema；
- 接入 registered rubric ID/version/hash；
- 移除旧 Controller result-forwarding 语义。

验收：Reviewer result 只能由 Controller commit。

### DR-2：固定审阅顺序

- 实现 scope、baseline、change、causal、implementation、measurement、counterexample 顺序；
- 只选最强 unresolved issue。

验收：fixture 可稳定定位同一 blocking dimension。

### DR-3：Decision gates

- 四种决定的 deterministic validator；
- 十一项 readiness checks 与四种决定的完整字段矩阵；
- readiness/problem/counterexample/evidence 的双向字段耦合；
- rejection category enum；
- sibling Direction dedup projection、`duplicateOfDirectionRef`/`duplicateComparison` 和 category/check matrix；
- counterexample resolution/evidence binding；
- one-question constraint。

验收：decision 与字段矛盾时 fail closed。

### DR-4：实验边界

- handoff schema；
- executable instruction 检测；
- `executionAuthorized = false` literal check。

验收：handoff 永远不会进入实验 task queue。

### DR-5：角色级测试

覆盖：

- stale revision；
- unfair baseline；
- causal gap；
- unresolved interface；
- measurement gap；
- contradiction；
- counterexample resolution 缺失或未引用 counterevidence；
- Direction 越界；
- 全部 review purpose；
- 未注册或 hash 不匹配的 rubric；
- 非零 tool/evidence budget 或 task/StageContract budget 不一致；
- valid testable；
- valid experiment_required；
- valid rejected 与全部 rejection category；
- fabricated/missing sibling ref、duplicate comparison 不完整和 category/check mismatch；
- invalid handoff；
- input-contract failure 不产生业务决定；
- terminal output self-check，以及 structure/binding/pre-Gate semantic
  failure 的同角色 `correctionFeedback` fresh replacement；
- tool/delegation attempt。

## 11. 完成标准

1. 每个 Turn 只审阅一个 Direction revision。
2. 输入只包含 committed、frozen、可追溯事实。
3. Reviewer 不搜索、不修订对象、不调度、不维护状态。
4. 四种决定均有可机械验证的字段关系。
5. `continue_search` 最多一个主要问题。
6. `testable` 与实验验证严格分离。
7. ExperimentHandoff 不可执行。
8. ReviewDelta 通过 Gate 后触发新的 Workflow integration Turn。
9. 角色级 fixture 全部通过。
10. 每次 attempt 使用固定 `high` effort，不能被 task 或 run config 覆盖。
11. Agent 在输出前完成契约自检，且不调用或依赖格式修复辅助 Agent。
12. shared schema、Skill schema、Python/TypeScript validator 对十一项 readiness checks、decision/rejection matrix 和 duplicate binding/comparison 完全一致。
13. 输入契约错误与四种业务审阅结论严格分离，前者不能提交 ReviewDelta。
