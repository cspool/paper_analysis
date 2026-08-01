# 正式快照、轮次授权、收敛探测与 Runtime Delta 合并设计

> 状态：2026-08-01 已实施并通过本地回归测试。  
> 前置设计：`05_minimal_controller_validation_and_semantic_handoff_design.md`、
> `06_outer_loop_memory_trajectory_and_atomic_direction_design.md`、
> `07_frozen_decision_snapshots_semantic_convergence_and_runtime_delta_dedup_design.md`。

## 1. 问题与边界

format v6 的最新多模态运行本身可校验，但在第 12 个 Decision 周期后反复出现
`PAUSED`。原因不是 Agent 格式错误，也不是语义重试：`maxRounds=8` 被当成整个
运行的固定累计上限；每次 resume 只越过当前暂停一次，下一完整周期又触发同一
条件。与此同时，机械需求闭合后的 Anchor 扩展没有获得冻结 research memory，
Worker 容易重复搜索；逐 token `output_delta` 虽已去掉双份 raw event，仍造成大量
JSONL 行和文件写入。

本次保持既定边界：

- Script 只校验会直接影响状态机的格式、核心字面量、引用和轮次授权；
- 是否存在新 Anchor、一次负面探测是否可信、Topic 是否可完成，仍由
  Worker、Reviewer、Decision 进行语义判断；
- Agent 输出协议不增加 saturation、coverage-complete 或 budget 字段；
- 控制台仍实时转发每个 Provider delta。

## 2. 修改前正式版本的冻结

修改代码和 Skill 前，完整冻结源运行及其当时实现：

```text
archive/learning_workflow/official_runs/
  multimodal_inference_latency_first_v6_20260801_round12/
    OFFICIAL_SNAPSHOT.md
    run_snapshot.tar.gz
    implementation_snapshot.tar.gz
```

快照保持真实生命周期 `PAUSED`：12 个已提交轮次、已准备的第 13 轮、36 个
Agent Turn、4 个已接受 Anchor、4 个已接受 Direction，以及 1 个待修订 Anchor。
“正式保存”表示字节级、可审计冻结，不等于伪造 `FINISHED`。

## 3. format v7：追加式轮次授权

### 3.1 状态

`state.json` 增加 Script 内部字段：

```json
{
  "pauseKind": "ROUND_BUDGET_EXHAUSTED | OPERATOR_REQUESTED | null",
  "roundBudget": {
    "authorizedThroughRound": 8,
    "lastAuthorizationRef": null
  }
}
```

`run.json.budgets.maxRounds` 重新解释为初始授权窗口和默认追加窗口，不再作为
永久累计上限。初始值仍不可变。

### 3.2 恢复

预算暂停后：

```bash
node scripts/simple_semantic_loop.ts resume \
  --additional-rounds N \
  --work-dir <run-dir>
```

Script 从当前已授权边界追加 N 轮，并写一次不可变记录：

```text
authorizations/rounds/<round-authorization-id>.json
```

记录包含来源 State revision、首个新增轮次、增加数量和新的授权终点。省略 N 时，
默认追加初始化 `maxRounds` 轮。因此一次人工授权可执行多个完整 Loop 周期；中间
不再每轮暂停。操作员主动 pause 不自动扩权，除非显式提供 N。

### 3.3 可观察状态

`status` 由 Script 机械投影：

- `pauseKind`；
- `nextRole` 与可机械确定的 `nextAction`；
- `finishEligibility.mechanicallyEligible` 和阻塞项；
- 初始授权、当前授权终点、剩余授权轮数、默认追加数和最后授权 Ref。

这些字段只供 Agent 外的人类和运维观察，不要求任何 Agent 回显。

## 4. 有界收敛探测

当现有 Anchor/Direction 已机械闭合且 Script 再绑定 `CREATE_ANCHOR` 时，该
Worker Task 被明确描述为 `CONVERGENCE_PROBE`，但 intent 不进入 Agent 输出协议。
Script 为该 Task 冻结：

```text
tasks/<binding-id>/research_memory_snapshot.json
```

并仅通过 `T01.inputs.researchMemory` 传给 Worker。该快照提供 accepted、
needsRevision、rejectedLessons 和动态 6L coverage，作用是去重与选择代表性角度，
不是来源证据，也不是 Script 的饱和判定。

语义闭环保持：

```text
Worker 有界检索
  ├─ 新颖且有证据 → READY_FOR_REVIEW → 正常扩展
  └─ 重复/无支撑 → BLOCKED_NO_RESULT
                         ↓
Reviewer 判断探测是否可信
  ├─ REJECT → 可作为负面收敛证据
  └─ REVISE → 搜索过窄、工具失败或遗漏明显候选
                         ↓
Decision 结合 Goal、全部结论和轨迹决定继续或完成
```

最新一次成功 Anchor 之后需要一轮新的可信探测。一轮可信负面探测通常足够；除非
Reviewer 指出实质 coverage 缺陷、工具失败或搜索过窄，Decision 不应要求无依据
的连续空探测。

## 5. Runtime Delta 合并

Provider 的每个 `item/agentMessage/delta`：

1. 立即发出 live console event；
2. 在内存中按 message item 聚合；
3. 累计达到 2,048 字符或 100 ms 时写一个规范化 `output_delta`；
4. message complete、Turn complete、timeout、close 或 provider failure 前强制
   flush；
5. Controller 按合并后的同序文本更新 `partial_output.txt`。

因此实时输出粒度不变，恢复得到的字符串不变，但 `runtime.jsonl` 行数和
`turn.json` 更新次数显著下降。非 delta provider event 仍完整保存。

## 6. 兼容性与验证

- 当前可写格式为 v7；v5、v6 只读，不原地迁移；
- v6 仍校验 Context-local 冻结 Decision sources；
- v7 Validator 机械校验 pause kind、授权边界和不可变授权 Ref；
- T01 只允许 `CREATE_ANCHOR` 使用可选 `researchMemory`；
- Worker、Reviewer、Decision 的核心输出协议不变；
- 回归测试覆盖一次授权连续两轮、可信负面探测、v5/v6 只读、实时 64 个 delta
  对应较少持久事件且重建文本完全一致。

正式 v6 快照不回写。继续运行应新建 v7 work directory；这是有意的审计边界，
避免用新语义静默改写旧运行历史。
