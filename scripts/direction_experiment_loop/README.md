# Direction Experiment Loop 使用说明

该脚本把 Learning Flow 中一个已审阅 Direction 转成可迭代的消融实验工作流。当前格式
为 v7，对应
[`07_atomic_contract_early_stop_checkpoint_and_runtime_control_design.md`](../../draft/experiment_workflow_agent_loop_plans/07_atomic_contract_early_stop_checkpoint_and_runtime_control_design.md)。

```text
START → fresh Experiment Decision
           ├─ RUN_LAB → 冻结一个原子 contract-N → persistent Direction Lab Goal
           │                                           ↓ result.md
           │                                  fresh Evidence Judge
           │                                           ↓
           ├─────────────────────────────────── Experiment Decision
           ├─ RUN_JUDGE ─────────────────────→ Evidence Judge → Decision
           ├─ COMPLETE_SUPPORT / COMPLETE_REJECT / RETURN_TO_LEARNING → FINISHED
           └─ BLOCKED → PAUSED
```

- `Experiment Decision`：`max` effort 的 fresh Turn。理解冻结 Direction、完整轨迹、
  Lab/Judge 结论和运行包络，每轮只冻结一个会改变下一决策的问题。
- `Direction Lab Goal`：`high` effort、无 token budget 的持久 Goal。只执行一个原子
  合同，在每个昂贵动作前检查合同 stop condition，并原子维护 checkpoint/result。
- `Evidence Judge`：`high` effort 的 fresh Turn。独立审查早停边界和本轮证据，只输出
  有效正、有效负、证据不足或无效及实际 scope。

Script 只维护状态、合同和 Cycle 绑定、文件握手、invocation、事件及转换。它不判断
科研 stop condition、baseline 强弱、证据语义或最终价值。

## 初始化

输入必须是 Learning run `objects/index.json` 可唯一定位的 Direction `WORK_RESULT`：

```bash
cd /data3/paper_analysis

node scripts/direction_experiment_loop.ts init \
  --direction-result /data3/paper_analysis/learning_outputs_codex/<source_run>/results/<direction_turn>.json \
  --max-cycles 5 \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/<direction_run>
```

若无法自动定位来源 run，增加：

```bash
--source-work-dir /data3/paper_analysis/learning_outputs_codex/<source_run>
```

初始化会冻结研究输入、`ATOMIC_DECISION_CONTRACT_V7` 政策以及三个 Skill/Ref 的 run-local
副本。后续活动 Skill 更新不会改变该 run 的执行方法。主要目录为：

```text
inputs/skills/           本 run 冻结的三个角色 Skill/Ref
decisions/               fresh Decision 原始输出和接受结果
contracts/contract-N/    不可变原子合同、revision 和 hash
lab_goals/               Goal 与每次 invocation 独立记录
judge_requests/          独立审阅请求
judges/                  fresh Judge 记录
snapshots/               状态、轨迹和 Lab runtime envelope
workspace/cycles/N/      每轮隔离源码、shards、raw、analysis、checkpoint、result
history.jsonl            Decision、Experiment、Judgment 有序索引
final/                   终态报告和 Learning handoff
```

项目级 `experiment_cache/{models,data,environments}` 可共享只读大对象。Lab 必须按
revision/content hash 引用，不能在 cache 内做本轮修改。

## 原子合同与 Stop Gate

每次 `RUN_LAB` 合同只回答一个问题，并包含：

- `objective`、`comparison`、`conditions`；
- 非空且按优先级排列的 `stopConditions`；
- 不超过当前单次 Lab watchdog 包络的 `estimatedMinutes`；
- `allowedWeakening`、`forbiddenWeakening`；
- 按不同退出路径定义的 `completionEvidence`。

calibration、independent confirmation 和 performance 通常拆成不同 Cycle：

```text
calibration → Judge → Decision
confirmation → Judge → Decision
performance → Judge → Decision
```

命中 stop condition 后，Lab 保存最窄证据并结束，不继续形式上的下游实验。Judge 审查
这个早停是否有效，Decision 再决定全局完成或下一合同。

## 启动

实验通常需要写代码、安装环境和运行工具：

```bash
node scripts/direction_experiment_loop.ts run --yolo \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/<direction_run>
```

不加 `--yolo` 时 provider 使用 `read-only` sandbox；`approvalPolicy` 始终为 `never`。
默认 timeout：Decision/Judge idle 5 分钟、hard 15 分钟；Lab idle 15 分钟、单次
invocation hard 6 小时。Lab hard timeout 是安全 watchdog，不是整个科研 Cycle 的预算。

初始化时可覆盖：

```bash
--decision-idle-timeout-ms 300000 \
--decision-hard-timeout-ms 900000 \
--judge-idle-timeout-ms 300000 \
--judge-hard-timeout-ms 900000 \
--lab-idle-timeout-ms 900000 \
--lab-hard-timeout-ms 21600000 \
--interrupt-grace-ms 15000
```

Decision 可用的最大 `estimatedMinutes` 等于 Lab hard timeout 减去 Script 预留的结果打包
时间；超过包络的合同会在同一冻结快照上要求 Decision 修正。

## 结果、检查点和恢复

每个 Cycle 的机械握手为：

```text
result.md 存在且非空、Cycle binding 未改变
  → 无论 provider complete / paused / blocked / timeout，都索引并进入 Judge

只有绑定正确的 checkpoint.json
  → PAUSED，保留同 provider thread；resume 创建新的 invocation 记录

两者都没有
  → 异常 PAUSED，不伪造 fallback 结果
```

普通恢复：

```bash
node scripts/direction_experiment_loop.ts resume --yolo \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/<direction_run>
```

Lab cycle 授权耗尽时：

```bash
node scripts/direction_experiment_loop.ts resume --yolo \
  --additional-cycles 2 \
  --work-dir /data3/paper_analysis/experiment_outputs_codex/<direction_run>
```

每次 resume 复用原 provider thread，但产生新的 invocation、deadline、prompt、compact
runtime 和 provider raw log。恢复只补 checkpoint 中缺失且 hash 未通过的 shard。

## 实时暂停

`pause` 不再等待长时间持有的 run lock，而是写入锁外控制请求：

```bash
node scripts/direction_experiment_loop.ts pause --work-dir <DIR>
```

活动 Controller 会中断当前 Lab Goal，然后优先接管已经提交的 `result.md`，否则索引
`checkpoint.json` 并暂停。SIGINT/SIGTERM 使用同一条优雅暂停路径。

## 审计与终态

```bash
node scripts/direction_experiment_loop.ts status --work-dir <DIR>
node scripts/direction_experiment_loop.ts events --work-dir <DIR>
node scripts/direction_experiment_loop.ts validate --work-dir <DIR>
node scripts/direction_experiment_loop.ts render --work-dir <DIR>
node scripts/direction_experiment_loop.ts pause --work-dir <DIR>
node scripts/direction_experiment_loop.ts cancel --work-dir <DIR>
```

主 `events.jsonl` 和 invocation `runtime.jsonl` 只保存小型规范事件；完整 provider raw
事件独立写入 `provider_raw.jsonl`。终态产生 `final/report.md`、`final/handoff.json` 和
`final/outcome.json`。

旧 format v2-v6 只保留只读审计；不要用 v7 Skill 恢复旧 persistent Goal。应从来源
Direction 初始化新的 v7 run，并把旧合同、结果和 Judgment 作为历史证据导入。

## 测试

```bash
node --test scripts/direction_experiment_loop/tests/*.test.ts
node --test scripts/simple_semantic_loop/tests/*.test.ts
```
