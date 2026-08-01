# Simple Semantic Loop 实现与验证记录

## 1. 实现范围

五份组件计划已经落地为一个确定性 Controller 和四类 fresh Turn：

| 计划 | 主要实现 |
|---|---|
| Workflow Turn Agent | `.codex/skills/learning-semantic-loop-workflow-turn/`、`scripts/simple_semantic_loop/prompt_templates/workflow_turn.ts`、`validators/workflow_proposal_validator.ts` |
| Evidence Reader Agent | `.codex/skills/learning-semantic-loop-evidence-reader/`、`prompt_templates/evidence_reader.ts`、`validators/evidence_packet_validator.ts` |
| Direction Reviewer Agent | `.codex/skills/learning-semantic-loop-direction-reviewer/`、`prompt_templates/direction_reviewer.ts`、`validators/review_delta_validator.ts` |
| Closure Reviewer Agent | `.codex/skills/learning-semantic-loop-closure-reviewer/`、`prompt_templates/closure_reviewer.ts`、`validators/closure_review_validator.ts` |
| Scheduler Script | `scripts/simple_semantic_loop.ts` 与 `scripts/simple_semantic_loop/` |

实现只注册四个 Turn role；不存在辅助 Agent、实验执行 Agent 或 Agent
之间的直接通信。

## 2. 已实现的不变量

- Controller 是 SQLite WAL/CAS 权威状态库和唯一写入者。
- 每个 attempt 使用 fresh ephemeral App Server thread；不 resume provider
  thread，不创建 Goal。
- Workflow Decision Turn 固定 `max`；Evidence、Direction、Closure 固定
  `high`，CLI、run config 和 proposal 不能覆盖。
- Workflow Agent 只提交 proposal 和 Stage 特定 Gate criteria；Controller
  编译 typed operands，注入不可移除的 mandatory checks，再将 Stage/Gate
  连同 contract/criteria hash 和 compiler/evaluator version 冻结。
- Gate evaluator 是 total/fail-closed；unresolved pointer、resolver exception、
  artifact byte/hash mismatch 和 evaluator version drift 均产生 failed check。
- Evidence 仅有两个只读 Obsidian 工具；query、path、term provenance、
  hit/read/context 和 no-delta accounting 均与 runtime trace 对齐。
- 四类 Agent 输出均视为不可信。structure、binding 或 pre-Gate semantic
  error 原子记录后，以同一 logical task 的 fresh Turn 纠错；输入包含
  hash-bound `correctionFeedback`，不包含上一份 raw 自由文本。
- output 最多三次、provider failure 最多两次，总 attempt 最多四次；过期 CAS、frozen Gate
  failure、安全和预算错误不伪装成 output correction。
- provider 完成且 raw 已落盘的中断 attempt 在启动时本地重放，不恢复
  provider history，也不重复模型调用。
- L1–L6 是固定修改坐标；Direction ModificationAtom 必须是 Topic layerScope
  子集。最终 renderer 展示 Anchor baseline/张力以及 Direction 修改、因果、
  公平比较、实现、反证和退化条件。
- Direction Reviewer 独立执行四分支决策；Closure Reviewer 独立检查
  13 项闭包事实。
- 完成只能经过 StopCandidate、机械 preflight、Closure accept、
  确定性 render/coverage validator 和原子 completed commit。
- 实验只形成不可执行 handoff；权限、协议、Stage registry 和 runtime
  event 四层均禁止执行实验。

## 3. 自动化验证

2026-07-29 的本地验证结果：

- 四个 Skill 均通过 `skill-creator/scripts/quick_validate.py`。
- `node --test scripts/simple_semantic_loop/tests/*.test.ts`：
  `59/59` 通过。
- `node scripts/simple_semantic_loop.ts doctor --no-provider`：全部本地检查
  通过；本轮没有为验证再次发起付费 provider probe。
- 先前 provider capability probe 已确认模型 `gpt-5.6-sol` 支持
  `low, medium, high, xhigh, max, ultra`。
- 四个 structured-output schema 均通过 App Server 子集检查；schema
  manifest SHA-256：
  `cc2ee8e1ffa946c80448d8f682c41ac1349b88e4508451dd4132c7f55aeb6301`。

测试覆盖正常闭环、experiment-required handoff、11 个语义 trigger、
动态 plan cycle/上限、Stage/Gate 不可变、CAS、run lock、crash
reconcile、captured-raw zero-provider replay、structure/binding/semantic
correction、provider/output 独立预算、Gate 编译/强制检查/total evaluation、
安全事件、Evidence runtime trace、L1–L6 subset、Direction 四决策、Closure
13 项检查、完整确定性渲染、completed 后拒绝写入，以及 fake App Server 下的
显式 YOLO policy、Agent delta、tool 状态和实时控制台转发。

## 4. 被中断 run 的只读恢复演练

对
`learning_outputs_codex/multimodal_inference_latency_first/`
做了副本演练，未修改真实 run：

- 原状态通过 `validate`：SQLite integrity `ok`、event cursor 连续；
- attempt `attempt-22d24eaa-6533-4b4e-a0a5-ae0100d599a0` 的 completed raw
  artifact 已存在，但 attempt 尚为 `running`；
- Controller 对旧 task 缺失的 nullable `correctionFeedback` 只做
  hash-verified recovery compatibility projection；
- raw Turn 在本地重放，provider 调用数为 `0`；
- 原 Evidence result 通过 output validator，但旧 frozen Gate 是旧 DSL，
  因而按原 Gate total/fail-closed，attempt 进入 `gate_failed` 并记录
  `stage_gate_failed`；
- Controller 没有静默重写旧 Gate，也没有重复读取 Evidence。后续只允许新的
  Workflow Decision Turn 根据 Gate failure 选择恢复路线。

两个临时审计副本已删除。

## 5. 真实 provider canary

工作目录：
`learning_outputs_codex/simple_semantic_loop_canary_20260729_v7/`

Run：
`run-7b3dd74b-c505-40e9-a251-6f3b80c7bce0`

该 canary 已实际验证：

- Workflow `max` fresh Turn；
- Evidence `high` fresh Turn；
- Obsidian Omnisearch 工具调用和只读 runtime trace；
- 无效结构化结果的同角色 fresh retry；
- retry exhaustion 后的显式 resume/reconcile；
- 已有 pending SearchNeed 的动态 Stage 绑定；
- 动态 Gate 失败诊断及修复后的 Evidence Gate pass/commit。

最终状态：

- lifecycle：`waiting_user`；
- snapshot/event cursor：`20/20`；
- canonical revision：`3`；
- pending/in-flight/unconsumed result：均为空；
- SQLite integrity：`ok`；
- event cursor：连续；
- `validate`：`valid=true`。

停止原因符合权限边界：本地 `idea_notes/` 未找到能形成 Anchor 的合格证据，
而扩大证据范围或改变验收目标需要用户授权。Controller 因此持久化一个
`ASK_USER` operator request 后停止，没有伪造 Anchor/Direction、误报
completed 或继续无界循环。该 canary 跨越了实现期的多次诊断修订；完整
raw Turn、validation、Gate、usage 和事件历史均保留在上述工作目录。
