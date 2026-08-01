# 归档：四角色 SQLite/Stage/Gate Simple Semantic Loop

> 归档日期：2026-07-30。该目录只用于设计与实现追溯；现行入口为
> `scripts/simple_semantic_loop.ts`，不要从本目录启动新 Run。

Simple Semantic Loop 由一个确定性 Controller 和四类按需启动的 fresh Turn
组成。Controller 负责持久状态、调度、校验、Gate、CAS 提交和停止；Agent
只处理一个结构化任务并退出。

| Turn | 推理强度 | 职责 |
|---|---:|---|
| Workflow Decision | `max` | 在脚本触发语义判断时提出一个决策或动态 Stage/Gate |
| Evidence Reader | `high` | 在授权的本地 Obsidian 路径中读取一个 SearchNeed 的证据 |
| Direction Reviewer | `high` | 独立审查一个 Direction |
| Closure Reviewer | `high` | 独立审查 StopCandidate 和 StopProof |

没有辅助 Agent、持久 Agent、Agent 间通信或实验执行 Agent。Workflow
Decision Turn 只能提出 proposal；只有 Controller 可以改变权威状态。

## 运行前检查

从项目根目录运行：

```bash
cd /data3/paper_analysis
node scripts/simple_semantic_loop.ts doctor
```

`doctor` 会检查：

- Node.js 22 或更高版本以及 `node:sqlite`；
- SQLite WAL、事务和 CAS；
- schema manifest 与四类输出 Schema；
- 四个当前 Skill 及其 role profile；
- `workflow=max`、其余 Turn 为 `high` 的能力映射；
- Codex CLI、指定模型和 reasoning effort；
- Obsidian MCP 只读能力配置；
- 输出目录可写。

仅检查本地依赖、不连接 provider：

```bash
node scripts/simple_semantic_loop.ts doctor --no-provider
```

默认模型是 `gpt-5.6-sol`。可通过 `doctor --model <model>` 检查另一个模型，
也可在初始化时用 `--model <model>` 固定本次 run 的模型。

## 示例：多模态推理加速

用户主题：

```text
多模态推理加速, 优先优化延迟, 保证较高吞吐
```

建议将“延迟优先”写进 objective，同时把吞吐、质量和比较公平性写成不可静默
放宽的 acceptance criteria：

```bash
cd /data3/paper_analysis

node scripts/simple_semantic_loop.ts init \
  --topic '多模态推理加速, 优先优化延迟, 保证较高吞吐' \
  --objective '识别多模态推理中的可验证加速方向；以端到端延迟为第一优化目标，同时避免显著牺牲稳态吞吐、输出质量或比较公平性。' \
  --acceptance '不得静默缩窄多模态推理范围；未知的模态、模型、阶段和负载应保留为待决问题。' \
  --acceptance '优先分析端到端延迟及关键阶段延迟；吞吐、质量、硬件、精度格式、batch 和 SLO 必须在公平条件下比较。' \
  --acceptance '每个 Direction 必须具备可追溯本地证据、明确 baseline、作用机制、实现边界、失败条件和测量方案。' \
  --acceptance '需要新实验才能确定的结论只生成 EXPERIMENT_REQUIRED handoff，不执行实验。' \
  --model gpt-5.6-sol \
  --work-dir /data3/paper_analysis/learning_outputs_codex/multimodal_inference_latency_first
```

`init` 只创建 Controller 状态和初始导出，不启动 Agent。若目标目录中已经存在
`workflow.db`，它会拒绝覆盖。

启动实际工作流：

```bash
node scripts/simple_semantic_loop.ts run --yolo \
  --work-dir /data3/paper_analysis/learning_outputs_codex/multimodal_inference_latency_first
```

`run` 会启动真实 Codex App Server Turn，可能产生模型调用费用。每个 attempt
都创建新的 ephemeral thread；不会 resume provider thread，也不会创建或使用
Codex Goal。

## 权限模式与实时控制台

默认模式已经设置 `approvalPolicy=never`，不会弹出交互审批，但 provider
Turn 使用 `sandbox=read-only`，因此不是 YOLO。

显式增加 `--yolo` 后，provider Turn 使用：

```text
approvalPolicy = never
sandbox = danger-full-access
```

这提供 Codex CLI 所称的 YOLO 权限。它不会扩大 Workflow 协议中的 role、
Stage、tool 和 path allowlist：Workflow、Direction 和 Closure Turn 仍不得调用
工具，Evidence Reader 仍只允许两个 Obsidian 只读工具。不过，YOLO 下违规工具
可能在 Controller 完成事后校验前已经产生外部效果，因此该模式是明确的 operator
风险选择，不是安全边界。

`run` 和 `resume` 默认实时向标准错误流转发：

- App Server 启动和 stderr；
- Turn role、attempt、effort、sandbox、thread 和 turn ID；
- Agent 输出增量；
- 工具开始/完成状态；
- Turn 状态、耗时、token usage 和工具调用数。

最终 Controller JSON 仍写到标准输出，因此可以分别重定向。如果只需要机器可读
输出，可增加 `--quiet`：

```bash
node scripts/simple_semantic_loop.ts run --quiet --work-dir <run-dir>
```

## Controller 实际执行过程

典型闭环如下：

1. Controller 从 SQLite 权威状态计算确定性转换；只有出现已注册的语义
   trigger 时才启动 Workflow Decision Turn。
2. Workflow Decision Turn 读取当前快照、触发原因、已提交产物和权限边界，
   输出一个 `WORKFLOW_DECISION_PROPOSAL`。
3. Controller 校验 proposal 的版本、哈希、动作权限、目标不变性和预算。Agent
   只能提出 Stage 特定的类型化 Gate 条件；Controller 会拒绝非法/矛盾条件，
   注入不可移除的 Schema、binding、权限、预算、No Experiment、引用与幂等
   检查，再把有效 Stage Contract 和完整 Gate 一起冻结、版本化和哈希。
4. `EVIDENCE_READ` Stage 启动一个 Evidence Reader Turn，产生
   `EVIDENCE_PACKET`；Controller 对其 Schema、引用、路径、工具轨迹和 Gate
   进行独立校验后再提交。
5. Workflow Decision Turn 将已提交且尚未消费的结果提议整合为 TopicFrame、
   SearchNeed、Anchor、Direction、SemanticDelta 或 plan patch；Controller
   再次校验并以 CAS 提交。
6. 每个 Direction 在需要时由独立 Direction Reviewer Turn 输出
   `REVIEW_DELTA`，Worker 不定义并裁决自己的成功。
7. 当计划已耗尽但目标仍开放、证据冲突、Gate 无唯一机械恢复路线、多条非等价
   路径同时可运行或长期无进展时，Controller 再触发 Workflow Decision Turn。
8. Workflow Agent 只能在十项闭包事实均满足时提出 StopCandidate/StopProof。
   Controller 随后执行机械 preflight，再启动 fresh Closure Reviewer Turn。
9. 只有 Closure Reviewer 接受、权威 revision 未变化且确定性渲染覆盖校验通过，
   Controller 才原子写入 `final.md` 并将 lifecycle 提交为 `completed`。

Agent 输出不是完成信号；完成只由第 9 步的 Controller 提交决定。

## Agent 输出不可信与异常式纠错

Controller 假定四类 Agent 都可能返回：

- 无法解析、缺字段或带额外字段的结构错误；
- JSON/Schema 合法，但 identity、state、hash 或 attempt 绑定错误；
- 结构与绑定合法，但动作、权限、引用、领域不变量、Stage 或 Gate 草案语义错误。

处理顺序固定：

```text
保存 raw Turn（untrusted）
→ 有限规范化
→ Schema / binding / role / domain / proposal preflight
→ 无效：原子记录 ValidationReport，不执行 Gate，不提交 canonical state
→ 同一 logical task + 新 attemptId + 新 provider thread
→ 注入 correctionFeedback 后重新执行完整 Turn
```

`correctionFeedback` 只包含前一 attempt ID、原输出 SHA-256、ValidationReport
ID/hash、错误类别，以及最多 32 条带 JSON Pointer 的 Controller 错误；每条
错误同时携带 `requiredRule` 和由 Controller 固定生成的 `validExamples`。
Prompt 将其表述为“实际错误 + 必须满足的规则 + 合法形式”，不会让 Agent
从模糊报错中猜格式，也不会把上一份任意自由文本输出重新注入 Prompt。新 Turn
必须返回完整替代对象，不是 JSON Patch，并重新校验整个替代对象而非只修一个
pointer。纠错前若权威状态已变化，Controller 会按 stale task 失败关闭，不会
把旧错误包应用到新状态。

固定预算是：

- 每个 logical task 最多 3 次输出契约尝试（即最多 2 次纠错）；
- 最多 2 次 Provider 失败尝试；
- 两类故障交错时总 attempt 上限为 4。

Agent、run config 和 `--yolo` 都不能放大这些上限。Provider 失败不会伪装成
输出错误；安全违规、预算违规和 stale binding 也不能靠改写 JSON 绕过。

只有通过上述 pre-Gate 校验的输出才运行已冻结 Gate。Gate 由 Controller
编译、保存和执行，不是 Workflow Agent 的最终判断。若合法结果未通过 Gate，
Controller 原子记录 Gate failure，并在需要语义选择时触发新的 Workflow
Decision Turn；它不会把 Gate failure 当成同任务“格式纠错”。

若进程在 provider 已完成、raw Turn 已落盘但尚未提交时被杀死，下一次
`run`/`resume` 会先校验并本地重放该 raw artifact，不重复调用 provider。
只有没有可验证 raw 结果的中断 attempt 才按独立的 Provider 恢复预算处理。

## 查看状态和过程

读取当前权威状态的导出：

```bash
node scripts/simple_semantic_loop.ts status \
  --work-dir /data3/paper_analysis/learning_outputs_codex/multimodal_inference_latency_first
```

读取事件日志。默认是一行一个 JSON，适合配合 `tail`：

```bash
node scripts/simple_semantic_loop.ts events \
  --work-dir /data3/paper_analysis/learning_outputs_codex/multimodal_inference_latency_first \
  | tail -n 20
```

以 JSON 数组输出全部事件：

```bash
node scripts/simple_semantic_loop.ts events \
  --json \
  --work-dir /data3/paper_analysis/learning_outputs_codex/multimodal_inference_latency_first
```

检查 SQLite 完整性、事件 cursor 连续性和 completed/final 绑定：

```bash
node scripts/simple_semantic_loop.ts validate \
  --work-dir /data3/paper_analysis/learning_outputs_codex/multimodal_inference_latency_first
```

`workflow.db` 是权威状态；`exports/` 是 Controller 在运行过程中原子刷新的可读
投影，不能反向编辑后当作状态输入。

## 暂停、恢复和取消

```bash
node scripts/simple_semantic_loop.ts pause --work-dir <run-dir>
node scripts/simple_semantic_loop.ts resume --yolo --work-dir <run-dir>
node scripts/simple_semantic_loop.ts cancel --work-dir <run-dir>
```

当前 `resume` 只会把以下 lifecycle 转回 `running`：

- `paused_budget`
- `paused_operator`
- `failed_retriable`
- `blocked_semantic`
- `blocked_external`

重要限制：协议已经支持 `ASK_USER` 并会持久化 `operator_requests`，但当前 CLI
还没有“提交用户答案”的命令。因此 run 若停在 `waiting_user`，不能仅靠
`resume` 正确继续；`waiting_external` 也没有外部结果提交接口。不要直接修改
SQLite 绕过这一边界，应先补齐对应的 Controller/CLI 输入事务。

`run` 或 `resume` 只有在 lifecycle 为 `completed` 时返回退出码 `0`；在
等待、暂停、阻塞或达到单次 transition 上限后返回退出码 `2`。运行错误返回
退出码 `1`。因此退出码 `2` 不等于完成，也不一定表示实现异常，应结合
`status` 和 `events` 判断。

`render` 不会绕过闭包审查生成结果；它只确认已经由 finalization 生成的
`final.md`：

```bash
node scripts/simple_semantic_loop.ts render --work-dir <run-dir>
```

## Run 目录中的产物

```text
<run-dir>/
├── workflow.db                  # SQLite 权威状态、事件、任务、attempt、结果和 CAS 版本
├── config.json                  # run/model/预算/路径配置
├── schema_manifest.json         # 本次 run 固定的协议 Schema 清单
├── prompts/                     # 每个 attempt 的完整冻结 Prompt
├── raw_turns/                   # provider 原始 Turn、usage 和工具事件
├── exports/
│   ├── workflow_state.json      # 当前状态快照
│   ├── workflow_plan.json       # 当前动态 plan
│   ├── topic.json
│   ├── anchors.jsonl
│   ├── directions.jsonl
│   ├── search_needs.jsonl
│   ├── stop_candidates.jsonl
│   ├── evidence_packets.jsonl
│   ├── direction_reviews.jsonl
│   ├── closure_reviews.jsonl
│   ├── tasks.jsonl
│   ├── attempts.jsonl
│   ├── events.jsonl
│   ├── validation.json
│   └── usage.json
├── artifacts/
│   └── manifest.jsonl           # Prompt、raw Turn 和最终产物的哈希清单
└── final.md                     # 仅在 accepted closure 后出现
```

每个 Agent attempt 都应能从 `prompts/`、`raw_turns/`、`exports/tasks.jsonl`、
`exports/attempts.jsonl`、`exports/validation.json` 和
`artifacts/manifest.jsonl` 交叉追溯。只有 Gate 通过且结果已由 Controller
提交的内容才进入 canonical 导出。

## 不启动 Agent 的验证

```bash
node --test scripts/simple_semantic_loop/tests/*.test.ts
```

自动化测试使用可编程的 fresh-Turn runtime 覆盖 Controller、协议、Gate、
恢复、安全和闭包逻辑，并用 fake App Server 验证 YOLO 参数与实时输出转发；
不会调用付费 Agent。它不能代替针对具体 topic 的真实 provider run。

实现计划与验证记录：

- [实现计划入口](../../draft/learning_workflow_simple_semantic_loop_implementation_plan.md)
- [Scheduler Script 实现计划](../../draft/learning_workflow_simple_semantic_loop_plans/scheduler_script_implementation_plan.md)
- [实现与验证记录](../../draft/learning_workflow_simple_semantic_loop_plans/implementation_verification.md)
