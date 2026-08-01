# Learning Workflow：Agent 编排与脚本协作设计约束

> 归档状态：旧版工作流设计，仅供设计追溯。

## 1. 核心约束

以下原则是后续实现的硬约束，而不是提示词层面的建议：

1. 每个 Skill 只对应一个 Agent。Agent 逻辑应尽量简单，工作流必须拆分到不需要上下文压缩的粒度。
2. 禁止 Agent 直接启动或管理其他 Agent。脚本独占不同类型 Agent 的生命周期、任务状态、Session 和消息转发管理。
3. 短暂 Agent 用于一次性任务或思考，一般负责读取知识库并回答给定输入，不负责状态管理。
4. 持久 Agent 负责根据脚本反复提供的当前情况作出流程推进决策。它需要整体理解所属 Stage 的运行逻辑，但不负责基于知识库自由联想或检索推理。
5. JSON 可以用于提高业务语义密度，但任务状态和 workflow 控制状态使用 Marker、LOOP 或特殊字符协议表达，避免让模型用复杂 JSON 同时管理语义与控制流。

参考实现：

- `.claude/skills/idea_question/SKILL.md`
- `.claude/skills/idea_answer/SKILL.md`
- `scripts/idea_review_orchestrator.ts`

需要复用的核心思想是：

```text
Skill 定义语义流程
Agent 执行一个简单角色
协议暴露本轮动作和恢复点
脚本管理 Session、任务状态和消息转发
```

## 2. Skill 与 Agent 的拆分

不再使用一个 `layered-exploration-workflow` Skill 同时描述 Discovery、Curator、Direction、Judge、Evidence 等角色。

建议拆分为以下 Agent 类型：

| Skill / Agent | 生命周期 | 知识库权限 | 职责 |
|---|---|---:|---|
| `anchor-stage-controller` | Stage 1 持久 | 无 | 根据当前 AnchorSpace 和上一轮结果规划下一轮、判断收敛 |
| `anchor-evidence-worker` | 单任务短暂 | 只读 | 针对一个探索角度读取知识库，返回证据与候选 Claim |
| `anchor-curator-worker` | 单任务短暂 | 无或只读 evidence packet | 将一批 Claim 转成 Anchor/Baseline/Entry/Edge delta |
| `direction-planner` | 每个 Anchor 一个持久 Session | 无 | 理解单个 Anchor，逐步构造显著不同的 Direction，判断该 Anchor 是否完成 |
| `direction-reviewer` | 每个 Direction 一个持久 Session | 无 | 按专家流程提问、评估回答、决定继续或完成 |
| `review-evidence-worker` | 每个问题一个短暂 Session | 只读 | 读取知识库或 evidence packet，回答一个明确问题 |

“一个 Skill 对应一个 Agent”是严格的一对一：

- Skill 不描述其他 Agent 的行为；
- Agent 不调用或启动其他 Agent；
- Agent 只知道自己接收什么、输出什么；
- 其他 Agent 是否启动，由脚本根据协议决定。

## 3. 持久 Agent 和短暂 Agent 的边界

### 3.1 短暂 Agent

短暂 Agent 每次只完成一个封闭任务：

```text
接收 Task
→ 检索或思考
→ 输出结果协议
→ Session 关闭
```

例如：

```text
task_id: AE-R03-L2-02
问题: 找到 MoE decode 中 expert imbalance 的 runtime baseline 和加速证据
```

短暂 Agent 不需要知道：

- 当前 Stage 1 是第几轮；
- 已有多少 Anchor；
- 是否达到 30；
- 下一步应该调用谁；
- 整个 workflow 是否完成。

这些都属于脚本和持久 Controller。

### 3.2 持久 Agent

持久 Agent 只负责某个有限 scope 内的流程决策：

- `anchor-stage-controller`：只负责 Stage 1，最多约 10 个全局 Round；
- `direction-planner`：只负责一个 Anchor；
- `direction-reviewer`：只负责一个 Direction。

不能让一个持久 Agent 跨越：

```text
Stage 1 → 30 个 Anchor → 全部 Direction → 全部 Review
```

否则必然触发上下文压缩和状态遗忘。

## 4. Stage 1 的调用方式

每个 Explore Round 只调用持久 Controller 一次，而不是每个 Worker 结果都调用一次：

```text
Anchor Controller
    │ 输出 ROUND_PLAN
    ▼
脚本生成并调度多个短任务
    ├── Evidence Worker 1
    ├── Evidence Worker 2
    └── Evidence Worker N
    │
    ▼
脚本聚合证据
    │
    ├── Curator Worker 1
    └── Curator Worker N
    │
    ▼
脚本验证、去重、更新 Top-30
    │
    ▼
形成 ROUND_RESULT
    │
    └── 重新输入 Anchor Controller
```

因此，Stage 1 持久 Agent 的上下文只包含若干个 Round，而不是数百个搜索任务。

## 5. 协议负责控制流，JSON 只承载语义对象

不要求 Agent 用一个复杂 JSON 同时表达：

- 业务内容；
- 当前任务状态；
- 下一步动作；
- Session 恢复点；
- workflow 完成状态。

后续采用“控制协议 + 语义载荷”两层输出。

### 5.1 控制层：Marker 与 LOOP

例如 Stage 1 Controller：

```text
___ANCHOR_ROUND_PLAN_START___
round: 3
goal: expand_new_anchor
task_count: 4
___ANCHOR_ROUND_PLAN_END___

___TASK_PAYLOAD_START___
[
  {
    "task_id": "AE-R03-01",
    "focus": "MoE decode runtime scheduling baseline",
    "layer": "L2",
    "value_axis": "exploration"
  }
]
___TASK_PAYLOAD_END___

[LOOP: §EVAL_ROUND | await=ROUND_RESULT | round=3]
```

其中：

- Marker 表示当前输出类型；
- `LOOP` 表示下一次恢复位置；
- `await` 表示脚本下一次必须输入的信号；
- JSON 只用来紧凑表达任务内容；
- Agent 不输出完整 `state.json`。

### 5.2 短暂 Worker 输出

```text
___EVIDENCE_TASK_RESULT_START___
task_id: AE-R03-01
status: complete

___SOURCES_START___
- /data3/paper_analysis/knowledge_notes/example.md:120-138
___SOURCES_END___

___CLAIMS_START___
[
  {
    "statement": "...",
    "layer": "L2",
    "quote": "..."
  }
]
___CLAIMS_END___

___GAPS_START___
- 未找到 strong baseline 的定量结果
___GAPS_END___
___EVIDENCE_TASK_RESULT_END___

[TASK_TERMINATED]
```

短暂 Agent 不需要输出 `LOOP`，因为它不会恢复；输出完整结果后脚本直接关闭 Session。

### 5.3 Stage 完成

```text
___ANCHOR_STAGE_COMPLETE___
reason: target_reached
proposed_anchor_count: 30
___ANCHOR_STAGE_COMPLETE_END___

[LOOP: §TERMINATED | done]
```

即使 Controller 声明完成，脚本仍然检查：

```text
accepted_anchor_count == 30
OR no_material_gain_streak >= 2
OR round/budget limit reached
```

Agent 的协议是请求或判断，脚本才是最终状态裁决者。

## 6. 脚本给持久 Agent 的每轮输入

不能只依赖 provider Session 的记忆。脚本每次都重复输入：

```text
本次执行语义
当前规范状态
上次持久 Agent 的完整规范化输出
本轮新收到的 Worker 结果
```

示例：

```text
本次执行语义：
从 §EVAL_ROUND 开始；已经收到 ROUND_RESULT。
评估 round 3 的有效增量，决定继续规划或结束 Stage 1。

___CURRENT_STAGE_STATE_START___
round: 3
active_anchor_count: 17
max_anchors: 30
no_material_gain_streak: 0
remaining_budget: 7
___CURRENT_STAGE_STATE_END___

___PREVIOUS_CONTROLLER_OUTPUT_START___
<上次 Controller 的 ROUND_PLAN，完整重复>
___PREVIOUS_CONTROLLER_OUTPUT_END___

___ROUND_RESULT_START___
<脚本验证、去重和提交后的规范结果>
___ROUND_RESULT_END___
```

Session 记忆只是辅助，脚本回注的状态才是权威信息。

## 7. 持久 Skill 使用 idea-question 风格伪代码

持久 Skill 使用明确任务块：

```text
§1 / §INIT
§2 / §PLAN_ROUND
§3 / §WAIT_RESULT
§4 / §EVAL_ROUND
§5 / §TERMINATE
```

每个任务块必须写清：

1. 接收什么输入；
2. 顺序执行哪些步骤；
3. 哪些是内部 `GOTO`；
4. 在哪里 `YIELD`；
5. 输出哪一种 Marker；
6. 下次 LOOP 从哪里恢复。

例如：

```text
§4 / §EVAL_ROUND

输入：ROUND_RESULT、当前 StageState、上次 ROUND_PLAN。

§4.1 检查脚本确认的 accepted delta
§4.2 检查 Anchor 数、no-gain streak 和剩余 frontier
§4.3 若满足完成条件 → GOTO §5
§4.4 否则形成下一轮唯一探索目标 → GOTO §2
```

Agent 不能停在“接下来应该继续搜索”这种叙述状态，必须执行到一个明确的 `YIELD` 或 `TERMINATE`。

## 8. 脚本独占生命周期管理

脚本负责：

```text
创建 Session
发送 Skill 和初始化输入
保存 session_id
解析 Marker
解析 LOOP
检查状态转移是否合法
启动短暂 Worker
转发规范化 Worker 结果
保存 checkpoint
恢复中断 Session
关闭 Session
```

Agent 只能通过协议请求工作：

```text
REQUEST_EVIDENCE
PLAN_NEXT_ROUND
REQUEST_REVIEW_ANSWER
STAGE_COMPLETE
```

Agent 不能：

- 使用 Task/Agent 工具；
- 运行 Claude/Codex 子进程；
- 创建或恢复 Session；
- 直接向另一个 Agent 发送消息；
- 写 `state.json`；
- 判断某个 Worker 是否已经启动。

脚本对 persistent session 和 task 分别记录状态：

```text
Session:
not_started → initializing → waiting_input
            → running → yielded → terminated

Task:
pending → dispatched → response_received
        → protocol_valid → domain_valid → committed
        ↘ failed_retriable
        ↘ failed_terminal
```

## 9. 协议失败处理

复用 `idea_review_orchestrator.ts` 已验证的策略：

1. 保存 raw stream；
2. 只提取角色允许的 Marker；
3. Marker 或 LOOP 缺失时，在同一 Session 发送一次 `[PROTOCOL_REPAIR]`；
4. Repair 禁止检索、禁止重做语义任务，只补全已经完成的协议输出；
5. 再失败则记录为 `failed_retriable`，脚本不猜测状态；
6. Resume 时检查：
   - protocol version；
   - session ID；
   - provider/model；
   - last LOOP；
   - last normalized output；
   - canonical task state。

## 10. 对当前实现的直接影响

当前实现需要调整：

- 当前单一 `layered-exploration-workflow` Skill 拆成多个单 Agent Skill；
- Discovery、Curator、Direction、Judge、Evidence 不再只是同一 Skill 下的参考角色；
- Discovery/Evidence 改为单任务短 Session；
- Stage 1 增加持久 Controller；
- Direction 构造改为每个 Anchor 一个受限持久 Agent；
- Review 从默认 stateless Judge 改为每个 Direction 一个持久 Reviewer；
- 当前复杂 JSON schema 不再承担 workflow 状态；
- Provider 层改为解析 Marker、payload 和 LOOP；
- 保留 JSON 作为 Anchor、Entry、Direction 等语义对象的中间表达；
- 所有 canonical JSON 仍由脚本校验后写入。

## 11. 后续实现顺序

后续应按以下顺序完善：

1. 定义 Agent 划分与每个 Skill 的唯一职责；
2. 定义每个角色允许输出的 Marker；
3. 定义 LOOP 状态迁移表；
4. 定义脚本 checkpoint 和 Session 生命周期；
5. 实现各个 Skill；
6. 实现 orchestrator；
7. 编写协议解析、状态恢复和异常输出测试；
8. 最后接入 DeepSeek/Codex provider 并执行端到端验证。
