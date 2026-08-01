# 三类 Agent 的编写与协作模式

> 归档状态：旧版 Learning Skill/Agent 协作草案，仅供设计追溯。

本文基于以下现有实现整理可复用的脚本与 skill 编写经验：

- `scripts/learning_scheduler.ts` 与 Question、Answer、Horizon Summary、Vertical Summary skills：通过四阶段任务图批量生成、调度和验收一次性子 Agent
- `scripts/idea_review_orchestrator.ts`：持久 Question/Answer Agent 的多轮协作
- `.claude/skills/idea_question/SKILL.md`：盲评、追问与最终判断
- `.claude/skills/idea_answer/SKILL.md`：独占 idea note、渐进获取证据并回答
- `draft/idea_review_protocol_spec.md`：当前 Q/A 协议和 loop 的详细规格

目标不是寻找一种通用 Agent 模板，而是先判断任务依赖哪种控制方式，再选择对应写法。

---

## 一、核心结论

Agent 可以按控制权和上下文用途分为三类：

| 类型 | 适合的任务 | 控制权主要位于 | 长上下文的用途 | 推荐写法 |
|------|------------|----------------|----------------|----------|
| 1. 状态驱动的一次性子 Agent | 可拆分、可并行、输入输出明确的批处理任务 | 调度脚本 | 仅完成当前任务 | 自包含输入 + 线性 workflow + 产物/DONE 信号 |
| 2. 协议驱动的复杂 workflow Agent | 多角色需要反复交换信息、按条件分支并逐步收敛的任务 | 持久调度 Agent 决定语义流程；脚本执行确定性调度 | 保留目标、判断依据和累计结果 | 显式非线性伪代码 + marker + 脚本提醒 |
| 3. 目标驱动的自主规划 Agent | debug、调查、开放式实现等路径无法预先枚举的任务 | Agent 自主规划，外部只规定边界 | 保存证据、假设、尝试和代码上下文 | 明确目标/约束/验收，少规定流程 |

最重要的边界是：

> 复杂 workflow 的主干应由伪代码显式定义；marker 只提供脚本调度所需信号，脚本不需要理解领域流程。

长上下文适合保留目标、判断依据和累计结果，但复杂 flow 容易在多轮中偏离。解决方式不是反复注入 JSON state，而是让 skill 用短伪代码定义非线性 flow，并由脚本根据上轮 marker 回注简短执行提醒。

---

## 二、Flow、Marker 与脚本调度

复杂协作中需要区分的不是四类“状态”，而是三个职责不同的控制层：

| 控制层 | 作用 | 推荐归属 |
|--------|------|----------|
| Flow 定义 | 定义非线性步骤、分支条件、循环和终止条件 | 调度 Agent 的 skill，以伪代码显式定义 |
| Marker 协议 | 声明本轮输出类型、应转发给谁、是否等待或完成 | Agent 输出；脚本只解析 marker |
| 调度执行 | 启停 Session、维护运行状态、按预定规则转发、记录日志 | 调度脚本 |

三者的关系是：

```text
skill 中的伪代码：
  决定“收到某类结果后，应继续追问、派发新任务，还是结束”

Agent 输出的 marker：
  告诉脚本“本轮产生了哪类可调度结果”

脚本：
  根据固定映射启动 Agent 或转发 marker 包装的内容
  并在下一次调用持久 Agent 时附上简短执行提醒
```

Marker 不需要携带完整业务状态，也不需要把 flow 转换成 JSON。它只需要提供脚本真正使用的信息，例如：

- `___SPAWN_ANSWER_AGENT___`：创建临时 Answer 子 Agent
- `___FORWARD_TO_QUESTION_AGENT___`：将结果转发给 Question Agent
- `___QA_REFERENCE_REQUEST___`：按白名单注入 reference
- `___JUDGMENT_COMPLETE___`：保存最终产物并结束
- `[LOOP: §DIM_EVAL | await=ANSWER_RESULT]`：提醒持久 Agent 下一次从哪个伪代码段继续

脚本维护的是 Agent 的外部运行状态，例如 `initializing/running/waiting/completed/failed`。它不需要解析 `DIM_QUEUE`、`review_material` 等领域判断，也不需要将这些内容序列化后反复塞回上下文。

---

## 三、类型 1：状态驱动的一次性子 Agent

### 3.1 适用条件

满足以下条件时，优先生成一次性子 Agent：

- 一个任务可以独立完成，不需要与其他 Agent 多轮对话
- 输入、输出文件和完成条件可以在启动时确定
- 任务可按问题、层次、文件或数据分片并行
- 失败后可直接使用相同输入重试
- 后续阶段只依赖产物，不依赖该 Agent 的隐藏上下文

`learning_scheduler.ts` 的 Question、Answer、Horizon Summary、Vertical Summary Agent 都属于此类。

### 3.2 控制模型

```text
调度脚本：
  建立任务池
  将任务标记为 pending

worker:
  领取 pending task
  标记 running
  生成“skill body + 当前任务参数”的自包含 prompt
  启动一次性 Agent
  等待进程结束
  校验输出文件和 DONE marker
  成功 -> done
  失败 -> pending，允许重试
```

这里的关键经验是：

> 领取任务不等于完成任务；进程退出也不等于产物有效。

当前 learning scheduler 使用 `pending/running/done`、输出文件和 DONE marker 共同确认完成，并在 resume 时用磁盘产物修复 checkpoint。这比依赖 Agent 自报进度或进程计数可靠。

### 3.3 脚本负责什么

- 解析用户输入，生成任务池
- 为每个子任务注入具体参数
- 控制并行度、超时、重试和进程清理
- 保存 `pending/running/done`
- 验证输出文件和 DONE marker
- 从上游产物生成下一阶段子任务
- checkpoint/resume 和真实进度展示

### 3.4 Skill 负责什么

- 当前单个任务的目标
- 可用输入和输出路径
- 完成当前任务所需的方法
- 质量要求、证据规则和输出格式
- 唯一完成信号

Skill 不需要描述 worker pool、重试、checkpoint 或其他子 Agent。这些属于脚本。

### 3.5 推荐 Skill 结构

```markdown
# <Role>

## 目标
完成一个明确任务，并写入指定输出文件。

## 输入
- task_id
- 输入文件
- 输出文件
- 用户约束
- 完成信号

## Workflow
1. 读取输入
2. 执行当前任务所需分析
3. 按输出模板写入结果
4. 自检
5. 在文件末尾写入完成信号

## 输出格式
<固定模板>

## 完成条件
- 输出文件存在且内容完整
- 文件末尾包含 [DONE] <task_id>
```

一次性 Agent 可以使用较长、较详细的 skill，因为它只需在线性执行中读取一次，不需要在多轮输入后持续记住 workflow 位置。

### 3.6 从 learning scheduler 提取的模式

1. `fillSkillInput` 将每个任务的动态参数直接嵌入 skill，使 prompt 自包含。
2. 问题空间文件生成 Answer 子任务；Answer 产物再生成 Horizon 子任务。Agent 输出用于生成后续任务，但任务图由脚本掌握。
3. checkpoint 记录真实任务状态；恢复时将遗留的 `running` 重置为 `pending`。
4. DONE marker 是产物契约，不是装饰文本。
5. 进度展示必须来自内部 worker 状态和产物校验，不能依赖不可靠的进程名统计。
6. Skill 不应一边声明“只读”，一边要求写入输出文件；应明确为“只允许写指定产物路径”，并由启动参数落实权限边界。

---

## 四、类型 2：协议驱动的复杂 Workflow（持久调度 + 临时执行）

### 4.1 适用条件

满足以下条件时，适合使用协议驱动的复杂 workflow：

- flow 存在条件分支、循环或动态派发，无法写成固定流水线
- 调度决策需要结合之前多个任务结果逐步收敛
- 实际执行任务可以被包装成输入输出明确的子任务
- 不同角色拥有不同权限、上下文或执行方法
- 脚本可以根据 marker 用预定规则完成启动和转发

当前 idea review 中：

- QA 不持有 idea note，负责盲评、提问和最终判断
- AA 独占 idea note 和检索工具，负责自包含回答
- 编排器负责在二者之间转发、注入 reference、校验协议和保存 review

它已经体现了伪代码 flow、marker 和脚本转发三部分，但当前使用两个持久角色 Session。对于更一般的复杂 flow，推荐进一步采用“持久调度 Agent + 临时执行子 Agent”。

### 4.2 核心控制模型

复杂 workflow 的 flow 应写在持久调度 Agent 的 skill 中。脚本不重写这套 flow，也不解析调度 Agent 的领域判断，只根据 marker 执行动作。

```text
持久调度 Agent：
  读取累计结果
  按 skill 中的伪代码判断下一步
  输出 marker 包装的派发请求、追问或最终结果
  等待脚本返回执行结果

调度脚本：
  解析 marker
  更新各 Agent 的 initializing/running/waiting/completed/failed
  根据固定规则启动临时子 Agent 或转发输出
  向持久调度 Agent 回传结果，并附上下一执行段提醒

临时执行子 Agent：
  接收自包含任务
  完成一次实际任务
  输出 marker 包装的结果
  结束 Session
```

这样做的原因是：

- 调度逻辑需要跨轮次结合累计结果，适合持久 Session
- 实际任务通常目标明确，适合临时 Session
- 临时子 Agent 不需要记忆整个复杂 flow，只需完成当前任务
- 脚本只做确定性调度，不需要承担领域推理
- marker 比完整 JSON state 更容易生成、解析和恢复

### 4.3 用伪代码显式定义非线性 Flow

复杂 flow 的 skill 应使用命名步骤、条件分支、`GOTO`、`YIELD` 和 `TERMINATE` 显式定义主干：

```text
§INIT
  理解目标与评估标准
  GOTO §PLAN_NEXT

§PLAN_NEXT
  根据当前已有结果判断：
    若证据不足 -> GOTO §DISPATCH_TASK
    若存在关键冲突 -> GOTO §DISPATCH_VERIFY
    若满足完成标准 -> GOTO §FINALIZE

§DISPATCH_TASK
  生成一个自包含的执行任务
  输出 ___SPAWN_WORKER___ ... ___SPAWN_WORKER_END___
  输出 [LOOP: §EVALUATE_RESULT | await=WORKER_RESULT]
  YIELD

§EVALUATE_RESULT
  读取脚本转发的 ___WORKER_RESULT___
  判断结果是否充分、冲突或需要继续深入
  GOTO §PLAN_NEXT

§DISPATCH_VERIFY
  生成针对冲突点的验证任务
  输出 ___SPAWN_VERIFIER___ ... ___SPAWN_VERIFIER_END___
  输出 [LOOP: §EVALUATE_RESULT | await=VERIFIER_RESULT]
  YIELD

§FINALIZE
  输出 ___WORKFLOW_COMPLETE___ ... ___WORKFLOW_COMPLETE_END___
  TERMINATE
```

伪代码应明确：

- 当前步骤收到什么输入
- 根据什么条件分支
- 何时生成哪类 marker
- 输出后从哪个步骤继续
- 何时结束

伪代码不需要描述脚本内部如何 spawn、记录 PID 或写日志。脚本也不需要将伪代码转换为自己的状态机；它只需在每次回传时提醒调度 Agent 从 `[LOOP: §...]` 指定位置继续。

### 4.4 Marker 只承载调度信号

Marker 的目标是让脚本可靠识别动作，而不是保存完整业务状态。

一个派发请求可以写成：

```text
___SPAWN_ANSWER_AGENT___
请回答当前问题，并提供来源与信息缺口：
<问题正文>
___SPAWN_ANSWER_AGENT_END___

[LOOP: §EVALUATE_ANSWER | await=ANSWER_RESULT]
```

临时 Answer 子 Agent 返回：

```text
___ANSWER_RESULT___
<原始 Markdown 回答>
___ANSWER_RESULT_END___
```

脚本只需要识别：

```text
___SPAWN_ANSWER_AGENT___ -> 启动临时 Answer Agent
___ANSWER_RESULT___      -> 转发给持久调度 Agent
___REFERENCE_REQUEST___  -> 按白名单注入 reference
___WORKFLOW_COMPLETE___  -> 保存结果并结束
```

不推荐在每轮输入中增加 JSON state：

- JSON 容易因长文本、转义或字段漂移导致解析失败
- 调度 Agent 已通过持久 Session 保留累计结果
- 伪代码和 `[LOOP: ...]` 提醒已经足以重新定位 flow
- 重复注入状态会挤占并污染核心逻辑上下文

需要记录来源、轮次等信息时，优先使用简单 marker 段或由脚本在日志中记录；不要把完整业务过程复制成一份 JSON 状态。

### 4.5 脚本负责什么

复杂 workflow 中，脚本负责不同 Agent 的运行状态和预定转发规则：

- 启动一个持久调度 Agent
- 根据 marker 启动对应类型的临时执行子 Agent
- 维护每个 Session 的 `initializing/running/waiting/completed/failed`
- 等待临时子 Agent 完成并提取 marker 包装的输出
- 按固定映射将结果转发给持久调度 Agent或其他指定 Agent
- 回注 `[LOOP: §...]` 对应的简短执行提醒
- 管理并行度、超时、预算、错误、重试和 Session 关闭
- 记录时间戳、Agent、输入、输出和最终产物
- 验证 marker 是否存在、是否属于允许类型

脚本不负责：

- 判断某篇论文或某个回答是否有价值
- 解析调度 Agent 的隐含推理
- 保存一份与持久 Session 重复的完整业务 JSON state
- 决定非线性 flow 的下一语义步骤
- 将开放式领域判断硬编码为脚本分支

因此，脚本是**确定性的调度器和消息代理**；持久调度 Agent 是**复杂 flow 的语义控制者**。

### 4.6 持久调度 Agent 负责什么

- 持续记住总目标、评价标准和累计结果
- 按 skill 伪代码执行非线性 flow
- 判断下一步需要派发什么任务
- 将执行任务写成自包含输入
- 评估临时子 Agent 的返回结果
- 判断继续、验证、追问、换方向或完成
- 用 marker 向脚本声明下一调度动作

持久调度 Agent 不直接承担大量检索、文件修改或逐项执行工作。它把这些工作派发给临时子 Agent，以减少复杂 flow 上下文被执行细节淹没。

### 4.7 临时执行子 Agent 负责什么

- 接收一个自包含、边界明确的任务
- 使用对应 skill、工具和上下文完成实际工作
- 输出结果、来源、缺口或完成信号
- 完成后立即结束 Session
- 不理解整个复杂 workflow，也不决定后续派发

临时子 Agent 可以按任务类型拆分，例如：

- Answer Agent：检索证据并回答一个问题
- Verification Agent：验证冲突、数字或来源
- Deep-Dive Agent：深入分析一个已选方法
- Summary Agent：将一组已完成结果整理成指定格式
- Editing Agent：根据明确输入修改一个文件

### 4.8 推荐的复杂 Flow 结构

```text
                      ┌─────────────────────────┐
                      │ 持久调度 Agent          │
                      │ 目标 + 累计结果 + flow  │
                      └────────────┬────────────┘
                                   │ marker: SPAWN_X
                                   ▼
                      ┌─────────────────────────┐
                      │ 调度脚本                │
                      │ 状态管理 + 规则转发     │
                      └───────┬─────────┬───────┘
                              │         │
                    spawn     │         │ spawn
                              ▼         ▼
                    ┌─────────────┐ ┌─────────────┐
                    │ 临时子 Agent A│ │ 临时子 Agent B│
                    │ 实际任务执行 │ │ 实际任务执行 │
                    └──────┬──────┘ └──────┬──────┘
                           │ marker result  │
                           └───────┬────────┘
                                   ▼
                         脚本转发给持久调度 Agent
```

这套结构适合 review、研究探索、复杂内容生成和需要多次验证的分析任务。它同时利用了持久 Session 的累计判断能力与临时子 Agent 的任务隔离能力。

如果某个执行角色确实需要跨轮复用大量上下文，可以保留为持久 Session；但这应是例外。默认优先让调度 Agent 持久，执行 Agent 临时。

---

## 五、类型 3：目标驱动的自主规划 Agent

### 5.1 适用条件

以下任务更适合让 Agent 在长上下文中自行规划：

- debug 一个未知根因
- 阅读代码库并实现一个目标
- 调查性能退化
- 修复测试失败
- 设计方案并根据新证据调整方向

这类任务的目标和验收条件可以明确，但执行路径无法预先写成固定 workflow。

### 5.2 为什么长上下文适合 debug

debug 的关键不是记住“当前处于状态机第几步”，而是持续积累：

- 已观察到的现象
- 已排除的假设
- 命令和测试结果
- 代码结构与依赖
- 当前最可能的根因
- 修改后是否通过验收

这些是业务证据。新证据可能改变原计划，因此 Agent 应拥有重新规划的自由。

### 5.3 推荐 Skill/Prompt 结构

```markdown
# Debug Agent

## 目标
修复 <问题>，直到 <验收条件> 成立。

## 已知现象
- ...

## 约束
- 不修改 ...
- 保持 ...
- 使用现有项目模式

## 可用工具与证据
- ...

## 工作原则
1. 先复现并定位根因
2. 根据证据更新假设
3. 做最小范围修改
4. 运行相关测试
5. 若失败，继续调查而非停在解释

## 完成条件
- ...
```

这里不需要把所有可能路径写成 `§A -> §B -> §C`。可以给一个开放 loop：

```text
while acceptance_not_met:
  inspect current evidence
  choose highest-value next action
  execute
  update hypotheses
  verify
```

状态输出只用于向外部报告阶段性进展，不用于驱动每一个内部动作。

### 5.4 外部脚本的边界

自主规划 Agent 通常只需要较轻的外部控制：

- 提供目标、权限、预算和超时
- 保存日志和最终修改
- 在必要时提供人工审批
- 用测试或验收命令判断是否完成

不要用脚本预先枚举 debug 步骤，否则会限制 Agent 根据新证据改变方向。

---

## 六、三类模式的选择顺序

设计新 Agent 时依次问：

1. **任务能否切成输入输出明确的独立单元？**

   能：使用一次性子 Agent，由 scheduler 管理状态。

2. **后续任务是否只依赖产物，而不依赖 Agent 隐藏上下文？**

   是：继续使用一次性子 Agent。

3. **是否存在需要结合累计结果判断的非线性调度 flow？**

   是：使用持久调度 Agent，用伪代码定义 flow；实际任务优先派发给临时子 Agent。

4. **执行路径是否会因新证据频繁改变？**

   是：使用目标驱动自主规划 Agent。

5. **哪些角色真正需要跨轮保留上下文？**

   默认只让调度 Agent 持久；执行角色只有在跨轮上下文复用收益明显时才保留持久 Session。

简化判断：

```text
可独立验收？
  ├─ 是 -> 一次性子 Agent
  └─ 否
      ├─ 需要非线性调度与多轮收敛 -> 持久调度 Agent + 临时执行子 Agent
      └─ 路径开放、靠新证据推进 -> 自主规划 Agent
```

---

## 七、脚本与 Skill 的统一职责边界

| 内容 | 调度脚本 | 持久调度 Agent / Skill | 临时执行子 Agent / Skill |
|------|----------|------------------------|--------------------------|
| 非线性 flow 和语义分支 | 不判断 | 用伪代码定义并执行 | 不理解全局 flow |
| Agent 运行状态 | 维护 `running/waiting/completed/failed` | 用 marker 声明调度动作 | 用 marker 声明结果 |
| 启动、并行、重试、超时 | 负责 | 只提出派发需求 | 不负责 |
| 输出转发 | 根据 marker 固定映射转发 | 指定需要的输出类型 | 返回指定结果类型 |
| 工具权限与角色隔离 | 负责落实 | 声明调度边界 | 声明并遵守执行边界 |
| 领域判断 | 不做 | 判断下一步和完成条件 | 完成当前局部任务 |
| 专家知识 | 按规则注入 | 决定何时需要 | 在局部任务中使用 |
| durable 产物 | 统一保存或验收 | 生成最终业务内容 | 生成局部结果 |

一个实用原则：

> 持久调度 Agent 决定“语义上下一步做什么”；脚本根据 marker 决定“技术上启动谁、转发给谁”；临时子 Agent 只负责“把当前任务做好”。

---

## 八、从现有实现得到的重构经验

### 8.1 先写协议规格，再改脚本和 Skill

重构复杂 workflow 时，先提取唯一事实来源：

1. 角色之间的交互逻辑
2. 持久调度 Agent 的非线性伪代码 flow
3. 临时子 Agent 的任务类型与执行方法
4. marker 类型与脚本转发规则
5. 专家知识注入逻辑
6. 终止与失败条件

`draft/idea_review_protocol_spec.md` 已包含 Q/A flow、marker 和转发逻辑。后续整理时应继续减少重复叙述，让伪代码主干、marker 契约和脚本规则彼此对应。

### 8.2 区分主干逻辑和细节约束

推荐顺序：

```text
角色边界
-> 非线性伪代码 flow
-> marker 与转发规则
-> 临时子 Agent 任务定义
-> 专家知识/reference
-> 质量约束和自检
```

不要一开始就把大量问题表、格式细节和例外塞在 workflow 前面。对持久调度 Agent 来说，主干伪代码应短、靠前，并通过脚本提醒快速重新定位。

### 8.3 协议修改必须成套更新

任何协议语义变化都应同时更新：

- orchestrator parser/formatter
- 对应 role skills
- protocol spec
- protocol version
- checkpoint resume 校验
- 最小协议测试

否则聊天中“已经约定”的行为不会成为可维护系统的一部分。

### 8.4 产物与运行状态分离

- durable 业务产物：如 `review_notes/<title>_review.md`
- 可恢复运行状态：如隐藏的 `.claude/idea-review-runs/<title>/`
- 原始日志：只用于诊断，不作为最终 review

这能避免临时调度目录变成不可维护的长期输出目录。

---

## 九、当前两个案例的评价

### 9.1 learning scheduler

适合继续采用“一次性子 Agent + 外部状态机”：

- 分层问题、单题回答、水平总结、垂向总结都有明确产物
- 每步可独立重试
- worker pool 和 checkpoint 能真实表达进度
- 后续阶段通过文件消费上游结果

需要长期保持的原则是：产物验证优先于 Agent 自报完成，`running` 不能提前计入 `done`。

当前 skill 中“只读”与“写入指定输出文件”的描述互相冲突，而 scheduler 启动 Agent 时也没有限制其只能写当前产物。后续应把权限契约改成“可读所需输入，只能写指定输出路径”，并在脚本层约束工具，而不是只靠 skill 提醒。

### 9.2 idea review Q/A

当前设计中合理的部分：

- QA/AA 信息隔离明确，blind QA 契约清楚
- 首轮固定总览，再按候选维度追问
- marker 包装的原始 Markdown 比将长 answer 塞入 JSON 稳定
- reference 白名单按需注入
- orchestrator 统一写 review、保存日志并校验协议
- skill 使用伪代码定义非线性 flow，`LOOP` 为脚本提供下一轮提醒
- 脚本按预定规则完成 Q/A 转发，不参与价值判断

当前仍脆弱的部分：

- QA 和 AA 都是持久 Session，AA 的检索与回答细节会持续累积，可能污染复杂 flow 的核心上下文
- 两个持久角色都需要反复定位自身 flow，调度结构比“一个持久调度 Agent + 临时执行 Agent”更重
- skill 较长，主干伪代码和大量约束仍可能互相稀释
- 脚本同时包含 Q/A 固定轮转、reference 注入和协议容错，通用调度规则与该 workflow 的特殊规则耦合
- `§INIT` 行为同时写在 skill 和 `buildQAInitMsg` / `buildAAInitMsg` 的附加提示中，存在双重真值
- `--max-budget-usd` 当前会被解析和展示，但 `spawnSession` 中对应 Claude 参数被注释，运行时并未真正应用该上限
- AA skill 声明不能写文件，但运行时仍允许 `Bash`，且未禁用 `obsidian_delete_note`；只读边界尚未由脚本完整执行

推荐演进方向：

1. 将 QA 明确为持久调度 Agent：保留盲评目标、累计回答、提问 flow 和最终判断。
2. QA 输出 `___SPAWN_ANSWER_AGENT___` 后，由脚本为当前问题启动一个临时 AA。
3. 临时 AA 接收 idea note、当前问题和 Answer skill，完成一次检索与回答后输出 `___ANSWER_RESULT___` 并结束。
4. 脚本只维护 QA 与临时 AA 的运行状态，按 marker 将回答转发回 QA，并附上 `[LOOP: §DIM_EVAL ...]` 提醒。
5. QA 根据伪代码判断继续追问、请求 reference、切换维度或输出最终 judgment。
6. 不增加 JSON state；累计判断保留在 QA 持久 Session，运行记录由脚本保存。
7. 若 AA 跨轮复用检索上下文的收益明显高于上下文污染和调度复杂度，再保留持久 AA 作为特殊配置。
8. 精简两个 skill 主体，将详细专家问题表继续放入 references。
9. 删除初始化行为的重复描述，让 skill 成为角色行为的唯一事实来源。
10. 让预算配置真正传入运行时，收紧 AA 权限，并增加 marker 路由与终止条件测试。

---

## 十、最终编写准则

1. 可拆分任务优先使用一次性子 Agent，不为“像多 Agent”而保留 Session。
2. 复杂 workflow 默认使用一个持久调度 Agent，实际任务优先交给临时执行子 Agent。
3. 非线性 flow 用短伪代码显式定义；脚本通过 `[LOOP: §...]` 提醒持久调度 Agent 继续位置。
4. Marker 只包装派发请求、执行结果、等待和完成信号，不承载完整 JSON state。
5. 脚本维护各 Agent 的运行状态，并根据预定 marker 规则启动和转发，不做领域判断。
6. debug 等开放任务只固定目标、约束和验收，允许 Agent 自行规划。
7. 持久调度 Agent 的 skill 主干要短、靠前、伪代码化；大型知识放 references。
8. 完成必须可由外部验证：DONE marker、合法 judgment、输出文件或测试结果。
9. 协议变化必须同步更新脚本、skills、spec 和 protocol version。
10. 判断复杂 flow 是否清晰：脚本能否只看 marker 就完成调度，而不需要理解 Agent 的领域内容。
