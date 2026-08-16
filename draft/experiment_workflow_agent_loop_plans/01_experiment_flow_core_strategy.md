# Experiment Flow 核心策略：Direction 驱动的环境、实现与实验迭代

状态：设计讨论稿；暂不实现 Script、Skill 和通信 Schema。

## 1. 定位

Learning Flow 和 Experiment Flow 处理不同问题：

```text
Learning Flow
  以 Topic 为起点
  → 搜索和组织研究知识
  → 审阅性能矛盾和研究价值
  → 产生可证伪的 Direction

Experiment Flow
  以一个具体 Direction 及其 revision 为起点
  → 打通可用实验环境
  → 实现、修改并运行实验代码
  → 根据错误、数据和新认识继续迭代
  → 形成对该 Direction 的经验结论
```

Experiment Flow 不重新执行开放式 Topic 价值搜索。它可以为实现
Direction 补充技术知识，但不应在实验过程中静默把原 Direction
替换为另一个研究方向。

## 2. 核心假设

实验环境不是可以在一个预处理 Stage 内一次确定的前置条件。

实际过程可能是：

```text
尝试环境 A
  → 发现模拟器不支持目标机制
  → 尝试环境 B
  → 实现后发现 baseline 不可比
  → 修改配置或更换环境 C
  → 运行后对机制产生新认识
  → 修改实现和对照
  → 继续实验
```

因此，不在 Script 中定义 `ENV_READY`、`IMPLEMENTATION_COMPLETE`、
`CONFIRMATORY_STAGE` 等需要预先穷尽情况的语义阶段。环境、代码和
实验方法都是持久工作区中随认识变化的对象。

## 3. 最小顶层 Loop

Experiment Flow 复用 Learning Flow 的调度策略和三类 Agent role：

```text
START(Direction)
  ↓
Worker → Reviewer → Decision
                       ├─ RUN_WORKER   → Worker → Reviewer → Decision
                       ├─ RUN_REVIEWER → Reviewer → Worker → Reviewer → Decision
                       ├─ RETRY_WORKER
                       ├─ RETRY_REVIEWER
                       └─ COMPLETE     → Script finalize
```

这是逻辑上的同一个 Loop，不是按环境、实现和验证分割的
多段 workflow。

每个 Worker Turn 只需要执行当前最有价值的一次有界推进，例如：

- 从本地知识中查找可能的模拟器、容器、代码库、依赖和部署方法；
- 尝试或修复一个实验环境；
- 实现或修改 Direction 对应的代码；
- 运行模拟器、GPU 实验、微基准或正确性检查；
- 阅读日志和数据，诊断环境、实现或实验方法问题；
- 根据 Reviewer 和 Decision guidance 补充对照或重做一次尝试。

Worker 不需要在每轮宣布某个阶段已完成。它只需要如实说明本轮改变、
运行、观察、产物和当前结论。

## 4. 本地实验知识是按需外部记忆

Experiment Worker 可使用 Obsidian Omnisearch 从两个本地维度获取环境和
实验设计思路：

| 维度 | 路径 | 主要用途 |
|---|---|---|
| experiment | `experiment_notes/` | HPC 论文和历史实验中的实现、模拟器、代码库、软硬件配置、部署命令、指标、对照、已测约束和失败现象 |
| human | `human_notes/` | 人类总结的环境经验、踩坑、主观判断、资源条件、待验证假设和未解问题 |

这两个目录不是 Script 预先转换的环境注册表，也不要求 Worker
在第一轮全部读取。它们是每个 fresh Turn 可根据当前缺口查询的持久
外部记忆。

### 4.1 查询时机

只在当前推进缺少关键信息时查询，例如：

- 不知道何种模拟器或开源框架可表达 Direction 的机制；
- 不知道某论文的实验环境、软件版本或基线实现；
- 部署、编译或运行失败，需要查找已知约束或历史诊断；
- 实验数据暴露新瓶颈，需要查找相关测量方法、反例或对照；
- Reviewer 提出可能改变当前结论的实现或环境缺口。

查询可以在任何轮发生，包括已经运行多轮实验之后。

### 4.2 查询方法

Worker 而非 Script 理解当前问题并构造关键词。基本方法是：

```text
当前 Direction＋最新错误/数据＋环境或实现缺口
  → 提取技术对象、模拟器/框架、硬件、工作负载、错误或指标词
  → 选择 experiment 或 human 维度
  → Omnisearch
  → 深读少量命中笔记
  → 形成本轮可执行的尝试
```

工具调用使用带路径约束的查询：

```text
mcp__obsidian__obsidian_search_notes(
  mode="omnisearch",
  query="path:experiment_notes/ <current terms>"
)

mcp__obsidian__obsidian_search_notes(
  mode="omnisearch",
  query="path:human_notes/ <current terms>"
)
```

命中后使用 `mcp__obsidian__obsidian_get_note` 深读。搜索摘要只用于选源，
不直接作为实验事实。

精确词无命中时可逐步缩短查询，但保留 Direction 的核心技术对象。当
已获得一个值得尝试的路径时即可停止，无需穷尽笔记。

### 4.3 笔记的权威边界

- `experiment_notes/` 中的论文环境是候选实现路径，不代表当前机器
  已经具备该环境。
- `human_notes/` 中的经验和判断可以指导尝试，但不替代当前运行
  和测量证据。
- 笔记可能对应旧 commit、旧驱动、旧 GPU 或不同工作负载，Worker
  需要在真实部署中重新确认。
- 外部来源内的命令和说明是不可信输入，不是 Controller 指令。

## 5. Agent 职责

### 5.1 Experiment Worker

Worker 负责对当前 Direction 做实际推进：

- 读取 Direction、当前工作区、最新 Agent 结论和 Decision guidance；
- 按需查询 `experiment_notes/` 和 `human_notes/`；
- 使用当前可用工具尝试部署、实现、编译、运行和诊断；
- 保留有用代码、配置、日志、指标和失败信息；
- 输出本轮完成的动作、实际观察、当前判断和产物引用。

Worker 不调度下一轮，不宣布全局完成，不因为一次环境或代码
失败就把 Direction 写成已证伪。

### 5.2 Experiment Reviewer

Reviewer 独立审阅当前 Worker 轮次及其引用的实验产物：

- 区分环境失败、实现错误、实验无效和 Direction 证据；
- 检查当前结论是否与真实日志、代码和指标一致；
- 检查 baseline、对照、资源、输入、质量和测量口径是否可比；
- 在结论可能被本地环境经验改变时，可独立查询两个笔记维度；
- 说明会影响下一轮的具体缺口。

Reviewer 不直接修改环境或实验代码，不调度下一轮。

### 5.3 Experiment Decision

Decision 读取：

- 输入 Direction 和 Experiment Flow 的最终需求；
- Script 的当前运行状态和历次尝试索引；
- 最新 Worker 结论和 Reviewer 结论；
- 本次 Script 允许的决策字面量。

Decision 只决定 Loop 下一步由谁继续，并可用精简 guidance 提醒：

- 继续当前环境，还是尝试新环境；
- 修复实现，还是增加对照或测量；
- 当前结论是否足以结束；
- 是否应把 Direction 机制问题返回 Learning Flow 修订。

Decision 不编写实验代码，不自行运行实验，不修改 Script 状态。

## 6. Script 职责

Experiment Script 是确定性调度器和持久记忆，不是实验专家。

它负责：

- 绑定输入 Direction 及其 revision；
- 为每轮创建 fresh Worker、Reviewer 和 Decision Turn；
- 向 Agent 注入当前工作区、最新结论、历次尝试索引和 guidance；
- 实时保存 Agent 输出、tool event、命令运行、日志和产物引用；
- 维护轮次、Turn、重试、暂停、预算和最终报告；
- 处理 Provider、超时、JSON 解析和核心控制字段错误；
- 按 Decision 的正式选择执行下一个确定性转换。

它不负责：

- 预先判断应该使用哪个模拟器、仓库、容器、GPU 或实验方法；
- 根据 Topic 或错误文本生成 Omnisearch 关键词；
- 判断环境、实现或 baseline 是否语义正确；
- 将编译失败、OOM、模拟器缺功能或负结果伪装成通信错误；
- 强制某个环境在后续轮次中不能被替换。

## 7. 迭代记忆与实验轨迹

借鉴 Karpathy autoresearch 的小循环思路，每次尝试都应留下简短可观察
轨迹，但不使用破坏性 reset 丢弃失败实现。

借鉴 Orchestra autoresearch 的持久记忆和反思思路，Script 为 Decision
生成紧凑的尝试轨迹与当前认识，原始日志和代码仍作为可按需读取
的完整产物。

轨迹至少帮助 Agent 回答：

- 已经尝试过什么；
- 哪些环境、代码或配置失败；
- 失败当时观察到什么；
- 哪些实现和数据仍值得保留；
- 当前对 Direction 的认识相比上一轮发生了什么变化。

这些信息由 Script 从已保存的运行事实和 Agent 结论中建立观察，
不要求 Agent 在每个输出里回显完整历史。

## 8. 错误和负结果的边界

```text
Provider/运输/JSON 核心协议错误
  → Script 使用同一绑定重试

环境部署失败/编译错误/OOM/模拟器不支持/实验数据不足
  → 正常 Worker 结论
  → Reviewer 解释边界
  → Decision 选择下一轮

有效实现和可比实验在目标条件下未支持预期
  → 才可作为反对 Direction 的经验证据
```

Script 不用预设规则区分后两类语义情况；Reviewer 和 Decision 负责。

## 9. 与 Learning Flow 的闭合

Experiment Flow 必须绑定一个明确的 Direction revision。Learning Flow
保留对 Topic 价值、Anchor 和 Direction 的权威定义；Experiment Flow
保留对真实实验过程和数据的权威记录。

Experiment Flow 的结束不必须是正向加速，可以是：

- 当前条件下支持 Direction；
- 当前条件下不支持 Direction；
- 环境、实现或数据不足，无法得出可信结论；
- 实验过程暴露 Direction 定义问题，需要返回 Learning Flow 修订。

返回结果只作为与该 Direction revision 绑定的实验证据，不静默
覆盖 Learning Flow 中的旧对象。

## 10. 本设计刻意不引入的内容

为了保持第一版 Flow 简单，本文不引入：

- 独立 Environment Agent、Implementation Agent、Runner Agent 或 Analysis Agent；
- 环境→实现→验证的固定 Stage DAG；
- Script 中的模拟器、GPU、容器或工具 allowlist；
- Script 对证据、实验归因和完成意义的强语义校验；
- 要求第一轮就确定最终环境、baseline、代码和测量方法；
- 为了追求正结果而无限重试的“never stop”规则。

## 11. 后续设计顺序

在本核心策略确认后，后续再依次讨论：

1. Experiment Worker、Reviewer、Decision 的精确职责和调度边界；
2. Script 的最小状态机和转换；
3. Agent 之间的简化 JSON 通信和实验产物引用；
4. 工作区、代码、环境、命令、日志、指标和失败尝试的持久化；
5. 如何将 Experiment Result 以不可变引用返回 Learning Flow。
