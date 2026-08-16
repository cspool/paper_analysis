# Experiment Decision、复现保真度与受控弱化设计

状态：已按本设计实现为 Direction Experiment Loop format v6；承接
`01_experiment_flow_core_strategy.md` 至
`05_bounded_direction_revision_and_learning_handoff_design.md`。活动实现位于
`scripts/direction_experiment_loop/`、`scripts/direction_experiment_loop.ts` 和三类
`direction-*` Skill；旧 format v2-v5 仅保留审计兼容。

## 1. 本次修改解决的问题

当前 Direction Experiment Loop 只有两类 Agent：

```text
Direction Lab Goal
  负责实现和运行实验

Evidence Judge
  同时负责审阅实验、选择下一步和终止整个 Flow
```

这把两个不同问题合并给了同一个 Turn：

1. 当前实验在其声明条件下是否有效；
2. 当前证据是否已经足以完成整个 Direction 的验证。

03“分任务视觉 Token 预算”暴露了这一冲突。Judge 合理地认为合成代理中的
`C0—M1` 配对在 `WEAKENED_PROXY_MECHANISM` 范围内成立，但随后直接输出
`SUPPORT`，Script 因而终止。这个结论没有回答更重要的问题：在 RTX 4090 上使用小型
真实模型和真实数据时，任务信息是否仍然优于长度或廉价输入信息。

问题不在 Controller、JSON 或 GPU，而在于当前 Judge 同时拥有“局部证据审阅权”和
“全局完成决策权”。

## 2. Experiment Flow 的核心目标

Experiment Flow 应围绕一个已冻结的 Direction revision，完成以下闭环：

```text
建立可运行且可追溯的 baseline
  ↓
实现 Direction 声明的唯一主要变化
  ↓
构造最强可信的简单 baseline
  ↓
执行同载体配对 A/B 与必要组件消融
  ↓
根据实验数据判断 Direction 是否具有增量价值
```

环境部署、兼容性修复和代理构建都是实现这个比较的手段，不是 Flow 的最终目标。
不能因为代理代码可以运行，就把实现完成等同于 Direction 获得支持。

## 3. 设计结论：恢复三角色 Loop

恢复 `01_experiment_flow_core_strategy.md` 中 Worker、Reviewer、Decision 分离的核心结构，
但使用实验域专属名称：

```text
Experiment Decision
  ↓ 理解消融验证目的并冻结本轮实验化 Direction 与实验合同
Direction Lab Goal
  ↓ 实现 baseline、variant 和消融并保存产物
Evidence Judge
  ↓ 独立评判本轮证据
Experiment Decision
  ├─ RUN_LAB
  ├─ RUN_JUDGE
  ├─ COMPLETE_SUPPORT
  ├─ COMPLETE_REJECT
  ├─ RETURN_TO_LEARNING
  └─ BLOCKED
```

三者分别对应：

- Direction Lab Goal：执行者；
- Evidence Judge：实验 Reviewer；
- Experiment Decision：工作流 Decision。

Script 仍然只是确定性调度器和持久状态库，不获得实验语义判断能力。

## 4. Experiment Decision

### 4.1 输入

每个 fresh Experiment Decision Turn 读取：

- 原始 Direction、Parent Anchor、来源审阅和证据；
- 原始 Direction revision 和当前实验合同 revision；
- Experiment Flow 的最终验证需求；
- Script 当前状态、revision 和实验轮次；
- 已冻结的历次实验合同；
- Lab 对 baseline、variant、实现、运行和消融的状态摘要、完整产物索引和简短结论；
- Judge 对每轮实验的独立评判、实际证据范围和未覆盖问题；
- 已尝试载体、弱化维度、负结果和外部阻塞；
- 本次 Script 允许的决策字面量。

原始日志、代码和 raw data 保留为按需读取证据，不直接塞入无限增长的 Prompt。Decision
必须能够在 Lab 摘要、Judge 结论或历史状态发生明显冲突时按引用深读关键产物，但不以
逐行代码审查或重新计算统计量代替 Judge。

### 4.2 职责

Experiment Decision 负责：

- 理解“通过 baseline、优化实现和配对消融验证或评判当前 Direction”的最终目的；
- 判断当前最重要的未决实验问题；
- 将原始 Direction 实验化为本轮明确的 baseline、variant、唯一变化和消融关系；
- 选择下一轮需要达到的证据范围；
- 确定本轮实验使用的软硬件类别、模型、数据、负载、规模、trace 和测量边界；
- 决定继续高保真实现，还是授权哪些维度受控弱化，并冻结弱化后的结论边界；
- 决定是否应从代理升级为真实小模型、真实数据或本地单卡性能实验；
- 决定是否需要加入更强的简单 baseline、额外消融或新的输入条件；
- 在不改变原始优化对象和核心因果 lever 的范围内，版本化调整实验化 Direction；
- 检查 Lab 是否回答了冻结合同，以及 Judge 是否完整审阅了实现、比较和证据边界；
- 在 Judge 漏审、结论冲突或只需重新评判现有证据时请求新的 Judge Turn；
- 判断问题属于实验实现、实验合同调整，还是需要返回 Learning Flow；
- 判断局部正负证据是否已经足以终止整个 Direction Experiment Flow。

Experiment Decision 不负责：

- 编写或修改实验代码；
- 执行命令和测量；
- 代替 Judge 审计 raw data、验证统计结果或给单轮实验作专业有效性裁决；
- 直接修改 Script 状态；
- 改变原始 Direction 的优化对象、Parent Anchor 或核心因果 lever；
- 静默放宽质量、正确性、吞吐和公平性约束。

Decision 的检查是元层检查：它判断实验实现和评判是否足以支持下一次调度，而不是重新
执行 Lab 或 Judge。若核心研究主张必须改变，Decision 选择 `RETURN_TO_LEARNING`，不在
Experiment Flow 内自行制造新的 canonical Direction。

### 4.3 最小决策集合

为保持控制面简单，Decision 只需要以下工作流选择：

```text
RUN_LAB
RUN_JUDGE
COMPLETE_SUPPORT
COMPLETE_REJECT
RETURN_TO_LEARNING
BLOCKED
```

`RUN_LAB` 可覆盖继续、修复、增强、降级、换载体或调整同一因果主张的实验化表达，不
需要为这些语义分别增加状态机字面量；区别由 Decision 形成的新实验合同 revision
表达。`RUN_JUDGE` 只重新审阅当前冻结合同或已有实验产物，不启动新的实验。

重试 Provider、运输和核心输出格式错误仍由 Script 处理，不进入 Experiment Decision。

## 5. 版本化实验化 Direction 与冻结实验合同

Experiment Decision 每次选择 `RUN_LAB` 时，先形成一个简洁、不可变的本轮实验合同。
合同也是原始 Direction 在本轮的实验化表达：它可以版本化调整 baseline、variant、
适用条件、载体和消融设计，但不得改变原始优化对象、Parent Anchor、核心因果 lever 或
不可变 guard。合同只冻结科学比较，不冻结所有实现细节。

它至少说明：

- 当前要消除的唯一关键不确定性；
- parent baseline、最近方法 baseline 和最强简单 baseline；
- variant 的唯一主要变化；
- 必须保留的因果接口、触发条件和 guard；
- 本轮目标证据范围；
- 指定或约束本轮软件载体、硬件、模型、数据和实验负载；
- 哪些模型、数据、软件、硬件、规模或 trace 维度允许弱化；
- 哪些维度承载核心主张，禁止弱化；
- 本轮成功、负结果、无效和证据不足分别意味着什么；
- 必须产生的核心产物。

合同不必预先规定所有命令、函数、兼容性 patch 和调试步骤。Decision 可指定必须使用
或优先使用的仓库、软件栈和硬件；未指定到具体实现时，Lab 只能在合同允许的载体类别
和弱化边界内选择最容易执行的实现路径并反复修复。

如果 Lab 发现必须改变 baseline 家族、证据目标或被禁止弱化的核心维度才能继续，应
保存现状并退出，由 Judge 评判，再由下一次 Experiment Decision 形成新合同 revision。
Lab 不得为了完成 Goal 自行改变这些条件。

## 6. Direction Lab Goal

Lab 只执行当前冻结实验合同：

1. 获取、复现或构建 baseline；
2. 验证 baseline 的正确性、来源和适用边界；
3. 在同一载体上实现 Direction 的唯一主要变化；
4. 实现合同指定的最强简单 baseline；
5. 公平校准 baseline，冻结确认性边界；
6. 验证触发、action/state/execution trace 和行为差异；
7. 执行同载体单变量 A/B 和必要组件消融；
8. 保存代码、patch、环境、命令、配置、raw data、统计和失败；
9. 输出最窄实验观察，不选择下一分支和全局结论。

Lab 可在一个 Goal 内反复部署、编译、修改和诊断，但不能自行：

- 从真实模型改成合成代理；
- 从本地性能证据改成机制代理证据；
- 更换合同指定或限制的软件、硬件、模型、数据和实验负载类别；
- 删除合同指定的竞争 baseline；
- 调整实验化 Direction、baseline ladder、variant 定义或核心因果 lever；
- 宣布整个 Flow `SUPPORT` 或 `REJECT`。

Lab 不再具有 `DIRECTION_REVISION` 模式。baseline ladder、实验条件、载体弱化、适用范围
和消融关系的调整由 Experiment Decision 通过新合同 revision 完成；Lab 只负责把已经
确定的合同实现并运行。若调整会改变核心研究主张，必须返回 Learning Flow。

## 7. Evidence Judge

Evidence Judge 只审阅一个已确定实验合同及其 Lab 产物。

它负责判断：

- Lab 是否执行了合同指定的 baseline 和 variant；
- baseline 是否正确、竞争性充分并获得公平校准；
- 组内唯一主要变化是否成立；
- 弱化是否保留了声明所需的因果接口、触发和 guard；
- trace、指标、统计和结论是否与原始产物一致；
- 结果是有效正证据、有效负证据、证据不足还是无效实验；
- 实际证据范围和未解决问题是什么。

当 Experiment Decision 选择 `RUN_JUDGE` 且尚无新 Lab 结果时，Judge 可以只对当前合同、
已有产物或冲突结论进行设计审计和复审。它仍然只输出评判，不定义新的实验条件。

Judge 不再：

- 选择下一 Lab Goal；
- 授权弱化或增强载体；
- 修改实验化 Direction 或原始 Direction；
- 直接终止整个 Flow。

Judge 的概念输出精简为：

```text
assessment = VALID_POSITIVE | VALID_NEGATIVE | INCONCLUSIVE | INVALID
evidence_scope = 当前证据实际覆盖范围
reason = 简短判断和边界
remaining_uncertainty = 会影响最终结论的主要未决项
```

具体 JSON Schema 在后续通信设计中确定。Script 只校验核心字面量和引用，不机械验证
Judge 的科学语义。

## 8. 高保真优先与受控弱化

### 8.1 原则

默认策略不是“先做最便宜的代理”，也不是“必须先完整复现论文环境”，而是：

> 优先选择当前资源下因果保真度最高、且能够实际推进的实现；遇到明确障碍后，再由
> Experiment Decision 授权特定维度的受控弱化。

论文环境可得时，应尽量复现 baseline、优化实现和原始测量。论文环境不可得时，不把
完整复现设为硬门槛，但必须说明弱化原因和结论边界。

### 8.2 保真度不是单一线性等级

Decision 应分别考虑：

- baseline 方法和参数域保真度；
- 优化接口与实现保真度；
- 模型保真度；
- 数据和任务分布保真度；
- 软件栈与系统行为保真度；
- 硬件和拓扑保真度；
- workload、并发和 trace 保真度；
- 指标、质量和统计保真度。

某一维度能否弱化取决于 Direction 的因果主张。例如：

- 控制器状态机机制可以先使用最小本地组件验证；
- 特殊互连或多 GPU 拓扑可以用模拟器表达；
- 真实任务可压缩性本身是主张时，不能用人工预设的 task profile 代替；
- 模型规模可以缩小，但仍应保留真实模型、真实 token 和真实质量指标；
- 数据量可以缩小，但子集必须覆盖关键输入和失败条件。

### 8.3 推荐的尝试方式

根据具体 Direction，Decision 从当前可行的最高保真路径开始，并按证据和失败动态选择：

```text
原论文或官方开源实现
  ↕
本地可运行的真实模型、真实数据和真实 GPU
  ↕
容易修改的开源平替或最小系统组件
  ↕
弱化代理
  ↕
模拟器
```

这不是必须顺序执行的固定 Stage，也不是严格的总排序。硬件机制可能更适合模拟器，
数据依赖机制则应优先保留真实模型和真实数据。

弱代理主要用于：

- 打通接口；
- 验证实现逻辑；
- 检查 trigger 和 trace；
- 排除明显行为等价或实现错误；
- 为更真实实验降低部署风险。

除非最终需求本来只要求机制可行性，否则弱代理的局部正结果不应自动终止 Flow。

## 9. 证据与终止边界

### 9.1 局部评判与全局完成分离

Evidence Judge 可以输出：

```text
VALID_POSITIVE + WEAKENED_PROXY_MECHANISM
```

它只表示代理条件下机制成立。Experiment Decision 必须进一步判断该证据是否满足最终
需求。若 Direction 的价值依赖真实模型、真实任务或真实单卡性能，下一步应升级证据，
不能直接 `COMPLETE_SUPPORT`。

### 9.2 COMPLETE_SUPPORT

只有在以下条件同时成立时使用：

- 决定性 baseline 足够强且公平；
- variant 的独特行为和归因成立；
- 必要正确性、质量、吞吐和资源 guard 通过；
- 当前证据范围已经覆盖 Direction 最终需求中的关键因果维度；
- 继续实验不太可能改变总体判断。

### 9.3 COMPLETE_REJECT

适用于：

- 最强简单 baseline 行为等价或支配 variant；
- 有效同载体实验触发 Direction 的预注册失败条件；
- 同一机制族的多次有效负结果已经收敛；
- 关键增量价值消失，继续只是在寻找有利区间。

环境失败、安装失败、无触发代理或无效实验不能直接构成科学拒绝。

### 9.4 RETURN_TO_LEARNING

当后续推进需要更换优化对象、核心因果 lever、Parent Anchor、信息来源或整体研究主张
时，Experiment Decision 返回 Learning Flow，而不是在实验过程中静默改写 Direction。

## 10. 03 Direction 的预期运行方式

对“分任务视觉 Token 预算开关”，新的 Loop 应按以下方式推进：

```text
Experiment Decision
  形成新的实验合同 revision，并冻结本地真实证据目标：
  - RTX 4090
  - 小型真实 VLM
  - 真实 benchmark 子集与原生质量指标
  - 允许缩小模型、数据量、请求数和 trace
  - 禁止用人工 task profile 代替真实任务异质性
  - baseline 包含 B0、M0、C0、C_length、必要时 C_input、M1
  ↓
Direction Lab Goal
  复现 baseline、实现策略、校准并执行同载体消融
  ↓
Evidence Judge
  判断真实实验是否有效以及 task 信息是否具有超越 length/input 的增量价值
  ↓
Experiment Decision
  ├─ 证据有效且充分 → COMPLETE_SUPPORT / COMPLETE_REJECT
  ├─ 实现或统计不足 → RUN_LAB
  ├─ 当前真实载体不可行 → 授权特定弱化后 RUN_LAB
  ├─ baseline 或条件定义仍不完整 → 形成新实验合同 revision 后 RUN_LAB
  ├─ Judge 漏审或结论冲突 → RUN_JUDGE
  └─ 主张需要改变 → RETURN_TO_LEARNING
```

若先运行合成代理，它只能验证表查询、Top-K、旁路和 prefill 路径是否形成预期 trace。
它不能证明真实任务存在相同预算异质性，也不能自动满足整个 Flow 的终止要求。

## 11. Script 状态机

Script 只实现以下机械转换：

```text
START
  → DECISION

DECISION / RUN_LAB
  → FREEZE_EXPERIMENT_CONTRACT_REVISION
  → LAB
  → JUDGE
  → DECISION

DECISION / RUN_JUDGE
  → JUDGE
  → DECISION

DECISION / COMPLETE_SUPPORT | COMPLETE_REJECT | RETURN_TO_LEARNING
  → FINALIZE

DECISION / BLOCKED
  → PAUSE
```

Script 负责冻结输入、Decision 产生的实验合同 revision、Agent 原始输出、校验结果、事件、
产物引用和状态。
它不判断：

- 哪个载体更忠实；
- 哪个 baseline 最强；
- 弱化是否保留因果逻辑；
- 当前证据是否足以完成；
- Direction 是否具有科研价值。

这些分别由 Experiment Decision、Lab 和 Judge 的语义职责闭合。Script 只机械检查合同
版本、核心引用和状态机字面量，不判断合同调整是否科学正确。

## 12. 与现有 v5 的主要差异

| 当前 v5 | 本设计 |
|---|---|
| Judge 同时审阅并调度 | Judge 只审阅，Experiment Decision 独立调度 |
| `SUPPORT + 弱代理范围` 可直接结束 | 弱代理只形成局部评判，由 Decision 判断是否满足最终需求 |
| Lab 可自行选择最小载体或在 Revision 模式调整实验定义 | Decision 冻结实验化 Direction、目标范围、载体和允许弱化维度，Lab 只实现 |
| “允许更小模型”只是通用许可 | 每轮明确哪些维度可以弱化、哪些承载核心主张 |
| 最强简单 baseline 主要由一次 Judge 判断 | Decision 根据 Judge 发现形成新的版本化实验合同 |
| Judge 审阅不完整时只能继续 Lab 或结束 | Decision 可用 `RUN_JUDGE` 对现有合同和证据重新审阅 |
| 两角色 Loop | Lab、Judge、Decision 三角色 Loop |

## 13. 本设计不引入的内容

为保持主干简单，不增加：

- Environment Agent、Implementation Agent、Runner Agent 或 Analysis Agent；
- 环境准备、baseline 完成、确认性实验等固定 Stage DAG；
- Controller 的实验语义校验；
- 必须完整复现论文环境的硬门槛；
- 代理、单卡、模拟器之间机械固定的升级顺序；
- 为寻找正结果而无限修改 Direction 或重复实验。

## 14. 后续设计顺序

若本策略确认，后续依次设计：

1. Experiment Decision Skill 的消融验证目标、实验化 Direction、输入、方法和边界；
2. Evidence Judge 从“审阅＋调度”改为只读实验评判；
3. Direction Lab Goal 移除 Direction revision 和载体弱化决策，只执行冻结合同；
4. 三类 Agent 的最小消息协议；
5. Script 三角色状态机、实验合同 revision、快照和 lineage 持久化；
6. 旧 v5 运行的 audit-only 保留与新格式迁移；
7. 用 03 Direction 验证真实小模型、真实数据、length/input baseline 和弱代理非终态规则。
