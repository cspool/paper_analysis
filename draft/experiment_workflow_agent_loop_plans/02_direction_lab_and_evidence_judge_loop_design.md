# Direction Lab Goal 与 Evidence Judge：双角色实验闭环设计

状态：已实现为 Direction Experiment Loop format v3；承接
`01_experiment_flow_core_strategy.md`。实现入口为
`scripts/direction_experiment_loop.ts`，两个 Skill 为
`.codex/skills/direction-lab-goal/` 与
`.codex/skills/direction-evidence-judge/`。

## 1. 设计结论

针对已经由 Learning Flow 形成并审阅过的 Direction，不再复用 Learning
Flow 的发现型角色和 `Topic → Anchor → Direction` 状态机。

新的实验闭环只使用两个实验域专属角色：

```text
Direction Lab Goal
  负责知识调研、环境、代码、运行、诊断和实验产物

Evidence Judge
  负责冻结下一实验目标、独立解释证据和决定闭环结果
```

对应 Skill 建议命名为：

```text
direction-lab-goal
direction-evidence-judge
```

这些名字不与 Learning Flow 的 Agent role 重合。第一版不增加独立的环境、
实现、检索、分析、审阅或调度 Agent。

## 2. Flow 的固定输入和目标

每次运行只绑定一个既有 Direction revision：

```text
direction_result_ref
direction_revision
parent_anchor_ref
source_review_ref
```

### 2.1 可读取的 Direction 目标包

上述四个引用不能只作为控制字段存在。Script 在初始化时必须把它们解析为
Lab Goal 和 Judge 都能直接读取的冻结目标包：

```text
inputs/
  direction_result.json          # 原始 Direction WORK_RESULT，权威语义输入
  parent_anchor_result.json       # 父 Anchor 的场景、baseline、性能矛盾和 6L 空间
  source_review_result.json       # Direction 最近一次独立审阅
  direction_target.md             # Script 对已知字段做的无解释可读投影
  evidence_manifest.json          # Direction/Anchor 的证据引用、用途和解析状态
  source_run.json                 # 来源运行及 Skill/Ref 版本
```

`direction_target.md` 不是新的 Agent 总结，也不改变原对象。它只把原始 JSON 的
既有字段按固定顺序展开，方便 Agent 在开始环境或代码工作前确认目标。原始
`direction_result.json` 始终是最终权威；投影与原始字段冲突时以原始 JSON 为准。

### 2.2 必须暴露的 Direction 内容

目标包至少让两个 Agent 能读取下列内容：

| 要理解的问题 | 读取内容 | 主要来源 |
|---|---|---|
| 正在优化什么 | Direction 名称、机制、目标指标、适用条件 | `content.name`、`content.mechanism`、`content.expectedEffects` |
| 与什么比较 | 执行 baseline、最近方法 baseline、唯一主要变化、不变量 | 父 Anchor `baseline`、Direction `baselineChange` |
| 优化位于哪里 | L1–L6 中的技术对象、执行事件、接口和边界 | 父 Anchor `scope6L` 及 Direction 的机制、实现和测量描述 |
| 为什么可能有效 | 已观察的性能 headroom、瓶颈资源、因果假设 | 父 Anchor `performanceTension`、Direction `mechanism` |
| 预期改变什么 | 延迟、吞吐、质量、资源和其他结果的方向与条件 | `expectedEffects` |
| 不能牺牲什么 | 质量、吞吐、公平性、资源和范围约束 | 父 Anchor `constraints`、Direction `tradeoffs` |
| 什么会否证它 | 预注册失败条件、退化边界和反例 | `failureConditions` |
| 应怎样测量 | baseline 复现、A/B、消融、指标、统计和环境计划 | `measurementPlan` |
| 当前审阅认可什么 | PASS/REVISE/REJECT、发现和未闭合缺口 | `source_review_result.json` |
| 每个判断依据什么 | `sourceRef`、`supports`、来源类别及可读取位置 | Direction/Anchor `evidence[]`、`evidence_manifest.json` |

`direction_target.md` 使用固定模板呈现这些内容：

```markdown
# Frozen Direction Target

## Identity
- Direction name / revision / source ref / parent Anchor

## Optimization direction
- Target performance problem
- Proposed mechanism
- Target metrics and applicable conditions

## Baselines and frozen change
- Parent execution baseline
- Nearest method baseline
- Unique baseline change
- Variables and interfaces that must remain unchanged

## Optimization layers and concrete objects
- L1 ...
- L2 ...
- L3 ...
- L4 ...
- L5 ...
- L6 ...
- Cross-layer interfaces and execution events

## Evidence-backed headroom and causal hypothesis
- Observed source facts
- Direction hypothesis that remains to be tested

## Expected effects, constraints and tradeoffs

## Failure conditions and counterexamples

## Measurement and ablation plan

## Source review
- Verdict / findings / query gaps

## Declared evidence
- [owner] sourceRef — supports
```

其中 L1–L6 直接展开父 Anchor 的 `scope6L`；不存在的层写为
`NOT_DECLARED`。Script 不根据自然语言关键词自动给机制分层。Direction 中明确
写出的跨层接口和事件只按原文列出，具体哪些层是本轮实验的主要可修改层，由
Judge 和 Lab 在阅读证据后理解。

若某个字段在源对象中不存在，Script 在投影中标为 `NOT_DECLARED`，不能自行补写。
字段缺失是否会阻止实验，由 Judge 判断。

### 2.3 证据清单与可追溯读取

`evidence_manifest.json` 对父 Anchor 和 Direction 的全部 `evidence[]` 做机械索引，
每项至少包含：

```json
{
  "owner": "ANCHOR | DIRECTION",
  "sourceRef": "原始来源引用",
  "supports": "源对象声明该来源支持什么",
  "resolvedPath": "可解析时填写本地路径",
  "sourceUnit": "章节或块定位",
  "sha256": "可读取本地文件的哈希",
  "resolution": "RESOLVED | UNRESOLVED"
}
```

Script 只做路径解析、哈希和索引，不判断证据是否真的支持主张。Lab Goal 和 Judge
应通过 `sourceRef` 深读相应来源单元，并明确区分：

```text
来源直接报告的 baseline、环境和测量
来源支持但尚未在当前机器复现的事实
Direction 作者提出的机制假设
当前实验新产生的观察
```

来源无法解析时不能静默丢弃。它保留为 `UNRESOLVED`，由 Judge 判断是非阻断的
引用问题，还是会使 baseline、优化层次或 headroom 无法确认的关键缺口。

绑定后不可由 Agent 静默替换。Flow 的目标是：

1. 找到或重建可比的 baseline；
2. 实现 Direction 中冻结的主要变化；
3. 通过 A/B、消融和诊断实验识别真正有效的优化点；
4. 判断当前条件下证据支持、不支持还是无法判断该 Direction；
5. 把实验结论和不可变产物引用返回 Learning Flow。

该 Flow 不创建新 Anchor，不扩展 Topic 的 6L 空间，也不把实验中临时想到的
新机制直接写成新的 Direction。若实验暴露 Direction 定义需要实质改变，则结束
当前绑定并返回修订建议。

## 3. 最小状态机

```text
START(frozen Direction)
          ↓
    Evidence Judge
          │
          ├─ RUN_LAB ─→ Direction Lab Goal ─┐
          │                                  │
          └──────────────────────────────────┘

          ├─ SUPPORT          → FINISHED
          ├─ REJECT           → FINISHED
          ├─ REVISE_DIRECTION → FINISHED_WITH_HANDOFF
          └─ BLOCKED          → PAUSED
```

Judge 每次既审阅已有证据，也在仍需推进时定义唯一的下一次 Lab Goal。因此，
不再设置相互独立的 Planner、Reviewer 和 Decision Turn。

Script 的可执行节点只有：

```text
JUDGE
LAB_GOAL
PAUSED
FINISHED
```

## 4. Direction Lab Goal

### 4.1 职责

Direction Lab Goal 是一次持久、面向实际工作区的实验推进。它可以在一个 Goal
内部反复尝试和修复，直到完成 Judge 冻结的目标或遇到真实阻塞。

它负责：

- 先读取冻结目标包，明确优化方向、baseline、主要变化、6L 对象与接口、
  约束、失败条件及其对应证据；
- 读取历史实验轨迹和本轮 `labGoal`；
- 深读 `evidence_manifest.json` 中与本轮环境、baseline、实现或测量直接相关的
  来源单元，不把 `supports` 摘要当成已复现事实；
- 按需调研本地现有实验、专家经验、代码入口、模拟器和部署方法；
- 获取、部署、重建或修改实验环境；
- 复现 baseline，并保存复现偏差和适用边界；
- 实现 Direction 的主要变化及必要测量插桩；
- 执行 A/B、消融、敏感性、正确性和失败边界实验；
- 根据错误、日志和新数据修改代码、配置或实验方法；
- 保存命令、配置、源码、日志、原始指标和分析报告；
- 如实报告完成项、观测、失败和局限。

它不负责：

- 判断整个 Direction 已经得到支持或被否证；
- 选择闭环的下一角色；
- 修改冻结 Direction 的因果接口；
- 为了得到正结果而事后改变成功标准；
- 在当前 Flow 内创建新 Anchor 或新 Direction。

### 4.2 一个 Goal 可以完成的工作

Judge 冻结的 `labGoal` 每次只针对当前最有信息价值的一次推进，例如：

- 查清最近 baseline 的仓库、commit、环境和测量方法；
- 在当前机器上打通或重建 baseline；
- 修复一次编译、依赖、OOM、模拟器能力或数据问题；
- 实现主要变化并通过正确性检查；
- 在冻结配置下完成一次 baseline/variant 配对；
- 运行一个区分机制解释的消融；
- 对异常结果增加最小诊断；
- 在新的负载或边界条件下检验收益是否消失。

环境调研、环境部署、实现和测量不是固定 Stage。若运行数据改变了对环境或
机制的认识，后续 Judge 可以重新定义新的 `labGoal`。

## 5. 本地实验与专家知识调研

01 中 Experiment Worker 的本地知识调研职责，全部并入 Direction Lab Goal，
不新增 Knowledge Agent。

### 5.1 两个外部记忆维度

| 维度 | 路径 | 用途 |
|---|---|---|
| experiment | `experiment_notes/` | 论文实验环境、模拟器、代码库、版本、部署配置、baseline、指标、消融方法、资源条件、已知失败和复现入口 |
| human | `human_notes/` | 人类专家经验、踩坑、环境限制、实现判断、资源现实、诊断思路、待验证假设和未解问题 |

它们是按需读取的外部记忆，不是 Script 预先构造的环境注册表，也不要求第一轮
穷尽搜索。

### 5.2 查询触发条件

Lab Goal 可在任何时点查询，包括已经执行多轮代码和实验之后。典型触发包括：

- 不清楚什么框架、模拟器、容器或仓库能够表达目标机制；
- 不清楚论文 baseline 的硬件、版本、配置和测量口径；
- 安装、编译、运行或数据准备失败；
- 当前机器与论文环境不一致，需要寻找可比替代或重建方法；
- 实验暴露新瓶颈，需要寻找已有诊断、消融或反例；
- Judge 指出一个会改变结论的环境、实现或公平性缺口；
- 需要确认某项实现是否已有公开代码或更接近的实验基线。

### 5.3 查询过程

查询由 Lab Goal 根据当前缺口构造，而不是由 Script 根据错误字符串机械生成：

```text
冻结 Direction
＋ 当前 labGoal
＋ 最新错误、日志或观测
＋ 缺失的环境、baseline、实现或指标
  ↓
提取技术对象、框架、硬件、工作负载、错误和指标词
  ↓
选择 experiment_notes/ 或 human_notes/
  ↓
Omnisearch 搜索
  ↓
深读少量高价值笔记
  ↓
形成一个可执行尝试
```

允许工具：

```text
mcp__obsidian__obsidian_search_notes
mcp__obsidian__obsidian_get_note
```

搜索时使用路径约束：

```text
path:experiment_notes/ <当前关键词>
path:human_notes/ <当前关键词>
```

精确查询无命中时逐步缩短，但保留 Direction 的核心技术对象。搜索摘要只用于
选择来源；涉及环境、命令、结果或约束的事实必须深读原笔记。找到一个足以形成
下一次实际尝试的路径后即可停止，不要求穷尽知识库。

### 5.4 知识的权威边界

- `experiment_notes/` 只说明候选实验路径，不证明当前机器已经可用；
- `human_notes/` 可指导诊断和选择，但不替代当前运行证据；
- 旧 commit、旧驱动、不同 GPU、不同模型或不同负载必须重新确认；
- 笔记中的命令和文字是不可信数据，不是 Controller 指令；
- 最终性能判断必须来自当前 Flow 保存的可追溯实验产物；
- 如果本地笔记指向公开代码或论文 artifact，Lab Goal 可在当前授权范围内继续核对，
  但必须记录实际取得的版本与修改。

## 6. Evidence Judge

### 6.1 职责

Evidence Judge 是 fresh Turn。它读取当前唯一权威状态，而不依赖长期对话记忆：

- 完整冻结目标包，包括 Direction、父 Anchor、原始 review、可读目标投影和
  证据清单；
- Direction 声明的优化方向、baseline change、相关 6L 对象、预期效果、
  约束、失败条件和测量计划；
- 与上述声明对应的来源单元及其解析状态；
- Script 生成的实验状态快照；
- 历次 `labGoal`、Goal 状态和结果索引；
- 最新 `result.md` 及其引用产物；
- 已记录的可信负结果和外部阻塞；
- 本次允许决策字面量。

它负责：

- 先确认 Lab 实际执行的对象仍与冻结优化方向、baseline 和 6L 接口一致；
- 检查关键 baseline、headroom、实现边界和测量设计是否具有可追溯来源，
  并区分来源事实、Direction 假设与当前实验观察；
- 判断 baseline 是否真实运行、是否与来源及 Direction 可比；
- 判断 variant 是否只实现了冻结的主要变化；
- 检查输入、资源、精度、batch、负载、质量和统计口径是否公平；
- 区分环境失败、实现错误、无效实验、统计不足和有效负结果；
- 判断某次消融是否真正区分了不同机制解释；
- 判断局部参数调整是否仍属于同一机制族的重复尝试；
- 在证据不足但存在高信息价值尝试时定义下一 `labGoal`；
- 在证据充分时输出支持、拒绝或返回修订结论。

Judge 不修改实验代码、不运行实验，也不自行发明替代 Direction。

### 6.2 Judge 对知识库的使用

知识调研的主要执行者是 Lab Goal。Judge 只在以下情况下独立查询
`experiment_notes/` 或 `human_notes/`：

- Lab 声称某环境或 baseline 不可获得，但该判断可能改变 `BLOCKED`；
- 某种比较口径、参考实现或已知反例会改变实验有效性；
- Lab 的来源引用无法支撑其环境或方法判断；
- 需要确认下一实验是否只是已经失败机制的重复包装。

Judge 的查询是审计，不是重新进行开放式研究搜索。

### 6.3 最小输出与证据级别

Judge 输出保留四个核心字段：

```json
{
  "decision": "RUN_LAB | SUPPORT | REJECT | REVISE_DIRECTION | BLOCKED",
  "evidenceScope": "WEAKENED_PROXY_MECHANISM | LOCAL_SINGLE_GPU_PERFORMANCE | SIMULATED_HARDWARE_MECHANISM | PAPER_EXTERNAL_VALIDITY",
  "reason": "对当前证据和边界的简短判断",
  "labGoal": "仅在 RUN_LAB 时填写的下一次有界实验目标"
}
```

`RUN_LAB` 时，`evidenceScope` 是下一次实验的目标级别；其他决策中，它是当前判断
实际覆盖的级别。这样 `SUPPORT + WEAKENED_PROXY_MECHANISM` 不会被机器误读为真实
单卡或完整论文环境已经验证。

`labGoal` 是 Script 冻结并原样传给 Direction Lab Goal 的简短语义合同。它应在
一段话中说明：

- 当前要消除的唯一关键不确定性；
- baseline 与 variant 的比较边界；
- 需要产生的核心证据；
- 什么观察意味着完成、失败或仍不可判断。

Script 只检查 `decision`、`evidenceScope` 字面量以及 `RUN_LAB` 时 `labGoal` 非空，
不机械校验实验语义。

## 7. Baseline、消融与优化点的推进逻辑

Flow 不强制固定阶段，但 Judge 通常按证据依赖选择下一目标：

```text
确认最近 baseline 与可复用环境
  ↓
复现 baseline 的关键趋势和正确性
  ↓
实现冻结的主要变化
  ↓
执行最小 A/B
  ↓
用消融区分收益来源、开销和失败边界
  ↓
必要时在相同因果接口内优化实现或参数
  ↓
形成条件化结论
```

这只是常见依赖关系，不是 Script Stage。Judge 可以根据真实进展跳过、返回或重排。

允许在当前 Direction 内尝试：

- 实现正确性修复；
- 不改变主要机制的工程优化；
- 预先定义的参数或策略变体；
- 用于归因的负控制和消融；
- 能暴露适用边界的负载、规模和硬件检查。

以下变化应输出 `REVISE_DIRECTION`，而不是在当前 Flow 内静默继续：

- 更换主要因果杠杆；
- 更换不可比 baseline 以制造收益；
- 放宽质量、吞吐或正确性约束；
- 根据结果事后重写主要成功标准；
- 将原联合包拆换成实质不同的新方法；
- 把当前负结果包装成另一个名称继续实验。

## 8. Direction Lab Goal 的产物合同

Lab Goal 不需要输出复杂 JSON。官方 Goal 状态与工作区产物共同构成运行事实。

建议工作区：

```text
workspace/
  direction/        # 冻结 Direction、父 Anchor、原 review、可读目标与证据清单
  knowledge/        # 选中的 experiment/human 笔记引用与适用性说明
  env/              # 环境清单、版本、容器和部署记录
  code/             # baseline、variant、patch 和插桩
  configs/          # 冻结配置与实验 manifest
  commands.md       # 可重放命令和失败历史
  raw/              # 原始日志、指标和输出
  analysis/         # 分析、图表和统计结果
  cycles/<n>/result.md
```

每个 `result.md` 至少记录：

- 本轮实际执行了什么；
- 实际使用的环境和版本；
- baseline 与 variant 的实际边界；
- 运行、正确性和测量结果；
- 失败、异常和未完成项；
- 当前能支持的最窄结论；
- 对应原始产物路径。

Script 不从这些自然语言内容推导科学结论，只保存和索引，供 Judge 阅读。

## 9. Script 的最小职责

新的脚本建议独立于 `simple_semantic_loop.ts`，例如：

```text
scripts/direction_experiment_loop.ts
```

Script 负责：

- 通过显式参数导入并快照一个 Direction result；
- 解析和快照其父 Anchor、原 review 与来源运行；
- 从既有字段机械生成 `direction_target.md`，完整呈现优化方向、baseline、
  baseline change、6L 内容、约束、失败条件和测量计划；
- 索引父 Anchor 与 Direction 的全部 `evidence[]`，生成
  `evidence_manifest.json`，保留 `sourceRef`、`supports`、解析路径和哈希；
- 把冻结目标包路径同时注入每次 Judge Turn 和 Lab Goal；
- 初始化共享实验工作区；
- 在 `JUDGE ↔ LAB_GOAL` 之间做确定性转换；
- 冻结每次 Judge 输入、决策和下一 `labGoal`；
- 启动一个 Goal 完成该 `labGoal`；
- 保存 Goal 状态、输出、tool event、命令轨迹和产物引用；
- 生成紧凑实验历史供下一 Judge Turn 阅读；
- 处理 Provider、超时、核心 JSON 和恢复问题；
- 在终态生成不可变实验报告及返回 Learning Flow 的 handoff。

Script 不负责：

- 根据关键词推断优化层次或替 Direction 补全缺失字段；
- 判断某条来源是否真的支持 Direction 声明；
- 判断 baseline 是否有效；
- 选择模拟器、仓库、容器、GPU、指标或统计方法；
- 生成 Omnisearch 查询；
- 区分环境失败与科学负结果；
- 判定两个实现是否属于同一机制族；
- 因为 Goal 状态为 `complete` 就认定 Direction 得到支持；
- 要求存在一个 active Anchor 才允许结束。

## 10. 持久状态和历史

Script 保存的最小控制状态：

```json
{
  "directionRef": "...",
  "directionHash": "...",
  "directionTargetRef": "inputs/direction_target.md",
  "evidenceManifestRef": "inputs/evidence_manifest.json",
  "lifecycle": "RUNNING | PAUSED | FINISHED",
  "node": "JUDGE | LAB_GOAL | null",
  "cycle": 3,
  "activeLabGoalRef": "...",
  "latestLabResultRef": "...",
  "finalDecision": null,
  "evidenceScope": null
}
```

另保存 append-only 历史：

```text
cycle
judgeDecision
evidenceScope
labGoalRef
goalStatus
resultRef
elapsed/usage
artifactRefs
```

紧凑历史由 Script 从事实和已有结论投影，不要求 Agent 在每轮重复完整历史。
原始日志、代码和测量仍通过路径按需读取。

`state.evidenceScope` 始终是最近一次已接受 Judge 决策的范围：若该决策为
`RUN_LAB`，它是当前实验目标；若为其他决策，它是当前判断边界。初始化前为
`null`，完成时自然成为最终范围，无需再维护重复的终态 scope 字段。暂停和失败的
stdout/outcome 也保留该值，便于外部调度器在不解释 reason 的情况下识别当前边界。

## 11. 完成与收敛

### SUPPORT

满足冻结 Direction 的关键成功条件，并且 baseline、主要变化、质量和资源比较可比。
结论必须限定在 `evidenceScope` 和实际测量条件内。

### REJECT

有效实现和可比实验命中核心失败条件，或同一机制族经过实质变化后仍重复失败。
环境失败、代码错误和数据不足不能单独作为 REJECT。

### REVISE_DIRECTION

实验显示原 Direction 的 baseline、主要变化、适用边界或成功条件本身需要实质改写。
当前 Flow 结束并把修订依据交回 Learning Flow。

### BLOCKED

存在需要用户授权、外部数据、硬件、账号、不可得代码或其他外部条件的明确阻塞。
Script 暂停；外部条件解决后从 fresh Judge Turn 恢复。

### 防止局部实验循环

Judge 必须比较历史尝试的：

- baseline；
- 主要因果杠杆；
- 实现变化；
- 测量边界；
- 已命中的失败条件。

只有新实验能够区分不同解释或实质改变最终判断时，才再次输出 `RUN_LAB`。
仅调阈值、改随机种子、替换相邻分数、重命名同一方案或增加无判别力样本，不构成
新的实验理由。

## 12. 返回 Learning Flow 的 handoff

终态输出建议为：

```json
{
  "directionRef": "...",
  "directionHash": "...",
  "directionTargetRef": "inputs/direction_target.md",
  "sourceEvidenceManifestRef": "inputs/evidence_manifest.json",
  "outcome": "SUPPORTED | NOT_SUPPORTED | REVISION_REQUIRED",
  "evidenceScope": "WEAKENED_PROXY_MECHANISM | LOCAL_SINGLE_GPU_PERFORMANCE | SIMULATED_HARDWARE_MECHANISM | PAPER_EXTERNAL_VALIDITY",
  "conditions": ["实际适用边界"],
  "summary": "最窄可信结论",
  "baselineRef": "...",
  "experimentResultRefs": ["..."],
  "finalJudgeRef": "...",
  "reportRef": "final/report.md"
}
```

`BLOCKED` 时不生成伪终态 handoff，只保留暂停状态和阻塞说明。

Learning Flow 可以把该 handoff 作为后续审阅证据，但不会由实验脚本直接覆盖原
Direction revision。

## 13. 与 01 的关系

02 保留 01 的以下原则：

- Direction revision 是不可变输入；
- 环境、实现和实验理解可以随数据动态变化；
- `experiment_notes/` 与 `human_notes/` 是按需外部记忆；
- Script 只负责确定性调度、持久化和核心协议；
- 环境失败、实现错误与科学负结果由智能 Agent 区分；
- 失败尝试应保留为可复用轨迹；
- 实验结果以不可变引用返回 Learning Flow。

02 对 01 的主要简化是：

```text
原设计：Experiment Worker → Experiment Reviewer → Experiment Decision

新设计：Direction Lab Goal ↔ Evidence Judge
```

缩减理由是当前输入已经是冻结 Direction，不再需要发现型工作流的通用角色分工。
执行与判断仍然分离；Judge 在冻结下一 Goal 后不参与实现，Lab Goal 也不判断全局
结论，因此双角色已能维持最小必要独立性。

## 14. 第一版刻意不加入

- 不加入独立 Knowledge Scout；
- 不加入独立 Environment、Implementation、Runner 或 Analyst；
- 不加入固定环境→实现→消融→确认 Stage DAG；
- 不让 Script 维护工具 allowlist 或实验语义规则；
- 不要求 Lab Goal 输出庞大 JSON；
- 不自动并行运行多个 Direction；
- 不允许当前实验 Flow 自行生成替代 Direction；
- 不把一次 Goal `complete` 当作科学完成；
- 不为了寻找正结果无限运行。

## 15. 后续设计顺序

本方案确认后，建议依次设计：

1. `direction-evidence-judge` Skill 的输入、方法和四字段输出；
2. `direction-lab-goal` Skill 的知识检索、工作区和结果报告方法；
3. `direction_experiment_loop.ts` 的最小状态机与恢复语义；
4. Direction 导入、父 Anchor/review 解析和不可变快照格式；
5. `direction_target.md` 与 `evidence_manifest.json` 的机械投影 Schema；
6. Lab workspace、cycle history 和最终 handoff 的具体文件合同；
7. 用第 01 个缓存感知 DAG Slack Direction 的现有负实验记录做迁移测试。
