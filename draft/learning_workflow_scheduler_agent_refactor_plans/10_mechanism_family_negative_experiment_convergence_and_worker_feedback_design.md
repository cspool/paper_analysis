# 同机制族负实验收敛与 Worker 反馈设计

> 状态：已实施并通过本地合同、E2E、恢复和 timeout 测试。本文记录 2026-08-03
> 对连续 EXP 运行的复盘、设计约束和对应实现。  
> 前置设计：`05_minimal_controller_validation_and_semantic_handoff_design.md`、
> `06_outer_loop_memory_trajectory_and_atomic_direction_design.md`、
> `09_concrete_6l_semantics_and_reviewer_value_integration_design.md`。  
> 核心边界：Script 不自行理解机制族、实验结论或研究价值；Reviewer 负责局部
> 证据归因，Decision 负责全局收敛，Worker 负责在负结果约束下继续探索。

## 1. 目的

09 已经把 EXP Goal 从异常兜底提升为由 Decision 选择的信息增益分支，但真实运行
暴露了新的外循环问题：当一个 Direction 被实验否证后，正常闭环可能不断创建同一
高层因果机制的替代实现，再为每个替代实现启动新 EXP。

本设计同时解决两个问题：

1. 避免 Learning Flow 在一个局部机制族中反复生成候选并连续实验；
2. 把可信负结果转化为后续 Worker 可见、可复用的研究约束，而不是只保存在某个
   EXP workspace 或一次 Decision 上下文中。

目标不是禁止失败实验。负结果是 Learning Flow 的重要产出；问题在于负结果必须
改变后续搜索分布和 Workflow 选择，而不能只导致“换一个信号再试一次”。

## 2. 真实运行暴露的行为

当前 continuation：

```text
learning_outputs_codex/
  multimodal_inference_latency_first_v8_exp_authorized_retry_20260803
```

共保存四条 EXP 记录：

| 记录 | 实验对象 | 状态与结论 |
|---|---|---|
| 继承记录 | `alpha=1.03` PPL 响应门 | `budgetLimited`；Hugging Face 授权阻塞，无科学结论 |
| 新 EXP 1 | 授权后的 `alpha=1.03` PPL 响应门 | 候选成本更高且判别近随机，否证当前 Direction |
| 新 EXP 2 | 相邻视觉 embedding 余弦响应门 | 成本显著更低，但判别近随机且特异度不足，否证当前 Direction |
| 新 EXP 3 | 上下文隐藏状态残差线性响应头 | 召回率、特异度和 balanced accuracy 未达到预注册下限，否证当前 Direction |

三个新 EXP 不是相同命令的机械重复，但都围绕同一个高层性能杠杆：在冻结
VideoLLM-online 主路径、上下文和 KV 行为基本不变时，用一个更廉价的逐帧响应门
替代 EOS 判定，以跳过静默帧上的完整解码。

当前状态机形成了以下链路：

```text
EXP 否证当前 Direction
  ↓
Decision 选择 RUN_REVIEWER
  ↓
Reviewer REJECT 当前 Direction
  ↓
该 active Anchor 不再拥有可接受 Direction
  ↓
Script 为闭合“每个最终 Anchor 至少一个 Direction”安排 Worker
  ↓
Worker 创建同一 Anchor 下的替代门信号
  ↓
Reviewer 记录 experiment query gap
  ↓
Decision 再次选择 RUN_EXP_GOAL
```

因此，这不是 Provider/runtime 重试错误，也不是 Script 把一个 Goal 的多个 Turn
误计为多个实验。它是现有 Requirement、`RUN_REVIEWER` 固定序列、Reviewer
实验缺口和 Decision 信息增益规则共同造成的局部搜索循环。

`--max-exp-goals 5` 只是全局机械上限，不是同机制族收敛策略。它可以限制最坏
次数，但不能判断第五次实验是否仍然只是局部重复。

## 3. 保持不变的架构原则

- 不增加新的顶层 Agent 类型；
- EXP Goal 仍只执行一个由 Decision 给出的有界实验目标；
- Worker、Reviewer、Decision 仍使用各自现有角色，不把实验实现并入普通 Turn；
- Script 仍是状态机、持久状态库、引用索引和执行器；
- Script 不通过关键词、embedding、规则树或 JSON 字段自行判断两个 Direction
  是否属于同一机制族；
- Agent 原始输出和完整实验 workspace 继续不可变保存；
- 全局 `maxExperimentGoals` 继续作为用户授权和成本边界；
- Script 的核心 Gate 仍只覆盖状态机所需字面量、消息边界和必要引用，不扩大为
  专业实验审查器；
- 负实验只在 Reviewer 独立核对后影响 Learning 对象和搜索约束；EXP Goal 自己
  不能宣布某个 Anchor、Direction 或机制族已被全局否证。

## 4. 同机制族的语义定义

机制族不能按技术名、字段名或使用的特征划分。应比较三个语义要素：

1. **baseline change**：替换、删除、重排或新增了 baseline 的哪个执行环节；
2. **causal lever**：预期通过减少什么计算、等待、通信、数据移动或资源竞争改变
   Goal 指标；
3. **preserved boundary**：模型、执行路径、状态更新、工作负载和硬件边界中哪些
   关键部分保持不变。

当这三项基本相同时，仅改变以下内容通常仍属于同一机制族：

- 阈值或超参数；
- PPL、余弦、距离、熵等打分函数；
- 从一个冻结中间特征换成另一个冻结中间特征；
- 把固定阈值换成小型分类头，但仍控制同一接口并依赖同一延迟假设；
- 对同一局部机制换模型名称、数据子集或包装代码。

当前运行可作如下分层：

```text
Anchor
  在线流式视频中逐帧 EOS 判定带来静默帧解码成本

高层机制族
  用廉价逐帧响应门替代 EOS 判定，避免静默帧完整解码

候选变体
  PPL 阈值
  相邻视觉 embedding 余弦阈值
  上下文隐藏状态残差线性头
```

“同机制族”不等于结论可以无限外推。所有负结论必须同时保存适用边界。例如，
冻结 VideoLLM-online 检查点、约 3 fps、当前数据定义和 KV 行为下的失败，不能直接
证明所有视频模型、训练式门或其他 workload 下的响应门都无效。

## 5. 负结果的证据等级

以下结果不能计为机制负证据：

- `budgetLimited`、授权失败、下载失败或环境尚未打通；
- runtime timeout 且没有完成冻结测量；
- EXP Goal 自己声明失败，但 Reviewer 尚未核对；
- 指标、baseline、输入或预注册判据发生未解释变化；
- 只说明当前实现有 bug，不能回答机制问题。

负结果进入收敛记忆至少需要：

1. EXP Result 已不可变保存；
2. Reviewer 核对它是否回答绑定的 experiment objective；
3. Reviewer 判断命中了哪个预注册失败条件；
4. Reviewer 明确负结论适用的实验边界；
5. Reviewer 区分它只否证当前 Direction，还是对更广机制族具有累积价值。

EXP 的 `goalStatus=complete` 只表示 Goal 生命周期正常完成，不表示结论为正或为负。
负证据必须来自 Reviewer 对结论的语义审查。

## 6. 角色职责

### 6.1 EXP Goal

EXP Goal 继续只负责：

- 冻结、实现和执行本次实验；
- 保存环境、代码、命令、原始观测、分析和审计材料；
- 回答 Decision 注入的有界经验问题；
- 如实报告支持、不支持、收窄、无结论或环境阻塞。

它不负责：

- 判断与历史 Direction 是否属于同一机制族；
- 决定是否继续另一个实验；
- 把当前负结果外推到整个 Anchor 或 Topic；
- 直接修改 Learning 的 canonical 对象或负结果记忆。

### 6.2 Reviewer

Reviewer 是负证据的局部语义入口。对于绑定 EXP 后的审阅，它必须：

- 核对 EXP 是否真正回答当前 Anchor/Direction 的判别问题；
- 根据预注册条件给出 `PASS | REVISE | REJECT`；
- 判断失败是参数局部、Direction 局部还是对机制族有累积意义；
- 明确适用边界，防止过度外推；
- 将负结果压缩成后续 Agent 可用的 lesson；
- 审查后续候选是否只是更换信号、阈值或包装后的同族重复；
- 只有当新候选改变了已经失败的因果假设，并有独立证据支撑时，才继续提出
  verdict-changing `experiment` query gap。

### 6.3 Decision

Decision 负责全局收敛，而不是重新分析原始测量。它必须：

- 读取当前 Anchor 下所有已审阅 EXP 结论，而不只读取当前 Direction 的结果；
- 在语义上比较 causal lever、baseline change 和 preserved boundary；
- 判断是否已经形成同机制族的累积负证据；
- 在继续同族、转向其他机制、转向其他 Anchor、重新审阅 Anchor 和完成之间选择；
- 若允许一次重开，在 guidance 中说明改变的是哪个失败假设，以及新观测如何改变
  Workflow 决策；
- 不把剩余实验额度、长时间投入或换了实现名字当成继续实验的理由。

### 6.4 Worker

Worker 必须把负结果当作搜索约束，而不是只在正文末尾加 caveat：

- 先读取 Script 注入的相关负证据索引和 Reviewer 结论；
- 创建或深化 Direction 时，比较它与已失败机制的 causal lever；
- 不得仅更换阈值、特征或小型头后把候选描述为全新 Direction；
- 若仍触及已收敛机制族，必须说明改变了哪个失败假设，并给出独立来源依据；
- 机制族关闭后，优先搜索不同性能杠杆、不同 6L 对象或不同 Anchor；
- 若当前 Anchor 内没有可信的异族 Direction，如实返回无结果，不为机械闭合编造
  新候选。

### 6.5 Script

Script 只执行可机械完成的工作：

- 保存 EXP Result、Review Result 及其父 Anchor/Direction Ref；
- 生成按 Anchor 索引的负证据导航文件；
- 将相关 Ref 注入后续 Decision、Worker 和 Reviewer Task；
- 保证同一个实验结论在进入下一次内容实验前已得到 Reviewer 审阅；
- 保存每个 Run、Anchor 和 Direction 已启动的 EXP 数量供 Agent 和用户观察；
- 执行全局实验上限、timeout 和状态机边界。

Script 不判断：

- 两个自然语言机制是否相同；
- 负结果是否足以关闭机制族；
- 某个新证据是否足以重开；
- Worker 是否真正提出了不同因果杠杆。

这些继续由 Reviewer 和 Decision 判断。

## 7. 最小负结果记忆

### 7.1 权威来源

负结果记忆不能替代以下权威文件：

- EXP Goal Result 和 `final_output.md`；
- EXP workspace 中的原始观测、统计和审计；
- Reviewer Result；
- 对应 Anchor、Direction 和 Work Task。

Script 生成的负证据索引只是可重建导航层，与 research memory 和 trajectory 相同，
不能反向覆盖权威结果。

### 7.2 Reviewer 的最小语义输出候选

如果后续实现确认现有 `summary/findings/queryGaps` 难以稳定表达跨轮 lesson，可以在
Reviewer Result 中增加一个可选、非核心控制对象：

```json
"negativeLesson": {
  "scope": "DIRECTION_ONLY | MECHANISM_FAMILY",
  "finding": "包含适用边界的核心失败结论",
  "nextConstraint": "后续 Worker 应避免什么，以及什么条件下才可重新考虑"
}
```

约束如下：

- 没有可信负结论时为 `null` 或不产生 lesson；
- `finding` 必须包含边界，不能写成全局禁令；
- `nextConstraint` 是研究约束，不是 Script 要解析的命令；
- `anchorRef`、`directionRef`、`experimentRef`、`reviewRef` 由 Script 自动附加，
  Agent 不回显；
- 该对象不成为 Worker/Reviewer wire message 的新核心 Gate；缺失不能被 Script
  误判为 JSON 传输失败；
- 若继续使用现有字段即可达到稳定效果，则优先不扩张 Reviewer JSON，而由 Script
  索引相关 Review summary/findings Ref。

在实施前必须用当前 Codex structured-output 限制验证该可选对象是否值得增加；
不能为了压缩记忆重新引入复杂、易错的 Agent JSON。

### 7.3 Script 派生索引

Script 可机械生成：

```text
observations/negative_experiment_index.json
```

每条只保存已知 Ref 和机械状态，例如：

```json
{
  "anchorWork": "results/...json",
  "directionWork": "results/...json",
  "experimentResultRef": "experiments/.../result.json",
  "reviewRef": "results/...json",
  "reviewVerdict": "REJECT"
}
```

不在该索引中增加由 Script 猜测的 `familyId`、`familyName`、`closed=true` 或
`sameMechanism=true`。Decision、Worker 和 Reviewer 根据 Ref 中的专业结论完成
语义分组。

## 8. 收敛规则：两次关闭，一次有条件重开

建议采用以下默认语义策略，而不是简单的“每族最多 N 次”：

### 8.1 第一次可信负结果

- 关闭被测 Direction；
- 不自动关闭整个机制族；
- Reviewer 记录失败假设和边界；
- Decision 可以允许 Worker 搜索一个实质不同的候选。

### 8.2 第二次同族可信负结果

当第二个候选已经改变了局部实现，但 baseline change、causal lever 和关键边界仍然
相同，且它也命中机制判别失败条件时：

- 默认认为该机制族在当前边界下已经收敛；
- 下一步不应自动启动第三个只换信号的 EXP；
- Decision 应优先让 Worker 转向不同因果杠杆、不同 6L 对象或其他 Anchor；
- 如果当前 Anchor 的价值只依赖该机制族，应重新审阅 Anchor，而不是强制生成
  新 Direction。

### 8.3 一次有条件重开

关闭后的机制族最多允许一次有明确依据的重开。必须同时满足：

1. 新候选改变的是已失败的因果假设，而不只是阈值、特征名称或实现包装；
2. 存在独立论文、实现、已有观测或机制证据，不是由连续失败临时猜出的变体；
3. Reviewer 明确认可其与已关闭变体的实质差异；
4. Decision 说明一个最小判别实验如何决定是否继续；
5. 重开实验仍受相同 baseline、公平性和 guardrail 约束。

重开候选再次得到可信负结果后，该更广机制族在本次 Run 内关闭。除非用户修改
Goal、实验边界发生实质变化或新的外部证据出现，不再启动第四个相邻实验。

### 8.4 对当前案例的解释

```text
PPL 门失败
  → 第一次可信负结果

视觉余弦门失败
  → 第二次同族失败
  → 关闭“冻结信号/阈值式逐帧响应门”

监督上下文残差头
  → 从无监督冻结分数变为监督、上下文相关判别
  → 可以被视为一次有条件重开，但必须显式说明依据

残差线性头再次失败
  → 更广“逐帧轻量响应门”机制族关闭
  → 不应继续第四个响应门 EXP
```

该规则允许有信息价值的救援实验，同时防止无限局部爬山。

## 9. EXP 后的状态机必须先完成原子语义整合

09 当前定义：

```text
EXP Goal → Decision
```

这一顶层关系保留。但当前 `RUN_REVIEWER` 分支会继续展开为：

```text
Reviewer → Worker → Reviewer → Decision
```

对于刚完成的 EXP，这个固定后缀会在 Decision 再次全局判断前自动生成替代
Direction，是局部实验链出现的直接原因之一。

建议为现有 Reviewer 增加一个 Script 派生的特殊模式，不增加新 Agent 和新顶层
Decision 字面量：

```text
EXP Goal
  → Decision
  → Decision 选择 RUN_REVIEWER
  → POST_EXP_REVIEWER
  → 保存 Reviewer 结论和负证据 lesson
  → Decision
```

`POST_EXP_REVIEWER` 的固定后缀只有 `Decision`，不自动附带 Worker。它只负责把
实验结果转化为可信的 Learning 语义结论。

新的 Decision 再选择：

- `RUN_WORKER`：在负结果约束下寻找异族 Direction 或其他 Anchor；
- `RUN_REVIEWER`：重新审阅父 Anchor 或从新角度审阅；
- `RUN_EXP_GOAL`：仅在重开条件成立时启动另一个实验；
- `FINISH_WORKFLOW`：负结果已使剩余信息增益很低且其他要求已闭合。

Script 只根据“当前有尚未语义整合的 EXP Result”和 Decision 的 `RUN_REVIEWER`
字面量选择 `POST_EXP_REVIEWER` 模式，不解释实验结论。

## 10. 负结果如何进入后续 Worker

### 10.1 Task 输入

Script 为绑定 Anchor 的 Worker Task 增加一个可选输入 Ref：

```text
negativeExperimentHistoryRef
```

该文件只列出同 Anchor 的相关 EXP Result、Review Result 和必要 object Ref。对于
创建新 Anchor 的开放探索 Task，可以注入 Topic 级压缩负结果导航，避免在其他
位置重新包装相同机制。

Agent 不回显这个 Ref，也不复制完整实验日志。

### 10.2 Worker 方法要求

Worker 读取历史后执行：

1. 确认当前 Task 是否触及已有负结果的 baseline change 和 causal lever；
2. 若属于关闭机制族，不得只改变阈值、信号、feature 或小型 head；
3. 优先搜索不同因果杠杆，例如减少逐帧视觉编码、改变帧调度、缓存策略、异步
   执行、batching 或其他具体 6L 对象；
4. 若候选确实改变失败假设，在现有 `mechanism`、`baselineChange` 和 `evidence`
   中解释差异和独立依据，不增加专门的自评分字段；
5. 如果找不到可信异族候选，返回 `BLOCKED_NO_RESULT`，并把已有负结果视为有价值
   的收敛证据。

### 10.3 Reviewer 的重复检查

Reviewer 同样读取 `negativeExperimentHistoryRef`，检查：

- Worker 是否只是重命名同族变体；
- 新候选是否真正改变已失败的因果假设；
- 独立证据是否足以支持一次重开；
- 新的 `experiment` query gap 是否仍会改变 verdict，而不是延续局部搜索惯性。

同族重复且没有重开依据时，应 `REJECT` 或要求转向，不再产生推动下一次 EXP 的
阻塞性 experiment query gap。

## 11. Anchor 生命周期与 Requirement 闭合

当前要求“每个最终 Anchor 至少包含一个可验证 Direction”只适用于最终 active
Anchor。它不能被解释为：一个 Anchor 的所有可信机制都被实验否证后，仍必须无限
生成替代 Direction。

因此，负结果收敛还需要一条 Anchor 出口：

```text
当前 Anchor 的主要机制族关闭
  ├─ 仍有证据支持其他因果杠杆
  │    → Worker 探索异族 Direction
  └─ 没有可信剩余杠杆，或原性能矛盾本身被实验推翻
       → Reviewer 重新审阅父 Anchor
            ├─ PASS/REVISE：保留 Anchor，但明确剩余边界
            └─ REJECT：Anchor 退出最终 active 集合
```

只有这样，机械 requirement 才不会反向驱动无限局部实验。

实现时应优先复用 `RUN_REVIEWER` 字面量：当 Decision 在已审阅的 family-level
负结果后再次选择 Reviewer，Script 可绑定父 Anchor 的 `REVIEW_ANCHOR_AFTER_EXP`
Task。Reviewer 仍使用现有 Anchor review contract；是否保留 Anchor由 Reviewer
语义判断，Script 只提交 verdict。

具体 Task 选择规则需要在代码修改前单独核对，确保普通 `RUN_REVIEWER` 的既有
闭环不被误改。

## 12. Decision 的重新打开与完成规则

Decision Skill 需要增加以下明确规则：

- 同一 Anchor 的实验历史必须按机制而不是 Direction 名称比较；
- `budgetLimited`、授权失败或无有效测量不计为负机制结果；
- 两个同族可信负结果后，默认转向，不把剩余 EXP quota 当成继续理由；
- 重新打开必须说明改变的失败假设和独立依据；
- 重开失败后，该机制族在本 Run 内关闭；
- 负结果可以支持 `FINISH_WORKFLOW`，也可以支持探索其他 6L 区域；它不是自动
  完成信号；
- 当一个 Anchor 已无可支持 Direction 时，应考虑重新审阅或淘汰 Anchor，而不是
  为满足 requirement 重复创建 Direction。

Decision 输出协议继续保持：

```text
decision = <Script 本次允许字面量>
guidance = <可选简短说明；RUN_EXP_GOAL 时为实验目标>
```

不增加 `familyDecision`、`negativeCount`、`closeFamily` 等新的 Decision 输出字段。
机制收敛理由保存在 Decision 原始输出和 Reviewer/EXP 权威 Ref 中，Script 派生
观察负责导航。

## 13. Script 的机械边界

建议 Script 增加的只是：

- EXP/Review/Anchor/Direction Ref 索引；
- `POST_EXP_REVIEWER` 的固定序列；
- `negativeExperimentHistoryRef` 的 Task 输入；
- 每个 Anchor/Direction 的 EXP 次数观察；
- family-level lesson 后的父 Anchor review 路由；
- 不允许未审阅的 EXP 结果直接触发另一个同对象实验；
- 全局 `maxExperimentGoals`、timeout 和暂停逻辑保持不变。

不建议 Script 增加：

- 关键词或向量相似度 family classifier；
- 按自然语言 family 名称计数的硬 Gate；
- 自动把第 N 个负结果解释为机制关闭；
- 自动从 EXP 报告抽取专业失败条件；
- 根据指标阈值直接拒绝 Direction；
- 因为 Reviewer 可选 lesson 缺失而把完整 JSON 判为无效。

若后续仍出现 Decision/Reviewer 忽略负历史的问题，再讨论由 Reviewer 明确引用
已有 lesson Ref 的小型协议；不在第一步引入复杂 `familyId`、图结构或硬编码分类。

## 14. 与 09 的关系

09 的以下结论继续成立：

- 纸面 gap 不等于本地实测 gap；
- Reviewer 可以提出会改变 verdict 的最小实验缺口；
- Decision 决定是否运行 EXP；
- EXP 是信息增益分支，不是完成前仪式；
- EXP 无 token budget，使用独立 idle timeout 和 hard cap；
- EXP 完成后回到 Decision；
- Script 不理解实验目标和专业结论。

10 对 09 增加的是：

1. EXP 的负结果必须先经 Reviewer 形成可复用语义；
2. 同一 Anchor 的历史实验必须持续反馈给 Decision、Worker 和 Reviewer；
3. Decision 需要执行机制族级收敛，而不只判断单个 Direction；
4. EXP 后的 Reviewer 使用原子 `Reviewer → Decision` 路径，不能自动生成替代
   Worker；
5. 一个 active Anchor 在机制被否证后必须能够重新审阅或退出最终集合；
6. 全局 EXP 数量上限不能替代语义收敛。

## 15. 对当前暂停运行的判断

当前运行已完成三次有科学结论的新 EXP，最后一个上下文残差线性头也得到明确负
结果。按本设计：

- 继承的 401 授权失败不计入负结果次数；
- PPL 门是第一次可信负结果；
- 视觉余弦门构成第二次同族失败；
- 上下文残差线性头可视为一次监督、上下文相关的有条件重开；
- 重开也失败后，“逐帧轻量响应门”机制族应在当前 Run 和实验边界下关闭；
- 下一步应先由 Reviewer 整合最后 EXP，再由 Decision 转向异族探索、父 Anchor
  复审或完成；不应启动第四个响应门实验。

当前目录保持 `PAUSED`，本文不修改其 state、run、Skill pin 或任何实验产物。

实现没有原地修改该目录；它通过下述预算重置 continuation 继承：

```text
learning_outputs_codex/
  multimodal_inference_latency_first_v10_negative_convergence_reset_20260803
```

新分支保留 4 条历史 EXP、全部对象和结论，从 round 46 的 Decision 开始；轮次重新
授权 6 轮，EXP 重新授权 5 次。历史 EXP 不占新授权，但仍进入语义上下文。

## 16. 后续实现顺序

若该策略确认，建议按以下顺序实施：

1. 先修改状态机中的 post-EXP Reviewer 序列，消除自动 Worker 后缀；
2. 增加按 Anchor 的实验与 Review Ref 索引，并注入三个 Learning 角色；
3. 修改 Reviewer Skill：负结果边界、同族重复、重开依据和父 Anchor 影响；
4. 修改 Decision Skill：两次关闭、一次重开、重开失败后收敛；
5. 修改 Worker Skill：读取负历史、异族探索和诚实 `BLOCKED_NO_RESULT`；
6. 设计父 Anchor 的 post-EXP 重新审阅路由；
7. 最后再决定是否需要可选 `negativeLesson` JSON；若现有 summary/findings 已足够，
   不扩张 wire schema；
8. 更新 observation、checkpoint、README 和最终报告，使用户能看到负结果如何改变
   后续搜索。

## 17. 验收场景

实施后至少覆盖以下测试：

1. 授权失败或环境阻塞不会被计为机制负结果；
2. 完成 EXP 后，Reviewer 审阅结束立即回到 Decision，不自动启动 Worker；
3. 同 Anchor 的历史负 EXP Ref 会注入后续 Worker、Reviewer 和 Decision；
4. 第一次负结果允许一个有依据的替代候选；
5. 第二次同族失败后，普通换信号候选被 Reviewer 识别为重复，不再产生新的
   experiment query gap；
6. 改变失败因果假设且有独立证据的候选可获得一次重开机会；
7. 重开实验再次失败后，不再启动同族 EXP；
8. 不同机制族或不同 Anchor 的高价值实验仍可由 Decision 选择；
9. 无剩余 Direction 的 Anchor 可以被重新审阅并退出 active 集合，机械 requirement
   不再制造无限替代；
10. Script 不需要理解任何专业机制文本即可完成保存、索引、注入和状态转换；
11. Agent JSON 核心字段和最小 Controller Gate 不因负结果记忆而重新膨胀；
12. checkpoint/final report 能展示被否证机制、适用边界和后续 Worker 如何避开它。

## 18. 本文的最终策略摘要

```text
可信负 EXP
  → Reviewer 确认失败条件和边界
  → Script 保存并按 Anchor 索引负证据
  → Decision 判断 Direction 失败还是机制族收敛
  → Worker 读取负历史并转向不同因果杠杆
  → Reviewer 阻止换名重复
  → 最多一次有独立依据的机制重开
  → 重开再失败则关闭该机制族
  → 必要时重新审阅或淘汰父 Anchor
```

真正的收敛不是“实验次数达到上限”，而是：负结果已经改变了可接受的假设空间，
后续 Worker 不再回到同一失败因果杠杆。

## 19. 实施记录

本轮按最小 Controller Gate 实施，没有增加 Agent 输出核心字段，也没有加入
`negativeLesson`、`familyId` 或 Script 机制分类器。

### 19.1 状态机

- 未审阅 EXP 存在时，Script 注入给 Decision 的允许集合只含
  `RUN_REVIEWER`；
- 该分支展开为 `POST_EXP_REVIEWER → Decision`，不包含自动 Worker；
- post-EXP Reviewer 的 Task 同时冻结当前 Work、旧 Review、EXP Result 和同
  Anchor 负历史；
- EXP 的 Reviewer Ref 回写到 Script-owned experiment record；
- 最新负 Direction 被 Reviewer `REJECT` 且父 Anchor 无 viable Direction 时，
  `RUN_REVIEWER` 预览为 `ANCHOR_REASSESS → Decision`；每个最新 EXP 只机械记录
  一次这种父 Anchor 复审。

### 19.2 负证据索引和注入

- `observations/negative_experiment_index.json` 保存 Run/Anchor/Direction EXP
  次数和已审阅负结果 Ref；
- 只有 `goalStatus=complete` 且后续 Reviewer 为 `REJECT` 的 EXP 进入负条目；
  `budgetLimited` 等非科学终态不进入；
- Script 可从旧历史中按“Review Turn 晚于 EXP 完成且 Work Ref 相同”恢复已有
  review 关联；
- 每个 DecisionContext 冻结
  `negative_experiment_history_snapshot.json`；
- Worker/Reviewer Task 冻结 `negative_experiment_history.json`，按 Anchor 对象的
  全部 revision 聚合，而非只匹配当前 Work Ref；
- Decision 的 `experimentContext.previousResultRefs` 同样覆盖该 Anchor 的全部
  revision 和 Direction 实验。

### 19.3 Agent 方法

- Decision Skill 实现“两次同族负结果默认关闭、一次有依据重开、重开失败后更广
  机制族收敛”，并明确剩余额度不是继续理由；
- Reviewer Skill 先判定 EXP 是否真正回答 frozen objective，再判断当前 Direction、
  同机制族累积价值和适用边界；它使用现有 `summary/findings/queryGaps` 保存 lesson；
- Worker Skill 把负历史作为搜索约束，比较 baseline change、causal lever 和
  preserved boundary；找不到异族杠杆时允许诚实 `BLOCKED_NO_RESULT`。

### 19.4 预算重置 continuation

`continue --reset-budgets` 允许从稳定 `PAUSED` 或 `FINISHED` source 创建不可变
分支。它保留历史实验用于认知，但用 `sourceExperimentCount` 将历史记录与新分支
授权消耗分开；新分支的 `experimentGoalsStarted` 从 0 计。旧 source 仍未被修改。

### 19.5 验证

- `node --test scripts/simple_semantic_loop/tests/*.test.ts`：50/50 通过；
- Decision、Worker、Reviewer、EXP 四个 Skill 均通过 `quick_validate.py`；
- `doctor --no-provider` 通过；
- 新 continuation 的 `validate` 全部检查通过且无 advisory；
- 新分支初始机械投影为：允许 `RUN_REVIEWER`、模式
  `POST_EXP_REVIEW`、序列 `REVIEWER → DECISION`；旧两条可信负结果已进入索引，
  最后一条残差头 EXP 保持待独立审阅。
