# Experiment Flow 的最强简单基线、行为等价性与消融归因设计

状态：已实现为 Direction Experiment Loop format v4、科学比较与载体政策
`COMPETITIVE_BASELINE_V3`，并同步重构 Learning Worker/Reviewer 与 Direction
Lab/Judge Skill。承接
`03_local_proxy_open_source_single_gpu_and_simulator_revision_design.md`。本文不改变
`Evidence Judge ↔ Direction Lab Goal` 双角色 Loop，也不增加 Controller 的语义判断；
它修正的是 Direction 对 baseline 的表达，以及 Lab/Judge 判断“优化是否真的带来增量
价值”的方法主干。

## 1. 本次修订解决的问题

03 已经允许使用弱化代理、真实单卡、局部开源实现和模拟器，并要求在同一载体中
比较 baseline/variant。但“存在一个可运行 baseline”仍不足以建立优化价值：

- baseline 可能是未经公平调优的弱策略；
- baseline 的可用参数域可能被不合理缩窄；
- variant 虽然形式更复杂，却可能与某个简单 baseline 产生完全相同的决策；
- 性能收益可能来自“启用某个阈值”而不是 Direction 声称的新增信息或机制；
- 一个联合实现可能同时改变多个可独立开关的组件，使结果无法归因；
- 大规模参数扫描和稳健性测试可能在确认主要变化是否真正生效之前就被执行。

最近的长度相关 MM-SP/TP 模拟实验体现了这一风险。实验把长度感知选择器与一个
被限制为始终选择 full-TP 的 batch-only baseline 比较，得到模拟 P95 改善；但一个
合法的全局 batch threshold 可以在全部被测事件上产生与长度感知选择器相同的动作
轨迹。因此现有结果最多支持“某个 batch threshold 优于始终 full-TP”，不能支持
“长度信息相对最强简单 batch-only 策略具有增量价值”。

本次修订的核心问题是：

> 在承认软硬件、模型和规模可以弱化的同时，如何保证实验仍然比较了公平、竞争性
> baseline，并且观测收益确实来自 Direction 声称的唯一主要变化？

## 2. 核心原则

实验载体可以弱化，科学比较不能被弱化。

```text
冻结 Direction
  ↓
识别执行 baseline、最近方法 baseline 和最强简单 baseline
  ↓
在校准数据上公平选择/调优决定性 baseline
  ↓
构建最小可信代理并验证 baseline 正确性
  ↓
检查 variant 是否产生独有的动作/状态/执行路径差异
  ↓
执行同载体、单一主要变化的配对消融
  ↓
只有独有变化产生增量收益时才允许 SUPPORT
```

必须同时满足以下原则：

1. **Baseline 可运行不等于 baseline 有竞争力。** 应比较与当前主张最接近、在同一
   合法参数域中经过公平选择的简单策略。
2. **形式差异不等于行为差异。** 新增特征、分类器、阈值表或控制器只有在实际事件
   上改变了相关动作、状态转换或执行路径时，才可能获得独立归因。
3. **A/B 性能差异不自动证明 Direction。** 必须先证明 A/B 唯一主要差异就是当前
   Direction 的因果 lever，而且该 lever 被实际触发。
4. **弱化只改变载体和结论范围。** 更小模型、数据子集、请求规模、短 trace、替代
   软件/硬件或模拟器均可使用，但 baseline 与 variant 必须在同一弱化载体中公平
   比较。
5. **语义判断属于 Agent。** Controller 只调度、保存、索引和检查控制字段，不判断
   baseline 是否最强、动作轨迹是否等价或消融是否足以支持科学结论。

## 3. Direction 应表达什么

### 3.1 三种 baseline 的关系

Direction 及其父 Anchor 可能涉及三种 baseline：

1. **Parent execution baseline**：Anchor 中暴露性能矛盾的当前执行方法或系统。
2. **Closest method baseline**：与 Direction 在目标、修改对象、机制、粒度和运行区间
   上最接近的已有方法。
3. **Strongest simple baseline**：在当前实验可用信息和合法参数域中，不引入
   Direction 新增复杂度、但能最有力解释或替代其收益的简单策略。

三者可以相同，也可以不同。真正的决定性实验 baseline 应是能够排除最强替代解释
的那个，而不是默认选择最容易击败的 parent baseline。

例如，若 Direction 声称“输入长度感知的多个阈值优于 batch-only 调度”，则最强
简单 baseline 不是“始终 full-TP”，而应是经过相同校准预算选择的单一全局 batch
threshold。只有长度感知策略相对这个 baseline 产生不同动作并获得增量收益，长度
信息才具有可归因价值。

### 3.2 保持 Direction JSON 精简

不为上述三个概念增加重复的调度字段。继续使用现有内容字段：

- Anchor 的 `baseline` 保存 parent execution baseline；
- Direction 的 `mechanism` 描述可修改对象、因果接口和新增信息；
- Direction 的 `baselineChange` 同时明确：最近/最强简单 baseline、唯一主要变化，
  以及二者必须不同的可观测动作或执行路径；
- `expectedEffects`、`failureConditions` 和 `measurementPlan` 定义效果、守护、等价
  失败条件与判别实验。

`baselineChange` 的语义最低要求是回答：

```text
比较对象是谁？
它如何被公平选择或调优？
variant 唯一新增了什么信息、状态或执行变化？
在哪类事件上二者应产生不同动作？
如果存在简单策略产生同样动作，Direction 应如何判定？
```

当前 `work-result-direction-v2` 没有独立的 `nearestMethodBaseline` 字段，而
`direction_target.md` 的机械投影却尝试读取该字段。实现时不应为了投影再增加重复
JSON 元素；应把该展示项改成“Closest/strongest baseline and unique change”，直接
展开 `baselineChange`。缺失的真实科学信息仍写为 `NOT_DECLARED`，不得由脚本推断。

### 3.3 Direction 的最低实验含义

一个可进入 Experiment Flow 的 Direction 至少应能够恢复：

- 明确的执行 baseline 与最近方法 baseline；
- 一个可切换、可测量的主要变化；
- 主要变化作用的因果接口和触发条件；
- 最强简单替代策略及其合法参数域；
- 主要指标、正确性/质量/吞吐守护与失败条件；
- baseline 复现、简单策略校准、行为差异检查和配对消融的方法；
- 允许弱化的环境/规模变量与不可弱化的因果不变量。

若这些内容无法从冻结 Direction、Anchor 和证据中恢复，Lab 不应自行补造科学绑定；
Judge 应选择一次有界修复，或在必须改写科学主张时选择 `REVISE_DIRECTION`。

## 4. “复现 baseline”的准确含义

Experiment Flow 要求复现的是**当前载体内可公平执行的科学 baseline**，不是默认
复现论文完整代码、硬件和绝对数值。

baseline 复现至少包括：

- baseline 的策略、控制分支或执行路径真实存在并可单独运行；
- 输入、输出、正确性和必要质量约束可以验证；
- 机制触发事件、队列/cache/拓扑状态或关键算子可观察；
- baseline 参数在其完整合法域内选择，而不是因 variant 的结构人为缩窄；
- calibration 与 confirmatory/holdout 数据严格分离；
- baseline 与 variant 使用相同载体、资源、输入和统计规则；
- compatibility patch 与 Direction 的主要修改分开记录。

以下内容不属于默认硬门槛：

- 与论文完全相同的 GPU、runtime、模型和数据规模；
- 逐项复现论文绝对延迟、吞吐或利用率；
- 重建与当前因果接口无关的完整服务系统；
- 获得未公开的最终策略或私有 patch。

只有当这些对象承载不可替代的因果机制时，才需要由代理或模拟器显式表达。

## 5. 最强简单 baseline 合同

### 5.1 选择原则

Lab 在正式测量前应构造一个有界的 baseline ladder：

```text
parent execution baseline
  ↓
closest existing method baseline
  ↓
strongest simple alternative under the frozen information boundary
```

不要求穷尽所有算法，但至少要包含一个能够解释当前收益的最强简单替代项。选择时
优先考虑：

- 相同目标和运行区间；
- 修改同一控制对象或执行接口；
- 使用 Direction 新增信息的严格子集；
- 参数数量更少、规则更简单，但可在完整合法域中调优；
- 能够产生与 variant 相同或近似动作的已知策略；
- 来源方法或当前实现中最自然的关闭/退化版本。

### 5.2 公平校准

baseline 与 variant 的选择过程必须预先冻结并可比较：

- 使用相同 calibration workload、输入支持、资源上限和正确性规则；
- 使用相同或明确公平的候选/搜索预算；
- baseline 的候选域按自身合法范围定义，不取多个异构分组支持域的交集来人为限制；
- 不允许 variant 使用 holdout 结果调参而 baseline 只用固定默认值；
- 不允许确认性结果出现后更换 baseline、成功条件或统计口径而不重新绑定实验；
- 保存 baseline 候选、选择规则、被选参数和未选原因。

当两个策略的表达能力不同，公平不一定意味着候选数逐项相同，但必须保证 baseline
获得足够预算以排除“未调好所以较差”的替代解释。

## 6. 行为等价与支配检查

### 6.1 检查对象

在大规模性能测量前，Lab 应针对当前 Direction 选择最接近因果接口的 trace：

- 调度策略：每个事件的设备、队列、batch、优先级或模式选择；
- cache 策略：命中/驻留/淘汰/spill 分支与状态转换；
- compiler/runtime：pass、kernel、fusion、layout 或执行图选择；
- 模型/算法：token、head、expert、layer、路径或停止决策；
- 硬件机制：资源分配、通信路由、同步和时序事件。

这里的“action trace”是语义概念，不新增固定 Controller JSON。Lab 根据 Direction
确定最小可观察轨迹并保存原始记录和摘要。

### 6.2 等价判断

若存在一个公平校准的简单 baseline，在全部有效测试事件上产生与 variant 相同的
相关动作、状态转换或执行路径，则：

- 不能把收益归因于 variant 新增的特征、状态或复杂策略；
- 若两者执行路径也相同，则无需继续昂贵性能 sweep；
- 若实现开销不同，只能比较实现开销，不能声称新增决策信息有效；
- 若等价仅因当前 workload 未触发差异，可由 Judge 请求一个有界、预先定义的触发
  workload；不得无限寻找能产生正结果的区间；
- 若冻结适用范围内仍保持等价，并命中 Direction 的失败条件，Judge 应在对应证据
  范围选择 `REJECT`；
- 若必须改变新增信息、baseline 或适用范围才能制造差异，应选择
  `REVISE_DIRECTION`。

### 6.3 支配判断

若最强简单 baseline 在相同正确性和 guard 条件下：

- 产生相同动作但复杂度更低；或
- 覆盖 variant 的有效动作，且主要指标不差；或
- 以更少状态、更低开销达到相同效果，

则 variant 在当前证据范围内被简单策略支配。不得因 variant 的形式更复杂或参数表
更丰富而判定其具有独立价值。

行为等价和支配属于 Lab/Judge 的语义判断。Controller 只保存它们引用的 artifact，
不尝试机械证明。

## 7. 消融实验合同

### 7.1 最低必要消融

每个性能结论至少需要一个同载体的单变量配对：

```text
A：公平选择并验证的决定性 baseline
B：仅启用 Direction 唯一主要变化的 variant
```

组内应冻结源码版本、build、硬件或模拟器、资源、模型、精度、输入、到达过程、
初始状态、seed、预热、测量窗口、指标、质量检查和统计规则。必须报告主要变化的
触发次数和行为差异次数。

### 7.2 独立组件与联合包

- 若组件可以独立开关并具有独立语义，应逐个消融，或只保留其中一个作为当前
  Direction；不得用不可归因的联合收益支持每个组件。
- 若组件在技术上不可分割，允许作为联合包比较，但必须说明不可分割原因，并且只
  支持包级效果。
- setup、兼容补丁、日志和测量接口是实验 enabler，不应被写成优化贡献。

### 7.3 实验顺序

Lab 应按由便宜到昂贵的顺序工作：

1. baseline 可运行性、正确性和接口检查；
2. 最强简单 baseline 的有界校准；
3. 主要变化的触发覆盖；
4. action/state/execution trace 差异检查；
5. 小规模同载体配对消融；
6. 只有前五步显示独有且有意义的变化后，才进行更多 seed、参数敏感性、真实单卡
   性能、模拟包络或论文环境外部有效性测试。

这可以避免在一个行为等价或被简单策略支配的 Direction 上消耗大量 GPU 和 Agent
时间。

## 8. Direction Lab Goal 的主干修改

Lab Goal 应在现有弱化载体策略之前和之中增加以下职责：

1. 从冻结输入恢复 parent baseline、最近方法 baseline、最强简单替代项和决定性
   A/B pair；不把 `NOT_DECLARED` 自动填成事实。
2. 选择最弱充分证据级别与最小可信载体，允许更小模型、数据子集、请求规模、短
   trace、替代软硬件、局部开源实现或模拟器。
3. 在当前载体中建立可运行且正确的 baseline；不要求完整论文环境。
4. 在 calibration 数据上公平调优最强简单 baseline，冻结选择规则和参数。
5. 实现 variant 的唯一主要变化，并把兼容修改与 Direction patch 分离。
6. 保存 baseline/variant 的相关 action trace，检查唯一主要变化是否真实触发以及
   是否存在简单策略等价或支配。
7. 只有通过上述检查后才执行确认性配对消融和必要的稳健性实验。
8. 在 `result.md` 中明确：
   - 决定性 baseline 及选择理由；
   - baseline 的合法参数域和校准预算；
   - baseline 正确性证据；
   - variant 唯一主要变化；
   - 触发与行为差异计数；
   - 等价/支配检查结果；
   - A/B 结果及必要组件消融；
   - 弱化维度、保留因果不变量和结论边界。

若小规模检查已证明完全等价，Lab 可以结束当前有界 Goal 并把该事实交给 Judge，
不需要为了“完成实验”继续执行无判别力的长时间 sweep。

## 9. Evidence Judge 的主干修改

### 9.1 `RUN_LAB`

以下任一关键问题尚未解决且存在有界动作时，选择 `RUN_LAB`：

- baseline 尚未正确运行或其参数域被不公平限制；
- 最近方法/最强简单 baseline 尚未识别或公平校准；
- variant 是否产生独有动作或执行路径尚不清楚；
- 现有 A/B 同时改变多个可独立组件；
- 一个小型行为等价或支配检查即可改变结论；
- 代理、单卡或模拟器可以检验尚未触发的必要机制。

`labGoal` 应只描述一个判别性问题，包括载体、决定性 baseline/variant 边界、允许
弱化项、必须保留的因果接口/guard，以及成功、失败或仍不确定时应形成的 artifact。

### 9.2 `SUPPORT`

除现有证据范围要求外，只有同时满足以下条件才能 `SUPPORT`：

- baseline 已通过正确性和接口验证；
- 已包含并公平校准最强简单 baseline；
- baseline 和 variant 的唯一计划差异就是冻结主要变化；
- 主要变化在被测 workload 中实际触发；
- variant 相对最强简单 baseline 产生可观察的独有动作或执行路径；
- 增量收益来自该独有变化，并通过冻结指标、guard 和统计条件；
- 不存在一个行为等价或支配 variant 的更简单策略；
- 结论严格限制在弱化代理、真实单卡、模拟器或论文外部有效性对应范围。

“variant 比未调优默认策略更快”或“形式上的阈值表不同”都不足以 `SUPPORT`。

### 9.3 `REJECT`

以下证据可在相应范围支持 `REJECT`：

- 公平校准的最强简单 baseline 与 variant 行为等价，且冻结主张要求新增信息产生
  独立作用；
- variant 被简单 baseline 支配；
- 有效单变量配对命中核心失败条件；
- 同机制的有意义变体在冻结适用范围内反复失败。

若等价仅由当前代理未覆盖触发区间造成，应先判断是否存在一个有界、非事后挑选的
触发测试；弱化环境失败或实现失败仍不能作为科学 `REJECT`。

### 9.4 `REVISE_DIRECTION`

当公平实验必须改变以下任一科学绑定时选择 `REVISE_DIRECTION`：

- 决定性 baseline；
- 新增信息或主要因果 lever；
- 必需适用范围或触发条件；
- guard、成功条件或可归因边界。

例如，长度特征没有产生独立动作，而需要增加方差、最大长度或新拓扑状态才能获得
差异时，应结束当前冻结 Direction，由 Learning Flow 创建新 revision，而不是在
原 Goal 中静默扩充特征。

### 9.5 `BLOCKED`

维持 03 的严格外部阻塞规则。未完成最强简单 baseline 或等价性检查通常是
`RUN_LAB`，不是 `BLOCKED`；只有代理、单卡、开源局部构建和模拟器均无法表达必要
因果接口，且确实缺少外部授权、数据、代码或硬件时才可阻塞。

## 10. Controller、协议和存储边界

本修订不增加 Controller 智能，也不扩大 Judge 的输出 JSON。继续使用：

```json
{
  "decision": "RUN_LAB | SUPPORT | REJECT | REVISE_DIRECTION | BLOCKED",
  "evidenceScope": "WEAKENED_PROXY_MECHANISM | LOCAL_SINGLE_GPU_PERFORMANCE | SIMULATED_HARDWARE_MECHANISM | PAPER_EXTERNAL_VALIDITY",
  "reason": "简短证据判断和边界",
  "labGoal": "仅 RUN_LAB 时填写"
}
```

Controller 只负责：

- 冻结并传递 Direction、Anchor、review、policy、history 和 `labGoal`；
- 启动 Judge 或持久 Lab Goal；
- 保存原始 Agent 输出、配置、命令、artifact、决策和状态；
- 校验 JSON 是否可解析、decision/evidenceScope 字面量、非空 reason 和条件性
  `labGoal`；
- 按合法决策推进状态机。

Controller 不负责：

- 判断哪个 baseline 最强；
- 解释参数域是否公平；
- 比较 action trace 的语义；
- 判定策略是否等价或支配；
- 判断某次消融是否足以支持科学结论。

上述语义结论由 Lab 写入 `result.md` 和对应 artifacts，由下一次 Judge 阅读并判断。
可以约定推荐文件名，例如 `baseline_audit.md`、`action_trace_equivalence.md` 和
`ablation_summary.md`，但它们不应成为 Controller 的强语义 Gate；现有
`result.md` 仍是 Judge 的统一入口。

## 11. 对当前 Direction 投影的修订

`direction_target.md` 继续是原始 JSON 的机械可读投影，原始 JSON 保持权威。建议将：

```text
Nearest method baseline: directionContent.nearestMethodBaseline
Unique baseline change: directionContent.baselineChange
```

合并为：

```text
Closest/strongest baseline and unique change: directionContent.baselineChange
```

这样不会为了显示目的增加重复 JSON 字段，也不会让 Agent 误以为一个不存在于
Direction contract 的字段是权威输入。父 Anchor 的 execution baseline 继续单独投影。

## 12. 当前模拟 Direction 的低成本纠正实验

现有模拟结果和原始 holdout 保持不可变，不覆盖或删除。针对长度相关 MM-SP/TP
Direction，只需执行一次低成本重新分析，无需新的 GPU 实验：

1. 在 calibration 数据上按 batch-only 策略自身的完整合法参数域选择最佳全局
   threshold，不能使用所有长度桶公共支持域的交集；
2. 冻结该 threshold 后重放现有 holdout；
3. 比较全局 threshold 与长度感知策略的逐事件动作轨迹；
4. 若动作完全相同，记录行为等价并停止性能归因；
5. 若存在差异，确认差异事件覆盖预注册触发条件，再进行配对指标比较；
6. 只有长度感知策略在这些独有事件上产生稳定增量收益，才能支持当前 Direction；
7. 若必须加入方差、最大长度或新硬件状态才能产生差异，输出
   `REVISE_DIRECTION`，不得修改当前冻结主张。

## 13. 实现落点

本次实现已修改：

- `.codex/skills/learning-loop-worker/`：在 Direction 方法中强化最强简单 baseline、
  等价失败条件和最低消融含义，同时保持现有精简 JSON；
- `.codex/skills/learning-loop-reviewer/`：审查 baseline 竞争性、参数域、公平校准和
  可归因性；
- `.codex/skills/direction-lab-goal/`：把 baseline ladder、行为等价/支配检查和由
  便宜到昂贵的实验顺序加入主干方法；
- `.codex/skills/direction-evidence-judge/`：把最强简单 baseline 和独有增量效果加入
  `RUN_LAB/SUPPORT/REJECT/REVISE_DIRECTION` 判断；
- `scripts/direction_experiment_loop/setup.ts`：修正 Direction 投影中的不存在字段，
  不增加新的调度协议字段；
- `scripts/direction_experiment_loop/contracts/experiment_policy.md`：补充竞争性 baseline
  与行为等价边界；
- 相应 README、静态合同测试和 E2E 测试。

不应修改：

- 双角色 Loop 拓扑；
- Judge 的最小 JSON 协议；
- Controller 的最小机械校验原则；
- 已冻结旧运行的输入、日志和实验产物；
- 03 已定义的弱化代理、单卡、开源局部构建、模拟器和证据范围语义。

## 14. 验收条件

实现完成后应满足：

1. Direction 可恢复 parent baseline、最强简单 baseline、唯一主要变化和判别事件，
   不新增重复控制字段；
2. `direction_target.md` 不再因读取非 contract 字段而无意义地产生
   `Nearest method baseline: NOT_DECLARED`；
3. Lab 在昂贵 sweep 前执行 baseline 正确性、竞争性校准、触发覆盖和行为等价检查；
4. Judge 不会仅凭 variant 击败未调优或人为受限 baseline 而 `SUPPORT`；
5. 行为等价或被支配的复杂策略能够在相应证据范围停止，而不是继续无界实验；
6. 弱化代理、真实单卡和模拟器仍被允许，且结论继续按 `evidenceScope` 限定；
7. Controller 仍只校验直接影响状态机的核心字段，不执行任何新增语义 Gate；
8. 旧运行保持可审计，新规则只作用于新 format 或明确重新启动的实验运行；
9. 测试证明协议和状态机未扩张，同时 Skill/Ref 已包含新的语义判断主干。

## 15. 与 03 的关系

03 回答：

> 在论文软硬件不可得时，能否用弱化代理、真实单卡、局部开源实现或模拟器继续
> 验证机制？

本文回答：

> 即使代理载体有效，如何证明收益不是来自弱 baseline、参数域不公平或与简单策略
> 行为等价，而确实来自 Direction 声称的唯一主要变化？

两者共同构成 Experiment Flow 的实验有效性边界：

```text
03：载体可以弱化，但必须保留因果接口和组内公平性
04：baseline 必须有竞争力，variant 必须产生独有且可归因的增量效果
```
