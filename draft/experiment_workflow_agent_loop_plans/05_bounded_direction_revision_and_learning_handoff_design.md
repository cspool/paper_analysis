# Experiment Flow 的有界 Direction 修订与 Learning Flow 回交设计

状态：已实现（Direction Experiment format v5，政策
`COMPETITIVE_BASELINE_REVISION_V4`）。承接
`04_strongest_simple_baseline_behavioral_equivalence_and_ablation_design.md`。本设计保持
`Evidence Judge ↔ Direction Lab Goal` 两类 Agent，不增加长期角色，也不让 Controller
获得语义判断能力。它增加的是一条有边界、可版本化、可审计的 Direction 修订分支，
使可修复的实验定义问题不必立即终止整个 Experiment Loop。

实现记录：Script、协议、持久化、Prompt、验证器、两类 Skill、冻结政策、README 和
串行启动命令已经统一到同一 v5 主干。新运行默认授权 5 个实验 cycle 和 2 个 Direction
revision；旧 v2/v3/v4 只读审计。Direction Loop 的 11 个端到端场景及 Learning Loop
的 51 个回归场景均已通过。

## 1. 当前问题

format v4 已能在执行昂贵实验前识别以下问题：

- 决定性 baseline 不是最强简单 baseline；
- baseline 没有在完整合法域内公平校准；
- variant 与某个简单策略行为等价；
- 成功或失败条件没有覆盖关键等价、支配或退化情形；
- 触发条件、行为 trace 或同载体配对消融没有被定义；
- 当前代理载体无法表达冻结因果接口，需要调整实验实现方式。

但当前 `REVISE_DIRECTION` 是终态。Judge 一旦发现上述问题，Script 就结束运行并生成
`REVISION_REQUIRED` handoff。即使问题只涉及 baseline、校准、行为差异或消融定义，
也必须人工修改 Direction、重新初始化新 run，随后重新进入 Judge。

03“分任务视觉 Token 预算开关”说明了这个断点：

- 原 Direction 将未经公平校准的固定 20% 预算作为 M0；
- 同一评分器实际上支持多个全局固定预算；
- M1 若对所有任务选择 30%，会与固定 30% 简单策略行为等价；
- 核心机制“分任务选择预算”没有改变，但决定性 baseline 和失败归因需要修复；
- v4 Judge 正确发现问题，却无法在同一 Loop 内修订后继续实验。

因此需要区分两类情况：

1. **实验定义可修复**：核心 Direction 不变，可以在 Experiment Loop 内生成新 revision；
2. **研究主张需要改变**：核心优化对象或因果机制变化，必须回交 Learning Flow。

## 2. 设计结论

增加“有界 Direction 修订”，但不允许 Experiment Loop 任意重写研究方向。

```text
Evidence Judge
  ├─ RUN_LAB
  │    ↓
  │  Direction Lab Goal：执行一个冻结实验目标
  │    ↓
  │  Evidence Judge
  │
  ├─ RUN_REVISION
  │    ↓
  │  Direction Lab Goal：只形成一个新 Direction revision
  │    ↓
  │  Script 冻结 revision 和 hash
  │    ↓
  │  新 Evidence Judge Turn 独立审查
  │
  ├─ SUPPORT / REJECT
  │    ↓
  │  完成 Experiment Loop
  │
  ├─ RETURN_TO_LEARNING
  │    ↓
  │  输出带证据的 Learning Flow handoff 后结束
  │
  └─ BLOCKED
       ↓
     外部条件确实无法替代时暂停
```

Lab 的两种工作模式必须由 Script 显式注入，不能在同一个 Goal 内同时修改 Direction
并执行确认性实验：

```text
goalKind = EXPERIMENT
goalKind = DIRECTION_REVISION
```

这样仍然只有两个 Agent 职能：

- Lab 负责把冻结研究主张变成可运行、可比较的实验定义或实验实现；
- Judge 负责独立判断修订是否仍属于同一 Direction，以及证据是否支持下一步。

## 3. Direction 中哪些内容可以内部修订

### 3.1 允许内部修订的内容

`RUN_REVISION` 只适用于不改变核心因果身份的实验化修订：

- parent execution baseline、closest method baseline 和 strongest simple baseline 的关系；
- 决定性 baseline 的合法参数域、候选族和公平 calibration；
- calibration、validation、confirmatory/holdout 数据的分离方式；
- variant 相对 baseline 的唯一主要变化；
- 机制触发条件和最小可观察 action/state/execution trace；
- 行为等价、简单策略支配和零差异失败条件；
- 同载体配对实验和可独立开关组件的消融顺序；
- 代理软件、真实单卡、弱化模型、数据子集、短 trace 或模拟器的载体选择；
- 在不改变主张的前提下收窄适用范围；
- 测量接口、正确性检查、资源和统计控制；
- 在观察确认性结果前，对成功/失败条件中明显遗漏的逻辑分支进行修复。

这些修订解决的是“如何公平验证同一个 Direction”，而不是生成一个新的优化方向。

### 3.2 必须回交 Learning Flow 的变化

以下变化不得由 Experiment Loop 内部完成，Judge 必须选择 `RETURN_TO_LEARNING`：

- 更换主要优化对象或跨层因果接口；
- 更换主要机制、加入新的核心算法或引入新的决定性信息源；
- 改变 Parent Anchor、Topic 或需要覆盖的 6L 区域；
- 扩大原 Direction 的适用范围；
- 删除或弱化 Topic、Anchor 已冻结的质量、正确性、吞吐或公平性约束；
- 为适应已经看到的结果而改变主要指标、成功阈值、失败条件或统计口径；
- 在简单 baseline 已支配 variant 后，通过增加新组件来挽救原主张；
- 将一个被否证机制改写成实质不同的下一研究假设；
- 需要重新进行知识库搜索和价值审查才能确定的新方向。

若最强简单 baseline 已完全解释收益，且原 Direction 的增量价值已经消失，应优先
`REJECT`，而不是通过 `RUN_REVISION` 增加复杂度寻找正结果。

## 4. Judge 的修订职能

Judge 读取当前 active Direction revision、所有既有 Lab 结果、负证据、运行状态和冻结
政策，并先判断问题属于哪一类：

```text
同一因果主张可以通过实验定义修复吗？
  ├─ 是 → RUN_REVISION
  └─ 否
       ├─ 现有证据已证伪 → REJECT
       ├─ 需要新的研究主张 → RETURN_TO_LEARNING
       └─ 存在不可替代的外部阻塞 → BLOCKED
```

Judge 返回 `RUN_REVISION` 时只给出一个有界的修订目标，说明：

- 当前 revision 的具体缺陷；
- 必须保持不变的核心因果身份；
- 应修改的 baseline、校准、触发、消融或判定条件；
- 新 revision 通过审查所需满足的最低条件；
- 不允许借修订引入的新机制或放宽项。

Judge 不直接生成完整 Direction，也不直接修改 Script 状态。修订内容由后续 Lab Goal
形成，再由新的 Judge Turn 独立审查，避免同一个 Turn 同时提出并批准修订。

## 5. Lab 的 Direction Revision 模式

当 `goalKind = DIRECTION_REVISION` 时，Lab 只做以下工作：

1. 读取 active Direction、Parent Anchor、来源证据、当前实验政策和 Judge 的冻结
   `nextGoal`；
2. 恢复不得改变的优化对象、因果 lever、目标指标和约束；
3. 按 `nextGoal` 修订 baseline、校准、行为差异、消融、适用条件或测量计划；
4. 显式保留所有已有负证据和旧 revision 的结论边界；
5. 生成一个完整、自洽的新 Direction revision artifact；
6. 不部署环境、不修改实验代码、不运行确认性实验；
7. 输出后退出，等待独立 Judge 审查。

Revision 模式不得把“实验实现方便”写成科学主张，也不得自行扩大研究范围。若修订时
发现 Judge 目标必须改变核心机制才能完成，Lab 应在结果中声明不能形成合法 revision，
由下一 Judge Turn 选择 `RETURN_TO_LEARNING`。

## 6. 精简消息协议

当前实现已将原先只适用于 Lab 的 `labGoal` 统一为 `nextGoal`，避免同时存在
`labGoal`、`revisionGoal` 等重复字段。Judge 输出保持一个简单 JSON：

```json
{
  "decision": "RUN_LAB | RUN_REVISION | SUPPORT | REJECT | RETURN_TO_LEARNING | BLOCKED",
  "evidenceScope": "DESIGN_AUDIT_ONLY | WEAKENED_PROXY_MECHANISM | LOCAL_SINGLE_GPU_PERFORMANCE | SIMULATED_HARDWARE_MECHANISM | PAPER_EXTERNAL_VALIDITY",
  "reason": "简短判断、证据边界或回交原因",
  "nextGoal": "仅 RUN_LAB 或 RUN_REVISION 时填写"
}
```

字段规则：

- `decision` 是 Script 唯一用于选择状态机分支的语义字面量；
- `nextGoal` 在 `RUN_LAB` 和 `RUN_REVISION` 时必须非空，其他决策忽略；
- `evidenceScope = DESIGN_AUDIT_ONLY` 表示尚未产生实验结果，只完成了比较设计审计；
- 其他 scope 表示判断实际依赖的最窄实验载体，不代表完整 Direction 已获支持；
- `reason` 和 `nextGoal` 由 Agent 负责语义质量，Script 只检查存在性和字面量。

Lab Revision 的可机读 artifact 只保留状态机和版本绑定所需字段：

```json
{
  "messageType": "DIRECTION_REVISION",
  "directionId": "保持不变",
  "baseRevision": 2,
  "revision": 3,
  "reason": "本次修订解决的问题",
  "content": {
    "完整的新 Direction 内容": "使用现有 Direction 内容合同"
  }
}
```

Script 只机械检查：

- JSON 可解析；
- `messageType` 正确；
- `directionId` 与当前 active Direction 相同；
- `baseRevision` 与当前 revision 相同；
- `revision = baseRevision + 1`；
- `content` 是非空对象；
- 输出绑定当前 revision hash 和本次冻结 `nextGoal`。

baseline 是否公平、核心机制是否被偷换、成功条件是否被放宽，均由下一 Judge Turn
判断，不能交给 Script 做强语义校验。

## 7. 状态机与持久化

Direction Experiment Loop 已升级为 format v5；旧 format v2/v3/v4
保持 audit-only，不能在新状态机中续跑。

State 增加最小指针：

```text
activeDirectionRef
activeDirectionHash
activeDirectionRevision
revisionCount
latestRevisionRef
```

原始输入保持不可变：

```text
inputs/direction_result.json
inputs/parent_anchor_result.json
inputs/source_review_result.json
```

每次修订单独保存：

```text
revision_goals/revision-3/next_goal.md
revisions/revision-3/proposal.json
revisions/revision-3/direction_target.md
revisions/revision-3/record.json
```

`record.json` 保存 provider Turn、输入绑定、hash、开始/结束时间、原始输出和校验结果。
Script 接受 revision 后只更新 active 指针，不覆盖旧 revision。所有 Lab 结果继续绑定其
实际使用的 Direction revision；后续 revision 不得把旧结果伪装成新定义下的确认性
证据。

当前实现独立设置：

```text
maxDirectionRevisions = 2
maxLabCycles = 现有配置
```

Direction 修订不消耗 Lab 实验 cycle，但消耗独立 revision budget。修订授权耗尽时，
Judge 仍要求改变定义，则 Script 以 `RETURN_TO_LEARNING` handoff 结束，而不是无限
生成 revision。

## 8. 防止结果驱动的主张漂移

### 8.1 尚未观察确认性结果

在只完成设计审计、环境检查或 calibration 前，可修复 baseline、合法域、触发条件、
等价失败条件和配对消融。新 revision 必须在确认性测量前冻结并哈希。

### 8.2 已经观察确认性结果

一旦某 revision 已产生 confirmatory/holdout 结果：

- 不得回写该 revision 的成功条件、阈值或决定性 baseline；
- 不得删除、覆盖或降级已有负结果；
- 实现错误可以修复，但旧结果必须标记为对应 revision 下的无效测量并保留原因；
- 同一机制的后续诊断可以形成新 revision，但必须使用未参与修订的新 holdout、seed
  或 trace；
- 若修订是为了改变已经失败的科学主张，应回交 Learning Flow，形成新的 Direction，
  而不是在 Experiment Loop 内追逐正结果。

Judge 必须把此前负证据注入新 revision 的审查和后续 Lab Goal，避免局部反复实验。

## 9. 03 Direction 的预期闭环

对“冻结分层负载下的分任务视觉 Token 预算开关”，新 Loop 应这样运行：

```text
Judge 审计 revision 2
  ↓
RUN_REVISION / DESIGN_AUDIT_ONLY
  ↓
Lab 形成 revision 3：
  - B0 保留无压缩执行 baseline
  - 固定 20% 保留为来源方法诊断项
  - 决定性简单 baseline 改为在独立 calibration 上选择的全局固定预算族
  - M1 必须在 holdout 上优于该全局固定预算
  - 若全部任务选择同一预算且 action trace 相同，则判行为等价
  - 至少两类任务触发不同预算，才证明分任务信息被实际使用
  ↓
Script 冻结 revision 3
  ↓
新 Judge 审查 revision 3
  ├─ 仍偷换机制或比较不公平 → 再修订一次或 RETURN_TO_LEARNING
  └─ 合格 → RUN_LAB
           ↓
         baseline 复现与公平 calibration
           ↓
         行为 trace 差异检查
           ↓
         同载体配对消融
           ↓
         Judge SUPPORT / REJECT / RETURN_TO_LEARNING
```

若公平校准后的全局固定 30% 在冻结事件上与 M1 完全等价，则应直接 `REJECT` 当前
“分任务策略具有增量价值”的主张；不得通过新增另一种复杂 selector 来维持该
Direction。

## 10. Script、Skill 和文档的实现落点

### 10.1 Script

- `scripts/direction_experiment_loop/types.ts`
  - 升级 format；
  - 增加 `RUN_REVISION`、`RETURN_TO_LEARNING` 和 `DESIGN_AUDIT_ONLY`；
  - 用 `nextGoal` 取代 `labGoal`；
  - 增加 active Direction revision 指针和 revision record 类型。
- `scripts/direction_experiment_loop/protocol.ts`
  - 解析新的最小 Judge JSON；
  - 只校验决策字面量、scope、条件性 `nextGoal`；
  - 机械解析 Revision artifact 的身份与版本字段。
- `scripts/direction_experiment_loop/controller.ts`
  - 增加 `RUN_REVISION` 分支；
  - 以 `goalKind = DIRECTION_REVISION` 启动 Lab；
  - 冻结 revision 后回到新 Judge Turn；
  - `RETURN_TO_LEARNING` 生成终态 handoff；
  - 分离 revision budget 与 Lab cycle budget。
- `scripts/direction_experiment_loop/prompts.ts`
  - Judge 始终读取 active revision、完整历史和负证据；
  - Lab Prompt 明确注入 `goalKind`；
  - Revision 模式禁止运行实验，Experiment 模式禁止改写 Direction。
- `scripts/direction_experiment_loop/setup.ts`
  - 初始化 revision 0/源 revision 的 active 指针与独立预算。
- `scripts/direction_experiment_loop/validation.ts`
  - 校验 active 指针、revision 链、artifact hash 和状态机引用；
  - 不增加 baseline 或机制的脚本语义校验。
- `scripts/direction_experiment_loop/store.ts`
  - 保存 revision goal、proposal、record、机械投影和事件索引。
- `scripts/direction_experiment_loop/contracts/experiment_policy.md`
  - 将“修改 Direction 一律终止”改为“实验定义内部修订、核心主张回交 Learning”的
    单一主干政策。

### 10.2 Agent Skills

- `.codex/skills/direction-evidence-judge/SKILL.md`
  - 增加内部修订与回交 Learning 的判别方法；
  - 禁止为了正结果反复修订；
  - 对新 revision 做独立的 baseline、因果身份和阈值漂移审查。
- `.codex/skills/direction-evidence-judge/references/judgment_contract.md`
  - 定义六种决策、`DESIGN_AUDIT_ONLY` 和修订收敛规则。
- `.codex/skills/direction-lab-goal/SKILL.md`
  - 将两种 `goalKind` 作为互斥主流程；
  - Revision 模式只生成新 Direction artifact；
  - Experiment 模式只执行冻结 revision。
- `.codex/skills/direction-lab-goal/references/lab_method.md`
  - 增加 revision 方法、不可修改项和已有负证据继承要求。

### 10.3 README 与命令文档

- 更新 Direction Loop 状态图；
- 说明 revision budget、Lab cycle budget 和 resume 语义；
- 说明旧格式只能审计；
- `serial_direction_learning_flow_commands.md` 使用新格式的新 work-dir，不能复用 v4
  已完成目录。

## 11. 必须覆盖的测试

1. Judge 返回 `RUN_REVISION` 后只启动 Revision 模式 Lab，不启动实验。
2. 合法 revision 被冻结为 `baseRevision + 1`，随后自动进入新的 Judge Turn。
3. stale `baseRevision`、错误 `directionId` 或缺失 `content` 被机械拒绝并以冻结错误信息
   重试。
4. 新 Judge 可以拒绝偷换因果机制的 revision，并选择 `RETURN_TO_LEARNING`。
5. `RUN_LAB` 和 `RUN_REVISION` 都使用 `nextGoal`，其他决策不依赖该字段。
6. 无实验的设计审计保存为 `DESIGN_AUDIT_ONLY`，不再误标为弱代理实验结果。
7. revision budget 与 Lab cycle budget 独立，任一预算耗尽不会隐式增加另一预算。
8. 达到最大 revision 次数后不会循环修订，而是生成 Learning handoff。
9. 已有 Lab 结果始终绑定旧 revision；新 revision 不覆盖旧 artifact、hash 或负结论。
10. 观察确认性结果后修改成功阈值不能被自动接受，Judge 必须回交 Learning Flow。
11. `SUPPORT` 仍要求当前 active revision 的竞争性 baseline、行为差异和配对消融证据。
12. 旧 format v2/v3/v4 运行在新实现中保持 audit-only。

## 12. 完成标准

format v5 的实现验收标准如下：

- Experiment Loop 能自行修复同一 Direction 内的 baseline、校准、行为差异和消融定义；
- 核心机制变化仍由 Learning Flow 负责；
- Script 不承担语义修订或科学判断；
- Lab 不在同一 Goal 中一边改主张一边跑确认性实验；
- 每次 revision 都版本化、不可覆盖、可追溯并绑定旧证据；
- 负结果不能通过改阈值或改 baseline 被抹除；
- 可修复的设计问题不再像 03 一样立即终止；
- 被简单策略等价或支配的 Direction 仍能快速 `REJECT`，不会借修订进入无限实验。
