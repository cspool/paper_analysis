# Loop 需求闭合设计

状态：format version 5 已采用；最小在线校验边界以 05 为准，Decision 全局
外循环、最小充分 Direction 和 Script 派生观察以 06 为准。  
闭合范围：`draft/todo_draft.md` 第 19–35 行的 Agent 映射白名单。

## 1. 闭合对象

工作流最终形成：

```text
Topic
  └─ Anchor 集合：动态定义当前 Topic 的 6L 空间
       └─ Direction 集合：每个 Anchor 内可证伪的性能优化路径
```

Anchor 包含：

- 具体场景；
- baseline；
- 可观察性能矛盾；
- 非空 6L 区域；
- 适用约束。

Direction 包含：

- 可修改对象和因果机制；
- 相对 bound Anchor baseline 的主要变化；
- 条件化预期影响；
- 权衡和失败条件；
- 可证伪测量计划。

最终采用的 Anchor 和 Direction 都必须有独立 Reviewer `PASS`。

## 2. 固定 Loop

首次启动：

```text
Script
  → CREATE_ANCHOR Worker
  → REVIEW_ANCHOR Reviewer
  → Decision
```

正常分支：

```text
RUN_WORKER
  → commit pending
  → Worker → Reviewer → Decision

RUN_REVIEWER
  → commit pending
  → pre-review
  → commit pre-review as the target revision's current R01
  → Worker → Reviewer → Decision

FINISH_WORKFLOW
  → commit pending
  → requirement 为空
  → deterministic render
  → stop
```

语义恢复：

```text
RETRY_WORKER
  → supersede pending Worker/Reviewer
  → same Worker TaskBinding → Reviewer → Decision

RETRY_REVIEWER
  → retain Worker
  → supersede Reviewer
  → same Reviewer TaskBinding → Decision
```

每条非完成分支最终都回到 Decision；这是 Loop，不是线性阶段链。

## 3. Decision 在 Loop 中的位置

Decision 总是在一个 JSON 和核心控制字面量合法的 Worker/Reviewer 结论对
之后触发。完整正文仍可能偏离 Ref 或存在语义错误。它
获得：

- G01 最终需求；
- 全部最新 committed Worker/Reviewer 结论；
- 当前 pending T01、W01、R01；
- pending 提交后的 remaining requirements；
- 本次允许 Decision。

Decision 的输出只确定下一条状态机分支：

```text
RUN_WORKER
RUN_REVIEWER
FINISH_WORKFLOW
RETRY_WORKER
RETRY_REVIEWER
```

“增加 Anchor”“增加 Direction”“继续深入当前对象”“从新角度审阅”都只能
作为 guidance。Script 不解析 guidance，而是按状态和 requirement 机械绑定
下一 T01。

## 4. Script 的机械 requirement

Script 只维护：

```text
ANCHOR_REQUIRED
ANCHOR_REVIEW_PASS_REQUIRED:<anchor-result-ref>
DIRECTION_REQUIRED:<anchor-result-ref>
DIRECTION_REVIEW_PASS_REQUIRED:<direction-result-ref>
```

关闭规则：

- 至少存在一个未拒绝 Anchor；
- Anchor 的最新 W01 为 `READY_FOR_REVIEW` 且 R01 为 `PASS`；
- 每个通过的 Anchor 至少存在一个未拒绝 Direction；
- Direction 的最新 W01 为 `READY_FOR_REVIEW` 且 R01 为 `PASS`。

`REVISE` 保持对应 pass requirement 打开；`REJECT` 将该对象排除出可闭合集合。

## 5. Script 的机械 Worker 选择

需要 Worker 时按固定优先级：

1. pre-review 指向一个对象：深化该对象；若被拒绝则创建同类替代；
2. active Anchor 尚未通过：深化该 Anchor；
3. active Direction 尚未通过：深化该 Direction；
4. 没有 active Anchor：创建 Anchor；
5. 通过的 Anchor 没有 Direction：为其创建 Direction；
6. 最低闭合已经满足而 Decision 仍选择 `RUN_WORKER`：创建新 Anchor。

第 6 条使 Anchor 集合能够继续扩展 Topic 的 6L 空间，同时不要求 Script
理解具体专业方向。

第 1 条使用的 pre-review 已先成为该对象当前版本的权威 R01。若其 verdict
为 `REJECT`，旧对象立即退出 active/可闭合集合，替代 Worker 的
`inputs.currentWork` 和 `inputs.latestReview` 分别指向被拒绝的 W01 与当前
R01；旧的 `PASS` 不得继续控制最终报告。

## 6. 19–21：多维知识查询

需求：

- 知识库包含 experiment、idea、knowledge、human 等表达维度；
- 不同缺口查询不同维度；
- 需要决定查询时机和关键词。

闭合方式：

```text
Worker 读取 T01 和现有 R01
  → 根据当前缺失字段或 query gap 选择维度
  → 构造当前对象的技术对象、场景、baseline、性能关系关键词
  → 使用 Obsidian omnisearch
  → 深读命中笔记
  → W01 evidence 保存实际 sourceRef

Reviewer 独立审阅 W01
  → 提出可能改变 verdict 的对象局部 query gaps
  → 标记 experiment / idea / knowledge / human resolution channels
```

Script 只保存任务、结果、引用和事件，不生成查询词，也不解释来源。

## 7. 23：Topic 相关性能优化潜力

```text
G01 限定 Topic 与最终目标
→ Worker 创建 Anchor
→ Reviewer 审阅其场景、baseline、性能矛盾、证据和 6L 区域
→ Worker 在通过的 Anchor 内创建 Direction
→ Reviewer 审阅机制、影响、权衡、失败条件和测量计划
→ Decision 判断是否继续扩展、增加审阅或完成
```

最终潜力不是一次 Agent 输出，而是多个已通过 Anchor/Direction 的闭环集合。

## 8. 24：人类可读最终产物

Decision 只有在 Script 注入 `FINISH_WORKFLOW` 时才能选择完成。Script 仅在
pending 提交后的 requirement 为空时提供该选项。

完成路径：

```text
FINISH_WORKFLOW
→ 幂等提交 pending
→ 再计算 requirement
→ 读取最终 PASS 的 W01/R01
→ 容错且确定性地渲染 final/report.md
→ 写 final/manifest.json
→ 写 O01
```

报告按 Anchor → Direction 层次组织，包含 baseline、性能矛盾、机制、预期
影响、权衡、失败条件、测量计划、证据和非阻塞审阅 caveats。
缺少可选 Ref 字段不能使完成路径崩溃；非标准正文保真写入 JSON 附录，而不
由 Renderer 重新作语义判定。

不再启动单独的 Final Report Agent。

## 9. 25：过程输出和通信

过程分为：

```text
Agent 通信：
G01 / T01 / D01 / W01 / R01 / E01

调用者结果：
O01

Controller 内部：
Run / TaskBinding / Turn / State / Object / Round /
Validation / Event / Runtime / FinalManifest
```

Agent 不回显 Controller 已知的 ID、revision、Round、Attempt、哈希或预算。

## 10. 28–29：Anchor 集合定义动态 6L 空间

```text
Topic6LSpace(current) =
  union(scope6L of every active, non-rejected Anchor)
```

Anchor 不是预先固定的六层清单，而是“场景 + baseline + 性能矛盾”定义的区域
中心。一个 Anchor 可覆盖任意非空 L1–L6 子集。

当最低 requirement 已经闭合但 Decision 仍选择 `RUN_WORKER`，Script 机械
创建新 Anchor Task；具体区域由 Worker 根据 Topic 和证据形成，再由 Reviewer
审阅。

## 11. 30：Direction

Direction 永远绑定一个 Anchor：

```text
CREATE_DIRECTION
  inputs.boundAnchor = <latest Anchor Work Result>

DEEPEN_DIRECTION
  inputs.boundAnchor = <latest Anchor Work Result>
  inputs.currentWork = <latest Direction Work Result>
  inputs.latestReview = <latest Direction Review Result>
```

TaskBinding 内部保存对象 ID、revision 和父 Anchor。Agent-visible JSON 不回显
这些控制字段。

## 12. 31：多轮搜索—思考—审阅

每个正常 Decision 增加一个 Round：

```text
Round n
  Worker/Reviewer 产生 pending
  → Decision
  → commit
  → Round n+1
```

格式/core-control 重试和语义重试只增加 Attempt，不增加 Round。所有
Attempt、原始输出和
替代关系保留审计。

## 13. 33–35：长期运行和持久记忆

持久存在：

- Controller state；
- workflow goal；
- TaskBinding；
- Agent 原始输出和已解析结果；
- 每个成功 Turn 的核心控制投影；
- Object/Revision 和 Round 索引；
- ValidationAudit；
- append-only EventLog；
- RuntimeLog；
- FinalManifest。

临时存在：

- Decision Turn；
- Worker Turn；
- Reviewer Turn；
- Turn 内部可选辅助 Agent。

中断恢复：

- 已捕获输出：本地重放和最小控制校验；
- 未捕获输出：原 Turn 记为 `RUNTIME_FAILED`，同一绑定启动 fresh Attempt；
- 协议/JSON/core-control 不合法：E01 + 同一冻结输入；
- core-valid 但 workflow 语义错误：Decision 正式 retry；
- Skill 保存方法，不保存运行状态。

## 14. 闭合证明

| todo | Worker | Reviewer | Decision | Script |
|---|---|---|---|---|
| 19–21 | 选维度、构造词、查询、深读 | 定义对象局部 query gaps | 判断是否继续内容/审阅 | 保存引用和结果 |
| 23 | 形成 Anchor/Direction | 审阅专业成立性 | 判断整体是否继续 | 维护对象集合 |
| 24 | 提供结构化内容 | 提供独立结论 | 选择完成 | 确定性人类报告 |
| 25 | W01 | R01 | 行协议 | G01/T01/D01/E01/O01 与内部审计 |
| 28–29 | 形成具体 Anchor 区域 | 审阅范围 | 决定是否继续扩展 | Anchor 集合定义动态 6L |
| 30 | 形成 Direction | 审阅 Direction | 决定下一分支 | 固定父 Anchor 绑定 |
| 31 | 每轮内容工作 | 每轮独立审阅 | 每轮回到决策点 | 固定回边和 Round |
| 33–35 | fresh Turn | fresh Turn | fresh Turn | 持久状态、顺序调度和恢复 |

当 O01 为 `FINISHED` 时，Script 能机械证明：

- 至少一个最终 Anchor；
- 每个最终 Anchor 的 W01 ready 且 R01 pass；
- 每个最终 Anchor 至少一个最终 Direction；
- 每个最终 Direction 的 W01 ready 且 R01 pass；
- 最终报告只引用这些已提交结果。
