## Information-Revealing Demonstrations for SFT

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Information-Revealing Demonstrations（信息揭示型示范轨迹）是VisGym提出的SFT数据筛选策略，针对部分可观察（POMDP）或未知动态（unknown dynamics）环境的VLM训练。核心洞察：标准demonstration仅展示到达目标的动作序列，但在未知状态转换规则和隐藏信息时，VLM无法从中学习状态表示。信息揭示型demonstration在到达目标前先执行结构化探索步骤暴露隐藏状态或环境动态，使VLM学习更准确的状态表示。

从算法pipeline角度拆解术语，给出具体例子。
两个关键案例（Sec 5.4）：

**Matchstick Rotation（未知动态/unknown scale）**：
- 标准：直接三次stochastic moves到目标 → 成功率32.9%
- 信息揭示型：先两次unit-scale探索步骤（'move', [1,0,0]和'move', [0,1,0]）暴露scale对应关系 → 最后对齐 → 成功率70.0%（+37.1%）

**Mental Rotation 3D Objaverse（部分可观察）**：
- 标准（Solve-Only）：沿每轴旋转一次直接对齐
- 信息揭示型（Rotate-Then-Solve）：先完整旋转每轴暴露3D几何，再对齐
- 验证：在Rotate-Then-Solve模型上继续训练Solve-Only → 性能恶化，确认改善来自demonstration的信息结构而非长度/数量

```
# 信息揭示型demonstration的通用结构：
# Phase 1: Exploration
for each unknown/hidden dimension:
    execute exploratory action (unit move, full rotation)
    observe effect → learn dynamics / expose geometry
# Phase 2: Exploitation
compute and execute goal-directed actions
stop()
```

术语一般如何实现？如何使用？
通过修改solver策略实现（配置为先exploration再solution）。仅用于partial observability或unknown dynamics任务（fully observable + known dynamics任务的标准demonstration已足够）。信息揭示型demonstration帮助VLM"学会学习"环境动态，而非仅模仿动作序列。

涉及论文标题：
- VisGym: Diverse, Customizable, Scalable Environments for Multimodal Agents
