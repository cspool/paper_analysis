## Fetch Directed Instruction Prefetching（FDIP，取指定向指令预取）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FDIP（Reinman et al., MICRO'99）利用解耦前端：在 FTQ 的 Fetch Head 之前增设 Prefetch Head，当取指流水线未占满 L1I 带宽时，沿 FTQ 中更远（更前瞻）的 fetch block 提前向 L1I 发指令预取，把 L1I miss 隐藏到需求访问之前，降低取指停顿、提升前端吞吐。Bumper 的 baseline 采用 FDIP；其有效性依赖 BPU 精度：移动应用 BPU MPKI 平均 8.0（主因 BTB 容量 miss），错误路径上的 FDIP 预取把大量 useless 代码行带入统一 L2C（平均 20.3% 容量）。关键实验事实：完全禁用 FDIP 使 useless 代码行占比从 20.3% 降到 4.0%，但性能 -14.6%——FDIP 收益巨大不能简单关闭，只能事后清理其污染（这正是 Bumper 的定位：Bumper 使 FDIP 到 L2C 的预取流量平均 -5.7%，并允许更大 FTQ 深度）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
流程：FTQ 追加新 fetch block → 取指流水线未占满 L1I 带宽（有空闲 miss 资源/带宽）→ Prefetch Head 沿 FTQ 发预取请求 → L1I miss → 向 L2C 请求 → Bumper 下以 RRPV=3 安装（l2_vulnerable_fill=1）→ 若该行在错误路径上则永不被提升、快速淘汰。Bumper 与 FDIP 正交且互补：Bumper 不改变预取决策，只加速淘汰 FDIP 错误路径预取带来的 useless 行，从而提升有用行生命周期（+52.5%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：FTQ 上维护第二指针（Prefetch Head）+ 与取指请求的带宽仲裁逻辑，预取不占用需求取指的关键资源。改进变体：UDP/UFTQ（Oh et al., ISCA'24）用效用（utility）过滤与动态 FTQ 尺寸调节；Bumper 论文还评估了"连续若干 fall-through fetch block 后节流预取"与 UDP 式节流，均收益有限——因为过滤 useful 与 useless 预取的代价高度不对称（误滤少量 useful 请求即严重伤害性能）。

涉及论文标题：
- Bumper: Hinting Instruction Usefulness for Robust Unified Caches
