## UDP / UFTQ（Utility-Driven Fetch Directed Instruction Prefetching）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
UDP（Oh et al., ISCA'24，UCSC/Columbia）提出两个机制优化 FDIP：UDP 过滤"效用低"的 FDIP 预取（utility-driven filtering），UFTQ 动态调整 FTQ 尺寸以平衡预取激进程度与精确度；核心观察是部分 off-path（错误路径）预取在控制流重新汇聚（merge point）到正确路径后仍然有用，因此不应一刀切关闭。Bumper 论文复现两个变体：FTQ-Size-OPT（每应用离线搜索的最优静态 FTQ 尺寸，作为 UFTQ 动态尺寸的上界参照，平均 +1.1% 加速）与 UDP（用 TAGE 分支预测置信度过滤 FDIP 预取，平均 +1.2%）。UDP 收益有限的根因：BTB miss 时该分支根本没有 TAGE 预测，置信度无法用于节流——而移动应用的 BPU MPKI 主要由 BTB miss 构成，错误路径预取大量来自此。Bumper 较 FTQ-Size-OPT 高 5.4%。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
流程：FTQ 的 Prefetch Head 发预取前 → UDP 过滤器评估该预取的效用/置信度 → 低效用则不发出（减少 useless 行进入 L2C）；UFTQ 侧则根据效用与及时性比例动态伸缩 FTQ 允许的预取窗口。Bumper 与 FTQ-Size-OPT 组合后平均最优 FTQ 从 26 增至 51：Bumper 通过快速淘汰 useless 行使更激进的 FDIP 变得划算，说明两者互补（Bumper+FTQ-Size-OPT 额外 +0.8%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：基于置信度/效用计数器的过滤逻辑 + FTQ 尺寸调节状态机，硬件代价约 8KB；原论文在开源 Scarab 模拟器（Sunny Cove 类核 + TAGE-SC-L）上验证。Bumper 论文的教训：预取侧过滤的"误伤代价不对称"（滤掉少量 useful 请求即严重损失性能），是选择在缓存侧做 commit 驱动的淘汰（Bumper）而非请求侧过滤的原因。

涉及论文标题：
- Bumper: Hinting Instruction Usefulness for Robust Unified Caches
