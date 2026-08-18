## TTP（Tree Traversal Prefetcher，树遍历预取器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- TTP 是面向 GPU 光线追踪 RT unit 的专用硬件预取器（NCSU，ISCA'26）：BVH 遍历是 memory-bound 负载，RT unit 内线程大部分周期在等节点数据读返回（100-300 cycles miss），而多数场景 DRAM 带宽未用满——存在 latency 隐藏空间。TTP 的核心洞察是"向上遍历的下一节点地址已经在硬件遍历栈里，预取零推测"：完全复用 RT unit 中每线程已有的遍历栈（栈存节点地址而非数据），不需要地址预测器、不需要改遍历算法、不需要改 BVH 树格式。DFS 模式用 2-bit 有限状态机（FSM）监控栈 push/pop：连续第 1 个 pop 发 1 个预取（栈顶），第 2 个连续 pop 发 2 个（栈顶 2 个），第 3 个发最多 16 个（栈顶 16 个）；一次 push 重置 pop streak 并停发——因为向下遍历（连续 push）的下一个节点地址取决于交点测试结果、无法提前得知，强行预取会产生错误路径带宽浪费。BFS 模式更简单：每次从 FIFO 队头 pop 时从队头预取 N 个节点（N=预取距离，实验 1/2/4，N=4 最优）。实现硬件：每线程 2-bit FSM 字段（加入 warp buffer）+ 预取指针（指向下一个待预取栈项，push 时重置回栈顶 T）+ 比较器（指针到 T−k 停止，k=1/2/16）；也可为栈项加 flag bit 记录已预取。默认仲裁 demand read 优先，预取在无 demand 时发送；节点 >32B 时按 sector（32B）拆块逐周期发送。评估：Vulkan-sim 2.0，DFS 平均 1.48x（峰值 1.89x）、功率 1.35x、能量 -8.70%；L1/L2 accuracy 98.92%/89.81%、coverage 31.54%/33.46%；带宽开销 18.22% 但 DRAM 总流量不变；面积用 FreePDK45 综合，128 状态机/SM 仅 1117 cells（每状态机 8.7 cells），对比每线程 192-bit 射线属性可忽略。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- TTP 在 RT unit 中的运转流程（DFS）：①warp 内某线程弹栈（pop）读节点 → ②该 pop 触发其 TTP FSM 状态迁移（S1/S2/S3，2-bit）；③FSM 计算 T−k（T=栈顶，k=1/2/16）作为本次预取深度，预取指针从栈顶向下取值 → ④demand read 空闲时把栈中节点地址作为预取请求发出（>32B 拆 32B sector 块，每周期一个）→ ⑤预取请求与 demand 共享内存访问队列送 L1/L2/DRAM，提前填充缓存 → ⑥等该节点真正被 pop 读取时命中 L1，miss 延迟被隐藏。一次 push 把指针重置回 T 防止重复预取。BFS 模式：队头 pop 时直接用队头后 N 个地址发预取。以图 5 的示例：O 在 P 被 pop 时预取（S1），随后 O 被 pop 时预取 N、L（S2）。TTP 是每线程引擎，预取只在 warp 被 RT 调度器选中时发送，配合 warp buffer（4 warps×32 线程）内 128 个 FSM 并行工作。limit study（perfect upward 1.79x vs perfect downward 1.35x）验证了"向上遍历预取是主要收益来源"的机理。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：在 Vulkan-sim 2.0 的 RT unit warp buffer 中新增每线程 2-bit FSM、预取指针与比较器（TTP 论文修改 github.com/yavuz650/vulkansim，论文声明；artifact 为 Docker 镜像 ttp-isca2026-ae:1.0，Zenodo DOI 10.5281/zenodo.19394324）。面积评估用 FreePDK45 综合 FSM。使用：完全对软件透明（无需改 shader/BVH/API），硬件开关式启用；与 Treelet prefetcher（需改遍历算法+重排树）和 Park et al.（叶子交点测试窗口内预取，触发不足仅 1.04x）对比均占优。用法要点：参数敏感度低——预取强度（1/2/16）与仲裁阈值（25/50/100 cycles）对结果几乎无影响，BFS 的 N 取 4 即可；对更大 L1（64KB/128KB，1.44x）、更大 GPU（30SM/64KB L1/3MB L2，1.50x）、更高分辨率（256x256，1.44x）、AO/SH shader（1.22x/1.18x）都保持稳健。限制：只针对 closest-hit/path tracing 的长遍历收益最大；wknd 等简单场景提升有限；TTP 修改版开源仓库未独立验证（论文声明为准）。

涉及论文标题：
- TTP A Hardware-Efficient Design for Precise Prefetching in Ray Tracing
