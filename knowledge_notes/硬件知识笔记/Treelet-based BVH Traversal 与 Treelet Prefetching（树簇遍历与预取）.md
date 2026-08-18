## Treelet-based BVH Traversal 与 Treelet Prefetching（树簇遍历与预取）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- treelet（树簇）是 BVH 遍历的访存优化范式：把全局 BVH 划分成若干紧凑子树（treelet），每个 treelet 尺寸适配 SM 的 L1 cache，节点在内存中连续存放，可整块从全局内存批量预取，改善节点访问的局部性与缓存复用，降低每节点访存延迟。Treelet Prefetching（Chou et al., MICRO 2023 [47]；Treelet Accelerated Ray Tracing on GPUs, ASPLOS 2025 [48]）是其典型硬件实现。GauTracer 把它作为 baseline RTA 的组成部分（与 [48] 一致），并在此基础上叠加 far-node pruning。
- 从硬件架构角度拆解术语，给出运转流程具体例子（GauTracer IV-D）：①treelet 划分：BFS 方式构建，每内部节点的子节点要么并入同一 treelet，要么提升为子 treelet 的根；叶节点并入父 treelet。②treelet 控制器：counter table 跟踪每 treelet 的射线人口，warp buffer 内的射线完整处理当前 treelet 后才切换到下一个，调度由 counter table 驱动。③遍历栈两级化：treelet stack（子 treelet 根）+ node stack（当前 treelet 内子节点），都存每射线寄存器，深度超限按 short-stack 策略溢到 local memory。④配置：每 treelet 256 节点 × 64B = 16KB，适配 64KB L1，ping-pong 双缓冲隐藏预取延迟。作用：GauTracer 基线 profiling 显示带 treelet 后 BVH 遍历大幅加速，shader 执行成为主瓶颈（占 72.9%），从而引出硬件 shader（RGIU/AGHU）的必要性。
- 术语一般如何实现？如何使用？：开源参考实现为 ubc-aamodt-group/treelet-prefetching-for-rt（MICRO 2023，https://github.com/ubc-aamodt-group/treelet-prefetching-for-rt）。GauTracer 在 Vulkan-Sim 的遍历逻辑中集成 treelet-based stack loop、修改 BVH 节点打包/解码；paper 把 treelet 配置与 far-node pruning 结合，并评估 closest-first 子节点排序（PRUNE+SORT）——排序带来的收益很小甚至降级（上层节点包围体重叠导致判断失误），说明 treelet 布局质量与自适应 closest-first 是未来方向。

TTP 补充视角（ISCA'26，与 Treelet 的对比）：Treelet Prefetcher（MICRO 2023）需要 (1) 用 treelet 遍历算法直接替换 DFS，(2) 预处理重排 BVH 树，(3) 在 BVH 节点加 treelet 归属信息，treelet 根被读时整体预取整个 treelet——软件/树格式改动大且预取过量。TTP 完全不需要这些：不改遍历算法、不改树组织、对软件透明，只在 RT unit 栈旁加 2-bit FSM。模拟对比（Vulkan-sim，LumiBench path tracing）：128x128 下 TTP 平均 1.48x vs Treelet 1.00x（ship/spnza/crnvl/fox 场景 Treelet 反而回退）；32x32 下 1.44x vs 1.14x；64x64 下 1.49x vs 1.00x；Treelet 的 DRAM 总流量增 1.38x（spnza 带宽 58%→80%），TTP 的 DRAM 流量不变；chsnt 场景 Treelet 开启即崩溃（故排除）。
涉及论文标题：
- GauTracer: Extending Ray Tracing Accelerator for Gaussian-based Scene Representation
- TTP A Hardware-Efficient Design for Precise Prefetching in Ray Tracing
