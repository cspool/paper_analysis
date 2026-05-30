## Tutel Adaptive Mixture-of-Experts at Scale

- baseline方法是什么？
  Baseline 是 **Fairseq MoE / DeepSpeed MoE 的静态执行框架**（Ott et al., 2019; Rajbhandari et al., 2022），遵循 GShard 计算逻辑（Lepikhin et al., 2021）。全栈执行例子：
  - **模型推理算法层**：GShard 风格的 Top-K 稀疏门控 MoE。Token t 经 gate = Softmax(W_g · x_t) 计算 E 个专家的路由概率，TopK 选出 K 个专家，dispatch → expert FFN → combine。Baseline 使用静态 capacity factor f = f_upper（固定上界），导致：(a) f 偏大时浪费计算；(b) f 偏小时丢弃 token。
  - **系统框架层**：Fairseq/DeepSpeed MoE 采用固定并行策略（如 EP+DP），运行时不可切换。切换到不同并行策略需要：不同的张量分片布局、参数迁移开销（Figure 4）、框架接口变更。并行策略在所有训练步中保持静态，不适应动态变化的 expert capacity。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：使用 NCCL 的 Linear All-to-All（Algorithm 1），所有 GPU 直接 P2P 通信。Dispatch/Combine 的 encode/decode 使用稠密 einsum 实现（Figure 20a），时间复杂度 O(T·E·C_g·D)，内存消耗大（Table 5: 32,768 tokens/step 时 57.9 GiB）。All-to-All 与 Expert FFN 顺序执行，无通信计算重叠。
  - **硬件架构层**：NVIDIA A100 80GB GPU + HDR InfiniBand。Linear All-to-All 在 scale-out 时消息大小 S/n 变得过小，无法饱和 InfiniBand 链路带宽（Figure 16），All-to-All 通信开销占比从 16 GPUs 的 33.7% 增长到 256 GPUs 的 56.7%（Table 2）。

  Baseline 的核心缺陷：
  (1) **静态并行不适应动态负载**：MoE 的 expert capacity 随训练步动态变化（实测 4.38× 波动，Figure 1），而 Fairseq/DeepSpeed 固定使用一种并行策略，不同并行策略在不同 capacity 下有 7.39%~27.76% 的性能差距（Figure 3）。
  (2) **静态流水线度/All-to-All 算法低效**：不同 scale 和模型配置的最优流水线策略不同（Figure 5），静态策略导致最坏情况下 23%~599% 的性能损失（Table 6b）。
  (3) **密集 encode/decode 计算冗余**：稠密 einsum 包含大量零乘加，且消耗大量 GPU 内存。
  (4) **Linear All-to-All 不可扩展**：大规模下小消息无法饱和网络带宽。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **TUTEL 自适应 MoE 全栈系统**，通过统一张量布局实现零成本并行切换、自适应流水线、稀疏 GPU kernel 和层次化 All-to-All。全栈执行例子：
  - **模型推理算法层**：保持 GShard 计算逻辑不变（数学等价），支持动态 capacity factor（capacity_setting 参数控制：正值=固定值，0=自适应最小不丢 token，负值=带上限自适应，Figure 10），以及动态 Top-ANY 路由（每步可调整 k 值）。
  - **系统框架层**：自适应并行切换——基于 ZeRO-DP Stage-3 风格的统一张量分片布局，DP（r=0）和 EP+DP+MP（r∈[1,⌈W/E⌉]）共享相同的 weight slicing 和 data layout（Figure 6/7/8）。运行时通过 O(1) 字典查表选择最优 r，无需参数迁移或数据重整（零成本切换）。通过通信复杂度分析（Table 4）将 7 种并行策略化简为 2 种。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：(a) Fast Encode/Decode（K0/K1/K2 CUDA kernels）——稀疏 SIMT-efficient 实现，每个 warp 处理一个 token，利用 warp shuffling、Blelloch scan、half2 向量化，复杂度从 O(T·E·C_g·D) 降至 O(T·k·D)，GPU 内存节省 20%~90%（Table 9）；(b) Flexible All-to-All——输出 layout 从 (W, E_g, C_g, D) 变为 (E_g, C, D)，消除 scale 对 expert matmul 的影响（Figure 11）；(c) 2DH All-to-All——4-phase 算法（stride memcpy → intra-node A2A → stride memcpy → inter-node A2A, Figure 17），聚合小消息为大消息，在大规模下大幅降低延迟（Figure 18），支持 MSCCL 编译优化和 LL128 协议；(d) 自适应多流流水线——token capacity 维度分区 + 多 CUDA stream 异步执行，重叠 All-to-All 通信与 Expert FFN 计算，动态选择流水线度 d∈{1,2,4,8} 和算法 a∈{Linear,2DH}。
  - **硬件架构层**：NVIDIA A100 GPU + HDR InfiniBand，NCCL 2.10.3-1 + RDMA SHARP plugin。2DH All-to-All 通过聚合小消息提升 InfiniBand 带宽利用率。

  **设计思路核心映射**：
  - 缺陷(1) "静态并行" → 方案：统一张量布局 + 零成本自适应切换 → 1.35×~14.57× MoE 层加速
  - 缺陷(2) "静态流水线" → 方案：字典式最优策略查找 + 多流异步调度 → 平均 9%~101% 提升，最坏情况 23%~599% 提升
  - 缺陷(3) "密集 encode/decode" → 方案：SIMT-efficient 稀疏 CUDA kernel (K0/K1/K2) → kernel 加速 + 20%~90% 内存节省
  - 缺陷(4) "Linear A2A 不可扩展" → 方案：2DH 层次化 All-to-All + Flexible layout + MSCCL 编译优化 → 大规模下 4.25× 提升（2,048 GPUs）
  - 最终效果：2,048 GPUs 上单 MoE 层 5.75× speedup（vs Fairseq），SwinV2-MoE 端到端训练 1.55× 推理 2.11× 加速
