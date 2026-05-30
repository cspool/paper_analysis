## Efficient and Adaptable Overlapping for Computation and Communication via Signaling and Reordering

- baseline方法是什么？
  两种现有 computation-communication overlap 方法：(i) **Decomposition-based**（CoCoNet、Async-TP、Domino、Centauri 等）——将 GEMM 输出 tensor 分解为多个 subtensor，异步交替执行第 i 个 subtensor 的通信和第 i+1 个 subtensor 的计算。为确保数据地址连续以直接调用通信 API（如 NCCL），分解仅限于单一维度，与 GEMM tile 的二维分块范式不对齐，无法实现 tile 级细粒度重叠；且小 GEMM shape 时碎片化计算无法充分利用 GPU 资源。(ii) **Fusion-based**（FLUX、Comet、TileLink、cuBLASMp 等）——将通信原语直接融合进 GEMM kernel 内部，通过指令调度实现 tile 级重叠。但需要为每种通信原语手动实现定制融合 kernel（AllReduce、ReduceScatter、All-to-All 各需独立实现），且融合时因协调计算和通信 pipeline 可能需修改 tiling 策略或计算逻辑导致性能退化。

  全栈执行例子（以 Llama3-70B TP=8 推理中 GEMM+AllReduce，A800 GPUs，使用 Decomposition-based Async-TP baseline 为例）：
  - **模型推理算法层**：TP=8 下每 GPU 计算 GEMM 部分结果 → AllReduce 求和得到完整结果。
  - **系统框架层**：PyTorch 调用 Async-TP，将 GEMM 输出沿单维度分解为多个 subtensor。框架层负责管理 subtensor 的通信调度和 CUDA stream 同步。论文未明确说明上层 serving 框架修改。
  - **编译框架层**：论文未明确说明。使用标准 cuBLAS + NCCL API 调用，无编译优化。
  - **kernel调度层**：GEMM 被分解为多个独立的小 GEMM kernel（fragmented GEMMs），每个小 GEMM 调用 cuBLAS 执行。Subtensor 通信通过 NCCL API。第 i 个 subtensor 的 NCCL 通信与第 i+1 个 subtensor 的 cuBLAS GEMM 并发执行。但分解为 subtensor 时仅沿单一维度（保证地址连续），而 GEMM tile 是二维分块——tile 完成顺序（wave pattern）与分解维度不匹配，先完成的 tile 不能立即通信，必须等整个 subtensor 完成。Tile 级重叠机会丧失。小 K 值时碎片化 GEMM 无法填满 SM。
  - **硬件架构层**：A800 GPU，NVLink 互联。GEMM 在 Tensor Core 上执行，通信通过 NVLink。Decomposition-based 的 subtensor 通信数据量较小，可能导致带宽利用不足。

  Baseline 核心缺陷：
  - (a) Decomposition-based **无 tile-wise overlapping**：限于单维分解，与 tile 的 2D 分区不对齐，无法利用 tile 是最小并行数据单元的事实。已完成 tile 无法立即触发通信。
  - (b) Decomposition-based **interfere 计算**：GEMM 被碎片化为多个小 kernel，小 K 值时 GPU 利用率不足，overlap 带来的通信隐藏收益被计算性能损失抵消。
  - (c) Fusion-based **通信不通用（无 communication agnosticism）**：每种通信原语需要定制融合实现（AllReduce、ReduceScatter、All-to-All 各不同），重复开发成本高。
  - (d) Fusion-based **干扰计算**：融合时协调计算-通信 pipeline 可能需要改变 tiling 策略或计算逻辑，引入额外调优需求，可能导致性能退化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **FlashOverlap —— 基于 signaling 的 computation-communication overlap**：核心思路是在 GEMM kernel 中嵌入轻量信号机制，当 tile 完成时发送信号触发通信，同时 GEMM 继续执行剩余 tile（interference-free computation）。在此基础上：(i) wave-wise signaling timing —— 利用 GEMM 执行的 wave pattern（多个 tile 几乎同时完成，差异 <5% wave 时长），以 wave 而非单个 tile 为信号单位提升通信带宽利用率；(ii) tunable wave grouping —— 将连续 wave 组合为 group，在重叠机会和通信分段之间优化；(iii) pre/post-communication reordering —— pre-reordering 将非连续地址的 tile 按执行顺序重排为连续地址以直接调用 NCCL API（communication agnosticism），post-reordering 在通信后恢复正确数据顺序；(iv) predictive search —— 剪枝设计空间并通过延迟预测器实时搜索最优 wave group partition。

  全栈执行例子（同样 Llama3-70B TP=8 GEMM+AllReduce，A800 GPUs，FlashOverlap）：
  - **模型推理算法层**：同一 TP=8 GEMM+AllReduce 计算逻辑不变。不修改模型结构和计算语义。
  - **系统框架层**：FlashOverlap 替换 vLLM/Megatron-LM/xDiT 中的原始 linear layer + 通信对。端到端 LLM 推理（vLLM）直接调用 FlashOverlap 的 GEMM+overlap 实现。论文未明确说明框架层调度修改。
  - **编译框架层**：论文未明确说明。GEMM kernel 基于 CUTLASS 模板，使用标准 CUDA 编译路径。
  - **kernel调度层**：
    1. **GEMM kernel（Stream A）**：单一 GEMM kernel 完整执行（不碎片化），main loop 不变。每个 tile 完成时，epilogue 中执行 pre-communication reordering（将 tile 数据按执行顺序散射到连续地址的通信 buffer），同时 atomicAdd 更新 counting table 中对应 group 的计数。
    2. **Wave pattern 利用**：T 个 wave 依次完成（T = tile_num / SM_num），counting table 记录每个 group 的完成 tile 数。
    3. **Signaling kernel（Stream B）**：周期性查询 counting table。当 group G_j 计数达到 |G_j|（该 group 的 wave 数 × wave 内 tile 数）时，调用 NCCL API 对重排后的连续 buffer 执行通信。同时 Stream A 中 GEMM 的后续 wave 继续执行（interference-free）。
    4. **Post-communication reordering**：通信完成后，fused 到后续 RMSNorm kernel 中根据 mapping table 恢复数据原始顺序。
    5. **Tuning**：predictive search 在已知 GEMM size 和 bandwidth curve 后，搜索使 overlap 延迟最小的 wave group partition，选择如 (1, 2, 2) 的 partition（分别在第 1/3/5 个 wave 后触发通信）。
  - **硬件架构层**：同一 A800 GPU。GEMM 在 Tensor Core 上执行（main loop 不受干扰），通信通过 NVLink。Two CUDA streams 实现 concurrency：while GEMM 的 wave 2-3 在 Tensor Core 上计算，wave 1 的数据通过 NVLink 通信。

  关键设计选择与 baseline 缺陷的对应：
  - **defect: Decomposition-based 无 tile-wise overlapping (a)** → 方案：Wave-wise signaling —— tile 完成后通过 counting table 信号立即识别（tile-wise），但以 wave 为单位触发通信以保证带宽利用率（兼顾 overlapping opportunity 和通信效率）。对比 decomposition-based 必须等整个 subtensor 完成才能触发通信。
  - **defect: Decomposition-based interfere 计算 (b)** → 方案：Signaling 机制在主 GEMM 外部运行（另一 CUDA stream），GEMM main loop 完整保留不变。Counting table 的 atomicAdd 在 epilogue 中仅增加 ~0.07% GEMM 开销（A800 tile-level）。GEMM 不碎片化，GPU 利用率与无 overlap 时相同。
  - **defect: Fusion-based 通信不通用 (c)** → 方案：Pre-communication reordering —— 将按执行顺序的非连续 tile 重排为连续地址，无需修改通信库即可直接调用 NCCL API。所有通信原语（AllReduce、ReduceScatter、All-to-All）复用同一套 signaling + reordering 机制，仅 reordering pattern 不同（tile 级/subtile 级/subtoken 级）。Communication agnosticism 得以实现。
  - **defect: Fusion-based 干扰计算（tiling 策略修改）(d)** → 方案：Signaling 不改变 GEMM 的 tiling 策略或计算逻辑。Main loop 完全由 CUTLASS profiler 最优配置驱动，epilogue 中仅增加 reordering scattering 操作（0.07%-0.68% 开销）。Post-communication reordering 融合到后续必须执行的 element-wise kernel 中（RMSNorm 开销 7.46%-9.63%）。
  - **defect: 固定 overlap 策略在变 workload 下非最优** → 方案：Predictive search —— 根据 GEMM size 和 bandwidth curve 自动搜索最优 wave group partition。搜索空间从 2^{T-1} 通过剪枝约束 |G_1|≤2, |G_P|≤4 降低，延迟预测器误差 <5%（平均 3.4%），搜索 partition 达到穷举 >99% 性能。RTX 4090 + AllReduce 上仅 4% 的 case 最优为单 wave group（即 baseline partition），平均 17.34% 性能差距证明了 tuning 的必要性。
