## MegaBlocks: Efficient Sparse Training with Mixture-of-Experts

- baseline方法是什么？
  - Baseline 为现有 MoE 训练框架（以 Microsoft Tutel 和 Megatron-LM SwitchMLP 为代表），它们使用固定 expert capacity 和 token dropping/padding 机制来实现 MoE 层的高效计算。以 Tutel (Hwang et al. 2022) 为例说明全栈执行路径：
    - **算法层**：MoE 训练时，Router 通过 top-k greedy selection 分配 token 到 expert → 定义固定 expert capacity（capacity_factor × num_tokens/num_experts）→ Permutation 将 token 按 expert 分组 → 对超出 capacity 的 token **直接丢弃**（不参与 expert 计算），对不足 capacity 的 expert **padding 零填充** → 使用 **batched matrix multiplication**（所有 expert 共享相同 batch size = capacity）并行计算所有 expert → Un-permutation 恢复 token 顺序 → 输出缩放。
    - **系统框架层**：基于 Megatron-LM（Shoeybi et al. 2019）+ PyTorch 实现。MoE 层使用 NCCL All-to-All 进行跨设备 token dispatching。Tutel 在此基础上实现了动态 capacity factor（运行时自适应计算最小不丢 token 的 capacity factor）和通信隐藏优化。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：使用 cuBLAS batched GEMM 作为 expert 计算原语。batched GEMM 要求所有矩阵乘法的尺寸相同，这强制了 expert capacity 约束。无自定义 GPU kernel 处理稀疏/动态计算。
    - **硬件架构层**：NVIDIA A100 SXM4 80GB GPU + 200 Gbps InfiniBand。8-way expert model parallelism。
  - Baseline 核心缺陷：
    1. **Token dropping 降低模型质量**：即使使用 load balancing loss，token routing 仍然高度不均衡。capacity_factor=1 时丢 token 导致 validation loss 仅降低 0.15（vs dense baseline），而不丢 token 可降低 0.26（1.73× 改善），足以超越更大 dense model（图 2）。完全避免丢 token 需要 capacity_factor 高达 11（Hwang et al. 2022），导致 computation 增加超过 2×（图 2）。
    2. **Padding 浪费计算和内存**：为满足 batched GEMM 的形状约束，不足 capacity 的 expert batch 需要 zero-padding。padding 在 MoE 层显著增加 activation 存储需求，导致 Tutel 被迫使用 2×–8× 更小的 micro_batch_size（表 3），降低 GPU 利用率和硬件效率。
    3. **Capacity factor 超参数调优成本高**：capacity_factor 在模型质量和计算效率之间构成 trade-off，需要为每个模型和任务调优。大型模型训练成本可达数十万美元，这阻碍了 capacity factor 的充分探索（Artetxe et al. 2021; Clark et al. 2022 完全放弃了 capacity factor 调优）。
    4. **Sequential expert 计算退化**：Megatron-LM 的 SwitchMLP（逐个 expert 顺序计算）虽避免丢 token，但随 expert 数量增加性能急剧退化——num_experts=128 时比 MegaBlocks 慢 20×（图 10），因为单个 expert 的计算量不足以饱和 GPU，小矩阵乘法序列化执行。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - MegaBlocks 通过"block-sparse 操作重表述 + 自定义 GPU block-sparse kernels"双层设计解决上述缺陷，实现真正的 dropless-MoE (dMoE)。全栈执行路径（以 MoE-Small 训练为例，8×A100）：
    - **算法层 — dMoE 的 block-sparse 重表述（§4, Figure 4）**：
      1. Router 分配 token 到 expert（与 baseline 相同：indices, weights = router(x)）。
      2. **make_topology(indices)**：构造图 3C 的 variable-size block diagonal matrix。每个 expert 的 token batch（size 可变）被分解为 ceil(num_tokens_expert/128) 个 128×128 固定 block，所有 expert 的 block 沿对角线排列，构成一个大的 block-sparse 矩阵。这一步取代了 baseline 的 capacity 约束——不再需要 token dropping 或 padding 到固定容量。
      3. **padded_gather**：按 expert 分组 token，仅 padding 到 128 的倍数（而非 padding 到固定 capacity），padding 量极小。
      4. **sdd(x, w1, topology)**：SDD（Sparse = Dense × Dense）操作。输出是 block-sparse 的 intermediate result——sparse output matrix 中只有分配给各 expert 的 token row 被计算（对应图 3C 的非零 block），未被分配的位置默认为零。等效于同时计算所有 expert 的第一层 FFN，但只计算实际需要的位置。
      5. **dsd(intermediate, w2)**：DSD（Dense = Sparse × Dense）操作。以 block-sparse intermediate 为左输入，计算所有 expert 的第二层 FFN，输出 dense tensor。
      6. **padded_scatter + scaling**：恢复 token 顺序并乘以 router probabilities。
    - **系统框架层**：
      1. 基于 Megatron-LM + PyTorch 构建，实现自定义 dMoE layer 替换标准 MoE layer。
      2. 支持 data parallelism 和 expert model parallelism（§5.3）。expert model parallelism 中先通信各设备将接收多少 token（避免 all-to-all 阶段的丢 token/padding）。
      3. 与 Megatron-LM 的 mixed-precision training（FP16 + FP32 accumulation）完全兼容。
    - **kernel 调度层 — 自定义 Block-Sparse GPU Kernels（§5.1）**：
      1. 基于 CUTLASS 2.5 扩展实现 SDD、DSD、DDS 三种 block-sparse GEMM kernel，支持所有 transposed/non-transposed 输入组合（满足前向+反向 6 种操作需求）。
      2. **Hybrid Blocked-CSR-COO 编码**：BCSR 作为主格式（高效行迭代），**额外物化行索引**使得 BCSR 兼具 BCOO 能力。SDD kernel 中每个 threadblock 通过 row_idxs[blockIdx.x] 和 column_idxs[blockIdx.x] 直接 O(1) 定位其 non-zero block 的坐标，无需搜索 row offsets。metadata 存储开销 <0.1%（128×128 block 含 16384 非零值仅需 1 个索引）。
      3. **Transpose Indices**：为支持向后传播中稀疏矩阵转置操作（SDD^T, DS^T D, DSD^T, DD^T S），构造转置元数据（等效 BCSC 编码：column offsets + 转置顺序的 non-zero block 偏移数组）。不显式转置非零数据，通过间接索引实现在转置顺序下迭代矩阵，避免 O(nnz) 的数据复制开销。
      4. **128×128 block size** 基于 CUTLASS tile benchmark 选择（图 5），实测在所有 tile dimension 配置中表现最优（与 cuBLAS 为 dense Transformer 选择的配置一致）。大 block size 提供足够的算术强度以充分利用 A100 Tensor Cores，同时摊销稀疏元数据开销。
      5. **Custom permutation kernel**：将 token padding（到 128 倍数）融合进 gather/scatter kernel，且在前向开始时一次性构造 block-sparse 和 transpose 元数据，摊销到后续 6 次矩阵乘法。
    - **硬件架构层**：NVIDIA A100 SXM4 80GB GPU × 8。CUDA 11.5 + CUTLASS 2.5。无硬件修改。
  - 对比 baseline 的改进映射：
    - **Token dropping 降低模型质量 → 无丢 token 的 dMoE（block-sparse 重表述）**：标准 MoE 需要固定 capacity → 超出 capacity 的 token 被丢弃。MegaBlocks 用 block-sparse 操作替代 batched GEMM，天然支持每个 expert 接收不同数量的 token（variable-size block），从根本上消除了 token dropping 的必要性。结果为更低的 loss（图 7）同时避免 capacity_factor 超参数调优。
    - **Padding 浪费计算和内存 → block-sparse 只计算实际需要的 tokens**：batched GEMM 要求所有 expert batch 等大小 → 大量 zero-padding 占用 activation 内存。block-sparse 操作只对实际分配的 token row 计算（sparse matrix 的非零 block），仅需将 token batch padding 到 128 的倍数而非固定 capacity。Tutel 因 padding 导致 micro_batch_size 被迫缩小 2×–8×，MegaBlocks 的 micro_batch_size 与模型大小自然匹配，端到端训练加速 1.38×–4.35×。
    - **Capacity factor 超参数调优成本高 → 无 capacity_factor 参数**：MegaBlocks 的 dMoE 从根本上不需要 capacity_factor——不需要在丢 token 和 wasteful computation 之间权衡。这不仅减少了 hyperparameter 搜索空间，还避免了"token dropping MoE 需要额外调优 capacity factor → 调优成本和训练成本叠加"的困境。图 8 显示即使与最优 capacity_factor 的 MoE 比较，dMoE 仍减少 1.18×–1.38× 训练时间。
    - **Sequential expert 计算退化 → block-sparse 并行计算所有 expert**：Megatron-LM 的 SwitchMLP 顺序计算 expert，随 expert 数量增加 GPU 利用率急剧下降。MegaBlocks 通过 block-sparse kernels 一次性并行计算所有 expert（单个 kernel launch），在 num_experts=128 时实现 20× 加速（图 10），在常规 num_experts=64 时端到端加速达 4.35×（图 7）。
    - **现有 block-sparse 库不适用 → 自定义 MoE-tailored kernels**：cuSPARSE blocked-ELL 格式要求所有 row 的非零数相同（与 MoE 冲突），Triton Blocksparse 假定稀疏拓扑在 iterations 间不变（与 MoE 的动态路由冲突）。MegaBlocks 的 custom kernels 专为动态稀疏拓扑设计，且 block-sparse 矩阵运算达到 cuBLAS 密集运算的 98.6% 吞吐量（图 9），实现了"sparse computation at dense throughput"的工程目标。
