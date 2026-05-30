## MoEBlaze: Breaking the Memory Wall for Efficient MoE Training on Modern GPUs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：MoEBlaze 设计并实现了两类自定义 CUDA kernel：
    1. **高效 Dispatch 数据结构构建 Kernel**（Section 4.2）：替代传统基于 multi-pass radix sort 的 token 调度方法，采用 3 步 atomic-free 的 GPU 并行构建流程：
       - **Step 1 - Build Dense Token-Expert Map**：分配 L×E 的 dense map，每个 warp 分配不相交的 token rows（i），将 top-k expert ID 写入 dense_token_map[i, e_{i,k}] = i。每个 (i,e) pair 最多写入一次（expert ID per token 唯一），guaranteed no intra-warp collision。
       - **Step 2 - Compute Expert Lengths**：CTA grid 按列（expert）映射，每个 CTA 专用于一个 expert e_i，通过 warp-level reduction 计数该列非零项，产出 expert_lengths 数组。
       - **Step 3 - Route Indices to Gates**：两阶段构建 location map：(i) tile-level scan——每 CTA 处理 contiguent tokens，shared memory 内做 exclusive scan（prefix sum）；(ii) 局部 scan 结果加 expert 的全局 expert_offsets，得到每个 entry 在 expert_token_indices 中的最终位置。最后通过并行 kernel 将 dense_token_map 中非零项直接写入对应位置，无原子操作。
    2. **Fused SwiGLU MoE Training Kernel**（Section 5）：将 SwiGLU FFN 的两个第一层 projection (W1, W2) 和激活 epilogue 融合为单 kernel：
       - **Forward**：一次性加载 input x，同时流式通过 W1 和 W2 的 GEMM，在 register/shared memory 中计算 SiLU(a) 并与 b 做 element-wise 乘法，仅写最终输出到 global memory，消除 a、b、σ(a)、SiLU(a)、final product 等中间结果的 global memory 写。
       - **Backward**：融合两个分支的 activation derivatives 计算，通过 tiled reduction 做 in-place 梯度聚合，消除临时 global buffers。采用 activation checkpoint 策略——forward 中不保存 SiLU 中间结果，backward 时 recompute SiLU（element-wise 操作，memory bandwidth bound，recompute 开销极低）。

  - 实验比较：(1) 训练速度（1.4×–6.2× speedup vs MegaBlocks，取决于配置和激活函数）；（2）激活内存消耗（最高 4× reduction）；（3）SiLU vs SwiGLU 两种激活函数的性能差异。

- 后端平台是什么，配置是什么。
  - 单张 NVIDIA H100 Tensor Core GPU（80GB HBM）。利用 H100 的硬件加速特性：warp-group matrix multiplication（WGMMA）、Tensor Memory Accelerator（TMA）。

- 评估性能的软件/脚本是什么。修改了什么。
  - 软件栈：PyTorch 2.0.1 + CUDA 12.1。Baseline：MegaBlocks（基于 block-sparse 操作的高性能 MoE 训练系统）。MoEBlaze 以自定义 CUDA kernel 替换了 MegaBlocks 的 token dispatch（sort-based bucketize）和 expert FFN 计算流程。
  - 内存测量：使用 PyTorch 的 saved tensor hooks 追踪训练中分配的所有中间激活张量大小。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未明确说明开源链接。
  - 评估原理：对比 MoEBlaze 与 MegaBlocks 在单层 MoE 的 forward+backward（Sparse-to-Sparse 阶段）的 wall-clock time 和 peak activation memory。
  - Kernel 输入到性能输出全过程（以 SwiGLU 为例）：
    1. **输入**：token 张量 x ∈ R^{L×d}（L = batch_size × seq_len），gate 权重 W_g，expert 权重 W1_i, W2_i, W3_i（每个 expert），所有数据在 HBM 中。
    2. **Gating**：W_g · x → softmax → TopK，生成 topk_experts（L×K 的 int32 索引）。
    3. **Dispatch 数据结构构建**（3-step kernel）：
       - Launch L×E grid → 填充 dense_token_map → 写入 HBM（L×E 个 int32，稀疏矩阵）
       - Launch E CTAs → warp-level reduction → expert_lengths[E] → expert_offsets[E+1]
       - Launch E CTAs → tile-level scan + global offset add → 并行写入 expert_token_indices[L×K]
       - 同时构建 token_expert_indices[L×K] 和 token_index_map[L×K]
    4. **Fused SwiGLU FFN Forward**（per expert）：
       - HBM → register：加载 x[token_ids]（on-the-fly gather via expert_token_indices）
       - Tensor Core (WGMMA)：x @ W1 → a，x @ W2 → b（两个 GEMM 在一个 kernel 内流式完成）
       - Register/shared memory：compute SiLU(a) = a·σ(a)，y_swi = SiLU(a) ⊙ b
       - Store to HBM：仅保存 a, b, y_swi 用于 backward（SiLU(a) 不保存，backward 时 recompute）
       - HBM → register：y_swi @ W3 → y_out，store to HBM
    5. **Backward**：
       - ∇W3 = y_swi^T · ∇y_out（GEMM）
       - ∇y_swi = ∇y_out · W3^T（GEMM）
       - Recompute SiLU(a) from a（element-wise，memory bandwidth bound）
       - ∇a = ∇y_swi ⊙ b ⊙ ∇SiLU(a)，∇b = ∇y_swi ⊙ SiLU(a)_recomp（fused tiled reduction）
       - FusedBwdW：∇W1 = a^T · ∇a，∇W2 = b^T · ∇b（fused kernel）
       - FusedBwdX：∇x = ∇a · W1^T + ∇b · W2^T（in-place aggregation）
    6. **测量**：PyTorch saved_tensors_hooks 追踪所有通过 ctx.save_for_backward 保存的 tensor → 计算总字节数作为 activation memory。Wall-clock time 由 CUDA events 测量，排除 optimizer 更新时间。
  - 性能要点：相比 MegaBlocks 的 sort-based dispatch（multi-pass radix sort 需要多次 global memory passes + 多次 kernel launch），MoEBlaze 的 3-step 构建仅需单次 kernel launch chain，利用 shared memory 内的 prefix sum 和 warp-level reduction 避免 atomics，大幅减少 global memory traffic 和 kernel launch latency。
