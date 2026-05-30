## XAttention: Block Sparse Attention with Antidiagonal Scoring

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  基于 FlashInfer 框架实现 block-sparse attention 的 GPU kernel，核心包含三部分 kernel 级实现：(1) **Strided Antidiagonal Scoring Kernel**：对每个 B×B 大小的 attention block，沿反对角线以步长 S 重排 Q 和 K 后计算近似注意力分数。Q 的 reshape 从 [B, d] 按 stride S 沿反对角线交错读取为 [S, B//S, d]；K 从全局内存按 stride S 分步读取。计算 Q_reshaped @ K_reshaped^T 得到 [S, S] 近似注意力矩阵，其反对角线和作为 block 重要性得分；(2) **Block Selection Kernel**：对 softmax 归一化后的反对角线分数执行 find_blocks——选择累计概率超过阈值 τ 的最小 block 子集。基于 cumulative probability threshold 实现动态稀疏度（Top-K 和 Top-Ratio 无法处理变化序列长度）；(3) **Block-Sparse Attention Kernel**：仅对选中的 block 子集执行完整的 FlashAttention 风格精确注意力计算，kernel 级别的 block tiling 跳过未选中区域。

  实验比较：(a) Prefill attention speedup vs FlashAttention (FlashInfer 实现)、MInference、FlexPrefill——256k 上下文下最高 13.5× 加速（S=16, τ=0.9, 密度 7.32%）；(b) Pattern selection 时间对比——XAttention 的 antidiagonal pattern selection 比 MInference 的 vertical-slash index search 快 24.9×，比 FlexPrefill 快 5.9×；(c) Attention time breakdown（Figure 5）：XAttention 将 pattern selection + sparse attention 的总开销控制在最低水平；(d) 消融研究：对比 antidiagonal vs random vs diagonal 模式的密度和准确率。

- 后端平台是什么，配置是什么。
  NVIDIA GPU（DGX 服务器）。基于 FlashInfer（https://flashinfer.ai/）注意力 kernel 库实现，使用其 FlashAttention 实现作为 dense baseline。精度为 BF16/FP16。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 FlashInfer 框架的 attention kernel 进行修改。核心修改：
  1. **Antidiagonal Scoring 实现**：在标准 FlashAttention prefill kernel 前插入轻量级 antidiagonal score 计算——对 Q 按 stride S 沿反对角线取子序列（Q[i::S,:] for i=S-1..0），对 K 按 stride S 正向取子序列（K[i::S,:] for i=0..S-1），计算 Q_reshaped @ K_reshaped^T / sqrt(d_h) / S 作为近似注意力分数。该步骤计算量仅为完整注意力的 1/S²。
  2. **Block Selection 实现**：基于反对角线 softmax 概率累积和选择 block，实现 greedy cumulative threshold 算法——按反对角线得分降序排列 block，从高到低累积直到超过 τ。
  3. **Sparse Attention 计算**：将选中的 block indices 传入 FlashInfer 的 block-sparse attention kernel，仅计算 mask 为 1 的 (query_block, key_block) 对。
  4. **Dynamic Threshold Prediction**：离线使用动态规划为每个 head 预测最优 τ 值，通过逐步降低 10% 的搜索策略（M=1000 steps）探索最优配置。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源：https://github.com/mit-han-lab/x-attention

  **评估原理**：使用 CUDA Event 测量 attention 模块的 prefill kernel 执行时间（含 pattern selection + sparse attention 两部分），与 FlashInfer 的 FlashAttention 实现对比。Density 定义为选中 block 占总 block 数的比例（Table 5：128k 时 S=4→6.20%, S=8→6.89%, S=16→7.32%）。

  **Kernel 输入**：Q, K, V ∈ R^{L×d}（prefill 阶段完整序列），block size B，stride S，threshold τ。Dynamic threshold prediction 模式下每个 head 独立 τ_h。

  **Kernel 执行流程**（以 prefill 128k tokens, B=64, S=16 为例）：

  ```
  // N_B = 128k / 64 = 2048 blocks

  // === Phase 1: Antidiagonal Scoring (轻量级) ===
  for each block b in 0..N_B-1 (parallel over blocks):
      // Q reshape: [64, d] -> [16, 4, d]
      Q_slice = Q[b*64:(b+1)*64, :]
      Q_reshaped = []
      for i = 15 down to 0:
          Q_reshaped.append(Q_slice[i::16, :])  // 取反对角线元素

      // K reshape: [L, d] -> [16, L//16, d]
      K_reshaped = []
      for i = 0 to 15:
          K_reshaped.append(K[i::16, :])

      // 近似注意力: [16, 4, d] @ [16, L//16, d]^T -> [16, 16, L//16]
      // 简化为 per-block 的反对角线得分
      A_approx = Softmax(Q_reshaped @ K_reshaped^T / sqrt(d_h) / 16)
      score[b] = sum of antidiagonal values in A_approx

  // === Phase 2: Block Selection ===
  sorted_blocks = argsort(scores, descending=True)
  cumsum = 0
  selected_blocks = []
  for b in sorted_blocks:
      cumsum += scores[b]
      selected_blocks.append(b)
      if cumsum >= τ: break
  // 例如 τ=0.9, density ≈ 7%, 选中 ~143/2048 blocks

  // === Phase 3: Block-Sparse Attention ===
  M = zeros(N_B, N_B)  // 2048 x 2048 block mask
  M[:, selected_blocks] = 1  // 每行 query block 只关注选中的 key blocks
  // 实际 kernel 实现中直接传入 selected_blocks 索引列表

  for each query_block in 0..N_B-1 (grid-level parallel):
      load Q_blk [64, d] into SRAM
      for each key_block in selected_blocks:
          if key_block > query_block: continue  // causal mask
          load K_blk [64, d], V_blk [64, d] into SRAM
          S = Q_blk @ K_blk^T / sqrt(d_h)  // [64, 64]
          P = online_softmax(S)
          O_blk += P @ V_blk
      write O_blk to HBM

  // 总计算量: N_B * |selected| * B² * d
  //           = 2048 * 143 * 64² * 128 ≈ 1.5 × 10^11 FLOPs
  // vs dense: 2048 * 2048 * 64² * 128 ≈ 2.2 × 10^12 FLOPs
  // 加速比 ≈ 14.5×（接近实测 13.5×）
  ```

  **Pattern Selection 效率**：
  - XAttention antidiagonal scoring: O(N_B × S × d × B/S) = O(L × d) per block → 轻量级
  - MInference vertical-slash index search: O(L × k_v × k_s) → 需遍历垂直/斜线索引
  - 论文报告 pattern selection 快 24.9× vs MInference, 5.9× vs FlexPrefill

  **稀疏度随序列长度变化**（Table 5, S=8）：
  - 4k: density 52.16%（短序列注意力较密集）
  - 32k: density 20.97%
  - 128k: density 6.89%（长序列注意力高度稀疏）

  **With/Without Dynamic Threshold**：
  - Fixed τ=0.9: S=8, 32k density 23.06%
  - Minimum τ (DP optimized, avg 0.8): S=8, 32k density 20.97%
