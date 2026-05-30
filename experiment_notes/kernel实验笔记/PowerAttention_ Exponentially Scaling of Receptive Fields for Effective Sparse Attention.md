## PowerAttention: Exponentially Scaling of Receptive Fields for Effective Sparse Attention

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  POWERATTENTION 使用 PyTorch FlexAttention（Dong et al., 2024）库实现自定义稀疏 attention mask，结合 Triton（Tillet et al., 2019）实现 kernel 级别的优化和序列并行训练（RingAttention, Liu et al., 2024）。核心 kernel 实现是通过 FlexAttention 的编程模型将自定义 mask（power-of-2 + window + sink）编译为优化的 GPU kernel，使用 256-token blocks 对齐 CUDA compute cores 的内存访问模式。FlexAttention 自动将 mask 定义转换为 block-sparse attention kernel 的 tiling 和内存访问策略。

  实验比较：(a) Kernel 前向时间对比：POWERATTENTION vs Full Attention vs MInference，测量 16K-128K context 下每次 attention forward pass 的时间（Figure 4b），128K 时 POWERATTENTION kernel 比 MInference 快 5.3×，比 Full Attention 快 21.6×；(b) 端到端延迟：prefilling 阶段和 decoding 阶段（1024 steps）的完整推理延迟（Figure 4a），128K 时 prefilling 比 Full Attention 快 3.0×，decoding 仅需 Full Attention 58% 的时间。

- 后端平台是什么，配置是什么。
  NVIDIA A800 GPU。模型 Qwen2-7B（28 layers, 32K context）。Kernel 配置：block_size=256 tokens（对齐 GPU compute core 内存访问），sparsity ratio ≈ 94%（每个 token 最多关注 ~10 blocks）。POWERATTENTION 内核因 O(N log² N) 时间复杂度，增长曲线接近滑动窗口的线性复杂度。

- 评估性能的软件/脚本是什么。修改了什么。
  使用 PyTorch FlexAttention 库定义稀疏 attention mask，Triton 结合 RingAttention 用于序列并行以扩展到更长序列。修改内容：(a) 使用 FlexAttention 的 `create_block_mask` 或等效接口定义 power-of-2 mask pattern；(b) 将 POWERATTENTION 的 mask（sink + window + power）编译为 block-sparse kernel，利用 FlexAttention 自动将 mask 映射到 GPU tiling；(c) 通过 Triton kernel 将 KV cache 分块在序列维度上并行计算。MInference 仅在 prefilling 阶段使用（按原论文建议），decoding 阶段回退到 FlashAttention。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未明确提供开源链接。实现基于 PyTorch FlexAttention 库。kernel 执行原理如下：

  **评估原理**：FlexAttention 提供一种编程模型，用户只需用 PyTorch 张量操作定义 attention score mask（`score_mod` 函数），框架自动将其编译为融合的 block-sparse CUDA kernel。对于 POWERATTENTION，mask 定义在 block 级别（block_size=256），因此所有计算和内存访问均为 block-aligned。

  **Kernel 输入**：Query tensor Q [M, d_k]，Key tensor K [N, d_k]（N 个 prefill token 或 1 个 decode token + KV cache），Value tensor V [N, d_v]，block-wise mask（从 FlexAttention score_mod 函数推导）。

  **Kernel 执行流程**（FlexAttention block-sparse 模式）：
  ```
  1. Mask 分析阶段（offline 或首次调用时）:
     - FlexAttention 接收 score_mod(q_idx, kv_idx) 函数
     - 将输入按 block_size=256 分块，预计算哪些 (query_block, kv_block) 对需要计算
     - 生成稀疏的 block 索引列表（仅 mask=1 的 block 对）
     - POWERATTENTION 的 block mask：sink(block 0) + window(5 blocks) + power-of-2 blocks
  
  2. Kernel 执行阶段（Triton grid）:
     for each query_block (Grid-level parallel):
       load Q_block [B_q, d_k] into SRAM
       for each kv_block in sparse_block_list[query_block]:
         load K_block [B_k, d_k], V_block [B_k, d_v] into SRAM
         S = Q_block @ K_block^T / sqrt(d_k)  # [B_q, B_k]
         online softmax + accumulate: o = softmax_update(S, V_block)
       write o to HBM
  ```
  POWERATTENTION 的 block 数 ≈ O(log n) per query，因此总计算复杂度 O(N log² N)，内存访问量远低于 Full Attention 的 O(N²)，与滑动窗口的 O(N) 接近。128K context 时 kernel 比 Full Attention 快 21.6×。
