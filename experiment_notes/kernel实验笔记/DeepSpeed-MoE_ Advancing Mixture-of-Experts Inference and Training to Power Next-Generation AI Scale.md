## DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是针对 MoE 推理的专用 CUDA kernel 优化，共三类：**(1) Gating Function Kernel Fusion**：将 top-k 选择、cumsum（Token ID per expert）、scatter 操作融合为单一 CUDA kernel，使用 dense token-to-expert mapping table 替代 sparse one-hot 掩码。cumsum 使用 Blelloch scan 算法并行化。**(2) Data Layout Transformation 替代 Sparse Einsum**：将 sparse einsum 的 token sorting（按 expert ID 排序输入 token）和 re-sorting（恢复原始 token 顺序）实现为基于 mapping table 的数据布局变换，复杂度从 S×E×M×c^e 降至 S×M×c^e，并融合 gating logits 概率缩放。**(3) Expert 参数内的并行运算优化**：利用 DeepSpeed inference 的高带宽利用率 Transformer kernel 处理 non-expert 层。实验比较：MoE kernel 延迟降低 6x+；per-GPU 吞吐随 GPU 数量增加而超线性增长（super-linear throughput scaling）；不同 MoE 模型规模（107B→2T params）的端到端推理延迟和吞吐量。

- 后端平台是什么，配置是什么。
  NVIDIA A100 GPU（Azure ND A100 instances），最多 256 GPUs。节点内 8 GPUs 通过 NVLink 互联，节点间 Mellanox InfiniBand 互联。软件栈：DeepSpeed-MoE，PyTorch distributed，NCCL / Microsoft SCCL 通信后端，自定义 CUDA kernels。

- 评估性能的软件/脚本是什么。修改了什么。
  DeepSpeed-MoE inference framework（开源）。主要修改：(a) 实现 MoE Gating 融合 kernel —— 单 kernel 内完成 top-k + cumsum (Blelloch scan) + scatter，使用 dense mapping table；(b) 实现 data-layout transformation kernel 替代 sparse-dense einsum，将 token 排序/反排序作为显式内存布局操作；(c) 在 token 反排序时融合 gating logits 的概率域缩放；(d) 优化所有 MoE 相关 kernel 为 dense representation 消除稀疏张量运算。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源，代码位于 https://github.com/microsoft/DeepSpeed。

  **MoE Gating + Token Routing Kernel 原理**：
  ```
  Input: S tokens, hidden dimension M, E experts, k=1 (top-1 gating)

  === Kernel 1: Fused Gating Kernel (单个 CUDA kernel) ===
  Input:  gate_logits[S][E]   // raw logits from gating linear layer

  // Step A: Top-k selection (k=1)
  For each token t in parallel:
    expert_id[t] = argmax(gate_logits[t])    // 选 logit 最高的 expert
    // 存入 dense mapping table, 不使用 sparse one-hot

  // Step B: Cumsum (Blelloch scan) - 计算每个 expert 处理多少 token
  // Parallel prefix sum on GPU:
  expert_counts[E] = {0}
  For each token t in parallel:
    atomicAdd(expert_counts[expert_id[t]], 1)
  // Blelloch scan to compute exclusive prefix sum:
  token_offset[0] = 0
  BlellochScan(expert_counts) → expert_offset[E]
    // expert_offset[i] = 起始位置 for expert i's tokens

  // Step C: Scatter - 计算每个 token 在其对应 expert 中的局部 ID
  For each token t in parallel:
    local_id[t] = atomicAdd(expert_offset[expert_id[t]], 1)

  Output: expert_id[S], local_id[S], expert_offset[E+1]
  // Complexity: O(S) parallel, S×E reduced to dense mapping table

  === Kernel 2: Data Layout Transformation (替代 Sparse Einsum) ===
  Input:  activations[S][M], expert_id[S], local_id[S], expert_offset[E]

  // Forward pass: 按 expert ID 重排 token 顺序 (sort)
  Output: sorted_acts[E][ce][M] = {0}   // ce = expert capacity
  For each token t in parallel:
    e = expert_id[t]
    pos = local_id[t]
    sorted_acts[e][pos] = activations[t]      // 直接 memcpy, 无 sparse 乘法

  // Expert FFN computation (per-expert, standard linear layers)
  For each expert e in parallel:
    expert_output[e] = W2_e @ GeLU(W1_e @ sorted_acts[e])

  // Backward pass: 恢复原始 token 顺序 (unsort) + probability scaling
  Input: expert_output[E][ce][M], expert_id[S], local_id[S], gate_probs[S]
  Output: final_output[S][M]
  For each token t in parallel:
    e = expert_id[t]
    pos = local_id[t]
    final_output[t] = gate_probs[t] * expert_output[e][pos]  // 融合概率缩放

  // 复杂度分析:
  // Sparse Einsum: S × E × M × ce = O(S·E·M·ce) → 含大量零乘法
  // 优化后: S × M × ce = O(S·M·ce) → 仅移动非零元素
  ```
  
  关键优化点：
  - Sparse einsum 中 (E-1)/E 的运算为与零相乘 → 完全消除
  - 从立方复杂度 S×E×M×ce 降至二次 S×M×ce
  - 多个 kernel launch 融合为单一 kernel → 减少 launch overhead
  - Dense mapping table 替代 sparse mask → 减少内存和计算开销
  - 组合优化实现 MoE kernel 延迟降低 6x+
