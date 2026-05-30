## SeerAttention-R: Sparse Attention Adaptation for Long Reasoning

- 属于算法pipeline的实现是什么？实验比较什么？
  SeerAttention-R 提出一种自蒸馏 Attention Gate (AttnGate) 稀疏注意力框架，在 post-training 阶段只训练轻量级 gating module（冻结原始模型权重），实现解码阶段 block-level 稀疏注意力。实验比较了 SeerAttention-R 与 Full Attention baseline 和 Quest（training-free 稀疏注意力）在长序列推理 benchmark 上的准确率，同时通过 oracle sparsity 实验验证 attention 本身存在稀疏性。消融实验研究了 block size、hybrid dense layers、threshold vs token budget 等设计选择。

- 硬件平台是什么，配置是什么。
  训练：AMD MI300x GPU，DeepSpeed ZeRO-2 优化。
  推理精度评估：论文未明确说明 GPU 型号（推理精度实验）。
  Kernel 性能 benchmark：NVIDIA H100 GPU（Section 4.4）。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen3-4B, Qwen3-8B, Qwen3-14B（GQA, g=4）, DeepSeek-R1-Distill-Qwen-14B。
  训练数据集：OpenR1-MATH-220K（HuggingFace），packed to 32k token sequences。
  Benchmark：AIME24, AIME25（各64样本pass@1），MATH-500（8样本pass@1），GPQA-Diamond（16样本pass@1）。
  Baseline对比：Full Attention（dense），Quest（training-free sparse，block size调整为64与SeerAttention-R对齐）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/microsoft/SeerAttention

  算法 Pipeline（伪代码级描述）：

  **Stage 1 - AttnGate 前向推理（稀疏块选择）：**
  ```
  输入: Q ∈ R^{1, num_kv_heads, d_head}   (decode阶段单token)
        K ∈ R^{seq_len, num_kv_heads, d_head}
        K_compression_cache ∈ R^{num_blocks, num_kv_heads, d_gate}

  // 1. Q分支：GQA query head 聚合
  Q_nope, Q_pe = split_rope(Q)
  Q_reshaped = reshape(Q_nope, [num_kv_heads, g * d_head])
  Q_gate = RoPE(W_q_gate @ Q_reshaped)  // [1, num_kv_heads, d_gate]

  // 2. K分支：pooling + 压缩（使用 K Compression Cache 加速）
  //    新token累积到block_size倍数时才更新cache
  K_new_block = concat[MaxPool(K[-block_size:]), MinPool(K[-block_size:]), AvgPool(K[-block_size:])]
  update K_compression_cache[-1] = W_k_gate @ K_new_block  // [1, num_kv_heads, d_gate]

  // 3. 块级注意力分数预测
  S = softmax(Q_gate @ K_compression_cache^T / sqrt(d_gate))  // [1, num_kv_heads, num_blocks]

  // 4. Top-K 选择 (token budget 方法)
  block_budget = token_budget / block_size
  selected_block_indices = topk(S, k=block_budget, dim=-1)  // [1, num_kv_heads, block_budget]
  // 始终选中最后一个未完成的block（K Compression Cache 未更新时）
  selected_block_indices = selected_block_indices ∪ {last_block_index}
  ```

  **Stage 2 - 块稀疏 Flash Decoding（使用selected blocks计算attention）：**
  ```
  // 对每个 KV head group，只对 selected_block_indices 中的块计算 attention
  对于每个选中的 block i:
    K_block = K_cache[i * b : (i+1) * b]
    V_block = V_cache[i * b : (i+1) * b]
    S_i = Q @ K_block^T / sqrt(d_head)     // [1, b]
    // FlashAttention tiled softmax & rescaling
    O = online_softmax_rescale(O, S_i, V_i)
  输出 O ∈ R^{1, d_head}
  ```

  **Stage 3 - 训练（自蒸馏）：**
  ```
  // 只训练 AttnGate 参数（W_q_gate, W_k_gate），冻结原始模型权重
  对于每个 training step:
    1. 修改版 FlashAttention-2 kernel 同时计算 full attention output 和 block-level ground truth
    2. Ground truth = 1D column-wise maxpool(full_attention_scores)
    3. 对 GQA 组内 query heads 的 ground truth 再做 maxpool → KV-head 级别
    4. Normalize ground truth to sum 1
    5. L_KL = KL(AttnGate(S) || ground_truth)
    6. 反向传播仅更新 AttnGate 参数
  ```

  训练配置：global batch size=16, 800 steps (0.4B tokens), AdamW lr=1e-3, cosine decay, MI300x GPUs, DeepSpeed ZeRO-2。

  K Compression Cache 内存开销：block_size=64 时仅需原始 KV cache 的 1/128 (<1%)。
