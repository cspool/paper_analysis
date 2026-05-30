## Lightning Attention 2

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Lightning Attention 2 由 Qin et al. (2024) 提出，是第一个在实际中实现线性注意力理论计算优势的 Triton kernel 实现。核心采用 tiling 策略将注意力计算分为两部分：(1) Intra-block：使用传统 softmax attention（利用局部性）；(2) Inter-block：应用线性注意力的右乘（right product）技巧——先计算 KV 累积值再与 Q 相乘。通过 IO-aware 设计将 KV state 保持在 SRAM 中以最小化 HBM↔SRAM 数据传输。在因果（causal/autoregressive）设置下，解决了之前线性注意力实现中 cumsum 操作无法发挥理论优势的问题。性能：8K context 比 FlashAttention-2 快 1.5×，32K context 快 3×；训练速度与序列长度无关（恒定），推理 per-token 速度也与 context 长度无关。

在 SUPRA 中的用法：SUPRA 使用 Lightning Attention 2 的 Triton kernel（`lightning_attn_ops`）进行训练时的并行线性注意力计算。输入为 RoPE 后的 MLP kernel 特征 q/k、scale 后的 k 和 v，以及 decay slope tensor。Kernel 内部处理带衰减的线性注意力：O_i = Σ_{j=1}^{i} γ^{i-j} (φ(q_i)·φ(k_j)) v_j，通过 tiling 分解为 intra-block 和 inter-block 计算。这使 SUPRA 的训练 throughput 达到约 4300 tokens/s/GPU（7B 模型，H100）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Lightning Attention 2 tiling 策略伪代码：
```
# 将序列分为 chunks of size B
# Intra-block (块内): 使用左乘 (QK^T)·V (softmax attention style)
# Inter-block (块间): 使用右乘 K^T·V 累积 (linear attention style)

Input: Q, K, V ∈ R^{N×d}, decay slope s ∈ R^h
KV_state = 0  # 存于 SRAM

For i in 0, B, 2B, ..., N-B:
    Q_block = Q[i:i+B], K_block = K[i:i+B], V_block = V[i:i+B]
    
    # Intra-block: 标准 causal attention (左乘, block 内)
    Attn_block = causal_softmax(Q_block @ K_block^T / √d) @ V_block
    
    # Inter-block: 线性注意力累积 (右乘, 跨 block)
    KV_state_decayed = KV_state * exp(-s * B)
    Linear_block = Q_block @ KV_state_decayed  # O(B·d²)
    
    # 更新 KV state (存回 SRAM)
    KV_state = KV_state_decayed + K_block^T @ V_block  # O(d²)
    
    # 合并输出
    O[i:i+B] = Attn_block + Linear_block

Return O
```

关键设计：KV_state 在 SRAM 中持续累加（不再写回 HBM），仅 Q/K/V blocks 从 HBM 读取和 O blocks 写回 HBM。Triton 实现利用 `tl.dot` 进行 block 内 MatMul，利用 Triton 的 automatic memory coalescing 优化 HBM 访问。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/OpenNLPLab/lightning-attention (Triton kernel)。SUPRA 在其 OpenLM fork 中集成了该 kernel，调用方式为 `lightning_attn_ops(q, k * qk_scale, v, slope_tensor)`。Lightning Attention 2 的局限性：(1) 仅支持固定 decay 的线性注意力（无法直接处理 data-dependent decay 如 Finch 的 w_t）；(2) block size 需 tuned 以获得最优性能。适用于使用固定或 learnable decay 向量的线性注意力模型（如 RetNet、SUPRA、TransNormer）。

涉及论文标题：
- Linearizing_Large_Language_Models

---
