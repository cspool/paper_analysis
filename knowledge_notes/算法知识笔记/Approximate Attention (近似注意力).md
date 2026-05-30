## Approximate Attention (近似注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Approximate Attention 是一类通过计算注意力矩阵的稀疏子集而非完整 N×N 矩阵来加速 Transformer 推理的技术。与 FULLATTN（精确注意力，如 FLASHATTN/RINGATTN/ULYSSES）保留完整计算结果不同，approximate attention 的核心思想是：注意力矩阵中大部分元素的 Softmax 值趋近于 0，仅少数 token pair 贡献有意义的 attention score，因此可以跳过无关计算，以微小精度损失换取显著加速。

Approximate attention 可大致分为三类：(1) 静态稀疏模式——如 MINFERENCE 为每个 attention head 预先分配固定的稀疏模式（如 diagonal, vertical+slash），仅计算模式指定的 attention score；(2) 基于 KV cache 压缩的——如 H2O、SNAPKV、LOCRET，通过选择保留重要的 KV pair 减少后续 attention 计算量；(3) 基于 anchor 的分布式近似——如 STARATTN 和 APB，在序列并行框架中通过 anchor block 和/或 passing blocks 近似全局 attention。

从算法pipeline角度拆解术语。

**APB 中的 Approximate Attention 流程（4-stage pipeline）**：

```
Stage 1: Context Splitting
  文档 d 按 H 个 host 均分，每 host 持有 B_h
  Anchor block A = {query, d[0:l_a]}（远小于 STARATTN）

Stage 2: Block Compression
  每 host 独立打分，选 Top-l_p KV pair:
  s = R([Q_h, K_h, V_h])           // retaining heads MLP 推理
  B_h^C = top_k(KV_h, l_p, by=s)   // 仅保留 l_p 个最重要 KV

Stage 3: Communication  
  (K^C_{1:H}, V^C_{1:H}) = AllGather(K_h^C, V_h^C)
  P_h = concat(K^C_{1:h-1}, V^C_{1:h-1})  // passing block

Stage 4: Computation
  K = [K_a, K_p^C, K_h], V = [V_a, V_p^C, V_h]
  A = flash_attn_with_mask(Q, K, V, M')   // 修改后 mask
  P_h 在 attention 后丢弃，不进入 FFN
```

**与 FULLATTN 的计算量对比**：
APB FLOPs/forward 远小于 FULLATTN（Table 9），因为：(a) 每 host 仅处理 l_b = n/H 长度，而非完整 n；(b) passing block 仅 l_p 长度（默认 l_p = l_b/8）；(c) anchor block 仅 l_a 长度（l_a = l_b/4 或 l_b/8）。

术语一般如何实现？如何使用？

APB 的 approximate attention 通过定制 FLASHATTN kernel（修改 attention mask）+ retaining heads（LOCRET 训练的 MLP）+ AllGather 通信实现。开源：https://github.com/thunlp/APB。STARATTN 通过大 anchor block + 无通信实现 approximate attention（https://github.com/NVIDIA/Star-Attention）。MINFERENCE 通过静态 head-specific 稀疏模式实现。Quest 通过 query-aware page-level selection + upper-bound criticality estimation 实现 approximate attention：先加载 per-page min/max Key metadata 计算 criticality score，Top-K 选择关键 page，仅对选中 page 执行 FlashAttention（https://github.com/mit-han-lab/Quest）。

涉及论文标题：
- APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs
- Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference

---
