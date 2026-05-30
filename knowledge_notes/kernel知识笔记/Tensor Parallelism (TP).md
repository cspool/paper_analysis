## Tensor Parallelism (TP)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Tensor Parallelism (TP, Shoeybi et al., 2019) 是 Megatron-LM 提出的模型并行策略，将单层 Transformer 内的矩阵乘法算子沿特定维度切分到多个 GPU。对于 attention 层的 QKV 投影和 FFN 层，TP 将权重矩阵按列切分，每 GPU 计算部分输出；对于后续的 output projection，按行切分，每 GPU 先本地计算再 all-reduce 求和。TP 优点是不需要 layer-wise 的通信流水线（PP 需要），但每层都需要一次 all-reduce。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Megatron-LM TP（2-way TP 简化示例）：

```
# Attention layer with TP=2 (沿列切割 W_q, W_k, W_v)
GPU0: Q0 = X @ W_q[:half_cols], K0 = X @ W_k[:half_cols], V0 = X @ W_v[:half_cols]
GPU1: Q1 = X @ W_q[half_cols:], K1 = X @ W_k[half_cols:], V1 = X @ W_v[half_cols:]

# 各自计算 attention (本地，无通信)
GPU0: Z0 = attention(Q0, K0, V0)
GPU1: Z1 = attention(Q1, K1, V1)

# Output projection (沿行切割 W_o), 需要 all-reduce
GPU0: Y0 = Z0 @ W_o[:half_rows]^T  →  Y = all_reduce(Y0 + Y1)
GPU1: Y1 = Z1 @ W_o[half_rows:]^T
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- FOLDMOE 在 attention 层使用 TP=8（intra-node，同一节点内 8 GPU）
- TP 与 FOLDMOE 的 attention-MoE pipelining 正交：TP 切分算子，FOLDMOE 沿 sequence 维度切分数据
- TP 通信为 all-reduce（通常通过 NVLink，intra-node），与 EP 的 A2A 通信（可能跨节点）独立
- FOLDMOE 将 TP 和 EP 组合使用，充分利用节点内高带宽 NVLink 和节点间网络
- HAP 对 TP 的推理性能分析：TP 在长上下文 prefill 场景下因 AllReduce 通信量 ∝ batch×seqlen 成为瓶颈——在 PCIe 低带宽（A6000/V100）下，TP 通信开销严重。TP 在短序列 decode 场景下因单 token 通信量极小且无负载不均衡，是最优策略。HAP 的 ILP 搜索在通信瓶颈场景下倾向于为 Attention 选 DP（无通信）、为 Expert prefill 选 EP（All-to-All 通信量低于 TP 的 AllReduce），在计算瓶颈场景下仍选 TP。

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining
- HAP: Hybrid Adaptive Parallelism for Efficient Mixture-of-Experts Inference
- IFMoE: An Inference Framework Design for Fine-grained MoE
