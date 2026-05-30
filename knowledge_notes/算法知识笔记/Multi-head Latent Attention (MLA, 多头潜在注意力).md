## Multi-head Latent Attention (MLA, 多头潜在注意力)

术语解释
MLA 是 DeepSeek-V2 提出的注意力机制，通过对 Key 和 Value 进行低秩联合压缩（low-rank joint compression），将 KV Cache 从每头存储完整 K/V 向量压缩为存储一个低维 latent vector，大幅减少推理时的 KV Cache 内存占用。

术语是什么？
传统 MHA 对每个 token 需要缓存所有头的 K 和 V 矩阵，KV Cache 大小随序列长度和头数线性增长。MLA 通过将 K 和 V 投影到低维 latent 空间：输入 h_t 经下投影矩阵 W^{DKV} 压缩为 latent vector c_t^{KV}，再分别经 W^{UK} 和 W^{UV} 上投影还原 K 和 V。推理时仅需缓存 c_t^{KV}（而非完整 K/V），实现 93.3% KV Cache 减少（DeepSeek-V2 报告）。

从算法pipeline角度拆解术语：

```
=== MLA 前向传播 ===
# Step 1: Q 投影（标准）
q_t = W_Q @ h_t  # [n_heads * d_head]

# Step 2: KV 低秩压缩（MLA 核心）
c_t_KV = W_DKV @ h_t  # [d_latent], d_latent << n_heads*d_head

# Step 3: K, V 还原
k_t_C = W_UK @ c_t_KV  # [n_heads * d_head]
v_t_C = W_UV @ c_t_KV  # [n_heads * d_head]

# Step 4: RoPE 仅部分维度
k_t_R = RoPE(W_KR @ h_t)
k_t = concat([k_t_C, k_t_R])

# Step 5: Attention 标准计算
output = softmax(q_t @ K_cache.T / sqrt(d_head)) @ V_cache

# KV Cache 对比：
# MHA: n_heads × d_head × 2(K+V) → e.g., 64KB/token
# MLA: d_latent × 1(latent) → e.g., 1KB/token (64x压缩)
```

术语一般如何实现？如何使用？
- DeepSeek-V2 提出，21B 激活参数达到 Llama3-70B 水平
- KV Cache 减少 93.3%，推理吞吐提升 5.76x
- EPS-MoE 因 MLA 使用 DP+EP（非 TP+EP），避免 MLA 的额外 TP 通信开销
- 适用：长上下文（128K token）内存受限推理场景

涉及论文标题：
- EPS-MoE: Expert Pipeline Scheduler for Cost-Efficient MoE Inference
