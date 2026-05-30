## Expert Sharding (Tensor Sharding of Experts / 专家张量切分)

术语解释
Expert Sharding 是 MoEShard (EuroMLSys '25) 提出的 MoE 推理优化策略，将每个 expert 的权重矩阵（W_i 和 W_o）按张量维度切分到所有 GPU（而非将完整 expert 分配到不同 GPU），使每个 GPU 持有所有 expert 的部分 shard，所有 GPU 处理全部 token 的 partial computation，通过 pointwise sum 恢复完整输出，实现与路由分布无关的 perfect load balancing。

术语是什么？
在传统 Expert Parallelism 中，不同 GPU 持有不同完整 expert，token 按 routing 结果发送到对应 GPU。当 routing 倾斜时，热门 expert 所在 GPU 过载、冷门 expert 所在 GPU 空闲。Expert Sharding 的核心洞察：expert 计算本质上是两次矩阵乘法（x · W_i · W_o），其中 W_i（shape [h_i, h_o]）可列切分、W_o（shape [h_o, h_i]）可行切分到 |G| 个 GPU，每个 GPU g 计算 x · W_i^g · W_o^g → 部分输出 y_g，最终 Σ y_g 等价于完整 expert 输出。因为所有 GPU 的计算量完全相等（均处理全部 token × 所有 expert 的 1/|G| shard），天然实现 perfect load balancing，无需 profiling、专家复制或 token dropping。

从kernel调度角度拆解术语：
```
# Expert Sharding 的 kernel 执行流程（MoEShard, Algorithm 1 + Section 3.2）
# 假设 4 GPU, 128 experts, batch_size=B, seq_len=S, hidden_dim=h

# --- Shard 准备（一次性，推理前完成）---
# 每个 expert e 的 W_i [h_i, h_o] 列切分为 |G| 份
# 每个 expert e 的 W_o [h_o, h_i] 行切分为 |G| 份
# GPU g 持有: {W_i^g[e] : shape [h_i, h_o/|G|]  for all e in E}
#           {W_o^g[e] : shape [h_o/|G|, h_i]  for all e in E}

# --- Forward Pass（per MoE block）---
# Step 1: Token Routing（每 GPU 独立）
m_expert = ROUTER(x)  # x: [B*S, h], m_expert: [B*S], token→expert 映射

# Step 2: Metadata Exchange（轻量 all-to-all broadcast）
m_sizes = count_per_expert(m_expert)  # size=[|E|], 每个 expert 收多少 token
broadcast m_sizes to all GPUs

# Step 3: Token Scatter（全复制 - 与 EP 的本质区别）
# 每 GPU 发送全部 token 给所有其他 GPU
# 每 GPU 发送 ≈ B*S*h*4 bytes → 88 MiB (batch=250, seq=120, h=768)
# NVLink 3.0 600 GiB/s → ~0.15ms, negligible
W[g][e] = tokens from GPU g assigned to expert e  # 2D 组织

# Step 4: Sharded Expert Computation（核心 kernel）
for e in E:                                    # 128 experts
    # Fusion opt 1: concatenate tokens for expert e from all GPUs
    tokens_e = cat([W[0][e], W[1][e], ..., W[|G|-1][e]])  # 合并同 expert token
    shard = LOAD_SHARD(rank, e)                 # W_i^rank[e], W_o^rank[e]
    partial = tokens_e @ W_i_shard @ W_o_shard  # partial output per GPU
    # Fusion opt 2: MegaBlocks sparse MM 将所有 expert 计算融合为 1 kernel
    split partial back to per-GPU results

# Step 5: Gather & Aggregate
send partial results back to source GPUs
y_final = sum(all partial outputs)  # pointwise addition across GPUs
```

与其他 sharding 策略的对比分析（Section 3.2）：
- W_i row-wise + W_o column-wise: W_i row 切分需在 x·W_i 后 cross-GPU sum（中间同步）
- W_i column-wise + W_o column-wise: x·W_i 后需 cross-GPU concat（中间同步）
- **W_i column-wise + W_o row-wise (MoEShard)**: 两次矩阵乘法之间无需中间同步，最优

术语一般如何实现？如何使用？
- 基于 PyTorch 实现，源码: https://github.com/sacs-epfl/moe-inference
- 要求所有 GPU 等容量等算力（同构集群），expert shard 数可被 GPU 数整除
- 代价：token 全复制（NVLink 高带宽吸收）和 partial output 求和（pointwise add, 可忽略）
- 适用场景：encoder-based MoE, batch size 较大（≥100），routing 高度倾斜
- 局限：小 batch（10）时因 token 全复制 overhead 可能慢于 DeepSpeed EP；decoder autoregressive 生成未验证

涉及论文标题：
- Accelerating MoE Model Inference with Expert Sharding

---
