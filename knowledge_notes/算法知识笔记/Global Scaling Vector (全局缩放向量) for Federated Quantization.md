## Global Scaling Vector (全局缩放向量) for Federated Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Global Scaling Vector (s_g) 是 FedWSQ 用于在 federated 设置中协调各 client 量化一致性的机制。s_g = [s_{g,1}, ..., s_{g,L}]^T 是一个 L 维向量，其中每个元素 s_{g,l} 是第 l 层 LMPU 的全局 scale factor。更新方式：每轮通信后，server 收集各 client 的 local scale vectors s_i（每层 LMPU 的标准差），通过 EMA 更新 s_g ← (1-β)s_g + β·(1/|S|)·Σ_{i∈S} s_i，其中 β 为 momentum 参数（默认 0.1）。Client 在量化 LMPU 前从 server 获取 s_g，将 LMPU 各层张量除以对应的 s_{g,l} 实现归一化。与 FedPAQ 的 per-tensor absmax scaling 不同，global scaling vector 提供了三层优势：(1) per-layer scaling 更精细地匹配各层 LMPU 量级差异；(2) EMA 聚合使 scale 平滑更新，抵抗 client 间波动；(3) 各 client 使用统一的 global scale，确保量化边界一致，server 端 dequantization 信息不丢失。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Global scaling vector 的更新和使用流程：

```python
# Server-side: s_g update (after LMPU aggregation)
# Input: s_i for i ∈ S_t (local scale vectors, each shape [L])
s_mean = (1/|S_t|) * sum(s_i for i in S_t)  # element-wise mean
s_g = (1 - beta) * s_g + beta * s_mean       # EMA update, beta=0.1

# Client-side: using s_g for DANUQ quantization
# Input: ΔW_i (LMPU, L tensors), s_g (global scale vector)
for l in range(L):
    ΔW_norm_l = ΔW_i[l] / s_g[l]          # normalize layer l
    ΔW̄_i[l] = DANUQ_quantize(ΔW_norm_l)    # quantize to B-bit
    s_i[l] = std(ΔW_i[l])                  # compute local scale
return (ΔW̄_i, s_i)

# Server-side: dequantization using transmitted scales
for l in range(L):
    Δ_i[l] = DANUQ_dequantize(ΔW̄_i[l]) * s_i[l]  # restore full-precision
```

**Annotations**: s_g 的维度 L = 模型层数（如 ResNet-18 约 20 层卷积+FC，每层一个 scale）。s_i 以 float32 传输，overhead = L × 4 bytes ≈ 80 bytes（可忽略不计）。β=0.1 使 s_g 在约 10 轮内适应分布变化。初始化 s_g^0 通常设为 1.0 或从第一轮 local training 计算。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Global scaling vector 是 FedWSQ 设计的协作量化机制。实现上，s_g 在 server 端维护为 PyTorch buffer（形状 [L]），每轮在聚合 LMPU 后更新。s_g 对训练稳定性至关重要：如果各 client 使用独立的 local scale（如 FedPAQ），相同数值在不同 client 的量化中可能映射到不同 QL，导致 server 聚合时信息破坏。s_g 通过 EMA 平滑更新既保证一致性又允许随时间适应分布变化。β 的选择需平衡响应速度与稳定性——过大的 β 会过度受单轮 client 子集影响，过小则无法跟踪分布漂移。FedWSQ 实验显示 β=0.1 在多数场景下表现良好。

涉及论文标题：
- FedWSQ Efficient Federated Learning with Weight Standardization and Distribution-Aware Non-Uniform Quantization
