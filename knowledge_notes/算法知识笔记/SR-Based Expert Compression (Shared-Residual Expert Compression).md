## SR-Based Expert Compression (Shared-Residual Expert Compression)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

SR-Based Expert Compression（共享-残差专家压缩）是 HybridEP 提出的用于在 MoE 跨 DC 训练中大幅减少 expert 参数传输量的压缩算法。核心思想是将 expert 参数分解为 shared expert（共享专家，所有 expert 的平均值，学习 expert 间的共性/冗余知识）和 residual expert（残差专家，expert - shared_expert，捕捉每个 expert 特有的知识差异），仅传输压缩后的残差。关键动机来自两个观察：(1) Expert 权重分布比 activation data 更集中、outlier 更少（Figure 4），具有更高的可压缩性；(2) Expert 间的主要差异集中在少数参数上（Figure 9a），残差的分布比原始权重更集中和稀疏。压缩算法分两阶段——SREncode（编码：计算残差 → Top-k 保留绝对值最大的 k 个元素 → value-index 稀疏格式存储）和 SRDecode（解码：从稀疏格式 scatter 恢复残差 → 与 shared_expert 相加恢复完整 expert，其中恢复和加法被 fused 以减少 overhead）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# SR-Based Expert Compression Pipeline (HybridEP)
# 配置: E experts, 每个 expert 参数量 P_E, 压缩比 CR, k = P_E // CR

# === 算法初始化 ===
shared_expert = mean(expert_0, expert_1, ..., expert_{E-1})  # 各专家均值
# shared_expert 通过 backward All-Reduce 在每次 iteration 同步梯度

# === Phase 1: SREncode (与 optimizer.step() 融合执行) ===
def SREncode(expert, shared_expert, CR):
    # Step 1: 残差分解
    residual = expert - shared_expert    # shape: [P_E]
    # residual 分布更集中、更稀疏 (Figure 9a "res" suffix)
    
    # Step 2: Top-k 稀疏压缩
    k = P_E // CR                       # 例如 P_E=4.7M, CR=50 → k≈94k
    abs_residual = abs(residual)
    _, topk_indices = topk(abs_residual, k)  # 选绝对值最大的 k 个位置
    topk_values = residual[topk_indices]     # 保留对应的值(含符号)
    
    # Step 3: 稀疏格式存储
    compressed = (topk_values, topk_indices)  # 存储为 value-index pairs
    return compressed
    # 压缩后数据量: k * (sizeof(FP16) + sizeof(INT32))
    # 例如: 94k * (2B + 4B) ≈ 0.56 MB vs 原始 4.7MB → 8.4× (与带宽和 CR 相关)

# === Phase 2: SRDecode (与 expert FFN computation 融合执行) ===
def SRDecode(compressed, shared_expert):
    values, indices = compressed
    
    # Step 1+2: 恢复残差 + 加回共享专家 (fused)
    expert_recovered = shared_expert.clone()        # [P_E]
    expert_recovered.scatter_(indices, values)      # 将 values 写入 indices 位置
    # 等价于: expert = shared_expert + residual_recovered
    #   其中 residual_recovered[i] = values[j] if i == indices[j] else 0
    
    return expert_recovered
    # 融合 overhead: SRDecode + expert FFN 融合可减少 ~45% overhead (Figure 15b)

# === 训练 iteration 中的使用 ===
# 前一步: SREncode
for expert in local_experts:
    compressed = SREncode(expert, shared_expert, CR)
    send_queue.push(compressed)

# 当前步: AG 通信 → SRDecode → Expert FFN
for layer_experts in send_queue:
    all_compressed = AllGather(layer_experts, domain_group)  # 域内收集
    for c in all_compressed:
        expert = SRDecode(c, shared_expert)
        output += gate_weight * expert_ffn(expert, tokens)
```

关键词: 为什么用 shared + residual 而非直接压缩？Figure 14 对比显示：HybridEP w/o S（直接 Top-k 压缩）的 loss 显著高于 baseline，而 HybridEP w/ S（shared expert + residual Top-k）的 loss 与 baseline 几乎一致（50× CR），证明 shared expert 对维护精度至关重要——shared expert 捕获了 expert 间的共性知识，residual 仅编码微小的专家差异。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 实现依赖于 PyTorch 的 scatter 操作和 CUDA 优化的 Top-k kernel。编码通过 `torch.topk(abs(residual), k)` 实现，解码通过 `tensor.scatter_(dim, indices, values)` 融合完成。
- Shared expert 占用的额外 GPU memory 通过 offloading 到 CPU（ZeRO-Offload 兼容策略）管理——local experts 被 offload 到 CPU memory 而 shared experts 保留在 GPU memory。
- SREncode 与 optimizer step 融合的关键：在 Adam optimizer 更新参数后，立即对更新后的 expert 执行 SREncode，利用 GPU 已经在更新 expert 参数时的高计算利用率，减少额外 kernel launch 开销。实验显示融合可减少 ~30% 编码 overhead（Figure 15a）。
- SRDecode 与 expert FFN 融合的关键：SRDecode 的 scatter 操作可与 FFN 的第一个 GEMM（gate projection）通过 CUDA stream 或 kernel fusion 重叠，减少 ~45% overhead（Figure 15b）。
- 压缩比 (CR) 是一个超参数，论文在 50× 下验证无精度损失（Figure 14），但更高压缩比下的行为未充分探索（论文未展示 >50× CR 的结果，仅注明 page limit）。

涉及论文标题：
- HybridEP: Scaling Expert Parallelism to Cross-Datacenter Scenario via Hybrid ExpertData Transmission
