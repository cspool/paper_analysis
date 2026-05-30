## Head-Level KV Cache Budget Allocation (注意力头级别的 KV Cache 预算分配)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Head-Level KV Cache Budget Allocation 是一种在 Transformer 推理中将有限的 KV cache 总预算按 attention head 的重要性非均匀分配的策略。与 token-level 压缩（如 H2O、A2SF 在每个 head 内独立选择保留哪些 token）和 layer-level 分配（如 PyramidKV 深层分配更少 cache）不同，head-level 分配认识到不同 attention head 对模型性能的贡献差异显著，因此应为重要 head 保留更多 KV pairs，不重要 head 保留更少或仅保留 local window。

CoKV 的预算分配公式：

$$c_i = B \cdot \frac{\mathcal{NSV}_i}{\sum_{j=1}^n \mathcal{NSV}_j} + s$$

其中 B 为共享预算总数（总 KV pairs 减去所有 head 的 local window 固定部分），NSV_i 为 head h_i 的归一化 SSV 分数（α 个最低分 head 的 NSV=0，仅保留 local window），s=8 为 local window 大小。最终每个 head 的 cache size c_i = 按 SSV 分数比例分配 + 固定 local window。

CoKV 实验发现：当平均 cache size 达 512 tokens/group（约 6.4% of full cache for 8K context），CoKV 平均准确率超越 Full KV Cache（说明 CoKV 成功识别并限制了有负面贡献的 heads）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Head-Level Budget Allocation 的完整流程（CoKV）**：

```
// === 预计算阶段（Offline） ===
// 对每个 task，计算 SSV 分数
SSV[task_id] = compute_SSV(model, validation_set, H={32,64,96,128}, M=250)

// === 推理阶段（Online） ===
B = total_shared_budget  // e.g., (avg_cache_size - s) * num_groups
s = 8                    // local window
α = optimal_alpha[task]  // 从验证集选取

// 加载指定 task 的 SSV
ssv = load_SSV(task_id)  // shape: [num_groups=256]

// Step 1: Min-max normalize with α-low cutoff
sorted_ssv = sort(ssv)
min_α = sorted_ssv[α]
max_ssv = sorted_ssv[-1]
nsv = zeros_like(ssv)
for i where ssv[i] > min_α:
    nsv[i] = (ssv[i] - min_α) / (max_ssv - min_α)

// Step 2: Proportional allocation
total_nsv = sum(nsv)
for each group i:
    if nsv[i] == 0:
        c_i = s              // only local window
    else:
        c_i = B * (nsv[i] / total_nsv) + s

// Step 3: Per-group token selection (SnapKV mechanism)
for each group i:
    A_win = softmax(Q_win @ K_prefix^T / sqrt(d_h))
    token_scores = A_win.max(dim=1).mean(dim=0)
    keep_idx = topk(token_scores, c_i)
    K_retain = K_prefix[keep_idx]
    V_retain = V_prefix[keep_idx]
    K_cache = cat([K_retain, K_local])
    V_cache = cat([V_retain, V_local])
```

**与其他分配策略的对比**：

| 策略 | 粒度 | 重要性依据 | 代表方法 |
|------|------|-----------|---------|
| Uniform | per-head | 无（均分） | SnapKV |
| Layer-level | per-layer | 层深度（金字塔形） | PyramidKV |
| Head-level (独立) | per-head | 个体 retrieval-reasoning 分 | HeadKV-R2 |
| Head-level (稀疏度) | per-head | 个体 concentration degree | Ada-SnapKV |
| Head-level (合作) | per-head | 合作博弈 Shapley Value | CoKV |

术语一般如何实现？如何使用？

Head-level 分配在推理前计算各 head 的 cache size c_i，推理时在每个 Transformer 层 prefill 完毕后按照各自的 c_i 独立执行 token eviction。与 GQA 兼容：GQA 中一组 query heads 共享同一 KV cache，CoKV 将每个 KV group 作为合作博弈的玩家，评估 group-level SSV 后按 group 分配 budget。CoKV 在 Mistral-7B 和 Llama-3-8B 上验证，128 tokens/group 时保留 Full KV 97.29% 的性能。代码开源：https://github.com/nawei1010/CoKV。

与 CoKV 的 head-level 分配不同，CompressKV 采用 **Error-Aware 层级自适应分配（Layer-Level）**：离线在 LongBench 上模拟极端压缩（每层仅保留 ≈32 tokens），计算每层 attention output 的 Frobenius norm 重建误差 e^(l) = Σ_t ||O_comp^l - O_full^l||_F / ||O_full^l||_F，跨数据集归一化平均后得到层级重要性分数。在线推理时按分数比例分配 cache budget 给各 layer（而非各 head），设置 per-layer 上下界 [m=32, M=3×B_per-layer]。优势：(a) 离线计算无在线开销；(b) 基于真实压缩误差而非 attention 统计量（entropy/variance），跨模型泛化性更好。

涉及论文标题：
- CoKV: Optimizing KV Cache Allocation via Cooperative Game
- CompressKV: Semantic Retrieval Heads Know What Tokens are Not Important Before Generation
- KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction

注：KVzip 采用不同的 non-uniform allocation 策略——不是按 head 各自分配固定 budget，而是在所有 head 的所有 KV pairs 中取全局 top r% 最高得分进行保留。这天然导致重要性高的 head 保留更多 KV pairs，无需显式计算 per-head budget。该方法比 CoKV 的 Shapley-value 分配更简单，在 KVzip 的实验中优于 uniform allocation（Figure 16）。

---
