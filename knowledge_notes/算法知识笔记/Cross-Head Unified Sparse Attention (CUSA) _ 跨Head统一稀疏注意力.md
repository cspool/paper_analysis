## Cross-Head Unified Sparse Attention (CUSA) / 跨Head统一稀疏注意力

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

CUSA 是 LessIsMore 论文提出的免训练稀疏注意力机制的核心组件。它解决长期困扰稀疏注意力方法的核心问题：推理（reasoning）任务中 token 重要性是全局属性，而非每个 attention head 的局部属性。CUSA 的工作流程：(1) 每个 attention head 基于精确 attention score 独立提案自己的 top-k 候选 token；(2) 通过 UnionFlatten 将所有 head 的候选聚合为一个统一候选集；(3) 全局排序后保留最高分的 K·(1-r) 个 token（r 为近邻保留比例）；(4) 所有后续 attention head 和 layer 共享这个统一的 token 索引集 ρ 进行稀疏注意力计算。关键设计在于聚合步骤——CUSA 不假设 head 功能完全相同（常规假设是对不同 head 不同 token subset），而是利用 LessIsMore 论文中观察到的跨 head 空间局部性（cross-head spatial locality）来消除个别 head 的噪声选择，降低 selection variance。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

CUSA 在 LessIsMore 的 decoding pipeline 中的具体执行过程：

**伪代码（基于 Algorithm 1）**：
```
Input: hidden state h, KV cache C, token budget K, static ratio r
for each decoder layer i:
    q, k, v = f(W_qkv, h)           # 标准 QKV 投影
    C.append(k, v)                   # 更新 KV cache
    if i is Full Attention Layer:    # 仅前 2 层
        o = FullAttention(q, C[:])   # 完整注意力
    elif i is Token Selection Layer: # 仅 1 层（如 Layer 12）
        o = FullAttention(q, C[:])
        P = q @ C.K^T                # [H, 1, L_kv]，注意力分数矩阵
        # 各 head 独立提案
        rho_head = TopKIndices(P[:, :-(K·r)], k=K·(1-r))
        # 跨 head 统一聚合
        rho_unified = UnionFlatten(rho_head)
        rho_recent = Recent(K·r)
        rho = rho_unified[:K·(1-r)] ∪ rho_recent
    else:                            # 其余所有层
        o = SparseAttention(q, C[rho])  # 仅对 rho 中的 token 计算注意力
    h = FFN(o)
return lm_head(h)
```

**GQA 下的 CUSA 张量计算**（以 DeepSeek-R1-8B 为例，hq=32, hkv=8, r=4）：
1. P = q @ C.K^T  # [32, 1, L_kv]（Token Selection Layer 中计算）
2. For each KV group g (0..7):
     P_g = P[4g:4g+4, :, :]        # [4, 1, L_kv]，4 query heads 共享 1 KV head
     对每个 query head h in [0..3]:
       idx_g_h = TopK(P_g[h], k=K·0.75)
     聚合 4 个 head 的提案: idx_g = union(idx_g_0, idx_g_1, idx_g_2, idx_g_3)
3. 全局统一: idx_all = unique(flatten([idx_g for g in 0..7]))
4. 按原始 attention score 排序: idx_sorted = sort_by_score(idx_all)
5. 保留历史 token: idx_hist = idx_sorted[:K·0.75]
6. 保留近邻 token: idx_recent = [L_kv-K·0.25, ..., L_kv-1]
7. rho = idx_hist ∪ idx_recent  # 所有 32 query heads 和后续 layer 共享

术语一般如何实现？如何使用？

CUSA 在 LessIsMore 中实例化为 TidalDecode 的上层机制，通过 2 个 Full Attention Layer + 1 个 CUSA Token Selection Layer + 剩余 Sparse Attention Layers 实现。CUSA 的关键属性：
- **低频重选**：token 选择仅在 1-2 层执行（token selection layer），产生的 ρ 跨后续层复用。图 4 验证：仅 Layer 2 选择 vs 每层都选，attention recall 几乎相同（~95% vs ~96%），说明全局 token 重要性跨层稳定。
- **GQA kernel 友好**：所有 query heads 共享统一 ρ，单次 global-to-shared memory 加载即可服务整个 KV group 的 query heads，消除 per-head 独立选择导致的冗余 KV loading（G2S 从 2.34MB 降至 1.04MB, Table 4）。
- **免训练**：不需要任何模型权重修改或 fine-tuning，纯推理时机制。
- **架构无关**：可集成到任何 selection-based 稀疏注意力框架中，兼容 GQA 和 MHA。

代码开源：https://github.com/DerrickYLJ/LessIsMore

涉及论文标题：
- Less Is More: Fast and Accurate Reasoning with Cross-Head Unified Sparse Attention
