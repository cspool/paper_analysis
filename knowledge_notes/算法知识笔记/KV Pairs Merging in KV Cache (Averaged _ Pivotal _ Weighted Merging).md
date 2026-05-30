## KV Pairs Merging in KV Cache (Averaged / Pivotal / Weighted Merging)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

KV Pairs Merging 是 LOOK-M 提出的一种 KV cache 压缩辅助机制，将对被 evicted（淘汰）的 KV pair 不直接丢弃，而是通过 many-to-one nearest-neighbor matching 将其信息合并到 conserved（保留）的 KV pair 中。该机制解决纯 eviction 方法的核心缺陷：被淘汰的 token 中可能包含关键上下文信息（尤其是视觉细节），直接丢弃会导致 hallucination 和 contextual inconsistencies。

LOOK-M 提出三种合并策略：(a) **Averaged Merging (A-Merge)**——将 evicted token 与其最近匹配的 conserved token 直接求均值，公式为 k_c = 1/(L_sim+1) × (k_c + Σ k_sim[i])；(b) **Pivotal Merging (P-Merge)**——先对每个 evicted token 与其 closest conserved token 做平均融合产生 "pivotal token"，再将所有 pivotal tokens 与 conserved token 平均，强调 conserved token 的权重比例，公式为 k_c = 1/(L_sim+1) × {k_c + 0.5 × Σ(k_sim[i] + k_closest)}；(c) **Weighted Merging (W-Merge)**——基于 similarity matrix S 中的相似度值动态分配权重，而非静态平均，公式为 k_c = 1/(L_sim+1) × {k_c + Σ(k_sim[i] × S[x][y])}。三种策略中 TP + P-Merge（text-prior + pivotal merging）在多模态长上下文任务上取得最佳效果。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**KV Pairs Merging 伪代码（LOOK-M，基于 Section 3.3）**：

```
# 输入：conserved K_c [N+M, D], evicted K_e [L-(N+M), D]
# 输出：merged conserved K_c' [N+M, D]

# Step 1: 计算 similarity matrix between K_e and K_c
S = zeros(len(K_e), len(K_c))
for i in range(len(K_e)):
    for j in range(len(K_c)):
        S[i,j] = cosine_similarity(K_e[i], K_c[j])   # (a^T b)/(||a|| ||b||)

# Step 2: 对每个 conserved token j，收集其 maximum similarity set
match_map = defaultdict(list)
for i in range(len(K_e)):
    j_star = argmax(S[i,:])          # 最相似的 conserved token
    match_map[j_star].append(i)      # many-to-one 关系

# Step 3: 合并策略（三选一）

# (a) Averaged Merging
for j in match_map:
    indices = match_map[j]
    K_c[j] = (K_c[j] + sum(K_e[i] for i in indices)) / (len(indices) + 1)

# (b) Pivotal Merging
for j in match_map:
    indices = match_map[j]
    pivots = [(K_e[i] + K_c[j]) / 2 for i in indices]  # pivotal tokens
    K_c[j] = (K_c[j] + sum(pivots)) / (len(indices) + 1)

# (c) Weighted Merging
for j in match_map:
    indices = match_map[j]
    weighted_sum = sum(K_e[i] * S[i,j] for i in indices)  # similarity-weighted
    K_c[j] = (K_c[j] + weighted_sum) / (len(indices) + 1)

# Value 合并使用 Key 的相同 similarity matrix 和加权权重（KV-pair alignment property）
```

**与 Token Merging (ToMe/AIM) 的关键差异**：

| 维度 | Token Merging (AIM/ToMe) | KV Pairs Merging (LOOK-M) |
|------|-------------------------|--------------------------|
| 合并对象 | 输入 token embedding（visual tokens） | KV cache 中的 Key/Value 张量 |
| 合并阶段 | LLM 输入前 / ViT 层间 | Prompt prefill 完成后、decode 前 |
| 合并目的 | 减少 LLM 输入 token 数 → 减少 FLOPs | 保留 evicted token 的上下文信息 → 补偿 eviction 精度损失 |
| 相似度度量 | 余弦相似度（embedding 间） | 余弦相似度（Key 向量间） |
| 合并粒度 | 二对一成对合并（每次减半迭代） | many-to-one nearest-neighbor 匹配 |

术语一般如何实现？如何使用？

在 LOOK-M 实现中，KV Pairs Merging 在每层 Transformer 的 prefill 之后、eviction 之后执行。合并基于 Key 向量的余弦相似度矩阵，相似度矩阵和加权权重在 Key 和 Value 之间共享（alignment property）。无需任何训练或微调，以即插即用方式集成到 HuggingFace Transformers 推理流程。三种合并策略通过配置开关选择，TP + P-Merge 为默认最优配置。代码开源：https://github.com/SUSTechBruce/LOOK-M。

**MEDA 的差异化使用**：MEDA 采用 A-Merge（平均合并）策略，但与 LOOK-M 的关键不同在于：(1) MEDA 的合并发生在 entropy-guided dynamic layer-wise budget allocation 之后——每层根据跨模态注意力熵获得不同的 KV cache budget；(2) MEDA 使用 text-prior 累积注意力分数进行 KV pair 选择——在合并前先通过加 max(A_s) 偏置确保关键文本 token 不被 evict；(3) MEDA 仅需 A-Merge 即可取得最佳效果，消融实验验证移除 merging 会导致 CLEVR-Change ROUGE-L 从 18.9 降至 18.2。MEDA 的合并与熵引导分配互补——前者保留被淘汰 token 的信息，后者确保不同层获得与实际注意力密度匹配的 KV cache 大小。

涉及论文标题：
- LOOK-M: Look-Once Optimization in KV Cache for Efficient Multimodal Long-Context Inference
- MEDA: Dynamic KV Cache Allocation for Efficient Multimodal Long-Context Inference
- Multi-head_Temporal_Latent_Attention

**MTLA 的 Hyper-network Dynamic Weighted Merging**：MTLA 提出第四种 KV merging 策略——通过 hyper-network 以输入内容为条件动态生成 per-position merge weight，而非基于相似度的静态合并。区别于 LOOK-M/MEDA 的 post-eviction merging（先逐出再合并），MTLA 的 merging 是架构内嵌的（architecture-level）——在 attention 计算前按固定 stride s 将 consecutively adjacent latent vectors 合并，无需 eviction 步骤。权重生成：w_i = Sigmoid(Linear(c_i) · Linear(pe_j))，其中 c_i 为第 i 个 token 的 latent vector，pe_j 为位置嵌入。Sigmoid gate 而非相似度，使合并策略 learnable 且 data-driven。该策略是 MTLA 的核心创新，使 KV cache 序列长度从 T 缩减至 T/s。

---
