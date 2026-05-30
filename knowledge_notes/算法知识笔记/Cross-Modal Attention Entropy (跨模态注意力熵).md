## Cross-Modal Attention Entropy (跨模态注意力熵)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Cross-Modal Attention Entropy 是 MEDA 提出的用于量化多模态 LLM 每层跨模态注意力分布特性的度量指标。它通过同时计算文本→视觉（T→V）和视觉→文本（V→T）两个方向的注意力熵，捕捉该层对跨模态 token pair 的注意力集中/分散程度。公式为：

$$E_{CM}^l = -(E_{TV}^l + E_{VT}^l)$$

其中：
$$E_{TV}^l = \frac{1}{|T|} \sum_{i=1}^{n_T} \sum_{j=1}^{n_V} A_{TV}^l[i,j] \log A_{TV}^l[i,j]$$
$$E_{VT}^l = \frac{1}{|V|} \sum_{i=1}^{n_V} \sum_{j=1}^{n_T} A_{VT}^l[i,j] \log A_{VT}^l[i,j]$$

A_TV 和 A_VT 分别是文本 query 对视觉 key 和视觉 query 对文本 key 的跨模态注意力矩阵（公式 4）。该度量源于信息论中的 Shannon 熵，被观察到能有效反映不同层注意力密度的差异：早期层（如 Layer 1）注意力分散、熵较高；深层（如 Layer 24）注意力集中于少数关键跨模态 token 对、熵较低（Figure 2）。这种层级差异指导后续的动态 KV cache 分配策略。
从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**跨模态注意力熵计算流程**：
```
# 输入：多模态 prompt X（含 text tokens X_T, visual tokens X_V）
# 对每层 l:

# Step 1: 标准 QKV 投影
Q^l = X W_Q^l      # [L, D]
K^l = X W_K^l      # [L, D]

# Step 2: 按 modality index 分离 text 和 visual
Q_T^l = Q^l[text_indices]      # [n_T, D]
K_T^l = K^l[text_indices]      # [n_T, D]
Q_V^l = Q^l[visual_indices]    # [n_V, D]
K_V^l = K^l[visual_indices]    # [n_V, D]

# Step 3: 计算跨模态注意力矩阵
A_TV^l = Softmax(Q_T^l · (K_V^l)^T / √D)    # [n_T, n_V]
A_VT^l = Softmax(Q_V^l · (K_T^l)^T / √D)    # [n_V, n_T]

# Step 4: 计算跨模态注意力熵
E_TV^l = -(1/|T|) Σ_i Σ_j A_TV^l[i,j] · log(A_TV^l[i,j])
E_VT^l = -(1/|V|) Σ_i Σ_j A_VT^l[i,j] · log(A_VT^l[i,j])
E_CM^l = E_TV^l + E_VT^l   # 注意：公式 (6) 最终带负号，但用于分配时直接用 exp(E_CM^l)

# Step 5: 用于动态 KV cache 分配
α_l = exp(E_CM^l) / Σ_k exp(E_CM^k) · L · ρ
S_l = α_l · S
```

**直觉**：低熵 → 注意力集中于少数关键 token pair → 层已完成跨模态信息聚焦 → 分配更少 KV cache。高熵 → 注意力分散于大量 token pair → 层仍在广泛处理跨模态交互 → 分配更多 KV cache。

术语一般如何实现？如何使用？

跨模态注意力熵在 prefill 阶段计算一次（O(n_T · n_V) per layer，相比 O(L²) 的 self-attention 可忽略），产生 per-layer 熵值向量后用于确定各层的 KV cache budget S_l。由于只在 prefill 执行一次且不需要训练或调优参数，它与任何 MLLM（LLaVA、InternVL、LLaVA-Video 等）和量化/稀疏技术兼容。代码开源：https://github.com/AIoT-MLSys-Lab/MEDA。

涉及论文标题：
- MEDA: Dynamic KV Cache Allocation for Efficient Multimodal Long-Context Inference

---
