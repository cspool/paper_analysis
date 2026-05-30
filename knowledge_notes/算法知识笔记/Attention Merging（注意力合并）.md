## Attention Merging（注意力合并）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Attention Merging 是 VisiPruner (EMNLP 2025) 提出的一种训练无关的视觉 token 压缩技术：在 MLLM 推理时，将某层的所有视觉 cross-attention weights 强制合并/聚焦到单个视觉 token 上，从而在保持 attention 分布稳定性（避免 softmax 归一化崩塌）的同时，将 N_text × N_v 的 cross-attention 计算降为 N_text × 1。具体操作：在 cross-attention matrix A ∈ R^{N_text × N_v} 上，对每一行 i（对应每个文本 token），将分散在所有视觉 token 上的 attention weights 求和后全部赋给一个随机选定的视觉 token k，其余位置的权重置零：A^{(l)}_{i,j} = Σ_{v∈V} A^{(l)}_{i,v} if j=k else 0。使用 attention merging 后，文本 token 不再需要与所有 N_v 个视觉 token 逐一交互，仅需与单个合并后的 representation 交互即可维持 attention 分布的数值稳定性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Attention Merging in VisiPruner (Shallow Layer 1)
# Input: Q_t [N_text, d], K_v [N_v, d], V_v [N_v, d]
# k: randomly selected visual token index

# Step 1: compute original cross-attention
scores = Q_t @ K_v.T / sqrt(d)        # [N_text, N_v]
A = softmax(scores, dim=-1)            # [N_text, N_v]

# Step 2: merge all attention weights to token k
A_merged = zeros_like(A)
for i in range(N_text):
    total_weight = sum(A[i, :])        # typically close to 1
    A_merged[i, k] = total_weight      # all weights -> single token

# Step 3: compute cross-attention output with merged attention
H_cross = A_merged @ V_v              # [N_text, d]
# 仅需 V_v[k] * total_weight，等价于用单个 token 的 value 加权
```

Annotations: k 可以是任意视觉 token index，VisiPruner 实验证实随机选择 token index=1/128/288/576 对性能几乎无影响（GQA 均保持 ~61.8），证明没有特定视觉 token 是关键的。该技术仅在 layer 1 发生——layer 2+ 可以直接跳过 cross-attention（system prompt 等文本 token 已可替代视觉 sink）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VisiPruner 在 LLaVA 推理代码中通过修改 HuggingFace LLaMA 模型的 attention mask 实现：在 layer 1 的 cross-attention 计算后，不直接使用 softmax 输出，而是先对 A 做"行求和→单列赋值"的后处理，再与 V_v 相乘。代码位于 GitHub repo `llava/cli_pruning.py` 中，通过配置 `shallow_mid_layer` 参数指定浅/中层分界。该技术的前提是经过 VisiPruner 的系统分析确认浅层视觉 token 仅起 attention sink 作用，因此可直接绕过——若无此分析直接使用可能破坏跨模态融合。Attention merging 可将浅层 cross-attention FLOPs 从 O(N_text × N_v) 降至 O(N_text)，在 LLaVA-v1.5 7B（576 visual + 74 text）中节省显著计算量。

涉及论文标题：
- VisiPruner__Decoding_Discontinuous_Cross-Modal_Dynamics_for_Efficient_Multimodal_LLMs
