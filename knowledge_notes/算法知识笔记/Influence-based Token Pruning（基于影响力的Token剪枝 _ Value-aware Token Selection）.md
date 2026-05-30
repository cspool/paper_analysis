## Influence-based Token Pruning（基于影响力的Token剪枝 / Value-aware Token Selection）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Influence-based Token Pruning 是 VisiPruner 提出的中层视觉 token 选择方法，核心思想是：不依赖 attention weights（易受 attention sink 污染且分布分散），而是直接评估每个视觉 token mask 后对最后一个输入 token（决定首个回答 token）的 attention output 的改变程度——即该 token 对模型残差流的"实际影响力"。使用两个互补指标：(1) Cosine Similarity：衡量 mask 前后 attention output 的方向变化——cosine 越低说明该 token 对输出方向的贡献越大；(2) L2 Distance：衡量 mask 前后 attention output 的幅度变化——L2 越大说明该 token 对输出大小的贡献越大。当某层的最低 cosine similarity < 0.995 时，将该层定义为 filtering layer（跨模态融合开始的信号）；在 filtering layer 中，L2 distance < 0.2 的视觉 token 被丢弃（其对输出几乎无影响）。该方法在 LLaVA-v1.5 7B 上将 576 个视觉 token 压缩至平均 10.3 个，GQA 仅降 0.7%（62.0→61.3），远优于 attention-based 方法。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Influence-based Token Selection at layer l
# Input: H_all [N_text + N_v, d], last_text_idx = -1
# Hyperparams: cosine_thresh=0.995, l2_thresh=0.2

Q, K, V = W_Q(H_all), W_K(H_all), W_V(H_all)
W = softmax(Q @ K.T / sqrt(d) + M)         # full attention weights
O = reshape(sum_over_heads(W @ V))          # [N_total, d]
O_last = O[last_text_idx]                   # attention output of last text token

for each visual token j:
    # Mask token j: set W'[last_text_idx, j] = 0 for all heads
    W_masked = W.clone()
    W_masked[last_text_idx, j, :] = 0
    O_masked = reshape(sum_over_heads(W_masked @ V))
    O_last_masked = O_masked[last_text_idx]
    
    cos_sim[j] = dot(O_last, O_last_masked) / (||O_last|| * ||O_last_masked||)
    l2_dist[j] = ||O_last - O_last_masked||_2

# Layer-level decision
if min(cos_sim) < 0.995:
    # This is a filtering layer — visual info starts contributing
    keep_mask = l2_dist >= 0.2
    H_v = H_v[keep_mask]                     # discard low-influence tokens
```

Annotations: 该方法在每层对所有视觉 token 逐个评估——每个 token 需要一次 masked attention forward（仅修改 attention weight 矩阵的一列），复杂度为 O(N_v × N_text × d)。但该方法仅在确定过滤层时执行一次（而非每层），因此在 LLaVA-1.5 7B 上平均仅需评估约 9-10 层中的一层。cosine < 0.995 是启发性阈值，从实验观察得出——对应视觉信息开始实质改变残差流方向的拐点。L2 < 0.2 同样来自实验 calibrate。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VisiPruner 在 PyTorch 中实现：通过 hook 拦截每层 attention 模块的 forward，对 attention weights 做 per-token masking 后重算 output。关键实现细节：(1) 仅需评估最后一个文本 token（idx=-1）对各视觉 token 的 cross-attention 影响——因为解码从该 token 开始，其 attention output 直接决定首个生成 token；(2) 在 attention weights 层面 mask（设 W'_{i→j}=0 across all heads），而非直接删除 token（后者会改变序列长度和 positional encoding）；(3) 评估在单层内完成——不需要 propagate 到后续层（与 leave-one-out 方法相比大幅节省计算）。该方法在 multiple MLLM architectures（LLaVA-v1.5 7B/13B, InternVL2.5 8B, Qwen2-VL 7B, MobileVLM-v2 3B）上验证有效，证明了 influence-based 选择优于 attention-based 选择的通用性。代码：https://github.com/EIT-NLP/VisiPruner。

涉及论文标题：
- VisiPruner__Decoding_Discontinuous_Cross-Modal_Dynamics_for_Efficient_Multimodal_LLMs
