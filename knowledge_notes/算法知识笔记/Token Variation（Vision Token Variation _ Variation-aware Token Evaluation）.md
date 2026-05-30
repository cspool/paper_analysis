## Token Variation（Vision Token Variation / Variation-aware Token Evaluation）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Variation（视觉 Token 变异性）是指 LVLM 推理过程中，visual token 表示在相邻 LLM transformer 层之间发生的变化幅度。核心假设：参与 LLM 推理计算的高重要性 token 会在跨层传播时产生显著表示变化（high variation），而被动经过的 token 保持相对稳定（low variation）。V2Drop（CVPR 2026）首次系统性地从 token variation 视角研究 token 压缩，证明 variation 信号与 token 的任务相关性高度一致：high-variation tokens 对应语义重要区域（问题相关物体），low-variation tokens（"lazy tokens"）对应无关背景区域。Variation 的度量指标包括 L1 Distance、L2 Distance 和 Cosine Similarity，其中 L2 Distance 提供了最佳的 performance-efficiency 平衡。该度量与位置无关（spatial-agnostic），因此不受 attention-based 方法的位置偏见影响。

论文给出理论支撑（Theorem 1）：在一阶 Taylor 展开下，||Δf_j|| ≈ ||J_j||_op · ||Δx_j^(t)||，即 token j 对模型输出的影响与其跨层变化量 ||Δx_j^(t)|| 近似成正比——variation 是 token importance 的计算高效代理。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Variation 计算发生在 LVLM decoder 的指定剪枝层。对 V2Drop 在 LLaVA-1.5-7B 上的典型配置（layers 3, 17, 22）：

```
# Step 1: Variation Computation at pruning layer l_k
for i in range(M_curr):  # M_curr = current vision token count
    # f_i^(l_k-1): token i at previous layer output
    # f_i^(l_k):   token i at current layer (after Attn+FFN+Residual)
    s[i] = ||f_i^(l_k) - f_i^(l_k-1)||_2  # L2 distance

# Step 2: Sort by variation (descending)
indices = argsort(s, descending=True)

# Step 3: Retain top-K with highest variation
F_v_retained = {f_indices[0], ..., f_indices[K_l - 1]}
# Drop: F_v_dropped = remaining tokens (low-variation lazy tokens)
```

计算开销：M=576, D'=4096 时每层约 7M FLOPs（3MD'），仅为单层 attention（32B FLOPs）的 0.022%；三层总计约 21M FLOPs（完整 forward 的 0.002%）。Variation 信息仅需简单的张量相减 + L2 norm，无需访问 attention map，天然兼容 FlashAttention。

与 Representation Shift 的区别：Representation Shift 度量 MLP 输入→输出的变化（Δ = ||MLP(LN(x)) - x||）；V2Drop 度量相邻层间完整 token 表示的变化（Δ = ||x^(l) - x^(l-1)||），更适合在多个 LLM layer 间进行渐进式剪枝。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：在指定 LLM transformer layer 的 residual add 之后，读取当前层和上一层的 visual token hidden states，计算 pairwise L2 距离。在 PyTorch 中：
```python
# hook at layer l after residual connection
with torch.no_grad():
    prev_hidden = hidden_states_buffer[l-1][vision_mask]  # [M, D]
    curr_hidden = hidden_states[l][vision_mask]            # [M, D]
    var_scores = torch.norm(curr_hidden - prev_hidden, dim=-1)  # [M]
    _, topk_idx = torch.topk(var_scores, k=K_l)
    # retain only top-k vision tokens
```

使用场景：(a) 任何 ViT-Projector-LLM 架构的 LVLM 推理加速；(b) 高分辨率图像（576+ tokens）和长视频（1024+ tokens）场景；(c) 单 GPU（A100/3090/4090）部署，需降低延迟和显存；(d) 作为其他压缩方法的排序信号（替代 attention score）。开源：https://github.com/xuyang-liu16/V2Drop（Apache-2.0）。

涉及论文标题：
- V2Drop__Variation-aware_Vision_Token_Dropping_for_Faster_Large_Vision-Language_Models
