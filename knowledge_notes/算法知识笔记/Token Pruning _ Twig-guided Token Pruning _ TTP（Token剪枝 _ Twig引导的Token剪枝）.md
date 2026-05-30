## Token Pruning / Twig-guided Token Pruning / TTP（Token剪枝 / Twig引导的Token剪枝）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Pruning 是一种通过移除冗余 token 来降低 Transformer 计算复杂度的加速方法。在 VLMs 中，视觉 token 数量通常远超文本 token 且含大量冗余信息，因此在 VLM 的早期层剪枝视觉 token 可显著减少后续层的计算量。传统方法（如 FastV）使用 VLM 早期层（如第 2 层）的 attention map，取文本 token 对视觉 token 的 attention scores 之和，选择 top-R 最重要的视觉 token 保留，丢弃其余。但早期层 attention 对多模态语义理解不充分（"attention signals in early layers are insensitive to the task"），导致剪枝后精度大幅下降。

Twig-guided Token Pruning (TTP) 是 TwigVLM 提出的改进方案：在 base VLM 第 K 层后附加 T 层 twig block，使用 twig 最后一层（深度 K+T，更靠近 prediction head）的 attention map 指导 token 剪枝。由于 twig 层更接近 loss 函数，其 attention 对多模态关系的理解更精准。TTP 流程：输入 tokens X → base VLM 前 K 层 → 得 X^(K)_Mb → twig block → 得 final twig layer attention map A^(K+T)_Ms → 选择 top-R 视觉 token → X̂^(K)_Mb = P(X^(K)_Mb, A^(K+T)_Ms, R) → 传入剩余 VLM 层。配合 FinalWipe 策略在 Kf 层后移除所有视觉 token，平均保留 token 数 R̄ = [M×K + R×(Kf-K)]/L。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# TTP: Twig-guided Token Pruning
# 输入: X = [X_v (visual tokens), X_q (text tokens)]
# K: pruning 位置层, T: twig 层数
# R: 保留的 visual token 数

# Step 1: 前向到共享层
X_K_mb = Mb.forward_layers(X, start=1, end=K)
           # 同时得到 twig 的 attention
X_K_ms, Attn_twig = Ms.forward_twig(X_K_mb)

# Step 2: 用 twig 最后层的 attention 计算重要性
# Attn_twig ∈ R^{(M+N)×(M+N)}: twig 最后层的 attention map
# 取 text tokens 对 visual tokens 的 attention scores
attn_scores = sum(Attn_twig[text_positions, visual_positions])
               # ∈ R^M, 每个 visual token 的重要性分数

# Step 3: Top-R 选择
keep_indices = topk(attn_scores, k=R)
X_kept_vis = X_K_mb[keep_indices]
X_kept = concat([X_kept_vis, X_K_mb[text_positions]])

# Step 4: 传入剩余层
output = Mb.forward_layers(X_kept, start=K+1, end=L)
```

与传统 Token Pruning (FastV) 的对比：
| 特性 | FastV-style Pruning | TTP (TwigVLM) |
|------|-------------------|---------------|
| 剪枝信号来源 | VLM 早期层 attention (如 layer 2) | Twig 最后层 attention (如 layer 5) |
| Attention 深度 | K (浅) | K+T (更深，更靠近 loss) |
| 信号质量 | 低 (attention 对任务不敏感) | 高 (attention 对多模态理解精准) |
| 额外参数 | 无 | T 层 twig block (~10% base VLM) |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：(1) 训练时，仅训练 twig block（冻结 base VLM），使用标准 AR loss + SFT 数据（如 LLaVA-665K），训练耗时约 10% 的 base VLM 训练时间。(2) 推理时，twig block 同时在 prefilling（TTP）和 decoding（SSD）阶段使用。开源实现：https://github.com/MILVLG/twigvlm，使用 `--twig-K 2 --twig-T 3` 配置。TTP 的 R 值通过 Eq.(6) 根据目标 pruning ratio 1-R̄/M 反算。TwigVLM++ 使用 P-Head 专门计算 token 重要性 scores（替代 attention map），进一步解耦剪枝与预测任务。

涉及论文标题：
- Growing_a_Twig_to_Accelerate_Large_Vision-Language_Models
- Representation_Shift__Unifying_Token_Compression_with_FlashAttention
- SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs

**Representation Shift-based Token Pruning**（来自 Representation Shift 论文）：
该论文提出了一种完全不依赖 attention map 的 token 重要性度量——representation shift（表示漂移），公式为 s = ||MLP(LN(x')) - x'||₂，即 token 经过 MLP 层的 L2 表示变化量。关键发现：(1) MLP 操作逐 token 独立，产生的 representation shift 比 attention-based 方法更具判别性；(2) L2 距离在所有深度上优于 L1 和 cosine distance。该方法的根本优势在于**无需 attention map**，因此可与 FlashAttention 完全兼容——FlashAttention 避免构建完整 attention map 以减少 HBM I/O，传统 attention-based pruning 无法使用。结合 FlashAttention + representation shift pruning 实现乘法级加速：FlashAttention 自身约 2.7× speedup，pruning 再额外约 2× speedup，总计 5.5× (UMT-L video-text retrieval)。方法进一步扩展到 CNN（通过行/列级表示变化剪枝）和 SSM（替换激活值基重要性分数），验证了模型无关性。
