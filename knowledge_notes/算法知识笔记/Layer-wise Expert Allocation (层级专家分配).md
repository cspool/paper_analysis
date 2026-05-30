## Layer-wise Expert Allocation (层级专家分配)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Layer-wise Expert Allocation（层级专家分配）是 MoLA 论文提出的核心创新：在结合 LoRA 和 MoE 进行参数高效微调时，不为 Transformer 每一层分配相同数量的 LoRA expert，而是根据各层的表示特性和冗余程度，灵活分配不同数量的 expert。其理论基础是：Transformer 底层处理 token-level 特征（词义、语法），expert 间高度相似（冗余大），不需要太多 expert；中层/高层处理抽象推理和任务特定模式，需要更多 expert 学习细粒度特征。对于有 m 层的 Transformer，每层 j 分配 N_j 个 expert，总 expert 数 ΣN_j 固定（与 baseline 等量分配相同参数量），仅分配方式不同。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子：
MoLA 提出五种基于不同假设的层级分配配置（以 LLaMA-2-7B 32 层为例，总 config sum=20）：

| 配置名称 | 分配 (层1-8, 9-16, 17-24, 25-32) | 假设 |
|------|------|------|
| MoLA-△ (Triangle) | 8, 6, 4, 2 | 底层需更多 expert 处理细粒度 token 特征 |
| MoLA-▽ (Inverted Triangle) | 2, 4, 6, 8 | 高层需更多 expert 处理抽象推理和任务特定模式 |
| MoLA-▷◁ (Hourglass) | 8, 2, 2, 8 | 底层和高层需更多 expert，中层做特征聚合/映射 |
| MoLA-✸ (Diamond) | 2, 8, 8, 2 | 中层表示学习最关键（effective representation learning） |
| MoLA-□ (Rectangle) | 5, 5, 5, 5 | 各层 expert 数相同，传统 MoE baseline |

伪代码（每层使用不同 N_j）：
```
# 配置: expert_config = [2]*8 + [4]*8 + [6]*8 + [8]*8  # MoLA-▽ 2468
for layer_j in 1..m:
    N_j = expert_config[layer_j]
    for module in [Wq, Wk, Wv, Wo, Wgate, Wdown, Wup]:
        W_r = Linear(d_model, N_j)               # 该层 router
        probs = Softmax(W_r @ x)                  # [B, L, N_j]
        topk_vals, topk_idx = TopK(probs, K=2)
        topk_vals /= sum(topk_vals)
        h = frozen_W0 @ x
        for idx, w in zip(topk_idx, topk_vals):
            h += w * (B[idx] @ A[idx] @ x)       # LoRA delta
        f_e = mean(indicator(token to expert e))
        P_e = mean(probs[:, e])
        L_aux += N_j * sum(f_e * P_e)            # per-layer LB loss
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 开源：https://github.com/GCYZSL/MoLA，基于 HuggingFace PEFT + Transformers。通过 expert_config 列表指定每层 expert 数量。
- 关键超参数：LoRA rank=8, top-K=2, LoRA alpha=16, LoRA dropout=0.05。总可训练参数 ~105.6M（LLaMA-2-7B 的 ~1.5%）。
- 最优配置因 base model 而异：LLaMA-2 和 Gemma 倾向 MoLA-▽（高层多 expert），Mistral 倾向 MoLA-✸（中层多 expert）。与各模型预训练层级质量（HT-SR PL Alpha Hill metric）高度相关（Pearson r=0.91 LLaMA-2, r=0.74 Mistral）。
- 核心发现：在固定总 expert 预算下，减少底层 expert（冗余高）、增加中高层 expert（冗余低），可提升性能且不增加参数。MoLA-▽ (2468) 以 62.5% 的参数量超越等量 MoLA-□ (8888)。

涉及论文标题：
- MoLA: MoE LoRA with Layer-wise Expert Allocation

---
