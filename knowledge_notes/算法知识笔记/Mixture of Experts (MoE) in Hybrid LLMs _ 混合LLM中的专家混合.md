## Mixture of Experts (MoE) in Hybrid LLMs / 混合LLM中的专家混合

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mixture of Experts (MoE) 是一种通过稀疏激活增加模型容量但不线性增加推理计算的架构设计。在 Transformer/Mamba LLM 中，将 FFN 层替换为 E 个独立专家子网络，每个 token 通过 router (gating network) 选择 top-k 专家激活。核心优势：(1) 总参数大但推理仅激活 top-k，计算量与激活参数成正比；(2) 专家在训练中自然分化出专业化。LongLLaVA 在 hybrid 架构中每隔一层使用 MoE FFN：E=16 experts, top-2 gating per token，总参 53B，推理激活 13B。Router 为 linear projection + softmax + top-k。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
class MoE_FFN(h):  # h: [B,L,d], E=16, k=2
    gate = softmax(Linear(d→E)(h), dim=-1)     # [B,L,16]
    w, idx = topk(gate, k=2); w = w/sum(w)     # renormalize
    out = zeros([B,L,d])
    for i in range(2):
        out += w[:,:,i,None] * expert[idx[:,:,i]](h)
    return out
# expert: SwiGLU_FFN (d → d_ff → d)
# Load balance: L_aux = E * Σ f_e * P_e
```

Annotations: E=16, k=2 → 12.5% active parameters；MoE 遵循 Jamba 的 layer-wise pattern (every other)；LongLLaVA-9B 仅保留 Expert-0 (差异极小)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LongLLaVA 的 expert selection 消融 (Appendix C)：不同 expert (0/5/12/15) 和聚合方式 (arithmetic/spherical) MMLU/BBH 差异极小 (<1%)，因此 LongLLaVA-9B 仅保留 Expert-0。Jamba 首次 hybrid Mamba-Transformer + MoE。Mixtral (Jiang et al., 2024a) 8 experts top-2。DeepSeek-V2 160 experts top-6 (细粒度 MoE)。关键训练挑战：load balancing 和 expert dropping。

涉及论文标题：
- LongLLaVA__Scaling_Multi-modal_LLMs_to_1000_Images_Efficiently_via_Hybrid_Architecture
