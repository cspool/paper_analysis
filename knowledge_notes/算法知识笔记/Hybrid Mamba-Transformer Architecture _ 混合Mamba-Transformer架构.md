## Hybrid Mamba-Transformer Architecture / 混合Mamba-Transformer架构

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hybrid Mamba-Transformer 架构是一种混合 LLM backbone 设计，在同一个模型中将 Transformer Attention 层与 Mamba (Structured State Space Model, SSM) 层按特定比例交替排列。核心动机：Transformer attention 具有 O(N²) 的计算复杂度和 KV cache 内存消耗（N 为序列长度），长上下文场景下效率极低；Mamba SSM 具有 O(N) 的线性复杂度且无需 KV cache，但在 In-Context Learning (ICL) 和复杂检索/推理任务上能力弱于 Transformer attention。Hybrid 架构通过在层维度上混合两者，利用 Transformer attention 层保留 ICL 和上下文检索能力，利用 Mamba 层的线性复杂度降低整体计算开销，达到效率与效果的平衡。LongLLaVA 使用 4 组 hybrid stack，每组以 Attention:Mamba = 1:7 的比例排列（即每 8 层中 1 层 Transformer Attention + 7 层 Mamba SSM），配合 MoE 每隔一层集成 (16 experts, top-2 gating)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Hybrid LLM Backbone (Attention:Mamba = 1:7, 4 stacks)
for stack in range(4):
    for layer in range(stack_size):
        idx = stack * stack_size + layer
        if idx % 8 == 0:                # Attention layer (1/8)
            H = RMSNorm(H); Q,K,V = W_Q(H),W_K(H),W_V(H)
            H = H + FlashAttention(Q,K,V, causal=True)
        else:                            # Mamba layer (7/8)
            H = RMSNorm(H)
            H = H + MambaBlock(H)        # selective scan
        if layer % 2 == 0:               # MoE layer
            gate = softmax(router(H)); w,idx = topk(gate,k=2)
            H = H + sum(w[i]*expert[idx[i]](H) for i in [0,1])
        else:
            H = H + SwiGLU_FFN(H)
```

Annotations: 每 8 层 1 Attention + 7 Mamba = Quasi-Linear 复杂度；仅 attention layers 需要 KV cache (12.5%)；MoE 16 experts top-2 gating。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LongLLaVA Table 1 效率对比 (100K tokens): Hybrid Prefill 25.5s / TP 37.6 / Mem 79.1GB vs Transformer 34.0s / 14.7 / 79.4GB vs pure Mamba 14.3s / 72.6 / 32.1GB。VL-ICL 5-shot: Hybrid 61.3 vs Mamba 53.2 vs Transformer 58.9。1:7 ratio 在 1.3B 模型上验证为最优（Table 2: 1:7 vs 1:3 性能差距极小但 1:7 更高效）。参考：Jamba 使用类似 hybrid 设计，256K tokens 仅 4GB KV cache。

**Hybrid Mamba-Transformer 在长视频理解中的设计 (TimeViper)**：TimeViper 采用 Nanov2-9B backbone（27 Mamba-2 + 4 Self-Attention + 25 MLP），Mamba:Attention ratio ≈ 17:1（27/56 Mamba + 4/56 Attention + 25/56 MLP）。Self-attention 层仅 4 层（占 7.1%），集中在特定深度位置（如第 14 层为第一个 attention 层）。这使得 LLM 整体接近 quasi-linear 复杂度，GPU 内存和 prefilling 时间随帧数近似线性增长（而非二次）。具体效果：vanilla 模型在 128 frames 即 OOM，+ToMe 扩展至 ~5K frames，+ToMe+TransV 扩展至 10K+ frames。在 4096 frames 时，TransV 减少 54.8% 内存和 15.7% prefilling time。定性分析揭示 Mamba 层 attention 模式多样性（sparsity/locality/globality），self-attention 层展示 "attention sink" 现象（大量 attention 集中在前几个 token 上），且 vision token 的 attention 随深度递减，印证了 vision-to-text information aggregation 现象。

涉及论文标题：
- LongLLaVA__Scaling_Multi-modal_LLMs_to_1000_Images_Efficiently_via_Hybrid_Architecture
- TimeViper__A_Hybrid_Mamba-Transformer_Vision-Language_Model_for_Efficient_Long_Video_Understanding
- TimeViper__A_Hybrid_Mamba-Transformer_Vision-Language_Model_for_Efficient_Long_Video_Understanding
