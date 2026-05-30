## MoE in Multimodal LLMs

术语是什么？
MoE in Multimodal LLMs是将Mixture-of-Experts应用于MLLM的LLM backbone FFN层的设计范式。MLLM由Visual Encoder（提取visual token）→ Projector（对齐到text embedding space）→ LLM Backbone（transformer with MoE FFNs）组成。典型MLLM MoE配置：Kimi-VL-A3B（64 experts/layer, k=6）、Qwen3-VL-MoE-30B-A3B（128 experts/layer, k=8）、InternVL-3.5-30B-A3B（128 experts/layer, k=8）。MLLM中MoE的独特挑战：(1) vision token数量大→MoE计算开销显著；(2) 不同模态token在FFN中行为不同（modality gap）；(3) 浅层expert贡献远大于深层。

从算法pipeline角度拆解术语：
```
MLLM Forward:
    V = VisualEncoder(image/video)             // [N_v, d_v]
    V' = Projector(V)                           // [N_v, d_model]
    X = concat([V', T])                         // vision + text tokens
    for each transformer layer l:
        X = Attention(LN(X)) + X
        π = softmax(Router(LN(X)))              // [M] routing probs
        y = Σ_{m∈topk(π,k)} π_m · Expert_m(LN(X))
        X = y + X
```

术语一般如何实现？如何使用？
开源MLLMs：Kimi-VL, Qwen3-VL-MoE, InternVL-3.5。router为`nn.Linear(d_model, M)` + Softmax。Expert为独立FFN。评估使用lmm-eval框架 + multimodal benchmarks。MoDES通过GMLG+DMT+Frontier Search实现MLLM特定的expert skipping。

涉及论文标题：
- MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

---
