## Multimodal Connector / Modality-Alignment Module

术语是什么？

Multimodal Connector（又称 projector / modality-alignment module）是将 image encoder 输出映射到 LLM token embedding space 的轻量级模块。因 encoder 和 LLM 独立预训练，embedding spaces 维度和语义对齐方式不同，connector 桥接这一 gap。最常见形式为 2-layer MLP（LLaVA 系列）。参数量极小——<0.1% 总参数，<0.4% TTFT（ModServe 论文）。

从算法pipeline角度拆解术语：

```
encoder_output: [N_tokens, d_enc]
  → W_1 @ x + b_1 → GELU → W_2 @ h + b_2
  → [N_tokens, d_llm]
```

维度示例: ViT-H/14 d_enc=1280, Llama 3.1 8B d_llm=4096。计算量远小于 LLM prefill。

ModServe 部署选择：connector 共置于 Text Instance（与 LLM 共享 GPU），Image Instance 仅输出原始 encoder output，connector forward 在 RDMA 传输后在 Text Instance 侧执行——避免为极轻量模块分配独立 GPU。

术语一般如何实现？如何使用？

实现方式因模型而异：LLaVA 使用 2-layer MLP+GELU，InternVL 使用 pixel shuffle+MLP，BLIP-2 使用 Q-Former（learnable query-based）。ModServe 兼容各种 connector。

涉及论文标题：
- ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving

---
