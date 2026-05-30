## Cross-Attention (CroAttn) LMM Architecture

术语是什么？

Cross-Attention LMM 是多模态大模型的第二种主流架构，在 LLM backbone 中插入专用 cross-attention layers 处理 image tokens。大部分层保持原有 self-attention（仅处理 text tokens），少数层（如 Llama3.2-11B 中 4/40 层）被替换为 cross-attention layers——text tokens 通过 cross-attention attend to image tokens，image tokens 不参与 self-attention。代表模型：Llama 3.2 Vision（4/40 CroAttn layers）、NVLM-X、Flamingo。

从算法pipeline角度拆解术语：

CroAttn LMM 推理（Llama3.2-11B）：
```
for l in 1..40:
  if l is self-attention layer (36 layers):
    Q,K,V from text tokens only
    A = softmax(Q_t@K_t^T/sqrt(d))
  if l is cross-attention layer (4 layers):
    Q_t from text tokens, K_i,V_i from image tokens
    A_cross = softmax(Q_t@K_i^T/sqrt(d))
    O = A_cross@V_i
  h = FFN(O)
```

关键特征：(1) prefill 复杂度大幅降低——self-attention 仅 O(N_t²)，cross-attention O(N_t·N_img)；(2) Image encoding 成为 TTFT 主要瓶颈——Llama3.2-11B 中 79% TTFT 来自 encoding（Insight 1）；(3) Image token 比例增加时 LLM prefill 时间反而减少（Insight 7），因此 CroAttn 对 image burst 更具弹性——autoscaling 仅需扩容 Image Instances。

术语一般如何实现？如何使用？

训练：先预训练 vision encoder 和 cross-attention adapter，再端到端 fine-tune。推理时 cross-attention K/V projection 需要额外权重和 KV cache 管理。ModServe 关键发现：CroAttn 模型 image burst 时 Text Instances 无需扩容——这是 41.3% cost saving（vs DecOnly 25%）的原因。

涉及论文标题：
- ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving
