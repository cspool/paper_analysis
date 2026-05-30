## Decoder-Only (DecOnly) LMM Architecture

术语是什么？

Decoder-Only LMM 是多模态大模型的一种主流架构，核心特征是复用未经修改的纯文本 decoder-only LLM（如 Qwen2, LLaMA）作为 backbone，将 image encoder 输出的 image tokens 与 text tokens 在 LLM 的 self-attention 中统一处理。Image tokens 通过 connector/MLP 映射到 LLM token embedding space 后，与 text tokens 拼接为一个统一序列输入 LLM。所有 transformer layer 中的 self-attention 对 text 和 image tokens 进行同等计算。代表模型包括 LLaVA-OneVision（Qwen2 backbone + SigLIP encoder）、InternVL（InternLM backbone + InternViT encoder）、NVLM-D（Qwen2-Instruct backbone + InternViT encoder）、DeepSeek Janus。

从算法pipeline角度拆解术语：

DecOnly LMM 推理 token 级计算过程：
```
输入: text_prompt (N_t tokens) + image

Step 1 - Image Preprocessing (CPU):
  raw image → resize/rescale/pad/normalize → tile segmentation

Step 2 - Image Encoding (GPU, ViT):
  tiles → ViT forward → image_tokens [N_img, d_enc]

Step 3 - Connector Projection:
  image_tokens → MLP → [N_img, d_llm]

Step 4 - LLM Prefill (all layers self-attention on all tokens):
  input_seq = [image_tokens | text_tokens]  // N = N_img + N_t
  for l in 1..L:
    Q, K, V = W_Q×h, W_K×h, W_V×h  // text+image 同等处理
    A = softmax(Q@K^T / sqrt(d))    // N×N full attention
    h = FFN(h)
```

关键特征：prefill 中 image tokens 参与每一层 self-attention，计算复杂度 O((N_img+N_t)²·L)。高分辨率图像产生大量 tokens（LLaVA-OV 每张 896×896 图像产生 7290 tokens），严重增加 prefill 延迟。ModServe Insight 3：DecOnly 模型 LLM prefill 延迟可达同规模 CroAttn 的 10×。

术语一般如何实现？如何使用？

两阶段训练：(1) pre-training——冻结 encoder 和 LLM，仅训练 connector；(2) instruction fine-tuning——解冻 LLM+connector。推理时 connector 极轻量（<0.1% 参数，<0.4% TTFT）。DecOnly 模型在 image-heavy workload 下因 prefill 高延迟而严重资源争用——ModServe 发现 stage decoupling 对 DecOnly 收益更大（InternVL 5.5× vs Llama3.2 3.3× throughput improvement）。

涉及论文标题：
- ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving
