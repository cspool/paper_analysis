## Any-to-Any Multimodal Models

术语是什么？

Any-to-Any Multimodal Models 是能够接受和生成多种模态（text、image、video、audio）的统一模型架构。与传统的 text-to-text LLM 或 multimodal-input text-output 模型不同，any-to-any 模型支持交叉模态的端到端训练和理解-生成的统一。代表性模型包括：Qwen-Omni 系列（text+image+video+audio 输入，text+audio 输出）、GLM-Image（text+image 输入，text+image 输出）、BAGEL（Mixture-of-Transformers 设计，分离 multimodal understanding 和 visual generation experts）、LongCat-Flash-Omni（560B MoE LLM backbone + LSTM/CNN audio decoder）等。

从算法pipeline角度拆解术语：

Any-to-Any 模型的典型 pipeline 组成（以 Qwen3-Omni 为例）：
```
Input: Text + Audio + Image + Video
  │
  ├─ Text → tokenizer → text token embeddings
  ├─ Audio → Whisper audio encoder → audio embeddings
  ├─ Image → ViT/SigLIP vision encoder → image embeddings
  └─ Video → Vision encoder + temporal aggregation → video embeddings
  │
  ▼
Multimodal Embedding Concatenation
  (所有模态 embeddings concat 后输入 LLM backbone)
  │
  ▼
LLM Backbone (AR Decoder):
  ├─ Thinker: 自回归生成 text tokens + 输出 hidden states
  │   每 step: self-attention over (text + multimodal) tokens
  │   → 产生 text output tokens + per-step hidden states
  │
  └─ Modality-specific Generator:
      ├─ Audio: Talker LLM → codec tokens → Vocoder → waveform
      └─ Visual: DiT → iterative denoising → image/video pixel output
  │
  ▼
Modality-specific Decoder Output
```

关键算法特征：
1. **共享 embedding space**：所有模态通过专用 encoders 映射到统一 embedding space，LLM backbone 在此空间做 cross-modal reasoning
2. **AR Semantic + Modality-Specific Synthesis**：LLM backbone 负责高层语义理解/生成 → modality-specific decoders 负责低层信号合成
3. **Multi-AR pipeline**：Thinker-Talker 等双 AR 设计意味着 pipeline 中有多个需要 KV cache management 的 autoregressive stages
4. **AR+DiT hybrid**：BAGEL 等模型将 AR understanding 和 DiT generation 耦合成单个 inference pipeline

术语一般如何实现？如何使用？

现有 any-to-any 模型多通过 HuggingFace Transformers 实现，开发者手动编排 multi-stage pipeline——每个 stage 的 generate loop 独立实现、cross-stage transfer 手动进行、无 framework-level batching/scheduling 优化。vLLM-Omni 是首个原生支持 any-to-any model serving 的框架——通过 stage graph abstraction 将 multi-stage pipeline 分解为独立 stages，每个 stage 由 vLLM engine 或 diffusion engine 服务，stage 间通过 Unified Connector 传输数据。

涉及论文标题：
- vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models
