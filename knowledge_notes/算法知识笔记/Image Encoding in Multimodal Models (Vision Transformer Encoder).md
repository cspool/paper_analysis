## Image Encoding in Multimodal Models (Vision Transformer Encoder)

术语是什么？

Image encoding 是 LMM 推理中将预处理图像 tiles 转换为 image token embeddings 的阶段。当前主流 LMM 使用 Vision Transformer (ViT) 作为 encoder，将每个 image tile 编码为固定数量的 visual feature tokens。不同 LMM 使用不同 encoder：ViT-H/14 (630M, Llama3.2)、SigLIP (400M, LLaVA-OV)、InternViT (6B, InternVL/NVLM-D)。

从算法pipeline角度拆解术语：

ViT encoding（处理 560×560 tile）：
```
conv2d (kernel=14, stride=14): [1,3,560,560] → [1,d,40,40]
flatten + [CLS]: [1, 1601, d]
for l in 1..L_enc:
  h = h + MHA(LayerNorm(h))
  h = h + FFN(LayerNorm(h))
→ image_tokens [1, 1601, d_enc]
```

ModServe Insight 2: Image encoding 是 compute-bound（SM activity ~100%, DRAM util <30%）。Encoder 最佳 TP 度通常为 TP-1——因 630M 模型分到 8 GPU 时 inter-GPU communication > compute savings，与 LLM backend（需 TP-4/8）形成鲜明对比。这一差异是 ModServe stage decoupling 的物理基础——允许 encoder 和 LLM 使用不同 TP 度。

术语一般如何实现？如何使用？

通过 HuggingFace Transformers 加载。不同 LMM tile 配置不同（Table 1）：Llama3.2 560×560 tiles → 4 tiles, LLaVA-OV 384×384 → up to 10 tiles, InternVL 448×448 → 5 tiles。Tokens/tile: ViT-H/14 1601, SigLIP 729, InternViT 256。ModServe 的 Image Instance 通过多个 TP-1 encoder 并行化同请求多 tiles 的 encoding（tiles 间无依赖，Insight 2）。

涉及论文标题：
- ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving
