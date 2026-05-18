## Final Image Cache for Diffusion Models（扩散模型最终图像缓存）

术语是什么？

Final Image Cache是MoDM提出的扩散模型serving缓存策略：缓存diffusion model生成的最终图像（PNG/JPEG标准格式），而非中间latent representations。使缓存内容成为model-agnostic的通用表示，可被不同模型族（Stable Diffusion、SANA、FLUX）复用。与Nirvana的latent caching（缓存多个中间去噪状态，2.5MB/张，model-specific）相比，final image caching仅需1.4MB/张存储，且消除了模型依赖性。

从系统架构角度拆解术语：

Final Image Cache在MoDM中运转流程：

```
1. 生成: 大模型完成全量denoising -> 输出final image -> 保存PNG/JPEG格式
2. Embedding: CLIP image encoder(image) -> 512维embedding e_I -> 存入cache
3. 存储: {image_file, embedding, prompt, timestamp}
   存储成本: 100K images -> 1.4MBx100K + 0.29GB(embeddings)
4. 检索: 新prompt -> CLIP text encoder -> cosine sim(q, each e_I) -> 最高sim图像
5. 复用: 检索图像加噪 -> 小模型refine T-k步
6. 维护: FIFO滑动窗口，丢弃最旧图像（基于temporal locality: >90% cache-hit检索4h内图像）
```

与latent cache对比：Storage 1.4MB vs 2.5MB per image；Model Compatibility跨任何模型族 vs 仅同一模型；Retrieval用text-to-image CLIP similarity（语义对齐）vs text-to-text similarity（视觉失配）；Cache Fragmentation单一pool vs 每模型独立cache。

术语一般如何实现？如何使用？

MoDM在Request Scheduler进程中维护cache。每次推理完成后异步调用CLIP image encoder提取embedding并写入。FIFO策略简化管理，基于DiffusionDB trace有效（>90%查询在4h窗口内）。cache检索在GPU上通过矩阵乘法实现cosine similarity，100K图像检索仅0.05s。FIFO还天然保证content diversity，避免utility-based cache导致的某些图像过度重复使用引入生成偏差。

涉及论文标题：
- MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models

---
