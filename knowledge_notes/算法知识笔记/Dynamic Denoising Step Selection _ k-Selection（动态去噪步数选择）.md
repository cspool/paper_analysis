## Dynamic Denoising Step Selection / k-Selection（动态去噪步数选择）

术语是什么？

Dynamic Denoising Step Selection (k-Selection) 是MoDM提出的自适应去噪步数跳过策略：基于query prompt与retrieved cached image之间的text-image similarity score，动态决定skip多少步denoising。相似度越高表示检索图像与目标越接近→skip更多步（k更大）；相似度越低→skip更少步（k更小），留更多步做refine。k从离散集合K={5,10,15,20,25,30}中选择，总去噪步数T=50，k cap在30以防止生成图像与缓存图像过度相似。

从算法pipeline角度拆解术语：

k-Selection算法（offline calibration + online inference）：

```
// === Offline Calibration (Fig.5a) ===
1. 用大模型生成10000张图像+prompts作为候选cache (100K cached images)
2. 对每个k in {5,10,15,20,25,30}:
    for each (query_prompt, query_image) in 10000 samples:
        cached_img = RetrieveBestMatch(query_prompt, cache)  // text-to-image retrieval
        sim = CLIPScore(query_prompt, cached_img)
        refined_img = AddNoise(cached_img, t_k) + Denoise(small_model, T-k steps)
        quality = CLIPScore(query_prompt, refined_img)
        记录(sim, k, quality) tuple
3. 对每个k: 找到满足quality >= alpha * Q_full_gen 的最小sim (alpha=0.95)

// === Online k-Decision Heuristic (Fig.5b) ===
Function k_decision(similarity):
    if similarity >= 0.30: return 30
    if similarity >= 0.29: return 25
    if similarity >= 0.28: return 15
    if similarity >= 0.27: return 10
    if similarity >= 0.25: return 5
    return None  // cache miss, similarity < tau_min

// === Quality Constraint (Eq.5) ===
Q_cache-hit(k) >= alpha * Q_full-gen  // alpha=0.95
// 确保cache-hit生成质量 >= 全量大模型质量的95%
```

heuristic在1000个独立test prompts上的表现：平均CLIPScore 28.50 vs full large-model pipeline 28.59（达到99.7% baseline quality），超过95% quality retention目标。

术语一般如何实现？如何使用？

MoDM在Request Scheduler中实现k-Selection heuristic。Offline calibration用DiffusionDB数据集+大模型生成图像执行一次（cost较高但仅需一次），建立(similarity threshold, k) lookup table。Online inference时直接查表O(1)决定k值。alpha=0.95作为quality degradation factor可由系统部署者调节以在quality和throughput之间trade off。cache hit threshold tau也随k变化（更大的k要求更高similarity）。

涉及论文标题：
- MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models

---

