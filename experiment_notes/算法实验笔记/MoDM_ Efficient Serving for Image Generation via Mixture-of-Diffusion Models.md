## MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models

- 属于算法pipeline的实现是什么？实验比较什么？
  提出基于final image缓存+text-to-image CLIP similarity retrieval的动态去噪步数选择算法。核心设计：(1) Image Caching替代Latent Caching：缓存final generated image（PNG/JPEG压缩后1.4MB/张，对比Nirvana latent caching 2.5MB/张），存储image CLIP embedding仅0.29GB/100K images，cache检索耗时0.05s/100K images；(2) Text-to-Image similarity retrieval：用CLIP image encoder提取cached image embedding，与query text embedding做cosine similarity检索，比text-to-text similarity在CLIPScore (mean 0.28 vs 0.22)和PickScore (mean 20.33 vs 19.52)上均更优；(3) 动态k-selection heuristic：基于text-image similarity score和quality constraint (α≥0.95, 公式5)决定跳过去噪步数k∈{5,10,15,20,25,30}，固定T=50步，cache-hit仅需refine T-k步；(4) Noise re-introduction：对检索图像按扩散模型noise schedule加噪（公式2: Ĩ=σ_tk·ε+(1-σ_tk)·I*），使图像重新进入去噪流程；(5) Quality-constrained retrieval policy：仅当Q_cache-hit(k)≥α·Q_full-gen(α=0.95)时接受cache hit，heuristic在1000 prompt测试集上平均CLIPScore 28.50 vs full pipeline 28.59 (99.7% baseline quality)。实验比较throughput/quality trade-off，对比Vanilla (全量T=50步大模型)、Nirvana (latent caching text-to-text retrieval)、Pinecone (检索无refine)。

- 硬件平台是什么，配置是什么。
  4×NVIDIA A40 GPU (48GB) 单节点 + 16节点各4×AMD MI210 GPU (64GB)。PyTorch + PyTorch RPC。

- 模型是什么。数据集和bench分别是什么。
  Large models: Stable Diffusion-3.5-Large (SD3.5L, 8B params), FLUX.1-dev (12B params)。Small models: Stable Diffusion XL (SDXL, 3B params), SANA (1.6B params)。Distilled baseline: SD3.5L-Turbo (10步)。数据集：DiffusionDB (2M images production dataset)、MJHQ-30k (MidJourney images)。模型均使用T=50 denoising steps生成1024×1024图像，除SD3.5L-Turbo使用10步。SDXL用FP16，其余用BF16。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源：https://github.com/stsxxx/MoDM。算法pipeline：
  1. Cache初始化：用大模型生成N张图像（如10000张），每张图像用CLIP image encoder提取embedding (512-dim)，存入cache。存储final image (1.4MB/张 PNG) + embedding (0.29GB/100K embeddings)。
  2. 推理时：新prompt→CLIP text encoder提取query embedding q→对cache中所有image embeddings e_I计算cosine sim (公式1)→取最高sim的cached image I*
  3. 若sim≥threshold: 按heuristic (Fig.5b)确定k（sim≥0.3→k=30; sim≥0.29→k=25; sim≥0.28→k=15; sim≥0.27→k=10; sim≥0.25→k=5）
  4. 用扩散模型noise schedule公式2对I*加噪到timestep t_k→生成Ĩ=σ_tk·ε+(1-σ_tk)·I*
  5. 将Ĩ送入小模型(SDXL/SANA)执行剩余T-k步去噪→输出final image
  6. 若sim<threshold: 发往大模型(SD3.5L/FLUX)执行全量T=50步推理
  7. 总compute savings公式4：C_total_saved = H_cache·ΣP(K=k)·[k/T·C_gen + (T-k)/T·(C_gen-C_small)]，同时跳过k步+用小模型进一步降低每步成本
