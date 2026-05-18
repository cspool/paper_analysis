## Mask-aware Cached Activation / Y-Activation Cache for Diffusion Models（面向扩散模型的Mask感知激活缓存）

术语是什么？通过联网搜索让回答具体和精准。

Mask-aware Cached Activation是FlashPS提出的针对图像编辑扩散模型推理的激活复用技术。核心思想：在generative image editing serving中，用户请求通常包含mask指定待编辑的局部区域，其余unmasked区域应保持不变。FlashPS预先计算模板在无编辑条件下的transformer block输出activation Y（非K/V cache），在线推理时对unmasked token直接从cache读取Y并注入transformer block输出，仅对masked tokens执行完整attention和feed-forward计算。选择缓存输出activation Y而非K/V cache的原因是：(1) unmasked token的输出Y在相同模板的不同编辑请求间高度相似；(2) Y的缓存量约为K/V的一半，显著降低cache footprint；(3) masked/unmasked cross-attention较弱，仅关注masked tokens的计算不显著影响质量（SSIM可达0.99）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FlashPS mask-aware cached activation的算法pipeline（以SDXL/Flux DiT为例）：

```
输入: latent (spatial tokens), mask (binary, 1=masked/需编辑), template_id
输出: edited_latent

// 离线预计算阶段（每个template执行一次）
1: cached_activations = {}  // {block_id: unmasked_Y}
2: latent_full = VAE.encode(template_image)
3: for each transformer_block in DiT:
4:     Y_full = transformer_block(latent_full, timestep, cond)
5:     cached_activations[block_id] = Y_full[mask == 0]  // 仅保存unmasked tokens的Y
6: store_to_host_memory(template_id, cached_activations)

// 在线推理阶段（每个编辑请求）
7: latent_noisy = add_noise(VAE.encode(input_image), sigma_t)
8: for t in denoising_steps (T→1):
9:     for each transformer_block in DiT:
10:        // DP决定此block是否使用cache（见Bubble-free DP Block Selection）
11:        if use_cache[block_id]:
12:            cached_Y_unmasked = async_load(template_id, block_id)  // CUDA stream异步从host memory加载
13:            Y_masked = transformer_block_compute_masked_only(
14:                latent_noisy[mask==1],    // 仅masked tokens参与attention作为query
15:                latent_noisy,             // 全量tokens作为key/value context
16:                timestep, cond)
17:            Y_full = merge(Y_masked, cached_Y_unmasked, mask)  // masked位置用Y_masked，unmasked用cached
18:        else:
19:            Y_full = transformer_block_full(latent_noisy, timestep, cond)  // 全量计算
20:        latent_noisy = Y_full
21:    latent_noisy = scheduler_step(latent_noisy, predicted_noise, t)
22: edited_image = VAE.decode(latent_noisy)
```

计算复杂度：标准全量attention+FFN对N个tokens为O(N²d + Nd²)。FlashPS仅对masked tokens（≈mN个，m为mask ratio）做attention中的Q·K^T和weighted sum of V，以及对masked tokens做FFN。unmasked tokens的Y直接来自cache（O((1-m)N·d)的读取开销）。理论加速比约为1/m（mask ratio 0.11时理论≈9x）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
1. **Cache存储**：cached activations存储在host memory（CPU DRAM），也可使用disk或distributed storage作为二级存储。FlashPS生产trace中仅970个templates，每个模板缓存所有transformer block的Y（FP16），总cache大小可控（host memory容纳）。
2. **CUDA Stream异步加载**：创建独立的CUDA stream用于cache load，与computation stream并行。cache load stream通过cudaMemcpyAsync从host memory拷贝cached Y到GPU HBM，computation stream同步执行masked token的attention/FFN计算。
3. **Y-cache vs K/V-cache选择**：论文量化分析——mask ratio 20%时，缓存K/V可将latency从2.27s降至2.06s（比缓存Y额外降低约10%），但cache size翻倍（需存K和V两个tensor）。缓存Y是cache size与speedup的帕累托最优。
4. **适用条件**：收益依赖mask区域较小（生产trace平均mask ratio=0.11）、模板复用率高（970模板各平均复用约35k次）、编辑确实保持unmasked region不变。对style transfer等全局改变任务收益下降。

涉及论文标题：
- FlashPS: Efficient Generative Image Editing with Mask-aware Caching and Scheduling

