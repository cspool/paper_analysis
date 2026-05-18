## Patch-Level Cache Reuse for Diffusion Serving

术语是什么？通过联网搜索让回答具体和精准。

Patch-Level Cache Reuse（patch级缓存复用）是一种在扩散模型逐step denoising过程中，以patch（而非全图）为粒度进行block输出的缓存和选择性重算的技术。核心观察是扩散模型denoising过程中相邻steps的block输出高度相似——但不同分辨率的图像在不同blocks上的相似性分布不同（Figure 7, 512/768/1024分辨率下skipped blocks集合显著不同），因此需要per-patch的动态复用决策。MixFusion在每个block每step前用Random Forest Cache Predictor（GPU端cuML）比较当前patch输入与上step缓存的输出，按MSE similarity threshold（默认0.1）生成reuse mask，标记哪些patch可跳过重算，仅重算unmasked patches。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

MixFusion中patch-level cache reuse workflow：

```
每个denoising step的每个block前：
1. Predict（Cache Reuse Predictor）：
   - Random Forest Classifier（GPU端cuML）输入：当前patch_input与cached_output的MSE
   - 输出：reuse_mask[N_patches]（0=重算, 1=复用）
2. Compute（Block Forward）：
   - pixel-wise operators: 仅recompute unmasked patches（masked patches直接用cached output）
   - context-dependent operators（如Self-Attention, Convolution）：
        masked patches用上step的完整block output填充（因相邻step output高度相似）
   - 产生partial output（masked部分不精确）
3. Combine（Cache Update）：
   - 用reuse_mask将masked区域替换为cache中上step输出→固定完整output
4. Update：
   - 更新cache中的input和output为当前step值，供下step使用

Cache系统Batching：
   输入：patch unique IDs + intermediate results
   → Common Set: IDs既在cache又在input中→比较后决定update
   → New Set: IDs仅在input中→batch insert
   → Expired Set: IDs仅在cache中→batch delete（对应已退出GPU的patch）
```

关键约束：每个block的cache操作必须在<2ms内完成（SD3每step 40-50ms含24 blocks→per block cache overhead <2ms，否则即使所有blocks都可skip也无法获得净收益）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MixFusion中cache predictor使用cuML Random Forest Classifier，训练数据为1K inference requests的input-output MSE similarity across all blocks and timesteps。Cache以map结构组织（patch unique ID → cached data），每个block维护独立cache。Patch-level caching比whole-image caching consistently more effective（因whole-image caching要求batch内所有patches都meet threshold才能skip block→成功率更低，Figure 20）。

涉及论文标题：
- MixFusion: A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models
