## Operator Taxonomy for Patch-Level Diffusion Inference

术语是什么？通过联网搜索让回答具体和精准。

Operator Taxonomy for Patch-Level Diffusion Inference（patch级扩散推理的算子分类）是MixFusion提出的将扩散模型中的算子按是否需要跨patch上下文(context)进行分类的体系，用于指导patch-level parallel serving的设计。扩散模型的denoising过程中，大多数算子操作在"pixel level"（局部）而非"image level"（全局）信息上，这些算子可被decompose为独立子算子在各patch上并行执行。算子分为两类：(a) Pixel-wise operators——Linear、FeedForward、Cross Attention等，仅依赖当前像素/patch内部信息，可直接在各patch上独立并行执行；(b) Context-dependent operators——Self-Attention和Convolution，需要跨patch上下文信息来保证输出一致性（Self-Attention需要all patches交互形成Cartesian product；Convolution需要相邻patch的边界像素）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Patch-level operator execution pipeline：

```
对每个denoising step的每个block：
1. // Pixel-wise operators: 直接对各patch独立批处理
2. for op in [Linear, FeedForward, CrossAttention]:
3.     // 所有N个patches形状相同→直接组成batch[N, ...]执行
4.     output = op(patches_batch)  // standard batched execution
5.
6. // Self-Attention: 按分辨率分组reconstruct为全图后批处理
7. for resolution_group in unique_resolutions:
8.     // 例如：resolution=1024的patches重组为完整feature map
9.     full_feature = reconstruct_from_patches(patches_of_this_resolution)
10.    attention_output = SelfAttention(full_feature)  // batched attention per resolution group
11.
12. // Convolution (仅U-Net, DiT无此操作): 需要跨patch边界
13. for each patch:
14.    // GroupNorm + boundary stitching fused in single kernel
15.    output = FusedGroupNormWithStitching(patch, neighbor_boundaries)
16.    // Padding with 0 when neighbor absent（如image边缘patch）
```

U-Net（SDXL）vs DiT（SD3）在operator taxonomy下的差异：
- U-Net含ResNet blocks（含Convolution）和Transformer blocks（含Self/Cross Attention）
- DiT仅含Transformer blocks（Self-Attention + Cross Attention + FFN）
- SD3无Convolution→patched inference自然100% accuracy
- SDXL Convolution kernel size 1-3（kernel>1时出现跨patch依赖）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MixFusion在PyTorch diffusion pipeline中实现此分类：(1) pixel-wise operators通过标准PyTorch batch execution处理——因为CSP格式确保所有patches相同shape，无额外修改；(2) Self-Attention利用CSP的ResolutionOffset按resolution分组——每个resolution group内patch重组为full feature map后通过xformers执行batched attention；(3) Convolution通过Patch Edge Stitcher（fused GroupNorm + boundary stitching CUDA kernel）处理跨patch边界。此operator taxonomy的设计使MixFusion能在保留generation quality的前提下实现patch-level parallelism——仅在必要时引入cross-patch context（Convolution的PES, Self-Attention的per-resolution reconstruction），其他计算完全并行。

涉及论文标题：
- MixFusion: A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models

术语是什么？

Text-to-Image Similarity-Based Cache Retrieval是MoDM提出的扩散模型缓存检索方法：用CLIP text encoder提取query prompt的text embedding，与cached image的CLIP image embedding做cosine similarity匹配，选取最高相似度的图像返回。与Nirvana的text-to-text similarity retrieval（比较prompt text embedding与cached prompt text embedding）相比，text-to-image retrieval直接度量图文之间的语义-视觉对齐，更好地匹配用户意图中的style、structure和content。

从算法pipeline角度拆解术语：

Text-to-Image Cache Retrieval算法流程：

```
// 离线: 为cache中每张图像预计算CLIP image embedding
for each cached image I_j:
    e_{I_j} = CLIP_ImageEncoder(I_j)           // 512维embedding

// 在线检索:
Function RetrieveCacheImage(query_prompt, cache, threshold_tau):
    q = CLIP_TextEncoder(query_prompt)          // 512维text embedding
    best_sim = -inf, best_image = None
    
    for each (I_j, e_{I_j}) in cache:          // GPU并行矩阵乘法
        sim = cosine_similarity(q, e_{I_j})     // q·e_{I_j} / (||q|| x ||e_{I_j}||)
        if sim > best_sim:
            best_sim = sim; best_image = I_j
    
    if best_sim >= threshold_tau:
        return (best_image, best_sim)           // cache hit
    else:
        return None                             // cache miss
```

检索性能：100K图像embedding存储仅0.29GB；GPU上cosine similarity计算0.05s/100K张；retrieval latency远小于denoising（>10s），不构成瓶颈。相似度阈值τ（0.25-0.30）虽低于Nirvana的text-to-text threshold（0.65-0.95），但因CLIP score本身捕捉图文semantic alignment，更低阈值仍能保证更好的视觉匹配（Fig.2验证：text-to-image mean CLIPScore 0.28 vs text-to-text 0.22）。

术语一般如何实现？如何使用？

MoDM使用OpenAI CLIP ViT-L/14模型提取embedding。image encoder部署在Request Scheduler侧，每张新生成图像异步提取embedding。检索过程在GPU上实现为单次矩阵乘法（Q·K^T），利用了GPU的并行计算能力。MoDM证明text-to-image retrieval在CLIPScore和PickScore上均优于text-to-text retrieval，且避免使用CLIP做bias（同时用PickScore交叉验证）。

涉及论文标题：
- MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models

---

