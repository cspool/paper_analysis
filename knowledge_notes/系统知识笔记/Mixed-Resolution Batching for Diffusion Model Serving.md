## Mixed-Resolution Batching for Diffusion Model Serving

术语是什么？通过联网搜索让回答具体和精准。

Mixed-Resolution Batching（混合分辨率批处理）是一种将不同分辨率（如512×512, 768×768, 1024×1024）的T2I扩散模型请求合并为单一batch并行处理的技术。传统扩散模型serving中，不同分辨率请求产生不同shape的latent tensors——在U-Net/DiT的各层中tensor shape mismatch阻止了PyTorch/CUDA的batch-level kernel fusion，导致请求只能sequential执行（3个不同分辨率SDXL请求在H100上sequential=17.8s vs batched=9.5s）。MixFusion通过将各分辨率图像沿height和width维partition为uniform size patches（patch size = 各resolution在该维度的最大公约数），使所有patches具有相同shape，从而可将heterogeneous请求在单batch内并发处理。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

MixFusion中mixed-resolution batching的workflow：

```
1. 请求到达：多个不同resolution请求（如512, 768, 1024）进入wait_queue
2. Patch Partition：
   - 计算height维度和width维度各resolution的最大公约数作为patch_size_h, patch_size_w
   - 各请求图像被均分为N_h × N_w个均匀patch（如1024×1024在patch_size=256时→16 patches）
   - 所有patches具有相同shape→可组成single batch
3. CSP格式管理：
   - request按resolution排序（相同resolution的patches连续存放）
   - RequestOffset[]: 每个请求的首patch在全局patch数组中的偏移
   - ResolutionOffset[]: 每个请求的Self-Attention reconstruction所需resolution偏移
   - RequestStart[]/RequestEnd[]: 每个patch在其所属请求中的起止范围
4. Batched Denoising：
   - pixel-wise operators (Linear, FeedForward, Cross Attention): 各patch独立→直接batch执行
   - Self-Attention: 按resolution分组reconstruct全图→batched attention per resolution group
   - Convolution: Patch Edge Stitcher处理跨patch边界
5. 输出：各patch完成denoising→按RequestOffset重组为各请求的完整latent→VAE decode
```

关键效益：batch size可随请求数线性扩展（而不受resolution diversity限制），GPU utilization显著提升。对比Distrifusion（patch数固定=GPU数，不支持mixed-resolution），MixFusion在8 GPU下throughput随batch size单调递增（3→12 batch, Figure 19），而Distrifusion在SD3上因同步开销throughput随batch size下降。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MixFusion用12.5K行Python + C++/CUDA实现，基于PyTorch + xformers加速attention。CSP格式的offset-based compression将patch mapping压缩为四个integer arrays，实现O(1) patch→request逆向查找。Patch size选择策略：取batch内各resolution在height/width维的最大公约数（如batch含512/768/1024→GCD(512,768,1024)=256→patch_size=256→各请求分别产生4/9/16个256×256 patches→total batch=29 patches）。该方法限制是patch数随resolution多样性增加可能使单batch的total patch数过大，超过GPU memory。论文evaluation使用max batch size=12（H100 80GB memory limit），patch size sensitivity study显示throughput随patch size增大而提升（splitting overhead减少），但过小patch导致performance下降（SDXL因Convolution overhead更显著）。

涉及论文标题：
- MixFusion: A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models
