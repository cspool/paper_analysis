## Mixture-of-Diffusion Models Serving（混合扩散模型服务）

术语是什么？

Mixture-of-Diffusion Models Serving是MoDM提出的扩散模型serving范式：在服务系统中同时部署多个不同尺寸的扩散模型（large model如SD3.5L 8B生成高质量图像但慢；small model如SANA 1.6B生成低质量图像但快），通过缓存最终生成图像实现跨模型复用：cache-hit请求用小模型refine以降低延迟，cache-miss请求用大模型全量推理以保证质量。核心洞察：扩散模型存在固有的latency-quality trade-off（大模型慢但质量高、小模型快但质量低），但通过final image缓存（而非model-specific latent缓存）可以在两模型间无缝传递生成内容。

从系统架构角度拆解术语：

MoDM的混合扩散模型serving运转流程：

```
1. 系统部署N个GPU Worker进程，每个Worker可加载一种模型变体（large或small），可动态切换
2. Request Scheduler接收prompt→提取CLIP text embedding→查询Image Cache
3. 若cache hit (text-image similarity >= threshold):
    → Router将请求发往Cache Hit Queue
    → 小模型Worker执行T-k步denoising refine（k由similarity决定）
4. 若cache miss (similarity < threshold):
    → Router将请求发往Cache Miss Queue
    → 大模型Worker执行全量T=50步denoising
5. Global Monitor每周期统计R(请求率)、H_cache(命中率)、k分布
   → Quality-Optimized Mode: max N_large s.t. throughput constraints
   → Throughput-Optimized Mode: 按workload比例分配N_large/N_small
6. PID Controller根据实时负载调整大/小模型GPU数量
```

关键设计要素：(1) 模型无关性——缓存final image而非latent使cache跨模型族(Stable Diffusion/SANA/FLUX)复用；(2) 动态模型切换——系统可根据负载动态将worker从SDXL切换到SANA，实验显示超过22 req/min时SDXL无法满足需求，系统自动切换SANA提升throughput；(3) 双层缓存策略——cache-large（仅缓存大模型生成图像保quality）vs cache-all（缓存全部图像提throughput），系统设计者可按需选择。

术语一般如何实现？如何使用？

MoDM从零实现Python serving系统，Request Scheduler、Global Monitor、各GPU Worker运行在独立进程中，通过PyTorch RPC通信。实现包含Request Scheduler中的CLIP embedding提取+cosine similarity检索、Global Monitor中的PID controller（Kp=0.6, Ki=0.05, Kd=0.05）、FIFO滑动窗口cache管理。DiffusionDB workload下MoDM-SANA达3.2x throughput vs Vanilla SD3.5L，46.7% energy savings。开源：https://github.com/stsxxx/MoDM。

涉及论文标题：
- MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models

---
