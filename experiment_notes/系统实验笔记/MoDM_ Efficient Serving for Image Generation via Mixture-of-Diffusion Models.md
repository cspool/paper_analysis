## MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models

- 属于Serving调度的实现是什么？实验比较什么？
  提出MoDM，一个基于final image缓存+混合扩散模型(mixture-of-diffusion models)的自适应serving系统。核心实现包含：(1) Request Scheduler：管理请求流，用CLIP text embedding做text-to-image similarity检索缓存图像，根据相似度动态选择跳过步数k∈{5,10,15,20,25,30}，cache-hit请求发往小模型refine、cache-miss请求发往大模型全量推理；(2) Global Monitor：基于PID控制器(Kp=0.6, Ki=0.05, Kd=0.05)动态分配GPU资源到大/小模型worker，支持Quality-Optimized Mode（最大化大模型数量受SLO约束）和Throughput-Optimized Mode（所有cache-hit用小模型、cache-miss用大模型）；(3) FIFO-based Cache Maintenance：滑动窗口缓存策略，>90% cache-hit请求检索到4小时内生成图像；(4) Cross-model compatibility：缓存final image而非latent，使同一cache跨Stable Diffusion/SANA等多模型族复用。实验比较throughput、tail latency (P99)、SLO compliance (2×/4× large model latency)、最大负载、energy consumption，对比Vanilla (仅大模型)、Nirvana (latent caching)、Pinecone (检索无refine)。quality指标用CLIPScore、FID、IS、PickScore。

- 硬件平台是什么，配置是什么。
  两套配置：(1) 单节点4×NVIDIA A40 GPU (48GB memory)；(2) 16节点集群，每节点4×AMD MI210 GPU (64GB memory)。软件环境：Python + PyTorch，PyTorch RPC做节点间通信。

- 开源Serving框架是什么。修改了什么。
  MoDM从零实现Python serving系统（非基于已有开源Serving框架修改），Request Scheduler、Global Monitor和每个worker运行在独立进程中，通过PyTorch RPC通信。核心新增：(1) CLIP-based text-to-image相似度计算与检索模块；(2) 基于相似度的k-selection heuristic (Fig.5b)；(3) PID controller resource allocation算法 (Algorithm 1)；(4) FIFO cache管理策略；(5) 图像noise re-introduction + small model refinement pipeline (公式2)。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源地址：https://github.com/stsxxx/MoDM。以SD3.5L (8B)为large model、SDXL (3B)为small model的serving为例：
  1. 部署：启动Request Scheduler进程（管理CLIP embedding提取和cache lookup）+ Global Monitor进程（PID controller资源分配）+ N个GPU Worker进程（各加载一个模型变体，可动态切换大/小模型）
  2. 请求到达：Scheduler提取prompt的CLIP text embedding→与cache中图像embedding计算cosine similarity→若sim≥threshold（如0.25-0.30），确定跳过步数k（例如sim≥0.3则k=30）→Retrieve cached image→加噪到timestep t_k（公式2: Ĩ=σ_tk·ε+(1-σ_tk)·I*）→发到cache-hit queue用小模型refine T-k步
  3. 若sim<threshold→cache miss→发到大模型全量T=50步推理
  4. Global Monitor每周期统计request rate R、cache hit rate H_cache、k分布→计算miss_workload和hit_workload→PID调整N_large和N_small分配
  5. 新生成图像存入FIFO cache替换最旧图像
  6. 在DiffusionDB workload下MoDM-SANA达到3.2× throughput提升，46.7% energy savings，P99 tail latency在4×A40上支持10 req/min不violate SLO

