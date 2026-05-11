**ASPLOS26**

# **Large Language Models (LLMs)**

- LLM Inference
  - Prefill-Decode Multiplexing
    - Towards High-Goodput LLM Serving with Prefill-decode Multiplexing [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790236)] [[arXiv](https://arxiv.org/abs/2504.14489)]
      - SJTU & HKU & NUS
      - Propose **MuxWise**, an LLM serving framework built on intra-GPU prefill-decode multiplexing.
      - Integrate a bubble-less multiplex engine, a contention-tolerant estimator, and an SLO-aware dispatcher.
      - 拆分Prefill到layer，Prefill和Decode共享KVCache，Stream同步粒度调整SM分区，动态分配SM。
    - Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790135)] [[arXiv](https://arxiv.org/abs/2504.19516)] [[Code](https://github.com/zejia-lin/Bullet)]
      - SYSU
      - Enable concurrent execution of prefill and decode requests.
      - Dynamically provision GPU resources based on real-time performance modeling.
      - 优化基于token预算的Prefill chunk和Decode输入打包，PD动态分配不同SM执行，减少由于chunk导致KV Cache的reload和PD打包的lock-latency。
    - TPLA: Tensor Parallel Latent Attention for Efficient Disaggregated Prefill & Decode Inference [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790237)]
      - PKU & Tencent YouTu Lab
      - Introduce tensor-parallel latent attention for disaggregated prefill/decode inference.
      - Combine latent attention with tensor parallelism to improve PD-disaggregated long-context serving.
      - low-rank Attn（Global LA）的优化TPLA。
  - Scheduling
    - QoServe: Breaking the Silos of LLM Inference Serving [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790206)] [[arXiv](https://arxiv.org/abs/2503.22562)]
      - MSR India
      - Introduce fine-grained QoS classification so applications can specify precise latency requirements, and adapt scheduling decisions to real-time system state.
      - Leverage the predictable execution characteristics of LLM inference to implement dynamic chunking for higher throughput under strict QoS guarantees.
      - Combine hybrid prioritization with selective request relegation to balance fairness, efficiency, and graceful degradation under overload.
      - 在线scheduler优化，应用分类，不同应用的请求有不同QoS要求，请求-chunk level的动态调度。
    - Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790219)] [[arXiv](https://arxiv.org/abs/2509.16495)]
      - Snowflake
      - Introduce **Shift Parallelism**, a runtime that switches across inference parallelism strategies for dynamic workloads.
      - Turn parallelism selection into a runtime control decision to jointly improve latency and throughput.
      - 多卡GPU的多种并行方式支持，动态切换（SP、TP）的配置。
    - XY-Serve: End-to-End Versatile Production Serving for Dynamic LLM Workloads [[Paper](https://dl.acm.org/doi/10.1145/3760250.3762228)]
      - Huawei & THU & Shanghai AI Lab
      - Present **XY-Serve**, an end-to-end serving system for dynamic production LLM workloads.
      - Coordinate scheduling, batching, and runtime resource management to sustain serving efficiency under workload variation.
      - 对齐昇腾tile加速器的垂直优化。
    - BlendServe: Optimizing Offline Inference with Resource-Aware Batching [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790133)]
      - UC Berkeley & UW & UC Davis & Rice
      - Present a resource-aware batching framework for offline inference.
      - Form batches against actual compute and memory bottlenecks to improve throughput.
      - 离线重排请求，互补资源需求的请求一起并发。
  - MoE Inference
    - MoE-APEX: An Efficient MoE Inference System with Adaptive Precision Expert Offloading [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790187)]
      - SJTU & CUHK
      - Introduce an MoE inference system with adaptive-precision expert offloading.
      - Jointly tune expert offloading and precision to reduce memory pressure during serving.
      - MoE取回SSD的Expert时，根据gating output（直接影响Expert Output）决定激活Expert的重要性，加载不同精度的SSD Expert到GPU。
  - Compression
    - ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790250)] [[arXiv](https://arxiv.org/abs/2603.17435)] [[Code](https://github.com/xxyux/ZipServ)]
      - HKUST-GZ & HIT-SZ & HKUST
      - Introduce hardware-aware lossless compression for LLM inference.
      - Reduce memory footprint while preserving exact model behavior and improving serving efficiency.
      - 权重压缩的kernel优化。
  - Speculative Decoding
    - DFVG: A Heterogeneous Architecture for Speculative Decoding with Draft-on-FPGA and Verify-on-GPU [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790153)]
      - SJTU & Eastern Institute of Technology, Ningbo & Southeast University & Ningbo Institute of Digital Twin, Eastern Institute of Technology, Ningbo
      - Propose a heterogeneous speculative decoding architecture with FPGA draft generation and GPU verification.
      - Pipeline draft and verify across devices to reduce end-to-end decoding latency.
      - SPD的pipeline到GPU（verify）+FPGA（分支并行的draft）系统
    - SwiftSpec: Disaggregated Speculative Decoding and Fused Kernels for Low-Latency LLM Inference [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790246)]
      - ByteDance Seed & UChicago
      - Introduce disaggregated speculative decoding together with fused kernels for low-latency LLM inference.
      - Combine system-level disaggregation and kernel-level optimization to make speculative decoding practical in deployment.
      - SPD pipeline在多GPU的执行优化。
  - Sparsity
    - SpeContext: Enabling Efficient Long-context Reasoning with Speculative Context Sparsity in LLMs [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790224)]
      - SJTU & Infinigence-AI & SII & THU
      - Introduce speculative context sparsity for long-context reasoning in LLMs.
      - Avoid uniform full-context processing by speculating over sparse context usage during long-input inference.
      - GPU+CPU DRAM的KVCache卸载优化。
  - Attention Mechanisms
    - I/O Analysis is All You Need: An I/O Analysis for Long-Sequence Attention [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790174)]
      - IIT & ICT, CAS & UCAS
      - Present an I/O-centric analysis framework for long-sequence attention.
      - Show that data movement, rather than FLOPs alone, dominates long-context attention cost.
      - IO角度设计优化SA的dataflow。
    - PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790200)]
      - TJU & Stevens Institute of Technology
      - Introduce prefix-aware attention together with a multi-tile kernel for LLM decoding.
      - Reduce decode latency by exploiting shared prefixes while keeping GPU resource usage under control.
      - IO角度设计优化SA的dataflow。
  - Value Level Parallelism (VLP)
    - Mugi: Value Level Parallelism For Efficient LLMs [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790189)]
      - CMU & UCF
      - Introduce value-level parallelism as a new execution dimension for LLM inference.
      - Exploit finer-grained parallel structure than conventional tensor or sequence parallelism.
      - LUT-GEMM HW。
  - KV Cache Offloading
    - REPA: Reconfigurable PIM for the Joint Acceleration of KV Cache Offloading and Processing [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790212)]
      - SJTU
      - Present a reconfigurable PIM architecture for jointly offloading and processing KV cache.
      - Co-design KV movement and KV computation to reduce host-memory bottlenecks during inference.
      - LLM Inference，PIM执行基于KVCache的计算，GPU执行主要推理计算。
    - STARC: Selective Token Access with Remapping and Clustering for Efficient LLM Decoding on PIM Systems [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790226)]
      - RPI & UMass Amherst & IBM Research
      - Introduce selective token access with remapping and clustering for PIM-based LLM decoding.
      - Reduce unnecessary KV accesses and improve data locality during decoding.
      - cluster粒度计算相似度，选择要加载的KVCache tokens，PIM系统。
- Language Processing Units (LPUs)
  - Hardwired-Neuron Language Processing Units as General-Purpose Cognitive Substrates [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790169)]
    - ICT, CAS & USTC & IS, CAS & Cambricon Technologies
    - Propose Language Processing Units (LPUs) as a language-centric hardware substrate for general-purpose cognitive workloads.
    - Specialize the architecture around language processing primitives to improve efficiency on language-centric tasks.
    - cluster粒度计算相似度，选择要加载的KVCache tokens，PIM系统。

# **Generative Recommenders (GRs)**

- GR Serving
  - BAT: Efficient Generative Recommender Serving with Bipartite Attention [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790131)]

    - ZJU & HKU & Alibaba & NUS & Aalto University
    - Introduce bipartite attention for generative recommender serving.
    - Tailor the serving design to recommendation-style generative workloads rather than generic LLM inference.
    - 核心瓶颈是 compute-bound GR prefill 与低 KV cache 复用率叠加，而不是单个 Transformer 算子本身不够快。
    - Bipartite Attention，利用推荐 prompt 中 user 与 item 语义近似 permutation-invariant 的观察，让系统可在 User-as-prefix 与 Item-as-prefix 两种注意力组织之间切换；通过 attention mask 和 position encoding 调整，使 item KV cache 不依赖具体用户上下文，从而可跨用户复用。

# **Diffusion Models**

- Video DiT Training
  - DSV: Exploiting Dynamic Sparsity to Accelerate Large-Scale Video DiT Training [[Paper](https://dl.acm.org/doi/10.1145/3760250.3762216)] [[arXiv](https://arxiv.org/abs/2502.07590)]
    - CUHK & StepFun
    - Exploit dynamic sparsity to accelerate large-scale video DiT training.
    - Use hybrid sparsity-aware context parallelism to rebalance workloads under heterogeneous attention sparsity.
- Diffusion Model Serving
  - TetriServe: Efficiently Serving Mixed DiT Workloads [[Paper](https://dl.acm.org/doi/10.1145/3779212.3790233)] [[arXiv](https://arxiv.org/abs/2602.05116)] [[Code](https://github.com/DiT-Serving/TetriServe)]
    - UMich & UW-Madison & NTU
    - Present a serving system for mixed DiT workloads.
    - Coordinate scheduling and batching across heterogeneous diffusion requests in a shared runtime.
    - 请求粒度的调度单位，改为step/round粒度的采样步/时间段调度，更细粒度的抢占点，调度设计。
- Mixture-of-Diffusion Models
  - MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models [[Paper](https://dl.acm.org/doi/10.1145/3760250.3762220)] [[Code](https://github.com/stsxxx/MoDM)]
    - UMich & Intel Labs
    - Introduce mixture-of-diffusion models for image generation serving.
    - Use specialization across diffusion sub-models to improve efficiency and quality-cost tradeoffs.
    - 大小Model混合生成图片，text2text相似性（Cache hit）跳过去噪步骤时使用的中间数据，使用缓存的最终图像加噪（逆向生成到当前step）而不是中间latent，后续使用小Model继续去噪。

PPoPP 2026

# **LLM**

- LLM inference
  - JanusQuant: Accurate and Efficient 2-bit KV Cache Quantization for Long-context Inference
    - WHU
    - 2bit的baseline 的瓶颈已经不是 2-bit 存储本身，而是 memory-bound 的 outlier 管理、缓存搬移和量化/反量化算子分裂（运行时开销）。
  - Laser: Unlocking Layer-Level Scheduling for Efficient Multi-SLO LLM Serving
    - SYSU
    - request的iter执行切分成layer，每个iter执行若干layer，提高动态调度灵活性。
  - High-Throughput Non-Uniformly Quantized 3-bit LLM Inference
    - CUHK & HKUST
    - 3bit在GPU上地址访问对齐Cache Line而低效。
  - Accelerating Sparse Transformer Inference on GPU
    - CUP-Beijing & BUAA
    - MHA经过mask稀疏的激活执行后续算子时低效，但为多种mask模式、GPU架构、并行参数手写kernel不现实。
- Attention
  - FlashAttention-T: Towards Fully Tensorized Attention by Exploiting Tensor-Vector Parallelism
    - USTC & ICT, CAS
    - 部分softmax调度到TC执行，减少TC和VU的重叠不完整问题。
  - MetaAttention: A Unified and Performant Attention Framework Across Hardware Backends [[arXiv](https://arxiv.org/abs/2502.15349)] [[Code](https://github.com/microsoft/AttentionEngine)]
    - SJTU, IPADS & PKU & MSRA
    - LLM、线性注意力、RetNet、Mamba2、DeepSeek MLA、Sigmoid Attention、Sparse GQA 等变体在 score 计算、normalization、tensor shape、是否 recurrent、是否稀疏等方面差异很大，编译框架。

# **Diffusion Models**

- Difflow: A Data-Characteristic-Aware Serving System for Diffusion Models
  - THU
  - prompt / image / ControlNet / LoRA 等输入可能在部分请求间重复，生成分辨率和视频帧形状可能不同，denoising loop 中又会出现跨迭代不变张量。编译层处理。
- MixFusion: A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models
  - UWaterloo & CMU & Rice
  - diffusion Serving，不同分辨率切分相同patch并行。

# **GNN**

- APERTURE: Algorithm-System Co-Optimization for Temporal Graph Network Inference
  - BUAA
- ElasGNN: An Elastic Training Framework for Distributed GNN Training
  - BUAA
- TAC: Cache-based System for Accelerating Billion-Scale GNN Training on Multi-GPU Platform
  - UCAS

# **Sparse Matrix**

- ASM-SpMM: Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration
  - SYSU
  - SpMM到ARM SME的对齐。
- Exploiting Efficient Mapping and Pipelined Execution for Accelerating SpMV on Tensor Cores
  - BUAA
  - SpMV到TC的对齐kernel库。
- VDHA: Vector-Driven Hash Aggregation for Sparse Matrix–Sparse Vector Multiplication on GPUs
  - THU
  - SpMSpV的kernel优化。

# **Quantization**

- RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization [[Artifact](https://github.com/thu-pacman/RoMeo)]
  - THU
  - SpMSpV的kernel优化。

# **Cache Management**

- Cacheman: A Comprehensive Last-Level Cache Management System for Multi-tenant Clouds
  - Alibaba Cloud

# **Misc**

- Scaling GPU-to-CPU Migration for Efficient Distributed Execution on CPU Clusters
  - GaTech
  - 单GPU的kernel到多CPU的迁移框架。
- zBuffer: Zero-Copy and Metadata-Free Serialization for Fast RPC with Scatter-Gather Reflection
  - XMU & Alibaba & SJTU

**HPCA26**

# **LLM**

- LLM inference
  - AUM: Unleashing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving [[Paper](https://www.cs.sjtu.edu.cn/~lichao/publications/AUM_Unleashing_HPCA-2026-Wang.pdf)]
    - SJTU & Alibaba
    - CPU cluster的调度优化：LLM负载使用AU，通用负载使用其他模块，资源隔离的调度优化。
  - GyRot: Leveraging Hidden Synergy between Rotation and Fine-grained Group Quantization for Low-bit LLM Inference
    - KAIST
    - 低比特 LLM 推理中，rotation-based quantization、fine-grained group quantization 和 asymmetric quantization 各自能提升量化精度，但直接组合会产生算法和硬件两层冲突。算法-硬件协同设计。
  - ELORA: Efficient LoRA and KV Cache Management for Multi-LoRA LLM Serving
    - SJTU & Huawei Cloud & HKUST
    - LoRA-LLM Serving的缓存（LoRA Cache、KVCache）管理优化，在线服务中，底座模型常驻 GPU，而不同任务、用户或场景会动态访问不同 LoRA；同时，多轮对话、翻译、personal agent 这类 workload 又会复用同一前缀的 history KV cache。把热门 LoRA 和 KV 放在 GPU 内存中可以减少 cold-start，但 GPU 内存有限，现有系统往往把 LoRA cache 和 KV cache 分区管理，导致 TTFT 和 TPOT 在动态负载下显著恶化。
  - Towards Resource-Efficient Serverless LLM Inference with SLINFER [[arXiv](https://arxiv.org/abs/2507.00507)]
    - SJTU
    - CPU-GPU系统的模型调度，从模型实例-GPU一对一绑定，到多模型实例共享Node（CPU/GPU），动态分时支持动态分区。
  - LILo: Harnessing the On-chip Accelerators in Intel CPUs for Compressed LLM Inference Acceleration
    - UIUC & Seoul National University & Intel
    - CPU LLM Serving的压缩、通信优化。
  - PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System [[arXiv](https://arxiv.org/abs/2412.20166)]
    - Hanyang University & SK hynix & KAIST
    - LLM Serving的PIM系统设计。
- Speculative decoding
  - Adaptive Draft Sequence Length: Enhancing Speculative Decoding Throughput on PIM-Enabled Systems
    - HUST
    - SpD的PIM硬件系统设计。
- Quantization
  - BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache [[arXiv](https://arxiv.org/abs/2503.18773)]
    - Edinburgh & MSRA
    - CC+TC作低比特量化计算和运行时。
  - AQPIM: Breaking the PIM Capacity Wall for LLMs with In-Memory Activation Quantization
    - Institute of Science Tokyo
    - GPU+PIM：PIM负责KVCache压缩和相关计算。
- Reasoning
  - The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective [[arXiv](https://arxiv.org/abs/2506.04301)]
    - KAIST
    - Agent系统评估框架。
  - PASCAL: A Phase-Aware Scheduling Algorithm for Serving Reasoning-based Large Language Models
    - KAIST
    - reasoning模型中，TTFT包含Prefill全部和Decode中reasoning token decoding，不包含Decode中answering token decoding，因此调度需要优先reasoning token decoding。
  - RPU - A Reasoning Processing Unit
    - Harvard
    - 面向LLM dataflow的加速芯片设计。
- RAG
  - VectorLiteRAG: Latency-Aware and Fine-Grained Resource Partitioning for Efficient RAG [[arXiv](https://arxiv.org/abs/2504.08930)]
    - GaTech
    - RAG Serving对齐CPU+GPU系统。
- VLM
  - Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models [[arXiv](https://arxiv.org/abs/2512.14661)] [[Code](https://github.com/dubcyfor3/Focus)]
    - Duke
    - VLM稀疏pipeline到SA加速器的对齐。
- Video LLM
  - V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval [[arXiv](https://arxiv.org/abs/2512.12284)]
    - KAIST
    - KV Cache在VLM Stream场景的GPU kernel优化。
- Misc
  - Towards Compute-Aware In-Switch Computing for LLMs Tensor-Parallelism on Multi-GPU Systems
    - SJTU & Huawei
    - LLM TP的多卡通信优化。
  - RoMe: Row Granularity Access Memory System for Large Language Models [[arXiv](https://arxiv.org/abs/2512.01541)]
    - Seoul National University & Meta
    - 内存控制器的重新设计，加大访问粒度。
  - LEGO: Supporting LLM-enhanced Games with One Gaming GPU
    - SJTU & Tongji University
    - 单GPU的游戏-LLM双负载场景

# **GPU**

- UVM
  - ARIADNE: Adaptive UVM Management for Efficient GPU Memory Oversubscription [[Artifact](https://zenodo.org/records/17852674)]
    - Yonsei University & DGIST
    - GPU UVM的缺页处理机制的优化
- Chiplet
  - COMET: Communication and Memory Co-Design for Fine-Grained AI Inference in MCM Accelerators
    - NUDT & PKU
    - MoE推理在chiplet架构的通信优化。
  - Deadlock-Free Bridge Module for Inter-Chiplet Communication in Open Chiplet Ecosystem
    - NUDT
    - 解决chiplet之间的cache一致性协议中的死锁。
  - LRM-GPU: Alleviating Synchronization Overhead for Multi-Chiplet GPU Architecture
    - SYSU
    - 多chiplet GPU架构的同步机制的开销优化
- Sparsity
  - Swift: High-Performance Sparse-Dense Matrix Multiplication on GPUs
    - Hunan University
    - SpMM的GPU kernel优化。
  - Uni-STC: Unified Sparse Tensor Core
    - CUP-Beijing & NUDT
    - GPU Sparse TC的新设计，面向SpMM（A稀疏）、SpMV（A稀疏）、SpMSpV（A和V稀疏）、SpGEMM（A和B稀疏）算子。
- Misc
  - QuCo: Efficient and Flexible Hardware-Driven Automatic Configuration of Tile Transfers in GPUs
    - University of Murcia & William&Mary & NVIDIA
    - ATT（如TMA）运行时配置的GPU硬件单元扩展。
  - μShare: Non-Intrusive Kernel Co-Locating on NVIDIA GPUs
    - TJU
    - GPU运行时的设计，运行时将kernel启动划分为half-plus block，允许SM并行资源需求不同的kernel，提高资源使用率。
  - FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive operators via Inter-Core Connection [[arXiv](https://arxiv.org/abs/2512.12949)]
    - SJTU
    - 编译框架：利用SM之间的DSMEM缓存融合kernel之间的中间数据，而不用写回GMEM。

# **VAR**

- VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy
  - HKUST
  - VAR模型的新pipeline和对应加速器设计。

**EuSYS26**

# **Large Language Models (LLMs)**

- LLM Inference
  - Speculative Decoding
    - AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding
      - CMU & Princeton & EPFL & AWS & Purdue
      - LLM的不同应用SLO需求的Serving框架优化。
  - Request Scheduling
    - FlexPipe: Adapting Dynamic LLM Serving Through Inflight Pipeline Refactoring in Fragmented Serverless Clusters [[Paper](https://doi.org/10.1145/3767295.3769316)] [[arXiv](https://arxiv.org/abs/2510.11938)]
      - SIAT, CAS & UCAS & UCSD & University of Macau
      - LLM Serving的集群运行时调度系统设计。inflight pipeline refactoring 根据实时 CV、吞吐、延迟和队列状态选择候选 granularity，在 burst 时拆细 stage，在稳定时合并 stage。
    - TokenFlow: Responsive LLM Text Streaming Serving under Request Burst via Preemptive Scheduling
      - SJTU & GMU & China Telecom Shanghai
      - 交互式Streaming指标的Serving系统的抢占优化。
    - AdaGen: Workload-Adaptive Cluster Scheduler for Latency-Optimal LLM Inference Serving
      - UVA & HPE Labs & UC Riverside
      - Serving的cluster调度，预测输出长度来分类/聚类请求，调度不同集合的请求。
    - SkyWalker: A Locality-Aware Cross-Region Load Balancer for LLM Inference
      - UC Berkeley & RUC & Rice
      - 多地域LLM Serving的调度。
    - PiLLM: Resource-Efficient LLM Inference Using Workload Prediction
      - ShanghaiTech & SenseTime & Beihang
      - 面向LLM Serving的token level资源管理机制，基于预测更激进分配内存同时减少OOM。
  - KV Cache and Memory Management
    - Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading [[Paper](https://doi.org/10.1145/3767295.3769319)] [[arXiv](https://arxiv.org/abs/2502.05370)]
      - Stevens Institute of Technology & Waterloo & Rutgers
      - 面向LLM Serving的token level资源管理机制，基于预测更激进分配内存同时减少OOM。
    - KUNSERVE: Parameter-centric Memory Management for Efficient Memory Overloading Handling in LLM Serving
      - SJTU
      - LLM Serving中的KVCache的缓存策略优化，丢弃跨实例重复的KVCache。
    - High Throughput and Low Latency LLM Serving via Adaptive KV Caching
      - University of Macau & SIAT, CAS & NTU
      - Serving调度框架，丢弃部分较旧token的KV Cache，利用空闲SM重算token的KV Cache。
  - Multiplexing
    - MFS: An Efficient Model Family Serving System for LLMs
      - HKUST & USTC & Inspur
      - 模型family的Serving系统，微调最大模型将model family调整为共享部分权重的一个嵌套模型，即经过tier-1就结束对应小模型，继续tier-2结束对应中模型，继续tier-3结束对应大模型推理。相当于将大模型能力切分到3个tier。
    - Efficient Multimodal Serving via Module Multiplexing
      - HKUST & SYSU & XJTU & MetaX
      - 多模态Serving，不同模块（VAE Encoder/Decoder、BackBone）的token走独立pipeline执行。
  - Sparsity
    - SAS: Sparse Attention Synthesizer for Efficient Language Model Inference
      - Amazon
      - sparse kernel自动生成框架。
  - Heterogeneous Environment
    - Scaling LLM Test-Time Compute with Mobile NPU on Smartphones
      - THU & USTC & MSR & AIR, THU
      - TTC在decoding时需要计算多个branch，多个branch的GEMV能充满NPU的GEMM Core。
    - TailorLLM: Collaborative End-Cloud Inference of Large and Small Language Models Based on Low-Rank Adaptation
      - BUPT
      - 端侧SLM、云中心LLM的协同调度，复杂请求到云端、简单请求在端侧处理。
  - Trusted Execution
    - TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone
      - SJTU
      - 移动端SLM的权重加密推理。
  - LLM-based Applications
    - AIMS: A Cost-Efficient Framework for LLM-based Agent Deployment in Cloud-Edge Hybrid Environments
      - UVA & Microsoft
      - 面向agent级联调用请求的云LLM-端SLM的调度优化。
    - From Imperative to Declarative: Towards LLM-friendly OS Interfaces for Boosted Computer-Use Agents
      - IS, CAS & UCAS & SJTU
      - 将Agent控制现有软件的面向人的接口（视觉定位，鼠标追踪等操作序列）整形为状态空间的建模和变换。

# **Diffusion Models**

- Image Editing
  - FlashPS: Efficient Generative Image Editing with Mask-aware Caching and Scheduling [[arXiv](https://arxiv.org/abs/2505.20600)] [[Code](https://github.com/Sylvia-16/FlashPS)]
    - HKUST & Alibaba
    - 将图像编辑请求中的mask引入单个请求的pipeline，mask部分token时修改的主力需要计算，unmasker token直接使用之前缓存的token值。Serving场景决定多个请求的调度设计，每个请求在每一步去噪后退出，并加入新请求。

# **Model Serving**

- Automated End-to-End Model Serving with Cooperative Compilation and Scheduling
  - NJU & Hunan University
  - 静态编译，把算子拆成tile，每个tile设计多种可能的micro-kernel（使用不同类型或分配粒度的计算单元），运行时根据空闲资源、kernel冒险等情况动态调度sub-kernel，达到运行时kernel fusion的效果，GPU Scheduler。

# **Resource Management**

- Serverless Computing
  - Efficient Data Passing for Serverless Inference Workflows: A GPU-Centric Approach
    - HUST & CUHK-Shenzhen & TeleAI & HKUST
    - Serverless Workflow的集群优化。
  - iRoute: Local Routing Table-based Workflow Management in Serverless Computing
    - TJU & THU & IEIT Systems & Inspur
  - DROPS: Managing Serverless Resource Pools in Microsoft Azure Functions
    - Waterloo & MSR & Microsoft
  - Squeezy: Rapid VM Memory Reclamation for Serverless Functions
    - NTUA & UIUC
  - Demystifying Serverless Costs on Public Platforms: Bridging Billing, Architecture, and OS Scheduling
    - UBC & Johns Hopkins
  - Fix: externalizing network I/O in serverless computing
    - Stanford
- GPU Cluster Management
  - Bridging the GPU Utilization Gap: Predictive Multi-Dimensional Resource Scheduling for AI Workloads
    - THU & Alibaba & SJTU
  - Untangling GPU Power Consumption: Job-Level Inference in Cloud Shared Settings [[Paper](https://hal.science/hal-05291033v1/file/GPU_power_Eurosys.pdf)]
    - ÉTS & Inria & OVHcloud & CNRS
    - Present practical job-level power estimation methods for GPUs under temporal sharing, spatial sharing, and passthrough deployment modes in cloud environments.
    - Show that GPU sharing can improve energy efficiency for small AI workloads, and identify substantial GPU underutilization in an IaaS GPU cluster.
