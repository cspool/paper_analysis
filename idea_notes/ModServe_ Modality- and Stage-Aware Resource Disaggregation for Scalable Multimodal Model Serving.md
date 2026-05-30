## ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving

- baseline方法是什么？
  Baseline是vLLM [27] monolithic LMM serving部署。在monolithic架构下，LMM的image preprocessor、image encoder和LLM backend（prefill + decode）被打包为单个serving instance，使用统一的TP配置和batch size。所有instance完整副本部署，text-only和image-text请求都由同一个instance处理。如需扩容，整个instance（包括所有stage）一起扩容。

  Baseline的三个核心缺陷：(1) **Image encoding成为TTFT瓶颈但无法独立优化**——CroAttn模型中image encoding占TTFT的65–79%（Insight 1），但monolithic将其与LLM backend绑定，无法独立scale out encoder和并行化encoding；(2) **统一batch size和TP配置导致资源浪费**——image encoding是compute-bound，应使用小batch和TP-1；LLM prefill在DecOnly中compute-bound（需大TP），在CroAttn中更轻量（需小TP）；decode是memory-bound（需大batch）。Monolithic强制所有stage使用相同配置（Insight 4、5）；(3) **无法应对modality-specific bursts**——生产环境image-text和text-only traffic表现出独立的burst pattern（Insight 6），monolithic无法针对性扩缩容，image burst时只能整体扩容，导致LLM backend过度provisioning。

  全栈执行例子（以Llama3.2-11B (CroAttn) monolithic部署在4×A100 TP-4上，处理含4张896×896图像的请求）：
  - 算法层：ViT-H/14 (630M) image encoder + Llama 3.1 (8B) LLM backbone with 4 cross-attention layers（共40 layers，其中4层为CroAttn）。Image tokens仅在CroAttn layers参与cross-attention，自注意力层仅处理text tokens。Connector MLP将image tokens映射到LLM token space。
  - 系统框架层：vLLM v0.7.2，PD colocated模式（prefill和decode在同一instance）。Image preprocessing (CPU) → image encoding (GPU, TP-4, batching the 4 images' tiles) → LLM prefill (GPU, TP-4, batch_size=1) → LLM decode (GPU, TP-4, continuous batching via PagedAttention)。所有4张图的tiles串行或小batch编码在4个GPU上（630M模型被TP分到4 GPU，inter-GPU通信开销 > 计算节省）。
  - 编译框架层：论文未明确说明。PyTorch eager execution on CUDA。
  - kernel调度层：论文未明确说明。NVIDIA A100 GPU kernel执行：ViT forward（matmul + attention + MLP kernels）、LLM prefill（self-attention + cross-attention + MLP kernels）。
  - 硬件架构层：DGX-A100 server，8× A100 80GB via NVLINK 3.0（600 GB/s），内部4 GPU用于TP-4。AMD EPYC 7V12 96-core CPU for image preprocessing。1900 GiB DRAM。

  在此执行下，若同时到达大量image请求：(1) 所有请求排队等待TP-4 GPU资源，image encoding无法并行到更多GPU；(2) 大image请求造成HoL blocking——4张图的encoding阻塞后续小请求；(3) 扩容时TP-4整体扩容，即使只需更多encoder也要带4个GPUs。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出ModServe，将LMM serving拆分为Image Instances和Text Instances两个独立资源池，通过stage-aware profiling、token-aware autoscaling和modality-aware routing解决上述三个缺陷。

  **解决缺陷1（image encoding瓶颈）**：将Image Instances与Text Instances物理分离。Image Instances可以独立扩展到不同GPU，甚至多张GPU（如每个TP-1），实现单请求内多图像跨多个Image Instance并行编码（Insight 2：images within a request have no compute dependency）。ModServe调度将大请求的图像split到多个Image Instance，类似request chunking效果，减少HoL blocking。

  **解决缺陷2（统一配置浪费）**：Image Pool Manager和Text Pool Manager各自根据offline LMM profile独立决定各stage的TP度、max batch size。例如，ViT-H/14 (630M) encoder在TP-1时吞吐最高（因TP-4下inter-GPU communication > compute saving），LLaMA 3.1 (8B) LLM backend在TP-4时TTFT最低。ModServe允许Image Instance TP-1、Text Instance TP-4共存于同一physical GPU pool，灵活资源分配。

  **解决缺陷3（modality-specific bursts）**：Token-aware autoscaling基于modality-specific load（image tokens/sec for Image, prompt tokens/sec for Text）而非请求速率，独立计算各池需要的instance数（⌈ML/MC⌉）。Image burst时仅scale out Image Instances，LLM backend不受影响。CroAttn模型在image burst时LLM prefill不增加（因image tokens仅参与4/40 cross-attention layers），因此Text Instances无需扩容——这正是ModServe达到最高41.3% cost saving的来源。

  全栈执行例子（以Llama3.2-11B (CroAttn) ModServe部署在16-server 128-GPU集群上，处理含4张896×896图像的请求）：
  - 算法层：同上（ViT-H/14 encoder + Llama 3.1 8B with CroAttn layers）。但encoder和LLM backend现在运行在不同GPU上，通过RDMA传输image tokens。
  - 系统框架层：ModServe on vLLM v0.7.2 + HuggingFace Transformers。
    - Image Instances (TP-1, batch_size=1): 接收image-text请求 → CPU image preprocessing (numactl NUMA-pinned) → GPU encoding (ViT-H/14 forward, 4 tiles) → image tokens注册RDMA buffer
    - Image Pool Manager (gRPC server): 接收实时load metric → 计算autoscaling决策（⌈image_tokens_per_sec / per_instance_max_capacity⌉）→ 管理Image Instance生命周期
    - Text Pool Manager (gRPC server): 同上，基于prompt tokens/sec → 选择目标Text Instance
    - Pull-based RDMA: Text Instance收到RDMA地址 → NCCL+GPU Direct RDMA pull image tokens → Connector MLP（共置于Text Instance GPU）
    - Text Instance (TP-4, mixed batch): LLM prefill (image tokens仅在4 CroAttn layers) + decode (PagedAttention)
    - Modality-Aware Router: image-text→least-image-token-load Image Instance; text→least-pending-text-tokens Text Instance (CroAttn按text tokens路由)
    - SLO-driven Priority Scheduler: 优先短请求，防止长image-text请求HoL blocking短text请求
  - 编译框架层：论文未明确说明。PyTorch eager + NCCL backend。
  - kernel调度层：论文未明确说明具体kernel。A100 GPU上执行ViT matmul/attention kernels（Image Instance）和LLM self-attention/cross-attention/MLP kernels（Text Instance）。Image token transfer使用GPU Direct RDMA over InfiniBand。
  - 硬件架构层：16× DGX-A100 servers（128× A100-80GB total）。NVLINK 3.0 intra-server，InfiniBand inter-server（支持GPU Direct RDMA，P99 image token transfer latency 5ms，<0.5% TTFT for CroAttn）。Image和Text Instance可colocate同server（如1× TP-4 Text Instance + 2× TP-2 Image Instance on 8-GPU server），但instance配置独立。

  结果：vs vLLM monolithic baseline，ModServe取得3.3×（Llama3.2 CroAttn）–5.5×（InternVL DecOnly）higher throughput（static allocation），25–41.3% cost saving（autoscaling with production traces）。PD disaggregation之上额外2.8× average TTFT reduction（图19）。

  设计思路核心：ModServe的本质是将LMM serving从"model instance as atomic unit"的粗粒度资源管理转变为"pipeline stage as decoupled resource pool"的精细管理。这个转变的可行性依赖于三个发现：(1) 各stage对batch/TP的响应曲线截然不同（使独立配置有意义）；(2) image encoding tokens间无依赖（使并行编码可行）；(3) 不同modality的traffic burst pattern独立（使针对性autoscaling高效）。
