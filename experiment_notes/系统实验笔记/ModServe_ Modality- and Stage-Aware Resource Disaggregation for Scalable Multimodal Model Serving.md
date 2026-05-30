## ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving

- 属于Serving调度的实现是什么？实验比较什么？
  实现是ModServe——一个模块化LMM serving系统，将多模态推理pipeline解耦为Image Instances（image preprocessing + encoding）和Text Instances（LLM prefill + decode）两个独立资源池。核心实现包括：(1) 离线LMM profile生成——对image encoder和LLM backend独立profiling，记录不同TP度、batch size、load下的性能数据；(2) Token-Aware Pool Autoscaling——基于token throughput（image tokens/sec, prompt tokens/sec）而非请求速率去动态扩缩容各阶段的instance数量；(3) Modality-Aware Request Routing——image-text请求路由到image token负载最少的Image Instance，大请求跨多个Image Instance并行编码，text请求路由到pending token最少的Text Instance；(4) SLO-driven Priority Scheduling——优先调度短请求以避免HoL blocking；(5) Pull-based RDMA Image Token Transfer——Image Instance完成编码后延迟传输，等所有image tokens就绪后由Text Instance通过RDMA pull。实现基于vLLM v0.7.2（Text Instance）和HuggingFace Transformers（Image Instance），约5000行Python代码。

  实验比较baseline：vLLM（monolith，将image preprocessor/image encoder/LLM backend打包为单个TP实例）。消融ablation：ModServe-Decoup（仅stage decoupling，无modality-aware scheduling和routing）、ModServe-Sched（decoupling + modality-aware scheduling）、ModServe（decoupling + scheduling + routing，即完整系统）。PD disaggregation兼容性对比：PD-Monolith（prefill实例同时host image encoder）vs PD-ModServe（image encoder完全解耦到独立GPU）。

  评估指标：TTFT (avg/P99/P90)、maximum throughput under SLO、GPU allocation cost（autoscaling场景下的GPU数）、image token transfer latency。关键结果：ModServe vs vLLM monolith实现3.3×–5.5×更高throughput（static allocation），25–41.3% cost saving（autoscaling下）。PD-ModServe vs PD-Monolith额外提供最高2.8× average TTFT reduction。

- 硬件平台是什么，配置是什么。
  集群：16台DGX-A100服务器（共128 GPUs）。每台DGX-A100：8× NVIDIA A100 80GB GPU via NVLINK 3.0，96 AMD EPYC 7V12 CPU cores，1900 GiB DRAM。跨服务器互联：InfiniBand（支持GPU Direct RDMA）。Characterization使用单台DGX-A100。部署使用BF16精度。

- 开源Serving框架是什么。修改了什么。
  基于vLLM v0.7.2（Text Instance）和HuggingFace Transformers（Image Instance）。修改/新增内容：
  - Text Instance：基于vLLM，复用其PagedAttention KV cache管理、continuous batching、tensor parallelism。新增：与Image Instance的pull-based通信接口，支持接收RDMA传输的image tokens。
  - Image Instance：基于HuggingFace Transformers加载image encoder模型（ViT-H/14, SigLIP, InternViT等），新增：image preprocessing pipeline（numactl限制单NUMA node）、image encoding engine（支持tile-level parallelization）、token序列化与RDMA地址注册。
  - Image Pool Manager & Text Pool Manager：新增轻量级gRPC server（部署在dedicated VM）实现：offline profiling数据查询、token-aware autoscaling决策（副本数 = ⌈当前load / 每instance最大capacity⌉）、TP配置管理、heartbeat-based failure detection。
  - Modality-Aware Router：新增per-request routing逻辑——image-text请求→最少pending image tokens的Image Instance；text请求→最少pending tokens的Text Instance（CroAttn按text tokens，DecOnly按total tokens）。
  - SLO-driven Priority Scheduler：替换vLLM默认FIFO调度，优先调度短请求以降低HoL blocking。
  - Pull-based RDMA Transfer：使用PyTorch distributed communication + NCCL backend + GPU Direct RDMA实现GPU-to-GPU image token传输。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源情况：论文代码未明确开源。但论文表明生产LMM inference traces已开源：https://github.com/Azure/AzurePublicDataset。

  作用：ModServe解决monolithic LMM serving的三个核心问题：(1) image encoding在TTFT中占比高（CroAttn模型可达79%），成为瓶颈；(2) 不同stage（image preprocessing/encoding vs LLM prefill/decode）对batching和TP的敏感度不同，monolithic统一配置浪费资源；(3) 生产环境多模态traffic存在image-driven bursts且与text traffic模式独立，monolithic无法针对性扩缩容。

  全过程（以Llama3.2-11B (CroAttn) serving一个image-text请求为例，8-A100 server上部署ModServe于16-server集群）：
  ```
  请求到达ModServe（含文本prompt + 4张896×896图像）
    → Modality-Aware Router: 识别为image-text请求
    → 选择Image Instance: 查询所有Image Instance的pending image token load
      → 4张图片分配给两个load最少的Image Instance（各2张）
    → Image Instance 1 & 2 并行执行:
      Image Preprocessing (CPU, numactl bound to single NUMA node):
        每张原始图像 → resize+rescale+pad+normalize
        → segmentation into tiles (Llama3.2: 560×560 tiles, 4 tiles/image)
        → tile-level transformations → 输出processed image tiles tensor [16, C, H, W]
      Image Encoding (GPU, TP-1, batch_size=1):
        ViT-H/14 (630M): 16 tiles → forward pass through ViT encoder
        → 输出 image tokens: shape [16, 1601] = 6404 tokens total（1601 tokens/tile × 4 tiles）
        → 注册RDMA memory region，发送RDMA地址给Pool Manager
    → Image Pool Manager: 聚合4张图的encoding完成信号
    → Text Pool Manager: Pull-based RDMA Transfer决策
      → 查询各Text Instance的pending text token load + queue size
      → 选择pending最少的Text Instance（TP-4, 4×A100）
      → 通知该Text Instance: RDMA addresses of Image Instances 1 & 2
    → Text Instance (GPU, TP-4, Llama3.2-11B):
      RDMA Pull: 从Image Instance 1 & 2的GPU memory读取6404 image tokens
        (InfiniBand RDMA, P99 latency 5ms)
      → Connector (MLP, 共置在Text Instance): image tokens → LLM token space mapping
      → LLM Prefill (40 layers total):
        Self-Attention Layers (36 layers): 仅text tokens参与self-attention
        Cross-Attention Layers (4 layers): text tokens attend to image tokens + text tokens
        → 高效prefill（image tokens不参与所有layer的self-attention，FLOPs显著减少）
      → First token生成 → TTFT测量完毕
      → Decode阶段 (memory-bound, continuous batching):
        自回归逐token生成: 使用PagedAttention管理KV cache
        → TBT测量（受compute insensitivity影响，与monolith相当）
      → 生成完成 → 返回text response给client
  ```

  Autoscaling循环（每5分钟，由Pool Managers驱动）：
  ```
  Pool Manager读取实时load指标:
    → 计算新instance数 = ⌈ML / MC⌉
        ML: 当前modality-specific load (image tokens/s for Image, prompt tokens/s for Text)
        MC: offline profile中的最大capacity（不违反SLO的最大吞吐）
    → 若不足: 启动新instance（warm-start from cached model profile）
    → 若过剩: gracefully drain后关闭instance
    → CroAttn text autoscaling基于text tokens only
    → DecOnly text autoscaling基于total tokens (text+image)
  ```

  对比Monolith部署（vLLM）执行同样请求：
  ```
  请求到达 → 单TP-4实例（含image preprocessor + encoder + LLM backend全在4 GPUs上）
    → Image Preprocessing (CPU) → Image Encoding (GPU, TP-4, ViT-H/14, 630M分到4 GPU)
      → 低效：小encoder分到4 GPU，inter-GPU communication overhead > 计算节省
    → LLM Prefill (GPU, TP-4): 所有40层处理text+image tokens
      → DecOnly模型更差：高resolution图像导致长序列prefill
    → 若同时有大量image请求 → HoL blocking, TTFT急剧退化 (Figure 1)
    → Autoscaling: 只能整体扩容TP-4实例（包含不必要的LLM backend扩容），浪费GPU
  ```
