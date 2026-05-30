# 实验_Serving调度

## LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving

- 属于Serving调度的实现是什么？实验比较什么？
  实现是LiquidServe——基于LiquidGEMM W4A8 GEMM kernel构建的端到端LLM serving系统。LiquidServe集成了：(1) FlashAttention-2用于runtime attention计算；(2) PagedAttention用于KV cache管理；(3) SmoothQuant per-token动态激活量化（FP16→INT8）；(4) KV cache per-channel静态INT8量化（scale factor离线计算）；(5) 权重离线两级量化（FP16→INT8→UINT4, group size=64）。实验比较baseline：QServe（W4A8 KV4, group size=128）、TensorRT-LLM（FP16/W4A16/W8A8/FP8）。评估指标：peak token generation throughput（input=1024, output=512, batch size 1-256）、固定batch size下的throughput、per-layer time breakdown（GEMM/Attention/Others）。消融实验：LiquidServe/wo（替换LiquidGEMM为QServe的W4A8 kernel）对比LiquidGEMM的系统级加速贡献。

- 硬件平台是什么，配置是什么。
  NVIDIA H800 GPU（80GB HBM），Intel Xeon Platinum 8457C CPU，2.9TB RAM。PyTorch 2.4.0，CUDA 12.4。

- 开源Serving框架是什么。修改了什么。
  未使用单一现有开源Serving框架作为主代码基。LiquidServe自建serving系统，集成多个开源组件：(1) FlashAttention-2 [6]——替换为FlashAttention-2而非FlashAttention-3（后者专为FP8优化，但LiquidServe使用INT8 activation）；(2) PagedAttention [12]——用于高效KV cache管理，支持内存分页；(3) SmoothQuant [29]——per-token动态激活量化，通过smooth scale除以激活后在线量化FP16→INT8。LiquidGEMM GEMM kernel使用CUTLASS和Cute编程原语构建，集成WGMMA指令、TMA异步数据搬运和barrier同步。修改/自建内容：(1) 自建LiquidGEMM kernel——fused dequantization+MMA mainloop，Dual-MMA packed layout，ImFP pipeline；(2) 离线量化pipeline——FP16→INT8→UINT4两级量化+per-token激活量化；(3) KV cache INT8量化——per-channel静态量化替代QServe的4-bit KV cache。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源情况：LiquidGEMM/LiquidServe未提供开源代码。论文说明LiquidGEMM已部署为ByteDance Seed生产LLM serving基础设施的primary GEMM kernel。

  作用：LiquidServe是一个W4A8量化的端到端LLM serving系统，实现最高达4.94x系统级加速（vs QServe）。核心优化在于LiquidGEMM kernel通过LiquidQuant硬件友好dequantization和ImFP pipeline解决W4A8 GEMM的dequantization瓶颈，使W4A8在实践中超越W8A8和FP8，而不仅仅是理论roofline分析上的优势。

  全过程（以LLaMA2-7B单层decoding为例）：
  ```
  Serving系统接收请求（input_len=1024, output_len=512, batch_size=128）
    → Prefill阶段：FlashAttention-2处理prompt tokens
      - KV cache: PagedAttention管理，per-channel INT8量化存储
    → Decoding阶段（逐token生成）：
      Layer处理循环（自回归）：
        Step 1 - Attention:
          FlashAttention-2: QKV projection (LiquidGEMM W4A8 GEMM + output projection)
          KV cache更新: PagedAttention append新token的KV
        Step 2 - FFN:
          gate_proj + up_proj: LiquidGEMM W4A8 GEMM（INT8 activation × UINT4 weight）
          SiLU activation: elementwise on FP16
          down_proj: LiquidGEMM W4A8 GEMM
        Step 3 - Activation Quantization:
          动态per-token: FP16 activation → smooth_scale除法 → clamp → INT8
          （fused到前一个kernel的epilogue中，overhead微小）
    → 输出token → 循环至output_len完成后返回

  LiquidGEMM kernel内部（每个GEMM调用）：
    Load WG: TMA从GMEM加载UINT4 weight tile到SMEM（Dual-MMA packed layout, LDG.128）
    → ImFP: Load WG将weight切分为fine-grained tasks写入SMEM
    → Compute WG_0: 从SMEM竞争获取task → LDS.128加载到RF → unpack 4-bit
      → IMAD + XOR dequantization (4 elements/2 instructions, CUDA Cores)
      → WGMMA.m64nNk32 MMA (Tensor Cores, INT8)
    → Compute WG_1: 同时处理另一task（dequantization与MMA跨WG自然重叠）
    → Epilogue: 第一级dequantization（INT8→FP16, per-channel scale）→ 写回GMEM
  ```

## HuntKTm: Hybrid Scheduling and Automatic Management for Efficient Kernel Execution on Modern GPUs

- 属于Serving调度的实现是什么？实验比较什么？
  实现是task scheduler，在multi-GPU系统中自动将多个任务（每个任务是包含多个kernel的独立GPU程序）调度到合适的GPU设备上，实现task-level并发执行。task scheduler包含三部分：(1) resource analyzer——编译期分析每个kernel的launch配置（thread数、register数、shared memory量）和memory object大小，汇总每个task的计算和内存资源需求；(2) lazy engine——运行时拦截CUDA API调用，延迟GPU相关操作（memory allocation/deallocation/data transfer），收集编译期无法确定的动态资源信息，精确预测task资源需求后发送给task dispatcher；(3) task dispatcher（Algorithm 1）——遍历可用GPU列表，基于三个维度（threads、registers、shared memory）评估SM可用量，选择拥有最多available SMs且内存和hardware queue充足的GPU。若所有GPU都无法满足需求，task挂起到pending queue等待资源释放。

  实验比较的task-level scheduling baseline包括：SA (Single-Assignment，每个GPU一次分配一个任务)、CASE (compiler-assisted scheduling framework，自动分析资源需求并调度)、HuntK (仅stream scheduler + SA)、HuntKT (stream scheduler + task scheduler，无memory management)。也对比了单任务kernel concurrency baseline：Taskflow（静态）、GrSched（动态）、Serial。评估指标包括system throughput（多GPU系统）、硬件资源利用率（DCGM采集的SM occupancy/FP32 utilization/memory bandwidth utilization）、memory reduction ratio、task-level kernel execution speedup。

- 硬件平台是什么，配置是什么。
  服务器配备4× NVIDIA A100 GPU (40GB HBM each, 6912 CUDA cores each)、2× AMD EPYC 7742 64核处理器、256 GB DDR4内存。操作系统Debian 10.2.1，NVIDIA driver 555.42.06。另一平台：4× NVIDIA RTX 4090 24GB GPU、2× Intel Xeon Gold 6338N CPU、1024 GB DRAM。NVIDIA MPS启用以实现跨进程space-sharing并发，NVIDIA persistence mode启用以减少GPU初始化开销。

- 开源Serving框架是什么。修改了什么。
  未使用现有开源Serving框架。HuntKTm自建task scheduler系统，基于CUDA Runtime和LLVM Compiler Infrastructure实现。修改/自建内容包括：
  - 通过function wrapper拦截所有CUDA runtime调用（cudaMallocAsync, cudaFreeAsync, kernel launch等），收集resource信息。
  - lazy engine维护deferred CUDA operation queue，在task被dispatch到具体GPU前暂缓所有GPU操作。
  - resource analyzer通过nvcc获取每个kernel的register和shared memory使用量。
  - task dispatcher实现Algorithm 1的资源感知调度算法，通过shared memory与lazy engine通信。
  - 调度后调用cudaSetDevice绑定task到目标GPU，使用cudaDeviceGetDefaultMemPool获取默认memory pool并用cudaMemPoolSetAttribute设置release threshold为预测的memory footprint。
  - 每个GPU的可用hardware queue数上限设为32（匹配CUDA runtime最大连接数）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源链接：https://github.com/Gemini321/HuntKTm

  作用：HuntKTm的task scheduler解决的是"多个无协作关系的GPU程序（tasks）如何自动在multi-GPU系统上高效共置"的问题。传统方式需要用户手动为每个程序指定目标GPU并确保不超出内存容量，HuntKTm通过编译期资源分析和运行时lazy execution实现全自动、内存安全的task-to-GPU调度。

  全过程（以W4 workload为例，包含约16个混合大小的task同时到达的场景）：
  ```
  用户编译多个multi-kernel CUDA程序（已通过stream scheduler自动转化为多stream版本）
    → LLVM pass (resource analyzer): 分析每个kernel的launch配置(num_threads, registers, shared_mem)
      和memory object大小，插入cudaTaskSchedule调用点在资源需求完全确定的位置
    → 编译生成带resource metadata的可执行程序
    → 运行时：W4的所有16个task同时到达，每个task启动lazy engine
    → lazy engine拦截所有CUDA API调用并存入deferred operation queue，暂不执行
    → 当程序执行到cudaTaskSchedule点：lazy engine聚合所有stream的首kernel computing需求
      和所有memory object大小，通过shared memory发送给task dispatcher
    → task dispatcher执行Algorithm 1:
      for each GPU g in 4×A100:
        if g.free_memory >= task.memory and g.free_hw_queues > 0:
          score = g.available_SMs - min(thread_score, reg_score, shmem_score)
          选择score最大的GPU
    → 若GPU可用：lazy engine → cudaSetDevice(target_gpu_id)
      → 初始化memory pool (size = predicted_peak_memory)
      → 顺序执行deferred operations (cudaMallocAsync → cudaMemcpyAsync → kernel launches)
      → 多stream的kernels并发在MPS环境下与同GPU其他task的kernels space-sharing执行
    → 若GPU不可用：task进入pending queue，等待其他task释放资源后被唤醒重试
    → DCGM监控每个GPU的硬件指标 → 计算system throughput = 总完成task数 / 总耗时
  ```

  HuntKTm通过该流程，相比CASE在多GPU A100系统上实现平均33.2% throughput提升，在4090系统上平均52.5%提升（因4090内存更紧张，memory management效果更显著）。

## Marconi: Prefix Caching for the Era of Hybrid LLMs

- 属于Serving调度的实现是什么？实验比较什么？
  实现是Marconi——首个面向Hybrid LLMs（Attention+SSM混合架构）的prefix caching系统。核心实现包括两部分：(1) Judicious Admission策略——通过radix tree记录历史请求，识别两种前缀复用模式（Purely Input: 系统提示词、few-shot示例等被多请求共享的前缀；Input and Output: 对话历史等从最后一个decoded token继续的前缀），仅缓存高复用概率的SSM状态，每个序列最多2个checkpoint；(2) FLOP-Aware Eviction策略——Utility Score = recency + α × flop_efficiency，其中flop_efficiency = 复用该state节省的总FLOPs / 该state占用的内存字节数，优先保留计算节省密度高的缓存条目。Marconi将SSM states和KV caches统一管理在单个radix tree中。

  实验比较baseline：fine-grained checkpointing（naive方案，每x token存一个SSM state checkpoint）、SGLang+（扩展SGLang支持Hybrid LLMs，使用LRU eviction）、vLLM+（扩展vLLM支持Hybrid LLMs）。评估指标：token hit rate (%)、Time To First Token (ms)、P95 TTFT reduction。消融实验：Marconi vs fine-grained checkpointing（评估judicious admission贡献）、FLOP-aware eviction vs LRU eviction（评估eviction policy贡献）。结果：Marconi vs fine-grained checkpointing token hit rate提升4.5×–34.4×，P95 TTFT降低36.1%–71.1%。FLOP-aware eviction单独（vs LRU）提升19%–219% token hit rate。

- 硬件平台是什么，配置是什么。
  Cloudlab节点，Ubuntu 22.04，32-core CPU，约20 GB磁盘空间，约7 GB traces数据。GPU硬件：论文未明确说明具体GPU型号（实验使用离线trace-based模拟而非实际GPU部署）。模型：NVIDIA Mamba2-Hybrid-7B（4 Attention + 24 SSM + 28 MLP layers），tokenizer使用meta-llama/Llama-2-7b-hf。

- 开源Serving框架是什么。修改了什么。
  基于radix-tree prefix cache架构（源自vLLM和SGLang的prefix caching设计），Marconi修改/扩展内容包括：
  - radix_cache_hybrid.py：核心caching逻辑，实现judicious admission和FLOP-aware eviction策略。在radix tree中统一管理Attention层的KV cache和SSM层的recurrent states。Tree节点分为intermediates（purely-input前缀，被多请求共享）和leaves（input-and-output前缀，对话末尾）。
  - radix_cache_vllm.py：vLLM适配版本，集成到vLLM serving framework。
  - policy_exploration.py：可插拔eviction policy框架，支持V1 (SGLang+ LRU)、V2 (Marconi)、V3 (offline-optimal oracle)。自定义policy可通过实现新的evict_policy_version加入。
  - config_tuner.py：自动调优α参数（FLOP efficiency的权重系数）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源链接：https://github.com/ruipeterpan/marconi。MLSys 2025 Outstanding Paper Honorable Mention。使用conda环境（environment.yml），Python 3.11.9。

  作用：Marconi解决Hybrid LLMs（Mamba2-Hybrid, Jamba等）中SSM层的prefix caching难题——SSM状态通过in-place recurrent更新，无法像Attention KV cache那样通过切片回滚到任意前缀位置。Naive checkpointing（每隔x token存一次SSM state）导致：(1) 缓存条目稀疏命中；(2) SSM state尺寸大（固定大小但比单token KV大几个数量级）；(3) 频繁缓存thrashing。Marconi通过judicious admission（仅缓存高复用概率状态，每序列至多2个checkpoint）和FLOP-aware eviction（权衡recency和compute savings）解决上述问题。

  全过程（以NVIDIA Mamba2-Hybrid-7B serving为例，LMSys conversational workload）：
  ```
  请求到达 → Marconi prefix cache处理:

    Step 1 - Speculative Admission:
      新请求token序列插入radix tree（tentative insertion）
      → 如果请求创建新branching point（intermediate node）:
         标记为purely-input前缀（如system prompt），高复用概率 → admit
      → 如果请求延伸到leaf node:
         标记为input-and-output前缀（如对话结束位置） → 仅缓存最后token的SSM state
      → 每个序列至多产生2个SSM state checkpoint

    Step 2 - Cache Lookup & Hit:
      radix tree从根节点匹配请求的token序列
      → 匹配到最深节点 → 获取已缓存的KV cache（Attention层）+ SSM states（SSM层）
      → 未匹配的tail tokens需重新prefill计算
      → Attention层: KV cache直接切片复用（传统prefix caching）
      → SSM层: 从checkpoint恢复recurrent state，从此state继续forward

    Step 3 - Eviction (当缓存容量满时):
      for each cached entry in radix tree:
        FLOP_efficiency = 复用该state节省的总FLOPs / state内存字节数
        Utility = recency_score + α × flop_efficiency_score
      淘汰Utility最低的entry
      → α由config_tuner根据workload模式自动调优

    Step 4 - 执行输出:
      Prefill剩余tail tokens（利用缓存的KV+SSM states加速） → 生成first token (TTFT)
      → Decoding阶段自回归生成后续tokens → 返回完整响应
  ```

  Evaluation配置：sweep各种cache size和request arrival patterns。Per-trace runtime: LMSys ~30s, ShareGPT ~5s, SWEBench ~5-10min（32-core CPU离线模拟）。完整实验sweep约12小时。图7（token_hit_rate.py）复现Marconi vs fine-grained checkpointing结果，图8（sglang_comparison.py）复现FLOP-aware eviction vs SGLang+ LRU比较。

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

## vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models

- 属于Serving调度的实现是什么？实验比较什么？
  实现是vLLM-Omni——一个面向any-to-any多模态模型的fully disaggregated serving system。核心实现包括：(1) Stage Abstraction——用户将复杂的多模态模型架构分解为stage graph，node代表model stage（AR LLM、DiT等），edge代表stage间数据传输函数；(2) Disaggregated Stage Execution Backend——每个stage由独立的execution engine serving（vLLM engine用于AR stages，专用diffusion engine用于DiT stages），支持per-stage request batching、灵活GPU分配和intra-stage serving优化（continuous batching、chunked prefill、execution graph compilation）；(3) Unified Connector——负责stage间中间数据传输，支持inline control queues（小payload）、system shared memory（大payload，单节点）和Mooncake-based RDMA（跨节点）；(4) Streaming Stage Output——支持下游stage在上游未完全完成时就开始incremental处理。

  实验比较baseline：Qwen-Omni系列（Qwen2.5-Omni、Qwen3-Omni）使用HuggingFace Transformers默认实现；BAGEL使用原始实现；MiMo-Audio使用原始实现；Diffusion模型（Qwen-Image、Qwen-Image-Edit、Wan2.2系列）使用Diffusers库。评估指标：Real-Time Factor (RTF)、Job Completion Time (JCT)、Tokens Per Second (TPS) for Thinker和Talker components。关键结果：Qwen3-Omni JCT降低91.4%，RTF降低90.7%，Thinker TPS提升12.97×，Talker TPS提升7.98×。BAGEL T2I speedup 2.40×，I2I speedup 3.72×。MiMo-Audio RTF从1.39降至0.12（11.58× speedup）。Diffusion model整体1.26× speedup。

- 硬件平台是什么，配置是什么。
  服务器配备2块accelerator设备（每块80GB memory），24 CPU cores，192 GB system memory。使用virtual setup环境，vLLM version 0.12.0。BAGEL和MiMo-Audio在单accelerator（80GB）上评估。

- 开源Serving框架是什么。修改了什么。
  基于vLLM v0.12.0。修改/新增内容：
  - Stage Abstraction Frontend：提供Python API让用户定义stage graph——每stage实现forward（step-centric batched forward）和preprocess（修改stage输入）函数，edges定义stage-transfer函数。
  - Orchestrator：管理stage执行和调度incoming requests，route requests through stage graph。
  - AR Stage Engine：基于vLLM engine，每stage独立运行自己的scheduler（batching）、KV-cache manager（per-stage KV管理）和model runner。新增per-request intermediate data dictionary（用户可在transform和preprocess函数中访问和更新）。preprocess在每个decode iteration调用，output processor负责执行transform function并传输数据到下游stage的设备。
  - Diffusion Engine：全新实现，集成flash attention、SAGE attention、TurboAttention等attention优化，TeaCache/cache-dit等caching策略，RingAttention context parallelism和Ulysses sequence parallelism。支持text-to-image（Z-Image、Qwen-Image、Flux）、image editing（Qwen-Image-Edit、LongCat-Image-Edit）和video generation（Wan2.2、HunyuanVideo）。
  - Unified Connector：单节点使用inline control queues（小payload）+ system shared memory（大payload）；多节点通过Ray orchestration + Mooncake-based connector（TCP/RDMA transport）。Connector同时处理intra-stage transfer（prefill-decode KV cache、encoder-prefill MM cache），兼容EPD disaggregation。
  - Streaming Stage Output：output processor异步stream partial outputs到下一stage，减少TTFT和enabling streaming responses。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  开源链接：https://github.com/vllm-project/vllm-omni（注意：URL带连字符）。vLLM-Omni v0.20.0（2026/05最新版本）。文档：https://docs.vllm.ai/projects/vllm-omni/en/stable/

  作用：vLLM-Omni解决的核心问题是"existing LLM serving frameworks（vLLM、SGLang）的step-centric abstraction无法表达multi-stage any-to-any pipeline"。传统框架仅支持单AR decoding或单DiT denoising stage，开发者需手动实现inter-stage transfer outside serving framework，导致：(1) 无法利用continuous batching、chunked prefill等serving优化；(2) 计算资源无法跨stage灵活分配，stage co-located as monolith导致资源浪费。

  全过程（以Qwen3-Omni音频生成请求为例，2×80GB accelerators）：
  ```
  用户请求到达vLLM-Omni server（含text + audio prompt，请求生成text + audio outputs）
    → Orchestrator接收请求，调度到stage graph执行:

    Stage 1 - Thinker (AR LLM, ~30B, TP-2 across both accelerators):
      vLLM engine接收stage input:
        multimodal encoder处理输入（audio→Whisper encoder, image→ViT, video→Vision encoder）
        → encoder embeddings + text token embeddings concatenated
      → vLLM model runner执行continuous batching:
        Prefill阶段: chunked prefill处理prompt tokens
          - PagedAttention管理KV cache
          - Thinker forward function + customized preprocess (concatenate MM embeddings)
        Decode阶段: 自回归生成text tokens + 产生hidden states
          - 每个decode iteration: preprocess → forward → 输出text token + hidden states
          - 使用execution graph compilation加速
      → Output Processor:
        对每个完成text生成的request:
          Thinker2Talker transform function:
            提取Thinker hidden states → 转换为Talker input embeddings
            → 通过unified connector传输到Talker stage设备(device-1)

    Stage 2 - Talker (AR LLM, smaller but compute-intensive, 置于device-1):
      vLLM engine接收streaming input:
        每iteration调用preprocess:
          连接Thinker hidden states + Talker input embeddings + original multimodal embeddings
      → AR decode生成audio codec tokens (平均545.4 tokens)
      → Output Processor:
        Talker2Vocoder transform:
          收集generated codec tokens → 通过unified connector传输到Vocoder(device-0)
        Streaming: 当Talker产出initial tokens时Vocoder即可开始处理

    Stage 3 - Vocoder (DiT/CNN, 置于device-0):
      Diffusion engine接收codec tokens:
        从partial tokens开始incremental denoising (streaming)
        → DiT denoising with flash attention + TeaCache caching
        → 生成audio waveforms
      → Final output: text response + audio waveforms返回client
  ```

  对比Baseline（HuggingFace Transformers monolithic执行）：
  ```
  请求到达 → 单进程加载所有model components:
    1. encoder处理multimodal input
    2. Thinker generate (自定义generate loop, 无可用的continuous batching/paged attention/chunked prefill)
    3. 等待Thinker完全完成 → 提取hidden states
    4. Talker generate (自定义generate loop)
    5. 等待Talker完全完成 → 提取codec tokens
    6. Vocoder generate waveforms
  → 返回结果
  关键性能损失:
    - Continuous batching不可用 → 多请求无法batch，GPU利用率低
    - Chunked prefill不可用 → 长prompt prefill latency高
    - Execution graph compilation不可用 → kernel launch overhead大
    - Stage co-located → Thinker(30B)和Talker竞争memory，无法独立扩缩容
    - 无streaming → Vocoder必须等Talker完全完成，增加end-to-end latency
  ```

  Qwen3-Omni关键数据流分析:
  ```
  Thinker TPS: baseline ≈ X tokens/s, vLLM-Omni = 12.97× X (execution graph compilation + chunked prefill)
  Talker TPS: baseline ≈ Y tokens/s, vLLM-Omni = 7.98× Y (continuous batching + KV cache management)
  Talker占总延迟大部分（生成545.4 audio tokens vs Thinker 150.9 text tokens）

  Thinker→Talker transfer: 5.49ms (shared memory) / 8.28ms (Mooncake RDMA) — negligible vs total latency (tens of seconds)
  Talker→Vocoder transfer: 0.53ms (shared memory) / 3.34ms (Mooncake) — negligible
  ```
