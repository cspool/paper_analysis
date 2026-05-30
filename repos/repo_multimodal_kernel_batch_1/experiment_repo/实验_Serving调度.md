# 实验_Serving调度

## Characterizing Concurrency Mechanisms for NVIDIA GPUs under Deep Learning Workloads

> **近似层次匹配说明**：本文并非修改开源Serving框架，而是直接使用CUDA API（priority streams、time-slicing、MPS）在单GPU上执行concurrent training+inference workload，以评估不同并发机制对inference serving性能的影响。这与Serving调度的核心关注点（多请求调度、SLO、吞吐量）紧密相关，但实现层面在CUDA/kernel级而非Serving框架级。

- 属于Serving调度的实现是什么？实验比较什么？
  论文实现了在单GPU上同时运行latency-sensitive inference serving和best-effort training的并发workload，通过三种NVIDIA GPU并发机制（priority streams、time-slicing、MPS）进行调度。实验比较各机制下inference请求的turnaround time（平均延迟）、variance（可预测性），以及training execution time（资源利用率proxy）。Inference请求使用两种模式：MLPerf server mode（Poisson过程到达，500请求）和MLPerf single-stream mode（连续请求，5000请求）。

- 硬件平台是什么，配置是什么。
  NVIDIA GeForce RTX 3090（Ampere microarchitecture）：82 SMs、1536 threads/SM、64KB registers/SM、1024KB shared memory/SM、24GB GDDR6X、6144KB L2 cache。

- 开源Serving框架是什么。修改了什么。
  论文未使用或修改开源Serving框架。所有并发调度通过CUDA runtime API直接控制：
  - **Priority streams**：将training和inference task置于同一OS进程的不同CUDA stream，inference stream设高优先级（-2到0三级），CUDA thread block scheduler优先从高优先级stream取blocks。
  - **Time-slicing**：两task作为独立进程运行，由CUDA application-level scheduler以约2ms固定时间片轮转调度。
  - **MPS**：启动MPS server，两task作为独立MPS client进程提交kernels，MPS server调度来自不同CUDA context的kernel blocks，允许spatial sharing（同一SM colocation），可设置per-client thread limit。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文未提供独立开源代码。以下是基于论文描述的各机制的调度全过程：

  **Priority Streams 全过程**（以ResNet-50 training + inference为例）：
  1. Host端：单进程内创建两个CUDA stream——stream_high（priority=0，inference kernels）和stream_low（priority=-2，training kernels）。
  2. Training task持续向stream_low提交kernel序列（如convolution、batch norm、ReLU等），Inference task根据请求模式向stream_high提交kernel序列。
  3. GPU端Thread Block Scheduler：当两种stream都有pending blocks时，总是优先从stream_high取blocks分配到SM。但已分配的low-priority blocks不会被抢占。
  4. **Compounded Delay问题**：当inference kernel执行完毕，下一inference kernel到达前有一段时间窗口。在此期间，training kernel抢占了GPU所有SM资源并填满blocks。下一inference kernel到达后必须等待当前执行中的training blocks完成才能被调度，造成约2-4×的turnaround time增加。
  5. 结果输出：每个inference kernel完成后记录timestamp，计算端到端turnaround time。

  **Time-Slicing 全过程**：
  1. Host端：两个独立进程各自创建CUDA context并提交kernels。
  2. GPU Application-Level Scheduler：以约2ms固定时间片round-robin分配GPU。每时间片内，整个GPU（所有82 SMs）专属于一个进程。
  3. 时间片切换：约145μs切换开销（通过global timer register测量，推测一半保存context、一半恢复context），但register和shared memory似乎不传输（推测为避免高开销）。
  4. 限制：两进程的kernel资源需求总和不能超过GPU硬件上限（即使不同时执行），否则第二进程OOM。
  5. 结果特点：turnaround time低且可预测（2ms延迟 + 切换开销），但utilization差（resource在时间片内空闲时无法被另一进程使用）。

  **MPS 全过程**：
  1. Host端：启动MPS control daemon和MPS server，两task作为MPS client进程连接server。
  2. MPS Server：接收来自两个CUDA context的kernel dispatch请求。调度策略为FCFS + leftover policy（优先调度最近到达kernel的所有blocks）。
  3. SM Spatial Sharing：两个进程的thread blocks可以在同一SM上colocated，只要两者thread总和不超过SM limit。
  4. 实验结果：MPS utilization最好（training execution time增加通常仅20-30秒），但inference degradation因无优先级而显著（如ResNet-152 turnarround time 2×）。

  **性能对比总结**（基于论文Figure 1/3）：
  - ResNet-50: Priority Streams +103% TT, MPS +78% TT, Time-Slicing +18% TT（但training time +90s）
  - VGG-19: Priority Streams worst（training含大量long-running kernels），Time-Slicing TT best
  - RNNT+BERT: Time-Slicing表现差（因memory transfer contention）

## A Survey of Resource-efficient LLM and Multimodal Foundation Models

- 属于Serving调度的实现是什么？实验比较什么？
  本文为综述论文，无原创Serving系统实验。§5.3-5.4系统梳理云侧与端侧LLM serving系统的优化技术：
  (i) **推理加速**（§5.3.1）——Kernel优化（FlashAttention/FlashAttention-2加速prefill、Flash-Decoding/FastGen加速decode）、Parallelism策略（TP+PP+DP+Expert Parallelism混合）、Request Batching与调度（Orca的iteration-level batching消除padding、SARATHI的chunked-prefill与decode混合调度、Splitwise的prefill-decode分离部署）；
  (ii) **内存节省**（§5.3.2）——vLLM的PagedAttention（block级按需分配KV cache消除碎片，up to 29×吞吐提升）、S-LoRA的Unified Paging支持多LoRA adapter、SGLang的RadixAttention支持跨请求KV cache复用、FlexGen的激活/参数offload到DRAM/NVMe；
  (iii) **新兴部署平台**（§5.3.3）——SpotServe on spot instances（动态调整并行策略应对抢占）、HexGen on heterogeneous GPUs（进化算法搜索placement和parallelism）；
  (iv) **端侧Serving**（§5.4）——Edge-cloud协作（EdgeFM）、端侧MoE（EdgeMoe的expert-wise bit-width adaptation、PC-MoE的参数委员会机制）、内存优化（LLMCad的speculative decoding + token tree、PowerInfer的热/冷神经元分离GPU/CPU计算）、I/O优化（STI的动态权重bit-width加载、LLM in a Flash的细粒度闪存管理）、Kernel优化（mllm-NPU利用移动NPU加速prefill）、LLMaaS范式（LMS的细粒度KV cache管理、ELMS的弹性SLO支持）。

- 硬件平台是什么，配置是什么。
  综述未进行统一实验。被引述系统的硬件平台包括：数据中心GPU（A100/H100/TPU）、消费级GPU、手机端（iPhone 12、安卓设备w/ NPU）、Raspberry Pi 5。

- 开源Serving框架是什么。修改了什么。
  综述表5总结开源框架：
  - **vLLM**（UC Berkeley）：PagedAttention block级KV cache管理，消除内存碎片。vAttention进一步直接依赖OS/CUDA做物理内存重分配，端到端吞吐再提升1.29×。
  - **DeepSpeed-Inference/MII**（Microsoft）：支持ZeRO优化、模型压缩、FastGen动态split-fuse schedule。
  - **TensorRT-LLM**（NVIDIA）：集成AWQ/GPTQ/SmoothQuant量化、speculative decoding、TP+PP并行、PagedAttention。
  - **HuggingFace TGI**：支持TP、bitsandbytes/GPTQ量化、PagedAttention。
  - **SGLang**：RadixAttention进行跨请求KV cache复用，prompt programming primitives。
  - **LightLLM**：token级别KV cache内存管理。
  - **MLC-LLM**：编译器加速的通用部署方案，支持native API。
  - **llama.cpp**：CPU端LLM推理，支持2-8bit整数量化（K-quant），3-4×加速。
  - **mnn-llm**（阿里）：将推理分为prefill/decoding两阶段分别优化。
  - **mllm**：面向多模态大模型的端侧推理引擎。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  综述全部材料开源：https://github.com/UbiquitousLearning/Efficient_Foundation_Model_Survey（LaTeX源码+参考文献）。以下以vLLM为例说明Serving框架输入到硬件执行的全过程：

  以vLLM的PagedAttention为例（§5.3.2）：
  ```
  用户请求 → HTTP API Server
    → Scheduler（iteration-level调度，管理request queue）
      → Block Manager（为每个request的KV cache分配逻辑block，
          映射到物理GPU内存block，类似OS虚拟内存）
      → Model Runner（batching requests with PagedAttention kernel）
        → PagedAttention CUDA Kernel:
          for each query token q_i:
            for each physical_block b in block_table[req_id]:
              // 从GPU HBM读取该block的K/V cache
              K_block = KV_cache[b]  // shape [block_size, num_heads, head_dim]
              // 计算block内attention
              scores = q_i @ K_block^T / sqrt(head_dim)
              p = softmax(scores)
              o_i += p @ V_block
          → output token → 追加到KV cache（可能需要新block分配）
    → 返回generated token到用户
  ```
  PagedAttention消除KV cache碎片，相比vanilla KV cache（预分配max_seq_len连续空间），内存利用率从约20-30%提升至接近100%。

## FlashInfer Efficient and Customizable Attention Engine for LLM Inference Serving

- 属于Serving调度的实现是什么？实验比较什么？
  实现FlashInfer v0.2作为LLM serving框架的attention backend，集成入SGLang v0.3.4和MLC-Engine进行端到端serving优化。核心Serving调度层面的设计：(i) **统一block-sparse KV-cache格式**——将PageAttention和RadixAttention等非连续KV-cache存储统一映射为BSR矩阵，query/output用ragged tensor存储消除padding，与上层serving框架的page table / radix tree管理透明对接；(ii) **Composable formats for prefix-caching**——利用多BSR格式分解KV-cache稀疏矩阵：共享prefix对应的dense submatrix用大$B_r$存储（3 queries共享同一KV-cache于shared memory），唯一suffix用$B_r=1$存储（各query在自己的threadblock内独立访问），无需数据移动仅需计算不同submatrix的indices/index pointer arrays；(iii) **Dynamic scheduling per generation step**——CPU端plan function处理变化的sequence length信息生成load-balanced调度计划，可跨层复用（同一步内所有decode attention共享plan），plan info cached on GPU workspace buffer；(iv) **CUDAGraph-compatible persistent kernel**——attention+contraction merged into single persistent kernel with fixed grid size，workspace buffer fixed offset保证CUDAGraph pointer不变，plan function (on CPU) not captured by CUDAGraph，run function (on GPU) captured in graph。

  实验比较：(i) **Standard LLM serving**——FlashInfer+SGLang vs SGLang-Triton，Llama 3.1 8B (1xH100)和70B (4xH100)，ShareGPT + Variable synthetic workload (seqlen uniform 512-2048)，测量P99 TTFT<200ms约束下的ITL和TTFT；(ii) **Long-context inference (Streaming-LLM)**——FlashInfer fused RoPE+attention kernel vs unfused kernels，Vicuna-13B on MT-Bench，变化Streaming-LLM recent window size；(iii) **Parallel generation with prefix-caching**——FlashInfer composable formats vs single format in MLC-Engine，Llama 3.1 8B/70B on ShareGPT，request rate=16，变化parallel tokens n∈{1,2,4,8,16,32,64}，测量ITL和TTFT。

- 硬件平台是什么，配置是什么。
  - Standard serving: 1× NVIDIA H100 80GB SXM (Llama 3.1 8B)，4× NVIDIA H100 80GB SXM (Llama 3.1 70B)
  - Long-context: NVIDIA A100 (Streaming-LLM, Vicuna-13B)
  - Parallel generation: H100 80GB SXM (Llama 3.1 8B/70B on MLC-Engine)
  - CUDA 12.4 + PyTorch 2.4.0，FP16存储和计算

- 开源Serving框架是什么。修改了什么。
  - 开源框架：SGLang v0.3.4 (https://github.com/sgl-project/sglang)，MLC-Engine (https://github.com/mlc-ai/mlc-llm)
  - FlashInfer修改：不修改上层serving框架的调度逻辑（continuous batching / radix tree管理 / page table管理保持不变），而是作为drop-in attention backend替换。在SGLang中替换Triton attention backend，在MLC-Engine中增加composable formats支持。
  - Serving框架通过FlashInfer的PyTorch编程接口（Listing 1）集成：创建workspace buffer→compile阶段：对每种attention spec+task_info组合创建AttentionWrapper→JIT compile kernel→dummy plan→capture CUDAGraph→runtime阶段：选择最优CUDAGraph→每generation step更新seqlen_info→attn.plan() (CPU)→g.replay() (GPU executed via CUDAGraph)。
  - Composable formats实现：prefix-caching场景下创建多个AttentionWrapper（不同block sizes对应shared prefix和unique suffix），framework根据KV-cache配置选择最优CUDAGraph。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  已开源：https://github.com/flashinfer-ai/flashinfer (Apache-2.0)，已集成进SGLang、vLLM、MLC-Engine。

  使用例子（SGLang + FlashInfer, Llama 3.1 8B, 1xH100, ShareGPT dataset, online serving）：
  1. **输入**：ShareGPT对话请求流（variable-length prompts），SGLang API server接收请求，continuous batching调度器维护运行中请求batch。请求按到达顺序被调度，P99 TTFT target < 200ms。
  2. **SGLang RadixAttention管理**：SGLang维护radix tree结构管理KV-cache pages（prefix-caching），KV-cache以non-contiguous page table形式存储在GPU HBM中。当请求到达时，SGLang的scheduler将请求加入running batch，更新page table。
  3. **FlashInfer作为attention backend**：
     a. **Compile阶段**（init时）：对每种task_info（attention variant spec + 平均query长度范围 + block size配置）调用`AttentionWrapper(attn_spec, task_info, workspace_buffer)`→JIT编译生成CUDA kernel（核心是BSR FlashAttention模板）→dummy plan→捕获对应CUDAGraph。多种配置对应多个CUDAGraphs（不同composable formats / tile size组合）。
     b. **Runtime text generation loop**（per step）：`seqlen_info.update()`（查询当前batch中各请求的query长度和KV-cache长度）→`attn.plan(seqlen_info)`（CPU端Algorithm 1计算load-balanced CTA调度plan，跨所有layer复用）→`g.replay()`（GPU端CUDAGraph执行：persistent attention kernel处理可变长度KV-cache chunks → persistent contraction kernel将partial outputs用⊕操作合并为final attention outputs）。
     c. **Attention计算细节**（decode step）：Q ragged tensor（各请求1 token）→ BSR KV-cache（page table映射为non-zero blocks）→ per-CTA根据plan info处理分配的KV chunks → partial attention states (O_partial, LSE_partial) → ⊕ compose → final O。
  4. **Prefill处理**：FlashInfer prefill kernel使用FA2/FA3 algorithm处理prompt tokens（query长度=prompt长度），BSR格式处理页表映射。FlashInfer prefill kernel配合causal masking。
  5. **输出**：生成tokens返回客户端。Throughput = total output tokens / total time。ITL = decode step平均时间。TTFT = prefill + 第一次decode时间。
  6. **性能**（Figure 7）：FlashInfer backend vs Triton backend——Llama 3.1 8B ITL reduction 29-69%，TTFT reduction 29-69%（ShareGPT）；Llama 3.1 70B ITL reduction 29-69%，TTFT reduction 29-69%。Variable synthetic workload similarly consistent speedup across all settings。
  
  并行生成例子（MLC-Engine + FlashInfer composable formats, Llama 3.1 70B, 4xH100, ShareGPT, n=4 parallel tokens）：
  1. **输入**：ShareGPT请求，每个请求要求并行生成n=4条回复（OpenAI API "n" parameter equivalent）
  2. **Prefix-caching**：所有4条回复共享输入prompt prefix的KV-cache。MLC-Engine的prefix-caching机制识别shared prefix
  3. **Composable format分解**：
     - Sub-matrix 1 (shared prefix): BSR with $B_r=3, B_c=1$。3 queries共享同一KV-cache page，在shared memory中复用加载的K/V tile（high-bandwidth SMEM access），减少global memory traffic
     - Sub-matrix 2 (unique suffixes): BSR with $B_r=1, B_c=1$。各query独立访问各自unique KV-cache，通过global memory / L2 cache
     - 无需KV-cache data movement，仅计算两种submatrices的`kv_indptr`和`kv_indices`数组
  4. **Attention computation**：每个submatrix使用独立的AttentionWrapper（不同$B_r$对应的不同CUDAGraph）。大$B_r$的shared prefix部分tensor core GEMM复用shared memory中KV tile（Q tile (3, head_dim) × K tile (l_kv, head_dim)），显著降低global memory bandwidth需求
  5. **输出**：ITL reduction 17.42%（8B为13.73%），TTFT reduction 22.86%（8B为16.41%）at n=4 (peak speedup)。n<4时block size增大不足抵消overhead，n>32时attention不再dominate computation（短sequence ShareGPT场景），speedup plateau
  
  - **作用**：FlashInfer作为通用attention backend统一解决LLM serving中KV-cache异构存储（paged/radix tree）、workload动态性（variable-length sequences per batch）、attention变体多样性（GQA/MQA/specialized masks/fused operations）三个核心问题。将上层serving框架的page table / radix tree数据结构直接映射为BSR sparse matrix格式，实现memory layout到compute kernel的端到端优化，无需修改框架的调度和内存管理逻辑。

## Cornserve Efficiently Serving Any-to-Any Multimodal Models (Cornfigurator)

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：Cornfigurator，一个面向通用 Any-to-Any（A2A）多模态模型推理 Serving 的自动化部署规划器。它根据模型定义（DAG 组件图）、配置空间（executor 类型及其配置）、workload（请求类型分布）和 GPU 预算，自动搜索最优的 colocation/disaggregation 组合、executor 配置（batch size、parallelism degree）和请求路由策略，以最大化每种请求类型的 goodput（满足 per-type 延迟目标的吞吐量）。规划器使用三层粗到细评估管道：network flow 估算吞吐量上限 → Monte Carlo 采样估算延迟 → request-level 模拟器精确建模排队动态，每层后剪枝淘汰劣化方案。
  - 实验比较：Cornfigurator 生成的部署方案 vs. vLLM-Omni（专家手动调优的固定策略）、Full Disaggregation（完全解耦的受限版本）、vLLM（monolithic）、ModServe（MLLM 专用解耦）、EPD（encoder-prefill-decode 解耦）在多种 A2A 模型上的 goodput。

- 硬件平台是什么，配置是什么。
  - 2× AWS p4de.24xlarge 实例，每实例 8× NVIDIA A100-80GB GPU（NVSwitch 互联），跨节点 400 Gbps 带宽。实验使用 8 GPU 和 16 GPU 配置。

- 开源Serving框架是什么。修改了什么。
  - 开源框架：Cornserve（https://github.com/cornserve-ai/cornserve），一个通用 A2A 分布式 Serving 平台。Cornfigurator 是基于 Cornserve 之上的自动化规划器（https://github.com/cornserve-ai/cornfigurator）。规划器本身约 5K 行 Rust 实现。它不修改 Cornserve 的内部机制，而是作为规划层，生成 physical plan（节点拓扑、executor 数量、配置、路由概率）交给 Cornserve 执行。Planning 流程：Profiler 先对每个 model component 在各种配置下进行 benchmark → Planner 枚举 logical subplans（simple/compound）→ 组合成 logical plans → 注解 GPU 分配和 executor 配置生成 physical plans → 三阶段评估选最优。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源：是。Cornfigurator 开源在 https://github.com/cornserve-ai/cornfigurator，Cornserve 开源在 https://github.com/cornserve-ai/cornserve。
  - 使用例子（Qwen 3 Omni 30B, 16 GPU, 2/3 audio output workload）：
    1. **输入**：Model definition（DAG: E_img, E_vid, E_aud, L_th(thinker LLM), L_ta(talker LLM), G_aud(vocoder)），Configuration space（executor 类型如 encoder/LLM/DiT executor，各支持的 batch size、parallelism degree），Workload（8 种 request type 的分布，如 T+I→T, T+I+V→T, T+I→A, T+I+V→A 等），Latency targets（per-type），GPU budget N=16。
    2. **Profiler**：对每个 executor type 在 A100-80GB 上 sweep batch size 和 parallelism degree，记录稳态吞吐和延迟（去除排队延迟），输出 per-executor-config 的 profile。
    3. **Planner 枚举**：从 model graph 生成 simple subplans（通过枚举每条 colocation edge 的 Keep/Merge 决策），合并共享节点的 subplans 为 compound subplans（k_c=2），组合为 logical plans（k_s=2），注解 executor 分配和路由概率生成 physical plans（约 483M 候选）。
    4. **粗到细评估**：
       - Phase 1 (Network flow, 3.48s)：计算每个 plan 在各 node 汇聚的瓶颈吞吐量 R_d，剪枝冗余配置（483M→1.95M）。
       - Phase 2 (Monte Carlo, 34.23s)：采样请求、按路由概率流经 plan 各 executor，累积 per-executor 处理延迟得出 per-type 延迟 CDF，计算 goodput，剪枝 Pareto-suboptimal plans（1.95M→25）。
       - Phase 3 (Simulator, 0.83s)：request-level 模拟器以 α·R_d 速率运行 workload，建模排队动态和 inter-type 竞争，计算最终 goodput（25→5）。
    5. **输出 Physical Plan**：一个 compound subplan 结构——一个分支用 disaggregated video encoder 处理 heavy video-input 请求，另一个分支用 monolithic 配置处理其余请求，共享 13×(L_ta+G_aud) talker+vocoder executor。
    6. **部署执行**：Cornserve runtime 接收 physical plan，在 16×A100 GPU 上按配置启动 executor 实例，根据路由概率将各 request type 的请求分发到对应 subplan 路径，各 executor 按配置的 batch size 和 parallelism 执行推理计算。
    - **作用**：自动为通用 A2A 模型找到最大化 per-type goodput 的部署方案，避免人工专家调优，1.12×–6.32× 优于 baseline。

## FastTree Optimizing Attention Kernel and Runtime for Tree-Structured LLM Inference

- 属于Serving调度的实现是什么？实验比较什么？
  FastTree 作为 SGLang 的 plugin 实现，属于 Serving 调度层优化。核心实现：(i) 利用 SGLang 的 radix tree 管理全局 KV cache（已有机制），FastTree 读取该 radix tree 结构后生成 context-queries grouping plan；(ii) tree structure-adaptive runtime 在每次 radix tree 结构变化时（因新请求到来/旧请求完成）重新执行 greedy heuristic 搜索最优分组方案；(iii) 将 attention 计算从原有的 per-query 分离模式替换为 FastTree 的 tree-structured attention kernel，query 按共享前缀聚合后批量计算 attention；(iv) 预处理 overhead（CPU greedy search + grouping plan generation）被 SGLang 的多步连续 decoding 摊销，且可与 GPU 计算 overlap。
  实验比较：FastTree+SGLang vs SGLang-Triton vs SGLang-FlashInfer 在 4 种 tree-structured workload 上的端到端 throughput（tokens/s）：(A) multi-level system prompt（随机替换 Meta AI system prompt 中的 country/language）；(B) multiple few-shot learning（系统 prompt + 8 组 20-shot examples + 16 questions）；(C) multi-chain reasoning（每个问题 4 chains）；(D) multi-document QA（Llama-2 report 拆分为多文档前缀）。所有 benchmark batch=128, gen_len=256 tokens。额外进行 breakdown analysis（decoding latency、CPU preprocessing overhead、GPU kernel execution time）。

- 硬件平台是什么，配置是什么。
  NVIDIA H100 GPU (80GB)，CUDA 12.2。CPU 端预处理轻量（BFS greedy search + virtual tree generation），在实验中 overhead 可忽略。

- 开源Serving框架是什么。修改了什么。
  - 开源框架：SGLang v0.2.13（https://github.com/sgl-project/sglang），已集成 radix tree KV cache 管理。
  - FastTree 修改：作为 plugin 替换 SGLang 中 decoding 阶段的 attention backend。SGLang 原有两种 attention 实现（Triton kernels from LightLLM、FlashInfer CUDA kernels）。FastTree 新增第三种 backend，在 decode 阶段使用 tree-structured attention kernel，prefill 阶段沿用 FlashInfer。不修改 SGLang 的核心调度逻辑（continuous batching / radix tree 维护），而是在 attention 计算层插入优化。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  已开源：https://github.com/PanZaifeng/FastTree-Artifact（Apache-2.0）。Docker 环境含 SGLang 0.2.13、FlashInfer 0.1.6、Triton 3.0.0。

  使用例子（Llama-2-7B, GQA=1, benchmark B multiple few-shot learning, batch=128, gen=256）：
  1. **输入**：128 个并发请求到达 SGLang API server。这些请求共享 3-level tree prefix——系统 prompt (Meta AI, 3193 tokens) → 8 组 few-shot example 组合（20-shot each）→ 16 个独立 question per 组合。SGLang 将收到的请求组织为 radix tree（root=系统 prompt, L1=example combinations, L2=questions），KV cache 按 tree 结构非连续存储（paged KV cache blocks）。
  2. **Radix tree 维护**：SGLang 的 continuous batching 机制持续调度新请求进入 batch，已完成的请求移出。当 batch 成员变化（请求加入/完成），radix tree 结构更新，触发 FastTree runtime 重新搜索。
  3. **FastTree runtime**：读取当前 radix tree → BFS greedy heuristic 做 binary edge assignment → 生成 virtual tree → node-centric query aggregation → 输出 (context, {queries}) grouping plan。开销 < 1ms（被 decoding 循环摊销）。
  4. **Attention kernel 替换**：SGLang 原按 query 分别调用 FlashInfer attention kernel（每个 query 单独 load KV cache from HBM，GEMV 计算）。FastTree 替换为：按 grouping plan 将共享同一 context prefix 的 queries 聚合 → 单 kernel 处理 → Q 矩阵 tile 在 shared memory 中复用 KV tile → tensor core GEMM 替代 CUDA core GEMV。
  5. **Decoding loop**：SGLang 连续执行多步 decoding（amortize scheduling overhead）。每步的 attention 计算被 FastTree 加速（平均 1.9× over FlashInfer on Llama）。
  6. **输出**：生成的 tokens 返回客户端。Throughput = total output tokens / total time（含 scheduling + prefill + decode + communication）。FastTree 相比 SGLang-FlashInfer throughput 提升 up to 2.2×。

  - **作用**：弥补 SGLang 在 radix tree 内存层面的优化与其 computation 层面仍然执行 per-query 分离计算的 gap。具体而言，SGLang 的 radix tree 减少了 KV cache 内存占用（更多请求可同时服务），但 attention 计算仍重复加载共享 KV cache 且无法利用 tensor core（decode 阶段 GEMV）。FastTree 在 scheduling→computation 交界处优化，使内存布局（tree）直接指导计算聚合（grouping），实现 memory-aware computation optimization。

## EPD-Serve A Flexible Multimodal EPD Disaggregation Inference Serving System On Ascend

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：EPD-Serve 是一个支持 Encode-Prefill-Decode 三阶段解耦的多模态推理 Serving 系统，部署于华为 Ascend NPU。核心实现包括：(i) E-P 异步特征预取机制——基于 Mooncake Store 构建共享多模态缓存池（MM Store），通过事件驱动的 hash 通知实现跨节点特征传输与 Encode 计算重叠；(ii) P-D 分层分组 KV cache 传输机制——按 Transformer 层将 KV cache 分组打包，延迟调度以对齐通信与 Prefill 计算，最大化通信-计算重叠（overlap ratio 从 baseline 15.27% 提升至 98.78% at seq_len=1024）；(iii) 模态感知多路径调度——根据请求模态（纯文本 vs 多模态）路由到不同管道（P-D vs E-P-D），配合实例级最少负载优先的全局调度策略；(iv) 灵活阶段解耦与物理共置——支持 E-P-D、EP-D、ED-P、E-PD、(E-P)-D、(E-D)-P、(E-PD) 等部署拓扑，逻辑隔离 + NPU 空间复用实现算子级并行（MatMul 用 AI Core、AllReduce 用 AI Vector 互补执行）。
  - 实验比较：(i) EPD-Serve 多种部署拓扑（E-PD、(E-PD)、(E-P)-D、(E-D)-P、EP-D、E-P-D、TP1、TP2）在吞吐、TTFT、TPOT、SLO 达成率上的全面对比；(ii) 传输优化消融实验——对比 Baseline(E-P-D) vs +E-P 异步预取 vs +P-D 分层分组 vs 两者全开；(iii) Encode 解耦效益分析——(E-PD) vs TP1/TP2/E-PD；(iv) Decode 解耦效益分析——EP-D/(E-P)-D/(E-D)-P vs TP1/TP2；(v) 全解耦效益分析——E-P-D vs (E-P)-D vs (E-D)-P vs EP-D at 10 req/s 高负载。

- 硬件平台是什么，配置是什么。
  - 华为 Ascend Atlas 800I A2 服务器，每 NPU 64 GB 片上内存。单机多 NPU 环境。所有对比实验在相同硬件配置下进行以保证公平性。

- 开源Serving框架是什么。修改了什么。
  - Baseline 框架：vLLM v0.11.0（默认 monolithic 架构——Encode/Prefill/Decode 串行在同一计算资源上执行）。
  - EPD-Serve 修改：(i) 将 E/P/D 拆分为独立实例进程，支持独立调度和弹性伸缩；(ii) 引入统一 Proxy 组件执行跨实例请求路由和负载均衡；(iii) 基于 Mooncake Store（参考 Mooncake [12]）实现 MM Store 多模态缓存池；(iv) 实现 E-P 异步特征预取和 P-D 分层分组 KV 传输模块；(v) 增加模态感知路由逻辑；(vi) 支持多种 flexible deployment topology 的物理共置。底层 PyTorch/Ascend 计算框架未修改。
  - 论文未声明 EPD-Serve 开源，未提供代码仓库 URL。关联工作 Mooncake Store 开源在 https://github.com/kvcache-ai/Mooncake。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 开源情况：论文未声明开源（2026年5月检索未找到公开代码仓库）。底层传输组件 Mooncake Store 关联开源。
  - 使用例子（openPangu-7B-VL on ShareGPT-4o, (E-P)-D 部署，2 NPU，10 req/s）：
    1. **输入**：API Server 接收请求，判断输入是否含图像/音频/视频模态。多模态请求路由到 E-P-D 完整管道，纯文本请求路由到 P-D 管道。AISBench 控制请求注入速率 10 req/s。
    2. **Encode + Prefill 阶段**（共置 NPU 1 (E-P)）：Vision Encoder (ViT 0.7B) 编码图像 I_m → 特征 V_m ∈ R^{n×3584}。完成后事件驱动异步发送特征 hash（非完整 tensor）到 Prefill 实例的 listener。Prefill listener 从 MM Store（hash→feature vector）检索并写入本地缓存 → 若 miss 则本地重算（fault-tolerant）。文本提示 I_t 编码为 V_t → 拼接 V_m+V_t 输入 LLM (7B) → 逐层 Prefill 计算 KVCache。
    3. **P-D 传输**：当 Prefill 开始计算 L+1 层时，L 层的 KVCache 异步传输至 Decode 实例。多层 KV 分组打包减少握手次数，延迟调度避免通信峰值。通过分层分组，KV 传输 overlap ratio 从 baseline 15.27% 提升至 98.78%（seq_len=1024）。
    4. **Decode 阶段**（独立 NPU 2 (D)）：接收分层到达的 KVCache，按自回归逐 token 生成 O_i+1。独立 Decode NPU 不受 Encode/Prefill 资源竞争影响，稳定低 TPOT。输出固定 64 tokens 或至 <eos>。
    5. **物理共置空间复用**：NPU 1 上 Encode 与 Prefill 共享 AI Core/AI Vector——当一个阶段等待通信时另一阶段利用空闲计算单元。MatMul（AI Core compute-heavy）与 AllReduce（AI Vector communication-heavy）交替执行实现算子级并行。
    6. **输出**：Proxy 收集各实例结果返回客户端。SLO 约束：TTFT ≤ 2000ms, TPOT ≤ 50ms（Decode-disaggregated 时）。Per-NPU effective throughput = 77.36 tokens/s（(E-P)-D at 10 req/s），SLO attainment rate = 26.17%。
  - **作用**：在 Ascend NPU 上实现多模态推理三阶段灵活解耦与物理共置优化。(E-P)-D 在 12 req/s 高并发下比 PD-disaggregated EP-D 提升吞吐 57.37-69.48%；全解耦 E-P-D (3 NPU) 在 10 req/s 下 SLO 达成率 94.34%；单 NPU 共置 (E-PD) 比 monolithic TP1 提升吞吐 12.87-14.88%、降低 TTFT 2.7-3.25%、降低 TPOT 69.58-70.39%。

## Efficient and Adaptable Overlapping for Computation and Communication via Signaling and Reordering

> **近似层次匹配说明**：本文核心创新在 kernel 级的 signaling + reordering 重叠设计，但端到端评估通过修改 vLLM/Megatron-LM/xDiT 框架将 FlashOverlap 集成到多 GPU Serving 系统中，以 throughput 为指标评测端到端性能提升，属于 Serving 调度层的集成与优化。

- 属于Serving调度的实现是什么？实验比较什么？
  FlashOverlap 通过替换原始 linear layer 和后续通信原语（AllReduce、ReduceScatter、All-to-All）为带 overlap 的 FlashOverlap 实现，集成到 vLLM（LLM 推理）、Megatron-LM（LLM 训练）、xDiT（text-to-video 生成）三个主流 Serving/训练框架中。实验比较集成 FlashOverlap 前后的端到端 throughput，以及与 decomposition-based（Async-TP）和 fusion-based（FLUX）方法的 throughput 差异。

- 硬件平台是什么，配置是什么。
  NVIDIA A800 GPU（NVLink pairwise 连接，1935GB/s HBM 带宽，312 TFLOPS FP16），用于所有端到端评估。软件环境：CUDA 12.1、NCCL 2.19.3、PyTorch 2.5.1、CUTLASS 3.6.0。

- 开源Serving框架是什么。修改了什么。
  - **vLLM**（LLM 推理）：替换 Llama3-70B TP=8 配置中 attention 和 FFN 后的 GEMM+AllReduce 对为 FlashOverlap 实现。chunk_size=16384。不修改请求调度逻辑（continuous batching 保留）。
  - **Megatron-LM**（LLM 训练）：替换 Llama3-70B TP=8 和 Mixtral-8x7B EP=4,TP=2 配置中的 GEMM+ReduceScatter 和 GEMM+All-to-All 对为 FlashOverlap 实现。input_token 分别为 16384 和 32768。层数设 8 和 4 以适配单节点内存。
  - **xDiT**（text-to-video 生成）：替换 Step-Video-T2V TP=4 配置中的 GEMM+AllReduce 为 FlashOverlap 实现。input_token=33792。
  所有框架的修改均为：定位到 linear layer 输出后的通信原语调用 → 替换为 FlashOverlap 的 GEMM-with-signaling + NCCL 通信双 stream 实现。框架的其余调度逻辑（batching、pipeline、memory management）保持不变。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  已开源：https://github.com/infinigence/FlashOverlap（ae 分支），Zenodo DOI: 10.5281/zenodo.17201530。

  **端到端 Serving 集成全过程**（以 Llama3-70B TP=8 LLM 推理 on vLLM + 8×A800 GPUs 为例）：
  1. **输入**：用户请求到达 vLLM API server。vLLM continuous batching 调度器将请求按可用 KV cache slot 批量组成 batch → tokenize → 转换为 input tensor。
  2. **模型前向传播**：vLLM 按 TP=8 分区执行 Llama3-70B 各 transformer layer。每个 layer 的 attention projection 和 FFN 输出均为 GEMM 在 8 个 GPU 上的部分结果（每 GPU 持有 M/8 行）。
  3. **FlashOverlap 执行的 GEMM+AllReduce**（替代原 sequential 路径）：
     a. **GEMM kernel (Stream A)**：CUTLASS GEMM 执行 M×N×K 矩阵乘法，main loop 完整不变。tile 按 wave pattern 顺序完成，epilogue 中 pre-communication reordering 将完成 tile 的数据散射到连续地址通信 buffer，同时 atomicAdd 更新 counting table。
     b. **Signaling kernel (Stream B)**：周期性查询 counting table，当某 wave group 的所有 tile 完成时，调用 `ncclAllReduce(sendbuf, recvbuf, data_size, ncclFloat16, ncclSum, comm, stream_B)` 对该 group 的数据执行 AllReduce。
     c. **Overlap 并发**：Stream A 中后续 wave group 的 GEMM 计算与 Stream B 中前一个 wave group 的 NVLink AllReduce 通信并发执行。
  4. **Post-communication reordering**：fused 到后续 RMSNorm kernel 中，根据 mapping table 将通信后的数据恢复原始顺序。
  5. **其余层正常执行**：FlashOverlap 不修改 attention、embedding、softmax 等其他算子。
  6. **输出**：vLLM 完成所有层后 decode 出 token → 逐 token 生成直到 <eos> 或 max_length → 返回响应。

  **作用**：在保持现有 Serving 框架调度逻辑不变的前提下，通过替换 GEMM+通信对的实现降低通信瓶颈。端到端 speedup：LLM 推理（Llama3-70B/vLLM）1.05×，LLM 训练（Llama3-70B/Megatron-LM）1.08-1.13×，LLM 训练（Mixtral-8x7B/Megatron-LM）1.05-1.05×，T2V 生成（Step-Video-T2V/xDiT）1.11-1.12×。T2V 因大 input token 数（33792）通信占比较高、加速最大。FlashOverlap 通过 interference-free computation 保证不退化：即使 overlap 效果有限的 case 也不会比 non-overlap baseline 更差。