## TokenFlow: Responsive LLM Text Streaming Serving under Request Burst via Preemptive Scheduling

- 属于Serving调度的实现是什么？实验比较什么？
  论文实现TokenFlow，一个面向实时LLM text streaming场景的buffer-aware preemptive scheduling serving系统。核心Serving调度实现包含：(1) Buffer-aware Request Scheduler：两步调度——第一步determine working set（根据GPU显存、请求KV footprint、等待队列长度、I/O队列长度和buffer安全条件决定可过量承诺的请求集合），第二步buffer balancing（在working set内按buffer size、weighted token generation quantity、required output rate和内存约束做greedy selection，再用局部交换优化优先级）。请求buffer越低、用户消费速率越高，越容易获得运行优先级；buffer较高的请求可被暂时抢占，继续由客户端buffer平滑输出；(2) Request Tracker：持续记录每个请求的buffer token数、生成时间戳、消费速率和资源占用；(3) Request Offload Manager：在scheduler决定抢占时将请求从running set移出，协调KV cache offload；(4) Hierarchical KV Cache Manager：使用并行CUDA streams（compute/load/evict）和Python multithreading，通过write-through KV cache、synchronous chunked writing和load-evict overlap降低抢占恢复开销。系统约3000行Python代码，基于SGLang实现。实验比较：(a) micro benchmark：burst/Poisson request distributions下的TTFT（mean/P99）、raw throughput、effective throughput（buffer加权：buffer<10%输出长度全计，10%-20%线性衰减，>20%不计）；(b) real trace benchmark：真实生产LLM service trace下的端到端性能；(c) 消融实验：w/o offload、w/o write-through、w/o evict-load overlap的影响。对比baseline：SGLang conservative scheduling（FCFS/prefill优先）、SGLang chunked prefill、Andes（QoE-aware scheduling with Token Pacer）。

- 硬件平台是什么，配置是什么。
  三套GPU配置：(1) NVIDIA H200（mem-frac=0.3）；(2) NVIDIA RTX 4090；(3) NVIDIA A6000。micro experiment中也报告Huawei Ascend 910B支持。H200和RTX 4090上进行受控实验，输入输出长度按短序列和长序列正态分布配置，H200输出长度相对RTX 4090放大2倍。

- 开源Serving框架是什么。修改了什么。
  基于SGLang实现，替换默认scheduler。修改包括：(1) 新增priority-based scheduler（buffer-aware两步调度）；(2) 新增Request Tracker模块（实时追踪buffer token数、生成速率、消费速率、资源占用）；(3) 新增Request Offload Manager（协调请求抢占和恢复时的状态转移）；(4) 新增Hierarchical KV Cache Manager（write-through strategy + 三并行CUDA stream pipeline：compute stream、load stream、evict stream + 动态chunk sizing + batched transfer + CUDA events协调非阻塞执行）。论文未开源代码仓库。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  未找到明确开源仓库（arXiv页面https://arxiv.org/abs/2510.02758未提供代码链接，EuroSys 2026 DOI: https://doi.org/10.1145/3767295.3769328）。系统原型基于SGLang实现，约3000行Python代码。以TokenFlow在H200上serving Llama3-8B为例：
  1. 部署：启动TokenFlow serving instance（SGLang + TokenFlow scheduler + Hierarchical KV Cache Manager），GPU显存配置mem-frac=0.3用于KV cache working set。
  2. 请求到达：R1、R2先到达，用户消费速率分别为20 tokens/s和30 tokens/s。LLM Executor为R1/R2生成token并写入客户端buffer，Request Tracker持续记录buffer token数。
  3. Burst到达：R3到达时触发burst。Scheduler检查R1/R2的实时buffer状态——若buffer不足以覆盖evict+load+schedule切换时间（buffer safety condition），暂不抢占。
  4. 安全抢占：当R1因消费速率较低积累足够buffer后（如buffer > 切换延迟×消费速率），scheduler判定R1可安全preempt。Request Offload Manager将R1移出running set。KV Cache Manager依赖后台write-through已同步的大部分KV cache，将剩余chunk快速evict到CPU memory，同时load stream将R3状态加载到GPU。Executor开始为R3生成首token→降低R3 TTFT。
  5. 恢复执行：当R1 client buffer接近耗尽（降至安全阈值），scheduler决定resume R1。Load stream从CPU memory加载R1的KV cache chunk回GPU，R1恢复decode。
  6. 效果：burst场景下P99 TTFT最多降低80.2%，mean TTFT最多降低48.4%，effective throughput最多提升52.9%。Poisson场景下effective throughput最多提升82.5%，TTFT最多降低53.7%。端到端真实trace实验中，mean TTFT平均降低52.6%，A6000和H200上effective throughput分别提升45.1%和37.1%。

