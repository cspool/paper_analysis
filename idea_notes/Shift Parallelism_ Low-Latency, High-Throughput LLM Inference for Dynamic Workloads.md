## Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads

- baseline方法是什么？
  Baseline方法分为三类：(1) **Tensor Parallelism (TP)**：将每层权重和计算切分到多GPU，column-parallel QKV/O无需通信，row-parallel linear需要all-reduce同步。全栈执行例子：算法层标准transformer attention+FFN→Serving框架层vLLM内置TP，Megatron-style column/row partitioning→kernel调度层NCCL all-reduce同步各GPU partial results→硬件架构层8×H200 NVSwitch互联。TP适合降低单请求延迟（TTFT/TPOT），但随着TP degree增加，communication-to-compute ratio（O(n)通信开销，n为sequence length）上升，高流量下combined throughput显著下降（Llama-70B场景TP的throughput为24.7k tok/s，比DP的45.9k tok/s低46%）。(2) **Data Parallelism (DP)**：跨请求复制完整模型，无GPU间通信，吞吐高（45.9k tok/s），但不能并行化单请求，interactive低并发场景TTFT和TPOT较差（TTFT 614ms vs Shift Parallelism 102ms，TPOT 22.5ms vs 10.1ms）。全栈执行例子：算法层标准transformer→Serving框架层vLLM多worker各持完整模型副本→kernel调度层无跨GPU通信→硬件架构层各GPU独立执行不同请求。(3) **静态双部署（TP节点 + DP节点）**：分别部署TP节点处理交互请求、DP节点处理批量请求，路由到对应节点。缺陷是加倍部署成本和系统复杂度，且TP与DP的KV cache memory layout不兼容，无法在同一请求生命周期内切换。(4) **原始Ulysses SP (training版本)**：来自训练的SP将sequence切分到多GPU并行prefill，无all-reduce通信（仅all-to-all），有接近DP的吞吐。但缺乏inference关键特性：不支持GQA（当KV head数小于GPU数时无法自然扩展）、小batch decode时因sequence partition不均导致load imbalance（batch size=9, SP=8时效率仅50%）、不能并行decode step导致TPOT变差。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**Shift Parallelism**：基于KV cache invariance（SP与TP共享相同attention head layout），在同一vLLM部署内保留base configuration（SP或mixed SP+TP）和shift configuration（full TP），运行时按batch token阈值动态切换。

  核心设计及其与Baseline缺陷的映射：
  
  **(1) SP for Inference通用化**：为SP添加GQA支持（通过fused all-to-all中KV cache replication处理Q head数与KV head数不匹配）、small batch load balancing（padding到SP degree倍数避免sparse communication）、任意(SP, TP)组合forward pass（Algorithm 1）。解决了原始SP无法覆盖Llama/Qwen等GQA模型、小batch decode效率低的问题。相对TP baseline，SP避免了all-reduce通信，在large batch prefill时TTFT更低（102ms vs TP 159ms for Llama-70B），combined throughput更高（37.2k vs 24.7k tok/s）。
  
  **(2) KV Cache Invariance与Head Ordering对齐**：约束base config与shift config使用相同attention head layout和ordering。对于任意(SP, TP)组合，通过SP_TP group确保shift model按base config的SP group order加载权重，维持KV cache coherence。这是Shift Parallelism的核心insight——解决了TP与DP因KV cache memory layout不兼容而无法动态切换的根本问题。请求在SP和TP之间切换时无需搬迁KV cache，切换成本极低（仅runtime CUDA graph选择）。
  
  **(3) 双配置动态切换机制**：Base configuration使用SP或(SP, TP)处理大batch（优化TTFT和throughput），shift configuration固定为(SP=1, TP=P)的full TP处理小batch（优化TPOT）。算法极简：batch token数 > shift threshold则选base，否则选shift（Algorithm 2）。解决了单并行策略在不同流量下的latency-throughput tradeoff：大batch时SP避免TP的all-reduce吞吐损失；小batch时TP避免SP的padding/load imbalance导致的TPOT恶化（SP TPOT 32.5ms vs TP 9.34ms vs Shift Parallelism 10.1ms）。
  
  **(4) 双模型权重加载**：separate models方式同时加载base model和shift model两套权重，共享同一KV cache。Shift model额外内存开销约1/SP（SP=8时为12.5%），替代方案on-the-fly weight slicing受Hopper FP8 tensor core限制需要矩阵转置而性能更差。相对TP+DP双部署，Shift Parallelism不需要复制整套节点，仅在同一部署内多加载约12.5%权重即可覆盖两种并行策略。
  
  **(5) vLLM插件集成**：通过ArcticInference插件系统编译并capture base和shift两套CUDA graphs，初始化时注册，运行时按threshold选择replay，无需修改vLLM核心代码。

  论文方法全栈执行例子（以8×H200，base=(SP=4, TP=2)，shift=(SP=1, TP=8)为例）：
  - 算法层：标准transformer架构（attention+MLP），论文未修改模型算法。SP沿sequence维度并行prefill，attention经all-to-all切换到head parallel layout。
  - Serving框架层：vLLM v0.9.2 continuous batching + ArcticInference插件。请求入队→scheduler检查当前batch token数→大于threshold选base model（Algorithm 1[SP,TP]），SP slice输入[seq/SP, d]→QKV projection [seq/SP, 3×h/TP]→SP all-to-all→attention [seq, h/(SP×TP)]→SP all-to-all→O projection→TP all-reduce→MLP→TP all-reduce→SP all-gather输出；小于等于threshold选shift model（Algorithm 1[1, SP×TP]），full TP并行整个batch。
  - 编译框架层：vLLM CUDA graph capture机制，base和shift各capture数百个graph（不同batch shape），初始化时注册，运行时replay。论文未修改编译框架本身。
  - kernel调度层：SP路径使用fused all-to-all（Q/K/V通信融合为单次collective），base config降低communication-to-compute ratio（相对TP的all-reduce），shift config用小batch full TP避免SP load imbalance。KV cache在base和shift之间共享不搬移。
  - 硬件架构层：8×H200 GPU，NVSwitch 900GB/s互联，使用FP8 tensor core（1,979 TFLOPS peak）。论文未修改硬件。SP的all-to-all通信量不随SP degree增长（Table 2），TP的all-reduce通信量与n（sequence length）成正比，使SP在大batch时比TP更高效利用NVSwitch带宽。
