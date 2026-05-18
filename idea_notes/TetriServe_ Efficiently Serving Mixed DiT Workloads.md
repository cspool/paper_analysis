## TetriServe: Efficiently Serving Mixed DiT Workloads

- baseline方法是什么？
  Baseline方法是**固定degree的sequence parallelism (SP)**，即xDiT对所有请求使用统一的SP度（SP=1/2/4/8），请求一旦以固定并行度开始执行便不可抢占，持有分配的GPU直到完成所有denoising steps。全栈执行例子：算法层DiT模型（FLUX.1-dev 12B参数）以Ulysses attention实现token序列跨GPU分布→系统框架层xDiT固定SP度分发请求，无deadline感知→编译框架层论文未明确说明→kernel调度层NCCL all-to-all collective通信，小分辨率下通信占比超30%导致scaling效率差→硬件架构层8×H100 NVLink 4.0。固定SP的缺陷：低SP度（SP=1/2）对大分辨率（2048×2048）处理太慢导致超时；高SP度（SP=4/8）对小分辨率（256×256）通信开销过大导致head-of-line blocking；无deadline感知导致高SLO violation；RSSP (Resolution-Specific SP)虽然每种分辨率选最优固定SP，但缺乏deadline感知和运行时adaptation，SAR仍低。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**step-level sequence parallelism + round-based deadline-aware scheduling**：在每个denoising step级别动态调整SP度，通过round离散化时间将NP-hard的连续时间调度转化为可解的per-round DP packing问题。全栈执行例子：算法层相同DiT模型→系统框架层TetriServe Scheduler在每个固定时长round内：(a) offline profiled cost model查找每种(分辨率, GPU数)的单步耗时，(b) 为每个请求计算满足deadline的最小GPU分配，(c) DP pack requests入round最大化surviving请求数，(d) GPU Placement Preservation保持同请求跨round同GPU集合，(e) Elastic Scale-Up利用空闲GPU为有余量steps的请求加速→编译框架层论文未明确说明→kernel调度层selective continuous batching仅对同小分辨率请求合并steps减少kernel launch overhead，NCCL process groups预warmup策略平衡启动延迟和显存占用→硬件架构层8×H100 NVLink 4.0 / 4×A40 PCIe 4.0。

  对应解决：(1) **固定SP对小分辨率效率差** → step-level动态SP，小分辨率请求只用1 GPU避免通信开销（通信占比从>30%降至最低）；(2) **固定SP对大分辨率不足** → deadline感知为紧急大分辨率请求临时scale-up到更多GPU加速；(3) **无deadline感知** → round-based DP调度显式建模每个请求的deadline和剩余steps，优先调度即将超时的请求；(4) **不可抢占导致head-of-line blocking** → round边界自然提供preemption点，调度器可在round间重新分配GPU；(5) **RSSP缺乏运行时adaptation** → TetriServe在运行时动态感知deadline紧迫度调整并行度，在tight SLO 1.0× Uniform workload下SAR比RSSP高0.10（0.42 vs 0.32）。

- baseline方法是什么？
  Baseline方法分为三类：(1) **Chunked prefill** (vLLM v0.8.5, SGLang v0.4.6)：用固定token budget将prefill chunk与decode token绑在同一个lock-step batch中，所有SM统一执行融合后的prefill+decode kernel。全栈执行例子：算法层标准transformer→Serving框架在SGLang中通过Flashinfer实现prefill/decode attention融合→GPU所有108 SM统一执行单batch→KV cache以PagedAttention管理。chunked prefill的缺陷：prefill compute efficiency仅70%-76%（wave quantization与attention bottleneck）；长prompt按N(N+1)/2次KV reload（16k token prefill以1k chunk的总prefill latency比unchunked高1.13x）；固定token budget难以同时兼顾TTFT和TPOT。(2) **Static overlap** (NanoFlow 1024 chunk)：在chunked prefill基础上做静态kernel overlap pipeline，固定overlap调度，无法适应动态workload。(3) **Disaggregated serving** (SGLang+Mooncake xP-yD)：prefill和decode放到不同GPU/节点，需要跨实例KV/state迁移和手动资源调参，存在KV cache pool减半导致的hit rate下降、GPU静态分配无法适应负载波动等问题。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**intra-GPU prefill-decode spatial-temporal orchestration**：在同一GPU内将prefill和decode拆成可并发执行的两个engine，通过layer-wise prefill、step-wise decode、动态SM partition和SLO-aware scheduling在满足TTFT/TPOT约束下提升GPU利用率。四个组件形成闭环：
  
  (1) **Performance estimator (SRM)**：用SM-scaling roofline model建模compute/memory/network bandwidth随SM数量变化的性能上界，少量offline concurrent sample校准prefill/decode interference，在线统计持续修正。解决了纯profiling开销大的问题（profiling overhead低于1小时，预测/更新开销微秒级）。
  
  (2) **SLO-aware task scheduler**：周期性读取全局状态，预测TTFT/TPOT，重排waiting requests，为每批prefill layer或decode step搜索合适SM分区。prefill队列堆积时提升prefill SM快速清队列，decode TPOT接近SLO边界时减少prefill SM或增加decode SM。解决了chunked prefill的固定token budget dilemma和disaggregated serving的静态资源分配问题。
  
  (3) **Resource manager**：通过libsmctrl_set_stream_mask修改CUDA stream metadata实现微秒级SM重分区（平均4.1us）。解决了已有系统无法快速适应动态workload的问题。
  
  (4) **Concurrent execution engine**：prefill和decode放在独立进程/worker中，共享CPU metadata buffer、统一GPU memory pool (CUDA IPC)、ZeroMQ metadata传递。GPU memory预先分配模型权重和KV cache，CUDA IPC共享避免KV copy。prefill以layer粒度执行，decode以CUDA Graph step执行。避免了chunked prefill的锁步等待和disaggregated serving的跨实例KV迁移开销。
  
  全栈执行例子：算法层标准transformer attention+FFN，论文未修改模型算法→Serving框架层基于SGLang v0.4.6+PyTorch 2.6.0，约4100行Python，prefill/decode拆为两个SGLang worker，ZeroMQ异步传递metadata→编译框架层未修改编译框架，但利用CUDA Graph优化decode iteration（单graph launch），prefill使用piecewise layer-wise执行→kernel调度层通过libsmctrl将prefill CUDA stream绑定到SM mask A（如70个SM），decode CUDA stream绑定到SM mask B（如38个SM），scheduler周期下发repartition命令更新mask→硬件架构层使用NVIDIA A100/H100/H20 GPU，利用MPS spatial sharing，SM mask以16 SM为粒度分组（A100 6种配置，H100 7种配置）。
