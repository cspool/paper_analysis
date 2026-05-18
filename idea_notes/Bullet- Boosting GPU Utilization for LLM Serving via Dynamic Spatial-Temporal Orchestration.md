- baseline方法是什么？
  Baseline方法分为三类：(1) **Chunked prefill** (vLLM v0.8.5, SGLang v0.4.6)：用固定token budget将prefill chunk与decode token绑在同一个lock-step batch中，所有SM统一执行融合后的prefill+decode kernel。全栈执行例子：算法层标准transformer→Serving框架在SGLang中通过Flashinfer实现prefill/decode attention融合→GPU所有108 SM统一执行单batch→KV cache以PagedAttention管理。chunked prefill的缺陷：prefill compute efficiency仅70%-76%（wave quantization与attention bottleneck）；长prompt按N(N+1)/2次KV reload（16k token prefill以1k chunk的总prefill latency比unchunked高1.13x）；固定token budget难以同时兼顾TTFT和TPOT。(2) **Static overlap** (NanoFlow 1024 chunk)：在chunked prefill基础上做静态kernel overlap pipeline，固定overlap调度，无法适应动态workload。(3) **Disaggregated serving** (SGLang+Mooncake xP-yD)：prefill和decode放到不同GPU/节点，需要跨实例KV/state迁移和手动资源调参，存在KV cache pool减半导致的hit rate下降、GPU静态分配无法适应负载波动等问题。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**intra-GPU prefill-decode spatial-temporal orchestration**：在同一GPU内将prefill和decode拆成可并发执行的两个engine，通过layer-wise prefill、step-wise decode、动态SM partition和SLO-aware scheduling在满足TTFT/TPOT约束下提升GPU利用率。四个组件形成闭环：
  
  (1) **Performance estimator (SRM)**：用SM-scaling roofline model建模compute/memory/network bandwidth随SM数量变化的性能上界，少量offline concurrent sample校准prefill/decode interference，在线统计持续修正。解决了纯profiling开销大的问题（profiling overhead低于1小时，预测/更新开销微秒级）。
  
  (2) **SLO-aware task scheduler**：周期性读取全局状态，预测TTFT/TPOT，重排waiting requests，为每批prefill layer或decode step搜索合适SM分区。prefill队列堆积时提升prefill SM快速清队列，decode TPOT接近SLO边界时减少prefill SM或增加decode SM。解决了chunked prefill的固定token budget dilemma和disaggregated serving的静态资源分配问题。
  
  (3) **Resource manager**：通过libsmctrl_set_stream_mask修改CUDA stream metadata实现微秒级SM重分区（平均4.1us）。解决了已有系统无法快速适应动态workload的问题。
  
  (4) **Concurrent execution engine**：prefill和decode放在独立进程/worker中，共享CPU metadata buffer、统一GPU memory pool (CUDA IPC)、ZeroMQ metadata传递。GPU memory预先分配模型权重和KV cache，CUDA IPC共享避免KV copy。prefill以layer粒度执行，decode以CUDA Graph step执行。避免了chunked prefill的锁步等待和disaggregated serving的跨实例KV迁移开销。
  
  全栈执行例子：算法层标准transformer attention+FFN，论文未修改模型算法→Serving框架层基于SGLang v0.4.6+PyTorch 2.6.0，约4100行Python，prefill/decode拆为两个SGLang worker，ZeroMQ异步传递metadata→编译框架层未修改编译框架，但利用CUDA Graph优化decode iteration（单graph launch），prefill使用piecewise layer-wise执行→kernel调度层通过libsmctrl将prefill CUDA stream绑定到SM mask A（如70个SM），decode CUDA stream绑定到SM mask B（如38个SM），scheduler周期下发repartition命令更新mask→硬件架构层使用NVIDIA A100/H100/H20 GPU，利用MPS spatial sharing，SM mask以16 SM为粒度分组（A100 6种配置，H100 7种配置）。
