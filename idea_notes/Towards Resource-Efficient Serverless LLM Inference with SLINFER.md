- baseline方法是什么？
  Baseline是ServerlessLLM (sllm, OSDI'24)，一个面向LLM serverless部署的系统。sllm的核心设计：(1) 每GPU独占分配给单个model instance，GPU资源不可共享；(2) event-driven分配：请求到达时若无运行中instance则在空闲GPU上启动新instance（cold-start经过fast model loading优化到~1s），否则排队等待；(3) 使用vLLM作为底层inference engine（paged-attention KV-cache管理+continuous batching），但vLLM默认将全部GPU memory分配给单一instance。

  全栈执行例子（以serverless场景下serving 64个7B LLM请求为例）：
  - 算法层：standard transformer autoregressive decoding，Llama-2-7B FP16，prefill+decode两阶段
  - 系统框架层：ServerlessLLM分配一个7B instance独占A100-80GB GPU → vLLM continuous batching处理该instance的in-flight requests → KV-cache静态预分配整个GPU memory → 其他model排队等待GPU释放。即使该instance仅用23% GPU memory（图5），剩余~60GB闲置。
  - 编译框架层：论文未明确说明（vLLM使用PyTorch CUDA backend，无编译框架自动生成）
  - kernel调度层：vLLM默认scheduler按batch内request到达时间执行continuous batching，无token-level精细化调度。GPU kernel为standard FlashAttention+GEMM，CPU核心仅用1 core（图10），其余31 core闲置。
  - 硬件架构层：NVIDIA A100-80GB GPU + Intel Xeon CPU (AMX idle)。GPU独占导致大量GPU memory over-provisioning，CPU matrix accelerator完全闲置。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出SLINFER，通过三个核心机制解决ServerlessLLM的三大缺陷：

  **缺陷1：GPU独占+over-provisioning → serving capacity低**
  → **SLINFER方案**：异构硬件抽象+弹性资源共享。将CPU/GPU统一为resource pool，instance不再独占整节点。CPU通过OpenVINO+AMX独立serve ≤13B LLM（第4代Xeon TTFT 567ms for 7B-1K），GPU上多model共享memory。实验证明64个7B model下SLINFER仅用0.9 GPU vs sllm的3.2 GPU。

  **缺陷2：无token-level compute调度 → SLO violation无法精细化预防**
  → **SLINFER方案**：Headroom-Driven Compute Subsystem。每scheduling cycle选最短headroom instance执行一个iteration，shadow validation在添加请求前虚拟模拟future compute探索三种SLO violation case（新请求prefill超时、现有请求被delay、aggregate decode超TPOT SLO）。performance quantification用linear/2D interpolation profiling，estimator偏差仅5.9%(TTFT)/3.9%(TPOT)。

  **缺陷3：静态内存+无协调 → OOM风险和fragmentation**
  → **SLINFER方案**：Hazard-Aware Memory Subsystem (watermark w=25% early scale-up + lazy scale-down, optimistic budgeting + pessimistic reservation station协调并发memory操作避免OOM) + Efficiency-Oriented Consolidator (proactive preemption让大batch instance抢占小邻居来scale-up, reactive bin-packing优先路由到大batch instance加速碎片回收)。

  论文方法全栈执行例子（以serving 64个7B LLM，一个Llama-2-7B请求到达为例）：
  - 算法层：standard transformer Llama-2-7B FP16 autoregressive decoding，同baseline
  - 系统框架层（核心创新）：SLINFER proxy收到请求→优先尝试CPU instance（通过OpenVINO backend）→compute subsystem对候选instance执行shadow validation：(a) 线性interpolation估计新请求prefill time，(b) 虚拟添加后仿真所有in-flight request headroom，(c) 检查三种SLO violation case均不发生→通过验证。memory subsystem检查node可用memory是否够容纳新请求的KV-cache（Mrequire=C·Σ(Ir+max(Or,Ō))），若需scale-up则检查optimistic budget→若不足则尝试compromise降级为Mrequire→若仍不足则evict最长headroom请求。请求加入后token-level调度器按headroom轮转instance执行iteration：每cycle选最短headroom instance执行一次decode/prefill→更新headroom→重复。
  - 编译框架层：论文未明确说明（vLLM PyTorch CUDA backend + OpenVINO CPU backend，无编译框架修改）
  - kernel调度层：SLINFER的compute subsystem替代vLLM默认scheduler，实现token-level跨instance scheduling。CPU instance用OpenVINO后端（AMX-accelerated matmul），GPU instance用vLLM CUDA backend（FlashAttention+GEMM）。对比baseline每instance独占GPU、batch size较小且CPU闲置，SLINFER实现instance sharing使average batch size提升74%→sub-linear compute growth特性带来更高吞吐。
  - 硬件架构层：NVIDIA A100-80GB GPU + Intel 4th Gen Xeon (AMX)。SLINFER充分发挥CPU的AMX matrix accelerator（7B 1K-input TPOT仅71ms vs SLO 250ms），CPU可独立serve ≤13B model短输入请求。GPU memory utilization接近1.0（vs sllm的three-tier阶梯分布），KV-cache scaling watermark机制使scaling overhead仅1.4%。

  Baseline缺陷→SLINFER方案映射：
  | Baseline缺陷 | SLINFER方案 | 效果 |
  |-------------|-----------|------|
  | GPU独占每instance仅用23% memory → 大量over-provisioning | 异构硬件抽象+弹性资源共享（CPU独立serve+GPU多model共享） | 64×7B: 0.9 GPU vs 3.2 GPU (sllm)，serving capacity +86-154% |
  | 无token-level调度 → SLO violation不可控 | Headroom-driven token-level scheduling + shadow validation | 128 models下SLO-met rate显著高于baseline，TTFT sub-second CDF |
  | 静态KV-cache分配 → memory resizing overhead大且无协调 | Watermark-based scaling (w=25%) + optimistic budget/pessimistic reservation | Scaling overhead从11.3%降至1.4%，无OOM |
  | 碎片化instance → 重复weight loading + 小batch | Proactive preemption + reactive bin-packing consolidation | Batch size +74% vs sllm, decode throughput +0-88% on GPU |
  | CPU完全闲置 → 浪费AMX加速能力 | OpenVINO + AMX backend, CPU优先调度, SLO不满足时fallback GPU | CPU可独立serve 7B/13B，3-4 CPU node ≈ 1 GPU node serving capacity |