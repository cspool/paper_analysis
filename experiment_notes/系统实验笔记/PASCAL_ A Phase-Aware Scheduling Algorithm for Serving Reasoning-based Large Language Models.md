## PASCAL: A Phase-Aware Scheduling Algorithm for Serving Reasoning-based Large Language Models

- 属于Serving调度的实现是什么？实验比较什么？
  论文提出PASCAL，一个面向reasoning-based LLM的phase-aware两级调度系统，核心实现包含：(1) Instance-level Scheduler：采用两种实例选择算法——Algorithm 1（为reasoning phase请求选择KV cache footprint最小的SLO-compliant实例）和Algorithm 2（为answering phase请求选择reasoning请求数最少的实例），并在phase transition时执行request migration；(2) Intra-instance Scheduler：每GPU实例内维护两级优先级队列——High-priority queue存放reasoning phase请求（优先调度、优先分配GPU memory），Low-priority queue存放answering phase请求（RR调度+token pacer平滑输出速率）；(3) Adaptive Migration：在phase transition时根据目标实例GPU memory可用性决定是否迁移请求，避免将请求迁移到已满实例导致不必要stall；(4) Conditional Demotion：当单个reasoning请求KV cache超过阈值（5000 tokens）时降级到low-priority queue。实验比较TTFT（tail和raw）、SLO violation rate（基于QoE, threshold=0.95）、serving throughput，对比FCFS（vLLM默认）和RR（round-robin with token quantum=500）两种baseline scheduler。消融实验：PASCAL(NoMigration)（禁用inter-instance migration）和PASCAL(NonAdaptive)（禁用adaptive migration）。

- 硬件平台是什么，配置是什么。
  模拟器环境：8 server nodes（instances），通过100 Gbps fabric互联，每节点配单张NVIDIA H100 96 GB GPU + Intel Xeon Platinum 8558 CPU + 256 GB DDR5 DIMM，PCIe 5.0互联。单instance仿真采用profile-based methodology（vLLM profiling data），验证用真实系统（相同CPU+GPU配置），MAPE 1.62%（end-to-end latency）、12.6%（mean TTFT）、6.49%（TPOT）。GPU memory约束：KV cache allocation capped at 50% of oracle capacity。

- 开源Serving框架是什么。修改了什么。
  基于vLLM v0.6.1（PyTorch 2.4.0, CUDA 12.1）。论文开发了cluster-level simulator模拟PASCAL调度行为，未直接修改vLLM源码。Simulator建模PASCAL的两级调度架构：(1) Instance-level scheduler实现Algorithm 1（instance selection for reasoning）和Algorithm 2（instance selection for answering），基于instance monitor采集的token pacer状态和queue occupancy做placement决策；(2) Intra-instance scheduler实现hierarchical priority queue（high/low），reasoning queue用RR、answering queue用RR+token pacer；(3) 模拟KV cache transfer（迁移时跨实例传输KV cache，P99 latency 0.14-0.25 sec）；(4) Adaptive migration逻辑根据当前实例和目标实例的GPU memory状态override迁移决策。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  论文未明确说明开源地址。以DeepSeek-R1-Distill-Qwen-32B + AlpacaEval2.0 serving为例：
  1. 部署：8-instance cluster，每instance运行vLLM server + PASCAL intra-instance scheduler（high/low priority queues）。Instance monitor持续采集各instance的token pacer状态和queue occupancy。
  2. 请求到达（reasoning phase）：新请求进入instance-level scheduler → Algorithm 1检查所有SLO-compliant instances（ti=TRUE）→ 选择KV cache footprint mi最小的instance（如Instance 3，mi最小）→ 路由到Instance 3的high-priority queue。
  3. Reasoning执行：Instance 3的intra-instance scheduler优先调度high-priority queue中的reasoning请求（RR policy, token quantum=500）。GPU memory优先分配给high-priority queue的KV cache。
  4. Phase transition检测：instance monitor检测到生成的token是`<\think>`（DeepSeek-R1的reasoning结束标记）→ 触发Algorithm 2。
  5. Instance selection for answering：Algorithm 2检查SLO-compliant instances → 选reasoning请求数ri最小的实例（如Instance 5）。Adaptive migration检查：若当前Instance 3 GPU memory充足但Instance 5已满→不迁移，继续在Instance 3执行answering phase。
  6. Answering执行：answering phase请求进入low-priority queue → RR scheduling + token pacer平滑输出速率（target TPOT=100ms）。若某reasoning请求KV cache超过5000 tokens→conditional demotion到low-priority queue。
  7. 效果：PASCAL相比FCFS tail TTFT降低up to 61%（AlpacaEval2.0）和72%（Arena-Hard）。SLO violation rate consistently低于或持平baselines。Throughput与baselines差异<3%。

