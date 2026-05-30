
## The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective

- baseline方法是什么？
  Baseline是conventional single-turn LLM serving（以ShareGPT chatbot workload为代表），即每个用户请求对应一次LLM inference，prefill+decode后返回结果。这是目前LLM serving系统（如vLLM continuous batching）优化的主要目标workload。

  全栈执行例子（以ShareGPT + Llama-3.1-8B-Instruct + vLLM on A100为例）：
  - 算法层：单次LLM inference，无需外部tool interaction、无迭代reasoning loop。模型接收prompt→prefill一次→decode生成response tokens→返回。
  - 系统框架/Serving层：请求到达vLLM→FCFS scheduler→continuous batching将多个请求的decode step合并→prefix caching可选但收益有限（因为单次inference prefix共享少）。ShareGPT latency集中分布在低位（95th percentile 9.7s），throughput可达6.4 QPS。GPU时间分配：prefill约占4.7%、decode约占74.1%、GPU无idle时段（无外部tool等待）。
  - 编译框架层：论文未明确说明（使用PyTorch 2.6 + CUDA 12.8默认编译路径）。
  - kernel调度层：论文未明确说明（使用vLLM默认CUDA kernel，无定制kernel）。
  - 硬件架构层：NVIDIA A100 40GB GPU，无定制硬件。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文不是提出新系统优化，而是构建系统级表征方法论（AgentBench框架），通过量化agent workload vs static LLM serving的成本差距，揭示baseline中"被忽略"的系统瓶颈，为未来scheduler、cache、routing设计提供目标。

  **缺陷1：Static LLM serving假设请求是一次或少次模型前向，无法反映agent的多轮迭代LLM call + tool call模式**
  → 论文测量：tool-augmented agent平均LLM calls是CoT的9.2×，LATS平均71.0次LLM calls/request。单请求内LLM与tool因数据依赖难以并行（LLMCompiler的DAG规划仅实现18.2% overlap）。这导致单请求延迟分布宽且带重尾（ReAct HotpotQA 95th percentile 20.7s vs ShareGPT 9.7s）。

  **缺陷2：Baseline serving optimization假设GPU持续执行LLM inference，不处理tool等待造成的GPU idle**
  → 论文测量：GPU runtime中tool latency可导致最多54.5% idle time（HotpotQA/MATH因Wikipedia API和Wolfram Alpha API在CPU/外部执行）。论文论证需要inter-request parallelism来填补这些idle gap：ReAct sequential execution仅0.10 QPS，concurrent execution提升到2.6 QPS（25×）。但即便如此，agent serving throughput仍远低于ShareGPT（2.6/1.2 QPS vs 6.4 QPS）。

  **缺陷3：Baseline KV cache优化针对短context、单次inference的静态LLM，不处理agent长interaction history导致的KV cache膨胀**
  → 论文测量：tool-augmented agent的KV cache memory/request平均是CoT的3.0×、最坏5.4×。但prefix caching在agentic workload中收益显著：prefill latency降低60.1%，end-to-end LLM latency降低15.7%，LATS memory requirement降低64.8%，serving throughput提升5.62×（vs ShareGPT仅1.03×）。prefix caching在serving场景下使KV cache平均/最大memory usage分别降低51.7%/63.5%。

  **缺陷4：Baseline不考虑test-time scaling的成本-收益递减，将模型推理视为固定成本**
  → 论文测量：test-time scaling存在sharply diminishing returns。Reflexion从16.9s→25.6s仅获4% accuracy gain，从56.0s→325.5s仅获同等marginal gain（31× cost）。sequential scaling峰值资源需求低但延迟长；parallel scaling可降低延迟但增加瞬时GPU memory和serving contention。8B模型配合LATS parallel scaling可接近70B性能但energy更低。8B ShareGPT 0.32 Wh/query vs 8B Reflexion 41.53 Wh (130.9×) vs 70B Reflexion 348.41 Wh (136.5×)。

  **缺陷5：Baseline无datacenter-level energy/power意识**
  → 论文projection：71.4M queries/day下70B Reflexion接近1.0 GW datacenter power；Google Search级13.7B queries/day下70B Reflexion达198.9 GW（接近美国电网平均负荷的40%）。

  论文方法全栈执行例子（以HotpotQA + ReAct agent + Llama-3.1-8B-Instruct + vLLM on A100为例）：
  - 算法层：ReAct agent workflow：LLM产生thought/action（如Wikipedia search）→tool执行→observation回写context→下一轮LLM。单请求平均多次LLM call（agent calls平均9.2× CoT），token组成包含instruction + few-shot + user query + LLM history + Tool history + output。
  - 系统框架/Serving层：Agent server entrypoint→ReAct worker→vLLM backend。prefix caching在multi-round LLM calls间复用shared prefix KV cache，prefill latency降低60.1%。多worker并发通过continuous batching填补单worker的tool-idle GPU时间。GPU execution breakdown：prefill 4.7%、decode 74.1%、tool-idle最高54.5%。
  - 编译框架层：论文未明确说明（PyTorch 2.6 + CUDA 12.8默认编译）。
  - kernel调度层：论文未明确说明（使用vLLM默认kernel）。
  - 硬件架构层：GCP A100 40GB GPU（8B单卡、70B 8卡），GPU utilization用DCGM测量。论文强调control-flow serialization、long-context KV cache pressure和idle-period underutilization是dynamic reasoning workload的特征，不是特定GPU microarchitecture独有。

## PASCAL: A Phase-Aware Scheduling Algorithm for Serving Reasoning-based Large Language Models

- baseline方法是什么？
  Baseline是vLLM默认的FCFS（First-Come-First-Served）scheduler和Round-Robin（RR）time-sharing scheduler，两者均不区分reasoning phase和answering phase。

  FCFS全栈执行例子（以DeepSeek-R1-Distill-Qwen-32B + vLLM 0.6.1 + H100 96GB, memory-constrained 50% KV cache capacity为例）：
  - 算法层：Reasoning-based LLM inference（CoT decoding）。模型生成reasoning tokens（hidden, r1-rN）→ 遇到`<\think>`标记 → 生成answering tokens（user-visible, t1-tM）。FCFS不区分两阶段。
  - 系统框架/Serving层（FCFS）：请求按到达顺序进入batch。当KV cache占满GPU memory时，新请求被block在队列中等待已运行请求完成→ Head-of-Line (HoL) blocking。即使短reasoning请求（128 tokens）也需等待长请求完成，latency up to 5.14× vs oracle。Reasoning phase被block→TTFT膨胀。Answering phase同样被block→TTFAT（Time-To-First-Answering-Token）延迟→SLO violation。
  - 系统框架/Serving层（RR）：RR分配固定token quantum（500 tokens），轮转调度所有请求。对short reasoning请求减轻HoL blocking（TTFT更低），但frequent preemption fragment execution→long reasoning请求（2048 tokens）latency up to 1.75× vs oracle。Answering phase通过preemption避免HoL blocking→SLO attainment较高。但RR不能enforce全局"reasoning-first"优先级→reasoning和answering在同一batch竞争。
  - 编译框架层：论文未明确说明（PyTorch 2.4.0 + CUDA 12.1默认路径）。
  - kernel调度层：论文未明确说明（vLLM默认kernel）。
  - 硬件架构层：NVIDIA H100 96 GB GPU + Intel Xeon Platinum 8558 CPU + 256 GB DDR5，PCIe 5.0。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出PASCAL，通过phase-aware两级调度架构（instance-level + intra-instance）解决reasoning-based LLM serving中baseline不区分reasoning/answering phase导致的TTFT膨胀和SLO violation问题。

  **缺陷1：FCFS/RR不区分reasoning phase和answering phase → reasoning latency（直接决定TTFT）被blocking或preemption不必要地延长**
  → **PASCAL方案**：Intra-instance hierarchical priority queue。High-priority queue存放reasoning请求，优先调度、优先分配GPU memory；Low-priority queue存放answering请求。原因：reasoning phase latency高度sensitive to interruption（影响TTFT），answering phase是threshold-sensitive（只需满足TTFAT≤0.25s + TPOT≤100ms即可）。RR用于high-priority queue保证short reasoning请求低TTFT；RR+token pacer用于low-priority queue保证answering phase SLO compliance。

  **缺陷2：FCFS/RR无跨instance感知 → 可能导致reasoning/answering请求分布不均，某些instance因memory压力导致answering phase SLO violation**
  → **PASCAL方案**：Instance-level scheduler双算法。Algorithm 1（reasoning请求）：选择SLO-compliant instances中KV cache footprint最小的instance，最小化新请求对现有请求干扰的同时减小attention执行时间。Algorithm 2（answering请求）：选择reasoning请求数ri最少的instance，避免answering被高优先级reasoning抢占memory。通过instance monitor持续采集token pacer状态和queue occupancy指导placement。

  **缺陷3：Baseline无phase transition处理 → reasoning完成后answering请求可能与同instance的reasoning请求竞争资源，导致answering phase stall**
  → **PASCAL方案**：Phase-aware request migration。在检测到phase transition token（如`<\think>`）时，根据Algorithm 2决定是否迁移到更合适的instance。Adaptive migration override：若当前instance GPU memory充足而目标instance已满，保留在当前instance避免不必要的KV cache transfer overhead（P99 transfer latency仅0.14-0.25 sec, negligible vs reasoning latency）。

  **缺陷4：RR的fairness-oriented priority导致长reasoning请求持续被preempt但短request answering phase却获得高优先级**
  → **PASCAL方案**：Conditional demotion。当单个reasoning请求KV cache超过5000 tokens时demote到low-priority queue，释放GPU memory给answering请求。Global "reasoning-first" priority确保所有实例一致：reasoning always preempts answering。

  PASCAL全栈执行例子（以DeepSeek-R1-Distill-Qwen-32B + AlpacaEval2.0 + 8-instance H100 cluster为例）：
  - 算法层：DeepSeek-R1-Distill-Qwen-32B CoT decoding，reasoning tokens → `<\think>` → answering tokens。PASCAL不修改模型或推理算法，仅通过调度区分两阶段。
  - 系统框架/Serving层：Request到达→Algorithm 1选instance（reasoning phase）→ high-priority queue → RR调度（token quantum=500）→ 检测`<\think>` → Algorithm 2选instance（answering phase）+ adaptive migration → low-priority queue → RR + token pacer。Tail TTFT vs FCFS: 减少up to 72%（Arena-Hard, 64.21 sec absolute reduction）；vs RR: 减少up to 33%（89.91 sec absolute reduction）。SLO violation rate consistently lower或comparable。Throughput差异<3%。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明。
  - 硬件架构层：8×H100 96GB GPU cluster，100 Gbps fabric互联。KV cache transfer: P99 0.14-0.25 sec（negligible vs tens-to-hundreds sec reasoning latency）。

## Adaptive Draft Sequence Length: Enhancing Speculative Decoding Throughput on PIM-Enabled Systems

- baseline方法是什么？
  Baseline是SpecPIM类PIM-enabled heterogeneous speculative decoding系统（PIM-SD）。PIM-SD在HBM-PIM+GPU异构系统上运行speculative decoding：(1) DLM (如OPT-1.3B) 在PIM/GPU上自回归生成固定长度d=8的draft tokens；(2) TLM (如OPT-66B) 并行验证所有draft tokens；(3) operator mapping通过离线design-space exploration基于初始batch size和fixed draft length确定，推理中不改变；(4) DLM prediction和TLM verification严格串行执行。

  全栈执行例子（以OPT-66B+OPT-1.3B, Dolly dataset, batch_size=64, d=8为例）：
  - 算法层：standard speculative decoding with fixed draft length d=8。DLM (OPT-1.3B) autoregressive生成8个draft tokens→TLM (OPT-66B) parallel verification with rejection sampling。当d=8时acceptance rate从d=4的~0.6降至~0.4(Fig.4a)，大量draft token被拒绝后丢弃，浪费DLM生成和TLM验证的计算。
  - 系统框架层：PIM-SD采用静态operator mapping，DLM prediction→TLM verification串行执行。每轮speculative iteration：所有请求先等DLM生成d=8 tokens（micro-batch同步屏障），再统一TLM验证。batch内请求间draft长度相同无bubble，但固定长度导致整体吞吐在BS=64时反而低于autoregressive baseline (Fig.3a)。
  - 编译框架层：论文未明确说明。
  - kernel调度层：PIM-SD离线分析后固定映射：DLM attention→PIM, TLM FC→GPU。当effective batch size和draft length变化时映射不变。例如当batch内部分请求提前完成drafting后effective batch size降低，DLM FC算术强度降低（从GPU带宽限制转为PIM计算限制，Fig.6），但operator仍固定映射在GPU上执行，导致suboptimal utilization。
  - 硬件架构层：AttAcc风格HBM-PIM架构：每bank 1 PE，bank-level并行。PIM-SD的Manager无runtime adaptive control，仅执行离线确定的mapping schedule。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出SADDLE，针对PIM-SD的三大缺陷分别设计解决方案：

  **缺陷1：固定draft length → 生成大量被TM拒绝的无效token，浪费计算和带宽**
  → **SADDLE方案**：运行时自适应draft length。Controller每生成draft token时读取采样概率p_t→维护累计接受概率H_t=∏p_i→当H_t<阈值τ（离线用验证集校准，选20%区间内平均draft length最高且≥90%验证成功率的τ）时停止该请求drafting。简单请求（高p_t token）自动获得更长draft，复杂请求（低p_t token）更早停止。运行时可根据系统负载动态调节τ：轻负载降低τ允许更长draft→提升并行度。

  **缺陷2：DLM-TLM串行执行 + 自适应draft长度引入的同步bubble → 请求间等待加剧延迟**
  → **SADDLE方案**：prediction-verification解耦异步pipeline + Shared Pool + Eager Pool。(a) Shared Pool跨micro-batch聚合draft tokens：各micro-batch不再单独等待所有请求完成DLM prediction，draft tokens逐生成即存入Shared Pool，当token数达GPU capacity C(=512)或GPU空闲时触发TLM并行验证；(b) Eager Pool乐观执行：TLM验证Shared Pool时，DLM基于"当前token将被接受"假设继续生成后续tokens暂存Eager Pool，验证通过后迁入Shared Pool，被拒绝则丢弃；(c) 异步重叠prediction和verification，消除串行pipeline的idle time。

  **缺陷3：算术强度动态变化 → 静态operator mapping suboptimal**
  → **SADDLE方案**：arithmetic intensity-aware operator scheduler。(a) predication后根据仍活跃请求数估算DLM FC有效micro-batch size→计算arithmetic intensity→与预标定PIM/GPU阈值比较动态remap；(b) verification前根据Shared Pool每请求token数估算TLM attention arithmetic intensity→同理动态remap。初始固定映射：DLM attention→PIM (低强度)、TLM FC→GPU (高强度的GEMM)。动态remap使SADDLE中14.89% ops在PIM、85.11%在GPU执行（vs 无scheduling时的9.51%/90.49%），吞吐提升1.21×。

  论文方法全栈执行例子（以OPT-66B+OPT-1.3B, Dolly, BS=64为例）：
  - 算法层（核心创新）：自适应draft length。每请求dynamically调整draft长度：DLM生成token x_t时获取p_t=DLM(x_t|x_{<t})→更新H_t=H_{t-1}·p_t→若H_t<τ=θ则停止该请求drafting。H_t基于DLM自身采样概率，无需额外训练或分类器。相比baseline固定d=8时acceptance rate ~0.4，SADDLE自适应停止在H_t低于阈值时，每个请求的draft length在[1, optimal]区间动态变化，减少无效token生成。
  - 系统框架层（核心创新）：异步pipeline。batch切成micro-batches→每micro-batch有独立Draft Generator。请求#0 (simple task) H_t始终>τ→持续draft→tokens入Shared Pool。请求#1 (complex task) H_t在第3 token后<τ→停止drafting。不等待请求#1继续：Shared Pool累计token数达C→TLM并行验证所有已存tokens。同时请求#0在TLM验证期间继续生成新tokens→Eager Pool暂存→验证通过后migrate到Shared Pool。
  - 编译框架层：论文未明确说明。
  - kernel调度层（核心创新）：动态operator mapping。prediction后Scheduler统计仍活跃请求数→估算DLM FC effective batch size→与roofline阈值比较→决定FC在PIM或GPU执行。verification前Scheduler统计Shared Pool每请求token数→估算TLM attention arithmetic intensity→同样动态remap。例如当请求#1停止drafting后effective batch size降低→DLM FC arithmetic intensity降到PIM compute-bound区→Scheduler将DLM FC从GPU remap到PIM执行。Operator mapping随每speculative iteration动态调整。
  - 硬件架构层（核心创新）：SADDLE Manager硬件。Controller以专用硬件（softmax unit + multipliers + comparators）低延迟计算H_t并比较τ（仅占end-to-end latency 0.83%）。Shared Pool (1KB CAM)和Eager Pool (1KB) 的token migration为lightweight on-chip memory operation，每verification iteration后刷新无容量压力。SFU在buffer die加速softmax/layer norm等非矩阵运算。PE沿用HBM-PIM design (16 FP16 MACs/bank)，面积overhead仅13.4% DRAM die。


## Towards Resource-Efficient Serverless LLM Inference with SLINFER
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

## VDHA: Vector-Driven Hash Aggregation for Sparse Matrix–Sparse Vector Multiplication on GPUs
- baseline方法是什么？
  Baseline是现有的GPU SpMSpV write-back策略：(1) **Atomic write-back**：每个partial product直接用global atomicAdd写入output vector y[ind]。this method在many-to-one scatter pattern下产生严重address contention，uncoalesced memory stores导致带宽利用率极低。在A100上sparsity=0.1时atomic write-back仅270 GB/s (peak 1555 GB/s的17%)，write-back占overall runtime的>30%，stall cycles中>45%是long scoreboard waits。(2) **Sort-based write-back**：buffer所有(row_idx, val) pairs→global sort by row index→sequential reduce duplicates→每个row仅写一次。sort阶段带宽仅~43.3 GB/s，write-back占overall runtime的>70%，需要large temporary buffers。
  
  全栈执行例子（以it-2004 web graph, sparsity=0.1, NVIDIA A100为例）：
  - 算法层：SpMSpV y=A*x，CSC格式矩阵→vector-driven (column-major) paradigm: 对x中每个nonzero遍历A对应column→生成partial products→write-back到y
  - 系统框架层：GPU graph frameworks (Gunrock/GraphBLAST)提供atomic-based SpMSpV kernel，或Adaptive SpMSpV根据matrix statistics在4种column-major kernel+2种row-major kernel间选择
  - 编译框架层：论文未明确说明（手工CUDA kernel，无编译框架自动生成）
  - kernel调度层：BlockAtomic (Gunrock-like): 多short column聚合到一个CTA→CTA内threads各自计算partial products→global atomicAdd写入y。GlobalAtomic: global prefix scan计算total NNZ→均匀分配每个CTA→CTA内threads global atomicAdd。BlockSort (FastSpMSpV-like): CTA内生成(row_idx, val) pairs→sort→reduce→coalesced global write。GlobalSort: global均匀分配→sort-reduce→write。Load balancing策略：Block-mapped按column数分配CTA（skewed分布下poor balance），Global-mapped按NNZ均匀分配CTA（较好balance但prefix-scan overhead）
  - 硬件架构层：NVIDIA A100 GPU。global memory bandwidth 1555 GB/s，L2 cache 40MB，per-SM 168KB shared memory。Atomic units处理global atomicAdd，uncoalesced scatter导致cache line浪费、L2 hit rate低、bandwidth saturation差

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**VDHA (Vector-Driven Hash Aggregation)**：通过shared-memory hash table做local aggregation减少global write conflicts，column decomposition+reordering增强locality，fetch-compute-writeback pipeline隐藏hash overhead。

  论文方法全栈执行例子（以it-2004, sparsity=0.1, A100为例）：
  - 算法层：SpMSpV y=A*x，CSC格式→vector-driven paradigm，hash-based write-back替代atomic/sort write-back
  - 系统框架层：VDHA CUDA kernel可集成到adaptive SpMSpV框架中（配合预测模型在VDHA和baseline间选择，best-of-7 fallback实现1.22× speedup）
  - 编译框架层：论文未明确说明（手工CUDA kernel，使用cp.async异步copy指令、atomicCAS等PTX级操作，无编译框架自动生成）
  - kernel调度层（核心创新）：
    (1) **Shared-memory hash aggregation**（解决baseline缺陷：global atomic scatter→low bandwidth utilization）：
    每个CTA维护2048-entry shared-memory hash table→partial products插入hash table做local accumulation→hash table满时bucket-order flush到global memory。对比baseline：BlockAtomic每个update都global atomic→>30% runtime in write-back with ~270 GB/s bandwidth；VDHA大部分updates在shared memory local aggregation→仅flush和fallback才global write→global atomic conflicts显著减少（atomic-unit utilization从22.99%降至12.82%）。hash table还提供partial ordering：entries按bucket order (0→2047) flush→warp内threads访问相对连续地址→改善memory coalescing（γ从0.744提升至2.607）。
    (2) **Column decomposition with reordering**（解决baseline缺陷：skewed column lengths→workload imbalance + poor hash locality）：
    长列按SPLIT_SIZE=256切分为segments→segments metadata按首row index排序（仅排序segment metadata而非nonzeros本身，O(S log S) cost where S<<N）→cross-column segment overlap增强hash table reuse。对比baseline：Global/Block mapping均不exploit long-column内部缺乏locality的特点→ρ仅51.0% (T=2048, density=100%)→hash table insufficient aggregation；VDHA切分+重排序后ρ提升至89.8%→更多updates在共享内存完成aggregation。local overlap ratio ρ从0.510→0.898，coalescing factor γ从0.744→2.607。
    (3) **Fetch-compute-writeback pipeline**（解决baseline缺陷：high memory latency→warp stall无法被occupancy掩盖）：
    double buffering + cp.async异步fetch下个segment→当前segment做hash aggregation→next segment ready时swap buffer。对比baseline：45%+ stall cycles是long scoreboard waits (pending global memory)→即使高occupancy也无法隐藏latency；VDHA将hash computation叠加到memory fetch latency上→stall ratio从>45%降至~15%→hash computation cost从16.7%降至12.3%。
  - 硬件架构层：NVIDIA A100 GPU。利用shared memory（per-SM 168KB）做hash table显式scratchpad（vs L1 cache implicit caching），2048-entry hash table=16KB per CTA→8 CTAs/SM→256 threads/CTA平衡occupancy。cp.async (Ampere+)实现asynchronous global→shared copy。atomicCAS支持intra-CTA shared memory atomic操作。FALLBACK_ITER机制控制linear probing worst-case latency避免warp divergence。

  Baseline缺陷→VDHA方案映射：
  | Baseline缺陷 | VDHA方案 | 效果 |
  |-------------|---------|------|
  | Global atomic scatter→low bandwidth (~270 GB/s) | Shared-memory hash local aggregation + coalesced flush | Atomic-unit utilization ↓ 22.99%→12.82% |
  | Skewed column lengths→imbalance + poor hash locality | Column split (SPLIT_SIZE=256) + segment reorder by first row index | Local overlap ρ ↑ 0.510→0.898, γ ↑ 0.744→2.607 |
  | Memory stall dominates (>45% long scoreboard) | cp.async double-buffering pipeline: fetch↔compute overlap | Stall ratio ↓ >45%→~15%, hash cost ↓ 16.7%→12.3% |
  | Sort-based→O(N log N) sort overhead (~43 GB/s sort bandwidth) | Hash aggregation O(N) with hash table (O(1) amortized insertion) | No global sort needed, hash computation mostly hidden |
  | No basis for kernel selection→suboptimal choice | Decision tree predictor (5 features, 91.3% accuracy) → fallback to best-of-7 | Adaptive geomean speedup 1.13×→1.22× on SuiteSparse |

## Exploiting Efficient Mapping and Pipelined Execution for Accelerating SpMV on Tensor Cores

- baseline方法是什么？
  Baseline是DASP（Dense MMA units Accelerated general Sparse matrix-vector multiplication, SC'23），目前state-of-the-art的Tensor Core加速SpMV方案。全栈执行例子：
  - 算法层：SpMV y=A*x，稀疏矩阵A经reorder按NNZ per row分为long/medium/short三类
  - 系统框架层：NVIDIA cuSPARSE等vendor library提供CSR/CSC/BSR/SELL等格式的SpMV kernel
  - 编译框架层：论文未明确说明（手工CUDA kernel，无编译框架自动生成）
  - kernel调度层：DASP使用m8n8k4 TC shape（Volta时代最小TC），将非零元素沿TC output matrix对角线映射。long row独占TC block经shuffle指令warp内归约后调用额外GPU function做warp-level归约、medium row分组到TC block coalesced写回、short row拼接互补长度（如NNZ=1+3→mma_k=4 TC block）但仍独立计算不减少TC执行数。数据访问串行——每个线程独立load sparse A（Fetch Sparse）→load column ID（Fetch CID）→根据CID sparsely load X→TC Computation，缺少asynchronous memory access导致warp stall严重、memory utilization仅20%-60%
  - 硬件架构层：NVIDIA A100/H100 GPU。m8n8k4 TC在A100/H100上不再由真实TC硬件执行，编译器fallback到ALU指令（CUDA Cores），Tensor Cores大量idle。NVIDIA Nsight profiling显示大量CUDA Core指令而非TC指令。TC result efficiency Eres较低（大TC shape下因mismatch降至1/4），约60% Ecomp

  其他baseline：Spaden（bitmap-based sparse format用TC但引入额外LOP3 bit-shift指令恢复非零位置）；SpMM-oriented方法DTC-SpMM/SMaT（用于SpMM时因SpMV的single-column特性Ecomp极低仅11.78%）；cuSPARSE（CSR格式CUDA Cores计算不利用TC）；FastLoad（CSC格式+load balancing但不用TC）；TileSpMV（tiled SpMV不用TC）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**Drawloom**：面向GPU Tensor Cores和Sparse Tensor Cores的SpMV加速框架，通过ArbitWeave自适应TC shape映射、ZCF稀疏存储格式和Multi-stage Register Pipeline三大组件解决DASP的三个关键缺陷。

  论文方法全栈执行例子（以pwtk矩阵0.2M rows/11.6M NNZ在A100 HBM FP16上为例）：
  - 算法层：SpMV y=A*x，SuiteSparse matrix→preprocessing重排序按NNZ per row分类
  - 系统框架层：Drawloom CUDA library替代cuSPARSE等vendor SpMV调用
  - 编译框架层：论文未明确说明（手工CUDA kernel + PTX intrinsic如LDG指令、async-copy、MMA指令，无编译框架自动生成）
  - kernel调度层（核心创新）：
    (1) **ArbitWeave mapping**（解决DASP缺陷1：suboptimal TC shape→ALU fallback）：
    选择现代GPU高效TC shape如A100/H100上的m16n8k16（494 TFlops），按结构比V=mma_m/mma_n=2切分row strip（2行一组）→压缩零值列对齐TC block的B矩阵columns保持高Ecomp→成果沿output matrix对角线输出保持高Eres。Long Mapping: nnz>T1的row strip独占TC block，TC blocks均匀分配warps（WarpLoad参数）+warp内shuffle归约+warp间reduction kernel。Medium Mapping: T2<nnz≤T1的多个row strip聚合到TC block。Short Mapping: nnz≤T2的row strip利用SpTC 2:4 sparsity加速——非零元配对编组到2:4 pattern box（每box最多2 NNZ）→50%存储压缩+metadata编码非零位置→SpTC硬件直接跳过zeros。首次同时调度TC和SpTC处理不同稀疏度矩阵区域。
    (2) **Zig-zag Chained Format (ZCF)**（解决DASP缺陷2：rigid sparse formats tied to specific TC shapes→poor portability）：
    Long/Medium使用ZCF for TC（ptr+Cid+Val，zcf_value_stride FP16=8/FP32&FP64=4对齐128-bit transaction，vectorized memory access减少IMAD index计算指令67.8%、branch指令50%、提升memory bandwidth 48.3%）。Short使用ZCF for SpTC（2:4压缩Val+metadata编码+remapped Cid）。Two-level load balancing: matrix-level按NNZ分组+WarpLoad分Long TC blocks到多warp。
    (3) **Multi-stage Register Pipeline**（解决DASP缺陷3：high memory latency+insufficient memory-compute overlap→long warp stall）：
    将Fetch Sparse→Fetch CID→Load X→TC Comp串行流程重构为FillSMEM（async-copy sparse A+Cid GMEM→SMEM）→FillREG（SMEM index→LDG load X到REG）→Comp（TC MMA m16n8k16）→EmptySMEM→EmptyREG五阶段pipeline。delaySMEM控制GMEM-SMEM overlap，delayREG控制SMEM-REG+Compute overlap。warp stall改善达3.02×-3.13×（多数representative matrix）、memory throughput提升达2.61×-2.75×。
  - 硬件架构层：NVIDIA A100/H100 GPU，利用async-copy硬件单元（A100+）、Tensor Cores MMA（m16n8k16真硬件执行而非ALU fallback）、Sparse Tensor Cores 2:4 sparsity硬件、shared memory双缓冲、128-bit vectorized global memory transaction。

  与DASP的缺陷-方案映射：
  1. DASP用m8n8k4→A100/H100上被编译为ALU指令→Drawloom ArbitWeave选择m16n8k16由真TC硬件执行，保持Ecomp≈60%同时显著提升Eres
  2. DASP rigid format tied to specific TC shape→Drawloom ZCF支持任意TC shape，vectorized access+减少分支和地址计算指令
  3. DASP串行memory-compute→Drawloom Multi-stage Register Pipeline利用async-copy+多级REG pipelining实现memory-compute overlap，消除warp stall
  4. 额外创新：Short row由SpTC而非CUDA Cores处理（DASP short row用CUDA Cores），首次实现TC+SpTC同时调度的SpMV方案

## MetaAttention: A Unified and Performant Attention Framework Across Hardware Backends

- baseline方法是什么？
  Baseline方法分为三类：(1) **手写专家库**：FlashAttention-2/3 for Softmax Attention（~2.7k行CUDA，针对标准headdim相同、softmax归一化、H100/A100优化）、FlashSigmoid for Sigmoid Attention（~1.9k行CUDA）、FlashMLA for DeepSeek MLA（~1.7k行CUDA、blockSize=64）、Mamba2 chunk kernel for Mamba2 SSM（~3k行Triton）、Flash-Linear-Attention v0.2.0 for Gated Retention/RetNet Recurrent（~0.4k行Triton）。全栈执行例子：算法层标准Softmax Attention Q×K^T→softmax→×V→Serving/系统框架层FlashAttention-3 kernel在H100上通过TMA异步加载、Tensor Cores MMA、register-level pipelining→编译框架层手写CUDA kernel无自动生成→kernel调度层固定tiling/pipeline策略→硬件架构层NVIDIA H100/A100 GPU。缺陷：(a) 仅支持特定attention变体——Gated-RetNet和ReLU-Attention上现有库表现极差或不支持（Fig.2：FLOPS utilization仅2.9%-7.1% vs Softmax-Attention的65.3%）；(b) 硬件移植成本高——FlashAttention v2在A100达70% peak throughput但H100仅30%，需register-level pipelining和ping-pong kernel design才能发挥H100性能；(c) 非标准shape不支持——如DeepSeek MLA的dimqk≠dimv、RetNet的非标准embedding维度、query seqlen=1解码场景；(d) 开发成本高——每种新attention变体需要专家手写上干行kernel代码（Table 5）。

  (2) **通用DL compiler**（Torch Inductor、TVM、Ansor-AF、Welder、Alcop、TensorRT）：将attention作为不透明算子序列做operator fusion。全栈执行例子：算法层attention展开为matmul+softmax+matmul→编译框架层Torch Inductor/TVM识别算子图并尝试fusion→kernel调度层生成fused kernel→硬件架构层GPU。缺陷：(a) 不理解attention语义——无法自动推导online softmax、chunk parallelism、memory-efficient pipelining等attention特有优化；(b) 在attention workload上性能远低于手写库（Fig.2：Torch Inductor在Softmax-Attention仅14.1% FLOPS utilization vs FlashAttention-3的65.3%）；(c) Ansor-style auto-tuning编译时间长。

  (3) **模板/接口型方法**（FlexAttention、FlashInfer）：预定义大部分attention计算，开放有限参数或user code injection。全栈执行例子：算法层parallel attention→编译框架层FlexAttention/FlashInfer暴露score_mod等有限callback→kernel调度层预编译fixed-pattern kernel→硬件架构层GPU。缺陷：(a) 仅支持parallel pattern——无法覆盖recurrent/linear attention（Mamba2、RetNet、YOCO等）；(b) 接口灵活性有限——非标准tensor shape（dimqk≠dimv、MLA head/head_kv、chunk-based state维护等）超出接口能力。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**MetaAttention**：统一attention抽象框架，通过relevance scoring + aggregation两个核心操作+ customizable functions模板，结合IntermediateTensor-based两层调度策略自动生成跨硬件后端优化kernel。

  论文方法全栈执行例子（以RetNet Parallel Attention在H100上为例）：
  - 算法层：用户选择Parallel Pattern→声明Q[batch,head,seq_len,256]/K[batch,head,seq_len_kv,256]/V[batch,head,seq_len_kv,512]→定义scores_Mod（scores*mask缩放）+scores_RowNorm（reduceAbsSum-based row normalization）→MetaAttention将RetNet的retention机制映射为relevance scoring（Q×K^T matmul）+Mod+RowNorm+aggregation（scores×V matmul）
  - 编译框架层：frontend trace customizable functions→tensor DAG（elementwise节点→SIMT fusion，row-reduce节点→intra-warp reduction）→scheduler enumerates output tile sizes（如[64,128]）→propagate tiles to Q/K/V/scores/state→TileResourceScheduling分配memory tier和pipeline stage→H100 DeviceConfig约束（basetile 128×128，register 65536×32bit，shared mem 228KB）→生成scheduling plan
  - kernel调度层：runtime选择Parallel Pattern kernel template→inline customized Mod/RowNorm online（online_prologue初始化row_sum_wo_clamp=0,row_sum=0→online_forward逐tile计算reduceAbsSum+更新全局row_sum+对前一tile rescale）→TMA异步加载K/V tile到shared memory→Tensor Cores执行Q×K^T MMA→on-chip执行customized functions→Tensor Cores执行scores×V MMA→遍历所有KV tiles
  - 硬件架构层：NVIDIA H100 SXM5（CUDA 12.4），利用TMA硬件单元异步加载、Tensor Cores做MMA、shared memory做tile缓存

  以Mamba2 SSM Recurrent Attention在H100上为例：
  - 算法层：用户选择Recurrent Pattern→声明Q/K/V shape、headv=80,dimqk=128,dimv=64→relevance scoring为Q×hidden_state matmul→aggregation为hidden_state += K[i]×V[i]
  - 编译框架层：scheduler将hidden_state、Q/K/V tile、临时Mod结果建模为IntermediateTensor→搜索chunk parallelism的chunk size和state tile memory placement→scheduling time约82秒
  - kernel调度层：runtime使用chunk parallelism将长序列切为并行chunk→chunk内维护recurrent state→elementwise+reduction逻辑融合到recurrent kernel→single fused kernel执行

  对应解决Baseline缺陷：
  (1) **手写库仅支持特定attention变体** → unified attention abstraction覆盖Parallel和Recurrent两种pattern，customizable functions（Mod/RowNorm/RowNorm online）表达Softmax/Sigmoid/ReLU/RetNet/Mamba2/MLA/Sparse GQA等变体。Table 3展示10种attention mechanism均可在MetaAttention中实现。自定义attention仅需22-90行代码（vs手写库0.4k-3k行）。
  
  (2) **手写库硬件移植成本高** → DeviceConfig + IntermediateTensor scheduling自动适应硬件差异。H100上使用TMA+Tensor Cores+CUTE/TileLang，AMD MI250上使用Matrix Cores+TileLang。编译时间控制在分钟级（46-89秒），无需为每种新GPU手写不同kernel。MI250上avg 3.3× forward/2.0× backward speedup over baseline证明跨后端能力。
  
  (3) **手写库不支持非标准shape** → MetaAttention不要求headdim_qk=headdim_v或特定head layout。Diff-Transformer-3B（dimqk=128≠dimv=256）上相对FlashAttention-3平均1.61× speedup（FA3需padding到同维度）。MLA（head=128, head_kv=1, dimqk=576≠dimv=512, query seqlen=1）上性能接近FlashMLA且比MLA Triton快4.6×。
  
  (4) **通用compiler不理解attention语义** → MetaAttention显式建模relevance scoring/aggregation/online RowNorm/recurrent state语义，支持online softmax（RowNorm online接口：prologue/forward/epilogue三段式）、chunk parallelism（recurrent pattern自动分块）、on-chip fusion（IntermediateTensor memory tier控制）。H100上相对PyTorch Inductor获得大幅speedup。
  
  (5) **模板库仅支持parallel pattern** → MetaAttention并行支持parallel和recurrent两种pattern。Mamba2 SSM forward/backward平均1.66×/1.78× over Flash-Linear-Attention。RetNet Recurrent、YOCO-13B、RFA-Big等recurrent变体均被覆盖，而FlexAttention/FlashInfer不支持。
  
  (6) **开发效率低** → attention-specific programming interface（pattern选择+shape声明+Mod/RowNorm/RowNorm online函数定义）将开发代码量从数百至数千行降至数十行（Table 5：MLA 90 vs 1700 LoC，Mamba2 27 vs 3000 LoC）。

  Trade-off：(a) MetaAttention用受约束的attention-specific model换取可优化性——不是任意Python/Tensor IR compiler，假设attention可分解为relevance scoring+aggregation+有限类型customizable functions；(b) scheduling time为分钟级（46-89s），短于Ansor等传统auto-tuning compiler，但仍不是零成本即时编译；(c) 在已有高度优化手写库且shape完全匹配的场景，目标为comparable performance，主要优势来自变体/非标准shape/跨后端/开发效率。

## MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models

- baseline方法是什么？
  Baseline方法分为三类：(1) **Vanilla System**：所有请求用单一large diffusion model（SD3.5L或FLUX）做全量T=50步去噪推理，无任何缓存。全栈执行例子：算法层diffusion model T=50步iterative denoising→系统框架层单一模型worker处理所有请求，无调度优化→编译框架层论文未明确说明→kernel调度层标准PyTorch diffusion推理kernel→硬件架构层A40/MI210 GPU。缺陷：每请求完整去噪过程计算量大、延迟高、吞吐低。(2) **Nirvana (latent caching)**：缓存diffusion model中间latent representations，text-to-text similarity检索复用，跳过少量去噪步数。全栈执行例子：算法层diffusion modeldenoising→系统框架层Nirvana text-to-text检索latent cache→编译框架层论文未明确说明→kernel调度层标准diffusion kernel→硬件架构层A40/MI210 GPU。缺陷：latent cache 2.5MB/张存储开销大；model-dependent，同一cache不能跨模型复用；text-to-text retrieval视觉对齐差（CLIPScore mean 0.22 vs text-to-image 0.28）；>90% cache hit rate下仅20% latency reduction；高请求率下SLO violation频繁。(3) **Pinecone (retrieval-only)**：基于CLIP text embedding similarity直接检索并返回最相似cached image，无refine。缺陷：无generative refinement导致CLIPScore显著低于生成方法，图像-文本对齐弱。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**MoDM (Mixture-of-Diffusion Models)**：基于final image缓存+CLIP text-to-image similarity检索+小型扩散模型refine+大型模型全量推理的混合serving系统，通过PID-driven的Global Monitor动态分配GPU资源实现延迟-质量自适应平衡。

  论文方法全栈执行例子（以SD3.5L作为大模型、SDXL作为小模型、4×A40为例）：
  - 算法层：final image caching替代latent caching（1.4MB vs 2.5MB/张，100K embedding仅0.29GB）；text-to-image CLIP similarity检索（cosine sim公式1）；quality-constrained dynamic k-selection heuristic (Fig.5b, α≥0.95)；noise re-introduction Ĩ=σ_tk·ε+(1-σ_tk)·I*重新进入去噪流程→小模型执行剩余T-k步去噪（公式4 compute savings）
  - 系统框架层：Request Scheduler管理CLIP embedding提取+缓存查取+请求路由(cache-hit→小模型queue, cache-miss→大模型queue)→Global Monitor PID controller (Algorithm 1, Kp=0.6,Ki=0.05,Kd=0.05) 统计R/H_cache/k分布→Quality-Optimized Mode (最大化N_large) 或Throughput-Optimized Mode (cache-hit全用小模型)→N个GPU Worker动态加载大/小模型→FIFO cache维护(>90% hits在4h内)
  - 编译框架层：论文未明确说明
  - kernel调度层：标准PyTorch diffusion kernel，未做kernel修改。cache检索GPU上cosine similarity计算0.05s/100K images
  - 硬件架构层：4×A40 (48GB) 或16×4×MI210 (64GB)，PyTorch RPC跨进程通信

  对应解决Baseline缺陷：
  (1) **Nirvana latent cache存储开销大+model-dependent** → final image cache模型无关（PNG/JPEG标准格式），1.4MB/张 vs 2.5MB/张，跨Stable Diffusion/SANA/FLUX多模型族复用
  (2) **Nirvana text-to-text retrieval视觉对齐差** → text-to-image CLIP similarity检索，CLIPScore mean 0.28 vs 0.22，PickScore mean 20.33 vs 19.52，检索到视觉上更符合prompt的图像
  (3) **Nirvana >90% hit rate仅20% latency reduction** → 混合大/小模型：cache-hit用小模型refine (每步compute cost更低)，cache-miss用大模型保证quality。DiffusionDB上MoDM-SANA达到3.2× throughput、46.7% energy savings、66.3% energy savings with SANA as small model
  (4) **单一模型无法应对负载波动导致SLO violation** → PID-driven Global Monitor动态调整N_large/N_small分配，支持Quality-Optimized和Throughput-Optimized双模式，请求率波动时自动切换small model类型（SDXL→SANA），在4×A40上支持10 req/min without SLO violation (2× threshold)，而Vanilla仅5 req/min、Nirvana仅6 req/min即出现显著SLO violation
  (5) **Pinecone retrieval无refine导致CLIPScore低** → cache-hit请求加噪后用小模型refine T-k步，保证visual quality接近全量大模型(99.7% baseline CLIPScore)，显著优于retrieval-only
  (6) **蒸馏小模型(SD3.5L-Turbo)静态降低质量** → MoDM的mix-of-models策略使cache-hit用小模型+缓存的high-quality image初始化，FID远低于standalone小/蒸馏模型(MoDM-SDXL FID 11.85 vs SDXL standalone 16.29 on DiffusionDB)

## TetriServe: Efficiently Serving Mixed DiT Workloads

- baseline方法是什么？
  Baseline方法是**固定degree的sequence parallelism (SP)**，即xDiT对所有请求使用统一的SP度（SP=1/2/4/8），请求一旦以固定并行度开始执行便不可抢占，持有分配的GPU直到完成所有denoising steps。全栈执行例子：算法层DiT模型（FLUX.1-dev 12B参数）以Ulysses attention实现token序列跨GPU分布→系统框架层xDiT固定SP度分发请求，无deadline感知→编译框架层论文未明确说明→kernel调度层NCCL all-to-all collective通信，小分辨率下通信占比超30%导致scaling效率差→硬件架构层8×H100 NVLink 4.0。固定SP的缺陷：低SP度（SP=1/2）对大分辨率（2048×2048）处理太慢导致超时；高SP度（SP=4/8）对小分辨率（256×256）通信开销过大导致head-of-line blocking；无deadline感知导致高SLO violation；RSSP (Resolution-Specific SP)虽然每种分辨率选最优固定SP，但缺乏deadline感知和运行时adaptation，SAR仍低。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**step-level sequence parallelism + round-based deadline-aware scheduling**：在每个denoising step级别动态调整SP度，通过round离散化时间将NP-hard的连续时间调度转化为可解的per-round DP packing问题。全栈执行例子：算法层相同DiT模型→系统框架层TetriServe Scheduler在每个固定时长round内：(a) offline profiled cost model查找每种(分辨率, GPU数)的单步耗时，(b) 为每个请求计算满足deadline的最小GPU分配，(c) DP pack requests入round最大化surviving请求数，(d) GPU Placement Preservation保持同请求跨round同GPU集合，(e) Elastic Scale-Up利用空闲GPU为有余量steps的请求加速→编译框架层论文未明确说明→kernel调度层selective continuous batching仅对同小分辨率请求合并steps减少kernel launch overhead，NCCL process groups预warmup策略平衡启动延迟和显存占用→硬件架构层8×H100 NVLink 4.0 / 4×A40 PCIe 4.0。

  对应解决：(1) **固定SP对小分辨率效率差** → step-level动态SP，小分辨率请求只用1 GPU避免通信开销（通信占比从>30%降至最低）；(2) **固定SP对大分辨率不足** → deadline感知为紧急大分辨率请求临时scale-up到更多GPU加速；(3) **无deadline感知** → round-based DP调度显式建模每个请求的deadline和剩余steps，优先调度即将超时的请求；(4) **不可抢占导致head-of-line blocking** → round边界自然提供preemption点，调度器可在round间重新分配GPU；(5) **RSSP缺乏运行时adaptation** → TetriServe在运行时动态感知deadline紧迫度调整并行度，在tight SLO 1.0× Uniform workload下SAR比RSSP高0.10（0.42 vs 0.32）。


## Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration
- baseline方法是什么？
  Baseline方法分为三类：(1) **Chunked prefill** (vLLM v0.8.5, SGLang v0.4.6)：用固定token budget将prefill chunk与decode token绑在同一个lock-step batch中，所有SM统一执行融合后的prefill+decode kernel。全栈执行例子：算法层标准transformer→Serving框架在SGLang中通过Flashinfer实现prefill/decode attention融合→GPU所有108 SM统一执行单batch→KV cache以PagedAttention管理。chunked prefill的缺陷：prefill compute efficiency仅70%-76%（wave quantization与attention bottleneck）；长prompt按N(N+1)/2次KV reload（16k token prefill以1k chunk的总prefill latency比unchunked高1.13x）；固定token budget难以同时兼顾TTFT和TPOT。(2) **Static overlap** (NanoFlow 1024 chunk)：在chunked prefill基础上做静态kernel overlap pipeline，固定overlap调度，无法适应动态workload。(3) **Disaggregated serving** (SGLang+Mooncake xP-yD)：prefill和decode放到不同GPU/节点，需要跨实例KV/state迁移和手动资源调参，存在KV cache pool减半导致的hit rate下降、GPU静态分配无法适应负载波动等问题。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**intra-GPU prefill-decode spatial-temporal orchestration**：在同一GPU内将prefill和decode拆成可并发执行的两个engine，通过layer-wise prefill、step-wise decode、动态SM partition和SLO-aware scheduling在满足TTFT/TPOT约束下提升GPU利用率。四个组件形成闭环：
  
  (1) **Performance estimator (SRM)**：用SM-scaling roofline model建模compute/memory/network bandwidth随SM数量变化的性能上界，少量offline concurrent sample校准prefill/decode interference，在线统计持续修正。解决了纯profiling开销大的问题（profiling overhead低于1小时，预测/更新开销微秒级）。
  
  (2) **SLO-aware task scheduler**：周期性读取全局状态，预测TTFT/TPOT，重排waiting requests，为每批prefill layer或decode step搜索合适SM分区。prefill队列堆积时提升prefill SM快速清队列，decode TPOT接近SLO边界时减少prefill SM或增加decode SM。解决了chunked prefill的固定token budget dilemma和disaggregated serving的静态资源分配问题。
  
  (3) **Resource manager**：通过libsmctrl_set_stream_mask修改CUDA stream metadata实现微秒级SM重分区（平均4.1us）。解决了已有系统无法快速适应动态workload的问题。
  
  (4) **Concurrent execution engine**：prefill和decode放在独立进程/worker中，共享CPU metadata buffer、统一GPU memory pool (CUDA IPC)、ZeroMQ metadata传递。GPU memory预先分配模型权重和KV cache，CUDA IPC共享避免KV copy。prefill以layer粒度执行，decode以CUDA Graph step执行。避免了chunked prefill的锁步等待和disaggregated serving的跨实例KV迁移开销。
  
  全栈执行例子：算法层标准transformer attention+FFN，论文未修改模型算法→Serving框架层基于SGLang v0.4.6+PyTorch 2.6.0，约4100行Python，prefill/decode拆为两个SGLang worker，ZeroMQ异步传递metadata→编译框架层未修改编译框架，但利用CUDA Graph优化decode iteration（单graph launch），prefill使用piecewise layer-wise执行→kernel调度层通过libsmctrl将prefill CUDA stream绑定到SM mask A（如70个SM），decode CUDA stream绑定到SM mask B（如38个SM），scheduler周期下发repartition命令更新mask→硬件架构层使用NVIDIA A100/H100/H20 GPU，利用MPS spatial sharing，SM mask以16 SM为粒度分组（A100 6种配置，H100 7种配置）。

## Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

- baseline方法是什么？
  Baseline方法分为两类：(1) **Chunked-prefill**（SGLang/SARATHI-Serve）：将prefill拆分为chunks，每个chunk与一次decode迭代融合执行，通过capping token budget（prefill chunk新token数 + decode batch大小）来保证decode SLO。执行例子：请求到达→prefill被拆成多个chunk→每个chunk与decode iteration一同发射到GPU→所有SMs同时执行chunk prefill+decode→chunk prefill需读取之前所有chunk产生的KV cache→decode迭代在每次融合执行后产生新token。全栈路径：算法层无特殊优化→Serving框架在SGLang中通过Flashinfer将prefill attention和decode attention融合为单一kernel→GPU所有SMs统一执行融合kernel→KV cache以PagedAttention方式管理。(2) **Disaggregated serving**（Splitwise/SGLang-PD静态、LoongServe动态）：将prefill和decode分配到不同GPU实例，各自有独立KV cache pool。执行例子：请求到达→prefill实例处理→KV cache迁移或recompute→decode实例迭代产生token。Splitwise GPU分配在初始化时固定；LoongServe根据序列长度动态伸缩GPU数量，但释放原始GPU上的KV cache，跨请求无法复用。

  Baseline的缺陷：
  - Chunked-prefill：过度小的token budget无法打满GPU导致利用率低，过度大的token budget导致TBT超过SLO。而且chunk prefill需重复读取KV cache，当reused context长时（如multi-turn场景>4K tokens），TBT显著膨胀甚至SLO violation。
  - Static disaggregation：GPU静态分配无法适应请求负载波动，decode/prefill实例一方繁忙时另一方空闲。分离的KV cache pool使有效cache容量减半，cache hit rate从36.6%降至4.2%。
  - Dynamic disaggregation (LoongServe)：为支持动态GPU伸缩而立即释放KV cache，跨请求无法复用，multi-turn场景需全量recompute KV cache。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**intra-GPU prefill-decode (PD) multiplexing**：在同一GPU内不同SM上空间复用prefill和decode。核心设计：(1) **空间分区替代时间融合**：decode阶段保留best-fit SMs（如60%）以满足TBT SLO，剩余SMs分配给prefill，避免了chunked-prefill的token budget dilemma。因为prefill和decode独立执行，prefill不再阻塞decode，decode SLO不再受prefill chunk大小影响。(2) **共享内存空间**：两个阶段在同一进程中共享GPU内存和KV cache pool，避免了disaggregation造成的cache pool缩减和hit rate下降。(3) **动态SM重配**：通过GreenContext以微秒级开销调整SM分区比例，适应请求负载和输入长度的动态变化。(4) **Layer-wise prefill执行**：将prefill按transformer layer切分为多个prefill layer，每层独立发射，消除了因prefill和decode延时差异导致的GPU气泡，同时支持长请求抢占短请求以保障TTFT SLO。(5) **Contention-tolerant worst-case估计**：通过solo-run predictor + contention guard提供decode的最坏延时估计，保守但安全地保障SLO。

  论文方法全栈执行例子：
  - 算法层：与baseline相同，标准的transformer attention+FFN，论文未修改模型算法。
  - Serving框架层：MuxWise dispatcher收到请求→estimator用solo-run predictor（公式1/2）预测prefill和decode延时→contention guard查表得到最大slowdown factor→dispatcher选择multiplexing plan（如decode 60% SMs，prefill 40% SMs）→multiplex engine将prefill拆为prefill layers（PLs），计算发射层数NPL=⌈(Td×NT)/TP⌉→发射decode graph到decode stream→异步发射NPL个prefill layers到prefill stream→query-based同步定期轮询CUDA events→prefill完成后合并入decode batch。
  - 编译框架层：论文未修改编译框架。但利用了CUDA Graph优化decode iteration（单graph launch<0.5ms），prefill使用piecewise CUDA graph（按layer切分）。
  - kernel调度层：通过GreenContext将CUDA streams绑定到特定SMs实现intra-process空间分区。SMs按16个为粒度分组（6种配置A100，7种H100），contention guard通过grid-sampling profiling（约7K样本对，12小时/模型-机器对）获得memory bandwidth竞争的最大slowdown（A100≤20%，H100≤30%）。Flashinfer提供融合的prefill+decode attention kernel。
  - 硬件架构层：使用NVIDIA A100/H100/H200 GPU，NVLink互联，论文未修改硬件。利用H100的Thread Block Cluster特性（要求16 SM粒度）。

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

## Hardwired-Neuron Language Processing Units as General-Purpose Cognitive Substrates

- baseline方法是什么？
  Baseline是通用GPU（NVIDIA H100）部署LLM推理和专用LPU（Cerebras WSE-3）预加载权重到片上SRAM。全栈执行例子：算法层gpt-oss 120B FP4 MoE transformer→Serving框架层TensorRT-LLM（H100）执行continuous batching和weight fetching→编译框架层CUDA/Triton编译器生成kernel→kernel调度层GPU warp scheduler执行GEMM kernel，从HBM反复读取权重（每次decoding step都需fetch数百亿参数）→硬件架构层H100 GPU（814 mm²，1.3kW，HBM3 3.35 TB/s带宽）。Baseline缺陷：(1) 权重反复从内存层次读取消耗大部分系统功耗；(2) 指令驱动的GPU控制单元开销（instruction decoding、scheduling、control flow）；(3) GPU软件栈（OS、runtime、library、compilers、frameworks）带来性能抖动和维护成本；(4) AI数据中心2028年预计占美国总电力12%，不可持续。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出HNLPU（Hardwired-Neuron LPU），将LLM权重物理固化到芯片金属导线中，实现极致专用化。两个核心创新：(1) Metal-Embedding (ME) 方法论：将CNN的multiply-accumulate重构为accumulate-multiply-accumulate Hardwired-Neuron架构（结合weight constancy优化、分配律提取公共乘法器、bit-serialization用CSA树以面积换时间），将权重参数从硅器件的2D网格嵌入金属导线的3D拓扑，密度提升15×，光罩成本降低112×；(2) Sea-of-Neurons架构：预制造HN Array（M0-M7层，60层光罩同质化共享），仅通过M8-M11的10层DUV光罩自定义金属嵌入编程权重，使参数更新respins仅需$37M（而非重新设计全部光罩$480M）。全栈执行例子：算法层相同gpt-oss 120B FP4 MoE transformer→无软件栈（纯硬件Token-In-Token-Out，Continuous Batching在硬件中实现，6级pipeline×36层=216最大batch size）→无编译框架（权重固化于金属层，模型=芯片）→无kernel调度层（HN Array直接以常数算术电路计算，零参数fetch开销，MoE稀疏激活仅4/128 experts活跃）→硬件架构层16芯片HNLPU系统（827 mm²/芯片，308W/芯片，4×4 CXL 3.0全连接fabric，总面积13,232 mm²）。解决Baseline缺陷：(1) 权重零memory access——权重物理嵌入金属导线作为电路的一部分，消除GPU反复fetch权重的能耗（1,047×/283×能效比H100/WSE）；(2) 零控制开销——无ISA、无指令译码/调度，近乎100%面积和时间投入有效计算；(3) 零软件栈——无OS/runtime/compiler/framework，确定性Token-In-Token-Out行为，消除软件维护成本和性能抖动；(4) 可持续性——碳足迹比H100集群降低357×（含制造和运营碳排放），TCO降低41.7-80.4×。

## ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression

- baseline方法是什么？
  Baseline是现有lossless compression方法（DFloat11/Huffman、DietGPU/rANS、nvCOMP/rANS）集成到decoupled inference pipeline中。全栈执行例子：算法层使用Huffman或ANS entropy coding对BF16 weight exponent field进行变长编码→编译框架层无特殊修改，使用标准NVCC编译→kernel调度层decompression kernel作为独立pre-processing stage执行，先完整解压权重到global memory buffer（Bitstream Partitioning→Symbol Extraction→Pointer Advancement三阶段），再启动标准cuBLAS_TC GEMM kernel读取解压后的权重→硬件架构层NVIDIA RTX4090/L40S GPU，解压阶段受限于variable-length bitstream的串行解码，SIMT warp内线程diverge，仅达到43.7%（DietGPU/ANS）至76.5%（DFloat11/Huffman）peak memory bandwidth。Baseline缺陷：(1) kernel-level mismatch：变长编码的data-dependent decoding与GPU SIMT lockstep执行模型冲突，导致control-flow divergence和compute underutilization；(2) system-level mismatch：decoupled pipeline的intermediate global memory buffer导致redundant memory traffic，decode阶段compute intensity相比标准GEMM下降62.0%（batch size 32, M=K=4096），使decode在memory-bound regime下性能严重退化；(3) 综合效果：DietGPU/nvCOMP/DFloat11仅达到cuBLAS的0.17×/0.19×/0.28×（RTX4090），压缩带来的memory saving被解压开销完全抵消。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出ZipServ，第一个hardware-aware lossless compression framework实现LLM推理加速。两个核心创新：(1) TCA-TBE fixed-length bitmap编码替代变长entropy coding——利用BF16 exponent的contiguous top-7分布（>95%权重覆盖，99.6%矩阵exponent连续），将每个8×8 tile编码为三个64-bit bitmap（每bit-plane独立），解压只需bitwise OR + POPC + integer ADD，完全消除分支和控制流divergence；(2) ZipGEMM fused decompression-GEMM kernel——将解压和GEMM融合为单一CUDA kernel，权重在register file内解压后直接送入Tensor Core mma指令，消除intermediate global memory buffer。

  全栈执行例子：
  - 算法层：TCA-TBE offline compressor分析每层weight exponent histogram→选择top-7连续exponent→按8×8 FragTile编码为3×64-bit bitmap + PackedSignMantissa (8-bit sign+mantissa) + FullValue fallback buffer。3-bit codeword (001-111) 通过base_exp + codeword算术implicit lookup恢复exponent。平均11.3 bits/element，~1.41×压缩率。
  - Serving框架层：集成vLLM (PyBind11)，stage-aware strategy：decode用fused ZipGEMM（load-compressed, compute-decompressed），prefill用decoupled decompression+cuBLAS_TC（摊销<4% overhead）。压缩权重减少的GPU memory自动分配给KV cache（LLaMA3.1-8B上KV cache从5.07GB→8.60GB, 1.70×），提升batch size和context length。
  - 编译框架层：论文未修改编译框架。ZipGEMM通过nvcc编译为.so，使用内联PTX指令（mma.m16n8k16、LDGSTS.128、LDSM.M88、cp.async）实现低层级硬件控制。
  - kernel调度层：ZipGEMM采用split-K tiling + 两级software pipeline。Coarse-level：tile double buffering，cp.async + __syncthreads() barrier重叠global→shared memory传输与计算。Fine-level：slice-wise interleaving，Tensor Core执行slice i时ALU并发load+decompress slice i+1到register。TCA-TBE的3层tiling（8×8 FT → 16×16 TT → 64×64 BT）直接对齐Tensor Core operand register layout (Ra0–Ra3)，消除runtime坐标变换。Decompressor的spatial bitmap indicator (bitwise OR) + dynamic addressing (POPC prefix sum) + arithmetic exponent reassembly (base_exp + codeword) 全程使用GPU native integer/popcount/shuffle指令，无分支、无shared memory table lookup、无bank conflict（仅~4.7K vs DietGPU百万级）。
  - 硬件架构层：NVIDIA RTX4090/L40S/RTX5090 GPU。用29.3% DRAM read reduction换取ALU增加（LOP3/IADD/POPC），但两级pipeline隐藏decode latency，Tensor Core利用率保持cuBLAS的71.6%。在RTX4090上peak 1.71×、L40S上peak 2.21× kernel speedup over cuBLAS。端到端平均1.22× over vLLM，首次证明lossless compression可以同时提供storage savings和LLM inference acceleration。

  解决Baseline缺陷的映射：
  (1) Kernel-level mismatch → TCA-TBE的fixed-length triple bitmap layout + warp-synchronous bitwise decoding消除divergence，Decompressor全程register-resident操作。
  (2) System-level mismatch → ZipGEMM的fused design将Compute Intensity从decoupled的严重退化提升至约50%高于标准GEMM，将bandwidth saving转化为wall-clock speedup。
  (3) 整体效果 → ZipGEMM是唯一超越cuBLAS_TC的压缩kernel（DietGPU/nvCOMP/DFloat11在RTX4090上仅0.17–0.28×）。stage-aware strategy根据prefill/decode自动切换执行模式保证全阶段性能最优。

## DFVG: A Heterogeneous Architecture for Speculative Decoding with Draft-on-FPGA and Verify-on-GPU

- baseline方法是什么？
  现有speculative decoding系统可分为三类：(1) **Dovetail** (CPU+GPU)：draft model部署在GPU，verifier部署在CPU，通过减少带宽消耗实现1.43× speedup，但draft和verify仍串行执行，无硬件感知动态draft；(2) **DuoDec** (GPU+CPU, DuoDecoding)：draft在CPU、target在GPU，支持hardware-aware draft budgeting，1.67× speedup，但无tree-based verify、无动态branching；(3) **SpecInfer** (Multi-GPU)：tree-based speculative decoding + verification，2.40× speedup，使用静态预定义branch配置构建token tree，不根据模型confidence动态调整branch数和token长度。全栈执行例子（以SpecInfer为代表）：算法层标准autoregressive token-by-token→Serving框架层Multi-GPU间分配draft和verify→编译框架层未修改→kernel调度层token tree的irregular causal mask导致sparse memory access，GPU vectorized computing未充分利用→硬件架构层所有模型部署在homogeneous GPU上，draft model（轻量）和verify model（重量）共享相同GPU资源，造成memory contention和compute underutilization（LLaMA-7B在RTX4090上utilization仅51.72%，有时<10%），draft model的bandwidth-intensive特性与verify model的compute-intensive特性在homogeneous硬件上冲突。

  Baseline的三个核心缺陷：(1) **资源利用不均衡**：draft model（lightweight, latency-sensitive, memory-light, bandwidth-intensive）和verify model（large, compute-intensive, memory-bandwidth-bound）在同构硬件（GPU-only/CPU-only）上资源需求冲突，memory contention + serialized workloads导致低利用率；(2) **固定pattern token tree低效**：静态预定义branch策略无法根据model confidence和hardware resource动态调整——高confidence位置无法增加branch fully exploit certainty，低confidence位置仍生成大量低质candidate导致极低acceptance rate并可能超出硬件处理能力；(3) **解耦执行和频繁rollback**：draft和verify串行/独立执行无充分协调，阶段切换时硬件idle（pipeline bubbles），token rejection后必须discard output从last accepted prefix重新生成，跨设备场景下transfer latency放大rollback开销。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**DFVG**，第一个完整的FPGA-GPU heterogeneous speculative decoding architecture。三个核心贡献对应三个baseline缺陷：

  **(1) Heterogeneous Architecture Design（解决资源不均衡）**：将speculative decoding拆分为FPGA-based drafting和GPU-based verification。FPGA低延迟streaming计算 + fine-grained parallelism非常适合draft stage（token生成可deeply pipelined），GPU高吞吐tensor operations适合verification（batch parallel forward pass）。FPGA上部署Multi Compute Core Overlay Processor（Verilog HDL, 300MHz, V80 FPGA）——systolic PE array执行draft model forward pass，support branch concatenation（多weight buffer使PE按branch选择路径）、DSP packing（单DSP双BF16×BF16乘法，吞吐翻倍）。GPU上执行TreeSort-Verify做batch parallel verification。FPGA draft model功耗仅75W runtime（远低于V80 TDP 190W），减轻GPU压力使其专注高吞吐verification。

  **(2) Hardware-Aware Dynamic Draft Generation（解决固定token tree低效）**：提出ADAPT算法——budget-constrained integer programming动态构建token tree。目标函数max Σ p_{i,j,l}·x_{i,j,l}（最大化expected accepted tokens），约束：总branch≤B（FPGA compute budget）、每层branch≤k_max（FPGA parallel support limit，设为8/16/32整数倍）、depth≥D_min=⌈T_verify/T_draft⌉（确保pipeline overlap）。求解采用Temperature-Controlled Probabilistic Sampling：Path Cumulative Probability通过softmax temperature T归一化后Gumbel sampling做非重复选择，T→0退化为top-k deterministic selection。算法复杂度O(D·k_max·log k_max)满足实时推理要求。实验结果：acceptance rate 75%-85%，draft length呈long-tail分布自动适配不同iteration的difficulty。

  **(3) Tightly-Coupled Heterogeneous Pipeline（解决解耦执行和rollback开销）**：Stage-Decoupled Scheduling实现FPGA draft和GPU verify的overlapped execution——FPGA持续生成drafts（即使部分将被rejected），GPU从不idle（either verifying or forwarding）。interrupt-driven coordination：FPGA写draft token到shared host buffer→update BAR status→raise interrupt to CPU→CPU trigger DMA→GPU fetch并verify→GPU返回accepted prefix→FPGA对比local sequence检测rollback→若required则reset KV cache从verified prefix恢复。通信仅传输compact token IDs+status metadata，占wall time的1.08%-3.2%。Pipe-Overlap ablation贡献：3.08×→3.26× overall speedup。

  论文方法全栈执行例子（以Qwen3-0.6B/8B + V80/RTX4090为例）：
  - 算法层：ADAPT algorithm每iteration求解budget-constrained tree construction。给定prefix→FPGA draft model (Qwen3-0.6B) forward pass输出vocab-level probability→temperature-controlled Gumbel sampling选k_i个candidate per layer→构建depth D≥⌈T_verify/T_draft⌉的token tree。
  - Serving框架层：CPU host controller（C++）协调跨设备pipeline。FPGA生成candidate tokens写入shared host memory→interrupt+PCIe DMA→GPU读取→TreeSort-Verify做block-parallel attention→acceptance decision (Eq.3) via probability ratio→GPU返回accepted prefix→FPGA检测rollback并恢复。Ping-pong buffering确保computation和communication overlap。
  - 编译框架层：FPGA使用Xilinx Vivado 2024.1综合Verilog HDL设计→生成bitstream。GPU使用CUDA 12.1编译TreeSort-Verify kernel和cuBLAS GEMM调用。论文未修改现有编译框架。
  - kernel调度层：FPGA侧Multi-Branch Mapping——shared prefix使多branch共享Linear weight loading→Q×K^T复用prefix+K仅改loading address→S×V最后round accumulation归并。PE array通过ping-pong KER/IFM loading重叠compute和data loading，matrix multiplication loading/computation均达86.2%-97.5%效率。GPU侧TreeSort-Verify——对token tree节点path-packing重排序→划分K个连续block→每个block独立调用cuBLAS GEMM (block-diagonal causal mask)，eliminate irregular sparse mask的memory divergence。
  - 硬件架构层：AMD V80 FPGA (300MHz, 10848 DSPs, HBM) + NVIDIA RTX 4090 (2230MHz, 512 Tensor Cores, 24GB) + PCIe Gen4×16 (64GB/s) + Intel Xeon 4310 host CPU。FPGA overlay processor占89.6% LUT、90.9% FF、8192 DSPs、18MB BRAM、67MB URAM (主要为KV-cache)。V80 runtime功耗仅75W。

  最终效果：Qwen3-8B上3.26× speedup、5.79× energy efficiency（vs AR baseline），显著超越SpecInfer (2.40×)、DuoDec (1.67×)、Dovetail (1.43×)。energy efficiency额外约1.7×来自FPGA极低runtime功耗（75W vs RTX4090 236W）。scaling perspective：从LLaMA-7B到OPT-13B，existing methods speedup明显下降，DFVG维持稳定加速（得益于FPGA-GPU重叠避免pipeline idling）。

## PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel

- baseline方法是什么？
  Baseline分为两类：(1) **Query-centric kernels**（FlashAttention v2.5.9, FlashInfer v0.2.5）：每个query和对应KV cache独立映射到一个CTA（one-query-per-CTA），GPU上多个CTA并行执行decode attention。全栈执行例子：算法层标准multi-head attention→Serving框架层vLLM v0.9.0 continuous batching→kernel调度层FlashAttention将每个query分配到一个CTA，CTA从global memory加载完整KV cache到shared memory，以tiling pipeline（tile size固定m=64,n=128 for FlashAttention; m=16,n=128 for FlashInfer）执行QK^T和PV→硬件架构层A100 GPU的SMs执行CTA，KV cache从HBM→L2→shared memory→register逐级搬运。NCU profiling显示FlashAttention的KV cache traffic比理论最小值高4.3-8.7×，比PAT高4.1-9.5×。(2) **KV-centric kernels**（FastTree, RelayAttention, DeFT, Cascade Inference）：将共享prefix的多个query与其KV cache放入同一CTA以减少重复KV读。全栈执行例子：算法层标准attention→Serving框架层vLLM block table管理→kernel调度层FastTree用compute-oriented cost model打包shared KV queries到CTA，用固定两种tile configs (64,32)和(16,32)串行执行→硬件架构层仍受限于one-size-fits-all tile design导致shared memory padding浪费（query数<𝑚时padding填充）和tail execution bubble（KV长度差异大时最后完成的CTA拖慢整体）。RelayAttention仅支持单层first-level prefix；RelayAttention++扩展到multi-level但依赖L2 cache而非kernel级复用；DeFT用fixed (32,16) tile和load balancing；Cascade Inference用fixed settings打包。

  Baseline的核心缺陷：(a) Query-centric内核在batch内多query共享prefix时重复从global memory加载相同KV blocks，是memory-bound decode attention的主要瓶颈；(b) KV-centric内核的one-size-fits-all tile设计无法同时适配动态变化的CTA query数和KV长度，造成shared memory/register浪费（I_mem）和execution bubble（I_exe）；(c) 两者均未充分利用workload-level shared prefix结构来系统性地减少global memory bandwidth压力。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出PAT，一个prefix-aware attention kernel实现，遵循pack-forward-merge执行范式。四个核心设计对应对冲baseline缺陷：

  **(1) Memory-centric pack scheduler（解决query-centric的redundant KV loads和KV-centric的compute-oriented packing）**：将vLLM block table转为prefix tree（internal node=shared prefix段含𝑙 tokens和𝑠个query），用memory-centric profit model比较split/merge两种scheme——intra-node profit=(s-1)*l*d vs overhead=8*s*d（intermediate写回/读回）；inter-node profit比较merge child到parent CTA的增量收益4*s_j*d - l_u*d，当child query数足够多且parent prefix短时merge更优。TreeHeuristic算法以𝑂(|𝑉|+|𝐸|)线性复杂度遍历prefix tree生成CTA partition。对比FastTree的compute-oriented cost model，PAT的memory-centric model使memory read/write分别降低10.9%和16.7%。

  **(2) Multi-tile kernel + runtime tile selector（解决KV-centric的one-size-fits-all资源浪费）**：Offline通过三约束求解可行(m,n) tile set——① register/shared memory上界约束（m*h*b + n*h*b + 中间结果 ≤ S_smem, per-thread register ≤ S_reg_thr, 总register ≤ S_register）；② bandwidth lower bound（n ≥ LB/(S*C*h*b)，保证in-flight data覆盖memory latency以饱和带宽）；③ CUTLASS constraint（m,n为2的幂且≥16）。A100上得11组可行配置，H100上得12组。Runtime tile selector constant-time决策：m用round-up规则（选≥当前CTA query数的最小可行m避免padding→消除I_mem）；n根据KV length profiling做piecewise决策（长KV偏大n降低CTA concurrency减少tail bubble→消除I_exe；短KV偏小n避免最后tile compute bubble）。相比于PAT-fixed（固定(64,128) tile），PAT的multi-tile降低attention latency 39%。

  **(3) Multi-stream forward + long-KV split（解决execution bubbles）**：为每种active tile配置创建独立CUDA stream并行执行，kernel launch overhead与前置kernel执行overlap。Long-KV split将KV length超过batch均值的CTA沿KV维拆分为多个子CTA，缩短最后完成CTA的时间。相比于PAT-serial（串行multi-kernel），multi-stream降低attention latency 4.8%。PTX profiling显示multi-stream显著减少execution bubble。

  **(4) Lightweight merge kernel with online softmax（解决multi-CTA splitting的merge overhead）**：每个CTA输出per-query/per-head的partial max score、log-sum-exp accumulator和partial value-weighted sum到global memory，merge kernel读取同一query所有partial intermediates以online softmax归并max/sum再归一化partial sum。Merge overhead已纳入pack scheduler的profit model。

  论文方法全栈执行例子（以Qwen3-8B + A100 + vLLM + Conversation trace为例）：
  - 算法层：标准GQA transformer attention (32 heads / 8 KV heads, head dim=128, FP16)，论文未修改算法
  - Serving框架层：vLLM v0.9.0 continuous batching + paged KV cache。每次decode step开始：vLLM维护batch中每request的block table → PAT pack scheduler读取logical block IDs → 构建prefix tree（Conversation trace有三层prefix: 45/351/2126 tokens for Qwen3 tokenizer） → TreeHeuristic遍历tree按profit model决策打包CTA → lazy update在block table未变时跨iteration复用
  - kernel调度层：packed CTA送入forward stage → tile selector为每个CTA选(m,n)（如q=20选m=32; KV=4096选n=128） → CTAs按tile config分组进入各自CUDA stream → cp_async+double buffering搬运K/V tile从global→shared memory → CTA内多query共享同一shared-memory KV tile → QK^T、online softmax stats、PV累加 → partial results写回global memory → merge kernel读回partial results以online softmax合并 → 输出final attention output
  - 硬件架构层：A100-80GB GPU (108 SM, 40MB L2, 1935 GB/s HBM bandwidth)。packing使共享KV block仅从HBM加载一次（而非每query一次）→ 直接减少memory-bound decode attention的主开销。Multi-tile配置在A100上维持83%-86% bandwidth utilization，multi-stream在多种tile config间并行减少execution bubble

  最终效果：synthetic batch下平均降低attention latency 53.5%；真实ToolAgent/Conversation trace下端到端TPOT降低17.0-93.1%（vs FlashAttention/FlashInfer/RelayAttention++）；分布式下Qwen2.5-72B-Instruct TPOT降低14.3-26.7%；MoE下Qwen3-30B-A3B TPOT降低5.53-16.9%。

## BAT: Efficient Generative Recommender Serving with Bipartite Attention

- baseline方法是什么？
  现有GR serving系统采用User-as-prefix attention作为事实标准：将输入序列组织为[U, I1,...,I_N, Instr]，仅user profile token U的KV cache可跨同一用户的multi-turn请求复用。全栈执行例子：算法层标准causal self-attention对所有token执行QK^T→Serving框架层vLLM管理paged KV cache，prefix caching以LRU策略在CPU/GPU memory中存储user prefix cache→编译框架层未特殊修改→kernel调度层FlashInfer提供融合attention kernel→硬件架构层A100 GPU处理compute-bound prefill。Baseline三个核心缺陷：(1) **User cache hit rate极低（仅18%）**——大规模用户基（10^8）中大量用户不活跃，item token占总token 33%以上但无法被User-as-prefix共享；(2) **User cache存储不可行**——Qwen2-1.5B模型下单个user平均1000 tokens需约29MB KV cache，缓存10^8用户需超过2.9PB存储，远超单机/集群本地内存容量；(3) **热门item跨用户共享机会未被利用**——trace分析显示约90%的访问集中在top 10%的热门item，但这些item的KV cache在User-as-prefix中依赖前置user context而无法跨用户复用。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**Bat**，基于关键洞察"user和item语义在推荐prompt中排列不变"设计Bipartite Attention，并配套co-design disaggregated KV cache pool、HRCS item cache placement和hotness-aware prompt scheduling。

  核心设计及其与Baseline缺陷的映射：

  **(1) Bipartite Attention算法（解决user cache共享受限问题）**：利用排列不变性提出Item-as-prefix attention作为User-as-prefix的替代——将输入重组为[I1,...,I_N, U, Instr]，调整attention mask使item间无cross-attention（遵循HSTU原则），调整position encoding使所有item共享相同起始位置ID，使item KV cache独立于user context可跨用户自由共享。相比baseline仅能复用user cache，Bipartite Attention解锁了item cache复用——热门item可跨成千上万用户共享，cache hit rate从18%提升到最高58%。

  **(2) Disaggregated KV Cache Pool（解决user cache存储不可行问题）**：设计compute-storage分离架构——KV cache worker管理paged memory pool（CPU memory + GPU memory），cache meta service集中追踪索引和热度，inference worker专注forward computation。User-prefix cache和item-prefix cache作为独立组件被主动管理而非被动存储。Item cache仅需约287GB（1M items, Qwen2-1.5B）vs user cache的430TB（10M users），使本地内存存储成为可行。

  **(3) HRCS Item Cache Placement（解决item cache引入的额外内存开销和跨节点通信问题）**：利用item access分布的skewness——通过offline profiling网络带宽B和prefill computation time t确定max allowed communication ratio R_max→按item access frequency CDF计算hotspot replication ratio r→top-r% hot item在所有KV cache worker间完整复制（eliminate IO bottleneck for high-frequency access），其余cold item均匀分片（minimize memory usage per machine）。相比fully replicate节省内存（在100Gbps下Bat throughput比Bat-Replicate高16%），相比fully shard减少网络开销（在10Gbps下Bat-Hash throughput仅为Bat-Replicate的78%）。

  **(4) Hotness-aware Prompt Scheduling（解决prefix selection非trivial问题）**：基于sliding-window频率估计𝑓_u（empirically validated用户连续行为具有相似性），greedy policy决策prefix选择——当user token数≥item token数且user的estimated frequency > cache中最冷user page频率时选User-as-prefix（高频用户的user cache值得保留），否则fallback到Item-as-prefix（低频用户利用item cache避免compulsory miss）。相比cache-agnostic policy（naively选token数更多的做prefix），hotness-aware在user cache空间受限时（25GB）throughput和cache hit rate显著更高。

  论文方法全栈执行例子（以Qwen2-1.5B, Books dataset, 4-node A100集群为例）：
  - 算法层：Bipartite Attention根据scheduler决策选择User-as-prefix（输入=[U, I1,...,I_N, Instr], Attn(q_{I,Instr}, k_{I,Instr}∪k_U, v_{I,Instr}∪v_U)）或Item-as-prefix（输入=[I1,...,I_N, U, Instr], Attn(q_{U,Instr}, k_{U,Instr}∪k_I, v_{U,Instr}∪v_I)）。Attention mask屏蔽item间cross-attention，position encoding使所有item共享相同起始位置。
  - Serving框架层：vLLM + FlashInfer + 自定义Bat组件。Hotness-aware prompt scheduler接收retrieval stage的requests（user ID + 100 candidate item IDs）→查询cache meta service获取cache状态和hotness→根据greedy rule决策前缀类型→concatenate prompts成batch→load-balanced分发到inference workers→cache meta service协调KV cache worker执行物理KV cache传输（RDMA/DMA）。
  - 编译框架层：论文未修改编译框架。
  - kernel调度层：FlashInfer提供优化attention kernel，KV cache以PagedAttention兼容的fixed-size pages管理。HRCS通过offline polynomial regression model估计prefill time，online按frequency CDF动态决定replication ratio。GPUDirect RDMA用于跨节点KV cache传输。
  - 硬件架构层：4节点×A100-40GB GPU (PCIe 3.0x16) + 100Gbps网络互联。生产环境16节点×H20 GPU + 200Gbps网络。

  最终效果：Bat在多个数据集和模型上相比UP(User-as-prefix)提升throughput最多1.6×，相比RE(Recomputation)提升最多2.3×；cache hit rate最高达58%；减少total computation最多58%；在200ms P99 latency SLO下相比UP和RE分别sustain 1.47×和1.57×更高request rate。

## Laser: Unlocking Layer-Level Scheduling for Efficient Multi-SLO LLM Serving

- baseline方法是什么？
  Baseline方法分为两类：(1) **Sarathi-Serve**：prefill-decode aggregation架构 + iteration-level chunked prefill + EDF调度。prefill被按固定token budget拆分为chunks，与decode iteration融合执行；为满足最严格TBT（50ms），每个instance限制prefill chunk size，导致prefill SLO violation显著。执行例子：请求到达→chunked prefill与decode融合batch→所有GPU SMs统一执行完整iteration（穿过所有模型layers）→下一iteration才能加入新请求或切换。全栈路径：算法层标准transformer→Serving框架层vLLM/Sarathi-Serve continuous batching + chunked prefill→编译框架层未修改→kernel调度层FlashAttention/FlashInfer融合attention kernel→硬件架构层A100 GPU，所有SMs统一执行。(2) **DistServe**：prefill-decode disaggregation架构，prefill和decode放到专用instance，但每个instance内部使用iteration-level scheduling + EDF。解决了Sarathi-Serve的prefill-decode资源争用，但无法在iteration内差异化执行multi-SLO请求。执行例子：请求到达→prefill instance处理（完整iteration）→KV cache迁移→decode instance批处理（完整iteration，统一TBT约束所有请求）。

  Baseline的缺陷：
  - **Prefill侧（Inflexible prefill chunking）**：chunk size是固定trade-off——增大chunk size（100→3200）使per-iteration latency增16×但per-token latency降45.4%。大chunk更高效但垄断compute导致head-of-line blocking（latency-critical请求被迫等待当前chunk完整iteration结束）；小chunk响应快但GPU utilization低。iteration-level无法在同一iteration内动态调整chunk大小或合并请求。
  - **Decode侧（Unified decode batching）**：不同SLO请求的TBT容忍度差异显著（relaxed请求可容忍7×更多并发），但iteration-level scheduler以统一TBT为目标配置batch size，所有请求被迫服从最严格SLO，严重限制了relaxed请求的batch容量，导致overall SLO violation增加。
  - **实例间调度**：DistServe的disaggregation改善了阶段隔离，但decode instance的batch sizing仍受multi-SLO constraint限制，无法充分利用relaxed请求的slack。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**layer-level scheduling**，将LLM serving调度粒度从完整的iteration细化到单个transformer layer，并配套设计Laser系统实现intra-instance layer-level scheduling与inter-instance SLO-aware dispatching的联合优化。

  核心设计及其与Baseline缺陷的映射：

  **(1) Layer-level chunked prefill（解决prefill chunking不灵活问题）**：prefill Scheduler在每个layer边界评估新请求的slack时间（TTFT target - estimated prefill latency）。若当前chunk剩余iteration time超过新请求slack，则在layer边界保存正在执行请求的intermediate state到GPU intermediate cache，然后决定：(a)把新请求推进到同一layer位置后与旧请求合并，提高batch size和GPU utilization；或(b)直接优先执行新请求，避免SLO violation。若无抢占且queue为空，动态合并新请求到当前chunk以提升效率。队列按EDF排序。这一机制同时解决了head-of-line blocking（可随时在layer边界抢占）和limited chunk-size execution（可在同一iteration内动态合并请求形成更大有效chunk）两个问题。

  **(2) Layer-level decode batching（解决统一decode batch问题）**：decode Planner为每个请求生成execution plan（L: 每iteration执行层数，O: 调度offset），利用不同请求的SLO slack差异：(a) strict请求（如50ms TBT）每iteration执行全部N层确保低延迟；(b) relaxed请求（如200ms TBT）每iteration只执行N/2或N/4层，多轮完成完整forward pass，释放batch slot给更多relaxed请求。Planner通过模块化latency model（stateless模块用分段线性函数，stateful attention用线性函数，Pearson系数>0.78）估计per-iteration latency，贪心策略优先减少最relaxed请求的层数并选offset平衡同group负载。更新仅在critical events触发且overlap执行，开销<3.8%。

  **(3) Inter-instance request dispatching（解决实例间SLO异构排布问题）**：prefill侧Global Controller选TTFT slack最大的实例，最大化SLO合并弹性。decode侧采用SLO-homogeneous group management——将decode instance按TBT target分组，新请求优先分到相近SLO group中TBT increment最小的instance，允许跨group分配以平衡负载。实例按arrival rate和SLO-compliant batch size的比值动态调整group大小。去中心化performance evaluation——每个decode instance本地运行ExecPlan评估接纳请求后的impact，仅向Controller返回aggregated TBT increment，避免中央调度瓶颈。

  **(4) Intermediate state management + latency model（支撑layer-level切换的系统机制）**：每个instance维护GPU intermediate cache（prefill 16384 tokens，decode 2048 tokens，Llama-70B上<256MB）存layer-level hidden states；fused CUDA kernel合并state caching/retrieval；KV cache migration按layer粒度异步overlap prefill computation。Latency model通过offline profiling（≤2秒）拟合参数，预测准确率94.6%-98.6%。

  论文方法全栈执行例子（以Qwen-14B + ShareGPT/HumanEval/LongBench mixed workload, 4×A100 80GB集群为例）：

  - **算法层**：标准transformer模型，论文未修改attention/FFN算子或模型结构。
  - **Serving框架层**：vLLM + Ray。Global Controller profiling→请求到达→prefill instance的Scheduler评估slack→layer boundary抢占/合并→EDF队列调度→prefill Executor执行layer-level chunked prefill→KV cache按layer粒度异步迁移→decode instance的Planner构建execution plan（L/O）→latency analysis估计per-iteration latency→贪心调整→Executor按plan在layer边界切换请求→intermediate cache管理intermediate states。
  - **编译框架层**：论文未修改编译框架。利用CUDA fused kernel优化state caching/retrieval。
  - **kernel调度层**：FlashInfer提供attention kernel。Laser新增的layer-level intermediate state switching通过fused CUDA kernel实现（合并caching和retrieval），switching overhead<1.5%。KV cache migration按layer粒度与prefill computation异步overlap。
  - **硬件架构层**：4台主机×4×A100 80GB，机内NVLink，机间100 Gbps LAN。1/2/4-way tensor parallelism分别对应14B/32B/70B模型。

  最终效果：Laser相比Sarathi-Serve和DistServe在Qwen-14B/32B和Llama-70B上分别提升goodput 43.4%/68.9%/56.6%。relaxed请求占比高时改进>86%；tight SLO (0.8×)下vs Sarathi-Serve goodput gain最高6.25×；99% attainment目标下throughput提升最高1.85×。prefill SLO violation rate最多降低21.6%，平均TTFT降低>10%；decode TBT violation rate降低最多17.2%（6.7% from layer-level batching + 10.5% from group-based assignment）。

## JanusQuant: Accurate and Efficient 2-bit KV Cache Quantization for Long-context Inference

- baseline方法是什么？
  Baseline方法分为两类：(1) **效率导向2-bit量化系统（Atom/QServe）**：采用per-token group quantization等硬件友好策略，QServe使用类似SmoothQuant的per-channel smoothing并通过离线校准将smoothing factor融入前层权重以降低runtime开销。全栈执行例子：算法层per-token group quantization→系统框架层离线校准smoothing factor融合到权重→编译框架层论文未明确说明→kernel调度层标准量化kernel，row-major访存→硬件架构层A100 GPU。缺陷：2-bit下每组仅4个量化值，K cache中outlier channel放大同组误差（Atom MSE 1.0352 vs QServe 0.5552 vs ideal 0.3734）；离线校准的静态smoothing factor无法适应不同请求和序列长度的per-channel absmax波动（可超4×），导致2-bit精度崩塌（Llama2-7B perplexity: Atom-2bit 103.05, QServe-2bit 11.36 vs FP16 5.47）。(2) **准确率导向2-bit量化方法（SKVQ/KVQuant/KIVI）**：通过channel reordering、dense-and-sparse quantization、recent token reservation维持2-bit准确率。全栈执行例子（以SKVQ attention layer为例）：算法层channel reordering + sliding-window recent token reservation + outlier detection/extraction→系统框架层Transformers/PyTorch→编译框架层论文未明确说明→kernel调度层多步分立kernel：retain recent FP16 tokens (tensor concatenation)→detect/extract outliers→quantize剩余KV→separate dequantization kernel→attention kernel→硬件架构层A100 GPU。缺陷：outlier handling overhead（prefill ~20%）、caching overhead from tensor concatenation（prefill ~20% / decode ~15%）、separate dequantization kernel overhead（prefill ~45% / decode ~80%），三项合计prefill阶段超85% runtime，decoding阶段超97% runtime，端到端延迟甚至高于FP16。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**JanusQuant**，通过RtSmooth runtime smoothing算法联合FAVP fast absmax positioning、memory-efficient ring buffer token cache和mixed-precision attention kernel三项系统优化的算法-系统协同设计。

  论文方法全栈执行例子（以Llama2-7B、128K context、单张A100-40GB decoding为例）：
  - **算法层**：RtSmooth对K cache执行per-token smoothing transformation + per-channel group quantization，对V cache执行per-token group quantization。每个token的smoothing factor = max(|K_i|)^0.5在运行时动态计算，缩小group内值域→降低quantization scale→降低error upper bound（ε_smooth ≤ s_smooth/2 < s/2）。Decoding中每g=32步将FP16 buffered recent tokens量化。无需channel reordering或sparse outlier path，保留positional alignment。
  - **系统框架层**：约2500行Python实现，基于PyTorch和Transformers。Custom attention module继承PyTorch nn.Module API，通过Pybind和FlashInfer调用CUDA kernels。支持Llama/Mistral/Vicuna/Qwen模型族。Pre-allocated ring buffer token cache替代sliding-window concatenation。
  - **编译框架层**：论文未明确说明。CUDA kernels编译为standalone .so。
  - **kernel调度层**：约3500行CUDA/C++。三类kernel：(1) Fused smoothing+quantization kernel：FAVP限制absmax计算到离线校准的稀疏channel集（<2% channels），将smoothing transformation、参数计算、KV cache INT2 packing融合，避免4.43× naive runtime smoothing overhead。(2) Ring buffer token cache kernel：预分配buffer+指针管理，以分段量化替代频繁tensor concatenation。(3) Mixed-precision attention kernel：将INT2 dequantization fused into attention，高效unpacking (lop3/or/sub, 3指令处理2值) + unified parameter block layout (memory transactions 20→8)。Task parallelism and async execution重叠计算与访存。
  - **硬件架构层**：NVIDIA A100-PCIE-40GB（单卡），PyTorch 2.4.0 + CUDA 12.6。

  对应解决Baseline缺陷：
  **(1) Atom/QServe 2-bit精度崩塌**：RtSmooth在运行时动态计算per-token smoothing factor适应不同请求和序列长度的KV cache分布变化，而非依赖静态离线校准→Llama2-7B perplexity 5.80 vs QServe-2bit 11.36，LongBench 8任务平均分保留99% FP16 accuracy。
  **(2) SKVQ/KVQuant outlier handling overhead (~20%)**：RtSmooth不需要检测和抽取outlier到单独稀疏路径，而是通过smoothing将outlier影响均化到group内→消除explicit outlier detection/extraction开销。FAVP将absmax扫描限制在<2% channels，进一步降低runtime smoothing成本。
  **(3) SKVQ/KIVI recent token reservation overhead (prefill ~20% / decode ~15%)**：Ring buffer预分配+指针切换替代tensor concatenation，避免decoding中每次token追加触发内存reallocation和data copy。缓存容量n*g token时每次decoding至少保留(n-1)*g个FP16 recent token。128K context下额外FP16 token仅占0.05%总token数。
  **(4) SKVQ/KVQuant separate dequantization overhead (~45%-80%)**：Mixed-precision attention kernel将dequantization融合进attention，消除独立dequantization kernel launch和全局内存往返。INT2-to-FP16高效unpacking降低compute intensity避免kernel成为compute-bound，unified parameter block减少memory transactions。Decoding 100 tokens总延迟：5.64× over FA2, 5.84× over KIVI, 4.45× over QServe, 2.50× over DuoAttention。KV cache memory 5.30× reduction over FP16。
  **(5) 整体trade-off**：JanusQuant接受额外smoothing factor存储（avg bit-width 3.008 vs 3.000 for KIVI/SKVQ）换取端到端加速，与KVQuant non-uniform方案（avg 2.320 bits）相比在压缩率上不占优但在实际decoding速度上显著领先。

## High-Throughput Non-Uniformly Quantized 3-bit LLM Inference

- baseline方法是什么？
  Baseline方法分为两类：(1) Non-uniform 3-bit quantization方法（SqueezeLLM、Any-Precision LLM、Bitsandbytes），使用K-means clustering将FP16权重映射为3-bit index+per-row centroids，在CUDA cores上完成dequantization和matrix-vector multiplication；(2) Uniform quantization方法（GPTQ），使用线性映射将权重量化为低比特整数。

  Baseline全栈执行例子（以SqueezeLLM 3-bit在A100上跑OPT-30B为例）：
  - **算法层**：K-means clustering将每行权重映射为8个FP16 centroids+3-bit index。dequantization做centroid[C[Wq]] pointer chasing lookup恢复FP16 weight，matmul在CUDA cores上执行matrix-vector multiplication（每token顺序乘权重矩阵的一列）。
  - **系统框架层**：论文未明确说明，SqueezeLLM提供Python inference API通过HuggingFace Transformers调用。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：SqueezeLLM dequantization和matmul均在CUDA cores上完成，bit packing采用padding（10个3-bit元素塞32-bit word，浪费2 bits）或spanning（32个3-bit元素跨3个32-bit word，破坏coalesced memory access）。dequantization产生大量bitwise/type-conversion指令和cache-unfriendly indirect memory access（decoding W†=C[Wq]需pointer chasing）。Tensor Cores完全未被使用，因为3-bit sub-byte operand layout与Tensor Core所需interleaved pattern不兼容。
  - **硬件架构层**：NVIDIA A100/L40 GPU。

  Baseline的核心缺陷：
  (1) **内存节省未转化为加速**：SqueezeLLM在OPT-30B上memory reduction 4.07×但latency反增3.01×（vs FP16 baseline），matmul时间占GPU总时间92%
  (2) **低效bit packing**：3-bit与32/64-bit GPU word不对齐，padding浪费内存带宽，spanning破坏coalesced access引入分支和warp divergence
  (3) **高dequantization overhead**：CUDA cores上centroid pointer chasing lookup产生大量指令，batch增大时instruction count指数增长（图3）
  (4) **Tensor Core underutilization**：dequantization和matmul均压在CUDA cores，Tensor Cores空闲等待（pipeline bubble），GPU最强算力单元未参与核心matmul

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**Quantix**，一个"离线布局转换+在线fused kernel"的GPU执行框架，在不改变non-uniform量化模型accuracy的前提下将内存节省转化为实际推理加速。

  论文方法全栈执行例子（以Quantix 3-bit在L40上跑LLaMA-65B为例）：
  - **算法层**：沿用SqueezeLLM等non-uniform quantization的K-means clustering结果（Wq和C）。Quantix的offline bit shuffling对Wq做两步变换：(a) bit dividing将每个3-bit index拆为1-bit matrix Wq,1和2-bit matrix Wq,2，使32个1-bit元素恰好填32-bit word、32个2-bit元素恰好填64-bit word，消除padding浪费和spanning跨界；(b) bit mapping按64×64 warp tile→16×16 Tensor Core tile层次，将每个thread负责的元素收集为连续linear segments W1'/W2'，确保128-bit coalesced vectorized access且匹配Tensor Core operand layout。bit shuffling是lossless w.r.t. quantized model（不改变centroids），完全保留原non-uniform量化精度。
  - **系统框架层**：Quantix fused kernel集成进HuggingFace Transformers替换SqueezeLLM默认inference backend。对uniform baselines (GPTQ/Marlin)使用AutoGPTQ library。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：Quantix fused dequantization-matmul kernel将数据搬运、dequantization和MMA融合为单一CUDA kernel的执行pipeline。具体：(a) inter-tile层：shared memory double buffering (Smem0/Smem1)，cp.async 128-bit预取下一K-tile的W1'/W2'/A到shared memory，与当前tile计算重叠；(b) intra-tile层：register double buffering (Reg0/Reg1)，CUDA cores在registers中做bit concatenation [1-bit]+[2-bit]→3-bit index→shift+mask按qi=(R>>3i)&0x7并行提取→centroid lookup得到FP16 W†，Tensor Cores同时消费另一register buffer的W†执行MMA；(c) in-register dequantization使FP16 weight仅在registers中存在并直接进入Tensor Core，消除了baseline中dequantize→write memory→read memory→matmul的多级内存路径和cache miss；(d) Split-K提升小矩阵并行度；(e) 128-bit UINT4 vectorized memory access最大化memory bandwidth。
  - **硬件架构层**：NVIDIA L40 GPU（面向inference优化）和A100 GPU。L40上kernel平均speedup 4.82× over FP16 cuBLAS。GPU utilization profiling显示Quantix维持>90% cache hit rate（baseline在batch增大时降至接近0%），有效规避"memory wall"。

  对应解决Baseline缺陷：
  **(1) 内存节省未转化为加速 → 全栈co-design将内存节省转为吞吐**：fused kernel+bit shuffling使3-bit Quantix在L40在kernel-level平均4.82× over FP16 cuBLAS，端到端LLaMA-65B可单GPU运行（FP16不行），最高相对SqueezeLLM 11.46× speedup
  **(2) 低效bit packing → bit dividing将奇数bit拆为1-bit+2-bit双路**：1/2均为32/64的因子，天然对齐GPU word边界。消除padding的内存浪费和spanning的跨界访问、branching和warp divergence。对比naive padding（10个3-bit→32-bit浪费2 bits）和spanning（32个3-bit→跨3个word），bit dividing后每路均perfectly packed
  **(3) 高dequantization overhead → in-register dequantization消除pointer chasing和cache miss**：bit concatenation和centroid indexing在registers内部完成，使用shift+mask (qi=(R>>3i)&0x7) 无分支；centroids也保持在registers中，无需indirect memory access。Ablation显示in-register dequantization是最大贡献组件（移除后性能降至40%）
  **(4) Tensor Core underutilization → fused kernel让CUDA cores做dequantization准备，Tensor Cores做matmul**：hierarchical pipeline使dequantization latency被Tensor Core compute隐藏。GPU utilization分析显示Quantix在batch size增长时Tensor Core利用率持续提升，而SqueezeLLM完全不用Tensor Cores。inter-tile+intra-tile双层double buffering消除了baseline中"Tensor Cores等dequantization"的pipeline bubble
  **(5) 关键trade-off**：non-uniform quantization用更多centroids换取更好accuracy（3-bit SqueezeLLM perplexity 6.15 vs GPTQ 7.55），但centroid overhead在compute-bound大batch场景下使4-bit Quantix可能慢于4-bit Marlin（uniform quantization dequantization更简单）。A100上因更高memory bandwidth，memory-saving benefit相对L40更小。过大批量时register pressure可能导致spilling影响ALU utilization

## Accelerating Sparse Transformer Inference on GPU (STOF)

- baseline方法是什么？
  Baseline分为两类：(1) **MHA fused kernel baseline**：FA2/FA3仅支持causal和sliding window等连续mask pattern，无法处理discrete/unstructured分布（如random attention、Bigbird）；FlashMask使用column-wise representation但仅支持columns上element连续的mask，无法表示discrete分布；FlexAttention支持任意mask但constrained to fixed optimizations仅达到suboptimal性能；ByteTransformer自定义kernel限制max seq_len=1024；SPLAT专注regular sparse kernels (R-SDDMM/R-SpMM)放弃MHA整体优化机会；PyTorch Native/MCFuser/ByteTransformer不支持sparse mask，需先做mask subtraction（减inf）再做full GEMM，无法减少计算量。(2) **Operator fusion baseline**：PyTorch Compile/AStitch仅fuse MI operators，CI operators分开用vendor library；Welder/DNNFusion fuse CI+MI但受限于category-based规则；Chimera/MCFuser fuse CI+CI chain但忽略硬件细节（bank conflict等），long sequence下性能差；Bolt基于CUTLASS template但fusion range扩展困难。所有baseline的共性缺陷：固定operator fusion scheme无法根据tensor dimensions（batch size/seq_len/hidden_dim）自适应——例如GEMM+Layernorm fusion在hidden_dim=512时最高加速39.1×，但在hidden_dim=1024时反而显著减速。此外，individual operator tuning的最优参数配置无法直接迁移到fused operator（因为search space fundamentally不同），operator-by-operator sequential tuning会使GEMM+Layernorm在A100上平均仅达2.4×而非post-fusion tuning的10.1×。

  全栈execution例子（以FA2在A100上跑BERT-Base + Bigbird mask、batch=16、seq_len=4096为例）：
  - 算法层：Bigbird mask (80.8% sparsity, unstructured) → FA2不支持Bigbird mask，PyTorch Native fallback为先GEMM得full score matrix→mask subtraction（将mask位置score设为-inf）→Softmax→GEMM，无计算量节省
  - 系统框架层：PyTorch fx graph capture MHA subgraph，但因mask不支持fused kernel，subgraph被拆分为fine-grained meta operators逐op执行
  - 编译框架层：PyTorch Compile对meta operators做通用compilation optimization（constant folding + instruction scheduling），CI operators用cuBLAS library，MI operators做通用fusion
  - kernel调度层：FA2 kernel仅支持causal/sliding window mask → fallback到cuBLAS GEMM + Softmax + cuBLAS GEMM多kernel launch，每次intermediate result write back to HBM再reload
  - 硬件架构层：NVIDIA A100 (108 SM, 80GB HBM2e)，大量HBM读写intermediate results → memory bandwidth成为瓶颈

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**STOF (Sparse Transformer Optimized Framework)**，通过customized MHA kernels + adaptive operator fusion实现sparse Transformer inference在GPU上的全栈优化。

  论文方法全栈execution例子（以STOF在A100上跑BERT-Base + Bigbird mask、batch=16、seq_len=4096为例）：
  - **算法层**：Bigbird mask (80.8% sparsity, unstructured) → two-level storage format表示为full OTs (BSR) + part OTs (64×uint64 bitmap_mask)，任意mask pattern统一表示。Mask sparsity直接转化为计算量跳过（仅加载/计算valid OTs）
  - **系统框架层**：STOF将模型分为MHA structure（sparse，custom kernel处理）和downstream operators（dense，template-based fusion处理）→ torch/cpp_extension将MHA kernel封装为PyTorch native function→ fx.GraphModule operate downstream graph
  - **编译框架层**：neural hashing发现频繁子图→predefined rules生成initial fusion scheme→hash encoding转为binary array→numerical decoding映射到Triton/TileLang compilation template→two-stage search engine确定optimal fusion scheme + kernel parameter
  - **kernel调度层**：Block-wise kernel（因seq_len=4096，valid OT ratio适中）：Q_i在register中resident→仅加载valid OTs的K_Tj/V_j→cp.async异步加载V_j与GEMM重叠→part OTs用bitmap_mask精确mask 8×8 IT→Softmax with scaling factor α做跨OT reduction→最终write back HBM。OT行主序（Softmax迭代）、IT列主序（bank conflict-free）、Q register resident（避免SMEM重复读写）
  - **硬件架构层**：NVIDIA A100 (108 SM, 80GB HBM2e) → block-wise kernel利用Tensor Cores (8×8 IT对齐mma.m16n8k16)，SMEM double buffering重叠memory access与compute → 端到端相对FA2加速~4.8×（(16,4096)时）

  对应解决Baseline缺陷：
  **(1) 现有MHA fused kernel无法表示任意mask pattern → two-level storage (BSR+bitmap)**：FlashMask仅支持column-continuous mask（4-array column-range表示无法处理discrete分布），FlexAttention fixed optimizations suboptimal。STOF通过OT/IT两层抽象：OT级BSR（full_row_ptr/full_col_idx + part_row_ptr/part_col_idx）表示全局skip blocks，IT级64×uint64 bitmap_mask表示block内精确element分布。causal (50% sparsity)到Bigbird unstructured (80.8% sparsity)均可统一表示，且存储格式本身直接驱动kernel skip逻辑（load_row_ptr/load_col_idx告知加载哪些OT、bitmap_mask告知IT内哪些element需mask）
  **(2) MHA kernel缺乏灵活kernel选择 → analytical model驱动的kernel selection**：row-wise kernel（小seq_len+高稀疏→row-sliced Q并行+warp内shuffle无sync overhead）vs block-wise kernel（大seq_len通用场景→partition Q/K/V到SMEM利用memory hierarchy）。公式1基于valid OT ratio和seq_len计算threshold（log penalty压制extreme sparse长seq），自动选择最优kernel
  **(3) Fixed operator fusion scheme无法适应diverse input scales → hash encoding + two-stage search**：Fixed category-based fusion（MI-only/CI+MI/CI+CI）在hidden_dim从512变为1024时可能从16.5×加速变为slowdown。STOF用hash encoding将fusion scheme表达为searchable binary expression（任意scheme可表示），two-stage search（fusion expansion through expand/seize/compete rules→parameter sampling with reward-based allocation）自动发现per-input-scale最优fusion方案。Tuning time比MCFuser快6.7×、比Bolt快6.9×
  **(4) Individual tuning参数不transfer到fused operator → co-tuning fusion scheme + kernel parameters**：直接复用individual operator最优参数到fused operator导致Bias+Layernorm仅2.4×而post-fusion tuning达10.1×（A100）。STOF的two-stage search在每次fusion expansion时对pre-fusion和post-fusion分别采样参数比较，performance cache复用避免重复，确保fusion scheme和kernel parameters在hierarchical space中co-optimized
  **(5) Downstream operator fusion忽略硬件细节 → compilation template with tile-level optimization**：Chimera/MCFuser的loop-based construction忽略bank conflict等GPU硬件细节。STOF的Triton/TileLang template内部：tile decomposition最大化data reuse、warp-level primitives做高效reduction、multi-stage pipeline重叠memory/compute、仅暴露关键参数（block_size/num_stages/num_warps/blkM/blkN/blkK）作为search space，既保证硬件效率又限制搜索复杂度
  **(6) Long sequence下baseline OOM → MHA compute skipping + memory saving**：seq_len 32k时所有baseline (PyTorch Compile/ByteTransformer/MCFuser)均OOM，STOF运行正常（64k才OOM）。seq_len 16k时STOF相对PyTorch Compile加速16.8×

## Difflow: A Data-Characteristic-Aware Serving System for Diffusion Models

- baseline方法是什么？
  Baseline方法分为三类：(1) **通用DNN框架**（PyTorch v2.1, PyTorch-Inductor v2.1, TensorRT v8.6）：以operator-by-operator方式执行扩散pipeline，对所有请求独立处理，batching要求完全相同的shape。全栈执行例子：算法层扩散模型denoising loop→系统框架层PyTorch eager execution或Inductor/TensorRT compile→编译框架层operator fusion / kernel selection但无数据属性感知→kernel调度层cuBLAS/cuDNN标准kernel→硬件架构层A100/H100 GPU。缺陷：(a) 无跨请求优化——大量相同prompt/input的冗余计算不被消除；(b) ragged shape请求无法batch——仅支持uniform shape batching，ragged请求需单独执行；(c) 无invariant tensor优化——loop-invariant计算在每次denoising iteration中重复执行。

  (2) **扩散专用框架**（Stable Fast v1.0, Diffusers）：手动优化pipeline针对特定uniform数据属性场景（如batch相同prompt）。全栈执行例子：算法层扩散pipeline→系统框架层Stable Fast/Diffusers提供预设优化pipeline→编译框架层manual optimized pipeline→kernel调度层标准kernel→硬件架构层A100/H100 GPU。缺陷：(a) 仅针对uniform properties预设场景优化——一旦prompt/control input/image shape等出现heterogeneity，退回通用执行；(b) 手动优化覆盖极窄——应用开发者和用户的grid search/correlative requests工作流产生大量异构数据属性，manual optimization枚举组合不可行；(c) 无ragged batching支持——不同shape请求只能串行或pad到最大shape（浪费计算和显存）。

  (3) **Katz[37]** (Diffusion Serving System)：面向ControlNet-as-a-service的多GPU serving，sequential执行每个ControlNet请求。全栈执行例子：算法层ControlNet+LoRA pipeline→系统框架层Katz multi-GPU worker→编译框架层论文未明确说明→kernel调度层标准kernel+multi-GPU通信→硬件架构层4×H100 GPU。缺陷：(a) 多GPU通信overhead（edit SDXL-Turbo单iteration场景下严重限制throughput）；(b) sequential execution无batching (latency ~0.03s/request但per-GPU throughput低)；(c) LoRA serving数学上不等价。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**Difflow (ChituDiffusion)**：一个data-characteristic-aware diffusion model serving系统，通过symbolic data property propagation将pipeline分解为dGraphs→编译为多版本dEngines→运行时DP调度dTasks batching。

  论文方法全栈执行例子（以edit应用：SDXL Turbo + ControlNet + 16 LoRA styles, A100为例）：
  - 算法层：SDXL Turbo diffusion pipeline(1 denoising step) + ControlNet(Canny edge spatial control) + 16 LoRA weights for diverse styles→固定latent noise和U-Net conditioning，每个request调用不同LoRA
  - 系统/Serving框架层：Application characteristics标注——prompts可能相同(symbolic)、input images vary、ControlNet conditioning inputs shared→Difflow compiler执行符号化属性传播→SDXL U-Net分解为3个dGraphs：dGraph1 (input-dependent U-Net layers)、dGraph2 (input-independent U-Net layers, 16 styles共享)、dGraph3 (post-processing)→每个dGraph编译为多个dEngines（redundant/ragged/uniform variants）→运行时2 unique dTasks for dGraph1, 3 for dGraph2→DP scheduler枚举dEngines寻找最优batching→data-aware batching dispatch到matched dEngine执行→asynchronous property inference与execution overlap
  - 编译框架层：Symbolic property propagation (Table 1规则覆盖elementwise/linear/convolution)→DFG unroll+stabilize denoising loop→group consecutive operators by identical output property expression→enumerate property conditions→prune conflicting+inessential(<5% speedup) cases→compile surviving conditions to specialized dEngines→Redundancy elimination rules (dimension-level computation+memory access)→Ragged operation regularization (transpose/reshape/im2col graph transforms)→Invariant tensor detection (four-state: constant/loop-invariant/loop-variant/unknown)
  - kernel调度层：Redundancy elimination——相同prompt请求的K/V tensors compress沿batch dim + concat Q tensors→FlashAttention标准kernel→broadcast恢复；Ragged batching——不同shape请求通过transpose/reshape转regular→标准Matmul/Conv kernel；Invariant tensor——dGraph2输出识别为constant→compile-time caching with multi-value support (16 LoRA outputs precomputed)
  - 硬件架构层：NVIDIA A100 40GB PCIe (CUDA 12.1) / H100 80GB PCIe

  对应解决Baseline缺陷：
  **(1) 通用DNN框架无跨请求优化 → dGraph decomposition + redundancy elimination**：Difflow通过符号化属性传播识别pipeline中的共享优化条件——dGraph decomposition将pipeline分片为共享相同优化条件的子图（如U-Net中的prompt-embedding processing、ControlNet conditioning），使跨请求冗余（如相同prompt的CLIP embedding、相同conditional inputs）被系统性地检测和消除。Redundancy elimination在dimension-level移除冗余计算和内存访问——以attention为例，相同K/V tensors compress + Q tensors concat → 一次attention计算替代多次独立计算。在refine/edit/video应用上（含correlative requests with shared inputs），A100上throughput最高提升2.1×，H100上2.2×。

  **(2) 扩散专用框架仅支持uniform properties → multi-version dEngine compilation**：Difflow通过selective dEngine generation为每个dGraph编译多个specialized engines，每个engine针对特定data property配置优化——涵盖uniform/redundant/ragged等不同property组合，通过pruning unsatisfiable+inessential conditions控制编译开销（SDXL U-Net从monolithic 16384 engines→4 engines, 从11天→4分钟）。运行时DP scheduler在选择dEngines时兼顾batch size (ragged dEngines批处理更多请求)和per-engine efficiency (uniform dEngines无ragged overhead)——如图15所示，raggedness ratio 25%-50%时混合使用两种dEngines比单一类型额外提升10%。

  **(3) ragged shape请求无法batching → ragged operation regularization**：Difflow不要求手写所有ragged operator kernel（开销大且auto kernel generators难以处理），而是通过graph transformation rules (transpose/reshape/im2col)将ragged data-sharing operations转化为regular operations——ragged dim与batch dim fuse后变为regular dim→直接使用cuBLAS/cuDNN等成熟kernel库。ragged data-independent ops通过round-robin tile-to-thread-block mapping并行执行。在venti (SD1.5) ragged-only workload上，Difflow达1.4× speedup（通过ragged batching实现并行+减少weight重复内存访问）；在grande (SDXL)上1.1× speedup（SDXL模型更大，batching gain收窄）。

  **(4) 手动优化枚举所有property组合不可行 → symbolic property propagation**：Difflow用symbolic boolean variables表示数据属性——初始由application characteristics提供（fixed/varying/symbolic）→通过operator-specific propagation rules (Table 1)传播→output property expression自动推断→operators with identical expressions grouped to dGraph→state space exponentially reduced。相比monolithic optimization需枚举2^n property combinations (n=input count)，dGraph decomposition将问题分解为多个小dGraph的独立枚举。

  **(5) loop-invariant计算重复执行 → invariant tensor elimination + multi-value caching**：Difflow的four-state detection (constant/loop-invariant/loop-variant/unknown)识别constants和loop-invariants→compile-time precompute→loop-invariants hoisted outside denoising loop→multi-value constants selective fixing (如16 LoRA weights对应16 precomputed outputs)。sequential execution (无batching)下IRE单独贡献1.3× speedup (Figure 13b)。

  **(6) Katz multi-GPU communication overhead → single-GPU data-aware execution**：Difflow在单GPU上通过dGraph decomposition+data-aware batching同时服务所有请求，避免Katz的多GPU通信overhead。在edit应用上Difflow per-GPU throughput显著高于Katz (normalized to per-GPU, 图10)。

  Trade-off：(1) 符号化分析假设数据属性可被有限symbolic variables表达——对极端irregular或不可预测的workload可能退化；(2) dEngine pruning (5% speedup threshold) 可能在edge cases下丢失marginal优化机会；(3) 性能模型虽R²=0.998但基于线性假设——对complex computation patterns (如video temporal/spatial attention)需手动调整input metrics；(4) dGraph decomposition依赖pipeline DFG结构——高度irregular control flow（如dynamic ControlNet activation per iteration）需runtime adaptation。

## MixFusion: A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models

- baseline方法是什么？
  **Baseline为NIRVANA + ORCA batching和Distrifusion。**
  全栈执行例子（以SDXL serving三个不同分辨率请求512×512/768×768/1024×1024为例）：
  - **算法pipeline**：NIRVANA使用approximate caching（offline profiling决定每步skipped blocks），对全图做cache reuse决策——block级skip基于固定pattern（每步预定义skip哪些block，不随分辨率变化调整）。Distrifusion将image切为固定数量patch分发到多GPU，各GPU独立执行denoising，异步AllGather交换stale cross-GPU context。
  - **Serving系统框架**：NIRVANA集成ORCA进行iteration-level selective batching——ORCA优先新到达请求以在strict deadline内完成更多请求，但当请求分辨率不同时tensor shape mismatch→无法batching→sequential execute→GPU underutilization（三个SDXL请求sequential 17.8s, concurrent batching 9.5s on H100）。Distrifusion在multi-GPU下每个GPU独立处理分配到的patch→通过异步通信overlap同步开销→但patch数固定=GPU数，无法将不同分辨率请求统一为同一patch size→不支持mixed-resolution batching。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：NIRVANA使用标准PyTorch diffusion pipeline kernel（无patch-level kernel）。Distrifusion使用异步AllGather进行cross-GPU boundary exchange，但stale context导致accuracy degradation（Figure 6, Distrifusion生成图像内容偏离original）。Naive stitching（fetch all required boundaries + concatenate with target patches）的overhead完全抵消patch parallelism收益（Figure 5, naive stitch latency > sequential）。
  - **硬件架构**：论文未明确说明。NIRVANA和Distrifusion均运行于NVIDIA H100 GPU。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **MixFusion通过CSP patch管理 + Patch Edge Stitcher + patch-level caching + SLO-aware scheduling四个层次协同解决baseline在混合分辨率场景下的缺陷。**
  全栈执行例子（同场景三个不同分辨率SDXL请求）：
  - **算法pipeline**：MixFusion将不同分辨率请求partition为uniform patch（patch size = 各resolution height/width维度的最大公约数）→pixel-wise operators（Linear, FeedForward, Cross Attention）对各patch独立→Self-Attention按resolution分组reconstruct全图后batched attention→Convolution通过Patch Edge Stitcher fuse入GroupNorm kernel处理跨patch依赖。Cache predictor（GPU端Random Forest per block）在每步每block前动态预测每个patch的reusability——非固定skip pattern，能适应resolution变化导致的skip block差异（Figure 7, 不同resolution的skipped blocks分布显著不同）。
  - **Serving系统框架**：CSP格式通过resolution reorder + offset compression将heterogeneous patches映射为统一批次→所有请求的patches在单batch内并行处理（3请求从17.8s sequential→9.5s batched）。SLO-Aware Scheduler用slack score量化紧急度，MLP Throughput Analyzer（200个resolution组合训练，error <3.7%）预测batch latency→schedulability test避免SLO violation→相比NIRVANA在90% SLO satisfaction下SDXL goodput提升5.33×，SD3 SLO satisfaction平均高30.1%。
  - **编译框架**：论文未明确说明。基于PyTorch + xformers。
  - **kernel调度**：Patch Edge Stitcher fused in GroupNorm kernel——每个TB normalized一个patch的同时将boundary pixels暂存shared memory→所有normalization完成后TB定位目标patch写回boundary→消除naive stitching的额外memory movement overhead。CSP O(1) patch定位→batch cache操作（Common/New/Expired三集合coalesce）→每个block的cache操作<2ms（SD3每step 40-50ms含24 blocks→per-block cache overhead ~1-2ms）。
  - **硬件架构**：论文未明确说明。运行于NVIDIA H100-80GB GPU。

  **对应解决Baseline缺陷的映射：**
  **(1) 混合分辨率无法batching → CSP + uniform patch partitioning**：NIRVANA因tensor shape mismatch只能sequential处理不同分辨率请求。MixFusion将image沿height/width均切为uniform patch→所有patch shape相同→单batch并行→GPU utilization显著提升（Figure 19, throughput随batch size线性增长，3→12 batch size）。
  **(2) 跨patch context exchange开销大 → Patch Edge Stitcher fused in GroupNorm**：Naive stitching的boundary fetch+concat开销完全抵消并行收益（Figure 5）。Patch Edge Stitcher将stitching fuse入GroupNorm→TB shared memory暂存boundary→overlap stitching与normalization→无额外synchronization开销→PSNR 28.82/SSIM 0.88 (4 patches, Table 4)远高于naive replicate (PSNR 9.54/SSIM 0.45)。
  **(3) 固定cache pattern不适应resolution变化 → patch-level dynamic cache reuse**：NIRVANA的offline fixed skip blocks在混合分辨率下失效（Figure 7不同resolution skipped blocks分布不同）。MixFusion的per-patch per-block Random Forest Cache Predictor在线比较input与cached data→动态决定每个patch是否重算→patch-level caching比whole-image caching consistently更高computation savings（Figure 20）。
  **(4) 混合分辨率latency难以预测 → MLP Throughput Analyzer**：传统offline profiling无法应对explosive combination of resolutions（M并发请求×N分辨率→∑C(N-1, i+N-1)个组合）。MixFusion用200个resolution组合训练MLP→在线预测batch latency（error <3.7%）→SLO-Aware Scheduler做schedulability test→避免SLO violation同时最大化goodput。
  **(5) Distrifusion accuracy degradation → up-to-date cross-patch context**：Distrifusion使用async AllGather的stale KV cache导致内容偏离original（Figure 6）。MixFusion在单GPU内通过CSP同步所有patch→Convolution通过Patch Edge Stitcher使用real-time boundary data→PSNR/SSIM显著高于Distrifusion（Table 4, SDXL PSNR 28.82 vs 10.96, SSIM 0.88 vs 0.49）。

  Trade-off：(1) Patch Edge Stitcher only covers GroupNorm+Conv fusion——对DiT（无Conv, SD3）模型无需stitching但受益于batching；(2) Cache Predictor (Random Forest on GPU)增加per-block prediction overhead——需控制在<2ms per block，否则cache收益被抵消；(3) CSP格式假设请求在GPU内完成整个denoising pipeline——不支持跨请求preemption；(4) SLO scale为3×时NIRVANA因ORCA的priority scheduling在SD3上略优于Mixed-Cache——此时SLO constraint极紧限制了batching收益空间。

## ASM-SpMM: Unleashing the Potential of Arm SME for Sparse Matrix Multiplication Acceleration

- baseline方法是什么？
  Baseline方法分为三类：(1) **ARM CPU通用SpMM库**：ArmPL v24.10、Armadillo v14.6.0、SuiteSparse Cholmod v5.3.3、Eigen v3.4.0、MP-SpMM，主要依赖通用多核和vector unit（SVE/Neon），无法使用SME matrix unit的outer-product算力。全栈执行例子：算法层稀疏矩阵乘法A×B→系统框架层调用ArmPL/Armadillo等BLAS库→编译框架层Clang编译但无SME指令生成→kernel调度层CSR/CSC等通用稀疏格式+SIMD vector kernel→硬件架构层ARM CPU P-core/E-core上的SVE/Neon vector unit。缺陷：SME matrix unit的ZA register和MOPA outer-product指令完全未被利用，导致理论算力浪费。(2) **GPU Tensor Core SpMM方法**：TCF、ME-TCF、DTC-LSH、AccOrder、TC-GNN等依赖inner-product、warp调度和GPU memory hierarchy，并常依赖固定block alignment、left-aligned tile和zero padding。全栈执行例子：算法层SpMM→系统框架层CUDA/cuSPARSE→编译框架层NVCC→kernel调度层Tensor Core MMA with block sparse format→硬件架构层GPU Tensor Cores。缺陷：这些格式和调度策略假设inner-product和GPU warp-level并行，与SME的predicate-driven vector outer-product不一致；固定block alignment导致冗余zero padding浪费SME的稀疏算力。(3) **静态负载均衡**：按行、非零元或core能力预分配任务。全栈执行例子：kernel调度层静态row partitioning将A的行均匀分配给各core→硬件架构层Apple M4上P-core和E-core共享SME unit。缺陷：在Apple M4这类P-core/E-core异构且SME单元按cluster共享的架构上，静态分配忽略非零分布不均和core速度差异，容易导致负载失衡。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**ASM-SpMM**：面向ARM SME的高性能SpMM库，通过OP-MCF稀疏格式+SME outer-product microkernel+multi-tile并发+prefetch pipeline+SVE/Neon混合执行+hetero-core动态work stealing调度六层协同设计。

  论文方法全栈执行例子（以Apple M4上GCN inference的SpMM为例）：
  - 算法pipeline：GCN layer中的稀疏邻接矩阵A × 稠密特征矩阵W的SpMM操作→ASM-SpMM替换PyG/DGL的默认SpMM实现。
  - 系统框架层：论文未明确说明（ASM-SpMM以library/kernel形式被GNN framework调用）。
  - 编译框架层：Clang 16.0编译，使用ARM SME intrinsics（svld1_f32/svmopa_za32_f32_m/svst1_hor_za32/_svprfw等）而非依赖compiler自动生成SME指令。
  - kernel调度层：OP-MCF格式转换→row window划分→compressed slot遍历→每个slot：svld1_f32加载sparse values到Z register→ColumnPositionMaskBit转predicate register→vectorized load dense B tile→svmopa_za32_f32_m outer product accumulate to ZA tile→循环间_svprfw预取下一slot的data/mask/column index/dense B fragments→多ZA tile并发→低密度block分配SVE/Neon vector path→interleaved scheduling隐藏vector latency。
  - 硬件架构层：Apple M4 CPU（512-bit SVL，8 double/window），ZA matrix register做outer-product累加，P-core cluster和E-core cluster各一个SME compute unit，predicate register控制有效非零计算。

  **对应解决Baseline缺陷的映射：**
  **(1) ARM CPU库无法使用SME outer-product算力 → OP-MCF格式+SME outer-product microkernel**：ArmPL/Armadillo等使用CSR+SIMD vector kernel，SME ZA register闲置。ASM-SpMM的OP-MCF按SME vector length切row window，column compaction将非重叠稀疏列合并为一个compressed slot，bitmask还原原始列有效行，使SME predicate register只参与有效非零计算；svmopa outer-product直接利用ZA tile做sparse value向量×dense tile外积累加，避免把SME当成普通SIMD。M4上geomean speedup 11.81× over ArmPL、15.12× over Armadillo、18.62× over Eigen。

  **(2) GPU Tensor Core格式不匹配SME → OP-MCF去掉硬性block padding**：TCF/ME-TCF等依赖fixed block alignment和left-aligned tile，SME上zero padding浪费算力。OP-MCF的masked multi-column merging消除padding——非重叠列合并为一个physical column无需对齐，mean NNZ per slot达到约4-6（vs CSR约1-2），显著提高SME外积的每slot有效计算密度。

  **(3) SpMM访存不规则导致cache miss → multi-tile并发+显式prefetch pipeline**：CSR/普通sparse kernel的LLC miss rate约30%-61%。ASM-SpMM用多ZA tile并发提高ZA/Z register占用率，_svprfw类prefetch指令显式预取sparse/dense operand将LLC miss rate降至23%-48%，prefetch pipeline隐藏不规则访存延迟。

  **(4) 低密度block在SME上利用率低 → SVE/Neon混合matrix-vector kernel**：稀疏尾部或碎片化block若强行走SME path会因padding浪费周期。ASM-SpMM将最稀疏block分配给SVE/Neon vector unit，通过interleaved instruction scheduling使vector工作量隐藏在SME固定执行窗口内。hybrid kernel在rCA、FY-RSR、ddi、ppi上比matrix-only快8%-18%。

  **(5) 异构多核静态负载均衡失衡 → hardware-aware task mapping+动态work stealing**：Apple M4有P-core/E-core异构且SME unit按cluster共享，静态分配忽略非零分布和core速度差异。ASM-SpMM runtime先做hardware-aware task mapping分配row window，再用progress monitoring检测core完成进度，从负载较重core窃取row window实现动态再平衡。LX2上12线程相对2线程达8x-11x scaling。

  Trade-off：(1) OP-MCF需要一次性格式转换，适合同一稀疏矩阵被重复执行的场景（GNN inference、迭代solver、超参搜索）；若矩阵频繁变化且复用次数很少，转换成本可能更难摊销。(2) 混合SME/SVE调度提升吞吐但引入register partition、资源竞争和workload partition开销，hybrid/theory仅0.78-0.90。(3) 设计依赖ARM SME硬件可用性，不同SME平台需调整tile size、resource allocation和调度参数。(4) ASM-SpMM面向FP64（论文在Apple M4和LX2上均以FP64为主评估），FP32/FP16等低精度下的OP-MCF和SME kernel行为需独立验证。

## RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization

- baseline方法是什么？
  Baseline是已有channel-wise mixed precision quantization方法（MixQ、QuaRot、Atom），以及uniform INT4量化。全栈执行例子（以Qwen3-8B在RTX 4090上W4A4推理为例）：
  - 算法层：INT4 uniform quantization→逐round-to-nearest量化activation和weight到4-bit→scaling factor保存→INT4 Tensor Core GEMM→dequantization恢复FP16；或MixQ：per-channel max value检测outlier channels→outlier channel→INT8/FP16 precision、normal channel→INT4→channel-wise mixed precision GEMM沿reduction dimension分解。QuaRot：Hadamard rotation平滑channel outliers→uniform INT4量化→仅处理channel维度outlier。
  - 系统框架层：SGLang/vLLM serving框架加载quantized model→prefill/decode pipeline使用FP16 GEMM或量化kernel
  - 编译框架层：论文未明确说明（手工CUTLASS/Triton kernel实现，无编译框架自动mixed precision生成）
  - kernel调度层：NVIDIA Tensor Core mma指令，INT4/INT8精度。Channel-wise方法沿reduction dimension分解GEMM→normal channel和outlier channel分别dense计算→结果合并。QuaRot使用fused dequantization kernel（dequant融合入GEMM）。Atom使用group-wise fine-grained quantization（group size 128）→引入额外dequantization/scaling overhead。
  - 硬件架构层：NVIDIA RTX 4090 GPU。INT4 Tensor Core峰值吞吐8×FP16，INT8 Tensor Core 2×FP16。Channel-wise mixed precision沿reduction dimension分解不会产生sparse computation pattern，与Tensor Core指令兼容。
  Baseline缺陷：(1) Channel-wise方法仅处理channel维度outlier→4-bit下仍有大量token-wise outlier无法表示→量化误差大（MixQ Qwen3-8B perplexity 14.76 vs BF16 9.72）。(2) QuaRot仅旋转但无mixed precision→旋转后仍有token-wise residual outliers→perplexity 11.53（优于MixQ但仍劣于RoMeo 10.97）。(3) Atom使用group-wise fine-grained quantization提高准确率但计算开销大→Qwen3-32B上Atom仅51.59 average zero-shot accuracy（显著低于RoMeo 70.66），且kernel实现性能差（仅3.63× average kernel speedup vs RoMeo 4.68×）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**RoMeo**：通过RTMPQ算法（Hadamard rotation + token-wise mixed precision）处理双维度outlier，配合permutation-free系统设计和异步并发执行实现与uniform quantization相当的计算效率。

  论文方法全栈执行例子（以Qwen3-8B在RTX 4090上W4A4 5%-outlier推理为例）：
  - 算法层（核心创新）：
    (1) **Hadamard Rotation**（解决baseline缺陷：channel-only处理无法消除token-wise outlier）：对activation右乘Hadamard矩阵H，利用H的正交性和+1/-1元素将channel-wise irregularity平滑并迁移到token维度。旋转后peak activation从1272降至58.5。后续token-wise mixed precision只需处理token维度outlier，消除双维度的复杂度。
    (2) **Token-wise Mixed Precision Quantization**（解决baseline缺陷：4-bit下residual token-wise outlier仍导致量化误差）：在旋转后的纯token-wise分布上做per-token max-based outlier detection→top-k选择5% outliers→outlier token用INT8（range=127 vs INT4 range=7，16×更大的表示范围）；normal token用INT4。同时weight矩阵也由于H^T预乘amplify non-uniformity而采用相同的mixed precision。四种精度组合的cross-precision乘法。
  - 系统框架层（核心创新）：
    (3) **Permutation-free Mixed Precision Computation**（解决baseline中不存在而token-wise特有的挑战：non-reduction dimension sparse computation无法利用Tensor Core）：预分配outlier buffer→整个activation矩阵统一量化为INT4→outlier token单独拷贝到outlier buffer量化为INT8→所有四种精度组合各自操作dense uniform-precision矩阵→每个thread block处理同种精度组合→无需permutation保留contiguous memory layout。Tolerate redundant computation（outlier token参与两次计算）来保证Tensor Core兼容。
    (4) **Asynchronous Concurrent Execution**（解决：outlier乘法tall-and-skinny矩阵导致GPU SM underutilization）：将量化分解为outlier/normal两个独立task→分解dependency graph→四个GEMM kernel之间无dependency可通过多CUDA stream异步并发执行。CUDA events仅在有真实依赖处同步（如quant→GEMM dependency）。掩盖kernel launch overhead + 提升SM利用率。
  - 编译框架层：论文未明确说明（手工CUTLASS/Triton kernel，JIT编译机制）
  - kernel调度层（核心创新）：
    (5) **Separate-kernels with CUTLASS**（解决：fused kernel无法对不同精度配置独立on-chip resource allocation）：INT4-INT4 kernel→shared memory需求小→compiler可用更多register做loop unrolling提升ILP；INT8-INT8 kernel→共享内存需求大→occupancy由shared memory restrict→compiler自动balance。separate-kernels的launch overhead和underutilization由async execution弥补。评估表明separate-kernels+async优于fused-kernel。
    (6) **Software Pipelining with cp.async**（解决：memory access latency导致warp stall）：使用PTX cp.async指令异步加载GMEM→SMEM→pipeline fill→steady state: wait oldest copy→mma compute→issue new copy→pipeline drain。
    (7) **Fused Triton kernels**（解决：在线outlier detection引入runtime overhead）：将per-token row-max + top-k selection + INT4/INT8 quantization + data packing融合为单一Triton kernel，减少kernel launch和内存round-trip。INT4→INT8 casting在SMEM内用两条binary arithmetic指令完成（避免昂贵type conversion指令）。
  - 硬件架构层：NVIDIA RTX 4090 GPU。利用INT4 Tensor Core (8×FP16)和INT8 Tensor Core (2×FP16)。cp.async (Ampere+)异步memory copy。Cross-precision accumulation使用INT32累加器（防止overflow）。

  Baseline缺陷→RoMeo方案映射：
  | Baseline缺陷 | RoMeo方案 | 效果 |
  |-------------|---------|------|
  | Channel-wise方法无法消除token-wise outliers | Hadamard rotation将channel-wise outlier迁移到token维度再处理 | Peak activation 1272→58.5 (rotation后)→18.6 (TO pruned后) |
  | 4-bit下residual outliers导致perplexity退化 | Token-wise INT8/INT4 mixed precision，5% outlier用INT8 | Qwen3-8B PPL: MixQ 14.76→RoMeo 10.97; QuaRot 11.53→RoMeo 10.97 |
  | Token-wise mixed precision在non-reduction dim无法利用Tensor Core | Permutation-free dense computation + redundant outlier copy | 无需permutation，所有GEMM为dense uniform-precision |
  | Tall-and-skinny outlier矩阵SM underutilization | Asynchronous four-kernel concurrent execution over CUDA streams | Batch=16 layer latency 6.73→3.39ms with async |
  | 在线outlier detection引入runtime overhead | Fused Triton outlier detection + quantization + packing kernel | Hadamard+Quant+Post-mul overhead仅~12% baseline latency |
  | INT4/INT8 fusion kernel无法fine-tune on-chip资源 | Separate CUTLASS kernels per precision + software pipeline | Separate+Async overall最优 vs fused kernel |

## AUM: Unleashing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving

- baseline方法是什么？
  Baseline是工业界当前的**AU-exclusive**和**AUV-oblivious sharing**两种方式：
  
  (1) **AU-Exclusive (ALL-AU)**：将整个AU-enabled CPU独占分配给LLM serving，不与其他workload共享。全栈执行例子（以GenA + llama2-7b chatbot, batch=16为例）：
  - 算法层：LLM serving使用xFasterTransformer框架，prefill phase对QKV mapping执行GEMM（dim=8192×4096×22016），decode phase执行GEMV（dim=16×4096×22016）。所有核心全部使用AMX加速矩阵运算
  - 系统框架层：AU-Exclusive不共享CPU资源→所有48×2=96物理核全部运行LLM serving→无co-located workload→CPU idle核心浪费、冗余硬件资源未利用→perf-per-watt低（比GPU A100差2.1×），perf-per-dollar略优于GPU但效率不足
  - 编译框架层：论文未明确说明（使用Intel oneDNN预编译AMX算子库）
  - kernel调度层：所有核心统一使用AMX，无operator级AU选择——prefill用AMX GEMM（40.57 TFLOPS），decode也用AMX GEMV但效率低（3.87 TFLOPS，tile register配置overhead大）。因TDP限制所有AU核心频率统一降至2.5 GHz（prefill导致最大降频）。无频率区域划分
  - 硬件架构层：Intel SPR Xeon 8475B，96物理核，每核AMX单元1024 BF16 ops/cycle。AU-exclusive导致大量物理核的AU idle（decode phase的AMX cycle ratio仅1.5%），但独占策略避免management complexity
  
  (2) **AUV-Oblivious Sharing (SMT-AU / RP-AU)**：
  - **SMT-AU**：使用SMT (Simultaneous Multi-Threading) 共享AU核心——将LLM serving和通用workload混合调度到同一物理核的hyperthread上。AU不跨hyperthread共享。缺陷：AU perf degradation >200%（由于memory contention），co-running OLAP >40% slowdown（Figure 9a）。compute-intensive shared app虽对AU干扰<10%但自身40% degradation（频率拖累，Figure 9b）。无法控制可变的AU行为和干扰
  - **RP-AU**：使用Intel RDT (CAT/MBA) 做application-aware资源分区——隔离L2 cache、LLC、memory bandwidth给AU和shared应用。缺陷：单独隔离某类资源只能轻微减轻AU slowdown但无法达到最优决策（Figure 10），因为AU的critical backend bound随资源类型不同而变化，单一维度分区不足以应对三维AUV。无法handle AU frequency interference和variable usage pattern

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**AUM (AU-aware resource Manager)**，通过system-layer管理精确灵活地收获AU未利用资源给shared应用，三维度处理AUV。全栈执行例子（以GenA + llama2-7b chatbot shared with SPECjbb为例）：
  
  - 算法层：LLM serving同上（xFasterTransformer, llama2-7b BF16）。AUM不改LLM算法但通过ARI判定AU选择——prefill高ARI用AMX、decode低ARI用AVX/GEMV替代AMX，避免小矩阵AMX tile register overhead
  - 系统框架层（核心创新）：AUM作为系统层resource manager daemon运行。Background AU Profiler离线构建AUV Model（3 division × 3 sharing × 5 config × 10 rep = 450次AU执行）→Runtime Controller在线决策（每次<1ms）。Processor分区为三个频率区域：High-AU (prefill, 2.1 GHz), Low-AU (decode, 2.8 GHz), None-AU (shared SPECjbb, 3.2 GHz)。按加权效率E_CPU = (1.8×P_H + 0.2×P_L + γ×P_N)/W_CPU最大化来切换分区分频
  - 编译框架层：论文未明确说明（使用xft + oneDNN预编译算子）
  - kernel调度层：AUM通过三个runtime阶段做kernel级调度：(1) Slack-aware SLO Analyzer：通过LAG = Σ(d_TPOT - e_token)跟踪每个decode token相对deadline的位置——LAG<0表示落后，需加速→更多AU资源；LAG≥0表示领先→可收获资源给shared；(2) Efficiency-aware Core Switcher：根据SLO slack动态切换core分区分频——若decode underperforming则增加Low-AU region core数量，若prefill SLO slack充足则减少High-AU region→释放core给shared；(3) Collision-aware Allocation Tuner：检测AU shared performance collision（δ_AU > threshold=2）→优先收获对AU干扰最小的资源（如decode低AU usage时先收LLC，因decode对LLC不敏感Figure 13；高affinity的memory bandwidth根据runtime adaptively收放Figure 18），用P_a (avg perf) aggressive收获或用P_t (tail perf) conservative归还
  - 硬件架构层：Intel Xeon processors with AMX。利用Intel RDT CAT/MBA硬件接口实现cache way和memory bandwidth的硬件级隔离。利用AU的SIMD特性（frontend bound仅1% vs 通用功能单元5%）和decode phase memory-bound特性（DRAM bound 59.9%）精确收获硬件资源

  Baseline缺陷→AUM方案映射：
  | Baseline缺陷 | AUM方案 | 效果 |
  |-------------|---------|------|
  | AU-exclusive导致大量AU核心idle（decode AMX cycle ratio仅1.5%）浪费硬件效率 | AU-aware sharing：将decode/low-AU核心上的冗余LLC/BW资源精确收获给shared应用 | CPU efficiency ↑ 8.8% vs ALL-AU |
  | AUV-oblivious SMT无法handle variable AU behavior（perf degradation >200%） | Usage-aware：ARI判定AU选择（AMX vs AVX）+ Frequency-aware：分区分频避免频率interference + Bound-aware：按AU资源affinity调资源 | AU perf SLO guarantee ↑ 11% vs SMT-AU/RP-AU, efficiency ↑ 4.7% |
  | Variable AU usage导致compulsory frequency reduction（prefill→2.5 GHz）拖累共享应用 | Processor Region Division：High/Low/None三区域独立频率管理→decode不与prefill共享频率惩罚 | Shared OLAP 40% degradation → <10% (Figure 9b对比) |
  | AUV-oblivious RP单维度资源分区无法最优（单独LLC/BW隔离仅轻微减轻干扰） | 三维度AU-aware：Usage × Frequency × Resource Bound joint optimization → Runtime Controller自适应调整所有维度 | 精确资源分配（Figure 18: AUM根据runtime info灵活分配LLC和BW，vs static allocation） |
  | 无runtime SLO adaptivity导致LLM serving无法应对dynamic workload | LAG-based SLO分析：实时量化每个request ahead/behind schedule→tune AU resource accordingly | Decode TPOT SLO guarantee比AUV-oblivious高7% |
  | 无AU behavior profiling机制导致resource management盲目 | Background AUV Model：离散化continuous variation为bucket→记录P_a/P_t/W_CPU→供online look up | Profiling cost可摊销（450次执行→覆盖数千核），runtime决策<1ms |
  | GenA→GenC代际提升但efficiency提升有限（Fig 15: 仅1.55×） | AUM leverage更强大AU和memory→更多resource tuning headroom | GenC上AUM efficiency提升19%/11%/17%（vs GenA上15%/7%/10%） |

## GyRot: Leveraging Hidden Synergy between Rotation and Fine-grained Group Quantization for Low-bit LLM Inference

- baseline方法是什么？
  Baseline是现有将rotation与group quantization简单组合的量化方案，以及使用浮点dequantization的硬件加速器。

  全栈执行例子（以LLaMA-3-8B W4A4推理为例）：
  - 算法层：现有rotation-based quantization（Quarot/SpinQuant）执行全局Hadamard rotation flatten activation/weight分布→per-channel或per-token symmetric/asymmetric quantization→GPTQ weight quantization。或LightRot执行local rotation (R=G=128) + asymmetric group quantization (FP16 SF/ZP)。缺陷：全局rotation将outlier分散到所有channel，与fine-grained group quantization的localized scaling冲突——rotation期望全局平滑，group quantization依赖local distribution capture。实验显示当G≤32时，加rotation反而增加perplexity（RTN下从7.40升至30.12 at G=32）。
  - 系统框架层：GPU上使用TensorRT-LLM或vLLM部署量化模型，GEMM在Tensor Cores上以INT4执行→dequantization在CUDA cores上以FP16执行（INT→FP conversion + FP scale/bias + FP accumulate）→mixed-precision execution path增加latency和能耗
  - 编译框架层：论文未明确说明（手工CUDA kernel或RTL，无编译框架自动生成）
  - kernel调度层：MANT PE (G=64, FP16 SF, flexible data format) / LightRot PE (G=128, FP16 SF+FP16 ZP)。每个PE执行INT4 dot product后→FP dequantization unit：partial sum→FP16 SX乘法→FP16 ZX×WSUM加法→FP16 SW乘法→FP accumulate。缺陷：(a) G越小→dequantization频率越高，FP dequantization开销急剧增长（Fig.2）；(b) asymmetric quantization增加ZX项→FP overhead进一步放大；(c) GPU上INT→FP type conversion增加指令数和register pressure
  - 硬件架构层：Tender（8-bit systolic array，无group quantization，W4A4下accuracy severe degradation PPL=23.85 on LLaMA-1-7B）。MANT accelerator（2D systolic PE array + FP16 dequantization per G=64）。LightRot accelerator（2D systolic PE array + FP16 SF/ZP dequantization per G=128 + outlier-aware permutation）。缺陷：FP16 dequantization unit area/energy占比高；small group size下dequantization frequency成为瓶颈

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**GyRot**：通过CoRFiG解耦rotation scope与group granularity、HAP利用Hadamard harmonic rows做outlier alignment、重公式化非对称量化+ceiling ZP rounding实现fully integer dequantization，在algorithm-hardware co-design层面同时解决accuracy和hardware efficiency问题。

  论文方法全栈执行例子（以LLaMA-3-8B W4A4KV4, R=1024, G=32为例）：
  - 算法层（核心创新）：
    (1) **CoRFiG**（解决baseline缺陷：全局rotation破坏fine group的local coherence→accuracy degradation）：
    将rotation scope限制为R=1024（而非全局全部channel），同时保持group size G=32。满足R=2^g·G关系（g=5），确保rotation在R范围内充分flatten分布，但每个outlier最多影响R个channel而非全部→保留group-level local variance。实验：G=32时，CoRFiG (R=1024, PPL=6.91) vs 全局rotation (PPL=7.04)，证明localizing rotation scope改善与fine group的协同。
    (2) **HAP**（解决baseline缺陷：rotation后per-group range扩大→scale factor精度需求上升；INT8 SF时PPL degradation严重）：
    利用Hadamard矩阵递归构造产生的harmonic rows（长度2^k的全+1或全-1向量），将全局outlier channel permute到harmonic rows上。Post-rotation后每个group内outlier乘以同符号（全+1或全-1），per-group range被tightly bounded→scale factor的精度需求大幅降低。实验：CoRFiG+HAP使INT8 SF从PPL=364.17降至6.80（Table IV），证明INT8 SF可与FP16 SF parity。
    (3) **重公式化非对称量化 + Ceiling ZP rounding**（解决baseline缺陷：HAP后per-group分布高度asymmetric→传统非对称量化zero-point分布long-tailed→INT8 ZP degradation）：
    将公式从x̂=⌊x/s_x+z_x⌉改为x̂=⌊(x+z_x)/s_x⌉，z_x从scaled domain计算（−min(x_g)/s_x，小s_x放大尾部）改为unscaled domain计算（−min(x_g)，无除s_x操作）→zero-point分布显著flatten（Fig.5）。ZP量化用⌈·⌉替代round→保证z_Q≥z→消除underflow clipping（Fig.6）。实验：reformulated asymmetric + ceiling rounding使INT8 ZP从PPL=7.93恢复至6.91（Table V）。
  - 系统框架层：GyRot accelerator以RTL实现→综合为ASIC（28nm, 1GHz）→替代GPU部署量化LLM推理。无Serving框架修改（专用硬件方案）
  - 编译框架层：论文未明确说明（RTL手工实现，无编译框架自动生成）
  - kernel调度层（核心创新）：
    GyRot PE：32-way INT4 dot product→fully integer dequantization pipeline (INT8 SX→INT8 ZX×WSUM→INT8 SW)→32-bit integer accumulator。FHT unit：5-stage 32-way add/subtract pipeline (160 units)，支持online Hadamard rotation up to 1024-dim。
    对比baseline PE：MANT/LightRot PE需FP16 multiplier+adder做dequantization→GyRot PE以INT8乘法替代FP16乘法→PE area减65.2%、energy减69.2% vs Tender。
  - 硬件架构层（核心创新）：
    GyRot accelerator: 8×8×32 tensor PE array（3D tensor organization vs 2D systolic of baselines）→2048 parallel ops/cycle。FVU集成nonlinear ops + FHT rotation→消除CPU-GPU data movement for online rotation。WSUM unit预计算per-group weight sum→broadcast共享避免per-PE重复计算。Total: 2.10 mm², 740.95 mW in 28nm。
    Speedup: 1.42–3.40× over Tender/MANT/LightRot。Energy efficiency: 1.20–3.64× improvement。关键：算法创新（CoRFiG+HAP+reformulated asym quant）使INT8 SF/ZP成为可能→硬件创新（integer dequantization PE）利用这一宽松精度需求实现高效全整数数据路径。

  Baseline缺陷→GyRot方案映射：
  | Baseline缺陷 | GyRot方案 | 效果 |
  |-------------|---------|------|
  | 全局rotation破坏fine group local coherence→G=32时PPL从7.40反升至30.12 | CoRFiG: rotation scope限制为R=1024, R=2^g·G=32G, decouple rotation与group granularity | G=32+R1024 PPL=6.91 vs global rotation 7.04 |
  | Rotation后per-group range扩大→INT8 SF精度不足→PPL degraded to 364.17 | HAP: harmonic row alignment→per-group range tightly bounded→relax SF precision requirement | INT8 SF PPL从364.17→6.80 (parity with FP16) |
  | HAP后per-group高度asymmetric→传统非对称量化long-tailed ZP分布→INT8 ZP clipping | Reformulated asym quant (bias-before-scale) + ceiling ZP rounding | INT8 ZP PPL从7.93→6.91 (near FP16 6.81) |
  | FP16 dequantization unit area/energy overhead大→small G加剧频率 | Fully integer dequantization (INT8 SF/ZP datapath in PE) | PE area ↓65.2%, energy ↓69.2% vs Tender |
  | GPU mixed-precision path (INT GEMM→FP dequant→FP accum) | Dedicated accelerator with fused integer dequantization in PE | 1.42–3.40× speedup, 1.20–3.64× energy efficiency |
  | LightRot R=G耦合→无法scale到更小G（如32） | CoRFiG解耦R和G→R=1024固定, G可独立选择32 | G=32正常工作, LightRot在G=32 INT8 SF/ZP下PPL=30.12 |

## PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System

- baseline方法是什么？
  PIM-based LLM inference系统（CENT[16]和NeuPIMs[21]）在长上下文decoding场景下存在三个系统性低效：
  (1) Head-First Partitioning (HFP)：将head-batch pair分配到PIM channel执行，默认batch/head并行足够填充所有channel。但长上下文下单个请求的KV cache足以占满一个channel容量，batch size被压低，Tensor Parallelism下不同请求token length不同造成channel执行时间不平衡（短请求channel早空闲），Pipeline Parallelism下每stage只激活与当前请求相关的少数channel（sparse channel activation）。论文在32K context CENT分析中MAC utilization下降48%。
  (2) Static PIM Command Scheduling：PIM primitive采用WR-INP→MAC→RD-OUT固定序列，传统scheduler只按保守时间间隔发射命令，不跟踪GBuf/OutReg entry级真实依赖，即使命令间无hazard也等待固定间隔。Attention的QK^T和SV因dh/dout小、数据复用低，I/O transfer频繁，静态调度导致MAC大量idle（小维度Attention MAC utilization低至14.7%）。
  (3) Static KV Cache Management：传统PIM指令的loop count和operand address在编译期固定，无法根据当前token length调整；系统必须按最大上下文Tmax为每个请求预留KV cache。真实workload请求长度差异大，静态预留导致平均容量利用率仅31.0%-40.5%。

  全栈执行例子（以CENT PIM-only系统、LLM-7B-32K decoding为例）：
  - 算法层：标准Transformer decoder，无算法改动（non-GQA，每层32 heads，dh=128，QK^T和SV均为GEMV操作）
  - 系统框架层：CENT PIM-only multi-node系统，HFP按head-batch pair将Attention GEMV分配到PIM channels。TP=2时两个module各持一半heads，PP=2时layer 1和layer 2分配到不同module。由于KV cache按Tmax预留，batch size受容量限制
  - 编译框架层：CENT使用fixed PIM instruction sequences（WR-INP→MAC→RD-OUT），loop count和operand address编译期固定为Tmax，无动态partitioning/metadata支持
  - kernel调度层：每个Attention head的QK^T固定映射到1-2个channel，其他channel idle（因无足够head-batch pair填充）。PIM controller按固定tWR-INP/tMAC/tRD-OUT间隔串行发射指令，即使命令间无hazard也等待，I/O transfer和MAC无法重叠
  - 硬件架构层：PIM module有16 channels × 16 banks，每个channel的PIM controller使用标准single-entry GBuf/output register，无dependency tracking logic，无on-module address translation

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**PIMphony**：通过TCP（Token-Centric Partitioning）、DCS（Dynamic Command Scheduling）、DPA（Dynamic PIM Access）三项协同设计，系统性地解决PIM在长上下文decoding中的channel utilization、I/O bottleneck和容量浪费。

  论文方法全栈执行例子（以PIMphony on CENT、LLM-7B-128K-GQA decoding为例）：
  - 算法层：无算法改动（同一Transformer decoder，GQA group size=8）
  - 系统框架层：TCP将Attention的token维度作为主并行维度。对QK^T，每个channel处理一段token的Key cache，与同一query做部分dot-product，score在module内拼接后Softmax。对SV，每个channel处理一段score/value的partial context，经module内reduction得完整context。不跨module同步（SV module内reduction开销<0.2% attention latency at 16K tokens）。DPA使KV cache按实际token length以1MB chunk lazy allocation，batch size不再受Tmax容量限制
  - 编译框架层：MLIR-based compiler自动识别Transformer decoder pattern→生成PIM-specific instruction sequences→embed TCP partition metadata（per-channel token segment range）、DCS dependency annotations（GBuf/OBuf entry-level hazard info）和DPA dynamic addressing（Dyn-Loop/Dyn-Modi编码）。IREE runtime HAL对接commercial PIM SDK，根据当前token length自适应dispatch
  - kernel调度层：TCP确保16-channel/16-bank配置下QK^T token length>256即可full channel activation（远超HFP的batch-dependent activation）。DCS通过D-Table/S-Table跟踪每GBuf/OBuf entry的依赖状态，命令到达时仅等待真正依赖的前序命令完成——无关WR-INP和MAC可乱序穿插、MAC和RD-OUT可在不同OBuf entry上并行。GQA下DCS利用dual-port GBuf/OBuf在MAC消费当前entry时预取下一批query/score，将row-reuse的KV reuse转化为真实吞吐。DPA的Dyn-Loop按runtime Tcur循环而非Tmax，Dyn-Modi按stride自动计算row/col，on-module dispatcher做VA→PA翻译
  - 硬件架构层：在AiMX PIM HUB侧新增dual-port OBuf（每bank面积0.47% of MAC unit），D-Table/S-Table（576B metadata）+ dependency-check unit（0.5% area/1.3% power overhead on PIM HUB control blocks），on-module dispatcher（<200KB buffer, 4% area overhead）。不改动DRAM bank array本身

  Baseline缺陷→PIMphony方案映射：
  | Baseline缺陷 | PIMphony方案 | 效果 |
  |-------------|-------------|------|
  | HFP按head/batch分配channel→长上下文下batch不足→MAC utilization降48% | TCP沿token维度partition→每个channel处理token segment→QK^T token>256即可full channel activation | PIM-only up to 11.3× speedup |
  | 固定WR-INP→MAC→RD-OUT timing→I/O和MAC无法重叠→小维度MAC util 14.7% | DCS entry-level dependency tracking→乱序issue→dual-port OBuf重叠数据搬运和计算 | MAC util从14.7%显著提升，DCS vs ping-pong up to 1.4× higher utilization |
  | 静态按Tmax预留KV cache→真实workload capacity util 31.0%-40.5% | DPA Dyn-Loop/Dyn-Modi+on-module dispatcher VA→PA翻译→1MB chunk lazy allocation | Capacity utilization提升至75.6% |
  | GQA row-reuse的KV复用被WR-INP transfer stalls抵消 | DCS dual-port GBuf/OBuf预取+MAC并行消费→隐藏input transfer overhead | GQA 128K模型收益更大（up to 11.3×） |
  | CENT在1M context退化至2% utilization（pipeline bubbles放大） | TCP+DCS+DPA协同消除三个瓶颈→Attention比FC更快→长上下文下系统utilization持续提升 | 1M context达46.6× speedup over CENT |
  | GPU A100虽有用但受HBM带宽/容量限制 | PIM内部带宽32TB/s (vs GPU ~2TB/s HBM)→PIMphony最大化此带宽利用 | PIMphony vs GPU-A100取得显著throughput优势，尤其non-GQA长上下文

## BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache

- baseline方法是什么？
  Baseline有两种：(1) **Non-fused low-bit KV cache attention**（如Kivi）：将mixed-precision attention分解为多个standalone kernel（量化kernel、dequantization kernel、attention kernel分离），每个kernel各自load/store intermediate data到global memory，无on-chip data reuse，导致高kernel launch overhead和global memory traffic。(2) **CUDA Cores-only fused attention**（如Atom、QServe）：将量化/dequantization和attention融合到单kernel，但dequantization和矩阵乘法（GEMV/GEMM）全部在CUDA Cores上通过FMA执行，Tensor Cores完全闲置。CUDA Cores承担dequantization（INT4→FP16）、scaling、element-wise运算等memory-bound任务，消耗instruction slots、register bandwidth、L1/L2 capacity，降低occupancy和tile sizes，剩余给compute-heavy matmul的资源极少。

  全栈执行例子（以QServe on A100, LLaMA-3.1-8B, GQA, 4-bit KV cache, 32K context, decoding step为例）：
  - 算法层：4-bit KV cache量化。K/V在prefill后量化为INT4，decoding时新token也即时量化。GQA下h_q=32, h_k=8，4个query heads共享1组K/V。
  - 系统框架层：QServe在FlashAttention kernel内fuse量化/dequantization操作。Page management管理KV cache内存分配。
  - 编译框架层：论文未明确说明。QServe使用CUDA实现，无编译框架自动代码生成。
  - kernel调度层：FlashAttention-style block-wise tiling (Tm×Tn tiles) + CUDA Cores FMA。Q tile(M=1) × K tile (N=128)→dequantization (INT4→FP16) + QK^T GEMV全在CUDA Cores→该warp沿N维串行处理每个tile，每tile都需dequant→频繁CUDA Cores stall。Nsight分析：dequantization overhead占近半数kernel时间（Fig 15a），CUDA Cores FMA consume 72.24% pipe utilization（Fig 15b），Tensor Cores utilization 0%。GQA下K/V tile需为32 query heads服务但CUDA Cores FMA compute-bound，GQA speedup仅1.4×（RTX4090，Fig 10）。
  - 硬件架构层：A100 GPU。Tensor Cores (312 TFLOPS FP16) 闲置，CUDA Cores (19.5 TFLOPS FP32) 成为瓶颈。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出BitDecoding，cooperative use of Tensor Cores + CUDA Cores for low-bit KV cache decoding。针对baseline三大缺陷：

  **缺陷1：Tensor Cores布局不匹配 → 已有系统无法在Tensor Cores上执行低比特数据**
  → **BitDecoding方案**：(a) Hardware instruction-induced layout：ldmatrix的thread-to-register mapping自动将低比特packed data隐式保留FP16 interleaved layout→Residual Kernel内每thread在register中做完计算→量化→pack→写global memory→Packing Kernel用相同ldmatrix/mma配置加载时layout天然正确。无需global reshape或手动layout transform。(b) Remapping for fast dequant：lop3-based 75316420 pattern→高效INT4/INT2→FP16转换（对比naive static_cast极慢）。(c) Residual block alignment：Nr=Pn×Wn×R确保packed tile精确填充Tensor Cores warp-tiling，消除fragment underfill。

  **缺陷2：Dequantization频繁stall Tensor Cores → 低GPU occupancy**
  → **BitDecoding方案**：(a) 新颖warp layout：Wm=1（decode Q len<16通常仅1 token），将资源重新分配给Wn→多个warp并发做dequantization→SM warp scheduler overlap dequantization与Tensor Cores mma，消除serialization bottleneck。(b) Asynchronous pipeline：register级软件pipeline——slice i做mma (Tensor Cores)同时slice i+1做ldmatrix+dequant (CUDA Cores)→producer-consumer持续流动。(c) Cooperative softmax：引入sTMP和sAcc shared memory buffer→跨warp reduction→sAcc通过ldmatrix重载P→保证后续PV mma layout正确。仅0.5% overhead换取correctness。

  **缺陷3：系统不支持多样的量化算法和attention变体**
  → **BitDecoding方案**：(a) Residual Kernel统一支持tensor-wise和channel-wise scaling：通过residual block内按seq_len维度做channel-wise、按hidden维度做tensor-wise reduction。Warp-level __shfl_xor_sync reduction + shared memory cross-warp aggregation计算scale/zero→half2 compact存储。(b) Query Transformation：Q reshape [1,(gq,hkv)]→[gq,hkv]，使MHA/MQA/GQA统一在Tensor Cores上高效执行，GQA的grouped queries形成大GEMM block。(c) 架构可移植性：Hopper用STSM+wgmma_SS；Blackwell直接用原生mxfp4 mma；layout-agnostic设计自动适配不同GPU世代的fragment layout。

  论文方法全栈执行例子（以LLaMA-3.1-8B, GQA (gq=4), 4-bit KC, 32K context, H100, decode step为例）：
  - 算法层：4-bit channel-wise KV cache + FP16 Q/score。Scale/zero compact为half2。Residual block size Nr = 8 × 4 × 4 = 128 (Pn=8 for mma.m16n8k16, Wn=4, R=4 for INT4→INT16)。
  - 系统框架层：BitDecoding提供Residual Kernel + Packing Kernel双kernel设计。Prefill: Residual Kernel量化→packed KV cache + residual FP16 cache。Decode: Packing Kernel做fused attention with low-bit data。
  - 编译框架层：论文未明确说明。CUDA手写PTX指令级实现，无编译框架。
  - kernel调度层（核心创新）：Query Transformation reshape Q [1, (4, 8)] → [4, 8]→4个query heads并行形成4×8 GEMM→full Tensor Cores occupancy。Packing Kernel: cp.async异步加载Q到shared mem (cp.async.cg)→加载Kpack (low-bit packed) 和Kp (scale/zero, half2)→ldmatrix加载packed data→lop3 75316420 remap→INT4→FP16 dequant (CUDA Cores, 4 warps并行)→异步overlap: slice i做QK^T mma (Tensor Cores wgmma_SS) 同时slice i+1做ldmatrix+dequant→cooperative softmax (sTMP跨warp reduction)→sAcc storer2s P→ldmatrix s2r重载→PV mma (wgmma_SS)→output。Hopper: STSM将dequantized K写入shared memory→wgmma_SS直接从shared memory读B矩阵。Tensor Cores utilization从baseline的0%提升至19.66%（Table III），dequantization overhead降至<15% (4-bit)，整体speedup 8.0× vs FP16 FlashDecoding-v2。
  - 硬件架构层：H100 (SM90, wgmma, TMA, STSM)。Blackwell B200 (SM100, native mxfp4 mma) 绕过所有dequant——Q (FP16) × K_packed (mxfp4) 直接硬件mma，speedup 8.6×。

  Baseline缺陷→BitDecoding方案映射：
  | Baseline缺陷 | BitDecoding方案 | 效果 |
  |-------------|----------------|------|
  | CUDA Cores-only FMA执行→Tensor Cores完全闲置 | Cooperative CUDA+Tensor Cores: ldmatrix自动induce layout + lop3快速dequant→Tensor Cores mma | TCs utilization 0%→19.66%, speedup up to 8.6× |
  | Dequant频繁stall→低GPU occupancy | 新颖warp layout (Wm=1, ↑Wn) + async pipeline: dequant与mma重叠 | dequant overhead <15% (4-bit), <35% (2-bit) |
  | 低比特layout不匹配Tensor Cores→无法正确执行 | Hardware instruction-induced layout + residual block alignment (Nr=Pn×Wn×R) | 零overhead layout transform, kernel correctness verified |
  | 不支持多样化量化算法和attention变体 | Residual Kernel统一channel-wise+tensor-wise scaling + Query Transformation支持MHA/MQA/GQA | GQA下speedup 3× vs QServe仅1.4× |
  | weight-optimized mpGEMM (Marlin/Ladder) 不适用于动态KV cache | Online fused quantization+packing in Residual Kernel, overhead 0.008ms/decode step (vs Marlin 0.41ms) | 2个量级quantization overhead降低 |

## AQPIM: Breaking the PIM Capacity Wall for LLMs with In-Memory Activation Quantization

- baseline方法是什么？
  Baseline是SOTA PIM-based LLM inference accelerator AttAcc! [56] 配合标准KV cache处理方法。(1) AttAcc!假设KV cache完全fit在PIM的有限on-chip memory中，不做压缩；(2) 当KV cache溢出HBM-PIM memory时，Baseline被迫通过offloading (KV cache→CPU memory via PCIe) 处理，产生巨大communication overhead（GPU-CPU通信占decoding latency的90~98.5%）；(3) 如果使用现有quantization方法(KVQuant/SKVQ等uniform/non-uniform quantization)，需在面积受限的BankPE中添加INT32 MAC units等额外ALU用于scaling/dequant，面积开销从FP16-only的50%增加到FP16+INT32的126%，严重损害memory density；(4) 如果使用sparse attention (SnapKV/StreamingLLM) 配合offloading，scattered memory access pattern与PIM对data locality的刚需冲突。

  全栈执行例子（以Mistral-7B-Instruct-v0.2, LongBench, S_in=4096, S_out=128/batch_size=16为例）：
  - 算法层：标准attention GEMV计算qK^T (q: [1,d] × K: [N,d]^T)，KV cache无压缩，长context下N增长导致KV cache size线性膨胀。当HBM-PIM capacity (4×16GB=64GB) 无法容纳全部KV cache时，触发KV offloading到CPU memory。
  - 系统框架层：GPU+HBM-PIM异构系统，GPU负责projection/FFN，PIM负责attention GEMV。无KV cache compression，decode阶段GPU需等待PIM完成attention后处理后续层。
  - 编译框架层：论文未明确说明。
  - kernel调度层：AttAcc!的attention kernel直接使用BankPE FP16 MAC units做标准GEMV (q×K^T)，每decode step需读取完整K matrix所有行。KV cache expand后offloading via PCIe → GPU-CPU communication latency 11.385ms/decode step (batch_size=32, gpu+cpu场景，图13)，远超matmul本身的0.181ms。
  - 硬件架构层：AttAcc!的HBM-PIM架构：BankPE在DRAM die bank旁（面积受限），BufferPE在buffer die（含accumulators和softmax units）。不支持compression、不支持intra-row indirection、无PQ相关commands。BankPE仅含FP16 MAC units，无INT/quantization专用ALU。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出AQPIM框架，通过algorithm-hardware co-design用PQ在线聚类量化解决PIM capacity wall。核心设计及其与Baseline缺陷的对应关系：

  1. **Capacity Wall (Baseline: KV cache溢出PIM capacity) → PQ-based online KV cache quantization**：利用PIM高内部带宽在线执行k-means clustering生成codebook（m=32 subvectors, K=512 centroids），在prefill期间与GPU computation并行完成（4次迭代收敛），compression ratio最高80%+，将KV cache footprint从~120GB降至~0GB水平（图13, gpu+pq: 0.181ms, aqpim: 0.047ms）。

  2. **Area Overhead (Baseline: 添加quantization ALU需126% area) → PQ eliminates dequantization, reuses existing FP16 MAC**：AQPIM将GEMV qK^T转换为query splits×codebook的lookup+summation（图5: query→m subvectors→multiply with codebook→inner product matrix→lookup indices→sum→qK^T approximation），无需dequantization step，完全使用现有BankPE FP16 MAC units (ADD/MUL/SUM)。仅添加intra-row indirection硬件0.0565mm² (0.43% of BankPE area)。

  3. **Random Access Penalty (Baseline: PQ lookup产生大量随机DRAM row activation) → Page-aware windowed clustering + Intra-row indirection**：算法端保证每个window内不超过512个centroid={512 inner product values}，完全fit在1KB HBM row buffer中；架构端添加GRF→MUX→column decoder的indirection datapath，将随机logical access转为单一row-buffer hit。

  4. **Accuracy Loss (Baseline: naive PQ equal treatment causes accuracy drop) → Importance-weighted k-means + Channel pre-sorting**：Weighted clustering使高attention score token获更小quantization error（µ_k = Σ w_n x_n / Σ w_n, w=last t tokens' attention scores）；Channel pre-sorting offline生成sorting matrices absorb到projection weights，将高cosine similarity channels聚合同一subvector减少信息损失。Ablation: Standard PQ avg 44.29 → AQPIM avg 50.00 (+5.71), compression scenario K=128 centroids。

  全栈执行例子（以Mistral-7B-Instruct-v0.2, LongBench, S_len=32768, batch_size=16):
  - 算法层：PQ-based attention → lookup+summation替代GEMV。Importance-weighted k-means (t=32, m=32 subvectors, K=512 centroids) → codebook+indices替代full KV cache。Channel pre-sorting (Wikitext-2-v1 offline) absorb到projection。Sink tokens (前8)+sliding window (最近32)保留full precision。
  - 系统框架层：GPU+HBM-PIM pipeline: GPU生成qkv→offload到PIM→PIM append PQ indices+PQ-attention→output回传GPU→GPU projection/FFN。Sequence-by-sequence pipelining隐藏GPU-PIM sequential gap。Head-wise HBM mapping + subvector-wise bank mapping。
  - 编译框架层：论文未明确说明。
  - kernel调度层：BankPE执行DC/ATNK/ATNV (FP16 ADD/MUL/SUM)，BufferPE执行CA/SFM (MIN/DIV/EXP)。4-iteration codebook generation在prefill期间与GPU并行→query×codebook inner product→intra-row indirection lookup from row buffer→summation for qK^T→softmax→attention reconstruction。256个centroid = 512 FP16 values = 1KB = 1 row buffer → 每window仅1次row activation。
  - 硬件架构层：HBM3-PIM with BankPE+BufferPE dual architecture, intra-row indirection (GRF→MUX→column decoder), new PIM commands (PIM_SET_CONFIG/PIM_MAC_AB/PIM_SFM/PIM_RET等), page-aware memory allocation。Area overhead仅0.43%。Decoding per-step latency 0.12× vs GPU baseline (8.33× speedup), energy 0.07× vs GPU baseline (图14, S_len=32768)。3.4× speedup over SOTA PIM (AttAcc!)。

## Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

- baseline方法是什么？
  Baseline是两类：**算法baseline** (FrameFusion token merging、AdapTiV/CMC token pruning) 和**硬件baseline** (vanilla systolic array、AdapTiV accelerator、CMC accelerator、GPU+FrameFusion)。

  **算法Baseline缺陷分析**：
  - FrameFusion [20]：merge temporally redundant tokens跨帧，但产生不规则稀疏模式，GPU Tensor Core难以高效利用，且runtime overhead增加up to 36.8%
  - AdapTiV [70]：intra-frame token-level similarity，仅支持静态图像，忽略视频language interaction；用sign-bit做轻量相似度检查，细粒度不足
  - CMC [56]：video-codec-inspired inter-frame redundancy search，利用H.264类压缩，但忽略language inputs；global token-wise操作，需等完整token输出写回DRAM后再压缩，带来高带宽和差locality。CMC虽有46% sparsity但仍有79% dense DRAM traffic
  - 三者共同缺陷：都仅关注visual redundancy（ViT导向），忽略cross-modal semantic intent（prompt对token重要性的影响）；都做token-level coarse granularity操作，无法捕获motion引起的sub-token partial alignment

  **全栈执行例子（CMC baseline, LLaVA-OneVision-7B在VideoMME上推理）**：
  - 算法层：CMC做inter-frame token-level similarity search → 46% sparsity但accuracy 62.11 vs original 63.32；AdapTiV做intra-frame token merging → 39.55% sparsity但accuracy 62.22；FrameFusion merge temporally redundant tokens → 固定70% sparsity但accuracy降至62.54
  - 系统框架层：CMC/adapTiV算法在PyTorch中实现 → GPU上运行 → token pruning/merging结果转换为不规则稀疏mask → 难以映射到高效batch GEMM
  - 编译框架层：论文未明确说明
  - kernel调度层：CMC hardware accelerator用global token-wise方法：systolic array输出完整M×N token矩阵→全部写回DRAM→外部codec unit读回DRAM做token-wise similarity search→压缩结果写回DRAM→下次GEMM前读回。Global执行阻止fine-grained scheduling，增加memory pressure。CMC需1.4MB额外buffer
  - 硬件架构层：Vanilla systolic array (32×32 PE, FP16, weight stationary) → 无compression → 全量tokens参与GEMM → 高DRAM traffic。AdapTiV accelerator集成token merging但still coarse token-pair操作，transfer uncompressed tokens before processing

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出Focus，通过**Multilevel Streaming Concentration**和**hardware-algorithm co-design**解决baseline四大缺陷：

  **缺陷1：忽略cross-modal semantic intent → static token importance metrics不准**
  → **Focus方案**：SEC从attention SoftMax(QK^T)中提取text-to-image (T×M) cross-modal attention block，计算每image token从所有text tokens和heads接收到的最大attention score。prompt问"dog"时attention集中在狗，问"flower color"时转向花位置。SEC基于实际cross-modal attention做prompt-aware pruning，不再依赖静态magnitude/saliency。

  **缺陷2：token-level coarse granularity → 无法捕获sub-token redundancy和motion-caused partial alignment**
  → **Focus方案**：SIC将token embedding分为32-dim vectors做vector-wise cosine similarity matching。2×2×2时空block内localized comparison（8 vectors per block）。64%的8-dim vectors超过cosine similarity 0.9（vs仅18%的3584-dim full tokens），实现82.8% sparsity（vs token-wise 73.0%）。Vector-level匹配允许每个vector匹配多个candidates，捕获richer sub-token similarity。

  **缺陷3：global token-wise compression → high DRAM traffic, poor locality, non-streamable**
  → **Focus方案**：所有compression on-chip、tile-local、streaming。SIC不等整层token就绪，每个GEMM m×n tile产生后立即压缩（vector-level similarity matching + deduplication），仅deduplicated vectors和similarity map写回DRAM。SEC top-k sorter与image attention GEMM完全重叠，不占用critical path。CMC 46% sparsity → 79% dense DRAM traffic vs Focus ~81% sparsity → 21% bandwidth。

  **缺陷4：irregular sparsity不利于硬件利用 → GPU/accelerator无法高效执行**
  → **Focus方案**：将算法稀疏性转化为hardware-friendly structured/tile-local稀疏。GEMM tile对齐（m=1024, n=32）、convolution-style layouter的conflict-free bank mapping（Bank=f%2×4+r%2×2+c%2）支持2×2×2 block内8 vectors无conflict并行读取。Similarity Scatter用2a-wide accumulator做concurrent reconstruction。SpMM-like execution：GEMM对compact vectors (p<1024)执行，按similarity map scatter partial sums恢复full output。Focus Unit面积仅3.21 mm²（2.7% of SA），功耗736 mW（0.9% of SA）。

  论文方法全栈执行例子（以LLaVA-Video-7B在VideoMME上推理为例）：
  - 算法层（核心创新）：Multilevel Concentration。Layer 3 attention: SEC提取text-to-image attention→importance vector→top-k sorter选40% tokens (约2500/6272)→保留tokens继续P(i)×V。Layer 9: SEC选30%。Layer 18: SEC选20%。Layer 26: SEC选10%。所有FC layers: SIC在每个GEMM tile后做2×2×2 block cosine similarity matching (threshold=0.9)→deduplicate vectors→similarity map记录映射。Scatter恢复full output→下一tile的Gather再压缩。最终avg sparsity 82.82%，accuracy 62.74 vs original 64.15 (-1.41%)
  - 系统框架层：Focus Unit作为modular component嵌入systolic-array accelerator memory interface → 拦截GEMM output和attention SoftMax → 不修改core compute pipeline。SEC在attention layer集成 → SIC在FC/O projection/PV GEMM层集成。Similar to pooling/activation function —— modular, scalable
  - 编译框架层：论文未明确说明
  - kernel调度层（核心创新）：Streaming dataflow。t=1: Attention SoftMax → SEC importance analyzer (parallel max units) → a-way bubble sorter与image attention GEMM (Q(i)K^T)重叠→sorting在attention GEMM完成前结束。t=2: SEC pruning + offset encoding → P(i)×V仅加载保留tokens。t=3: FC GEMM tile (1024×32) → convolution-style layouter重组为FHW布局 → SIC similarity matcher (7 pairwise cosine per key, 8×m cycles max vs GEMM 112×m cycles) → deduplicated vectors + similarity map写回DRAM。t=4: 下一层GEMM对p个compact vectors计算 → Similarity Scatter根据similarity map复制partial sums到full output positions → 2a-wide accumulator并发累加 → tile完成后Similarity Gather再次压缩。Scatter-Gather循环贯穿所有FC层
  - 硬件架构层（核心创新）：Focus Unit (SEC+SIC) integrated near systolic array memory interface。SEC: importance analyzer (并行max units + on-chip 25KB buffer) + a-way bubble sorter + offset encoder (lightweight registers)。SIC: convolution-style layouter (16KB buffer for 256-vector window, conflict-free bank mapping) + similarity matcher (dot-product unit + L2-norm precompute buffer, <1% SA area) + similarity map buffer + scatter accumulator (2a-wide=64)。Total on-chip buffer 734KB。TSMC 28nm: 3.21 mm², 736 mW —— 4.47× speedup vs SA, 7.90× vs GPU, DRAM traffic 4.9× reduction vs dense SA

## V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval

- baseline方法是什么？
  Baseline是GPU上的fixed top-k KV cache retrieval方法（FlexGen、InfiniGen、InfiniGenP、ReKV），用于streaming video LLM。核心设计：(1) KV cache offloading：将完整KV cache offload到CPU memory或storage，缓解GPU memory压力。(2) Fixed top-k selection：按固定token budget选择"最重要"的KV token取回GPU memory，利用GPU的规则并行能力和可预测资源分配。(3) KV prediction hiding：在前一层attention中计算KV prediction，提前prefetch selected KV cache，尝试将fetch latency与computation重叠。

  全栈执行例子（以COIN benchmark, Llama-3 8B, AGX Orin GPU, 40K KV cache length, FlexGen baseline为例）：
  - 算法层：fixed top-k KV cache retrieval。FlexGen将KV cache offload到storage→generation时按固定比例选择token→通过PCIe 3.0 x4 (4 GB/s)从SSD取回selected KV→GPU memory仅保留selected cache。InfiniGen仅在generation stage做retrieval，InfiniGenP扩展到prefill，ReKV做frame-level selection。token selection ratio均匀应用于所有layers和heads。
  - 系统框架层：VideoLLM-Online streaming video LLM framework。Vision Tower (SigLIP)→MLP Projector→LLM decoder layers。每新frame到达后sequential prefill处理，无framework-level KV retrieval优化。
  - 编译框架层：论文未明确说明（使用PyTorch/CUDA backend）。
  - kernel调度层：GPU上执行fixed top-k selection。对每层Q×K^T计算attention score→对所有tokens排序→选top-k个token indices→通过PCIe从CPU/storage fetch selected KV entries→full attention computation。40K cache sequence length时：KV retrieval占85% total latency，其中KV prediction computation 40%、KV cache fetch 39%——尽管retrieval仅占23%的计算量。fixed top-k的问题：(a) layer/head间token importance分布差异大，固定k导致不重要层/head over-fetching浪费PCIe bandwidth，关键层/head under-fetching导致accuracy degradation；(b) streaming video prefill中query tokens多（每帧多个token），每个query token需要独立retrieval，token budget远大于text generation场景；(c) PCIe带宽远低于GPU memory带宽（4 GB/s vs 204 GB/s），无论baseline如何优化，数据搬运始终是核心瓶颈。
  - 硬件架构层：NVIDIA AGX Orin (204 GB/s LPDDR5, PCIe 3.0 x4 4 GB/s, 32 GB, ~40W)。GPU memory有限，连续视频输入数分钟内KV cache超过容量→KV cache必须offload到SSD/CPU→PCIe成为系统性瓶颈。Fixed top-k method为GPU规则并行设计，在streaming prefill workload下因不规则访问和条件分支导致GPU严重underutilization（roofline分析仅达6.6%理论峰值）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出V-Rex，软件-硬件协同方案，通过ReSV动态retrieval算法+DRE硬件加速单元系统性地解决fixed top-k retrieval在streaming video LLM prefill中的三大缺陷：

  **缺陷1：Fixed top-k导致over-fetching/under-fetching → 浪费PCIe带宽或降低accuracy**
  → **V-Rex方案**：ReSV的WiCSum dynamic thresholding替代fixed top-k。对每layer/head独立计算Query×KeyCluster^T ScoreCluster→按每row score×token count加权求weighted sum→从高分bucket累计直到超过threshold Thr-wics即停止。每layer/head自适应选择不同数量token（论文Fig.20显示layer间retrieval ratio从4.2%到44.0%变化，head间也存在显著变异），比ReKV平均少检索3.0× token，同时accuracy仅比baseline下降0.8%。动态threshold消除了固定k在低重要度layer/head的over-fetching和高重要度layer/head的under-fetching mismatch。

  **缺陷2：KV prediction computation随序列增长 → GPU上不规则的clustering和sorting导致严重underutilization**
  → **V-Rex方案**：DRE的KVPU专用硬件加速ReSV不规则计算。(a) HCU用bit-level XOR accumulator做Hamming distance clustering，替代昂贵的cosine similarity计算——hash-bit仅≤0.5%原始dimension，XOR+popcount远轻于高维dot-product；(b) WTU用early-exit sorting pipeline（bucket sort→cumulative sum→threshold check，从高分bucket开始）终止sorting在中途（平均仅需处理16% rows的scores即可达到threshold）；(c) 同时利用视频帧间高度时空相似性（相邻帧key cosine similarity热力图验证），hash-bit Hamming distance与cosine similarity相关性0.8，足以支撑准确聚类。消融实验中AGX+ReSV仅达2.8× speedup（KV prediction仍占48% latency），+KVPU后达6.0× speedup和9.2× energy reduction，证明GPU无法高效执行ReSV的不规则操作，必须专用硬件加速。

  **缺陷3：PCIe bandwidth bottleneck + irregular KV fetch → 数据搬运无法与computation有效重叠**
  → **V-Rex方案**：KVMU通过两层memory优化最大化PCIe带宽利用并隐藏fetch latency。(a) Hierarchical memory system：recent KV→V-Rex on-chip memory (fast access)→old KV offload to CPU/storage→selected entries prefetch回V-Rex memory→attention computation。三级hierarchy使retrieved data仅占1% DRAM bandwidth，可与attention和FFN并发执行。(b) Cluster-wise memory mapping：KVMU将同hash cluster的token连续存储于offloaded KV cache中，使得一次PCIe fetch可transfer cluster内多个token（平均32 tokens/cluster），提高PCIe有效带宽。Bandwidth analysis显示KV prediction短时spike 600 GB/s可hidden in attention，KV retrieval因PCIe bottleneck仅占1% DRAM bandwidth可全程concurrent with LLM computation。

  论文方法全栈执行例子（以COIN benchmark, Llama-3 8B, V-Rex8 edge deployment, 40K KV cache length为例）：
  - 算法层（核心创新）：ReSV动态KV cache retrieval。Video frame到达→Vision Tower+MLP→LXE decoder layer L。QKV generation→RoPE→hash-bit generation (Nhp=32 hyperplanes, binarize)→HCU计算current hash-bit与HC table中KeyCluster hash-bit的Hamming distance（XOR+popcount，Thhd=7）→聚类结果更新HC table (cluster id/token idx/KeyCluster/token count)。LXE计算Query×KeyCluster^T (仅对representative KeyCluster，非完整key cache)→ScoreCluster送入WTU→preprocess计算每row weighted sum/threshold→高分bucket sort→cumulative sum check vs Thr-wics=0.3→early-exit→输出selected cluster→HC table映射回token indices。KVMU通过PCIe 3.0 x4从SSD prefetch selected K/V entries（cluster-wise continuous address→bulk transfer）。Light Attention：仅对selected cluster tokens做attention计算→与KV prediction for layer L+1重叠。
  - 系统框架层：V-Rex accelerator = LXE + DRE。LXE处理主LLM计算+ReSV中规则矩阵运算（hash-bit generation的MatMul, Query×KeyCluster^T）。DRE处理ReSV中不规则计算（HCU bit-level clustering, WTU early-exit thresholding）和KV memory管理（KVMU hierarchy+cluster mapping）。video processing为iterative prefill：每frame sequential通过全部decoder layers，每层内部KV prediction→prefetch→execution流水化。
  - 编译框架层：论文未明确说明。
  - kernel调度层（核心创新）：DRE runtime pipeline。Step 1: LXE VPE生成frame hash-bit→Step 2: HCU做Hamming distance clustering (NHCU_h=1 XOR accumulator with 16-wide inputs)→Step 3: LXE DPE做Q×KeyCluster^T→Step 4: WTU做WiCSum thresholding (early-exit sorting, 1 core 16-wide)→Step 5: KVMU prefetch selected KV from storage/CPU via PCIe (cluster-wise contiguous mapping)→Step 6: Light Attention on selected tokens。Step 1-4为KV prediction (与当前layer attention重叠)，Step 5为KV retrieval (与attention+FFN重叠)，Step 6为execution。消融实验证明：ReSV alone (AGX+ReSV) 2.8× speedup → +KVPU 6.0× → +KVMU 8.1× speedup and 10.2× energy savings。
  - 硬件架构层（核心创新）：V-Rex DRE硬件。单核14nm RTL：1.89mm², 2.61W。HCU: bit-level XOR accumulators+hash-bit MEM (current/key cache)+HC table updater→仅0.01mm² (0.28% area), 2.99mW (0.11% power)。WTU: score MEM+token count MEM+bucket sorters (upper/lower)+multipliers+adder tree+bucket range updater→0.02mm² (1.23%), 39.04mW (1.49%)。KVMU: KV reordering+hierarchical memory controller+cluster mapping logic→0.01mm² (0.53%), 15.01mW (0.58%)。DRE总计2.0% area, 2.4% power overhead。V-Rex8: 15.12mm², 35W total (vs AGX Orin 200mm², 40W)。KVMU overhead：HC table平均占1.67% of full KV cache (32 avg tokens/cluster)。实现3.9-8.3 FPS real-time streaming edge inference，相对AGX+FlexGen达1.9-19.7× speedup和3.1-18.5× energy efficiency。

## RPU - A Reasoning Processing Unit

- baseline方法是什么？
  Baseline是NVIDIA H100/H200 GPU系统上的低batch LLM decode推理。H100系统特征：(1) 内存层使用HBM3e（如单个stack 1280GB/s、48GB、BW/Cap≈27），带宽和容量强绑定，大量容量在低batch decode中未使用；(2) 系统层为monolithic die设计，compute-to-bandwidth ratio约200 OPs/Byte，仅30-40% TDP分给memory interface，decode时大量compute/cache资源闲置（power trace: decode avg 239.9W vs prefill 634.2W, BW utilization仅32.2%）；(3) 微架构层为shared memory NUMA + 统一memory access (UMA) + randomized address mapping，kernel launch和synchronization overhead在低batch小kernel下显著（小矩阵VMM kernel执行时间tens of microseconds, overhead同量级），无法持续饱和HBM带宽。

  全栈执行例子（以Llama3-70B FP8, 4×H100, BS=32, 16K prefill / 2K decode为例）：
  - 算法层：标准transformer decode（FP8 weights, BF16 activations），weight matrices column-sharded across 4 GPUs via tensor parallelism。
  - 系统框架/Serving层：vLLM + NVIDIA Dynamo。Decode phase下batch中每个query串行生成token→attention sequential computation→KV cache随seq len增长持续膨胀→memory bandwidth bound。
  - 编译框架层：PyTorch 2.2 compiled dense-linear kernels。
  - kernel调度层：GPU CUDA kernel launch model→host-driven offload→每个VMM kernel launch开销、tensor-parallel通信延迟（NVLink/NVSwitch）、cross-SM synchronization。小batch下kernel执行时间短（tens of microseconds），launch+sync overhead占显著比例，导致仅32.2% BW utilization。
  - 硬件架构层：H100 SXM (132 SMs, 80GB HBM3, 3.35TB/s peak BW)，HBM energy per bit ~3.44pJ/bit [43]，monolithic die ~814mm²，UMA memory controller + L2 cache (50MB) → 长距离片上数据移动增加energy。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出RPU（Reasoning Processing Unit），通过三层协同设计系统性地解决H100 baseline在低batch LLM decode中的memory wall瓶颈。

  **缺陷1：HBM高容量过度配置 → 为带宽买容量造成成本/能耗浪费（memory overprovisioning paradox）**
  → **RPU方案（HBM-CO）**：保留HBM的shoreline bandwidth架构但削减主要贡献容量的结构（ranks/banks/subarrays）→BW/Cap从27提升到341（候选Pareto-optimal），energy per bit从~3.44降至1.45pJ/bit（2.4× improvement），带宽/美元提高5×，总模块成本降低35×（去掉192×未使用容量）。Chiplet-based modular memory architecture允许在系统level灵活scale bandwidth and capacity across many smaller stacks。

  **缺陷2：GPU monolithic die的power/area provisioning偏向compute而非bandwidth → memory-bound decode闲置大量compute/cache资源**
  → **RPU方案（Chiplet Compute Fabric + Power/Area Reprovisioning）**：将compute die从monolithic拆为多个chiplet→相同compute die area下暴露近10× memory IO shoreline（~600mm vs H100 ~60mm）→每chiplet tightly coupled with HBM-CO stacks→70-80% TDP分配给memory interfaces→compute-to-bandwidth ratio调至32 OPs/Byte（H100 ~200 OPs/Byte）→Roofline向左下移以匹配低batch decode的low arithmetic intensity。ISO TDP下RPU提供2×+ bandwidth。

  **缺陷3：GPU的host-driven offload + shared memory NUMA + barrier synchronization → 低batch小kernel无法持续饱和HBM带宽（32.2% utilization）**
  → **RPU方案（Decoupled Microarchitecture + NUMA at All Scales）**：每core私有HBM-CO channel + local SRAM buffer，全NUMA无shared memory→消除coherence overhead。Memory/Compute/Network三pipeline硬件解耦→Pipeline Arbiter用buffer entry粒度valid counter实现data-driven同步（非global barrier）→memory pipeline在compute/network stall时继续预取weights/KV cache到on-chip buffer→compute后续"catch-up"消耗已预取数据。BS=1时RPU完全饱和memory bandwidth（roofline performance）。BS=32时decoupled pipeline吸收compute-bound weight layers和memory-bound KV cache layers间的phase imbalance→overall latency improvement up to 1.6×。

  **缺陷4：GPU通用编程模型和runtime scheduling overhead → 确定性执行不足，难以在低batch下持续利用峰值带宽**
  → **RPU方案（Custom ISA + Deterministic Compilation）**：RPU ISA将优化dataflow硬化为CISC-style指令，每条指令执行固定streaming schedule→compiler static order所有DMA和compute指令→消除runtime scheduling overhead。Autonomous execution消除GPU的host-driven offload模型→每core独立执行long-running instruction loop→只在layer边界trigger host interrupt。Pipeline Arbiter flags嵌入每条指令→software-defined但hardware-enforced同步→deadlock-free保证。

  论文方法全栈执行例子（以Llama3-8B, 64-CU RPU, MXFP4 weights, FP8 KV cache, BF16 activations, BS=1, Seq Len=16K为例）：
  - 算法层：不修改模型或推理算法，使用MXFP4 block quantization（Stream Decoder on-the-fly dequantize to BF16）压缩off-chip weight storage。
  - 系统框架/Serving层：Prefill和decode分离（Dynamo/Splitwise execution model）→prefill由GPU处理→KV cache转入RPU HBM-CO memory→RPU执行decode。Host仅在layer transition接收interrupt，不参与per-token kernel offload。
  - 编译框架层：Python compiler trace PyTorch model→lowering torch.nn.Linear到三阶段micro-kernel（Loading/Looping/Launching）→pre-shard weights by C=64 column-wise→pre-quantize to MXFP4→generate synchronized memory/compute/network instruction streams with Pipeline Arbiter flags→static order所有操作。
  - kernel调度层（核心创新）：Decoupled three-pipeline execution。wQKV: network DMA broadcast activation（network latency limited）→memory DMA prefetch weights→compute waits→memory keeps prefilling buffer（~80KB ahead）。QK^T: network gather Q/K/V shards→distributed max collective + exp-sum reduction→compute stalls briefly→memory pipeline prefetches KV$/weights。wUp/wGate: compute runs at full utilization→memory prefetches deep ahead（~6MB/CU at BS=32）。Pipeline Arbiter在buffer entry粒度用valid counter协调，memory始终前进不因compute/network stall而停顿。
  - 硬件架构层（核心创新）：RPU chiplet system。每Core: 4×8×8 TMAC (BF16+FP32), HP-VOPs (FP32 vector ops), Stream Decoder (4-8bit on-the-fly dequant), Memory/Compute/Network DMA + Pipeline Arbiter, 1MB SRAM buffer, 32GB/s HBM-CO channel。每CU: 16 cores, 16MB on-chip memory, 512GB/s BW。每Package: 4 CUs, 2TB/s BW, 64MB on-chip memory。Ring network topology: in-package UCIe-S short-reach (<10mm), off-package PCB-routed interconnect。Llama3-405B at 428 CUs: 1.0ms/token, sustaining >200 TB/s tensor-parallel memory bandwidth。

## RoMe: Row Granularity Access Memory System for Large Language Models

- baseline方法是什么？
  Baseline是conventional HBM4-based memory system，配置为每cube 32 channels（64 pseudo channels）、8 Gbps data rate、2 TB/s bandwidth、32B access granularity、1KB row size。MC使用FR-FCFS scheduling policy + open-page policy + per-bank refresh。bank states包含Idle/Activating/Active/Reading/Writing/Precharging/Refreshing七个状态，需管理15个timing parameters，bank FSM数量等于每PC所有bank数。MC必须做bank group interleaving和PC interleaving来最大化带宽利用率。每64-bit data channel需要10 row C/A pins + 8 column C/A pins。在LLM serving场景中，decode阶段的weight/KV cache/activation以KB-MB级连续块被顺序访问，但传统HBM4将这些大块访问拆成128个32B cache-line transactions。

  全栈执行例子（DeepSeek-V3 decode阶段读取一段weight block，HBM4 baseline）：
  - 算法层：LLM decoder block执行GEMV（decode阶段），需要读取weight矩阵连续block（如12MB weight chunk），activation为单token向量。
  - 系统框架/Serving层：accelerator DMA engine发出memory read requests→MC address mapping将物理地址映射到channel/PC/bank group/bank/row/column→每个12MB block被分解为~384K个32B cache-line read requests→request queue需depth≥45才能充分look ahead做bank-level parallelism。
  - 编译框架层：论文未明确说明（使用底层硬件DMA直接管理memory transfer，无编译框架介入）。
  - kernel调度层：论文未明确说明（无kernel级优化，所有调度在硬件MC层完成）。
  - 硬件架构层（核心）：MC command scheduler对每个32B request执行：检查bank state→若row-buffer hit直接发RD；若miss则发PRE→ACT→RD→维持open-page等后续访问→page policy判断何时发PRE。同时在不同bank group间交错（tCCDS=1ns间隔）和不同PC间交错。bank FSM需跟踪128 banks/channel的状态，timing constraints包括tRCDRD/tRAS/tRP/tCCDS/tRRDS/tFAW等15个参数。每通道18 C/A pins承载RD/WR/ACT/PRE/REF/MRS命令。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出RoMe，核心思路是将HBM接口从cache-line粒度的column-level RD/WR替换为row-level RD_row/WR_row（4KB粒度），并通过VBA、command generator、simplified MC三个组件重构整个memory hierarchy。

  **缺陷1：Baseline将LLM大块连续访问拆成大量32B碎片事务，导致MC调度复杂且queue压力大**
  → RoMe将AGM_C从32B提升到4KB，一个RD_row替代128个32B RD命令。request queue从需要≥45 entries降到2 entries即可饱和带宽。DRAM访问从column-level变为row-level后，ACT/PRE/WR/RD之间的复杂时序不再暴露给MC。

  **缺陷2：Baseline的bank group和pseudo channel是为了在cache-line粒度下扩展带宽而引入的，但迫使MC做BG/PC interleaving和复杂状态追踪**
  → RoMe提出VBA，从MC-DRAM接口中移除bank group和PC概念。单个VBA由两个不同BG的bank以time-multiplexed方式组成，两个PC并发工作。一个VBA即可提供满带宽，MC不再需要跨bank group和PC搜索ready bank。bank FSM从每PC所有bank（~128个）降到仅5个。

  **缺陷3：Baseline需要管理7个bank states、15个timing parameters、open/close/adaptive page policy的复杂决策**
  → RoMe MC只发RD_row/WR_row/REF三种命令，bank states缩为4个（Idle/Writing/Reading/Refreshing），timing parameters缩为10个。Row granularity本身保证了row-buffer locality，不再需要page policy——每次row access后自动precharge。调度简化为跨VBA交错+oldest-first公平性。RoMe MC scheduling logic面积仅为conventional MC的9.1%。

  **缺陷4：Baseline每通道需18 C/A pins（10 row + 8 column），随着HBM代际演进C/A-to-DQ pin ratio持续上升**
  → RoMe移除column C/A pins（8个），减少address bits（PC bit + 1个bank bit），C/A pins从18降到5（节省72%）。省下的13 pins/channel × 32 channels = 416 pins聚合后增加4个新channel，仅需额外12 pins。HBM cube从32 channels扩到36 channels，带宽从2 TB/s提升到2.25 TB/s (+12.5%)。

  **缺陷5：Baseline在LLM decode阶段受memory bandwidth-bound限制，但HBM带宽扩展受限于DRAM core frequency和access granularity**
  → RoMe通过row granularity释放C/A pins→增加channel count→直接提升bandwidth。Command generator放置在logic die中，将row-level command静态展开为传统DRAM command序列，内部处理tRRDS/tCCDS等时序约束。decode阶段TPOT在DeepSeek-V3/Grok 1/Llama 3上分别降低10.4%/10.2%/9.0%。DRAM energy降低1.9%/0.7%/0.7%，主要来自ACT数量减少（仅需minimal ACT）和interposer command traffic减少。

  论文方法全栈执行例子（以DeepSeek-V3 decode阶段读取一段weight block，RoMe）：
  - 算法层：不修改模型或推理算法，使用相同LLM architecture（MLA+MoE）和weight format（BF16）。
  - 系统框架/Serving层：accelerator DMA engine发出4KB-granularity memory requests→RoMe MC只做address mapping（channel/SID/VBA）和oldest-first scheduling→避免连续访问同VBA以保持带宽→MC发出RD_row命令。
  - 编译框架层：论文未明确说明（无编译框架介入）。
  - kernel调度层：论文未明确说明（无软件kernel修改，所有优化在硬件路径）。
  - 硬件架构层（核心创新）：(1) RoMe MC: 接收4KB request→address mapping→选择一个空闲VBA→发RD_row命令→仅需4 bank states + 10 timing params + 5 bank FSMs + 2-entry queue。(2) Command Generator in HBM logic die: 接收RD_row→插入tRRDS−tCCDS intentional delay→对VBA内Bank A发ACT→对Bank B发ACT→按tCCDS间隔交替发RD（两个BG的数据传输错开填满通道）→发PRE→VBA回到Idle。(3) VBA: 两个不同BG的bank以time-multiplexed方式工作，两个PC并发接收数据→有效row size 4KB→36 channels/cube提供2.25 TB/s带宽。(4) Refresh优化: MC每2×tREFIpb发一次per-bank refresh→command generator对VBA内两bank间隔tRREFD发两个REFpb→每VBA stall从2×tRFCpb降到tRFCpb+tRREFD。RoMe MC scheduling logic面积仅为conventional MC的9.1%，command generator占logic die 0.003%，总chip area overhead仅0.10%。

## LEGO: Supporting LLM-enhanced Games with One Gaming GPU

- baseline方法是什么？
  Baseline有三种：(1) SmallModel：用同系列小模型（Llama3-3B替换8B、Mistral-4B替换7B）降低推理开销，运行时按游戏平均rendering headroom将LLM inference切为等大小subtasks，渲染完成后提交一个推理subtask；(2) LayerSkip（LITE/CALM）：基于token-level confidence threshold做per-token early exit/layer-skipping决策，跳层策略确定后用与SmallModel相同调度方式。LITE对每层定义预定义confidence threshold→当某token在某层confidence超过threshold→跳过剩余层输出该token；(3) PilotFish式time-division GPU sharing：在渲染任务结束后利用帧间idle time dispatch LLM subtask，但不利用intra-rendering headroom。

  全栈执行例子（以LITE + Llama3-8B + BlackMyth + 200 APM on RTX 4090为例）：
  - 算法层：LITE对Llama3-8B fine-tune→每层定义confidence threshold。推理时每个decode token经过各transformer layer→计算confidence score→若在某层超过threshold→early exit跳过剩余层→head映射到vocabulary→输出token。不同token退出深度不同，平均跳层数无per-request保证。
  - 系统框架/Serving层：LITE + PilotFish scheduling。游戏以60 FPS渲染→每16.6ms一帧。LLM action到达（每300ms, 200 APM）→PilotFish在每帧渲染完成后dispatch等大小LLM subtask→仅利用inter-rendering headroom。scheduler不进行headroom预测，不拆分intra-rendering subtask。
  - 编译框架层：论文未明确说明（llama.cpp默认CUDA编译路径）。
  - kernel调度层：论文未明确说明（llama.cpp默认CUDA kernel，无定制kernel）。
  - 硬件架构层：Nvidia RTX 4090，无定制硬件。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出LEGO，algorithm-system co-design：算法侧用resource-oriented layer-skipping adaptor做知识蒸馏补偿；系统侧用headroom-maximizing LLM scheduler做细粒度共置调度。

  **缺陷1：SmallModel固定用小模型导致精度永久下降，即使GPU有空余headroom也无法恢复精度**
  → LEGO保留原始大模型主体权重不变，只在资源不足时按需跳过特定层并用adaptor补偿。当GPU headroom充足时（如100 APM），仅需跳≤5层即可满足90% case，精度接近原始模型。相比之下SmallModel的精度drop固定为平均20.41%（MMLU/ARC-C/SQuAD-2.0）。

  **缺陷2：LITE/CALM的token-level confidence-based跳层无法为每个请求提供资源预算保证，导致SLO violation**
  → LEGO反转跳层决策逻辑：先由资源预算决定跳层数量（scheduler预测headroom→选择层数），再在该约束下通过adaptor做知识蒸馏补偿。LITE即使在平均时间对齐latency target时仍有47.1%推理超时；LEGO确保每个action请求在execution window内完成，99th-percentile APM全部达到目标。

  **缺陷3：LITE-S强行加入SLO约束需要提前跳层，跳过了其自身机制认为重要的层，导致27.2% accuracy drop**
  → LEGO的adaptor在离线阶段通过similarity heatmap选择相似度最高的连续层段做蒸馏，而非逐token动态决定。跳层时用训练好的FFN adaptor替代跳过的层段，在跳12层时相比LITE减少86.3% accuracy loss。

  **缺陷4：PilotFish只利用inter-rendering headroom，高APM下GPU空闲不足导致大量跳层甚至FPS/APM violation**
  → LEGO scheduler同时利用inter-rendering（帧间）和intra-rendering（帧内）headroom。测得intra-rendering headroom平均0.24ms/gap（总平均1.39ms/frame, 最大3.1ms）。将LLM subtask粒度降到layer级（decode ~0.4ms）和sublayer级（prefill: attention ~0.5ms, FFN ~1.0ms）以填充这些极短的空隙。headroom usage提升最高28.6%。

  **缺陷5：逐帧headroom预测误差大，naive时间序列模型预测误差>3%且开销高（ARIMA~1s, SVM>50s)**
  → LEGO采用以execution window为单位的LR预测（前三个window总headroom→预测下一个window），最大预测误差仅1.3%、平均0.6%。因为window跨12-36帧，单帧波动被平滑。LR推理开销仅1.3ms (3-input) 或 0.9ms (runtime fit)。

  论文方法全栈执行例子（以LEGO + Llama3-8B + BlackMyth + 200 APM on RTX 4090为例）：
  - 算法层：离线训练adaptor。构建layer similarity heatmap→跳4层选L25-L29、跳8层选L23-L31等→每种跳层配置训练FFN adaptor（MSE loss）。在线推理：scheduler决定跳N层→替换对应transformer层段为adaptor→adaptor输入f_k输出f_{k+n}近似原表示→剩余层正常前向→输出token。
  - 系统框架/Serving层（核心创新）：llama.cpp front-end集成到UE4→修改traversal function加入调度→scheduler用LR基于前3个execution window总headroom预测下个window→选择跳层策略→prefill以attention/FFN sublayer为粒度利用intra-rendering gap→decode以transformer layer为粒度→rendering subtask完成时提交fine-grained LLM subtask（T_subtask ≤ T_minimal safety check）→整帧完成后切换coarse-grained subtask利用inter-rendering gap→每token生成后更新预测→检测QoS violation则动态调整跳层。
  - 编译框架层：论文未明确说明（llama.cpp默认CUDA编译路径，修改限于traversal调度逻辑）。
  - kernel调度层：论文未明确说明（使用llama.cpp默认CUDA kernel，无新定制kernel。调度发生在traversal layer，非kernel level）。
  - 硬件架构层：Nvidia RTX 4090消费级GPU，无定制硬件。论文强调LEGO不依赖RTX 4090特殊特性，可部署到其他gaming GPU。

## Deadlock-Free Bridge Module for Inter-Chiplet Communication in Open Chiplet Ecosystem

- baseline方法是什么？
  Baseline class为三类state-of-the-art inter-chiplet deadlock avoidance方法：MTR (boundary router turn-restriction based)、DeFT (virtual channel isolation for upward/downward traffic) 和RC (in-chip permission network for injection control)。它们共同将deadlock处理与chiplet内部NoC实现绑定，不能满足开放chiplet生态中"plug-and-play chiplet"的目标。

  全栈执行例子（以PARSEC workload + MTR baseline + 4-chiplet homogeneous mesh + gem5/Garnet为例）：
  - 算法层：论文未明确说明（PARSEC application不涉及ML算法）。
  - 系统框架/Serving层：论文未明确说明（gem5 full-system simulation，非serving框架）。
  - 编译框架层：论文未明确说明（无编译框架修改）。
  - kernel调度层：论文未明确说明（无GPU kernel调度）。
  - 硬件架构层：MTR在chiplet边界router实施turn restriction，禁止特定方向的包转向以打破跨chiplet CDG环。例如，限制从interposer vertical channel进入chiplet的包只能转向特定方向，避免形成cyclic dependency。执行流：In-Req从interposer到达chiplet boundary router → turn restriction检查 → 若方向违反限制则被阻塞或用其他vertical channel → 进入chiplet内部NoC → 触发cache controller查找 → 可能产生Out-Rsp → 从chiplet出口router发回interposer。MTR成本低，但会限制vertical channel选择造成load imbalance，且依赖TSV/interposer wiring layout，portable性差。
  - 芯片设计层：4个homogeneous chiplet通过shared interposer互连，chiplet和interposer均采用4×4 mesh。MTR的边界turn restriction与具体TSV/interposer wiring绑定，不同floorplan需要重新设计限制规则。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出DFBM (Deadlock-Free Bridge Module)，放在chiplet NoC与interposer NoC边界作为独立bridge module，包含Credit Management (CM) 和Cross-VN Deadlock Buffer (CVN-DB) 两大核心模块，通过transaction-aware packet injection control实现deadlock avoidance，且不要求了解chiplet内部NoC拓扑细节。

  **缺陷1：MTR的turn restriction限制vertical channel选择，导致non-uniform vertical channel distribution下load imbalance**
  → DFBM不依赖转向限制，而是通过CM的Expected Credit Table预测事务后续响应数并在入口预留credit。包可以自由选择任何可用的vertical channel，对非均匀vertical channel分布更稳健。在non-uniform条件下（12条vertical channel分布不均），DFBM相对MTR延迟降低1%-4%。

  **缺陷2：DeFT要求每个VN至少2个VC用于upward/downward traffic隔离，增加路由器资源和面积（约+48%）**
  → DFBM不要求片内NoC做额外VC隔离，而是在bridge module中通过CM管理credit、CVN-DB管理共享buffer。DeFT需要per-VN的VC dedicating，DFBM用CVN-DB共享deadlock buffer将面积开销从~5% (dedicated per-VN buffer) 降到~2.5% (shared buffer)，且开销位于成本更低的interposer侧。

  **缺陷3：RC的permission network需要片内NoC加专用控制网络，增加验证复杂度并产生vendor lock-in**
  → DFBM继承RC的injection control思路但把控制逻辑外置到bridge module，不要求供应商修改片内NoC。CM的two-stage admission arbitration (Stage1 coherence extraction + Stage2 credit/CVN-DB check) 替代RC的片内permission network。同时DFBM不是持续节流，而是根据expected credits、CVN-DB occupancy和congestion状态动态调节，低负载下延迟更接近原生路径。

  **缺陷4：Recovery类方法 (UPP/Steered Bubble) 允许deadlock先发生再恢复，需检测逻辑和escape channel，引入架构修改和验证负担**
  → DFBM是avoidance机制，不依赖死锁检测后再打断环。通过在最坏情况下预留足够credit保证chiplet-to-interposer出口方向的吸收能力，从根源上防止跨chiplet CDG形成环。无需deadlock检测逻辑、escape channel或directional bubble routing。

  **缺陷5：所有baseline都将deadlock处理与chiplet内部NoC实现绑定，破坏开放生态的模块化目标**
  → DFBM作为独立bridge module实现了"plug-and-play"目标：不同供应商的chiplet无需了解彼此内部NoC实现细节即可直接互连。DFBM只需要供应商提供少量协议参数（coherence state machine依赖、cache controller最大outstanding request数、NoC VC数），而不需要修改片内NoC路由算法、VC分配或拓扑。

  论文方法全栈执行例子（以PARSEC workload + DFBM + 4-chiplet mesh + gem5/Garnet为例）：
  - 算法层：论文未明确说明（PARSEC application，非ML场景）。
  - 系统框架/Serving层：论文未明确说明。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明。
  - 硬件架构层：DFBM two-stage admission pipeline执行流：外部chiplet发来In-Req → DFBM CM Stage1提取coherence type → 查询Expected Credit Table得到最大Out-Rsp/Out-Fwd-Req数K → CM Stage2检查current reserved credit + CVN-DB occupancy → 若credit充足则admit In-Req进入chiplet（消耗reserved credit），若不足则block → chiplet内部cache controller处理请求产生Out-Rsp/Out-Fwd-Req → 响应包返回DFBM → 若interposer方向空闲则直接发出，若拥塞则进入CVN-DB → CVN-DB按Response > Forward-Request > Request优先级drain → 包成功发出后CM更新credit。DFBM在请求进入chiplet前就为其后续响应预留边界吸收能力，保证chiplet-to-interposer vertical channel不成为不可释放的依赖点。
  - 芯片设计层：DFBM位于chiplet NoC与interposer NoC边界（interposer侧），连接两侧已有VC接口不引入专用控制线。CM和CVN-DB均在interposer side实现，面积开销（~2.5% with CVN-DB）由成本更低的interposer承担。对非确定性coherence transition，dummy packet只在chiplet-DFBM接口内局部传输，对全局延迟和带宽影响有限。

## LRM-GPU: Alleviating Synchronization Overhead for Multi-Chiplet GPU Architecture

- baseline方法是什么？
  Baseline是MCM-GPU[2]的同步机制，沿用传统monolithic GPU的软件驱动coherence protocol。同步变量不被本地cache维护一致副本，acquire操作invalidate L1和L1.5 cache，release操作确保dirty data写回LLC，atomic同步操作绕过本地cache直接到LLC执行。MCM-GPU同时采用L1.5 cache（仅缓存remote data）、first-touch page allocation和distributed CTA scheduling来优化普通数据访问的NUMA问题，但这些优化不解决同步瓶颈。

  全栈执行例子（以lock-based synchronization + MCM-GPU + 4-chiplet GPU为例）：
  - 算法层：CUDA kernel使用atomicCAS实现spin lock，多个SM竞争同一lock，acquire/release包围临界区访问共享数据A。
  - 系统框架/Serving层：论文未明确说明（GPGPU-Sim直接运行CUDA kernel，无Serving框架层）。
  - 编译框架层：论文未明确说明（CUDA 11.1, O3编译，无定制编译pass）。
  - kernel调度层：SM0 (chiplet0) 发出atomicCAS(lock)→绕过L1/L1.5直接路由到LLC执行→成功acquire→SM0发出acquire同步→invalidate chiplet0 L1.5（保守地清掉可能stale的数据）→执行load/store修改共享数据A→release同步→write-through L1.5保证dirty data已写回→SM1 (chiplet0) acquire→再次invalidate L1.5（即使连续同步在同一个chiplet内，L1.5仍被重复清空）→SM2 (chiplet1) 的atomicCAS(lock)→跨chiplet发送到LLC→受inter-chiplet 768GB/s有限带宽和网络排队影响。每次acquire/release都触发L1.5 invalidate/flush（即使大部分同步变量owner未变），多SM对同一lock的atomicCAS spin产生大量跨chiplet重试流量。
  - 硬件架构层：MCM-GPU的coherence行为——L1/L1.5 write-through，LLC write-back。Acquire: invalidate L1+L1.5。Release: write-through保证LLC不持有stale data，无需额外flush。Atomic同步: bypass L1/L1.5直接到LLC执行。该方案在4-chiplet系统上与等价monolithic GPU对比平均性能下降50.5%（其中额外L1.5 invalidation导致22.5%损失，remote atomic access导致23.5%损失）。
  - 芯片设计层：4-chiplet GPU，每chiplet 64 SMs + 2MB L1.5 + LLC slice + DRAM partition，inter-chiplet concentrated hierarchical crossbar 768GB/s，DRAM 64 channels/3TB/s。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出LRM-GPU，核心洞察是利用同步行为的locality：(1) 同步变量ownership locality——当同步操作在一个chiplet内发生时，后续同步操作很可能也在同一chiplet；(2) 同步atomic数据locality——跨chiplet的atomic操作可能同时访问同一地址，可在网络内合并。基于此，LRM-GPU实现两个机制：

  **缺陷1：MCM-GPU每次acquire/release都保守地invalidate/flush L1.5，即使连续同步发生在同一chiplet也重复付费**
  → LRM-GPU引入Lazy Release Consistency：在LLC实现sync-val directory（64 entries, 0.4KB），为每个同步变量记录owner chiplet。Acquire时：
  - 若directory无记录→分配entry+记录owner+invalidate本地L1.5
  - 若owner=本地chiplet→直接从LLC读同步变量，不做coherence action（延迟到owner真正变化）
  - 若owner=远端chiplet→flush远端L1.5+更新owner+invalidate本地L1.5
  - 若directory满→LRU evict+flush被驱逐owner的L1.5
  Release时类似：local owner仅写同步变量到LLC，延迟coherence action。这利用chiplet内连续同步的时间locality减少L1.5 invalidate/flush。相比MCM-GPU减少30% L1.5 cache invalidation。

  **缺陷2：跨chiplet atomic同步操作受inter-chiplet有限带宽严重制约，大量同地址atomicCAS spin或atomicAdd update产生冗余跨chiplet流量**
  → LRM-GPU在每个chiplet网络中嵌入AMU (Synchronization Atomic Merge Unit)，检测并合并同地址cross-chiplet atomic请求。多个atomicAdd(addr,1)合并为atomicAdd(addr,n)；多个对同一lock的atomicCAS在比较值相同时只发送一个可能成功的请求。AMU的merge table (2K entries, 16 banks, CAM+SRAM dual-port) 通过countdown timer控制合并窗口，到期或SM list满后发送合并请求，响应通过multicast unit广播给所有参与SM。相对MCM-GPU减少28% inter-chiplet traffic（AMU单独贡献12%）。相对HMG减少52% inter-chiplet traffic。AMU总功耗301.44mW/面积1.84mm² (40nm)，仅占系统0.13%能耗。

  **缺陷3：相比hLRC（多级cache追踪同步变量+write-back），同步变量跨SM迁移时需写回远端cache，高竞争下增加等待时间和重试流量**
  → LRM-GPU不同：同步变量不被缓存在L1/L1.5中（与传统GPU一致直接bypass到LLC），避免跨SM迁移时的多级cache write-back等待；仅在有跨chiplet ownership变化时才对L1.5做coherence action，不跟踪每个SM的注册状态。hLRC虽减少56% L1.5 invalidation，但因同步变量write-back等待和重试导致性能反而不如MCM-GPU。

  **缺陷4：相比HMG（完整cache coherence protocol + hierarchical sharer tracking），所有写入数据需向sharer发送invalidation request，在atomic密集workload上产生大量write-invalidation traffic**
  → LRM-GPU只跟踪同步变量的owner chiplet（而非所有数据的sharer），directory仅64 entries/0.4KB（HMG 12K entries/chiplet），状态空间小得多。在MST和pagerank等atomic密集benchmark上，HMG因大量write-invalidation performance degradation，而LRM-GPU无此问题。相对HMG平均加速1.22×，减少52% inter-chiplet traffic，减少32% energy。

  论文方法全栈执行例子（以lock-based synchronization + LRM-GPU + 4-chiplet GPU为例）：
  - 算法层：CUDA kernel使用atomicCAS实现spin lock，SM0→SM1 (chiplet0) →SM2 (chiplet1) 顺序执行。与baseline相同的应用代码，差异全在硬件同步路径。
  - 系统框架/Serving层：论文未明确说明。
  - 编译框架层：论文未明确说明（CUDA 11.1, O3编译，无定制编译pass）。
  - kernel调度/硬件层：SM0 acquire lock→directory无X→分配entry+owner=chiplet0+invalidate L1.5→SM0修改A=1后release→仅写X到LLC不flush L1.5（owner未变）→SM1在同chiplet0 acquire→directory命中owner=chiplet0→不invalidate L1.5直接读X+访问最新A→SM1修改A=2后release→仅写X→SM2 (chiplet1) acquire→directory发现owner=chiplet0→flush chiplet0 L1.5（write back A=2到LLC）→更新owner=chiplet1→invalidate chiplet1 L1.5→SM2读最新A=2→SM2 release。跨chiplet atomic路径：SM0/SM1同时发atomicAdd(addr,1)→AMU merge table合并→单一atomicAdd(addr,2)跨chiplet发送→响应广播。关键差异：MCM-GPU在SM0和SM1的acquire时各invalidate一次L1.5（共2次），LRM-GPU仅第一次invalidate 1次；MCM-GPU的atomic请求每次单独跨chiplet，LRM-GPU先merge再发送。
  - 硬件架构层：sync-val directory在LLC中——64 entries, 2-bit owner/chiplet, 48-bit tag address, 1-bit valid→每entry 51 bits→总0.4KB。AMU per chiplet——16 channels, 2K merge entries/16 banks, CAM key查找+SRAM data存储 dual-port, countdown timer控制合并窗口。
  - 芯片设计层：multi-chiplet inter-chiplet network中每个chiplet嵌入AMU，merge table以TSMC 40nm定制电路实现。Chiplet数从4增至8时加速比从1.33×降至1.21×（locality下降）。Inter-chiplet latency 8-48 cycles范围内性能几乎不变（GPU warp switching隐藏延迟，性能对bandwidth敏感而非latency）。

## Swift: High-Performance Sparse-Dense Matrix Multiplication on GPUs

- baseline方法是什么？
  Baseline是已有GPU SpMM方法：Sputnik（ROMA格式+vector memory instruction处理未对齐稀疏访问）、ASpT（adaptive tiling处理矩阵不规则性）、RoDe（CSR行拆分为regular block和residual part优化pipeline）、cuSPARSE v12.2（通用稳定实现）。这些baseline在不同稀疏分布下可以提升格式效率或负载均衡，但均未同时解决warp内sparse矩阵A和dense矩阵B的coalesced memory access问题。

  全栈执行例子（以cuSPARSE CSR-based SpMM + N=128 + A100为例）：
  - 算法层：SpMM C=A×B，A为M×K稀疏矩阵(CSR格式)，B为K×N稠密矩阵(row-major)，C为M×N稠密输出。CSR格式下每个warp处理一行或多行，通过rowPtr定位value/colIdx范围。
  - 系统框架/Serving层：论文未明确说明（直接CUDA kernel调用，无上层框架）。
  - 编译框架层：论文未明确说明（CUDA 12.2 NVCC默认编译路径）。
  - kernel调度层：cuSPARSE CSR SpMM kernel——warp内线程按CSR行分配，每个线程迭代rowPtr[start]到rowPtr[end]范围的非零元。读取A的value和colIdx后，colIdx作为索引访问B的行（B为row-major时colIdx×N+jump导致不连续地址），warp内不同线程的colIdx分布随机→B的访问地址跳跃→接近warpSize次memory transaction→数据加载开销平均超过整体性能32%。
  - 硬件架构层：NVIDIA A100 GPU，128 SM，40/80GB HBM2e，1555 GB/s bandwidth。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Swift，通过"sparsity-based sorting + dense row rearrangement + warp-size blocking + dual-kernel path (regular coalesced + irregular load-balanced) + segment-sum atomic reduction"系统性地解决baseline的memory coalescing缺失问题。

  **缺陷1：CSR/常规格式下warp内线程的colIdx分布随机→访问dense B时地址不连续→memory coalescing差→数据加载占>32%时间**
  → Swift第一步sparsity-based sorting：按稀疏矩阵A每列NNZ升序排序列，并同步重排B的行。排序后相邻列NNZ相近、colIdx连续性增强，配合column-major B布局使warp内线程访问B地址更连续。

  **缺陷2：稀疏矩阵非零元分布高度不规则→单一路径处理浪费coalescing机会或负载不均**
  → Swift第二步blocking：按warpSize=32将排序后A划分为regular block（列宽恰好=32的完整block）和irregular block（不足32列或长列残留）。为两类block生成独立索引结构（regular: blkPtr/blkColIdx/value/rowIdx/positionIdx/offsetIdx; irregular: irrPtr/irrValue/irrRowIdx/colIdxIndex/blkStart/blkStop）。

  **缺陷3：regular block的coalesced访问仍需要高效的partial sum reduction→直接atomicAdd写回C产生大量atomic冲突**
  → Regular kernel使用segment sum优化：warp内各lane将乘积写入shared memory→用positionIdx/offsetIdx前缀和索引对rowIdx相同的partial sum做local segment sum→仅发生少量atomicAdd写回。这在不破坏coalesced加载的前提下大幅降低了global atomic竞争。

  **缺陷4：长列（高NNZ列）直接分配整个warp导致warp间负载不均**
  → Irregular kernel将长列按sub-column/block拆分：warpId先通过colIdxIndex判断任务类型（独立短列或长列子块）→blkStart/blkStop定位范围→lane以步长32遍历→对每个非零元循环访问B并atomicAdd写回C。长列被拆解为多个均匀子块由不同warp分担，消除单warp阻塞其他warp的问题。

  论文方法全栈执行例子（以Swift CSC-based SpMM + N=128 + A100为例）：
  - 算法层：SpMM C=A×B，A为M×K稀疏矩阵(CSC格式，经Swift预处理)，B为K×N稠密矩阵(column-major+行重排后)，C为M×N稠密输出。
  - 系统框架/Serving层：论文未明确说明（直接CUDA kernel launch，无上层框架）。
  - 编译框架层：论文未明确说明（CUDA 12.2 NVCC编译，生成test可执行文件）。
  - kernel调度层：预处理——CPU端CSC A→统计列NNZ→升序排序A的列并重排B行→warpSize=32划分regular/irregular→生成双套索引。Regular kernel——thread block=32×8线程，每32-lane warp处理一个sparse block(32列)+B中连续32列，lane读sparse value/rowIdx→colIdx=blkColIdx+laneID→B[colIdx:k:k+N]连续加载（coalesced column-major access）→乘积入shared memory→segment sum归并同rowIdx的partial→atomicAdd写回。Irregular kernel——warp通过colIdxIndex查任务→blkStart/blkStop取子块→lane stride=32遍历→每非零元访问B→atomicAdd写回C。
  - 硬件架构层：NVIDIA A100 GPU，利用shared memory(每thread block 48KB可用)做segment sum buffer，warp scheduler通过dual-path减少stall。关键trade-off：预处理（sorting+blocking+索引生成）有额外开销，NNZ>10^6时Swift预处理成本大于Sputnik/RoDe、接近ASpT；但当非零元分布较均匀时memory coalescing收益远大于预处理开销。

## Uni-STC: Unified Sparse Tensor Core

- baseline方法是什么？
  Baseline包括DS-STC（面向SpGEMM的outer-product路径）、RM-STC（面向SpGEMM的row-row路径）和NV-DTC（NVIDIA dense tensor core）。DS-STC以A的半列和B的半行为基本单位组织outer-product任务，RM-STC以scalar-vector组合组织row-row任务。两者都采用"gather data"模式：将sparse data gather成固定形状的T2/T3子任务后送入MAC array，遇到长行、长列、对角集中或双边稀疏矩阵时，固定形状任务导致大量低MAC utilisation周期（DS-STC和RM-STC分别有61.68%和62.78%的周期低于50% utilisation），且大中间乘积网络持续搬运带来能耗。

  全栈执行例子（以RM-STC执行SpMV为例）：
  - 算法层：输入CSR格式稀疏矩阵A和dense vector x，RM-STC用row-row scalar-vector组合方式沿N维拼接任务到固定MAC array
  - 编译框架层：论文未明确说明（无专用编译框架，硬件直接解析CSR-derived任务）
  - kernel调度层：warp读取CSR row一行的非零值→硬件decoder将非零位置映射到固定shape T2/T3任务→送入MAC array执行partial product→结果通过shfl_gather聚合
  - 硬件架构层：RM-STC硬件含hardware decoder解析CSR格式生成固定shape任务、固定width network搬运中间乘积、accumulator做partial sum。缺陷：遇到长行时一行被拆成多个独立任务而无法跨K维拼接；遇到窄行时MAC利用率低；decoder面积开销大

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Uni-STC的核心设计哲学从"gather data"转为"gather tasks"。方法包括四层协同设计：(1) BBC格式用CSR组织4x4 tile+两级bitmap描述tile内非零，避免CSR hardware decoder和跨kernel格式转换；(2) TMS/DPG将固定形状的T1→T3→T4 task做三级灵活分解，T3在M/N/K三维对称便于统一多kernel支持，T4的1x1x4 dot-product片段可以跨多低负载任务灵活拼接；(3) SDPU的segmented dot-product+merge-forward在写C前预合并最多4个partial products，减少中间乘积网络流量；(4) TMS的task ordering在outer-product与row-major间动态选择提升A/B tile复用，dynamic DPG activation按需power-gate多余DPG。

  全栈执行例子（以Uni-STC执行SpGEMM为例）：
  - 算法层：输入BBC格式矩阵A和B（一次offline转换，<100ms），warp通过stc.load收集tile metadata/values到Uni-STC的Matrix A/B Buffer
  - 编译框架层：论文未明确说明（UWMMA通过汇编级stc.load/stc.task/stc.numeric指令序列编程，无高级编译器）
  - kernel调度层：stc.task触发TMS→从Meta Buffer读top-level bitmap→按K维动态拼接T3 task（避免RM-STC沿K维无法拼接的缺陷）→8个DPG并行将T3转为T4 sparse dot-product task→以Z-shaped顺序入Dot-product queue→stc.numeric检查READY驱动SDPU执行→相邻低负载T4拼接成segmented dot-product
  - 硬件架构层：两层精简network仅搬运控制信息（T4 code）而非全量中间乘积→SDPU内局部执行短向量点积→merge-forward预合并partial products后写C→dynamic power-gating按需关闭闲置DPG和network。量化效果：Uni-STC相对DS-STC和RM-STC几何平均speedup 3.35x和2.21x，energy reduction 1.97x和1.27x，SuiteSparse全量矩阵上的MAC低利用率周期比例显著降低

## QuCo: Efficient and Flexible Hardware-Driven Automatic Configuration of Tile Transfers in GPUs

- baseline方法是什么？
  Baseline是当前GPU ATT编程的最佳实践：程序员手动进行wavefront specialization（将workgroup内wavefront分为dedicated producer wavefront执行ATT memory transfer + consumer wavefronts执行computation），手动选择tile sizes、queue slots、LDS partitioning和synchronization barriers。NVIDIA提供cuda::pipeline API包装TMA的producer-consumer wavefronts为reusable queues，CUTLASS3+CuTe和ThunderKittens提供高级ATT抽象（TMA pipelines和asynchronous I/O），但获得hardware-specific peak ATT performance仍需程序员deep understanding of underlying GPU microarchitecture和手动tuning——不同kernel间最优ATT配置差异可达1.2×，不同GPU间可达1.4×。

  全栈执行例子（以手动ATT/Fine-Tuned执行Matrix-Matrix kernel on MGPUSim R9 Nano为例）：
  - 算法层：矩阵乘法kernel [512,2048]×[2048,128]，CI>4（compute-bound），程序员需理解kernel compute-to-memory ratio和data reuse pattern决定wavefront specialization策略
  - 系统框架/Serving层：论文未明确说明（无serving framework，直接launch kernel）
  - 编译框架层：论文未明确说明（kernel binary通过AMD ROCm编译，但无QuCo时编译框架不参与ATT配置）
  - kernel调度层：程序员手动编写producer wavefront（通过ATT descriptors指定global memory addresses、tile dimensions、memory strides、LDS destination）→调用Push/Wait_For_Push→consumer wavefronts通过Peek/Pop配合asynchronous transaction barriers（arrive+wait两阶段）消费LDS中tile→程序员需遍历exhaustive DSE（对Matrix-Matrix为2.6×10^14种组合）找最优配置，每次尝试需一次kernel launch profiling
  - 硬件架构层：GPU Compute Unit内含ATT engine（bypass L1 cache, 直接issue read memory requests到global memory, 自行生成address/transfer count, 直接写LDS）→Sync Unit管理asynchronous barriers→ATT descriptors指针从LDS读取。缺陷：参数选择kernel-specific且architecture-specific（R9 Nano的最优配置在MI-100上差1.4×），手动tuning laborious且impractical for large design spaces（Whisper Tiny 2.1×10^17 kernel launches）

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出QuCo（Queue Configurator），一个嵌入GPU die的轻量级硬件单元，在kernel launch时通过RISC-V firmware自动计算ATT队列最优配置，无需程序员手动tuning。方法对应baseline缺陷如下：

  **缺陷1：手动tile size选择需要遍历exponentially large design space（64-8192 elements × 1-8 slots × 多queue组合，从25到2.6×10^14种组合）→programmer effort impractical**
  → QuCo Algorithm 1+2自动计算：遍历tile size candidate→对每个tile计算merit factor（processing time / memory transfer time）× cost function→processing time含scheduling roundtrip overhead（wavefront dispatch limitations建模）→memory time含ATT/DRAM/L2 latency + bandwidth inverse penalty + cache-line inverse penalty→选weighted merit最优tile→按CI调整（低CI放大tile提升memory throughput，高CI缩小tile平衡overlap）。无需profiling，单次pass完成。

  **缺陷2：手动slot选择缺乏系统化方法，过多slots导致memory contention，过少导致underutilization和idle cycles**
  → QuCo Algorithm 3：streaming queues用hardware-aware Little's Law（slots = memory transfer time / tile compute time）推导ideal queue depth → CU-aware rounding（更多CU→减少slots降低memory system pressure） → LDS capacity check with CI-based fallback（低CI多slots提升throughput，高CI少slots）。stationary queues均分remaining LDS capacity后同样round。通过GST获取精确hardware参数（bandwidth/latency/LDS size/CU count）确保architecture-specific optimal。

  **缺陷3：post-compilation binary缺乏跨GPU portability——同一ATT配置在不同GPU上性能差异可达1.4×**
  → QuCo的GST（256-byte vendor-written table）存储每个GPU型号的architectural parameters（memory latencies/clock/LDS size/CU count/arithmetic throughput）。同一compiled binary在不同GPU上执行时，QuCo firmware读取该GPU的GST→自适应计算tile/slots→preserve same binary across GPU family。evaluation证明QuCo on MI-100/R9 Nano/Radeon 530均near-optimal。

  **缺陷4：DVFS和多租户环境下静态tuned配置失效——frequency变化或partitioned resources打破assumptions**
  → QuCo-HW在每个kernel launch时重新读取当前operating frequency和resource availability→动态调整queue配置适应runtime变化。DVFS实验：QuCo-HW在3种频率变化scenario下vs QuCo-SW（假定固定频率）获up to 17% improvement。Multi-tenant场景下QuCo根据实际available resources动态调整。

  **缺陷5：per-layer/model-level手动tuning不scale——DNN模型如Whisper Tiny有827层，每层特性不同，Semi-Tuned复用早期层配置导致后续层性能下降**
  → QuCo per-kernel invocation自动reconfigure queue参数（tile size从256到1024、slots从2到4不等，因层CI和dimensions而异）。Whisper Tiny layer-wise ablation证明QuCo在所有unique层上均outperform ATT/Semi-Tuned和ATT/Fine-Tuned。

  全栈执行例子（以QuCo执行Matrix-Matrix kernel on MGPUSim R9 Nano为例）：
  - 算法层：与baseline相同kernel binary（无代码改动），host code仅需：driver.InitQuCo(HIGH, 512, 64) → driver.RegisterQueue(K, 4, TYPE_STREAMING)注册8个streaming queues + 1个stationary queue
  - 编译框架层：论文未明确说明（kernel编译路径不变，QuCo在post-compilation runtime介入）
  - kernel调度层：kernel launch时QuCo RISC-V firmware从GST读取R9 Nano参数（64 CUs, 1.0GHz, LDS size, DRAM bandwidth, latencies 190/300/450）→计算CI>4（compute-bound）→Algorithm 1选tile 512（对高CI缩小tile平衡overlap）→Algorithm 3计算streaming queue slots = 2（Little's Law + CU-aware rounding从4降为2降低memory contention）→stationary queue slots = 2（remaining LDS均分后round）→写入ATT descriptors到LDS→ATT unit加载descriptors开始异步tile load→consumer wavefronts消费LDS中tile不感知配置过程。结果：QuCo performance在ATT/Fine-Tuned（exhaustive DSE）的1.04%以内，但仅需单次pass无任何tuning努力。
  - 硬件架构层：QuCo hardware（RISC-V core 0.027mm² + memory 0.014mm² @28nm）与main compute pipeline解耦，不干扰wavefront scheduling和memory requests→配置完成后idle，多kernel workload中reconfiguration overhead ~6300-8300 cycles远小于kernel execution time

## μShare: Non-Intrusive Kernel Co-Locating on NVIDIA GPUs

- baseline方法是什么？
  Baseline是NVIDIA GPU的默认硬件调度器行为：stacked co-location。GPU的dispatch unit在kernel launch后将同一kernel的所有block按顺序分配到SM cores，当kernel线程数超过GPU总thread capacity时，block占满所有SM。由于同kernel所有block的hardware resource需求相同，产生"1 more, 5 less"资源利用模式——1种主要hardware resource高利用率（avg 30.19%），其余5种极低（avg 5.07%）。NVIDIA-SMI报告81.16% utilization但Nsight Compute仅报告9.28% low-level hardware utilization。

  全栈执行例子（以PyTorch + NVIDIA A40 GPU + 默认FCFS kernel调度为例）：
  - 算法层：推理模型（如BERT/ResNet50）由多个CUDA kernel组成（matrix multiplication、layer normalization、convolution等），每个kernel的blocksize由PyTorch/TVM/TensorRT在dedicated GPU场景下静态preset（枚举+选最优，如roll kernel preset 512）
  - 系统框架/Serving层：PyTorch sequential launch kernel到CUDA stream → GPU Command Processor串行接收 → dispatch unit按block粒度调度到SM。跨kernel的并发通过multi-stream/persistent kernel尝试有限。INFless/Orion等Serving系统在inter-SM层面做spatial/temporal sharing，但intra-SM仍为stacked co-location
  - 编译框架层：论文未明确说明（PyTorch 2.2.0 + CUDA 11.8默认编译路径，blocksize preset在模型导出时固化）
  - kernel调度层：硬件dispatch unit left-over scheduling——SM满足线程数要求即可入驻block。同一kernel blocks因需求同构→全部stack进同一批SM→同构资源需求→single-dominant resource pattern。例如：vectorized kernel (layer norm) block在SM内LDST利用率58.02%，其余5种HW avg < 15%。Roll kernel block INT32利用率33.25%。两者stacked co-locate时各自独占SM→无法互补
  - 硬件架构层：NVIDIA A40 GPU（84 SMs, 每SM 1536 threads, 32 FP64 cores/64 FP32/64 INT32/4 Tensor cores/16 SFU/32 LDST units）。硬件scheduler闭源，不暴露intra-SM resource allocation接口

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出μShare：通过非侵入式修改kernel launch参数（blocksize）间接操纵闭源GPU硬件scheduler，实现intra-SM scattered kernel co-location。

  **缺陷1：Baseline的stacked co-location导致intra-SM硬件资源"1 more, 5 less"——同kernel blocks因需求同构独占SM，剩余5种hardware units闲置**
  → μShare提出half-plus blocksize shaping：将kernel blocksize设为略超半SM thread容量（A40: 768+32=800），使同一SM无法容纳两个该kernel block（800×2=1600 > 1536）→同kernel blocks被迫散布到不同SM→每SM剩余threads（1536-800=736）可容纳另一kernel的小block（512-640 threads）→实现scattered co-location。实验：当dominant resource不同的kernel配对时half-plus提升throughput 19.94%；若dominant resource相同则下降10.37%

  **缺陷2：Static preset blocksize在co-location场景下non-optimal——PyTorch/TVM/TensorRT在dedicated GPU下枚举选最优blocksize（如roll kernel=512），但SM内可用资源动态变化时该blocksize不再optimal**
  → μShare发现roll kernel与vectorized kernel（blocksize=256）co-locate时最优blocksize从512 shift到1024（1.98× throughput improvement）。μShare的block shaper根据实时kernel launch slack动态调整blocksize：slack positive→half+32（最小warp粒度）、slack negative→逐步+32加速执行

  **缺陷3：48.37%的kernel blocksize不可修改（cuBLAS/cuDNN闭源wrapper + tiling kernel如Conv2d修改后产生CUDA internal error）→仅partial co-location**
  → μShare提出time-shifted launching：对unmodifiable kernel保持default blocksize，检查6种low-level hardware resource combined utilization + shared memory/register availability→满足互补条件则立即launch→不满足delay β=10μs后重检→更新slack重新排序kernel set→若延迟导致其进入top-x slack最小的kernel→升级为half-plus shaping。实验：unmodifiable kernels从100%→48.37%时throughput从47.59→58.81单调提升；worst-case (100% unmodifiable) 回退到resource-coupled co-location（等效INFless）

  **缺陷4：Existing intra-SM sharing方法需要侵入性修改——kernel fusion (Tacker/T3/Rammer/COMBO)需重写合并kernel代码、persistent kernel (ISPA/Plasticine/Elastic kernel)需空non-terminating kernel驻留、hardware modification (CCWS/Prema/PriorityRR)需重新设计GPU scheduler→在public cloud不feasible**
  → μShare完全非侵入：仅在Linux userspace通过LD_PRELOAD劫持kernel launch函数→无需修改kernel代码、无需修改GPU硬件scheduler、无需额外CPU-GPU通信。Kernel interception overhead仅60.35ns/kernel，CPU overhead 6.85% of single core。PyTorch仅需设置C10_LAUNCH_BOUNDS与CUDA limit一致

  **缺陷5：A800/A100/H200等2048 threads/SM的GPU不支持half-plus（即使max blocksize=1024，两个1024 block仍可在1SM内stacked co-locate）**
  → μShare改用1/3-plus shaping：blocksize=2048/3+α≈704→同kernel两个1/3-plus block (704×2=1408) 可入1SM→剩余threads (2048-1408=640) < 704→不能放第三个large block→只能放small block (512-640) from complementary kernel→实现scattered co-location。尽管从half(1/2)提升到2/3上限可能导致轻微thread分配不均衡，但在A800上仍获得16.45%-52.29% throughput improvement

  **缺陷6：Baseline使用NVIDIA-SMI的"active time ratio"夸大了GPU utilization（81.16% vs Nsight Compute 9.28%）和SLO管理粗糙（仅inter-request层面调整batch size）**
  → μShare引入dual-level SLO guarantee：(1) 请求侧：exponential decay-based SLO slack feedback control → 基于monitored latency保守增加（linear, bj+1 = bj + k×s→j）、激进减少（exponential, bj+1 = max{bj - e^(λ×s→j), 1}）batch size；(2) kernel侧：基于kernel launch slack调整blocksize加速kernel执行。v7配置(k=0.05, λ=-0.2) SLO violation rate 0.84%低于INFless/Orion的2.05%/1.12%，同时throughput仍提升19.28%-44.83%

  全栈执行例子（以PyTorch + μShare + NVIDIA A40 + roll kernel与vectorized kernel scattered co-location为例）：
  - 算法层：与原模型相同（无kernel代码修改）。模型kernel分类：roll kernel（INT32 dominant, blocksize=512 default, modifiable 51.63%）+ vectorized kernel（LDST dominant, blocksize modified to half-plus 800）+ cuBLAS gemm（Tensor core dominant, blocksize unmodifiable 48.37%）等
  - 系统框架/Serving层：Kernel Interceptor通过LD_PRELOAD+dlsym拦截cudaLaunchKernel → 读取blocksize参数 → 计算kernel launch slack（s_k = tLaunch - tIntercept）。Batch Manager根据实时SLO slack (SLO=200ms) 采用conservative-increase/aggressive-decrease调整batch size。离线Profiler对每个kernel记录9-tuple resource profile
  - 编译框架层：论文未明确说明（PyTorch 2.2.0 + CUDA 11.8默认编译，model权重无修改）
  - kernel调度层：Block Shaper将vectorized kernel blocksize设为800 (half+32) → 1SM内800 threads → 2个vectorized block需1600 threads > 1536上限 → 只放1个 → 剩余736 threads放1个roll block (512) → LDST (vectorized) + INT32 (roll) 互补执行。同时cuBLAS gemm (Tensor core) 根据time-shifted launching在vectorized kernel接近完成时启动 → Tensor core + LDST/INT32 互补。结果：intra-SM 6种hardware avg utilization从10.90% (INFless) 提升到15.10% (μShare)
  - 硬件架构层：NVIDIA A40 GPU（84 SMs, 1536 threads/SM），硬件scheduler left-over策略保持不变。μShare仅通过控制stage 1 (launching) 的blocksize参数间接影响stage 2 (scheduling) 和stage 3 (execution) 的behavior。修改在CPU userspace完成（dlopen/dlsym/mmap），GPU硬件执行路径不变

## FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection

- baseline方法是什么？
  Baseline是现代GPU上compute-intensive operator chain的传统kernel fusion方法。以GPT-6.7B FFN（两个连续GEMM：M×K×N→M×N×L）为例的全栈执行：
  - **算法层**：标准Transformer FFN，两个GEMM间有一个激活函数（ReLU/SiLU），权重为预训练固定值。M=128(batch token), N=16384(intermediate), K=4096(hidden→intermediate), L=4096(intermediate→hidden)
  - **系统框架/Serving层**：PyTorch 2.6 + torch.compile（减少kernel launch overhead），或SGLang。cuBLAS将每个GEMM作为独立kernel launch，BOLT/Chimera尝试融合，TensorRT做graph optimization
  - **编译框架层**：cuBLAS/CUTLASS将两个GEMM编译为两个独立kernel（或BOLT使用CUTLASS模板融合、Chimera做block order探索）。中间结果tensor C [M×N=128×16384=~2M floats≈8MB] 必须存到某处。单SM SMEM上限约227KB，远小于8MB C matrix
  - **kernel调度层**：cuBLAS中第一个GEMM kernel将完整结果C写回HBM global memory；第二个GEMM kernel再从HBM读取C作为输入。形成"write-then-read" round-trip。BOLT尝试在SMEM内fusion但受限于模板固定block执行顺序；Chimera可在reg/SMEM中保留部分C tile但大模型下SMEM capacity不够→fusion failure，回退到global memory round-trip
  - **硬件架构层**：NVIDIA H100 GPU，HBM带宽3TB/s，FP16算力~1000 TFLOPS。H100具备DSM（SM-to-SM NoC, cluster内最多16 SMs），但baseline软件栈不使用DSM——中间数据只能走global memory路径

  **Baseline缺陷**：(1) 单SM SMEM约227KB的严格容量限制，当中间tensor超过阈值时fusion直接失败(图5)；(2) 即使partial fusion可能，fixed block order（BOLT）和忽略DSM导致中间结果必须round-trip HBM；(3) 全局memory访问约为FlashFuser的2.4倍(图11)，FFN占LLM推理40%–60%时间且memory-bound

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出FlashFuser编译框架，通过DSM扩展on-chip memory pool，实现大中间tensor的cross-SM fusion。以同一GPT-6.7B FFN为例：
  - **算法层**：与baseline相同（标准FFN/Gated FFN/conv chain的GEMM运算），无算法修改
  - **系统框架/Serving层**：FlashFuser作为offline compiler预编译多版本fused kernel→运行时通过binning/table lookup按动态变化的M选kernel。与SGLang集成实现端到端加速（平均1.32× speedup for small models, 1.24× overall）
  - **编译框架层**：核心创新——三个层次化设计。
    (a) dsm_comm primitives：定义DSM-aware通信原语（dsm_all_exchange→cluster内All-Reduce聚合partial C tile；dsm_shuffle→ring communication交换C切片；dsm_reduce_scatter→scatter-reduce聚合partial E；inter_cluster_reduce→TMA cp.reduce.async.bulk跨cluster）。编码cluster size四维参数(clsm,clsn,clsk,clsl)和派生变量clsshuffle/clsreduce
    (b) Dataflow Analyzer：将loop schedule/tile/resource mapping扩展到reg→SMEM→DSM→global四层。贪心策略优先放高层缓存，容量不足逐层spill——关键：当C matrix tile超出SMEM 227KB时，不再直接fusion fail，而是spill到DSM（cluster内多SM SMEM聚合可达3.6MB=16×227KB）。分析各层数据搬移量供cost model评估
    (c) Fusion Search Engine：用C_l=V_l/B_l minimax cost model + 5条DSM-aware pruning rules在2.75×10^13→1.15×10^6搜索空间中找到最优plan。Top-K=11候选实测选最佳
  - **kernel调度层**：执行分三阶段——
    GEMM0: K维spatial partition(clsk=2)两平行block计算C partial→dsm_all_exchange沿K维All-Reduce得完整C tile→C tile留在DSM不写global memory（避免Chimera的fusion failure和cuBLAS的HBM round-trip）
    GEMM1: dsm_shuffle在Shuffle Group内ring communication交换C切片→各block与D tile乘得E partial
    Store: dsm_reduce_scatter cluster内scatter-reduce + inter_cluster_reduce TMA原子聚合→最终E写global memory
    关键trade-off：DSM bandwidth随cluster size变化（越大cluster带宽越低），需cost model平衡；spatial partition最大化并行vs sequential执行最小化DSM通信
  - **硬件架构层**：NVIDIA H100 GPU。FlashFuser利用DSM（SM-to-SM NoC on-chip path）替代global memory round-trip——DSM latency始终低于global memory，DSM bandwidth（除最大cluster size 16外）高于HBM bandwidth。dsm_comm通过TMA数据移动+mbarrier many-to-many synchronization实现，替代默认all-to-one cluster-sync（如CUTLASS），支持仅同步必要的CTA子集以构建higher-level collectives

  **解决效果**：通过DSM将on-chip memory从单SM SMEM 227KB扩展到cluster级~3.6MB，使8MB C matrix的tile可以留在片上。减少58% global memory access，GEMM chain kernel speedup 3.1×-5.4× over baselines，整体1.24× end-to-end。关键不是简单"多一层缓存"，而是把DSM变成可搜索、可建模、可codegen的编译器抽象层次

## VAR-Turbo: Unlocking the Potential of Visual Autoregressive Models through Dual Redundancy

- baseline方法是什么？
  Baseline是CPU/GPU上朴素VAR（Visual Autoregressive）推理：VAR每次迭代仅decode一个visual token，一张256×256图像需256至4096次串行Transformer invocation。每次invocation内完整执行attention（O(N²)复杂度）和FFN，不利用图像token的空间冗余和不同Transformer层间的计算冗余。MaskGIT类parallel decoding使用启发式mask schedule但质量和硬件友好性不足；speculative decoding类方法引入draft model额外开销且仅2-3 token/step并行度。ViTCoD、AdapTiV等ViT ASIC accelerator仅对attention或FFN做单侧冗余利用，仍为串行解码。

  全栈执行例子（以Vanilla VAR 256×256 + V100 GPU为例）：
  - 算法层：VAR自回归next-token prediction——每step选1个visual token→完整transformer forward（QKV attention O(N²) + FFN，所有token参与计算）→写回→下一step——重复256次。注意力图随着层数加深趋于相似但未利用此特性。单张256×256图像生成常需10-60秒。
  - 系统框架层：论文未明确说明（V100单卡PyTorch推理，无多请求调度）。
  - 编译框架层：论文未明确说明（PyTorch默认编译路径）。
  - kernel调度层：GPU上标准attention和FFN kernel（cuBLAS/cuDNN），无定制dataflow。TopK排序使用通用Bitonic Sort+Merge Sort。
  - 硬件架构层：Intel Xeon Platinum 8168 CPU、NVIDIA V100 GPU；ASIC baseline ViTCoD/AdapTiV（28nm）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文从信息论角度刻画VAR模型的双重冗余（Dual Redundancy）：(1) Image Redundancy——图像token的空间相关性远高于语言token（entropy/redundancy分析），使跨迭代并行解码成为可能；(2) Model Redundancy——attention可解释为低通滤波，随着层数加深，深层attention map趋于相似形成"模型惯性"。基于此提出三项算法优化+两项硬件dataflow。

  **缺陷1（Inter-Iteration）：每次仅解码1个visual token，数百至数千次串行Transformer调用导致高延迟**
  → PD (Draft-Free Parallel Decoding)：无需draft model，在每轮中对所有masked token位置预测并按置信度TopK一次性解码多个token。原理：图像token比语言token有更高的空间相关性，熵值更低，因此可直接用模型自身的置信度判断而非依赖draft model。PD-aware training选择sampling temperature、masking ratio r(t)、guidance scale。256×256下仅需8-32步（减少>80%），一轮最多解码64 token。

  **缺陷2（Intra-Iteration - 浅层）：浅层attention全局计算存在大量token间相似性冗余**
  → TA (Token Aggregation)：利用attention作为低通滤波的insight——浅层仍处于"Learning Region"，但局部窗口内token高度相关。将token分local window→Small Attention（OP dataflow）将窗口内token聚合为representative token→Big Attention（Row dataflow）对浓缩后的代表token做全局建模。减少41% attention MAC，质量下降<0.5%。

  **缺陷3（Intra-Iteration - 深层）：深层attention map趋于相似、token重要性分化，但baseline对所有token执行完整计算**
  → DB (Dynamic Bypass)：深层"Inert Region"中注意力趋于均匀化，大部分token信息增量低。轻量MLP对每token打分→Radix Sort Core选TopK重要token→仅这些token进入完整attention+FFN→被bypass token通过token restoration（Token_i × JudgeWeight_i + Token_i）将原有信息补回下一层输入。额外减少58% MAC。schedule function控制逐层skip rate（α=0.3, β=-0.4, max=0.55）。

  **缺陷4（硬件）：TA引入Small/Big Attention异构执行模式，PD和DB需大K TopK（N=4096时K=1936），通用排序方案在大K上延迟高（TopK仅3.5%操作但占20.9%延迟）**
  → Unified Attention Core：同一PE array通过Snooper+Fat Tree动态切换Row（Big Attention）和OP（Small Attention）dataflow，避免为两类attention放独立core造成低利用率。
  → Radix Sort Core：将大K TopK转为固定4阶段流水线（CountBin→PrefixSum→SelectBin→Filter），加Locality-aware Scheduling根据mask history优先调度高置信区域，消除全局排序的反复读写重排开销。

  论文方法全栈执行例子（以VAR-Turbo-Balance 256×256 8步为例）：
  - 算法层：VQGAN tokenization→全masked V0。每轮：Transformer预测所有masked token→PD Gumbel sampling+置信度TopK→释放K(t)个token→mask更新。浅层0-15层TA（Small Attention local window→representative token→Big Attention全局），深层DB（MLP打分→TopK→attention+FFN→bypass token restoration）。8步完成（vs baseline 256步），FID 2.67 vs baseline 2.65。
  - 系统框架层：论文未明确说明。
  - 编译框架层：论文未明确说明。
  - kernel调度/硬件架构层：28nm VAR-Turbo accelerator (7.09mm², 1.98W) + HBM2 32GB/s。Unified Attention Core Row/OP dataflow+Radix Sort Core 4-stage TopK+MLP/Non-Linear/SIMD Cores层间pipeline。vs V100 GPU 210.3× speedup, 423.5× energy efficiency improvement。

## AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding

- baseline方法是什么？
  Baseline是现有LLM serving系统的continuous batching策略（vLLM、TensorRT-LLM、Sarathi-Serve），将同一batch中所有请求同质化处理，iteration-level scheduling导致batch内各请求per-token latency近似一致。当strict SLO请求（如coding copilot TPOT < 50ms）与relaxed SLO请求（如summarization TPOT < 150ms）混部时，要么缩小batch牺牲吞吐满足严格请求，要么保持大batch导致latency violation。vLLM + Priority等优先级策略通过限制batch或抢占非紧急请求照顾严格请求，但损害整体SLO attainment。vLLM-Spec和SpecInfer虽使用speculative decoding，但策略静态（固定speculation length/width），缺少per-request SLO aware allocation，不能随请求分布和系统负载动态调节。

  全栈执行例子（以vLLM + Llama3.1-70B + coding copilot(50ms SLO) + chatbot(100ms SLO) + summarization(150ms SLO) 混部为例）：
  - 算法层：所有请求统一使用自回归decode（每轮出1 token）或固定speculation length speculative decoding（每轮尝试出n个token）。同一batch中三个请求sync barrier等待后一起进入下一decoding step。
  - 系统框架/Serving层：vLLM continuous batching + FCFS scheduler + PagedAttention。batch composition由到达顺序决定，非SLO-aware。coding copilot因strict SLO可能在batch中因其他请求的decode时间积累而violate TPOT。
  - 编译框架层：论文未明确说明（使用PyTorch + CUDA默认编译路径）。
  - kernel调度层：vLLM默认CUDA kernel（FlashInfer/PagedAttention），无per-request token-level调度。
  - 硬件架构层：NVIDIA A100 80GB GPU，无定制硬件。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文将multi-SLO serving形式化为带budget约束的token tree构造问题，提出SLO-customized speculative decoding的speculate-select-verify三阶段pipeline。

  **缺陷1：Baseline continuous batching将batch内请求同质化，所有请求在一次decoding iteration中同步前进相同步数，无法为strict SLO请求提供更快的decoding速度**
  → AdaServe允许同一batch中不同请求在同一次LLM verification中接受不同数量的draft token：strict SLO请求在SLO-customized selection中优先获得足够高概率节点（保证一次verification可能接受多个token），relaxed SLO请求仅获少量节点（不浪费验证budget）。打破continuous batching的"所有请求同速前进"限制。

  **缺陷2：Baseline speculative decoding策略静态（固定speculation length/width），不随per-request SLO需求和系统负载动态调节**
  → AdaServe根据每个请求的当前latency、已生成token数、TPOT SLO阈值和全局活跃请求数动态计算per-iteration SLO推进目标。同时根据活跃请求数动态调节speculation depth d和beam width w：高负载时减小d/w降低speculation overhead，低负载时增大d/w提升吞吐。

  **缺陷3：Baseline优先级策略通过限制batch或抢占照顾strict请求，损害整体SLO attainment和goodput**
  → AdaServe的SLO-customized + throughput-optimized两阶段token allocation：先用budget满足所有请求SLO需求（SLO优先），剩余budget再按全局path probability最大化总吞吐。严格请求不通过牺牲其他请求满足，而是通过speculation depth/width和token tree节点分配的精细粒度实现。

  **缺陷4：Draft model和target LLM colocate时speculation overhead高，尤其在动态调节depth/width时kernel launch overhead显著**
  → AdaServe用CUDA Graph优化draft model decoding：从第二个speculation step到第d步，若活跃请求数相同则复用预捕获CUDA graph，消除重复kernel launch overhead。实验显示CPU selection overhead仅占总serving time的0.41%/0.31%。

  论文方法全栈执行例子（以AdaServe + Llama3.1-70B + coding copilot/chatbot/summarization 混部为例）：
  - 算法层：每轮decoding iteration：(1) draft model beam search生成candidate token tree（每请求d步×w beam），记录draft logits近似path probability；(2) SLO-customized selection：coding copilot（TPOT SLO最紧）优先获高prob节点，累计≥3 expected accepted tokens达SLO目标；chatbot获中等节点（≥1 token）；summarization仅获最低需求节点；(3) throughput-optimized selection：剩余budget全局最高prob节点分配；(4) target LLM tree-based verification并行验证所有请求selected trees。
  - 系统框架/Serving层：FlexFlow Serve + SLO-customized scheduler。Request manager维护per-request latency/token/SLO状态。FlashInfer batched prefill kernel改造用于speculation+verification。CUDA Graph复用减少draft decoding kernel launch开销。Dynamic d/w根据活跃请求数调节。
  - 编译框架层：论文未明确说明。
  - kernel调度层：FlashInfer batched prefill kernel改造用于tree-based parallel verification；CUDA Graph预捕获draft model的固定shape decoding steps。
  - 硬件架构层：4×NVIDIA A100 80GB GPU（NVLink），无定制硬件。

## PiLLM: Resource-Efficient LLM Inference Using Workload Prediction

- baseline方法是什么？
  Baseline分为两层：(1) 跨GPU层面：KServe/云服务常用autoscaling依赖GPU utilization等硬件指标，但LLM中长输入/长输出可能在利用率未明显变化时显著拉长TTFT/TPOT，导致扩容滞后；固定最大资源分配能保SLO但代价是长期GPU空转。(2) 单GPU层面：vLLM的greedy resource utilization策略追求即时显存利用率但可能带来高eviction（可达68.39%）；PastFuture和SGLang等更保守策略能压低eviction但显存利用率和batch size偏低。

  Baseline全栈执行例子（以vLLM greedy + utilization-based autoscaling为例）：
  - 算法层：LLaMA-3.1 8B，标准prefill+decode pipeline，KV cache按paged attention分块为per-request固定block。
  - 系统框架/Serving层：vLLM continuous batching，greedy per-request KV cache reservation（每个请求预留max output length的KV空间），跨GPU通过KServe/Kubernetes HPA根据GPU utilization触发autoscaling。
  - 编译框架层：论文未明确说明。
  - kernel调度层：LightLLM token-level KV cache管理，标准attention kernel。
  - 硬件架构层：8×NVIDIA H800 GPU（NVLink），无定制硬件。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出PiLLM，一个predictable inference系统。核心方法分两部分：(1) 用滑动窗口统计+中心极限定理预测批级输入/输出长度分布，再转为FLOPs和KV cache内存需求；(2) 基于该predictor，实现跨GPU的elastic dispatch和单GPU的batch-aware KV cache scheduling。

  **缺陷1：Utilization-based autoscaling依赖硬件指标间接反映负载，长请求导致扩容滞后**
  → PiLLM直接估算LLM真实工作量（prefill/decode FLOPs），而非等待GPU utilization间接反映负载。滑动窗口维护输入/输出长度均值方差，对批级平均长度构造带error bound的预测值，用离线校准系数转FLOPs后计算所需prefill/decode实例数。能更快应对长请求造成的计算峰值。

  **缺陷2：固定最大分配过度预留GPU，长期空转浪费**
  → PiLLM用管理窗口内的预测工作量削减不必要实例。spike reaction兜底：当无法满足deadline时尝试组成最大可行batch并快速激活新实例。不同workload上实现1.62x-3.06x GPU节省，prefill SLO满足率≥97.9%。

  **缺陷3：Per-request KV cache reservation（vLLM greedy）显存利用率高但eviction可高达68.39%；保守策略（PastFuture/SGLang）eviction低但显存利用率低**
  → PiLLM的batch-aware KV cache scheduler利用大数定律：单个请求输出长度难以预测，但多个请求的平均行为方差随batch size增大而下降。不对每个请求预留worst-case剩余长度，而是在batch级别更新共享KV cache预算，以可控error bound允许更高显存overcommit。结果显存利用率78.93%-96.05%，eviction rate仅0.01%-0.53%。

  **缺陷4：固定per-request KV cache block分配不灵活，无法池化共享**
  → PiLLM将KV cache组织为token slot linked list而非固定block。请求需要新token空间时从共享池分配，释放后的slot回池。配合批级memory pool，某些请求生成少于预测值释放余量供其他请求使用。

  论文方法全栈执行例子（以PiLLM disaggregated prefill/decode + LLaMA-3.1 8B为例）：
  - 算法层：LLaMA-3.1 8B，标准prefill+decode pipeline。新增控制面的统计采集、批级资源预测、跨GPU实例数管理和批级KV cache memory pool。预测公式：输出长度上界 μ_d + σ_d/√|B| * Φ⁻¹(1-ε)，离线校准系数转FLOPs/memory。
  - 系统框架/Serving层：基于LightLLM扩展，三层架构——API layer（输入长度采集）、global scheduling layer（predictor + inter-GPU manager + dispatcher）、execution layer（prefill/decode instances + intra-GPU manager + batch memory pool）。Disaggregated prefill/decode，prefill compute-bound独立缩放，decode memory-bound独立缩放。
  - 编译框架层：chunk-based workspace、AOT compilation和CUDA graph降低动态内存分配与JIT抖动。
  - kernel调度层：LightLLM token-level KV cache（token slot linked list），无定制kernel。
  - 硬件架构层：8×NVIDIA H800 GPU（NVLink），无定制硬件。

## FlexPipe: Adapting Dynamic LLM Serving Through Inflight Pipeline Refactoring in Fragmented Serverless Clusters

- baseline方法是什么？
  Baseline涵盖两类系统：(a) serverless-based LLM serving：ServerlessLLM使用DeepSpeed parallelism做分布式推理，侧重低延迟serverless部署和快速资源provisioning；Tetris通过tensor sharing实现memory-efficient hosting，无专门pipeline parallelism。(b) offline-optimized pipeline系统：AlpaServe基于历史请求模式做pipeline architecture优化，使用statistical multiplexing with model parallelism；MuxServe使用spatial-temporal multiplexing做多租户serving。这些系统的共同特征是pipeline拓扑在部署前由离线优化决定，服务过程中基本保持不变——主要靠调整batch size或replica数应对负载变化，pipeline stage数和切分方式不会在运行时改变。

  Baseline全栈执行例子（以AlpaServe + OPT-66B + 4-stage pipeline为例）：
  - 算法层：OPT-66B标准Transformer decoder。AlpaServe离线分析历史请求分布，用整数规划确定pipeline并行度和stage placement。4-stage配置下per-stage compute约69.94ms，inter-stage communication约6.3ms，max batch 128。无运行时stage粒度调整能力。
  - 系统框架/Serving层：AlpaServe在OSDI 2023提出的statistical multiplexing框架，pipeline配置在模型加载时固定。收到请求后按FCFS或SLO-aware调度分发到固定stage sequence。当请求burst到来时，只能通过增加整个pipeline的data-parallel replicas扩容（pipeline-level scaling），每个新replica需要完整加载全部stage参数。
  - 编译框架层：论文未明确说明。Alpa依赖ML编译器（如XLA、TVM）做子图编译，但不涉及运行时pipeline重构。
  - kernel调度层：使用标准CUDA kernel + NCCL all-reduce/point-to-point通信。无定制kernel。
  - 硬件架构层：GPU集群。Baseline假设tensor parallelism依赖的NVLink/InfiniBand高速互联可在同机获得多卡；但在碎片化serverless环境中，同机4卡可用概率仅0.02%，78%的tensor parallelism请求被迫退化为pipeline parallelism。

  Baseline的根本缺陷：
  (1) 静态pipeline无法适应请求分布CV剧烈变化：请求CV在180s/3h/12h窗口间波动可达7×。4-stage pipeline在CV=1时性能良好，但当CV升至4时pipeline stall cycle ratio指数增长22×，goodput下降37%。静态pipeline只能为一种CV值优化。
  (2) 碎片化GPU资源无法被静态pipeline有效利用：GPU平均订阅率216%，仅8.7%概率获得单张>85%空闲显存的GPU，0.02%概率同机获得4卡。Tensor parallelism依赖的同机高速互联几乎不可用，而粗粒度pipeline在碎片化GPU上stage放置困难。
  (3) 弹性扩容冷启动慢且资源预留浪费：生产集群维持历史峰值75%的GPU为always-on，但平均SM utilization仅17%。弹性扩容需从持久存储重载参数，多秒级延迟违反交互式LLM应用的亚秒SLO。
  (4) 多模型multiplexing时bursty workload的CV²型干扰未被建模，静态sharing策略在资源波动时性能恶化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  FlexPipe提出Inflight Pipeline Refactoring：将pipeline stage粒度从部署前的一次性决策转变为运行时连续控制变量。系统由三段协同设计构成：

  (1) Fine-Grained Model Partitioning：离线对LLM计算图做operator-level profiling（computation time, parameter size, activation size），用带约束的动态规划在GPU显存上限、inter-stage bandwidth、target computation-communication overlap cycle约束下求解最优切分，同时保留Transformer attention block层级结构边界以支持未来stage合并。多粒度候选granularity set覆盖从粗（4-stage）到细（32-stage）的pipeline配置。

  (2) Inflight Pipeline Refactoring：运行时持续监控CV、队列长度、吞吐/延迟profile。通过多目标优化公式 g*=argmax[α·T_k/T_max+(1-α)·L_min/L_k·exp(-|ν_t-ν_k|/σ)] 选择最优granularity。CV升高时执行Pipeline Expansion：申请新GPU、加载split stage、异步KV cache迁移（token-level validity mask C(t)=∪KV_i(t)⊗M_valid，优先RDMA、fallback sendfile）、更新gateway routing。CV回落时执行Stage Consolidation：layer-wise merge state、重建pipeline、释放多余instance。决策延迟<5ms。

  (3) Adaptive Pipeline Scaling with Topology-Aware Allocation：Hierarchical Resource Graph在server/rack/cluster三级协调资源。Memory-aware elastic scaling将参数保留在host memory，affinity-based scheduling优先选择最近承载过该模型的服务器，将cold start转warm start。GPU分配建模为约束优化：max throughput efficiency − γ(CV_i)·multiplexing penalty（penalty ∝ CV²）。

  对应解决Baseline缺陷的具体设计：
  - 对缺陷(1)：FlexPipe把pipeline深度变成CV的函数。高CV时自动切换到细粒度pipeline（如16-stage），单stage compute从69.94ms降至18.67ms，max batch从128扩至512，利用distributed buffering吸收burst；低CV时合并回粗粒度降低inter-stage communication。CV=4时16-stage pipeline的latency仅为4-stage的1/3。
  - 对缺陷(2)：FlexPipe放弃依赖同机高速互联的tensor parallelism，全面采用pipeline parallelism（point-to-point O(1) vs all-reduce O(n²)）。Fine-grained stage使单stage显存需求大幅降低（32-stage下参数加载仅5.43s vs 4-stage的47.14s），更容易在碎片化GPU中找到placement。
  - 对缺陷(3)：Host memory参数缓存+affinity scheduling将扩容初始化延迟平均降72%。Always-on GPU从峰值75%降至30%（降70%），剩余按需弹性分配，资源分配等待时间降85%。
  - 对缺陷(4)：Multiplexing penalty γ(CV_i)=γ_0·(1+α·CV_i²)建模bursty workload的并发资源尖峰。高CV时保持stage隔离避免干扰，低CV时允许GPU共享。实现43% utilization下维持12,000 req/s（vs static系统85% utilization仅1,543 req/s），资源效率最高8.5× better。

  论文方法全栈执行例子（以FlexPipe + OPT-66B在82-GPU Kubernetes集群为例）：
  - 算法层：OPT-66B标准Transformer，无算法修改。控制面新增CV-based pipeline granularity selection（公式4）、stage-level data parallelism M(g_k)计算（公式5）、multiplexing penalty modeling（公式9）、SLO约束scaling granularity validation（公式12）。
  - 系统框架/Serving层：FlexPipe自研~7K LoC serving runtime（非基于已有框架）。三组件协同——API Manager + Resource Monitor → Performance Controller（granularity adaptation + pipeline refactoring）→ Scheduler（HRG topology-aware allocation + memory-aware warm start）。核心机制：fine-grained stage切分→CV驱动的granularity selection→Pipeline Expansion/Stage Consolidation→KV cache一致性迁移（C(t)=∪KV_i(t)⊗M_valid）→gateway routing update。数据传输用RDMA/sendfile替代NCCL。
  - 编译框架层：论文未明确说明。离线partitioning阶段做operator-level profiling和计算图分析，但不涉及编译框架修改。
  - kernel调度层：无定制kernel。使用标准CUDA kernel。KV cache迁移使用RDMA或sendfile系统调用做kernel-space zero-copy传输。
  - 硬件架构层：Kubernetes集群42台服务器82张NVIDIA A100 GPU，100Gbps网络，部分支持RDMA。无定制硬件。

## TokenFlow: Responsive LLM Text Streaming Serving under Request Burst via Preemptive Scheduling

- baseline方法是什么？
  Baseline包括三套：(1) SGLang conservative scheduling（默认FCFS/prefill优先continuous batching调度器）；(2) SGLang with chunked prefill（在SGLang上启用chunked prefill的改进版本）；(3) Andes（QoE-aware scheduling with Token Pacer，论文在SGLang中用recompute-based preemption方式实现）。

  SGLang FCFS全栈执行例子（以Llama3-8B + SGLang on H200为例）：
  - 算法层：Llama3-8B标准Transformer decoder，无算法修改。prefill阶段将prompt tokens批量处理，decode阶段自回归逐token生成。
  - 系统框架/Serving层：请求按FCFS到达顺序进入SGLang scheduler → prefill阶段获得GPU优先调度（prefill-prioritized）→ decode阶段continuous batching合并多个请求的decode step → KV cache提前分配（pre-allocation）在GPU显存中 → 请求完成后释放KV cache。在burst负载下形成head-of-line blocking：早到请求持续占用GPU生成token（约30 tokens/s），后到请求在队列中等待→TTFT激增。SGLang H200 micro-benchmark显示burst时TTFT可超过用户可接受阈值。
  - 编译框架层：论文未明确说明（SGLang默认编译路径，PyTorch + CUDA）。
  - kernel调度层：SGLang默认CUDA kernel，无定制KV cache I/O调度。普通write-back策略：仅在显存压力下被动evict KV cache到CPU memory。
  - 硬件架构层：NVIDIA H200/RTX 4090/A6000 GPU。GPU显存直接持有KV cache，CPU memory作为被动溢出空间。

  Baseline的根本缺陷：
  (1) 调度与用户消费速率脱节：FCFS/prefill优先调度不感知"用户只按固定速率消费token"这一text streaming特性。早到请求即使已生成大量超出用户阅读/听取速率的token，仍继续占用GPU→资源错配。过快生成的token堆在客户端buffer中无实际价值→raw throughput虚高，effective throughput低。
  (2) Head-of-line blocking在burst下恶化：突发请求到达时，已在运行的请求持续占用GPU和KV cache→新请求排队→TTFT膨胀。SGLang在burst load上升时TTFT可超过用户可接受阈值。
  (3) 抢占无KV cache协同：Andes引入QoE-aware scheduling和Token Pacer改善感知延迟，但其抢占机制带来频繁context switch，缺少与KV cache管理的协同。直接抢占会把KV cache搬移变成显存和I/O瓶颈，吞吐和资源利用受损。
  (4) Compute-bound到memory/I/O-bound的相变：随着更多请求交替在GPU上运行，KV cache总量超过GPU显存容量→系统从compute-bound转向memory/I/O-bound→PCIe带宽成为瓶颈。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  TokenFlow提出buffer-aware preemptive scheduling + hierarchical KV cache management，把调度目标从最大化raw token throughput改为最大化effective throughput和响应性。核心由三段协同设计构成：

  (1) Buffer-Aware Two-Step Scheduling：第一步determine working set——根据GPU显存、请求KV footprint、等待队列长度、I/O队列长度和buffer安全条件决定可过量承诺的请求集合。第二步buffer balancing——在working set内按buffer size、weighted token generation quantity、required output rate和内存约束做greedy selection，再用局部交换优化优先级。请求buffer越低、用户消费速率越高，越容易获得运行优先级；buffer较高的请求可被暂时抢占，继续由客户端buffer平滑输出。

  (2) Streaming QoS Metric：用effective throughput替代raw throughput作为核心优化目标。effective throughput根据text streaming体验对token加权：buffer小于总输出长度10%的token全计（用户急需消费），10%-20%之间线性衰减，超过20%的token不计入有效吞吐（用户尚未消费到的"过早token"无实际价值）。

  (3) Hierarchical KV Cache Manager：GPU显存作为CPU memory上大容量KV cache的高速cache。Write-through策略（每次decode iteration同步新KV chunk到host）替代普通write-back策略（仅在抢占时写回）→抢占时大部分KV cache已在host同步，剩余chunk通过load-evict overlap与load stream重叠传输。三并行CUDA stream pipeline（compute/load/evict）+ 动态chunk sizing + batched transfer + CUDA events协调非阻塞执行。

  对应解决Baseline缺陷的具体设计：
  - 对缺陷(1)：TokenFlow把"请求是否继续占用GPU"变成随buffer实时变化的在线决策。Scheduler根据每个请求的实时buffer token数、用户token消费速率和I/O状态决定admission/preemption/resumption。buffer较低或尚未获得首token的请求优先获得GPU→让GPU时间投入到"用户真正要消费的token"。
  - 对缺陷(2)：预占式调度让GPU在早到请求积累足够buffer后安全转向紧急请求（新到达或即将耗尽buffer的请求）。利用客户端buffer覆盖切换延迟→burst场景下P99 TTFT最多降低80.2%，mean TTFT最多降低48.4%。
  - 对缺陷(3)：Scheduler和KV Cache Manager双向协作——scheduler在决策时考虑I/O overhead和recompute/load代价，memory manager在后台提前write-through可能被抢占请求的KV cache。真正发生preemption时不必完整同步所有cache→抢占恢复延迟大幅降低。
  - 对缺陷(4)：Write-through策略将KV cache持续同步到CPU memory，GPU显存作为高速cache。当working set超过GPU显存时，evict/load操作通过CUDA streams与compute overlap，避免同步I/O stall。消融实验：去掉offload完成时间从66.00s恶化到127.28s（恶化93%），说明分层内存管理是收益核心。

  论文方法全栈执行例子（以TokenFlow + Llama3-8B + H200为例）：
  - 算法层：Llama3-8B标准Transformer decoder，无算法修改。控制面新增Streaming QoS metric（effective throughput加权公式）和buffer safety condition（buffer >= 切换延迟 × 消费速率）。
  - 系统框架/Serving层：TokenFlow基于SGLang ~3000行Python代码。五组件协同——Request Tracker（实时追踪buffer/消费速率/资源占用）→ Buffer-aware Scheduler（两步调度：working set determination + buffer balancing，周期性reschedule）→ Request Offload Manager（请求状态转移管理）→ LLM Executor（SGLang continuous batching engine）→ Hierarchical KV Cache Manager（write-through + 三CUDA stream并行pipeline）。请求生命周期包含多次可能的pause/resume循环：早到请求积累buffer → 安全抢占 → GPU转向紧急请求 → 原请求buffer耗尽前恢复 → 循环。
  - 编译框架层：论文未明确说明（SGLang默认PyTorch + CUDA编译路径）。
  - kernel调度层：Hierarchical KV Cache Manager使用PyTorch CUDA stream API管理三并行stream（compute/load/evict）。Write-through在每次decode iteration后将新KV chunk写入write buffer并异步同步到host。Load-evict overlap通过CUDA events协调：preempted请求已同步chunk直接释放，未同步chunk与load操作重叠。动态chunk sizing根据compute duration预估选择传输大小，最大化compute-I/O overlap。
  - 硬件架构层：NVIDIA H200/RTX 4090/A6000 GPU + host CPU memory。GPU显存为高速cache层，CPU memory为大容量KV cache存储层。PCIe作为KV cache搬移通道，write-through策略持续占用PCIe带宽但换取抢占时更低的上下文切换延迟。Huawei Ascend 910B也被报告支持。

## Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading

- baseline方法是什么？
  Baseline是四种SOTA MoE serving系统（均经论文作者改造以公平比较）：(1) MoE-Infinity：使用request-level Expert Activation Matrix记录每个request历史上激活的expert集合，同步进行expert prediction和prefetching，prefetch与inference forward不重叠；(2) DeepSpeed-Inference：expert-agnostic layer-wise parameter offloading，按层on-demand从CPU加载参数到GPU，无expert级别的prefetch/cache；(3) Mixtral-Offloading：layer-wise speculative expert prefetching，使用LRU cache管理GPU侧expert，预测粒度为layer级而非iteration级；(4) ProMoE：stride-based speculative prefetching + per-layer NN predictor，需要针对每个MoE模型训练predictor（论文因ProMoE未开源只做best-effort复现）。共同根因：baseline依赖request-level热度统计或固定规则（stride/LRU），忽略不同iteration、不同layer、不同prompt语义之间的细粒度差异，导致expert hit rate低或GPU cache被低价值expert占据。

  全栈执行例子（以MoE-Infinity + Mixtral-8x7B + RTX 3090 24GB为例）：
  - 算法层：Mixtral-8x7B MoE decoder-only模型，每层8 experts×top-2 routing。Gate network每个token选择2个expert。MoE-Infinity在request history中统计每个request过去激活的expert，构建Expert Activation Matrix。新请求到达时，基于历史矩阵做expert prediction→prefetch predicted experts→forward。
  - 系统框架/Serving层：MoE-Infinity codebase。Request到达→查询Expert Activation Matrix→predict requested experts→同步prefetch（blocking forward）→MoE layer gate network top-2选择→若命中GPU cache则直接计算，若miss则on-demand CPU→GPU loading。问题：request-level聚合冲淡iteration-level清晰模式（entropy analysis显示coarse-grained patterns比fine-grained更不可预测），同步prefetch增加延迟。
  - 编译框架层：论文未明确说明（PyTorch + CUDA默认编译路径）。
  - kernel调度层：MoE-Infinity在GPU侧管理expert cache，使用CUDA Runtime APIs做expert weight的CPU↔GPU传输。Expert loading通过PCIe 4.0（32GB/s），单次expert loading耗时与expert size成正比。
  - 硬件架构层：6×RTX 3090（24GB each, pairwise NVLinks, PCIe 4.0 32GB/s）+ AMD Threadripper 3955WX（32 cores, 480GB CPU memory）。Expert parallelism将不同expert分布到不同GPU。inactive expert占据GPU显存（72%-84% of parameters为inactive），造成数十GB显存浪费。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出FineMoE，核心是用iteration-level expert probability distribution（Expert Map）替代request-level expert activation count，结合semantic和trajectory similarity做细粒度expert prefetch/cache决策。

  **缺陷1：Request-level Expert Activation Matrix冲淡iteration-level pattern → expert prediction hit rate低**
  → FineMoE方案：Expert Map Store记录每个inference iteration、每个MoE layer上gate network对所有experts的概率分布。Entropy analysis显示fine-grained patterns比coarse-grained patterns更可预测（coarse-grained统计将早期decode iteration的清晰模式与后续iteration混成高熵统计）。概率分布不仅表示expert是否被选中，还表达gate network对expert的置信度→低相似度时多预取降低miss，高相似度时少预取节省显存。

  **缺陷2：固定规则（stride/LRU）忽略prompt语义和iteration间expert使用变化 → 预测与当前context脱节**
  → FineMoE方案：Dual-Mode Expert Map Search。前d层：semantic-based search——从embedding layer获取prompt semantic embedding，与历史embedding做cosine similarity取最相似历史iteration的前d层expert probability map。d层之后：trajectory-based search——收集当前iteration已过层的gate probability trajectory，与历史expert maps做cosine similarity。两种模式分别在prompt语义相似度和execution trajectory相似度上匹配，使预测适应当前上下文。

  **缺陷3：同步prefetch阻塞inference forward → 增加TTFT/TPOT**
  → FineMoE方案：Publisher-Subscriber异步路径。Context Collection→Expert Map Searcher→Prefetch Publisher→Cache Subscriber与inference forward完全解耦。forward执行时search和prefetch在后台异步进行，不阻塞compute。仅在cache miss时暂停普通prefetch task执行on-demand loading。

  **缺陷4：LRU eviction与MoE layer-wise顺序执行特征不匹配 → 驱逐即将使用的expert**
  → FineMoE方案：Probability-Aware Prefetch/Eviction Priority。Prefetch priority = p/(l-l_now)：概率越高、距离当前layer越近，越优先预取。Eviction priority = 1/(p×freq)：低概率、低访问频率expert优先驱逐。直接量化"未来哪些expert更可能且更快被用到"，而非LRU的时间局部性假设。

  **缺陷5：粗粒度预测无法在hit rate和GPU memory之间fine-tune → cache capacity利用不高效**
  → FineMoE方案：Similarity-Aware Dynamic Selection。delta = clip(1-score, 0, 1)，按概率从高到低选择最少expert使累计概率超过delta。低相似度时delta高→多预取→降低miss risk；高相似度时delta低→少预取→节省GPU cache。动态均衡hit rate和memory footprint。

  FineMoE全栈执行例子（以Mixtral-8x7B + LMSYS-Chat-1M + 6×RTX 3090为例）：
  - 算法层：Mixtral-8x7B MoE decoder模型不变（每层8 experts×top-2 routing）。控制面新增Expert Map Store（iteration-level expert probability map）和Similarity-Aware Selection（delta=clip(1-score,0,1)）。FineMoE不改变gate network或模型权重，仅在expert placement层面优化。
  - 系统框架/Serving层：基于HuggingFace Transformers + MoE-Infinity codebase改造。六组件协同：(1) Context Collection：从embedding layer提取semantic embedding（前d层），收集已过层gate probability作为trajectory；(2) Expert Map Searcher：订阅context，semantic search（前d层）+ trajectory search（d层之后），PyTorch native ops做pairwise cosine similarity；(3) Similarity-Aware Expert Selector：按similarity score算delta，选累计概率超delta的最少expert；(4) Expert Cache（C++/CUDA Runtime）：按p/(l-l_now)异步prefetch，按1/(p×freq) evict；(5) On-Demand Loading：cache miss时暂停普通prefetch，立即CPU→GPU加载缺失expert；(6) Map Updater：iteration结束写回新map，redundancy score去重。多GPU expert parallelism用hash map+round-robin分配。整体在6GB GPU cache下vs DeepSpeed-Inference/Mixtral-Offloading/ProMoE/MoE-Infinity的TPOT分别降低36%/25%/16%/29%。
  - 编译框架层：论文未明确说明（PyTorch + CUDA默认编译路径）。
  - kernel调度层：Expert Cache用CUDA Runtime APIs实现expert weight的CPU↔GPU传输和GPU cache管理。Prefetch通过异步task pool+scheduler执行，与inference forward CUDA stream重叠。On-demand loading使用高优先级task打断普通prefetch。Expert weight传输通过PCIe 4.0（32GB/s）。
  - 硬件架构层：6×RTX 3090（24GB, NVLinks, PCIe 4.0 32GB/s）+ AMD Threadripper 3955WX（32 cores, 480GB CPU memory）或A100 80GB。GPU显存作为高速expert cache，CPU memory作为大容量expert weight存储。Expert Map Store在CPU memory（1K maps <200MB CPU memory，32K maps仍低于200MB）。PCIe 4.0带宽32GB/s决定expert loading latency下限。


## High Throughput and Low Latency LLM Serving via Adaptive KV Caching

- baseline方法是什么？
  Baseline是三种vLLM中已有的或复现的KV cache管理策略：(1) vLLM-Recompute：GPU显存满时丢弃preempted requests的全部KV cache，等请求重新获得显存后整体重算所有历史token的KV，问题是长序列下重算开销高且粒度为整请求；(2) vLLM-Swap：GPU显存满时将preempted requests的全部KV cache swap到host memory，再通过PCIe恢复，问题是host-GPU带宽（PCIe 4.0 x16约32GB/s）成为瓶颈，长序列swap延迟高；(3) HCache：缓存hidden states加速状态恢复，但恢复仍发生在request-level或layer subset粒度，decode阶段仍倾向于保留当前并发请求的完整KV cache。共同缺陷：all-or-nothing恢复策略未利用"旧token可部分重算、SM有空闲、显存更稀缺"的资源不平衡——论文实测Llama2-13B在A100上显存常接近饱和而GPU SM utilization处于很低区间。

  baseline全栈执行例子（以Llama2-13B + vLLM-Recompute + ShareGPT单请求在A100上decode为例）：
  - 算法层：Llama2-13B MHA decoder模型不变，每层40 heads × 128 head_dim，每token每层产生约40×128×2个float16 KV元素。
  - 系统框架/Serving层：vLLM continuous batching + PagedAttention。请求进入decode后KV cache按PagedAttention block分配填充。当GPU显存满时scheduler选择一批请求preempt，丢弃其全部KV cache blocks。请求重新被调度时从token 0开始逐token recompute所有layer的KV（O(L×T)重算开销），然后继续decode新token。batch size受限于GPU显存必须容纳所有running request的全部KV cache + 模型权重。
  - 编译框架层：PyTorch torch.compile默认路径，无特殊kernel编译优化。
  - kernel调度层：PagedAttention默认GEMM kernel + FlashInfer/PagedAttention kernel，无融合。KV swap路径：cudaMemcpy host↔GPU同步传输（一条stream阻塞）。
  - 硬件架构层：A100-80GB GPU，108个SM，PCIe 4.0 x16连接host。GPU显存容纳模型权重(~26GB for Llama2-13B FP16) + batch内所有请求的完整KV cache。SM在重算期间利用率短暂上升但大部分时间因显存瓶颈batch size受限而idle。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出eLLM adaptive KV caching系统，核心设计：(1) token-wise/layer-wise KV cache管理：从整请求粒度细化为token和layer粒度，每个请求缓存较新的(1-r) token KV、释放较早的r token KV（或swap到host），用map table追踪每个(token, layer)的缓存/swap/recompute状态，layer-granular block按F=4层划分减少内存碎片；(2) cross-request batched recomputation：将多个请求的uncached token聚合成batch统一重算KV，提高GEMM/attention效率——避免小batch单独重算效率差；(3) request-level SLSQP在线优化：将batch size b和uncached token ratio r作为优化变量，在GPU显存和TPOT SLO约束下用SciPy SLSQP求解器最大化throughput；(4) layer-wise communication-computation overlapping：双CUDA stream，Stream A异步传输cached KV（host↔GPU），Stream B执行recomputation+decode；(5) kernel fusion：将compute-intensive K1（旧token KV recompute）和memory-intensive K2（新token decode）融合，减少launch overhead并动态分配线程；(6) closed-loop adaptation：layer-level实际额外显存Mo反馈给request-level，形成b和r的闭环调整。

  论文方法全栈执行例子（以Llama2-13B + eLLM + ShareGPT单请求在A100上decode为例）：
  - 算法层：Llama2-13B MHA decoder模型不变。eLLM不修改模型权重或attention公式，仅在KV cache存储策略和kernel执行层面优化。
  - 系统框架/Serving层：基于vLLM扩展。request-level optimizer用SLSQP求解b和r（如r=0.4）。历史序列前40% token的KV被释放或swap，后60% token的KV保留在layer-granular block中（F=4层一组block）。进入transformer layer i时layer-level scheduler通过map table精确定位cached token物理block，确定uncached token在本层是recompute还是swap。本轮可容纳更大batch size（因每请求KV residency降低约40%）。
  - 编译框架层：PyTorch + 预编译CUDA shared libraries。对K1+K2 fused kernel预生成32组线程variant（32-1024 step 32），运行时按K1/K2计算量比选择.so。
  - kernel调度层：Stream A异步cudaMemcpyAsync传输layer i+1的swapped KV，Stream B执行fused kernel K1+K2——K1为layer i+1的uncached旧token重算KV（利用cross-request batching提高GEMM效率），K2用layer i完整历史KV对current token decode。线程按K1/K2 FLOP比例分配（如70:30），取整到32的倍数。K1临时KV使用后立即释放，额外workspace仅约1 layer KV。Ttoken completion后新token KV写入对应layer-granular block，layer-level计算Mo反馈给request-level。
  - 硬件架构层：A100-80GB GPU，108 SM。eLLM用compute换memory——SM在recompute K1时利用原本idle的计算资源，换出显存容纳更多并发请求。PCIe 4.0 x16的swap传输与computation并行避免成为单一瓶颈。

  **缺陷对应关系**：
  - vLLM-Recompute的整请求全量重算 → token-wise partial recomputation只重算已释放的token/layer，且cross-request batching提高重算效率
  - vLLM-Swap的整请求host↔GPU全量传输 → 按uncached ratio部分传输+cached token保持GPU resident，减少PCIe传输量
  - 三者共享的"decode阶段每个活跃请求必须保留完整KV cache"假设 → 直接降低每个active request的KV residency，从源头提高并发容量（用空闲SM compute换显存）
  - 粗粒度KV block → layer-granular block（F=4）减少碎片
  - 重算与decode串行 → fused kernel + dual stream overlap消除idle bubble


## MFS: An Efficient Model Family Serving System for LLMs

- baseline方法是什么？
  Baseline是Orca serving system + 独立部署model family中多个模型（如分别部署Llama2-7B和Llama2-13B）。Orca的selective batching假设batch内所有请求共享同一模型结构，因此无法高效跨不同规模模型batching；独立部署小模型和大模型时，模型权重和KV cache都需分别保存，显存占用随模型数增加而上升；speculative sampling中draft model（7B）和target model（13B）参数/层结构不一致，小模型生成的token后大模型无法直接复用其KV cache，切换模型需重复prefill/decode计算。

  全栈执行例子（以Orca分别部署Llama2-7B + Llama2-13B + 独立speculative sampling为例）：
  - 算法层：两个独立LLM（Llama2-7B-chat、Llama2-13B-chat），各自独立推理，无参数/KV cache共享。Speculative sampling时小draft model生成候选token→大target model需从头prefill这些token→无法复用draft的KV cache。
  - 系统框架/Serving层：Orca selective batching按模型类型分组→7B请求batch和13B请求batch完全独立→GPU compute和显存无法跨模型共享。两份模型权重独立加载→显存占用=W(7B)+W(13B)。两份KV cache独立管理→跨模型不共享。每个模型各自维护waiting queue→各自调度各自batch。
  - 编译框架层：论文未明确说明（PyTorch默认编译路径）。
  - kernel调度层：论文未明确说明（使用Orca默认CUDA kernel，未提出新kernel）。
  - 硬件架构层：NVIDIA A100 GPU（2卡）/ NVIDIA 3090 GPU（8卡），无定制硬件。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出MFS，用Knowledge Precipitation将model family最大模型微调为嵌套multi-tier模型，配合group batching、attention fusion、shareable KV cache和tier-aware speculative sampling。

  **缺陷1：Orca selective batching假设batch内请求共享同一模型结构→无法跨不同规模模型高效batching→GPU compute underutilization**
  → MFS方案：Group batching。multi-tier模型使不同tier共享公共前缀层→tier-level scheduler将需要相同前缀层的不同tier请求组成group batch→公共层一起执行→GPU compute利用率提升。decode阶段通过attention fusion拼接QKV进一步提升GPU并行度（以少量冗余attention计算换并行度）。

  **缺陷2：独立部署多模型→模型权重和KV cache双重占用GPU显存→显存压力限制并发数**
  → MFS方案：Shareable KV cache + 嵌套模型权重共享。所有tier共享低层Transformer参数（单份权重替代多份独立模型），KV cache manager将公共前缀层的KV cache标记为multi-tier shareable→高tier可直接复用低tier KV cache→GPU memory footprint降低47.8%。

  **缺陷3：独立draft/target speculative sampling→小模型KV cache对大模型不兼容→target验证需重新计算前若干层prefill**
  → MFS方案：Tier-aware speculative sampling。draft和target是同一嵌套模型的不同tier→低tier产生的hidden states和KV cache对高tier完全兼容→高tier验证时直接继承低tier KV cache，无需重新prefill→GPU utilization从23.9%提升到59.8%。

  **缺陷4：直接构造multi-tier模型的baseline方法（strawman early-exit/PEFT/deep pruning）无法同时保证质量、层连续性和KV cache兼容性**
  → MFS方案：Knowledge Precipitation + layer-only tier split + step-by-step fine-tuning。(a) 只切layer不切head→保持attention中所有head一致性，确保跨tier KV cache可共享；(b) step-by-step逐层沉淀知识而非一次性joint training→避免多tier loss梯度冲突导致低tier性能不可控；(c) latency-aligned切分→使tier延迟匹配目标小模型用户体验。

  论文方法全栈执行例子（以Llama2-13B → 3-tier MFS + 2×A100 serving为例）：
  - 算法层：Knowledge Precipitation将Llama2-13B fine-tune为嵌套3-tier模型（tier-1=前18层/~3B等效, tier-2=前32层/~10B等效, tier-3=全部40层/13B）。每个tier有独立lm_head和training loss，低tier通过高tier梯度反向传播获得知识。Step-by-step fine-tuning：先train tier-3→加tier-2 head co-train→加tier-1 head co-train，逐层沉淀。
  - 系统框架/Serving层：front-end接收请求和所需tier→request pool→tier-level scheduler维护三个tier队列→group batching将需要相同前缀层的请求合并执行→attention fusion在decode阶段拼接QKV提升GPU并行度→shareable KV cache manager记录所有tier公共前缀层KV为可共享→tier切换时直接复用。可选speculative sampling：tier-1 draft若干token→tier-3 verify，复用tier-1 KV cache。
  - 编译框架层：论文未明确说明（使用PyTorch默认路径）。
  - kernel调度层：论文未明确说明（未提出新GPU kernel，attention fusion使用标准attention实现但拼接QKV输入）。
  - 硬件架构层：训练16×H800 SXM5 80GB + 400Gbps InfiniBand、推理2×A100/8×3090，无定制硬件。

## Efficient Multimodal Serving via Module Multiplexing

- baseline方法是什么？
  Baseline是传统unimodal serving系统（Triton/ TF-Serving/ Clipper/ TurboTransformers）和Gpulet（spatio-temporal GPU sharing）。Triton将多模态模型当作monolithic computation graph，所有模块顺序执行，使用统一batch size和100% SM allocation。Gpulet将每个模块当作独立模型并发执行，但强制共享固定batch size且无模块间协同调度。

  全栈执行例子（以BLIP VQA + Triton + RTX 3090为例）：
  - 算法层：BLIP由三个模块组成：ViT-B/16 visual encoder (86M params)、BERT-based text encoder (110M params)、BERT-based text decoder。请求以统一batch size（如B=8）进入。
  - 系统框架/Serving层：Triton将BLIP作为单一模型图→一次接收B个请求→先对B个图像做preprocessing（resize/normalize/tensor conversion/H2D transfer，耗时可达visual encoder计算时间的40%）→visual encoder全部batch推理（GPU接近满利用率）→text encoder全部batch推理（GPU利用率骤降，因text encoder计算强度远低于visual encoder）→text decoder全部batch推理（GPU继续低利用率）。同一batch内所有请求必须等待最慢模块完成。如VQA中单图对应多个问题，每个问题独立通过全流程，visual encoder被重复计算。
  - 编译框架层：论文未明确说明（使用PyTorch默认编译路径）。
  - kernel调度层：论文未明确说明（使用vLLM默认CUDA kernel）。
  - 硬件架构层：NVIDIA RTX 3090/A100，无定制硬件。GPU active SM在超过50%时间低于10%（论文Figure 4）。

  Baseline缺陷总结：
  1) **Inference heterogeneity**：visual encoder latency可达text encoder的8×，text encoder在B≈8后饱和而text decoder到B>32仍受益，统一batch无法同时匹配各模块最优工作点。
  2) **Preprocessing heterogeneity**：图像预处理（loading/resize/normalize/H2D）可达visual encoder计算时间的40%，串行执行让GPU在预处理期间空闲。
  3) **Input imbalance / reuse缺失**：VQA2中单图至少对应3个问题（最多246个），每个请求重算visual encoder浪费compute和latency。
  4) **Gpulet虽支持spatio-temporal sharing**，但将模块视为独立竞争模型，无synergistic scheduling、无stage-level pipeline、无modal cache reuse。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出EEVEE，核心是module multiplexing：将多模态模型的modality-specific modules作为独立调度单元，配置独立batch size和SM allocation，在同一GPU上通过NVIDIA MPS并发执行。系统由offline scheduler（greedy search生成multiplexing strategies）和online controller（modal cache管理）两部分组成。

  **缺陷1→方法**：inference heterogeneity导致统一batch下GPU利用率波动剧烈
  → EEVEE方案：Module-level scheduling + balanced SM allocation。每个模块运行独立vLLM process，拥有各自最优batch size（text decoder可用大batch、visual encoder用小batch）；SM allocation按模块计算强度动态划分（从短latency模块转移SM到长latency模块直到stage内各模块latency趋近平衡）。这样让visual encoder获得充足SM的同时text decoder以更大batch充分占用剩余SM，填充原baseline中的GPU bubble。

  **缺陷2→方法**：preprocessing heterogeneity导致GPU等CPU
  → EEVEE方案：Stage-level parallelism。将预处理和推理视为独立阶段，visual encoder的新请求预处理可与text decoder的推理在同一GPU上并发执行。GPU不再因等待CPU完成预处理而闲置。

  **缺陷3→方法**：input imbalance导致visual encoder被重复计算
  → EEVEE方案：Modal cache control + request-aware reuse。对encoder-decoder模型（BLIP）缓存cross-attention消费的视觉token KV pairs，对decoder-only MLLM（LLaVA）缓存decoder消费的视觉token KV pairs。后续引用同一图像的问题通过64-bit hash查找缓存，跳过visual encoder重算。Cache支持compression（按attention score剪除低重要性token），在GPU memory紧张时优先使用compressed critical cache，full cache保留在host memory。

  **缺陷4→方法**：Gpulet类spatio-temporal sharing将模块视为独立竞争模型
  → EEVEE方案：Synergistic scheduling algorithm。将模块间的directed dependency、egress module batch multiplier、SLO latency约束和SM总量约束（per-stage SM总和=100%）建模为约束优化问题。Greedy search从monolithic顺序执行的可行策略初始化，逐步增加各模块batch size，每次将增量放在对batch latency负面影响最小的stage，直到违反SLO。这避免了随意并发导致的资源竞争退化（论文strawman实验：不合适策略使throughput从22.2降到14.1 req/s，合适策略可达34.4 req/s）。

  论文方法全栈执行例子（以BLIP VQA + EEVEE + RTX 3090为例）：
  - 算法层：BLIP模块不变（ViT-B/16 + BERT encoder + BERT decoder），但执行方式从整模型顺序变为模块级并发。Visual encoder batch size按复用需求设为较小值（如B_v=2），text decoder batch size可扩大（如B_t>2）。Modal cache compression按30% ratio剪除低attention视觉token，score仅轻微下降。
  - 系统框架/Serving层：三个独立vLLM process（visual encoder、text encoder、text decoder）通过NVIDIA MPS并发执行→offline scheduler根据SLO、模型结构和GPU硬件生成strategy（如visual encoder 70% SM，text decoder 30% SM）→controller管理GPU shared memory中的modal cache（64-bit hash索引、per-module hashmap、global LRU eviction）。用户请求包含图像+Q1：visual encoder处理图像→controller缓存visual tokens→text decoder生成A1；同一图像Q2到达→controller从cache加载visual tokens→跳过visual encoder→cache loading与Q1的text decode或新图像的encode overlap→生成A2。Stage-level parallelism使Q2的preprocess与Q1的inference重叠。
  - 编译框架层：论文未明确说明（使用PyTorch默认编译路径，vLLM后端）。
  - kernel调度层：论文未明确说明（未提出新GPU kernel，SM allocation通过CUDA MPS active thread percentage环境变量在CUDA初始化前设置）。
  - 硬件架构层：NVIDIA RTX 3090/A100，PCIe 3.0 32GB/s，使用NVIDIA MPS（非MIG）实现GPU spatial sharing。GPU active SM接近90%。

## Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

- baseline方法是什么？
  **模型scaling baseline**：直接增大模型参数（Qwen2.5 3B/7B base model, Llama3.2 3B base model）在手机上运行，但这带来更高的内存占用（3B model dmabuf ~2090MiB, total ~2.4GiB）、带宽需求和功耗。**小模型单路径baseline**：1B/1.5B模型用单路径autoregressive decode，成本低但数学推理准确率不足（Qwen2.5-1.5B MATH500 base accuracy ~50%）。**系统baseline**：llama.cpp OpenCL backend使用Snapdragon Adreno GPU的Q4_0 GEMM kernel，在decode batch size=1时更快，但大batch test-time scaling下无法利用Hexagon HMX矩阵tile的空置行。**QNN baseline**：Qualcomm闭源QNN框架仅支持per-tensor/per-channel粗粒度量化，Llama3.2-1B-Instruct的QNN W4A16在MATH500上准确率从AutoAWQ的15.9骤降至2.1、GSM8K从32.6骤降至3.4，精度损失使test-time scaling不可行。

  **全栈执行例子（baseline: llama.cpp OpenCL + Qwen2.5-1.5B单路径decode）**：
  - 算法层：Q4_0 group quantization (group size 32, conventional column-major layout)，无tile layout transformation，单token autoregressive decode via greedy sampling。
  - 系统框架/Serving层：llama.cpp main executable → OpenCL backend → Adreno GPU command queue。每个decode step：CPU准备activation (shape [1, hidden_dim]) → clEnqueueWriteBuffer → GPU OpenCL kernel执行GEMV（GEMM退化，32-wide warp仅1 lane有效计算）→ clEnqueueReadBuffer → CPU sample next token。
  - 编译框架层：论文未明确说明（OpenCL runtime编译，无自定义compiler pass）。
  - kernel调度层：Adreno GPU OpenCL Q4_0 matmul kernel：column-major weight layout → per-group dequantization (INT4→FP16 via unpack+convert) → GEMV inner product → 单token activation仅利用GPU warp的1/32 lane → Adreno GMEM/TCM未针对HMX tile优化。无FlashAttention on GPU（使用标准attention实现）。
  - 硬件架构层：Snapdragon Adreno GPU (OpenCL)，Hexagon NPU的HMX matrix unit在decode阶段闲置（非目标硬件）。CPU处理所有logits/lm_head、sampling。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **方法概述**：将"更多test-time compute"转化为decode batch parallelism，用Hexagon NPU HMX matrix unit原本空置的tile行做多候选生成(Best-of-N/Beam Search)，让1B/1.5B小模型+test-time scaling达到或超过3B/7B base模型的accuracy-latency/accuracy-energy Pareto frontier。核心技术创新：(1) Hardware-aware tile quantization：权重按HMX FP16 tile layout重排→在memory order上做group size 32量化→group coalescing适配HVX 1024-bit register；(2) LUT-based Softmax/dequantization：用HVX vgather/vlut16查表替代多项式exp和mask-unpack-convert。

  **缺陷→方法映射**：
  - 缺陷1：模型scaling增大memory/带宽/功耗→方法：不增大模型参数，用test-time scaling增加decode batch parallelism。Best-of-N中Qwen2.5-1.5B TTS结果超过3B base accuracy，Beam Search中Llama3.2-1B达到与3B相当效率。1.5B模型batch size=8 decode energy低于3B model batch size=1。
  - 缺陷2：QNN粗粒度量化破坏fine-grained quantization精度→方法：tile-group quantization保留group size 32的fine-grained量化精度，但将layout重塑为HMX tile format。Tile quantization group与conventional group的WinoGrande/MMLU/Wiki PPL差异远小于量化本身的精度损失。
  - 缺陷3：HVX通用计算能力弱（单thread FP16 GEMM仅33 GFLOPS vs HMX 12 TFLOPS），多项式exp和传统dequantization成为瓶颈→方法：LUT-based computation。Softmax用64KiB FP16 LUT + vgather替代exp2 Taylor展开，加速1.26-2.19× vs F32 exp。Dequantization用vlut16直接INT4→FP16 + LUT广播scale，相比conventional layout加速9.65-19.04×。
  - 缺陷4：GPU OpenCL backend在大batch下无法利用HMX tile空洞→方法：将decode batch映射到HMX 32×32 tile的多行。batch size从1增至16时decode throughput随batch增大显著提升（因HMX tile行利用率从1/32升至B/32），而HMX计算时间几乎不增。

  **论文方法全栈执行例子（Qwen2.5-1.5B Best-of-N B=8, OnePlus 12 Snapdragon 8 Gen 3）**：
  - 算法层：Best-of-N parallel sampling (B=8条候选路径) + Skywork-1.5B-PRM outcome reward scorer。Weight采用Q4_0 tile-group quantization: HuggingFace weight→HMX 32×32 tile layout重排（tile级column-major + tile内2-row permutation）→在memory order上group size 32量化（等于2×16 tile片段）→8 group coalesce为128-byte super-group→HVX vlut16查表INT4→FP16 dequantization→HMX FP16 tile-level inner product。
  - 系统框架/Serving层：llama.cpp CPU backend → rpcmem/dmabuf shared memory → FastRPC remote NPU session → Hexagon NPU operator library。CPU维护8条候选路径的KV cache和sampling状态，lm_head/logits在CPU侧计算。NPU侧thread pool轮询共享内存中的operator request。
  - 编译框架层：Hexagon SDK 6.0.0.2 LLVM toolchain编译，无QNN依赖，使用reverse-engineered FP16 HMX指令。离线模型转换脚本：HuggingFace→HMX layout GGUF→llama-quantize(Q4_0/Q8_0/F16混合)。
  - kernel调度层：每个transformer layer的linear层执行时→DMA搬weight tile和activation tile入TCM→HVX vlut16 dequantization+scale broadcast（产生HMX-compatible FP16 tiles）→HMX执行32×32 tile MM（8条候选路径占8 activation rows, 24 rows空置）→accumulate到FP32 internal accumulator→output tile写回TCM。Attention执行：FP16 FlashAttention tile-level Q/KV分块→HMX算QK^T和PV→HVX rowmax+rowsum+LUT_Exp (64KiB TCM LUT)→FP16 online softmax (critical accum in FP32)。DMA→TCM→HVX→HMX→TCM→DMA形成闭环比。
  - 硬件架构层：Snapdragon 8 Gen 3 Hexagon V75 NPU。HVX (1024-bit VRF, 4-6 units)负责dequant/LUT/scale/reduction，HMX (1-2 units, ~12 TFLOPS FP16)负责tile MM，TCM 8MiB承载LUT(64KiB)+weight tiles+activation tiles+KV cache tiles，DMA ~60GB/s搬移DDR↔TCM。CPU负责lm_head/sampling/KV cache管理。Total device power <5W (1.5B model decode)，dmabuf 1056MiB (1.5B, ctx=4096)，total memory ~1.3GiB。

  **关键trade-off**：依赖Qualcomm Hexagon SDK、FastRPC、reverse-engineered FP16 HMX指令，移植到其他NPU需重新适配tile layout和指令。Decode speed受runtime dequantization overhead限制。CPU logits计算在batch size=16时占比≥50%，削弱scaling收益。仅评估数学推理Best-of-N/Beam Search，对不适合并行采样的任务收益不成立。

## TailorLLM: Collaborative End-Cloud Inference of Large and Small Language Models Based on Low-Rank Adaptation

- baseline方法是什么？
  Baseline是cloud-only LLM推理服务（Llama3-70B纯云端部署），用户所有请求都发送到云端LLM执行decoder-based autoregressive inference。这是当前LLM推理服务的主流部署方案，提供稳定高精度体验，但面临cloud computing costs急剧上升的问题（推理服务成本已接近或超过预训练成本）。

  全栈执行例子（以Llama3-70B cloud-only text summarization请求为例）：
  - 算法层：单个用户请求→Llama3-70B autoregressive decode，每token经全部decoder layers（RMSNorm→Grouped Query Attention→FeedForward Network with SiLU gate），无任何端侧参与。全精度推理。
  - 系统框架/Serving层：请求到达云端服务器（4×RTX 3090）→排队等待GPU→prefill处理prompt→逐token decode→返回response。单请求无法利用GPU并行计算能力（sequential token generation成为瓶颈），无请求间batching时GPU利用率低。
  - 编译框架层：论文未明确说明（使用默认PyTorch/CUDA推理路径）。
  - kernel调度层：论文未明确说明（使用默认GPU kernel，无定制kernel）。
  - 硬件架构层：NVIDIA RTX 3090 GPU (24GB)。cloud-only下所有计算集中在云端GPU，成本按API定价累积（输入$2.50/1M token + 输出$10.00/1M token）。

  论文同时将token-level end-cloud collaboration（HSL, speculative decoding）和model partitioning（Petals）作为对比baseline。HSL方法：端侧SLM生成draft tokens→云端LLM逐token验证（每5 token验证一次），频繁端云交互导致累计延迟（单次QA可能有15+次通信），token验证频率高时接近cloud-only延迟。Petals方法：将Llama3-70B按5:65拆分到端云两侧，受限于端侧显存。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **方法概述**：TailorLLM提出task-level（而非token-level）端云协同推理框架。核心理念：观察用户请求在task-level具有高度集中性（少数任务覆盖>70%请求）和时间周期性，用LoRA微调SLM使其在high-frequency任务上达到接近LLM的精度→端侧自主完成这些任务→减少云端调用频率和端云通信轮次。两个关键算法：RFLoRA（离线低秩微调，冻结共享A矩阵，仅传任务特定B+magnitude，减少~50%传输）和AdapterMgr（在线imitation learning管理端侧LoRA cache，用Mamba时序模型+Belady最优策略学习做近最优替换决策）。

  **缺陷→方法映射**：

  **缺陷1：Cloud-only方案成本过高，GPU并行能力在single-request autoregressive decode中未充分利用**
  → 方法：Task-level任务分流。高频简单任务（text translation/sentiment analysis/summarization等）在端侧经LoRA增强的SLM完成，复杂/稀有任务（数学推理/新类别）发往云端LLM。实验：约70%请求端侧处理，cloud cost减少69.8%，end-to-end latency减少62%（vs cloud-only）。Llama3-1B推理速度22.6ms/token vs Llama3-70B 5.3ms/token，但由于端侧无网络RTT，总体延迟更低。

  **缺陷2：Token-level collaboration（HSL）频繁端云通信（单QA 15+次），在弱网环境下累计延迟严重，且cloud validation部分抵消成本节省**
  → 方法：Task-level替换token-level。端侧SLM经LoRA增强后在特定任务上独立完成推理（而非仅生成draft tokens等待cloud验证），单次请求最多1次端云通信（仅在卸载任务时）。实验：RTT 20→200ms时TPOT仅1%退化（HSL 22%退化、Petals 46%退化），TTFT始终保持低位。TailorLLM端侧hit rate决定cloud cost节省程度（AdapterMgr hit rate接近Belady最优）。

  **缺陷3：标准LoRA adapter传输开销大（每个adapter ~22MB for Llama3-1B），在无线网络下更新端侧LoRA library成本高，限制可存储的adapter数量**
  → 方法：RFLoRA参数解耦。发现A矩阵跨任务收敛（domain-invariant encoder），B矩阵任务特异（domain-specific transformation），且权重可分解为direction+magnitude。设计：冻结共享A矩阵（所有任务共用一份），仅传输B矩阵+magnitude参数m（~11.56MB vs 22MB，减少~50%）。解耦还使magnitude分量不依赖frozen A就能独立优化方向分量。实验：RFLoRA达到81.6% accuracy (3.4M trainable params, 0.273%) vs LoRA 81.2% (0.454%)、DoRA 82.1% (0.484%)。Trainable params约为DoRA的56%，但精度仅差0.5pt。11.56MB adapter在端侧可存储更多任务（同等存储空间下数量翻倍）。

  **缺陷4：静态LoRA部署无法适应用户需求的动态变化（任务周期性+新任务出现），LRU简单策略预测精度低**
  → 方法：AdapterMgr基于imitation learning的动态cache管理。用Mamba (SSM)捕捉用户历史访问序列的长程时序依赖（支持训练时并行、推理时recurrent），融合端侧cache state双模态信息（projection fusion），以Belady最优替换策略（evict longest reuse distance）为训练标签，BCE loss引导模型不仅学习最优action还学习区分"正确/错误"替换决策。滑动窗口H=100，cache capacity w=5，embedding dim d=128。实验：AdapterMgr在所有数据集上hit rate最接近Belady上界，在用户请求越动态（cycle 200 vs cycle 30）时相对LRU优势越明显。

  **论文方法全栈执行例子（以Llama3-1B + RFLoRA + AdapterMgr text summarization请求为例）**：
  - 算法层：RFLoRA离线训练：云端对text summarization数据集fine-tune Llama3-1B→权重W分解为magnitude m + direction W/||W||_c→direction分量施加LoRA ΔW=BA (A共享冻结,B可训练)→仅B(11.56MB)+m传输到端侧。端侧预存一份共享A矩阵。推理时加载B→合并W' = m·(W_0+BA)/||W_0+BA||_c→SLM执行autoregressive decode (22.6ms/token)。
  - 系统框架/Serving层：用户text summarization query到达→Task Classifier (Contriever semantic encoding→UMAP降维→HDBSCAN聚类)判定为已知summarization类别→Allocator查表确认精度达标+local cache命中→加载对应LoRA→端侧独立完成推理（0次云端通信）。AdapterMgr后台：维护历史访问序列H=100→Mamba提取时序特征→融合当前cache state→MLP输出替换概率→若B矩阵已在cache则hit（不替换），若miss且cache满则evict max eviction-probability slot并下载新B。
  - 编译框架层：论文未明确说明（使用默认PyTorch/CUDA路径，端侧Tesla T4 GPU推理）。
  - kernel调度层：论文未明确说明（使用默认GPU kernel，无定制kernel）。
  - 硬件架构层：Cloud-side 4×RTX 3090 (24GB)，End-side Tesla T4 (16GB→10GB limited)。端侧Llama3-1B推理占用~2.8GB显存，LoRA switching <1ms，task classification 0.45-1.53ms。端侧energy消耗经llama.cpp在手机上评估约等于轻量2D游戏功耗。

  **关键trade-off**：依赖LoRA adapter的wireless传输（每个~11.56MB），在弱网/低带宽下仍有传输延迟。SLM模型更新时所有已训练LoRA需重新训练。对新任务类型需积累足够样本才能形成新类别（HDBSCAN基于密度聚类），冷启动阶段依赖云端。仅评估1B/70B模型组合，未验证其他SLM/LLM组合。

## TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone

- baseline方法是什么？
  Baseline有三个层次：(1) REE-LLM-Memory：未修改llama.cpp在REE中运行，全部参数预加载在内存中，代表无保护、内存低效但接近理论最优性能的baseline；(2) REE-LLM-Flash：未修改llama.cpp在REE中运行，inference start时用pipelined restoration加载参数（buddy system allocation，不解密），代表实用但无保护的REE baseline；(3) Strawman：每次请求在TEE中冷启动、动态扩展secure memory、加载/解密参数、CPU-only计算，提供安全性和内存效率但缺少pipeline和NPU支持。Strawman是论文最直接对比的baseline。

  全栈执行例子（Llama-3-8B 512-token prompt, Strawman TEE cold start on RK3588）：
  - 算法层：llama.cpp decoder-only transformer（RMSNorm→Grouped Query Attention→FFN with SiLU gate），8-bit quantized weights，CPU-only inference。
  - 系统框架/Serving层：每次请求→启动llama.cpp→初始化metadata和tokenizer→CMA分配7.9GB连续物理内存（~4.182s）→从NVMe SSD加载加密模型参数（~4.054s）→OpenSSL AES解密（~0.892s）→CPU prefill 512 tokens（~164s量级）→decode。总TTFT包含约11.6s纯restoration overhead。参数在TEE secure memory中，但动态CMA allocation完全暴露在critical path上。
  - 编译框架层：论文未明确说明（llama.cpp使用默认C++编译路径，无定制编译器）。
  - kernel调度层：无pipeline——串行执行allocation→loading→decryption→computation，无operator重叠，无preemptive scheduling。CPU仅做computation无NPU参与。TZASC/TZPC在启动时一次性配置，无动态world switch。
  - 硬件架构层：RK3588 (4×A76 @2.4GHz + 4×A55 @1.8GHz, 16GB LPDDR4X, 3-core NPU ~6 TOPS, NVMe SSD)。Strawman中NPU闲置（CPU-only），TEE和REE无法共享NPU。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **方法概述**：TZ-LLM提出两个核心机制：(1) Pipelined Parameter Restoration：将LLM computation graph扩展为computation operator前插入allocation/loading/decryption三类restoration operator的DAG pipeline，配合priority-based greedy + preemptive micro-operator scheduling和partial parameter caching，将restoration latency隐藏到computation latency下；(2) Co-driver NPU：将完整NPU driver拆为REE control plane（scheduling/power/frequency）和TEE data plane（secure job context/MMIO launch/interrupt completion），通过shadow job+TZPC/TZASC/GIC完成安全NPU time-sharing。

  **缺陷→方法映射**：

  **缺陷1：Strawman TEE cold start中CMA allocation（~4.182s）+ flash I/O（~4.054s）+ decryption（~0.892s）完全串行暴露在TTFT critical path上，使TTFT增加~11.6s**
  → 方法：Pipelined Restoration将restoration与computation重叠。按computation graph拓扑顺序，为最早operator先分配/加载/解密参数→立即开始prefill→后续operator的restoration在后台推进。实验：vs strawman TTFT降低77.1%-91.1%（固定prompt length），real-world benchmark上降低76.1%-90.9%。Pipeline scheduling距critical path lower bound仅0.01%-9.9% overhead。

  **缺陷2：Strawman CPU-only prefill极慢（512-token Llama-3-8B ~164s）且NPU闲置**
  → 方法：Co-driver NPU使TEE安全使用NPU加速。REE掌握control plane（统一调度、电源/频率管理），TEE仅执行最小data plane（secure context验证、MMIO launch、interrupt处理）。Shadow job使secure job融入REE调度队列。Decoding阶段vs strawman提升0.9%-23.2%（NPU acceleration），vs REE-LLM仅1.3%-4.9%额外overhead（来自TEE/REE driver通信）。

  **缺陷3：Static secure memory partitioning（warm start方案）需长期预留约8GB参数内存，在24GB以内移动设备上内存效率差**
  → 方法：Partial Parameter Caching仅在REE memory pressure允许时缓存早期prefill operator参数（而非全部模型），inference后按reverse topological order释放。Pipeline-aware extend/shrink接口利用LLM参数first-in-last-out模式保证TZASC连续secure memory。实验：在达到cache比例阈值前TTFT近似线性下降；不使用时自动释放，不长期锁住完整模型内存。

  **缺陷4：NPU baseline中TEE和REE各放完整NPU driver会导致Rockchip NPU detach-attach ~32ms world-switch overhead，且将约60K LoC驱动代码纳入TEE TCB**
  → 方法：Co-driver将data plane缩小为TEE user-mode driver（仅secure job context/MMIO launch/interrupt completion），control plane留在REE。TZPC/TZASC/GIC硬件配置完成NPU secure mode切换（无driver detach-attach）。NPU time-sharing对REE NN应用额外slowdown最高3.8%，对LLM额外slowdown最高3.0%。TEE TCB仅增加约112 LoC（CMA mapping + TZASC/TZPC配置），不引入完整NPU driver（~60K LoC）。

  **论文方法全栈执行例子（Llama-3-8B 512-token prompt, TZ-LLM on RK3588）**：
  - 算法层：llama.cpp decoder-only transformer（同baseline），8-bit quantized weights，OpenSSL AES参数解密。计算本身无算法修改（不涉及稀疏/量化/蒸馏等算法pipeline优化）。
  - 系统框架/Serving层：TZ-LLM TA在TEE中运行→提取computation graph→按拓扑顺序调度restoration+computation operators→REE file system异步I/O直接写入allocated未保护内存→TEE OS动态扩展TZASC region映射入TA→OpenSSL解密参数→computation ready即开始CPU/NPU prefill→后续operator restoration在后台重叠执行。Partial parameter caching缓存早期prefill参数（内存压力允许时），用完按reverse order释放。
  - 编译框架层：论文未明确说明（llama.cpp使用默认C++编译路径，无定制编译器）。
  - kernel调度层：Priority-based greedy scheduler维护restoration operator队列→computation优先→否则执行earliest-computation关联restoration→alloc/decrypt切为micro-operator可抢占→computation就绪即抢占→消除pipeline bubble。Co-driver NPU路径：LLM TA准备secure execution context→提交shadow job→REE scheduler选中→smc通知TEE data plane driver→TZPC阻止REE访问NPU MMIO+GIC路由interrupt到TEE→等待non-secure job完成→TZASC允许NPU访问secure memory→sequence number验证→MMIO launch NPU job→interrupt完成→切回non-secure mode。
  - 硬件架构层：RK3588 SoC (Orange Pi 5 Plus)：4×A76负责CPU computation和scheduling，3-core NPU (~6 TOPS)通过co-driver安全执行矩阵运算。TZASC配置secure memory region（利用连续物理内存特性），TZPC配置NPU MMIO secure access，GIC路由NPU interrupt到TEE。CMA从REE Linux分配连续物理内存→TZASC保护后成为secure memory。NVMe SSD存储加密模型参数。stress-ng模拟memory pressure（Llama-3-8B对应6GB压力），stress threads与LLM threads pin到不同core。

  **关键trade-off**：pipelined restoration依赖LLM参数访问顺序足够确定（对MoE/early-exit等非确定workload可能预取未使用参数）。TZASC要求连续物理内存→仍受CMA allocation和内存碎片影响（虽然overhead变为transient可与I/O/compute重叠）。Co-driver依赖具体NPU driver可分离出小data plane（已在Rockchip实现，仅调查了Qualcomm开源NPU driver可行性，未跨平台完整验证）。安全模型不覆盖物理DRAM攻击/side channel/cryptographic attack/DoS。参数tensor size与secure NPU job execution time向REE泄漏模型结构级信息。TEE内解密会引入额外CPU overhead（vs REE-LLM-Flash在TTFT上5.2%-28.3% overhead）。

## AIMS: A Cost-Efficient Framework for LLM-based Agent Deployment in Cloud-Edge Hybrid Environments

- baseline方法是什么？
  Baseline有两个层次：(1) HybridLLM [8]：使用classifier对每个subtask独立判定走SLM或LLM，subtask之间无依赖感知、无位置感知。每次路由决策仅考虑当前subtask的内容特征，忽略subtask在agent reasoning workflow中的阶段（early/mid/late）和subtask间的因果链（前一个subtask的SLM输出可能导致后续subtask偏离LLM路径）；(2) Minions [31]：confidence-based routing，SLM先尝试执行每个subtask，用average log-probability衡量uncertainty，低置信度时escalate到云LLM。同样逐subtask独立决策，不建模subtask依赖和位置效应。两个baseline的共同缺陷：将AI agent的subtask序列视为彼此独立的单次请求集合，忽略agent workflow中subtask间的强依赖关系和cascading effect——一个早期subtask的错误路由可能改变后续整个subtask链。

  全栈执行例子（以HotpotQA "maternal grandfather of Titanic director"请求 + HybridLLM + Qwen3-4B/GPT-5 + RTX 5090为例）：
  - 算法层/Serving层：AutoGen agent生成subtask ST1="Identify Titanic director"→HybridLLM classifier判定ST1简单→路由到Qwen3-4B SLM→SLM输出"James Cameron"（正确）。但ST1的SLM执行导致agent state与LLM路径产生微小差异→ST2="Find James Cameron's mother"→SLM可能输出"Shirley Lowe"但遗漏full name细节→ST3="Find Shirley Lowe's father"因缺少中间名而搜索到错误人物。HybridLLM的classifier在每个subtask独立评估时可能都判定"简单"，但累积状态偏移（early divergence accumulation）最终导致最终答案错误。HybridLLM在HotpotQA上accuracy仅76.35%，SLM usage 68.40%，对后期subtask（late-stage）的LLM→SLM切换导致精度损失高达9.53%（vs early-stage仅5.25%），但classifier因无位置感知而对所有stage一视同仁。
  - 编译框架层：论文未明确说明（llama.cpp默认编译路径）。
  - kernel调度层：论文未明确说明（使用llama.cpp默认CUDA kernel）。
  - 硬件架构层：NVIDIA RTX 5090（本地SLM执行），云端LLM API（GPT-5/Claude Sonnet 4），无定制硬件。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **方法概述**：AIMS提出Adaptive Iteration-level Model Selector，将AI agent的subtask调度从"独立per-subtask routing"升级为"position-aware、dependency-aware的workflow-level routing"。五个核心组件：User Request Classifier（全请求级过滤）、Subtask Similarity Evaluator（subtask级fast-path）、S-L Similarity Evaluator（SLM-LLM距离预测回退路径）、Convergence Detector（未来收敛点搜索）、Subtask Decomposer（复杂subtask分解为SLM友好粒度）。所有estimator基于offline profiling数据用ModernBERT/Qwen3-0.6B + LoRA fine-tune（2小时/A100），在线推理仅需2GB VRAM。

  **缺陷→方法映射**：

  **缺陷1：HybridLLM/Minions将subtask视为独立单元，忽略subtask间依赖→早期错误路由产生cascading divergence，最终输出错误**
  → 方法（SSE + SP_SLM/SP_LLM预测比较）：AIMS不在当前subtask输出上做决策，而是用subtask predictor预测下一个subtask的输出，比较SLM和LLM对"next subtask"的预测相似度。这捕捉了路由决策对未来subtask链的影响——如果SLM生成的next subtask与LLM路径分歧，即使当前subtask输出相似也不会走SLM。Trace分析：HybridLLM failure中early divergence accumulation占53.18%，说明忽略依赖是主要失败模式。AIMS通过预测式比较提前阻断cascading error。

  **缺陷2：HybridLLM/Minions对所有subtask使用相同路由标准，不考虑stage-specific sensitivity→late-stage subtask精度敏感却与early-stage同等对待**
  → 方法（position-aware adaptive threshold κ(i)）：AIMS根据subtask序列位置动态调整相似度阈值。κ = threshold_base + min(ID, 5) · 0.02（base=0.6）。早期subtask（ID=1,2）κ≈0.62-0.64（宽松），允许更多SLM offload；后期subtask（ID≥5）κ=0.70（严格），更倾向LLM保证精度。实验验证：LLM→SLM切换在Late stage精度损失9.53% vs Early 5.25%——阈值自适应地反映了这一风险梯度。Trace分析：Minions failure中late-stage sensitivity占45.86%，因其无位置感知在后期仍aggressively offload。

  **缺陷3：HybridLLM/Minions缺乏"SLM虽当前subtask输出不同但未来可能收敛"的机制→过早放弃SLM机会导致cloud cost高**
  → 方法（SLE + CD）：AIMS引入S-L distance概念——LLM subtask的SLM"追赶距离"。通过Distance Predictor预测当前LLM subtask需多少额外SLM subtask才能匹配，再通过SP_SLM预测匹配点的输出与LLM比较。若高S-L similarity则SLM沿路径执行。CD进一步迭代搜索未来收敛点（选取最后一个以最大化SLM使用）。实验：SLE+CD联合贡献accuracy +3.72%、SLM usage +8.06%（消融实验中移除CD后）。Trace分析：HybridLLM/Minions因lack of convergence handling分别损失17.18%/24.07%的case。

  **缺陷4：复杂subtask超出SLM能力→直接走LLM损失offloading机会**
  → 方法（SD subtask decomposition）：AIMS用fine-tune的Subtask Decomposer将复杂subtask拆为更细粒度的子subtask序列（如"Verify Shirley Cameron's father, including corroborating biographical details"→拆为Search/Extract/Bio details/Confirm 4步），仅在所有子subtask都适合SLM时才整组offload。这避免了逐个子subtask单独路由可能导致的额外LLM调用。实验：SD单独贡献accuracy +1.58%、SLM usage +5.54%（消融实验中移除SD后）。SLM total usage 83.58% vs HybridLLM 52.20%。

  **缺陷5：HybridLLM以"request"为粒度做路由，忽略agent workflow是多subtask迭代过程→无法处理整请求级别的简单case**
  → 方法（URC request-level pre-filter）：AIMS在subtask routing之前先用URC判断整请求能否直接走SLM。若全请求输出similarity>0.7则跳过所有subtask routing。实验：w/o URC的SLM usage下降13.40%（from 83.58% to 70.18%），accuracy仅微降0.80%，说明URC在不牺牲精度前提下批量捕获简单请求的SLM机会。

  **论文方法全栈执行例子（HotpotQA "maternal grandfather of Titanic director" + AIMS + Qwen3-4B/GPT-5 + RTX 5090）**：
  - 算法层/Serving层（AIMS routing pipeline）：URC预测request similarity<0.7→进入subtask routing。ST1="Identify Titanic director"→SSE预测next subtask similarity→κ(1)=0.62（宽松）→SLM执行ST1，输出"James Cameron"。ST2="Find director's mother"→SSE similarity<κ(2)=0.64→SLE predict d=1（SLM多需1步可达LLM对应）→SP_SLM predicted ST3 vs SP_LLM predicted ST2 similarity>κ→SLM执行ST2→SLM自动生成ST2.5="Search James Cameron biography for mother's name"（S-L distance的额外subtask）。ST3="Confirm maternal grandfather"→SSE/SLE均失败→CD迭代搜索：SP_SLM/SP_LLM forward predict 3步→第3对similarity>0.7→收敛点在第3个future subtask→SLM执行ST3及后续2步。ST6（新生成的final confirmation）→SSE/SLE/CD全失败→SD分解为"Search Shirley Lowe's father"+"Extract father's full name"+"Find birth/death dates"+"Confirm maternal grandfather"→4个子subtask全部通过SSE→SLM整组执行。最终accuracy 90.75%、SLM usage 81.85%。
  - 编译框架层：论文未明确说明（llama.cpp默认编译，estimator PyTorch + LoRA fine-tune）。
  - kernel调度层：论文未明确说明（llama.cpp默认CUDA kernel，无定制kernel）。
  - 硬件架构层：NVIDIA RTX 5090 GPU本地运行Qwen3-4B（4-6GB VRAM），云端GPT-5 API。Estimator推理约2GB VRAM additional。调度决策overhead占总时间3-7%。网络hop latency平均0.58s。Cloud cost 0.17× vs All-LLM（83% savings）。

  **关键trade-off**：offline profiling需对每个SLM-LLM pair重新收集subtask binary tree数据并fine-tune estimator（约2小时/A100），换model pair时需重新profiling。Decomposition仅在全部分解子subtask适合SLM时才offload整组——保守策略确保了accuracy但可能错失部分子subtask可SLM而另一部分需LLM的混合case。Adaptive threshold依赖论文原实验数据拟合的参数（base=0.6, step=0.02, max=5），换模型对/dataset可能需重新调参。当前不支持显式online latency/货币budget约束——SLA-aware扩展留作future work。仅验证text-based agent（非多模态agent）。

## From Imperative to Declarative: Towards LLM-friendly OS Interfaces for Boosted Computer-Use Agents

- baseline方法是什么？
  Baseline是Microsoft UFO-2的GUI-only模式（UFO2-as + action sequence）。该baseline完全通过GUI与OS应用交互：LLM必须观察当前屏幕/控件树，规划并输出可见控件上的细粒度操作序列——点击菜单项、切换tab、展开dropdown、拖动滚动条、观察反馈再继续。这是一个命令式（imperative）交互范式：LLM承担从高层task规划到底层UI导航和操作的完整责任。

  全栈执行例子（以PowerPoint "将所有幻灯片背景设为蓝色" + UFO-2 GUI-only + GPT-5 medium + Windows为例）：
  - 算法/Agent层：LLM接收task→规划操作序列→每步输出具体动作（click Design tab、click Format Background、select Solid fill、click Fill Color、select Blue、click Apply to All）。每步需一次LLM推理调用，6+步长操作链易级联失败。成功仅44.4%，mechanism-level失败占53.3%。
  - 系统框架/Serving层：UFO-2 agent framework管理LLM调用轮次——每轮等待LLM输出→通过UIA/屏幕坐标执行动作→观察结果→构造下一步prompt。即使注册UIA event handler暴露完整control tree，LLM仍需将控件信息按alphabet labels嵌入prompt后，输出可见控件上的操作序列（而非声明意图）。平均步数8.16，时间392s。
  - 编译框架层：论文未明确说明（Agent级别不涉及编译框架）。
  - kernel调度层：论文未明确说明（不涉及GPU kernel）。
  - 硬件架构层：论文未明确说明（Windows通用PC，无定制硬件）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Declarative Model Interface (DMI)，核心是policy-mechanism separation：LLM只负责"语义上要做什么"的policy声明，DMI负责"如何导航到控件并完成交互"的mechanism执行。DMI分三步：(1) 离线构建UI Navigation Graph→转path-unambiguous forest；(2) 压缩为LLM友好core topology层级文本（将冗长XPath-like控件ID替换为连续整型ID）；(3) 在线通过access/state/observation三类声明式原语执行交互。

  全栈执行例子（同PowerPoint任务 + DMI + GPT-5 medium + Windows）：
  - 算法/Agent层：LLM从prompt中core topology层级文本理解功能树结构→声明访问目标控件ID（`Blue`对应leaf ID 128、`Apply to All`对应leaf ID 132，shared subtree中的需附带entry_ref_id）→声明内容只含语义层面"我要访问什么"，不含"如何到达"。
  - 系统框架/Serving层：DMI executor接收声明式JSON→解析控件ID为XPath-like primary_id|control_type|ancestor_path→求解唯一导航路径（确定性图搜索，非LLM规划）→fuzzy matching当前窗口控件→按OK>Close>Cancel优先级关闭浮窗→沿路径点击导航控件→到达后通过UIA control patterns执行状态设置（SelectionPattern选色、InvokePattern点击Apply to All）。成功74.1%（+29.6%），步数4.61（-43.5%），时间239s（-39%）。失败81.0%为policy-level。
  - 编译框架层：论文未明确说明。
  - kernel调度层：论文未明确说明。
  - 硬件架构层：论文未明确说明。

  **缺陷1：LLM必须生成完整GUI操作序列，视觉定位和精确交互是LLM弱项**
  → DMI将控件定位从LLM的视觉/文本推理转为确定性UIA树匹配（XPath-like: primary_id|control_type|ancestor_path）。不用LLM识别"蓝色按钮在哪里"，而通过UIA automation_id和ancestor_path做确定性映射。消融实验验证：为baseline注入DMI static navigation knowledge（文本描述或JSON），只要不启用声明式接口，SR反而从44.4%降到42.0%，证明**接口形式本身**决定性能——知识有用但不能通过命令式接口有效利用。

  **缺陷2：GUI-only对所有控件类型一视同仁，未利用UIA intrinsic semantics**
  → DMI对UIA控件分类处理：navigation控件（菜单、tab、dropdown）用DFS路径求解而非LLM规划；interaction控件通过control patterns做结构化状态读写（ScrollPattern替代拖拽坐标、TextPattern替代逐行选择、SelectionPattern替代多选点击）；transient窗口用OK>Close>Cancel优先级关闭以恢复导航状态。

  **缺陷3：长操作链级联失败，每步LLM推理延迟累计**
  → DMI通过state原语将多步复合交互压缩为单次声明：`set_scrollbar_pos(80%)`替代拖拽坐标定位、`select_lines(start, end)`替代多次点击+拖拽、`select_controls([id1,...])`一次性多选。interface lifting使步骤减少43.5%，降低累计延迟和级联风险。

  **缺陷4：全量5K+控件信息撑爆prompt context window**
  → DMI分层core topology压缩：默认只提供主干文本（Excel约30K tokens），通过`further_query`按需扩展缺失分支。连续整数ID替换冗长XPath-like ID进一步压缩token。

  **关键trade-off**：DMI以离线建模成本（每Office应用自动<3h + 人工约1.5人日）和在线prompt token开销（core topology 15K-30K tokens）换取执行稳定性和成功率。覆盖范围限于UIA可完整枚举的标准控件应用；自由绘图、精确位置调整、游戏/专业图形软件保留GUI slow-path fallback。需应用版本固定、UIA信息可用。

## FlashPS: Efficient Generative Image Editing with Mask-aware Caching and Scheduling

- baseline方法是什么？
  **Baseline**: Diffusers——标准扩散模型serving系统，使用static batching和mask-agnostic全图生成。FISEdit利用mask sparsity做图像编辑加速，但仅适配SD2.1、不支持不同mask ratio请求batching。TeaCache复用denoising中间activation跳过计算，面向通用生成任务，不能利用mask精确区分应保持不变区域。
  **Baseline全栈执行例子**（以Diffusers处理InstructPix2Pix编辑请求为例）：
  - 算法层：整个latent的所有spatial tokens通过DiT transformer全量计算attention和feed-forward，mask区域和unmask区域无差别处理。一个512×512 latent（约4096 tokens for SDXL）需要full attention over all token pairs。
  - 系统层：static batching——请求到达后被组为一组，等整批所有请求完成全部denoising steps后才一起输出。请求数负载均衡（request-count），不考虑不同请求mask ratio差异。
  - 编译框架层：HuggingFace Diffusers pipeline + PyTorch + FlashAttn，编译/框架层无mask-aware支持。
  - kernel层：标准attention kernel（FlashAttn）和FFN kernel，对全量tokens执行计算，无mask-aware sparsity。
  - 硬件层：NVIDIA A10/H800 GPU，标准HBM访存，无cache loading stream。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **FlashPS方法**：(1) Mask-aware cached activation——按mask划分tokens，unmasked区域直接复用缓存的block输出activation Y，仅对masked tokens做主要计算；(2) DP bubble-free block selection——对每个transformer block比较cache loading+masked计算 vs 全量计算的完成时间，O(N) DP决定每个block是否使用cache；(3) Denoising-step continuous batching + disaggregated preprocessing；(4) Mask-aware load balancing。
  **FlashPS全栈执行例子**（同一InstructPix2Pix请求，mask ratio=0.11）：
  - 算法层：输入latent按mask划分为masked tokens（~450 tokens）和unmasked tokens（~3650 tokens）。对每个transformer block，若DP决定使用cache：unmasked tokens的output activation Y从cache直接读取（已在模板预计算时存储），masked tokens在attention中作为query仅与所有tokens的K/V计算，但unmasked tokens的最终Y直接从cache填入。若DP决定跳过cache（因cache loading延迟大于全量计算）：该block对所有tokens执行标准full attention+FFN。block输出合并masked计算结果与cached unmasked activations后送入下一block。
  - 系统层：请求通过mask-aware scheduler路由——scheduler根据mask ratio=0.11用线性模型估算该worker computation latency（低，因计算量与m成正比）和cache loading latency（与unmasked tokens数成正比），选pipeline latency最低的worker。进入worker队列→preprocessing进程将image/mask编码为latent。GPU主进程在下一denoising step边界将请求加入running batch。请求完成50 steps denoising后立即退出batch→postprocessing进程解码为输出图像→返回客户端。新请求在step边界加入而不等待整批完成。
  - 编译框架层：HuggingFace Diffusers + PyTorch，修改attention operator（只对masked tokens执行attention计算，unmasked位置从cache直接注入）。保留FlashAttn作为attention backend。
  - kernel层：CUDA stream-based async cache loading——cache load stream从host memory异步加载cached Y到GPU HBM，computation stream并行对masked tokens执行FlashAttn attention + FFN。kernel级merge操作将masked计算结果与cached unmasked activations拼接回完整Y。
  - 硬件层：NVIDIA A10/H800 GPU。Cache存储在host memory（PCIe加载）或分布式存储。CPU preprocessing/postprocessing进程与GPU denoising主进程并行运行。

  **设计如何解决Baseline缺陷**：
  - 缺陷1（全图重复计算）：mask-aware cached activation将计算量降为与mask ratio m成正比，理论speedup≈1/m。生产trace m=0.11时理论可加速约9×，实际因cache loading开销mask ratio 0.2下SDXL加速2.2×、Flux加速1.9×。缓存Y而非K/V减少cache footprint约2×但对质量影响可控（SSIM 0.99）。
  - 缺陷2（static batching排队延迟）：denoising-step级continuous batching + disaggregated预处理将P95延迟相比static batching降低35%，相比LLM-style naive continuous batching降40%。
  - 缺陷3（忽略mask ratio差异）：mask-aware load balance相比request-count/token-count负载均衡将高流量下tail latency降低26%。
  - 缺陷4（FISEdit的模型支持局限）：FlashPS在SD2.1/SDXL/Flux三类模型上均验证，相比FISEdit最高加速4×。

  **关键trade-off**：FlashPS以host memory/storage容纳GiB级template activation cache换取GPU HBM容量压力降低，但引入PCIe/storage→HBM加载开销，需CUDA stream pipeline重叠隐藏。收益依赖mask区域小、模板复用高、编辑确实保持unmasked region不变；对style transfer等全局改变任务收益下降。系统增加的scheduler/cache engine/异步batch/跨进程通信开销为毫秒级。

## Automated End-to-End Model Serving with Cooperative Compilation and Scheduling

- baseline方法是什么？
  Baseline是operator-level compilation和coarse-grained scheduling的DNN serving系统。编译侧：Ansor/MetaSchedule依赖长时间搜索，Roller减少搜索空间但仍需性能模型评估，cuDNN依赖手工优化库——它们都以完整operator kernel为编译和调度基本单位，无法为推理期动态调度提供足够细粒度和多版本选择。服务侧：CUDA Stream/MPS允许并发但不能保证GPU内部空间共享；Triton是通用serving平台但调度受kernel launch/queue机制约束；Paella改善kernel执行顺序控制但仍以较粗的kernel/operator为调度对象。

  Baseline全栈执行例子（以TVM + CUDA Stream serving ResNet-50 on A100为例）：
  - 算法层：ResNet-50 DNN inference，operators按ONNX computation graph顺序执行。每个operator对应一个完整kernel，kernel整体执行或整体不执行。
  - 系统框架/Serving层：CUDA Stream并发——多个请求的不同operator kernel被放入不同CUstream，GPU scheduler按stream优先级和时间顺序发射kernel到硬件队列。kernel间存在tail effect（前序kernel结束阶段pipeline bubble导致GPU idle 1-3μs）和cold start（后续kernel preamble包括thread block dispatching/资源分配/prologue bubble）。MPS进一步允许不同进程context共享GPU，但空间共享由GPU inner scheduler控制，不可精确管理。
  - 编译框架层：TVM compiler将ONNX model转为computation graph→schedule primitives生成tensor program→codegen生成CUDA kernel。Kernel编译针对单个operator，ILP/TLP/arithmetic intensity trade-off固定在编译期确定，无法根据推理期GPU并发状态调整。
  - kernel调度层：operator kernel在CUDA stream中排队，GPU thread block scheduler (GigaThread Engine) 管理SM分配——block-fused和warp-fused colocation均需编译期预先融合（如HFuse），无法运行时动态决定。Kernel launch通过CUDA runtime API，host-device同步开销大。
  - 硬件层：NVIDIA A100 GPU (Ampere, 108 SM, 40GB HBM2e)。SM内warp scheduler以warp粒度选择ready warp执行。当单个operator kernel的instruction pattern单调时（如仅FP密集或仅memory密集），部分GPU pipeline units空闲，GPU utilization低。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Infera，通过cooperative compilation and scheduling将编译和调度从"动态耦合"改为"静态提供调度空间、运行时动态选择"。

  Infera全栈执行例子（以mixed-model BERT+ViT inference on A100为例）：
  - 算法层：与baseline相同，DNN inference，但operator被切分为tile/micro operators，下游operator的某些tile可在上游operator完整结束前开始执行（sub-operator级data asynchrony）。
  - 系统框架/Serving层：Infera inference server pipeline——JDU将job按GPU可用显存和estimated remaining time分发到目标GPU→TSU按priority（ddl_rq用EDF、rt_rq用FIFO、gcfs_rq用GCFS按nice value分配instruction budget）生成VTB→TEU三阶段执行（SelectKernels→FuseKernels→LaunchKernel）。Priority preemption：高优先级任务到达→TSU暂停调度保存状态→TEU保存HKQ/DKQ kernel context→in-flight kernel通过flag快速终止。
  - 编译框架层：Infera compiler基于TVM 0.16.0——ONNX→TVM Relay→tile-based TensorIR→为每个micro operator生成多个ILP/TLP/intensity配置的kernel candidates（register 64/96/128, shared memory 48/80/112/144 KiB, pipeline stage 2/3/4）→cut-and-patch instruction scheduling在SASS级别优化ILP→warp specialization（4 mainloop warps+4 data copy warps）→打包为CUDA binary static library。Zero-tuning：完全基于静态分析，无需GPU profiling或性能模型评估。
  - kernel调度层：TEU SelectKernels在data dependency DAG中选zero in-degree且maximize asynchronous wavefront的data blocks→online regression model估计IPC→选#inst/IPC最小且TLP≥4的kernel→FuseKernels在CUDA binary level做warp-level horizontal fusion（prologue恢复special registers+shared memory offset+barrier重组）→LaunchKernel通过HKQ→GDRCopy→DKQ→daemon kernel CDP fire-and-forget launch（<10μs, 避免HoL）→SM内不同task的warps空间共享GPU pipeline units。
  - 硬件层：NVIDIA A100 GPU。daemon kernel独占一个SM用于device-side kernel launch和DKQ管理。GDRCopy bypass DMA engines实现<100ns小数据/<5μs典型kernel传输延迟。Driver-level placeholder kernel slot覆盖避免cuModuleLoad的global host-device同步。Preemption latency：Infera-P约10μs（保存HKQ/DKQ），Infera-R约5μs（仅保存不暂停），比REEF-N快约2.5×。

  **缺陷1：Operator-based compilation无法提供sub-operator级parallelism → operator内部分tile必须等整个operator完成**
  → Infera方案：tile-based partition将大operator切分为micro operators/tiles，每个tile独立编译和调度。下游operator的某些tile在上游operator完成对该tile的输入后即可开始执行（data dependency DAG中zero in-degree节点），扩大data asynchrony。Multi-version kernel generation为每个tile提供不同ILP/TLP/intensity trade-off：register limit 64/96/128控制ILP vs TLP，shared memory limit 48/80/112/144 KiB控制tile size和intensity。编译完全并行化且zero-tuning，比Ansor/MetaSchedule编译时间低2-3个数量级。

  **缺陷2：Monotonous instruction pattern导致GPU pipeline unit利用率低（FP unit忙时memory unit空闲，反之亦然）**
  → Infera方案：multi-version kernels在编译期覆盖ILP、TLP和arithmetic intensity的多种组合，runtime SelectKernels根据当前GPU occupancy、kernel hazard和data/structure hazard估计选择最适合的kernel版本。Warp-level binary fusion将不同task的不同类型kernel（如BERT MatMul+ViT Conv）的warps在SM内空间共存，使不同instruction pattern的warps同时利用GPU的不同pipeline units，降低scoreboard和throttle stall cycles。

  **缺陷3：Baseline serving (Stream/MPS/Triton/Paella) 调度粒度粗，无法精确控制GPU内部kernel空间共享和公平性**
  → Infera方案：JDU基于GPU estimated remaining time做load-balancing dispatch；TSU 64-priority runqueue支持deadline (EDF)/real-time (FIFO)/normal (GCFS)三类调度策略，aging mechanism防止starvation；VTB + instruction budget让normal任务按nice value公平共享调度周期；TEU三阶段pipeline (SelectKernels→FuseKernels→LaunchKernel)在每个调度周期动态组织micro-kernels。在multi-model mixed serving中（uniform requests/uniform models）至少快1.6×，bursty lognormal scenarios最高快3.5×。

  **缺陷4：CUDA runtime kernel launch有head-of-line blocking和host-device同步开销 → tail effect/cold start造成GPU idle**
  → Infera方案：daemon kernel常驻一个SM，通过CUDA Dynamic Parallelism fire-and-forget launch直接在device side发射kernel，消除stream tracking overhead和HoL，launch latency <10μs。GDRCopy host-device kernel code transfer bypass DMA engines。Driver-level placeholder kernel slot覆盖避免cuModuleLoad的global同步。Kernel completion通过cudaGetLastError异步检查。

  **关键trade-off**：Infera以高系统复杂度（SASS级代码改写、driver-level placeholder slot覆盖、GDRCopy、CDP daemon kernel、device-side queue、自研调度器约17k LoC）换取更细粒度的GPU控制。daemon kernel独占一个SM带来<1/#SM的设备侧开销。host侧kernel fusion、queue管理和fused kernel缓存带来CPU和host memory开销（最大吞吐前约13% CPU和600 MiB host memory）。实现强依赖NVIDIA GPU执行模型、SASS格式和底层driver行为，跨GPU代际和跨厂商可移植性需额外验证。tile粒度也有trade-off：过细增加调度/fusion/launch元数据开销，过粗退化回operator-based scheduling。

## Latent Wavelet Diffusion for Ultra-High-Resolution Image Synthesis

- baseline方法是什么？
  Baseline是标准Latent Diffusion Models (LDMs)，如Flux-1.dev（flow-matching backbone）、SD3-F16（MMDiT backbone + 16ch VAE）、PixArt-Sigma-XL（DiT backbone）、Sana-1.6B（linear DiT）、URAE（基于Flux的参数高效适配）。这些模型在训练时对所有spatial regions使用uniform denoising supervision：L_fm = ||(ε-z_0) - v_Θ(z_t,t,y)||²₂，每个空间位置(i,j)在每一timestep接受等量优化信号。这导致两个缺陷：(1) 计算浪费——低细节平滑区域（如天空、纯色背景）被过度监督；(2) 高细节区域监督不足——纹理、边缘、毛发等高频结构得不到足够关注，在UHR（2K-4K）下产生模糊或纹理坍缩。此外，标准VAE在高分辨率下latent representation包含跨尺度不一致的高频伪影，进一步恶化UHR生成质量。

  Baseline全栈执行例子（以Flux-1.dev在A100上生成4K图像为例）：
  - 算法层：Flux flow-matching模型，给定text prompt y，VAE encoder将输入图像x映射到latent z_0∈R^{C×H×W}，前向扩散z_t = (1-t)z_0 + tε，模型预测velocity field v_Θ(z_t,t,y)，loss对所有(H,W)位置等权：每个空间位置在每个timestep接受相同的||(ε-z_0)-v_Θ||²损失信号。平滑区域和高细节区域在训练中获得完全相同频次的refinement。
  - 系统框架层：PyTorch DataParallel/FSDP训练pipeline，HuggingFace Diffusers加载预训练模型，标准训练loop：encode→sample noise→forward→uniform loss→backward→optimizer step。无空间或时间自适应调制。
  - 编译框架层：PyTorch eager mode + torch.compile（可选），使用标准CUDA kernel（FlashAttention for DiT backbone）。无额外编译优化。
  - kernel层：标准PyTorch CUDA kernel——DiT attention用FlashAttention-2，卷积/线性层用cuBLAS。所有操作以完整tensor/layer为粒度，无空间选择性计算。
  - 硬件层：NVIDIA A100 GPU (64GB HBM2e, 108 SM)。训练时GPU memory约57.9-60.1GB（取决于backbone），训练batch=1-8。推理时相同资源消耗。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Latent Wavelet Diffusion (LWD)，通过在训练阶段引入signal-derived frequency saliency来增强latent diffusion model的UHR生成质量。核心insight：图像中不同区域的结构复杂度分布不均匀，但现有denoising model对所有位置等同refine。LWD通过wavelet-derived spatial saliency maps + time-dependent masking实现频率感知的spatially adaptive supervision，两步走：(1) 用scale-consistency loss微调VAE以净化latent空间；(2) 用wavelet-masked flow matching objective微调扩散模型。

  LWD全栈执行例子（以LWD+Flux在A100上生成4K图像为例）：
  - 算法层：Stage 1——Flux-VAE经scale-consistency loss微调：L_VAE = ||D(z)-x||²₂ + α||D(E(z_down))-x_down||²₂ + β D_KL(q||p) + λ L_LPIPS。该loss惩罚跨尺度不一致的高频分量（即伪影），使latent空间的频谱分布对齐clean natural image的RGB频谱。Stage 2——对每个training step，z_t输入Haar DWT→计算HF energy E(c,i,j) = (1/C) Σ_c[(z_LH)²+(z_HL)²+(z_HH)²]→bilinear upsample+min-max normalize得Awavelet→time-dependent binary mask M_t(i,j)=1 if T·(Awavelet+l)≥t else 0→最终L_masked = ||M_t ⊙ [(ε-z_0)-v_Θ(z_t,t,y)]||²₂。高频区域（纹理/边缘/轮廓）在更多timestep收到监督信号，平滑区域在较少timestep收到监督。l=0.3确保所有区域至少30%监督覆盖。Haar wavelet因其最紧凑support（2 coefficients）提供最精确空间定位，避免Daubechies的mask边界模糊和FFT高通的Gibbs ringing伪影。
  - 系统框架层：与baseline相同——PyTorch训练pipeline，额外依赖pytorch-wavelets库（Cotter, 2019）做Haar DWT。Mask计算开销极小（Haar DWT per step），训练memory仅增~3%（Sana 90.5%→93.9%），每step时间几乎不变。完全不需要推理期修改——训练好的LWD模型权重可直接替换baseline checkpoint，推理pipeline不变。
  - 编译框架层：论文未明确说明（与baseline相同，PyTorch eager mode）。DWT通过pytorch-wavelets的CUDA kernel实现，mask生成与loss modulation为纯PyTorch tensor操作。
  - kernel层：论文未明确说明（与baseline相同，标准PyTorch CUDA kernel）。Wavelet DWT通过pytorch-wavelets库，基于标准卷积操作实现。
  - 硬件层：NVIDIA A100 GPU (4×A100, 64GB each)。训练period不变，推理period不变。LWD+URAE 4K训练约24h (batch=1)，LWD+Diff4K 2K约48h (batch=8)。推理期与baseline identical——相同参数量、相同inference time、相同GPU memory。

  **缺陷1：Uniform supervision——所有空间位置接受相同频次refinement，低细节区域浪费计算，高细节区域监督不足**
  → LWD方案：wavelet energy map Awavelet捕捉每个空间位置的局部high-frequency能量。time-dependent masking M_t(i,j)使高Awavelet区域在更多timestep参与loss计算（mask=1更长时间），低Awavelet区域仅在l·T个timestep强制参与（l=0.3）。这本质上是spatial curriculum learning——模型在训练早期关注所有region建立全局结构，后期将capacity集中在细节丰富区域。GLCM score从baseline 0.79提升到0.74（更接近真实分布），表明masking改善了而非简单增加高频纹理。

  **缺陷2：标准VAE latent空间包含跨尺度不一致的高频伪影，污染wavelet-based saliency的准确性**
  → LWD方案：scale-consistency loss在VAE微调中引入多尺度reconstruction约束——对原图x和其降采样版本x_down分别reconstruct，强制encoder在跨尺度下保持结构一致性。这抑制了"spurious high-frequency noise"（跨尺度不一致的伪影），使后续DWT提取的HF energy对应真实结构而非噪声。消融证明：仅VAE-SC (+2.5% Aesthetics） 和仅Wavelet Masking (+2.3% FID) 各自有效，但组合 (Full LWD) 效果最优（+4.1% CLIPScore, +3.5% Aesthetics），验证了两阶段协同的必要性。

  **缺陷3：UHR生成中纹理坍缩——现有方法（Diffusion-4K wave loss等）虽引入频率约束但uniform施加，不区分空间区域**
  → LWD方案：相较于Diffusion-4K将wavelet loss作为uniform空间上的passive frequency signal，LWD将频率能量转化为active spatial condition——直接在flow-matching loss上施加spatially adaptive binary masking，使模型"知道哪里需要更多学习"。在4K Aesthetic-Eval上LWD+SD3-F16在FID/CLIPScore/Aesthetics三项指标和GLCM上全面持平或超越SD3-Diff4k-F16，且不需Diff-4K的额外wave loss计算开销。LWD+URAE在4K HPD上MAN-IQA 0.4011（最高）、GLCM 0.74（最高）。

  **缺陷4：UHR训练资源消耗巨大——full UHR fine-tuning需大量GPU内存和长时间训练，训练数据稀缺**
  → LWD方案：通过加速收敛缓解——仅需baseline原始训练iteration的10-50%即达收敛（如LWD+URAE仅2k steps vs baseline建议值远超此数），大幅降低训练成本。LWD在50K 2K + 20K 4K LAION子集上即可有效训练，不依赖专有UHR数据集。训练期memory overhead仅~3%（额外存储DWT中间tensor，尺寸为latent map级别）。

  **关键trade-off**：LWD以训练时marginal overhead（每step DWT+mask计算，约3% memory）换取UHR生成质量significant improvement，且零推理开销。但继承LDM通用限制——VAE compression导致fine-grained semantic detail的丢失可能限制需要精确spatial alignment的任务（论文建议未来工作可探索更高保真度的latent space或latent+pixel联合监督）。Haar wavelet对sharp edge/discontinuity定位好，但可能对gradual texture transition不敏感。GLCM score在full LWD下轻微下降（0.79→0.74）反映perceptual realism对raw texture complexity的trade-off——论文认为这是有意义的trade-off，经FID/Aesthetics/HPS等感知指标验证。

## Spectral Regularization for Diffusion Models

- baseline方法是什么？
  Baseline是standard diffusion model training，使用pointwise signal-domain reconstruction objective（DDPM的L_DDPM = E[||ε−ε_θ(x_t,t)||²₂]，EDM的L_EDM = E[λ(σ)||ε−ε_θ(x_σ,σ)||²₂]）。这些objective均在signal domain（pixel/waveform space）以MSE/L2形式定义reconstruction error，不显式约束error在frequency bands或spatial scales间的分布。核心问题：L2 objective只控制total spectral energy of error（Parseval identity: ||x||²₂ = ||X(ω)||²₂恒成立），但对error的spectral distribution完全agnostic——small overall loss仍可对应disproportionate high-frequency errors。

  全栈执行例子（以FFHQ 64×64 unconditional + VP-EDM + A6000 GPU为例）：
  - 算法层：Standard EDM training/sampling。模型NCSN++/DDPM++接收noisy image x_σ = x₀ + σε→预测denoised ε_θ(x_σ,σ)→L2 loss对所有frequencies赋予equal weight。High-frequency components在低noise regime（σ小）才被学习，但此时effective regularization更弱、sample更少→HF errors更易overfit。结果：生成的FFHQ样本可能出现over-smoothing、incorrect frequency balance、degraded fine-scale texture。
  - 系统框架/Serving层：论文未明确说明（research training setting，非production serving）。
  - 编译框架层：论文未明确说明（标准PyTorch + CUDA训练路径）。
  - kernel调度层：论文未明确说明（标准PyTorch FFT via cuFFT backend）。
  - 硬件架构层：NVIDIA A4000/A6000 GPU，无定制硬件。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出loss-level spectral regularization框架，在standard denoising objective上augment differentiable Fourier-domain和wavelet-domain L1 penalty，不改动diffusion process、architecture或sampler。这是一种"soft inductive bias"——通过训练objective而非硬约束来引入spectral awareness。

  **缺陷1：Standard L2 denoising loss对所有frequencies赋予equal weight，无法区分低频和高频的reconstruction quality**
  → Fourier Amplitude Loss (LA_F)：在predicted clean sample x̂₀和ground-truth x₀间计算Fourier amplitude spectrum的L1差异。L1 intentionally breaks Parseval invariance——Parseval identity（||x||²₂=||X(ω)||²₂）仅对L2成立，对L1不成立，因此L1 amplitude loss可直接控制error在frequency bands间的distribution，而非仅控制total energy。Checkerboard toy experiment（64×64 grayscale, dominant high spatial frequencies）验证：baseline MSE model产生attenuated/broadened spectral responses和visible smoothing，而spectral regularizer correctly concentrates energy near correct frequency bands。

  **缺陷2：Fourier spectrum捕获global frequency structure，但缺乏spatial/temporal localization——对non-stationary signals（textured images、audio transients、edges）的local spectral structure不敏感**
  → Wavelet Coefficient Matching Loss (LW)：对Haar和bior1.3两种wavelet，在multi-scale decomposition的所有scales和orientations上计算wavelet coefficient L1 difference。Wavelet提供localized、scale-aware control：Haar强调sharp discontinuities和edge-like features，bior1.3由于higher-order vanishing moments提供smoother multi-scale consistency。Audio实验验证：Haar wavelets achieve lowest MR-STFT distance（improved multi-resolution temporal coherence），而不同wavelet type对应不同的sharpness-vs-smoothness trade-off。

  **缺陷3：Phase information对perceptual quality critical，但直接使用phase difference会导致training instability（branch-cuts、low-amplitude band noise）**
  → Amplitude-Phase Coupled Loss (LAP_F)：phase penalty通过amplitude weighting引入：LAP_F = E[||A₀−Â₀||₁·(1+||ϕ₀−ϕ̂₀||₁)]。关键设计：large phase discrepancies in bands with vanishing amplitude are perceptually insignificant（被amplitude因子抑制），而similar discrepancies in dominant bands correspond to coherent structural distortions（被amplitude因子放大）。Audio实验验证：Amplitude-phase loss produces most balanced gains——达highest UTMOS和PESQ、lowest NDB（mode coverage best），说明phase coupling稳定fine-scale structure且不引入instability。

  **缺陷4：修改diffusion process或architecture的spectral方法（如wavelet-domain diffusion、frequency-based noise control）需要specialized implementation，与现有DDPM/DDIM/EDM pipeline不兼容**
  → Loss-level-only design：所有spectral regularization仅在training objective层面添加auxiliary term，不需modified forward process、basis-specific parameterization或architecture change。与DDPM/DDIM/EDM fully compatible。Image实验仅需5 optimization steps fine-tuning（非full retraining），证明spectral bias是data-efficient的——在strong pretrained EDM baseline上仍获0.02-0.07 FID improvement，尤其在高分辨率unconditional setting（FFHQ/AFHQ）收益最大。

  **缺陷5：Time-domain waveform loss in audio diffusion难以capture perceptually important spectral structure**
  → Audio实验：DiffWave在LJSpeech上fine-tune 150K steps with spectral losses。Fourier amplitude regularization yields strongest FAD improvement（1.994→1.462 at λ=10⁻⁴），证明matching global magnitude statistics足以恢复驱动perceptual distance的dominant spectral structure。所有spectral losses在FAD和PESQ上均outperform DiffWave baseline，指示explicit spectral-domain biasing有效correct weakly-constrained spectral mismatches。

  论文方法全栈执行例子（以FFHQ 64×64 unconditional + VP-EDM + Fourier amplitude loss fine-tuning + A6000为例）：
  - 算法层：加载pretrained VP-EDM checkpoint→仅5 step fine-tuning。每step：采样timestep t→forward process得x_t→DDIM一步得x̂₀→对x̂₀和x₀做PyTorch FFT→计算amplitude L1 loss→总loss L_total = L_EDM + λ·LAF→backprop update。采样阶段与standard EDM完全相同，无extra compute。
  - 系统框架/Serving层：论文未明确说明（属于training-time modification, sampling time unchanged）。
  - 编译框架层：论文未明确说明（标准PyTorch FFT + PyWavelets库，training-time auxiliary loss仅增加可忽略的computational overhead）。
  - kernel调度层：论文未明确说明（PyTorch FFT底层使用cuFFT）。
  - 硬件架构层：NVIDIA A4000/A6000 GPU，无定制硬件。论文强调compute overhead "negligible"——FFT和DWT在GPU上高度优化。

## ELF: Embedded Language Flows

- baseline方法是什么？
  Baseline是现有扩散语言模型（DLMs），分为两类：(1) Discrete DLMs（MDLM、Duo等）直接在离散token space定义扩散过程，使用masked/uniform transition matrices，需要categorical reparameterization，CFG难以有效应用；(2) Continuous DLMs（Diffusion-LM、CDCD、FLM、LangFlow等）将token映射为continuous representation但每步均施加token-level cross-entropy supervision（per-step discretization），导致denoising trajectory被耦合到vocabulary-level prediction，限制了flow dynamics的灵活性。Latent Diffusion LMs（LD4LG等）虽然避免per-step vocabulary supervision，但依赖DDPM-style formulation + 单独训练的decoder。

  全栈执行例子（以MDLM/FLM + OWT unconditional generation为例）：
  - 算法层：MDLM使用absorbing-state masking in discrete token space，128/256/1024 steps of iterative unmasking，无法使用CFG。FLM在embedding space做Flow Matching但每步施加cross-entropy loss，将embedding projection回token space做监督，flow trajectory受token-level constraint限制。
  - 系统框架/Serving层：论文未明确说明（所有方法均为research-stage DLM training，未涉及serving deployment）。
  - 编译框架层：论文未明确说明（PyTorch/JAX standard training loop，无定制compilation）。
  - kernel调度层：论文未明确说明（TPU/GPU standard matmul and attention kernels）。
  - 硬件架构层：Google TPU v5p（训练），NVIDIA GPU（baseline推断），无定制硬件。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出ELF，通过以下关键设计解决baseline缺陷：

  **缺陷1：Discrete DLMs在离散token space定义扩散，需categorical transitions，难以apply CFG**
  → ELF将language generation完整体迁移到连续空间：tokens→T5 encoder→continuous embeddings→Flow Matching denoising全程在continuous embedding space→仅在t=1 final step做discretization。这使得CFG（原为continuous quantities设计）可自然应用。ELFB with CFG scale=3 in 32步达到Gen.PPL 24 vs MDLM 1024步 27、Duo 1024步 34。

  **缺陷2：现有continuous DLMs每步施加per-step token-level cross-entropy supervision，将denoising trajectory耦合到vocabulary prediction**
  → ELF在除最后一步外的所有steps使用pure MSE loss in embedding space（L_MSE = ‖(x̂−x)/(1−t)‖²），不做per-step discretization。仅在t=1 final step使用CE loss via shared-weight unembedding。Denoising mode概率0.8 vs decoding mode 0.2，确保主trajectory在continuous space自由演化。

  **缺陷3：Latent Diffusion LMs需要单独训练decoder（autoregressive/non-autoregressive），增加inference component和训练复杂度**
  → ELF使用shared-weight network：同一网络在所有t<1执行denoising（x-prediction+MSE loss），在t=1执行decoding（x-prediction+unembedding+CE loss）。通过binary "mode" token区分。无需单独decoder，减少参数量和训练stage。

  **缺陷4：DDPM-style discrete-time formulation限制sampling flexibility**
  → ELF使用continuous-time Flow Matching（rectified flow），支持ODE和SDE sampler灵活切换。Logit-normal time schedule在few-step regime显著优于uniform schedule。SDE sampler在32步即可达到1024步ODE的质量（Gen.PPL 24 vs 26.6），data efficiency提升10×（45B vs 524B tokens）。

  **缺陷5：x-prediction在high-dim embedding space中比v-prediction/ϵ-prediction更稳定但未被充分exploit for language**
  → ELF证明x-prediction在512/768/1024-dim embedding space中唯一保持稳定（v-prediction在高维退化，ϵ-prediction collapse）。x-prediction使shared-weight design成为可能（denoising和decoding均predict clean embeddings）。

  论文方法全栈执行例子（以ELF-B + OWT + 32-step SDE + TPU v5p×64为例）：
  - 算法层：T5-small encoder frozen→bottleneck 512→128→hidden 768→DiT with SwiGLU/RMSNorm/RoPE/qk-norm→Flow Matching x-prediction (80% MSE) + shared-weight decoding (20% CE)→training-time CFG (ω∈[0.5,5]) with self-conditioning→SDE sampler with γ=1.5, CFG scale=3→32步Gen.PPL 24.08, Entropy 5.15。
  - 系统框架/Serving层：论文未明确说明（research-stage，无serving deployment）。推断时支持batch generation。
  - 编译框架层：论文未明确说明（标准PyTorch/JAX training loop on TPU）。
  - kernel调度层：论文未明确说明（TPU v5p standard matrix multiply and attention kernels）。
  - 硬件架构层：Google TPU v5p × 64，1.5h/epoch，5 epochs total，45.2B effective training tokens。无定制硬件。
