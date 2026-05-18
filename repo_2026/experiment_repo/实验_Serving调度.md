## The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective

- 属于Serving调度的实现是什么？实验比较什么？
  论文实现了一个AI agent serving系统（基于vLLM 0.6.6 + PyTorch 2.6 + CUDA 12.8），系统性表征agent workload的latency、GPU utilization、KV cache memory、throughput、energy consumption和datacenter-wide power demand。核心Serving调度实现包含：(1) Agent Worker进程：每个用户请求进入server entrypoint后启动agent worker，worker根据agent类型（CoT/ReAct/Reflexion/LATS/LLMCompiler）循环执行LLM inference和tool call，LLM请求发送到vLLM backend、tool可能是Wikipedia API/WebShop navigation/Wolfram Alpha API/Python code execution等；(2) vLLM backend采用默认FCFS scheduler + continuous batching，开启prefix caching，多个worker的LLM请求在vLLM后端通过continuous batching合并执行；(3) 请求流量按Poisson arrival distribution模拟。实验比较：(a) single-request层面：LLM/tool invocation count、end-to-end latency breakdown、GPU idle time（因tool等待导致最多54.5% GPU idle）、端到端延迟分布；(b) serving层面：throughput (QPS)、95th percentile tail latency (ShareGPT可达6.4 QPS、ReAct HotpotQA/WebShop仅2.6/1.2 QPS)、prefix caching对throughput的提升（agentic workload平均5.62× vs ShareGPT仅1.03×）、KV cache memory usage（prefix caching使平均/最大KV cache memory分别降低51.7%/63.5%）；(c) test-time scaling：iteration budget、few-shot example数、sequential vs parallel scaling、model size（8B vs 70B）对accuracy-latency trade-off的影响；(d) 能耗估算：GPU energy/query（单次agent query能耗比ShareGPT高62.1×-136.5×）和datacenter-wide power projection（70B Reflexion在当前流量下接近1.0 GW）。baseline对比对象：(1) conventional chatbot (ShareGPT) 作为非agent baseline；(2) 五种agent类型之间相互对比（CoT/ReAct/Reflexion/LATS/LLMCompiler）。

- 硬件平台是什么，配置是什么。
  8B实验：GCP a2-highgpu-1g实例（12 vCPUs、85GB memory、单张NVIDIA A100 40GB GPU）；70B实验：GCP a2-highgpu-8g实例（96 vCPUs、680GB memory、8张NVIDIA A100 40GB GPU）。GPU utilization用NVIDIA DCGM测量。

- 开源Serving框架是什么。修改了什么。
  基于vLLM 0.6.6（OpenAI-compatible API模式）。论文未修改vLLM源码，而是在vLLM之上构建了agent serving layer：(1) Agent Worker：每个worker根据agent workflow决定是调用vLLM backend做LLM inference还是执行tool；(2) Server Entrypoint：接收用户请求、路由到agent worker pool；(3) tool system：本地code interpreter、Wikipedia API、Wolfram Alpha API、WebShop navigation等外部工具接口。论文指出vLLM作为LLM backend保持默认FCFS scheduler不变，agent workflow的实现基于各agent原作者的开源实现（ReAct github.com/ysymyth/ReAct、Reflexion github.com/noahshinn/reflexion、LATS、LLMCompiler github.com/SqueezeAILab/LLMCompiler），并适配到统一评测框架。对于LATS，论文进一步优化其实现以支持concurrent LLM inference和parallel tool invocation（原版本顺序执行，加重端到端延迟）。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源地址：https://github.com/VIA-Research/AgentBench。AgentBench包含论文使用的AI agent implementations和benchmarking utilities，支持ReAct、Reflexion、LATS、LLMCompiler的运行配置。
  以HotpotQA + ReAct agent serving为例：
  1. 部署：启动vLLM server（Llama-3.1-8B-Instruct, prefix caching enabled）→启动Agent Server entrypoint（配置ReAct worker pool、Wikipedia API tool interface）。
  2. 请求到达：用户query "Which film came out first, A Separation or The Salesman?" 进入server entrypoint→启动ReAct worker。
  3. ReAct worker构造prompt：instruction ("Solve a question answering task...") + few-shot examples + user query。第一次LLM call发送到vLLM backend → LLM生成 thought/action观察→action是Wikipedia search("A Separation film")。
  4. Tool execution：worker调用Wikipedia API（平均1.2s latency）→返回"A Separation is a 2011 Iranian drama film..."。
  5. 工具返回追加到context → 第二次LLM call（prompt现在包含之前的thought+observation作为LLM history和Tool history tokens）→vLLM prefix caching复用与前次shared prefix的KV cache（prefill latency降低60.1%）→LLM继续生成thought/action或final answer。
  6. 多请求并发：多个ReAct worker异步运行→worker#A在等Wikipedia API时GPU空闲→worker#B的LLM call通过continuous batching填补idle gap→GPU utilization提升。
  7. Serving层面：HotpotQA ReAct concurrent execution达成2.6 QPS（vs sequential的0.10 QPS即25×提升），95th percentile tail latency为20.7s（vs ShareGPT的9.7s）。
  8. 能耗：单个ReAct Reflexion请求GPU energy 41.53 Wh（8B），是ShareGPT 0.32 Wh的130.9×。

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

## MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models

- 属于Serving调度的实现是什么？实验比较什么？
  提出MoDM，一个基于final image缓存+混合扩散模型(mixture-of-diffusion models)的自适应serving系统。核心实现包含：(1) Request Scheduler：管理请求流，用CLIP text embedding做text-to-image similarity检索缓存图像，根据相似度动态选择跳过步数k∈{5,10,15,20,25,30}，cache-hit请求发往小模型refine、cache-miss请求发往大模型全量推理；(2) Global Monitor：基于PID控制器(Kp=0.6, Ki=0.05, Kd=0.05)动态分配GPU资源到大/小模型worker，支持Quality-Optimized Mode（最大化大模型数量受SLO约束）和Throughput-Optimized Mode（所有cache-hit用小模型、cache-miss用大模型）；(3) FIFO-based Cache Maintenance：滑动窗口缓存策略，>90% cache-hit请求检索到4小时内生成图像；(4) Cross-model compatibility：缓存final image而非latent，使同一cache跨Stable Diffusion/SANA等多模型族复用。实验比较throughput、tail latency (P99)、SLO compliance (2×/4× large model latency)、最大负载、energy consumption，对比Vanilla (仅大模型)、Nirvana (latent caching)、Pinecone (检索无refine)。quality指标用CLIPScore、FID、IS、PickScore。

- 硬件平台是什么，配置是什么。
  两套配置：(1) 单节点4×NVIDIA A40 GPU (48GB memory)；(2) 16节点集群，每节点4×AMD MI210 GPU (64GB memory)。软件环境：Python + PyTorch，PyTorch RPC做节点间通信。

- 开源Serving框架是什么。修改了什么。
  MoDM从零实现Python serving系统（非基于已有开源Serving框架修改），Request Scheduler、Global Monitor和每个worker运行在独立进程中，通过PyTorch RPC通信。核心新增：(1) CLIP-based text-to-image相似度计算与检索模块；(2) 基于相似度的k-selection heuristic (Fig.5b)；(3) PID controller resource allocation算法 (Algorithm 1)；(4) FIFO cache管理策略；(5) 图像noise re-introduction + small model refinement pipeline (公式2)。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源地址：https://github.com/stsxxx/MoDM。以SD3.5L (8B)为large model、SDXL (3B)为small model的serving为例：
  1. 部署：启动Request Scheduler进程（管理CLIP embedding提取和cache lookup）+ Global Monitor进程（PID controller资源分配）+ N个GPU Worker进程（各加载一个模型变体，可动态切换大/小模型）
  2. 请求到达：Scheduler提取prompt的CLIP text embedding→与cache中图像embedding计算cosine similarity→若sim≥threshold（如0.25-0.30），确定跳过步数k（例如sim≥0.3则k=30）→Retrieve cached image→加噪到timestep t_k（公式2: Ĩ=σ_tk·ε+(1-σ_tk)·I*）→发到cache-hit queue用小模型refine T-k步
  3. 若sim<threshold→cache miss→发到大模型全量T=50步推理
  4. Global Monitor每周期统计request rate R、cache hit rate H_cache、k分布→计算miss_workload和hit_workload→PID调整N_large和N_small分配
  5. 新生成图像存入FIFO cache替换最旧图像
  6. 在DiffusionDB workload下MoDM-SANA达到3.2× throughput提升，46.7% energy savings，P99 tail latency在4×A40上支持10 req/min不violate SLO

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

## Adaptive Draft Sequence Length: Enhancing Speculative Decoding Throughput on PIM-Enabled Systems

- 属于Serving调度的实现是什么？实验比较什么？
  提出SADDLE，面向PIM+GPU异构系统的adaptive draft sequence length speculative decoding系统，核心Serving调度实现包含：(1) 运行时自适应draft length调整：Controller在DLM每生成一个draft token时读取采样概率p_t，维护累计接受概率H_t=∏p_i，当H_t低于离线校准的阈值τ时停止该请求drafting，避免生成低置信度token被TLM拒绝后浪费计算；(2) Shared Pool跨micro-batch聚合draft tokens：各micro-batch的Draft Generator将draft tokens存入Shared Pool (1KB CAM)，当token数达GPU verification capacity C或GPU空闲时触发TLM并行验证，避免micro-batch内因最长draft请求同步等待；(3) Eager Pool乐观执行：TLM正在验证Shared Pool中token时，DLM基于"当前token将被接受"的乐观假设继续生成后续draft tokens暂存到Eager Pool (1KB，按micro-batch划分)，验证通过后迁入Shared Pool否则丢弃并以TLM修正token重新开始；(4) prediction-verification解耦异步pipeline：DLM prediction与TLM verification并行执行，通过Shared Pool/Eager Pool token migration消除同步气泡。实验比较throughput、energy efficiency、latency breakdown、communication cost、GPU/PIM utilization、area overhead，对比GPU-AD (autoregressive on GPU)、GPU-SD (speculative on GPU)、PIM-AD (PIM attention+GPU FC autoregressive)、PIM-SD (SpecPIM类PIM-enabled speculative decoding)。在OPT-66B+OPT-1.3B、Llama3.1-70B+Llama3.2-1B、OPT-175B+OPT-6.7B三组TLM/DLM组合上，SADDLE相比GPU-AD/GPU-SD/PIM-AD/PIM-SD平均吞吐分别提升3.36×/2.88×/1.94×/1.71×。消融实验：(Ssaddle-d)仅自适应draft length反而比PIM-SD低1.22×；(Ssaddle-p)+Shared Pool比Ssaddle-d高1.52×；(Ssaddle-s)+Eager Pool+动态operator mapping进一步提升1.24×和1.13×。异步pipeline实现端到端延迟降低1.73×（与PIM-SD对比），monitoring/decision-making仅占0.83% latency。

- 硬件平台是什么，配置是什么。
  SADDLE系统含8个SADDLE PIM devices，每device配1个NVIDIA A100 GPU (centralized processor) + 5个HBM3 stacks (每stack 16GB, 5.2Gbps/pin)，总GPU显存640GB、总HBM 640GB，A100 DGX聚合带宽16TB/s，PIM内部带宽144TB/s (9× DGX)。GPU baselines在8×A100 DGX上评估(DeepSpeed Inference)。PIM baselines用相同数量GPU和40 HBM stacks (各16GB)。SADDLE Manager配1KB Shared Pool (CAM)、1KB Eager Pool (每micro-batch最多512 tokens)、1KB SRAM (存logits和累计接受概率)。Controller集成softmax unit、multipliers和comparators。HBM3 PIM chip每bank附1个PE (16 FP16 multipliers + 16 FP16 adders)，buffer die上集成SFU(softmax/layer norm/activation)。互联：NVLink或CXL。

- 开源Serving框架是什么。修改了什么。
  GPU baselines基于DeepSpeed Inference实现。PIM baselines基于AttAcc风格HBM-PIM架构，PIM-SD采用SpecPIM的离线静态operator mapping（基于初始batch size和max sequence length的design-space exploration）。SADDLE自身通过cycle-accurate simulator (修改Ramulator2 + ATTACC) 评估，非基于开源Serving框架修改。SADDLE核心新增：(1) Draft Generator per micro-batch (Controller + Eager Pool)；(2) Shared Pool跨micro-batch聚合验证；(3) Controller的H_t累积概率计算和阈值比较hardware module；(4) Scheduler的arithmetic intensity估算和动态remapping逻辑。模型权重和KV cache按pipeline parallelism分配到S组PIM devices，batch切成> S个micro-batches占满pipeline。KV cache mapping：每attention head分配一个HBM stack，K^T column-wise partitioning across BGs + row-wise across banks，V row-wise across BGs + column-wise across banks。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源：论文未提供SADDLE代码或模拟器开源链接（HPCA 2026）。引用Dolly dataset开源仓库 https://github.com/databrickslabs/dolly 仅用作评测数据集。SADDLE speculative decoding scheduling使用流程：
  1. 离线阶段：用验证集校准阈值τ——对每个请求运行完整prediction-verification pipeline，记录每draft step j的H_j和验证结果，估算H_j上的条件成功率曲线，选20%区间内平均draft length最高且≥90%验证成功率的τ
  2. 推理启动：batch切成micro-batches，每个micro-batch分配Draft Generator。DLM逐token生成draft→Controller读取p_t更新H_t→若H_t<τ则停止该请求drafting→draft token存入Shared Pool
  3. 当Shared Pool token数≥GPU capacity C(=512)或GPU空闲时→TLM并行验证所有Shared Pool tokens（跨micro-batch聚合）。同时DLM对H_t仍高于τ的请求继续生成新token→暂存Eager Pool
  4. TLM验证返回：若请求所有旧draft tokens被接受→Eager Pool中该请求新tokens迁入Shared Pool→进入下一轮verification。若有token被拒绝→该请求Eager Pool tokens全部丢弃→用TLM修正token重开drafting
  5. Scheduler在prediction后根据活跃请求数估算DLM FC算术强度→决定GPU或PIM执行；在verification前根据Shared Pool每请求token数估算TLM attention算术强度→决定GPU或PIM执行
  6. 以OPT-66B+OPT-1.3B, Dolly, BS=64为例：SADDLE自适应draft length+异步pipeline相比PIM-SD固定d=8时吞吐提升1.71×，有效避免了固定draft length在batch增大时因大量draft token被拒绝导致的吞吐下降

## Towards Resource-Efficient Serverless LLM Inference with SLINFER

- 属于Serving调度的实现是什么？实验比较什么？
  提出SLINFER，一个面向small- to mid-sized LLM的resource-efficient serverless inference方案，实现异构硬件（CPU+GPU）上的弹性、按需资源共享。核心实现包含三大子系统：(1) Headroom-Driven Compute Subsystem：基于request headroom (公式1: headroom=ST+TTFTSLO+TPOTSLO·O-CT) 的token-level调度，每次调度cycle选最短headroom的instance执行一个iteration，通过shadow validation虚拟添加新请求并模拟future compute流程来避免SLO violation，使用linear interpolation做prefill time quantification、2D linear interpolation (batch size×token length) 做decode time quantification；(2) Hazard-Aware Memory Subsystem：watermark-based KV-cache scaling（early scale-up + lazy scale-down，watermark w=25%），optimistic budgeting + pessimistic scheduling的inter-instance orchestration机制避免OOM，intra-instance scaling compromise当full scale-up不可行时降级为Mrequire而非Mrecommend，极端memory不足时evict最长headroom请求；(3) Efficiency-Oriented Consolidator：proactive consolidation with preemption（允许大batch instance preempt小batch邻居来scale-up）+ reactive consolidation with bin-packing（新请求优先路由到大batch instance，加速小batch instance回收）。实验在4×32-core Intel Xeon 6462C CPU + 4×A100-80GB GPU上，使用Llama-3.2-3B/Llama-2-7B/Llama-2-13B模型，Azure LLM Conversation Trace + Azure Serverless Trace workload，SLO为TTFT ≤ min(max(0.5, L/512), 8)s、TPOT ≤ 0.25s。对比baselines：ServerlessLLM (sllm, GPU-only)、sllm+c (加CPU support)、sllm+c+s (加CPU+time-sharing)。主要指标：Nodes Used、Decode Speed、SLO-met Req、TTFT CDF、memory utilization CDF、batch size CDF。SLINFER在serving 128 models时SLO-met requests比sllm提升86%-154%，比sllm+c提升47%-62%，比sllm+c+s提升18%-70%。

- 硬件平台是什么，配置是什么。
  4×32-core Intel Xeon 6462C @3.3 GHz CPU（第4代Xeon，支持AMX）+ 4×NVIDIA A100-80GB GPU，逻辑上分离为两个物理节点各2 GPU。CPU用于独立inference（通过OpenVINO 2024.6.0）或辅助GPU offloading。Ubuntu 22.04, CUDA 12.4, Conda。

- 开源Serving框架是什么。修改了什么。
  基于ServerlessLLM [26] (OSDI'24) 和 vLLM 0.5.2 [37] (SOSP'23) 修改。主要修改：(1) 新增Headroom-Driven Compute Subsystem：实现token-level调度器替代vLLM默认continuous batching，每cycle选最短headroom instance执行一个iteration，shadow validation仿真prefill/decode timeline检查SLO violation风险；(2) 新增Hazard-Aware Memory Subsystem：在paged-attention基础上增加watermark-based KV-cache scaling逻辑、optimistic budget + pessimistic reservation station机制协调多instance并发memory操作；(3) 新增Efficiency-Oriented Consolidator：preemption-based proactive consolidation和bin-packing-based reactive consolidation；(4) CPU backend集成OpenVINO 2024.6.0替代vLLM GPU backend用于CPU inference；(5) 统一硬件抽象层：将CPU/GPU节点统一为resource pool，proxy层路由请求优先到CPU instance，CPU无法满足SLO时fallback到GPU。源码约需200GB磁盘空间，完整实验约26小时。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  已开源：GitHub https://github.com/BarrinXu/SLINFER，Zenodo DOI: 10.5281/zenodo.17846442。使用流程（以3B模型实验为例）：
  1. 环境准备：每台GPU机器安装Conda环境SLINFER-GPU (Python 3.11)，安装ServerlessLLM model loader (`sllm-store`)、modified vLLM (`pip install -e .`)、transformers 4.46.3等依赖。每台CPU机器安装Conda环境SLINFER-CPU，安装OpenVINO版modified vLLM。
  2. 模型下载：从HuggingFace下载Llama-3.2-3B-Instruct/Llama-2-7b-chat-hf/Llama-2-13b-chat-hf到`$PROJECT_BASE/huggingface_models/`。
  3. 启动系统：GPU机器上启用NVIDIA MPS (`nvidia-cuda-mps-control -d`)，启动4个GPU instance wrapper（每GPU一个，如`python vllm_batch_starter.py --model llama-3.2-3b --device gpu --worker_num 8 --port 8000 --gpu 0`），启动ServerlessLLM model loader (`sllm-store-server`)，启动root gateway (`python gateway.py`)。CPU机器上各启动CPU instance wrapper (`--device aliyun --cpu_kv_gb 16`) 和dist gateway。
  4. 运行实验：`python test_3B_extreme_lite.py` (26分钟快速测试) 或 `python test_3B_full.py` (396分钟完整测试)，生成JSON结果后用`python draw.py`生成PDF图表。
  5. SLINFER作用：event-driven请求到达时，compute subsystem通过shadow validation选择合适instance（优先CPU），headroom-based token-level调度在instance间轮转执行iteration；memory subsystem在请求加入/完成时动态scale KV-cache（early scale-up预留watermark空间，lazy scale-down减少ping-pong）；当instance无法scale-up容纳新请求时，consolidator尝试preempt小邻居或创建新instance并bin-packing路由后续请求。

- 属于Serving调度的实现是什么？实验比较什么？
  提出TetriServe，一个面向DiT (Diffusion Transformer) serving的deadline-aware round-based调度系统。核心实现包含：(1) Deadline-Aware Round-Based Scheduler：将连续时间离散化为固定时长round，每个round内通过offline profiling的cost model确定每个请求在最小GPU hour消耗下满足deadline所需的最少GPU数量（step-level sequence parallelism），然后用动态规划(DP)进行request packing以最小化下一round将超时的请求数；(2) GPU Placement Preservation：跨round保持请求在同一GPU集合上执行，消除状态迁移延迟；(3) Work-Conserving Elastic Scale-Up：利用placement后空闲的GPU给有余量steps的请求增加并行度；(4) Selective Continuous Batching：仅对相同小分辨率请求进行step级batching，不牺牲SLO；(5) VAE Decoder Sequential Execution：顺序执行VAE decoder以避免高分辨率下大batch的activation memory峰值。实验比较SLO Attainment Ratio (SAR)、end-to-end latency CDF、平均并行度、对arrival rate/step granularity/resolution的sensitivity，对比baselines：xDiT (SP=1/2/4/8固定并行度)和RSSP (Resolution-Specific SP, 对每种分辨率选最优固定SP)。

- 硬件平台是什么，配置是什么。
  两个GPU集群：（1）8×NVIDIA H100-80GB HBM3, NVLink 4.0 (900 GB/s inter-GPU bandwidth)；（2）4×NVIDIA A40-48GB, GPU对之间NVLink连接, host通过PCIe 4.0连接。软件环境：NVIDIA NGC container, CUDA 12.5, NCCL 2.22.3, PyTorch 2.4.0。

- 开源Serving框架是什么。修改了什么。
  基于xDiT (git-hash 8f4b9d30) 的sequence parallelism engine，复用vLLM的async logic和MuxServe/SGLang的process launcher。TetriServe在xDiT之上新增5,033行Python和C++代码实现：C++编写的scheduler核心决策循环（达到毫秒级控制面延迟）、Round-based DP调度算法、Future-like latent transfer抽象（异步非阻塞传输中间latent）、NCCL communication process groups预warmup策略、以及selective continuous batching逻辑。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源地址：https://github.com/DiT-Serving/TetriServe。以FLUX.1-dev serving为例：300个prompt从DiffusionDB采样，以Poisson过程按12 req/min到达。请求包含四种分辨率(256×256, 512×512, 1024×1024, 2048×2048)和对应SLO deadline（1.5s/2.0s/3.0s/5.0s）。TetriServe在8×H100上运行：Scheduler在每个round开始时，根据offline profiled的cost model查表获取每种(分辨率, GPU数)组合的单步耗时，为每个请求计算满足deadline的最小GPU分配，再通过DP pack requests入round。Execution Engine的8个GPU worker执行分配的diffusion steps，Latent Manager处理中间latent。请求完成后VAE decoder顺序执行解码。对比xDiT固定SP=1/2/4/8：在Uniform workload下TetriServe SAR平均高出10%（tight SLO 1.1×时高出28%），Skewed workload下平均高出15%（1.2× SLO时高出32%）。

- 属于Serving调度的实现是什么？实验比较什么？
  提出Bullet，一个面向LLM serving的intra-GPU prefill-decode spatial-temporal orchestration系统。核心实现包含四个模块：(1) Performance estimator：基于SM-scaling roofline model (SRM) 建模compute/memory/network bandwidth随SM数量变化的性能上界，用少量offline concurrent sample校准prefill/decode interference，在线统计持续修正；(2) SLO-aware task scheduler：周期性读取全局状态，预测TTFT/TPOT，重排waiting requests，为下一批prefill layer或decode step搜索合适SM分区；(3) Resource manager：通过libsmctrl_set_stream_mask修改CUDA stream metadata，使后续kernel限制在指定SM子集执行；(4) Concurrent execution engine：prefill和decode放在独立进程/worker中，共享CPU metadata buffer、统一GPU memory pool (CUDA IPC)、ZeroMQ metadata传递来协同执行。prefill以layer为粒度运行并同步回CPU决策，decode使用CUDA Graph发射一个step的小kernel。实验比较TTFT、TPOT、throughput、P90 SLO attainment、normalized input/generation latency以及Nsight Systems采集的SM/Tensor Core/memory bandwidth utilization，对比vLLM v0.8.5 (1024 chunk)、SGLang v0.4.6+FlashInfer v0.2.7 (1024/2048 chunk)、NanoFlow (1024 chunk)和基于SGLang+Mooncake的xP-yD disaggregated-prefill配置。

- 硬件平台是什么，配置是什么。
  三类服务器：(1) 8×A100-80GB GPU（108 SM/GPU，NVLink 600 GB/s）；(2) 8×H100 GPU（132 SM/GPU，600 GB/s）；(3) 8×H20 GPU（78 SM/GPU，intra-node 400 GB/s，inter-node 200 GB/s）。CUDA 12.4。Artifact Appendix的复现实验聚焦单张NVIDIA A100 80GB、Debian 5.10、CUDA 12.4、Python 3.12.9、PyTorch 2.6.0、CMake 3.17、GCC 10.2.1、约20GB磁盘。

- 开源Serving框架是什么。修改了什么。
  基于SGLang v0.4.6 + PyTorch 2.6.0修改，约4100行Python代码，集成修改版libsmctrl做GPU resource allocation。主要修改：(1) prefill/decode engine拆为两个SGLang worker进程，MPS用于spatial sharing；(2) GPU memory由初始化进程预先分配模型权重和KV cache，通过CUDA IPC在两个engine间共享；(3) CPU侧用OS-managed shared memory保存全局状态与metadata；(4) 请求metadata用ZeroMQ异步传递；(5) scheduler下发repartition command后，resource manager立即修改相应CUDA stream的SM mask；(6) prefill以若干layer为粒度运行并同步回CPU决策，decode使用CUDA Graph发射step。对比的disaggregated baseline基于SGLang+Mooncake构建xP-yD配置（如1P1D、3P1D、6P1D、30P6D）。开源地址：https://github.com/zejia-lin/BulletServe，Zenodo DOI: https://doi.org/10.5281/zenodo.17937105。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  已开源，GitHub仓库（zejia-lin/BulletServe）为研究原型。使用步骤：
  1. 环境准备：Debian 5.10，CUDA 12.4，Python 3.12.9，PyTorch 2.6.0，CMake 3.17，GCC 10.2.1
  2. 克隆仓库：`git clone https://github.com/zejia-lin/BulletServe`
  3. 构建libsmctrl等依赖
  4. 启动MPS服务：`nvidia-cuda-mps-control -d`
  5. 启动Bullet engine：通过`--enable-bullet-engine`和MPS启动参数运行
  6. 复现实验：运行`artifact_evaluation/run_all.sh`生成日志、JSON结果和图
  Bullet通过在同一GPU内将prefill和decode分配给不同SM子集并发执行，从机制上减少chunked prefill的重复KV reload和lock-step等待。例如在A100+Llama3.1-8B上，ShareGPT 20 req/s下mean TTFT 0.16s（相比SGLang-1024好54.9x），SLO compliance提升1.49x；SM active cycles 86.2%（比SGLang高11.2%），Tensor Core utilization高11.8%，memory-bandwidth utilization高19.3%。

## Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

- 属于Serving调度的实现是什么？实验比较什么？
  MuxWise提出intra-GPU prefill-decode (PD) multiplexing，在同一GPU内的不同SM上空间复用prefill和decode阶段。实现包含三个模块：(1) bubble-less multiplex engine：将prefill按layer切分执行，通过query-based同步消除GPU气泡；(2) contention-tolerant estimator：利用solo-run predictor + contention guard提供worst-case延时估计以保障SLO；(3) SLO-aware dispatcher：为decode分配best-fit SMs来满足TBT SLO，剩余SMs分配给prefill最大化goodput。实验比较goodput（SLO约束下的峰值吞吐）、99%-ile TTFT和TBT，对比chunked-prefill (SGLang)、NanoFlow、LoongServe和SGLang-PD四种baseline。

- 硬件平台是什么，配置是什么。
  主测试平台：8×A100-80GB GPU（NVLink 600 GB/s）。附加测试：8×H100-SMX5-80GB GPU，8×H200-SMX5-141GB GPU。NVIDIA driver 570.124.06，CUDA 12.8。单GPU测试也评估了Llama-8B在单块A100上的表现。

- 开源Serving框架是什么。修改了什么。
  基于SGLang v0.4.10post2修改，PyTorch 2.6.0。主要修改：(1) 集成GreenContext实现intra-process SM空间分区，支持运行时低开销（微秒级）重配分partition ratio；(2) 将prefill阶段改为layer-wise执行（piecewise CUDA graph），每层可独立调度；(3) 实现query-based同步机制：定期轮询CUDA events，异步发射decode batch和prefill layer，event完成后立即合并；(4) 添加contention guard离线profiling和在线estimator模块；(5) SLO-aware dispatcher根据worst-case estimation动态选择multiplexing plan（SM配置和prefill layer数量）。开源地址：https://github.com/ykcombat/sglang.git (branch: slo_config)，Zenodo: https://zenodo.org/records/18062118。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  已开源，代码在GitHub（ykcombat/sglang, slo_config分支），提供Docker镜像。使用步骤：
  1. 克隆仓库并切换到slo_config分支：`git clone https://github.com/ykcombat/sglang.git && cd sglang && git checkout slo_config`
  2. 构建SGLang：`pip install -e "python"`
  3. 启动MuxWise服务：`./start_pdmux.sh`（启动带PD multiplexing的SGLang server）
  4. 启动chunked-prefill对比：`./start_chunk.sh`（通过设置`$CHUNK_SIZE`环境变量控制token budget）
  5. 运行benchmark：`./bench_pdmux.sh`（测试ShareGPT和LooGLE workload）
  MuxWise通过intra-GPU PD multiplexing，将decode阶段的SMs保留用于满足TBT SLO（如100ms），剩余SMs分配给prefill。例如对Llama-70B在8×A100上，decode分配约60% SMs，prefill分配约40% SMs，避免chunked-prefill的token budget dilemma（需要4K tokens才能打满GPU，但SLO只允许256）。Layer-wise prefill避免了chunked-prefill因重复读KV cache导致TBT膨胀的问题，并支持长请求的抢占调度。

## Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads

- 属于Serving调度的实现是什么？实验比较什么？
  提出Shift Parallelism，将训练用Ulysses Sequence Parallelism (SP) 改造为inference可用的并行方式，并与Tensor Parallelism (TP) 在同一vLLM部署中按batch token数阈值动态切换。核心实现：(1) SP for Inference：为SP添加GQA支持（通过all-to-all中KV cache replication处理Q head数与KV head数不匹配的场景）、small batch load balancing（padding到SP degree倍数）和任意(SP, TP)组合forward pass（Algorithm 1）；(2) KV Cache Invariance：约束base config与shift config使用相同attention head layout和ordering，使请求在SP与TP之间切换时无需搬移KV cache；(3) Dual Configurations：base configuration使用SP或mixed (SP, TP)处理大batch优化TTFT和throughput，shift configuration固定为(SP=1, TP=P)的full TP处理小batch优化TPOT，运行时按batch token数是否超过shift threshold选择执行路径（Algorithm 2）；(4) Dual Model Loading：base model和shift model分别加载权重并独立CUDA graph capture，共享KV cache，shift model内存开销约1/SP。实验比较TTFT、TPOT、completion time、combined throughput (tokens/sec)、不同arrival rate下的latency-throughput曲线、不同input sequence length的峰值吞吐和最低延迟，对比vLLM内置TP和DP、独立SP (Ulysses)，以及在production trace中对比SGLang v0.4.6、TensorRT-LLM v0.18.2。

- 硬件平台是什么，配置是什么。
  AWS p5en.48xlarge单节点8×H200 GPU，每卡141GB HBM、4.8TB/s带宽、FP8 tensor core峰值1,979 TFLOPS，GPU间通过NVSwitch互联，标称900GB/s。Artifact Appendix注明通用NVIDIA DGX-H200节点即可复现。

- 开源Serving框架是什么。修改了什么。
  基于vLLM v0.9.2（Artifact Appendix使用vLLM v0.10.1，ArcticInference插件commit `5e08f0f`），通过ArcticInference插件系统集成。主要修改：(1) 实现SP for inference forward path：支持GQA（QKV投影中Q head与KV head数不一致时通过all-to-all send/receive buffer完成KV cache replication）、fused all-to-all（Q/K/V通信融合到单次all-to-all）、padding-based load balancing（小batch时padding到SP degree倍数）；(2) 实现combined (SP, TP) forward pass：SP按sequence维度切分输入，attention前后通过all-to-all在sequence parallel与head parallel layout间转换，MLP路径使用TP all-reduce；(3) 实现Shift Parallelism runtime：base model和shift model分别编译和CUDA graph capture，运行时根据batch size选择执行，通过SP_TP group确保shift model按base config的SP group order加载权重；(4) vLLM插件系统：编译并capture base和shift两套CUDA graphs，初始化时注册，运行时按threshold选择replay。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  已开源，GitHub仓库（snowflakedb/ArcticInference），Apache-2许可证，vLLM插件实现，Zenodo可复现实验包（DOI: 10.5281/zenodo.18240909）。使用步骤：
  1. 安装vLLM v0.10.1：`pip install vllm==0.10.1`
  2. 克隆ArcticInference：`git clone https://github.com/snowflakedb/ArcticInference`
  3. 安装插件依赖
  4. 下载模型（如RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic）
  5. 启动vLLM server带Shift Parallelism：`--enable-shift-parallel --shift-parallel-threshold <N>`
  6. 复现实验：运行`ArcticInference/benchmark/reproducibility`中脚本
  Shift Parallelism通过在一个部署中保留base (SP-biased)和shift (full TP)两套配置，运行时根据每轮batch token数是否超过阈值动态选择。例如在8×H200上，base config设为(SP=4, TP=2)，当batch token多时选base用SP降低TTFT和提高throughput；当batch token少（如decode阶段仅有少量活跃请求）时选shift (SP=1, TP=8)用full TP降低TPOT。切换不搬移KV cache因为两者attention head layout一致。在Llama-70B-FP8、4k input/250 output场景下，Shift Parallelism的median TTFT为148ms、median TPOT为51ms、peak throughput为69,147 tok/s，对比vLLM throughput-opt DP (1,355ms/83ms/75,535 tok/s) 和vLLM latency-opt TP (3,930ms/85ms/51,162 tok/s)，以17% throughput牺牲换来9.15x TTFT降低和1.63x TPOT降低。

## ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression

- 属于Serving调度的实现是什么？实验比较什么？
  提出stage-aware inference strategy：根据prefill（compute-bound）和decode（memory-bound）阶段的不同特性选择不同执行路径。decode阶段使用fused ZipGEMM kernel，实现"load-compressed, compute-decompressed"执行模型——直接从DRAM读取压缩权重，在register file内解压并直接送入Tensor Core计算，消除中间global memory buffer，提高compute intensity约50%（相比标准GEMM）。prefill阶段使用decoupled pipeline——先用高效decompression kernel解压到global memory，再用cuBLAS_TC做高吞吐GEMM，利用prefill高算术强度摊销解压开销（<4% overhead）。实验比较end-to-end latency和throughput，对比vLLM、Transformers和DFloat11；还比较memory consumption和KV cache扩展能力。

- 硬件平台是什么，配置是什么。
  (1) 1× RTX4090跑LLaMA3.1-8B；(2) 2× L40S跑Mistral-24B (tensor parallelism)；(3) 4× L40S跑LLaMA3.1-70B (tensor parallelism)。batch size 8/32，output length 128/256/512/1024/2048 tokens。

- 开源Serving框架是什么。修改了什么。
  基于vLLM扩展，约1.0K行Python glue code通过PyBind11集成自定义CUDA kernel（ZipGEMM + Decompression kernel）。修改：(1) vLLM model loader：支持加载TCA-TBE压缩格式的权重；(2) linear execution module：根据prefill/decode阶段选择不同kernel——decode用fused ZipGEMM，prefill用decoupled decompression+cuBLAS_TC；(3) weight memory management：压缩权重从14.96GB降至10.83GB (LLaMA3.1-8B)，释放的内存自动分配给KV cache (PagedAttention)，KV cache从5.07GB扩展到8.60GB（1.70×增加）。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  已开源：https://github.com/HPMLL/ZipServ_ASPLOS26.git。约3.5K行代码（2.5K CUDA/C++ + 1.0K Python）。使用流程：
  1. 环境准备：CUDA 12.4，PyTorch，vLLM
  2. 克隆仓库编译：`git clone https://github.com/HPMLL/ZipServ_ASPLOS26.git && cd ZipServ && mkdir build && cd build && cmake .. && make` 生成ZipGEMM .so
  3. 离线压缩模型：`python compress.py --model Llama-3.1-8B-Instruct --output compressed_model/` 产生TCA-TBE格式
  4. 加载到vLLM推理：压缩权重以PyBind11调用CUDA kernel，decode阶段自动使用fused ZipGEMM
  5. 端到端效果：LLaMA3.1-8B在RTX4090上，output 2048 tokens batch 32下，throughput 1105 tok/s（1.66× over vLLM），平均1.22× end-to-end加速；内存节省25-29%。例如Mistral-24B weight从43.92GB压缩至31.30GB (28.7% reduction)，释放的~12.6GB内存用于扩展KV cache支持更大batch size或更长context。

## DFVG: A Heterogeneous Architecture for Speculative Decoding with Draft-on-FPGA and Verify-on-GPU

- 属于Serving调度的实现是什么？实验比较什么？
  提出Cross-Device Compact Pipeline Scheduling（跨设备紧密耦合流水线调度），实现FPGA draft generation和GPU verification的overlapped execution。核心机制：(1) Stage-Decoupled Scheduling：FPGA持续生成draft tokens（包含多分支动态长度），GPU同时验证上一轮候选tokens——当GPU验证时FPGA不等待而继续生成新draft，当FPGA未完成时GPU从prefix继续forward生成新tokens；(2) 五阶段流水线：FPGA生成候选→GPU立即验证→GPU返回决策→若FPGA未完成则GPU继续forward→若token被rejected则FPGA rollback同时GPU从prefix forward；(3) interrupt-driven coordination：FPGA写入新draft token IDs到shared host buffer（BAR空间），更新状态寄存器，通过interrupt通知CPU；CPU触发DMA使GPU从同一buffer读取并验证；(4) lightweight cross-device alignment：仅传输compact token IDs + status metadata，通信占wall time的1.08%-3.2%，不成为瓶颈。实验比较：(1) end-to-end speedup vs AR、SpS、DuoDecoding、SpecInfer（2.44×-3.26×）；(2) wall time breakdown（draft 92%-96%，verify 96%-98%，communication 1.08%-3.2%）；(3) energy efficiency（4.33×-5.79×）；(4) framework对比（vs vLLM、LLaMA.cpp、GPT-Fast）。

- 硬件平台是什么，配置是什么。
  服务器：Intel Xeon 4310 CPU + NVIDIA RTX 4090 GPU (512 Tensor Cores, 2230 MHz, 24GB DRAM, 1008 GB/s BW, 330 FP16 TOPS, 450W TDP) + AMD V80 FPGA (300 MHz, 10848 DSPs, 43MB SRAM, 64GB DRAM, 76 GB/s BW, 225W TDP)。附加配置：NVIDIA A100 GPU (432 Tensor Cores, 1410 MHz, 80GB, 1935 GB/s, 312 FP16 TOPS, 400W TDP) + AMD U200 FPGA (6480 DSPs, 84MB SRAM, 32+32GB DRAM, 51+820 GB/s DDR/HBM BW, 190W TDP)。互联：PCIe Gen4×16 (64GB/s)。

- 开源Serving框架是什么。修改了什么。
  不是基于现有Serving框架修改，而是从零构建的自定义heterogeneous runtime。CPU host controller使用C++编写，支持non-blocking draft和verify streams，通过interrupt和DMA协调FPGA-GPU流水线。FPGA侧使用Verilog HDL实现的custom micro-architecture + 扩展的compiler支持communication/dynamic draft configuration/rollback recovery指令。GPU侧使用CUDA 12.1实现TreeSort-Verify framework，含unified KV-cache management和multi-GPU synchronization via NCCL。Xilinx Runtime (XRT) 2024.1管理FPGA-PCIe通信。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  已开源：https://github.com/ShaoqiangLu/DFVG（MIT License）。使用流程：
  1. 环境：Ubuntu 20.04，CUDA 12.1，Xilinx Vivado 2024.1，XRT 2024.1
  2. Setup FPGA：`source /opt/xilinx/xrt/setup.sh`
  3. 编译FPGA+GPU：`cd fpga/ && make synthesize && make implement && cd ../gpu/ && make all`
  4. 下载模型：`python scripts/download_models.py`
  5. 运行实验：`python scripts/run_experiments.py --config configs/llama7b.yaml`
  Pipeline调度作用：FPGA draft model生成多分支候选tokens（在hardware budget约束下），GPU verify model以batch方式并行验证所有candidates并做acceptance decision，两者通过PCIe ping-pong buffer + interrupt实现tightly overlapped execution。例如Qwen3-8B/V80+RTX4090配置下，draft阶段占wall time的92%-96%，verify阶段占96%-98%（两者overlap执行，不串行相加），通信开销仅1.08%-3.2%。Pipeline通过Pipe-Overlap ablation达到3.26× speedup（相比无overlap的3.08×）。

## PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel

- 属于Serving调度的实现是什么？实验比较什么？
  PAT作为vLLM v0.9.0的off-the-shelf plugin集成到LLM serving框架中，修改attention backend以利用跨请求共享prefix减少decode attention延迟。核心Serving层面实现：(1) pack scheduler将vLLM的block table（每行是一个query的KV block IDs）转为prefix tree，internal node表示多query共享的prefix段，用memory-centric profit model决定split/merge策略生成CTA partition，最小化全局KV cache重复加载；(2) lazy update机制使scheduler仅在block table变化时重新调度（如request arrive/depart或新KV block分配），否则复用上次调度结果，并与pre-attention tasks（metadata preparation、QKV projection）异步重叠执行；(3) Python侧通过pybind11暴露kernel API，约1.2k行Python代码集成为vLLM后端。实验比较end-to-end serving指标：mean TTFT、mean TPOT、P99 TPOT，在两模型（Llama-3-8B, Qwen3-8B）和两个真实trace（ToolAgent: tool/agent workload含系统prompt共享, Conversation: 三层prefix结构）上，对比RelayAttention++、FlashAttention、FlashInfer三种baseline。PAT减少mean TPOT 17.0-93.1%，减少TTFT 9.3-99.8%。进一步在Qwen2.5-72B-Instruct (4×A100 TP=2/PP=2)上减少TPOT 14.3-26.7%，Qwen3-30B-A3B (MoE, 单卡A100)上减少5.53-16.9%。

- 硬件平台是什么，配置是什么。
  NVIDIA A100-SXM4-80GB单卡（端到端online serving主测试）。NVIDIA H100-SXM4-80GB（kernel benchmark扩展）。分布式实验：4×A100 (Qwen2.5-72B-Instruct, TP=2 PP=2)。软件：CUDA 12.4, PyTorch 2.7.0, vLLM v0.9.0。

- 开源Serving框架是什么。修改了什么。
  基于vLLM v0.9.0修改，将PAT作为自定义attention backend集成。主要修改：(1) 新增pack scheduler模块，在每次decode step前读取vLLM paged KV cache的block table，构建prefix tree并生成CTA partition；(2) 替换vLLM默认的FlashAttention backend为PAT的pack-forward-merge pipeline；(3) pack scheduler运行在asynchronous CPU thread上与pre-attention tasks重叠；(4) 复用vLLM的paged KV cache机制（KV entries以fixed-size blocks在block table中管理），pack scheduler仅操作logical block IDs，不重写KV paging实现；(5) 启用方式：设置环境变量`VLLM_ATTENTION_BACKEND=PAT`。PAT不改动vLLM的continuous batching语义或request-level scheduling逻辑。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  已开源：https://github.com/flashserve/PAT（MIT License）。vLLM集成使用流程：
  1. 环境准备：x86-64 Linux, ≥64GB RAM, 200GB disk, A100-80GB, CUDA 12.4, PyTorch 2.7.0, vLLM v0.9.0
  2. 克隆仓库：`git clone https://github.com/flashserve/PAT.git`
  3. 启动vLLM server：`VLLM_ATTENTION_BACKEND=PAT python -m vllm.entrypoints.openai.api_server --model Qwen3-8B --gpu-memory-utilization 0.95`
  4. Pack scheduler在每次decode step从vLLM获取block table → 构建prefix tree → profit model决策CTA partition → lazy update复用上次结果直到block table变化 → 异步重叠scheduler和QKV projection等pre-attention任务
  5. PAT的pack scheduler平均latency比pre-attention task latency低42.3%-49.6%，因此异步执行时不增加end-to-end延迟
  PAT的作用：在vLLM的continuous batching框架下，通过prefix-aware attention kernel利用跨请求共享prefix，减少decode attention的global memory KV cache加载次数。例如在Conversation trace (Qwen3-8B, 8 req/s)下，PAT相比FlashAttention减少mean TPOT 89.5%、TTFT 99.6%；相比FlashInfer减少TPOT 93.1%、TTFT 99.8%。

## BAT: Efficient Generative Recommender Serving with Bipartite Attention

- 属于Serving调度的实现是什么？实验比较什么？
  提出Bat，一个面向生成式推荐(GR)的高效Serving系统。核心实现包含四个模块：(1) Bipartite Attention算法层（见算法pipeline），自适应选择User-as-prefix或Item-as-prefix attention以最大化KV cache复用；(2) Compute-storage disaggregated KV cache pool：将user-prefix cache和item-prefix cache作为独立组件主动管理，通过KV cache worker池化多机内存，cache meta service集中管理索引和热度；(3) Hot-replicated Cold-sharded (HRCS) item cache placement：利用item访问分布的skewness——hot item（top 10%占90%访问）在所有worker间复制，long-tail item均匀分片，最小化跨节点通信开销并保持cache效率；(4) Hotness-aware prompt scheduling：基于sliding-window频率估计(𝑓_u)，greedy决策每个request的attention模式——当user token数≥item token数且𝑓_u大于cache中最冷user page的频率时选User-as-prefix，否则fallback到Item-as-prefix。实验比较系统QPS、cache hit rate、P99 latency，对比RE(Recomputation)、UP(User-as-prefix)、IP(Item-as-prefix)三种baseline。

- 硬件平台是什么，配置是什么。
  主测试平台：4节点集群（浙江大学），每节点Intel Xeon Silver 4214 CPU (2×24 threads @2.20GHz)、200GB内存、1×A100-40GB GPU (PCIe 3.0x16)、100Gbps网络。每节点部署1个inference worker和1个KV cache worker。生产测试平台：16节点集群，每节点1×NVIDIA H20 GPU、Intel Xeon Platinum 8469C CPU (2 socket×48 cores×2 threads)、500GB host memory、200Gbps网络。10Gbps和100Gbps两种网络带宽设置（用于评估HRCS cache placement）。

- 开源Serving框架是什么。修改了什么。
  基于vLLM + FlashInfer构建。主要修改：(1) inference engine基于vLLM，定制attention module集成Bipartite Attention机制，使用FlashInfer作为高性能backend优化执行；(2) 新增disaggregated KV cache pool架构——KV cache worker管理paged memory（CPU memory），以user/item粒度存储prefix KV cache，物理组织为兼容PagedAttention的fixed-size pages，内置transfer engine支持GPUDirect RDMA；(3) 新增cache meta service——逻辑集中式进程追踪各worker/tier上每个user/item KV entry的索引和热度，接收prompt scheduler的cache read/write/check请求并协调KV cache worker的物理传输；(4) 新增hotness-aware prompt scheduler——调度器定期查询批量user/item的KV cache状态，用sliding-window频率估计决策attention pattern，将tokens拼接成batch分发到inference worker做load-balanced执行；(5) max-batched-tokens limit通过offline profiling确定以保证latency SLA。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  论文未明确说明代码开源地址。Bat系统使用流程：
  1. 离线初始化：分配pooled memory给item cache，剩余给user cache；运行HRCS算法(Algorithm 1)——profile网络带宽(B tokens/sec)→estimate prefill computation time via polynomial regression→根据超参数α计算max allowed communication ratio R_max→按item access frequency CDF确定hotspot replication ratio r→top-r% hot item在所有worker间复制，其余均匀分片。
  2. 在线Serving：hotness-aware prompt scheduler接收retrieval stage的ranking requests（含user ID和candidate item IDs）→查询cache meta service获取相关user/item的KV cache状态和hotness→根据decision rule选择prefix type：若𝜏_u≥𝜏_i且𝑓_u>min_{p∈C_u}𝑓_p则选User-as-prefix，否则选Item-as-prefix→concatenate tokens成batch→load-balanced分发到inference workers→触发cache read/write操作。
  3. Bat在Books dataset (Qwen2-1.5B)上相比UP提升throughput最多1.6×；在Industry dataset上相比RE提升2.3×；cache hit rate最高58%；降低total computation最多58%。

## Laser: Unlocking Layer-Level Scheduling for Efficient Multi-SLO LLM Serving

- 属于Serving调度的实现是什么？实验比较什么？
  提出Laser，一个面向multi-SLO LLM serving的layer-level scheduling系统，将调度粒度从iteration降为layer。核心实现包含：(1) layer-level chunked prefill：prefill Scheduler在每个transformer layer边界评估新请求的TTFT slack，决定是否抢占当前chunk并优先执行latency-critical请求，或动态合并relaxed请求到当前chunk以提升GPU利用率；(2) layer-level decode batching：decode Planner为每个请求生成execution plan（L: 每iteration执行的层数，O: 调度offset），通过模块化latency model预测per-iteration latency，贪心调整relaxed请求的层数以最大化batch容量；(3) inter-instance request dispatching：Global Controller对prefill侧选择slack最大的实例，对decode侧按SLO-homogeneous group管理实例并按TBT increment最小原则分配请求；(4) intermediate state cache与fused CUDA kernel：GPU上维护layer级中间状态缓存（prefill 16384 tokens，decode 2048 tokens），fused kernel合并state caching/retrieval。实验比较SLO attainment和goodput（90% SLO attainment下的throughput），对比Sarathi-Serve（prefill-decode aggregation + chunked prefill + EDF）和DistServe（prefill-decode disaggregation + iteration-level scheduling + EDF）。

- 硬件平台是什么，配置是什么。
  4台物理主机，每台4×NVIDIA A100 80GB GPU，主机间100 Gbps LAN，机内GPU NVLink互联。模型：Qwen2.5-14B (1-way TP)、Qwen2.5-32B (2-way TP)、LLaMA-3-70B (4-way TP)。

- 开源Serving框架是什么。修改了什么。
  基于vLLM + Ray构建。主要修改：(1) 新增layer-level chunked prefill：prefill Executor支持在layer边界保存/恢复intermediate state（hidden states），Scheduler实现EDF队列+slack-based抢占决策+动态chunk合并；(2) 新增layer-level decode batching：decode Planner实现latency analysis（分段线性model stateless module + 线性model stateful attention）+ execution plan construction（贪心减少relaxed请求层数，offset平衡同group负载），Executor支持按L/O plan在layer边界切换请求；(3) 新增Global Controller：prefill instance selection（slack-aware）、decode group-based assignment（SLO-homogeneous groups + decentralized performance evaluation）、instance group management（根据arrival rate动态调整group大小）；(4) intermediate cache：GPU memory内维护layer-level intermediate state缓存（类似KV cache），state manager索引active requests，fused CUDA kernel合并caching/retrieval；(5) KV cache migration按layer粒度异步进行，与prefill computation overlap；(6) 新增offline profiling流程：系统初始化时测量token count和context length变化下的module latency，拟合latency model参数，profiling在2秒内完成。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  论文未明确说明代码开源地址。Laser使用流程：
  1. 系统初始化：Global Controller离线profiling各serving instance的layer-level module latency（stateless模块用分段线性函数，self-attention用线性函数），拟合latency model系数，建立instance group。
  2. 请求处理：(a) 新请求到达Global Controller → prefill侧选择TTFT slack最大的prefill实例；(b) prefill instance的Scheduler计算新请求slack，若当前chunk剩余iteration时间危及新请求SLO则在layer边界抢占，判断是否合并后共同执行；若无需抢占且队列为空则动态合并；否则EDF入队；(c) prefill完成后KV cache按layer粒度异步迁移到decode instance。
  3. Decode阶段：(a) Global Controller按group-based算法分配decode请求到SLO-homogeneous group内TBT increment最小的实例；(b) decode Planner触发execution plan更新（仅request arrival/departure或latency接近最严格SLO时），latency analysis估计per-iteration latency → 若超标则选SLO最relaxed的请求减少其L并选最优offset → 若低于目标则尝试恢复strict请求全层执行；(c) Executor按plan执行decode computation，在layer边界切换请求。
  4. 在4×A100 80GB × 4 host集群上，Qwen-14B+ShareGPT/HumanEval/LongBench混合workload下Laser相比DistServe和Sarathi-Serve分别提升goodput 43.4%和1.67×；99% attainment目标下提升最高1.85×；relaxed请求占比高时改进从19.4%增至>86%；tight 0.8× SLO下goodput gain最高6.25× vs Sarathi-Serve。
  Laser的作用：将LLM serving的调度粒度从完整的模型forward pass（iteration）细化到单个transformer layer，使系统能在layer边界进行抢占、合并和差异化执行。prefill侧避免长prompt的head-of-line blocking和低效小chunk；decode侧允许relaxed请求部分执行layer以释放batch容量给更多relaxed请求。适用于共享foundation model、多应用多SLO、prefill-decode disaggregation的production serving环境。

## Difflow: A Data-Characteristic-Aware Serving System for Diffusion Models

- 属于Serving调度的实现是什么？实验比较什么？
  提出Difflow（原名ChituDiffusion），一个利用扩散模型请求中数据属性局部性(data property locality)的serving系统。核心Serving/调度实现包含：(1) dGraph Identification：编译时通过符号化数据属性传播(symbolic data property propagation)将扩散pipeline分解为dGraphs，每个dGraph内操作共享相同优化条件；(2) dTask Scheduling：运行时将异构请求按dGraph分解为细粒度dTasks，通过动态规划(DP, Algorithm 1)识别并批处理具有相同数据属性的dTasks——schedule对所有dEngines枚举property requirements，为每种requirement找到最大batch (GetLargestBatch)，递归组合剩余dTasks为batches，用基于线性回归的性能模型(OLS regression, R²=0.998)估计执行时间，选择最优batching plan；(3) Data-Aware Batching：针对不同raggedness ratio混合使用regular dEngines和ragged dEngines（第7.7节：raggedness 0%时uniform batching更优，100%时ragged batching达1.5×, 25%-50%混合batching额外提升10%）；(4) Asynchronous Property Inference：通过tensor fingerprint技术在无实际执行情况下推断dTask输出数据属性，使调度与执行重叠以隐藏scheduling overhead；(5) 调度窗口分析：调度开销低于dEngine runtime的10%，大窗口下GPU idle time <5%含cold start。实验比较throughput（req/s），对比PyTorch v2.1、PyTorch-Inductor v2.1、TensorRT v8.6、Stable Fast v1.0、Katz (per GPU)，覆盖5个UNet-based应用(refine/edit/video/venti/grande)和3个DiT-based应用(refine-mix/refine-dit/edit-dit)。

- 硬件平台是什么，配置是什么。
  两台服务器：(1) NVIDIA A100 40GB PCIe GPU；(2) NVIDIA H100 80GB PCIe GPU。UNet结构模型使用CUDA 12.1，DiT系列模型使用CUDA 12.8。开源release使用PyTorch 2.9。

- 开源Serving框架是什么。修改了什么。
  基于Diffusers[61]、Triton[59]、Stable Fast[1]、FlashAttention[21,22]组件实现(C++和Python)。核心修改/新增：(1) dGraph分解pipeline：符号化数据属性传播规则(Table 1, 覆盖elementwise/linear/convolution算子)将pipeline分片为共享优化条件的dGraphs；(2) dTask调度与DP batching：Algorithm 1的动态规划调度，融合dTasks with identical inputs，枚举dEngine batching plans；(3) 异步property inference：tensor fingerprint技术(Φ operat-commutative hash)推断dTask输出属性；(4) 性能模型：OLS regression on input-related metrics预测执行时间(R²=0.998)；(5) 四个ragged data-independent operation kernels (Triton+CUDA)支持ragged batching；(6) 支持UNet和DiT架构，通过DFG作为universal IR实现architecture-agnostic。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源地址：https://github.com/thu-pacman/chitu/tree/Diffusion。使用流程：
  1. 部署：加载diffusion pipeline定义（如SDXL+Refiner或SD15+ControlNet）→ChituDiffusion编译器执行符号化属性分析→将pipeline分解为dGraphs→每个dGraph编译为多个dEngines（针对不同数据属性配置）
  2. Application characteristics配置：开发者提供pipeline输入的prior knowledge——哪些输入cross-request相同(fixed property)、哪些可能相同(symbolic variable)、哪些经常变化
  3. 请求到达：Scheduler aggregate scheduling window内请求→按dGraph分解为dTasks→Async property inference计算tensor fingerprints→DP scheduler枚举dEngines寻找最优batching plan→dTasks batches dispatch到matched dEngines执行
  4. 以edit应用(SDXL Turbo + ControlNet + 16 LoRA styles)为例：pipeline被分解为3个dGraphs——dGraph1(输入依赖的U-Net层)、dGraph2(输入独立的U-Net层, 16 styles共享)、dGraph3(后处理)。dGraph2的input-independent计算被识别为loop-invariant→multi-value compile-time caching。运行时仅有2个unique dTasks for dGraph1和3个for dGraph2，通过data-aware batching exploit shared inputs实现1.71× speedup。
  5. 在A100上refine应用达2.1× speedup, H100上达2.2×；DiT refine-mix上相对Stable Fast达3.0×（因Stable Fast无法compile HunyuanImage refiner pipeline）。

Difflow的作用：通过利用扩散模型请求中数据属性的局部性（相同prompt的冗余计算、相同shape的批处理、loop-invariant tensors），将pipeline分解→编译优化→运行时动态调度三个层次协同，在不牺牲等价性的前提下最大化throughput。相比仅支持uniform batching的现有框架，Difflow通过dGraph重组+多版本dEngine编译+DP调度实现对异构数据属性请求的高效批处理。

## MixFusion: A Patch-Level Parallel Serving System for Mixed-Resolution Diffusion Models

- 属于Serving调度的实现是什么？实验比较什么？
  提出MixFusion，一个面向混合分辨率扩散模型serving的patch级并行serving系统。核心实现包含四个模块：(1) Compressed Sparse Patch (CSP) 格式：受CSR格式启发，将不同分辨率请求的图像partition为均匀patch，通过resolution reordering + offset-based compression实现高效patch定位；(2) Patch-Tailored Diffusion Operators：Self-Attention模块按resolution分组后reconstruct为全图再batch执行；Convolution通过记录每个patch的邻接patch依赖实现跨patch边界stitching；(3) Patch-Level Cache Manager：在每个block每步执行前通过Random Forest Cache Predictor（GPU端cuML）动态预测各patch是否需要重算，将cache query/delete/update/insert操作合并为batch执行（Common Set/New Set/Expired Set三集合分类）；(4) SLO-Aware Scheduler：以slack score（(DDL-C-P)/SA）量化请求紧急度→最紧急或最大吞吐提升的请求被加入active batch→MLP Throughput Analyzer（CPU端Scikit-learn, 3层MLP）预测batch latency→schedulability test阻止SLO violation。实验比较SLO Satisfaction、Goodput对比NIRVANA+ORCA（state-of-the-art diffusion serving）、Distrifusion（分布式patch并行）和Mixed-Cache（MixFusion变体用FCFS调度），以及CLIP/FID quality；ablation包括scalability（2/4/8 GPU）、workflow distribution（单分辨率主导）、SLO scale（3×/5×/10× baseline latency）。

- 硬件平台是什么，配置是什么。
  单台服务器：NVIDIA H100-80GB GPU + AMD EPYC 9534 64-core CPU。软件栈：Ubuntu 18.04, CUDA 12.3, PyTorch 2.2.2。多GPU scalability评估使用单节点内2/4/8 H100 GPU（数据并行，新请求dispatch到workload最低GPU）。

- 开源Serving框架是什么。修改了什么。
  基于PyTorch实现，遵循vLLM系统设计原则，总计12.5K行Python + C++/CUDA代码。核心修改/新增：(1) Stable Diffusion pipeline重构为Preparation/Denoising/Postprocessing三阶段，支持跨可变步数的batch denoising；(2) 集成xformers加速attention；(3) CSP格式的patch管理（resolution reorder + RequestOffset/ResolutionOffset/RequestStart/RequestEnd arrays）；(4) Patch Edge Stitcher fused入GroupNorm kernel（Section 4.3）；(5) Patch-level caching系统（Random Forest Cache Predictor on GPU via cuML, batched cache operations）；(6) MLP Throughput Analyzer（CPU端Scikit-learn）用于在线latency prediction；(7) SLO-aware scheduling algorithm（Algorithm 1）。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源地址：https://github.com/desenSunUBW/mixfusion。以SDXL serving为例使用流程：
  1. 部署：加载SDXL模型→MixFusion将pipeline分解为Preparation/Denoising/Postprocessing三阶段→Denoising中每个block前注入Cache Predictor→patch管理使用CSP格式
  2. 请求到达：多个不同分辨率请求（如512×512, 768×768, 1024×1024各一个）→Scheduler的MLP Throughput Analyzer预测合并后的batch latency→计算slack score→决定是否将新请求加入active batch
  3. Patch处理：patch size取各resolution在height/width维度的最大公约数→CSP格式通过RequestOffset记录每个请求的首patch偏移，通过RequestStart/RequestEnd记录每个patch所属请求
  4. Denoising执行：pixel-wise operators（Linear, FeedForward, Cross Attention）直接对各patch独立执行→Self-Attention时按resolution分组reconstruct全图后batched attention→Convolution时Patch Edge Stitcher fused in GroupNorm kernel处理跨patch边界
  5. Cache预测：每个block前Random Forest Cache Predictor比较当前输入与上步cache→生成mask标记reusable patches→block仅重算unmasked patches→masked patches用上步cache填充→batch cache query/update
  6. 完成所有denoising步后VAE decoder生成最终图像
  7. 在H100上以90% SLO satisfaction为目标：SDXL上MixFusion相比NIRVANA达到5.33× higher goodput，SD3上相比NIRVANA SLO satisfaction平均高30.1%

## RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization

- 属于Serving调度的实现是什么？实验比较什么？
  提出RoMeo LLM serving系统，集成RTMPQ算法到SGLang serving框架。核心系统级实现：(1) 异步并发执行（Asynchronous Concurrent Execution）：将activation量化分解为outlier token量化和normal token量化两个独立任务，四个cross-precision乘法kernel（W4A4/W4A8/W8A4/W8A8）各自仅依赖一个量化任务，通过fine-grained task dependency graph在多个CUDA stream上异步并发执行。使用CUDA events确保仅有真实依赖关系处才同步。(2) CUDA Graph工作流捕获：使用CUDA Graph捕获整个serving工作流，消除kernel launch overhead、memory allocation cost和PyTorch framework overhead，捕获后的graph无同步重复执行确保GPU连续运行。(3) SGLang集成：将RoMeo quantized modules集成到SGLang v0.5.5的离线benchmarking pipeline，测量prefill throughput（tokens per second），支持tensor parallelism（TP=2 for 14B, TP=4 for 32B）。实验比较prefill throughput和end-to-end serving speedup，对比BF16 baseline和QuaRot。Qwen3-8B (TP=1, batch=64) 上RoMeo prefill throughput 20073.60 tok/s vs BF16 10545.13 tok/s (1.90× speedup)。

- 硬件平台是什么，配置是什么。
  NVIDIA GeForce RTX 4090 GPU (24GB memory)。多GPU serving: Qwen3-14B使用2×RTX 4090 TP，Qwen3-32B使用4×RTX 4090 TP。软件: Python 3.12, PyTorch 2.8.0, CUDA 12.8, SGLang v0.5.5。

- 开源Serving框架是什么。修改了什么。
  基于SGLang v0.5.5修改。核心修改：(1) 将RoMeo的PyTorch nn.Module quantized linear layer替换SGLang原生FP16 linear layer；(2) 集成RTMPQ算法的online outlier detection + mixed precision quantization + cross-precision GEMM到SGLang的prefill pipeline；(3) 使用CUDA Graph捕获包括quantization→GEMM→dequantization→post-overwrite的完整workflow；(4) 异步并发执行使用多个CUDA streams管理quantization和GEMM kernel的并行执行。所有evaluation通过SGLang官方offline benchmarking scripts完成，固定input sequence length=128。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源：https://github.com/thu-pacman/RoMeo。Serving使用流程（以Qwen3-8B在单RTX 4090上serving为例）：
  1. 部署：加载Qwen3-8B PyTorch模型→RoMeo替换所有Linear为RTMPQ quantized modules→Hadamard rotation online融入FWT→offline weight outlier identification and quantization→JIT编译cross-precision CUDA kernels并缓存
  2. 请求到达（SGLang offline benchmark，batch size 8-64，seq_len=128）：Activation Hadamard Rotation (FWT)→per-token max value计算→top-k outlier token selection→online quantization（outlier→INT8, normal→INT4, 5% outlier ratio）→asynchronous concurrent GEMM execution across CUDA streams→dequantization with per-token scaling→post-mul overwrite of outlier positions
  3. CUDA Graph模式：graph捕获quantization→GEMM→dequantization→overwrite的完整执行图→重复replay无同步开销
  4. 多GPU场景：Qwen3-14B用TP=2，每GPU加载一半模型→RoMeo替换→NCCL all-reduce通信开销部分抵消量化加速
  5. 在batch=64时RoMeo prefill throughput达20073 tok/s (1.90× over BF16 10545 tok/s)；Qwen3-14B TP=2上batch=64达9064 tok/s (1.32× over BF16 6848 tok/s)

## AUM: Unleashing the Efficiency Potential of Shared Processors with Accelerator Units for LLM Serving

- 属于Serving调度的实现是什么？实验比较什么？
  提出AUM，一个AU-aware资源管理器，用于在AU-enabled CPU上与通用workload共享LLM serving，最大化处理器效率。核心实现包含两个协作组件：(1) **Background AU Profiler**（离线）：通过ARI (Arithmetic Intensity) 判定不同operator的AU使用率U_AU、按AU使用率将处理器划分为High-AU/Low-AU/None-AU三个频率区域、用CAT/MBA profiling不同AU使用率下的最小资源需求R_AU，最终汇总为离散AUV Model；(2) **Runtime AU Controller**（在线）：Slack-aware SLO Analyzer通过LAG分析实时量化每个serving request领先/落后于deadline的程度来设定decode SLO、Efficiency-aware Core Switcher按加权perf-per-watt最大化切换core分区分频配置、Collision-aware Allocation Tuner根据AU性能偏差δ_AU调整资源分配并优先收获对AU干扰最小的资源（如低AU operator的LLC和带宽）。实验比较CPU perf-per-watt efficiency和AU application SLO guarantee (TTFT SLO和TPOT SLO)，对比EXCLUSIVE (ALL-AU)、AUV-oblivious sharing (SMT-AU, RP-AU) 和AU-aware ablations (AU-UP, AU-FI, AU-RB)。

- 硬件平台是什么，配置是什么。
  三台商用AU-enabled CPU平台：(1) **GenA**: Intel 4th Sapphire Rapids (SPR), 2×Xeon 8475B (48核/socket, 2 socket), 2.7 GHz基频, AU TFLOPS: AVX-512 25.6 / AMX 206.4, L1-I 32KB, L1-D 48KB, L2 2MB/core, LLC 97.5MB/socket, DDR5 1TB (233.8 GB/s)；(2) **GenB**: Intel 4th Sapphire Rapids, 2×Xeon Max 9468 (48核/socket, 2 socket), 2.1 GHz基频, AU TFLOPS: AVX-512 25.6 / AMX 206.4, LLC 105MB/socket, HBM 128GB (588 GB/s)；(3) **GenC**: Intel 6th Granite Rapids (GNR), 1×Xeon 6982P-C (120核/socket, 1 socket), 2.8 GHz基频, AU TFLOPS: AVX-512 32 / AMX 344, L1-I 64KB/core, L1-D 48KB/core, L2 2MB/core, LLC 504MB/socket, MCR 768GB (600 GB/s)。主eval平台为GenA。GPU对比平台：单NVIDIA A100 80GB。

- 开源Serving框架是什么。修改了什么。
  基于Intel xFasterTransformer (xft) 实现LLM serving原型，通过两个Python组件实现AUM管理：(1) Background Profiler在专用节点上重复实验记录AUV Model（450次AU-enabled执行收敛：3 division × 3 sharing × 5 config × 10 repetition）；(2) Runtime Controller作为daemon监控SLO并tune资源分配，决策耗时<1ms。工具链使用Linux perf、pmu-tools、pqos、turbostat做AU行为表征。修改：(1) 修改xft支持基于ARI的AU选择（prefill用AMX、decode用GEMV/AVX）；(2) 通过Intel RDT (CAT/MBA) 接口实现LLC way和memory bandwidth的动态分区；(3) 实现LAG-based SLO analyzer (Algorithm 1) 和efficiency-weighted core switching决策逻辑。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  论文未明确说明开源地址（HPCA 2026已录用，可能后续开源）。使用流程（以chatbot场景，GenA + SPECjbb shared为例）：
  1. 部署：GenA上运行xft serving llama2-7b/13b BF16, batch_size=16。AUM的Background Profiler离线运行→收敛后构建AUV Model table (Table III示例：High U_AU=0-11 cores, F=2.1GHz, R_LLC=2way等)→Runtime Controller作为Python daemon启动
  2. 请求到达：LLM requests到达xft→prefill tokens (高AMX usage, TTFT SLO dTTFT=250ms)→decode tokens (低AMX usage, TPOT SLO dTPOT=100ms)。AUM计算每个request的LAG = Σ(dTPOT - e_token)
  3. Runtime Controller每个control iteration：Slack Analyzer计算SLO_H = dTTFT - t_wait, SLO_L = dTPOT + LAG→Core Switcher按max E_CPU = (1.8×P_H + 0.2×P_L + γ×P_N)/W_CPU选择最佳分区分频→Allocation Tuner监控P^m，若满足SLO则用P^a aggressive harvest资源给SPECjbb，否则用P^t conservative return资源
  4. 资源调整通过CAT设置LLC ways、MBA限制memory BW→每次调整<1ms，vs 100ms-scale token latency可忽略
  5. 效果：AUM相比ALL-AU baseline efficiency提升8.8%（平均），比SMT-AU/RP-AU提升4.7%；chatbot场景下SLO guarantee 93.6%（比AUV-oblivious高11%），decode TPOT SLO比AUV-oblivious高7%

## μShare: Non-Intrusive Kernel Co-Locating on NVIDIA GPUs

- 属于Serving调度的实现是什么？实验比较什么？
  提出μShare，一个非侵入式GPU kernel co-location inference serving系统，通过在Linux userspace拦截kernel launch并动态调整blocksize，实现不同kernel在同一SM内的scattered co-location。核心Serving调度实现包含：(1) Kernel Interceptor：通过LD_PRELOAD劫持CUDA kernel launch函数（cudaLaunchKernel、cuBLAS/cuDNN封装函数），使用dlopen+dlsym获取原始函数地址和参数，在传递参数前修改blocksize；(2) Block Shaper：对modifiable kernel设置half-plus blocksize（A40: 800=768+32, 即1536/2+32；A800: 704=683+32, 即2048/3+32），使同kernel的block不能堆叠在同一SM内，剩余线程供其他kernel block使用；对unmodifiable kernel（cuDNN/cuBLAS/tiling kernel）使用time-shifted launching——根据profiled resource utilization延迟启动kernel使其与SM上互补资源需求的kernel co-locate；(3) Batch Manager：feedback-based自适应batch size，每个time window结束后计算SLO slack s→j = Σ 2^(1-i) * (tSLO - ti)，positive slack线性增加batch（bj+1 = bj + k×s→j），negative slack指数减少batch（bj+1 = max{bj - e^(λ×s→j), 1}）；(4) Profiler：离线分析每个kernel的9-tuple资源特征{rFP32, rFP64, rINT32, rLDST, rSFU, rTensor, rmem, rreg, tLaunch}。实验在co-located multi-model serving场景比较μShare vs INFless (ASPLOS'22) 和Orion (EuroSys'24)。指标：system throughput (QPS)、normalized throughput (QPS/unit batch)、SLO violation rate（box plot, 20次重复实验）、end-to-end latency (CDF)、6种low-level hardware unit utilization timeline (Nsight Compute)、A800 GPU可移植性。额外对比Tacker (kernel fusion intra-SM) 和CUDA Graph优化。消融实验：μShare shape 1024（固定blocksize）、μShare w/o shape（无blocksize调整）、μShare w/o batch（无batch size反馈）。不同unmodifiable kernel比例（100%/89.67%/79.35%/69.02%/58.70%/48.37%）下的throughput伸缩。Co-locate scientific computing workload (Parboil benchmark) + inference models。

- 硬件平台是什么，配置是什么。
  8台服务器，每台配备Intel Xeon Gold 6338 CPU（128逻辑核，2.00GHz base/3.20GHz max, 251GB memory）+ (A) NVIDIA A40 GPU（84 SMs, 44.784GB memory, 每SM 1536 threads/102,400B shared memory/65,536 registers, CUDA 11.8）或 (B) NVIDIA A800 GPU（108 SMs, 80GB memory, 每SM 2048 threads/167,936B shared memory/65,536 registers, CUDA 12.1）。Inference framework: PyTorch 2.2.0。

- 开源Serving框架是什么。修改了什么。
  基于PyTorch 2.2.0作为GPU推理框架，μShare组件编译为.so shared libraries通过LD_PRELOAD加载到PyTorch进程。修改内容：(1) Kernel Interceptor：使用libdl的dlopen()打开CUDA动态链接库（libcudart.so/libcublas.so/libcudnn.so）→dlsym()获取原始kernel launch函数地址→创建同名函数通过LD_PRELOAD先加载→拦截kernel参数（blocksize/gridsize/sharedMem/stream）；(2) Block Shaper：使用shm_open()创建共享内存区域→通过mmap()映射修改共享内存中的kernel参数→kernel_process()接口上传modified blocksize→返还参数到dlsym()获取的原始函数地址恢复执行；(3) 设置PyTorch的C10_LAUNCH_BOUNDS(blocksize)宏与CUDA limit一致（1024）；(4) Batch Manager：基于exponential decay algorithm监控时间窗口内的SLO slack动态调整batch size。基线系统INFless (ASPLOS'22) 使用MPS+memory control实现SM和memory资源的不均匀分配实现inter-SM spatial sharing；Orion (EuroSys'24) 通过控制kernel launch time实现compute-intensive和memory-intensive kernel的耦合共置。两者均采用stacked co-location，不修改kernel blocksize。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  论文未明确提供开源代码仓库链接（HPCA 2026）。以10模型co-located serving on NVIDIA A40为例说明μShare使用流程：
  1. Offline profiling：部署每个模型（Llama2-7b/GPT-2/Bert/ResNet50-v1.5/MobileNet v2/Swin Transformer/ViT/Yolostiny/Resnet101/EfficientNet B7）→确定满足200ms SLO的最大batch size→Nsight Compute记录6种low-level hardware utilization→Nsight Systems记录kernel launch time→输出每kernel的9-tuple资源profile
  2. 部署：编译μShare的.so文件→设置LD_PRELOAD环境变量→启动PyTorch推理服务（每模型4 replica, 共40 replica分布于8 GPU）
  3. 请求到达：使用Azure INFless production trace→Batch Manager batch多个请求→发送batch到PyTorch→PyTorch sequential launch kernels
  4. Kernel intercept：μShare Kernel Interceptor通过LD_PRELOAD劫持cudaLaunchKernel等函数→读取kernel blocksize/stream参数→计算kernel launch slack（sk = tLaunch - tIntercept）
  5. Half-plus shaping：对前x个slack最小的kernel（|X| = x, x为最小满足前x个kernel的总block数超过SM数的值）→设置blocksize = 800 (A40, half+32)→large block占超过半SM threads→阻止同kernel blocks stacked→剩余threads供small block kernel使用
  6. Time-shifted launch：对kernel set Y，检查6种hardware资源combined utilization ≤ 100%且shared memory/registers充足→满足则直接launch→不满足则delay β=10μs后重检→更新slack并重排→若进入top-x则升级为half-plus shaping
  7. SLO反馈：每个time window结束后计算SLO slack→positive slack linear增加batch size→negative slack exponential减少batch size→保守增加、激进减少
  8. 结果：μShare system throughput 3046 QPS（vs INFless 1722 QPS/Orion 1192 QPS），提升26.90%–54.09%；average low-level hardware utilization 15.10%（vs INFless 10.90%/Orion 9.37%），提升38.53%–61.15%；SLO violation rate 3.35%（vs INFless 2.05%/Orion 1.12%），μShare v7 (k=0.05, λ=-0.2) SLO violation 0.84%同时throughput仍提升19.28%-44.83%

## LEGO: Supporting LLM-enhanced Games with One Gaming GPU

- 属于Serving调度的实现是什么？实验比较什么？
  提出headroom-maximizing LLM scheduler，实现LLM推理与游戏渲染在单张消费级GPU上的细粒度共置调度。核心调度设计：(1) 修改llama.cpp的traversal function加入调度逻辑：游戏引擎(UE4)监控rendering task状态变量→渲染完成后启动inference subtask→decode阶段以Transformer layer为调度粒度(~0.4ms/layer)、prefill阶段以self-attention和FFN sublayer为调度粒度(~0.5-1.0ms)；(2) Headroom Prediction：以LLM inference execution window为时间单位（100 APM=600ms含36帧、200 APM=300ms含18帧、300 APM=200ms含12帧），用LR模型以前三个window总rendering headroom预测下一个window总headroom，最大预测误差1.3%、平均0.6%、推理开销1.3ms；(3) Feedback-driven intra-rendering调度：监控rendering subtask的start/completion→当rendering subtask完成且下一subtask未开始时提交fine-grained LLM subtask→利用intra-rendering headroom（平均0.24ms/gap, 总平均1.39ms/frame）；(4) Inter-rendering调度：渲染任务完成→切换coarse-grained LLM subtask（含多个transformer layers）→利用帧间headroom；(5) Safety constraint：T_subtasks ≤ T_minimal（T_minimal为游戏所有rendering task中最小的inter-rendering headroom），保证利用intra-rendering headroom不会导致渲染任务latency violation；(6) Sudden spike处理：每个token生成后用最新workload数据更新预测→检测QoS violation风险→动态调整后续token的layer-skipping策略。实验比较LEGO vs SmallModel（同系列小模型）和LayerSkip（LITE/CALM），在三种游戏（BlackMyth/FFXVI/RDR2）×两种LLM（Llama3-8B/Mistral-7B）×三种APM（100/200/300）共18个场景下，测量99th-percentile FPS和APM。LEGO在所有场景同时满足FPS和APM目标。

- 硬件平台是什么，配置是什么。
  Windows 11, CUDA driver 566.36, CUDA SDK 12.1, DirectX 12.1。Intel i9-13900KF @ 3.00 GHz, Nvidia RTX 4090 (24GB)。所有游戏配置4K高画质60 FPS。

- 开源Serving框架是什么。修改了什么。
  基于llama.cpp (github.com/ggml-org/llama.cpp, commit fc83a9e) 作为LLM inference framework。游戏引擎为Unreal Engine 4，图形库DirectX 12。核心修改：(1) 只将llama.cpp front-end集成到UE4中，其他功能通过dynamic library调用；(2) 修改llama.cpp的traversal function（computation graph creation与traversal分离）加入调度逻辑：游戏引擎监控rendering task状态变量→渲染完成时dispatch inference subtask→decode阶段dispatch transformer layers、prefill阶段dispatch self-attention/FFN sublayers；(3) 在dynamic library中注册新的schedulable traversal function，保证推理执行正确；(4) 集成LR headroom predictor：运行时取前三个execution windows的总headroom预测下一个window。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源：论文未提供官方开源仓库。以BlackMyth + Llama3-8B + 200 APM为例说明调度流程：
  1. 部署：LEGO scheduler集成到UE4游戏引擎中→游戏启动时加载llama.cpp dynamic library→注册schedulable traversal function→离线准备好的adaptor模型权重加载到GPU memory
  2. 运行时：游戏以60 FPS渲染→每16.6ms一帧→scheduler监控每个rendering subtask的start/end状态
  3. LLM action到达（每300ms）：scheduler用LR模型预测接下来300ms window内的总rendering headroom（基于前三个window的headroom历史）→选择跳层策略（如跳4层）→选择合适的adaptor
  4. Prefill阶段：scheduler将self-attention和FFN sublayer作为调度粒度→当rendering subtask完成且下一subtask未开始时→提交一个attention/FFN subtask→执行约0.5-1.0ms→检查下一rendering subtask状态→若已开始则等待→否则继续提交
  5. Decode阶段：每个token生成时→以Transformer layer为粒度（~0.4ms）→利用intra-rendering headroom填充→整帧渲染完成后进入inter-rendering headroom→提交coarse-grained subtask（多个transformer layers）
  6. Safety check：每个提交的LLM subtask满足T_subtask ≤ T_minimal（该游戏最小inter-rendering headroom）→保证不阻塞渲染
  7. Sudden spike处理：生成每个token后用最新workload更新headroom预测→若检测到QoS violation risk→调整后续token的跳层策略
  8. Headroom usage：LEGO相比SmallModel在100/200/300 APM下分别提升25.2%/28.6%/18.8%；相比LayerSkip在200/300 APM下分别提升14.0%/16.2%

## AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding

- 属于Serving调度的实现是什么？实验比较什么？
  提出AdaServe，一个支持multi-SLO的LLM serving系统，核心实现是将multi-SLO serving形式化为带budget约束的token tree构造问题，并用SLO-customized speculative decoding实现细粒度per-request token分配。核心Serving调度实现包含：(1) SLO-customized scheduler：维护active request pool，执行speculation→SLO-customized selection→throughput-optimized selection→verification四阶段pipeline；(2) speculate阶段：用小draft model对每个请求做d步beam search，每步扩展w个候选token，形成有限candidate token tree，记录由draft logits近似的path probability；(3) SLO-customized selection阶段：先计算每个请求本轮至少需接受的expected accepted tokens以追上TPOT SLO，然后在该请求候选树中按path probability从高到低加入节点直到累计概率满足SLO所需或达per-request token limit；(4) throughput-optimized selection阶段：满足各请求SLO推进后若还有剩余token budget，全局排序所有剩余候选节点按path probability选择最高节点补入token tree；(5) verification阶段：所有请求selected draft token trees提交target LLM做tree-based parallel verification，接受匹配目标分布的token并返回verified tokens。实验比较vLLM（PagedAttention + continuous batching）、Sarathi-Serve（chunked prefill + co-batching）、vLLM-Spec(n)（固定speculation length n=4/6/8）、SpecInfer（原生speculative decoding引擎），在multi-SLO混合workload（coding copilot + chatbot + summarization）下评估SLO attainment、goodput、violation reduction。

- 硬件平台是什么，配置是什么。
  主评测节点：4×NVIDIA A100 80GB GPU，NVLink互连；CPU AMD EPYC 7763 (64 cores/128 threads)，256GB DRAM。Llama3.1-70B-Instruct使用4-way tensor parallelism + 4×A100 80G；Qwen2.5-32B-Instruct使用2-way tensor parallelism + 2×A100 80G。draft model与target LLM colocate在其中一张GPU上（Llama-3.2-1B-Instruct、Qwen2.5-0.5B-Instruct）。Artifact Appendix推荐复现配置：8×A100-SXM4-40GB或AWS p4de.24xlarge，CUDA 12.4，Docker + NVIDIA container runtime。

- 开源Serving框架是什么。修改了什么。
  基于FlexFlow Serve构建。核心修改：(1) 新增SLO-customized scheduler模块，实现speculation、SLO-customized selection、throughput-optimized selection和verification调度逻辑；(2) 将FlashInfer batched prefill kernel改造用于speculation steps和LLM tree-based verification；(3) 对draft model decoding使用CUDA Graph优化：从第二个speculation step到第d步，若活跃请求数相同则复用预捕获CUDA graph减少kernel launch overhead；(4) Request manager维护per-request状态（当前latency、已生成token数、TPOT SLO阈值），供scheduler决策使用。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源：https://github.com/zikun-li/AdaServe-Artifact-Evaluation（artifact evaluation repo），Zenodo DOI: 10.5281/zenodo.17052619。以Llama3.1-70B-Instruct + coding copilot/chatbot/summarization混合workload为例说明调度流程：
  1. 部署：4×A100 80G，4-way tensor parallelism部署Llama3.1-70B-Instruct + Llama-3.2-1B-Instruct draft model colocate on GPU。FlexFlow Serve启动，加载SLO-customized scheduler。
  2. 请求到达：coding copilot（HumanEval, TPOT SLO 1.2×baseline latency）、chatbot（Alpaca, 50ms/token）、summarization（CNN/DailyMail, 150ms/token）三类请求按真实trace缩放后的到达率进入request pool。
  3. 每轮decoding iteration：scheduler从request pool取出活跃请求→计算每个请求本轮的SLO推进目标（expected accepted tokens ≥ 已消耗latency / TPOT_SLO - 已生成token数）→execution engine跑d步beam search生成candidate token tree（每步w个beam）→记录draft logits近似的path probability。
  4. SLO-customized selection：coding copilot（SLO最紧）优先获得高概率节点，累计概率达SLO所需；summarization（SLO最宽）仅获少量节点，节省budget。
  5. Throughput-optimized selection：若budget有剩余，全局选最高path probability节点分配给各请求，最大化总expected accepted tokens。
  6. Verification：所有请求selected token trees提交target LLM做tree-based verification→LLM并行验证→接受匹配token→返回corrected tokens。
  7. 结果：在不同RPS下AdaServe SLO attainment提升可达2.1×/1.6×，未满足SLO请求数减少4.3×/3.2×，goodput提升1.9×/1.7×。CPU selection overhead仅占总serving time的0.41%/0.31%。

## PiLLM: Resource-Efficient LLM Inference Using Workload Prediction

- 属于Serving调度的实现是什么？实验比较什么？
  提出PiLLM，一个基于workload prediction的LLM serving系统，核心实现包含两层调度：(1) Inter-GPU Elastic Dispatch：global scheduler用滑动窗口统计输入/输出长度分布，通过中心极限定理预测批级长度上界（输出长度预测为 μ_d + σ_d/√|B| * Φ⁻¹(1-ε)），再用离线校准系数将长度转为prefill/decode FLOPs和KV cache内存需求，动态决定prefill/decode实例数；请求分发器优先放入idle instance，否则选预计最早完成的active instance，无法满足deadline时进入spike reaction快速激活新实例；(2) Intra-GPU Batch-Aware KV Cache Scheduler：不为每个请求预留worst-case剩余输出长度，而是在batch级别更新共享KV cache预算，利用大数定律使批级预测方差随batch size增大而下降，在保持低eviction的同时允许更高显存overcommit；KV cache组织为token slot linked list而非固定block。实验比较：(a) Inter-GPU：GPU Saving Factor和SLO Satisfaction Rate，对比utilization-based autoscaling和fixed maximum allocation两类baseline；(b) Intra-GPU：KV cache memory utilization、eviction rate、average batch size，对比vLLM greedy、PastFuture和SGLang。PiLLM在不同workload上实现1.62x-3.06x平均GPU节省，prefill SLO满足率≥97.9%，decode SLO满足率100%；显存利用率78.93%-96.05%，eviction rate 0.01%-0.53%（vLLM eviction可达68.39%）。

- 硬件平台是什么，配置是什么。
  8×NVIDIA H800 GPU（NVLink all-to-all互连），2×Intel Xeon 6448Y CPU，1TB DDR内存。软件栈：PyTorch 2.1、CUDA 12.1。

- 开源Serving框架是什么。修改了什么。
  基于LightLLM扩展实现。选择LightLLM的原因是它已有token-level KV cache管理，适合做批级KV cache sharing和disaggregated prefill/decode。核心修改：(1) API layer添加输入长度采集；(2) 新增global scheduling layer，包含input predictor、inter-GPU manager、request dispatcher和output predictor；(3) Execution layer包含prefill instances和decode instances，每个instance内部有intra-GPU manager、batch memory pool、running batch和waiting queue；(4) KV cache从固定per-request block改为token slot linked list，请求从共享池分配新token空间；(5) 使用chunk-based workspace、AOT compilation和CUDA graph降低动态内存分配与JIT抖动。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源地址：论文未明确说明开源仓库（EuroSys 2026）。作者Ruihao Gong主页（https://xhplus.github.io/publication/）未为PiLLM标出Code链接。以LLaMA-3.1 8B在8×H800上的disaggregated prefill/decode serving为例：
  1. 部署：启动PiLLM的API layer、global scheduler（含predictor + inter-GPU manager + dispatcher）、多个prefill instances和decode instances。LightLLM加载LLaMA-3.1 8B权重，按disaggregated模式配置。
  2. 请求到达：API layer接收请求并采集输入长度→global scheduler将长度统计写入滑动窗口。
  3. 预测阶段：global scheduler用当前窗口的输入长度统计和decode instance定期回传的输出长度统计，预测这批请求的平均输入/输出长度上界。用离线校准系数将长度转换为prefill FLOPs、decode FLOPs和KV cache需求。
  4. Inter-GPU调度：根据预测FLOPs与目标阶段延迟计算prefill/decode所需实例数。若idle instance足够直接分发；否则dispatcher选预计最早完成且满足deadline的active instance；若无合适实例则进入spike reaction组成最大可行batch并快速激活新实例。
  5. Intra-GPU执行：prefill instance处理输入token，KV状态逐层转移到decode instance。decode instance自回归生成，batch scheduler将batch的预测剩余token作为共享内存预算——某些请求生成少于预测值释放余量，另一些请求使用更多token slot，只要batch总量不越界即可维持高显存利用率并避免频繁eviction。
  6. 反馈：decode instance将完成请求的输出长度周期性回传给predictor更新统计窗口。
  7. 效果：在BurstGPT workload上PiLLM实现3.06x GPU节省，prefill SLO满足率99.2%，decode SLO满足率100%。单GPU显存利用率95.84%，eviction rate 0.01%。

## Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading

- 属于Serving调度的实现是什么？实验比较什么？
  论文提出FineMoE，一个细粒度(fine-grained) expert offloading系统，用于MoE-based LLM serving。核心Serving调度实现包含：(1) Expert Map Store：存储每个inference iteration、每个MoE layer上gate network对所有experts的概率分布（而非request-level的expert激活计数），用ndarray存储semantic embeddings和expert maps，容量默认1K maps（约200MB CPU memory）；(2) Expert Map Searcher：前d层使用semantic-based search（从embedding layer获取prompt semantic embedding，与历史embedding做cosine similarity取最相似历史iteration的前d层expert probability map），d层之后使用trajectory-based search（收集当前iteration已过层的gate probability trajectory，与历史expert maps做cosine similarity）；(3) Similarity-Aware Expert Selection：根据similarity score动态设置threshold delta=clip(1-score,0,1)，按概率从高到低选择最少expert使累计概率超过delta，且至少选择top-K所需数量；(4) Publisher-Subscriber异步路径：Context Collection→Expert Map Searcher→Prefetch Publisher→Cache Subscriber，将map search、prefetch和map update与inference forward解耦；(5) Expert Cache：C++/CUDA Runtime APIs实现prefetching/caching/offloading/on-demand loading，prefetch priority按p/(l-l_now)排序（概率越高、距离当前layer越近优先），eviction priority按1/(p×freq)（驱逐低概率、低访问频率expert）；(6) Expert Map去重：store满时用semantic和trajectory redundancy score去重。实验比较四个SOTA baseline：(a) MoE-Infinity（request-level Expert Activation Matrix + 同步prefetch）；(b) ProMoE（stride-based speculative prefetching + per-layer NN predictor，best-effort复现）；(c) Mixtral-Offloading（layer-wise speculative prefetching + LRU expert cache）；(d) DeepSpeed-Inference（expert-agnostic layer-wise parameter offloading，在MoE-Infinity codebase中实现并加入expert cache公平比较）。核心指标：TTFT、TPOT、expert hit rate、cache limit sensitivity、online end-to-end latency、overhead。

- 硬件平台是什么，配置是什么。
  主实验平台：6×NVIDIA GeForce RTX 3090（每张24GB GPU memory），GPU间pairwise NVLinks，CPU-GPU通过PCIe 4.0连接（带宽32GB/s），CPU为AMD Ryzen Threadripper PRO 3955WX（32 cores），480GB CPU memory。补充实验：NVIDIA A100（80GB HBM2e，2TB/s峰值带宽）。多GPU推理采用expert parallelism，expert通过hash map和round-robin分配到不同GPU。

- 开源Serving框架是什么。修改了什么。
  基于HuggingFace Transformers，复用/修改MoE-Infinity codebase构建FineMoE原型。Expert Map Store和Expert Map Searcher用Python/PyTorch/NumPy实现（ndarray存储，Tensor相似度计算）。Expert Cache在MoE-Infinity基础上用C++/CUDA Runtime APIs实现prefetching/caching/offloading/on-demand loading。新增GPU侧task pool与异步线程调度expert prefetching和on-demand loading。Baseline改造：MoE-Infinity评测前准备对应历史Expert Activation Matrix；ProMoE基于MoE-Infinity做best-effort prototype；DeepSpeed-Inference offloading逻辑在MoE-Infinity codebase中实现并加入expert cache公平比较。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  未找到FineMoE官方开源仓库。相关页面：https://2026.eurosys.org/papers.html（EuroSys 2026论文列表）；https://intellisys.haow.us/publications/（作者论文页仅提供PDF）。FineMoE复用/修改MoE-Infinity codebase，MoE-Infinity开源地址https://github.com/TorchMoE/MoE-Infinity。以Mixtral-8x7B + LMSYS-Chat-1M serving为例：
  1. 部署：启动6×RTX 3090 GPU集群，expert按hash map和round-robin分布到不同GPU。Expert Map Store预加载LMSYS-Chat-1M 70% prompts的semantic embedding和expert maps（1K maps, <200MB CPU memory）。Expert Cache初始化GPU cache limit（如6GB）。
  2. 请求到达：prompt "Summarize this paper" 到达。Prefill或第一个decode iteration开始前，Context Collection从embedding layer获取prompt semantic embedding。
  3. Expert Map Search（前d层）：Expert Map Searcher接收semantic embedding → 在Expert Map Store中计算cosine similarity → 找到最相似历史iteration → 获取其前d层expert probability map。
  4. Similarity-Aware Selection：similarity score s ∈ [0,1] → delta = clip(1-s, 0, 1)。若s低（相似度低），delta高，系统选择更多高概率expert预取以降低mis-prediction；若s高，delta低，只预取较少expert节省GPU cache。至少预取top-K所需数量。
  5. Trajectory-Based Search（d层之后）：当inference forward执行到d层后，收集已过层的gate probability distribution作为expert trajectory → 与历史expert maps做cosine similarity → 取对应目标层概率分布。
  6. Expert Prefetching：Expert Cache按p/(l-l_now)排序预取，把CPU memory中的expert weights异步搬到对应GPU。异步publisher-subscriber路径确保map search/prefetch与forward解耦。
  7. MoE Layer执行：gate network执行top-K选择。若所需expert已在GPU cache中→直接计算；若miss→FineMoE暂停普通prefetch task，立即从CPU到GPU on-demand loading缺失expert。
  8. Cache Eviction：cache超限时按1/(p×freq)驱逐低概率、低访问频率expert。
  9. Map Update：iteration结束后新产生的expert map和semantic/trajectory context写回Expert Map Store。Store满时用redundancy score去重保留覆盖性更好的maps。
  10. 效果：FineMoE相比DeepSpeed-Inference/Mixtral-Offloading/ProMoE/MoE-Infinity，平均TTFT分别降低74%/67%/56%/53%，平均TPOT分别降低46%/38%/27%/22%。Expert hit rate相比Mixtral-Offloading/ProMoE/MoE-Infinity分别提升14%/37%/68%。6GB cache limit下TPOT分别降低36%/25%/16%/29%。平均降低inference latency 47%，提升expert hit rate 39%。

## FlexPipe: Adapting Dynamic LLM Serving Through Inflight Pipeline Refactoring in Fragmented Serverless Clusters

- 属于Serving调度的实现是什么？实验比较什么？
  论文提出FlexPipe，一个在碎片化serverless集群中进行运行中pipeline重构(Inflight Pipeline Refactoring)的动态LLM serving系统。核心Serving调度实现包含三大组件：(1) Fine-Grained Model Partitioning：离线对LLM计算图做operator-level profiling（记录每个operator的computation time、parameter size、activation size），用带约束的动态规划算法在通信带宽、GPU显存上限和计算-通信重叠目标下寻找最优stage切分边界，同时保留Transformer attention block等层级结构边界以便未来合并；(2) Inflight Pipeline Refactoring：运行时持续监控请求分布的coefficient of variation (CV)、队列长度、吞吐/延迟profile，根据多目标优化公式（权衡吞吐最大化与延迟最小化，并用指数项确保所选granularity与当前请求模式对齐）在候选pipeline granularities（η_k stages, b_k batch size）中选择最优配置。当CV升高时拆细pipeline stage（Pipeline Expansion），申请新GPU instance加载fine-grained stage，异步迁移KV cache（优先RDMA、fallback到sendfile），更新gateway routing；当CV回落时合并相邻stage（Stage Consolidation），layer-wise merge state，释放多余instance或保留参数在host memory中以待warm restart。决策延迟<5ms。多粒度data parallelism框架决定每种granularity的并行实例数；(3) Adaptive Pipeline Scaling：使用Hierarchical Resource Graph（HRG）在server/rack/cluster三级协调GPU memory、PCIe带宽、网络带宽和存储I/O资源，避免扩容时资源争抢。Memory-aware elastic scaling使用host memory参数缓存和affinity-based scheduling（优先选择最近承载过该模型的服务器）将cold start转为warm start。GPU资源分配建模为约束优化问题，最大化throughput efficiency并惩罚multiplexing overhead（penalty ∝ CV²）。实验比较两类baseline：(a) serverless-based系统：ServerlessLLM（DeepSpeed parallelism分布式推理）和Tetris（memory-efficient hosting无专门pipeline parallelism）；(b) offline-optimized系统：AlpaServe（基于历史请求模式配置pipeline）、MuxServe（statistical multiplexing多租户serving）。也纳入throughput-latency optimization和interference mitigation相关系统对比。核心指标：goodput、end-to-end latency、prefill latency、pipeline stall recovery time、GPU resource efficiency、initialization latency、resource allocation wait time、always-on GPU reservation ratio。在CV=1下FlexPipe比AlpaServe/ServerlessLLM低38.3% overall latency（相同goodput 12,000 requests）；CV=4下比AlpaServe低66.1%、比MuxServe低80.6% total latency，保持98.3% maximum throughput。CV=4下stall recovery仅9ms（比MuxServe/ServerlessLLM快约82%）。CV=4下FlexPipe以43% GPU utilization达到12,000 req/s（Tetris以85% utilization仅1,543 req/s），资源效率最高8.5× better。生产case study：always-on GPU reservation从历史峰值75%降至30%，资源分配等待时间降85%，实例初始化延迟平均降72%。

- 硬件平台是什么，配置是什么。
  Kubernetes v1.23.7集群，42台服务器、82张NVIDIA GPU（A100级别），每台至少256GB内存，100Gbps网络互联。部分服务器支持RDMA。工作负载来自Microsoft Azure Functions traces，prompt用Splitwise corpus生成。评估模型：WHISPER-9B、LLAMA2-7B、BERT-21B、OPT-66B（120GB）。论文还使用Alibaba GPU集群trace（C1: 468 GPUs, C2: 1175 GPUs）进行资源碎片化分析。

- 开源Serving框架是什么。修改了什么。
  FlexPipe从零实现约7K LoC（含3.2K LoC动态operator-level模型partitioning/merging工具），非基于已有开源Serving框架修改。核心实现包含：(1) 离线partitioning工具：对PyTorch模型计算图做operator-level profiling，生成可执行model stages和granularity profile；(2) 运行时系统：API Manager（接收请求、监控QPS/SLO）、Scheduler（GPU分配）、Resource Monitor（GPU/memory utilization、CV、latency采集）、Performance Controller（granularity选择、pipeline refactoring决策、SLO约束验证）、Pipeline Refactoring模块（parameter migration、KV cache consistency、routing metadata更新）；(3) 数据传输：RDMA优先的hierarchical data transfer（不支持RDMA时fallback到sendfile系统调用），避免NCCL的连接建立overhead（数秒级）；(4) Hierarchical Resource Graph：标注scaling event marker的三级resource graph（server/rack/cluster），用于topology-aware resource coordination。论文明确指出未使用NCCL做KV cache迁移，因为其connection establishment overhead达数秒且可能引发带宽争抢。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源状态：论文未给出FlexPipe官方代码仓库或artifact链接。论文脚注给出了Alibaba cluster trace数据链接（https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-GenAI），但此为trace数据而非FlexPipe系统源码。论文声明原型约7K LoC，无法确认源码是否公开。

  以OPT-66B在82-GPU Kubernetes集群上的FlexPipe serving为例：
  1. 离线阶段：FlexPipe partitioning工具加载OPT-66B计算图，对每个operator profiling computation time、parameter size、activation size。动态规划算法在GPU显存上限(≤M_GPU)、inter-stage bandwidth(B)、target computation-communication overlap(C)约束下求解最优切分，同时保留attention block层级边界以支持未来stage合并。生成候选granularity set G={g_1,...,g_K}，每个g_k=(η_k, b_k)对应stage数和batch size，并profile各granularity的throughput T_k和latency L_k。
  2. 部署：API Manager启动，加载初始pipeline配置（如4-stage coarse pipeline）。Scheduler通过HRG在碎片化GPU中分配每个stage的GPU instance（严格禁止同一模型的多个stage分配到同一GPU以保证performance isolation）。Resource Monitor开始采集QPS、CV、latency、GPU/memory utilization。
  3. 稳定期运行（CV≈1）：系统以4-stage pipeline运行，每stage compute约69.94ms，inter-stage communication约6.3ms。请求进入API Manager→dispatch到stage-0 GPU→逐stage前向传播→输出token。
  4. Burst检测：Resource Monitor检测到请求到达率CV上升（如CV→4），队列长度增长。Performance Controller按公式g*=argmax[α·T_k/T_max+(1-α)·L_min/L_k·exp(-|ν_t-ν_k|/σ)]评估所有候选granularity，选择更细的pipeline（如16-stage）。
  5. Pipeline Expansion（Fig. 6a）：Scheduler在HRG中查找可用GPU，优先选择host memory仍保留该模型参数副本的服务器。新GPU instance加载split stage。系统在旧pipeline继续服务的同时，通过RDMA/sendfile异步迁移KV cache（token-level validity mask: C(t)=∪KV_i(t)⊗M_valid）。迁移完成后更新gateway routing metadata，后续micro-batch导入细粒度pipeline。细粒度stage（16-stage）使per-stage compute降至18.67ms、max batch从128增至512，通过distributed buffering吸收burst。
  6. 稳定恢复（CV回落）：Performance Controller检测CV下降→选择更粗的granularity（如合并回8-stage）。执行Stage Consolidation（Fig. 6c）：layer-wise merge相邻fine-grained stages的state→重建pipeline→更新load balancing路由→释放多余GPU instance（参数可保留在host memory中作为warm cache）。
  7. Multi-model scaling：当多个模型同时扩容时，HRG通过scaling event marker识别争抢模式，将不同模型的scaling操作分布到不同server/rack，避免GPU memory、PCIe、网络和存储I/O同时竞争。
  8. 效果：CV=4时FlexPipe以43% GPU utilization维持12,000 req/s（vs Tetris 85% utilization仅1,543 req/s）。Always-on GPU从峰值75%降至30%。Stall recovery在CV=4时仅9ms。

## High Throughput and Low Latency LLM Serving via Adaptive KV Caching

- 属于Serving调度的实现是什么？实验比较什么？
  论文提出eLLM，一个面向高吞吐和低延迟LLM serving的adaptive KV caching系统，基于vLLM扩展实现（约3,500行Python代码和1,700行CUDA kernel-level优化代码）。核心Serving调度实现包含：(1) request-level optimizer：scheduler启动时收集实时request metadata（sequence length、队列大小、最大等待时间），用SciPy SLSQP求解受GPU显存和TPOT SLO约束的优化问题，输出近似最优batch size b与uncached token ratio r；(2) token-wise/layer-wise KV cache管理：将vLLM的KV block从所有layer/多token放在一个物理block重塑为按F个连续layer划分的更小单元，维护seq_id/token_id/layer_id/logical_block_id/physical_block_id/#filled等map table元数据，每个SequenceGroup维护自己的uncached-token ratio r，较新的(1-r) token保留GPU KV cache，较早的r token释放显存（或swap到host memory）；(3) layer-wise execution pipeline：使用两个CUDA stream实现host-GPU KV传输与GPU重算并行，将下一层未缓存token的KV recomputation K1与当前层当前token的decode K2融合为fused kernel；(4) closed-loop adaptation：layer-level计算overlap/fusion额外显存开销Mo后反馈给request-level optimizer，联合优化缓存比例、batch size和fused kernel线程分配。实验比较：(a) baseline对比：vLLM-Recompute（显存满时丢弃整请求KV cache后整体重算）、vLLM-Swap（preempted request KV cache整体swap到host memory再通过PCIe恢复）、HCache（缓存hidden states加速状态恢复），均集成到vLLM中复现；(b) ShareGPT数据集上Llama2-13B单卡throughput分别提升2.64×/2.61×/1.91×，Llama2-70B 4卡TP分别提升2.0×/2.0×/1.6×；(c) TTFT最多降低2.63×，TPOT SLO attainment达97.3%-98.6%；(d) L-Eval长上下文上平均只缓存约53% prefix length，memory saving超47%，最高throughput提升3.03×，TTFT降低1.79×；(e) ablation显示禁用Comm-Com Overlapping或Kernel Fusion均退化，但保留token-wise caching后仍优于多数baseline；(f) 系统overhead：request-level optimizer 90%耗时<10ms，map table layer lookup平均<0.015ms。

- 硬件平台是什么，配置是什么。
  主实验服务器：4×NVIDIA A100-80GB GPU通过PCIe 4.0 x16连接（无NVLink）；CPU为96-core Intel Xeon Gold 6342 @ 2.80GHz，host memory 256GB。软件栈：Docker环境，CUDA 12.4，NVIDIA Driver 550.107.02。Llama2-13B使用1张A100，Llama2-70B使用4张A100以tensor parallel运行。host memory为Llama2-13B分配40GB，为Llama2-70B分配160GB存放swapped KV caches。

- 开源Serving框架是什么。修改了什么。
  基于vLLM开源Serving框架和PagedAttention机制。修改内容包括：(1) KV block结构重塑：从原有所有layer/多token共享物理block改为按F个连续layer划分的layer-granular block（默认F=4），降低粗粒度block内部碎片；(2) 新增map table元数据结构：维护seq_id、token_id、layer_id、logical_block_id、physical_block_id、#filled等，支持部分token/部分layer的缓存、swap和recompute精确定位；(3) request-level optimizer组件：引入SciPy SLSQP在线求解batch size b与uncached token ratio r的约束优化；(4) layer-level异步执行：引入torch.cuda.stream双CUDA stream管理，预编译多组CUDA shared libraries（32到1024线程、步长32）；(5) 保留vLLM核心API以兼容第三方模型。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源状态：论文正文提及基于vLLM扩展实现，作者实验室页面（https://cds-macau.github.io/publication/conference-paper/ellm/）确认论文发表于EuroSys 2026，但截至本次分析未找到eLLM官方GitHub仓库或artifact appendix。使用例子（以Llama2-13B在A100上处理ShareGPT请求为例）：
  1. 请求到达后进入vLLM等待队列，eLLM scheduler启动request-level optimizer：收集queue中所有请求的sequence length、队列大小、最大等待时间以及当前GPU显存占用和FLOPS信息。
  2. SLSQP求解约束优化（Eq.5）：maximize throughput = b / (compute_time(b,r) + overhead)，subject to GPU memory ≤ M_GPU + M_saved(r) - M_overhead(b,r)，且predicted TPOT ≤ SLO阈值。输出本轮batch size b和uncached token ratio r。
  3. 对batch中每个请求，若r=0.4，则其较早的40%历史token不再长期保留KV cache（可swap到host或释放），较新的60% token保留在GPU layer-granular block中。map table追踪每个token在每个layer的缓存状态。
  4. 进入transformer layer i时：layer-level scheduler通过map table定位cached token的物理block，对uncached token根据预算确定在本层是recompute还是从host swap。Stream A传输cached KV到SM，Stream B执行fused kernel K1（为layer i+1旧uncached token生成KV）+ K2（用layer i完整历史KV对新token decode attention）。完成后临时重算的旧token KV立即释放，新token KV写入对应layer-granular block。
  5. layer-level完成后计算实际overlap/fusion额外显存Mo，反馈给request-level optimizer，下一轮可能进一步增加b或降低r，形成closed-loop adaptation。

## MFS: An Efficient Model Family Serving System for LLMs

- 属于Serving调度的实现是什么？实验比较什么？
  MFS实现了一个面向LLM model family的multi-tier serving system。核心Serving调度实现包含：(1) Multi-tier model structure：通过Knowledge Precipitation将model family中最大模型微调为嵌套式multi-tier模型，低tier对应小模型、高tier对应大模型，前若干层/tier之间共享参数和hidden states；(2) Group batching：tier-level scheduler将不同tier但共享公共前缀层的请求组成group batch一起执行，替代Orca的selective batching（后者假设batch内请求共享同一模型结构）；(3) Attention fusion：decode阶段将不同请求的QKV矩阵拼接后做统一attention计算，牺牲少量冗余attention换取GPU并行度，提升小batch decode吞吐；(4) Shareable KV cache：KV cache manager记录multi-tier可共享状态，tier切换时复用已有KV cache而不重复计算；(5) Tier-aware speculative sampling：低tier作为draft model快速提出token，高tier作为target model验证，直接继承低tier KV cache。分布式部署支持intra-layer parallelism和inter-layer parallelism，同一tier的层放在同一partition以减少同步开销。
  实验比较：(a) batching场景：MFS vs Orca的request execution time、JCT（MFS JCT最多提升31.2%）、per-token response latency（最多降低56.1%）；(b) KV cache sharing：GPU memory footprint对比（MFS最多降低47.8%）；(c) speculative sampling：GPU utilization对比（Orca约23.9% vs MFS约59.8%）；(d) end-to-end：合成generative request trace下的median end-to-end latency per generated token。

- 硬件平台是什么，配置是什么。
  训练：2台服务器，每台8×NVIDIA H800 SXM5 GPU (80GB)、2×56核Intel Xeon Gold CPU、2TB内存，8×400Gbps NDR InfiniBand互联。推理评估：(1) 2×NVIDIA A100 GPU服务器，2×48核Intel Xeon Gold CPU、512GB内存；(2) 8×NVIDIA 3090 GPU服务器，80核Intel Xeon Gold CPU、256GB内存。

- 开源Serving框架是什么。修改了什么。
  基于Orca scheduler。修改包括：(1) 替换Orca selective batching为group batching：将不同tier但共享公共前缀层的请求组成group batch，共同执行前缀层；(2) 新增tier-level scheduler：为不同tier维护独立请求队列，按优先级/公平队列/外部策略选择下一批执行，支持plug-and-play调度策略；(3) 新增attention fusion模块：decode阶段将group batch内各请求QKV拼接为一个larger attention operation，以冗余compute换GPU parallelism；(4) 新增multi-tier KV cache manager：管理跨tier可共享的KV cache，支持tier切换时KV cache复用；(5) 新增tier-aware speculative sampling模块：低tier作draft、高tier作target，共享lower-tier KV cache。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  未找到MFS官方开源仓库（论文EuroSys 2026 DOI确认，但正文/参考文献未提供MFS代码入口）。公开页面：https://www.cse.ust.hk/~kaichen/papers/mfs-eurosys26.pdf。
  以Llama2-13B转3-tier MFS模型（tier-1=~7B等效/tier-2=~10B等效/tier-3=13B等效）在2×A100上serving为例：
  1. 离线：对Llama2-13B-chat做Knowledge Precipitation fine-tuning（数据guanaco-llama2 ~9.85k对话，AdamW lr=2e-5，1 epoch/2500 iterations，约24h on 16×H800），产出嵌套3-tier模型，tier边界通过测量各层推理latency确定（如tier-1取前24层对齐7B latency）。
  2. 部署：加载multi-tier模型到2×A100→tier-level scheduler初始化三个tier队列→KV cache manager初始化可共享缓存空间→front-end监听请求。
  3. 三类请求到达：R1（低latency需求，选tier-1）、R2（中等质量需求，选tier-2）、R3（高质量需求，选tier-3）→写入request pool→scheduler检测到三者都需要tier-1公共前缀层→组成group batch。
  4. Group batching执行tier-1公共前缀层：batch内合并R1/R2/R3→decode阶段attention fusion拼接三个请求的QKV→GPU并行度提升→产生的KV cache标记为multi-tier shareable。
  5. R1在tier-1完成后采样输出并返回；R2/R3继续tier-2→复用tier-1的hidden states和KV cache→R2完成后输出；R3继续tier-3→复用前两个tier的KV cache。
  6. 若启用speculative sampling：tier-1作为draft生成若干候选token→tier-3作为target验证→直接继承tier-1的KV cache（vs Orca独立部署需为target重新计算全部KV cache）。
  7. 效果：相比Orca独立部署Llama2-7B+13B两份模型，MFS单一嵌套模型：GPU显存降低（KV cache sharing最多-47.8%），group batching + attention fusion提升GPU利用率（speculative场景23.9%→59.8%），JCT提升31.2%，per-token latency降低56.1%。

## Efficient Multimodal Serving via Module Multiplexing

- 属于Serving调度的实现是什么？实验比较什么？
  论文实现EEVEE，一个基于module multiplexing的多模态模型serving系统（约5,000行Python，基于vLLM + NVIDIA MPS）。核心Serving调度实现包含：(1) Module-level Scheduling：将多模态模型（如BLIP的visual encoder、text encoder、text decoder）拆为独立进程，每个模块配置独立的batch size和SM allocation，不再使用统一batch顺序执行；(2) Stage-level Parallelism：将预处理和推理作为可独立调度的阶段，visual preprocessing与text inference可重叠执行，减少GPU等待CPU的空闲；(3) Request-aware Reuse via Modal Cache：对encoder-decoder模型（BLIP）缓存cross-attention消费的视觉token KV pairs，对decoder-only MLLM（LLaVA）缓存decoder消费的视觉token KV pairs；支持modal cache compression（按attention score剪除低重要性token）；(4) Synergistic Scheduling Algorithm：离线greedy search算法从monolithic顺序执行初始化，逐步增加各模块batch size，将增量放在对batch latency负面影响最小的stage，用balanced SM allocation将SM从短latency模块转移到长latency模块；(5) Controller：每GPU上的controller进程管理GPU shared memory、模块间通信（CUDA IPC）、参数read-only copy共享、cache LRU eviction和host memory spill；(6) 多GPU支持：大模块使用intra-operator model parallelism、小模块使用data parallel replication。实验比较Triton（monolithic graph、统一batch顺序执行）和Gpulet（spatio-temporal GPU sharing，但将各模块视为独立竞争模型，无协同调度和cache reuse）。五种模型：CLIP、BLIP、BLIP-2、LLaVA-1.5、InternVL2.5-8B。三种任务：VQA (VQA2)、NLVR (NVLR2)、image-text matching (COCO)。工作负载：Poisson distribution + Azure trace采样。指标：max capacity (queries/sec under SLO)、average/P99 latency、SM active ratio。消融：stage-level parallelism / module-level scheduling / request-aware reuse分别提升capacity 146%/152%/116%。调度算法对比simulated annealing、genetic algorithm、random greedy，相同搜索步数下提升最高30.10%。

- 硬件平台是什么，配置是什么。
  高端服务器：Intel Xeon Gold 6230R CPU (104 cores)、503GB内存、8×NVIDIA A100 40GB GPU。低端服务器：Intel Xeon E5-2698 v4 CPU (80 cores)、377GB内存、8×NVIDIA GeForce RTX 3090 GPU。所有GPU通过PCIe 3.0连接CPU，峰值带宽32GB/s。软件环境：Ubuntu 20.04.5、Docker 23.0.1、CUDA 11.8、Python 3.8、PyTorch 2.0.1。GPU利用率使用NVIDIA Nsight的SMs Active指标测量。

- 开源Serving框架是什么。修改了什么。
  基于vLLM作为后端inference engine。论文未修改vLLM源码，而是在vLLM之上构建module multiplexing serving layer：(1) 为多模态模型的每个模块启动独立的vLLM process，通过CUDA环境变量（MPS active thread percentage）在CUDA初始化前配置不同的SM allocation；(2) 使用NVIDIA MPS实现模块间GPU资源并发共享（而非MIG，因MIG仅提供刚性分区且仅限最新GPU）；(3) 单机内模块间通信使用shared memory + CUDA IPC handler；(4) 每GPU上运行controller进程管理共享GPU memory、模块输出缓存（64-bit hash索引、per-module hashmap、global LRU eviction、host memory spill）。Gpulet baseline同样集成到vLLM上，但将每个模块当做独立模型并强制共享固定batch size，无module-level协同调度。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源地址：论文未提供开源仓库URL，正文和参考文献中未找到EEVEE官方GitHub、artifact appendix或复现包。相关公开页面：https://2026.eurosys.org/papers.html（EuroSys 2026 accepted papers）；https://zicongs-homepage.webflow.io/publications（作者主页）；https://doi.org/10.1145/3767295.3769389（ACM DOI）。论文仅说明原型约5,000行Python、基于vLLM和NVIDIA MPS实现。截至分析时无法确认EEVEE已公开官方实现。
  以BLIP VQA serving为例（基于论文描述和vLLM+MPS文档重建）：
  1. 部署：启动vLLM backend→为visual encoder、text encoder、text decoder各启动一个vLLM process→配置CUDA_MPS_ACTIVE_THREAD_PERCENTAGE为各模块分配不同SM份额→启动controller进程管理GPU shared memory→scheduler加载离线生成的multiplexing strategy。
  2. 请求到达：用户提交图像+问题Q1→front-end将请求送入scheduler→scheduler按离线策略决定：visual encoder以batch size较小的配置处理图像（例如2请求/batch，70% SM），同时text decoder以较大batch size继续处理上一批请求的文本解码。
  3. Visual encoder处理图像生成visual tokens→controller将输出存入GPU shared memory，以64-bit hash索引→标记为可复用modal cache→超额时按LRU evict到host memory。
  4. Text encoder/decoder消费visual tokens和问题文本→生成答案A1。此时同一GPU上visual encoder和text decoder通过MPS并发执行，SM按离线策略动态分配。
  5. 同一图像的后续问题Q2到达→controller根据hash查找modal cache命中→从GPU或host memory加载缓存的visual tokens/KV cache→跳过visual encoder重计算→cache loading可与新图像的encoder compute overlap。
  6. 若GPU memory紧张→使用compressed critical modal cache（按attention score剪除30%低重要性token）→优先放入GPU memory以支持快速访问→full modal cache保留在host memory备用。
  7. 效果：相比Triton（monolithic serving）在3090 server Azure workload上平均max capacity提升157%，高负载LLaVA平均latency降低约90%，BLIP VQA GPU active SM接近90%。

## TailorLLM: Collaborative End-Cloud Inference of Large and Small Language Models Based on Low-Rank Adaptation

- 属于Serving调度的实现是什么？实验比较什么？
  论文实现TailorLLM，一个基于LoRA的task-level端云协同LLM推理系统。核心Serving调度实现包含三个在线模块：(1) Task Classifier：使用Contriever semantic encoder提取高维语义特征→UMAP降维→HDBSCAN层次密度聚类实现无监督动态任务分类，支持开放类别识别（无需重训），在15分类benchmark中准确率>95%；(2) Task Allocator：根据分类结果查表判断SLM是否满足任务精度要求+本地cache是否有对应LoRA，两者都满足则端侧推理，否则卸载到云端LLM；(3) AdapterMgr：基于imitation learning的LoRA library动态管理算法，使用Mamba (SSM)提取用户历史访问序列的时序特征，融合当前端侧LoRA cache状态（双模态embedding+projection融合），以Belady最优替换策略为学习目标，通过BCE loss训练，在端侧存储空间有限（capacity w=5）条件下动态决策LoRA的加载/淘汰。滑动窗口H=100，embedding维度d=128。实验比较：(a) end-to-end性能：cloud computing cost ($/1k queries)、multitasking accuracy、end-to-end latency (s/query)、total cost including transmission；(b) RTT影响：20/50/100/200ms下TTFT和TPOT；(c) microbenchmarks：AdapterMgr hit rate vs Belady/LRU/Parrot on MovieLens和构造数据集（cycle 30/200）；(d) ablation：TailorLLM-LoRA (标准LoRA替代RFLoRA)、TailorLLM-LRU (LRU替代AdapterMgr)。对比baseline：cloud-only (Llama3-70B)、end-only (Llama3-1B)、HSL (token-level speculative decoding，每5 token验证)、Petals (模型拆分5:65)、TailorLLM-LoRA、TailorLLM-LRU。

- 硬件平台是什么，配置是什么。
  Cloud-side: 4×NVIDIA RTX 3090 GPU (24GB GDDR6X)，Ubuntu 20.04 LTS。End-side: NVIDIA Tesla T4 GPU (16GB，通过visible resource constraint限制为10GB模拟资源受限edge设备)，Ubuntu 20.04 LTS。Tesla T4算力约为RTX 3090的1/6。端云通过无线网络连接，标准RTT 47ms。

- 开源Serving框架是什么。修改了什么。
  论文未基于现有开源Serving框架（如vLLM/SGLang），而是自行构建端侧和云侧prototype系统。系统组成：(1) 端侧：部署Llama3-1B + LoRA adapter加载/切换 + Task Classifier (Contriever+UMAP+HDBSCAN) + AdapterMgr (单层Mamba Block) + Allocator逻辑；(2) 云侧：部署Llama3-70B + LoRA library存储。端云间通过无线网络传输LoRA adapter parameters (每个adapter经RFLoRA压缩后约11.56MB)。论文未明确说明是否开源。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  论文未明确说明开源。以TailorLLM serving流程为例：
  1. 离线阶段：对9个下游任务数据集(80% fine-tune/20% test)用RFLoRA训练task-specific LoRA adapters→存入云端LoRA library。RFLoRA冻结共享A矩阵，仅存储任务特定的B矩阵+magnitude参数m（相比标准LoRA减少约50%传输参数）。
  2. 在线推理：用户query到达→Task Classifier提取Contriever semantic embedding→UMAP降维到3维→HDBSCAN密度聚类判定任务类别（已知类别/新类别/uncertain标记为-1）。
  3. Allocator查表：若SLM在该任务accuracy达标且本地cache有对应LoRA→加载LoRA到SLM→端侧完成推理（Llama3-1B 22.6ms/token）。否则卸载到云侧Llama3-70B推理（5.3ms/token）。
  4. AdapterMgr后台运行：维护滑动窗口H=100的历史访问序列→Mamba提取时序特征→融合当前cache state→MLP+Softmax输出每个cache slot的替换概率→决定是否从云端下载新LoRA替换某个slot。以Belady最优策略为训练目标。
  5. 效果：约70%请求端侧处理，cloud cost减少69.8%，end-to-end latency减少62%（vs cloud-only）。TTFT在不同RTT下保持低水平，TPOT在RTT 20→200ms时仅1%性能退化（vs HSL 22%、Petals 46%）。

## AIMS: A Cost-Efficient Framework for LLM-based Agent Deployment in Cloud-Edge Hybrid Environments

- 属于Serving调度的实现是什么？实验比较什么？
  论文实现AIMS（Adaptive Iteration-level Model Selector），一个面向云边混合环境的AI agent subtask调度框架。核心Serving调度实现包含五个离线训练的estimator和四阶段在线决策流程：(1) User Request Classifier (URC)：用ModernBERT fine-tune预测整个请求在SLM和LLM上的输出相似度，若相似则全请求走SLM，否则进入subtask-level分配；(2) Subtask Similarity Evaluator (SSE)：用SP_SLM和SP_LLM两个subtask predictor（Qwen3-0.6B + LoRA fine-tune）预测当前subtask的下一个subtask，比较两者输出的SBERT cosine similarity，超过自适应阈值κ则分配SLM。阈值κ = threshold_base + min(ID, 5) · 0.02（threshold_base=0.6），随subtask位置后移而收紧；(3) S-L Similarity Evaluator (SLE)：用Distance Predictor（ModernBERT + LoRA）预测当前LLM subtask的S-L distance d，再用SP_SLM预测第(i+d)个subtask输出与LLM第i个输出比较——若相似则SLM处理当前及后续d个subtask；(4) Convergence Detector (CD)：从当前subtask起迭代预测SLM和LLM的未来subtask序列，寻找S-L similarity超过κ的收敛点，选取最后一个收敛点以最大化SLM使用；(5) Subtask Decomposer (SD)：将复杂subtask用Qwen3-0.6B fine-tune的模型分解为更简单的子subtask序列，仅在全部子subtask都适合SLM处理时才将整组分配给SLM，否则交LLM。在线决策遵循fast-path/slow-path模式：URC→SSE（fast-path）快速判定可走SLM的请求/subtask，SLE/CD/SD（slow-path recovery）在fast-path失败时寻找更多SLM机会。整个estimator栈约需2GB VRAM。实验比较：(a) accuracy和SLM usage：9个benchmark上vs HybridLLM（classifier-based independent subtask routing）、Minions（confidence-based routing）、All-SLM、All-LLM、Oracle、Random；(b) end-to-end latency breakdown（SLM latency/LLM latency/method&network overhead）；(c) 归一化remote cost（All-LLM=1.0）；(d) 泛化性：跨模型对（Qwen3-4B+GPT-5 / Gemma3-4B+Claude Sonnet 4）、跨硬件（RTX 5090 / iPhone 15）；(e) 消融实验：逐一移除SD/CD/SLE/SSE/URC组件；(f) 训练数据量消融（0%-100% traces）；(g) estimator性能（准确率+latency ratio）；(h) 参数敏感性（τ_req 0.50-0.78、τ_sse 0.60-0.80）；(i) trace-level failure analysis（early divergence accumulation / late-stage sensitivity / lack of convergence and decomposition）。

- 硬件平台是什么，配置是什么。
  本地设备：NVIDIA RTX 5090 GPU，SLM用llama.cpp执行占用4-6 GB VRAM。云端LLM通过public API访问。离线fine-tuning：cloud-based Nvidia A100（全部estimator fine-tune约2小时）。移动端泛化测试：iPhone 15。Estimator栈约需2GB VRAM，可在8-16GB显存游戏本上运行。

- 开源Serving框架是什么。修改了什么。
  基于AutoGen构建agent stack。论文在AutoGen之上实现了AIMS scheduler层：(1) 新增request-level classifier（URC），在agent收到请求后先判定是否整请求走SLM；(2) 新增subtask-level routing逻辑（SSE/SLE/CD/SD），替代AutoGen默认的固定模型选择，对agent workflow中每个subtask的生成和执行位置做动态决策；(3) 新增offline profiling pipeline：生成subtask binary tree（每个节点用SLM和LLM分别处理、递归分支至深度15或完成），收集SLM/LLM的subtask输出、S-L distance、subtask decomposition数据用于训练estimator。agent的tool execution部分（如Wikipedia search/WebShop navigation/code interpreter）未修改，AIMS专注于"语言模型调用"这一环节的SLM/LLM选择。论文未明确说明是否开源。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  论文未提供明确开源仓库URL（EuroSys '26 DOI: https://doi.org/10.1145/3767295.3803622，未在正文/artifact appendix中找到GitHub链接）。以AIMS在RTX 5090上serving HotpotQA请求为例（基于论文描述和AutoGen文档重建）：
  1. 离线profiling：对WorFBench和GSM8K的1000条subtask traces→生成SLM/LLM binary tree→收集输出similarity/S-L distance数据→fine-tune URC (ModernBERT)、DP (ModernBERT)、SP_SLM/SP_LLM (Qwen3-0.6B)、SD (Qwen3-0.6B)，全部用LoRA fine-tune，约2小时/A100。
  2. 线上请求："Who was the maternal grandfather of the director of Titanic (1997)?"→URC预测整请求similarity score若>0.7则SLM处理；否则进入subtask routing。
  3. AutoGen agent生成第一个subtask ST1="Identify the director of Titanic (1997)"→SSE用SP_SLM/SP_LLM预测ST1的next subtask→比较similarity→若>κ（early stage κ≈0.62）则SLM执行ST1。
  4. ST1 SLM执行完生成ST2="Find the director's mother"→SSE判定similarity<κ→SLE预测d=1（SLM多需1个subtask可达LLM对应输出）→SP_SLM预测第3个subtask vs SP_LLM预测第2个subtask→similarity>κ→ST2和ST3走SLM。
  5. ST4="Verify and confirm maternal grandfather"→SSE/SLE均失败→CD迭代预测无收敛点→SD将ST4分解为4个子subtask（Search/Extract/Find bio details/Confirm name）→全部子subtask通过SSE→整组走SLM。
  6. 效果：AIMS accuracy 90.75% vs HybridLLM 76.35% (+14.40%)、Minions 84.20% (+6.55%)，SLM usage 81.85% vs HybridLLM 68.40%、Minions 74.10%。端到端延迟13.33s（All-LLM 15.82s、HybridLLM 12.98s）。Scheduler决策overhead占总时间3-7%，网络hop latency平均0.58s可忽略。83% cloud cost savings vs All-LLM。

## TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone

- 属于Serving调度的实现是什么？实验比较什么？
  论文实现TZ-LLM，一个基于Arm TrustZone TEE保护端侧LLM模型机密性的安全推理系统。核心Serving调度实现包含：(1) Pipelined Parameter Restoration：将LLM computation graph扩展为computation operator前插入allocation/loading/decryption三类restoration operator的pipeline，按拓扑顺序调度。CPU上采用priority-based greedy policy：若CPU computation operator ready则优先执行，否则执行与最早computation operator相关的restoration operator。为减少operator时间错配引起的pipeline bubble，allocation和decryption被切成micro-operator支持preemptive scheduling。配合pipeline-aware extend/shrink接口利用LLM参数first-in-last-out模式保证TZASC连续secure memory。(2) Partial Parameter Caching：在REE memory pressure允许时缓存早期prefill operator的参数，inference后按reverse topological order逐步释放，消除pipeline起始bubble而不长期锁住完整模型内存。(3) Co-driver NPU设计：REE driver负责scheduling/power/frequency等control plane；TEE user-mode data plane driver只负责secure job context/MMIO launch/interrupt completion，通过shadow job机制使REE scheduler将secure job放入统一队列，TEE通过initialized/not-issued状态和monotonic sequence number防重放和重排序攻击。NPU world switch通过TZPC/TZASC/GIC配置实现。(4) 工程机制：TA多线程shadow thread、framework initialized-state checkpoint。实验比较：TTFT、decoding speed、pipeline critical path overhead、不同cache proportion下TTFT、NPU sharing throughput、CMA allocation对REE应用的干扰。对比baseline：REE-LLM-Memory（无保护全部参数预加载）、REE-LLM-Flash（无保护pipelined restoration不解密）、Strawman（TEE cold start无pipeline无NPU）。

- 硬件平台是什么，配置是什么。
  Orange Pi 5 Plus开发板，SoC为Rockchip RK3588，包含4×Cortex-A76 @2.4GHz + 4×Cortex-A55 @1.8GHz，16GB LPDDR4X，1TB NVMe SSD，3-core NPU（峰值~6 TOPS）。LLM TA使用4个Cortex-A76 CPU core和全部3个NPU core。为触发CMA memory migration，用stress-ng模拟REE memory pressure：四个模型对应压力分别为13GB/11GB/10GB/6GB，stress threads与LLM threads pin到不同CPU core。

- 开源Serving框架是什么。修改了什么。
  基于llama.cpp作为LLM inference backend。修改包括：(1) llama.cpp扩展约1.2K LoC实现pipelined restoration（computation graph提取、restoration operator插入、priority-based + preemptive scheduling）；(2) 扩展约1K LoC集成NPU data plane driver；(3) 使用OpenSSL做参数解密。REE侧：OpenHarmony v4.1 / Linux v5.10，扩展约364 LoC（NPU driver 167 LoC shadow job scheduling + TZ driver 197 LoC CMA allocation/deallocation）。TEE OS原本约17K LoC，扩展约62 LoC支持CMA page memory mapping，约50 LoC支持动态TZASC/TZPC配置。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源：Zenodo artifact DOI 10.5281/zenodo.17213486（EuroSys '26 artifact，包含prototype system源码和复现实验脚本，MIT License），Hugging Face papers页面可交叉确认arXiv:2511.13717。论文Artifact Appendix明确写出artifact内容。以TZ-LLM serving流程（Llama-3-8B 512-token prompt）为例：
  1. 启动请求→LLM TA根据llama.cpp提取的computation graph拓扑顺序→为最早prefill operator所需参数调用extend_allocated→从REE Linux CMA获取连续物理内存。
  2. REE file system异步将加密参数读入尚未TZASC-protected的allocated memory（避免bounce buffer）→TA调用extend_protected→TEE OS扩展TZASC region→映射内存入TA→解密参数。
  3. 第一个computation operator参数ready→开始CPU/NPU prefill，同时后续operator的allocation/I/O/decryption在后台推进。
  4. CPU computation ready时→scheduler抢占长allocation/decryption micro-operator优先执行computation→减少NPU/CPU等待。
  5. Decoding阶段：TA使用co-driver NPU→REE scheduler将secure shadow job排入统一队列→选中后smc通知TEE data plane driver→TEE配置TZPC+TZASC+GIC→等待non-secure job完成→验证sequence number→MMIO launch secure NPU job→interrupt完成→切回non-secure mode→REE标记完成继续调度。
  6. 效果：vs strawman TTFT降低77.1%-91.1%；vs REE-LLM-Flash TTFT overhead 5.2%-28.3%；decoding overhead 1.3%-4.9% vs REE-LLM。NPU time-sharing对REE NN应用额外slowdown最高3.8%。CMA对并发Geekbench性能下降最高6.7%。

## From Imperative to Declarative: Towards LLM-friendly OS Interfaces for Boosted Computer-Use Agents

- 属于Serving调度的实现是什么？实验比较什么？
  论文实现Declarative Model Interface (DMI)，一个LLM友好的OS声明式接口中间层，基于UFO-2 agent框架修改。核心实现分两段：(1) 离线阶段：通过Windows UI Automation (UIA) 从目标应用中抽取UI Navigation Graph (UNG)，带环和merge node的图转为path-unambiguous forest，再压缩为LLM友好的core topology层级文本描述（Excel约30K tokens, Word约15K, PowerPoint约15K）；(2) 在线阶段：DMI为LLM提供三类声明式原语——access（声明目标控件ID）、state（设置控件状态如滚动条位置、选中范围）、observation（读取文本），由DMI executor负责导航路径求解、控件模糊匹配和交互执行。LLM不再生成细粒度GUI操作序列（点击、拖拽、输入），只需声明意图。DMI原型超18K行Python，基于pywinauto调用Windows UI Automation。
  实验比较：(a) UFO-2 GUI-only baseline（UFO2-as, 使用action sequence）vs GUI+DMI，在OSWorld-W中27个单应用Office场景下对比成功率(SR)、平均LLM调用步数(Steps)、完成时间(Time)；(b) 不同LLM模型下的DMI增益：GPT-5 medium/minimal, GPT-5-mini medium；(c) 消融实验：为baseline注入DMI navigation forest静态知识但不启用声明式接口，SR反而从44.4%下降到42.0%；(d) 失败分析：将失败分为policy-level（语义规划错误）和mechanism-level（控件定位/导航/交互错误），验证DMI是否将错误从mechanism转移到policy。

- 硬件平台是什么，配置是什么。
  实验平台为Windows系统，运行Microsoft 365 builds的Word、Excel、PowerPoint。LLM使用OpenAI GPT-5和GPT-5-mini API（reasoning effort: minimal/medium）。论文未明确说明具体Windows机器硬件配置（CPU/RAM等）。每个任务最多30步，运行3次取平均，不做模型微调。

- 开源Serving框架是什么。修改了什么。
  基于Microsoft UFO-2 open-source computer-use agent framework（https://github.com/microsoft/UFO）。修改内容：(1) 添加DMI execution layer，在UFO-2 agent workflow的原GUI action path之外增加declarative action path；(2) DMI层内实现UNG构建模块（DFS + differential capture发现导航边、merge node处理、shared subtree识别）、forest压缩模块（core topology生成、控件ID整型映射）、在线executor模块（path resolution、fuzzy matching、load重试、OK>Close>Cancel优先级的浮窗关闭策略、fallback到GUI slow-path）；(3) baseline和DMI都注册UIA event handler确保完整control tree暴露，避免lazy loading造成差异。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源地址：https://github.com/dmi-interface/DMI（MIT License）。仓库集成到Microsoft UFO Windows computer-use agent framework，提供面向Microsoft Office Word、Excel、PowerPoint的预构建core topology。
  DMI作为Agent-OS交互中间层使用：
  1. 离线构建：每个应用运行UNG builder（自动exploration <3h，人工配置约1.5人日）→生成core topology→保存forest。
  2. 部署：在UFO-2中配置DMI executor，加载预构建core topology。
  3. 任务执行流（以PowerPoint "将所有幻灯片背景设为蓝色"为例）：
    a. Task到达UFO-2→LLM接收prompt（task + core topology层级描述 + 声明式API说明）。
    b. LLM生成声明式action JSON：声明控件ID（如`Blue`选项leaf ID、`Apply to All` leaf ID，shared subtree中需附带entry_ref_id）。
    c. DMI executor：解析控件ID→求解唯一导航路径（确定性图搜索）→fuzzy matching当前窗口控件→按OK>Close>Cancel优先级关闭浮窗→沿路径点击导航控件→通过UIA ScrollPattern/TextPattern/SelectionPattern等control patterns执行状态设置。
    d. DMI在每次LLM调用前通过passive mode采集DataItem截断文本，`get_texts()` active mode按需返回完整文本。
  4. GPT-5 medium下：SR 44.4%→74.1%（+29.6%绝对），Steps 8.16→4.61（-43.5%），Time 392s→239s（-39%）。GPT-5-mini medium：SR 17.3%→43.2%（约2.5×）。失败分析：GUI+DMI失败81.0%为policy-level，GUI-only mechanism-level失败占53.3%。
  5. DMI保留GUI imperative path作为slow-path fallback用于非标准控件场景。

## FlashPS: Efficient Generative Image Editing with Mask-aware Caching and Scheduling

- 属于Serving调度的实现是什么？实验比较什么？
  属于Serving调度的实现包括三部分：(1) 将continuous batching迁移到diffusion model的每个denoising step级别——已完成请求在denoising step后退出，新请求在下一个step边界加入，避免等待整个batch完成；(2) 将CPU密集的image preprocessing/postprocessing拆到独立进程（disaggregated preprocessing），避免打断GPU denoising主进程；(3) mask-aware load balancing——scheduler根据请求mask ratio用离线拟合的线性模型估算computation latency和cache loading latency，选择预计pipeline latency最低的worker路由请求。实验比较baseline：static batching（Diffusers默认）、naive continuous batching（LLM-style，不拆CPU进程）、request-count负载均衡、token-count负载均衡。指标包括端到端延迟、P95尾延迟、throughput、queue time。

- 硬件平台是什么，配置是什么。
  SD2.1: NVIDIA A10 GPU。SDXL和Flux: NVIDIA H800 GPU。在线服务评测使用单台8-GPU机器（每个worker独占1张GPU），SD2.1最大batch size=4，SDXL/Flux最大batch size=8。请求到达按Poisson process生成，不同RPS测试。每条请求的mask ratio按生产trace分布采样。

- 开源Serving框架是什么。修改了什么。
  基于HuggingFace Diffusers / PyTorch，保留FlashAttn等已有优化。前端FastAPI接收请求（image template、mask、输入条件）。Scheduler与workers间使用ZeroMQ通信，continuous batching的request queues和load balance scheduler用asyncio实现。主要修改：(1) worker内实现denoising-step级continuous batching逻辑——在每步去噪后检查已完成请求并退出，从队列拉新请求加入running batch；(2) 拆分出独立preprocessing/postprocessing进程——编码/解码latent不与GPU denoising竞争，通过进程间通信传递数据；(3) scheduler扩展为mask-aware——接收worker状态上报，用离线拟合的线性模型估算各worker的computation latency和cache loading latency，选择pipeline expected latency最低的worker路由。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  开源地址：https://github.com/Sylvia-16/FlashPS；Zenodo: https://zenodo.org/records/17176576。提供Docker镜像jiangxiaoxiao/flashps，包含Diffusers定制包、scheduler、baseline、scripts和mask_ratio_distribution目录。
  FlashPS Serving调度使用流：
  1. 部署：在每个GPU上启动一个FlashPS worker进程，包含model executor、cache engine和continuous batching逻辑。集群侧启动scheduler进程，通过ZeroMQ与所有worker通信。
  2. 请求到达：FastAPI前端接收请求→scheduler调用mask-aware load balance→选取预计延迟最低且有batch slack的worker→将请求路由到该worker的队列。
  3. Preprocessing阶段：独立进程将输入image和mask编码为latent（VAE encode），结果传递给GPU主进程。
  4. Denoising loop：GPU主进程在每个denoising step组成running batch——对每个transformer block执行mask-aware计算（DP决定哪些block用cache）。已完成所有steps的请求立即退出batch，交postprocessing进程解码为输出图像；新请求在step边界加入running batch。
  5. Postprocessing阶段：独立进程执行VAE decode等操作，生成最终编辑图像返回客户端。
  6. 效果：disaggregated continuous batching相比static batching和naive continuous batching将P95延迟分别降低约35%和40%；mask-aware load balance在高请求流量下相比request-granularity/token-granularity baselines将tail latency降低最多26%。

## Automated End-to-End Model Serving with Cooperative Compilation and Scheduling

- 属于Serving调度的实现是什么？实验比较什么？
  提出Infera inference server（约17k LoC C++ kernel-space module），通过cooperative compilation and scheduling实现端到端DNN inference serving。核心Serving调度实现包含：(1) Job Dispatch Unit (JDU)：从inference job queue中按FCFS dequeue jobs，基于GPU可用显存和estimated remaining time（通过#inst/IPC估算）选择目标GPU，支持overload throttling和periodic task migration；(2) Task Schedule Unit (TSU)：任务5状态机（New→Blocked↔Ready→Running→Exit），内存管理通过mm_wq异步swap-in/out（LRU后台回收，4个CUstream处理分配/释放/双向传输）。Priority scheduling：64个runqueue（priority 0-63），ddl_rq(0)用EDF、rt_rq(1-39)用FIFO、gcfs_rq(40-63)用GCFS按nice value分配instruction budget。Aging mechanism逐步提升长期未选normal任务优先级。Preemption：高优先级任务到达→发送preemption signal→TSU暂停调度保存状态→TEU响应保存HKQ/DKQ/shared memory queue；(3) Task Execution Unit (TEU)：三阶段pipeline——SelectKernels（两阶段：先选data block最大化asynchronous wavefront，再按#inst/IPC+TLP≥4+hazard分析在线回归模型选kernel）→FuseKernels（warp-level binary fusion：prologue恢复special registers+shared memory offset+barrier重组+preemption/locking/progress flags）→LaunchKernel（HKQ→GDRCopy→DKQ→daemon kernel CDP fire-and-forget launch，覆盖placeholder kernel slots实现device-side launch）；(4) Daemon Kernel：常驻一个SM，用CUDA Dynamic Parallelism device-side发射kernel，维护shared-memory double-ended queue，fire-and-forget launch消除stream tracking overhead和HoL问题（launch latency <10μs）。Preemption：host端暂停HKQ→DKQ传输、保存HKQ kernel，device端保存DKQ和shared memory kernel；in-flight kernel通过flag快速终止，kernel idempotent execution保证安全。实验对比Stream、MPS、Triton、Paella，single-model serving speedup 1.14×–1.40×（平均1.28×），multi-model serving speedup至少1.6×（uniform requests/uniform models）最高3.5×（lognormal requests/lognormal models）。

- 硬件平台是什么，配置是什么。
  Intel Xeon Gold 6330 CPU, 512 GB RAM, NVIDIA A100-PCIE-40GB GPU, Linux 6.1.0, CUDA 12.0。Latency-sensitive CPU线程使用real-time scheduling或绑定isolated CPU core并禁用中断；GPU daemon kernel独占一个SM。

- 开源Serving框架是什么。修改了什么。
  Infera inference server从零开发（C++ kernel-space module，约17k LoC），不基于现有开源serving框架。实现包括：JDU（GPU dispatch+load balancing）、TSU（priority scheduling+task state machine+memory management）、TEU（kernel select/fuse/launch pipeline）、daemon kernel（CDP-based device-side launch）、preemption机制、GDRCopy-based low-latency data transfer、driver-level placeholder kernel slot覆盖。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  未找到明确开源仓库（EuroSys 2026, DOI: 10.1145/3767295.3769392）。以multi-model mixed inference为例：
  1. 部署：启动Infera inference server（加载编译好的模型库和权重到model pool，host memory标记为cudaHostAllocWriteCombined/cudaHostAllocPortable）→ JDU初始化GPU dispatch表→ TSU初始化per-GPU task scheduler（64 runqueues）→ TEU初始化kernel fuser thread pool和daemon kernel
  2. 请求到达：用户提交inference job（如normal priority, nice=0）→ JDU检查GPU可用显存，选择estimated remaining time最小的GPU→ job在目标GPU上创建task（New状态）
  3. 内存管理：Task检查显存需求→若权重/input tensor不在GPU memory→进入Blocked状态→mm_wq等待异步swap-in（优先级排序，同优先级FIFO）→LRU后台回收旧模型腾空间→数据就绪→进入Ready状态→加入gcfs_rq
  4. 调度周期：TSU开始scheduling cycle→从gcfs_rq选择tasks组成VT→按nice value分配instruction budget→生成VTB→TEU执行VTB
  5. TEU三阶段：(a) SelectKernels：扫描data dependency DAG选择zero in-degree且maximize asynchronous wavefront G(u)的data blocks→在线回归模型估计IPC→选#inst/IPC最小且TLP≥4的kernel；(b) FuseKernels：多个CUDA binary kernel warp-level fusion→prologue恢复%tid等special registers→shared memory offset添加→barrier重组→insert preemption/locking/progress flags；(c) LaunchKernel：fused kernel入HKQ→GDRCopy copy code+args到device→覆盖placeholder kernel slot→入DKQ→daemon kernel CDP fire-and-forget launch
  6. Multi-model scenario：BERT+ViT+Inception混合请求→bursty lognormal arrival→Infera tile-level调度允许cross-model micro-kernel colocation→warp-level fusion保证SM内空间共享→比baseline快1.6×-3.5×。Infera-P preemption约10μs，Infera-R约5μs，Infera-P比REEF-N快约2.5×，比EffiSha快超一个数量级
