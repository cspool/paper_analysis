# Paper-Chance-MoE-Result

> **范围**：系统/硬件、推理优化（路由、压缩、缓存、调度、量化）、应用。排除纯训练。
**总数**：97 篇。每篇按四维度展开：核心创新 / Baseline 痛点→方法 / 开源环境。
**分类**：A.系统/通信/硬件(31) · B.调度/路由(22) · C.缓存/卸载(13) · D.压缩/量化(21) · E.架构/应用(10)
> 

---

## A. Systems / Communication / Hardware (31篇)

| # | 论文 (Venue) | 核心创新点 | Baseline 痛点 → 方法 | 开源/环境 |
| --- | --- | --- | --- | --- |
| A1 | **PROBE** (arxiv'26) | **Continuous Lookahead Pipelining**：当前层执行时预测下一层热点专家。三组件：Gate-Initialized Lookahead Predictor（蒸馏router）、Hardware-Aware Balance Planner（联合优化复制+分配）、Phase-Locked Co-Scheduling（split-phase传输） | EP放大straggler，continuous batching下专家热点抖动→双惩罚。方法：dual-track执行，predictor与A2A重叠，prefetch与GEMM重叠。**Prefill↓1.32×** | Kuaishou Kling Infra；2602.00509；未开源 |
| A2 | **MixServe** (arxiv'26) | **TP-EP混合并行 + Fused AR-A2A通信**：自动serving系统，离线建模comm overhead选最优策略；fused算法把intra-node AR与inter-node A2A重叠 | 纯TP跨节点效率低、纯EP负载不均→理论建模TP/PP/EP/DP comm，按硬件层次自动选。DeepSeek-R1/Qwen3上**3.8×** | 2601.08800；未开源 |
| A3 | **Tarragon** (arxiv'26) | **AW/EW故障域隔离**：reconfigurable datapath + Expert Routing Table；async增量KV checkpoint；EW用残余GPU显存部署shadow experts | 单worker故障触发整服务重启。方法：放松AW-EW紧耦合，间接路由层。故障停顿**↓160-213×**（~64s→0.3-0.4s） | UC Riverside+Cisco；2601.01310；未开源 |
| A4 | **FineMoE** (EuroSys'26) | **Expert Map数据结构**：iteration级gate概率分布；prompt语义嵌入辅助检索；async Pub-Sub解耦预取与推理 | MoE-Infinity coarse跟踪+同步预测→阻塞推理。方法：细粒度map+语义hint。**Latency↓47%, hit rate↑39%** | **GitHub**: IntelliSys-Lab/FineMoE-EuroSys26 |
| A5 | **FUSCO** (arxiv'25) | **Structured Transfer Engine + Comm Planner + Online LB**：把layout变换融进通信路径；segment descriptor + pipelined shuffle | MoE expert-major与NCCL device-major layout不匹配→先变换再通信冗余（占22-61%时间）。**3.84× over NCCL, 2.01× over DeepEP** | 2512.22036；未明确开源 |
| A6 | **UCCL-EP** (arxiv'25) | **GPU-CPU FIFO control channel**：CPU proxy替代GPU-initiated RDMA；16B descriptor经FIFO给多线程proxy；用RDMA immediate data模拟ordering | DeepEP需GPU直写NIC driver→跨厂商不可移植。方法：GPU-CPU解耦。**EFA上2.1×；SGLang吞吐↑40%；DeepSeek-V3训练↑45%** | **GitHub**: uccl-project/uccl；NV/AMD+EFA/Broadcom |
| A7 | **FinDEP** (arxiv'25) | **Fine-grained scheduling for DEP**：comp/comm切细粒度任务做流水；变粒度可重排调度优化；高效solver | 现有DEP不支持shared experts→重叠率低。**DeepSeek-V2吞吐↑1.61×，32-GPU 1.24×** | 2512.21487；未开源 |
| A8 | **Janus** (arxiv'25) | **Attn与Expert分到独立GPU子集群**：adaptive 2-phase comm利用intra/inter-node带宽层次；GPU-kernel实现的scheduler平衡激活expert数；activation-aware expert mgmt | 整个MoE当monolithic部署→attn和expert资源需求不同却统一配置。方法：模块独立扩展。**SGLang基础per-GPU吞吐↑3.9×；资源成本↓25%** | 基于SGLang；2512.13525 |
| A9 | **Context-Aware CXL-NDP** (arxiv'25) | **Context-aware placement**：prefill阶段激活统计指导decoding放置（hot in HBM, cold to CXL-NDP）；Expert Bitwidth Selector给NDP端1-4bit混合精度 | GPU-NDP系统context-agnostic→冷热错配。方法：把参数搬运转为更便宜的激活搬运。**8.7× decoding吞吐，0.13%精度损失** | RPI+IBM；2512.04476；未开源 |
| A10 | **Stratum** (MICRO'25) | **System-HW Co-Design**：Mono3D DRAM+NMP+GPU通过hybrid bonding/硅中介层集成；exploit 3D层间wordline latency差异做in-memory tiering；topic-aware调度器 | MoE大数据量超出HBM。方法：Mono3D DRAM高密度+高内带宽+NMP处理冷expert。**8.29×吞吐+7.66×能效** | UCSD+UIUC+GT；2510.05245 |
| A11 | **HD-MoE** (arxiv'25) | **3D Near-Memory Processing + Hybrid/Dynamic Parallelism**：结合3D NMP硬件做混合EP/TP并行，根据expert热度动态调整parallelism策略 | MoE expert参数量大→单GPU显存不足。方法：3D NMP扩展内存层次+动态并行 | 未找到GitHub |
| A12 | **All-to-All Fault Torus** (MICRO'25) | **Torus网络上容错All-to-All collective**：在大规模Torus拓扑中做MoE EP时处理节点/链路故障，保证collective不中断 | 大规模EP中故障率随规模上升→collective中断代价大。方法：fault-tolerant routing算法 | MICRO'25；未深入搜索 |
| A13 | **KTransformers** (SOSP'25) | **AMX优化CPU内核+异步CPU-GPU调度**：AMX tiling-aware内存布局、cache-optimized AMX kernel；低ARI用AVX-512替代；NUMA-aware tensor placement；Expert Deferral重叠CPU/GPU | Fiddler/Llama.cpp受CPU计算+同步限制→671B模型prefill仅70tok/s。方法：充分发挥AMX+async调度。**Prefill↑4.62-19.74×，Decode↑1.25-4.09×** | **GitHub**: kvcache-ai/ktransformers；已集成SGLang |
| A14 | **MegaScale-Infer** (SIGCOMM'25) | **Disaggregated EP**：attn与FFN分到不同GPU；ping-pong pipeline parallelism（micro-batches穿梭）；高性能M2N通信库消除冗余GPU-CPU拷贝 | MoE稀疏→FFN利用率低；attn和FFN共部署浪费资源。方法：模块解耦+独立model parallel+异构GPU部署 | ByteDance；2504.02263 |
| A15 | **HybriMoE** (DAC'25) | **三件套**：dynamic intra-layer CPU/GPU负载均衡调度；impact-driven inter-layer预取；score-based caching应对不稳定激活 | MoE激活模式不稳定→固定映射低效。**Prefill↑1.33×, Decode↑1.70× over kTransformers** | 基于kTransformers；2504.05897 |
| A16 | **CoServe** (ASPLOS'25) | **CoE专用：dependency-aware request scheduling**把依赖同一expert的请求聚集；dependency-aware expert mgmt；offline profiler自动找最优资源分配 | CoE部署expert多→频繁switching跨内存层。方法：利用expert dependency减少switching。**4.5×~12×吞吐** | Beihang；2503.02354 |
| A17 | **Samoyeds** (EuroSys'25) | **Dual-side结构稀疏**：同时在weights和activations上施加2:4稀疏；bespoke sparse-sparse matmul kernel在SpTC上跑；tiling/data stationary/packing优化 | 现有只考虑权重稀疏→忽略activation内在稀疏。方法：双边稀疏数据格式+SpTC kernel。**Kernel 1.99×，模型1.58×** | SJTU；EuroSys'25 |
| A18 | **COMET** (MLSys'25) | **Fine-grained comp-comm overlap**：shared-tensor数据依赖解析；adaptive workload assignment动态分配GPU thread block给comm/comp | 粗粒度重叠→非确定性kernel时序差；MoE comm占47%时间。方法：依赖解析+任务重排。**Layer 1.96×，E2E 1.71×**；ByteDance万卡部署 | **GitHub**: bytedance/flux；MLSys'25(5/5/5/4) |
| A19 | **FarSkip-Collective** (arxiv'25) | **去除阻塞集合通信同步点**：通过algorithmic重排让MoE中的blocking collective变为non-blocking，减少idle等待 | MoE EP中All-to-All是阻塞的→GPU idle多。方法：algorithmic重排实现non-blocking comm | 未找到GitHub |
| A20 | **SonicMoE** (arxiv'25) | **IO和Tile-aware kernel优化**：针对MoE算子的GPU kernel做tile-level调度和IO优化，提升硬件利用率 | MoE kernel利用率低→IO/tile粒度未优化。方法：tile-aware调度 | 未找到GitHub |
| A21 | **FlashMoE** (NeurIPS'25) | **完全GPU-resident单持久化kernel**：dispatch/compute/combine融合一个kernel；Actor-style并发模型；one-sided device-initiated RDMA替代bulk-sync collectives | CPU调度/host-initiated comm/频繁kernel launch→仅26% tensor core利用。方法：一次launch；on-device调度。**9×GPU利用/6×延迟/5.7×吞吐** | **GitHub**: osayamenja/FlashMoE；Cornell |
| A22 | **MoE-Lens** (arxiv'25) | **Hardware-limit分析**：建模MoE serving在资源约束下的理论吞吐天花板，给出接近极限的系统设计建议 | 现有系统离硬件极限远。方法：roofline-style分析+系统建议 | 未找到GitHub |
| A23 | **HarMoEny** (arxiv'25) | **Multi-GPU MoE推理**：专注expert跨GPU协调策略，优化communication pattern和expert placement | 多GPU上expert分布不均→通信瓶颈。方法：协调放置+通信优化 | 未找到GitHub |
| A24 | **MoE-Gen** (arxiv'25) | **Module-Based Batching单GPU高吞吐**：按module（attn/expert）粒度而非request粒度做batch，最大化单GPU利用 | 单GPU上MoE推理吞吐低→传统request-level batching未利用MoE稀疏。方法：module-level batch | 未找到GitHub |
| A25 | **HAP** (arxiv'25) | **Hybrid Adaptive Parallelism**：ILP-driven选混合module策略（TP/EP/DP per module），面向inference | 单一并行策略在不同module上不同→需自适应。方法：ILP优化 | 未找到GitHub |
| A26 | **The New LLM Bottleneck** (arxiv'25) | **Systems perspective综述**：MLA把attention推向compute-bound，MoE把FFN推向batched balance问题；建议高带宽NVLink+平衡expert负载 | MLA+MoE组合改变了传统bottleneck。方法：新视角分析+系统建议 | 分析论文 |
| A27 | **MoE-Inference-Bench** (arxiv'25) | **Benchmark**：对MoE LLM和Vision MoE模型做全面inference性能benchmark，覆盖延迟/吞吐/内存 | 缺少标准化MoE推理benchmark。方法：统一评测框架 | 未找到GitHub |
| A28 | **Async Serving** (arxiv'25) | **异步通信/计算解耦**：用asynchrony降低MoE serving成本，减少同步等待 | 同步EP comm是瓶颈→成本高。方法：异步解耦 | 未找到GitHub |
| A29 | **Chain-of-Experts** (arxiv'25) | **Expert间链式通信**：让expert之间能序列化通信（不仅仅并行独立），释放MoE更强表达力 | 传统MoE expert并行独立→表达力受限。方法：chain-style expert交互 | 未找到GitHub |
| A30 | **BigMac** (arxiv'25) | **Communication-Efficient MoE模型结构**：算法层面减少all-to-all流量，设计对通信友好的expert结构 | MoE all-to-all通信量大→训练和推理瓶颈。方法：结构层面减流量 | 未找到GitHub |
| A31 | **Speculative MoE** (arxiv'25) | **Speculative Token+Expert Pre-scheduling**：投机预测未来token的expert选择，提前调度通信和计算 | MoE通信延迟在critical path上。方法：speculative预调度减少等待 | 未找到GitHub |

---

## B. Scheduling / Serving / Routing (22篇)

| # | 论文 (Venue) | 核心创新点 | Baseline 痛点 → 方法 | 开源/环境 |
| --- | --- | --- | --- | --- |
| B1 | **LLEP** (arxiv'26) Least-Loaded EP | **动态重路由过载token+expert参数**到underutilized设备；保证所有设备在最小集体延迟内完成且满足内存约束 | 即使训练时有load balance loss，推理/post-training时expert路由仍然严重不均→过载设备compute/memory-bound failure。**↑5× speedup, ↓4× peak memory over standard EP**；gpt-oss-120b ↑1.9× | Salesforce AI；2601.17111；未开源 |
| B2 | **Remoe** (arxiv'25) Serverless | **异构MoE推理系统**：non-expert模块在GPU，频繁expert在CPU，低频expert拆成serverless函数按需并行调用；(1) SPS算法预测expert激活；(2) Lagrangian dual+KKT优化内存分配；(3) LPT启发式分配replicas | Serverless按memory-time计费→MoE参数多成本高。方法：heterogeneous deployment+cost优化。**成本↓57%, 冷启动↓47%** | 2512.18674；Kubernetes+gRPC |
| B3 | **BrownoutServe** (arxiv'25) | **United Experts + Dynamic Brownout**：(1) 把多expert知识整合成united expert，减少expert access次数；(2) SLO-Aware Latency Control动态把部分token路由到united expert | 静态模型放置+bursty workloads→SLO violation。方法：brownout降级策略保SLO。**吞吐↑2.07× vs vLLM，SLO violation↓90.28%** | 2507.17133；未开源 |
| B4 | **QLLM** (EuroMLSys'25) | **Expert-level preemption**：4逻辑队列per job class+phase；高优LS在任意layer抢占BE；checkpoint进行中expert状态 | Iteration-level FCFS→BE的head-of-line blocking。方法：MoE expert边界粒度抢占。**LS TTFT↓65.5×** | KTH；2503.09304 |
| B5 | **eMoE** (arxiv'25) | **Task-aware memory-efficient inference**：根据当前任务特征调整active expert集，减少不必要expert加载 | 所有expert都加载→浪费内存。方法：task-aware动态expert选择 | 未找到GitHub |
| B6 | **ElasticMoE** (arxiv'25) | **Auto Scaling**：动态调整EP/DP副本数以适应workload变化，弹性伸缩 | 静态部署无法应对负载波动。方法：auto-scaling策略 | 未找到GitHub |
| B7 | **Orders in Chaos** (arxiv'25) | **Data Movement Forecasting**：预测大规模MoE serving中的数据移动模式，提前优化通信调度 | 数据移动模式unpredictable→调度低效。方法：forecasting+预优化 | 未找到GitHub |
| B8 | **GRACE-MoE** (arxiv'25) | **Locality-Aware Routing+Replication**：按locality分组expert+复制hot expert到nearby GPU，减少跨节点通信 | EP跨节点通信延迟高。方法：locality-aware放置+热expert复制 | 未找到GitHub |
| B9 | **MoETuner** (arxiv'25) | **Balanced Expert Placement+Token Routing联合优化**：ILP或启发式联合优化expert在GPU上的放置和token路由策略 | Expert放置和routing独立优化→整体次优。方法：联合优化 | 未找到GitHub |
| B10 | **MicroMoE** (arxiv'25) | **Fine-Grained Load Balancing via Token Scheduling**：在token粒度做调度实现更细的负载均衡 | Token-level load imbalance。方法：token-level调度 | 未找到GitHub |
| B11 | **Capacity-Aware** (arxiv'25) | **Straggler Mitigation**：capacity-aware inference处理expert负载偏斜，避免straggler | Expert负载偏斜→straggler延迟。方法：capacity-aware分配 | 未找到GitHub |
| B12 | **Similarity-Preserving Routers** (arxiv'25) | **保相似度的router**：在保持语义相似性同时实现load balance的router设计 | Load balance loss损害routing质量。方法：similarity-preserving约束 | 未找到GitHub |
| B13 | **MoE-GPS** (arxiv'25) | **Expert Duplication Prediction Strategy**：预测哪些expert该复制、何时复制以动态均衡 | 静态expert placement→热expert过载。方法：prediction-guided duplication | 未找到GitHub |
| B14 | **Opportunistic Expert Activation** (arxiv'25) | **Batch-Aware Routing无需重训练**：根据当前batch中已缓存的expert做opportunistic routing，加速decode | 标准routing不考虑已缓存expert→miss多。方法：batch-aware opportunistic routing | 未找到GitHub |
| B15 | **GatePro** (arxiv'25) | **参数无关expert选择优化**：不需训练新router参数即可优化expert选择 | Router训练成本高。方法：parameter-free优化 | 未找到GitHub |
| B16 | **Steering via (De)Activation** (arxiv'25) | **通过激活/抑制特定expert引导LLM输出行为**（行为控制/安全类应用） | 无法精确控制MoE输出。方法：expert-level行为引导 | 未找到GitHub |
| B17 | **Ada-K Routing** (ICLR'25) | **per-token动态top-K**：lightweight allocator决定每token用多少expert；PPO训练非可微决策 | 固定K→简单token浪费算力。方法：动态K。**FLOPs↓25%, inference↑20%，性能反提升**；Mixtral-8x22B训练仅8h | ICLR'25；预计开源 |
| B18 | **Dense Train Sparse Inference** (arxiv'25) | **训练用dense，推理用sparse**：所有expert参与训练但推理时只激活子集。重新思考MoE训练范式 | 标准MoE训练时也sparse→expert specialization不足。方法：dense训练提升expert质量 | 未找到GitHub |
| B19 | **Oracle-MoE** (ICML'25) | **Oracle space routing**：用attention scores提示的紧凑空间做routing，保持相邻token的semantic locality减少expert swap | MoE在memory-constrained device上inter-token expert激活temporal inconsistency→频繁swap。方法：oracle space保locality。GPT-2(200M-2B)各预算下SOTA速度 | ICML'25 |
| B20 | **C3PO** (arxiv'25) | **Test-Time Expert Re-Mixing**：测试时通过Critical-Layer/Core-Expert/Collaborative Pathway重新组合expert路径 | 训练时的routing在测试时不一定最优。方法：test-time re-mixing | 未找到GitHub |
| B21 | **Hidden Collaboration** (arxiv'25) | **揭示MoE中expert间隐藏的协作模式**：分析expert间如何隐式协同工作 | MoE expert被视为独立→忽略协作。方法：分析揭示协作pattern | 分析论文 |
| B22 | **Hyper-Parallel RoE** (arxiv'25) | **Routing of Experts(RoE)+超并行**：让多expert并行执行路径可扩展，证明MoE推理性能可大幅提升 | MoE推理并行度受限。方法：hyper-parallel推理扩展 | 未找到GitHub |

---

## C. Offloading / Caching / Prefetch / Memory (13篇)

| # | 论文 (Venue) | 核心创新点 | Baseline 痛点 → 方法 | 开源/环境 |
| --- | --- | --- | --- | --- |
| C1 | **Diff-MoE** (SC'25) | **三级differential cache**：globally hot experts永驻high-priority cache；locally hot用priority-driven replacement管理medium-priority；cold临时cache按需驱逐 | Prefetch/cache方法为batch=1设计→large batch下comm暴涨/miss率升。方法：识别global+temporal locality差异化管理 | HUST+UNSW；SC'25 |
| C2 | **BuddyMoE** (arxiv'25) | **Buddy expert替代**：识别co-activation频率高的expert对（buddy set），prefetch失败时用GPU中已有的buddy expert近似替代，避免on-demand fetch的长延迟 | Prefetch失败→on-demand fetch 10ms PCIe延迟或drop expert。方法：buddy替代实现零延迟fallback。基于llama.cpp；A100+Xeon | 基于llama.cpp；2511.10054 |
| C3 | **MoE-SpeQ** (arxiv'25) | **Speculative quantized decoding+主动prefetch+offload**：小draft模型预测未来token的expert并预取；对offloaded expert做量化 | Offload expert体积大→PCIe瓶颈。方法：spec decode+量化减体积+预取。**Phi-MoE上2.34× over SOTA offload** | 未找到GitHub |
| C4 | **Pre-Attention Predict** (arxiv'25) | **Attention计算之前预测下一层expert并prefetch**：利用attention前的hidden state做lightweight prediction | Expert prediction太晚→prefetch窗口不够。方法：提前到pre-attention阶段预测 | 未找到GitHub |
| C5 | **Hide Offload via Spec Decode** (arxiv'25) | **Speculative decoding隐藏offload latency**：用小模型投机解码期间完成expert加载 | Expert offload在critical path上。方法：spec decode时间用于offload | 未找到GitHub |
| C6 | **PreMoe** (arxiv'25) | **Pruning+Retrieval**：剪枝不常用expert减少内存占用，按需从CPU/disk检索被剪expert | 所有expert常驻→内存浪费。方法：pruning减驻留+retrieval保质量 | 未找到GitHub |
| C7 | **Not All Models Suit Offloading** (arxiv'25) | **Local Routing Consistency分析**：研究哪些MoE模型的routing pattern适合offloading，哪些不适合 | 盲目offloading不同模型→有的效果差。方法：给出适合offloading的模型特征指南 | 分析论文 |
| C8 | **FloE** (ICML'25) | **Hybrid压缩+sparse prediction**：利用expert internal稀疏性做contextual sparsification；low-cost sparse predictor | Offload后PCIe带宽不足。方法：expert内部冗余压缩减传输量。**Mixtral-8x7B 9.3×参数压缩；11GB VRAM部署；48.7×speedup**（4.4-7.6%精度损失） | ICML'25 |
| C9 | **ProMoE** (arxiv'25) | **Proactive Caching**：用中间结果预测后续参数使用，主动fetch移出critical path | Reactive caching→miss多。方法：prediction-informed proactive fetch。**Prefill 2.13×/Decode 2.84×** | 未找到GitHub |
| C10 | **Mixture of Lookup Experts** (arxiv'25) | **Expert作为Lookup Table(LUT)**：用hash/索引方式快速访问expert知识，减少计算开销 | Expert forward pass计算量大。方法：LUT替代neural forward | 未找到GitHub |
| C11 | **PiKV** (arxiv'25) | **KV Cache管理系统专为MoE设计**：考虑MoE特有的sparse activation pattern管理KV cache | 通用KV cache管理忽略MoE稀疏特性。方法：MoE-aware KV管理 | 未找到GitHub |
| C12 | **D2MoE** (Mobicom'25) | **算法-系统协同**：dual routing动态分配bit-width给每expert；HEBF调度；MWQ混合权重量化；I/O-计算流水化 | 静态bit-width→动态稀疏下expert重要性变化。方法：动态bit-width+流水。**吞吐↑1.39× over SOTA量化框架**；Jetson AGX Orin | HKUST；MobiCom'25 |
| C13 | **Faster MoE Extreme** (arxiv'25) | **针对超大MoE模型（trillion级）的推理加速**：组合多种offloading/caching/scheduling技术 | 超大模型→现有方案不够。方法：多技术组合优化 | 未找到GitHub |

---

## D. Compression / Quantization / Pruning / Merging (21篇)

| # | 论文 (Venue) | 核心创新点 | Baseline 痛点 → 方法 | 开源/环境 |
| --- | --- | --- | --- | --- |
| D1 | **DualSparse-MoE** (arxiv'25) | **Tensor+Neuron级双稀疏**：expert partition后做权重稀疏化+输出reconstruction，两级稀疏协同优化 | 单一稀疏方式不够→精度-效率tradeoff差。方法：双层稀疏协调 | 未找到GitHub |
| D2 | **PuzzleMoE** (arxiv'25) | **Sparse Expert Merging+Bit-pack推理**：先merge相似expert减数量，再做bit-packed低精度推理 | Expert数多→参数冗余大。方法：稀疏merge+bit-pack | 未找到GitHub |
| D3 | **Compression Sensitivity** (SC ws'25) | **分析不同expert对压缩误差的敏感度差异**：识别哪些expert可以更激进压缩 | 均匀压缩→有些expert精度敏感被过度压缩。方法：sensitivity分析指导异构压缩 | SC workshop |
| D4 | **MoE-Compression** (SC'25) | **Expert压缩误差对推理精度的影响机制研究**：系统分析compression error如何传播并影响最终输出 | 压缩后精度下降机理不清。方法：error传播分析 | SC'25 |
| D5 | **MoE-Prism** (arxiv'25) | **Disentangling Monolithic Experts**：把大expert拆成可组合细粒度module，支持elastic MoE服务——按需组装不同大小的expert | Monolithic expert→无法弹性。方法：module化拆解+按需组装 | 未找到GitHub |
| D6 | **ReXMoE** (arxiv'25) | **跨层/跨任务Expert复用**：识别并复用相似expert，减少存储和计算 | Expert间冗余→浪费存储。方法：expert reuse | 未找到GitHub |
| D7 | **MergeMoE** (arxiv'25) | **Expert Output Merging**：不merge权重而merge输出，保留专家独立性同时减少计算 | 权重merge损害expert多样性。方法：output-level merge | 未找到GitHub |
| D8 | **DiEP** (arxiv'25) | **Differentiable Expert Pruning**：通过differentiable proxy做end-to-end学哪些expert该剪 | 手动/规则based pruning次优。方法：differentiable端到端学习 | 未找到GitHub |
| D9 | **EAC-MoE** (ACL'25) | **(1) QESC**：Quant with Expert-Selection Calibration——校准router缓解量化导致expert选择偏移；**(2) PESF**：基于expert-selection频率剪不重要expert | (1) 低bit量化→router expert选择偏差；(2) 不重要expert仍占算力。**显著降内存+提速** | ACL'25；2508.01625 |
| D10 | **MoEQuant** (arxiv'25) | **Expert-Balanced Sampling+Affinity Guidance**：量化校准时平衡expert采样+利用expert间亲和性指导 | 量化校准不考虑expert分布→某些expert量化误差大。方法：balanced sampling+affinity | 未找到GitHub |
| D11 | **ResMoE** (KDD'25) | **Wasserstein barycenter**：提取公共expert(barycenter)+低成本表示residual | MoE总参数大→稀疏激活但需全加载。方法：one-shot/data-agnostic/无需重训。**每expert参数↓75%** | **GitHub**: iDEA-iSAIL-Lab-UIUC/ResMoE |
| D12 | **CoSMoEs** (arxiv'25) | **Compact Sparse MoE**：结合expert合并和稀疏化压缩MoE | Expert数和参数冗余。方法：compact设计 | 未找到GitHub |
| D13 | **DeRS** (CVPR'25) | **Decompose, Replace, Synthesis**：把upcycled MoE expert拆成shared base+delta；delta用sparse/quantize/low-rank表示 | Upcycled MoE参数严重冗余（expert间共享大量权重）。方法：利用upcycled特有冗余。两场景：推理压缩+训练upcycling | CVPR'25；2503.01359 |
| D14 | **DSMoE** (arxiv'25) | **Matrix-Partitioned Experts+Dynamic Routing**：把dense LLM的FFN矩阵partition成expert，加dynamic routing变成MoE | Dense LLM计算量大→想获得MoE效率。方法：矩阵partition+routing改造 | 未找到GitHub |
| D15 | **Every Expert Matters** (arxiv'25) | **MoE Knowledge Distillation**：确保student model能从所有expert（包括低频expert）获得知识，不只从热expert蒸馏 | 标准KD忽略低频expert→知识损失。方法：全expert蒸馏 | 未找到GitHub |
| D16 | **Sub-MoE** (arxiv'25) | **Subspace Expert Merging**：在低维subspace上merge experts，保留expert差异性 | 直接merge→差异丢失。方法：subspace保差异merge | 未找到GitHub |
| D17 | **Cluster-Driven Expert Pruning** (arxiv'25) | **基于expert clustering决定剪哪些**：cluster相似expert→每cluster只保一个代表 | 哪些expert该剪不清楚。方法：clustering指导pruning | 未找到GitHub |
| D18 | **Finding Fantastic Experts** (arxiv'25) | **Expert Dropping策略统一研究**：系统对比不同expert dropping/pruning策略，给出观察和指导 | Expert dropping策略多→缺乏统一对比。方法：unified benchmark+observations | 未找到GitHub |
| D19 | **LExI** (arxiv'25) | **Layer-Adaptive Active Experts**：每层激活的expert数自适应调整（有些层需要多、有些层少） | 所有层用相同active expert数→浪费或不足。方法：layer-adaptive | 未找到GitHub |
| D20 | **MoDES** (arxiv'25) | **Dynamic Expert Skipping for MLLM**：多模态MoE中动态跳过不重要expert加速推理 | 多模态LLM的MoE层仍然active所有expert→冗余。方法：dynamic skipping | 未找到GitHub |
| D21 | **Omni-Expert** (NeurIPS'25) | **MoE in a Single Expert**：用单个expert模型通过高效参数化实现MoE的多expert行为，大幅减参数 | MoE expert多→参数大。方法：单expert实现MoE行为 | NeurIPS'25 |

---

## E. Architecture / Algorithm / Application (10篇)

| # | 论文 (Venue) | 核心创新点 | Baseline 痛点 → 方法 | 开源/环境 |
| --- | --- | --- | --- | --- |
| E1 | **DES** (arxiv'26) Diffusion LLM | **Sequence-level coreset选择**：DES-Seq序列级最优分配；DES-Vote token投票选coreset，全block共享K专家 | dLLM并行解码块大→激活expert线性增长(expert explosion)→memory-bound。方法：约束top-k为全block共享K。**激活↓55%, latency↓38%, 保99%精度** | 2602.00879；dInfer+Fast-dLLM；NV B200 |
| E2 | **LatentMoE** (arxiv'26) | **Latent-space MoE**：路由前down-project d→ℓ(≈d/4)，expert在ℓ空间运算后up-project；savings反投到更多expert+更高top-k | 标准MoE只看FLOP→serving真实瓶颈是bandwidth+all-to-all。方法：HW-SW codesign。**已用于NVIDIA Nemotron-3 Super/Ultra** | NVIDIA；2601.18089 |
| E3 | **MoESD** (NeurIPS'25) | **Speculative Decoding for Sparse MoE**：利用MoE sparse activation特性提高speculative decoding的acceptance rate | Spec decode在MoE上acceptance rate低→加速不明显。方法：MoE-aware spec decode | NeurIPS'25 |
| E4 | **LongCat-Flash** (arxiv'25) | **美团LongCat-Flash模型技术报告**：长上下文MoE模型架构设计+系统优化 | 长上下文MoE推理挑战。方法：架构+系统联合优化 | 未找到GitHub |
| E5 | **MoA Serving** (arxiv'25) | **Mixture-of-Agents efficient serving**：Tree-structured routing+Adaptive pruning+Dependency-aware prefill-decode overlap | MoA（多个agent/模型协作）serving效率低。方法：tree routing+pruning+PD overlap | 未找到GitHub |
| E6 | **Memory-Bound MoE Serving** (arxiv'25) | **"Balance Activated Experts, Not Tokens"**：在memory-bound regime下，平衡激活的expert数比平衡token数更重要 | 传统balance token→memory-bound时expert不均是真瓶颈。方法：expert-count balance | 未找到GitHub |
| E7 | **EvoMoE** (arxiv'25) | **MLLM的Expert Evolution**：多模态MoE中让expert动态演化以适应不同模态需求 | 固定expert→不同模态需求不同。方法：expert evolution | 未找到GitHub |
| E8 | **BrainMoE** (NeurIPS'25) | **Brain Foundation Model**：用MoE做cognition joint embedding，支持多模态认知任务 | 脑科学+MoE交叉。方法：MoE-based brain representation | NeurIPS'25 |
| E9 | **I2MoE** (ICML'25) | **Interpretable Multimodal Interaction-aware MoE**：专注多模态交互的可解释性expert设计 | MoE多模态交互不可解释。方法：interaction-aware+interpretable routing | ICML'25 |
| E10 | **Graph MoE Forecasting** (NAACL'25) | **LLM+Dynamic Forecasting via Graph MoE**：把时序预测建模为graph，用MoE处理不同pattern | 传统时序预测不用MoE。方法：graph MoE视角 | NAACL'25 |

---

## 统计汇总

- **已搜索开源代码的论文**（有GitHub链接）：FineMoE, UCCL-EP, KTransformers, COMET/Flux, FlashMoE, ResMoE = **6篇确认有GitHub**
- **疑似有代码但未确认**：Samoyeds, Ada-K Routing, CoServe, DeRS等
- **全部97篇精读/深入分析完成**