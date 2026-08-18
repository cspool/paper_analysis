## TTFT / TPOT 与 SLO Attainment（LLM serving 延迟指标与 SLO 达标率）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TTFT（Time to First Token）= 请求到达→首个 token 返回的时间，反映 prefill 阶段延迟；TPOT（Time per Output Token）= 后续 token 间平均间隔，=(E2EL−TTFT)/(输出 token 数−1)，反映 decode 阶段延迟；E2EL 为请求总延迟。SLO Attainment = 满足 SLO 约束的请求占比（典型约束：TTFT(r)≤S_TT(r) 且 TPOT(r)≤S_TP(r)）；goodput = 满足 SLO 的完成请求速率，比裸吞吐更能刻画"服务有效产出"。ConServe 的 SLO 定义：以无争用基线（到达率 0.01）测得的 25×TTFT 与 TPOT 作为阈值。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
ConServe 的指标使用：在线 serving（Poisson 到达率扫描）看 mean TTFT、p99 TTFT、decode throughput、SLO attainment；离线看端到端吞吐（prefill+decode 的 tokens/s）。论文观察：(1) TTFT 低到达率平坦、近饱和急剧上升（排队放大）；(2) p99 TTFT 近饱和比 mean 更陡（长前缀 prefill 的异构性放大尾部）；(3) decode 吞吐差距与单步延迟不同源——即便 L2 hit，VA 仍需翻译，TLB miss 的页走查在每 token 关键路径上，连续布局减少翻译次数；(4) SLO attainment 增益随模型 KV footprint 增大（Yi-34B +19% vs Yi-6B +12%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TTFT/TPOT 是 LLM 服务标准 SLO 维度（业界经验阈值：chatbot TTFT ~500 ms、TPOT 对齐人类阅读速度；TTFT p95 <2 s 交互级）。使用方式：容量规划（到达率扫描找饱和点）、百分位（p95/p99）与 goodput 度量、SLO 感知调度（如 SCORPIO/Tropical 以 SLO 达标为目标调度）。Web 证据：BentoML LLM Inference Handbook（https://origin.bentoml.com/llm/llm-inference-basics/llm-inference-metrics）给出 TTFT/TPOT 公式与 SLO 约束示例；SCORPIO 论文给出 Adherence=|R^good|/|R| 的正式定义。

DynoPipe 补充视角（ISCA'26）：评估指标为 TTFT、token 吞吐、E2E 延迟（mean/P50/P99）与 TPOT，并显式讨论"TPOT 与吞吐表面矛盾"——TPOT 反映单 token 顺序解码（边云穿越开销使 LLaMA2-7B TPOT 54→73ms），吞吐反映 pipeline 并行的并发处理（10.1× 提升来自排队占比从 62%→26% 的减少）。TTFT 较 FlexNN 降 98.5%（LLaMA3.1-8B，QPS 2-8 下 68.53s→0.74s；摘要写作 98.9%），MAF trace 下 P99 较 CloudOnly/FlexNN/EdgeShard 降 54%/60%/16%（LLaMA2-7B）、66%/76%/90%（Whisper-V2）；2.2× 的跨分位延迟变化证明动态编排的尾部控制。

ENEC 补充视角（无损压缩驱动的单卡端到端推理指标）：ENEC 把 TTFT/TPOT 用作"权重压缩能否消除 CPU-NPU 传输瓶颈"的端到端度量（非多请求调度场景）：单张 910B2（64GB HBM）装不下 Qwen3-32B/Falcon-40B 全量权重，baseline 采用"大部分权重在 NPU + 部分 CPU offload"，每层前向需从 CPU 经受限 NPU-CPU 带宽拉取权重，内存访问占 prefill/decode 78%–85% 执行时间；ENEC 把权重离线无损压缩（CR≈1.35）后由 AIV 实时解压（217 GB/s），并让下一层解压与当前层 forward 重叠。指标口径：固定输入/输出长度，10 次 warm-up + 50 次 test 取平均 TTFT/TPOT，逐层解压 overlap。结果：Qwen3-32B TTFT 4.1×、TPOT 3.9× 优于无压缩 baseline（相对 HANS 1.7×/1.6×）；Falcon-40B TTFT 6.3×、TPOT 4.9×（batch=4 时某些层权重传输占执行时间近 80%）。说明 TTFT/TPOT 也可作为"数据搬运瓶颈是否被消除"的延迟证据，而不只是 SLO 调度目标。

Raptor 补充视角（ISCA'26）：指标口径与部署绑定——TPOT（Time Per Output Token）是 decode 每个 token 的平均延迟，反映会话级交互性；interactivity = 1/TPOT（1/s，越高越好）；吞吐分两级：tok/s/usr ≈ 1/TPOT（每用户每秒 token），tok/s/card = U/TPOT（每加速卡每秒 token，U 为并发用户）；TTFT 度量 prefill 初始响应。Raptor 在 decode 内存受限域评估"吞吐-交互性"权衡（图 14/17/18/19/20）：每曲线扫 batch（增大 batch 升 tok/s/card、降 interactivity），batch=32 由 Poisson 到达分析（110 req/s、1ms per-batch 延迟下 max/avg batch 均 <32）定为实际工作点。结果口径：Raptor 3D-DRAM 平均 9.96× 更低 TPOT（更高 interactivity）与 4.71× 更高 tok/s/card vs HBM，2.44× vs SRAM；4K 上下文现实网络（0.5µs/1TB/s）下 4.38× vs HBM、3.15× vs SRAM。注意与 SLO 调度类用法的区别：Raptor 用这些指标做"内存基板×并行度"的设计空间对比，而非在线调度目标。

HybridSpec 补充视角（ISCA'26）：论文按序列长度分箱的 ShareGPT 负载评估 TTFT/TPOT（mean/median/P99），发现所有方案先稳定、过饱和点后急剧上升，且长序列饱和点更低（每请求迭代多、负载重）；HybridSpec 的 TTFT/TPOT 增长更慢、可服务更高请求率——同 SLO 约束下比 GPU baseline 多支撑 2.97× 请求率。容量视角：HB-ATTEN 的 TTFT 随请求率/序列长度剧增（HB DRAM 容量耗尽、请求挂起），HB-ATTEN-QT 靠 4-bit KV 量化缓解；HybridSpec 把 target KV 全放 LPDDR5X、HB 只放小的 draft KV，避免容量 stall。调度消融用 SLO 权衡度量（PFS vs FIFO 的 TTFT-TPOT 曲线，图 18）。

从系统架构角度拆解：SLO 指标同时是"设计评估"与"调度目标"——HybridSpec 用它对比四 baseline 的饱和点与请求率支撑能力，也用它度量仲裁策略的权衡（FIFO TPOT 略优但 TTFT 差、高负载反超）。

实现与使用：用 vLLM benchmark_serving.py 从 ShareGPT 生成请求负载、按请求率（1-4/设备）控制负载（比固定 batch 更贴近在线到达）；请求率数百则属集群级资源、不在单设备评估。

P3-LLM 补充视角（ISCA'26，量化运行时开销的 TTFT 论证）：P3-LLM 用 TTFT SLO 论证其动态输入感知 KV cache 平滑的运行时开销可忽略——chatbot 场景 250ms 的 time-to-first-token 约束下，平滑因子计算（prefilling 阶段每通道绝对最大值）在 A6000 上跑 Llama-3.1-8B 全层 <5ms（即使 32K context），运行时开销 <2% TTFT。即 TTFT 在此处作为"在线量化/平滑算法能否满足交互式延迟预算"的度量，而非多请求调度目标——与 serving 系统的 SLO 调度类用法不同，但共用同一延迟指标衡量算法流水线的可部署性。

PowerWeave 补充视角（ISCA'26，TTFT/TPOT 作为空间 DVFS 的 slack 度量）：PowerWeave 把 TTFT（prefill 延迟，compute-bound、对频率敏感）与 TPOT（decode 每 token 延迟，memory-bound、对频率不敏感）作为每个频率域 SLO 的量化 slack 来源——Governor 用 slack 公式 s₂=((1−s₁)×l₁)/SLO 把当前延迟换算成可容忍退化，DVFS Controller 据此为 prefill 域/decode 域分别选频率（TTFT 逼近 SLO 时只升 prefill 域，TPOT 有 slack 时 decode 域保持低频）。SLO 配置采用 MLPerf Inference 5.1（interactive：0.5s TTFT / 30ms TPOT，用于 ShareGPT 短输入；server：2s TTFT / 100ms TPOT，用于 scientific papers 长输入）+ Azure trace 用 DynamoLLM SLO（<256 token：250/100ms；<1024：400/100ms；≥8192：2000/100ms TTFT/TPOT）。多 SLO（TTFT 与 TPOT、不同输入长度）时 Governor 取最保守 slack。效果：disaggregated prefill 平均节能 28%（vs LithOS 13%），Qwen3-32B-FP8 因 TTFT slack 极小而 decode 被强制过供给，相对 LithOS 达 8× 节能——TTFT/TPOT 的 slack 差异直接转化为空间 DVFS 的收益空间。

Rearchitecting-the-Datacenter-Lifecycle 补充视角（ISCA'26，TBT 与 goodput 在容量规划中的用法）：论文用 TTFT 与 TBT（Time-Between-Tokens，decode 每 token 延迟，对应本条目 TPOT 的另一口径）作为 roofline 模型的预测输出与 SLO 约束——2K 序列、batch 8、TP1/TP4/TP8 下用 vLLM 实测 Llama3 1B–405B 在 T4/V100/A100/H100/H200 上的 TTFT/TBT（按 H200 归一化），发现 decode（TBT）比 prefill（TTFT）对 GPU 代际更不敏感（计算需求低）；同一 SLO（TTFT≤400ms、TBT≤100ms）约束下，把负载提升到违反 SLO 前的最大 RPS 定义为 goodput，作为"该 GPU 代际能服务多少请求"的容量指标，进而驱动刷新决策（70B 模型 A100 goodput/Watt 比 H100 低 ~3×10^10，小模型则 A100 反超）。TBT/TTFT 在此是跨代际硬件对比与 15 年容量规划的度量，而非单实例在线调度目标。

- Symbiotic MLLM Serving: Dynamically Balancing Parallelism Across GPUs and Resources Within GPUs
RESONATOR 补充视角（ISCA'26，MLLM 的 TTFT/TPOT 与 encoder 共享/动态并行）：RESONATOR 用 TTFT/TPOT/E2E latency/throughput 四指标评估 MLLM serving，TTFT 的 MLLM 特性是"vision encoder 进入 prefill 关键路径"——encoder 延迟占比随图像分辨率上升（Figure 3，高分辨率下成为 prefill dominant 瓶颈），故 TTFT 改进来自两处：Inter-GPU 动态 DP/TP（简单请求走 DP 减少 HoL 阻塞、复杂请求走 TP）与 Intra-GPU 共享（encoder 与 prefill 并行填洞）。TPOT 反映 decode 阶段，改进来自回收 TP 通信等待间隙的 SM 给 co-located decode。SLO 用法：论文用"TPOT P99 在 1 秒窗口内 ≤250ms"作 decode 质量 SLO——encoder-decode 共存下 stream-only 共享违例率 20%（21/103 窗口，最坏 TPOT P99 28s），加 Intra-GPU Sharing Engine 后 5%（5/100 窗口，最坏 479ms），median TPOT P99 从 193ms→118ms；典型数值：Kimi-VL-16B@10RPS mean TTFT 11.6s vs SGLang 43.5s/vLLM 59.7s（P99 14.7s vs 70.0/104.8s），Qwen2-VL-72B@8RPS mean TPOT 200ms vs 342/600ms。消融显示 TTFT 与 TPOT 由不同机制主导：TTFT 靠 Inter-GPU 并行（13.7×@4RPS），TPOT 靠 Intra-GPU 共享（2.6×@2RPS、完整系统再降到 42.7ms）。

Tetris 补充视角（ISCA'26，TTFT/TBT 作为 CDSP 的调度目标与评估指标）：Tetris 以 TTFT（prefill，反映 CDSP 调度的直接收益）与 TBT（Time-Between-Tokens，decode 每 token 延迟，对应本条目 TPOT 的另一口径）为核心 SLO，报告 P50/P99；TTFT 被拆为排队延迟+prefill 计算延迟（延迟模型估计），CDSP 调度的目标即最小化全局 TTFT 分布。三个 TTFT 洞察驱动设计：(1) 短请求大 SP 过度供给→TTFT 变差（资源浪费+ring 通信无法掩盖）；(2) 长请求小 SP→prefill 数十秒级（SP 从 16 降到 1 时 256K prefill 差距可达 43s）；(3) 高负载下排队延迟主导 TTFT→过度 SP 扩张反而恶化全局分布，需 improvement rate 控制。评估方法：真实线上 trace（ByteDance Doubao，Short/Medium/Long 三条长度分布）按 25× light-load 延迟归一化，压力测试缩放到达时间戳模拟负载；结果 TTFT 最高降 4.35×、median TBT 降 40.1%、容量提升 45%（对比 LoongServe/Fixed-SP）。
Understanding Inference Scaling 补充视角（ISCA'26，reasoning 负载的逆缩放定律）：并发（max_num_seqs）从 1K→10K 时 TTFT 与 TPOT 反向演化——TTFT（队列受限）在最大并发处最小（≈2.7s，新请求更快准入）；TPOT（带宽+容量受限）随并发线性恶化（0.08s→0.48s，KV 压力引发抢占加重 decode）。净效果是 E2E 凸曲线、≈2K 并发为最优甜点（低并发区 E2E 由排队延迟主导、高并发区由执行稀释+抢占主导）。结论：最优工作点是"TTFT 降低不再补偿 TPOT 恶化"的 batch 大小，motivate 用 TTFT/TPOT/KV 占用/HBM 带宽做反馈信号的在线 batch 调优。这与 SLO 调度类用法互补——此处 TTFT/TPOT 被用于刻画并发-容量权衡的凸性而非达标率。
涉及论文标题：
- Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism
- Understanding Inference Scaling for LLMs Bottlenecks, Trade-offs, and Performance Principles

- HybridSpec: Exploiting Hybrid-Bonding Memory to Accelerate LLM Serving through Heterogeneous Architecture and Speculative Decoding
- ConServe: Contiguity-Preserving Memory Management for Multi-Turn LLM Serving
- DynoPipe: Heterogeneous Edge-Cloud LLM Serving with Dynamically Orchestrated Pipeline Boundaries
- ENEC: A Lossless AI Model Compression Method Enabling Fast Inference on Ascend NPUs
- Early Silicon of Raptor: The First 3D-DRAM Accelerator for Generative Inference
- MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition
- P3-LLM An Integrated NPU-PIM Accelerator for Edge LLM Inference Using Hybrid Numerical Formats
- PowerWeave: Unlocking Energy-Efficient ML on GPUs with OS-Level Spatial Power Management
- Rearchitecting the Datacenter Lifecycle for AI
STEP 补充视角（ISCA'26，TTFT/TPOT 作为专家卸载/预取有效性的延迟证据）：STEP 在 batch=1 实时推理下把 TTFT（prefill，输入 512 token 定长 trace）与 TPOT（decode）作为"专家取数瓶颈是否被消除"的度量：baseline 下专家取数占执行 88%、offloading 时间常超计算时间（Fig.2a），STEP 经空间剪枝 + 临时共享专家预取 + 自适应窗口把命中率提到 85.5–98.8%（CNN/DM）并隐藏传输，prefill TTFT 相对 llama.cpp 几何平均 3.12×、decode 1.54×；不同 CER（25/50/75%）下优势在低 CER 更显著（冗余计算削减是关键）。与 SLO 调度类用法不同：此处 TTFT/TPOT 是"预取/剪枝调度策略有效性"的量化证据（命中率↑→TPOT↓），非多请求在线调度目标；decode 延迟分解（Fig.12）显示 STEP 相对标准 MoE 显著压缩 expert offloading 占比。

- SMoE: An Algorithm-System Co-Design for Pushing MoE to the Edge via Expert Substitution
- STEP: Adaptive Spatio-Temporal Expert Prefetching for Low-Latency and Memory-Efficient MoE Inference

MERIDIAN 补充视角（ISCA'26，RAG 的 TTFT 与延迟分解）：RAG 推理的 TTFT 被长文档上下文主导——retrieved context 可达上万 token，attention 随序列长二次增长，prefill 长上下文占计算主导并直接恶化交互式应用的 TTFT；KV 预计算（TurboRAG/BlockAttention）把文档编码移出关键路径，TTFT 可降最高 98%。MERIDIAN 进一步把 TTFT/延迟的"通信分量"压缩：集中式下文档 KV 经 PCIe 搬运占推理时间平均 48.60%、最坏 86.45%（1→4 H100 升到 72.72%），MERIDIAN 去中心化后通信占比 ≤6.34%；per-request 延迟分解（图 11，通信/prefill/decode 三阶段）显示 CPU-GPU 与先前 PIM baseline 的通信最多占 93.40%，而 MERIDIAN 只做 query 广播与轻量全局归约。指标口径：吞吐 tokens/s（batch 2/4/8/16）、每请求延迟（图 10）、阶段延迟分解（图 11）、能量（图 12）与准确率（图 13，差 <0.4pp）——TTFT/TPOT 类 SLO 不是本论文主指标，但 prefill/decode 阶段分解是其通信/计算效率论点的直接证据。

SMoE 补充视角（ISCA'26，TPOT 作为专家卸载调度目标而非 SLO 达标率）：SMoE 把 TPOT 用作"专家替换/缓存调度能否消除 PCIe 加载与 CPU 计算开销"的端到端度量——边缘低 batch（1–3）下 TPOT 被非驻留专家主导（Qwen/A6000 上 low-score 专家加载占 TPOT 42%、top-score 加载 29%、GPU 计算 29%；PCIe 比 GPU 计算慢 10–100×），SMoE 通过替换+预取+缓存淘汰把 GPU 命中率提到 >60%，TPOT 相对最优 baseline 降 24%（batch=1）/35%（batch=3）、S3 下 48%（batch=3）/34%（batch=1），TTFT 平均降 11%。指标口径：batch=1/3 解码测 TPOT、batch=1 prefill 测 TTFT；用 GPU expert cache ratio 反映显存利用效率；精度用 OpenCompass（Gaokao/triviaQA/WiC/Race-mid/gsm8k/MMLU）+ MT-Bench（GPT-4 Score）+ HumanEval（pass@1）。与 SLO 调度类用法不同：此处 TPOT 是"调度策略有效性"的量化证据（命中率上升→TPOT 下降的因果链），而非多请求在线调度目标。
