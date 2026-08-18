## Prefill / Decode（预填充与逐 token 解码两阶段）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LLM 推理的两个阶段：prefill（提示词全序列一次性前向，矩阵乘为 GeMM 形态、计算密集、吞吐优先）与 decode（逐 token 自回归，每步 GeMV 形态、带宽/访存受限、延迟优先），两阶段共用 KV Cache。CompAir 的两阶段硬件敏感度分析：prefill 是 compute-bound → SRAM-PIM 收益大（0.5K 长度 3.29–5.46×，加解耦列译码器后 4.1–7.89×）；decode 收益依赖 batch——batch=1 无 SRAM 收益（无复用机会），batch=64 时 2.67–6.28×；序列长度增大时 decode 的相对优势稳定在 ~2.5×、而 Curry ALU（非线性在途）贡献上升——128K 长上下文下 GPT3-175B/Qwen-72B 达 2.13–2.73×。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# prefill: X ∈ R^{seq×d}，一次前向
for layer in layers:
    Q,K,V = X @ W_qkv          # GeMM（计算受限）
    A = softmax(Q @ K^T) @ V   # attention
    X = FFN(RMSNorm(X + A))    # GeMM + 非线性
# decode 第 t 步: x_t ∈ R^d
Q_t,K_t,V_t = x_t @ W_qkv      # GeMV（带宽受限）
attn over cached K,V; FFN
```
CompAir 的映射：prefill 的 GeMM 与 batched decode 的 Q/K/V、FFN 交给 SRAM-PIM（权重驻留）；decode 的 QK^T/SV（GeMV、K/V 输入相关）与 attention 交给 DRAM-PIM；Softmax/RoPE 由 NoC 在途完成。batch=1 decode 是纯 GeMV，SRAM-PIM 无复用、收益为零。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：vLLM/SGLang 等 serving 框架以 continuous batching 混排 prefill 与 decode 请求；硬件侧按阶段切换映射（CompAir 的静态分派规则：GeMM→SRAM、GeMV→DRAM）。使用方式：prefill 优化看重算力与带宽平衡（解耦列译码器提升 bank 读带宽）、decode 优化看重 batch 复用（SRAM-PIM）与非线性开销（NoC 在途计算）；TP 提高对两阶段都削薄每 bank 的 batch/复用（TP≤8 最优）。CHIME 的 AFD 视角：两阶段按设备拆分——prefill attention 在 GPU、decoding attention 卸载到 DIMM-PIM（CHIME 的 sub-batch 调度让一个子批的 decode attention 与另一子批的 prefill+FC 并发）；decode 阶段是带宽密集且不随 batch 复用（各请求 KV 独立），正适合 PIM。

ConServe 补充视角（ISCA'26）：multi-turn 的 prefill 是增量 prefill——新 turn 只 prefill 新增 token（每轮 512），同时 attention 读全部历史 KV（ShareGPT 新 turn 99% token 来自历史）；因此 multi-turn prefill 的访存主体是历史 KV 的读，其虚拟布局（连续 vs 散页）决定翻译局部性：FlashInfer-paged 相对 native 的 prefill 时间随轮数从 1.2× 升至 1.75× 后饱和。decode 每 token 读全部历史 KV（内存受限），即便数据 L2 命中，VA 仍需翻译——TLB miss 的页走查在 L2 hit 时也加长每 token 关键路径，这是 ConServe 连续布局带来 decode 吞吐 +19.4%~25.6% 的来源。

EVA 补充视角（ISCA'26）：decode 的 GEMV 形态是"权重主导访存 + 计算单元低利用率"的双重低效（M=1 时 32×32 阵列仅 1 lane 活跃）。EVA 通过向量量化（VQ）+ 码本驱动 GEMM 重构把 decode 从 GEMV 重写为 GEMM（输入向量×码本 → 输出码本 → 冲突无关查找+累加），使 decode 阶段也能填满矩阵单元（M 维从 1 扩到 V=K/d>512），同时与 prefill 兼容（同一阵列 INT8/FP16 重配）。这为"decode 必然 memory-bound"提供了一条计算侧重构路径：权重压到 2-bit 且查表转 GEMM 后，decode 关键路径从"权重带宽"转移到"加法树吞吐"（VQ-GEMM 256 cycles vs EU 4096 cycles）。

Raptor 补充视角（ISCA'26）：两阶段计算-访存特性直接决定内存基板选型——prefill 是计算受限（大矩阵乘、强权重复用），decode 是带宽/容量受限（自回归逐 token、每步读写全部累积 KV cache、复用有限）。Raptor 把 XPU 逻辑（10 PFLOPS）固定、只换内存基板评估 decode 域：3D-DRAM（100TB/s/32GB）相对 HBM（18TB/s/192GB）与 SRAM（150TB/s/4GB）在 decode 吞吐与交互性上占优（4.71×/2.44× vs HBM/SRAM，9.96× 更低 TPOT vs HBM）；MoE（DeepSeek-V3/GPT-OSS/Kimi K2）在 batch 扫描下呈现 attention 内存受限→专家加载主导→专家全激活→attention 计算主导四段行为；speech 模型（Whisper/Canary，上下文仅 448 token）为纯带宽驱动，SRAM 反而最高、3D-DRAM 次之。即"decode 内存受限"在不同基板上表现不同：SRAM 卡容量（高 TP/PP、collective 大、网络敏感）、HBM 卡带宽（TPOT 高）、3D-DRAM 两维均衡。

HybridSpec 补充视角（ISCA'26，算术强度视角的两阶段异构分配）：decode 每 token 前向的算术强度随 batch/token 数缩放——GPU 计算-带宽比 100-300，需数百 token/前向才达计算饱和（Fig.4 实测：吞吐随 token 数上升后平台化，与模型规模/系列无关）。SD 使两阶段跨两个模型：prefill/verification 在 target（XPU，高强度、容量敏感）、decode 在 draft（HB 栈，低强度、带宽敏感）；draft prefill 单独执行（参数不同、开销可忽略），target prefill 与 verification 联合批。prefill 长度差异大（几十到几千 token）造成计算不均衡 → CHK 切块；TTFT（prefill 延迟）与 TPOT（decode 延迟）是评估主指标。

从算法pipeline角度拆解：一次请求 = target prefill（XPU，计算密集）→ draft decode 多轮（HB 栈，memory-bound，每轮 1-token×批）→ target verification（XPU，一次 batched 前向）→ 循环至 EOS；算术强度在两单元间被"极化+匹配"。

实现与使用：硬件选型按阶段算术强度定（计算密集→强算力 XPU、访存密集→高带宽 HB）；调度上把 prefill 与 verification 的竞争用 PFS/CHK 仲裁（见系统架构层）。

- M100 补充视角（ISCA'26，车规 NPU 上的两阶段）：LLaMA2-7B（输入 1,024 token）在 M100（12/14 cluster 激活）上 decode 用 W4A16（权重 4-bit INT/激活 FP16）：21.34ms vs Thor-U 20ms（0.94×）——两平台 DDR 带宽同为 273 GB/s，decode 为带宽受限故性能相当（Thor-U 优势来自 NVIDIA 对开源模型的高度优化）；prefill 用 W8A8：79ms vs 154ms（1.95×）——prefill 计算密集，M100 的 tensor 级并行（TCU 8×64 MAC）+ 数据流同步（计算与搬运重叠、低同步开销）优势显现。MindVLA LLM 组件 decode 0.1ms（3×）、prefill 0.84ms（2.1×）。
MoE serving 补充视角（ISCA'26，Patterns behind Chaos）：现代 serving 趋向细粒度分离——传统 LLM 把 prefill 与 decode 分离到不同机器（DistServe 及后续），MoE 更进一步（MegaScale-Infer 把 attention 与 MoE 操作拆到不同机器取最优 batch），因此本文以 decode 阶段 MoE 层吞吐作为评估指标。核心发现（Insight 1）：prefill 阶段与 decode 阶段的专家选择高度一致——跨层/跨 token 专家对热图（Figure 6a-d）与单专家频率分布（Figure 7a）在前后缀两阶段基本相似，Spearman ρ≥0.7（多数层强相关、少数中等）；top-5/10/20 热门专家跨阶段重叠约 60%/75%/90%（Figure 7b）。含义：prefill 收集的专家信息可用于指导 decode 初始阶段（前 ~1000 token）——这正是 prefill-guided expert placement 与 PD 分离部署下"prefill 机器把专家选择信息传给 decode 机器"的基础；decode 初期生成 token 少、历史上下文稀缺，prefill 信息是唯一可用的预测来源。

  - SHyLA 补充：LLM 自回归推理 = prefill 阶段（并行处理全部 prompt token，GEMM 为主，compute-bound）+ decode 阶段（逐 token 生成，GEMV 为主、memory-bound，依赖 KVCache）。SHyLA 从系统/内存角度给出容量模型：微批大小 b、系统批大小 = 微批 × pipeline 深度；系统吞吐（系统输出 token/时间窗）随更大的微批提升（增强 Weight 复用），而每用户吞吐（=1/TBT，Time Between Tokens 的倒数）随批变大下降（每迭代计算增加）——即系统吞吐与每用户吞吐在此处互相矛盾，是 SHyLA 两阶段 DSE（SLO 约束下）要权衡的核心。数据放置上 prefill/decode 可用 PD aggregation（prefill/decode 共享 chiplet）或 PD disaggregation（分离实例）；decode 阶段 GEMV 配对（fused ATTN）避免中间 QK^T 结果写 DRAM。
- STAGE 补充视角（ISCA'26）：STAGE 用 DeepSeek-R1 推理架构（prefill/decode 分离）作为真实应用案例（Table VIII）：144 GPU 分为 4×36 / 2×72 / 1×144 集群，MoE 层用专家并行（EP）、其余层用数据并行（DP），总 batch=2048。结论：prefill 处理长序列大 batch、compute-bound，偏好低 EP 度（减少 AllToAll 开销）；decode 短序列/步，偏好高 EP 度大集群以增大有效 batch 提升吞吐——例如 decode step time 227.5→163.7 ms（36→144 GPU）、吞吐 62.5→86.9 tokens/s/GPU；prefill 吞吐 7097→3911 tokens/s/GPU（低 EP 更高）。

Tetris 补充视角（ISCA'26，CDSP 下两阶段的并行度异构分配）：prefill 与 decode 的并行偏好相反——prefill 受益于小 TP（SP 分配更灵活，调整 SP 只重分 token、不需重分片权重）与可变 SP（长请求大 SP、短请求小 SP）；decode 受益于大 TP（压低计算延迟，TP=8 vs TP=1/2/4 的 decode 延迟低 1.93×-5.73×，LLaMA3-8B 实测），且 SP 对 decode 不如 TP 有效（(SP8,TP1)/(SP4,TP2)/(SP2,TP4) 相对 (SP1,TP8) decode 延迟高 1.83×/1.41×/1.15×，因 decode attention 计算量小、无法掩盖 ring 通信）。Tetris 据此采用 prefill-decoding 解耦：prefill 统一 SP 池（TP=1 for 8B）+ decoding 大 TP DP（TP=8），同一请求的 prefill 用 CDSP 多 chunk 不同 SP、decode 固定在大 TP 实例上 continuous batching。评估指标即两阶段各自 SLO：TTFT（prefill，含排队+计算）与 TBT（decode 每 token 延迟，P50/P99）。

Understanding Inference Scaling 补充视角（ISCA'26，reasoning 负载的 prefill/decode 资源发散量化）：推理模型（OSL≫10k、CoT 长轨迹）使系统 >99% 墙钟时间花在 decode 阶段。遥测（8B/405B/671B，batch 100–2000）显示两阶段如同"两台机器"：prefill compute-bound——SM 占用高、HBM 带宽仅 ≈30%（8B）/≈20%（405B、671B），算术强度高（GEMM 权重复用），KV footprint 低且瞬时；decode bandwidth-bound——HBM 带宽饱和 ≈85%（8B）/≈65%（405B），算术强度塌缩（每 token 需读全部权重+活跃 KV），KV 持续累积。该发散是论文主张"prefill/decode 架构解耦"（prefill 高 TFLOP 加速器 + decode 内存中心化层次）的物理依据。
涉及论文标题：
- Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism
- Understanding Inference Scaling for LLMs Bottlenecks, Trade-offs, and Performance Principles
- P3-LLM An Integrated NPU-PIM Accelerator for Edge LLM Inference Using Hybrid Numerical Formats
- Scalable Synthesis of Distributed LLM Workloads Through Symbolic Tensor Graphs
- HybridSpec: Exploiting Hybrid-Bonding Memory to Accelerate LLM Serving through Heterogeneous Architecture and Speculative Decoding
- Bridging Efficiency and Scalability in LLM System via 3D Hybrid PIM with 2D In-Transit Computation
- CHIME: A Case for Efficient Long-Context Attention-FC Disaggregated Inference with DIMM-PIM
- ConServe: Contiguity-Preserving Memory Management for Multi-Turn LLM Serving
- EVA: Accelerating LLM Decoding via an Efficient Vector Quantization Architecture
- Early Silicon of Raptor: The First 3D-DRAM Accelerator for Generative Inference
- M100: An Orchestrated Dataflow Architecture Powering General AI Computing
- MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity

MERIDIAN 补充视角（ISCA'26，KV-precomputed RAG 的两阶段变形）：RAG 的 prefill 只编码短 query（平均 ~16 token），文档 KV 已预计算复用，因此 prefill 退化为低算术强度的 skinny GEMM（17×d_model，batch 复用有限）；decode 每步做 query KV 与整份文档 KV（数千到上万 token）的注意力 GEMV，加上小 batch 下 FFN 也退化为 GEMV——attention 与 FFN 全程 memory-bound（roofline 图 3），H100 类加速器算力严重闲置。MERIDIAN 的对应：(a) 算法层——文档注意力分解把文档侧注意力移到 PIM 就地执行，decode 每步只搬 query 向量、不回搬文档 KV；(b) 硬件层——PIM 基板为 skinny GEMM 用"buffer 级复制（共享算术单元）"实现权重复用（对比 PAPI 的 full-datapath 复制），decode 的 GEMV 与非线性（GeLU/Swish 用 LUT 分段线性、softmax 专用精度硬件）全在内存侧执行。实测：MERIDIAN 通信占比 ≤6.34%（baseline 最高 93.40%），吞吐相对 CPU-GPU baseline 5.36×/6.64×、相对 PIM baseline（CENT/PAPI）3.98×/3.32×。
- Patterns behind Chaos: Forecasting Data Movement for Efficient Large-Scale MoE LLM Inference
