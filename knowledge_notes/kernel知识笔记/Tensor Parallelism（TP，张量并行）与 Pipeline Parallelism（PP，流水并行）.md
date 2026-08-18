## Tensor Parallelism（TP，张量并行）与 Pipeline Parallelism（PP，流水并行）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TP = 把单层权重按输出/头维度切分到多个设备，激活广播、输出经归约汇总，通信频繁但同步细粒度；PP = 按层把模型切成多段串成流水，通信少但流水气泡多。CompAir 的发现：CENT 原实验用全 PP（单 token 延迟显著增大），CompAir 改用 8 设备 TP=8 的均衡配置；扫描 TP=1..32 发现高 TP 下 bank 利用率骤降（每 bank 分得的 batch 复用机会减少）、吞吐退化，TP≤8 为最优范围（Llama2-13B 在此范围端到端 1.5–2.14×）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CompAir 的 TP 映射例子（GQA attention）：
```
# TP 沿 seqlen 维切分 K^T/V：每 bank 处理 seqlen/TP 段
for bank b: Q_b = Q[:, b*L/TP:(b+1)*L/TP]   # L=seqlen
# SRAM-PIM 视角：seqlen 映射为 batch 维、输出维对齐 GQA group size(8)
scores_b = Q_b @ K_b^T     # QK^T 依 TP/seqlen 决定 DRAM 或 SRAM
out_b = softmax(scores_b) @ V_b
```
权衡：TP 增大 → 每 bank 的 batch 变小 → SRAM-PIM 权重复用优势被稀释，同时数据搬移增多 → CompAir-NoC 归约/在途计算收益上升。结论：TP≤8；SRAM-PIM 的性能优势来自数据复用，过度并行削弱复用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Megatron-LM 式 f/g 切分 + all-reduce；PIM 场景用 NoC 归约树替代 all-reduce（见归约树条目）。使用方式：按设备数与模型大小选 TP/PP 组合；注意 bank 利用率与 batch 复用（SRAM-PIM 权重驻留场景对 TP 尤其敏感——TP 提高直接削薄每 bank 的有效 batch）。AttAcc 基线（4×A100 + 4×HBM3-PIM）与 CENT 均以 TP/PP 做设备级并行，CompAir 的 96 设备配置在该均衡策略下与 AttAcc 吞吐相当、能耗 3.52× 更低。

ConServe 补充视角（ISCA'26）：Yi-34B-200K（Hq=56、Hkv=8、L=60）以 TP=2 跨 2×A100 NVLink 部署。TP 切分下每卡只持有部分层的 KV，ConServe 的 VA slice 按分片内层数 L_shard 分层分段（每 token KV 字节 B_tok=2·L_shard·H_shard·d_head·b），slice 大小与 resize 触发频率都随 TP 分片减小；跨卡通信由框架负责，allocator 本身不涉及 TP 通信。

DynoPipe 补充视角（ISCA'26）：PP 用于边云异构域（边缘 RTX 3090 ∥ 云端 A40），与同构数据中心 PP 的关键差异是显式建模边界 stage 开销 T_boundary（Eq.1 的 I_boundary·T_boundary，含激活序列化/反序列化、跨域传输、状态同步、格式转换）与动态 split point（LRP 按遥测切换）。TP 跨边云被判定不可行——AllReduce 开销从 NVLink 0.8ms 增至边云链路 25ms（94% 效率退化，LLaMA2-70B attention），故约束"每请求单一边云边界"（跨域延迟比域内高 10-50×）。切分只在完整 transformer block 之间（残差块内解析、跨域只传 fully-resolved hidden state、无额外 buffering/重算），数值一致性 <10⁻⁶ 相对误差。

HybridSpec 补充视角（ISCA'26，HB 栈 logic block 间 TP 的细粒度计算-通信重叠）：权重按输出/输入通道交替切分到 4 个 logic block（常规 TP 切分），但执行方式改进——不先算完所有计算再发起集合通信（图 7(a)），而是把计算分解为 tile、每个 tile 与 ring-based 通信流水重叠（图 7(b)），大部分通信延迟被计算掩盖（图 7(c) 时间线）。本质是 kernel 层的"通信计算 overlap + 环形通信"优化，把 TP 的 all-reduce 瓶颈从"每层一次同步"细化为"逐 tile 流水"。设备级 TP 只用于 HB 栈内部（draft 模型）；更大模型（Qwen3-32B/OPT-66B）在 HybridSpec 中走数据并行（每设备独立完整请求、无跨设备通信），与 GPU baseline 需 TP+all-reduce 形成对比——这是 HybridSpec 相对 GPU 的优势来源之一。

从kernel调度角度拆解（伪代码示意）：
```
# 权重沿输出通道切分到 4 个 block；计算按 tile 流水
for tile_t in tiles:                        # 沿序列/输出维分 tile
    partial = GEMM_tile(block_i, tile_t)    # 本地计算
    ring_send(block_i, partial, next)       # 与下一 tile 计算重叠
    acc = ring_recv(block_i, prev)          # 环形聚合部分和
```
每个 block 边算边把部分和沿 ring 传给邻居，末轮聚合出完整输出；通信延迟被后续 tile 计算隐藏。

实现与使用：device 级 TP 的标准实现是 Megatron-LM 式 f/g 切分 + all-reduce；HybridSpec 在 block 级用 tile+ring 流水替代整层同步 all-reduce；评估经事件驱动模拟器（silicon-derived 参数）验证通信隐藏效果。

BusyBarn 补充视角（ISCA'26，wafer-scale 上的混合并行映射）：BusyBarn 把混合并行用作 wafer-scale 部署的核心手段——把 die 划分为 die 组，die 组之间做 Pipeline Parallelism（PP），每个 die 组内部联合应用 Sequence Parallelism（SP）、Context Parallelism（CP）与 Tensor Parallelism（TP）。分层依据：TP/CP 相对 PP 的通信比更高 [41]，故把高频通信的并行放在组内（die 内 mesh 近距、低延迟），把低频的 PP 放组间（跨 D2D 链路）。inter-die 映射用 SA 把 transformer block 层按 Hamiltonian Loop 排到 die 组（适配自回归 PP 的递归数据依赖，见"层次化 SA 映射（Hamiltonian Loop / ZigZag）"条目）；intra-die 映射在组内用第二个 SA 把 SP/CP/TP 算子的数据切片分配到 core。端到端模型（GPT-NeoX-20B、OPT-30B、Qwen3-32B、Llama-3-70B、Qwen3-MoE-30B、Qwen2-MoE-57B）以三种 wafer 拓扑（HW1 5×5/HW2 7×12/HW3 8×8）评估，混合并行 + BALD 通信使端到端相对 Tangram ZigZag+Gemini+XY-YX-FT baseline 加速 1.08–2.14×（几何均值 1.40×）。

- PIPEWEAVE: Synergizing Analytical and Learning Models for Unified GPU Performance Prediction

PIPEWEAVE 补充视角（ISCA'26，把 TP/PP 的 E2E 延迟预测当作 kernel 序列建模问题）：在分布式 E2E 推理中，TP 与 PP 不仅改变 kernel 切分方式，还引入通信 kernel——TP 产生 All-Reduce（activation 归约），PP 产生 Send/Recv 原语。PIPEWEAVE 对计算 kernel 逐个用解析特征+MLP 预测（Workload Generator 按 SGLang/vLLM 的 kernel 调用逻辑生成串行 kernel 序列、假定无重叠、求和得 E2E 延迟）；对通信 kernel 用简化方法：跨不同网络拓扑与通信量 profiling 建性能基线库，再用 Random Forest 回归估计通信延迟。评估覆盖 SGLang（Qwen3-32B TP=2、Llama3.1-70B TP=4/8）与 vLLM（Llama3.1-70B TP=4&PP=2），20 种配置平均 MAPE 6.6% vs Neusight 34.7%。论文还观察到 E2E 误差可能低于 kernel 级误差（如某 baseline E2E 0.5%）：E2E 聚合大量 kernel 造成系统性误差抵消，且 E2E workload 维度更窄、常落在 baseline 预测"甜蜜点"——因此 
ShadowUpdate 补充视角（ISCA'26，UVM 评估用的 column-wise TP + output-tiled GEMM 微 kernel）：ShadowUpdate 为评估 LLM 在 multi-GPU UVM 下的表现，构造 QKV projection 与 FFN 微 kernel：GEMM 核心用 output-tiled kernel（输出矩阵沿 M 与 N 维分 tile、迭代 K 维累加，仿 cuBLAS 结构），并按输出维做 column-wise tensor parallelism——把权重矩阵沿输出维切分到各 GPU，每个 GPU 算部分和、输出跨 GPU 拼接，从而在 MGPUSim 上模拟多 GPU 执行。五个模型（Llama3-8B、Llama2-7B、Mistral-7B-v0.3、Deepseek-llm-7b-chat、Qwen-14B）的 QKV 平均 1.37×、FFN 平均 1.42×；因 GEMM 主导且 tile 执行 + 相同 job 分配策略下页级 UVM 行为一致，各模型收益稳定。另外评估了三种 CTA/线程块调度方案对 ShadowUpdate 的影响：分布式 CTA 调度（baseline，1.40×）、CTA Clustering（相邻 CTA 共调度提 L1 局部性，降迁移频率，仍 1.24×）、LADM（联合优化线程块与数据放置，1.26×）——调度改善共享/远程访问但无法消除共享页迁移引发的 re-fault。

RoCC 补充视角（ISCA'26，Column-Linear/RowLinear 张量并行的 CC 卸载）：RoCC 论文用两种线性张量并行构造评估 workload：Column-Linear（权重按输出列切分，各 rank 算自己分片后 AllGather 收集）与 RowLinear（权重按输入行切分，各 rank 出部分和后 AllReduce 归约），外加 Expert Parallelism 的 AllToAll。关键量化：PyTorch distributed+NCCL 实测 CC 占 tensor 并行执行时间 40%-70%（输入尺寸不同而变），GEMM 与 CC 顺序执行时 CC 是主要瓶颈；RoCC 把 TP 产生的 AllGather/AllReduce/AllToAll 全部卸载到 ROP，SM 全容量算 GEMM，CC-only 延迟（大消息）AllReduce 35%、AllGather 11%、AllToAll 25% 加速，RoCC-Overlap 相对顺序 baseline 平均 51% 加速。
  - SHyLA 补充（inter-chiplet 空间切分）：16-chiplet MCM 采用 Megatron 式分区——QKV Generation/FFN1 的 Weight 按宽切分（本地算完整 IA 和）、Attention Output/FFN2 按高切分（需 all-reduce 累加）；ATTN 沿 head 维（MHA）或 attention group（GQA，g<pt 时 sequence parallelism）；MoE 把专家并行 pe 嵌套在 pt 维内。pipeline 并行只在 stage 最后 block 末尾通信且可与 tensor 通信重叠。片间通信（每 block 2 次 all-reduce，经 ICNT_BW 429GB/s）相对片内访存可忽略。DSE 部署空间含 (pp, pt, pe) 与 prefill/decode 各自的并行度；PD aggregation 下 (pt,pp)=(8,2)。
- STAGE 补充视角（ISCA'26）：STAGE 用符号张量表示把 TP/PP 编码进算子形状，自动推导通信。TP 分 Row/Column 两种切分：Column TP 把权重沿输出维切 w[H,4H/tp]、每设备算 y[B,4H/tp] 局部输出、需 AllReduce 汇聚 partial sum；Row TP 切 w[H/tp,4H@1/tp]、输入侧先 AllGather 恢复完整 x、输出 y[B,4H@1/tp] 是 partial sum 再 ReduceScatter。生产者/消费者分布不一致时（如 producer 输出 [a,c@1/tp] 而 consumer 期望 [a,c]），通信匹配器自动插入 AllReduce；TP 与 SP 组合用 AllGather+ReduceScatter 替代 AllReduce。PP 用图级分布实现：把计算图按层均分为 PP 段（规则脚本按 num_stacks 划分），在跨图边上按源/目的 rank 插入 send/recv 对，不产生张量级通信。STAGE 的 Table III 用符号记法（如 x[B,tp]→w[H,4H/tp]→y[B,4H/tp]）系统枚举了 dp/tp/fsdp/hp 等分布的张量表示，并在 Table VI/VII 中验证 TP/PP 配置下算子时间与通信量误差（如 GPT-3-5B TP8-w/SP 总误差 6.7%、通信误差 0.237%）。


- Symbiotic MLLM Serving: Dynamically Balancing Parallelism Across GPUs and Resources Within GPUs
RESONATOR 补充视角（ISCA'26，encoder 的 DP/TP 动态选择 + logical sharding）：RESONATOR 把 TP/DP 用于 MLLM 的 vision encoder（ViT-675M/MoonViT，权重小、并行选择灵活），关键差异是"TP 度逐 batch 运行时选择而非 AoT 固定"——PRISM 把请求排队建模为 MCKP 按 GPU 预算选每请求 TP 度（低分辨率 1 GPU/DP 最优、高分辨率 4-TP 最优，Figure 5：encoder compute 随序列长 ~二次增长、TP 通信随 ~线性增长）；TP 切换用 logical sharding（strided GEMM 只改 ld 参数，不搬权重）实现近零开销。TP 的通信是 NVLink all-reduce 类，其等待间隙的 SM 被 Intra-GPU 引擎回收给 co-located decode（TPOT 收益来源之一）。Data Parallelism（DP，各 GPU 处理不同请求、无跨设备通信）在低分辨率/高并发下最优——DP 与 TP 的价值随负载在 latency（TP）与 concurrency（DP）间权衡，是"无单一静态并行度最优"（Figure 10 landscape：低 RPS 8TP 最优、高 RPS 低分辨率 8DP 最优、高分辨率 2DP-4TP 最优）的实证。
涉及论文标题：
- Reducing Page Faults via Invalidation-based Mapping Propagation in Multi-GPU Systems
- Scalable Synthesis of Distributed LLM Workloads Through Symbolic Tensor Graphs
- HybridSpec: Exploiting Hybrid-Bonding Memory to Accelerate LLM Serving through Heterogeneous Architecture and Speculative Decoding
- Bridging Efficiency and Scalability in LLM System via 3D Hybrid PIM with 2D In-Transit Computation
- ConServe: Contiguity-Preserving Memory Management for Multi-Turn LLM Serving
- DynoPipe: Heterogeneous Edge-Cloud LLM Serving with Dynamically Orchestrated Pipeline Boundaries
- MERIDIAN: In-Memory Acceleration for RAG with Document Attention Decomposition
- MTIA 300: Meta's First Training Chip Featuring Built-in NICs and Collective Offloading Engines
- Mapping and Communication Optimizations with Fault Tolerance for Wafer-Scale LLM Inference
- Rearchitecting the Datacenter Lifecycle for AI
- RoCC Harnessing Raster Operations Pipeline for Efficient Tensor Collective Communication
- SHyLA 3D-Stacked NVM-DRAM Hybrid LLM-Inference Architecture Exploiting Data and Memory Heterogeneity

MERIDIAN 补充视角（ISCA'26，PIM 设备上的 TP/PP/Hybrid 模型映射）：MERIDIAN 的调度器在 32 个 CXL Type-3 PIM 设备上做模型映射——(1) tensor 并行：CEC 的 FC 层跨设备分片（每设备存部分权重、出部分输出再聚合），轻量算子（attention/激活）集中到主设备（主设备持 query/生成 token 的 KV cache）以减通信；DAC 按 attention head 切分文档 KV、每 head 分配到最少设备（避免广播、支持 head 级并行），设备内 KV 张量均匀分布到各 DRAM bank。(2) pipeline 并行：decoder 分成多 stage、DAC/CEC 分级匹配，batch 拆 micro-batch 顺序穿流水（两 stage 时两个 micro-batch 并行）。(3) hybrid：stage 跨多设备时内部用 tensor 并行，接口可配组合与粒度。与 GPU TP（Megatron 式列/行切分 + all-reduce）的差异：通信对象不同——CEC 聚合部分和（tensor 并行同步 partial results）、DAC 只传紧凑注意力统计量；pipeline 只传轻量激活，因此扩展性更好（32 设备 pipeline 4.19× vs tensor 3.68×）。ICE 交错调度进一步让 DAC/CEC 并发推进（见"Interleaved Cluster Execution"条目）。

MTIA 300 补充视角（ISCA'26，LLM 推理的 TP8-TP8 与 DP8-EP8 配置）：MTIA 300 评估 DeepSeek-R1（8 加速器，InferenceMax + vLLM）用了两种分片策略：(1) **TP8-TP8**——每层权重矩阵切分到全部 8 设备，每设备算每操作的一个切片、每个并行区域后 AllReduce 同步（对应本条目标准 TP）；(2) **DP8-EP8**——稠密层（attention + 共享 MLP）8 设备复制、各自处理独立 batch（DP），MoE 部分每设备拥有 1/8 专家、token 经 AllToAll 在设备间路由（对应"Expert Parallelism"条目）。BF16 做 attention/KV cache、FP8 做 MoE 计算（两平台原生支持）。结果：decode 主导下靠高 HBM 带宽在高并发（>64）优于 H200；低并发差距小（MTIA 300 小 batch 通信开销 vs H200 NVLink、部分 kernel 未按 batch/token 维充分并行致 PE 未充分利用）。
