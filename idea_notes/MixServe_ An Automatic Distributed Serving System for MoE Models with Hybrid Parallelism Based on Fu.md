## MixServe: An Automatic Distributed Serving System for MoE Models with Hybrid Parallelism Based on Fused Communication Algorithm

- baseline方法是什么？
  Baseline 为 vLLM（TP+PP 和 DP+EP 两种配置）和 Tutel（TP+EP），均为现有 MoE 模型推理服务系统。以 vLLM DP+EP（DeepSeek-R1, 4-node Ascend 910B, TP=8+DP=4, EP=32）为例说明全栈执行路径：
  - **算法层（MoE 推理）**：每层 Decoder 执行 Attention（QKV projection + attention score + output projection）→ Gating（top-K routing）→ Expert FFN（MLP）。Attention block 使用 TP（intra-node AR 同步），MoE block 使用 EP（inter-node A2A dispatch/combine）。AR 通信量 O(bs·h/d)，每层需 RS+AG 两阶段各 1 轮（Broadcast algorithm）。A2A 通信量 O(bs/d·hk)，需 d-1 轮（Pairwise algorithm）。
  - **系统框架层**：vLLM 使用 PagedAttention + continuous batching 管理 KV cache 和请求调度。TP 限于 intra-node（利用 NVLink/HCCS 高带宽），EP 跨 node（利用 InfiniBand/RoCE）。Tutel 额外支持 hybrid TP+EP（TP=4+DP=4, TP=4+EP=4），但仅限 intra-node TP + inter-node EP 的固定组合。两者的并行策略由用户手动指定（基于经验直觉），无自动策略选择机制。
  - **编译框架层**：论文未明确说明（标准 PyTorch + CUDA kernel）。
  - **kernel 调度层**：NCCL/HCCL collective communication library 处理 AR 和 A2A 原语。通信算子同步执行——AR（RS+AG）和 A2A（Dispatch+Combine）各自串行，互不重叠。inter-node A2A 的低带宽（RoCE 200 Gbps vs intra-node HCCS 480 Gbps）成为瓶颈。无 intra-node 和 inter-node 通信间的重叠设计。
  - **硬件架构层**：NVIDIA H20（96 GB, NVLink 4.0 900 GB/s intra-node, InfiniBand 400 Gbps inter-node）/ Ascend 910B（64 GB, HCCS 480 Gbps intra-node, RoCE 200 Gbps inter-node）。Intra-node 带宽显著高于 inter-node（2×-∞），但现有策略将所有通信统一处理，未利用带宽层次差异。
  - Baseline 核心缺陷：
    1. **缺乏系统性理论分析**：并行策略选择基于经验直觉和实践，未考虑模型超参数、网络拓扑和硬件资源配置间的复杂交互。TP degree、DP degree、EP degree 的组合空间巨大，人工枚举不可行。
    2. **无法有效利用通信带宽层次**：AR-based TP 在 inter-node 场景效率低（Fig. 3：d>8 时通信开销激增），A2A-based EP 存在负载不均衡（尤其高并行度时）。现有策略将所有通信统一处理，错失了利用 intra-node（高带宽）和 inter-node（低带宽）差异优化性能的机会。
    3. **Intra-node 和 inter-node 通信串行执行**：NCCL/HCCL 中 AR 和 A2A 作为独立 collective 算子串行执行，intra-node 通信期间 inter-node 链路 idle，inter-node 通信期间 intra-node 链路 idle，网络带宽利用率低。
    4. **并行度固定且无法自适应**：用户需手动指定 TP/DP/EP degree，无法根据 cluster 配置和 workload 特征自动调整。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MixServe 通过 **Automatic Analyzer（理论通信建模 + 自动策略选择）+ Hybrid TP-EP Partitioner（解耦重组 AR/A2A）+ Fused AR-A2A Communication Algorithm（intra/inter-node 通信异步重叠）** 三层设计解决上述缺陷。全栈执行路径（以 DeepSeek-R1 在 4-node Ascend 910B, TP=8+DP=4, TP=8+EP=4 为例）：
  - **算法层（理论通信建模）**：
    1. 形式化定义并行策略（§III-B1 context-free grammar）：为每层 Decoder 的 Attention block 和 MoE block 独立定义 intra-node 和 inter-node 并行策略。Attention 支持 TP+DP，MoE 支持 TP+EP。PP 跨 Decoder layer 正交叠加。
    2. 细粒度 AR 和 A2A 通信开销建模（式 1-3）：AR(size,degree) = RS(size/degree,degree) + AG(size/degree,degree) ∝ size/degree；A2A(size,degree) ∝ (size/degree) × (degree-1)。量化 DP vs EP 的三种 trade-off case（d_DP = / > / < d_EP，Fig. 6）。
    3. Token 生成延迟模型（式 4-7）：τ(computation) ∝ Ψ/(d_TP·d_EP) · b/d_DP · sh；λ(communication) = 2×AR + 2×A2A（含 d_DP < d_EP 时的 hidden states 冗余修正）；Δtsvc = l[τ+λ] + (d_PP-1)·P2P；M/M/1 排队模型预测 queuing delay Wq。
    4. 内存约束（式 8）：Ψ_Attn/d_TP + Ψ_MoE/(d_EP·d_TP) + KV cache < M。
    5. 理论性能指标（式 9-11）：TTFT = Wq + Δtsvc|s=Lin；ITL = Δtsvc|s=1；Throughput = (Lin+Lout)/(Wq + TTFT + Lout·ITL)。
  - **系统框架层（Hybrid TP-EP Partitioner + 自动策略选择）**：
    1. **Offline Stage**：Automatic Analyzer 以模型超参数 + 网络硬件配置为输入 → 用 profiling 数据（不同 batch/seq len 的 compute/comm latency）校准理论模型 → 枚举所有满足 n_proc × n_node = d_TP × d_EP 的 (d_TP, d_EP, d_DP) 组合 → 在内存约束（式 8）下选最小化 TTFT/ITL 或最大化 Throughput 的策略 → 输出最优 (d_TP, d_EP, d_DP)。
    2. **Online Stage**：Partitioner 按最优策略切分 Attention weights（intra-node TP + inter-node DP）和 MoE weights（intra-node TP + inter-node EP）→ Weight Loader 加载对应 shards → 初始化 mixed parallel communication groups → 向 forward method 注入 RS/AG/A2A 通信算子。
    3. **DP-EP Trade-off 自动管理**：考虑延迟/吞吐需求 + 内存约束，自动选择 d_DP = / > / < d_EP 的最优配置。d_DP = d_EP 时最平衡（rank 一一对应），d_DP > d_EP 时 expert weights 冗余换高吞吐，d_DP < d_EP 时 hidden states 冗余但 effective dropping 降低通信开销（Fig. 6）。
  - **编译框架层**：论文未明确说明。
  - **kernel 调度层（Fused AR-A2A Communication Algorithm）**：
    1. **解耦重组 AR/A2A**：将 AR 分解为 RS+AG，A2A 分解为 Dispatch+Combine → 重组为 RS→A2A→AG 三段式流程。hybrid TP-EP 使 TP group 和 EP group 分别映射到 intra-node 和 inter-node（d_TP = n_proc, d_EP = n_node），通信量从纯 EP 的 AR(bsh,n_proc) + 2×A2A(bshk,n_node) 降至 AR(bsh,n_proc) + AG(bshk/n_proc,n_proc) + 2×A2A(bshk/n_proc,n_node)（式 12-13）。
    2. **Fused RS-Combine（Alg 1）**：intra-node RS 与 inter-node A2A pairwise 异步重叠——每 node 内 TP rank 持有 hidden states 分片 → 发送到下一 node 同 TP rank（inter-node isend/irecv）→ 同时在本 node 内做 RS → 下一轮用接收的 hidden states 继续 → n_node rounds 后 intra-node AG 汇总。时间 O(n_node)，空间 O(bsh·n_proc)（临时存储）。
    3. **Fused AG-Dispatch（Alg 2）**：intra-node AG 与 inter-node Dispatch 异步重叠——local TP rank 做 expert routing → 发送到下一 node 同 TP rank → 同时在本 node 内做 AG → 下一轮继续 → n_node-1 rounds 后无需末轮通信（local shards 在 TP/EP group 内）。时间 O(n_node)，空间 O(1)。
  - **硬件架构层**：同 baseline。无硬件修改。MixServe 利用 intra-node（NVLink 900 GB/s / HCCS 480 Gbps）和 inter-node（InfiniBand 400 Gbps / RoCE 200 Gbps）之间的带宽层次差异，将 TP 通信限于 intra-node 高带宽域，EP 通信限于 inter-node，通过异步重叠隐藏低带宽的 inter-node 通信延迟。
  - 对比 baseline 的改进映射：
    - **经验策略选择 → Automatic Analyzer 理论建模 + 自动搜索**：vLLM/Tutel 需手动指定 TP/DP/EP degree → MixServe 通过 profiling 校准的理论模型（含 compute + communication + queuing latency）自动枚举并选择最优 (d_TP, d_EP, d_DP)。消融实验（Fig. 11）验证了不同硬件平台下最优 DP-EP 配置不同（Ascend 910B 上 d_DP = d_EP 最优，H20 上 d_DP < d_EP 最优），证明了自动选择的价值。
    - **通信统一处理、无法利用带宽层次 → Hybrid TP-EP 解耦重组**：纯 EP 策略中 inter-node A2A 通信量 O(bshk/n_node) 且需 n_node-1 轮 → hybrid TP-EP 将每轮通信量降至 O(bshk/(n_proc·n_node))，降低了通信规模和通信量（式 12→13）。同时 TP group（intra-node 高带宽）与 EP group（inter-node 低带宽）精确对齐硬件层次。
    - **串行通信、带宽利用率低 → Fused AR-A2A 异步重叠**：baseline 中 RS→AG→Dispatch→Combine 串行执行，总延迟 = sum(各算子延迟) → MixServe 将 intra-node RS/AG 与 inter-node A2A 重叠，总延迟 ≈ max(RS+A2A, AG+Dispatch) + O(n_node)。消融实验（Fig. 12）显示异步通信显著降低 TTFT 和 ITL 并提升吞吐量，加速效果约等于 inter-node 通信开销。
    - **实验结果**：DeepSeek-R1 上 TTFT 1.08×~3.80× 加速、ITL 1.03×~1.66× 加速、Throughput 5.2%~50.3% 提升。Ascend 910B 上 DeepSeek-R1 TTFT 2.67× vs vLLM TP+PP、1.70× vs vLLM DP+EP；H20 上 DeepSeek-R1 Throughput +50.3% vs vLLM TP+PP。
