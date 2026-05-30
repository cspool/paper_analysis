## MixServe: An Automatic Distributed Serving System for MoE Models with Hybrid Parallelism Based on Fused Communication Algorithm

- 属于Serving调度的实现是什么？实验比较什么？
  MixServe 提出自动分布式 MoE 推理服务系统，包含三个核心机制：
  1. **Automatic Analyzer（§III-B）**：离线阶段基于模型超参数（hidden dim h, head数, num_layers l, num_experts, top-k）+ 网络硬件配置（计算能力、intra-node/inter-node 带宽和拓扑），通过理论通信模型（AR 通信量 O(bs·h/d)，A2A 通信量 O(bs/d·hk)，式 1-3）和 profiling 数据，自动推导最优 TP-EP-DP 并行策略。包含 queuing-aware latency model（M/M/1 排队模型）预测 TTFT/ITL/Throughput（式 4-11）。
  2. **Hybrid TP-EP Partitioner（§III-C）**：将 Attention block 按 intra-node TP + inter-node DP 切分，MoE block 按 intra-node TP + inter-node EP 切分。解耦 AR 为 RS+AG，重组为 RS→A2A→AG 三段式通信流程，降低通信量和通信规模。自动枚举满足 n_proc × n_node = d_TP × d_EP 的所有可行策略并选最优。
  3. **Fused AR-A2A Communication Algorithm（§III-D）**：通过异步机制将 intra-node RS/AG 通信与 inter-node A2A 通信重叠执行。(a) Fused RS-Combine（Alg 1）：intra-node RS 与 inter-node A2A pairwise 异步重叠，最后 intra-node AG 汇总，时间 O(n_node)，空间 O(bsh·n_proc)；(b) Fused AG-Dispatch（Alg 2）：intra-node AG 与 inter-node Dispatch 异步重叠，时间 O(n_node)，空间 O(1)。
  实验比较了 MixServe 与 vLLM（TP+PP 和 DP+EP 两种配置）和 Tutel（TP+EP）在不同硬件平台（NVIDIA H20 + Ascend 910B）、不同模型（DeepSeek-R1 671B, Qwen3-235B-A22B）、不同并行度（TP=4/8, DP=2/4/8, EP=2/4/16/32）下的 TTFT、ITL、Throughput。还包括 DP vs EP trade-off 消融实验（d_DP = d_EP / d_DP > d_EP / d_DP < d_EP）和同步 vs 异步通信消融实验。

- 硬件平台是什么，配置是什么。
  - **NVIDIA H20 集群**：2 台服务器，每台 8×NVIDIA H20 GPU（96 GB）。Intra-node：NVLink 4.0（最高 900 GB/s）。Inter-node：InfiniBand（400 Gbps）。
  - **Ascend 910B 集群**：4 台 Atlas 800T A2 服务器，每台 8×Ascend 910B NPU（64 GB）。Intra-node：HCCS 全互联（最高 480 Gbps）。Inter-node：RoCE（最高 200 Gbps）。
  精度：论文未明确说明（通常 BF16/FP16）。

- 开源Serving框架是什么。修改了什么。
  基座框架：**vLLM**（Ascend 910B 集群）和 **Tutel**（H20 集群）。
  具体修改：
  - **Offline Stage**：新增 Automatic Analyzer 模块——获取模型超参数 → 用不同 batch size/seq length 的预设 prompt 收集 profiling 数据 → 以网络硬件配置（计算能力 + intra/inter-node 带宽和拓扑）为输入计算理论值 → 综合 observations 和理论值输出最优并行策略（§III-B1 定义的 context-free grammar 形式）。
  - **Online Stage**：新增 Weight Loader 和 Partitioner——根据 offline 输出的最优策略加载对应 weight shards → 初始化 mixed parallel communication groups → 向 MoE model 的 forward method 注入 collective communication operators（RS/AG/A2A）。
  - **Fused AR-A2A Communication**：解耦 TP group 的 AR 为 RS+AG → 重组 EP group 的 A2A → 形成 RS→A2A→AG 三段式流程 → 实现 Async RS-Combine（Alg 1）和 Async AG-Dispatch（Alg 2）算法，使 intra-node 和 inter-node 通信异步重叠。
  - **Serving service**：基于 vLLM 的 memory management 和 request scheduling 机制管理 K/V cache 和调度请求。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文未声明开源。经 web search（arxiv 2601.08800）未发现公开代码仓库。基于 vLLM 和 Tutel 实现。

  **MixServe 推理全流程（以 DeepSeek-R1 在 4 节点 Ascend 910B 集群、TP=8 + DP=4/EP=4 为例）**：
  1. **Offline 策略分析**：MixServe 读取 DeepSeek-R1 模型超参数（671B, 256 experts, h=7168, l=61, top-k=8）→ 用不同 batch size（1/2/4/8/16）和 seq length（128/256/512/1024/2048/4096）的预设 prompt 做 profiling 获取 compute/comm latency 观测值 → 输入集群配置（4 node × 8 NPU, HCCS 480 Gbps intra-node, RoCE 200 Gbps inter-node）→ Analyzer 搜索满足内存约束（式 8）的最优 (d_TP, d_EP, d_DP) → 输出 TP=8 + DP=4, TP=8 + EP=4。
  2. **Weight Loading**：Weight Loader 根据 TP=8 切分每节点内 8 NPU 上的 Attention 参数（沿 hidden dim），DP=4 在 4 节点间复制 Attention 参数。MoE 参数按 TP=8（intra-node）+ EP=4（inter-node，256 experts / 4 = 64 experts per node，node 内 8 NPU TP 共享）加载。Partitioner 按此方案注入通信算子。
  3. **请求接入**：用户请求到达 vLLM serving service。Scheduler 使用 continuous batching 管理请求，PagedAttention 管理 KV cache。最大 batch size=16，最大 seq len=4096。
  4. **Attention 计算（Layer ℓ）**：每节点 8 NPU 对 Attention block 做 TP 前向：各 NPU 计算自己持有的 Q/K/V/O 分片 → intra-node AR（RS + AG）同步 hidden states。4 节点间 DP 独立计算（各自有完整 batch 子集）。
  5. **Gating + Expert Routing**：Attention 输出经 gating network 计算 top-8 expert → token 准备 dispatch。
  6. **Fused RS-Combine（MoE block 输入侧）**：每节点内 token hidden states 先做 intra-node RS → 同时启动 inter-node A2A pairwise（4 node, 3 rounds）：每轮中 intra-node RS 结果与 inter-node recv 结果异步处理 → top-k weights 加权 → 最后 intra-node AG 汇总完整 hidden states。
  7. **Expert FFN**：各 NPU 对本地 experts 执行 FFN（MLP 前向），利用 TP 组内共享的 expert 参数。
  8. **Fused AG-Dispatch（MoE block 输出侧）**：FFN 输出在 node 内做 intra-node AG（shard hidden states 到 TP group）→ 同时 inter-node Dispatch 发送到对应 EP rank → 除首轮 pairwise 和末轮 AG 外，其余轮次 intra/inter-node 通信重叠。
  9. **Iteration 完成**：所有 61 层 Decoder 完成 forward → output token 生成。TTFT = queuing delay + prefill latency（s=prompt_len），ITL = decode latency（s=1）。
  10. **Pipeline Parallelism**：论文提到 PP 可用于跨 Decoder layer 分配，式 6 包含 PP 的 P2P 通信项，但实验配置中未明确标注 PP degree。

- 属于Serving调度的实现是什么？实验比较什么？
  FineMoE 基于 Megatron-LM 实现了 FineEP 策略，通过 per-micro-batch 的 token scheduling 实现细粒度 GPU 负载均衡。核心包含四个机制：
  1. **Token Scheduling（§5）**：将负载均衡建模为线性规划问题（LPP 1），目标 min(max GPU load)，约束为每个 expert 的 replica loads 之和等于其 total load。使用 HiGHs 求解器在单 CPU 线程求解（变量数 O(|E|d)），利用 warm-start 跨 micro-batch 复用求解器状态。
  2. **Locality-Aware Routing（§5.2, Algorithm 1）**：优先将 token 路由到本地 expert replica（同一 GPU），减少 all-to-all 通信量。从本地到远程依次分配 token 直到各 replica 达到目标负载 x_e^g。
  3. **Distributed Scheduling（§5.3）**：所有设备通过 all-gather 收集全局 load 信息后各自独立执行确定性调度算法，比集中式减少一次通信操作。
  4. **Overlapping（§5.4）**：CPU 上的 LPP 求解与 GPU 上的 token permutation 操作重叠执行。
  实验比较 FineMoE（w/ and w/o Adaptive Replacement）与 Megatron-LM、SmartMoE、FlexMoE、DeepSpeed 在 GPT（32×1.3B/16×3.2B/8×6.7B）和 Mixtral（16×2B/8×7B）模型上的端到端训练吞吐量、负载均衡能力（Zipfian skewness s∈[0,2]）、执行时间分解、调度开销、DeepEP 集成、ablation study（warm solving/locality-aware routing/overlapping）。

- 硬件平台是什么，配置是什么。
  4 节点，每节点 8×NVIDIA H100 80GB SXM GPU（共 32 GPU），900 GB/s NVLink intra-node，2×400 Gbps InfiniBand NIC per node。训练精度 BF16。PP degree = 节点数（仅用于 inter-node），DP degree = 8，EP degree = 4，FineEP 参数 d=2（1 FineEP group / DP group）。禁用 TP（因其高通信开销）。使用 selective activation recomputation（仅 recompute MoE FFN），distributed optimizers（类 ZeRO-1）。

- 开源Serving框架是什么。修改了什么。
  基座框架：**Megatron-LM**（github.com/NVIDIA/Megatron-LM）。
  修改内容：
  - 修改 MoELayer forward：在 gate network 之后、all-to-all dispatch 之前插入 FineEP Token Dispatcher。
  - 新增 Place Manager（Python，device 0）：生成 expert placement（symmetric: Cayley graphs / asymmetric: greedy + Monte Carlo sampling），broadcast 到所有设备；后台监控 load 分布并触发 adaptive replacement。
  - 新增 Token Dispatcher（C++）：LPP 求解（HiGHs solver）+ locality-aware token routing（Algorithm 1）+ communication-aware scheduling（Appendix A.1）。
  - 扩展 all-to-all 通信组：从 EP group（EP_degree=4）扩展为 FineEP group（d×EP_degree=8）。
  - 实现 Distributed Scheduling：all-gather 收集 `input_e^g` → 各 GPU 独立求解 LPP → 各 GPU 独立 route tokens。
  - 实现 Adaptive Replacement（§6.4）：监控 expert load 分布 → 时间序列预测（moving averages）→ Equation 3 评估 → 触发 asymmetric placement 生成和模型 reinitialization。
  - 额外实现 SmartMoE 和 FlexMoE 在 Megatron-LM 上的版本用于公平对比。
  - 集成 DeepEP（high-performance all-to-all backend）与 FineEP。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文未声明开源。经 web search（2025）未发现公开仓库。基于 Megatron-LM 实现（Python + C++ token scheduling）。

  **FineMoE 训练单个 micro-batch 的全流程（GPT 32×1.3B, DP=8, EP=4, d=2, B=4, seq_len=2048）**：
  1. **初始化**：Placement Manager（GPU 0）用 Cayley graph 生成 symmetric expert placement（8 GPUs × 4 experts/GPU），broadcast 到所有 GPU。各 GPU 加载分配的 expert replicas + optimizer states。
  2. **Micro-batch 加载**：Data loader 将 B=4 sequences（2048 tokens）分发给 8 个 DP rank。
  3. **Attention + Gate（Layer ℓ）**：各 GPU 独立执行 self-attention（DP）→ gate network（top-2 routing）→ 产出 `{input_e^g}`（GPU g 上路由到 expert e 的 token 数量，e∈[0,31]）。
  4. **Token Scheduling（与 GPU token permutation 重叠）**：
     a. All-gather：8 GPU 交换 `{input_e^g}` 信息（~32×8 integers，latency ~数 us）。
     b. LPP Solving（CPU, HiGHs）：求解 min max_g Σ x_e^g s.t. Σ_g x_e^g = load_e → 得到 `{x_e^g}`（目标 replica load）。
     c. Locality-Aware Routing（Algorithm 1）：先 route local tokens（同一 GPU 上有 replica 的 expert）→ 再 route 远程 tokens。产出 token-to-(GPU, replica) mapping。
  5. **All-to-All Dispatch**：8 GPU FineEP group 内 all-to-all（NCCL 或 DeepEP），tokens 发送到目标 GPU。
  6. **Expert FFN**：各 GPU 对收到的 tokens 执行本地 expert FFN（SwiGLU: W_gate·x → SiLU ⊗ (W_up·x) → W_down·result）。
  7. **All-to-All Combine**：反向 all-to-all，FFN 输出返回原始 GPU。
  8. **Residual + Next Layer**：MoE output + attention residual → Layer ℓ+1。
  9. **Backward**：autograd 反向传播 + DP gradient all-reduce（BF16 all-to-all + FP32 local reduction）。
  10. **Adaptive Replacement（每 ~50 iterations）**：Placement Manager 检查 load 分布 → 若 Equation 3 预测性能下降 → 生成新 asymmetric placement（greedy replica count + Monte Carlo placement）→ reinitialize（迁移 expert 参数 + optimizer states，~数百 ms）。
