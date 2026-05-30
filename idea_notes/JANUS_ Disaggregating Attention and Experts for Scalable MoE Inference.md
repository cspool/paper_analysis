## JANUS: Disaggregating Attention and Experts for Scalable MoE Inference

- baseline方法是什么？
  - **Monolithic MoE Inference（SGLang, vLLM, LINA）**：Attention 和 MoE 层共址于同一 GPU，使用共享并行度配置（如相同的 TP/EP degree），通过复制完整模型实例进行 scale。全栈执行例子（DeepSeek-V2 decode, 8×H100 TP=8）：
    - **模型推理算法层**：Token → Embedding → 每层: Attention (MLA, 512 context) → MoE Gate (Top-K routing) → Expert FFN (每 token 激活少量 expert, 其余 idle) → Shared Expert → LM head。所有 layer 使用相同并行度。
    - **系统框架层**：SGLang/vLLM monolithic deployment。模型实例包含完整 attention + MoE 权重。Scaling 单元 = 完整模型副本（如 DeepSeek-V3 最小 16 H100）。Elasticity 粗粒度：加载全部参数 + 重建并行组。
    - **编译框架层**：论文未明确说明。PyTorch eager mode + NCCL。
    - **kernel 调度层**：cuBLAS GEMM + NCCL AllReduce/All-to-All。MoE expert 参数占据 >90% 显存，仅少量 expert 被每 token 激活，大量显存被 idle expert 占用。KV cache 与 expert 参数共享同一 GPU memory budget。
    - **硬件架构层**：8× H100 80GB NVLink。Expert 参数完全 resident GPU HBM。
  - **Disaggregated MoE Inference（MegaScale-Infer, xDeepServe, EaaS）**：Attention 和 MoE 层部署于不同节点，实现独立配置。但仍有三个缺陷：① Expert scheduling 聚焦 token 负载均衡（如 EPLB 均匀分配 tokens），不直接 minimize a_max（各 GPU 的最大 distinct activated expert 数），token-balanced 配置下仍存在 straggler；② Resource scaling coarse-grained（MegaScale-Infer 限制为 balance attention/MoE 执行时间的配置，EaaS 仅提供弹性通信通道，xDeepServe 无 scaling policy）；③ Communication 使用 attention 侧 gating + pairwise all-to-all（每 attention instance 与所有 MoE instance 通信），产生 O(m×n) 次小消息传输。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **JANUS 方法**：Disaggregate attention/MoE + Adaptive two-phase communication + AEBS scheduling + SLO-aware fine-grained scaling。三大核心设计：
    1. **Adaptive Two-Phase Communication（解决 cross-sub-cluster 通信瓶颈）**：不同于 baseline 的 O(m×n) pairwise 传输或 attention-side gating，JANUS 将 gating 移至 MoE 侧（发送完整 activation 而非 per-expert packed tensor），通过 Phase-1 intra-node aggregation + Phase-2 inter-node bulk transfer 减少跨节点传输次数。案例选择（Case-1 直接 vs Case-2 中继）自适应流量负载。NVSHMEM one-sided put 消除同步开销，元数据打包进 signal 避免额外传输。
    2. **AEBS - Activated-Expert-Balanced Scheduling（直接解决 a_max straggler）**：发现 MoE 层 latency ∝ a_max（各实例最大 distinct activated expert 数），而非 token count。AEBS 直接 minimize a_max 而非 token imbalance。单副本 expert 固定分配，多副本 expert 贪心选负载最低实例。Synchronization-free：每个 MoE instance 独立运行相同 deterministic kernel，消除 CPU-GPU 同步和跨 GPU 协调，开销 <90μs。
    3. **Fine-Grained SLO-Aware Resource Scaling（解决 coarse-grained elasticity）**：构建 Roofline-based TPOT model + Monte Carlo â_max estimator + Little's Law 稳态求解。枚举完整 (n_a, n_e) 二维搜索空间（不限于 attention/MoE balanced 配置），选择满足 SLO 的最小 GPU 配置。联合优化 expert placement（Algorithm 3: activation-aware co-activation minimization）。15 min 间隔增量调整。

  - 全栈执行例子（JANUS, DeepSeek-V2, 1A6E 配置 vs Monolithic SGLang 8GPU）：
    - **模型推理算法层**：与 baseline 相同 MoE 模型。JANUS 通过 AEBS 改变 expert activation 的调度方式——从 "random/EPLB token balancing" 变为 "expert-count balancing"。Roofline 分析指导 MoE 层 latency 建模 T_moe = β·a_max + c_e（线性依赖 a_max）。
    - **系统框架层**：JANUS modifies SGLang。Attention instances (n_a=1) 与 MoE instances (n_e=6) 独立部署。对比 baseline monolithic SGLang (8 GPUs 统一 TP=8)，JANUS 使用 7 GPUs (1A+6E) 达到更高 throughput。Scaling 控制器每 15min 根据 workload 变化调整配置 (e.g. 1A6E → 2A6E → 4A6E → 5A10E)。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：AEBS GPU kernel 替代 baseline 的 EPLB/random scheduling。NVSHMEM one-sided put 替代 NCCL collective (baseline 的 pairwise/all-to-all)。MoE 侧 gating (EGate) + 2PC 替代 baseline attention-side gating + 1PC。关键性能：2PC+EGate 比 2PC+AGate throughput 提升 4-34%（因避免 per-link routing metadata 传输）。
    - **硬件架构层**：H100 80GB × 8 per node, NVLink + IB。JANUS 将 attention 和 MoE 分离到不同 GPU，expert 参数仅驻留 MoE GPU。与 baseline 所有 GPU 同时持有 attention + expert 参数不同，JANUS 各 GPU 内存仅用于单类型参数，减少 memory pressure。

  - **Baseline 缺陷 → 方法设计映射**：
    | Baseline 缺陷 | JANUS 设计 | 效果 |
    |-------------|-----------|------|
    | 统一并行度导致 attention/MoE 资源错配 | 解耦 attention/MoE 子集群 | 1A6E 配置在低负载下比 8GPU monolithic 高 4.7× TPG |
    | Token-balanced 调度未 minimize a_max | AEBS: expert-count balanced scheduling | a_max 降低 2-5 experts, MoE latency ↓ |
    | Coarse-grained scaling (full replica) | Fine-grained (n_a, n_e) 二维搜索 | 24h trace GPU-h 节省 39% vs SGLang |
    | O(m×n) pairwise 通信 | Two-phase aggregation + MoE-side gating | 2PC+EGate throughput +4-34% vs 2PC+AGate |
    | 无 expert placement 优化 | Activation-aware co-activation minimization | 减少高频共激活 expert 在同一 GPU 的冲突 |
