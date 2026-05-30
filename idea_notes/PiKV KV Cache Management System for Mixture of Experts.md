## PiKV KV Cache Management System for Mixture of Experts

- baseline方法是什么？
  - Baseline 是标准 MoE 推理中的 dense KV cache 管理方案：每个 GPU 保存全部 token 的完整 KV cache 副本（或通过模型并行复制），所有 GPU 间需全局同步 KV 状态。推理时每个 token 生成需要 attend 到所有 prior tokens 的完整 KV cache，导致 O(BLhE) 的注意力计算复杂度和 O(EL) 的 per-device 内存消耗。KV cache 以未压缩的全精度格式（FP16/BF16）存储，无自适应驱逐策略（采用简单 LRU 或 sliding window）。
  - 全栈执行例子（Baseline: Dense KV Cache + Static Eviction, MoE 7B, 16 experts, 128K context, multi-GPU）：
    - **算法层**：MoE gating（Softmax TopK）选 k 个 experts per token → 所有 prior tokens 的 (K,V) 不分 expert 归属全部参与注意力计算 → C_dense = BLhE FLOPs。Router 按输入语义独立决策，无 KV locality 感知。
    - **系统框架层**：vLLM/类似 serving 框架管理 continuous batching → PagedAttention 将 KV 按 block 管理 → 每个 decode step 需从 GPU VRAM 加载全部 pages → CPU/GPU 间通过 LRU 策略交换（无 query-aware 评分）。当序列长度增长到 128K 时，KV cache > 24GB，超出单 GPU HBM 容量。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：FlashAttention kernel 执行 exact attention，需完整加载全量 KV → KV 加载成为 compute-bound 外的额外瓶颈。GPU SM 大量时间花在等待 HBM→SRAM 的 KV 数据传输上，Compute Utilization 不足。
    - **硬件架构层**：multi-GPU 节点（如 8×H100），NVLink/NVSwitch 互联。跨 GPU 的 KV cache 同步需通过 All-Gather 或 RingAttention 通信，O(BLhE) 的 KV 数据在 GPU 间交换。通信延迟在 autoregressive decoding 中累积。
    - **Baseline 核心缺陷**：(a) KV cache 全量存储——内存与 expert 数 E 和序列长度 L 线性相关，E=16, L=128K 时 >24GB；(b) 全量注意力——计算与 E 成正比，大部分 expert 的 KV 与当前 query 无关却被加载；(c) 静态驱逐——无法区分高价值 token（heavy hitters）与低价值 token；(d) 未压缩——全精度存储浪费带宽；(e) 同构架构——所有 KV 管理在 GPU 上，metadata 开销占用 GPU 算力。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：PiKV 通过四个协同模块将 KV cache 从"全量静态存储"升级为"稀疏动态检索系统"：(1) Expert-Sharded Storage → 内存 O(EL) → O(L/G+KS)；(2) PiKV Routing → 注意力 O(BLhE) → O(BLhk)；(3) PiKV Compression → 带宽 ρ× 降低；(4) PiKV Scheduling → query-aware 自适应驱逐。可选 FPGA offload（§3.5）。
  - 全栈执行例子（PiKV Enhanced, MoE LLM, multi-GPU + optional FPGA）：
    - **算法层（解决"全量注意力"缺陷）**：
      - PiKV Routing 将 KV 查询空间从 E 个 experts 缩至 k 个：C_sparse = BLhk ≪ C_dense = BLhE，理论加速 E/k。
      - 支持 7 种路由策略：Base hash (O(1))、TopK softmax (O(E log k))、Load-Balanced TopK (O(E))、Cache-Aware (含 miss penalty, O(E))、Entropy-Penalized LB (O(E))、RL-Adaptive (learned, O(k²))、Hierarchical coarse→fine (O(E + k log k))。
      - Cache-Aware Router (R_P) 的 penalty term -λ log(1+miss_e) 使 routing 决策倾向于 KV cache 命中率高的 expert，减少不必要的 KV miss 和重加载。
      - **对比 baseline**：baseline 的 attention 需要加载所有 E 个 expert 的 KV；PiKV 仅加载 k 个。
    - **系统框架层（解决"全量存储+静态驱逐"缺陷）**：
      - Expert-Sharded Storage：hash s(t,e) = (t mod N_tok) ⊕ (e mod N_exp) 将 KV 按 expert 和 token 分片到不同 GPU。每个 GPU 仅存储 O(L/G + L/E) tokens，而非 O(EL)。
      - PagedKVCache 三级存储（GPU VRAM → CPU DRAM → SSD）：hot pages 留在 GPU，warm pages 在 CPU，cold pages 在 SSD。
      - DistributedKVCachePool：RDMA-based 跨节点 cache 池化，自动 load balancing。
      - CacheAwarePrefillScheduler：在 TTFT SLO 约束下优化 prefill 阶段的 cache 复用。
      - LoadBalanceDecodingScheduler：在 TBT SLO 约束下最大化 decoding 吞吐。
      - **对比 baseline**：baseline 的 vLLM PagedAttention 仅 CPU/GPU 二级 + LRU 驱逐；PiKV 三级 + expert sharding + SLO-aware scheduling。
    - **编译框架层**：论文未明确说明。vLLM 的 Triton kernel 编译不受 PiKV 修改影响。
    - **kernel调度层（解决"未压缩+metadata开销"缺陷）**：
      - PiKV Compression：支持 8 种压缩方案，压缩比 ρ=1.0-4.0×。在 KV 写入时执行 C(K,V)=(K̂,V̂)∈R^d'×2，读取时执行解压 D(K̂,V̂)。LoRA 方案：K̂ = W_d W_u K（rank-r matvec），K 重构 = K̂ + W_d W_u K̂ + b。Pipeline 中 T_step(ρ) = (dkB/ρ)(2/β + η/γ)，Speedup(ρ1→ρ2) = ρ2/ρ1。
      - Optional FPGA offload：metadata-heavy routing/compression/scheduling 在 FPGA 上执行，GPU 仅接收打包好的 {(K̂,V̂,idx)}_i∈P_t。
      - **对比 baseline**：baseline 无压缩，FP16 全精度存储和传输；PiKV 压缩减少 HBM 带宽和 PCIe 传输量。
    - **硬件架构层（解决"同构架构"缺陷）**：
      - GPU+FPGA 异构（§3.5）：FPGA SmartNIC (Alveo U55C) 通过 CXL Type-3 链接 disaggregated DDR 内存。GPU 仅执行 encoding + attention 的核心计算。
      - 32B MMIO command queue (AXI-Lite) → PiKV-CTRL → routing/compression/scheduling engines 并行执行。
      - On-chip 资源约 224 KB（BRAM_Γ + BRAM_meta + URAM_W），单 U55C SLR 内。
      - T_fpga = T_route + k(T_Γ + K(T_ddr + T_codec))，B_step ≈ (2kd'|P_t|/ρ_link) + k log E。
      - **对比 baseline**：baseline 纯 GPU 架构，KV 管理 metadata 开销与 attention 计算竞争 GPU SM；PiKV-FPGA 将 metadata 卸载，GPU 专注计算。
  - 解决 Baseline 缺陷的方式总结：
    1. **KV 内存过大**：Expert sharding (O(EL)→O(L/G+L/E)) + 压缩 (ρ× 降低) + 调度 (仅保留 K 个 pages)。
    2. **注意力计算浪费**：稀疏路由 (E→k experts per query)。
    3. **静态驱逐误删高价值 token**：Query-aware utility scoring (u_i = Σ α_j φ_j) + 自适应阈值 (θ ← θ + γ(η*-η))。
    4. **带宽瓶颈**：压缩 (d→d/ρ) + FPGA offload (metadata 与 payload 分离)。
    5. **同构架构**：GPU+FPGA 异构，KV metadata 在 FPGA，KV payload 在 CXL-attached DDR，GPU 仅接收精炼后的 KV 子集。
