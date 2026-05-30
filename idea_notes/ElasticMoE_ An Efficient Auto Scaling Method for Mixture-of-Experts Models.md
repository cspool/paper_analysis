## ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts Models

- baseline方法是什么？
  Baseline 方法包括四种自动缩放策略，均基于 vLLM 实现：(1) **Horizontal (Replica)**：启动完整独立推理实例副本（如 DP2-TP2-EP4 的 4 NPU 副本需再加 4 NPU），旧实例继续服务，新实例冷启动。粗粒度（最小增量 = 完整副本规模），高延迟（容器启动+权重加载+通信初始化+KV cache 分配，可达数十秒到分钟级），参数冗余（每个实例独立复制 expert 权重，内存浪费）；(2) **Vertical (Cold Restart)**：停止旧实例，在新 NPU 集合上重启扩展配置（如 4→6 NPU 仅需 6 而非 10），细粒度但引入 downtime（旧实例销毁期间无服务）；(3) **Vertical (Extravagant)**：新实例在独立 NPU 上并行启动，旧实例继续服务（4→6 需 10 NPU 临时总量），无 downtime 但资源翻倍，成本高；(4) **Vertical (Colocated)**：新实例在同组 NPU 上启动，4 颗共享 NPU 需临时持有两份模型权重和 KV cache，为避免 OOM 需提前缩小 KV cache，降低吞吐量。

  **Baseline 全栈执行例子（以 DeepSeek V2 Lite 模型，4 NPU 响应流量突发，Horizontal 缩放为例）**：
  - **算法层**：MoE gating 选择 top-k experts（64 experts，6 activated/token），attention + FFN 标准 Transformer decode
  - **系统框架层**：vLLM 部署在 DP2-TP2-EP4 配置（4 NPU）。当流量突发触发缩放时：(a) 检测到 SLO 持续低于阈值 → 触发 scale-out；(b) Kubernetes/Ray Serve 调度新容器/进程；(c) 新实例从磁盘加载完整模型权重到新 NPU 4-7；(d) 初始化通信组（HCCL init_process_group）；(e) 分配 KV cache 内存；(f) 新实例就绪后 Coordinator 更新路由表分流请求。整个过程需要数十秒到分钟级延迟，期间 4 NPU 的旧实例过载持续违反 SLO
  - **编译框架层**：论文未明确说明（使用标准 PyTorch CANN 编译路径）
  - **Kernel 调度层**：Ascend CANN API 管理 NPU 内存分配（aclrtMalloc），HCCL 管理通信（all-to-all for EP, all-reduce for TP），GEMM 在 NPU 计算单元执行。缩放时每个新 NPU 独立执行磁盘→HBM 加载（最慢链路）
  - **硬件架构层**：Huawei CloudMatrix384，Ascend 910C NPU（64 GB HBM），Unified Bus 互联。旧实例 4 NPU 满负荷运行，新 4 NPU 冷启动时从磁盘串行加载专家权重，Unified Bus 闲置

  **Baseline 核心痛点 (L1-L5)**：
  L1：高缩放延迟——新实例需冷启动（权重加载+通信初始化+KV cache 分配），数十秒到分钟级，无法应对突发短流量
  L2：高 downtime——Vertical Cold Restart 需销毁旧实例再启动新实例，in-flight 请求丢失，新请求排队积压
  L3：粗粒度缩放——Horizontal 必须启动完整副本（DeepSeek V3 最小 32 NPU），微小流量波动也需大量过度分配
  L4：低效 expert 重分配——Horizontal 每个实例独立复制 expert 权重，EP 度局限在实例内，无法跨实例统一 token 路由和负载均衡
  L5：高峰值内存——Vertical Colocated 在共享 NPU 上临时持有两份模型权重和 KV cache，OOM 风险或需缩 KV cache 牺牲吞吐

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ElasticMoE 提出三个核心机制解决上述五个缺陷：

  **(1) HBM 管理与推理执行解耦 (HMM + IMM) → 解决 L1, L5**
  - HMM 以持久守护进程独立管理模型权重和 KV cache，权重仅从磁盘加载一次
  - IMM 推理进程不直接加载权重，而是通过 zero-copy IPC 获取 HMM 的引用句柄
  - 旧实例终止不触发权重卸载，新实例无需磁盘 I/O 即可附加已有权重
  - 消除缩放期间的冗余内存分配峰值

  **(2) In-Place 缩放 + 零拷贝复用 + P2P 传输 → 解决 L2, L3**
  - 固定 TP 度不变，仅调整 DP 和 EP 度：共享 NPU 上 attention 权重和 KV cache 布局不变，可直接 zero-copy 复用
  - 新增 NPU 通过 HCCL P2P 传输获取权重（比磁盘快约一个数量级），绕过 host memory
  - "Scale-while-serve" 模型：旧实例持续服务 in-flight 请求，新实例在后台准备，准备就绪后无缝切换流量（零 downtime）
  - 支持增量为 2 NPU 的细粒度缩放（vs Horizontal 的 32-320 NPU 完整副本）

  **(3) 虚拟 Expert 管理 (vpage-remap) → 解决 L4**
  - 专家权重按非连续物理页存储但通过虚拟地址映射为连续逻辑张量（满足 GEMM kernel 要求）
  - EP 度变化时只需更新虚拟→物理映射而非全量拷贝/重新分配大缓冲区
  - 支持跨 NPU 灵活重新分配 expert 权重，避免 Horizontal 的 expert 复制冗余
  - 降低 peak memory 和使用中的延迟

  **论文方法全栈执行例子（以 DeepSeek V2 Lite，4→6 NPU scale-up 为例）**：
  - **算法层**：同 baseline，MoE gating 和 FFN 计算不变
  - **系统框架层**：
    1. Coordinator 的 SLO-aware Estimator 检测 SLO<90% → 触发 scale-up (DP2→DP3, EP4→EP6, TP2 固定)
    2. HMM 分析当前 vs 目标配置，生成最小代价计划：NPU 0-3 上 attention/KV cache 零拷贝复用，NPU 4-5 通过 HCCL P2P 从 NPU 0-1 接收 attention 权重
    3. Expert 权重全局 remap：p2p-copy 迁移到 NPU 4-5，vpage-remap 更新映射（旧映射保持活跃）
    4. IMM 从 LRU cache 取 pre-initialized 6-NPU 实例 → zero-copy attach HMM 权重和 KV cache → 标记 ready
    5. Coordinator 停止向旧实例路由新请求 → 等待 in-flight 完成 → 旧实例标记 inactive → 流量切到新实例
    全程旧实例持续服务（无 downtime），新实例准备期间与旧实例共享同一份 KV cache
  - **编译框架层**：论文未明确说明（CANN API 管理 NPU 计算图编译，PyBind11 桥接 C++/Python）
  - **Kernel 调度层**：
    - IpcSafeAllocator 拦截 torch.ones/empty/full → CANN IPC-compatible aclrtMalloc
    - p2p-copy: HCCL isend/irecv/broadcast + aclrtMemcpyAsync，经 Unified Bus 直接 NPU-to-NPU
    - zero-copy: rtIpcSetMemoryName → rtSetIpcMemPid → UNIX socket → rtIpcOpenMemory → torch::from_blob
    - vpage-remap: aclrtMallocPhysical (非连续物理页) → aclrtReserveMemAddress (连续虚拟地址) → aclrtMapMem (映射)
  - **硬件架构层**：Ascend 910C NPU × 6，Unified Bus 全互联。P2P 传输经 Unified Bus 而非 PCIe/host memory，延时极低。旧实例 4 NPU 和新实例 6 NPU 共享 NPU 0-3 上的 attention 权重和 KV cache 物理内存（通过 IPC 引用）

  **五个缺陷的对应解决**：
  | Baseline 缺陷 | ElasticMoE 解决方案 |
  |---|---|
  | L1 高缩放延迟 (数十秒到分钟) | HMM 持久权重 + IMM pre-initialized 实例 + P2P 快于磁盘 + zero-copy 消除重复加载 → scale-up 2.43s |
  | L2 高 downtime (Cold Restart) | Scale-while-serve: 旧实例持续服务直到新实例就绪 → 0 downtime |
  | L3 粗粒度 (32-320 NPU 增量) | 仅调 DP+EP，支持 2 NPU 增量细粒度缩放 → 灵活匹配需求 |
  | L4 低效 expert 重分配 | 全局 EP 重配置 + vpage-remap 无拷贝 expert 迁移 → 避免 expert 复制，提升 KV cache 容量 |
  | L5 高峰值内存 (OOM 风险) | 共享 NPU zero-copy 复用（非复制），新 NPU 仅 P2P 传输必要权重 → peak memory 仅比 Cold Restart 高 2-3%，比 Extravagant 低 35-40% |

  实验效果：(a) scale-up latency 为最佳 baseline 的 ≈0.11×（≈9× 改善），scale-down latency 为最佳 baseline 的 <0.15×；(b) 零 downtime；(c) peak memory 接近 Cold Restart 最优值（仅高 2-3%）；(d) 缩放期间 throughput 达 Cold Restart 的 ≈2×；(e) 在递增 RPS 下维持 SLO≥90% 到 ~8.7 RPS，远超 Cold Restart 和 Colocated baselines；(f) scale-down 后 SLO-per-NPU 最高（成本效率最优）。
