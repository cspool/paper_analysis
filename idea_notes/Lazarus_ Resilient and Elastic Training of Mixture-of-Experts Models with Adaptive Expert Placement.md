## Lazarus: Resilient and Elastic Training of Mixture-of-Experts Models with Adaptive Expert Placement

- baseline方法是什么？
  - **DeepSpeed MoE (DS) with checkpoint-based fault tolerance**：传统 Expert Parallelism (EP) 将 experts 等分到 EP size 个 GPU 上，每个 expert 只有一个 replica。故障发生时，所有 GPU 必须等待故障节点被替换（可能需要数小时到数天），然后从 checkpoint 重新开始训练。每 50 steps 进行一次 checkpoint 保存到 NFS server。EP 要求 GPU 数是 EP size 的整数倍，故障后可能有多余 GPU 空闲。All-to-all 通信在 EP group 内进行，使用 padded all-to-all（padding 到最大的 expert token 数）。
  - 全栈执行例子（DS baseline，GPT-L 16 experts，EP size=4，10 GPU 集群）：
    - **算法 Pipeline**：token → Gate(top-1) → route to expert → Expert FFN → combine。无 adaptive allocation，每个 expert 1 replica。
    - **系统框架**：DeepSpeed MoE (PyTorch + DeepSpeed v0.13)。EP size=4 → 8 GPU 使用（4 GPU/EP group × 2 groups），10 GPU 中 2 GPU 闲置。Data parallelism across EP groups。
    - **编译框架**：论文未明确说明。PyTorch eager mode。
    - **Kernel 调度**：NCCL all-to-all collective（padded），在 EP group 内固定大小传输。Training step: all-to-all dispatch (padded) → expert computation → all-to-all combine (padded) → checkpoint save every 50 steps (blocking I/O to NFS server, ~10s overhead for GPT-L)。
    - **硬件架构**：5 nodes × 2× RTX 3090 GPU/node，100 Gbps Mellanox ConnectX-5 NIC。故障恢复：失败后 NCCL timeout → 等待 replacement node（hours）→ 从 NFS 加载 checkpoint → 重启 NCCL groups → 继续训练。丢失当前 step 到上次 checkpoint 间的所有训练进度。

  - **DS(FT) baseline**：与 DS 相同的 EP 分配，但使用 Lazarus 的 reconfiguration runtime 进行快速故障恢复。如果完整 replica of all experts 仍然存在，则重新配置 EP groups 并从其他节点获取 expert states。否则必须从 checkpoint 重启。GPT-L 上 EP size=4，只有 <8 GPU 时无法利用超过 4 GPU，且丢失超过一个 EP group 时需 checkpoint 重启。

  - Baseline 的核心缺陷：
    1. **无弹性 (Inelastic)**：必须等待 replacement nodes 才能继续训练，无法利用剩余 GPU 继续推进。
    2. **GPU 浪费**：需要 GPU 数为 EP size 的整数倍，如 GPT-L 的 EP size=4 且 10 个 GPU 时只能使用 8 个。
    3. **Expert load 不均衡未处理**：传统 EP 等分 experts，不根据 expert 负载分配更多 replicas 给 popular experts，导致 GPU 间计算不均衡（up to 87% tokens routed to 2 experts）。
    4. **Checkpoint 开销巨大**：随着模型增大，checkpoint 和 restart 开销越来越显著，频繁故障下可能占据 >50% 训练时间。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **Lazarus**：通过自适应 expert replica 分配（根据 expert 负载动态调整 replica 数）+ 可证明最优的 MRO expert placement 算法（最大化故障恢复概率）+ Flexible Token Dispatcher CUDA kernel（处理非对称 placement 下的高效 token dispatch）+ 高效 reconfiguration runtime（最小化迁移开销），实现 MoE 训练的高吞吐、高弹性、强容错。
  - 对应解决 Baseline 缺陷：
    1. **解决无弹性**：自适应 replica 分配使得每个 expert 有 ≥f 个 replicas（f 为容错阈值），只要每个 expert 至少有一个 replica 存活即可恢复训练。所有剩余 GPU 全部利用，无需等待 replacement nodes。
    2. **解决 GPU 浪费**：不要求 GPU 数为 EP size 整数倍，N 个 GPU 全部使用。任何 GPU 数下均能完全利用。
    3. **解决 Expert load 不均衡**：Eq. 1 的 r_e 分配公式使 popular experts 获得更多 replicas（r_e ∝ t_e），更多 computation resource 给热门 experts，加速训练。无故障时 Lazarus 的 GPT-M 吞吐 45 samples/s vs DS 的 34 samples/s。
    4. **解决 Checkpoint 开销**：故障恢复无需从 checkpoint 重启，expert states 通过 NCCL send/recv 从 other nodes 并行获取（如 GPT-L 160 expert states 仅需 7.6s 传输），总 reconfiguration 时间 20~40s，远小于 checkpoint restart。

  - 全栈执行例子（Lazarus，GPT-L 16 experts，10 nodes，f=2）：
    - **算法 Pipeline**：token → Gate(top-1) → route to expert e → **Flexible Token Dispatcher CUDA kernel**（Algorithm 1：计算每 rank 处理容量 → 优先本地处理 → 按剩余容量比例分发 overflow tokens → reshuffle activations）→ **flexible all-to-all (no padding)** → Expert FFN（popular experts 有多个 replicas 并行处理）→ flexible combine → output。Expert loads 周期性收集（每 200 steps rebalance），动态调整 r_e。
    - **系统框架**：Controller（CPU node，Python async）管理集群，Agent（per GPU node）relay。PyTorch + DeepSpeed components + Lazarus runtime。NCCL groups: expert gradients all-reduce + non-expert gradients all-reduce + all-to-all (flexible, non-padded)。Controller 每 200 steps 或故障时重新计算分配和 placement（<100ms CPU 计算）。
    - **编译框架**：论文未明确说明。
    - **Kernel 调度**：Flexible Token Dispatcher CUDA kernel（Algorithm 1，对所有 E 个 experts 和 N 个 ranks 并行执行）→ flexible all-to-all collective（各 rank 发送/接收不同数量的 tokens，无 padding）→ expert computation → all-gather T_{e,j} (E integers per rank, negligible overhead) → periodic rebalance。流水线：all-to-all 与 computation 可在不同 streams 上执行。故障时：NCCL timeout (10~20s) → reconfig NCCL groups (5~15s) → NCCL batched send/recv 并行状态迁移 → 恢复训练。
    - **硬件架构**：5 nodes × 2× RTX 3090 GPU (10 emulated nodes)，100 Gbps NIC。Lazarus 将 expert replica slots per GPU 设为 6（GPU memory limit）。MRO placement: 将 16 experts 按 c=6 分为 ⌈16/6⌉=3 组，每组内最大化 expert overlap。故障时重新路由到剩余 alive nodes 上的 expert replicas。如 4 node failures：Lazarus 41% recovery prob vs spread placement 12%。
