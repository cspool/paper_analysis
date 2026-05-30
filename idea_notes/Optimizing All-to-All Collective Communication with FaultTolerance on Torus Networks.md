## Optimizing All-to-All Collective Communication with FaultTolerance on Torus Networks

- baseline方法是什么？
  - Baseline 是 Ring 算法 + Pipeline 调度（Ring+Pipeline）：在 N-D torus 网络上执行 All-to-All 集合通信，将数据沿顺时针/逆时针双向拆分传输，采用 store-and-forward 方式逐跳转发以避免网络拥塞；跨维度调度采用 Pipeline 方式将数据分块后在 X→Y→Z 固定顺序上流水线化执行。
  - 全栈执行例子（Baseline: Ring+Pipeline, 3D torus, 4×4×4, HalfRing+DimRotation 论文方法对比）：
    - **算法层**：每个节点将 All-to-All 数据按 N-1 个阶段拆分，每个阶段对应不同的跳距。以 4 节点环为例，共 3 个阶段（跳距 1/2/3）。每个阶段再拆分若干子阶段完成逐跳 store-and-forward 转发。Ring 算法同时利用顺时针和逆时针链路双向通信，但在大跳距阶段存在非最短路径问题——如节点 1 到节点 4 逆时针仅 1 跳，但顺时针需 3 跳，造成额外链路带宽消耗。每个子阶段执行单跳传输（如 Stage 3-1: 节点 1→节点 2 转发紫色数据块），共需 N(N-1)/2 次单跳传输。
    - **系统框架层**：论文未明确说明具体分布式训练框架。通信算法在 MPI 层或 PyTorch Distributed 的 collective communication 后端执行（论文 real machine 实验使用 PyTorch Distributed 模块、Ascend torch_npu）。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**（通信调度层）：All-to-All 按维度分解为 N 个顺序 phase（3D torus 下为 X→Y→Z）。Pipeline 调度将数据分为多个 chunk（如 6 chunks），所有 chunk 使用相同的 X-Y-Z 维度顺序。每个 phase 内使用 Ring 算法在对应维度的环上执行通信。Pipeline 不可避免地引入气泡（bubbles）——当某些 chunk 完成某维度通信后等待其他 chunk 完成才能进入下一维度，链路利用率 < 100%。Chunk 大小选择困难：过大导致维度间重叠不足，过小导致调度开销增加。
    - **硬件架构层**：N-D torus 网络拓扑。以 TPUv4 为例，4×4×4 3D-torus 含 64 个 TPU，每个 TPU 有 6 条链路（±X/±Y/±Z），链路带宽 56 GB/s。数据通过 ICI (Inter-Chip Interconnect) 在 torus 网络上进行单跳 store-and-forward 传输。Ring 算法在单维环上约需 N(N-1)/2 次单跳传输，每次传输数据量 S/N（S 为每节点数据量）。
  - Baseline 缺陷根因（三个核心问题）：(1) **非最短路径导致额外带宽消耗**：Ring 算法固定使用双向传输，大跳距阶段存在绕远路——N 节点环中最大跳距 ⌊N/2⌋，但实际最短路径可能仅需 1 跳，多跳转发消耗额外链路带宽，降低整体吞吐；(2) **Pipeline 调度引入气泡**：多维 torus 上 Pipeline 调度因固定维度顺序产生气泡，各维度链路利用率无法达到 100%，且最优 chunk 数量难以确定；(3) **无容错机制**：Ring 算法要求相邻节点间直接链路存在，任意链路故障导致整个环的 All-to-All 通信中断——而大规模训练运行数周，链路故障概率不可忽略。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：针对 N-D torus 网络的 All-to-All 集合通信，从"单维算法"和"多维调度"两个正交维度进行优化。无故障场景下提出 HalfRing（利用双向链路的最短路径算法）+ DimRotation（轮转维度顺序的无气泡调度）；故障场景下提出 FoldedRing（折叠环容错算法）+ MATE（多维度加速调度，利用健康维度链路加速故障环通信）。
  - 全栈执行例子（Paper method: HalfRing+DimRotation 无故障 / FoldedRing+MATEe 有故障, 3D torus, TPUv4）：
    - **算法层**：
      - **HalfRing（取代 Ring）**：在每个阶段，HalfRing 根据收发节点间的实际距离选择传输方向——最短路径方向。由于所有节点均沿最短路径通信，每个阶段仅消耗一个方向的带宽（顺时针或逆时针），另一方向留给配对的另一阶段使用。N=2k+1 个节点有 2k 个阶段，可配对为 k 对同时执行；N=2k 个节点有 2k-1 个阶段，剩一个未配对阶段将数据对半拆分后双向发送以充分利用带宽。对比 Ring 算法，HalfRing 在 N 为偶数时传输时间为 N/8·S/B（Ring 为 (N-1)/2·S/2B），比值 1~2×；N 为奇数时比值为 1.5~2×。关键原理：HalfRing 通过逐跳 store-and-forward 显式编排，保证无死锁（无多跳传输）、无活锁（无绕路）、无网络争用（无链路共享）。
      - **FoldedRing（取代 Ring 处理故障）**：当环上某链路故障时，故障链路两端节点之间通过所有逆时针物理链路构建逻辑补偿连接——形成"折叠环"。故障链路的顺时针方向所有健康链路保持不变，逆时针方向的全部链路被"折叠"来替代故障链路。由此 Ring 算法的逻辑通信模式得以恢复。代价是传输时间翻倍（Table 1: FoldedRing 传输时间为 (N-1)/2·S/B，而 Ring 为 (N-1)/2·S/2B，即 0.5× 性能）。
      - **MATE/MATEe（加速故障环通信）**：MATE 利用 N-D torus 中同一维度其他健康环的链路，通过其他维度链路（如 Y-dim）构建故障环上相邻节点的双向连接，使故障环也能使用 HalfRing 执行剩余数据传输。在 2D torus 例中（Fig 9），MATE 通过 Y-dim 链路连接故障 X-dim 环上节点（如 (0,1)→(0,2)→(1,2)→(1,1) 三条红色链路构成一条逻辑 X-dim 连接）。MATE 将故障环通信拆分为正常 phase（仅 FoldedRing）+ 加速 phase（利用构建的逻辑连接执行 HalfRing），可额外利用 N-1 个平面的链路同时加速传输。MATEe 增强版在正常 phase 也传输部分数据（按 HalfRing/FoldedRing 性能比静态分配），减少加速 phase 数据量。
    - **系统框架层**（调度层）：All-to-All 按维度分解为 N 个 phase。DimRotation 调度（取代 Pipeline）：将数据分为恰好 N 个 chunk（N 为维度数），第 i 个 chunk 的维度执行顺序为维度 i → i+1 → ...（循环轮转）。3D torus 下：chunk 1: X→Y→Z，chunk 2: Y→Z→X，chunk 3: Z→X→Y。三个 chunk 在三个维度上形成完美的无冲突全覆盖，实现 100% 链路利用率，零气泡。DimRotation 的 chunk 数固定为 N（最小充分数量），调度开销远小于 Pipeline。对于异构带宽或 mixed-radix torus，总时间受限于性能最差的维度——DimRotation 确保总时间不超过最差维度上完整数据的通信时间。
      - MATE 调度（多层 phase 结构）：每个 chunk 的正常 phase 后插入加速 phase M/M_e。正常 phase 在故障维度上使用 FoldedRing（MATEe）或跳过（MATE），加速 phase 利用其他维度链路构建逻辑连接后使用 HalfRing 传输。对多故障场景（同环多故障、多环各一故障、异维各一故障），MATE 为每个故障分配独立加速 phase，或当故障不在相同维度链路冲突时允许并行加速 phase。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**（通信执行层）：HalfRing/FoldedRing 算法在大规模网络中显式编排每个子阶段的单跳传输对（发送方→接收方+转发方），形成确定的通信时间表（Schedule[chunk][phase][i][j]）。该时间表可离线预计算，在实际执行时直接下发到通信后端（MPI/nccl/torch.distributed），CPU 开销从 kernel launch 中大幅削减（real machine 实验显示 startup time 显著降低）。每个单跳传输的数据量固定（S/N 或减半），保证链路负载完全均衡。
    - **硬件架构层**：同一 N-D torus 硬件。对比 baseline：Ring+Pipeline 受限于绕远路传输 + Pipeline 气泡 → 链路利用率不足；HalfRing+DimRotation 通过最短路径 + 无气泡调度 → 每维度每时刻所有链路处于活跃传输状态（Fig 13 维度利用率显示 DimRotation 三轴 100% 利用率，Pipeline 存在周期性下降）。故障场景下：FoldedRing+Pipeline 受限于故障环传输速度减半 + 气泡 → 性能降至 fault-free 的 0.55×；MATE/MATEe 利用其他维度链路（额外平面数 = N-1）加速后 → 性能超过 fault-free baseline（1.36×/1.37×）。
  - 解决 Baseline 缺陷的方式：
    1. **针对"非最短路径导致额外带宽消耗"**：HalfRing 根据收发间实际距离选择最短路径方向，每个阶段仅使用单向带宽，配对阶段利用对向带宽，消除了 Ring 算法在大跳距阶段的绕远路浪费。最短路径 + 全带宽利用 = 单维环上带宽和延迟均为最优（optimal in both bandwidth and latency），理论加速比 1~2×（取决于 N 的奇偶性）。
    2. **针对"Pipeline 调度引入气泡"**：DimRotation 将固定维度顺序替换为轮转顺序——N 个 chunk 各以不同维度作为起始维度循环执行，恰好覆盖所有维度在任意时刻的并行传输需求。气泡被完全消除，链路利用率达到 100%。Chunk 数量固定为 N（最小充分数量），调度开销可控。等价于在 N 维 torus 上实现了最优的 collective 调度。
    3. **针对"无容错机制"**：整套方案正交地扩展了容错能力——FoldedRing 在单维环上通过"折叠"全部反向链路构建故障链路的逻辑替代路径，保持 Ring 通信模式完整；MATE 利用 torus 的多维正交特性——健康维度链路可在不冲突的前提下构建故障环相邻节点的逻辑连接——将故障环通信部分卸载到健康环，使性能甚至超过 fault-free baseline。理论依据：N-D torus 中每个平面包含故障环时可额外提供 1 组双向链路，共 N-1 个加速平面可用。MATE 同样适用于 OCS 故障、多故障（同环、多环、异维）等更复杂场景，且加速 phase 可并行化。
