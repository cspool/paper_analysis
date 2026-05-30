## UltraAttn: Efficiently Parallelizing Attention through Hierarchical Context-Tiling

- baseline方法是什么？
  **Ring-based Context Parallelism（Ring Attention, ZigZag-Ring Attention, Striped Attention）**：现有的 context parallelism 系统沿单一维度（通常 $c_q$）用 stripe-like partition 将 attention workload 划分为带状分配给各 GPU，并通过 ring-based communication pattern 在 GPU 间轮转 KV。具体执行过程：在 ring attention 中，每个 GPU 持有部分 Q（沿 $c_q$ 维划分）和部分 KV（沿 $c_{kv}$ 维划分），通过 send/recv 在环中轮转所有设备持有的 KV，执行 per-step 的 attention 计算（每次一个 step：本地 Q × 当前持有的 KV → 部分 attention → 将 KV 传给下一个 rank → 从上一个 rank 接收新的 KV）。zigzag ring attention 在此基础上调整 stripe 顺序以获得更好的 causal attention 负载均衡；striped attention 对 Q 和 KV 做交织划分。

  全栈执行例子（Llama2-7B causal attention training, CP=64, S=512K, 8 nodes × 8 H100）：
  - **模型推理算法层**：causal attention 计算 $O = \text{Softmax}(\text{Mask}(QK^T/\sqrt{d_k}))V$。64 GPUs 在 $c_q$ 维度并行，每个 GPU 负责约 512K/64 = 8K tokens 的 Q 块。KV 在所有 GPU 间通过 ring 轮转。
  - **系统框架层**：PyTorch + Megatron-LM（或类似框架）管理 context parallel group。每个 GPU 发射 FlashAttn kernel 对本地 Q × 当前持有的 KV 做 attention。通过 NCCL send/recv 在 ring 中传输 KV。
  - **编译框架层**：论文未明确说明。使用 PyTorch 原生框架，无自定义编译 pass。
  - **kernel调度层**：
    1. Stripe-like partition：attention workload 沿 $c_q$ 维划分为 64 个 stripe（每个 GPU 1 条）。Stripe 形状为 $1 \times N$（1 维划分），沿 Q 和 KV 的 projection lengths 均为 O(N)。
    2. Ring-based communication：每个 step，每 GPU 同时执行三个任务——本地 attention block 计算（FlashAttn kernel）、发送当前 KV 到下一个 rank、从上一个 rank 接收新的 KV。Step 数 = CP = 64（对于 ring attention, step=CP 意味着 KV 轮转完整一圈）。
    3. Fine-grained kernel split：每个 step 的计算量极小（约 8K × 8K attention），以最大化 computation-communication overlap，但导致极低的单 kernel device utilization（SM occupancy 低）。
    4. Inter-node bottleneck：跨节点的 ring 连接在 8 节点时仅利用 2 个 NIC 的单向带宽（图 3a，ring 经过每个 node 时仅使用出/入各 1 NIC），浪费 75% 的 NIC 带宽。
    5. Redundant communication：zigzag ring attention 约 25% KV 传输是浪费的（对应当前 GPU 不需要的部分 KV blocks），标准 ring attention 接近 50% 浪费。
  - **硬件架构层**：8 节点 × 每节点 8× H100-NVLink-80GB（共 64 GPU）。节点内 NVLink 450GB/s 双向。节点间 8× 400Gb/s InfiniBand EDR，每 GPU 与 1 NIC 有 PCIe-5.0 affinity。Ring 通信未利用网络拓扑异构性——NVLink（intra-node）和 InfiniBand（inter-node）带宽差异巨大，但 ring pattern 将它们同等对待。

  Baseline 缺陷：
  - (a) **High Communication Traffic**：stripe-like partition（$1 \times N$ 形状）的 workload projection sum 是 O(N)，而 ideal curled-up partition（$\sqrt{N} \times \sqrt{N}$）的 projection sum 是 $O(\sqrt{N})$，相差一个数量级。stripe 瘦长形状导致每 GPU 需要接收大量不必要的 Q/KV 投影。
  - (b) **Inflexible Kernel Granularity**：ring-based 系统为最大化 computation-communication overlap 将 attention 拆分为极细粒度 kernel（每 step 一个小 kernel），但过度细粒度导致单 kernel SM utilization 极低。kernel granularity 存在 U 型性能曲线——太细则设备利用率低，太粗则重叠机会少。
  - (c) **Bandwidth Waste of Ring Communication**：ring 模式将所有 KV 轮转到每个设备，但在 block sparse attention 中许多设备只需部分 KV，导致 ~25%（zigzag ring）到 ~50%（ring）的冗余通信。跨节点时 ring 仅使用 2 NIC 单向带宽（75% NIC 带宽浪费）。
  - (d) **Poor Strong Scalability**：随着 CP 增加（固定 context length），每 GPU 的 computation volume 反比下降（$O(1/CP)$），但 ring pattern 的 communication volume 几乎不变，导致逐渐 communication-bound。
  - (e) **No Support for Irregular/Block Sparse Attention**：ring-based 系统的 step-by-step structure 在 block sparse attention 下遭遇 severe in-step load imbalance——每 step 只有部分 GPU 需要计算 attention block，其余 GPU 空等。
  - (f) **Sub-optimal Kernel Scheduling**：现有系统（FlexFlow、Tofu 等）使用 BFS-based 方法寻找 feasible topological order，但无法保证最优调度顺序。通信 contention（共享同一带宽的 kernel 重叠执行会延长各自执行时间）未被避免。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **UltraAttn：Hierarchical Context-Tiling System**，通过三层 context-tiling（node-level, device-level, kernel-level）+ ILP-based runtime 实现高效的 context parallelism for irregular attention。

  对应关系：
  - (a) → **Device-Level 2D Context-Tiling**：将 attention workload 沿 Q 和 KV 两个维度同时划分（而非仅一维），形成 curled-up ($\sqrt{N} \times \sqrt{N}$) 形状的 tile。通过 ILP 在 $P \times P$ grid 上分配每个 block $B_{r,c}$ 到设备 $U_g$，最小化 MCV（Maximum Communication Volume）。形式上：定义 partition degree P，将 workload 划分为 $P \times P$ grid，ILP 变量 $x_{r,c,g}$ 控制分配，约束 Allocate Uniqueness（每 block 到唯一设备）+ Computation Balance（$\sum_{FB} x \times 1 + \sum_{CB} x \times 0.5 \le \tau$），最小化 MCV。从 stripe-like 的 O(N) projection sum 降为 curled-up 的 $O(\sqrt{N})$，通信量降低一个数量级。
  - (b) → **Kernel-Level Context-Tiling**：在 parallel dependency graph（DAG）上通过三种 transformation（computation batching, peer-to-peer comm batching, collective comm batching）进行图变换，使用贪心策略自适应选择最优 kernel granularity。三种 substitution 的 transformation candidates 按 gain（融合后减少的时间）降序排序，贪心应用到 DAG → ILP runtime 评估执行时间 → 保留改善的变换。从而在 kernel overlap（灵活性）和单 kernel device utilization（效率）之间找到最优平衡点。对 dense attention 特别有效（kernel fusion 机会多），小 $\frac{S}{CP}$ 和 Nh 时增益更大。
  - (c) → **Node-Level Context-Tiling + Groupwise Peer-to-Peer**：将 context-tiling 解耦为 node-level 和 device-level 两层。Node-level tiling 将每个 node 视为集成设备，仅在 node 间执行 minimized 通信。节点间使用 groupwise peer-to-peer（图 3b），每个 node pair 通过多个 NIC 并行通信，充分利用全部 NIC 带宽（vs ring 的 75% 浪费）。Device-level tiling 在 node 内使用 NVLink peer-to-peer。两层建模类比统一——workload block ↔ node 的 distributed attention computation，peer-to-peer ↔ groupwise peer-to-peer，profiling 方式对应。
  - (d) → **2D Partition + Communication Minimization ILP**：当 CP 增加时，2D curled-up partition 的 communication volume 也随 $1/\sqrt{CP}$ 下降（vs ring 的常数），保持 communication 与 computation 的比例。ILP 显式最小化 MCV 确保 communication 不成为 bottleneck。实验验证 UltraAttn 在 CP=16→64 时实现 near-linear strong scalability（图 10）。
  - (e) → **ILP 自适配不规则 Workload**：ILP formulation 通用处理任意 FB/CB/EB 集合定义的 attention pattern（causal, full, strided, global+local, star, streaming），自动计算最优 workload-to-device 分配。Device-level tiling 打破 ring-based 的 step-by-step 结构，允许更灵活的通信模式（不限于顺序轮转），从而消除 in-step load imbalance。对于 block sparse attention，UltraAttn 在 CP=64 时获 $10.2\times$（strided）到 $5.7\times$（global+local）加速。
  - (f) → **ILP-based Runtime Kernel Scheduling**：将 DAG 的 kernel 调度建模为 ILP——按共享带宽分组 kernel 到不同 CUDA stream（避免 contention），对每 stream 求解 ILP（变量 $S_v$ + $Order_{uv}$，约束 Stream Exclusivity + Dependency，目标 minimize $End\_Time$），获得理论最优执行顺序。对比 FlexFlow BFS-based scheduling，ILP 方案在涉及复杂 kernel 依赖和 comparable computation-communication duration 的场景下有显著优势。

  全栈执行例子（UltraAttn strided attention training, CP=64, S=512K, Nh=1, 64 GPU）：
  - **模型推理算法层**：strided attention（Figure 2c diagonal stripes pattern），$O = \text{Softmax}(\text{Mask}(QK^T/\sqrt{d_k}))V$。与 baseline 相同的 attention 计算逻辑，但 mask pattern 为 stride-based sparse。
  - **系统框架层**：UltraAttn 作为 PyTorch 库（~10K LoC Python）。运行时：读取 attention pattern → 执行 hierarchical context-tiling → 生成 parallel dependency graph → ILP runtime 调度 → CUDA graph 编译 → GPU 执行。Context remap 预处理（$\phi(t_i) = \lfloor i \cdot 16/S \rfloor \mod 4$）增强 locality。
  - **编译框架层**：论文未明确说明。UltraAttn 直接使用 FlashAttn 2.5.7 作为 computation backend，NCCL 2.21 C-level API 作为 communication backend。无自定义编译 pass。
  - **kernel调度层**：
    1. Adaptive Workload Partition：计算 P 使 $DLI_{P,CP} \le \theta_{DLI}$，将 strided pattern 划分为 $P \times P$ grid
    2. Node-Level ILP（$CP_{node}=8$）：将 $P \times P$ grid 的 blocks 分配到 8 nodes，minimize MCV → node 间 groupwise peer-to-peer 通信计划
    3. Device-Level ILP（$CP_{device}=8$）：每个 node 内的 blocks 分配到 8 GPU，minimize MCV → GPU 间 peer-to-peer 通信计划
    4. DAG 构建：computation kernel（FlashAttn, 矩形节点）+ recv kernel（NCCL recv, 椭圆节点）+ send kernel（NCCL send, 菱形节点）构成 DAG
    5. Kernel-Level Tiling：贪心选择变换（computation batching/comm batching）应用到 DAG
    6. ILP Runtime：按共享带宽分组 kernel → 各 stream 内 ILP 求解最优顺序 → CUDA stream graph
    7. CUDA Graph 执行：各 stream 并行执行，stream 内串行，computation（FlashAttn forward）与 communication（NCCL send/recv）按 ILP 最优 schedule 交错
  - **硬件架构层**：8 nodes × 8 H100-NVLink-80GB。Node-level tiling 使用 groupwise peer-to-peer（每 node pair 利用 8 NIC 并行），device-level tiling 使用 NVLink peer-to-peer。最终 64 GPU 的 distributed attention 模块强可扩展性接近线性（图 10a）。

  关键设计选择与 baseline 缺陷的对应：
  - **defect (a): stripe-like high traffic** → 2D context-tiling：从 $1 \times N$ partition 变为 $\sqrt{N} \times \sqrt{N}$ curled-up partition，projection sum 从 O(N) 降至 $O(\sqrt{N})$
  - **defect (b): inflexible kernel granularity** → 贪心 kernel-level tiling + ILP runtime 评估：遍历三种 substitution 的 bounded 搜索空间，贪心选择，通过 ILP runtime 准确评估执行时间
  - **defect (c): ring bandwidth waste** → node-level tiling + groupwise peer-to-peer：仅在 node 间传必需数据，充分利用所有 NIC
  - **defect (d): poor strong scalability** → 2D tiling 的 communication 也随 CP 增加而下降：$1/\sqrt{CP}$ communication scaling vs ring 的 constant
  - **defect (e): no support for irregular attention** → ILP formulation 通用处理任意 FB/CB/EB pattern：打破 step-by-step 结构消除 in-step load imbalance
  - **defect (f): sub-optimal kernel scheduling** → ILP runtime 形式化 kernel scheduling：stream exclusivity + dependency constraints，理论上最小化 $End\_Time$
