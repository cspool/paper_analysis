## FasterMoE modeling and optimizing training of large-scale dynamic pre-trained models

- baseline方法是什么？
  - **Baseline 方法**：
    1. **ZeRO Optimizer（数据并行 baseline）**：使用 DeepSpeed 的 ZeRO stage 3，将 optimizer states、gradients、parameters 按 tensor 维度切分到所有 worker。MoE 模型被复制到所有 worker 上。每 iteration：forward（本地 GPU 全模型计算）→ backward → all-reduce 同步梯度 → 参数更新。ZeRO stage 3 虽然能容纳大模型，但引入了大量通信开销（梯度同步和参数 gather/scatter），导致 DDL-Roofline 中 R_CC 极低，训练效率差。
    2. **FastMoE（expert parallelism baseline）**：使用 expert parallelism，expert 分布在不同 worker 上。每 iteration：非 MoE 层 = 数据并行（本地计算），MoE 层 = all-to-all-v 发送 tokens 到目标 expert worker → 各 worker 计算本地 expert → all-to-all-v 返回输出 → 重组序列。所有通信和计算操作按同步模式执行（先通信完再计算，或先计算完再通信），导致通信和计算硬件交替闲置。
    3. **GShard / BASE Layer（修改 expert selection 的 baseline）**：GShard 使用辅助 loss 做 load balancing + top-2 gate；BASE Layer 使用 matching 算法分配 tokens 到 experts。两者都修改了 expert 选择以平衡计算负载，但未考虑网络拓扑对通信性能的影响。
  - **全栈执行例子（以 MoE-GPT 3.42B, 16 experts, 16×V100, johnny 集群，FastMoE baseline 为例）**：
    - **训练算法层**：MoE-GPT 3.42B, H=2048, α=2, 12 层，每层 16 experts，每个 expert 为 2 层 FC（GeMM: (B_w, H)×(H, αH) 和 (B_w, αH)×(αH, H)）。Gate 使用 softmax top-2 选择 expert。每 token 激活 2 个 expert。
    - **系统框架层**：FastMoE（基于 PyTorch），每 worker 持有 1 个 expert（64 GPUs / 16 experts 时每个 expert 有多个副本或部分 workers idle）。非 MoE 层使用数据并行，MoE 层使用 expert parallelism。All-to-all 通过 NCCL/MPI 同步原语实现。
    - **编译框架层**：PyTorch eager mode，无编译框架修改。
    - **kernel调度层**：NVIDIA cuBLAS 执行 GeMM。All-to-all 由 NCCL group 调用完成。调度顺序：all-to-all（发送 tokens）→ 同步 barrier → GeMM FC1 → GeMM FC2 → all-to-all（返回输出）→ 同步 barrier。通信和计算严格串行。
    - **硬件架构层**：16× V100-PCIE，2 节点，节点内 PCIe switch 互连，节点间 Infiniband 50Gb/s。
  - **Baseline 痛点**：
    1. **动态负载不均衡（skewed expert selection）**：训练数据自然服从偏斜分布，热门 expert 接收远超平均的 tokens（观察到 4/16 experts 处理约 20% tokens，3.2× 平均值），导致其所在 worker 重载而其他 worker 空闲，且此模式随 training iteration 动态变化。
    2. **同步执行模式低效**：All-to-all 通信和 GeMM 计算严格串行执行，当通信进行时 GPU 计算单元闲置，反之亦然。在非均匀 token 分布下，通信和计算的不均衡进一步放大资源浪费。
    3. **网络拓扑与 expert 选择不匹配**：All-to-all 通信在树形拓扑的上层链路（跨节点）产生严重拥塞——跨节点流量 T_n = M(N-1)/N · BH ≈ M 倍于节点内流量 T_w = (MN-1)/MN · BH。现有方法（GShard, BASE Layer）仅均衡计算负载，未考虑网络拓扑对通信的影响。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **论文方法（FasterMoE）**：通过性能模型引导的三个系统性优化，分别解决上述三个痛点：
    1. **Dynamic Shadowing（解决痛点 1）**：在每 iteration 运行时，通过轻量级 Algorithm 1 判断是否将热门 expert 参数广播到所有 worker（影子化），使热门 expert 的 tokens 在各 worker 本地计算，消除热门 worker 的计算瓶颈。核心决策基于性能模型——当 token 传输开销大于模型传输开销（B_max > rαH），或减少的计算延迟大于增加的通信开销时，启用影子化。
    2. **Fine-grained Smart Scheduling（解决痛点 2）**：将 all-to-all 通信拆分为 n 个 group 的 pairwise exchange 操作序列 S/C/R，分别在独立的 communication stream 和 computation stream 上异步执行，打破同步 barrier。通过将最快的操作（同 group 的 S_{i,0} 和环通信的 R_{i,n-1}）放在首尾，最小化首尾通信开销对整体延迟的影响。使程序从 DDL-Roofline 的半理想曲线跃升至理想曲线附近。
    3. **Topology-aware Gate（解决痛点 3）**：修改 expert 选择策略，限制跨节点 token 数为 L = W_net/(M·W_local) · B，将超出限制的 token 重新分配给本地节点内的 expert。同时保留 best-fit 的 token-expert 对，减少对模型质量的影响。
  - **全栈执行例子（FasterMoE w/ all optimizations, MoE-GPT 3.42B, 16 experts, 16×V100, johnny 集群）**：
    - **训练算法层**：与 baseline 相同的 MoE-GPT 模型结构。差异：(a) 每 iteration 开始前执行 SelectShadowExperts 算法，在每 worker 上基于 token-to-expert 分配矩阵 T 判断影子化哪些 expert；(b) Gate 使用拓扑感知逻辑，限制跨节点 tokens 上限 L；(c) 被影子化的 expert 参数在 forward 开始时 broadcast 到所有 worker，backward 结束时 reduce 梯度并 update 在原 worker。
    - **系统框架层**：FasterMoE（基于 FastMoE 扩展），在 FastMoE 的 transformer.py 中实现动态影子化决策逻辑（fastermoe/fmoe/transformer.py:34），通过环境变量 FMOE_FUSE_GRAN 控制分组粒度。Communication stream 和 computation stream 分别为独立 CUDA stream。使用 grouped pairwise exchange 算法替代 coarse-grained all-to-all。
    - **编译框架层**：论文未明确说明（沿用 PyTorch eager mode）。
    - **kernel调度层**：不同：S/C/R 操作序列在 comm stream 和 comp stream 上交错执行（图 8b/c）。S_{i,0} 接收本地 group tokens 最快（无跨节点连接），排在首位；R_{i,n-1} 为环通信（充分利用带宽），排在末位。comp stream 从 C_{i,0} 到 C_{i,n-1} 连续执行。两个 stream 间的依赖：C_{i,j} 等 S_{i,j} 完成，R_{i,j} 等 C_{i,j} 完成。
    - **硬件架构层**：与 baseline 相同硬件。执行特征：(a) 跨节点通信量减少至 W_net/W_local · BH（拓扑感知门控）；(b) GPU SMs 在通信期间执行 GeMM 计算（stream overlap）；(c) 热门 expert 所在 worker 的计算负载转移到全部 worker 分担（影子化）。
    - **关键性能对比**：
      - johnny 集群：vs ZeRO stage 3 加速 6.63×，vs FastMoE 加速 2.20×（影子化 1.95× + 调度 1.40× 联合）。
      - trevor 集群（64 GPU）：vs ZeRO stage 3 加速 17.87×，vs FastMoE 加速 5.72×（影子化 4.74× + 调度 1.40× 联合）。
      - 收敛加速：vs GShard 1.37× 更快收敛，vs BASE Layer 2.19× 更快收敛（FasterMoE w/ topo. gate, MoE-GPT）。
