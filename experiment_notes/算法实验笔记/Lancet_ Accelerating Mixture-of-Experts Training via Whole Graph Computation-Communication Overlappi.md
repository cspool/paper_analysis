## Lancet: Accelerating Mixture-of-Experts Training via Whole Graph Computation-Communication Overlapping

- 属于算法pipeline的实现是什么？实验比较什么？
  - Lancet 提出两种算法 pipeline 优化：
    1. **Weight Gradient Computation Scheduling（反向传播）**：将 MoE 模型训练反向传播中的 weight gradient computation (dW) 算子调度到与 all-to-all 通信重叠执行。分析 IR 依赖图，用 BFS/DFS 识别与每个 all-to-all 无依赖路径的 dW 指令集合。然后采用 best-fit greedy 算法：顺序遍历 all-to-all 指令，对每个 all-to-all 从可用 dW 池中贪心选取总执行时间最接近 current unoverlapped time 的 dW 算子集，使 all-to-all 被 dW 计算最大程度覆盖。
    2. **Operator Partition with Dynamic Programming（前向传播）**：将前向传播中的 non-MoE 计算（如 self-attention、前一个/后一个 Transformer layer 的 FFN）分区并与 all-to-all + expert 计算组成 computation-communication pipeline。使用 DP 公式 `T(n) = min_{1<i<n-1} {T(i) + min_{1<k<K} P(i,n,k)}` 搜索最优 partition range（包含哪些 non-MoE 算子和多少个 partitions）。其中 `P(i,n,k)` 为指令 i 到 n 被分为 k 个 partition 并经 pipeline scheduling 后的端到端时间。Partition axis 通过约束满足问题（CSP）求解，使用 OR-Tools。Pipeline scheduler 按 stage 组织 partitioned 算子并模拟时间线得到 P(i,n,k)。
  - 实验比较：
    - Lancet vs DeepSpeed 0.5.8、Tutel 0.3、RAF（无 Lancet 优化的 baseline）
    - 训练吞吐量（tokens/s 或 iteration time）、通信重叠度（non-overlapped communication time 减少量）
    - 两种 gating 方法：Switch gate（允许 pre-MoE 和 post-MoE 分区）和 Batch Prioritized gate（只允许 post-MoE 分区）
    - Ablation study：仅 scheduling vs 仅 pipelining vs 两者组合的加速比
  - 结果：non-overlapped communication 减少最多 77%（vs Tutel on V100），端到端加速最高 1.3x

- 硬件平台是什么，配置是什么。
  - **A100 Cluster**: Amazon EC2 p4de.24xlarge × 8 nodes，每 node 8× NVIDIA A100 80GB GPU，4×100 Gbps NIC
  - **V100 Cluster**: Amazon EC2 p3dn.24xlarge × 8 nodes，每 node 8× NVIDIA V100 32GB GPU，1×100 Gbps NIC
  - 软件环境：Ubuntu 20.06, CUDA 11.3, NCCL 2.12.12 (PXN enabled), Docker
  - Weak scaling 评测，从 1 node (8 GPUs) 扩展到 8 nodes (64 GPUs)

- 模型是什么。数据集和bench分别是什么。
  - **模型**：
    - **GPT2-S-MoE**：12 layers, hidden dim 768, 每 GPU 2 experts
    - **GPT2-L-MoE**：24 layers, hidden dim 1024, 每 GPU 2 experts
    - 基于 Huggingface Transformers v4.18.0 的 GPT-2，每隔一个 Transformer block 的 FFN 替换为 MoE layer
    - Expert 数量随 GPU 数量线性扩展（per-GPU 保持 2 experts）
  - **数据集**：WikiText (Merity et al., 2016)
  - **Benchmark Metrics**：training iteration time, throughput, non-overlapped communication time
  - Input sequence length 固定 512；A100 上 GPT2-S-MoE batch size 24/GPU, GPT2-L-MoE 48/GPU；V100 上 16 和 8

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源**：GitHub https://github.com/hikettei/Lancet (Apache-2.0)，AWS Labs 镜像 https://github.com/awslabs/Lancet-Accelerating-MoE-Training-via-Whole-Graph-Computation-Communication-Overlapping
  - **Weight Gradient Computation Scheduling 伪代码**：
```
Input: instruction sequence I (model IR)
  G = CreateDependencyGraph(I)
  Ia = [all-to-all instructions in I]
  For each all-to-all Ij^a:
    Wj = {Ik in I | no directed path between Ij^a and Ik}
  t^a, t^W = profile execution time of all instructions
  W_used = {}; Asg = {}
  For each all-to-all Ii^a in Ia:
    tu = ti^a  // unoverlapped time
    While tu > 0 AND W_i \ W_used != empty:
      jmin = argmin_j{|tu - tj^W| | Ij^W in W_i, Ij^W not in W_used}
      tu = tu - t^W_{jmin}
      W_used.add(I^W_{jmin}); Asg[I^W_{jmin}] = Ii^a
  I' = ReorderInstructions(Asg)
    // place dW instructions right after their assigned all-to-all
```
  - **DP Partition Range Selection**：
```
T(n) = min_{1<i<n-1} { T(i) + min_{1<k<K} P(i,n,k) }
// P(i,n,k): pipeline time of instructions i..n partitioned k ways
// For each P(i,n,k):
//   1. PartitionAxisInferencer (CSP) -> axes for all tensors
//   2. PipelineScheduler -> simulate timeline -> end-to-end time
// K is max partitions (default 8), limited by batch dim size
```
  - **Partition Axis CSP** (以矩阵乘 Y=XW 为例)：
    - `(ax1=0 ∧ ax2=-1 ∧ ay1=0) ∨ (ax1=-1 ∧ ax2=1 ∧ ay1=1)`
    - ax1=0 表示沿 X 的 row 维度分区（W 不变，Y 沿 row 分区）
    - ax2=1 表示沿 W 的 column 维度分区（X 不变，Y 沿 column 分区）
    - -1 表示不分该维度
  - **Pipeline Scheduling**：将 partitioned 指令按 stage 组织（所有 computation 连续执行为一个 stage，所有 communication 为一个 stage），各 partition 按 partition index 顺序调度，每个指令的 start time = max(依赖指令 end time, 同类型前一个 partition 指令的 end time)
