## Lancet: Accelerating Mixture-of-Experts Training via Whole Graph Computation-Communication Overlapping

- 属于编译框架的实现是什么？实验比较什么？
  - Lancet 是 **compiler-based MoE 训练优化系统**，在 RAF compiler（基于 Apache TVM 扩展的开源训练编译器）上实现 13K LoC C++。核心是两个编译 Pass：
    1. **Weight Gradient Computation Schedule Pass**：以 model IR（指令序列）为输入，通过依赖分析识别可与每个 all-to-all 重叠的 dW 指令集合，然后用 best-fit greedy 算法选择 dW 指令重新排序到 all-to-all 之后，使 dW 计算与 all-to-all 通信并发执行。
    2. **Operator Partition Pass**：接收 schedule pass 处理后的 IR，进一步优化前向传播中的 all-to-all。使用 DP 算法搜索最优 partition range，Partition Axis Inferencer（OR-Tools CSP solver）推导每个指令的分区轴，Pipeline Scheduler 模拟 partitioned computation-communication pipeline 的时间线来评估 P(i,n,k) 代价并反馈给 DP。
    3. **Caching Op Profiler**：profile 并缓存每种（partitioned）操作在确切 shape 下的执行时间。Communication Cost Model 通过 profile 不同 input size 下的通信开销构建，cost 之间线性插值。
  - 实验比较：
    - Lancet vs DeepSpeed 0.5.8（无 Tutel kernels）、Tutel 0.3（overlap all-to-all + experts）、RAF（无 Lancet 修改的 baseline）
    - 主要指标：training iteration time、throughput、non-overlapped communication time reduction
    - Cost model 预测精度（3.83% 误差）、优化时间（大部分模型 < 20 min）

- 硬件平台是什么，配置是什么。
  - A100 Cluster: 8× p4de.24xlarge nodes, 每 node 8× A100 80GB + 4×100Gbps NIC
  - V100 Cluster: 8× p3dn.24xlarge nodes, 每 node 8× V100 32GB + 1×100Gbps NIC
  - Ubuntu 20.06, CUDA 11.3, NCCL 2.12.12 (PXN enabled), Docker

- 开源编译框架是什么。修改了什么。
  - **RAF compiler** (https://github.com/awslabs/raf, Apache TVM 扩展，Yu et al., 2023) — 提供完整的 DL 训练编译
  - Lancet 在 RAF 上新增两个 optimization pass（IR 级别变换）：
    1. **Weight Gradient Computation Schedule Pass** (`src/pass/dist_optimization/`)：
       - 新增 CreateDependencyGraph 构建指令依赖图
       - 新增 BFS/DFS dependency analysis 识别 dW 指令集 `W^{Ia}`
       - 新增 best-fit greedy scheduler 实现 dW-to-all-to-all 分配
       - 新增 ReorderInstructions 按分配结果重排 IR
    2. **Operator Partition Pass**：
       - 新增 DP-based partition range selection
       - 新增 Partition Axis Inferencer (CSP, OR-Tools solver)
       - 新增 Pipeline Scheduler（时间线模拟器）
       - 新增 PartitionAxisConstraint (F_Z) 为 Transformer 所有计算算子注入 partition rules
    3. **Irregular All-to-All**：基于 NCCL Send/Recv primitives 实现（双趟 All-to-All：第一趟交换 data size，第二趟传输实际数据）
    4. **MoE dispatching ops**：基于 Tutel kernel 实现
  - Lancet 通过 RAF 的 optimization pass manager 启用，用户无需修改现有代码
  - 三个超参可通过环境变量设置：ρ（最大分区数）、γ（group size）、ι（最大 partition range）

- 开源情况。基于开源文档和论文，使用例子解释编译框架如何使用？作用是什么？至少具体到编译框架输入到输出的全过程。
  - **开源**：GitHub https://github.com/hikettei/Lancet (Apache-2.0)，AWS Labs 镜像 https://github.com/awslabs/Lancet-Accelerating-MoE-Training-via-Whole-Graph-Computation-Communication-Overlapping
  - **编译框架使用流程（端到端）**：
```
1. Input: PyTorch 训练的 MoE 模型 forward/backward 计算图
    ↓
2. RAF compiler 将模型转为 IR (sequence of instructions)
   I = [I_1, I_2, ..., I_N]
   每条指令 I_n = (x^n, y^n, f^n) —— 输入张量、输出张量、算子
    ↓
3. Caching Op Profiler
   - Profile 所有 (partitioned) computation op 在不同 shape 下的执行时间
   - Profile 通信操作在不同 input size 下的执行时间 -> Communication Cost Model
    ↓
4. Pass 1: Weight Gradient Computation Schedule Pass
   a. CreateDependencyGraph(I) -> G = (I, E)
   b. 对每个 all-to-all I_a: 用 BFS/DFS 标记无依赖路径的 dW 指令 W^{I_a}
   c. Best-fit greedy: 分配 W^{I_a} 中的 dW 给 I_a 使 overlap 最大化
   d. ReorderInstructions(Asg) -> I'  (dW 指令移到对应 all-to-all 之后)
    ↓
5. Pass 2: Operator Partition Pass (只处理 forward pass)
   a. Dynamic Programming: T(n) = min_{1<i<n-1}{T(i)+min_{1<k<K} P(i,n,k)}
   b. 对每个候选 range (i,n) 和 partition count k:
      - PartitionAxisInferencer: 构建 CSP (Z 约束 + D 张量依赖)
        find a s.t. F_Z^{f^i}(a_x^i, a_y^i)=1 ∧ a_yj^i=a_xl^k ∀(i,j,k,l)∈D
        使用 OR-Tools 求解 -> 所有 tensor 的 partition axis
      - PipelineScheduler: 
        * 将 partitioned 指令按 stage 组织 (communication stage / computation stage)
        * 模拟时间线：start_time = max(dependency_end, prev_same_type_partition_end)
        * 报告 P(i,n,k) = 最后一个指令的 end_time
   c. 选择最优 (i,n,k) 组合 -> IR 转换
    ↓
6. Output: 优化后的 IR，包含：
   - 重排的 dW 指令（与 all-to-all 重叠）
   - 分区并 pipeline 的前向算子
   - 生成的不规则 all-to-all kernels
    ↓
7. RAF 将优化后的 IR compile 为可执行代码
    ↓
8. Execution: 训练循环中，Lancet 优化后的计算图自动执行
   - Forward: partitioned non-MoE + MoE computation/communication pipeline
   - Backward: dW 计算与 all-to-all 重叠执行
```
  - **使用命令示例**（基于 GitHub README）：
    - `python create_nccl_profiles.py` — 生成通信 cost model
    - `python run_exp_configs.py --lancet-profile` — profile 阶段
    - `python run_exp_configs.py --lancet-opt` — 应用 Lancet 优化
    - `python run_exp_configs.py` — benchmark（无 Lancet 优化）
