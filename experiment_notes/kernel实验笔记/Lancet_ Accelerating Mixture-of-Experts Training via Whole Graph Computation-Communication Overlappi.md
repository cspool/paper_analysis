## Lancet: Accelerating Mixture-of-Experts Training via Whole Graph Computation-Communication Overlapping

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - Lancet 在 kernel 调度/运行时计算层面的实现：
    1. **Irregular All-to-All (All-to-Allv) 实现**：由于 Lancet 的 partition 方案沿 batch 维度分区导致每个 partition 发送给各 expert 的 token 数不均匀，实现了不规则形状的 all-to-all。采用双趟 NCCL 通信：第一趟 All-to-All 交换各 GPU 间实际要传输的 data size，第二趟 All-to-All 按照已知 size 传输实际数据。通过 grouped NCCL Send/Recv primitives 组合实现（非直接调用 ncclAllToAll）。
    2. **Tutel MoE Dispatching Kernel**：MoE 的 token-to-expert dispatch/gather 操作基于 Tutel 的 fast dispatching kernel 实现，处理动态 routing 后的 token 重排。
    3. **Pipeline Scheduling 内核**：partitioned computation/communication 被组织成 stage-based pipeline。相同类型的指令（all computation 或 all communication）组成一个 stage，各 partition 按 partition index 顺序执行。每个 kernel 的 start time 由依赖关系（数据依赖 + pipeline 顺序依赖）决定。
    4. **Caching Op Profiler + Communication Cost Model**：对每个 (partitioned) computation kernel 在不同 shape 下 profile 并缓存执行时间；通信 cost model 通过在不同 message size (1KB, 2KB, ..., max) 下 profile NCCL 通信操作构建，cost 间线性插值；对不规则 all-to-all，使用 static-shape approximation（以 C/n 容量的 uniform-shaped profile 值近似 n-partition 不规则 all-to-all 的通信时间）。
  - 实验比较：Lancet vs DeepSpeed vs Tutel vs RAF 在 V100/A100 集群上的训练吞吐量、iteration time decomposition（Non-overlapped Computation/Communication + Overlapped 的时间分解）

- 后端平台是什么，配置是什么。
  - **A100 GPU** (p4de.24xlarge): NVIDIA A100 80GB × 8 per node × 8 nodes, 4×100Gbps NIC per node
  - **V100 GPU** (p3dn.24xlarge): NVIDIA V100 32GB × 8 per node × 8 nodes, 1×100Gbps NIC per node
  - CUDA 11.3, NCCL 2.12.12 (PXN enabled), Ubuntu 20.06 Docker
  - all-to-all 通信基于 NCCL Send/Recv primitives，未使用 ncclAllToAll 高层 API

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估脚本：`run_exp_configs.py`（GitHub 仓库提供）
    - `--lancet-profile`：profile 阶段，生成 execution time cache
    - `--lancet-opt`：应用 Lancet 优化并运行
    - 无 flag：baseline 运行（RAF 无 Lancet 优化）
  - 通信 profiling：`create_nccl_profiles.py` 生成 NCCL communication cost model
  - 修改/新增内容：
    1. **Irregular All-to-All Kernel**：13K LoC C++ 中的通信层实现，基于 NCCL `ncclSend`/`ncclRecv` 组合成 grouped communication。Input/Output buffer 固定 shape (G × C)，实际数据只填充 buffer 的部分区域（由 gating 结果决定），避免传输 padding token。
    2. **Tutel-based MoE Dispatch**：复用 Tutel 的 fast expert dispatch kernel 进行 token permutation
    3. **Partition 约束函数 F_Z**：为所有 Transformer 计算算子（MatMul, LayerNorm, Attention, Activation 等）定义 partition axis 约束规则
    4. **Pipeline Scheduler**：新的 kernel launch 调度器，根据 partition index 和数据依赖编排所有 partitioned kernel/communication 的 launch 顺序

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源**：https://github.com/hikettei/Lancet (Apache-2.0)，AWS Labs 镜像 https://github.com/awslabs/Lancet-Accelerating-MoE-Training-via-Whole-Graph-Computation-Communication-Overlapping
  - **Irregular All-to-All Kernel 执行原理**：
```
Input: Gating result -> per-expert token counts per GPU
  Step 1: 在每 GPU 上根据 gating function 决定每个 expert 收到的 token 数
          Input buffer (shape G×C) 仅填充实际 token
  Step 2: 第一趟 All-to-All (data size exchange)
          - 每 GPU 对每个 target GPU 发送实际要传输的 data size
          - 实现: grouped NCCL Send/Recv (每个 target/src GPU 一个 Send/Recv)
          - 此时仅传输 int 类型的 count 信息，通信量极小
  Step 3: 第二趟 All-to-All (actual data transfer)
          - 每 GPU 根据 Step 2 获知的 size 发送/接收实际 token data
          - 实现: grouped NCCL Send/Recv with known sizes
          - 由于不传输 padding tokens，总体通信量低于 uniform-shaped all-to-all
  Step 4: Expert computation
          - 各 GPU 上每个 expert 处理收到的 Ci 个 token (Ci ≤ C)
          - Tutel kernel 执行 expert FFN (可能含 padding 到 block size)
  Step 5: Reverse irregular all-to-all (同上 Steps 1-3 的逆过程)
          - 将 expert output 发送回原 GPU
```
  - **Pipeline 执行全过程**（前向传播中 3 个 partition 的例子）：
```
Timeline:
  Partition 0:  [Non-MoE Compute_0] [All-to-All_0] [Expert_0] [Non-MoE Compute_0_post]
  Partition 1:     [Non-MoE Comp_1] [All-to-All_1] [Expert_1] [Non-MoE Compute_1_post]
  Partition 2:        [Non-MoE Comp_2] [All-to-All_2] [Expert_2] [Non-MoE Compute_2_post]

  其中 Non-MoE Compute_i 与 All-to-All_{i-1} 和 Expert_{i-1} 重叠执行
  All-to-All_i 与 Non-MoE Compute_{i-1}_post 和 Expert_{i-1} 重叠执行
  Stage 结构：
    Stage 0 (Compute):  [NMC_0, NMC_1, NMC_2] 按 partition index 顺序 launch
    Stage 1 (Comm):     [A2A_0, A2A_1, A2A_2]
    Stage 2 (Compute):  [Expert_0, Expert_1, Expert_2]
    Stage 3 (Compute):  [Post_0, Post_1, Post_2]
  Pipeline Scheduler 计算每个 kernel 的 start_time：
    start(NMC_i) = max(end(NMC_{i-1}), end(Post_{i-1} of prev layer))
    start(A2A_i) = max(end(NMC_i), end(A2A_{i-1}))
    start(Expert_i) = max(end(A2A_i), end(Expert_{i-1}))
```
