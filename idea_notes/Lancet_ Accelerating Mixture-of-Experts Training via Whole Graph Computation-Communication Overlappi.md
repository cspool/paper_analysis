## Lancet: Accelerating Mixture-of-Experts Training via Whole Graph Computation-Communication Overlapping

- baseline方法是什么？
  - **DeepSpeed/Tutel 的 All-to-All + Expert 重叠方案**：现有 MoE 训练优化（Tutel, FasterMoE）的 focus region 仅限于 all-to-all 通信和 expert 计算之间。通过沿 capacity 维度分区 all-to-all 和 experts 并组成 computation-communication pipeline，使 expert 计算与 all-to-all 通信重叠执行。但 all-to-all 通信时间通常远超 expert 计算时间（可达 3.36x），因此重叠仅能隐藏 expert 计算时间，all-to-all 通信本身仍是瓶颈。其他计算（self-attention、前后 Transformer layer 的 FFN、backward 的 dW 计算）不参与重叠，处于整体执行时间的 critical path 上。非 MoE 模型的 communication scheduling（如 P3, ByteScheduler）依赖 all-reduce 同步参数，不适用于 MoE 中 all-to-all 与其他计算之间有直接数据依赖的情形。
  - 全栈执行例子（Baseline Tutel，GPT2-S-MoE 在 8×A100 上前向传播）：
    - **训练算法层**：Top-k routing + expert capacity C 限制 + 超量 token drop。Switch gate 或 Batch Prioritized gate。
    - **系统框架层**：PyTorch + DeepSpeed/Tutel。Expert parallelism with all-to-all。Tutel 沿 capacity 维度将 all-to-all 分为 m 个 micro-batch，与对应的 expert 计算重叠。非 MoE 计算（self-attention, FFN before/after MoE layer）串行等待 all-to-all 完成。
    - **编译框架层**：PyTorch eager mode。Tutel 使用自定义 CUDA kernels 实现 MoE dispatch/gather。无编译器级别优化。
    - **kernel 调度层**：NCCL all-to-all (uniform-shaped C×E) + Tutel CUDA kernels for expert dispatch/combine + cuBLAS GEMM for expert FFN。Partitioned all-to-all 的每个 micro-batch 通信量与专家计算量按比例缩放。
    - **硬件架构层**：8× A100 80GB per node，NVLink intra-node + 100Gbps NIC inter-node。All-to-all 跨节点通信成为 40% 训练时间的瓶颈。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **Lancet 方法**：将 focus region 从仅 all-to-all + expert 扩展到整个训练图的两种新重叠机会：
    1. **前向传播 — Non-MoE 计算分区与 Pipelining**：将 self-attention 和前后 Transformer layer 的 non-MoE 计算沿 batch 维度分区，与 all-to-all + expert 组成更大的 computation-communication pipeline。解决"all-to-all 是瓶颈而 expert 计算不足够填满 overlap"的问题——引入更多 computation（self-attention、FFN）参与重叠，使 all-to-all 有足够的计算量可以隐藏。
    2. **反向传播 — Weight Gradient Computation (dW) 调度**：dW 计算依赖于 activation gradient (dX) 但不依赖于传输 activation gradient 的 all-to-all，因此无依赖路径。将 dW 指令重排序到 all-to-all 之后，使 dW 的 GEMM 计算与 all-to-all 通信并发执行。
    3. **解决 Partition 的数学等价性**：沿 batch 维度分区而非 capacity 维度分区（避免 token drop 差异），但 batch 分区导致 expert capacity 也要缩小从而可能引入额外 token drop。Lancet 实现特殊 gating operator 在各 partition 间传递 capacity 信息（第一个 partition 使用多少 C，后续 partition 动态调整 remaining C），保证 "所有 partition token-to-expert mapping 和 token dropping 与不分区的原版完全一致"。由此引入不规则形状 all-to-all（每个 partition 向每个 expert 发送的 token 数从 0 到 C 不等），通过双趟 NCCL Send/Recv 实现。
    4. **DP 搜索最优 Partition Range**：并非所有 non-MoE 算子都值得分区（GPU kernel launch overhead + SM under-utilization）。DP 算法在 O(N'GK) 复杂度下探索 partition range 和 partition count，Pipeline Scheduler 模拟每个候选方案的时间线并提供反馈。
  - 对应解决 Baseline 缺陷：
    - **All-to-all 是瓶颈（通信时间远长于 expert 计算）** → 引入 non-MoE 计算（self-attention, FFN）和 dW 计算参与重叠，增加可与 all-to-all 重叠的计算量，使 all-to-all 通信被更大范围的计算覆盖。
    - **Focus region 局限于 all-to-all+expert 导致 sub-optimal** → 扩展到 whole training graph，在 forward 中 pipelining non-MoE ops，在 backward 中 scheduling dW。
    - **Partition overhead（kernel launch + SM underutil）** → DP 自动搜索最优 partition range 和 count，避免 over-partitioning。
  - 全栈执行例子（Lancet，GPT2-S-MoE 在 8×A100 上前向传播，Switch gate，3 partitions）：
    - **训练算法层**：Switch gate routing。Special gating operator 在 partition 间传递容量信息，保证 token assignment 数学等价。Batch 维度分区（而非 capacity 维度）。
    - **系统框架层**：PyTorch 模型 → RAF compiler IR。Data parallel + Expert parallel。Lancet 优化通过 RAF pass manager 自动注入，无需修改 Python 代码。
    - **编译框架层**：RAF compiler IR 级别变换。Weight Gradient Computation Schedule Pass（依赖图 BFS+ 贪心分配）→ Operator Partition Pass（DP+CSP+PipelineScheduler）。Pass 输出是重排的 dW 指令和分区的 forward 算子 IR → RAF 编译为可执行代码。
    - **kernel 调度层**：Irregular All-to-All kernel（双趟 NCCL Send/Recv group）+ Tutel MoE dispatch kernel + cuBLAS GEMM for expert FFN。Pipeline scheduler 将 partitioned computation kernel 和 communication 按 stage 交错 launch：Partition 0 的 Non-MoE compute 先 launch，然后 Partition 0 的 All-to-All 与 Partition 1 的 Non-MoE compute 重叠 launch，依次类推。dW GEMM load 与 backward All-to-All 重叠。
    - **硬件架构层**：A100/V100 GPU。Irregular all-to-all 不传输 padding tokens → 总通信量低于 uniform all-to-all。Pipeline 中不同 partition 的 computation/communication 交错执行，提高 GPU SM 和 NIC 利用率。实现 non-overlapped communication time 减少 77%（V100 vs Tutel），吞吐量提升至 1.3x。
