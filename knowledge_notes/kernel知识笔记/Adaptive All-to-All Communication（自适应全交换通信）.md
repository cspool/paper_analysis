## Adaptive All-to-All Communication（自适应全交换通信）

术语是什么？
Adaptive All-to-All Communication 是 ScaleMoE 论文提出的 MoE 分布式训练通信优化技术。传统 MoE 训练框架（DeepSpeed, Tutel）在 expert parallelism 的 all-to-all 通信中，为统一各 GPU 的 message size 而大量使用 zero padding——由于 expert selection 高度不均衡，zero ratio 从训练初期的 88% 迅速升至 98%，导致通信量严重膨胀。Adaptive All-to-All 在运行时监控每个 GPU 的 per-expert token 选择计数，通过一次 all-gather 操作聚合所有 GPU 的计数信息，计算精确的 input slice（第 i 列发给 GPU-i 的 token 数）和 output slice（第 j 行从 GPU-j 接收的输出数），然后使用 NCCL alltoallv 仅传输有效数据，消除所有 zero padding。all-gather 的额外通信开销（44.50ms）相对于被消除的 GB 级 zero 传输可忽略不计。

从kernel调度角度拆解术语：
Adaptive All-to-All 在每个 MoE 层的通信 kernel 调度流程：
```
// 4 GPUs, 4 experts, 每 GPU 10 tokens 为例

// Step 1: Monitoring（本地 GPU kernel）
GPU-1: expert_counts = {E1:4, E2:1, E3:3, E4:2}
GPU-2: expert_counts = {E1:2, E2:6, E3:1, E4:1}
GPU-3: expert_counts = {E1:0, E2:3, E3:7, E4:0}
GPU-4: expert_counts = {E1:1, E2:0, E3:2, E4:7}

// Step 2: All-gather counts（通信 kernel, overhead 44.50ms）
global_counts = all_gather([GPU-1_counts, ..., GPU-4_counts])
// 构建 dispatch matrix: dispatch[i][j] = GPU-i 发往 GPU-j 的 token 数

// Step 3: Adaptive All-to-All dispatch（通信 kernel）
// NCCL alltoallv: 每个 send/recv buffer 大小不同
for each GPU i:
    for each target GPU j:
        send tokens to GPU-j with size = dispatch[i][j]  // 精确大小，无 zero pad

// Step 4: Expert FFN（计算 kernel）

// Step 5: Adaptive All-to-All combine（通信 kernel，对称反向）

// 对比 Baseline: 所有 GPU 按 max(dispatch[i][j] ∀i,j) 统一 buffer size
// e.g., max=7 → GPU-1 发往 GPU-2 仅 1 token 却传输 7-token 等价数据量
// Adaptive: GPU-1→GPU-2 仅传输 1 token 的数据
```

术语一般如何实现？如何使用？
实现于 PyTorch v2.0 + DeepSpeed。在 DeepSpeed 的 MoE dispatcher 中 hook 入监控逻辑，在原有的 all-to-all 调用前插入 all-gather 计数步骤，然后用不等长 buffer 的 `torch.distributed.all_to_all` 变体替换原有等长 all-to-all。与 Megatron-LM 的 alltoallv 功能等价但 dispatcher-agnostic（通过 hook 集成，最小化框架修改）。ScaleMoE 在 32×A100 GPU 上评估：all-to-all 通信开销减少 up to 81%，端到端 speedup 1.71-1.84×（homogeneous）和 2.88-3.31×（heterogeneous）。

- ScaleMoE: A Fast and Scalable Distributed Training Framework for Large-Scale Mixture-of-Experts Models
