## Disaggregated Expert Parallelism (DEP)

术语解释
Disaggregated Expert Parallelism (DEP)，也称为 Attention-FFN Disaggregation (AFD)，是一种专为 MoE 大模型高吞吐推理设计的并行策略：将 attention 层和 expert 层物理分离到两组不同的 GPU 上独立扩展。

术语是什么？
DEP 将多 GPU 系统划分为两个专用功能组：Attention Group (AG) 存储并处理所有 Transformer 的标准组件（Self-Attention 层和 Shared Expert），Expert Group (EG) 存储所有 sparse MoE experts。AG 内参数全复制（无 group 内通信），EG 内每个 expert 的 token 路由天然限制在单 GPU 上（无 expert 间通信）。跨 group 通信仅发生在两个方向：Attention-to-Expert (A2E) 和 Expert-to-Attention (E2A)。这种解耦使得 AG 和 EG 可以独立扩展资源，克服了单体并行方案中 attention 和 expert 无法分别优化的局限。

从系统架构角度拆解：
在一次 MoE model forward pass 中，DEP 的执行流程为：
1. AG 接收输入 tokens，每个 AG GPU 上本地计算 attention 层（MLA/MHA）和 shared expert（如存在）
2. A2E 阶段：AG 将处理后的 tokens 按 gating 结果路由到持有对应 expert 的 EG GPU
3. EG 接收 tokens 后，每个 GPU 对其持有的 E/eg 个 expert 分别执行 Feed-Forward Network 计算
4. E2A 阶段：EG 将 expert 输出收集并返回给 AG，AG 继续下一层的 attention 计算
核心时序问题：数据依赖导致 AG 计算→A2E 通信→EG 计算→E2A 通信→AG 计算形成串行链，不优化时 GPU 空转严重。MegaScale-Infer 的 PP-Pipe 通过 micro-batch pipelining 缓解此问题。FinDEP 进一步通过 fine-grained partitioning (r1 × r2 两级流水线) 最大化 AG、EG、A2E、E2A 四类任务的重叠。

术语一般如何实现？如何使用？
DEP 的实现依赖：(1) GPU 分组——通过进程组 (NCCL communicator) 将 P 个 GPU 划分为 ag 个 AG 设备 + eg 个 EG 设备；(2) 参数放置——attention 参数和 shared expert 参数全复制到所有 AG 设备，E 个 routed experts 分布到 eg 个 EG 设备；(3) 通信原语——A2E 使用 NCCL send/recv 或 all-to-all 将 token 从 AG 路由到 EG，E2A 反向收集结果；(4) 调度——需配合 pipelining（如 PP-Pipe 的 micro-batch 调度或 FinDEP 的 r1×r2 fine-grained 调度）隐藏通信延迟。典型配置：DeepSeek-V2 使用 (ag=3, eg=5) 或 (ag=8, eg=24)；Qwen3-MoE 使用 (ag=4, eg=4)。

涉及论文标题：
- Efficient MoE Inference with Fine-Grained Scheduling of Disaggregated Expert Parallelism

---
