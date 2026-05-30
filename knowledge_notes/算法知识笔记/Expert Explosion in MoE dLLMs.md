## Expert Explosion in MoE dLLMs

术语解释
Expert Explosion 是 MoE dLLM 并行解码中的一种现象：随着并行 token 数（block size N）增加，unique activated experts 数量近乎线性增长，导致 HBM→SRAM weight fetching 成本主导延迟，使推理陷入 memory-bound 状态。

术语是什么？
在 MoE dLLM 中，每个 token 通过独立 routing 函数选择 Top-K experts。当 N 个 tokens 并行处理时，unique expert load = |∪_{n=1}^N S_n|，其中 S_n 是第 n 个 token 选择的高分 expert 集合。在均匀路由假设下，|∪| 期望值为 M·(1-(1-K/M)^N)，随 N 增长迅速接近 M。

延迟模型：L_MoE = b·|∪_{n=1}^N S_n| + a·(N·K)，其中 b 是 HBM→SRAM weight fetching cost（主导项），a 是 marginal compute cost。N 增大时 b 项呈线性/次线性增长，导致总延迟上升。

Roofline 分析：MoE dLLM 的 operational intensity 低于同容量的 dense 模型（因为 sparse activation 降低了计算密度），使其更 memory-bound。现代 GPU 的 FLOPs/byte ratio 增速超过 memory bandwidth 增速（Ma & Patterson, 2026），加剧此瓶颈。

从算法pipeline角度拆解术语：
```
# Expert Explosion 量化
M = 128  # total experts
K = 8    # top-k per token
for block_size N in [8, 16, 32, 64]:
    # 每 token 独立选择
    expert_sets = []
    for token n in range(N):
        S_n = TopK(router_gate(token_n), K)  # |S_n| = K
        expert_sets.append(S_n)
    
    unique_experts = len(set.union(*expert_sets))  # ≈ M*(1-(1-K/M)^N)
    # N=8:  unique≈47, N=16: unique≈72, N=32: unique≈98, N=64: unique≈118
    memory_traffic = unique_experts * expert_size_bytes  # dominates latency
```
实验结果（LLaDA2.0-Mini 16B, N=32）：vanilla 产生 ~84 unique experts/layer，expert weight footprint ~0.98 GB/layer。

术语一般如何实现？如何使用？
- 识别：通过 MoE kernel latency profiling（Nsight Systems）观测 HBM traffic 与 unique expert count 的线性关系
- 缓解方向：
  - Dynamic Expert Sharing (DES)：序列级 coreset selection 减少 |∪S_n|
  - Expert offloading/CXL-NDP：将冷 experts 卸载到外部存储
  - Expert quantization：降低每 expert 的 weight footprint
  - 与 AR batching 中 expert popularity skew 的问题不同，expert explosion 是 dLLM 并行解码独有的

涉及论文标题：
- Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs
