## Preemptive Expert Migration（抢占式专家迁移）

术语是什么？
Preemptive Expert Migration 是 Pre-gated MoE (ISCA '24) 提出的 CPU-GPU expert 参数迁移调度策略。利用 pre-gate function 提前知道下一个 MoE block 需要的激活 experts，在当前 block 的 GPU expert computation 执行期间，通过独立 CUDA stream 异步发起 cudaMemcpy 将下一个 block 所需的激活 experts 从 CPU DRAM 迁移到 GPU HBM。

从系统架构角度拆解术语：
执行流程（除第一个和最后一个 MoE block）：

```
// Block 0 (例外，无法重叠):
gate_0(x_0) → select A_0 → cudaMemcpy(A_0, CPU→GPU) → expert_exec(A_0)
pre_gate_0(x_0) → 确定 A_1

// Block 1..N-1 (核心优化):
Stream 1 (compute): expert_exec(A_N, x_N)     // ~2ms
Stream 2 (comm):   cudaMemcpy(A_{N+1}, CPU→GPU) // ~2ms, 与 compute 重叠
pre_gate_N(x_N) → 确定 A_{N+1}                 // ~0.05ms, 可忽略
```

重叠条件：expert_exec 时间 ≥ expert_migration 时间。Switch-Base 128 experts Top-1 激活场景下，expert execution ~2ms，expert migration (1 expert, ~85MB) ~2.7ms via PCIe 32GB/s，重叠部分有效。当激活 expert 数增多时 computing 时间增长慢而 communication 时间线性增长，重叠效果递减。100% expert 激活（=Dense 行为）时重叠效果完全消失。

术语一般如何实现？如何使用？
在 FasterTransformer 上实现（GitHub: https://github.com/ranggihwang/Pregated_MoE）。利用两个 CUDA stream：compute stream 执行 cuBLAS GEMM，communication stream 执行 cudaMemcpy。结果：Pre-gated MoE 达到 GPU-only（oracular 上界）的 81% 吞吐，峰值 GPU 内存仅为其 23%，Switch-Large (26.4B) 在 GPU-only OOM 时仍可运行。

涉及论文标题：
- Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference
