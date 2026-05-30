## All-to-All Communication (Pairwise Algorithm)

术语是什么？
Pairwise All-to-All 是 All-to-All collective communication 的一种经典实现算法。在 MoE 的 Expert Parallelism 中，A2A 用于将 tokens 从各 GPU dispatch 到拥有对应 expert 的 GPU（Dispatch），并在 expert FFN 计算后将结果 collect 回原 GPU（Combine）。Pairwise 算法需要 N-1 轮通信（N 为参与设备数），每轮中每个 rank 向特定目标 rank 发送数据并从特定源 rank 接收数据，形成成对交换。MixServe 利用 Pairwise 算法的多轮特性，在每轮中将 intra-node RS/AG 与 inter-node send/recv 重叠执行。

从kernel调度角度拆解术语：
以 N=4 设备为例的 Pairwise A2A 通信流程：
```
Round 1: rank0↔rank1, rank2↔rank3  (step=1)
Round 2: rank0↔rank2, rank1↔rank3  (step=2)
Round 3: rank0↔rank3, rank1↔rank2  (step=3)
```
每轮通信量 O(bs/d · hk)（b=batch, s=seq_len, h=hidden_dim, k=top-k, d=degree）。总通信量 ∝ (size/degree) × (degree-1)。

与 Ring All-to-All 对比：
- Ring：数据沿环形链路传递，N-1 步，每步传输 size/N 数据，总传输量 = size × (N-1)/N
- Pairwise：每步直接 send/recv，N-1 步，总传输量 = size × (N-1)/N，但每步的通信对可以并行（利用 full-duplex 链路）

术语一般如何实现？如何使用？
- NCCL/HCCL 实现：NCCL 的 All-to-All 通过 P2P send/recv 组合实现（因无原生 A2A 原语），支持 Pairwise 和 Ring 两种算法。
- MixServe 中的使用：Pairwise 算法的多轮特性使每轮中的 intra-node RS/AG 可与本轮 inter-node send/recv 重叠。Fused AR-A2A 算法要求使用 Pairwise（而非 Ring）以确保每轮的 send/recv targets 可预测。
- 关键参数：d（参与设备数）越大则轮数越多、通信开销越大。MixServe 通过 hybrid TP-EP 将 d_EP 降至 n_node（而非纯 EP 的 n_node × n_proc），减少轮数。

涉及论文标题：
- MixServe: An Automatic Distributed Serving System for MoE Models with Hybrid Parallelism Based on Fused Communication Algorithm

---
