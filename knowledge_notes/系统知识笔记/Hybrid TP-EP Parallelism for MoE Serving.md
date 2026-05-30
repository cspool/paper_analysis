## Hybrid TP-EP Parallelism for MoE Serving

术语是什么？
Hybrid TP-EP Parallelism 是 MixServe 提出的混合并行策略：将 MoE 模型的 Attention block 按 intra-node Tensor Parallelism (TP) + inter-node Data Parallelism (DP) 切分，MoE block 按 intra-node TP + inter-node Expert Parallelism (EP) 切分。关键创新在于解耦 AR（All-Reduce）为 RS（Reduce-Scatter）+ AG（All-Gather），并将 EP 的 A2A（All-to-All）重组为 RS→A2A→AG 三段式通信流程。TP group 严格限制在 intra-node（利用 NVLink/HCCS 高带宽），EP group 限制在 inter-node（利用 InfiniBand/RoCE），使并行度与硬件带宽层次精确对齐。与纯 EP 策略相比，hybrid TP-EP 降低了 A2A 通信的每轮通信量和通信规模。

从系统架构角度拆解术语：
以 2 节点、每节点 4 NPU、TP=4 + DP=2、TP=4 + EP=2 为例的推理流程：

```
Offline: Automatic Analyzer 输入模型超参数 + 集群配置，输出最优 (d_TP, d_EP, d_DP)
Online Weight Loading:
  - Attention: 按 [TP=4, DP=2] 切分 → intra-node 4-way TP, inter-node 2-way DP
  - MoE: 按 [TP=4, EP=2] 切分 → intra-node 4-way TP, inter-node 2-way EP

Per-Layer Forward (MoE block):
  1. Hidden states 在 TP group 内分片
  2. Intra-node RS: 各 NPU reduce-scatter 交换 partial hidden states
  3. Inter-node A2A: Pairwise 交换 expert tokens
     步骤2和3异步重叠 (Fused RS-Combine)
  4. Expert FFN: 各 expert 独立前向计算
  5. Intra-node AG + Inter-node Dispatch: 结果汇总并路由回原节点
     步骤5的两个子操作异步重叠 (Fused AG-Dispatch)
```

DP 和 EP degree 的三种关系：
- d_DP = d_EP：最平衡，DP rank 与 EP rank 一一对应
- d_DP > d_EP：expert weights 冗余复制，增加内存换更高吞吐
- d_DP < d_EP：hidden states 冗余，但通过 effective dropping 降低通信开销

术语一般如何实现？如何使用？
- MixServe 基于 vLLM（Ascend 910B）和 Tutel（H20）实现。
- Automatic Analyzer 在 offline 阶段枚举所有满足 n_proc × n_node = d_TP × d_EP 的 (d_TP, d_EP, d_DP) 组合。
- 消融实验显示最优 DP-EP 配置因硬件平台而异：Ascend 910B 上 d_DP = d_EP 最优，H20 上 d_DP < d_EP 最优。
- 对比纯 EP：hybrid TP-EP 将 inter-node A2A 通信量从 2×A2A(bshk, n_node·n_proc) 降至 2×A2A(bshk/n_proc, n_node)。

涉及论文标题：
- MixServe: An Automatic Distributed Serving System for MoE Models with Hybrid Parallelism Based on Fused Communication Algorithm

---
