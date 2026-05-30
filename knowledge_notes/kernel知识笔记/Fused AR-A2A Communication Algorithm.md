## Fused AR-A2A Communication Algorithm

术语是什么？
Fused AR-A2A Communication Algorithm 是 MixServe 提出的将 All-Reduce (AR) 和 All-to-All (A2A) 两种 collective communication 算子融合并异步重叠执行的通信算法。核心思路是将 AR 分解为 Reduce-Scatter (RS) + All-Gather (AG) 两个子操作，将 A2A 分解为 Dispatch + Combine 两个子操作，然后重组为 RS→A2A→AG 三段式通信流程，利用异步机制使 intra-node 通信（RS/AG，高带宽）与 inter-node 通信（A2A Dispatch/Combine，低带宽）在时间上重叠，从而隐藏低带宽链路的延迟。该算法包含两个变体：(1) Fused RS-Combine（Alg 1），将 intra-node RS 与 inter-node A2A Combine 异步重叠；(2) Fused AG-Dispatch（Alg 2），将 intra-node AG 与 inter-node A2A Dispatch 异步重叠。

从kernel调度角度拆解术语：
以 4 节点、每节点 m 个 GPU/NPU、n_node=4 为例的 Fused RS-Combine 伪代码：

```
Require: n-node cluster, m GPUs/NPUs per node; input X [bs/d_EP, h] per node; global rank r
Ensure: output Y [b/d_DP, s, h] per node
1: Y = empty(b/d_DP, s, h)
2: [X_1,...,X_m] = split(X, m, dim=-1)  // 沿 hidden dim 切 m 份
3: r_TP = r mod m  // 计算 TP group 内 rank
4: S_1 = X_{r_TP}  // 暂存本地分片
5: for i = 1 to n-1 do async:  // inter-node A2A pairwise (异步)
6:     r_to = (r_TP + i*m) mod mn
7:     isend(X_{r_TP}, r_to)   // 发送到下一节点同 TP rank
8:     r_from = (r_TP - i*m) mod mn
9:     S_{i+1} = irecv(r_from)  // 从上一节点同 TP rank 接收
10: for i = 1 to n do async:  // intra-node RS + top-k 加权 (异步)
11:     S_i = await reduce_scatter(S_i, TP_group)
12:     Y_i = Y_i + topk_weights(S_i)  // 累加加权结果
13: Y = all_gather(Y_{r_TP}, TP_group)  // 最终 intra-node AG 汇总
```

Fused AG-Dispatch 同理：将 intra-node AG 与 inter-node Dispatch 重叠。首轮 pairwise 和末轮 AG 不可重叠，其余 n_node-2 轮通信完全重叠。时间复杂度 O(n_node)，RS-Combine 空间复杂度 O(bsh·n_proc)，AG-Dispatch 空间 O(1)。

术语一般如何实现？如何使用？
- MixServe 基于 vLLM（Ascend 910B）和 Tutel（H20）实现，通过向 MoE model 的 forward method 注入 RS/AG/A2A 通信算子。
- 异步机制：使用多个 CUDA/HCCL stream 并行执行 intra-node RS/AG 和 inter-node A2A isend/irecv，通过 await 同步点确保数据一致性。
- 性能收益：在 Ascend 910B 上，DeepSeek-R1 TTFT 加速 2.67× vs vLLM TP+PP，1.70× vs vLLM DP+EP；H20 上 Throughput +50.3% vs vLLM TP+PP。消融实验（Fig. 12）显示异步重叠的收益约等于 inter-node 通信开销。
- 适用场景：多节点 MoE 推理服务，intra-node 带宽显著高于 inter-node 带宽（如 NVLink 900 GB/s vs InfiniBand 50 GB/s，或 HCCS 60 GB/s vs RoCE 25 GB/s）。

涉及论文标题：
- MixServe: An Automatic Distributed Serving System for MoE Models with Hybrid Parallelism Based on Fused Communication Algorithm

---
