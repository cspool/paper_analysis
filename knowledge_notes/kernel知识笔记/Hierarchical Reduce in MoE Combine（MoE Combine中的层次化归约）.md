## Hierarchical Reduce in MoE Combine（MoE Combine中的层次化归约）

术语是什么？
Hierarchical reduce 是 GPU-initiated token-level EP 通信中 combine 阶段的优化：将 expert outputs 的 weighted sum 归约分解为两层——(1) intra-node reduce：在节点内对同一 token 的多份 expert output 先做本地加权归约；(2) inter-node reduce：归约后的结果通过 RDMA 发回原 token GPU 做最终归约。相比所有 expert GPU 各自独立发送 output，大幅减少跨节点网络传输量。

从kernel调度角度拆解术语：
```
// 无 hierarchical reduce:
//   每个选中的 expert GPU 独立 RDMA output 回原 GPU
//   原 GPU 收到 top_k 份 outputs 后做 weighted sum

// 有 hierarchical reduce (DeepEP/UCCL-EP HT mode):
//   Phase 1 (intra-node):
//     node_i 上所有 local expert outputs:
//       同 token 的 outputs 先做 local weighted sum
//       各 node 输出 1 份 intra-node reduced result
//   Phase 2 (inter-node via RDMA):
//     各 node intra-node result → RDMA → 原 token GPU
//   Phase 3 (final reduce on GPU):
//     原 GPU kernel 对 M 份 inter-node results 做加权 sum
//     M = distinct nodes among selected experts (M << top_k)
//   网络传输量: M 份 (vs top_k 份无 hierarchical reduce)
```

术语一般如何实现？如何使用？
在 UCCL-EP HT mode 中，GPU kernel 在 combine 阶段利用 routing metadata 判断哪些 expert outputs 分布在同一 node 内，先在 NVLink/xGMI domain 内完成 intra-node reduce，仅将结果通过 CPU proxy 发起的 RDMA 跨节点传输。此优化依赖 GPU-initiated fine-grained 通信能力：GPU kernel 需在 transfer 前读取并处理 routing 信息。

涉及论文标题：
- UCCL-EP Portable Expert-Parallel Communication
