## SparseAllGather

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

SparseAllGather 是 Hecate/FSSDP 提出的稀疏通信原语，用于在 MoE 训练中从 sharded expert parameters 按需物化 (materialize) 一个临时的 expert placement。形式上，SparseAllGather 操作在 logical input buffer（划分为等大小的 chunks C = {C_0, C_1, ...}，每个 chunk = 一个 expert 的参数）上，从 pre-condition placement 𝒫₀ 转换到 post-condition placement 𝒫₁（𝒫₀ ⊆ 𝒫₁，即物化目标是 shard 的超集）。𝒫₀ 为 surjective（每个 chunk 唯一归属于某 source device）。其通信量上界 O(λS)，其中 λ = |Ĉ|/|C| 为需跨 device 通信的 expert 比例（稀疏度），S 为总参数大小。当 λ << 1 时，远小于 FSDP AllGather 的 O(S)。

在 NCCL 实现中，SparseAllGather = ncclGroupStart/End 包裹的一组 ncclBroadcast：对每个 (expert, target_device) 需要物化的对，从持有该 expert 的 source device 向需要该 expert 的所有 target devices broadcast。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
SparseAllGather NCCL 实现 (spAG(P_0, P_1)):
-------------------
输入:
  P_0: 初始分片 placement (|D| 个 source device 各持有一部分 expert)
  P_1: 目标 placement (P_0 ⊆ P_1)
  expert 参数分布在各 device 上, 每个 expert 参数大小 = expert_size

输出: P_1 中每个 device 获得所需 expert 的参数副本

NCCL 执行:
  ncclGroupStart()
  for each chunk c (expert) that needs to be materialized:
      // 需要从 source device 发送到 ≥1 个 target device
      if (c, d_target) in P_1 and (c, d_target) not in P_0:
          d_src = unique source device from P_0 holding chunk c
          // Broadcast: 从 d_src 到所有需要 c 的 target devices
          sub_comm = NCCL subgroup containing d_src and all relevant targets
          ncclBroadcast(chunk_c_data, root=d_src, comm=sub_comm)
  ncclGroupEnd()

通信量分析:
  Ĉ = {c | c 至少需发送到一个新 device}
  λ = |Ĉ| / |C|  (稀疏度, 通常 λ << 1)
  vol(spAG) = O(λ·S)  // S = |C| × expert_size
                       // 最坏: bottleneck device 接收 λ·S 数据
  vs FSDP AllGather: O(S) vs FSSDP SparseAllGather: O(λS)
  当 λ << 1: O(λS) << O(S)

Scheduling 约束:
  t = T_attn_fwd × bw / expert_size
  // t = 可在 Attention forward 时间内隐藏通信的最大 expert 数
  // spAG 延迟 ≤ T_attn_fwd → 完全重叠, 零 critical path 开销
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Hecate 用 NCCL group calls 的 Broadcast 实现 SparseAllGather。每个 Broadcast 操作针对一个 expert chunk 到一个 sub-communicator（包含 source device 和所有 target devices）。
- 通信与 Attention computation 重叠（不在 critical path 上）：Forward 中 spAG 重叠于 Attention forward。Backward 中 spAG（下一层 re-materialize）重叠于 Attention backward。
- 更高效的实现可利用数据稀疏性和网络拓扑信息（如 TACCL、GC3 等 collective synthesizer），动态生成针对当前 placement 和拓扑优化的 sparse collective 算法（留作 future work）。
- FSSDP 的 Calibration stage 可选地追加一次 on-critical-path 的 spAG：MoE gate 输出后对比预测 load vs 实际 load，若追加物化的收益 > 通信开销则执行。

涉及论文标题：
- Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism
