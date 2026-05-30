## SparseReduceScatter

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

SparseReduceScatter 是 Hecate/FSSDP 提出的稀疏通信原语，与 SparseAllGather 对称配对使用，用于将 MoE training backward pass 中 replicated expert 产生的 gradients reduce（求和）回持有对应 MoE shard 的 source device。形式上，spRS(𝒫₀, 𝒫₁) 从 pre-condition 𝒫₀（gradients 分布在多个 device 上）转换到 post-condition 𝒫₁（每个 chunk 的 reduce 结果唯一存在于一个 device，𝒫₁ surjective 且 𝒫₁ ⊆ 𝒫₀）。通信量上界 O(λS)，其中 λ 为需跨 device reduce 的 expert 比例。

在 NCCL 实现中，SparseReduceScatter = ncclGroupStart/End 包裹的一组 ncclReduce：对每个需 reduce 的 chunk c，在持有 c 的 replica 的所有 device 之间执行 reduce 操作，结果写入持有该 MoE shard 的 root device。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
SparseReduceScatter NCCL 实现 (spRS(P_0, P_1)):
-------------------
输入:
  P_0: gradient 分布 placement (包含 replica, 每个 expert 的梯度可能在多个 device)
  P_1: 目标 placement (每个 expert 的梯度 reduce 到唯一 source device)
  |P_0| = 总 expert gradient 副本数 (≥ |P_1|)

输出: P_1 中每个 source device 持有其 MoE shard 中 experts 的 reduced gradients

NCCL 执行:
  ncclGroupStart()
  for each chunk c (expert) that has replicas:
      if (c, d_src) in P_1:  // d_src 是该 expert 的 gradient 目标
          // Reduce: 所有持有 c 梯度副本的 device → d_src
          sub_comm = NCCL subgroup containing d_src and all devices with replica of c
          ncclReduce(chunk_c_grad_data, root=d_src, comm=sub_comm, op=SUM)
  ncclGroupEnd()

通信量分析:
  // 与 spAG 对称且等价
  vol(spRS) = O(λ·S) = vol(spAG)
  总 FSSDP 通信量 = vol(spAG) + vol(spRS) = O(2λS)

与 Rearrangement AllReduce 比较:
  // 同一 placement P' 下, rearrangement 系统需要 AllReduce 同步 DP group
  Vol(AllReduces) = Σ_i 2(|D_i|-1)/|D_i| · S/|C|
  当 |D_i| 大时 → O(2λS) = vol(spAG) + vol(spRS)
  // FSSDP 实现相同 placement 的通信量等价于 AllReduce
  // 但消除了 rearrangement 的 expert 参数+优化器状态迁移开销

Scheduling:
  spRS(layer l) + spAG(layer l+1) 同时重叠于 Attention backward
  (Attention backward 约 2× Forward → 足够隐藏两个 sparse collective)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Hecate 用 NCCL group calls 的 Reduce 实现 SparseReduceScatter。每个 spRS 调用与其对称的 spAG 配对——spAG(𝒫, 𝒫') 物化 placement，spRS(𝒫', 𝒫) 将梯度 reduce 回 source。
- 在 MoE layer backward 中，expert backward 计算完成后立即执行 spRS（可与 Attention backward 重叠，如上所述）。
- spRS 的通信量取决于 placement 的稀疏度 λ。Hecate 的 topology-aware sparse materialization (Algorithm 1) 在搜索 placement 时最小化跨 node replica，间接降低 spRS 的通信开销。

涉及论文标题：
- Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism
