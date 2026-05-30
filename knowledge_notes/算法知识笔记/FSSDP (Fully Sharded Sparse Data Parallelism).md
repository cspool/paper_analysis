## FSSDP (Fully Sharded Sparse Data Parallelism)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FSSDP (Fully Sharded Sparse Data Parallelism) 是 Hecate 系统提出的 MoE 训练新范式，受 FSDP (Fully Sharded Data Parallelism) 启发但针对 MoE layer 的稀疏性重新设计。FSSDP 分为两个阶段：
(1) **Sharding Phase**：将每个 MoE layer 的 parameters 和 optimizer states 划分为 |𝒟| 个不相交的 MoE shards，每个 shard 包含一组 expert 的完整参数和优化器状态，唯一分配给一个 device。全局仅保留一份 optimizer states 副本，实现最小且均衡的内存占用。
(2) **Materialization Phase**：每次 iteration，用 SparseAllGather 从 shards 稀疏物化 (sparsely materialize) 一个临时的 expert placement ——即"从零构建"而非"从上一个 placement 迁移"。Forward 后释放物化参数（可选 re-materialization 在 backward 重新物化），backward 后用 SparseReduceScatter 将 replicated expert 的 gradients reduce 回持有对应 MoE shard 的 device。每次 iteration 都能工作在针对当前 expert load 分布最优的 placement 下，无需在 iteration 间迁移 expert 状态，因此不存在 rearrangement 系统的 memory/timeliness trade-off。

FSSDP 与 rearrangement 系统的关键区别：对于同一 expert placement 𝒫'，FSSDP 的 spAG(𝒫, 𝒫') + spRS(𝒫', 𝒫) 的通信量上界 O(2λS) 与 rearrangement 系统为同步 replicated expert gradients 的 AllReduce 总通信量完全相同（λ 为需跨 device 通信的 expert 比例），但 FSSDP 消除了 rearrangement 系统在 iteration 间 expert 参数+优化器状态迁移的额外通信开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FSSDP 在单 MoE layer l 一个 iteration 中的完整执行流程：

```
=== SHARDING PHASE (每 100 iterations 左右低频执行) ===
// 输入所有 MoE layers 的 expert load 分布 F^g
// Heterogeneous Sharding (Algorithm 2):
1. J ← 各层 top-t overloaded experts (可被 sparse materialization 选中)
2. J' ← E^g - J  // underloaded experts
3. 每 device 分配 |E^g|/|D| 个 slots
4. 先放置 J' (underloaded experts): layer by layer,
   优先最 overloaded 的 layer, 每 expert 选 least-loaded node/device
5. 再填充 J (overloaded experts): 任意分配到剩余 slots
// 输出: P^g = {P_0, ..., P_L} 各层 sharding plan

=== MATERIALIZATION PHASE (每 iteration 执行) ===
// Sparse Materialization (Algorithm 1):
Input: P (当前 sharding plan), F (预测的 expert load),
       t (overlap degree), m (memory capacity per device)
t = T_attn_fwd * bw / expert_size  // 可在 attention 时间内
                                   // 隐藏通信的最大 expert 数
m = 每 device 可额外容纳的 expert 参数数

if t <= m:
    // 物化 top-t overloaded expert 到所有 device
    P' ← P ∪ (D × E^topT)
else:
    totSlots ← |D| * m
    for e in sorted overloaded experts (by load descending):
        n ← 按负载比例分配 replica slots
        P^e ← 分配 n 个 replica 到 nodes/devices (优先有空闲 slots 的 node)
        P' ← P' ∪ P^e

=== FORWARD PASS ===
// Communication-Overlap:
[Attention Forward] ← spAG(P, P') 与此重叠
[MoE Gate] → Calibration (可选): 用实际 token assignment
    重新运行 Algorithm 1, 若收益>通信开销则追加额外的 spAG
[Token Dispatch: Topology-aware All-to-All]
    - 同 node 内有 expert replica → 优先 intra-node dispatch
    - 无同 node replica → 跨 node, 均匀分配到 replica devices
[Expert FFN Computation on materialized parameters]
[Release materialized parameters (for re-materialization mode)]

=== BACKWARD PASS ===
[Attention Backward] ← spRS(P',P) (layer l 的梯度 reduce)
                    ← spAG(P, P') (layer l+1 的 re-materialize)
                    两者与此重叠 (backward 约 2× forward 时间)
[Expert Backward Computation on re-materialized parameters]
[spRS(P',P) for this layer (若未在 attention backward 中完成)]

=== OPTIMIZER STEP ===
各 device 在其 MoE shards 上用 reduced gradients 更新 optimizer states
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FSSDP 的实现需要以下组件：
- **稀疏通信原语**：SparseAllGather 和 SparseReduceScatter。在 Hecate 的 prototype 中，用 NCCL group calls 实现（spAG = 一组 Broadcast, spRS = 一组 Reduce），每组包含多个同步调度的 point-to-point 或 collective 操作。更高效的实现可利用数据稀疏性和网络拓扑（留作 future work）。
- **Scheduler**：基于滑动窗口 (w=5) 预测 expert load，在 overlap degree t 和 memory capacity m 约束下搜索近似最优 placement。
- **Dispatcher**：拓扑感知的 token 路由，优先 intra-node 通信。
- **Communicator**：管理稀疏 collective 和 All-to-All 的通信队列。
- FSSDP 适用于大规模 MoE 训练场景（64+ experts × 32+ GPUs），expert load imbalance 越严重收益越大（低带宽 inter-node 环境下加速比更显著）。

涉及论文标题：
- Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism
