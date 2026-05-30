## Sequence Migration (序列迁移)

术语解释
Sequence Migration 是 LUFFY 分布式 MoE 训练系统中提出的通信优化技术：在 MoE 的 combine phase 中，不再将 token 全部拉回原 GPU 重构序列，而是将整个序列迁移到其大部分 token 被 expert 处理所在的目标 GPU 上重构，从而将跨 GPU 的 token 拉取路径隐藏为 intra-GPU 路径。

术语是什么？
在标准 Expert Parallelism 的 MoE 训练中，每个 GPU 处理本地序列的 attention，然后将 token 通过 all-to-all dispatch 发送到持有对应 expert 的 GPU，expert 计算后再通过 all-to-all combine 将所有 token 拉回原 GPU 重构序列。Sequence Migration 的核心观察是：由于 biased expert activation（每个序列通常只激活少数几个 expert），序列的大部分 token 往往集中在少数 GPU 上被处理。与其将所有 token 拉回原 GPU，不如将序列迁移到 token 最集中的 GPU 上重构。

从系统架构角度拆解术语：
Sequence Migration 在 LUFFY 系统架构中的执行流程：

```
=== Sequence Migration 在两轮 MoE Block 之间的调度 ===

Block b:
  GPU 0: Attention(seq_0, seq_1) → tokens → Dispatch → Expert 0,1
  GPU 1: Attention(seq_2, seq_3) → tokens → Dispatch → Expert 2,3
  GPU 2: Attention(seq_4, seq_5) → tokens → Dispatch → Expert 0,1
  GPU 3: Attention(seq_6, seq_7) → tokens → Dispatch → Expert 2,3

  All-to-All Dispatch: tokens routed to target experts
  Expert FFN Computation (各 GPU 并行)

Combine Phase 决策:

Step 1 - Controller 收集分布信息:
    token_to_gpu[t] = GPU where token t was processed
    token_to_sequence[t] = which sequence token t belongs to
    
Step 2 - 对每个 sequence i，估算迁移到各 GPU 的 combine 流量:
    for GPU j in all_GPUs:
        # 计算 sequence i 中在 GPU j 以外被处理的 token 数
        f_{i,j} = count({t in seq_i | token_to_gpu[t] != j})
    
Step 3 - 选择 top-q 候选 GPU (最小 combine 流量):
    H^i = {j_1, j_2, ..., j_q} where f_{i,j} is minimized

Step 4 - Attention Cost Model 评估候选:
    for GPU j in H^i:
        B_{j←i} = current_batch[j] ∪ {seq_i}
        L_{j←i} = max(len(s) for s in B_{j←i})
        # Cost model: T_att(B, L) = (3BLd² + 2BL²d) / P
        ΔT = T_att(B_{j←i}, L_{j←i}) - T_att(B_j, L_j)
    
    j* = argmin(ΔT)  # 选择 attention 成本增长最小的 GPU

Step 5 - 迁移执行:
    sequence_to_gpu[i] = j*  # 更新哈希表
    # Combine 阶段按新映射路由 token

Block b+1:
  GPU 0: Attention(seq_0, seq_3, seq_5) ← 新增 seq_3, seq_5
  GPU 1: Attention(seq_2, seq_7)        ← 新增 seq_7
  ...
  # 相似长度的 sequences 聚集 → padding zeros 减少
```

术语一般如何实现？如何使用？
- 需要一个集中式 Controller 节点收集分布信息并执行迁移算法
- 三张哈希表管理路由状态：token_to_sequence、token_to_gpu、sequence_to_gpu
- 通过 `torch.distributed.rpc` API 指导 GPU 间的 token 交换
- Cost model 参数 P（GPU 速度）通过 profiling attention 层多次获得，平均估计误差 ~5%
- 候选数 q 的权衡：大 q→更多 flexibility 用于 attention batch 优化；小 q→更侧重最小化 combine 流量
- 与 Expert Transfer 的关键区别：不移动 expert 参数（保持最大 expert parallelism），而是改变序列的物理位置
- 适用于 MoE 训练场景，因 gate routing 使每个序列的 token 分布有偏向性

涉及论文标题：
- Communication-Efficient Sparsely-Activated Model Training via Sequence Migration and Token Condensation

---
