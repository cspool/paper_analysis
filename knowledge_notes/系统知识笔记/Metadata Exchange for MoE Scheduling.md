## Metadata Exchange for MoE Scheduling

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Metadata Exchange 是 HarMoEny 在 all-to-all token dispatch 之前插入的轻量级全局信息同步步骤（Algorithm 1, Step 2）。每 GPU 将其本地 token-to-expert assignment（m_expert，整数 tensor）广播给所有其他 GPU，使所有 GPU 获得全局一致的 token 分布视图（m_all）。这使所有 GPU 能 deterministic 地独立并行计算相同的 rebalanced token-to-GPU schedule S，无需额外同步。

从系统架构角度拆解术语：

```
# HarMoEny Algorithm 1, Steps 2-3
Step 2: SENDMETADATATOGPUs(m_expert)
  # m_expert[i]: [num_local_tokens] int tensor, values ∈ [0, num_experts)
  # Total metadata ≈ |G| × num_local_tokens × sizeof(int)
  # On 8×V100, ~512 tokens/GPU → ~4KB total
  # Communication: torch.distributed.all_gather → m_all

Step 3: S_initial = INITIALASSIGN(m_all)
  # S_initial[g_from, e, g_to]: token count tensor
  S = REBALANCE(S_initial)
  # All GPUs compute S independently — deterministic because m_all is identical
```

关键性质：
- **Size**: ~4KB per batch， vs all-to-all token communication（MBs of hidden state embeddings）
- **时效性**: Per-batch execution，替代 profiling-based 方案的分钟级优化
- **Enables determinism**: m_all 全局一致保证 REBALANCE 的 greedy algorithm 在相同输入产生相同输出

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

使用 PyTorch 的 `torch.distributed.all_gather` 同步所有 GPU 的 m_expert。前提：expert placement 在所有 GPU 已知一致（round-robin initialization），各 GPU 可从 m_all 直接推导 S_initial。HarMoEny 开源实现中集成在自定义 MoE Layer 的 forward pass 内。

涉及论文标题：
- HarMoEny: Efficient Multi-GPU Inference of MoE Models
