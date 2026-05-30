## Shuffled-Reduce-Scatter (SRS / 混洗规约分散)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Shuffled-Reduce-Scatter (SRS) 是 Sem-MoE 在 Attention-TP 场景中实现的融合通信原语，将 speculative token shuffling（基于预测的 token 重排）嵌入标准 reduce-scatter 集合通信操作中。传统 Attention-TP 流程为：attention → allreduce → gate → all-to-all dispatch。SRS 将后三个操作融合：在 reduce-scatter 阶段按预测的 expert device assignment 对 token 进行重排，使每个 device 直接获得应路由给它的 token 子集，消除后续 all-to-all dispatch。Shuffling 逻辑嵌入 ring-based communication schedule，额外 overhead 仅约 1%。配套的 argsort kernel（Triton 实现）比 PyTorch 原生快 25%。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# SRS Kernel: Fused Shuffle + Reduce-Scatter
Input: X ∈ R^{B×H}          # post-attention hidden states
       T ∈ R^{t×E}           # token-to-device schedule table
       A ∈ R^{E²×E}          # 2-gram inter-layer device transition table
       C_p, A_p              # confidence scores

Step 1: Predict target device per token (两表竞争)
  for each token_id j in batch:
    if C_p[j] > A_p[prev_seq]:
      dev_ids[j] = T[j]          # token-level prediction
    else:
      dev_ids[j] = A[(d_prev1, d_prev2)]  # inter-layer prediction

Step 2: Compute shuffle indices (Triton argsort, 25% faster than PyTorch)
  shuffle_indices = argsort(dev_ids)

Step 3: Group, align, concatenate
  shuffle_indices = concat(align(group_by_key(shuffle_indices)))

Step 4: Shuffle + Ring-based Reduce-Scatter
  X_shuffled = X[shuffle_indices]
  X_local = reduce_scatter(X_shuffled)  # integrated shuffle overhead ≈ 1%

Output: X_local_i per device  # ready for local gate + expert FFN
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Triton 实现，调度表驻留 GPU memory（<12 MB for DeepSeek-V2），O(1) 查表。两表竞争机制确保鲁棒性：token-level 和 inter-layer 预测均低置信度时 fallback 到标准 all-to-all。结合 DeepEP 通信后端。

涉及论文标题：
- Speculative MoE: Communication Efficient Parallel MoE Inference with Speculative Token and Expert Pre-scheduling
