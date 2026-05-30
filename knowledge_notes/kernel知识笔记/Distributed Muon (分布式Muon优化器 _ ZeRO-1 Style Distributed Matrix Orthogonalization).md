## Distributed Muon (分布式Muon优化器 / ZeRO-1 Style Distributed Matrix Orthogonalization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Distributed Muon 是 Liu et al. (2025) 提出的基于 ZeRO-1 范式的分布式 Muon 优化器实现。核心挑战：标准 ZeRO-1 对 AdamW 高效是因为 AdamW 计算更新是逐元素的（element-wise），各 DP rank 可独立在本地分片上计算。但 Muon 需要全梯度矩阵才能执行 Newton-Schulz 正交化——若直接按 ZeRO-1 分片，每个 rank 只有 1/DP 的梯度矩阵，无法完成正交化。解决方案：在 DP 组内引入 bf16 DP Gather 操作，将分片梯度恢复为全矩阵执行 Newton-Schulz，计算完成后仅保留本地分片的更新结果。额外通信开销为 Distributed AdamW 的 1~1.25 倍，在实践的多 DP 组场景下接近下限（约 1.0x）。内存方面，Muon 仅需 1 个动量 buffer（vs AdamW 的 2 个），内存消耗减半。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Distributed Muon 的通信-计算流水线（Algorithm 1）：

```
Algorithm: Distributed Muon (per optimizer step, per DP rank)

Input: Full gradients G, DP-partitioned momentum m, DP-partitioned params p, momentum μ

# Phase 1: Gradient synchronization (标准 ZeRO-1)
1. g = reduce_scatter(G, dp_group)          # fp32, 通信量 = 4×|G| (每个 rank 得 1/DP)

# Phase 2: Local momentum update
2. g' = update_with_momentum(g, m, μ)        # 本地计算: g' = μ*m + g

# Phase 3: DP Gather — Muon 特有的额外操作
3. G_full = gather(g', dp_group)             # bf16 DP Gather, 通信量 = 2×|G| (bf16 vs fp32)
                                              # 注意: 仅在 DP 组内 gather，非全局 gather

# Phase 4: Newton-Schulz on full matrix
4. U = Newton-Schulz(G_full)                 # bf16, N=5 步, 本地计算
                                              # G_full ∈ R^{A×B}, U ≈ (G_full G_full^T)^{-1/2} G_full

# Phase 5: Discard non-local partitions
5. u = U[local_partition]                    # 仅保留对应本 rank 参数分片的更新

# Phase 6: Apply update locally
6. p' = apply_update(p, u)                   # p' = p - lr*(0.2*u*sqrt(max(A,B)) + λ*p)

# Phase 7: All-gather updated params
7. P = all_gather(p', dp_group)              # fp32, 通信量 = 4×|P|

# Phase 8: Return update RMS for logging
8. return sqrt(u².mean())
```

通信分析：
- Distributed AdamW 通信量：4 (fp32 reduce-scatter G) + 4 (fp32 all-gather P) = 8 单位
- Distributed Muon 通信量：4 (fp32 reduce-scatter G) + 2 (bf16 DP gather) + 4 (fp32 all-gather P) = 10 单位
- 比率：10/8 = 1.25x (上界)。若有 TP 启用，需额外 bf16 TP gather
- 多 DP 组下 DP gather 通信进一步分摊，实际接近 1.0x
- Newton-Schulz 在 bf16 下计算使 DP gather 通信量减半至 fp32 的 50%

端到端延迟：优化器延迟（含 Newton-Schulz 5 步迭代 + DP gather）通常为模型 forward-backward 时间的 1%~3%，可忽略。可通过 overlap gather 与 Newton-Schulz 计算、overlap reduce-scatter 与参数 gather 等工程优化进一步降低。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：
- 框架集成：基于 Megatron-LM 的 ZeRO-1 分布式优化器框架实现，充分利用其 TP/PP/EP/DP 并行策略
- 开源状态：Moonshot AI 承诺将 Distributed Muon 以 PR 形式贡献给 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)；社区已有 CPU 友好复现 [bird-of-paradise/muon-distributed](https://huggingface.co/datasets/bird-of-paradise/muon-distributed)
- Megatron-LM 集成细节：
  - 使用 `dist_group` (DP) 和 `tp_group` (TP) process group handles
  - 通过 `param_groups (buffer_idx)` 和 communication `buckets (bucket_idx)` 组织参数
  - `dist_meta` 和 `global_buffer_size` 管理 ZeRO-1 分片的 "虚拟缓冲区"
  - DP all_gather 需要 bucketing 以摊销延迟；TP all_gather (on-node, NVLink 高速) 不需要 bucketing
- 关键工程注意事项：
  - DP gather 仅需在 DP 组内操作（非全局），因每个 rank 只在 DP 组内分片
  - Newton-Schulz 输入需保持 2D shape（从 flat buffer unpack 后恢复原始矩阵维度）
  - NCCL backend 用于 GPU 间通信；CPU 复现可用 gloo backend
  - 需处理通信与计算的 overlap：DP gather 可与后续计算流水线化
- 内存优势：Muon 仅需 1 个 fp32 momentum buffer per parameter（vs AdamW 的 m + v 两个），在大型 MoE 模型中节省可观内存

涉及论文标题：
- Muon is Scalable for LLM Training
