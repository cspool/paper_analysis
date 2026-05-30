## Replay in Continual Pre-training (CPT中的数据回放)

术语解释
Replay 是持续学习（Continual Learning）中缓解灾难性遗忘的经典技术：在训练新任务/分布时，将一定比例的旧数据混合到训练 batch 中。在 LLM CPT 语境下，Replay 由 Ibrahim et al. (2024) 验证为 CPT 中最重要的防遗忘技术之一。本文验证了 Replay 对 MoE CPT 同样有效。

术语是什么？
Replay 的具体实现：每个 training batch 中 X% 的 samples 从旧分布（pre-training data）采样，(100-X)% 从新分布（CPT data）采样。例如 "40% Replay" = 每 batch 中 410 samples 来自 FineWeb，614 samples 来自 German CC（batch_size=1024）。

**Compute-equivalent Replay**：为保证不同 replay 比例下的计算量可比，增加 replay 时不增加总 token 预算，而是减少新数据量。例如：200B German CPT at 0% replay = 200B German tokens；at 40% replay = 120B German + 80B FineWeb tokens（总计仍 200B）。

从算法pipeline角度拆解术语：
```python
# CPT with Replay (40% for German)
replay_pct = 0.4
for step in range(95370):
    # Replay portion
    n_replay = int(batch_size * replay_pct)      # 410
    batch_replay = sample(fineweb_loader, n_replay)

    # New data portion
    n_new = batch_size - n_replay                 # 614
    batch_new = sample(german_loader, n_new)

    # Mixed batch
    batch = concat([batch_replay, batch_new])
    # Shuffle batch if needed
    loss = model(batch)
    optimizer.step()
```

术语一般如何实现？如何使用？
- **Replay 比例调优**：本文测试 0%/10%/40%（German）和 30%（Stack）。更高的 replay → 更好的防遗忘（FineWeb val loss 更低），但牺牲部分 adaptation（German/Stack val loss 略高）
- **MoE 与 Dense 的 replay 效果相同**：MoE 的 replay 行为与 FLOP-matched dense 模型一致（本文 Figure 6）
- **Replay 对 MRI 的影响**：Replay 对 SBTk 的 MRI 几乎无影响（SBTk 本身已经固有鲁棒）；对 PBTk 仅轻微减小分布偏移时的 MRI spike（因为 PBTk 恢复很快，replay 的边际收益不大）
- **数据比例**：DeepSeek-CoderV2 使用较高的 replay 比例（30-40%），本文遵循此设定

涉及论文标题：
- Continual Pre-training of MoEs How robust is your router

---
