## Skip Loss Spikes & Sample Retry

术语解释
训练异常自动处理策略：检测到 loss spike 时跳过当前更新，将触发数据随机重注入后续 batch；持续 spike 则自动降学习率。

术语是什么？
MoE 训练中 loss spike 分 narrow（数步影响小）和 wide（多步可致 benchmark 随机水平）。策略：skip→save→re-inject→retry→降 lr 级联。

从算法pipeline角度拆解术语：
```
if is_spike(loss, loss_ema):
    skip_update(); save_data(batch)
    inject_to_future_randomly(batch)
    if retry_count > 0 and is_spike(loss, loss_ema):
        lr *= decay_factor
    retry_count += 1
else:
    backward(); step(); retry_count = 0
```

术语一般如何实现？如何使用？
- spike 检测基于 loss 偏离 EMA 的倍数阈值
- Ling 在 DLRover 中实现，配合自动 checkpoint recovery

涉及论文标题：
- Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs

---
