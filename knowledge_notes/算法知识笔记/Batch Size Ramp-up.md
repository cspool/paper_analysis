## Batch Size Ramp-up

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Batch Size Ramp-up 是一种训练策略，在训练初期使用较小的 batch size，随着训练进行逐步增大到目标值。动机来自 McCandlish et al. (2018) 的发现：过大的 batch size 会损害优化过程（尤其训练初期），存在临界 batch size B_crit 随训练 loss 降低而增大。Joint MoE Scaling Laws 采用直接 grid search 选择过渡点，而非使用 noise scale 做 B_crit 预测。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

论文使用的具体 ramp-up 策略：

```
# Phase 1: batch_size=64K tokens,  training tokens 0→0.5B
# Phase 2: batch_size=128K tokens, training tokens 0.5B→1.0B
# Phase 3: batch_size=256K tokens, training tokens 1.0B→end

# 实现
if tokens_trained < 0.5B:
    batch_size = 64K
elif tokens_trained < 1.0B:
    batch_size = 128K
else:
    batch_size = 256K
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 被当代 LLM 训练广泛采用：Gopher, LLaMA 3 等
- 实现方式：修改 DataLoader 的 batch size 参数，在指定 token 计数后触发 ramp
- 与 LR schedule 配合：batch size 增大时 linear scaling rule (lr ∝ batch_size) 建议同步调高 LR，但 ramp-up 通常不调 LR（已在 peak）
- 价值：早期小 batch 帮助模型快速学习基本模式，后期大 batch 最大化硬件利用率

涉及论文标题：
- Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient
