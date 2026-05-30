## Hot-swappable Training in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Hot-swappable Training 是 GatePro 的特性——localized competition 机制可在训练期间随时启用或禁用，无需额外参数、架构修改或学习率调整。与 auxiliary loss 不同（其启用/禁用改变 loss landscape），GatePro 的 hot-swap 在 gate logit 层面操作，不影响 loss 计算。论文验证了"训练遗产效应"（training legacy effect）：GatePro 训练阶段建立的 expert diversity 对后续标准 MoE 训练持续产生正面影响。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
use_gatepro = True  # hot-swap flag

for step, batch in enumerate(dataloader):
    for layer in model.layers:
        x = layer.moe_gate(x, use_gatepro=use_gatepro)
    ...

    if step == hotswap_step:  # e.g., at 400B tokens
        use_gatepro = False   # disable, continue as standard MoE
```

Hot-swap 实验（Table 3, 0.7B/14B, 256 experts, 500B tokens）：
- 100B GatePro → 400B MoE: MMLU-Pro 28.7%
- 400B GatePro → 100B MoE: MMLU-Pro 30.0%
- 500B GatePro (Full): MMLU-Pro 30.1%
越长的 GatePro 训练产生越好的最终性能，400B+100B 接近 Full GatePro。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实际价值：(1) 资源优化——计算密集的早期训练用 GatePro 建立 diversity，后期切换 standard 节省开销；(2) 灵活部署——不同阶段按需切换，无需重新初始化或修改 checkpoint；(3) 实验探索——研究者可测试不同时间窗口的 diversity 提升效果。与 auxiliary loss 相比，hot-swap 不影响 loss landscape，切换对训练稳定性无影响。

涉及论文标题：
- GatePro Parameter-Free Expert Selection Optimization for Mixture-of-Experts Models
