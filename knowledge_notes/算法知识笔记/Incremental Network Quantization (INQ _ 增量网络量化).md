## Incremental Network Quantization (INQ / 增量网络量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Incremental Network Quantization (INQ) 是由 Zhou et al. (ICLR 2017) 提出的 training-aware 量化方法，将全精度预训练 CNN 逐步转换为低精度（幂次或整数）模型。核心思想：不一次性量化所有权重，而是将权重分成两组——已量化组（冻结）和未量化组（可微调）。在每一步中，逐步扩大已量化比例（如 25% → 50% → 75% → 87.5% → 100%），未量化权重通过 fine-tuning 补偿前面步骤的量化误差。

INQ 的三种权重划分策略：(1) 随机划分；(2) 按 |weight| 最大权重优先量化（剪枝思路，大权重对输出影响最大，先量化再微调小权重补偿）；(3) 按量化误差最小优先。

在 FQ 论文中，INQ 被用作 FQ 量化的 fine-tuning schedule（25%→50%→75%→87.5%→100%，每步 fine-tune 3 epochs，最后一步 10 epochs，LR=0.001 每 3 epochs 衰减），使模型逐步适应 FQ 的量化误差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
INQ 在 FQ pipeline 中的执行流程：

```
# Given: pre-trained & pruned model with FQ hyperparameters
# Schedule: partitions = [0.25, 0.50, 0.75, 0.875, 1.0]

for partition in partitions:
    # Step 1: Select weights to quantize (largest |weight| first)
    weights_to_quantize = top_k(|W|, partition * |W|)  # 按权重绝对值排序

    # Step 2: Quantize selected weights (frozen)
    for each w in weights_to_quantize:
        w_hat[w] = FQ_quantize(w)  # Q[θ] or Q^{shift}(θ)
        w.freeze()                  # 不参与后续梯度更新

    # Step 3: Fine-tune remaining (unquantized) weights
    for epoch in range(epochs_per_step):
        # Forward: use w_hat for quantized, w for unquantized
        # Backward: update only unquantized weights (STE through quantized)
        train_one_epoch()

    # Step 4: Update FQ hyperparameters (GMM fit) every k epochs
    if step % k == 0:
        update_GMM_hyperparams()

# Final step (partition=1.0): all weights quantized, final fine-tune
```

**Annotations**: `top_k` 选择权重绝对值最大的比例为 `partition`；freeze 通过 mask 实现，反向传播时梯度不更新已量化权重；GMM 更新间隔 k 指数增长以减少采样方差。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
INQ 不需要特殊的框架支持，可在标准 PyTorch/TensorFlow 训练循环中实现。核心是维护一个 binary mask 区分已量化（冻结）和未量化（可训练）的权重。INQ 同样适用于 shift quantization 以外的量化方案（整数、三元、二值等），只要量化和微调可以交替进行。FQ 论文中使用 INQ 的随机划分策略（按权重绝对值最大优先量化），Mayo 框架 (https://github.com/deep-fry/mayo) 包含 INQ 实现。

涉及论文标题：
- Focused Quantization for Sparse CNNs
