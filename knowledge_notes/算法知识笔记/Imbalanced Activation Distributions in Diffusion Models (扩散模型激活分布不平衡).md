## Imbalanced Activation Distributions in Diffusion Models (扩散模型激活分布不平衡)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
扩散模型激活分布不平衡是指：在全精度扩散模型的部分层中，激活值绝大多数集中在零附近（如 [-0.6, 1.7]），但同时存在稀疏但数值较大的离群值（如总范围 [-10, 34]）。这些稀疏大值对生成质量至关重要——替换最大值 token 为零会导致图像严重退化，而替换随机 token 几乎无影响。这种分布对低比特量化构成双重挑战：(1) 大值的量化：若缩小裁剪范围以适应小值，大值会被严重裁剪（clipping error 大）；(2) 小值的量化：若扩大范围覆盖大值，大量小值的量化精度急剧下降（rounding error 大）。在 4-bit（仅 16 个量化级）下，此矛盾尤为尖锐——PTQ 方法无法找到合适的 trade-off，导致生成失败。该现象在 LDM 和 Stable Diffusion 中普遍存在（附录 Fig. 5 展示了多个模型和数据集上的不平衡分布）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
分析激活分布不平衡的流程：
```
# 检测激活分布不平衡
for layer in model.layers:
    for t in sample_time_steps():
        x_T = randn()
        a = model.forward(x_T, t, stop_at=layer)
        hist = histogram(a, bins=100)
        # 分析分布特性
        range_total = max(a) - min(a)           # 总范围（如 44）
        pct_99_range = percentile(a, 99) - percentile(a, 1)  # 99%分位范围
        imbalance_ratio = range_total / (pct_99_range + eps)
        # QuEST 发现：imbalance_ratio 在某些层 >> 10
        # 大值稀疏但重要 → 低比特量化困难
```
QuEST 的解决方案不是直接修改激活分布（不可操作），而是通过微调权重间接调整：微调后激活范围从 [-10, 34] 缩小到 [-4, 14]，标准差从 0.171 降至 0.157，分布更紧凑但均值保持一致——既减少了大值的稀疏性又保护了小值的量化精度。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
该发现的实际意义：(1) 它是 QuEST 方法的核心 motivation——解释了为何需要微调而非仅调量化参数；(2) 通用性——在条件 LDM-4（ImageNet）、无条件 LDM-4（LSUN-Bedrooms）和 Stable Diffusion 中均观测到，说明是扩散模型的固有特性而非特定模型的偶然现象；(3) 与 LLM 中的激活异常值（outlier）问题有相似性——如 LLM.int8() 和 SmoothQuant 也处理类似的分布不平衡——但扩散模型的不平衡更温和（数值范围小），且与批次和时序动态耦合；(4) 可用作诊断工具——在量化任何扩散模型前先检查各层激活分布，识别潜在的问题层。

涉及论文标题：
- QuEST Low-bit Diffusion Model Quantization via Efficient Selective Finetuning
