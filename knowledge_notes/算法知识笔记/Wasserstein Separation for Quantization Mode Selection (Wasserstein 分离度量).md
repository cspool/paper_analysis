## Wasserstein Separation for Quantization Mode Selection (Wasserstein 分离度量)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Wasserstein Separation 是 FQ 中用于自适应选择量化模式的决策度量。当剪枝后某层的权重分布没有明显的多峰结构（即 GMM 的两个分量高度重叠），recentralized quantization 与普通 shift quantization 效果几乎等价——因为两个分量差异过小，具体选择哪个分量来量化某个权重对结果影响极小。此时浪费 1 bit 来编码 component selection 反而不如将该 bit 用于增加 shift quantization 的精度。

FQ 使用 2-Wasserstein 距离（归一化后）来衡量两个高斯分量的分离程度：
`W(c₁,c₂) = ((μ_c₁ - μ_c₂)² + (σ_c₁ - σ_c₂)²) / σ²_global`
其中 σ²_global 是整层权重的方差。当 `W < w_sep`（默认 2.0），说明分量重叠严重，退化为 n-bit shift quantization（等效精度比 (n-1)-bit recentralized 高 1 bit）；否则使用 recentralized quantization。

选用 2-Wasserstein 而非 KL 散度的原因：KL 散度非对称，不同方向距离不一致，不适合作为阈值决策度量；Wasserstein 距离对称且同时考虑了均值和方差的差异。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Wasserstein 决策在整个 FQ pipeline 中位于 GMM 拟合之后、量化执行之前：

```
# After EM fitting
μ_-, σ_-, μ_+, σ_+, σ²_global ← from GMM fitting

# Compute Wasserstein Separation
W = ((μ_+ - μ_-)² + (σ_+ - σ_-)²) / σ²_global

# Decision
if W < 2.0:
    # 分量高度重叠 → shift quantization (n-bit, full precision)
    mode = "shift"
    # 可利用全部 n bits 做 shift quant (无 component bit)
else:
    # 分量充分分离 → recentralized quantization ((n-1)-bit internal)
    mode = "recentralized"
    # 用 1 bit 区分 component, 剩余 (n-1) bits 做 shift quant

# Example from paper:
# ResNet-18 block22/conv1 (Figure 3): W ≈ 1.2 < 2.0 → shift mode
# ResNet-50 block3f/conv1 (Figure 2): W ≈ 5.3 ≥ 2.0 → recentralized mode
```

**Annotations**: 阈值 w_sep=2.0 是经验值，通过在 CIFAR-10 9 层 CNN 上 grid search（1.0~3.5, step=0.1, 每个值训练 100 次取平均）确定。论文实验显示，w_sep 设置恰使 1 个重叠层使用 shift、其余 8 层使用 recentralized 时平均准确率最高。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Wasserstein 分离度量作为逐层自适应判定，在 FQ 实现中完全自动化——给定量化 bit-width n 和阈值 w_sep，框架自动为每层计算 W 并选择最优模式。用户不需要手动干预。实现代码在 Mayo 框架中。

涉及论文标题：
- Focused Quantization for Sparse CNNs
