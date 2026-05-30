## Entropy-to-Bit (E2B) Mechanism

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Entropy-to-Bit (E2B) 机制是 Granular-DQ 的第二阶段组件，对 GBC 阶段分配了高 bit-width (8-bit) 的 patch 进行细粒度 bit-width 再调整，进一步降低平均 bit-width。理论基础来自 Shannon 信息论：图像 patch 的像素熵反映信息密度和像素分布复杂度，低熵 patch（平坦区域、纯色背景）对量化精度要求低，可用更低 bit-width。

E2B 流程：(1) 训练集所有 LR patch 计算像素熵 $\mathcal{H} = -\sum_i \mathcal{P}(x_i) \log \mathcal{P}(x_i)$，$\mathcal{P}(x_i)$ 由 Gaussian 加权核密度估计得到；(2) 熵值升序排列为 $\mathbf{H}$；(3) 分位数阈值（$t_1=0.5, t_2=0.9$）将 $\mathbf{H}$ 划分为子区间，映射到候选 bit codes [4, 5, 8]；(4) 对 GBC 分配了 8-bit 的 patch，据其熵值 $E$ 查找落入区间确定最终 bit-width。

从算法pipeline角度拆解术语，给出具体计算过程。

```
# E2B + ATC
# 预处理: 训练集熵分布
for each LR_patch:
    r_i = x_i - bin_values
    P(x_i) = Σ exp(-r_i²/(2σ²)) / (ΣΣ exp(-r_i²/(2σ²)) + ε)
    H.append(-Σ P(x_i) log(P(x_i)))
H = sort(H)  # 升序

# 阈值 (分位数)
I_t1, I_t2 = ceil(M*0.5), ceil(M*0.9)  # t1=0.5, t2=0.9

# Bit 适配 (GBC 阶段 b_i==8 的 patch)
if E <= H_t1: b_i = 4     # 低熵 → 低精度
elif E <= H_t2: b_i = 5   # 中熵 → 中等精度
else: b_i = 8              # 高熵 → 高精度

# ATC: 首 epoch EMA 动态校准
t^(j) = t^(j-1) · 0.9997 + Norm(E) · (1-0.9997)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

E2B 仅需首 epoch 计算（LR 样本跨 epoch 一致）。候选 bit [4,5,8] 经消融实验选出最优配置。ATC 使用 EMA (γ=0.9997) 平滑校准阈值。E2B 量化使用 QuantSR 方案。

涉及论文标题：
- Thinking in Granularity Dynamic Quantization for Image Super-Resolution by Intriguing Multi-Granularity Clues
