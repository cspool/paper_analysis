## Percentile-based Initialization for Quantization Bounds（百分位数量化边界初始化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

百分位数量化边界初始化是一种用于 PTQ 中确定量化裁剪范围的技术：使用张量分布的百分位数（而非全局 min/max）作为量化裁剪边界的初始估计。在 PMQ-VE 中，BMFQ 阶段的下界 lb 初始化为 p_{0.1}（第 0.1 百分位数），上界 ub 初始化为 p_{99.9}（第 99.9 百分位数），搜索空间限定在 lb ∈ [p_{0.1}, p_{10}] 和 ub ∈ [p_{90}, p_{99.9}]。

其核心优势在于抑制长尾分布中的 outliers 对量化分辨率的影响：当激活张量存在极端的正值或负值 outliers 时（如 Transformer 中的 post-GELU 激活），使用 min/max 作为量化边界将导致大部分正常值被压缩到极少的量化 bin 中，造成严重的分辨率浪费。百分位初始化通过直接裁剪掉最极端的 0.1%-10% 值，使量化 bin 集中在激活分布的主体区域，显著降低 MSE。

与 MinMax（p_0 和 p_100）相比，百分位初始化的搜索范围更窄且更接近最优解，收敛更快。与仅使用 MSE 搜索边界但不使用百分位约束的传统方法相比，百分位初始化提供了一种抗离群的先验。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在 PMQ-VE 的 BMFQ 中：

```
X_i ∈ R^{C×H×W}              # 第 i 帧的激活
p_vals = torch.quantile(X_i.flatten(), 
                        torch.tensor([0.1, 10.0, 90.0, 99.9]) / 100.0)
p_01 = p_vals[0]              # 0.1 百分位 → lb 下界
p_10 = p_vals[1]              # 10 百分位 → lb 上界
p_90 = p_vals[2]              # 90 百分位 → ub 下界  
p_999 = p_vals[3]             # 99.9 百分位 → ub 上界

# 搜索空间定义
S_i = {(lb, ub) | lb ∈ [p_01, p_10], ub ∈ [p_90, p_999]}

# 搜索起点
lb_0 = p_01, ub_0 = p_999
```

与其他方法的对比：MinMax 将搜索范围设为 [min, max]（无 outlier 抑制）；2DQuant 使用对称/非对称的单向 shrink 搜索（探索范围受限）；PMQ-VE 的百分位初始化 + 回溯搜索在不缩小有效搜索范围的同时避免了 outlier 污染。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 PyTorch 中使用 `torch.quantile(input, q)` 计算分位数；历史方法（如 Percentile [Li et al. 2019]）也使用百分位数但直接作为最终量化边界（无搜索优化），效果远低于搜索方法。PMQ-VE 将百分位仅作为搜索起点和边界约束，而非最终量化参数。代码开源：https://github.com/xiaoBIGfeng/PMQ-VE。

涉及论文标题：
- PMQ-VE Progressive Multi-Frame Quantization for Video Enhancement

---
