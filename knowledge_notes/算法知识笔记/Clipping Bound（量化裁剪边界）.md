## Clipping Bound（量化裁剪边界）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Clipping Bound（裁剪边界）是量化器中定义数值截断范围的参数，通常包含下界 l 和上界 u。在量化公式 `v_q = s * round(clip(v, l, u) / s)` 中，所有小于 l 的值被映射为 l，所有大于 u 的值被映射为 u，只有 [l, u] 区间内的值保持正常的量化分辨率。裁剪边界的设置直接影响量化的信息保真度：边界太宽（如 [min, max]）会导致长尾分布中的离群值占据过多的量化区间，使得密集区域的表示精度不足（有效位利用率低）；边界太窄会截断过多信息，导致显著的信息丢失。2DQuant 论文发现 SwinIR 的权重和激活呈现"对称+非对称共存+长尾"的分布特征，需要针对每种分布类型采用不同的边界搜索策略。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 2DQuant 的 DOBI 阶段，clip bounds 的搜索过程：
```
Data: v (张量), K=100 (搜索点数), N (bit数)
Result: l_best, u_best
l ← min(v), u ← max(v)
min_mse ← +∞
if v 对称:    # 钟形分布 → 双界同时收缩
    Δl ← (max(v) - min(v)) / (2K)
else:         # 指数分布 → 固定下界不动
    Δl ← 0
Δu ← (max(v) - min(v)) / (2K)
for i in 0..K:
    l_i ← l + i*Δl, u_i ← u - i*Δu
    v_q ← fake_quantize(v, l_i, u_i, N)
    mse ← ||v - v_q||_2
    if mse < min_mse:
        min_mse ← mse; l_best ← l_i; u_best ← u_i
```
DQC 阶段进一步用梯度下降微调每个量化器的 (l, u)，优化目标从 MSE（值层面）转向任务目标（输出+特征蒸馏 loss）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Clipping bound 的确定方法主要有三类：(1) 统计方法——MinMax（取 min/max）、Percentile（取 p-百分位数）、MSE 最小化（在 min 到 max 之间搜索/优化）；(2) 训练方法——通过 STE 梯度回传直接学习最优边界值（如 PACT、LSQ、2DQuant 的 DQC）；(3) 分析/启发式方法——如基于分布的峰度/偏度自动调整。在 PyTorch 量化 API 中，Observer 模块（如 `MinMaxObserver`、`MovingAverageMinMaxObserver`、`HistogramObserver`）负责在校准阶段统计并确定 clip bounds。

涉及论文标题：
- 2DQuant Low-bit Post-Training Quantization for Image Super-Resolution
- PMQ-VE Progressive Multi-Frame Quantization for Video Enhancement

在 PMQ-VE 中，Clipping bound 被扩展为 per-frame 概念：对多帧激活张量 X∈R^{N×C×H×W}，每帧 X_i 拥有独立的 (lb_i, ub_i)，通过 BTBI（回溯搜索）在百分位约束的搜索空间 [p_{0.1}, p_{10}] × [p_{90}, p_{99.9}] 中找到使 ||X_i - X̂_i||_2 最小的边界。这与传统 per-tensor 量化中所有帧共享统一边界的做法不同——PMQ-VE 的逐帧边界可适配各帧特有的激活分布，解决了视频增强中帧间激活分布差异导致的量化次优问题。

---
