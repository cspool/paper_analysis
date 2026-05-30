## BMFQ（Backtracking-based Multi-Frame Quantization / 回溯式多帧量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

BMFQ（Backtracking-based Multi-Frame Quantization）是 PMQ-VE 框架粗阶段（coarse stage）的核心方法，负责为多帧视频增强模型中的每个帧独立估计最优量化裁剪边界。BMFQ 包含两个关键设计：(1) **逐帧百分位初始化**——对每帧激活 X_i，将搜索空间限定在 [p_{0.1}(X_i), p_{10}(X_i)] × [p_{90}(X_i), p_{99.9}(X_i)]，以此抑制长尾分布中的 outliers；(2) **BTBI 回溯搜索算法**——从百分位初始点出发，递归探索候选 (lb, ub) 配置，通过评估量化误差 ||X_i - X̂_i||_2 指导搜索方向，在遇到误差上升时回溯到上一节点尝试其他方向（剪枝+回溯），终止条件为所有候选被评估或满足收敛阈值 ε。

BMFQ 相比传统 PTQ 边界搜索方法（如 2DQuant 的对称/非对称单向搜索、DBDC+Pac 的顺序调整）的优势在于：(a) 不受 uniform shrink 方向限制，探索更丰富的候选空间；(b) 百分位初始化使搜索起点不受 outliers 影响；(c) 回溯机制允许算法从局部极值逃逸。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

BTBI 伪代码（BMFQ 核心搜索算法）：

```
输入: X_i (第 i 帧激活), 步长 ΔL, ΔU, 收敛阈值 ε
输出: 最优 (lb*, ub*)

# 百分位初始化搜索空间
lb_0 = p_{0.1}(X_i)          # 下界起始于 0.1 分位数
ub_0 = p_{99.9}(X_i)         # 上界起始于 99.9 分位数
visited = {}                  # 已访问节点集合
error_min = +∞

Function Backtrack(lb, ub):
    if (lb, ub) in visited or lb < p_{0.1} or lb > p_{10}
       or ub < p_{90} or ub > p_{99.9}:
        return                                           # 越界剪枝
    visited = visited ∪ {(lb, ub)}
    X̂_i = QuantizeDequantize(X_i, lb, ub)               # uniform quantizer
    err = ||X_i - X̂_i||_2                                 # MSE 评估
    if err > error_min + ε:
        return                                           # 误差剪枝
    if err < error_min:
        error_min = err; lb* = lb; ub* = ub             # 更新最优
    foreach (δ_l, δ_u) in {(±ΔL, 0), (0, ±ΔU)}:         # 四个搜索方向
        Backtrack(lb + δ_l, ub + δ_u)                    # 递归搜索

Backtrack(lb_0, ub_0)
return (lb*, ub*)
```

搜索复杂度为 O(|S|)，其中 S 为搜索空间网格点数。由于剪枝和 visited 集合的存在，实际搜索节点数远小于网格点总数。

在 PMQ-VE 的完整流程中，BMFQ 仅需少量校准数据（无需标签），运行在蒸馏微调之前，为 PMTD 提供合理的初始量化边界。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

BMFQ 在 PyTorch 中实现，使用 fake quantization 模拟量化效果。百分位计算通过 `torch.quantile` 完成，量化误差评估在 FP32 精度下进行（反量化后的值与原始值比较）。BMFQ 运行在校准阶段，仅使用训练集中少量样本（如 Vimeo-90K 的子集），不需要反向传播或梯度计算——所有搜索基于前向 MSE 评估。代码开源：https://github.com/xiaoBIGfeng/PMQ-VE。

消融实验（STVSR 2-bit）：Baseline（MinMax，无 BMFQ，无 PMTD）12.67dB → +Per-Frame 19.64dB → +BMFQ 27.56dB（+7.92dB）。

涉及论文标题：
- PMQ-VE Progressive Multi-Frame Quantization for Video Enhancement

---
