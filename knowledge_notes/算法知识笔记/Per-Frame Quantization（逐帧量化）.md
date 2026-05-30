## Per-Frame Quantization（逐帧量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Per-Frame Quantization（逐帧量化）是一种针对多帧视频增强模型的量化策略，为输入张量中每个帧独立分配量化裁剪边界（lb_i, ub_i），而非对所有帧使用统一的 per-tensor 量化参数。其动机来自于多帧视频增强 Transformer 中不同帧的激活分布存在显著差异——网络对不同帧分配不同的注意力权重，导致各帧的激活值范围、分布形态（对称/非对称、长尾程度）各不相同。使用统一的 per-tensor 量化范围会造成某些帧的动态范围利用不足（过宽裁剪→分辨率浪费）或截断过多（过窄裁剪→信息丢失），而逐帧量化使每帧获得适配其自身激活统计的最优量化分辨率。

PMQ-VE 论文通过实验统计展示了这一分布差异：对 RSTT（STVSR）中各帧的激活值分别绘制分布图，发现不同帧的 min/max 激活值差异可达数倍。采用 per-frame 量化后，每帧独立搜索 [lb_i, ub_i]，保证量化误差在帧间均匀分布。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在 PMQ-VE 的 BMFQ 阶段，per-frame 量化的具体流程如下：

```
输入: 多帧激活张量 X ∈ R^{N×C×H×W}，N 帧
输出: 每帧的量化边界 {(lb_i, ub_i)}_{i=1..N}

for i = 1 to N:                          # 逐帧处理
    X_i = X[i, :, :, :]                  # 第 i 帧的激活
    # 百分位初始化（抑制 outliers）
    lb_0 = percentile(X_i, 0.1)          # 下界初始值
    ub_0 = percentile(X_i, 99.9)         # 上界初始值
    # 在搜索空间 [p0.1, p10] × [p90, p99.9] 内查找最优 (lb, ub)
    (lb_i, ub_i) = BTBI(X_i, lb_0, ub_0, ΔL, ΔU)
    # 执行逐帧量化
    X̂_i = fake_quantize_per_channel(X_i, lb_i, ub_i, N_bits)

# 后续层的计算使用量化后的 X̂ = [X̂_1; X̂_2; ...; X̂_N]
```

与传统的 per-tensor quantization（所有帧共享同一对 [lb, ub]）对比，逐帧量化使每帧的量化误差 ||X_i - X̂_i||_2 独立最小化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

逐帧量化的实现不需要特殊的硬件支持——在 fake quantization 框架下，每帧的裁剪边界仍然在 PyTorch 张量操作层面实现。与 per-tensor 量化的唯一区别在于搜索/优化阶段为每帧独立维护一组 (lb_i, ub_i)，前向推理时对每帧独立执行 clamp+round+dequantize。在 PMQ-VE 的实现中，逐帧量化仅应用于激活（权重仍使用 per-channel 量化），且仅对注意力模块中的关键线性层和 MatMul 层执行。代码开源：https://github.com/xiaoBIGfeng/PMQ-VE。

涉及论文标题：
- PMQ-VE Progressive Multi-Frame Quantization for Video Enhancement

---
