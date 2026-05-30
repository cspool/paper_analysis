## Hessian Sharpness and Trace in Quantization Robustness

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hessian Sharpness（Hessian 矩阵最大特征值 λ_max，也称"锐度"）和 Hessian Trace（Hessian 矩阵对角线之和，即所有特征值之和）是衡量神经网络 loss landscape 局部几何性质的关键二阶统计量。Sharpness 反映 loss 盆地在最陡方向上的曲率——值越大，loss 对权重扰动的敏感度越高。Trace 反映所有方向上的平均曲率——值越大，整体上 loss 曲面越不平坦。该论文将这两个指标与 PTQ 鲁棒性建立了因果关系：量化的本质是对权重施加离散扰动 W → Ŵ = W + ΔW。如果模型处于尖锐的 loss 区域（高 sharpness/trace），同样的 ΔW 导致更大的 loss 增加，因此量化误差更大。论文通过受控实验证明：(1) lr 衰减时 sharpness 和 trace 同时激增（Fig.9），与量化误差激增的时间模式完全一致；(2) 较大的峰值学习率产生更低的 sharpness 和 trace（Fig.26），对应更低的量化误差；(3) Weight averaging 也降低 sharpness，解释了其对 PTQ 的益处。论文由此假设：lr 衰减使模型进入更尖锐的 loss 区域，使其对量化扰动更敏感。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Hessian Sharpness 和 Trace 的估计（基于 PyHessian + Hutchinson estimator）：
```
def estimate_hessian_stats(model, val_loader, n_ht=100, n_pi=50):
    # Hutchinson estimator for trace (unbiased)
    traces = []
    for _ in range(n_ht):
        z = torch.randint(0, 2, shape) * 2 - 1  # Rademacher
        Hv = autograd_hvp(loss_fn, model_params, z)
        traces.append(dot(z, Hv))
    trace_est = mean(traces)

    # Power iteration for λ_max (sharpness)
    v = torch.randn(shape); v /= norm(v)
    for _ in range(n_pi):
        Hv = autograd_hvp(loss_fn, model_params, v)
        lambda_max = dot(v, Hv)
        v = Hv / norm(Hv)
    return lambda_max, trace_est
```
论文的发现模式：(a) 恒学习率阶段：仅 top eigenvalue 短暂上升但其余保持小值→量化误差温和增加；(b) Decay 阶段：所有 eigenvalue 同时急剧上升→量化误差飙升；(c) 较大峰值 lr → 更小的 sharpness 和 trace → 更低量化误差。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyHessian (Yao et al., 2019) 是标准的 PyTorch Hessian 分析工具（https://github.com/amirgholami/PyHessian），通过 Hessian-vector product (HVP) 自动微分实现无显式构建完整 Hessian 矩阵的统计量估计（160M 参数模型的完整 Hessian 有 2.56×10^16 个元素无法存储）。Hutchinson estimator 使用 Rademacher 分布（Gaussian 也有效但方差更大）；power iteration 收敛到 λ_max 需要 50+ 次 HVP 迭代；HVP 通过对 loss 做两次反向传播实现，内存和时间为标准前向+反向的约两倍。论文在验证集（100 个 FineWebEdu 序列，每个 2048 tokens）上计算统计量。该技术可推广到监控任何训练过程中的泛化和鲁棒性指标变化趋势。

涉及论文标题：
- Training Dynamics Impact Post-Training Quantization Robustness
