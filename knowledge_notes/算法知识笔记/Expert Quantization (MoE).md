## Expert Quantization (MoE)

术语解释
MoE专家量化是将MoE模型中expert的高精度权重（FP16/FP32）转换为低精度表示（INT8/INT4/INT2/INT1）的技术，以显著减少内存占用和数据传输量。

术语是什么？
MoE量化的核心挑战是expert之间重要性不均，需为不同expert分配不同的量化策略：
- **MC-MoE**：基于expert的访问频率φ_i、激活权重w_i和量化损失ε_ij构建整数规划模型，为每个expert分配最优位宽（1/2/3 bit）。目标函数：min Σ φ_i^α · w_i^β · (ε_ij · x_ij)^γ
- **MoE-CSP**：将expert权重量化为4或8 bit，设计专用CUDA kernel处理量化权重+浮点计算
- **MoQE**：观察到expert的FFN层量化到2 bit对模型质量影响小，而self-attention量化显著损害性能
- **QMoE**：极致压缩至1 bit，实现可扩展压缩算法和自定义GPU kernel
- **CMoE**：二值权重网络（1 bit权重）+ 4 bit激活量化
- **HOBBIT**：动态精度选择——根据gating输出计算expert重要性分数，低于阈值用低精度版本，高于阈值用高精度版本
- **EdgeMoE**：通过校准数据集统计分析确定每个expert的最优位宽

从算法pipeline角度拆解术语。
```
# MC-MoE: Adaptive Bit-width Allocation
for each expert e_i in layer:
    φ_i = n_i / N                    # 访问频率
    w_i = sum(σ_j) / N               # 平均激活权重
    for bit j in {1, 2, 3}:
        W_q = quantize(W_i, j)       # 量化到j bit
        ε_ij = ||W_i - W_q||_F       # Frobenius范数量化损失

# 整数规划求解最优位宽分配
min Σ_i Σ_j φ_i^α · w_i^β · (ε_ij · x_ij)^γ
s.t. Σ_j x_ij = 1, x_ij ∈ {0, 1}
```
结果总结（Table 3）：
- 内存减少：4x-150x（取决于位宽和方法）
- 精度损失：0%-23.81%
- 推理加速：0.95x-26x（取决于是否有专用kernel）

术语一般如何实现？如何使用？
- 训练后量化（PTQ）：在小型校准集上确定量化参数，无需重训练
- 量化感知训练（QAT）：训练中模拟量化效果，精度更高但成本大
- 需要专用反量化kernel才能真正实现加速（否则仅节省内存，不加速计算）
- 结合offloading使用效果更佳——低精度expert加载延迟更低

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models
- Compression Error Sensitivity Analysis for Different Experts in MoE Model Inference
- EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models

**补充（来自 EAC-MoE）**：EAC-MoE 的 QESC 方法提出 MoE 量化特有的 **expert-shift** 退化机制——量化后 router 因激活噪声而选错 expert，其退化程度可达与权重误差同量级。QESC 通过逐层 TopK-MSE Loss 校准 router 而非依赖静态 expert 频率分配位宽，避免了 PMQ 等方法的跨任务过拟合问题（详见 Expert-Shift Problem 和 QESC 词条）。

**补充（来自 Compression Error Sensitivity Analysis）**：该论文将量化作为 baseline 对比方法，汇总了 MC-MoE/MoE-CSP/MoQE/QMoE/CMoE/MoE-MPTQS/HOBBIT/EdgeMoE 等八种 MoE 量化方案的性能数据（Table 1），指出低比特量化（1-4 bit）的共同缺陷是引入不可控、不可预测的误差，导致生成质量不稳定。这一观察直接驱动了用 error-bounded lossy compression（SZ3/CuSZp）替代量化来压缩 MoE expert 的提议——error-bounded 方法通过有界误差保证实现精度可控的压缩。

---
