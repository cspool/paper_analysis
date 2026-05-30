## Post-Training Quantization (PTQ / 后训练量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Post-Training Quantization (PTQ) 是一种在模型训练完成后对权重（有时也包括激活）进行低精度压缩的方法，无需重新训练或微调。典型流程包括：（1）用少量校准数据（calibration set）前向传播模型，收集各层的激活分布或权重统计信息；（2）基于这些统计信息计算量化参数（scale s 和 zero-point z）；（3）将 FP16/FP32 权重量化到 INT4/INT8 等低精度格式。PTQ 的核心优势是快速、不需大量计算，适合部署场景。

与 Quantization-Aware Training (QAT) 的区别：PTQ 不需要训练，校准数据仅需几百到几千条序列，量化过程只需一次前向传播，总耗时通常在分钟级；QAT 在训练时模拟量化噪声并通过反向传播调整权重，需要完整训练 pipeline 和大规模数据。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

以 per-channel 对称均匀量化的 PTQ 为例：

```
# 输入：FP16 权重矩阵 W ∈ R^{o×c}，校准数据 X ∈ R^{b×c}
# 输出：量化权重 W_q ∈ Z^{o×c}，scale s ∈ R^o

# Step 1: 确定量化参数 (per-channel)
for i in range(o):
    s[i] = max(abs(W[i,:])) / (2^{bit-1} - 1)  # symmetric, max range

# Step 2: 量化
W_q = clamp(round(W / s), q_min, q_max)  # q_min = -2^{bit-1}, q_max = 2^{bit-1}-1

# Step 3: 反量化（推理时）
W_hat = W_q * s  # 近似原始权重

# Step 4: 推理计算
output = X @ W_hat  # 或使用 INT4/FP16 dequant kernel
```

在 GPTQ 中，量化不是独立逐行完成的，而是逐列量化并使用 Hessian 补偿误差：H = X X^T，对每列 j，量化 W[:,j] → 计算误差 ΔW[:,j] = W_q[:,j] * s - W[:,j] → 用 H^{-1} 将误差按比例补偿到剩余列的权重上。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

主流实现：
- **GPTQ (Frantar et al., 2022)**：基于 OBQ 的 Hessian 误差补偿方法，每次量化一列后用 Hessian 逆补偿剩余列，支持 4-bit/3-bit 量化。开源：https://github.com/IST-DASLab/gptq
- **AWQ (Lin et al., 2023)**：利用激活分布选择平滑系数和剪枝权重，量化损失为 L = ||WX - W_hat X||_F^2。开源：https://github.com/mit-han-lab/llm-awq
- **SmoothQuant (Xiao et al., 2022)**：通过数学等效变换将量化难度从 activation 迁移到 weight，支持 W8A8 量化。
- **QuaRot (Ashkboos et al., 2024)**：使用 Hadamard 变换消除 outlier 实现免旋转的量化。

PTQ 在 MoE 模型上的挑战：由于每个 expert 只处理部分 token，校准集分布不均会导致部分 expert 校准不足（inter-expert imbalance）；同时 MoE 的 gating coefficient 使不同 token 对同一 expert 具有不同重要程度（intra-expert imbalance），这些因素是 MoEQuant 论文的核心动机。

涉及论文标题：
- MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance
