## Weight Standardization (WS / 权重标准化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Weight Standardization (WS) 是一种对神经网络中卷积层或全连接层权重向量进行标准化的训练技术，由 Qiao et al. (2019) 提出。对于第 l 层的权重向量 w_{n,m}（其中 m 索引输出通道），WS 将其标准化为 w̃_{n,m} = (ρ/σ(w_{n,m})) · (I - P_1) w_{n,m}。具体流程：(1) 先减均值（通过投影矩阵 P_1 = 1·1^T/d 去除 DC 分量），(2) 除以标准偏差 σ（归一化到单位方差），(3) 乘以缩放系数 ρ（可调超参数）。标准化后的 WSP (Weight-Standardized Parameter) 才被用于前向计算（卷积或矩阵乘法）。WS 通常与 Group Normalization (GN) 搭配使用，放在 GN 层之前。在 FL 场景中，WS 的核心价值在于其对梯度的隐式过滤作用：反向传播时梯度经历双重投影 ∂L/∂w = (ρ/σ)(I - P_1)(I - P_{w̃}) ∂L/∂w̃，依次移除与 WSP 对齐的分量（local overfitting 方向）和 mini-batch 均值分量（local data bias 方向），仅保留对全局收敛有益的方向。FedWSQ 传输的是 PSP (Pre-Standardized Parameter) 而非 WSP，区别于 FedWon（传输 WSP 强制统计一致但丢失本地适应性信息）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
WS 在 FedWSQ local training 中的伪代码（每层每个输出通道 m）：

```python
# WS Forward (applied before each layer's computation)
w = local_model.layer[l].weight[:, m]  # shape: (I_l,), PSP vector
w_mean = w.mean()                        # scalar
w_centered = w - w_mean                  # (I - P_1)w, projection onto span{1}^⊥
w_std = w_centered.std()                 # σ(w)
w_tilde = (rho / w_std) * w_centered     # WSP vector, Eq.(5)

# Use w_tilde for forward computation:
y_l[m] = w_tilde^T @ x_l                 # Eq.(3)

# WS Backward (gradient filtering, automatic via autograd):
# ∂L/∂w = (rho/σ) * (I - P_1) * (I - P_{w̃}) * ∂L/∂w̃    # Eq.(6)
# Step 1: (I - P_{w̃}) removes component aligned with w̃
# Step 2: (I - P_1) removes mean component
# Result: gradient projected onto span{w̃, 1}^⊥
# Only directions orthogonal to both w̃ and 1 survive
```

**Annotations**: ρ 为超参数（FedWSQ 默认 ρ=0.001），控制标准化后的参数 scale。w̃ 满足 zero-mean 和 ρ-scaled unit variance。两个投影矩阵 P_1 和 P_{w̃} 的连续作用等价于将梯度投影到 span{w̃, 1}^⊥。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
WS 作为 plug-and-play 技术，可直接插入任何 CNN/MLP 架构的卷积层或全连接层前。在 PyTorch 中可通过自定义 `weight_standardization` 函数实现，在 `forward()` 中对每层权重调用标准化后再执行 F.conv2d 或 F.linear。WS 通常与 GroupNorm (GN) 搭配（替换 BatchNorm），因为 BN 的 batch 统计在 FL 中不可靠（各 client 数据量不同）。推荐在 GN 之前应用 WS。超参数 ρ 不敏感（FedWSQ 实验表明 1e-4 到 1e-1 范围内准确率变化 <3%），因为 inference 时 normalization 层会消除常数 scale 的影响。在 FL 中，FedWSQ 传输 PSP 而非 WSP——这一设计选择使 client 隐式通过梯度过滤受益于 WS 的稳定性，同时保留 client-specific 的本地参数信息，避免 FedWon (传输 WSP) 导致的信息丢失。

涉及论文标题：
- FedWSQ Efficient Federated Learning with Weight Standardization and Distribution-Aware Non-Uniform Quantization
