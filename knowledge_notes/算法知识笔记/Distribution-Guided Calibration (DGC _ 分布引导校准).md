## Distribution-Guided Calibration (DGC / 分布引导校准)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DGC（Distribution-Guided Calibration）是 PTQ4ARVG 提出的一种基于分布熵的校准集选择方法。核心观察：ARVG 模型中网络激活跨样本高度相似（尤其无条件样本），这种 sample-wise 冗余导致随机采样校准集时量化参数的校准不匹配（mismatched calibration）。DGC 使用 Mahalanobis 距离 ρ(x) = √((x-u)^T S^{-1} (x-u)) 衡量每个样本对整体分布熵的贡献，其中 u 和 S 分别为校准池的均值和协方差矩阵。ρ(x) 越大，表示该样本与整体分布的差异越大，对分布熵的贡献越大。DGC 选择 top 50% 高熵样本作为最终校准集，消除冗余样本，确保校准集与真实分布匹配。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
输入: 校准池 X_pool = {x_1, ..., x_N} (N 张 ImageNet 图像)
输出: 校准集 X_cal (N/2 张高熵样本)

# Step 1: 计算校准池的分布统计量
u = mean(X_pool)         # 均值向量 (在特征空间)
S = cov(X_pool)          # 协方差矩阵

# Step 2: 计算每个样本的 Mahalanobis 距离
for i = 1 to N:
    d = x_i - u                          # 偏差向量
    ρ[i] = sqrt(d^T · S^{-1} · d)       # Mahalanobis 距离 (Eq. 17)

# Step 3: 选择高熵样本
sorted_indices = argsort(ρ, descending=True)
X_cal = {X_pool[i] for i in sorted_indices[0:N/2]}  # top 50%

# Step 4: 使用 DGC 选择的校准集进行 PTQ
quantize_model_with_calibration(X_cal)
```

Mahalanobis 距离 vs Euclidean 距离的关键区别：Mahalanobis 距离考虑了特征之间的相关性（通过协方差矩阵 S），能识别在相关结构下真正"异常"的样本，而非仅测量绝对距离。这使得 DGC 能有效识别对分布熵贡献大的样本。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DGC 是一种校准集预处理方法，可插拔到任意 PTQ 流程。在 PTQ4ARVG 中，DGC 与 GPS 和 STWQ 协同使用：先用 DGC 选择 128 张高熵 ImageNet 图像，然后用 GPS 计算 scaling factor，最后用 STWQ 设定 per-token 量化参数。PTQ4ARVG 的实验表明：(1) DGC 在所有指标（IS/FID/sFID/Precision）上一致优于 random sampling 和 uniform sampling；(2) DGC 随校准集大小增大持续提供一致改进，展现强鲁棒性；(3) DGC 的消融实验中，加入 DGC 后的完整 PTQ4ARVG（GPS+STWQ+DGC）在 RAR-B W6A6 上将 FID 从 6.67（GPS+STWQ only）进一步降至 5.13。DGC 的计算开销很小——仅需对校准池计算一次均值和协方差。开源：https://github.com/BienLuky/PTQ4ARVG。

涉及论文标题：
- PTQ4ARVG Post-Training Quantization for AutoRegressive Visual Generation Models
