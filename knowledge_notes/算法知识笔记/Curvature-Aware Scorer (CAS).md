## Curvature-Aware Scorer (CAS)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Curvature-Aware Scorer (CAS) 是 CurveStream 框架的核心感知模块，用于在无限长流视频中对每帧的语义转换强度进行实时评估。CAS 是一个 training-free 的评分器，使用冻结的轻量视觉编码器（DINOv2-small，~22M 参数，输出 384 维特征）提取每帧的全局特征表示 F_t ∈ R^D 并进行 L2 归一化，投影到单位超球面上。CAS 的核心创新在于同时融合**一阶运动强度**和**二阶几何曲率**来构造综合曲率分数 CS_t = M_t + λ·C_t：(1) Motion Variation M_t = 1 - cos(F_t, F_{t-1}) 衡量相邻帧之间的特征位移模长（一阶信息）；(2) Geometric Curvature C_t = 1 - cos(d1, d2) 衡量特征位移向量 d1 = F_{t-1} - F_{t-2} 与 d2 = F_t - F_{t-1} 之间的角度偏差（二阶信息）。C_t 在微分几何视角下严格等价于 1/2 ||T2 - T1||²（单位切向量变化的平方），即特征轨迹流形曲率的离散近似。这一几何特性使得 C_t 在恒速物理运动（如平滑相机平移/旋转）中自然趋近于 0，仅在特征演化方向发生突变时产生显著尖峰，实现了从低层次物理运动到高层次语义转换的数学解耦。λ 作为几何惩罚项的平衡系数（论文默认 λ=0.2），用于调节曲率分量在综合评分中的权重。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CAS 在 CurveStream pipeline 中的位置：连续视频帧 I_t → DINOv2-small 编码 → L2 归一化 → CAS 计算 CS_t → 馈入 HVMM 进行记忆路由。
伪代码：
```
# 输入: t 时刻帧 I_t, 前两个时刻的特征 F_{t-1}, F_{t-2}
F_t = DINOv2_small(I_t)        # 冻结视觉编码器, shape: (D,)
F_t = F_t / ||F_t||_2          # L2归一化到单位超球面

# 一阶 Motion Variation
M_t = 1 - dot(F_t, F_{t-1})    # 余弦距离, F_t和F_{t-1}已归一化

# 二阶 Geometric Curvature
d1 = F_{t-1} - F_{t-2}         # shape: (D,)
d2 = F_t - F_{t-1}              # shape: (D,)
C_t = 1 - dot(d1,d2)/(||d1||*||d2||)  # 位移向量角度偏差

# 综合曲率分数 (λ=0.2)
CS_t = M_t + 0.2 * C_t          # scalar ∈ [0, 2.2]
```
CAS 的核心几何解释（论文 Appendix C）：将位移向量归一化为单位切向量 T1 = d1/||d1||, T2 = d2/||d2||，则 C_t = 1 - ⟨T1, T2⟩ = 1/2||T2 - T1||²。这表明 C_t 是切向量变化平方的一半，直接度量特征演化方向的变化率而非模长变化——这是区分"语义突变"（方向变）和"平滑运动"（模长大但方向不变）的数学基础。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CAS 作为 training-free 模块，不需要任何训练。在 CurveStream 中，CAS 使用 DINOv2-small 作为特征提取器（论文默认使用该模型），DINOv2-small 以 ViT-S/14 架构在 142M 张无标签图像上自监督预训练，输出 384 维特征。CAS 的计算开销极低：每帧仅需一次 DINOv2-small 前向传播（~0.5G FLOPs）加上少量向量运算（dot product + L2 normalization = O(D) ≈ O(384)）。论文消融实验（Table III）验证了曲率度量的有效性：Uniform Sampling=69.04%, Cosine Similarity=73.28%, Optical Flow=46.54%, Pyramid Optical Flow=75.69%, Curvature=77.31%（训练无关方法中最优）。CAS 独立使用（无 HVMM）在 StreamingBench 上带来 +9.12% 的绝对提升（Table IX），在 OVOBench 上带来 +8.39% 的提升（Table X）。

涉及论文标题：
- CurveStream__Boosting_Streaming_Video_Understanding_in_MLLMs_via_Curvature-Aware_Hierarchical_Visual_Memory_Management
