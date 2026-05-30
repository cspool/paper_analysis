## Feature Manifold Curvature (Geometric Curvature C_t)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Feature Manifold Curvature（特征流形曲率）是 CurveStream 提出的几何度量 C_t，用于衡量连续视频帧在潜空间特征流形上的轨迹弯曲程度。该度量定义在视频帧经视觉编码器映射到的高维特征空间（单位超球面流形）上：将每帧 I_t 经冻结编码器编码为 F_t 并 L2 归一化后，连续时间步的帧在流形上形成一条离散参数化曲线。C_t 近似计算该曲线的二阶几何曲率（方向导数），定义为相邻时间步特征位移向量 d1 = F_{t-1} - F_{t-2} 和 d2 = F_t - F_{t-1} 之间的余弦距离：C_t = 1 - ⟨d1,d2⟩/(||d1||·||d2||)。在微分几何中等价于 C_t = 1/2 ||T2 - T1||²，其中 T1, T2 是归一化后的单位切向量（即瞬时演化方向）。C_t 的核心理论优势：(1) 免疫恒速运动噪声——平滑相机运动下特征匀速演化，T1≈T2 → C_t≈0；(2) 对语义突变正交敏感——当场景发生突变（新实体进入、镜头切换、动作边界），特征轨迹方向急剧偏转，T2 投射到近乎与 T1 正交的子空间 → C_t 急剧增大。这种"对方向敏感、对模长免疫"的特性使 C_t 成为理想的语义转换检测器。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
C_t 在 CAS 内部的运动学建模（论文 Appendix C）：
```
# 运动学视角：特征位移向量 = 离散速度向量
d1 = F_{t-1} - F_{t-2}    # t-1 时刻的瞬时速度
d2 = F_t - F_{t-1}        # t 时刻的瞬时速度

# 归一化为单位切向量（方向）
T1 = d1 / ||d1||           # t-1 时刻演化方向
T2 = d2 / ||d2||           # t 时刻演化方向

# 几何曲率 = 切向量变化平方的一半
C_t = 1 - dot(T1, T2)      # = (1/2) * ||T2 - T1||²
```
典型场景对比：
- 恒速相机平推（背景整体位移）: d1 ≈ d2（模长接近，方向相同）→ T1 ≈ T2 → C_t ≈ 0 → 被 HVMM 归为 Discard 或 Blurred Memory
- 新物体突然出现（语义突变）: d1（背景运动方向）与 d2（包含新物体的特征位移）方向显著不同 → T1 ⊥ T2（近似正交）→ C_t → 1（尖峰）→ 被 HVMM 归为 Clear Memory

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
C_t 在 CurveStream 中仅需保存最近 3 帧的特征 F_t, F_{t-1}, F_{t-2}（滑动窗口），不需要全历史序列，使其适用于无限长流视频。计算复杂度和内存开销均为 O(D)，D = 384（DINOv2-small 输出维度）。C_t 与传统物理度量（cosine similarity, optical flow）的关键区别：cosine similarity 仅衡量模长变化（混淆语义突变与全局平移），optical flow 对像素噪声极度敏感（论文 Table III 中仅 46.54% accuracy），而 C_t 通过二阶方向微分天然解耦了运动幅度和方向变化。C_t 的鲁棒性验证：当 λ 在 [0.2, 1.0] 范围内变化时，accuracy 波动仅 3.33%（Table IV），证明曲率度量本身信号稳定。

涉及论文标题：
- CurveStream__Boosting_Streaming_Video_Understanding_in_MLLMs_via_Curvature-Aware_Hierarchical_Visual_Memory_Management
