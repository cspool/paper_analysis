## Functional Connectivity (FC, 功能连接)

术语解释
Functional Connectivity (FC) 是 fMRI 数据分析的核心表示，通过计算不同脑区 BOLD 时间序列之间的 Pearson 相关系数，量化脑区之间的功能同步程度。FC 矩阵是脑网络分析的基础数据结构。

术语是什么？
给定 fMRI 数据经脑图谱（如 AAL 116 ROIs）分区后，每个脑区提取一条 BOLD 时间序列 x_i(t)。两个脑区 i 和 j 之间的 FC 定义为：

$$FC_{ij} = \frac{\sum_t (x_i(t) - \bar{x}_i)(x_j(t) - \bar{x}_j)}{\sqrt{\sum_t (x_i(t) - \bar{x}_i)^2 \sum_t (x_j(t) - \bar{x}_j)^2}}$$

结果 FC ∈ R^{M×M}，M=脑区数（通常 116~400），是对称矩阵，值域 [-1, 1]。FC 矩阵反映了全脑功能网络拓扑结构——高 FC 表示两个脑区在时间上高度同步，可能参与相同的认知过程或属于同一功能网络（如 Default Mode Network, DMN）。

从算法pipeline角度拆解术语。
```
# FC 计算 Pipeline
# 输入: 4D fMRI (x, y, z, t) + T1w MRI
# 输出: FC matrix [M, M]

# Step 1: 组织分割 (T1w MRI)
tissue_seg = FSL_FAST(T1w_MRI)    # 白质、灰质、脑脊液

# Step 2: 脑图谱配准与分区
atlas = AAL116  # 或 Schaefer400、C200
for region_r in atlas.regions:    # r = 1..M
    # Step 3: 提取区域BOLD时间序列
    bold_r[t] = mean(fMRI_4D[x,y,z,t] for (x,y,z) in region_r)
    # bold_r: [T] 时间序列, T = 扫描时间点数

# Step 4: 计算Pearson相关 → FC矩阵
for i in range(M):
    for j in range(M):
        FC[i,j] = pearson_corr(bold_i, bold_j)
```

术语一般如何实现？如何使用？
- 预处理工具：FSL (FMRIB Software Library)，用于组织分割、运动校正、空间标准化
- 脑图谱选择：AAL (116 ROIs) 基于解剖标志；Schaefer (400 ROIs) 基于功能边界；C200 基于连接组
- FC 应用：作为 BrainMass 等 brain foundation model 的输入进行自监督预训练（mask reconstruction）；作为 BrainMoE Cognition Adapter 中 cross-attention 的 Key-Value source
- BOLD vs FC：BOLD 保留时间维度信息（timeseries），FC 通过相关性压缩为静态网络。小数据集上 FC 常优于 BOLD 作为输入特征（更高的 SNR）

涉及论文标题：
- BrainMoE Cognition Joint Embedding via Mixture-of-Expert Towards Robust Brain Foundation Model

---
