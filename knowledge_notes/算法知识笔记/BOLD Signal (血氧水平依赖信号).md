## BOLD Signal (血氧水平依赖信号)

术语解释
BOLD (Blood-Oxygen-Level Dependent) 信号是 fMRI 测量的核心生理信号，反映神经元活动引起的局部血氧浓度变化。当脑区激活时，局部血流增加带来的氧合血红蛋白变化导致 MR 信号强度改变，形成 BOLD 对比度。

术语是什么？
BOLD 信号是 fMRI 的基础测量量：(1) 神经元活动→局部代谢需求增加→血管扩张→脑血流增加→氧合血红蛋白（diamagnetic）相对脱氧血红蛋白（paramagnetic）比例上升→MR T2* 信号增强→fMRI 记录的信号变化；(2) BOLD 响应具有 hemodynamic delay（约 4-6 秒延迟，~12 秒恢复基线）；(3) 4D fMRI 数据：3D 空间体积 × 时间维度 = [x, y, z, t]，每个 voxel 的时间序列即为 BOLD 信号。

从算法pipeline角度拆解术语。
```
# BOLD 信号的使用
# 输入: raw 4D fMRI [x, y, z, time_points]
# 预处理步骤:
1. motion_correction(fMRI)         # 头动校正
2. slice_timing_correction(fMRI)   # 层时间校正
3. spatial_normalize(fMRI, MNI)   # 空间标准化到MNI模板
4. spatial_smooth(fMRI, FWHM=6mm) # 高斯平滑, 提高SNR
5. bandpass_filter(fMRI, 0.01-0.1Hz) # 保留神经活动频段

# 两种使用路径:
# Path A: BOLD → FC
for region in atlas:
    bold_ts[region] = mean(BOLD[voxels_in_region], axis=space)
FC = corr(bold_ts)  # [M, M]

# Path B: 直接使用 BOLD latent features
# (如 BrainJEPA, BrainLM)
Z = ViT_encoder(BOLD_4D_masked)  # spatiotemporal encoding
```

术语一般如何实现？如何使用？
- BOLD 作为 brain foundation model 的输入：BrainLM 和 BrainJEPA 直接对 BOLD timeseries 做 spatiotemporal masking 和 reconstruction，保留时间动态信息
- BOLD vs FC 权衡：BOLD 含时间信息但维度高（噪声大），FC 压缩为静态矩阵但丢失时序动态
- 预处理管线差异：不同研究使用不同预处理 pipeline（FSL, SPM, DPABI），导致 BOLD/FC 特征分布差异，影响模型跨数据集泛化——这也是 BrainMoE 需要鲁棒设计的原因之一

涉及论文标题：
- BrainMoE Cognition Joint Embedding via Mixture-of-Expert Towards Robust Brain Foundation Model

---
