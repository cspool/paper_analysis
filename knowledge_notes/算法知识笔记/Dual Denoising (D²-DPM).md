## Dual Denoising (D²-DPM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dual Denoising（D²-DPM，双重去噪）是 AAAI 2025 提出的针对量化扩散模型的后训练去噪机制，"双重"指在逆扩散采样过程中先后处理两类噪声：(1) **量化噪声去噪**：在每个时间步从量化模型输出 ε̂_θ(x_t, t) 中减去估计的量化噪声，恢复扩散噪声分布；(2) **扩散噪声去噪**：用恢复后的扩散噪声分布执行标准逆扩散采样。核心流程：校准阶段用 BRECQ 量化模型后，收集量化输出-噪声对 (ε̂, Δε)_t，为每个时间步 t 估计 ε̂ 和 Δε 的联合高斯分布参数；推理阶段在每一步用 TSQNM 条件化地预测量化噪声的均值和方差，进行修正后再执行 DDIM/DDPM 采样更新。提供两种变体：S-D²（随机双重去噪，减去完整估计噪声）和 D-D²（确定性双重去噪，仅减去条件均值，额外方差吸收到扩散项中）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 校准阶段
{calib_data} = collect_calibration(model_fp)  # 收集 M×T 组校准样本
model_q = BRECQ(model_fp, q_params, calib_data)  # AdaRound 量化
{(ε̂, Δε)^i} = collect_quant_error(model_fp, model_q)  # 收集 S×T 组量化误差对
μ[T×2], Σ[T×4] = gaussian_modeling({(ε̂, Δε)^i})  # 估计每时间步的联合高斯参数

# 推理阶段（双重去噪循环）
for t in T, ..., 1:
    ε̂_θ = model_q(x_t)  # 量化噪声估计网络前向
    μ_cond, σ²_cond = TSQNM(ε̂_θ, μ[t], Σ[t])  # 条件化量化噪声预测
    
    if S-D²:  # 随机双重去噪
        z ~ N(0, I)
        Δε' = μ_cond + sqrt(σ²_cond) * z
        ε' = ε̂_θ - Δε'  # 减去完整估计噪声，恢复分布
        # 用标准 SDE 采样
        
    if D-D²:  # 确定性双重去噪
        ε' = ε̂_θ - μ_cond  # 仅减去条件均值
        g'_eff = sqrt(g²(t) - g⁴(t)*σ²_cond/σ²_t)  # 额外方差吸收到扩散项
        # 用调整后的扩散系数采样
    
    x_{t-1} = DDIM_update(x_t, ε', α, Σ)  # 执行逆扩散更新
```

关键张量计算（LDM-4 W4A8, ImageNet 256×256, scale=3.0, η=0.0, steps=20）：
- 量化输出 ε̂_θ ∈ R^{4×64×64}（latent 空间维度）
- 假设各向同性+元素不相关 → σ²_ε̂, σ²_Δ, σ_ε̂Δ 均为标量（大大简化联合分布参数估计）
- 条件均值 μ_{Δε|ε̂} 和条件方差 σ²_{Δε|ε̂} 对所有元素使用相同的标量修正

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/TaylorJocelyn/D2-DPM。基于 LDM（latent-diffusion）和 BRECQ 框架构建。适用于 LDM-4/LDM-8 等 U-Net 架构的扩散模型，支持 W8A8 和 W4A8 配置。关键结果：W4A8 D-D² 在 ImageNet 256×256 上 FID=9.71（FP=11.13），即量化模型 FID 比全精度模型低 1.42。局限性：需要为每个时间步存储联合分布参数（μ[T×2], Σ[T×4]），需要校准阶段额外前向传播来收集量化噪声统计。

涉及论文标题：
- D2-DPM Dual Denoising for Quantized Diffusion Probabilistic Models
