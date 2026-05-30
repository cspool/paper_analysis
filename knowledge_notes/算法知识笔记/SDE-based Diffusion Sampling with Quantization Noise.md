## SDE-based Diffusion Sampling with Quantization Noise

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
这是 D²-DPM 提出的将量化噪声纳入 SDE 框架进行理论分析的视角。核心推导：当量化噪声 Δε_θ ~ N(μ_Δ, σ²_Δ I) 被引入噪声估计网络后，反向 SDE 采样方程变形为 d𝐱 = [𝐟(𝐱,t) + g(t)²(ε_θ + μ_Δ)/σ_t]dt + [g(t) + g(t)²σ_Δ√(dt)/σ_t]dw̄。量化噪声的 μ_Δ 叠加到 drift term（影响采样方向），σ_Δ 增大 diffusion coefficient（影响采样波动和收敛性）。基于此分解，D²-DPM 设计了分离式修正：通过条件均值修正 drift term，通过条件方差修正/吸收到 diffusion term 中。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
量化扩散模型的 SDE 采样方程分解：

**原始全精度 SDE 采样（Anderson 1982）**:
$$d\mathbf{x} = \left[ \mathbf{f}(\mathbf{x}, t) - g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) \right] dt + g(t) d\bar{\mathbf{w}}$$

**量化后 SDE 采样（含量化噪声）**:
$$d\mathbf{x} = \left[\mathbf{f}(\mathbf{x}, t) + g(t)^{2} \frac{\boldsymbol{\epsilon}_{\theta} + \boldsymbol{\mu}_{\Delta}}{\sigma_{t}}\right] dt + \left[g(t) + \frac{g(t)^{2} \sigma_{\Delta} \sqrt{dt}}{\sigma_{t}}\right] d\bar{\mathbf{w}}$$

关键分解：
- Drift term 修正：μ_Δ → 用条件均值 μ_{Δε|ε̂} 从 ε̂_θ 中减去
- Diffusion term 修正：σ_Δ → S-D² 通过减去完整估计噪声恢复；D-D² 通过调整有效 g(t) 吸收

D-D² 的有效扩散系数调整：
$$g'(t) = \sqrt{g^{2}(t) - \frac{g^{4}(t)\sigma_{\Delta}^{2}(t)}{\sigma_{t}^{2}}}$$

当 ODE 采样（η=0, g(t)→0）时，D-D² 的额外标准差实际上将 ODE 转换为了隐式的 Langevin SDE，产生更好的误差缓冲。这解释了 D-D² 在无随机性容量时反而优于 S-D² 的实验现象。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
该理论分析主要用于：(1) 理解量化噪声如何在采样过程中传播和积累；(2) 指导设计分离式修正策略（分别修正均值和方差）；(3) 判断采样器（ODE vs SDE）对额外方差的吸收能力。实际实现时不需要显式求解 SDE——D²-DPM 在 DDIM 采样迭代中嵌入修正步骤：先通过 TSQNM 预测量化噪声的参数，再对输出做去噪修正，最后用修正后的 ε' 执行标准 DDIM 更新。其直接对应关系为：DDIM 是 ODE 求解器（η=0），DDPM 是 SDE 求解器（η=1），η 控制随机性容量。

涉及论文标题：
- D2-DPM Dual Denoising for Quantized Diffusion Probabilistic Models

---
