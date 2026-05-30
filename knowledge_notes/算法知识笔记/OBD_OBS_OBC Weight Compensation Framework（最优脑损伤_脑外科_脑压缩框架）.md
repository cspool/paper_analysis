## OBD/OBS/OBC Weight Compensation Framework（最优脑损伤/脑外科/脑压缩框架）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
OBD (Optimal Brain Damage, LeCun et al. 1990) → OBS (Optimal Brain Surgeon, Hassibi & Stork 1993) → OBC/OBQ (Optimal Brain Compression, Frantar & Alistarh 2022) → GPTQ (Frantar et al. 2022) 是一条从神经网络剪枝演化到 LLM 量化的理论链。核心思想：使用二阶 Taylor 展开 δE = g δw^T + ½ δw H δw^T 估计参数移除（剪枝或量化）对损失函数的影响，并通过 Hessian 矩阵信息将误差补偿到剩余参数中。

各阶段演变：
- **OBD**：假设模型已收敛（g≈0）、参数独立（仅 Hessian 对角）、损失近似二次。Saliency s_k = ½ h_{kk} w_k²。缺陷：忽略参数间交互。
- **OBS**：使用全 Hessian 矩阵。Lagrangian 约束优化得 δw = −w_q/[H^{-1}]_{qq} · [H^{-1}]_{q,:}，Saliency ΔE = ½ w_q²/[H^{-1}]_{qq}。缺陷：O(n³)，每次剪枝需重算 H^{-1}。
- **OBC**：限制到权重矩阵逐行优化（每行独立，H = 2XX^T）。扩展至量化：δw = −(w_q − ŵ_q)/[H^{-1}]_{qq} · [H^{-1}]_{q,:}。通过 Gaussian 消元高效更新 H^{-1}。
- **GPTQ**：OBC 的 LLM 适配，lazy batch 更新 + Cholesky 分解：δw = −(w_q − ŵ_q)/T_{qq} · T_{q,q:}。
- **GPTAQ**（本文）：将 OBC 的优化目标从对称校准 `||(w+Δw)X − wX||²` 扩展为非对称校准 `||(w+Δw)X − wX̃||²`（X̃ 是全精度输入）。这引入残差 r = wX̃ − wX，推导得 δw = −(ŵ_q − w_q)/H_{qq}^{-1} · H_{q,:}^{-1} + r X^T H_{-q}^{-1}（两分量：量化误差补偿 + 前层累积偏差补偿）。通过残差分解 R = Σ W_{:,q} ΔX_{q,:} 和 Theorem 4.2（P = ((ΔX X^T L) ⊙ M_U) L^T）实现高效 GPU 并行。
- **FOEM**（本文）：指出因累积补偿，一阶项不可忽略。保留 g 重新推导，通过 g ≈ β(W−𝕎)H 近似后 Hessian 自动消去，仅增加权重差分运算。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
OBS Lagrangian 求解推导（从 OBS 到 FOEM 的演变基础）：

```
# OBS 约束优化（剪枝参数 q 为零）
min_{δw} ½ δw H δw^T
s.t. e_q δw^T + w_q = 0
ℒ = ½ δw H δw^T + λ(e_q δw^T + w_q)
∂ℒ/∂δw = δw H + λ e_q = 0  →  δw = −λ e_q H^{-1}
∂ℒ/∂λ = e_q δw^T + w_q = 0  →  λ = w_q / [H^{-1}]_{qq}
∴ δw = −w_q/[H^{-1}]_{qq} · [H^{-1}]_{q,:}  （最优补偿）
   ΔE = ½ w_q²/[H^{-1}]_{qq}                （剪枝 saliency）

# OBC/OBQ: 量化版本（w_q → ŵ_q）
δw = −(w_q − ŵ_q)/[H^{-1}]_{qq} · [H^{-1}]_{q,:}

# GPTQ: Cholesky 分解 H^{-1} = LL^T, T = L^T
δw = −(w_q − ŵ_q)/T_{qq} · T_{q,q:}

# FOEM: 保留一阶项 g
ℒ = g δw^T + ½ δw H δw^T + λ(e_q δw^T + w_q − ŵ_q)
∂ℒ/∂δw = g + δw H + λ e_q = 0  →  δw = −(g + λ e_q) H^{-1}
∂ℒ/∂λ = e_q δw^T + w_q − ŵ_q = 0  →  λ = (ŵ_q − w_q − g H^{-1} e_q^T)/[H^{-1}]_{qq}
∴ δw = −(w_q − ŵ_q − g H^{-1} e_q^T)/[H^{-1}]_{qq} · [H^{-1}]_{q,:} − g H^{-1}
# 梯度近似: g ≈ β(W − 𝕎)H, 代入后 H/H^{-1} 消去
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
OBD/OBS 最初用于小网络剪枝（LeNet、AlexNet）。OBC/OBQ 和 GPTQ 扩展至 LLM 量化。GPTQ 的 AutoGPTQ（https://github.com/PanQiWei/AutoGPTQ）是最广泛使用的实现，FOEM 集成于 gptqmodel 库。实践中 Hessian 从少量校准数据（128 samples, seq_len=2048）计算：H = 2XX^T，加 λI 正则化防止奇异。Cholesky 分解后保留上三角 T 用于补偿。该框架直接用于 weight-only 量化（W4A16/W3A16, group_size=128），配合旋转技术（QuaRot、SpinQuant）可扩展至 weight-activation 量化（W4A4KV4）。GPTAQ 进一步扩展 OBC 为非对称校准：在校准中同时使用全精度前向的激活 X̃ 和量化后激活 X，计算 ΔX = X̃ − X，引入残留误差补偿项 `W_{:,q} P_{q,:}` 纠正前层累积的激活偏差。核心优势：仅需少量校准数据（无需训练数据），一次前向计算 Hessian 即可。

SpQR 将 OBS 的敏感度准则直接嵌入量化过程：敏感度 s_ij = (w_ij − quant(w_ij))² / (2[H⁻¹]_jj)，在 GPTQ 逐列量化过程中动态计算（而非预处理）。关键创新：(1) 敏感度捕获了权重间的相关性——某权重大rounding error可被其他未量化权重的连续值优化补偿；(2) outlier检测通过leave-one-out error对比（E_base − E_ol > τ）在量化过程中完成，outlier不仅是"初始敏感"权重，还包括能补偿其他权重量化误差的权重；(3) 调整后的权重值（而非原始值）被保留为16-bit，包含GPTQ误差补偿过程的累积效应。

涉及论文标题：
- First-Order Error Matters: Accurate Compensation for Quantized Large Language Models
- GPTAQ: Efficient Finetuning-Free Quantization with Asymmetric Calibration
- Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs
- SpQR A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression
