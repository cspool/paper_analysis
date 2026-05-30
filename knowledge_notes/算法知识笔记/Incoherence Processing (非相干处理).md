## Incoherence Processing (非相干处理)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Incoherence Processing（非相干处理）是 Chee et al. (2023) 在 QuIP 中首次提出、QuIP# 进一步改进的一种 LLM 量化前预处理技术。其核心思想是：在量化之前，通过将权重矩阵 W 和代理 Hessian H 与随机正交矩阵做共轭变换（双侧乘法），使 W 和 H 变为 "μ-incoherent"——即矩阵所有元素的幅值高度集中、不存在离群值。形式化定义（Definition 2.1）：Hessian H ∈ R^{n×n} 是 μ-incoherent 的，若其特征分解 H = QΛQᵀ 满足 max|Q_ij| ≤ μ/√n；权重矩阵 W ∈ R^{m×n} 是 μ-incoherent 的，若 max|W_ij| ≤ μ‖W‖_F/√(mn)。非相干性使得量化时每个坐标的误差贡献均匀，任何单一方向不会被过度惩罚，从而在理论上保证量化误差有界（provably bounded error）。QuIP# 的 incoherence processing 使用 Randomized Hadamard Transform (RHT)：对 H 和 W 做 Ŵ ← Had(S_U · Had(S_V · W^T)^T), Ĥ ← Had(S_V · Had(S_V · H)^T)，其中 S_U, S_V 是随机 ±1 对角矩阵，Had 是正交 Hadamard 矩阵。变换保持代理损失不变（tr((UWV^T)(VHV^T)(VW^TU^T)) = tr(WHW^T)）。推理时，量化后的模型通过两次 Hadamard 变换撤销量化的预处理效果：y ← Had(S_U ⊙ decompress_multiply(Ŵ, C, Had(S_V ⊙ x)))。与启发式离群值抑制方法（AWQ 的 per-channel scaling、OmniQuant 的可学习变换）相比，incoherence processing 有严格的理论保证，且不增加 per-group scaling 的额外存储开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 QuIP# 的 incoherence processing 流程（Algorithm 3: IP-RHT）为例：
```
# 输入: W ∈ R^{m×n}, H ∈ R^{n×n}
# S_U ∼ U{±1}^m, S_V ∼ U{±1}^n  (随机符号向量)

# Step 1: 对 W 做双边 RHT 变换
W_tmp = diag(S_V) @ W^T         # m×n → n×m, 逐行符号翻转
W_tmp = Had(W_tmp)               # Fast Walsh-Hadamard Transform (O(m log n))
W_tmp = W_tmp^T                  # n×m → m×n
W_hat = Had(diag(S_U) @ W_tmp)^T # 逐行符号翻转 + FWHT → Ŵ

# Step 2: 对 H 做双边 RHT 变换
H_tmp = diag(S_V) @ H            # n×n
H_tmp = Had(H_tmp)^T
H_hat = Had(diag(S_V) @ H_tmp)^T # Ĥ

# Step 3: 输出 Ŵ, Ĥ, S_U, S_V
# 推理时撤销量化:
# y = Had(S_U ⊙ decompress_multiply(Ŵ_quantized, C, Had(S_V ⊙ x)))
```
QuIP# RHT vs QuIP Kronecker 的理论优势：Lemma 3.1 证明 RHT 实现 μ_H = √(2log(2n²/δ)) 和 μ_W = 2log(4mn/δ)，而 Kronecker 方法的 μ_W^{Kron} = A²log²(4Cmn/δ)²（log 依赖 vs log² 依赖）。时间复杂度 O(n log n) vs O(n√n)。消融实验验证：QuIP# 仅 RHT 替换 Kronecker 即显著降低困惑度（Llama 2 70B 2-bit: 4.58 vs 5.90 Wikitext2）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Incoherence processing 实现要点：(1) Hadamard 矩阵获取：使用 Neil Sloane 维护的已知 Hadamard 矩阵库（http://neilsloane.com/hadamard/）；(2) 非 2 的幂维度处理：分解 n = p×q（p 为最大 2 的幂，q 存在已知 Hadamard 矩阵），V = H_p ⊗ H_q，复杂度 O(q²p log p)；(3) 若无合适 Hadamard 分解，QuIP# 提供 RFFT（Randomized Fast Fourier Transform）替代方案——使用 FFT + 随机复数相位，仅需 n 为偶数，理论界类似，实践略逊于 RHT（2-7B: 8.30 vs 8.22 WikiText2）；(4) 开源实现：https://github.com/Cornell-RelaxML/quip-sharp，提供 CUDA 加速的 FWHT kernel。在更广泛的量化文献中，incoherence processing 的思想已被后续工作采纳：RoMeo（旋转混合精度量化）、GyRot（旋转+分组量化）、SpinQuant 等均使用正交旋转矩阵抑制离群值，只是旋转矩阵类型（Hadamard vs 随机正交 vs Cayley 参数化）和变换粒度（全局 vs block-wise）有所不同。QuaRot 的关键创新在于将 incoherence processing 从仅用于权重量化扩展到同时处理激活值量化：通过计算不变性定理，QuaRot 将离线 Hadamard 变换融入权重矩阵，使跨层激活值自动经过 incoherence processing（X→XQ），从而从根源上消除激活值离群值（图 1 直观对比），使得激活值的 4-bit 量化成为可能。这与 QuIP# 的关键区别：QuIP# 在推理时需为每个权重矩阵执行两次在线 Hadamard 变换（撤销量化预处理），而 QuaRot 将大部分 Hadamard 变换离线融入权重，仅保留每层 1.5 次在线变换（down-projection 和 out-projection 各一次 head Hadamard），大幅降低推理开销。

涉及论文标题：
- QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks
- QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs
- RoSTE: An Efficient Quantization-Aware Supervised Fine-Tuning Approach for Large Language Models

在 RoSTE 中，incoherence processing 的思想被应用于 QA-SFT 训练而非仅 PTQ 推理。RoSTE 将旋转矩阵 R_i 的选择整合为 bilevel optimization 的下层问题：选择 R_i 以最小化 weight-activation quantization error surrogate loss（12），同时上层通过 STE 优化量化权重。RoSTE 的关键洞察是：并非所有层都受益于旋转——某些层旋转后可能引入新的 outlier（如 Pythia 的末层）。因此采用自适应策略逐层在 I（无旋转）和 H（Walsh-Hadamard 旋转）间选择，仅当旋转降低该层量化误差时才应用。旋转分为 between-block (R1) 和 in-block (R2, R3, R4)，其中 R1/R2 可离线吸收到权重中，R3/R4 通过 fast Hadamard CUDA kernel 在线执行。

---
