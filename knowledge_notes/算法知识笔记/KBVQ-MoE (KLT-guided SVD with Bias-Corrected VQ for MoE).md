## KBVQ-MoE (KLT-guided SVD with Bias-Corrected VQ for MoE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KBVQ-MoE 是第一个专为 Mixture-of-Experts (MoE) 架构设计的向量量化（VQ）框架，由 Houmo AI 提出（ICLR 2026 Poster, arXiv:2602.11184）。框架集成两个创新模块：(1) **IDRE（Input-driven Redundancy Elimination）**：KLT 引导的 SVD 分解，将 expert 权重投影到输入相干空间，提取主导共享分量保留全精度，仅对 expert-specific 残差做 VQ；(2) **BCOS（Bias-Corrected Output Stabilization）**：对 VQ 量化的 expert-specific 输出做 channel-wise affine compensation（scale + bias），使每个 channel 的 mean/variance 与 FP16 对齐。总压缩率公式：`ratio = (16(m+ln)min(m,l)k + m·l·b·n + 2^(bv+4)·v·n + 32ln) / (16nml)`，其中 m×l 为权重维度，n 为 expert 数，k 为 SVD 截断秩比例，v 为 VQ 子向量长度，b 为位宽。典型配置（k=1/128, v=4, b=2）下有效位宽 ~2.08 bits，压缩率 87%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
KBVQ-MoE 完整 pipeline：
```
=== 离线校准阶段 ===
输入: expert weights {W^(i)}, 校准数据 X, 截断秩 k, VQ 子向量长度 d

# IDRE: KLT-guided SVD 冗余消除
1. C_X = X^T X / (B-1)                        # 输入协方差
2. C_X = U_KLT Λ_KLT U_KLT^T                   # KLT 特征分解
3. U_X = U_KLT Λ_KLT^{1/2}                      # 输入相干基
4. for i in 1..n: W̃^(i) = W^(i) U_X             # 投影到相干空间
5. W̄ = [W̃^(1); ...; W̃^(n)]                      # (n·oc)×ic 统一表示
6. W̄ = (U Σ V^T)^T                            # SVD 分解
7. 选 top-k: U_k = U_{:,1:k}, V_k = V_{:,1:k}
8. 划分 V_k 按 expert: V_k = [Σ_k V_k^(1); ...; Σ_k V_k^(n)]
9. for i in 1..n:
     U_share = U_X^{-1} U_k                     # ic×k 共享映射
     W_share^(i) = (U_share (V_k^(i))^T)^T      # 共享分量 (FP16)
     W_quant^(i) = W^(i) - W_share^(i)          # expert-specific 残差

# VQ 量化 (仅对 W_quant)
10. for i in 1..n:
      将 W_quant^(i) 划分为 d 维子向量 {z}
      K-means++ 初始化 codebook C = {c_1,...,c_K}
      训练 codebook via K-means (100 iters)
      for each z: q = argmin_j ||z - c_j||^2; z_q = c_q
      → W_quant,VQ^(i)

# BCOS: Bias 校正
11. for each expert i:
      ŷ = (W_share^(i) + W_quant,VQ^(i)) x      # 量化输出
      从 calibration 估计 μ_y, σ_y (原始), μ_ŷ, σ_ŷ (量化)
      s_j = σ_{y_j} / σ_{ŷ_j} - 1              # per-channel scale
      b_j = μ_{y_j} - (1+s_j) μ_{ŷ_j}          # per-channel bias

=== 推理阶段 ===
12. y_corr = (1+s) ⊙ ((W_share + W_quant,VQ) x) + b
13. W_share 以 FP16 计算, W_quant,VQ 通过 index→codebook 查表解码
```

关键超参数：k = ic/128（SVD 截断秩，经验最优，average bit-width 增加约 0.08）；d = 4（VQ 子向量长度）；K-means 100 iterations + K-means++ 初始化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
KBVQ-MoE 针对 MoE LLM 的 decoder-only 架构设计。校准数据：从 RedPajama 采样 256 条（seq len=4096, seed=42）。适用模型：Qwen1.5-MoE-A2.7B, Qwen3-30B-A3B, Mixtral-8x7B, DeepseekV2-Lite。已知限制：(1) SVD 截断秩 k 需经验选择，无自适应机制；(2) 仅验证 decoder-only MoE，未测试 encoder-decoder 或多模态 MoE；(3) 未测试 1-bit 极端量化。开源：arXiv:2602.11184, ICLR 2026 Poster，论文未注明代码仓库。推理加速：Qwen1.5-MoE-A2.7B 2-bit decode speed 35.24 tok/s vs BF16 22.31 tok/s (1.58× speedup)，BCOS overhead <0.1% FLOPs。评测工具：LM-Evaluation-Harness (v0.4.0)。

涉及论文标题：
- KBVQ-MoE KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models

---
