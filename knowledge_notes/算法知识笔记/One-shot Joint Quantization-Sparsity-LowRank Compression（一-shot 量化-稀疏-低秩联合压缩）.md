## One-shot Joint Quantization-Sparsity-LowRank Compression（一-shot 量化-稀疏-低秩联合压缩）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
One-shot Joint Quantization-Sparsity-LowRank Compression 是 SLiM 框架提出的将量化、稀疏化和低秩近似三者整合为统一 one-shot pipeline 的压缩范式。与各自独立执行（如先剪枝再量化，误差独立累积）不同，SLiM 的三阶段 pipeline 将误差视为统一的可补偿信号：(1) SLiM-Quant 最小化初始量化误差 E_Q；(2) Wanda 在量化权重上施加稀疏，引入 E_S；(3) SLiM-LoRA 对总误差 E_Q+E_S 做显著性加权 SVD，通过低秩适配器闭式解补偿。关键洞察：总压缩误差的显著性（而非 Frobenius 范数）决定对模型输出的影响，因此低秩适配器应优先修正高显著性通道的误差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# SLiM 整体 pipeline（per layer）
# 阶段一: 量化
α* = SLiM-Quant(W, q=4)         # 概率化最优 scaling factor
W_Q = round(clip(W/α*)) × 2^3   # 4-bit symmetric uniform quant

# 阶段二: 稀疏化
X_norm = ||calibration_X||_2
W_C = Wanda_prune(W_Q, X_norm, sparsity=0.5, pattern="2:4")
E_C = W_C - W   # 总误差 = E_Q + E_S

# 阶段三: 低秩补偿
x = mean(calibration_X) + shift  # 显著性向量
S_C = diag(x) @ E_C              # 误差显著性 [d_in, d_out]
L_tilde, R = SVD(S_C, rank=0.1d) # 低秩近似
L = diag(1/x) @ L_tilde          # 逆显著性变换

# 可选阶段四: 适配器量化 + PEFT 微调
L_Q = AbsMax_group_quantize(L, group_size=128, bits=4)
R_Q = AbsMax_group_quantize(R, group_size=128, bits=4)
# 冻结 W_C, 仅微调 L_Q, R_Q (STE, AdaFactor, 300K tokens C4)

# 推理: Y = SparseMarlin(X, W_C) + X @ L_Q @ R_Q (或 X @ L @ R)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
全流程实现于 SLiM 开源库。压缩配置：4-bit 权重量化 + 50% 2:4 稀疏 + rank=0.1d 低秩适配器。内存缩减（含量化适配器）约 5×（0.19-0.20× dense），FLOP 缩减约 1.5×（低秩适配器引入少量额外计算）。校准数据：128 条 C4 序列。整个压缩过程对 LLaMA-2-7B 约需 39 分钟（单 H100 GPU）。压缩后可选 PEFT 微调（300K tokens, ~14h 单 GPU）进一步缩小与 dense 模型的精度差距。

涉及论文标题：
- SLiM One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression

---
