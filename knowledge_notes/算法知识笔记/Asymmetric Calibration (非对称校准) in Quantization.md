## Asymmetric Calibration (非对称校准) in Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
非对称校准（Asymmetric Calibration）是 GPTAQ 提出的 PTQ 校准范式改进。传统 GPTQ 使用**对称校准（Symmetric Calibration）**：每层独立最小化 `||(w+Δw)X − wX||²`，其中 X 是前一层量化后的输出——即假设"当前层的输入已经是正确的"，仅优化当前层的局部量化误差。非对称校准将该目标改为 `||(w+Δw)X − wX̃||²`，其中 X̃ 是前一层全精度模型输出的激活（"ground truth" reference），X 是前一层量化后的实际输出。两者的差异 ΔX = X̃ − X 来自前层权重和激活量化的累积误差，沿网络深度逐渐放大（GPTAQ Fig.2a 验证）。GPTAQ 通过引入残差 r = wX̃ − wX = wΔX，在 Lagrangian 约束优化框架中推导出包含两项的最优权重更新：量化误差补偿项（与 GPTQ 相同）+ 残留误差补偿项 `r X^T H_{-q}^{-1}`。实验验证（Table 5）：仅用第二项（残留补偿）的 GPTAQ' 就能在零样本准确率上超越 GPTQ（69.0% vs 67.1%），两项联合的完整 GPTAQ 最优（69.6%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
非对称校准的核心差异：

```
# 对称校准（GPTQ）：target = wX（量化后输入的输出）
min ||(w+Δw)X - wX||²              # wX = 当前层的"局部正确"输出
# → δw = -(ŵ_q - w_q)/H_{qq}^{-1} · H_{q,:}^{-1}

# 非对称校准（GPTAQ）：target = wX̃（全精度输入的输出）
min ||(w+Δw)X - wX̃||²              # wX̃ = 全精度模型的"全局正确"输出
# 引入 r = wX̃ - wX = wΔX（输入偏差在输出空间的投影）
# Lagrangian: L = ||Δw X - r||² + λ(e_q Δw^T + w_q - ŵ_q)
# → δw = -(ŵ_q - w_q)/H_{qq}^{-1} · H_{q,:}^{-1} + r X^T H_{-q}^{-1}
#        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^
#        量化误差补偿项 (GPTQ term)                  残留误差补偿项 (new)
```

**Annotations**: X 是量化后激活（actual），X̃ 是全精度激活（reference）。r = wΔX 对应当前层权重的全精度输出与量化后输入的输出的偏差——即使权重未量化，仅因输入偏差就会产生输出偏差。第二项 `r X^T H_{-q}^{-1}` 将这一偏差通过 Hessian 逆矩阵回传到剩余未量化权重中。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GPTAQ 的非对称校准通过 4 个优化步骤高效实现：(1) **任意顺序处理**——放弃每列选择最优 q 的贪心策略，改为从左到右固定顺序，支持所有输出通道并行；(2) **残差分解**——R = Σ_{q=1}^n W_{:,q} ΔX_{q,:}，将 R 分解为 n 个独立神经元分量，预计算一次 ΔX 后消除重复的 R 评估；(3) **Cholesky 重构化 + Theorem 4.2**——P = ((ΔX X^T L) ⊙ M_U) L^T，将 P 矩阵计算融合为一行 GPU 友好代码；(4) **Lazy-Batch 更新**——block 后批量更新 block 外列。使用时需注意：激活量化应在权重量化之前执行（A→W 顺序），使 ΔX 包含激活量化信息；X̃ 临时存储通过逐 block 进出 GPU 优化（Algorithm 2），LLaMA2-7B 约 12GB 临时内存。GPTAQ 实现仅比 GPTQ 多约 20 行代码（主要在 P 矩阵计算和 lazy-batch 中的第二项），额外延迟大维度时 30-40%、小维度时 <10%。

涉及论文标题：
- GPTAQ: Efficient Finetuning-Free Quantization with Asymmetric Calibration
