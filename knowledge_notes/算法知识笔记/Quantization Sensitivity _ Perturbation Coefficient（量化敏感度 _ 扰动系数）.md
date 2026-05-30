## Quantization Sensitivity / Perturbation Coefficient（量化敏感度 / 扰动系数）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Quantization Sensitivity（量化敏感度）在 MxMoE 中以 Perturbation Coefficient Δ_{i,j,k} 量化——对 MoE block 中第 i 个 expert 的第 j 个 linear block，使用量化方案 k 量化后，MoE block 输出与全精度输出的 Euclidean distance：Δ = ||Ô - O||₂。该度量直接反映了该 linear block + 该量化方案的组合对最终输出的扰动程度。Δ 越大，说明该 linear block 对该量化方案越敏感，需要更高精度。在校准集（128 条 WikiText2 序列）上统计 Δ 值。该 metric 假设量化输出扰动与最终 loss 扰动正相关（Choukroun et al. 2019），因此最小化中间输出扰动可有效保持最终模型精度。与 Hessian-based 方法（如 HAWQ）相比，基于输出距离的度量更简单高效，不需要计算 Hessian 矩阵。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
计算 Δ_{i,j,k} (Pseudocode):
  输入: MoE block (FP16), X_cal calibration data
  for each linear-block (i,j) in MoE block:
      for each scheme k in S:
          # 保存原始权重
          W_orig = W_{i,j}
          # 临时量化该 linear block
          W_q = GPTQ_quantize(W_{i,j}, scheme k)
          # 计算 MoE block 输出
          Ô = MoE_block_forward(X_cal,
              weights: replace W_{i,j} with W_q, others FP16)
          O = MoE_block_forward(X_cal, all FP16)
          # 扰动 = 输出欧氏距离
          Δ_{i,j,k} = ||Ô - O||₂
          # 恢复权重
          W_{i,j} = W_orig
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Δ 值在 ILP 问题中作为目标 L = Σ Δ_{i,j,k}·x_{i,j,k} 的系数，驱动 solver 将更低的 Δ 分配给更敏感的 block。校准数据量影响 Δ 估计的准确性——128 条 sequence × 4096 tokens 在实践中平衡了准确性和校准开销。MxMoE 指出可能存在跨层依赖导致的敏感度估计偏差（如 Qwen2-MoE 在 3.25-bit 下略逊于 GPTQ），建议使用跨层 loss 而非单层 loss 来改进。

涉及论文标题：
- MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design

---
