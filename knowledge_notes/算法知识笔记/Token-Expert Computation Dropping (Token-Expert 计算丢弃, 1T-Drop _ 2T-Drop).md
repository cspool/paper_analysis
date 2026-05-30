## Token-Expert Computation Dropping (Token-Expert 计算丢弃, 1T-Drop / 2T-Drop)

术语解释
MoE 推理加速策略：对每 token 在每层 MoE 中，按归一化 gating scores 选择性丢弃 token-expert FFN 计算。1T-Drop 用单一阈值丢低 score 的 token-expert 对；2T-Drop 引入双阈值 + expert reconstruction，对 major (高重要性)/minor (低重要性) sub-expert 采用不同丢弃策略，trade off accuracy 与 speedup。

术语是什么？
观察：不同 benchmark 的 gating score 分布高度一致（图 6b/c），为跨任务 threshold-based dropping 提供泛化基础。极低阈值（~0.05）的丢弃甚至略提升 accuracy（低 score experts 输出可能是噪声）。1T-Drop 归一化 gating scores 后丢弃低于阈值的计算。2T-Drop 先 expert partition + neuron reconstruction → 对 major sub-expert 使用低阈值 T²_major（保守保留），对 minor sub-expert 使用高阈值 T²_minor（激进丢弃）→ 中间档的 expert 仅计算 major half。

从算法pipeline角度拆解术语：
```
=== 1T-Drop ===
s_norm[j] = s_selected[j] / Σ_k s_selected[k]   # normalize
mask[j] = s_norm[j] >= T_drop^1
y = Σ_{j: mask[j]} s_selected[j] · FFN_{e_j}(x)

=== 2T-Drop with Reconstruction ===
# Offline: Importance_e[n] = Σ|Swish(x·W1^n) ⊙ (x·W3^n)|
# Sort → major_expert (top 50%), minor_expert (bottom 50%)

# Online:
For each activated expert e_j:
  if s_norm[j] < T_major^2: skip
  elif s_norm[j] < T_minor^2: compute FFN_major only
  else: compute FFN_full

# Threshold setting: T_major^2 = T_drop^1 - 0.01, T_minor^2 = T_drop^1 + 0.01
# Keeps similar drop rate but higher accuracy
```
22-27% drop rate → 1.17-1.23× MoE speedup, 1.07-1.12× end-to-end. Tensor-level dropping 适配 GPU grouped-GEMM，区别于 neuron-level sparsity 在低稀疏率下难转换为实际 speedup。

术语一般如何实现？如何使用？
- SGLang 框架实现：gating function 后添加 normalize+threshold+mask 逻辑；Triton grouped-GEMM kernel 集成 skip/major-only/full 变粒度模式
- Calibration: MMLU 做 neuron importance profiling；threshold sweep 确定 optimal value
- 部署：支持 single GPU, TP, EP；drop rate 直接 proportional 转换为 speedup
- 局限：GSM8K 对 drop rate 最敏感；drop rate-threshold 非线性；per-layer threshold 待探索

涉及论文标题：
- DualSparse-MoE: Coordinating Tensor/Neuron-Level Sparsity with Expert Partition and Reconstruction
