## Group-GEMM（分组通用矩阵乘法）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Group-GEMM（Group General Matrix Multiply）是并行执行多个形状不同但独立的 GEMM 操作的 GPU 计算模式。在 MoE 模型中，每个 token 的路由机制选择 top-k 个 expert，不同 expert 收到的 token 数不同，因此 per-expert GEMM 的形状（m 维度 = token 数）不同。Group-GEMM 将所有这些 shape 不同的 GEMM 打包为单次 kernel launch 并行执行。与 Batched-GEMM（所有子问题形状完全相同）不同，Group-GEMM 处理的是异构 shape 子问题，需要更精细的 tile 分解和调度策略。NVIDIA CUTLASS 提供高效的 Group-GEMM 实现。

在 MxMoE 中，Group-GEMM 被扩展为支持混合精度：同一 kernel launch 内不同 expert 的 GEMM 可以使用不同的精度（如 W4A16, W8A8, W4A4），进一步增加了 tile shape 和计算模式的异构性。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MoE block 中 Group-GEMM 的计算结构：

```
MoE block 包含 E 个 expert，每个 expert 3 个 linear block (gate/up/down)

给定输入 X ∈ R^{T×d} (T tokens):
1. Gating: 每个 token 分配给 top-k expert
   → per-expert token 数 T_e，Σ T_e ≤ T×k

2. Group-GEMM 并行执行:
   for expert e in activated experts:
       X_e = gather(X, tokens_assigned_to_e)  // [T_e, d]
       # 3 个 GEMM 可进一步融合或分开发射
       gate_e = X_e × W_gate_e^T  // [T_e, d] × [d, d_inter]
       up_e   = X_e × W_up_e^T    // [T_e, d] × [d, d_inter]
       down_e = (SiLU(gate_e) ⊙ up_e) × W_down_e^T  // [T_e, d_inter] × [d_inter, d]
   
   # 所有 expert 的 GEMM 打包为 Group-GEMM 并行执行
   # 不同于顺序执行 (for-loop)，CUTLASS Group-GEMM
   # 在单 kernel 内将所有 tile 调度到 SM 上并行处理

3. Final output:
   F = Σ_e w_e · scatter(down_e)  // 加权聚合回 token 顺序
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CUTLASS 使用 `cutlass::gemm::kernel::GroupedGemm` 和 `cutlass::gemm::device::GroupedGemm` 模板类实现。MxMoE 在此基础上扩展：每种精度实现独立的 micro-kernel（CTA 级 CUDA device function），由 kernel generator 自动组合不同 micro-kernel 为统一 fused kernel，tile scheduler 按 greedy LPT 调度 tile 到 SM。

涉及论文标题：
- MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design

---
