## Fused GemmAdd (融合的 GEMM 加法)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Fused GemmAdd 是 LongCat-Flash 训练基础设施中将 FP32 gradient accumulation 融合到 Grouped GEMM epilogue 中的 kernel 优化。在 MoE training backward pass 中，dw（weight gradient）计算通过 GEMM 完成，随后需要与 optimizer state 或 existing gradient 做 FP32 加法规约——这个加法步骤原本作为独立 kernel，产生额外的 HBM write-back 和 re-read，成为 bandwidth-bound 瓶颈。

Fused GemmAdd 将 FP32 addition 嵌入到 GEMM 的 epilogue 阶段（在 Tensor Core 输出数据还未写入 HBM 前、仍在寄存器/SMEM 中时完成加法），消除中间 write-back，并通过 tile GEMM pipeline 隐藏加法延迟。此外避免 BF16 数据写入 HBM 后重新读取时的精度损失。LongCat-Flash 在 fused GroupedGemmAdd benchmark 上取得 3.12x-3.86x 加速。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Fused GemmAdd 原理

// 传统流程 (2 kernels):
// Kernel 1: GEMM
C_bf16 = GEMM(A_bf16, B_bf16)  // BF16 GEMM 输出 [m, n] in HBM
// Kernel 2: Accumulate (独立 kernel, bandwidth-bound)
C_fp32 = convert_to_fp32(C_bf16)  // Load from HBM
C_fp32 += existing_gradient_fp32   // FP32 add (HBM → Register → HBM)

// Fused GemmAdd 流程 (1 kernel):
// GEMM + Addition fused in epilogue:
for each tile in tiles:
    // Tensor Core: C_tile_bf16 = A_tile_bf16 × B_tile_bf16
    C_tile_bf16 = tc_mma(A_tile_bf16, B_tile_bf16)

    // Epilogue (融合阶段):
    // 不写回 HBM，直接在寄存器/SMEM 中完成:
    C_tile_fp32 = bf16_to_fp32(C_tile_bf16)        // 精度提升
    existing_fp32 = load_from_hbm(existing_gradient, tile_offset)  // 加载已有梯度
    C_tile_fp32 += existing_fp32                     // FP32 加法
    store_to_hbm(C_tile_fp32, tile_offset)           // 写回 HBM

    // 下一 tile 的 GEMM 与当前 tile 的 epilogue 通过 double-buffer 重叠
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
1. Epilogue fusion：需要修改 GEMM kernel 的 epilogue 阶段。CUTLASS/Triton 均支持自定义 epilogue（如 bias addition, activation, quantization），GemmAdd 本质上是 "+ existing gradient" 的 epilogue 算子。
2. 精度保持：FP32 epilogue 避免了 BF16 → HBM 写回 → BF16 → FP32 转换的精度损失。MoE training 对梯度精度敏感（专家数量多、每个 expert 的 token 少 → per-expert gradient magnitude 小）。
3. 适用场景：(1) Grouped GEMM 的 epilogue（每个 expert 独立梯度累加）；(2) ScatterAdd 替代——某些场景下 fused GEMM+add 可替代 ScatterAdd 的梯度聚合。

涉及论文标题：
- LongCat-Flash Technical Report
