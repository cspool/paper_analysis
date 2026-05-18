## Fully Integer Dequantization Datapath（全整数解量化数据路径）

术语是什么？

Fully Integer Dequantization是GyRot PE中使用的全整数解量化数据路径。传统group quantization的dequantization在浮点域执行（INT GEMM result → FP convert → FP scale × partial_sum → FP bias → FP accumulate），GyRot通过CoRFiG+HAP+重公式化非对称量化使scale factor (SX/SW)和zero-point (ZX)可用INT8精度表示，从而在PE内部以全整数完成dequantization——消除type conversion和FP arithmetic overhead。

从kernel调度角度拆解术语：

GyRot PE内fully integer dequantization的计算流程：

```
// 每个group (G=32) 的计算
Input:  X_Q[0:31]  // 32× INT4 activation
        W_Q[0:31]  // 32× INT4 weight
        SX         // INT8 activation scale factor
        ZX         // INT8 activation zero-point
        SW         // INT8 weight scale factor
        WSUM       // INT13 precomputed: Σ_{i=0}^{31} W_Q[i]

// Stage 1: INT4 dot product (1 cycle)
partial_sum = Σ_{i=0}^{31} X_Q[i] * W_Q[i]  // → 13-bit signed

// Stage 2: Integer dequantization (pipelined, 3 cycles)
// Step 2a: Multiply SX (INT8 × INT13 → INT21)
scaled = SX * partial_sum

// Step 2b: Multiply ZX × WSUM and subtract (INT8 × INT13 → INT21)
bias = ZX * WSUM
debiased = scaled - bias

// Step 2c: Multiply SW (INT8 × INT21 → INT29)
result = SW * debiased

// Stage 3: 32-bit integer accumulation (across groups)
accumulator += result  // 32-bit int, no FP conversion

// 最后: 32-bit int → FP16 (仅output writeback时转换一次)
output = fp16(accumulator)

// 对比传统FP dequantization:
// partial_sum → fp16_convert(partial_sum) → fp16_mul(SX_fp16) → 
// fp16_add(ZX_fp16 * WSUM_fp16) → fp16_mul(SW_fp16) → fp32_accumulate
// 每group需: 1× int→fp, 2× fp mul, 1× fp add, 1× fp accumulate
// GyRot: 0× type convert (all int), 2× int mul, 1× int sub, 1× int accumulate
```

术语一般如何实现？如何使用？

实现要素：(1) WSUM预计算：per-group weight sum Σŵ_i在weight加载时计算一次，存储在weight buffer的metadata bank，broadcast给整行PE共享——避免per-PE重复计算。(2) Multiplier设计：SX乘法器(INT8×INT13)、ZX×WSUM乘法器(INT8×INT13)、SW乘法器(INT8×INT21)均为定点整数乘法器，面积和功耗显著低于FP16乘法器。(3) 与重公式化非对称量化的配合：传统公式的dequantization顺序为SW·SX·(partial_sum − ZX·WSUM)，GyRot重公式化后变为SW·(SX·partial_sum − ZX·WSUM)——SX先乘partial_sum获得更大中间值范围，减少后续ZX减法引入的精度损失。(4) 硬件开销：GyRot-INT的dequantization+accumulation占PE area的4.2%、power的16.0%——远低于MANT（FP16 SF, G=64）和LightRot（FP16 SF+ZP, G=128）的FP dequantization unit。

涉及论文标题：
- GyRot: Leveraging Hidden Synergy between Rotation and Fine-grained Group Quantization for Low-bit LLM Inference

