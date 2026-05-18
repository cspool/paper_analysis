## GyRot PE (Integer Tensor Processing Element / 整数张量处理单元)

术语是什么？

GyRot PE是GyRot accelerator的基本计算单元，执行32-way INT4 dot product + fully integer dequantization。每个PE采用3D tensor组织（8×8×32），区别于传统2D systolic PE。PE支持INT4 weight × INT4 activation的32路并行点积，内嵌INT8 dequantization pipeline，输出32-bit整数累加器。

从kernel调度角度拆解术语：

```
// GyRot PE microarchitecture (Fig. 7b)
// 配置: 32-way INT4 dot product, minimum group size G=32

每个cycle的PE操作流程:
  ⃝1 32-way INT4 dot product:
    // 从input buffer读32× INT4 activation (X0~X31)
    // 从weight buffer读32× INT4 weight (W0~W31)
    // 32个4b×4b乘法 + adder tree → 13-bit partial sum

  ⃝2 Dequantization stage (pipelined):
    // SX (INT8) × partial_sum (INT13) → Multiply unit
    // ZX (INT8) × WSUM (INT13) → Multiply unit (parallel)
    // scaled - ZX×WSUM → Subtract unit
    // result × SW (INT8) → Multiply unit

  ⃝3 Integer accumulation:
    // 32-bit accumulator per PE
    // 跨group累加 (intra-group通过dot product, inter-group通过accumulator)
    // Output时转FP16写buffer

// PE array: 8×8×32 = 2048 parallel ops/cycle
// 8×8 systolic array, 每PE 32-way dot product
// output-stationary dataflow: partial sums stay in PE accumulator
```

GyRot PE与GPU上group quantization kernel的关键区别：
- GPU: Tensor Cores执行INT4 GEMM → CUDA cores做FP dequantization (INT→FP convert + FP scale/bias + FP accumulate) → mixed-precision path
- GyRot PE: 单一PE内integer domain完成全部计算 → fused INT datapath

术语一般如何实现？如何使用？

- RTL实现：SystemVerilog，Samsung 28nm工艺，Synopsys Design Compiler综合，1GHz目标频率
- PE Array: 8×8×32 tensor organization → output-stationary systolic dataflow
- 相比baseline PE：Tender (8-bit systolic, no group quant), MANT (G=64, FP16 SF), LightRot (G=128, FP16 SF+ZP)
- 面积/功耗：GyRot-INT PE相对Tender面积减65.2%、能耗减69.2%
- WSUM broadcast：WSUM unit用8×32-way adder-tree预计算per-group weight sum，整行8个PE共享，减少per-PE重复计算

涉及论文标题：
- GyRot: Leveraging Hidden Synergy between Rotation and Fine-grained Group Quantization for Low-bit LLM Inference

