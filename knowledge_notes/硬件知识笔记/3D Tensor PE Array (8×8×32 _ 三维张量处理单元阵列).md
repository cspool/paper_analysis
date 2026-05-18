## 3D Tensor PE Array (8×8×32 / 三维张量处理单元阵列)

术语是什么？

GyRot accelerator采用的3D tensor PE array组织：8×8 systolic PE grid，每个PE内部执行32-way INT4 dot product（第三维），总计2048 parallel operations/cycle。区别于传统2D systolic array（每PE 1-way或2-way），第三维通过per-PE内32路并行乘法器+adder tree实现。

从硬件架构角度拆解术语：

```
// GyRot PE array组织:
// 2D systolic: 8 rows × 8 columns of PEs
// 3D tensor:   each PE has 32-way dot product (third dimension)

Systolic dataflow (output-stationary):
  // 每个cycle:
  //   - Activation X从input buffer水平广播到整行8个PE
  //   - Weight W从weight buffer垂直广播到整列8个PE
  //   - 每个PE: 32 activation × 32 weight → 32-way dot product → 13-bit

Inter-group accumulation:
  // 每个PE维护32-bit accumulator
  // Group 0 (32 elements): PE执行dot product → dequantize → accumulate
  // Group 1 (32 elements): PE执行dot product → dequantize → accumulate到同一accumulator
  // ...所有groups处理完 → accumulator → FP16 → output buffer

对比2D systolic PE:
  // Tender/MANT/LightRot: 2D systolic array, 每PE 1-way或2-way dot product
  // GyRot: 3D tensor with 32-way per PE → 更高的compute density
  // 但引入了per-PE adder tree和wider datapath
```

硬件实现参数：PE Array (INT Tensor): 0.26 mm² (12.4% of total), 410.24 mW (55.4%)。PE Array (Dequant + Accum): 0.09 mm² (4.2%), 118.40 mW (16.0%)。Total PE Array: 0.35 mm² (16.6%), 528.64 mW (71.4%)——PE array占总面积小但占总功耗大（计算密集型）。

术语一般如何实现？如何使用？

- 设计选择：32-way per PE对应minimum group size G=32——每个group恰好填满一个PE的一次dot product。若group size>32需跨cycle处理。
- Output-stationary好处：partial sum留在PE accumulator中，减少data movement（vs weight-stationary需要移动partial sum）。
- 与FP dequantization accelerator的对比：MANT (G=64, 2D systolic, FP16 SF)和LightRot (G=128, 2D systolic, FP16 SF+ZP)的PE array为2D结构，dequantization需要在每个PE中嵌入FP乘法器。GyRot的3D tensor组织以更高per-PE density换取更少PE数，配合integer dequantization实现最高area/energy efficiency。
- 多bank memory：Input buffer (64KB data + 8KB metadata)和Weight buffer (64KB data + 4KB metadata)采用multi-bank结构，提供足够bandwidth喂给8×8 PE array (需per-cycle 8×32 INT4 activations + 8×32 INT4 weights + metadata)。

涉及论文标题：
- GyRot: Leveraging Hidden Synergy between Rotation and Fine-grained Group Quantization for Low-bit LLM Inference

