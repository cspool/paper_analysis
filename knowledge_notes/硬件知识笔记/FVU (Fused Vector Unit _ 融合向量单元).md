## FVU (Fused Vector Unit / 融合向量单元)

术语是什么？

FVU是GyRot accelerator中集成的融合向量处理单元，负责在线执行非线性操作（SwiGLU、RMSNorm、embedding等）和Hadamard rotation。FVU将nonlinear/element-wise operation unit与FHT (Fast Hadamard Transform) rotation unit融合，使rotation紧接nonlinear层执行——消除CPU-GPU data movement。

从硬件架构角度拆解术语：

```
// FVU在GyRot accelerator中的工作流程:
Input: activation from global buffer (after previous linear layer output)

Stage 1: 加载activation + 执行nonlinear/element-wise操作
  // 从global memory读output activation
  // 执行: SwiGLU (gated activation), RMSNorm, embedding lookup等
  // 这些层不能将rotation fuse进weight (rotation-invariance不跨非线性层成立)

Stage 2: FHT rotation unit
  // 5-stage pipeline, 32-way parallel, 160 add/subtract units (32/stage)
  // 支持scalable rotation: 通过local register file + two-stage scheme
  //   最大32×32 = 1024维 (对应CoRFiG R=1024)
  // Partial gating: 支持sub-32 power-of-two sizes (2, 4, 8, 16, 32)
  //   → 不浪费energy on unused lanes
  //
  // FHT计算 (以32-dim为例):
  //   Stage 1: 16 parallel add/sub (w0±w16, w1±w17, ...)
  //   Stage 2: 16 parallel add/sub on stage1 results
  //   ...
  //   Stage 5: 16 parallel add/sub → output 32 rotated values
  //   Total: 5×32=160 add/subtract operations → O(n log n) vs O(n²) naive

Stage 3: 量化
  // Rotated activation → asymmetric group quantization (reformulated)
  // → 送入PE array for INT4 GEMM
```

FVU硬件参数：面积0.07 mm² (3.5% of total)，功耗6.78 mW (0.9% of total)。FHT unit：5-stage pipeline，每stage 32 parallel add/subtract units，总160 units。通过local register file + two-stage scheme支持scalable rotation up to 1024-dim。Partial gating机制支持power-of-two sub-32 sizes，按需关闭未使用的FHT lanes。

术语一般如何实现？如何使用？

- 为什么需要FVU：大多数rotation可offline fuse进weight（X·H·H^T·W → X·W'），但当非线性层介于linear层之间时（如SwiGLU activation、RMSNorm、embedding），rotation-invariance不成立，需要online rotation。FVU通过融合nonlinear op + FHT避免将数据搬回CPU做rotation。
- FHT vs naive matrix multiply：Hadamard矩阵乘法的naive实现是O(n²)，FHT利用Hadamard递归结构只需O(n log₂ n)次加法/减法。GyRot的FHT unit 160 add/subtract units支持5-stage 32-way处理，相比FP乘法器显著减少area/energy。
- HAP permutation无FVU开销：HAP所需的channel permutation是permutation-invariant操作（非线性层不变），可完全fuse进weight matrix的output channel重排（offline），无runtime cost。
- 面积和功耗：FVU仅占总面积的3.5%和总功耗的0.9%，是GyRot中overhead最小的模块。

涉及论文标题：
- GyRot: Leveraging Hidden Synergy between Rotation and Fine-grained Group Quantization for Low-bit LLM Inference

