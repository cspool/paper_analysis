## "1 more, 5 less" Resource Utilization Pattern (一多五少资源利用模式)

术语是什么？

"1 more, 5 less"（一多五少）是μShare对GPU kernel执行过程中intra-SM硬件资源利用模式的实证描述：在SM的6种主要硬件资源（FP32 cores、FP64 cores、INT32 cores、LD/ST units、SFU units、Tensor cores）中，单个kernel通常大量使用其中1种（"1 more"，avg 30.19% utilization），而其余5种的使用率极低（"5 less"，avg 5.07% utilization）。这一模式源于每个CUDA kernel的计算特性高度专一（如matrix multiplication主要用Tensor cores、layer normalization主要用LDST、roll kernel主要用INT32），导致kernel独占SM时仅一种硬件单元饱满工作。当发生stacked co-location（同kernel多个block在同一SM）时，SM内的"1 more, 5 less"模式被放大，整体low-level hardware utilization仅9.28%（Nsight Compute），远低于NVIDIA-SMI报告的81.16%。

从硬件架构角度拆解术语：

A40 SM的6种硬件资源及其在"1 more, 5 less"模式中的角色：

```
SM硬件组成 (NVIDIA A40, Ampere architecture):
  FP32 cores:     64 units  // 通用浮点计算，非GEMM的FP32运算
  FP64 cores:     32 units  // 双精度浮点，科学计算/inference极少用
  INT32 cores:    64 units  // 整数运算（地址计算、循环索引、roll kernel的位移运算）
  Tensor cores:   4 units   // 矩阵乘法加速器（FP16/BF16/TF32/INT8/INT4）
  SFU units:      16 units  // 特殊函数单元（sin/cos/exp/sqrt等超越函数）
  LD/ST units:    32 units  // 加载/存储单元（global/shared memory access）

// 6802次kernel执行的分析（top 20 most frequent kernels, 6063 executions）：
// Kernel: CUTLASS Gemm (1293次, 223538μs)
//   "1 more": Tensor core = 80.49%
//   "5 less": FP32=5.93%, FP64=0%, INT32=11.44%, LDST=17.51%, SFU=0%
//   avg of "5 less": (5.93+0+11.44+17.51+0)/5 = 6.98%
//
// Kernel: CUDNN NHWC (116次, 9560μs)  
//   "1 more": LDST = 55.10%
//   "5 less": FP32=7.59%, FP64=0%, INT32=14.53%, SFU=0%, Tensor=0%
//   avg of "5 less": (7.59+0+14.53+0+0)/5 = 4.42%
//
// Kernel: Layer Norm (128次, 27497μs)
//   "1 more": LDST = 58.02%
//   "5 less": FP32=13.43%, FP64=0%, INT32=33.08%, SFU=11.03%, Tensor=0%
//   avg of "5 less": (13.43+0+33.08+11.03+0)/5 = 11.51%
```

"NVIDIA-SMI vs Nsight Compute discrepancy": 6802次执行的对比显示NVIDIA-SMI报告81.16% utilization，而Nsight Compute报告仅9.28% low-level hardware utilization。原因：NVIDIA-SMI使用"active time ratio"——只要SM中至少1个thread active就计为100%，极大夸大了实际硬件利用。即使在stacked co-location下SM内只有1种资源在使用，NVIDIA-SMI仍报告高利用率。

术语一般如何实现？如何使用？

"1 more, 5 less"是μShare通过实证发现的现象，用于驱动scattered co-location的设计决策：
1. **Profiling工具**：NVIDIA Nsight Compute CLI测量per-kernel的6种hardware unit active cycle ratio
2. **配对选择**：基于每kernel的dominant resource（"1 more"），选择dominant resource互补的kernel进行co-location。实验证实：dominant resource不同时half-plus提升throughput 19.94%（前19对），相同时则下降10.37%（后4对）
3. **Generalizability**：论文指出这一模式不限于inference workload，scientific computing workload（主要用FP64 cores）与inference（主要用FP32/LDST/Tensor cores）的co-location同样受益（系统throughput提升18.18%-28.62%）

涉及论文标题：
- μShare: Non-Intrusive Kernel Co-Locating on NVIDIA GPUs

