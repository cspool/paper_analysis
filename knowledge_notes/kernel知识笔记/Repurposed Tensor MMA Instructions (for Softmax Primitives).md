## Repurposed Tensor MMA Instructions (for Softmax Primitives)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Repurposed Tensor MMA Instructions是FlashAttention-T提出的核心技术：通过特殊的operand value assignment方法，将原本专用于GEMM（矩阵乘加D=AB+C）的tensor MMA指令（如HMMA.1688.F32.TF32 / HGMMA.64x8x8.F32.TF32）重新定向（repurpose）以执行软最大（softmax）计算中的关键原语——element-wise scaling、fused multiply-add（FMA）和row-sum reduction。这使得这些操作可以在tensor unit（Tensor Core）上执行，而非原本的vector unit（CUDA Core）。

核心理念：tensor MMA指令计算D(v,t) = Σ_k A(v,k)·B(k,t) + C(v,t)。通过精心设计fragment B的赋值（全部设为0/α/1组合），可以利用MMA指令的accumulation逻辑实现非GEMM操作：
- **Element-wise scaling (D=α·A)**：设置C=0，B包含α值的pattern，使D(v,t) = α·A(σ(v),t)（带一个permutation σ）
- **Fused multiply-add (D=α·A+C)**：同上B赋值 + 设置C fragment为offset值，利用tensor unit accumulator
- **Row-sum reduction**：设置A=D（待求和矩阵），B=全1 pattern，C=0，利用MMA内建的多线程累加来跨thread求和

从kernel调度角度拆解术语：

以Ampere HMMA.1688.F32.TF32为例（|A|=|D|=4 elements per thread），repurposed element-wise scaling的fragment级操作：

```
// 目标: 对input fragment A的4个元素做scaling α，输出D
// Fragment layout (HMMA.1688): A[0..3], B[0..3], C[0..3], D[0..3] per thread

// 1. 赋值fragment B以实现scaling（图5a, σ=(1 2), d_C(σ)=1）
B(0,t) = α;  B(1,t) = 0;   B(2,t) = 0;   B(3,t) = 0;
C(0..3,t) = 0;  // accumulator清零

// 2. 执行HMMA.1688.F32.TF32:
// D = A * B + C → D(v,t) = Σ_k A(σ(v),k) * B(k,t)
// 结果: D(0,t)=α·A(0,t), D(1,t)=α·A(2,t), D(2,t)=α·A(1,t), D(3,t)=α·A(3,t)
//      ↑ permutation σ = (1 2) swaps elements 1 and 2

// 3. 恢复non-permuted输出（1次swap per thread, Cayley distance = 1）
swap(D(1,t), D(2,t));  // → D = [α·A(0), α·A(1), α·A(2), α·A(3)]

// Row-sum reduction（图5b）:
// A(v,t) = D(v,t) [input matrix fragment], B全1 pattern, C=0
// D'(0,t) = Σ_{t∈κ_i} D(0,t) + D(2,t)   [sum of row 0 elements across quad-pair κ_i]
// D'(1,t) = Σ_{t∈κ_i} D(3,t) + D(1,t)
// intra-thread: s(0,t) = D'(0,t) + D'(2,t), s(1,t) = D'(1,t) + D'(3,t)
// → 与标准的intra+inter-thread summation等价，但消除了显式thread同步和all-reduce
```

关键设计要素：
- **Cayley distance minimization**：permutation σ引入swap overhead，通过求解constrained optimization problem（minimizing d_C(σ) subject to MMA mapping constraints）找到最优B赋值
- **Zero copy overhead**：repurposed MMA直接操作GEMM输出fragments（同一register space），不需要额外数据搬运
- **Algorithmic constraint**：scaling/FMA repurposing要求scaling factor α在tensor MMA指令的所有行上uniform（对应X-row tile，X=16 for HMMA, X=64 for HGMMA）
- **Architecture support**：支持所有现代NVIDIA GPU架构的同步HMMA（Ampere, Hopper），也支持Hopper异步WGMMA（B fragment必须在shared memory中）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Repurposed tensor MMA的实现需要：(1) 理解目标架构的MMA fragment layout（通过microbenchmarking或查阅PTX ISA文档获取μ mapping）；(2) 求解最优B赋值（可empirical或使用SAT/Z3 solver）；(3) 预先生成通用B fragments并复用（降低生成overhead和register bank conflict）。在FlashAttention-T中，Ampere ILP使用HMMA.1688.F32.TF32，Hopper TLP使用HGMMA.64x8x8.F32.TF32（仅row-sum reduction repurposing）。repurposed MMA的effective throughput与原始vector throughput相当（~16 elements/cycle on A100）——这也是FlashAttention-T采用tensor-vector parallelism（而非all-tensorized）的原因：当前硬件上repurposed MMA不提供额外吞吐，但允许tensor和vector并行执行来缩短总wall-clock time。当未来tensor unit吞吐继续提升（如Blackwell doubled FP16 throughput），repurposed MMA将提供更大的绝对加速。

涉及论文标题：
- FlashAttention-T: Towards Fully Tensorized Attention by Exploiting Tensor-Vector Parallelism
