## Compute-Communication Fused Kernel (Overlap Taxonomy)

术语是什么？
Compute-Communication Fused Kernel是将计算（如GEMM）和通信（如All-Scatter/All-Gather）融合到单个GPU kernel中执行的技术。Iris定义了两大类overlap策略：(1) Unfused——计算和通信在不同kernel中执行，通过CUDA stream并发或CU分区实现overlap；(2) Fused——计算和通信在同一个kernel中交织，通过workgroup specialization或sequential ordering实现。各策略在实现复杂度、资源利用和性能之间提供不同trade-off。

从kernel调度角度拆解术语：
四种overlap pattern对比（GEMM+All-Scatter）：

**Unfused Bulk-Synchronous**: 两个独立kernel顺序执行，中间global barrier。GEMM全完成后通信才开始——GPU资源交替闲置，存在execution bubble。

**Unfused Producer-Consumer**: 两个kernel在不同stream上并发。GEMM kernel使用256 CU，通信kernel使用48 CU。Producer通过atomic_cas(release)通知tile就绪，consumer通过atomic_cas(acquire) spin-lock等待。避免全局barrier但需手动CU分区。

**Fused Sequential**: 单kernel内GEMM tile产出后立即iris.store scatter。最简单的fused模式，仅需几行代码改动。但GEMM和通信在同一workgroup内顺序执行（先算后传），tail latency会增加。

**Fused Workgroup Specialization**: 单kernel内通过pid划分worker角色——256个GEMM workgroup做计算（完成后atomic_cas(release)发信号），48个COMM workgroup spin-lock等信号后iris.put。GEMM和通信在不同CU上并发执行，通信可完全隐藏在GEMM后面（尤其是小N大K场景，N/8后每个tile通信量极小）。代价：资源分配受GEMM（资源密集型）约束。

术语一般如何实现？如何使用？
开发者根据workload特性选择pattern：通信占比小选Fused Sequential；大K小N选Workgroup Specialization（最高效）；需避免worst-case resource allocation选Unfused Producer-Consumer。所有pattern通过Iris device-side API在Triton kernel中实现。

涉及论文标题：
- Iris: First-Class Multi-GPU Programming Experience in Triton
