## Vector Interval / Vector Interval Bottleneck

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Vector Interval（向量间隔）是FlashAttention-T (PPoPP'26) 命名并量化的GPU fused attention kernel中的性能瓶颈。在fused attention kernel的一个iteration中，warpgroup先执行QK^T GEMM（Tensor Core MMA指令），随后执行softmax（vector unit / CUDA Core），最后执行PV GEMM（Tensor Core MMA）。在softmax执行期间，由于当前实现中softmax完全依赖vector unit（CUDA Core），高吞吐的tensor unit（Tensor Core）处于idle状态，等待vector unit完成softmax后才能继续执行PV GEMM——这段tensor unit空转等待的时间窗口称为vector interval。Vector Interval Ratio定义为t_vec/t_iter，其中t_vec是vector interval的周期数，t_iter是整个iteration的周期数。

关键量化数据（FlashAttention-T Table 1, h=128, s=4096）：
- FlashAttention-2 on A100 (FP16-FP32): t_vec=924 cycles, t_iter=3100 cycles, ratio=29.8%
- FlashAttention-3 on H100 (FP8-FP32): t_vec=1126 cycles, t_iter=3106 cycles, ratio=36.3%

显示问题随硬件升级恶化——H100的FP8 GEMM吞吐是A100 FP16的4×（~989 vs ~312 TFLOPS），但vector unit吞吐提升有限，导致c/k比（tensor/vector throughput ratio）增大。Head dim=64（如gpt-oss）时ratio可达42%。

从kernel调度角度拆解术语：

Vector interval的成因与GPU fused attention kernel的调度结构直接相关。以FlashAttention-2的sequential scheduling为例（Ampere, per warpgroup）：
```
// Iteration j 的timeline（图3a）:
// |<------------------------- t_iter = 3100 cycles ------------------------->|
// |<-- QK^T GEMM -->|<---- softmax (t_vec = 924 cycles) ---->|<-- PV GEMM -->|
// Tensor Unit: BUSY |              IDLE (vector interval)    |    BUSY      |
// Vector Unit: IDLE |              BUSY (softmax primitives) |    IDLE      |
```
Softmax原语包括：② rowmax (vector REDUX)，③ FMA (exp rescale with new max)，④ mul (scale old accumulators)，② add (accumulate row sums)，⑤ exp (MUFU.EX2)。这些全部在vector unit执行（有效吞吐约16 elements/cycle via FMA+FADD instruction pairing）。

FlashAttention-3的pipelined scheduling（图3b）通过异步WGMMA实现了部分overlap——warpgroup 1的vector softmax与warpgroup 2的WGMMA GEMM并行。但仍有t_vec=1126 cycles的non-overlapped softmax部分，因为exp和rowmax等操作无法被WGMMA完全覆盖。

FlashAttention-T解决vector interval的核心思路：**将softmax中可tensorize的操作（scaling, FMA, row-sum reduction）offload到tensor unit，仅保留不可tensorize的操作（exp, rowmax）在vector unit，然后通过ILP（Ampere）或TLP（Hopper）并行执行tensorized和vectorized部分**。结果表明：
- ILP on Ampere: vector interval ratio 1.17-2.18× lower than baseline
- TLP on Hopper: vector interval ratio reduced to 2.7%

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Vector interval的测量方法：(1) 使用NVIDIA cycle-counting routines（clock64()指令）在kernel关键路径前后插入cycle counter采样；(2) 在Ampere ILP调度中，由于tensor和vector指令交错，直接测量困难——FlashAttention-T使用公式估计：t'_vec = t'_softmax - (t_vec - t'_softmax)，其中t'_softmax是FA-T的softmax时间，t_vec是baseline的vector interval时间，差值(t_vec - t'_softmax)代表被tensor unit利用的cycles；(3) 在Hopper TLP调度中，t'_vec可直接测量因为tensorized row-sum和vectorized rescaling在不同warpgroup上独立执行。

Vector interval概念不仅适用于attention kernel，也可推广到任何存在异构执行单元（tensor unit + vector unit）且workload被耦合调度限制的场景，如FFN中的activation function（GELU/SiLU在vector unit但linear projection在tensor unit）。

涉及论文标题：
- FlashAttention-T: Towards Fully Tensorized Attention by Exploiting Tensor-Vector Parallelism
- FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision
