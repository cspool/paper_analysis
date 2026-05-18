## RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization

- baseline方法是什么？
  Baseline是已有channel-wise mixed precision quantization方法（MixQ、QuaRot、Atom），以及uniform INT4量化。全栈执行例子（以Qwen3-8B在RTX 4090上W4A4推理为例）：
  - 算法层：INT4 uniform quantization→逐round-to-nearest量化activation和weight到4-bit→scaling factor保存→INT4 Tensor Core GEMM→dequantization恢复FP16；或MixQ：per-channel max value检测outlier channels→outlier channel→INT8/FP16 precision、normal channel→INT4→channel-wise mixed precision GEMM沿reduction dimension分解。QuaRot：Hadamard rotation平滑channel outliers→uniform INT4量化→仅处理channel维度outlier。
  - 系统框架层：SGLang/vLLM serving框架加载quantized model→prefill/decode pipeline使用FP16 GEMM或量化kernel
  - 编译框架层：论文未明确说明（手工CUTLASS/Triton kernel实现，无编译框架自动mixed precision生成）
  - kernel调度层：NVIDIA Tensor Core mma指令，INT4/INT8精度。Channel-wise方法沿reduction dimension分解GEMM→normal channel和outlier channel分别dense计算→结果合并。QuaRot使用fused dequantization kernel（dequant融合入GEMM）。Atom使用group-wise fine-grained quantization（group size 128）→引入额外dequantization/scaling overhead。
  - 硬件架构层：NVIDIA RTX 4090 GPU。INT4 Tensor Core峰值吞吐8×FP16，INT8 Tensor Core 2×FP16。Channel-wise mixed precision沿reduction dimension分解不会产生sparse computation pattern，与Tensor Core指令兼容。
  Baseline缺陷：(1) Channel-wise方法仅处理channel维度outlier→4-bit下仍有大量token-wise outlier无法表示→量化误差大（MixQ Qwen3-8B perplexity 14.76 vs BF16 9.72）。(2) QuaRot仅旋转但无mixed precision→旋转后仍有token-wise residual outliers→perplexity 11.53（优于MixQ但仍劣于RoMeo 10.97）。(3) Atom使用group-wise fine-grained quantization提高准确率但计算开销大→Qwen3-32B上Atom仅51.59 average zero-shot accuracy（显著低于RoMeo 70.66），且kernel实现性能差（仅3.63× average kernel speedup vs RoMeo 4.68×）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出**RoMeo**：通过RTMPQ算法（Hadamard rotation + token-wise mixed precision）处理双维度outlier，配合permutation-free系统设计和异步并发执行实现与uniform quantization相当的计算效率。

  论文方法全栈执行例子（以Qwen3-8B在RTX 4090上W4A4 5%-outlier推理为例）：
  - 算法层（核心创新）：
    (1) **Hadamard Rotation**（解决baseline缺陷：channel-only处理无法消除token-wise outlier）：对activation右乘Hadamard矩阵H，利用H的正交性和+1/-1元素将channel-wise irregularity平滑并迁移到token维度。旋转后peak activation从1272降至58.5。后续token-wise mixed precision只需处理token维度outlier，消除双维度的复杂度。
    (2) **Token-wise Mixed Precision Quantization**（解决baseline缺陷：4-bit下residual token-wise outlier仍导致量化误差）：在旋转后的纯token-wise分布上做per-token max-based outlier detection→top-k选择5% outliers→outlier token用INT8（range=127 vs INT4 range=7，16×更大的表示范围）；normal token用INT4。同时weight矩阵也由于H^T预乘amplify non-uniformity而采用相同的mixed precision。四种精度组合的cross-precision乘法。
  - 系统框架层（核心创新）：
    (3) **Permutation-free Mixed Precision Computation**（解决baseline中不存在而token-wise特有的挑战：non-reduction dimension sparse computation无法利用Tensor Core）：预分配outlier buffer→整个activation矩阵统一量化为INT4→outlier token单独拷贝到outlier buffer量化为INT8→所有四种精度组合各自操作dense uniform-precision矩阵→每个thread block处理同种精度组合→无需permutation保留contiguous memory layout。Tolerate redundant computation（outlier token参与两次计算）来保证Tensor Core兼容。
    (4) **Asynchronous Concurrent Execution**（解决：outlier乘法tall-and-skinny矩阵导致GPU SM underutilization）：将量化分解为outlier/normal两个独立task→分解dependency graph→四个GEMM kernel之间无dependency可通过多CUDA stream异步并发执行。CUDA events仅在有真实依赖处同步（如quant→GEMM dependency）。掩盖kernel launch overhead + 提升SM利用率。
  - 编译框架层：论文未明确说明（手工CUTLASS/Triton kernel，JIT编译机制）
  - kernel调度层（核心创新）：
    (5) **Separate-kernels with CUTLASS**（解决：fused kernel无法对不同精度配置独立on-chip resource allocation）：INT4-INT4 kernel→shared memory需求小→compiler可用更多register做loop unrolling提升ILP；INT8-INT8 kernel→共享内存需求大→occupancy由shared memory restrict→compiler自动balance。separate-kernels的launch overhead和underutilization由async execution弥补。评估表明separate-kernels+async优于fused-kernel。
    (6) **Software Pipelining with cp.async**（解决：memory access latency导致warp stall）：使用PTX cp.async指令异步加载GMEM→SMEM→pipeline fill→steady state: wait oldest copy→mma compute→issue new copy→pipeline drain。
    (7) **Fused Triton kernels**（解决：在线outlier detection引入runtime overhead）：将per-token row-max + top-k selection + INT4/INT8 quantization + data packing融合为单一Triton kernel，减少kernel launch和内存round-trip。INT4→INT8 casting在SMEM内用两条binary arithmetic指令完成（避免昂贵type conversion指令）。
  - 硬件架构层：NVIDIA RTX 4090 GPU。利用INT4 Tensor Core (8×FP16)和INT8 Tensor Core (2×FP16)。cp.async (Ampere+)异步memory copy。Cross-precision accumulation使用INT32累加器（防止overflow）。

  Baseline缺陷→RoMeo方案映射：
  | Baseline缺陷 | RoMeo方案 | 效果 |
  |-------------|---------|------|
  | Channel-wise方法无法消除token-wise outliers | Hadamard rotation将channel-wise outlier迁移到token维度再处理 | Peak activation 1272→58.5 (rotation后)→18.6 (TO pruned后) |
  | 4-bit下residual outliers导致perplexity退化 | Token-wise INT8/INT4 mixed precision，5% outlier用INT8 | Qwen3-8B PPL: MixQ 14.76→RoMeo 10.97; QuaRot 11.53→RoMeo 10.97 |
  | Token-wise mixed precision在non-reduction dim无法利用Tensor Core | Permutation-free dense computation + redundant outlier copy | 无需permutation，所有GEMM为dense uniform-precision |
  | Tall-and-skinny outlier矩阵SM underutilization | Asynchronous four-kernel concurrent execution over CUDA streams | Batch=16 layer latency 6.73→3.39ms with async |
  | 在线outlier detection引入runtime overhead | Fused Triton outlier detection + quantization + packing kernel | Hadamard+Quant+Post-mul overhead仅~12% baseline latency |
  | INT4/INT8 fusion kernel无法fine-tune on-chip资源 | Separate CUTLASS kernels per precision + software pipeline | Separate+Async overall最优 vs fused kernel |
