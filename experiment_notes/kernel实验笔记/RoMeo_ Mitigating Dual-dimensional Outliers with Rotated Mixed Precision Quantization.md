## RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  提出RoMeo的cross-precision混合精度GEMM kernel系统和辅助kernel。核心kernel/runtime设计：(1) Permutation-free Mixed Precision Computation：预分配dedicated outlier buffer（大小基于预定的outlier token数量），整个activation矩阵量化为INT4，outlier token embedding复制到outlier buffer量化为INT8。所有四个cross-precision矩阵乘法（W4A4/W4A8/W8A4/W8A8）各自操作dense uniform-precision矩阵，每个GPU thread block处理一种精度组合，完全避免permutation overhead。outlier token同时参与INT4和INT8两次计算（tolerate redundant computation以保证contiguous memory layout符合Tensor Core指令要求），高精度结果最终overwrite对应输出位置。(2) Separate-Kernels with Software Pipelining：采用separate-kernels而非fused-kernel实现，原因为不同精度组合的shared memory需求不同（INT8-INT8 kernel需2× shared memory vs INT4-INT4），separate允许compiler对每种kernel独立分配on-chip资源。Kernel内部使用software pipeline（Algorithm 2）：通过cp.async PTX指令异步加载global→shared memory（pipeline fill阶段），steady state每iteration等待oldest copy完成→Tensor Core mma计算→发射新async copy，最终drain阶段完成所有remaining mma。对于INT4→INT8的cross-precision计算，在shared memory内使用两个binary arithmetic指令做类型转换（而非昂贵的type conversion指令）。(3) Fused Triton Kernels：开发fused Triton kernel用于在线outlier identification（per-token row-max + top-k selection）、量化（round + scaling）和INT4 data packing，减少kernel launch次数和内存往返。(4) 在线动态Outlier Detection：因token-wise outlier来自输入语言特征，需要运行时在线检测；kernel执行在线row-max reduction然后top-k selection确定outlier set，而非离线静态分析。实验比较kernel-level speedup (normalized to BF16)，在Qwen3-8B/14B/32B和Llama-3.1-70B的QKV/O/UG/D四种linear layer matrix shape上，对比INT8 kernel、Atom group-wise INT4 kernel、QuaRot INT4 kernel。RoMeo geomean speedup 4.68× over BF16，与QuaRot 4.55×相当但额外计算5%高精度outlier。消融实验展示U-ker→U-ker+Pipe→S-ker→S-ker+Pipe→S-ker+Pipe+Async五组配置在batch=16/64下的layer-level latency breakdown。

- 后端平台是什么，配置是什么。
  NVIDIA GeForce RTX 4090 GPU (Ada Lovelace, 24GB memory, Compute Capability 8.9, peak INT4 Tensor Core throughput 8× over FP16)。Python 3.12, PyTorch 2.8.0, CUDA 12.8。Kernel编译：CUTLASS (cross-precision CUDA kernels), Triton (fused outlier detection/quantization/packing kernels), HadaCore (FWT Hadamard变换)。JIT编译机制：首次执行时编译对应模型维度的kernel并缓存compiled binary后续复用，auto-tune tiling size和pipeline stage数量。

- 评估性能的软件/脚本是什么。修改了什么。
  自研cross-precision CUDA kernels（基于CUTLASS）+ fused Triton kernels。Baseline kernels包括：BF16 (PyTorch half-precision matmul)、INT8 (CUTLASS INT8 matmul kernel)、Atom (group-wise INT4 mixed precision kernel)、QuaRot (INT4 matmul with fused dequantization kernel)。所有kernel使用CUDA Graph捕获消除launch overhead后，CUDA events测量平均latency。Kernel benchmark覆盖Qwen3和Llama-3.1模型实际weight tensor shapes（QKV_proj, O_proj, UpGate_proj, Down_proj），M dimension固定为4096，batch size可变。NVIDIA Nsight Compute用于profiling各kernel的shared memory/register使用和occupancy。

- 开源情况。评估软件/脚本如何使用？基于开源文档和论文，使用例子解释。
  开源：https://github.com/thu-pacman/RoMeo。Kernel使用流程（以Qwen3-8B Down_proj layer [4096×4096] GEMM为例）：
  1. JIT编译阶段：RoMeo首次加载模型→根据weight shape (4096,4096)和outlier budget (5%, ~205 outlier tokens)→auto-tune确定tiling size (TM/TN)和pipeline stages (Nstage)→编译四个separate CUTLASS cross-precision kernels + Triton outlier detection/quantization/packing kernels→缓存compiled binaries
  2. 在线执行阶段（batch=64, seq_len=128→M=8192 tokens）：
     a. Triton outlier detection kernel：per-token row-max reduction→top-k outlier selection→生成outlier index mask (8192 entries, ~410 are outliers)
     b. Triton quantization kernels：normal tokens→INT4 quantize + pack（W4A4 kernel input）；outlier tokens→copy to outlier buffer→INT8 quantize（W8A4 kernel input）；weight offline已per-column mixed precision quantized
     c. Asynchronous concurrent GEMM：四个CUDA streams各自launch W4A4/W4A8/W8A4/W8A8 kernel→每个kernel内部cp.async pipeline异步加载global A/B tile到shared memory→INT4→INT8 casting in shared memory→mma指令计算→结果scale by per-token scaling factor→写回global memory
     d. Post-mul overwrite kernel：高精度outlier结果overwrite W4A4结果的对应位置→完成所有精度组合的结果合并
  3. 性能：4096×4096×4096 GEMM RoMeo kernel ~0.56ms，BF16 baseline ~2.62ms (4.68× speedup)。Pipeline+Async优化后batch=16 layer latency从6.73ms降至3.39ms (2.0× over BF16)

