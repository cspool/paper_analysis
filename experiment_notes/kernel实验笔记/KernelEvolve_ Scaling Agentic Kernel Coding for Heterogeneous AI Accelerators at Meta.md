## KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是KernelEvolve通过LLM agent驱动的图搜索自动生成针对多后端平台的优化Triton kernel。具体包括：(1) 跨平台kernel合成——从统一operator specification自动生成NVIDIA GPU (CUDA/Tensor Cores/TMA)、AMD GPU (Infinity Cache/ROCm)和MTIA (SFU/PE array/inter-PE communication)的优化kernel实现；(2) 算子融合（Operator Fusion）——如conv1d将5个kernel融合为2个（消除NCHW↔NHWC layout转换和中间tensor materialization），WuKong Optimized FM将2次bmm融合为单kernel（消除intermediate HBM round-trip），InterFormer PFFN将FFN+GELU+RMSNorm融合为single-pass kernel（从3次memory pass减少到1次），MapIdTransform将4个PyTorch算子（bucketize+clamp+gather+where）融合为单kernel（消除3个中间tensor materialization）；(3) Shape-specific tiling——针对生产shape定制tile大小以保持SRAM residency；(4) 硬件特定优化——NVIDIA H100上利用TMA+Warp Specialization+Tensor Cores+double-buffering+differentiated cache modifiers（.ca/.cg），AMD上利用Infinity Cache-aware tiling，MTIA上利用SFU LUT operations+inter-PE broadcast/reduction+runtime barriers+compile-time loop unrolling+SIMD-vectorized counting+adaptive block sizing+cb_multiplier/use_dual_core compilation options；(5) Cross-operation tile reuse——PFFN中同一tile加载后完成全部算子链（matmul+bias+GELU+RMSNorm）再写回HBM；(6) Batched parallel execution——Batch Event Truncate中将多feature并行处理替代sequential per-feature loop；(7) Register-resident computation——中间结果保持在寄存器中，无intermediate tensor allocation。

  实验比较：
  - OSS Operator：160个ATen operators，对比KernelEvolve-generated Triton kernel vs PyTorch torch.compile baseline
  - KernelBench Level 1/2/3：共250个problems
  - Convolutional Transformer conv1d：Triton kernel vs PyTorch conv1d vs PyTorch conv2d workaround，FP16/FP32，多种batch sizes (32-2048) 和 shapes
  - Cross-platform conv1d：5个硬件平台 (H100, A100, MI300, MI350, MTIA v3)
  - WuKong Optimized FM：Fused Triton kernel vs PyTorch torch.compile (two extern_kernels.bmm)，多种 (B,N,D,K) production shapes
  - InterFormer PFFN：Fused Triton kernel vs PyTorch torch.compile (extern_kernels.bmm + triton_per_fused_rms_norm_add_gelu)，多种 (B,N,D,K) shapes
  - MapIdTransform：Fused Triton kernel vs PyTorch (bucketize+clamp+gather+where)，MTIA v2i/v3，多种 (UniqueIDs × Batch) 配置
  - MBDT：Fused Triton kernel vs PyTorch torch.compile，MTIA v2i/v3，多种 Batch × Features × Borders 配置
  - Batch Event Truncate：Batched Triton kernel vs non-batched PyTorch sequential，多种 feature counts (1/5/9/32) 和 event lengths

- 后端平台是什么，配置是什么。
  NVIDIA H100 GPU（Hopper, TMA, WGMMA, mbarriers, 多级cache hierarchy），NVIDIA A100 GPU（Ampere），AMD MI300 GPU（Infinity Cache, CDNA架构），AMD MI350 GPU（CDNA4），MTIA v2i（8×8 PE array, per-PE: dual RISC-V cores + SFU/DPE/RE/SIMD/MLU/CP fixed-function units, on-chip SRAM, custom inter-PE communication），MTIA v3（next-gen, improved compute throughput and native operator coverage）。LLM backends：Claude 4.5, GPT-5, Meta CWM, Llama on Twine（64K-1M token context windows）。Profiling工具：TritonBench, Triton MPP (NVIDIA), Triton Proton (intra-kernel), NCU (NVIDIA kernel-level), Torch Profiler (system-level), MTIA Insight (MTIA-specific: PE utilization, DPE/SFU/MLU utilization, cache hit rates, memory bandwidth, per-PE read/write counters)。

- 评估性能的软件/脚本是什么。修改了什么。
  使用TritonBench (https://github.com/meta-pytorch/tritonbench) 作为主要benchmark框架，通过BenchmarkOperator wrapper进行correctness验证（torch.allclose with precision-dependent tolerances）和speedup测量。Torch Profiler用于system-level timeline capture（CPU/GPU time, kernel launch overhead）。NCU用于NVIDIA kernel-level hardware metrics（occupancy, memory throughput, instruction mix）。Triton Proton用于intra-kernel instruction-level profiling。Triton MPP (Multi-Pass Profiler) 统一跨工具profiling数据采集。MTIA Insight用于MTIA-specific metrics（PE utilization, fixed-function accelerator metrics, cache analysis, memory bandwidth, load-store throughput）。Evaluation Code Generator自动将LLM生成的kernel artifact转换为各profiling tool的instrumented evaluation script，通过FaaS平台dispatched到remote hardware执行。
  
  修改：KernelEvolve的核心修改是构建了完整的agentic kernel优化pipeline，而非修改单个profiling工具。它通过Evaluation Code Generator自动合成evaluation harness（输入：标准化的PytorchModel+TritonModel+get_inputs() artifact；输出：各profiling tool的executable Python script），将profiling从手动操作转化为图搜索反馈信号。关键架构决定：evaluation代码是deterministically generated（保证reproducibility），而kernel逻辑是LLM-generated（允许创造性优化），两者通过标准接口解耦。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  KernelEvolve框架是Meta内部生产系统，论文未提供完整开源链接。系统依赖的开源组件：
  - TritonBench: https://github.com/meta-pytorch/tritonbench（kernel benchmark框架）
  - KernelBench: LLM kernel generation benchmark
  - KernelAgent: https://github.com/meta-pytorch/KernelAgent
  - KernelLLM: https://huggingface.co/facebook/KernelLLM
  - TLX: https://github.com/facebookexperimental/triton

  评估原理：
  1. KernelEvolve的Evaluation Framework将kernel评估抽象为多维度fitness signal采集：correctness（binary pass/fail）、performance（speedup ratio）、profiling metrics（occupancy, memory throughput, instruction mix, intra-kernel pipeline behavior）。这些信号综合决定图搜索节点的fitness score和下一轮优化方向。
  2. Evaluation Code Generator（deterministic）将标准化的dual-implementation artifact（PytorchModel + TritonModel + get_inputs()）转换为各tool的instrumented script，确保across-kernel-variant的reproducible profiling。
  3. FaaS-based remote evaluation将kernel评估dispatch到对应硬件平台的remote worker（NVIDIA/AMD/MTIA），利用pre-deployed Bento interpreter environments消除per-kernel compilation overhead。
  4. Fitness Function: F(v) = t_pytorch / t_triton，correctness失败或compilation/runtime错误的kernel F(v)=0。

  全过程（以MapIdTransform kernel在MTIA v2i上为例）：
  ```
  输入：KernelEvolve生成的Triton kernel artifact
    kernel_n.py:
      class PytorchModel(nn.Module):  # PyTorch reference: bucketize+clamp+gather+where
      @triton.jit def mapid_kernel(...):  # Fused Triton: binary search + clamp + match
      class TritonModel(nn.Module):  # Grid launch wrapper
      def get_inputs():  # Test cases: various (UniqueIDs, Batch) sizes

  Step 1 - Evaluation Code Generation (Section 3.4.3):
    → Deterministic code generator解析artifact
    → 生成 TritonBench harness:
        class MyOperator(BenchmarkOperator):
          def get_input_iter(self): return get_inputs()
          def run(self, inputs): return TritonModel()(*inputs), PytorchModel()(*inputs)
    → 配置: baseline=True (correctness), speedup measurement enabled
    → 生成 MTIA Insight instrumentation:
        mtia_insight.start(); kernel[grid](x, ...); metrics = mtia_insight.stop()
    → 生成 Torch Profiler script: torch.profiler.profile() around kernel launches

  Step 2 - FaaS Remote Evaluation (Section 3.4.6):
    → Dispatch evaluation request to FaaS endpoint: meta_kernel_mtia_interpreter
    → Remote MTIA worker: load pre-deployed Bento interpreter (Triton-MTIA compiler + MTIA Insight + runtime)
    → 执行evaluation script:
        a) TritonBench:
           - PyTorch baseline执行: torch.bucketize → torch.clamp → torch.gather → torch.where
             (部分ATen ops在MTIA v2i上缺少native支持 → CPU fallback + host-device sync)
             执行时间: t_pytorch
           - Triton kernel执行: 单次launch, fused binary search in registers + coalesced block-parallel
             执行时间: t_triton
           - Correctness验证: torch.allclose(pytorch_output, triton_output, atol=..., rtol=...)
        b) MTIA Insight profiling:
           - PE utilization (%), SFU LUT utilization (%)
           - Memory bandwidth: DRAM read/write bytes per PE
           - Per-PE CPU runtime, cache hit rates (I/D-cache, LLC)
           - Load-store throughput per PE
        c) 返回结构化结果:
           {correctness: true, speedup: 3.48, pytorch_ms: 1.623, triton_ms: 0.466, ...}

  Step 3 - Feedback Loop (Section 3.1):
    → Fitness score: F(v) = 1.623 / 0.466 = 3.48
    → Context Memory Sub-Agent分析:
        - 诊断: 4 PyTorch ops → 1 fused kernel, 4× memory traffic reduction
        - 发现: speedup随batch size增大（0.78× at 2000 → 4.07× at 50000）
        - 建议: 小batch时kernel launch overhead主导, 可探索persistent kernel
    → Metadata store更新: (id, pid, score=3.48, is_buggy=false, path_ref)
    → 分析报告写入: object store overview.md

  输出（记录在metadata store中）：
    - correctness: 100% pass across all test shapes
    - speedup: 3.48× on MTIA v2i (100 × 10000), 4.07× peak (10000 × 50000)
    - 跨平台: 1.05-1.36× on MTIA v3（baseline更强）
  ```
