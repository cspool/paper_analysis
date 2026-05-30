## MTIA (Meta Training and Inference Accelerator)

术语是什么？
MTIA (Meta Training and Inference Accelerator)是Meta自研的AI加速器芯片系列，专为推荐系统training和inference workloads设计。与通用GPU不同，MTIA是domain-specific accelerator，其架构针对recommendation-specific computation patterns（embedding lookup、feature interaction、data preprocessing）深度定制。目前有v2i和v3两代。

从硬件架构角度拆解术语：
**MTIA v2i**采用8×8 Processing Element (PE) Array通过Network-on-Chip互联。每PE含：dual RISC-V cores (Core A + Core B) + 专用fixed-function units——MLU (Memory Layout Unit, 数据格式转换)、DPE (Dot Product Engine, 矩阵乘)、RE (Reduction Engine, 归约)、SE (SIMD Engine, 向量)、CP (Command Processor, 控制流) + SFU (Specialized Function Units with LUT-based activation: exp/gelu/log/sigmoid/tanh)。Memory: custom on-chip SRAM hierarchy with distinct on-chip/off-chip bandwidth profiles。Inter-PE communication: 硬件neighbor PE数据传输和同步primitives。

**Programming Model**: Triton-MTIA (Triton DSL的MTIA扩展)，支持libdevice API (tl.extra.libdevice.gelu(x)→SFU LUT)、compilation options (cb_multiplier for Circular Buffer扩展、use_dual_core for双核heterogeneous执行)、custom type system (TensorView/CoreID/ExecutionGrid)、advanced sync (cross-PE broadcast/reduction via direction attribute、runtime barriers、explicit tensor copies)。编译输出: Triton-MLIR→MTIA-MLIR→LLVM-IR→RISC-V binary。可选C++ emission (emit_cxx=True)暴露compiler intermediate representation用于debugging。

**vs NVIDIA/AMD**: 非通用GPU——专用memory hierarchy（非multi-level cache+tensor cores式）、专用programming model（非CUDA/ROCm通用模型）、domain-specific accelerators（针对recommendation inference patterns而非通用GEMM/attention为主）。

术语一般如何实现？如何使用？
通过Triton-MTIA compiler toolchain编译Triton kernel为RISC-V binary，mtia_triton_launcher在硬件上执行。KernelEvolve通过knowledge injection使LLM能生成MTIA-specific optimized kernels。MTIA v2i缺少多个ATen ops (clamp.out, gather.out, sort.values_stable, _unique2等)，KernelEvolve-generated kernels填补了coverage gaps，实现从disaggregated CPU+MTIA到monolithic MTIA-only deployment的架构转变。

涉及论文标题：
- KernelEvolve: Scaling Agentic Kernel Coding for Heterogeneous AI Accelerators at Meta

---
