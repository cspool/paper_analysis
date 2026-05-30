## TileLang: A Composable Tiled Programming Model for AI Systems

- baseline方法是什么？
  **Baseline方法有两类：编译器/DSL方法（Triton, TVM）和手写kernel方法（FlashAttention-3, FlashMLA, AITER, Marlin等）。**

  **Triton [20]**（tile-level compiler）：提供block-level编程原语但隐藏thread行为、memory layout和address-space annotations于自动生成的策略之后。对专家开发者存在三个关键痛点：(1) **无法自定义memory layout**——Triton的tl.dot等向量化操作不支持用户自定义PTX/custom tile operator注册，对于量化weight的GEMM kernel，无法实现硬件对齐的自定义data layout；(2) **Pipeline控制受限**——Triton仅暴露num_stages参数，不允许用户定义完全自定义的pipeline（如warp specialization）；(3) **低精度支持不完善**——sub-byte类型操作需要通过uint32 bitwise手工解包，且解包后的register layout与Tensor Core要求的MMA layout不兼容，需通过shared memory做layout conversion（成为性能瓶颈）。

  **TVM [7, 8]**（schedule-oriented compiler）：要求用户显式区分computation和schedule，需手动注册新tensor指令和指定buffer layout。schedule程序的编写和理解困难，且primitive-style scheduling不支持现代GPU的关键优化（如cp.async/TMA based software pipelining）。

  **手写kernel方法**：FlashAttention-3使用手写CUDA（TMA + wgmma.mma_async + warp specialization），但固定tile sizes导致对小sequence length suboptimal。FlashMLA同样是手写CUDA kernel。这些方法的code line数多（FlashMLA的手写实现远超TileLang的~70行Python），且通用性差。

  **Baseline全栈执行例子（以Triton FP16 GEMM, H100 GPU为例）：**
  - 算法层：标准FP16矩阵乘法 C[M,N] = A[M,K] × B[K,N]，FP32 accumulate
  - 系统框架层：Triton kernel，@triton.jit编译，auto-tuning搜索tile大小和num_stages
  - 编译框架层：Triton将Python kernel编译为PTX → SASS。Thread binding和memory layout由Triton编译器自动决定，用户无法干预。Pipeline仅通过num_stages参数控制overlap深度
  - Kernel调度层：Triton自动生成global→shared copy（cp.async）+ shared→register load + Tensor Core MMA。但thread-level的register分配、shared memory bank conflict避免策略均由compiler heuristics决定，无用户控制。对于量化GEMM：load packed uint32 → bitwise unpack in registers → store to shared memory for layout conversion → ldmatrix reload → MMA——shared memory layout conversion是额外开销
  - 硬件架构层：NVIDIA H100 (Hopper)，Tensor Core wgmma.mma_async，TMA hardware unit

- 论文方法是什么？如何对应解决Baseline的缺陷？

  **TileLang方法：Python-embedded tiled DSL + JIT compiler，核心创新是解耦dataflow与scheduling space，将thread binding、memory layout、tensorization、pipeline封装为composable annotations和primitives。**

  **解决Triton的"隐式scheduling"限制**：TileLang显式暴露四种scheduling space为用户可控的annotations/primitives：
  1) **Thread Binding**: 通过Layout Inference Pass自动推断，但用户可通过T.Fragment手动指定thread→buffer映射。Fragment Layout (f: K^n → K²) 精确描述block级register file到thread的partitioning。
  2) **Memory Layout**: T.annotate_layout允许用户自定义shared/global memory layout（如自定义swizzle模式）。Layout抽象基于composable IterVar algebra (f: K^n → K^m)，支持stacking和composition（图5）。T.use_swizzle一键启用L2 cache友好的swizzle thread block ordering。
  3) **Tensorization**: 两种路径——Tile Library-based (CUTLASS cute/AMD CK, T.call_extern) 和 Direct PTX injection (T.ptx)。用户可直接注册custom tile operator（Python中定义Lower和InferLayout接口）。对于专家：可通过T.import_source + T.call_extern注入C++ template实现的DP4A/MMA等指令（图10a）。
  4) **Pipeline**: T.Pipelined(num_stages=N)自动推导pipeline schedule，同时允许用户显式指定producer/consumer order。自动支持Ampere cp.async、Hopper TMA+warp specialization、AMD CDNA async DMA。

  **解决Triton的"低精度layout conversion瓶颈"**：TileLang的Layout Inference Pass自动为GEMM的A_shared/B_shared应用MakeSwizzleLayout（消除bank conflict），为C_local应用MakeMMASTMatrixLayout（Tensor Core要求的register layout）。在Dequantized Matmul中（图17），weight以packed u8形式加载到register，通过View做零开销类型reinterpret（u8→i4）+ layout transform（tile layout→MMA layout），完全在寄存器内完成，消除Triton的shared memory layout conversion额外往返。

  **解决TVM的"schedule编程困难"**：TileLang采用dataflow-centric编程范式——用户仅需描述tile-level dataflow（T.copy, T.gemm, T.reduce, T.atomic），编译器自动完成所有scheduling推导。仅当默认优化不够时，用户才通过annotations精准控制。

  **解决手写kernel的"通用性缺失"**：所有kernel（GEMM, FlashAttention, Linear Attention, MLA, Dequantized Matmul）共享同一TileLang编程范式。FlashAttention仅需~70行Python代码即达FlashAttention-3的98%性能。Dequantized Matmul通过同一程序模板参数化支持INT2/INT4/NF4等多种量化格式。

  **TileLang方法全栈执行例子（FP16 GEMM, H100 GPU，对应图11）：**
  - 算法层：标准FP16矩阵乘法 C[M,N] = A[M,K] × B[K,N]，FP32 accumulate
  - 系统框架层：TileLang Python程序（~30行），@tilelang.jit decorator，tilelang.compile(program, target="cuda")
  - 编译框架层：五阶段pipeline——Parser (Python AST → TileLang AST) → IR Builder (→ TVM Tensor IR) → Optimization (Layout Inference + Thread Binding + Pipeline Derivation + Vectorization) → Codegen (→ CUDA C with TMA/wgmma.mma_async instructions) → nvcc → binary
  - Kernel调度层：
    a) T.Kernel(N//block_N, M//block_M, threads=128) → grid (N/128, M/128), block (128 threads)
    b) T.alloc_shared → shared memory buffers (A_shared[128,32] f16, B_shared[32,128] f16)
    c) T.alloc_fragment → register file C_local[128,128]（block-level allocation）
    d) Layout Inference: Gemm(priority=highest) → A_shared=SwizzleLayout, B_shared=SwizzleLayout, C_local=MMA_MatrixLayout → Copy(priority=lower) → auto parallelize + vectorize
    e) Thread Binding: C_local[128,128]通过Fragment Layout分发到128个threads（2 warps × 64 threads），每个thread持有部分register elements
    f) T.Pipelined(K // 32, num_stages=2): 推导Copy-GEMM interleaved pipeline → Hopper自动TMA + wgmma.mma_async + warp specialization（producer: TMA copy, consumer: wgmma.mma_async, mbarrier同步）
    g) Loop body执行: TMA load A/B tiles → mbarrier.arrive → mbarrier.wait → wgmma.mma_async(A_shared, B_shared, C_local) → 循环K维 → T.copy(C_local → C[global]) with thread binding + vectorized store
  - 硬件架构层：NVIDIA H100 (Hopper)，TMA hardware unit（异步global↔shared copy），wgmma.mma_async warp-group MMA，mbarrier同步

  对比Triton baseline：TileLang在同等简洁语法下（~30行 vs Triton ~25行 GEMM），实现了(vs Triton) 1.13× speedup on H100，通过custom swizzle layout消除shared memory bank conflict，通过TMA+warp specialization实现更高效的pipeline overlap。关键差异在于TileLang将scheduling从dataflow中解耦，让编译器在更大搜索空间中自动寻找最优调度。
