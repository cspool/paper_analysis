## TileLang: A Composable Tiled Programming Model for AI Systems

- 属于编译框架的实现是什么？实验比较什么？
  实现是TileLang，一个面向AI系统的可组合tiled编程模型和JIT编译器。核心由五阶段编译流水线组成：(1) Parser——将TileLang Python程序解析为Python AST再转换为TileLang AST；(2) IR Builder——将AST转换为TVM IR（复用TVM的语法树和基础设施）；(3) Optimization——执行图优化和调度变换，包括Layout Inference自动并行化、动态参数简化、自动Pipeline推导、动态shape的loop tail splitting等passes；(4) Codegen——将优化后的IR翻译为后端代码（LLVM IR / CUDA C/C++ / HIP C/C++）；(5) Runtime——加载和执行编译后的kernel。

  实验比较：(1) FlashAttention性能——vs FlashAttention-3, Triton, PyTorch（H100，speedup 1.36×/1.41×/1.70×）；(2) Linear Attention性能——vs Triton（H100，平均speedup 1.77×和2.10×）；(3) MLA性能——vs Torch, Triton, FlashInfer, FlashMLA（H100达1075.9× over Torch, 98% of FlashMLA）；vs Torch, Triton, AITER（MI300X达129.2× over Torch, 95% of AITER）；(4) GEMM性能——vs Triton和vendor库cuBLAS/rocBLAS（RTX 4090/A100/H100/MI300X, 0.97-1.10× vs vendor, 1.03-1.25× vs Triton）；(5) Dequantized Matmul——vs cuBLAS, Marlin, BitsandBytes（A100, INT2 7.65× over cuBLAS, INT4 1.04× vs Marlin, NF4 1.62× vs BitsandBytes）。

- 硬件平台是什么，配置是什么。
  评估平台：NVIDIA H100 (80 GB, Hopper, CUDA 12.4)，NVIDIA A100 (80 GB, Ampere, CUDA 12.4)，AMD Instinct MI300X (192 GB, CDNA3, ROCm 6.1.0)，NVIDIA RTX 4090 (Ada Lovelace)。所有平台运行Ubuntu 20.04。

- 开源编译框架是什么。修改了什么。
  TileLang是自研编译框架，基于TVM IR基础设施但进行了大量修改和扩展：(1) 新增TileLang AST前端和Parser，支持Python-embedded tile级编程语言；(2) 新增Layout Inference Pass——通过LayoutMap和优先级层次（GEMM > element-wise > copy）自动推断buffer layout和thread binding；(3) Fragment Layout系统——扩展TVM的Layout抽象，支持block级register file到thread的映射，提供repeat/repeat_on_thread/replicate四种primitive操作组合构建复杂fragment layouts；(4) 自动Pipeline推导——分析Copy和GEMM的依赖关系，生成结构化pipeline schedule，自动插入cp.async/cp.async.commit/cp.async.wait指令或TMA/mbarrier；(5) 自动Warp Specialization——在Hopper架构上自动将producer/consumer线程分组并插入mbarrier同步；(6) 支持内联PTX和C++ source injection（T.ptx, T.import_source, T.call_extern）；(7) 集成CUTLASS (NVIDIA) 和Composable Kernel (AMD) 作为tile library后端。

- 开源情况。基于开源文档和论文，使用例子解释编译框架如何使用？作用是什么？至少具体到编译框架输入到输出的全过程。
  开源地址：https://github.com/tile-ai/tilelang

  作用：TileLang的核心价值是以统一的Python tile级DSL表达各种AI kernel（GEMM、FlashAttention、Linear Attention、MLA、Dequantized Matmul），通过解耦dataflow与scheduling（thread binding、memory layout、tensorization、pipeline），让用户只需描述数据流逻辑，编译器自动处理并行化、pipeline、向量化、warp specialization等优化。同时保留thread级控制接口供专家手动调优。

  全编译过程（以GEMM kernel为例）：
  ```
  输入：Python TileLang程序（~30行）
    @tilelang.jit
    def Matmul(A: T.Tensor, B: T.Tensor, C: T.Tensor):
      with T.Kernel(N // block_N, M // block_M, threads=threads) as (bx, by):
        A_shared = T.alloc_shared(block_M, block_K)    # shared memory allocation
        B_shared = T.alloc_shared(block_K, block_N)
        C_local  = T.alloc_fragment(block_M, block_N)   # register file allocation
        T.clear(C_local)
        for k in T.Pipelined(K // block_K, num_stages=2):
          T.copy(A[by*block_M, k*block_K], A_shared)    # global→shared copy
          T.copy(B[k*block_K, bx*block_N], B_shared)
          T.gemm(A_shared, B_shared, C_local)           # Tensor Core matmul
        T.copy(C_local, C[by*block_M, bx*block_N])      # register→global store

  Step 1 — Parser:
    TileLang Python程序 → Python AST → TileLang AST
    识别T.Kernel context, T.alloc_shared, T.alloc_fragment, T.Pipelined等语义节点

  Step 2 — IR Builder:
    TileLang AST → TVM Tensor IR（中间表示）
    将tile operators（copy, gemm, pipelined）lowering为TVM IR表示
    T.alloc_shared → memory scope="shared" buffer
    T.alloc_fragment → memory scope="local" buffer (register files)
    T.Pipelined → 带pipeline annotation的loop IR

  Step 3 — Optimization（核心编译passes）:
    a) Layout Inference Pass: 按优先级(Gemm > Element-wise > Copy)推断所有buffer的Fragment Layout
       - Gemm buffer A_shared: MakeSwizzleLayout (避免bank conflict)
       - Gemm buffer B_shared: MakeSwizzleLayout
       - Gemm buffer C_local: MakeMMASTMatrixLayout (Tensor Core所需的thread→register映射)
       - Copy source/target: 根据thread binding自动parallelize和vectorize
    b) Thread Binding: 将block级register files按fragment layout分发到各thread
    c) Pipeline Derivation: 分析loop body的依赖关系，生成cp.async + commit + wait指令序列
       在Ampere: 自动插入cp.async, cp.async.commit_group, cp.async.wait_group
       在Hopper: 自动TMA + mbarrier + warp specialization
    d) Vectorization: 自动应用vectorized load/store (如128-bit loads)
    e) Dynamic Shape Handling: loop tail splitting for dynamic dimensions

  Step 4 — Codegen:
    优化后TVM IR → CUDA C/C++ (NVIDIA) / HIP C/C++ (AMD) / LLVM IR (CPU)
    T.gemm → cutlass::gemm_ss (CUDA) / composable_kernel::gemm (HIP) / 手写PTX
    T.copy → vectorized memory copy loops with thread binding
    Pipeline IR → cp.async + commit + wait / TMA instructions

  Step 5 — Runtime:
    Generated CUDA/HIP code → nvcc/hipcc compile → binary (.cubin/.hsaco)
    tilelang.compile(program, target="cuda") 返回可调用的kernel函数
    Runtime管理kernel cache，避免重复编译
  ```
