## Tile-based Programming Language (TileLang DSL)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

TileLang 是一个面向 AI 系统的 Python-embedded 领域特定语言（DSL），允许用户在 tile（瓦片/分块）粒度上编写高性能 GPU kernel。其核心设计哲学是"dataflow-centric"（以数据流为中心）：用户仅需使用 tile-level operators（T.copy, T.gemm, T.reduce, T.atomic 等）描述数据在各级硬件内存层次（global memory → shared memory → registers）之间的流动路径和计算逻辑，编译器自动处理并行化、线程绑定、内存布局、pipeline 重叠和向量化等底层调度优化。TileLang 的 tile 是一等公民——tile 代表一块有形状的数据（如 block_M × block_K），可由 warp、thread block 或等效并行单元所有和操作。与 Triton 只提供 block-level 原语且隐藏 thread 行为不同，TileLang 显式暴露 T.alloc_shared（shared memory 分配）、T.alloc_fragment（register file 分配）等硬件内存分配原语，并允许专家用户通过 T.ptx、T.call_extern 等接口手动注入 PTX 指令或 C++ 模板代码以获取极致性能。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

TileLang 的编译流程（五阶段 pipeline）：
```
输入: Python TileLang 程序（~30行 GEMM kernel）
  @tilelang.jit
  def Matmul(A, B, C):
    with T.Kernel(N//block_N, M//block_M, threads=128) as (bx, by):
      A_shared = T.alloc_shared(block_M, block_K)
      B_shared = T.alloc_shared(block_K, block_N)
      C_local  = T.alloc_fragment(block_M, block_N)
      T.clear(C_local)
      for k in T.Pipelined(K // block_K, num_stages=2):
        T.copy(A[by*block_M, k*block_K], A_shared)
        T.copy(B[k*block_K, bx*block_N], B_shared)
        T.gemm(A_shared, B_shared, C_local)
      T.copy(C_local, C[by*block_M, bx*block_N])

Step 1 — Parser: Python 源码 → Python AST → TileLang AST
  - 识别 T.Kernel context（grid_size, threads）
  - 识别 T.alloc_shared / T.alloc_fragment 语义（memory scope = shared / local）
  - 识别 T.Pipelined 循环标注（pipeline depth = num_stages）

Step 2 — IR Builder: TileLang AST → TVM Tensor IR
  - T.copy → memory copy IR node with source/destination scope annotations
  - T.gemm → call_extern("cute::gemm_ss", ...) 或 direct MMA IR node
  - T.Pipelined → loop IR with pipeline annotation

Step 3 — Optimization (核心编译 passes):
  a) Layout Inference: 按优先级(Gemm > Element-wise > Copy)推断所有 buffer 的 Fragment Layout
     - A_shared, B_shared: MakeSwizzleLayout (消除 bank conflict)
     - C_local: MakeMMASTMatrixLayout (Tensor Core 要求的 register layout)
  b) Thread Binding: 将 block 级 register files 按 Fragment Layout 分发到各 thread
  c) Pipeline Derivation: 分析 Copy-GEMM 依赖 → 生成 cp.async.commit + wait 序列
  d) Vectorization: 自动应用 128-bit vectorized load/store
  e) Loop Tail Splitting: 处理动态 shape 的尾部迭代

Step 4 — Codegen: 优化后 IR → CUDA C (NVIDIA) / HIP C++ (AMD)
  - Hopper: TMA + wgmma.mma_async + mbarrier + warp specialization
  - Ampere: cp.async + mma.sync + __syncthreads
  - CDNA: s_waitcnt + buffer_load_dword + HIP-wrapped async copy

Step 5 — Runtime: nvcc/hipcc 编译 → .cubin/.hsaco binary
  tilelang.compile(program, target="cuda") → 可调用 kernel 函数
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

开源: https://github.com/tile-ai/tilelang

使用方式: Python pip 安装 `tilelang` 包，通过 `@tilelang.jit` decorator 标记 kernel 函数，自动 JIT 编译。支持 `tilelang.compile(program, target="cuda"/"hip")` 显式编译。TileLang 的 GEMM 后端默认调用 CUTLASS（NVIDIA）和 Composable Kernel（AMD）作为 tile library，用户也可通过 T.call_extern 注册自定义 tile operator。TileLang 支持所有 Python 命令式构造（if-else, for, while），关键区别在于要求函数参数和变量声明显式类型标注（因 Python 动态类型不适合 device code generation）。

涉及论文标题：
- TileLang: A Composable Tiled Programming Model for AI Systems

---
