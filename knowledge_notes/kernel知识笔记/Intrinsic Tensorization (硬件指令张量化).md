## Intrinsic Tensorization (硬件指令张量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Intrinsic Tensorization 是 TileLang 中利用 GPU 硬件特殊指令（Tensor Cores mma、DP4A、IMAD 等）进行高性能矩阵运算的机制。现代 GPU 提供多种精度和吞吐量的指令路径：例如 NVIDIA RTX 3090 上，IMAD（标量融合乘加）≈ 17.8 TOPS，DP4A（向量化 4 元素点积）≈ 71.2 TOPS，MMA（Tensor Core 矩阵乘法）≈ 284 TOPS。选择合适的指令取决于输入 shape 和 data type。TileLang 提供两种互补的 Tensorization 方式：(1) Tile Library-based — 通过 T.call_extern 调用 CUTLASS cute (NVIDIA) 或 Composable Kernel (AMD) 的 tile API（如 cute::gemm_ss），自动选择最优指令；(2) Direct PTX/C++ injection — 通过 T.ptx 直接发射内联 PTX 指令（如 mma.m16n8k32.row.col.s32.s8.s32），或通过 T.import_source + T.call_extern 注入 C++ 模板实现的 DP4A/IMAD 等指令。

从 kernel 调度角度拆解术语，比如术语所在 kernel 调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

TileLang 中两种 Tensorization 方式的对比：
```
// 方式 1: Tile Library-based (默认，图 10c)
T.gemm(A_shared, B_shared, C_local)
// ↓ Lower to:
T.call_extern("cute::gemm_ss", 
  A_shared[SwizzleLayout], B_shared[SwizzleLayout], C_local[MMA_MatrixLayout])
// ↓ Codegen to CUDA C:
cutlass::gemm_ss<128, 128, 32, 2, 2>(AShared, BShared, C_local);
// CUTLASS 内部自动选择 mma.m16n8k32 或 wgmma.m64n64k16 等最优指令

// 方式 2: Direct PTX injection (图 10b，专家使用)
T.ptx("mma.m16n8k32.row.col.s32.s8.s32", 
      C_local_regs, A_regs, B_regs, C_local_regs)
// ↓ 直接生成 PTX:
asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s32.s32 "
             "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
             : "=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3)
             : "r"(a0),"r"(a1), "r"(b0), "r"(c0),"r"(c1),"r"(c2),"r"(c3));

// 方式 3: C++ Source Injection (图 10a，低精度场景)
T.import_source("dp4a_kernel.cuh")  // 注入 C++ 模板实现
T.call_extern("dp4a_kernel<int8_t>", A, B, C)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

默认使用 Tile Library-based 方式（简单、vendor-optimized 性能），但其局限是：(1) cute::gemm_ss 内部管理 shared→register data flow，外部无法 annotate/override layout；(2) CUTLASS 模板展开占约 90% TileLang 编译时间（NVCC 12.8 trace 验证）。Direct PTX/Injection 方式提供完全控制但需用户为每种硬件指令实现完整的指令集封装。TileLang 计划未来构建 self-hosting Tile Library 用 TileLang 自身替代 CUTLASS 依赖，同时保留 Direct Injection 路径。对于 AMD GPU，TileLang 使用 Composable Kernel (CK) 和手写 HIP 代码。

涉及论文标题：
- TileLang: A Composable Tiled Programming Model for AI Systems
