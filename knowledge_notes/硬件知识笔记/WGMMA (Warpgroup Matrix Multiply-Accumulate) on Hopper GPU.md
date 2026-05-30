## WGMMA (Warpgroup Matrix Multiply-Accumulate) on Hopper GPU

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
WGMMA (Warpgroup Matrix Multiply-Accumulate) 是 NVIDIA Hopper 架构 (H100, SM90) 引入的新型异步 Tensor Core 指令。与 Ampere 的 wmma/mma 指令（每 warp 独立同步发射，发射后 stall 等结果）不同，WGMMA 将同一 CTA 内 4 个 warp 编组为 "warpgroup"（128 threads），单条 WGMMA 指令协作完成大规模矩阵乘法（如 m64n256k16 for FP16）。核心特性：(1) 异步执行——指令发射后立即返回（non-blocking），Tensor Core 在后台异步计算，warp 可继续发射后续指令；(2) 两种数据源模式——SS（A/B/C 均来自 Shared Memory）和 RS（A/C 来自 Register，B 来自 Shared Memory）；(3) FP8 支持——Hopper FP8 Tensor Core 提供 2× FP16/BF16 吞吐（~1978 TFLOPS FP8 vs 989 TFLOPS FP16 on H100）；(4) k-major layout constraint for FP8——FP8 WGMMA 仅支持 B 操作数在 K 维度连续（k-major），与 FP16 可接受 k-major 或 mn-major 不同。

从硬件架构角度拆解术语：
FlashAttention-3 中 WGMMA 的关键使用模式（以 BF16 forward 为例）：
1. SS-WGMMA for QK^T：Q_i (FP16, in SMEM, B_r×d) × K_j^T (FP16, in SMEM, d×B_c) → S_ij (FP32 accumulator in registers, B_r×B_c)。Q 和 K 均从 SMEM 读取（SS mode），accumulator 写入各线程的 register file。
2. RS-WGMMA for PV：P̃_ij (FP16, in registers, B_r×B_c) × V_j (FP16, in SMEM, B_c×d) → O_i_accum (FP32, B_r×d)。P̃ 来自 register file（RS mode），V 来自 SMEM。
3. FP8 WGMMA additional constraints：(a) Q/K 需 k-major 布局（contiguous in head dim），V 需 mn-major 布局（contiguous in seqlen dim），否则需 in-kernel transpose；(b) FP32 accumulator → FP8 operand layout 需通过 byte_perm + shfl_sync 进行 register data exchange 转换 ownership pattern。

WGMMA 的异步特性是 FlashAttention-3 实现 GEMM-softmax overlapped pipelining 和 pingpong scheduling 的硬件前提。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
WGMMA 通过 CUDA PTX 指令编程：`wgmma.fence`（fence 之前的 shared memory 写入）、`wgmma.commit_group`（将 WGMMA 加入异步 group）、`wgmma.mma_async`（发射异步 MMA）、`wgmma.wait_group N`（等待 group N 完成）。CUTLASS 3.x 提供 C++ abstraction（`WarpgroupMMA`、`Pipeline`），自动管理 fence/commit/wait 同步。使用时需注意 accumulator layout 与 Ampere MMA 不同——跨世代 portable kernel 需处理 layout 差异。

涉及论文标题：
- FlashAttention-3 Fast and Accurate Attention with Asynchrony and Low-precision
- FlashAttention-T: Towards Fully Tensorized Attention by Exploiting Tensor-Vector Parallelism
