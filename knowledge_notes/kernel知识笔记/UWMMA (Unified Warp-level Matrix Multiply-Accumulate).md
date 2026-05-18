## UWMMA (Unified Warp-level Matrix Multiply-Accumulate)

术语是什么？

UWMMA 是 Uni-STC 论文提出的统一 warp 级矩阵乘累加指令序列接口，用于替代传统 GPU dense tensor core 的 MMA 指令。UWMMA 由三类指令组成——stc.load（同步收集 BBC 格式的 metadata 和数值）、stc.task（异步触发 TMS/DPG 硬件生成 task queue）、stc.numeric（检查 READY/BUSY 状态驱动 SDPU 执行）——将稀疏计算的控制流从"gather data"转为"gather tasks"模式。与传统 MMA 指令要求 A/B 数据在调用前完全就绪不同，UWMMA 的异步 task generation 允许 task 分解与 SM 后续操作重叠执行。

从kernel调度角度拆解术语：

UWMMA 指令序列的伪代码（以 SpMM 为例）：
```
// 一个 warp 执行 SpMM C[i_row_batch][j_col] = A[i_row_batch][k] × B[k][j_col]
// 假设 A 以 BBC 格式存储，B 以 BBC 格式存储

warp:
  // Phase 1: Load
  stc.load.meta rA_meta, [A_bbc_base + row_meta_off]   // load A 的 top-level CSR tile metadata
  stc.load.val  rA_vals, [A_bbc_base + val_off]         // load A 的 tile values
  stc.load.meta rB_meta, [B_bbc_base + col_meta_off]    // load B 的 tile metadata
  stc.load.val  rB_vals, [B_bbc_base + val_off]         // load B 的 tile values
  stc.load.meta rC_meta, [C_bbc_base + row_meta_off]    // load C 的 tile metadata (用于 T4 code 中编码 C 写入目标)

  // Phase 2: Task Generation (async)
  stc.task.gen.mm                                      // 触发 TMS → DPG task generation pipeline
  // TMS: 从 Meta Buffer 读 top-level bitmap → 沿 K 维匹配 A/B tile
  //      → 拆分 16×16×16 T1 为多个 4×4×4 T3 → 写入 Tile queue
  // DPG: 从 Tile queue 弹 T3 → 读 bottom-level bitmap → overlay A/B bitmap
  //      → 生成 1×1×4 T4 task code → 以 Z-shaped 顺序入 Dot-product queue

  // Phase 3: Numeric Execution
  check_ready:
    stc.poll.ready r_status                              // 检查 Dot-product queue 是否有 READY T4 task
    if r_status == BUSY: goto check_ready               // task generation in progress, stall

  stc.numeric.mm                                        // 驱动 SDPU 执行
  // SDPU: 弹出 T4 task code → 解码 A/B 操作数地址
  //      → 执行 1×1×4 segmented dot-product → 累加到 accumulator
  //      → merge-forward 合并 partial products → 写 C

  // Phase 4: Write-back
  stc.writeback rC_vals, [C_bbc_base + val_off]         // 更新 C 的 BBC value 区
```

指令设计的关键特性：
- **异步 task generation**：stc.task 提交后 warp 可继续执行其他指令，通过 stc.poll.ready 检查状态
- **统一 opcode**：同一套 load/task/numeric 指令覆盖 SpMV、SpMSpV、SpMM、SpGEMM，仅在 `.gen.mm`/`.gen.mv`/`.gen.spspv`/`.gen.gemm` 后缀上有区分
- **软件控制 BBC 数据路径**：ValPtr_Lv2 直接提供给 Uni-STC，使 TMS 能控制 tile 内数据转发而无需复杂硬件解码

术语一般如何实现？如何使用？

UWMMA 需要 GPU SM 侧扩展：instruction decoder 新增 opcode 解析、warp scheduler 支持 UWMMA 指令分发。软件编译时，程序员通过类似 CUDA PTX inline assembly 的方式嵌入 UWMMA 指令序列。数据路径要求 register file 提供足够 operand port（Ampere 类需每线程每周期最多 16 个 FP64 source + 4 个 FP64 destination operands）。BBC 格式矩阵的 ValPtr_Lv2 指针在 stc.load 时直接传入 Uni-STC 硬件 buffer，后续 task generation 通过指针索引而非逐元素搬运。

涉及论文标题：
- Uni-STC: Unified Sparse Tensor Core

