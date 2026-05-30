## setmaxnreg（动态寄存器重分配）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
setmaxnreg 是 NVIDIA Hopper 架构引入的 PTX 指令（§9.7.17.1），允许在同一个 CTA 内动态重新分配不同 warpgroup 的寄存器数量。传统 GPU 架构中，CTA 内所有 warps 均分 register file（如 H100 每 SM 256KB = 65536 个 32-bit registers，若 CTA 含 4 warps 则每 warp 分配 16384 regs / warp size 32 = 最多 512 regs/thread）。setmaxnreg 打破了这一限制——允许 programmer 指定某些 warps 持有更多 register（如 consumer warp 需要大量 register 存储 WGMMA accumulator和 S/P tiles），而其他 warps 释放 register（如 producer warp 仅需1个thread发射 TMA，register 需求极低）。FlashAttention-3 中 producer warpgroup 使用 setmaxnreg.dealloc 释放 register 给 consumer warpgroups 使用，consumer 使用 setmaxnreg.alloc 获取更多 register 以支持更大的 tile size 和更深 pipeline。

从kernel调度角度拆解术语：
FlashAttention-3 中 setmaxnreg 使用流程：
```
// CTA内有5个warps: 1 producer warp + 4 consumer warps (2 warpgroups)
if (warpgroup == PRODUCER):
    setmaxnreg.dealloc(32)   // 释放大量register（TMA只需1 thread, ~32 regs足够）
    // 执行 TMA load pipeline...
else if (warpgroup == CONSUMER_0 or CONSUMER_1):
    setmaxnreg.alloc(255)    // 申请更多register用于WGMMA + S/P tile缓冲
    // 执行 GEMM-softmax pipeline...
```
效果：Consumer warpgroups 可获得接近255 registers/thread（Hopper上限），从而支持更大的 $B_r \times B_c$ tile 和 2-stage pipeline 所需的额外 $\mathbf{S}_{\text{next}}$ 缓冲。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
setmaxnreg 通过 CUDA PTX 内联汇编或 CUTLASS 3.x 的 warp-specialized pipeline abstractions 使用。需注意：(1) dealloc 必须在 warp 执行任何使用寄存器之前调用，alloc 必须在实际使用额外寄存器之前调用；(2) 总寄存器使用量不能超过SM的物理register file（256KB/SM on H100）；(3) 过度 dealloc 可能导致 producer warp 寄存器不足而 spill 到 local memory（L1 cache），造成性能损失。CUTLASS 3.x 的 Pipeline 抽象自动管理 setmaxnreg 调用。

涉及论文标题：
- FlashAttention-3 Fast and Accurate Attention with Asynchrony and Low-precision
