## LLVM Compiler Hints (AMD GPU sched_barrier / sched_group_barrier / s_setprio)

术语是什么？
LLVM compiler hints 是 AMD GPU LLVM backend 提供的编译器内建函数（builtins），允许开发者在 HIP C++ 代码中向编译器传递指令调度提示，指导编译器如何排序和优先级化生成的 GCN/CDNA ISA 指令。三种关键 hint：(1) `__builtin_amdgcn_sched_barrier(mask)` — 建立指令簇之间的硬边界，mask 指定哪些类型的指令不可跨越该边界；(2) `__builtin_amdgcn_sched_group_barrier(mask, size, sync_id)` — 创建"super group"调度流水线，按 sync_id 分组，同 sync_id 的不同 instruction group 之间强制执行顺序；(3) `__builtin_amdgcn_s_setprio(0-3)` — 设置当前 wave 相对其他 wave 的硬件资源争用优先级。

从编译框架角度拆解术语：
这些 hint 在 AMD GPU kernel 开发中被用于精确控制指令发射顺序。以论文的 8-WAVE PING-PONG attention kernel 为例：
```
// Cluster 1: compute cluster, wave 优先级提升
__builtin_amdgcn_s_setprio(1);               // 提升当前 compute wave 优先级
mma_ABt(C_accum[0][0], A_tile, B_tile, ...);  // 执行 MFMA
__builtin_amdgcn_s_setprio(0);               // 恢复默认优先级
__builtin_amdgcn_sched_barrier(0);            // 硬边界，禁止指令跨边界重排
```
对于 mixed vector+matrix 指令的 attention kernel，`sched_barrier_pairs` 宏交替分组 MFMA 和 vector ops（如 softmax 的 exp2/sub/max），将 N 对 compute/memory 指令按 sync_id 分入不同 super group，确保 LLVM 按开发者意图交错发射而非完全自由排序。论文也指出限制：asm volatile 包裹的代码对编译器是黑盒，某些指令（如 v_cvt_pk_bf16_f32）缺少 LLVM builtins。

术语一般如何实现？如何使用？
LLVM compiler hints 通过 `<llvm/amdgcn>` 头文件提供的 `__builtin_amdgcn_*` 函数使用，在 HIP C++ kernel 源码中直接调用。mask 的常用位标志：MFMA_MASK (0x08)、VMEM_MASK (0x20)、DS_MASK (0x100) 等。Modular AI 的 GEMM kernel 当前依赖 sched_group_barrier，但需逐条指令考虑调度，而 HK 在 cluster 级别使用 hint + tile 原语组合，平衡可编程性和性能。

涉及论文标题：
- HipKittens: Fast and Furious AMD Kernels

---
