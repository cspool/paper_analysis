## buffer_load_dword (AMD Asynchronous Global-to-LDS Load)

术语是什么？
buffer_load_dword 是 AMD CDNA3/4 的异步 HBM→LDS（shared memory）加载指令，等价于 NVIDIA TMA。数据绕过 register file 直接写入 LDS，wave 可继续执行其他操作。变体：dword (4B)、dwordx3 (12B)、dwordx4 (16B)。通过 s_waitcnt vmcnt(N) 等待未完成的 load。

从kernel调度角度拆解术语：
在 HipKittens GEMM hotloop 中：
```
G::load(Bs[t][0], g.b, {0, 0, col*2, tile+2}); // 异步发射
// ... 执行 MFMA 和其他操作 ...
asm volatile("s_waitcnt vmcnt(6)");  // 等待直到 <=6 个 vm 操作未完成
__builtin_amdgcn_s_barrier();
```
全局地址的 swizzle 在 HBM 地址阶段完成（与 NVIDIA TMA 在 shared memory 地址上 swizzle 不同）。buffer_load_dwordx4 每个 thread 加载 16 bytes，最小化指令发射数但可能引起 shared memory alignment 问题（如 FP6 kernel 需 16-byte aligned ds_read_b128）。

术语一般如何实现？如何使用：
通过 HIP 内嵌汇编使用。截至 Sep 2025，Triton AMD 仍未默认使用 buffer_load（需 PR #8013 手动启用）。HipKittens 的 G::load 模板封装了指令选择、地址计算和 swizzle。

涉及论文标题：
- HipKittens: Fast and Furious AMD Kernels

---
