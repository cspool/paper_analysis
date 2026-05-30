## 8-WAVE PING-PONG Schedule (AMD)

术语是什么？
8-WAVE PING-PONG 是 HipKittens 论文提出的 AMD GPU kernel 调度的核心模式之一。它在一个 thread block 中使用 8 个 wave——每 SIMD 驻留 2 个 wave。8 个 wave 分为两组各 4 个（每组包含每个 SIMD 各 1 个 wave），通过 conditional barrier 交替执行 compute 和 memory 角色：组 A 的 wave 发射 MFMA 计算指令时，组 B 的 wave 发射 buffer_load_dword 从 HBM 预取下一 tile 到 LDS；完成后角色互换（ping-pong）。

从kernel调度角度拆解术语：
以 BF16 GEMM kernel (256x256 output tile per thread block, K_STEP=64) 的调度流程为例：

```
// Prologue: 8 waves 协作 preload
G::load(Bs[t0][0], g.b, {0, 0, col*2, 0});
G::load(As[t0][0], g.a, {0, 0, row*2, 0});
// 条件 barrier: 一半 wave 被 stall
if (warp_row == 1) { __builtin_amdgcn_s_barrier(); }
// Leader wavegroup 继续 preload, 完成后释放 follower
// ...
__builtin_amdgcn_s_barrier();

// Hotloop: ping-pong 交替
for (tile = 0; tile < num_tiles - 2; ++tile) {
    // Leader compute cluster (follower 同时做 memory):
    load(B_tile_0, st_subtile_b);       // LDS to register
    load(A_tile, st_subtile_a);
    G::load(As[t_next][1], ...);        // async HBM to LDS
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_s_setprio(1);      // 提升 compute wave 优先级
    mma_ABt(C[0][0], A_tile, B_tile_0);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    
    // 下一阶段: 角色交换 (Leader 做 memory, follower 做 compute)
    // 对称的 load+mma 操作
}
```

8-WAVE 允许使用大 tile 原语（类似 NVIDIA wave specialization），代码紧凑（GEMM hotloop 约 48 LOC），适合 compute 和 memory 持续时间平衡的 workload。在 MI355X 上，8-WAVE BF16 GEMM 达到 1610 TFLOPS，FP8 GEMM 达到 3222 TFLOPS，均匹敌 AITER 手写汇编。

术语一般如何实现？如何使用？
HipKittens 通过 C++ template 调度模板实现，开发者设置每 SIMD 2 个 wave，在 conditional stagger 后用 s_barrier 交替。适合 GEMM、attention forward 等 compute-memory 平衡 kernel。对 attention backwards 等 memory-heavy workload，4-WAVE INTERLEAVE 更优。

涉及论文标题：
- HipKittens: Fast and Furious AMD Kernels

---
