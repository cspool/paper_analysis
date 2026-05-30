## Implicit Fine-Grained Pipeline (ImFP)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Implicit Fine-Grained Pipeline (ImFP) 是LiquidGEMM提出的W4A8 GEMM kernel pipeline执行机制。采用single-producer multiple-consumer模型：一个专门的Load WG（Warp Group）通过TMA从GMEM加载weight到SMEM并切分为fine-grained tasks，多个Compute WG竞争获取这些tasks并各自完成dequantization+MMA。关键创新：(1) 同一Compute WG负责dequantization和MMA，消除ExCP（显式粗粒度pipeline）的SMEM↔RF round-trip数据搬运；(2) dequantization和MMA的overlap通过跨Compute WG实现（WG_0做dequant时WG_1做MMA），而非同一WG内串行；(3) task scheduling由硬件管理（atomic竞争获取），无需软件barrier同步。

从kernel调度角度拆解术语：
ImFP执行流程（每thread block = 3 WGs: 1 Load + 2 Compute）：
```
// Load WG (4 warps, TMA):
for each K_tile iteration:
    TMA: GMEM → SMEM[pong]  // async weight load (Dual-MMA packed)
    cp.async.bulk.wait_group
    for each MMA fragment in tile:
        smem_task_queue.push({frag_addr, frag_meta})  // metadata only
    swap(ping, pong)

// Compute WG_0 & WG_1 (各4 warps, CUDA + Tensor Cores):
while true:
    task = smem_task_queue.try_pop()  // atomic竞争, 无barrier
    if !task: break
    LDS.128: RF = SMEM[task.addr]  // 32 UINT4, single instruction
    unpack_4bit(RF)                 // 8 elem to 2 regs
    dequant_LQQ(RF)                 // IMAD + XOR, CUDA Cores
    WGMMA(C_frag, A_frag, RF)       // INT8 MMA, Tensor Cores
// WG_0和WG_1处理不同tasks，dequant与MMA自然跨WG重叠
```

与ExCP对比：ExCP需要Load WG→Dequant WG→MMA WG三阶段，Dequant WG从SMEM读到RF dequant后写回SMEM，MMA WG再读到RF做MMA——产生SMEM↔RF round-trip × 2和barrier同步开销。

术语一般如何实现？如何使用？
基于CUTLASS/Cute warp-specialized kernel框架实现。Task queue用SMEM中的metadata数组+atomic counter。每block 1 Load WG + 2 Compute WGs（12 warps = 384 threads）。配合Dual-MMA packed layout使LDS.128充分利用带宽。消融实验：ExCP在small batch退化（round-trip+sync开销），ImFP在所有batch size持续提升。

涉及论文标题：
- LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving

---
