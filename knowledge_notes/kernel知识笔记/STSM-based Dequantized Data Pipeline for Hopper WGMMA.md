## STSM-based Dequantized Data Pipeline for Hopper WGMMA

术语是什么？通过联网搜索让回答具体和精准。

STSM (Store to Shared Memory) 是 NVIDIA Hopper 架构新增的 PTX 指令，允许将 register 中的数据直接写入 shared memory 而不经过 global memory。BitDecoding 利用 STSM 为 Hopper 的 WGMMA 建立 dequantized data pipeline：WGMMA 的 wgmma_RS 模式要求 B 矩阵在 shared memory 中，而 dequantized K/V 通常在 register 中。BitDecoding 用 STSM 将 dequantized FP16 K/V 从 register store 到 shared memory → wgmma_RS 直接从 shared memory 读取 B 矩阵。由于 WGMMA 是异步执行的（non-blocking），STSM 的 shared memory 写入与 Tensor Cores 的 wgmma 计算可以自然 overlap——这一特性被 BitDecoding 用于实现 register→shared→wgmma 的无缝 data flow。

从 kernel 调度角度拆解术语：

```
// === STSM + WGMMA Pipeline (Hopper H100 decode step) ===
// 目标: 在Hopper上高效执行dequantized K × Q^T

// Step 1: ldmatrix加载packed low-bit K到register
ldmatrix.sync.aligned.m16n8k16.shared.b16 reg_K_packed, [smem_K_packed];

// Step 2: Dequantization in CUDA Cores (register)
for each thread:
    reg_K_fp16[thread_idx] = lop3_remap_dequant(reg_K_packed[thread_idx], scale, zero);

// Step 3: STSM将dequantized FP16 K写入shared memory
// (替代Ampere上需要的shared memory barrier + ldmatrix reload)
stsm [smem_K_fp16], reg_K_fp16;  // register → shared memory
__syncwarp();  // 保证STS完成

// Step 4: WGMMA读取shared memory作为B矩阵
// wgmma_RS: A=Q(register), B=K_fp16(shared memory), C=accumulator(register)
wgmma.fence;
wgmma.m64n64k16 C_acc, Q_reg, smem_K_fp16;
wgmma.commit_group;

// Step 5: 异步等待WGMMA完成
wgmma.wait_group 0;

// 关键：STSM与上一轮WGMMA可异步重叠
//  producer: STSM写下一块K_deq到shared memory
//  consumer: WGMMA消费当前shared memory中的K_deq
```

术语一般如何实现？如何使用？

STSM 通过 CUDA inline PTX 实现（`asm volatile("stsm ...")`）。在 BitDecoding 的 Hopper 版本中，Packing Kernel 内部通过 warp-specialized pipeline：部分 warps 负责 ldmatrix + dequantization + STSM，部分 warps 负责 wgmma computation。两者通过 shared memory 交替 ping-pong buffer 通信，无需 barrier（得益于 wgmma 的异步特性）。对比 Ampere 架构（无 WGMMA, 无 STSM）：dequantized data 保留在 register 直接参与 mma，无 STSM step——但 mma 是同步阻塞的，无异步 overlap 机会。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache

