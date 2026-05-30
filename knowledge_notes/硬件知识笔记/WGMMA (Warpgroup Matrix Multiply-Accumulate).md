## WGMMA (Warpgroup Matrix Multiply-Accumulate)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

WGMMA 是 NVIDIA Hopper 架构（H100, SM90, Compute Capability 9.0）引入的新型 Tensor Cores 指令。将同一 CTA 内多个 warp 编组为 "warpgroup"（通常 4 个 warp = 128 threads），warpgroup 协作执行单条 wgmma 指令完成大规模矩阵乘法。核心创新：异步执行模型——wgmma 指令发射后立即返回（non-blocking），TC 在后台异步执行乘累加，warp 可继续发射后续指令。两种数据源模式：wgmma_RS（A 和 C 来自 register，B 来自 shared memory）、wgmma_SS（A、B、C 均来自 shared memory）。

从硬件架构角度拆解术语。

BitDecoding 中 WGMMA 的 Hopper 执行流程：
1. **Warp specialization**：部分 warps 负责 ldmatrix + dequantization + STSM（将 dequantized FP16 K/V 写入 shared memory），部分 warps 负责 wgmma computation
2. **异步 overlap**：Warp 发射 wgmma 后不 stall——可立即开始下一 tile 的 ldmatrix 和 dequantization，与 TC 后台计算重叠
3. **共享内存通信**：dequantized data 经 STSM→shared memory→wgmma_SS 消费，无需 barrier（得益于 wgmma 异步特性）
4. **性能**：H100 上 WGMMA FP16 m64nNk16 达 989 TFLOPS；BitDecoding-v3 在 H100 上达 8.0× speedup（v2 仅 4.1×，v3 增加 WGMMA 异步）

术语一般如何实现？如何使用？

CUDA 中通过 `__hmma_mma_sync` 内联 PTX 或 cutlass 3.x 使用 WGMMA。异步同步通过 `wgmma.commit_group` 和 `wgmma.wait_group` 管理。与 Ampere mma 的关键区别：Ampere mma 同步阻塞（warp stall 等结果），WGMMA 异步非阻塞。BitDecoding 在 Hopper 版本中利用此实现 dequantization-TC computation overlap。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs Decoding with Low-Bit KV Cache

---
