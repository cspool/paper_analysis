## Tensor Core (TC)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Tensor Core (TC) 是 NVIDIA GPU 中专用于矩阵乘累加（MMA）的硬件单元，自 Volta 架构（V100, 2017）开始引入。TC 执行单条指令完成 D = A × B + C 的分块矩阵运算，提供远超 CUDA Cores 的矩阵计算吞吐。以 A100 为例：TC FP16 达 312 TFLOPS vs CUDA Cores FP32 仅 19.5 TFLOPS（~16× 差距）。TC shape 由 mma_m × mma_n × mma_k 定义。现代 GPU TC shape 演进：Ampere m16n8k16 (FP16)、m16n8k8 (TF32)；Hopper 增加 m16n8k16 (FP64) 和 WGMMA m64nNk16；Blackwell 增加原生 MXFP4/NVFP4 支持。

从硬件架构角度拆解术语。

TC 在 BitDecoding 中的核心作用：
1. **TC 承担 attention 的矩阵乘法**（QK^T 和 PV），而 CUDA Cores 执行 dequantization
2. **Layout induction**：利用 TC 的 ldmatrix 指令隐式建立 interleaved fragment layout，使量化后数据可直接匹配 TC 寄存器期望
3. **TC utilization**：通过 warp parallelism strategy (Wm=1, Wn↑) 和 cooperative softmax 提升 TC 利用率。BitDecoding 实测 TC utilization 从 10.91%（W_n=1）提升至 19.66%（W_n=4）
4. **与 CUDA Core-only 的对比**：QServe/Atom 完全在 CUDA Cores 上执行（FMA），TC 利用率 ~0%，而 BitDecoding 将 matmul 卸载到 TC 后获得 3-8× speedup

术语一般如何实现？如何使用？

TC 通过 CUDA PTX 的 `mma.sync.aligned.m16n8k16` 等指令或 CUTLASS 的 `warp::mma` 抽象编程。低精度模式（INT8/INT4/FP8/MXFP4）提供更高吞吐。在 LLM 推理中，TC 主要用于 GEMM（QKV projection、FFN）和 attention 中的 QK^T、PV matmul。BitDecoding 证明了 TC 也可用于低比特 KV cache 场景中的 matmul。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs Decoding with Low-Bit KV Cache
