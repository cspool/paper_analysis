## NVFP4 / MXFP4 (Micro-Scaling FP4 Formats on Blackwell)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

MXFP4 (Microscaling FP4) 是一种由 Microsoft、AMD、NVIDIA、Intel 等联合提出的 4-bit 浮点微缩放格式。其核心思想：将同质数据块（如 32 个相邻元素）共享一个公共缩放因子（scale），单个元素值用 4-bit 浮点表示（E2M1 / E1M2 等格式）。计算时硬件用"scale × fp4_value"重建完整精度值。NVFP4 是 NVIDIA 的对应实现（Blackwell 原生的封闭格式），与 MXFP4 在 bit layout 上可能不同但理念一致。Blackwell Tensor Cores 原生支持 MXFP4/NVFP4 mma 指令，无需软件 dequantization——硬件直接读取 packed 4-bit 数据和 block-scale factor 后在 TC 内完成乘累加，消除了 KV cache 解码中的 dequantization 瓶颈。

从硬件架构角度拆解术语。

Blackwell 原生 MXFP4 attention 执行流程（以 BitDecoding on RTX 5090 为例）：
1. **数据存储**：KV cache 以 packed MXFP4 格式存储在 HBM。每 block（e.g., 32 个 4-bit 值）附带一个 FP8/E5M2 共享 scale factor
2. **数据加载**：TMA 或 cp.async 将 packed 4-bit data + block-scale 从 HBM 异步拷贝到 shared memory
3. **TC 执行**：Blackwell TC 的 MXFP4 mma 指令直接消费 packed 4-bit data（无需 dequantization），TC 内部自动用 block-scale 乘以 4-bit 值后完成乘累加
4. **关键性能**：Blackwell B200 MXFP4 TC 峰值可达 20 PFLOPS（FP16 的 4× 以上），BitDecoding on RTX 5090 达 8.6× speedup over FP16 FlashDecoding-v2

术语一般如何实现？如何使用？

MXFP4 格式由 OCP Microscaling Formats 规范定义（https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf）。NVFP4 是 NVIDIA 闭源实现。在 CUDA 中可通过 `cutlass` 3.x 使用 `cutlass::float_e2m1_t` 等类型。Blackwell GPU 推理时，将 KV cache 以 NVFP4 格式存储，调用原生 mxfp4 mma 指令。BitDecoding 在 Blackwell 上跳过 lop3-based dequantization，直接使用 NVFP4 路径。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs Decoding with Low-Bit KV Cache

---
