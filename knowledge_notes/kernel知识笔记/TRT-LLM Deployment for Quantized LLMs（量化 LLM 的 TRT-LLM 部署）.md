## TRT-LLM Deployment for Quantized LLMs（量化 LLM 的 TRT-LLM 部署）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TRT-LLM（TensorRT-LLM）是 NVIDIA 的 LLM 推理优化框架，提供高性能 GPU kernel（特别是量化 GEMM）和完整的推理运行时。对于 INT4 量化场景，TRT-LLM 通过 CUTLASS INT4 GEMM kernel 实现对 W4A4 和 W4A8 矩阵乘法的硬件加速（利用 Tensor Core 的 INT4 MMA 指令如 mma.sync.aligned.m16n8k32）。FlatQuant 的推理栈使用 CUTLASS INT4 kernel 执行量化矩阵乘法（权重和激活均已量化为 INT4），并在 prefill 和 decoding 阶段通过 CUTLASS 的 tiling、software pipeline 和 epilogue fusion 实现高效推理。此外 FlashInfer 库用于 KV cache 量化操作。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
FlatQuant 中 CUTLASS INT4 GEMM 的推理流程：

```
输入:
  - A_q [M, K]: INT4 激活 (来自融合仿射量化 kernel 的输出)
  - B_q [N, K]: INT4 权重 (离线预量化)
  - scale_A [M]: per-token FP16 scale
  - scale_B [N]: per-channel FP16 scale

输出: Y [M, N] FP16

CUTLASS INT4 kernel 内部流程:
1. 从 Global Memory 加载 A_q tile, B_q tile → Shared Memory
2. 从 Shared Memory → Register (INT4 packed)
3. mma.sync.aligned.m16n8k32: 在 Tensor Core 上执行 INT4 MMA
   累积 int32 → 输出到 Register
4. Epilogue: int32 → FP16 反量化 (乘以 scale_A × scale_B)
5. Write Y tile → Global Memory
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CUTLASS 以 C++ 模板库形式提供（https://github.com/NVIDIA/cutlass），用户通过模板参数配置 tile size、warp tile、MMA 指令、pipeline stage 数等。FlatQuant 直接调用 CUTLASS INT4 kernel 进行量化推理。在开源实现中，通过 PyTorch 的 custom op 机制（torch.library）将 CUTLASS kernel 封装为 Python 可调用函数。

涉及论文标题：
- FlatQuant: Flatness Matters for LLM Quantization

---
