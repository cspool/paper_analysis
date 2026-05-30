## NVIDIA Ampere Architecture (NVIDIA Ampere 架构)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NVIDIA Ampere 是 NVIDIA 的第八代 GPU 微架构（2020 年发布，SM 8.0），代表产品包括 A100、A40、RTX 3090 等。Ampere 在 AI 推理方面引入了多项关键创新：(1) **第三代 Tensor Cores**：原生支持 TF32/FP16/BF16/INT8/INT4/INT1 混合精度计算，以及 2:4 结构化稀疏（Sparse Tensor Cores）；(2) **Multi-Instance GPU (MIG)**：将单个 A100 切分为最多 7 个独立 GPU 实例；(3) **HBM2e 内存**：A100 最高 80GB，带宽 2 TB/s；(4) **PCIe 4.0 + NVLink 3.0**；(5) **INT4 sparse GEMM**：Ampere 是首个硬件原生支持 INT4 + 2:4 sparse 组合的架构，使 LLM 推理的联合压缩技术具备实际硬件加速价值。OBR 实验在 A100-SXM4-80GB 上进行。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
A100 SM 内部结构（与 LLM 推理相关部分）：

```
A100 SM (Streaming Multiprocessor) 内部:
┌─────────────────────────────────────────────┐
│  Instruction Cache / Warp Scheduler (4×)     │
├─────────────────────────────────────────────┤
│  Register File (65536 × 32-bit)             │
├──────────────────┬──────────────────────────┤
│  INT32/FP32 CUDA │  Tensor Cores (4×)       │
│  Cores (64×)     │  - FP16/BF16/TF32/INT8/  │
│                  │    INT4/INT1              │
│                  │  - 2:4 Sparse 支持        │
├──────────────────┴──────────────────────────┤
│  Shared Memory (164 KB / SM)                │
│  L1 Data Cache (192 KB, configurable)       │
└─────────────────────────────────────────────┘

// 内存层次:
// L1 Cache (192 KB/SM) → L2 Cache (40 MB shared)
// → HBM2e (80 GB, 2 TB/s)
```

A100 上 INT4 2:4 sparse GEMM 的执行路径：
1. HBM2e 加载 INT4 packed W + metadata + INT4 A → L2 Cache
2. L2 → L1/Shared Memory（每个 threadblock tile 的数据切片）
3. Shared → Register → Tensor Core (mma.sp.sync)
4. Tensor Core 输出 FP32 accumulators → Shared/Register → HBM2e

A100 关键吞吐指标（FP16 dense vs sparse）：
- 峰值 FP16: 312 TFLOPS（dense）→ 624 TFLOPS（sparse, 理论）
- 峰值 INT8: 624 TOPS → 1248 TOPS（sparse）
- 内存带宽: 2039 GB/s (80GB version)

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
A100 可通过云服务（AWS p4d, GCP a2-megagpu, Azure ND A100 v4）或本地部署（DGX A100 8-GPU）使用。编程：CUDA 11.0+ toolkit，CUTLASS 2.x，TensorRT 8.x+。LLM 推理框架（vLLM, TensorRT-LLM）针对 A100 优化了 INT4 和 sparse kernel。OBR 的推理效率实验在单张 A100-SXM4-80GB 上进行，比较了 INT4 2:4 sparse vs INT4 dense vs FP16 dense。

涉及论文标题：
- Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs
