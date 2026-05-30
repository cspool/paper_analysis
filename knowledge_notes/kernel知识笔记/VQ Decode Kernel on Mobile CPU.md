## VQ Decode Kernel on Mobile CPU

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VQ Decode Kernel on Mobile CPU 是 GPTVQ 推理引擎中负责将 VQ 压缩权重在线解码为 native compute data type 的软件 kernel。它的设计目标是解码延迟低于 DRAM 带宽，使 VQ footprint 减小转化为实际的 token rate 提升。kernel 流程：从 DRAM 加载 block tuple（packed 6-bit indices + 64-entry INT8 LUT + FP16 scale）→ 进入 CPU cache → 解包 6-bit indices → TBL 指令查表（每维一次）→ 合并两维 → scale 反量化 → SIMD GEMM。在 Snapdragon X Elite 上实测：VQ 2D 3.125 bpv 解码延迟 = 0.96× vs INT4 数据传输（Table 6），端到端 token rate 26.15 tok/s（+10% vs Ours INT4, +45.7% vs llama.cpp INT4），footprint 3.52GB（-19% vs INT4 4.33GB）。

在 NVIDIA GPU（RTX 3080）上也实现了 VQ decode kernel，使用 CUDA vector types（char4/uchar4 和自定义 char128 agglomeration）并行加载和解码。VQ 4D 2.125 bpv 在 GPU 上实现相对 footprint 0.53× + 相对延迟 0.71×（vs INT4），即同时减小 footprint 和延迟。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
CPU 端 VQ decode + GEMM 完整调度流程：

```
Sequence Diagram: CPU VQ Decode + GEMM (per Transformer layer)
┌─────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  DRAM   │     │ CPU Cache│     │ NEON Regs│     │ SIMD ALU │
└────┬────┘     └─────┬────┘     └─────┬────┘     └─────┬────┘
     │                │                │                │
     │ Block Load ───>│                │                │
     │ (indices+LUT   │                │                │
     │  + scale)      │                │                │
     │                │ TBL LUT ──────>│                │
     │                │ load (dim0)    │                │
     │                │                │ TBL dim0 ─────>│
     │                │                │ (6b→8b)        │
     │                │                │<───────────────│
     │                │ TBL LUT ──────>│                │
     │                │ load (dim1)    │                │
     │                │                │ TBL dim1 ─────>│
     │                │                │ (6b→8b)        │
     │                │                │<───────────────│
     │                │                │ ADD dim0+dim1─>│
     │                │                │<───────────────│
     │                │                │ MUL scale ────>│
     │                │                │ (int→fp16)     │
     │                │                │<───────────────│
     │                │                │                │
     │                │                │ SIMD GEMM ────>│
     │                │                │ (fp16 weights  │
     │                │                │  × fp16 act)   │
     │                │                │<───────────────│
     │                │                │                │
     │<───────────────│ output write   │                │
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
GPTVQ 的推理引擎是 Qualcomm 自研的 C 语言实现：(1) 使用 ARM NEON intrinsics 实现 TBL 查表和 SIMD GEMM；(2) 利用 polyhedral compiler（Polly）进行细粒度循环向量化优化；(3) 粗粒度并行利用 transformer 的结构特性（如 multi-head attention 的 head 间并行）。移动端部署在 Snapdragon X Elite（Windows + Clang 18.1）。GPU kernel 在 CUDA 上实现，针对 RTX 3080+ 验证。代码尚未开源（论文声明 "will be made available in the future"）。关键设计原则：解码延迟必须 < DRAM 带宽节省的延迟，即 (T_decode - T_saved_bandwidth) < 0。GPTVQ 的 2D VQ 配置（6-bit index, 64-entry codebook）经过与 TBL 指令协同设计，确保此条件成立。

涉及论文标题：
- GPTVQ: The Blessing of Dimensionality for LLM Quantization

---
