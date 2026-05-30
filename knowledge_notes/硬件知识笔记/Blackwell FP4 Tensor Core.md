## Blackwell FP4 Tensor Core

术语解释
NVIDIA Blackwell 架构 GPU 中第五代 Tensor Core 原生支持的 4-bit 浮点矩阵乘累加（MMA）硬件单元，可直接对 MXFP4/NVFP4 格式执行低精度 GEMM，反量化融合在 MMA 指令内部完成。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Blackwell FP4 Tensor Core 是 NVIDIA Blackwell 架构（SM 10.0+，RTX 5070Ti/5090/PRO 6000 及 B200 数据中心 GPU）的第五代 Tensor Core 新功能。核心指令 `am16n8k64` 支持对 4-bit 浮点数据（MXFP4, NVFP4, INT4）执行矩阵乘法，每周期处理 16×8×64 tile。关键特性：(1) 原生支持 MX block-scaled 格式——MMA 指令直接接受带 E8M0 scale 的 MXFP4 元素，反量化融合在 Tensor Core 内完成；(2) FP4 吞吐为 FP16 的 4×、FP8/INT8 的 2×；(3) 专用 Tensor Memory (TMEM) 靠近计算单元缓存 tile 数据；(4) B200 单 GPU FP4 峰值 20 PFLOPS。

从硬件架构角度拆解术语，给出在硬件架构中运转流程的具体例子。
MicroMix GEMM kernel 中 Blackwell FP4 Tensor Core 运转流程：
```
1. 数据加载：DMA 从 HBM 加载 MXFP4 激活/权重 tile (E2M1元素 + E8M0 scale) 到 TMEM/Shared Mem
2. MMA 执行（Tensor Core 内部）：
   a. 读取 A_frag/B_frag 的 4-bit 元素 + E8M0 scale
   b. 反量化：A_val = A_elem × 2^{scale_A}（纯移位，E8M0 特性）
              B_val = B_elem × 2^{scale_B}
   c. 乘法：A_val × B_val → 中间乘积
   d. 累加：与 FP32 accumulator 相加
   e. 输出 BF16/FP32 tile 到 register file
3. 写回：结果 tile → shared memory → HBM
```
对比非 Blackwell：传统 INT4 kernel（Atom/QuaRot）INT8 Tensor Core MMA → INT32 部分和 → CUDA Core 额外 dequant（乘 scale + 类型转换），CUDA Core 成瓶颈。Blackwell 融合 dequant 消除此瓶颈。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
通过 CUDA PTX (`mma.sp.sync.aligned` 变体)、CUTLASS 3.x+ MXFP4 GEMM 模板、或 TensorRT-LLM/vLLM 高级库间接访问。MicroMix 利用 CUTLASS 模板实例化 MXFP4/6/8 MMA pipeline，解耦设计支持任意混合精度比例。适用场景：LLM 推理加速（2.29-3.38× vs FP16）、低精度训练（ViT FP4 训练）、极限吞吐 AI 工作负载。限制：小矩阵加速比有限；FP4 需混合精度策略保留关键通道精度。

涉及论文标题：
- MicroMix Efficient Mixed-Precision Quantization with Microscaling Formats for Large Language Models
