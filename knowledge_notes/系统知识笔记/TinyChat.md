## TinyChat

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TinyChat 是 MIT HAN Lab 为 AWQ 量化 LLM 设计的轻量级端侧推理系统。它是一个 PyTorch 前端 + 设备特定后端（CUDA/PTX、ARM NEON、x86 AVX）的推理框架，专为 W4A16 weight-only 量化场景优化。核心设计：(1) On-the-fly dequantization——INT4→FP16 反量化融合进 GEMM/GEMV kernel 主循环，避免将反量化权重写回 DRAM；(2) SIMD-aware weight packing——针对 ARM NEON 128-bit 和 x86 AVX 做交错排布以高效利用 SIMD 解包；(3) Kernel fusion——将 LayerNorm 的所有操作、QKV 三个投影、attention 计算与 KV cache 更新融合为极少 kernel launch。与 llama.cpp/exllama 等只能支持 LLaMA 系列的系统不同，TinyChat 通过 PyTorch 前端复用同一套 forward pass 代码支持 Llama-2、OPT、Falcon、MPT、Mistral、StarCoder、StableCode、VILA 等多种模型架构。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
TinyChat 在 RTX 4090 上运行 Llama-2-7B W4A16 的一次 decode token 生成全过程：
```
输入: 1 token → Embedding Lookup (FP16)

┌── TinyChat Decoder Layer (32 层) ─────────────────────────────────┐
│                                                                    │
│  [Kernel 1: Fused RMSNorm]                                        │
│  hidden_state → RMSNorm kernel (fused multiply/division/sqrt)     │
│  输出: normed_hidden [4096] FP16                                   │
│                                                                    │
│  [Kernel 2: Fused QKV Projection + RoPE]                          │
│  packed W_Q [4096×4096 INT4] + scales → on-the-fly dequant (GEMV)│
│  packed W_K [4096×4096 INT4] + scales → GEMV                      │
│  packed W_V [4096×4096 INT4] + scales → GEMV                      │
│  → Q [128×32] FP16, K [128×32] FP16, V [128×32] FP16             │
│  → RoPE applied on-the-fly (fused into kernel)                    │
│                                                                    │
│  [Kernel 3: Fused Attention + KV Cache Update]                    │
│  Q @ K_cache^T → scores → softmax → scores @ V_cache              │
│  → new K, V 写入预分配 KV cache → attention output [4096] FP16   │
│                                                                    │
│  [Kernel 4: Fused Output Projection (GEMV)]                       │
│  packed W_O [4096×4096 INT4] + scale → on-the-fly dequant GEMV   │
│  → residual add → hidden_state [4096] FP16                        │
│                                                                    │
│  [Kernel 5: Fused RMSNorm + Gated MLP]                            │
│  RMSNorm → packed W_gate + W_up + W_down (3 组 INT4 权重)         │
│  → gate: SiLU ⊙ up → W_down GEMV                                  │
│  → residual add → layer output [4096] FP16                        │
└────────────────────────────────────────────────────────────────────┘

→ LM Head (FP16 Linear) → Sampling → 输出 token
```

关键架构决策：
- 每个 Transformer Block 仅 ~5 次 kernel launch（vs HuggingFace PyTorch 的数十次，每次 launch overhead ~0.01ms on 4090）
- INT4 权重读取量为 FP16 的 1/4，decode 阶段 arithmetic intensity 从 ≈1 → ≈4 FLOPs/Byte
- KV cache 直接预分配为连续 GPU memory，更新在 attention kernel 内完成（无额外 memory copy）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TinyChat 开源：https://github.com/mit-han-lab/llm-awq/tree/main/tinychat。使用方式：
```python
from tinychat.models import LlamaForCausalLM
model = LlamaForCausalLM.from_pretrained("path/to/awq_quantized_llama2_7b")
output = model.generate("Hello, world!", max_new_tokens=200)
```
支持平台与加速比（vs HuggingFace FP16）：
- RTX 4090 (桌面): 3.2-3.9× (Llama-2-7B: 194 vs 52 tokens/s)
- RTX 4070 (笔记本 8GB): 可运行 Llama-2-13B @ 33 tokens/s (FP16 下 7B 都无法装入)
- Jetson Orin (移动 64GB): 3.5× (可运行 Llama-2-70B)
- Raspberry Pi 4B: Llama-7B @ 0.7 tokens/s (极边缘设备)

对比其他系统（Jetson Orin）：TinyChat 比 llama.cpp 快 1.7×，比 AutoGPTQ 快 2.6-3.0×。TinyChat 的设计原则是"最小化 kernel 数量 + 最大化 memory bandwidth 利用率"，后端仅实现 attention、layer norm、linear projection 三类 kernel，保持代码简洁（vs FasterTransformer 的完整重写）。已被 Apache TVM 社区的 MLC-LLM 项目借鉴（同期工作）。

涉及论文标题：
- AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FasterTransformer 是 NVIDIA 开发的高性能 Transformer 推理框架，提供针对 GPU 高度优化的 Transformer 层前向实现。它不是一个完整的 Serving 系统，而是一个推理后端/库，提供了 Transformer 各组件（Attention、FFN、LayerNorm、Beam Search 等）的 fused CUDA kernel 实现。FasterTransformer 后被整合到 NVIDIA TensorRT-LLM 中（2023 年）。AFPQ 论文使用 FasterTransformer 作为推理系统的后端，在其上实现了 NF4-asym 的自定义 dequantization kernel，以评估非对称 FP 量化对推理延迟的实际影响。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
AFPQ 基于 FasterTransformer 的推理系统流程（W4A16 NF4-asym）：
```
输入请求 → Tokenizer → Embedding Lookup (FP16)
  │
  ▼
┌─────────────────────────────────────────┐
│ FasterTransformer Decoder Stack (N 层)  │
│ ┌─────────────────────────────────────┐ │
│ │ 1. FP16 Activation 输入              │ │
│ │ 2. LayerNorm (FP16 fused kernel)    │ │
│ │ 3. Q/K/V 投影:                      │ │
│ │    - packaged NF4 weights → LUT     │ │
│ │    - scale_pos/scale_neg dequant    │ │
│ │    - FP16 GEMM with activation      │ │
│ │ 4. Attention (fused MHA kernel)     │ │
│ │ 5. O 投影 (同步骤 3 的 dequant+GEMM)│ │
│ │ 6. Residual + LayerNorm             │ │
│ │ 7. FFN (FC1+FC2, 同 dequant+GEMM)   │ │
│ │ 8. Residual                         │ │
│ └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
  │
  ▼
LM Head (FP16 Linear) → Logits → Sampling → 输出 Token
```
AFPQ 修改的部分：在 FasterTransformer 中新增了 NF4-asym dequantization kernel，替换了原有的 INT4 dequant kernel。修改发生在每个 Linear 层（Q/K/V/O/FC1/FC2）的权重加载后、GEMM 执行前。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FasterTransformer GitHub：https://github.com/NVIDIA/FasterTransformer（已于 2024 年归档，被 TensorRT-LLM 取代）。使用方式：C++ API 或 Python 绑定，通过配置文件指定模型结构、精度和并行策略。在 AFPQ 论文中的使用：(1) 加载 FP16 模型基线；(2) 将量化后的 NF4-asym 权重和 scale 参数加载到 GPU 显存；(3) 注册自定义 dequantize kernel；(4) 运行推理测量延迟。AFPQ 报告的延迟（A6000 GPU, batch=1, input_len=128, output_len=20）：LLaMA2-7B NF4-asym 265.54ms（FP16 415.06ms, 1.56x speedup），LLaMA2-13B NF4-asym 485.42ms（FP16 788.01ms, 1.62x speedup）。

涉及论文标题：
- AFPQ Asymmetric Floating Point Quantization for LLMs

---
