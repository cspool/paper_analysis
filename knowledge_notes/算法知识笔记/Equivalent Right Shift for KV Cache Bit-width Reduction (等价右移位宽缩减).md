## Equivalent Right Shift for KV Cache Bit-width Reduction (等价右移位宽缩减)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Equivalent Right Shift 是 PM-KVQ 中实现 KV Cache 位宽缩减的核心整数运算：将 2b-bit 量化整数降为 b-bit，结果等效于先反量化到 FP16 再重新量化为 b-bit——但全程仅用整数域乘加和移位。操作公式：`X_b = ((2^{2b} - 2^b + 1)(X_{2b} + 2^{b-1})) >> 3b`，参数更新 `Z_b = Z_{2b}`, `S_b = (2^b + 1)S_{2b}`。

三种策略消融（DeepSeek-LLaMA-8B, AIME-2024 pass@1）：Direct Right Shift 12.08% < Modified Right Shift 28.75% < Equivalent Right Shift 38.33%（对比 FP16 44.17%）。

从算法pipeline角度拆解术语：
- 16→8: `X_8 = (65281*(X_16+128)) >> 24`, `S_8 = 257*S_16`
- 8→4: `X_4 = (241*(X_8+8)) >> 12`, `S_4 = 17*S_8`
- 4→2: `X_2 = (13*(X_4+2)) >> 6`, `S_2 = 5*S_4`

术语一般如何实现？可使用 GPU 整数张量操作或 CPU 标准整型指令实现，无需浮点单元或自定义 CUDA kernel。代码开源：https://github.com/thu-nics/PM-KVQ。

涉及论文标题：
- PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

---
