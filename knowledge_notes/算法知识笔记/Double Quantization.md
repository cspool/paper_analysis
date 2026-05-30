## Double Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Double Quantization 是 QLoRA (Dettmers 2023) 引入的量化常量压缩技术。在 block-wise 量化中，每个 block 需要存储一个 scale factor（FP32），以 block_size=64 为例，scale 相当于每参数 32/64 = 0.5 bit 的额外开销。Double Quantization 对第一层量化产生的 scale factors 执行第二次量化：将 scale factors 按 block_size=256 分组，每组量化为 8-bit FP8，再引入第二层 scale factor c₂^FP16。这使 scale 的每参数开销从 0.5 bit 降至 8/64 + 32/(64×256) ≈ 0.127 bit，减少约 75%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 第一层量化：权重 → 4-bit NF4
for each block w_i of size 64:
    s_i = absmax(w_i)                           # FP32 scale factor (first level)
    w_quant_i = NF4(w_i / s_i)                  # 4-bit quantized weights

# 第二层量化（Double Quant）：scale factors → 8-bit FP8
# 每 256 个 s_i 组成一组
for each group of 256 scale factors {s_1,...,s_256}:
    c₂ = absmax({s_1,...,s_256})               # FP16 second-level scale
    s_FP8_i = FP8(s_i / c₂)                     # 8-bit quantized scales

# 存储内容（块 i）:
# - w_quant_i: 64 × 4 bit = 256 bit
# - s_FP8_i: 8 bit
# - 对应的 c₂：每 256 块共享一个 FP16 = 16 bit
# 每块总开销 = 256 + 8 + 16/256 ≈ 264 bit
# 相比 FP16 (64×16=1024 bit) 压缩近 4x
```
在 IR-QLoRA 中，Double Quantization 同样应用于 ICQ 引入的 τ*，与 scale factor 执行相同的 FP8 量化以减少存储开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
bitsandbytes 库实现 Double Quantization：`bnb_4bit_use_double_quant=True`。在 HuggingFace 中使用：`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)`。内存节省：对于 65B 模型，Double Quantization 单独节省约 3GB 显存。在 llama.cpp 社区中，类似的 super-block 方法独立探索了相同思想。在 BOF4 论文中，作者未使用 Double Quantization（即未进一步量化 quantization constants），因为 signed normalization（BOF4-S）会额外需要一个 sign bit 来编码归一化常数的符号，可能与 Double Quantization 不兼容。

SpQR 推广 Double Quantization 为**双层量化（Bilevel Quantization）**：核心差异在于 (1) 第一层：权重按极小 group（β₁=8~16，远小于 QLoRA 的 64）分组，每组独立计算 3-bit scale s_q 和 zero-point z_q；(2) 第二层：将 s_q 和 z_q 分别按 β₂=16 分组，以 3-bit 格式量化，并引入第三层 16-bit statistics s_s（scale of scales）、z_s（zero of scales）、s_z（scale of zeros）、z_z（zero of zeros）；(3) 总统计量开销 = (b_s+b_z)/β₁ + 64/(β₁β₂)，例如 β₁=16, β₂=32, b_s=b_z=3 时仅 0.5 bits/param；(4) 优化：去除"max>0, min<0"约束，允许全正/全负 group 使用非整数零点。与 QLoRA 的 Double Quantization（仅对 scale 做一次 FP8 量化）不同，SpQR 的双层量化在更小 group 上同时对 scale 和 zero-point 做二次量化，使极小 group 的存储开销可控。

涉及论文标题：
- Accurate LoRA-Finetuning Quantization of LLMs via Information Retention
- Improving Block-Wise LLM Quantization by 4-bit Block-Wise Optimal Float (BOF4)
- SpQR A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression

---
