## MXINT (Micro-exponent Integer Format)（微指数整数格式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MXINT (Micro-exponent Integer) 是 Darvish Rouhani et al. (ISCA 2023) 提出的低精度数值格式，属于 Microscaling (MX) 格式家族。核心设计：一个 block 内的所有元素共享一个小的共享指数（micro-exponent），block 内每个元素使用整数尾数表示。具体地，对 block size B 的权重块，先计算块内最大绝对值决定共享指数 shared_exp = ⌊log₂(max|w|)⌋，然后每个权重除以 2^shared_exp 并舍入到 N-bit 整数范围。QERA 论文使用 emulated MXINT：4-bit 下 block size=32（avg 4.25 bits/elem），3-bit 下 block size=32（3.25 bits），2-bit 下 block size=16（2.50 bits）。MXINT 的精度不仅来自更细粒度的 scaling（每 block 独立指数），还来自较小的 block size 带来的更均匀的块内数值分布。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MXINT 在 QERA 中的量化-反量化流程：

```
# MXINT 量化 (block_size=32, N=4-bit, range=[-7,7])
for each block of 32 consecutive weights in W:
    max_abs = max(abs(block))
    shared_exp = floor(log2(max_abs))       # block 共享指数
    scale = 2^shared_exp
    for each weight w in block:
        w_int = round(w / scale)             # 归一化并舍入
        w_int = clamp(w_int, -(2^{N-1}-1), 2^{N-1}-1)  # [-7, 7]
    # 存储: shared_exp (8-bit) + 32个(N-1)-bit尾数 + 32个1-bit符号位
    # 每元素实际位宽: 8/32 + N ≈ N + 0.25 bits

# MXINT 反量化
for each block:
    shared_exp = block_metadata.shared_exp
    scale = 2^shared_exp
    for each stored integer w_int in block:
        w_recovered = w_int * scale          # 恢复为近似FP值
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 QERA 中 MXINT 通过 PyTorch 手动实现量化/反量化函数（mxint_quantize / mxint_dequantize），非硬件原生支持。MX 格式家族的完整标准（OCP Microscaling Formats）包括 MXINT8、MXFP8、MXFP6、MXFP4 等变体，AMD、Arm、Intel、Meta、Microsoft、NVIDIA、Qualcomm 等已采用。优势：(1) 相比纯 INT 量化有更大动态范围（通过共享指数）；(2) 相比纯 FP 量化硬件实现更简单（block 内无 FP 乘加逻辑）；(3) block size 可灵活调节精度和开销的权衡。主要应用于 LLM 权重量化（W4A16）、KV cache 量化等低精度推理场景。

涉及论文标题：
- QERA: an Analytical Framework for Quantization Error Reconstruction
