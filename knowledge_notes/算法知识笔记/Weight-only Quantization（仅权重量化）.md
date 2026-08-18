## Weight-only Quantization（仅权重量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
weight-only 量化只压缩模型权重、保留激活高精度（FP16），是面向 LLM decode 阶段（内存受限、权重主导访存）的主流压缩范式（AWQ、GPTQ、SqueezeLLM）。由于 decode 每 token 只做小 GEMV、激活轻量而权重张量占主导内存流量，权重位宽降低几乎线性加速 decode（AWQ 论文观察）。解析型 weight-only 量化（INT4/INT8，线性缩放+取整）可原位逐元素反量化、无额外依赖；EVA 把这一趋势推到 2-bit 级：采用 VQ 型（非解析）weight-only 量化，精度保持优于解析法，但查表反量化引入访存冲突（EVA 的动机之一）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
decode 每步 y = xW：x∈R^{1×K}（FP16），W 以低比特存储。量化后内存流量 ∝ 权重比特数；AWQ/GPTQ 4-bit 时 decode 近线性提速，VQ 2-bit 再砍一半。EVA 精度对比（Table V，WikiText-2 ppl，L-2 7B）：EVA-A16W2（激活 FP16、权重 2-bit VQ）6.69 保持竞争性，而解析法 2-bit（FIGNA-INT2 AWQ）崩到 2.2e5——说明 2-bit 级只有非解析 VQ 能保精度；4-bit 时 EVA-A16W4（AQLM 4×8）5.43 优于 AWQ 4-bit 5.60。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：AWQ（激活感知，按激活幅度选保护通道）、GPTQ（二阶 Hessian 逐层量化）、SqueezeLLM 等框架；EVA 采用 AQLM/GPTVQ 的 VQ 型 weight-only 方案。使用方式：配合 serving/硬件加速器使用；EVA 硬件为 FP16 激活 + 2/3/4-bit VQ 权重设计（A16W2/W3/W4），在 32×8 FP16 重配阵列上执行，同时保持 INT8 prefill 兼容（同一阵列重配）。

- M100 补充视角（ISCA'26，车规 LLM 推理的量化选择）：LLaMA2-7B decode 阶段采用 W4A16（4-bit 权重 + FP16 激活）以压缩权重访存（decode 为带宽受限、权重主导访存）：M100 21.34ms vs Thor-U 20ms（两平台 DDR 带宽同为 273 GB/s）；prefill 阶段改用 W8A8（权重+激活 8-bit INT）以利用整数张量计算（compute-bound）：M100 79ms vs 154ms（1.95×）。选择逻辑与常规认知一致：带宽受限阶段用 weight-only，计算密集阶段用 weight-activation 量化。
涉及论文标题：
- EVA: Accelerating LLM Decoding via an Efficient Vector Quantization Architecture
- M100: An Orchestrated Dataflow Architecture Powering General AI Computing
