## FP8-S0E4M4（无符号 8-bit 浮点格式，注意力分数量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FP8-S0E4M4 是 P3-LLM 提出的无符号 8-bit 浮点格式：0 个符号位 + 4-bit 指数 + 4-bit 尾数，指数 bias 为 −15。用于量化 softmax 之后的注意力分数（attention-scores）。设计依据两条观察：(1) softmax 输出恒在 [0,1]，无需符号位；(2) FP16 有 5-bit 指数（bias −15，指数范围 [−14,15]），而注意力分数恒 <1，正指数完全用不上，有效指数范围仅 [−14,−1]（14 个值），4-bit 指数足够覆盖，省下的 1-bit 给尾数提升数值保真。相比 INT8 与 FP8-E4M3（表 II 实验）：INT8 量化注意力分数带来明显 perplexity 退化，FP8-S0E4M4 达到 near-lossless（Llama-2-7B Wikitext-2：FP16 5.15 → FP8-S0E4M4 5.15，INT8 5.19，FP8-E4M3 5.16）。注意 FP8-S0E4M4 与工业标准 FP8-E4M3/E5M2（NVIDIA/AMD 支持）是不同格式：FP8-S0E4M4 无符号且 4+4 划分专为 [0,1] 分布设计。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
注意力分数量化与融合流程（解码阶段）：
```
# softmax 后得到 P ∈ [0,1]^T（FP16）
P_scaled = P * (S^V / S^V_max)      # 融合 per-value-head 缩放（二级缩放防越界）
P8 = round_to_fp8_s0e4m4(P_scaled)  # 直接保留 FP16 高 4 位 mantissa + 指数重映射（无缩放因子）
O = PCU_GEMV(P8, V_INT4) * S^V_max  # P·V 在低精度 PCU 执行，结果乘回 S^V_max
```
要点：FP8-S0E4M4 不需要量化缩放因子（格式本身覆盖所需数值范围），因此"直接截位"即可量化，省去 scale 存储与乘法；S^V_max 的二级缩放保证融合后 P_scaled 仍在 [0,1]（不破坏无符号假设），P·V 完成后乘回。硬件侧（见本库硬件架构 PCU 条目）：该格式的 8-bit 尾数（5-bit 含隐藏位）+ 符号位恰好匹配 PCU 的 6-bit 定点乘法器输入宽度，使 P·V 不必回退到 FP16 单元。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：在 PyTorch 中作为 fake-quant（论文开源仓库 https://github.com/yc2367/P3-LLM 的 `--p_bits 8` 路径）——对 FP16 注意力分数直接取高 4 位 mantissa 并重映射指数到 bias −15 的无符号 4-bit 指数表示；硬件上用 6-bit 定点乘法器消费（尾数 5-bit 含隐藏位 + 1 符号位），4-bit 指数仅用于移位乘积。适用场景：任何需要对 [0,1] 有界张量做 8-bit 量化的低精度 MAC 硬件（PIM、NPU 低精度单元）；该格式表明"为操作数分布定制指数位分配"比通用 FP8 更优。开源状态：算法代码已开源，硬件 RTL 未开源。

涉及论文标题：
- P3-LLM An Integrated NPU-PIM Accelerator for Edge LLM Inference Using Hybrid Numerical Formats
