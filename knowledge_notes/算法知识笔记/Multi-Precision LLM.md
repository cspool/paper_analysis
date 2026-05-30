## Multi-Precision LLM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Precision LLM 是一种量化范式，允许单个部署模型支持多种推理精度（如 2/3/4-bit），从而根据运行时 SLO 动态切换比特宽度。与 Fixed-Precision LLM（每精度需独立模型）相比，通过共享精度间的量化表示大幅减少存储。源自 CNN 的 Any-Precision Networks (AAAI 2021)，LLM 时代转向 PTQ 方式。代表性工作：(1) Any-Precision LLM (ICML 2024 Oral)：聚类式非均匀量化 + Incremental Upscaling，从 3-bit 逐步分裂 centroid 至 8-bit，但 2-bit 退化严重且不硬件友好；(2) Matryoshka Quantization (2025)：MSB slicing，低比特是高位比特的子集；(3) AnyBCQ：BCQ-based 多精度，共享比特平面 + 独立 scale，2-bit 性能强且硬件友好。存储优势：LLaMA-3.1-8B 存三个独立模型需 9.85GB，AnyBCQ 单模型仅 4.99GB（↓49%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
AnyBCQ Multi-Precision LLM 推理流程（用户指定精度 p ∈ {2,3,4}）：

```
def multi_precision_inference(model, input_ids, p):
    hidden = embed(input_ids)
    for layer in model.layers:
        for linear in layer.linears:
            W_bits = load_bitplanes(linear, num_planes=p)  # 仅加载 p 层
            alpha = load_scales(linear, precision=p)       # α_i^{(p)}
            output = zeros(batch, hidden_dim)
            for i in range(p):
                partial = bitplane_gemm(W_bits[i], hidden)  # B_i ∈ {-1,+1}
                output += alpha[i] * partial
        hidden = attention(hidden) + mlp(hidden)
    return lm_head(hidden)
```

Multi-Precision（全模型统一精度切换）与 Mixed-Precision（不同层/不同 token 不同精度）的区别：前者如 AnyBCQ 从 2-bit 切换到 4-bit，后者如 DP-LLM 逐层动态精度、PMPD 逐解码步精度下降。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
多精度模型优化：(1) Any-Precision LLM：K-means 基础种子 → 分裂 centroid → 存储多套 centroid table；(2) AnyBCQ：BCQ 基础精度 → 逐比特冻结 B_i → 残差提取新比特平面 → 优化 α；(3) Matryoshka Quantization：QAT 联合优化多个比特宽度 loss。部署要求：运行时精度切换、按需比特平面加载、多精度 kernel。限制：Multi-Precision 高精度（如 4-bit）通常略逊于 Fixed-Precision 同精度模型，因共享比特平面的约束缩小了优化空间。

涉及论文标题：
- AnyBCQ Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs

---
