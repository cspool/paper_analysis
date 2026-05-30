## Quantization-Aware Fine-Tuning (QAF, 量化感知微调)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Quantization-Aware Fine-Tuning（QAF，量化感知微调）是介于 PTQ（后训练量化）和 QAT（量化感知训练）之间的范式：模型首先通过 PTQ 方法（如 GPTQ）被量化，然后在冻结的量化权重上训练轻量级适配器进行微调。与 QAT 在训练中模拟量化的全参数训练不同，QAF 保持量化权重冻结（或通过适配器间接调整），以较低的微调成本实现量化模型的性能恢复和任务适配。代表性工作包括 QLoRA（16-bit LoRA 适配器）、QA-LoRA（适配器调整零点因子实现无损合并）、LoTA-QAF（三值适配器直接调整量化权重实现无损合并）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QAF 的通用流程（以 LoTA-QAF 为例）：
```
# Stage 1: PTQ 量化（离线，一次性）
W_int, s, z = gptq_quantize(W_fp16, calibration_data)
# 得到量化权重和量化参数

# Stage 2: QAF 微调（在线，trainable）
freeze(W_int, s)  # 量化参数冻结（或通过适配器间接更新）
init_adapters()   # 初始化适配器（LoRA/TA/QA-LoRA）
for batch in fine_tuning_data:
    # 前向：量化权重 + 适配器调整
    if method == "LoRA":  # QLoRA 方式
        y = (dequant(W_int, s, z) + α/r * A@B)^T @ x
    elif method == "QA-LoRA":
        z' = z + adapter_output  # 仅调整零点
        y = (s * W_int + z')^T @ x
    elif method == "LoTA-QAF":
        W'_int = clamp(W_int + ternary_adjustment, 0, 2^N-1)
        z' = z + s * μ_offset
        y = (s * W'_int + z')^T @ x
    loss = criterion(y, label)
    loss.backward()
    adapter_optimizer.step()  # 仅更新适配器

# Stage 3: 合并 & 推理（适配器合并入量化权重）
merge_adapters_into_quantized_weights()
inference_with_pure_low_bit_weights()  # 无适配器开销
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
QAF 的三种主要模式：(1) QLoRA 模式——16-bit 适配器在量化权重上训练，推理时适配器与量化权重分开计算，存在混合精度开销；(2) QA-LoRA 模式——适配器结构对齐分组量化的零点因子，训练后无损合并进零点，但仅能间接调整量化权重；(3) LoTA-QAF 模式——三值适配器直接在量化网格内调整 W_int 和 z，训练后无损合并，推理零开销。QAF 的核心评估场景有两种：性能恢复（performance-recovery，用通用数据微调恢复量化造成的性能损失）和任务特化（task-specific，在特定下游任务上微调）。在 LoTA-QAF 实验中，QAF 展示了在低比特（尤其是 2-bit/3-bit）场景下显著恢复 PTQ 方法未充分利用的性能潜力（如 Qwen 2.5 32B 2-bit 微调后提升 16.97%）。

涉及论文标题：
- LoTA-QAF: Lossless Ternary Adaptation for Quantization-Aware Fine-Tuning
