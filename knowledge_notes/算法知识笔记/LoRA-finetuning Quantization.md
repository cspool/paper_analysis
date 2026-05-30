## LoRA-finetuning Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LoRA-finetuning quantization 是将 LLM 量化与低秩适配（LoRA）参数高效微调相结合的两阶段范式：(1) **PTQ 阶段**：对预训练 LLM 执行后训练量化（如 NormalFloat、GPTQ），将 FP16 权重压缩至 2-8 bit，获得量化器参数（scale factors, zero points）；(2) **LoRA 微调阶段**：在量化后 LLM 的线性层上附加低秩适配器（LoRA），冻结量化权重不动，仅训练 LoRA 的 ℓ₁, ℓ₂ 矩阵和少量参数。相比全模型微调，该范式大幅降低训练显存和时间；相比纯 PTQ，LoRA 微调可恢复量化造成的精度损失。代表方法：QLoRA（Dettmers 2023）、QA-LoRA（Xu 2023）、LoftQ（Li 2023）、IR-QLoRA（本文）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Phase 1: PTQ（离线，执行一次）
for each linear_layer in LLM:
    w_block = split_by_block(layer.weight, block_size=64)
    for block in w_block:
        s = absmax(block)                               # scale factor
        w_quant[block] = NF4(block / s)                 # NormalFloat 4-bit 量化
        s_FP8, s_FP16 = double_quantize(s)             # 双重量化 scale

# Phase 2: LoRA 附加
for each quantized_linear_layer:
    layer.lora_A = nn.Linear(h, r, bias=False)          # ℓ₁: down-project
    layer.lora_B = nn.Linear(r, o, bias=False)          # ℓ₂: up-project
    init: lora_B.weight = 0                              # 零初始化保证训练起始不变

# Phase 3: 微调（仅更新 LoRA）
for batch in dataset:
    y = quantized_linear(x) + α · lora_B(lora_A(x))     # FP16 计算
    loss = cross_entropy(y, target)
    loss.backward()  # 梯度仅流经 LoRA，量化权重保持冻结
    update(lora_A, lora_B)

# Phase 4: 推理部署（可选合并）
W_merged = dequant(w_quant) + α · ℓ₂^T · ℓ₁^T         # 合并 LoRA 到量化权重
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
QLoRA (Dettmers 2023) 为该范式的奠基工作，开源代码在 https://github.com/artidoro/qlora。基于 HuggingFace Transformers + PEFT + bitsandbytes 实现。使用方式：通过 `BitsAndBytesConfig` 设置 NF4 量化 → `prepare_model_for_kbit_training()` → PEFT `LoraConfig` 添加 LoRA → 标准 Trainer 微调。关键配置：block_size=64（量化块大小）, LoRA rank r=64（适配低秩维度）, α=16（LoRA 缩放系数）。QLoRA 可在单张 48GB GPU 上微调 65B 模型（4-bit 量化下仅需 ~18GB 显存）。后续改进包括 QA-LoRA（integer 量化+量化感知 LoRA）、LoftQ（交替量化-LoRA 初始化）、IR-QLoRA（信息保留）。

涉及论文标题：
- Accurate LoRA-Finetuning Quantization of LLMs via Information Retention
- LoftQ: LoRA-Fine-Tuning-aware Quantization for Large Language Models
- S²Q-VDiT Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation
- QA-LoRA Quantization-Aware Low-Rank Adaptation of Large Language Models

QA-LoRA 对 LoRA 进行了重要修改以适配量化场景：标准 LoRA 中 A ∈ R^{D_in × D_int}，B ∈ R^{D_int × D_out} 均无约束；QA-LoRA 通过组内求和聚合操作 QA(x) 将输入维度从 D_in 降至 L（L = D_in/g，g 为量化组大小），使 A 矩阵尺寸缩减为 L × D_int。这一约束使 A 的行向量在量化组内共享，从而满足合并后权重仍可表示为 group-wise INT 量化格式的数学条件。

---
