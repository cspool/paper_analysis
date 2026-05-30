## LLM-QAT (LLM Quantization-Aware Training，LLM量化感知训练)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LLM-QAT 是 Liu et al. (2023) 提出的面向 LLM 的无数据量化感知训练方法。与传统 QAT 需原始训练数据不同，LLM-QAT 的核心创新：使用预训练 LLM 自身生成训练数据（self-generated data），通过从预训练模型中采样 token 序列构建蒸馏数据集，然后在量化模型上执行知识蒸馏——教师为全精度 LLM，学生为量化 LLM。该方法的关键优势：(1) 无需访问预训练数据，保护数据隐私且降低数据获取成本；(2) 通过全参数微调（full-parameter fine-tuning）在量化约束下重新学习权重分布，比 PTQ 更好地补偿量化误差；(3) 可支持 INT4/INT8 等不同位宽，且通过蒸馏保留了全精度模型的效用。属于 QAT w/ FT 类别。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```python
# LLM-QAT Pipeline
# Step 1: 从预训练模型生成蒸馏数据
synthetic_data = []
for i in range(num_samples):
    prompt = random_start_token  # 或从词汇表随机采样
    seq = full_precision_model.generate(prompt, max_length=512)
    synthetic_data.append(seq)

# Step 2: 量化模型初始化
quantized_model = quantize(full_precision_model, bits=4)  # INT4 量化

# Step 3: QAT 蒸馏训练（全参数）
for seq in synthetic_data:
    # 教师（FP16）前向
    with torch.no_grad():
        teacher_logits = full_precision_model(seq)

    # 学生（Simulated INT4）前向，使用 STE 通过量化操作
    student_logits = quantized_model(seq)

    # 蒸馏损失（KL 散度 + 交叉熵）
    loss = KL_div(student_logits, teacher_logits) + CE(student_logits, seq_labels)
    loss.backward()  # STE 梯度通过伪量化操作

    # 更新所有参数
    optimizer.step()
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/facebookresearch/LLM-QAT。基于 HuggingFace Transformers + PyTorch，使用标准的 fake quantization（前向量化-反量化模拟，反向 STE）。训练时需多张 GPU（如 8× A100）进行全参数微调。在 Q-resafe 的安全评估中，LLM-QAT 在 benign 数据集（Risk-I）上 INT4 ASR=16.9%，表现优于 QLoRA（ASR=42.3%），因为全参数微调比 LoRA 更好地保留了预训练模型的能力包括安全能力；但在有害数据集上 ASR 仍高达 82.9%。

涉及论文标题：
- Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models
