## LMMs-Engine

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LMMs-Engine 是一个简洁、统一的多模态模型预训练和微调框架，由 LMMs-Lab 维护。它提供统一的训练接口，支持多种视觉语言模型架构（LLaVA 系列、Qwen-VL 系列等）的预训练、SFT 和评估。在 LongVT 中，LMMs-Engine 用于 Cold-Start SFT 和 RFT 两个阶段的训练。核心特性：(1) Stream Packing：将多个训练样本拼接至固定 buffer size（51200 tokens），消除 padding tokens 带来的冗余计算；(2) Dynamic Batching：动态调整批次大小以最大化 GPU 利用率；(3) Liger Kernel 集成：使用 Liger 的融合 kernel（fused linear + cross entropy 等）加速训练并减少显存占用。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
LMMs-Engine 的 Stream Packing 原理：
```
# 传统 padding: 每个样本独立 padded 到 max_length
batch_padded = [
    [sample_1_tokens, PAD, PAD, ..., PAD],  # 大量浪费
    [sample_2_tokens, PAD, ..., PAD],
    ...
]
# GPU 在这些 PAD tokens 上做无用计算

# LMMs-Engine Stream Packing:
buffer = []  # 固定大小 buffer (51200 tokens)
for sample in iterable_dataset:
    if len(buffer) + len(sample.tokens) <= BUFFER_SIZE:
        buffer.append(sample.tokens)
    else:
        # buffer 满，作为单个序列训练
        concat_tokens = concat(buffer)
        # 注意: 需要 attention mask 防止跨样本 attention
        loss = llm(concat_tokens, attention_mask=cross_sample_mask)
        update(loss)
        buffer = [sample.tokens]  # 开始新的 buffer
```
训练配置：AdamW optimizer, lr=5e-5, cosine schedule, warmup 300 steps (SFT) / 160 steps (RFT), weight_decay=0.0 (SFT/RFT)。LMMs-Engine 在 SFT 阶段使用 32 GPU、RFT 阶段使用 64 GPU。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LMMs-Engine 的设计理念是简洁和统一：提供标准化的 `train.py` 入口，通过 YAML 配置文件指定模型、数据集、训练超参数。相比 LLaMA-Factory 等更重量级框架，LMMs-Engine 更轻量、更专注于 LMM 训练场景。GitHub: https://github.com/LMMs-Lab/lmms-engine。支持 models: Qwen-VL 系列、LLaVA 系列、InternVL 系列。支持 data formats: image QA, video QA, interleaved image-text, tool-augmented multi-turn conversations。

涉及论文标题：
- LongVT__Incentivizing__Thinking_with_Long_Videos__via_Native_Tool_Calling
