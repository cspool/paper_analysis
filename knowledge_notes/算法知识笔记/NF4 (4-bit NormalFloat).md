## NF4 (4-bit NormalFloat)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NF4（4-bit NormalFloat）是一种基于信息论最优化的 4-bit 量化数据类型，由 Dettmers et al. (2023) 在 QLoRA 中提出。其设计原理：假设预训练神经网络的权重服从零均值正态分布 N(0, σ²)，则最优的 4-bit 量化方案是将该分布的累积分布函数（CDF）的 2^N 个等概率分位点映射到对应的 4-bit 索引。具体地，NF4 的 16 个量化级别为 Q^NF4_map(q_i) = Φ⁻¹(i/(2^N+1)) = Φ⁻¹(i/17)，其中 Φ⁻¹ 为标准正态分布的分位函数。这 16 个值分别对应 -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0。实际使用时，权重按 block_size=64 分组，每组归一化到 [-1,1] 后映射到最近 NF4 级别。QA-LoRA 论文指出 NF4 缺乏 CUDA 算子级别优化，导致 QLoRA 的 NF4 微调速度慢于 QA-LoRA 的 INT4 微调。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# NF4 量化流程
NF4_LEVELS = [Φ⁻¹(i/17) for i in range(1, 17)]  # 16 values
# 对每个 block_size=64 的 block:
for block in W.reshape(-1, 64):
    s = absmax(block)           # block-wise scale
    W_norm = block / s          # 归一化到 [-1, 1]
    for each w in W_norm:
        idx = argmin |w - NF4_LEVELS[i]|  # 最近邻查找, 得 0-15 的 4-bit index
        W_q = idx                  # 存储为 4-bit index
    scales.append(s)
# 反量化: w_deq = s * NF4_LEVELS[idx]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
NF4 通过 bitsandbytes 库（`bnb_4bit_quant_type="nf4"`）实现，配合 HuggingFace Transformers 使用。双重量化（Double Quantization）将 block-wise scale s 进一步以 FP8 量化（s_FP8）并保留 FP32 残差（s_FP32 = s - s_FP8），将 scale 存储从 0.5 bit/参数降至 0.127 bit/参数。

涉及论文标题：
- S²Q-VDiT Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation
- QA-LoRA Quantization-Aware Low-Rank Adaptation of Large Language Models
- QeRL Beyond Efficiency - Quantization-enhanced Reinforcement Learning for LLMs

QeRL (Huang et al., NVIDIA, 2025) 发现 NF4 在 RL 训练中存在严重性能瓶颈：NF4 反量化需通过 lookup table 将 4-bit index 映射回浮点值才能做矩阵乘法，使 QLoRA 的 rollout 速度比 BF16 LoRA 还慢 0.7-0.8×。此外 QeRL 实验显示 NF4 量化后 7B 模型 GSM8K 原始准确率下降 5.8 点（70.5% vs 76.3% BF16），RL 训练后恢复至 85.0%，仍低于 NVFP4+AQN 的 90.8%。

---
