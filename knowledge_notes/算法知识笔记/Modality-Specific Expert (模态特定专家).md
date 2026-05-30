## Modality-Specific Expert (模态特定专家)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Modality-Specific Expert（模态特定专家）是 Uni-MoE 提出的 MoE 多模态 LLM 训练策略中的核心概念：在将 MoE 架构引入多模态 LLM 时，不同专家（FFN）在不同的单模态/跨模态数据上分别预训练，使其发展出对不同模态的偏好和处理专长。训练完成后，每个 expert 在其专业模态的 token 上被 router 优先激活（如音频 tokens → Expert 4，图像 tokens → Expert 2）。

Uni-MoE 定义了 8 个单模态专家训练任务（Task1-Task8），分别训练不同用途的专家：Task2 用 LLaVA-Instruct-150K (T-I，文本-图像) 训练图像专家；Task3 用 LLaVA-Instruct-150K (I-A，语音-图像) 训练语音-图像专家；Task7 用 RACE-Audio + LibriSpeech 训练长语音专家；Task8 用 WavCaps/AudioCaps/MELD/Clotho 训练音频专家。

与标准 MoE（所有专家初始相同，简称 pure MoE）的对比：pure MoE 中专家缺乏模态区分性，routing 分布在各模态间更均匀（Figure 8, Figure 10），无法有效利用多模态数据的结构差异；modality-specific 专家天然形成模态偏好分布（Figure 4-5），router 学习到特定路由模式。

从算法pipeline角度拆解术语：

Modality-Specific Expert 训练流程（对应 Algorithm 1 Stage 2）：

```
# 对每种模态 M，独立训练对应专家
for each modality M:
    # 复制阶段一训练好的权重
    copy_weights_from_stage1()
    
    for each step:
        (x, y) = sample(D_M)                    # 采样该模态的跨模态指令数据
        x_M = Connector(x)                       # 模态投影
        prediction = LLM(x_M, E[h(i_M)])        # 前向，激活目标专家 E
        loss = CE(prediction, tokenize(y))
        # 仅更新: LoRA (MLP in LLM) + 投影层参数
        θ = θ - α ∇_θ loss

# 得到: {Expert_1→image, Expert_2→image-text, Expert_3→speech-image, Expert_4→audio, ...}
```

阶段三加载这些预训练专家到 MoE layers，通过 LoRA 联合微调。

术语一般如何实现？如何使用？

在 Uni-MoE 的具体实现中：(1) 阶段二从阶段一 checkpoint 初始化，每个专家独立训练 1 epoch；(2) 使用 LoRA（rank=64, alpha=16）仅微调 MLP 在 LLM 中的参数和投影层；(3) 学习率 4e-5（LoRA）和 3e-5（投影层），global batch size=16，2 块 A100 GPU；(4) 训练后将各专家 FFN 权重分别保存，阶段三加载到 MoE layers 的不同 expert slots。这种方法的优势在于：(a) 各专家发展出明确的模态专长，实现自然 load balancing；(b) router 更容易学习到有意义的 token-to-expert 映射；(c) 在混合多模态数据上训练收敛更快更稳定（Figure 3 蓝色线 vs 橙色线）。

涉及论文标题：
- Uni-MoE Scaling Unified Multimodal LLMs with Mixture of Experts
