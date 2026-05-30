## Post-Compression Fine-tuning (MoE 压缩后微调恢复)

术语解释
Post-Compression Fine-tuning 是对 Expert Trimming 压缩后的 MoE 模型进行轻量 fine-tuning 以恢复性能的技术。压缩后的模型（尤其是 Layer/Block Drop 后）在残差连接路径上出现结构不连续，fine-tuning 帮助剩余 layers 适应新的"跳过连接"模式。

术语是什么？
与完整预训练不同，Post-Compression Fine-tuning 仅需少量高质量指令数据，使用标准 LM loss 继续训练压缩后的模型几个 epoch。由于压缩后的模型已经保留了大部分原始模型的知识，fine-tuning 主要是"适应"而非从头学习。

He et al. (2025) 的设置：在 Alpaca-GPT4 数据集上 full-finetune 3 epochs，lr=8e-6，warmup ratio=0.03，cosine schedule，global batch size=32。

效果：DeepSeek-MoE-16B Block Drop B4/28 后 fine-tuning，性能 gap 从 -5.5% 缩小至 -0.6%（接近原始模型）。Layer Drop L4/28 从 -6.5% 恢复至 -1.0%。

术语一般如何实现？如何使用？
- 数据量需求小（Alpaca-GPT4 ~52K samples），不需要大规模预训练数据
- Full fine-tuning 比 LoRA 更有效（因为层结构发生了变化，需要全参数适应）
- Warmup ratio 0.03 + cosine schedule 避免初始训练不稳定
- 压缩比越高，fine-tuning 恢复效果越显著，但绝对性能仍随压缩率下降
- 适用于 Expert Drop / Layer Drop / Block Drop 所有三种 Expert Trimming 方法

涉及论文标题：
- Demystifying the Compression of Mixture-of-Experts Through a Unified Framework
