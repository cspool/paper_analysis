## Fine-grained Expert Segmentation

术语解释
由 DeepSeekMoE (2024) 提出，将标准 MoE 中每个专家的 FFN 中间维度 (d_ffn) 切分为更小的粒度，增加专家数量同时减小每个专家的尺寸，提升知识分解精度。

术语是什么？
标准 MoE 中 d_expert = d_ffn。Fine-grained 将 d_expert 缩小为 d_ffn / m（如 1/8），专家数扩大 m 倍。DeepSeekMoE-145B: d_expert = 1/8 d_ffn, 16→128 experts, top-2→top-16。

核心优势：(1) 更精细的知识分解 (2) 更灵活的专家组合 (3) 解决"知识混杂"问题。
LLAMA-MoE 验证：激活 4/16 experts (d_expert=688) 优于 2/8 (d_expert=1376)。

术语一般如何实现？如何使用？
- DeepSeekMoE, Qwen1.5-MoE, DBRX 均采用此策略
- 需配合 shared experts 使用以补偿单个专家容量不足
- DeepSeek-V3: d_expert=2048, shared=1, routed=256, top-8

涉及论文标题：
- A Survey on Mixture of Experts in Large Language Models
- Aria An Open Multimodal Native Mixture-of-Experts Model（ARIA: 24.9B total / 3.5B activated per text token, 66 experts/layer = 2 shared + 64 routed, expert FFN dim=1664, hidden dim=2560, 每 token 激活 6 routed + 2 shared；所有 expert 为 modality-generic，expert specialization 在 multimodal 预训练中自然涌现）
- Dense Backpropagation Improves Training for Sparse Mixture-of-Experts（使用 32c4 fine-grained MoE 配置：32 total experts × 4 active, 1.96B total params, 565M active params, hidden dim=1024）
- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
