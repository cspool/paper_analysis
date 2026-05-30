## Llama-MoE

术语解释
Llama-MoE 是 Zhu et al. (EMNLP 2024) 提出的一种从 dense LLaMA 模型通过 continual pre-training 构建 MoE 模型的方法。通过将 LLaMA 的 FFN 层转换为 MoE 层并继续训练，在保持原始 LLaMA 能力的同时获得 MoE 架构的计算效率优势。

术语是什么？
Llama-MoE 的构建方法：
- 从 LLaMA checkpoints 出发，将部分 FFN 层替换为 MoE 层
- 使用 Noise Top-k Gating 作为路由机制
- 通过 continual pre-training 训练新增参数（router + experts），同时保留原始权重
- 支持多种配置：k/N 表示每 token 从 N 个 experts 中选 k 个（如 4/16、2/8、2/16）
- 激活参数远少于总参数（sparse activation）

Kim et al. (2025) 使用的变体：Llama-MoE-3.5B (4/16)、Llama-MoE-3.5B (2/8)、Llama-MoE-3.0B (2/16)，其中 3.5B/3.0B 为总参数量，括号内为 (激活 expert 数 / 总 expert 数)。

术语一般如何实现？如何使用？
- Continual pre-training 使用公开语料（如 RedPajama）
- Router 使用 Noise Top-k Gating + auxiliary load balancing loss
- 在 KD 场景中作为 MoE teacher，distill 到 dense student (Sheared-Llama)
- 与 Sheared-Llama 共享 LLaMA tokenizer，满足 KD 中 teacher/student 共享 tokenizer 的要求

涉及论文标题：
- Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models

---
