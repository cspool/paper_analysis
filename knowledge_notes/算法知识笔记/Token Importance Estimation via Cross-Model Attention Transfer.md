## Token Importance Estimation via Cross-Model Attention Transfer

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Token Importance Estimation via Cross-Model Attention Transfer 是 SPECPREFILL 的核心机制假设：同一模型家族中不同规模的模型（如 Llama-3.1-8B 和 405B），其对 prompt token 重要性的注意力分布具有可迁移性。因此可用小型 speculator 的注意力分数作为 proxy 估计哪些 token 对大型 base model 重要，无需对大模型做任何额外 forward pass 或训练。

从算法pipeline角度拆解术语：

给定 prompt 长度 M，speculator 层数 L，头数 H，look-ahead N 步：
1. 对每个 prompt token i，第 j 步解码产生的注意力：a_{ij} = Softmax(Q_{M+j} K^T)_i
2. 聚合：importance(i) = (1/N) Σ_j max_{l} max_{h} a_{ij}^{(l,h)}
3. 基于 importance 选择 Top-K chunks（而非 Top-K tokens，利用邻近 token 重要性相关）

术语一般如何实现？如何使用？

实现要求 speculator 和 base model 同 tokenizer、同家族。speculator FLOPs 仅为主模型 14.24%（70B）或 2.96%（405B）。适用场景：prompt 含冗余 token 的长上下文任务；不适用信息密集短 prompts（如数学题）。

涉及论文标题：
- Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation

---
