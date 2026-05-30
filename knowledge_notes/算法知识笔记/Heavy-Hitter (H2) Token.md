## Heavy-Hitter (H2) Token

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Heavy-Hitter (H2) Token 是 H2O 论文提出的概念，指在 Attention 运算中贡献大部分 Attention Score 累积值的少数 token。H2O 的实验显示，约 20% 的 token（H2 token）贡献了绝大部分注意力分数，若将其移除则准确率急剧下降。H2 token 构成了 KV Cache Token Pruning 的理论基础：只需保留 H2 token 即可保持模型性能。

H2 的识别依赖 A2S：累积较高的 token 即为 H2 token。然而 A2SF 论文指出，由于 Causal Mask 的偏差，H2O 识别的 H2 往往偏向早期 token——不一定真正重要。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**H2 Token 在 A2SF 的 re-interpretation**：

H2O 假设：A2S 高 → 重要（语言现象）
A2SF 揭示的偏差：A2S 高 ← 位置靠前 + 累积次数多（数学必然）

因此 A2SF 不否定 H2 概念本身，而是修正识别 H2 的方式——用带衰减的 A2SF 替换原始 A2S，使真正的 H2 token（每步都输出高分而非仅靠位置优势）脱颖而出。

术语一般如何实现？如何使用？

H2 Token 选择是 A2S/A2SF 方法的输出步骤。在实现中，`top_k(A, K)` 返回的 K 个 token 即为 H2 token。H2O 为每个 head 独立维护 H2 列表，允许不同 head 关注不同重要 token。A2SF 继承了这一 head-wise 设计。

**KV-Distill 对 H2 方法的分析**：KV-Distill 将 H2O 分为两个范式——H2A（问题感知）：将问题和上下文拼接后计算累积注意力，利用问题信号扫描上下文中的关键信息；H2I（问题无关）：仅在上下文内部计算累积注意力。KV-Distill 实证发现：H2I 在问题无关范式下性能急剧下降——例如 LLAMA-3 SQuAD 上 H2I 25% retention 准确率仅 56.6%（vs uncompressed 87.6%，H2A 84.0%）。这说明 H2 的注意力累积机制在没有问题信号引导时无法有效识别对未知问题重要的 token。

**NACL 对 H2O/H2 的分析**：NACL 揭示了 H2O 的 attention bias 问题——H2 token 的高 A2S 分数部分源于位置偏差（初始和最近 token 天然高 attention），而非真正的语义重要性。在 LongBench passkey retrieval 中，H2O 30% budget 仅 PR-Zh=3.7/PR-En=5.0（vs Full 8.0/10.1），证明 H2 token 选择遗漏了中间位置的关键信息。NACL 通过 proxy tokens（仅用末尾 ~10% token 的 attention）替代全量累加，配合 head-wise RANDOM EVICTION，将 passkey retrieval 提升至 NACL 30%=6.8/9.0。

涉及论文标题：
- A2SF: Accumulative Attention Scoring with Forgetting Factor for Token Pruning in Transformer Decoder
- KV-Distill: Nearly Lossless Learnable Context Compression for LLMs
- NACL: A General and Effective KV Cache Eviction Framework for LLMs at Inference Time
- ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs

---
