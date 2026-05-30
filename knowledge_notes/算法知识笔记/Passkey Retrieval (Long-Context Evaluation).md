## Passkey Retrieval (Long-Context Evaluation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Passkey Retrieval 是由 Mohtashami & Jaggi (2023) 提出的简单合成长上下文评估任务，也是 ∞Bench（Zhang et al., 2024a）的核心组件。任务设计：在大量无关噪声文本（如重复的 "The grass is green. The sky is blue..."）中隐藏一个 5 位数字 "passkey"，模型需要在读取全部上下文后回答 "What is the passkey?"。Stuffed Mamba 论文广泛使用此任务评估 Mamba-2 的长上下文召回能力，核心发现：(1) 8K 训练长度的模型在 ≤8K 内近乎完美但 >16K 后降至 ~0%；(2) 该任务对长度泛化的敏感度远高于 validation loss——不同 LR 的 loss 相似但 Passkey 精度差异巨大；(3) 370M Mamba-2 在 256K 训练后达到近乎完美的 Passkey Retrieval。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Stuffed Mamba 使用的 Passkey Retrieval prompt 模板：
```
There is important info hidden inside a lot of irrelevant text. Find it and memorize it.

[重复噪声文本: "The grass is green. The sky is blue..." × N]

The passkey is 34847. Remember it. 34847 is the passkey.

[重复噪声文本: "The grass is green. The sky is blue..." × M]

What is the passkey? The passkey is
```
评估指标: 模型输出是否精确包含 5 位 passkey（greedy decoding, FP32）。Needle 位置均匀分布：n 个样本的 needle 分别插入在位置 T×i/n (i=0,...,n-1)。论文使用 accuracy=N_correct/N_total，与 ∞Bench 的 passkey 设置一致。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
使用方式：作为长上下文能力的主要验证指标（而非次要指标），因其对长度泛化的敏感度远高于 perplexity。Stuffed Mamba 的实践：(1) 用 Passkey Retrieval 而非 validation loss 进行 LR 选择和 checkpoint 选择；(2) 在 T_forget 搜索中使用 Passkey 准确率 >95% 作为 T_recall 的定义；(3) 均匀 needle 位置确保评估所有深度（开头到结尾）。注意：greedy decoding 给出最佳结果（其他 decoding 参数显著降低精度）；BF16 精度下 Δ_t 和 α_t 有 ~1e-3 误差但不影响主要结论。局限性：Passkey 是简单合成任务，完美准确率不一定翻译为真实长上下文任务的表现。Stuffed Mamba 验证了继续预训练后 370M 在 256K 达到近乎完美（首个 <1B 模型在此长度达到此性能）。

涉及论文标题：
- Stuffed_Mamba__State_Collapse_and_State_Capacity_of_RNN-Based_Long-Context_Modeling

---
