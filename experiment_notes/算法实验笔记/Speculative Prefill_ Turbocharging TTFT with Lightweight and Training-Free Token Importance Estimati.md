## Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 SPECPREFILL，一种训练无关（training-free）的 token 重要性估计框架。核心算法：利用一个较小的"推测器"模型（speculator，如 Llama-3.1-8B-Instruct BF16）计算 prompt 中每个 token 的注意力分数，通过以下步骤筛选出局部重要 token：1) Look-ahead decoding（向前解码 N=8 步以缓解 attention sink 和 proximity bias）；2) Max-mean aggregation（对 [N, L, S, H] 注意力张量，在 H 和 L 维度取 max 以突出显著 token，在 N 维度取 mean 以公平贡献）；3) Chunk selection + 1D average pooling（将 context 分块，块内平均 token 分数后选 Top-K 块，利用邻近 token 相似性降低方差）；4) Position ID restoration（保持原始非连续 position IDs 送入主模型）。仅将筛选出的 token 子集送入主模型（Llama-3.1-70B-Instruct BF16 / 405B-Instruct FP8）进行 prefill，跳过其余 token 的 attention + MLP 计算和 all-reduce 通信，TTFT 加速正比于 token 丢弃率。

  实验比较：(1) Long context 质量：LongBench 六类任务（Single-Doc QA, Multi-Doc QA, Summarization, Few-Shot, Code, Synthetic），对比 Baseline（Llama 原始模型）、RAG-LLAMA（sentence-level RAG）、LLMLingua（文本级压缩）、MInference（sparse attention）；(2) Synthetic context probing：RULER suite（4K-128K），10% 保持率；(3) Standard short tasks：MMLU, IFEval, GSM8K 8-shot, HumanEval, MBPP, Arc Challenge, GPQA 8-shot；(4) Efficiency：端到端 QPS 实验（LongBench 数据集、vLLM server + OpenAI API client）、合成数据 TTFT 测量（不同 batch size × sequence length）。
  实验变体：SPECPREFILL（仅原始 attention 分数）、SPECPREFILL Full（所有技术，无 look-ahead）、SPECPREFILL Full LAH（所有技术 + 8 步 look-ahead）。
