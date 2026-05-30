## LongBench

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

LongBench（Bai et al., ACL 2024）是首个中英双语多任务长上下文 benchmark，21 数据集/4750 样本，6 大类：Single-Doc QA、Multi-Doc QA、Summarization、Few-shot Learning、Synthetic、Code。英文 avg 6711 词，中文 avg 13386 字符。提供 LongBench-E 子集（均匀长度分布：0-4k/4k-8k/8k+）。

在 WindowKV 中用于评估 Qwen2.5-1.5B 和 LLaMA3-8B 在 KV size=512/1024/2048 下的 16 子任务。WindowKV 以 12% KV cache 取得最多次 SOTA。后续版本：LongBench v2 (ACL 2025, 503 多选题 8K-2M 词) 和 LongBench Pro (2026, 1500 样本 8K-256K)。

术语一般如何实现？如何使用？

官方仓库 https://github.com/THUDM/LongBench。WindowKV 使用标准 prompt，贪心解码。QA 类 F1，Summarization Rouge-L，Few-shot Accuracy，Code Edit Similarity。

涉及论文标题：
- WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference

---
