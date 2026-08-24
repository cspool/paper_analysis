# 6 Conclusion

*SuperWriter* addresses the challenge of long-form text generation by introducing a structured writing process—planning, writing, and refining—guided by the *SuperWriter*-agent. This approach teaches the model to "think before writing" and produces high-quality supervision signals. Combined with a hierarchical DPO strategy, the model learns to align its output across all writing stages.

Experiments show strong results: *SuperWriter*-LM outperforms all same-size models on Writing-Bench and even exceeds the 671B DeepSeek-R1 model [\[12\]](#page-10-2) in key domains. It also wins over 98% of real-user comparisons against top open-source baselines. These results confirm the value of multi-stage generation and structured preference learning for improving writing quality.

