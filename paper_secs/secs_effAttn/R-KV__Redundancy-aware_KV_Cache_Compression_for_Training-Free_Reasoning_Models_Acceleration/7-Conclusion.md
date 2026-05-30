# 7 Conclusion

We introduced R-KV, a novel decoding-time KV cache compression method tailored to the challenges of complex reasoning in large language models (LLMs). Reasoning models often generate long, redundant outputs that impose substantial memory and computational burdens during inference. R-KV addresses this by jointly scoring token importance and redundancy, enabling the retention of essential reasoning content while discarding repetitive or uninformative tokens. This dynamic and attention-guided strategy allows R-KV to preserve nearly full model performance using only 10–34% of the original KV cache—substantially outperforming prior compression methods.

Extensive throughput and efficiency analysis demonstrate that R-KV enables up to 13× larger batch sizes and over 9× speedup in long-sequence generation scenarios compared to FullKV, with particularly strong gains under constrained memory budgets. With its training-free and modelagnostic design, R-KV provides a scalable and deployment-ready solution for reasoning LLMs, especially in streamlining the rollout phase of reinforcement learning workflows.

