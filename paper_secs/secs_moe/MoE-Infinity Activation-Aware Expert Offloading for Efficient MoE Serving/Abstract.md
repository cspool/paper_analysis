# Abstract

This paper presents MOE-INFINITY, an efficient MoE inference system designed for personal machines with limited GPU memory capacity. The key idea for MOE-INFINITY is that on personal machines, which are often single-user environments, MoE-based LLMs typically operate with a batch size of one. In this setting, MoE models exhibit a high degree of activation sparsity, meaning a small number of experts are frequently reused in generating tokens during the decode phase. Leveraging this idea, we design a sparsityaware expert cache, which can trace the sparse activation of experts during inference and carefully select the trace that represents the sparsity pattern. By analyzing these selected traces, MOE-INFINITY guides the replacement and prefetching of the expert cache, providing 3.1–16.7× per-token latency improvements over numerous state-of-the-art systems, including vLLM, Ollama, DeepSpeed and BrainStorm across various MoE models (DeepSeek and Mixtral) when handling different LLM tasks.

Code: [https://github.com/Efficie](https://github.com/EfficientMoE/MoE-Infinity) [ntMoE/MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity)

