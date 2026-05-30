# 5 Conclusions

This study provides the largest-scale empirical analysis of training-free sparse attention to date, covering three model families (Qwen 2.5, Llama 3.1, Gemma 3), model scales (4B–72B parameters), sequence lengths (16K–128K tokens), sparsity levels up to 0.95 (i.e., 1/20 attention budget), and nine diverse long-sequence tasks. We organise the rapidly evolving landscape of sparse attention methods into a taxonomy along four design axes and introduce novel benchmarks consisting of natural texts that are fully controllable yet challenging. Our analysis yields three key insights.

Evidence of effectiveness. Sparse attention enables larger models to outperform smaller dense ones at equivalent computational cost, improving the Pareto frontier. Thus, sparsity becomes crucial for optimal LLM scaling.

Practical deployment guidance. Method selection should be task-aware: fine-grained token selection (e.g., Vertical-Slash) excels at retrieval, chunkbased methods (e.g., Block-Sparse) suit reasoning and aggregation, and Quest provides robust decoding across most scenarios.

Design recommendations. Longer sequences tolerate higher sparsity while maintaining accuracy. This suggests that fixed-budget methods deployed in production are suboptimal; future designs should adapt sparsity levels to sequence length, possibly growing the token budget sublinearly.

## Limitations

First, we evaluate only training-free sparse attention methods. Training-based approaches could reduce train-inference mismatch, but require substantial computational resources and access to proprietary training data.

Second, our experimental coverage, while extensive, is bounded. We evaluate three model families (Qwen 2.5, Llama 3.1, Gemma 3) that met our methodological requirements for controlled scaling experiments with native long-context support; other families may exhibit different behaviour. We test only instruction-tuned models; reasoning models with extended chain-of-thought capabilities (e.g., o1, DeepSeek-R1) may have different attention patterns and sparsity tolerance. Our nine tasks, though selected to span diverse dispersion levels, processing scopes, and data naturalness, do not exhaustively cover all long-context scenarios—openended tasks like summarisation were excluded due to unreliable automated metrics. Additionally, experiments at 128k tokens are limited due to low baseline performance and lack of robustness across models; more conclusive evidence on how sequence length affects sparse attention scaling requires stronger long-context models.

Third, we report hardware-agnostic computational costs (FLOPs and memory access) rather than wall-clock timings. Actual speedups depend on hardware, batch size, and implementation quality, which vary across deployment environments.

Fourth, we do not investigate interactions between sparse attention and other model efficiency techniques such as quantisation, weight pruning, or mixture-of-experts sparsity. These methods are often combined in practice, and their joint effects on attention sparsity tolerance remain unexplored.

