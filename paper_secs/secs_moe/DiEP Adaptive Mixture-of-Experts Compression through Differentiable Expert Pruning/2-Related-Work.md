# 2 Related Work

### 2.1 Sparse Mixture-of-Experts Models (SMoE)

It selectively activates a small subset of specialized networks (experts) for each input, enabling efficient model scaling [\[3,](#page-9-7) [18\]](#page-9-8). In early research, Shazeer et. al. [\[39\]](#page-11-0) introduced the Sparsely-Gated MoE layer, demonstrating the effectiveness of selective expert activation. [21](#page-10-9) advanced SMoE by implementing a distributed architecture that enabled efficient scaling across multiple devices. Recent studies have further refined SMoE architecture based on SOTA LLMs [\[42\]](#page-11-4). Mixtral models [\[19\]](#page-9-3) demonstrated successful scaling with a balanced approach of using two experts per token; Qwen-MoE [\[42\]](#page-11-4) and DeepSeek-MoE [\[8,](#page-9-1) [14\]](#page-9-9) explored larger expert pools with selective activation. They have attracted great attention from the AI community. Despite these advances, current SMoE-LLM architectures require huge memory to load trillion parameters and suffer from low expert utilization during inference.

