# SeerAttention-R: Sparse Attention Adaptation for Long Reasoning

 $\begin{array}{cccccccccccccccccccccccccccccccccccc$ 

Microsoft Research <sup>2</sup> The University of Hong Kong
Huazhong University of Science and Technology
Peking University <sup>5</sup> Tsinghua University

#### **Abstract**

We introduce SeerAttention-R, a sparse attention framework specifically tailored for the long decoding of reasoning models. Extended from SeerAttention, SeerAttention-R retains the design of learning attention sparsity through a self-distilled gating mechanism, while removing query pooling to accommodate autoregressive decoding. With a **lightweight** plug-in gating, SeerAttention-R is **flexible** and can be easily integrated into existing pretrained model without modifying the original parameters. We demonstrate that SeerAttention-R, trained on just 0.4B tokens, maintains near-lossless reasoning accuracy with 4K token budget in AIME benchmark under large sparse attention block sizes (64/128). Using TileLang, we develop a highly optimized sparse decoding kernel that achieves near-theoretical speedups of up to 9x over FlashAttention-3 on H100 GPU at 90% sparsity. Code is available at: https://github.com/microsoft/SeerAttention.

## 1 Introduction

Recent reasoning-focused models such as OpenAI o1 [28], DeepSeek-R1 [23], and Qwen3 [66] demonstrate that models' capabilities improve significantly through test-time scaling. By generating longer sequences during inference, these models are able to think and reason more effectively before producing an answer. Empirically, longer generations correlate with stronger reasoning performance. For instance, Qwen3-14B [66] outperforms DeepSeek-R1-Distill-Qwen-14B [23] while producing longer responses on average. Similarly, harder benchmarks such as AIME24 [48] require more tokens per generation than easier ones like MATH-500 [25].

However, deeper reasoning introduces increasing efficiency challenges. Due to the auto-regressive nature of decoding, later tokens must attend to a longer context, increasing compute and memory demands for the KV cache. As a result, the per-token generation cost grows linearly, while the overall generation cost increases quadratically.

Sparse attention offers a promising approach to addressing the long-sequence efficiency challenges. While it has been studied in general language modeling, its application to reasoning models, which require prolonged decoding, remains underexplored. Our experiment using oracle sparsity (Section 4.2) shows that attention in reasoning models is also inherently sparse, activating only a subset of important tokens is sufficient to maintain the model's reasoning capability. The key challenge lies in effectively identifying and leveraging this intrinsic sparsity.

<sup>\*</sup> Equal contribution.  $\diamond$  Corresponding author.

In this work, we extend SeerAttention [\[19\]](#page-13-2) to SeerAttention-R, a sparse attention framework aimed to improve the long decoding efficiency of reasoning models. SeerAttention was originally designed to improve prefill efficiency by selectively activating important attention blocks through a lightweight, self-distilled attention gating mechanism at post-training time. SeerAttention-R retains the core design of self-ditilled attention sparsity and introduces modifications to support efficient decoding. Specifically, it removes sequence-level pooling of query to accommodate auto-regressive decoding and adopts a shared sparsity design aligned with Grouped Query Attention (GQA) to enhance hardware efficiency. SeerAttention-R can be integrated into any standard transformer-based pretrained model by adding the learnable gate to the attention layer, without fine-tuning original model parameters.

We apply SeerAttention-R to multiple reasoning-focused open-source models, including Qwen3-4B, 8B, 14B [\[66\]](#page-16-0) and DeepSeek-R1-Distill-Qwen-14B [\[23\]](#page-13-0), and evaluate them on several reasoning benchmarks: AIME24, AIME25 [\[48\]](#page-15-0), MATH-500 [\[25\]](#page-13-1), and GPQA-Diamond [\[50\]](#page-15-1). Since SeerAttention-R only requires training the gating module, the distillation is lightweight with just 0.4B tokens from OpenR1-MATH-220K [\[17\]](#page-13-3) being sufficient. Across all models and tasks, SeerAttention-R consistently outperforms the Quest [\[57\]](#page-15-2) baseline and maintains near-lossless accuracy under a 4k token budget. Notably, the accuracy gap further diminishes as model size increases. More importantly, this learnable approach enables more coarse-grained sparse attention (e.g., a block size of 64 or 128), which further reduces the overhead from sparse attention scheme and improve hardware efficiency.

We implement the block sparse flash decoding kernel using both TileLang [\[1\]](#page-12-0) and Triton [\[59\]](#page-15-3), and benchmark it on an H100 GPU with FlashAttention-3 (FA3) [\[51\]](#page-15-4) as the baseline. Across a range of combination of sequence lengths, batch sizes, and sparsity levels, our TileLang-based kernel consistently outperforms both Triton and FA3. The gains are especially pronounced at large sequence lengths and batch sizes. For example, at batch size 16 and sequence length ≥ 32k, our TileLang kernel achieves near-theoretical speedups of up to 8.6× at 90% sparsity over the FA3 baseline, and delivers a 1.7× speedup compared to the Triton counterpart.

