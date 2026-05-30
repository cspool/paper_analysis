# Nuno Gonçalves <sup>1</sup> Marcos Treviso <sup>2</sup> André F. T. Martins <sup>123</sup>

## **Abstract**

The computational cost of softmax-based attention in transformers limits their applicability to long-context tasks. Adaptive sparsity, of which  $\alpha$ -entmax attention is an example, offers a flexible data-dependent alternative, but existing implementations are inefficient and do not leverage the sparsity to obtain runtime and memory gains. In this work, we propose ADASPLASH, which combines the efficiency of GPU-optimized algorithms with the sparsity benefits of  $\alpha$ -entmax. We first introduce a hybrid Halley-bisection algorithm, resulting in a 7-fold reduction in the number of iterations needed to compute the  $\alpha$ -entmax transformation. Then, we implement custom Triton kernels to efficiently handle adaptive sparsity. Experiments with RoBERTa and ModernBERT for text classification and single-vector retrieval, along with GPT-2 for language modeling, show that our method achieves substantial improvements in runtime and memory efficiency compared to existing  $\alpha$ -entmax implementations. It approaches and in some cases surpasses—the efficiency of highly optimized softmax implementations like FlashAttention-2, enabling long-context training while maintaining strong task performance.<sup>1</sup>

### 1. Introduction

Central to the success of transformers (Vaswani et al., 2017) lies the attention mechanism, where each token in a sequence attends directly to every other token. Attention probabilities are computed through the **softmax** transformation, which always assigns a nonzero probability to every token. However, for long context inputs, the accumulation of small probabilities can lead to dispersion (Veličković et al., 2025).

Proceedings of the 42<sup>nd</sup> International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

![](_page_0_Figure_10.jpeg)

<span id="page-0-1"></span>Figure 1. Runtime (Fwd+Bwd) as a function of input sparsity for non-causal attention. While the highly-optimized FlashAttention-2 maintains a constant runtime across varying levels of sparsity, ADASPLASH effectively leverages sparsity to obtain speed-ups, eventually outperforming FlashAttention-2 as sparsity grows.

In fact, previous research shows that attention probabilities tend to peak around a small number of tokens (Voita et al., 2019; Treviso et al., 2022), which suggests that model performance and computational efficiency can be increased by leveraging attention sparsity. This has motivated methods that predefine sparse masks (Beltagy et al., 2020; Zaheer et al., 2020b), rely on clustering-based strategies (Kitaev et al., 2020), or low-rank approximate attention (Choromanski et al., 2021; Peng et al., 2021; Xiong et al., 2021; Chen et al., 2021). Some of these techniques show the potential of sparsity to mitigate memory and computation bottlenecks, but they often require architectural modifications or crude approximations, limiting their flexibility and generality.

A related line of research explores adaptive and differentiable sparse activations as surrogates of softmax, such as **sparsemax** (Martins & Astudillo, 2016) and, more broadly, the  $\alpha$ -entmax family (Peters et al., 2019; Correia et al., 2019). By assigning zero probability to irrelevant tokens, these activations eliminate their residual influence, reducing the dilution of attention scores and potentially improving both performance and interpretability. Unfortunately, existing algorithms and implementations for these adaptive sparse activations do not exploit the sparsity, being slower than softmax-based attention and struggling to scale

<sup>&</sup>lt;sup>1</sup>Instituto Superior Técnico, Universidade de Lisboa, Portugal <sup>2</sup>Instituto de Telecomunicações, Lisbon, Portugal <sup>3</sup>Unbabel, Lisbon, Portugal. Correspondence to: Nuno Gonçalves <nuno.m.goncalves@tecnico.ulisboa.pt>.

<span id="page-0-0"></span><sup>&</sup>lt;sup>1</sup>Code: https://github.com/deep-spin/adasplash

effectively with context length, primarily due to the lack of hardware-optimized implementations like FlashAttention-2 [\(Dao,](#page-9-4) [2024\)](#page-9-4) or support from programming models like FlexAttention [\(Dong et al.,](#page-9-5) [2024\)](#page-9-5).

This paper addresses this problem by providing new algorithms and implementations to improve the computational efficiency of the family of α-entmax activations. Our main contributions include a faster and GPU-friendly algorithm for calculating α-entmax, alongside a Triton kernel [\(Tillet](#page-11-6) [et al.,](#page-11-6) [2019\)](#page-11-6) for computing entmax-based attention, which we call ADASPLASH. In particular, ADASPLASH advances the goal of supporting training of adaptively sparse models with longer context lengths, as shown in Figure [1.](#page-0-1) We demonstrate the potential and scalability of our approach through experiments with synthetic data and with several natural language processing benchmarks for encoder-only and decoder-only models, achieving substantial improvements over previous α-entmax implementations and approaching (sometimes surpassing) the efficiency of softmaxbased attention with FlashAttention-2, with strong performance on downstream tasks.

