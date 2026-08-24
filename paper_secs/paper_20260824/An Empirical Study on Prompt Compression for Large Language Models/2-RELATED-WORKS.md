# 2 RELATED WORKS

### 2.1 LLM'S LONG CONTEXT PROCESSING METHOD

Given the performance limitations and computational overhead of LLMs [\(Wang et al., 2024\)](#page-13-2), how to effectively apply LLMs to tasks involving lengthy textual inputs is a persistent challenge. Various solutions have emerged to address this issue, encompassing techniques such as length extrapolation [\(Chen et al., 2021;](#page-9-1) [Shaw et al., 2018\)](#page-12-2), attention approximation [\(Winata et al., 2019;](#page-13-3) [Wang et al.,](#page-13-4) [2020\)](#page-13-4), attention-free transformers [\(Gu et al., 2021;](#page-10-3) [Orvieto et al., 2023\)](#page-12-3), model compression [\(Lee](#page-11-3) [et al., 2023;](#page-11-3) [Ma et al., 2023\)](#page-11-4), and hardware-aware transformers [\(Dao et al., 2022;](#page-9-2) [Liu & Abbeel,](#page-11-5) [2023\)](#page-11-5).

In this paper, we focus mainly on the prompt compression techniques, especially those that do not rely on the internal states or parameters of LLMs and operate in a text-in, text-out manner. These methods present several advantages: they can be seamlessly integrated with different model architectures without requiring additional modifications, and they are particularly beneficial for online models, helping to reduce the economic costs associated with API calls.

### 2.2 PROMPT COMPRESSION

Figure [1](#page-1-0) illustrates the concept of prompt compression, and the compression ratio ρ for prompt compression is defined as:

<span id="page-2-0"></span>> **[图片提取文字 (无描述)]:**
> Optimization LLM Train LLM Compressor Compressor Scorer Reward Scorer (a) Reinforcement Learning (RL) (b) LLM Scoring (c) LLM Annotation
![](_page_2_Picture_1.jpeg)

Figure 2: Categories of prompt compression methods. These methods can be grouped into three main categories: (a) RL-based methods, which use heuristic rewards to optimize the compressor, (b) LLM scoring-based methods, which use another language model to score each token in a single autoregressive step and decide to keep or discard each token based on its score, and (c) LLM annotation-based methods, which use LLMs to annotate data for training a small model specifically designed for prompt compression.

$$\rho = 1 - \frac{L_c}{L_o}.\tag{1}$$

Here L<sup>c</sup> is the compressed context length and L<sup>o</sup> is the original context length. Many prompt compression methods have been developed to handle long prompts in LLMs. KiS [\(Laban et al.,](#page-11-6) [2021\)](#page-11-6) and SCRL [\(Ghalandari et al., 2022\)](#page-10-4) leverage reinforcement learning (RL) to train models for text compression without the need for ground-truth data, optimizing specific objectives such as fluency and simplicity. Recently, with advances in LLMs, some methods [\(Li et al., 2023;](#page-11-1) [Jiang et al.,](#page-10-5) [2023;](#page-10-5) [2024;](#page-10-2) [Pan et al., 2024\)](#page-12-0) employ pre-trained language models and various strategies to identify and prune redundant or less informative content.

Besides text-based methods, there are techniques aimed at compressing or trimming the hidden states or KV caches [\(Liu et al., 2023b;](#page-11-7) [Zhang et al., 2023;](#page-13-5) [Xiao et al., 2024;](#page-13-6) [Ge et al., 2024\)](#page-10-6). However, these methods are separate from our study and are not easily applicable to various model architectures or closed-source LLMs.

