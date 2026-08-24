# From Hours to Minutes: Lossless Acceleration of Ultra Long Sequence Generation up to 100K Tokens

Tong Wu\*♠, Junzhe Shen\*♠♡, Zixia Jia♠, Yuxuan Wang♠ and Zilong Zheng♠⊠ ♠ NLCo Lab, BIGAI 

Capable LUMIA Lab, Shanghai Jiao Tong University

Generating ultra-long sequences with large language models (LLMs) has become increasingly crucial but remains a highly time-intensive task, particularly for sequences up to 100K tokens. While traditional speculative decoding methods exist, simply extending their generation limits fails to accelerate the process and can be detrimental. Through an in-depth analysis, we identify three major challenges hindering efficient generation: frequent model reloading, dynamic key-value (KV) management and repetitive generation. To address these issues, we introduce TOKENSWIFT, a novel framework designed to substantially accelerate the generation process of ultra-long sequences while maintaining the target model's inherent quality. Experimental results demonstrate that TOKENSWIFT achieves over 3× speedup across models of varying scales (1.5B, 7B, 8B, 14B) and architectures (MHA, GQA). This acceleration translates to hours of time savings for ultra-long sequence generation, establishing TOKENSWIFT as a scalable and effective solution at unprecedented lengths. Code can be found at github.com/bigai-nlco/TokenSwift.

<span id="page-0-0"></span>> **[图片提取文字 (无描述)]:**
> TokenSwift 100K tokens AR 100000 AR 100K tokens Time: 90 min TokenSwift Time: 4.9 hours Fast Generation 80000 Slow Generation (TokenSwift) **Generated Tokens** (AR) 60000-40000-20000 0 50 100 200 150 250 300 Time (minutes)
![](_page_0_Figure_6.jpeg)

*Figure 1.* Comparison of the time taken to generate 100K tokens using autoregressive (AR) and TokenSwift with prefix length of 4096 on Llama3.1–8b. As seen, TokenSwift accelerates the AR process from nearly 5 hours to just 90 minutes.

## 1. Introduction

Recent advances in large language models (LLMs), amplified by their long context capacities (Wu et al., 2024; Ding et al., 2024), have demonstrated remarkable proficiency in intricate reasoning (Jaech et al., 2024; Guo et al., 2025), agentic thinking (Shinn et al., 2023; Yao et al., 2023; Li et al., 2024a), and creative writing (Wang et al.,

<sup>\*</sup> Equal contribution.

Correspondence to: Zilong Zheng <zlzheng@bigai.ai>.

[2023;](#page-15-3) [Mikhaylovskiy,](#page-14-1) [2023\)](#page-14-1), *etc*. These advancements necessitate the ability to generate lengthy sequences, *e.g*., o1-like [\(Jaech et al.,](#page-13-1) [2024\)](#page-13-1) reasoning tends to generate protracted chain-of-thought trajectories before reaching final conclusions. However, a critical challenge impeding the practical deployment of such applications is the extensive time required to produce ultra-long sequences. For instance, generating 100K tokens with LLaMA3.1- 8B can take approximately five hours (Figure [1\)](#page-0-0), a duration that is impractically long for the development of sophisticated applications, let alone recent gigantic models such as LLaMA3.1-405B [\(AI@Meta,](#page-12-0) [2024\)](#page-12-0) and DeepSeek-600B [\(Liu et al.,](#page-14-2) [2024a\)](#page-14-2). Addressing this bottleneck is essential for harnessing the full potential of LLMs in real-world scenarios.

A straightforward solution is to take advantage of recent success in speculative decoding (SD) [\(Leviathan et al.,](#page-14-3) [2023;](#page-14-3) [Chen et al.,](#page-12-1) [2023\)](#page-12-1), which employs a *draft-then-verify* strategy to expedite generation while preserving *lossless* accuracy; see Appendix [A](#page-17-0) and Section [5.1](#page-10-0) for detailed background and relevant literature. However, these methods are generally tailored for generating short sequences, *e.g*., TriForce [\(Sun et al.,](#page-15-4) [2024a\)](#page-15-4) and MagicDec [\(Chen et al.,](#page-13-3) [2024a\)](#page-13-3) are limited to generating 256 and 64 tokens, respectively. Directly extending their generation length to 100K tokens would inevitably encounter failures due to KV cache budget constraints. Furthermore, even when applied to optimized KV cache architectures such as Group Query Attention (GQA), these methods yield only marginal acceleration gains for short-sequence generation, as evidenced in Tables [1](#page-2-0) and [3.](#page-6-0) This observation leads to a pivotal research question:

*Is it possible to achieve model-agnostic lossless accelerations, akin to those seen in short-sequence SDs, for generating ultra-long sequences, with minimal training overhead?*

To answer this question, we conduct an in-depth analysis ([§2\)](#page-1-0) and identify three key challenges: **(1)** *frequent model reloading*: frequently reloading model for each token generation introduces a significant delay, primarily due to memory access times rather than computation. **(2)** *Prolonged Growing of KV Cache*, the dynamic management of key-value (KV) pairs, which grow with the sequence length, adds complexity in maintaining model efficiency. **(3**) *repetitive content generation*, the issue of repetitive generation becomes more pronounced as the sequence length increases, leading to degraded output quality.

Building on these insights, we introduce our framework TOKENSWIFT, which utilizes *n*-gram retrieval and dynamic KV cache updates to accelerate ultra-long sequence generation. Specifically, we employ *multi-token generation* and *token reutilization* to enable the LLM (*i.e*. target model) to draft multiple tokens in a single forward pass, alleviating the first challenge of frequent model reloading ([§3.2\)](#page-3-0). As the generation progresses, we *dynamically update* the partial KV cache at each iteration, reducing the KV cache loading time ([§3.3\)](#page-4-0). Finally, to mitigate the issue of repetitive outputs, we apply *contextual penalty* to constrain the generation process, ensuring the diversity of output ([§3.4\)](#page-4-1).

In [§4,](#page-5-0) we conduct extensive experiments to evaluate TOKENSWIFT across different model scales and architectures. In summary, we highlight our advantages as:

- 1. To the best of our knowledge, TOKENSWIFT is the **first** to accelerate ultra-long sequence generation up to 100K with lossless accuracy of target LLMs, while demonstrating significant superiority over enhanced baselines.
- 2. TOKENSWIFT consistently achieves over **3**ˆ speedup compared to AR across varying prefix lengths, model architectures, and model scales in generating 100K tokens, reducing the AR process from nearly 5 hours to 90 minutes on LLaMA3.1-8b.
- 3. TOKENSWIFT achieves progressively higher speedup compared to AR as the generation length increases, while enhancing diversity in ultra-long sequence generation (as measured by *Distinct-n* [\(Li et al.,](#page-14-4) [2016\)](#page-14-4)).

