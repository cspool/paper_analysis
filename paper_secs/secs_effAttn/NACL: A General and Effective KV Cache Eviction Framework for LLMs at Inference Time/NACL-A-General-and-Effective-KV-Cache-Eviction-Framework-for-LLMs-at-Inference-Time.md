# NACL: A General and Effective KV Cache Eviction Framework for LLMs at Inference Time

Yilong Chen<sup>1,2\*</sup>, Guoxia Wang<sup>3\*</sup>, Junyuan Shang<sup>3†</sup>, Shiyao Cui<sup>1</sup>, Zhenyu Zhang<sup>3</sup>, Tingwen Liu<sup>1,2†</sup>, Shuohuan Wang<sup>3</sup>, Yu Sun<sup>3</sup>, Dianhai Yu<sup>3</sup>, Hua Wu<sup>3</sup>

<sup>1</sup> Institute of Information Engineering, Chinese Academy of Sciences
 <sup>2</sup> School of Cyber Security, University of Chinese Academy of Sciences
 <sup>3</sup> Baidu Inc.

{chenyilong, cuishiyao, liutingwen}@iie.ac.cn {wangguoxia, shangjunyuan, zhangzhenyu07, wangshuohuan, sunyu02}@baidu.com

## **Abstract**

Large Language Models (LLMs) have ignited an innovative surge of AI applications, marking a new era of exciting possibilities equipped with extended context windows. However, hosting these models is cost-prohibitive mainly due to the extensive memory consumption of KV Cache involving long-context modeling. Despite several works proposing to evict unnecessary tokens from the KV Cache, most of them rely on the biased local statistics of accumulated attention scores and report performance using unconvincing metric like perplexity on inadequate short-text evaluation. In this paper, we propose NACL, a general framework for long-context KV cache eviction that achieves more optimal and efficient eviction in a single operation during the encoding phase. Due to NACL's efficiency, we combine more accurate attention score statistics in PROXY-TOKENS EVICTION with the diversified random eviction strategy of RANDOM EVICTION, aiming to alleviate the issue of attention bias and enhance the robustness in maintaining pivotal tokens for long-context modeling tasks. Notably, our method significantly improves the performance on short- and long-text tasks by 80% and 76% respectively, reducing KV Cache by up to  $5\times$  with over 95% performance maintenance. The code is available at https: //github.com/PaddlePaddle/Research/ tree/master/NLP/ACL2024-NACL.

#### 1 Introduction

Large Language Models (LLMs) with longer context window (Touvron et al., 2023; Xiong et al., 2023; Jiang et al., 2023; Anthropic, 2023; OpenAI, 2023) have emerged recently for better conducting long conversations, summarizing long documents,

![](_page_0_Picture_11.jpeg)

Figure 1: Traditional eviction algorithms perform stepby-step greedy search for tokens for eviction. Our framework searches globally for tokens within a chunk and then performs one single eviction.

or debugging code at the repository level (Bai et al., 2023). However, their deployment is costly and infeasible on fixed memory hardware, mainly due to the surprisingly large memory consumption of KV Cache mechanism. For instance, a 7 billion-parameter model with an input batch size of 4 and a sequence length of 32k results in 64GB of KV cache,  $4.7 \times$  larger than the model weights.

To mitigate the pressure on the scarce GPU memory from using KV cache, a number of studies (Zhang et al., 2023; Liu et al., 2023c; Ge et al., 2023; Xiao et al., 2023) have explored sparsity among Transformer attention blocks to evict unnecessary tokens from the KV cache. For instance, H2O (Zhang et al., 2023) utilized the local statistics of accumulated attention scores to retain a balance of recent and heavy hitter tokens during generation. Window Attention based methods (Xiao et al., 2023) proposed to keep the initial tokens which is proven vital for generation fluency. This line of work reduced the memory footprint of KV cache

Equal contribution. Work done at Baidu Inc.

<sup>&</sup>lt;sup>†</sup>Corresponding author. <sup>‡</sup> Project lead.

for efficient inference with negligible loss in generation quality. In addition, the above methods do not require costly retraining which is more suitable for current open-sourced LLMs [\(Touvron et al.,](#page-10-0) [2023;](#page-10-0) [Jiang et al.,](#page-9-0) [2023\)](#page-9-0), compared to those that need specific attention mechanism adaptation [\(Beltagy](#page-9-5) [et al.,](#page-9-5) [2020;](#page-9-5) [Kitaev et al.,](#page-9-6) [2020;](#page-9-6) [Shazeer,](#page-10-4) [2019;](#page-10-4) [Ainslie et al.,](#page-8-1) [2023\)](#page-8-1).

However, we argue that the performance reported in the above methods is over-optimistic, as the evaluation metric and tested benchmark is not sufficient. LLMs may fail in real-life long-context modeling tasks [\(Bai et al.,](#page-9-2) [2023\)](#page-9-2), though they can achieve low language modeling perplexity which is untrustworthy used as the golden metric in current studies [\(Xiao et al.,](#page-10-3) [2023;](#page-10-3) [Han et al.,](#page-9-7) [2023\)](#page-9-7). Furthermore, the local statistics of accumulated attention score is observed to be biased (see Fig. [2\)](#page-3-0), especially in long context input, meaning that it should be carefully used as the only strategy for measuring the importance of tokens.

To fill the gap, we propose NACL, a general and effective KV cache eviction framework to unleash the power of LLMs for long-context modeling with limited memory budgets. NACL specifically formulates the eviction task in encoding phase which is different from the commonly used one-tokenin one-token out eviction procedure in generation phase. In encoding phase, the eviction can be effectively implemented to apply only once on the whole input by progressively evicting KV caches layer by layer. The one-eviction formulation benefits current eviction policies in a more efficient and optimal way, as multiple costly eviction operations can be combined, then the global statistics of attention scores can be utilized.

Based on the above formulation, we present PROXY-TOKENS EVICTION which exploits the global statistics of attention scores gathered from proxy tokens for eviction. In practise, the proxy tokens can be selected from the question input, commonly located at the end of a long text. Intuitively, these proxy tokens are more capable of retaining the task-specific tokens in KV cache. As a result, PROXY-TOKENS EVICTION alleviates the *attention bias problem* (see Sec. [4\)](#page-3-1) occurred in methods using local statistics [\(Zhang et al.,](#page-10-2) [2023;](#page-10-2) [Oren et al.,](#page-9-8) [2024\)](#page-9-8) or task-irrelevant proxy tokens [\(Liu et al.,](#page-9-3) [2023c\)](#page-9-3).

However, PROXY-TOKENS EVICTION also relies heavily on the statistic of attention scores

which may be untrustworthy in long-context input. Thus, we incorporate RANDOM EVICTION, a random eviction policy, into PROXY-TOKENS EVICTION. RANDOM EVICTION randomly samples tokens to evict from the probability distribution in PROXY-TOKENS EVICTION with different seed on attention heads and layers. This diversified randomness enhances the model's robustness to maintain potentially important tokens in long text generation.

We conducted extensive experiments on a single NVIDIA A100 (80GB) GPU on representative open-sourced LLMs: LLaMA2-base, LLaMA2- Chat [\(Touvron et al.,](#page-10-0) [2023\)](#page-10-0), and evaluated them on both short- and long-text modeling tasks from lmeval-harness [\(Gao et al.,](#page-9-9) [2021\)](#page-9-9) and LongBench [\(Bai](#page-9-2) [et al.,](#page-9-2) [2023\)](#page-9-2). The experiments show that NACL performs KV cache eviction efficiently with negligible degradation on model quality (i.e., saving the inference memory usage of KV cache by up to 5× with over 95% maintenance). Specifically, NACL achieve 80% and 75% performance improvement on short- and long- text modeling tasks, respectively, with 50% KV cache reduction, compared to current eviction methods.

## 2 Related Work

