# 1 Introduction

Chain-of-Thought (CoT) prompting [\(Nye et al.,](#page-10-0) [2021;](#page-10-0) [Wei et al.,](#page-10-1) [2022;](#page-10-1) [Kojima et al.,](#page-9-0) [2022\)](#page-9-0) has emerged as a cornerstone strategy for enhancing Large Language Models (LLMs) in complex reasoning tasks. By eliciting step-by-step inference, CoT enables LLMs to decompose intricate problems into manageable subtasks, thereby improving their problem-solving performance [\(Yao et al.,](#page-10-2) [2023;](#page-10-2) [Wang et al.,](#page-10-3) [2023;](#page-10-3) [Zhou et al.,](#page-10-4) [2023;](#page-10-4) [Shinn](#page-10-5)

<span id="page-0-0"></span>> **[图片提取文字 (无描述)]:**
> (a) Original CoT LLM Pruning (b) TokenSkip **Efficiency**↑
![](_page_0_Picture_8.jpeg)

Figure 1: In contrast to vanilla CoT that generates all reasoning tokens sequentially, TokenSkip enables LLMs to *skip* tokens with less semantic importance (*e.g.,* ) and learn shortcuts between critical reasoning tokens, facilitating controllable CoT compression.

[et al.,](#page-10-5) [2023\)](#page-10-5). Recent advancements, such as OpenAI's o1 [\(OpenAI et al.,](#page-10-6) [2024\)](#page-10-6) and DeepSeek-R1 [\(DeepSeek-AI et al.,](#page-8-0) [2025\)](#page-8-0), further demonstrate that scaling up CoT lengths from hundreds to thousands of reasoning steps could continuously improve LLM reasoning. These breakthroughs have underscored CoT's potential to advance LLM capabilities, expanding the boundaries of AI-driven problem-solving.

Despite its effectiveness, the increased length of CoT sequences introduces substantial computational overhead. Due to the autoregressive nature of LLM decoding, longer CoT outputs lead to proportional increases in both inference latency and memory footprints of key-value cache. Additionally, the quadratic computational cost of attention layers further exacerbates this burden. These issues become particularly pronounced when CoT sequences extend into thousands of reasoning steps, resulting in significant computational costs and prolonged response times. While prior research has explored methods for selectively skipping reasoning steps [\(Ding et al.,](#page-8-1) [2024;](#page-8-1) [Liu et al.,](#page-9-1) [2024\)](#page-9-1), recent findings [\(Jin et al.,](#page-9-2) [2024;](#page-9-2) [Merrill and Sabharwal,](#page-9-3) [2024\)](#page-9-3) suggest that such reductions may conflict

<sup>\*</sup>[Corresponding Author](#page-10-5)

with test-time scaling (OpenAI, 2024; Snell et al., 2025), ultimately impairing LLM reasoning performance. Therefore, striking an optimal balance between CoT efficiency and reasoning accuracy remains a critical open challenge.

In this work, we delve into CoT efficiency and seek the answer to an important question: "Does every token in the CoT output contribute equally to deriving the answer?" We empirically analyze the semantic importance of tokens within CoT outputs and reveal that their contributions to the reasoning performance vary, as depicted in Figure 2. Building on this insight, we introduce TokenSkip, a simple yet effective approach that enables LLMs to skip less important tokens within CoT sequences and learn shortcuts between critical reasoning tokens, thereby allowing for controllable CoT compression with adjustable ratios. Specifically, as shown in Figure 1, TokenSkip constructs compressed CoT training data with various compression ratios, by pruning unimportant tokens from original LLM CoT trajectories. Then, it conducts a general supervised fine-tuning process on target LLMs with this training data, facilitating LLMs to automatically trim redundant tokens during reasoning.

We conduct extensive experiments across various models, including LLaMA-3.1-8B-Instruct and the Qwen2.5-Instruct series, using two widely recognized math reasoning benchmarks: GSM8K and MATH-500. The results validate the effectiveness of TokenSkip in compressing CoT outputs while maintaining robust reasoning performance. Notably, Owen2.5-14B-Instruct exhibits almost **NO** performance drop (less than 0.4%) with a 40% reduction in token usage on GSM8K. On the challenging MATH-500 dataset, LLaMA-3.1-8B-Instruct effectively reduces CoT token usage by 30% with a performance decline of less than 4%, resulting in a 1.4× inference speedup. Further analysis underscores the coherence of TokenSkip in specified compression ratios and its potential scalability with stronger compression techniques.

TokenSkip is distinguished by its low training cost. For Qwen2.5-14B-Instruct, TokenSkip finetunes only 0.2% of the model's parameters using LoRA. The size of the compressed CoT training data is no larger than that of the original training set, with 7,473 examples in GSM8K and 7,500 in MATH. The training is completed in approximately 2 hours for the 7B model and 2.5 hours for the 14B model on two 3090 GPUs. These characteristics make TokenSkip an efficient and repro-

ducible approach, suitable for use in efficient and cost-effective LLM deployment.

To sum up, our key contributions are:

- 1. To the best of our knowledge, this work is the *first* to investigate the potential of enhancing CoT efficiency through *token skipping*, inspired by the varying semantic importance of tokens in CoT trajectories of LLMs.
- We introduce TokenSkip, a simple yet effective approach that enables LLMs to skip redundant tokens within CoTs and learn shortcuts between critical tokens, facilitating CoT compression with adjustable ratios.
- 3. Our experiments validate the effectiveness of TokenSkip. When applied to Qwen2.5-14B-Instruct, TokenSkip reduces reasoning tokens by 40% (from 313 to 181) on GSM8K, with less than a 0.4% performance drop.

### 2 Background and Preliminaries

In this section, we discuss the relevant research background and present preliminary studies on token efficiency in CoT sequences, exploring its impact on the reasoning performance of LLMs.

### <span id="page-1-0"></span>2.1 Token Importance

We first investigate a critical research question to CoT efficiency: "Does every token in the CoT output contribute equally to deriving the answer?" In other words, we would like to know if there is any token redundancy in CoT sequences that could be eliminated to improve CoT efficiency.

Token redundancy has been recognized as a longstanding and fundamental issue in LLM efficiency (Hou et al., 2022; Zhang et al., 2023; Lin et al., 2024; Chen et al., 2024). Recently, it has garnered intensive research attention in prompt compression (Li et al., 2023; Jiang et al., 2023; Pan et al., 2024), which focuses on removing redundant tokens from the input prompt to reduce API token usage. To address this issue, Selective Context (Li et al., 2023) proposed to measure the importance of tokens in a piece of text based on the semantic confidence of LLMs:

$$I_1(x_i) = -\log P(x_i \mid \boldsymbol{x}_{\leq i}; \boldsymbol{\theta}_{\mathcal{M}_L}), \quad (1)$$

where  $x = \{x_i\}_{i=1}^n$  is the given text,  $x_i$  denotes a token, and  $\mathcal{M}_L$  denotes the LLM used to compute the confidence of each token. Intuitively, such a measurement could be seamlessly applied to CoT

<span id="page-2-0"></span>Problem: Marcus is half of Leo's age and five years younger than Deanna. Deanna is 26. How old is Leo?

Chain-of-Thought: Let'sbreakitdownstepbystep:1. Deannais26yearsold.2.Marcusisfiveyearsyoungerthan Deanna,soMarcusis26-5=21yearsold.3.Marcusishalf ofLeo's age ,soLeo's age istwiceMarcus's age .4.Since Marcusis21,Leo'sageis2x21=42.(Selective Context)

Chain-of-Thought: Let'sbreakitdownstepbystep:1. Deannais26yearsold.2.Marcusisfiveyearsyoungerthan Deanna,soMarcusis26-5=21yearsold.3.Marcusishalf ofLeo's age ,soLeo's age istwiceMarcus's age .4.Since Marcusis21,Leo'sageis2x21=42.(LLMLingua-2)

Final Answer: 42.

Figure 2: Visualization of token importance within a CoT sequence, with darker colors indicating higher values. This figure compares two token importance measurements: Selective Context and LLMLingua-2.

tokens generated by LLMs. We show an example of this measurement in Figure [2.](#page-2-0)

Despite its simplicity, LLMLingua-2 [\(Pan et al.,](#page-10-10) [2024\)](#page-10-10) argued that there exist two major limitations in the aforementioned measurement that hinder the compression performance. Firstly, as shown in Figure [2,](#page-2-0) the intrinsic nature of LLM perplexity leads to lower importance measures (i.e., higher confidence) for tokens at the end of the sentence. Such position dependency impacts the factual importance measurement of each token. Furthermore, the unidirectional attention mechanism in causal LMs may fail to capture all essential information needed for token importance within the text.

To tackle these limitations, LLMLingua-2 introduced utilizing a bidirectional BERT-like LM [\(De](#page-8-3)[vlin et al.,](#page-8-3) [2019\)](#page-8-3) for token importance measurement. It utilizes GPT-4 [\(OpenAI,](#page-10-11) [2023\)](#page-10-11) to label each token as "*important*" or not and trains the bidirectional LM with a token classification objective. The token importance is measured by the predicted probability of each token:

<span id="page-2-2"></span>
$$I_2(x_i) = P(x_i \mid \boldsymbol{x}_{\leq n}; \boldsymbol{\theta}_{\mathcal{M}_B}), \qquad (2)$$

where M<sup>B</sup> denotes the bidirectional LM.

This study applies LLMLingua-2 as the importance measurement to CoT tokens. Similar to plain text, we observe that the semantic importance of tokens within CoT outputs varies, as shown in Figure [2.](#page-2-0) For instance, mathematical equations tend to have a greater contribution to the final answer, consistent with recent research [\(Ma et al.,](#page-9-8) [2024\)](#page-9-8). In contrast, semantic connectors such as "*so*" and

<span id="page-2-1"></span>Revovering the Compressed Chain-of-Thought

Compressed CoT: break down Deanna 26 Marcus five younger 26 - 5 21 Marcus half Leo's age twice Marcus Marcus 21, Leo's age 2 x 21 = 42.

Recovered Compressed CoT: Let's break it down step by step. Deanna is 26 years old. Marcus is five years younger than Deanna: M = D - 5. Marcus's age: M = 26 - 5 = 21. Marcus is half of Leo's age: M = L / 2. Leo is twice Marcus's age: L = 2M. Leo's age: L = 2 x 21 = 42.

Figure 3: Recovering the compressed CoT for GSM8K math word problem using LLaMA-3.1-8B-Instruct.

"*since*" generally contribute less. These findings highlight the token redundancy in CoT outputs of LLMs and the substantial potential to enhance CoT efficiency by trimming this redundancy.

## 2.2 CoT Recovery

We further explore the following research question: *"Are LLMs capable of restoring the CoT process from compressed outputs?"* The answer is yes. As shown in Figure [3](#page-2-1) and detailed in Appendix [A,](#page-11-0) examples restored from compressed CoTs using LLaMA-3.1-8B-Instruct demonstrate that LLMs could effectively comprehend the semantic information encoded in the compressed CoT and restore the CoT process. This capability ensures that the interpretability of compressed CoTs is maintained. Additionally, when required by users, the complete CoT process can be recovered and presented.

In summary, the empirical analysis above underscores the potential of trimming redundant tokens to enhance CoT efficiency, as well as the ability of LLMs to restore CoT from compressed outputs. However, enabling LLMs to autonomously skip redundant CoT tokens and identify shortcuts between critical reasoning tokens presents a non-trivial challenge. To the best of our knowledge, this work is the *first* to explore CoT compression through *token skipping*. In the following sections, we present our proposed methodology in detail.

