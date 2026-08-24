# XGrammar-2: Dynamic and Efficient Structured Generation Engine for Agentic LLMs

Linzhang Li<sup>∗</sup> blemiade\_qinchuan@sjtu.edu.cn Shanghai Jiao Tong University China

Ziyi Xu xzy2022@sjtu.edu.cn Shanghai Jiao Tong University China

Yixin Dong∗† yixind@andrew.cmu.edu Carnegie Mellon University USA

Alexander Jiang akj2@andrew.cmu.edu Carnegie Mellon University USA

Guanjie Wang irfnfnkemed@sjtu.edu.cn Shanghai Jiao Tong University China

Tianqi Chen† tqchen@cmu.edu Carnegie Mellon University, NVIDIA USA

## Abstract

Modern LLM agents increasingly rely on dynamic structured generation, such as tool calling and response protocols. Unlike traditional structured generation with static structures, these workloads vary both across requests and within a request, posing new challenges to existing engines. We present XGrammar-2, a structured generation engine for dynamic agentic workloads. Our design is based on two key ideas: first-class support for tag-triggered structure switching, and fine-grained reuse across requests with different output structures. Concretely, XGrammar-2 introduces TagDispatch for dynamic structural dispatching and Cross-Grammar Cache for substructure-level cache reuse across grammars. It further improves efficiency with an Earley-based adaptive token mask cache, just-intime compilation, and repetition state compression. Experiments show that XGrammar-2 achieves over 6× faster compilation than prior structured generation engines, and incurs near-zero end-toend overhead in modern LLM serving systems.

## CCS Concepts

• Computing methodologies → Intelligent agents.

## Keywords

Agents, Structured Generation, Large Language Models

#### ACM Reference Format:

Linzhang Li, Yixin Dong, Guanjie Wang, Ziyi Xu, Alexander Jiang, and Tianqi Chen. 2026. XGrammar-2: Dynamic and Efficient Structured Generation Engine for Agentic LLMs. In ACM Conference on AI and Agentic Systems (ACM CAIS '26), May 26–29, 2026, San Jose, CA, USA. ACM, New York, NY, USA, [14](#page-13-0) pages.<https://doi.org/10.1145/3786335.3813124>

## 1 Introduction

Modern LLM agents demonstrate strong capabilities and increasingly rely on complex tool calling and code generation [\[26\]](#page-9-0). These

<sup>†</sup>Corresponding authors.

![](_page_0_Picture_20.jpeg)

[This work is licensed under a Creative Commons Attribution 4.0 International License.](https://creativecommons.org/licenses/by/4.0) ACM CAIS '26, San Jose, CA, USA

© 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2415-2/26/05 <https://doi.org/10.1145/3786335.3813124>

agentic applications impose strong requirements on structured generation, especially for small [\[27\]](#page-9-1) or compressed models. Constrained decoding [\[8,](#page-9-2) [15\]](#page-9-3) is widely adopted to guarantee structural validity by masking invalid tokens at each generation step, enabling reliable downstream applications with minimal overhead.

> **[图片提取文字 (无描述)]:**
> Generation Structure Output of LLM <function=get weather> Tool Calling: {"citv": ".\*"}</function> OK, I need to call a function. Llama Tool Calling <function=get\_weather> {"city": (defined by request's "Shanghai"} <function=search> tool set) {"url": ".\*"}</function> <|channel|>analysis <|channel|>analysis </message/>OK, I need to think <lmessagel>".\*" Response Protocol: about it. OpenAl Harmony <|channel|>commentary <|channel|>commentary Response Format to=fucntions.get weather to=functions.get weather <|message|>{"city": "Shanghai"} <|message|>{"city": ".\*"} Dispatched Structure Free-form Text Taa
![](_page_0_Picture_24.jpeg)

Figure 1: Some examples of tool calling and response protocols.

However, existing constrained decoding methods [\[9,](#page-9-4) [12,](#page-9-5) [33\]](#page-10-0) largely assume all structures are static and known in advance. Nowadays, a key characteristic of agentic LLM applications is the extensive use of tool calling to handle complex tasks. Each LLM request may contain dozens or even hundreds of possible tools, which greatly violates the structure assumption: the output structure becomes highly dynamic, both across requests and within a single request. This structural dynamism poses significant efficiency and expressiveness challenges to existing constrained decoding systems. We classify the challenge of structural dynamism into two categories:

Inter-request dynamism. In agent serving scenarios, each request may expose a different set of tools and schemas, often with per-tool access control [\[20,](#page-9-6) [25\]](#page-9-7). As a result, the space of possible output grammars becomes combinatorially large, and each grammar can itself be complex. Prior approaches typically preprocess the entire grammar and cache it at the request level to reuse identical structures. Under dynamic tool sets, such caching becomes ineffective, forcing expensive per-request preprocessing and significantly increasing time-to-first-token (TTFT).

<sup>∗</sup>Both authors contributed equally to this research.

> **[图片提取文字 (无描述)]:**
> Dispatching FSM Intra-request Dynamism Other Other Optimizations Characters Dispatching Root TagDispatch( (§3.6)(Tag 1, Grammar 1), TagDispatch (§3.2) (Tag 2, Grammar 2), Tag 1 Tag 2 Tag 3 Grammar 2 Grammar 1 Grammar 3 Inter-request Dynamism Cross-grammar Cache (§3.3) Prior LLM Output Token Mask Cache Pool Current Parser State Parse Retrieve Token Retrieve by • "+" [0-9] "+" • [0-9] **Adaptive Cache** Sub-Structure Mask Cache The pool is initially "+" • [0-9] on Earley empty and Compared by Parser (§3.4) managed by JIT • "+" [a-z] "+" • [a-z] Hash Value A rule in the grammar. compilation The dot means the system (§3.5). current state postion. Token Mask Cache Token Mask Accepted Rejected Uncertain Tokens Earley Parser State Token Mask Cache Tokens (Need to be Checked at Runtime) Tokens
![](_page_1_Figure_2.jpeg)

Figure 2: Overview of our approach. We design a new dynamic dispatching semantics, TagDispatch ([§3.2\)](#page-2-0), to efficiently support intra-request dynamism. To leverage the sub-structures across different grammars, we designed a cross-grammar caching algorithm ([§3.3\)](#page-2-1) based on the Earley parser ([§3.4\)](#page-3-0) to handle inter-request dynamism. We also design a JIT compilation method ([§3.5\)](#page-4-0) to optimize the efficiency for the inter-request dynamism. We also introduce a repetition state compression algorithm ([§3.6\)](#page-4-1) to handle repetition structures.

Intra-request dynamism. Within a single request, the model needs to follow a response protocol such as OpenAI Harmony [\[16\]](#page-9-8), and choose from many candidate tools. This requires the structural constraint to switch depending on the previous LLM output. For example, generating a tool name determines the JSON schema of the subsequent arguments [\[19,](#page-9-9) [28\]](#page-9-10), while a channel tag token constrains the following content to a specific channel, such as reasoning or output. Such dispatching is difficult to express efficiently with the Backus-Naur Form (BNF)-like grammars used by existing constrained decoding methods, and the large number of tools further challenges efficient mask generation.

To address these challenges, we propose XGrammar-2, a structured generation engine for dynamic agentic workloads. Our design is based on two key ideas: first-class support for tag-triggered structure switching in agent outputs, and fine-grained reuse across requests with different output structures. For the former, we introduce TagDispatch, a first-class grammar construct for expressing tag-triggered structural dispatching within a request. For the latter, we design a cross-grammar cache that reuses shared substructures across different grammar combinations. To make this design efficient in practice, we further develop an Earley-based adaptive

token mask cache, together with just-in-time compilation and repetition compression, to reduce compilation overhead and improve end-to-end efficiency.

We implement XGrammar-2 as a structured generation engine compatible with modern LLM inference systems. XGrammar-2 supports tool-calling formats across major models and enforces strict compliance with the OpenAI Harmony Response Format [\[16\]](#page-9-8). Experimental results show that XGrammar-2 achieves over 6× toolcalling compilation speed improvement compared to prior state-ofthe-art methods, while introducing near-zero latency overhead. We have incorporated XGrammar-2 into open-source serving frameworks such as SGLang [\[37\]](#page-10-1) and vLLM [\[17\]](#page-9-11), improving output reliability in agentic tasks. XGrammar-2 is open-source and has been adopted in both industry systems and open-source inference engines.

## 2 Background

