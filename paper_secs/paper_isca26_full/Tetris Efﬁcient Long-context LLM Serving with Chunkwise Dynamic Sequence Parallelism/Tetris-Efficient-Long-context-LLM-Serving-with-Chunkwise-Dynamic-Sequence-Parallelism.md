# Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism

Cong Li\*1,4, Yuzhe Yang<sup>2</sup> , Xuegui Zheng<sup>2</sup> , Qifan Yang<sup>2</sup> , Yijin Guan<sup>3</sup> , Size Zheng<sup>2</sup> , Li-Wen Chang<sup>2</sup> , Shufan Liu<sup>2</sup> , Xin Liu<sup>2</sup> , Guangyu Sun†1,4 <sup>1</sup>*School of Integrated Circuits, Peking University*, <sup>2</sup>*ByteDance Seed*, <sup>3</sup>*ByteDance*, <sup>4</sup>*Beijing Advanced Innovation Center for Integrated Circuits* {*leesou, gsun*}*@pku.edu.cn*, {*yangyuzhe.gilsaix, zhengxuegui.0, yangqifan, yijin.gyj*}*@bytedance,com*, {*zheng.size, liwen.chang, liushufan.amos, liuxin.ai*}*@bytedance.com*

*Abstract*—With the advancement of large language models (LLMs), their context windows have rapidly expanded. To meet diverse demands from varying-length requests in online services, existing state-of-the-art systems adjust resource allocation by tuning the sequence parallelism (SP) allocation. However, current dynamic SP allocation lacks flexibility to (1) support stage-specific parallelism requirements in LLM inference, (2) mitigate the global latency degradation from excessive SP allocation, and (3) exploit resource fragments arising from SP size variation.

To tackle this problem, we propose Chunkwise Dynamic Sequence Parallelism (CDSP), a fine-grained parallelism strategy that assigns SP sizes across *intra-request* token segments. Based on CDSP, we build Tetris, an LLM serving system that (1) efficiently integrates CDSP into disaggregated cluster architecture to satisfy parallelism heterogeneity, (2) dynamically regulates SP size expansion based on real-time load conditions, and (3) adaptively explores chunking plans to utilize fragmented resources while meeting per-request demands. Compared with state-ofthe-art systems, Tetris achieves up to 4.35× lower time-to-firsttoken (TTFT) under max sustainable loads, reduces median timebetween-tokens (TBT) by up to 40.1%, and increases the max request capacity by up to 45%.

## I. INTRODUCTION

Large Language Models (LLMs) have achieved outstanding performance in a wide range of generative tasks, such as chatbot [14], [30], code completion [12], [25], and reasoning [41], [42]. Such capability drives many cloud companies to deploy online LLM services [2], [4], [14], [30]. As LLMs continue to advance, their context lengths have notebly expanded. For example, OpenAI's GPT-4o [29] supports 128K contexts, Anthropic's Claude-3 [3] supports 200K, and Google's Gemini-2.5 pro [13] supports up to 1M tokens.

With the growth of sequence length, LLM inference requires proportionally more resources. To augment resource provision for long-context requests, sequence parallelism (SP) has been widely applied [5], [11], [16]–[18], [20], [21], [40], [43], [44]. Among these implementations, ring-attention-based SP [21] (also known as context parallelism, CP [11], [40], [44]) has been introduced to LLM serving [43], [44]. Specifically, it scatters long sequences across multiple LLM instances and performs distributed attention computation through peer-topeer (P2P) KV cache transmission. By overlapping cache transmission with attention computation, ring attention demonstrates better scalability than tensor parallelism (TP), especially when populating resources beyond a single node [44].

The expansion of context window also widens request length gaps, thereby amplifying variability in per-request resource demands. To cope with this, existing state-of-the-art long-context LLM serving system, LoongServe [43], proposes elastic sequence parallelism (ESP). ESP dynamically adjusts SP allocation *in the granularity of request batch* to satisfy diverse resource demands. In contrast, non-SP systems have to statically configure resource allocation at startup due to the high overhead of model weight resharding, limiting their ability to respond to highly variable resource demands when serving long-context LLMs.

Although LoongServe has surpassed existing best-performing non-SP systems [1], [19], [23], [47], its *coarse-grained SP allocation* fails to fully optimize online long-context LLM serving's performance: First, ESP enforces a uniform TP size across all instances. However, prefill benefits from smaller TP for better resource allocation flexibility, while decoding prefers larger TP to minimize compute latency. Second, LoongServe assigns requests to fixed batches and exhaustively optimizes per-batch latency. However, since this local-optimal strategy lacks global load awareness, its excessive SP expansion fails to optimize system's overall latency distribution. Third, dynamic SP allocation leads to varying queuing delays across instances. However, since ring attention requires synchronous computation across instances, such an imbalance results in idle slots and degrades overall resource efficiency.

To tackle these problems, we first propose Chunkwise Dynamic Sequence Parallelism (CDSP), a *fine-grained intrarequest SP allocation* strategy. It splits each request's prompt into multiple chunks and assigns each chunk a distinct SP size, enabling efficient utilization of resource fragments while fully optimizing prefill latency. Based on CDSP, we build Tetris, a system for efficient online long-context LLM serving. Tetris efficiently integrates CDSP into prefill-decoding disaggregated cluster by extending attention load-balancing strategy and KV cache transfer management, thereby fully accommodating the parallelism heterogeneity across different stages. For online scheduling, Tetris regulates SP size allocation based

<sup>\*</sup> Work done during Cong Li's internship at Bytedance Seed.

<sup>†</sup> Corresponding author.

on real-time request arrival pressure, thus preventing excessive SP expansion from degrading global latency. In addition, Tetris integrates a load-aware chunk partitioning strategy that dynamically determines the optimal execution plan for each request, maximizing the benefits of CDSP. To summarize, we have made the following contributions:

- We identify existing dynamic SP allocation strategy's rigidity in handling inter-request resource variability under online long-context LLM serving scenarios.
- We propose CDSP for intra-request fine-grained SP allocation and build Tetris's inference engine to fully satisfy
  the heterogeneous demands in long-context LLM serving.
- We propose real-time load-aware SP size allocation and chunk partitioning strategies in Tetris's scheduler to optimize the service's overall latency distribution.

Extensive experiments on workloads collected from a *real-world online long-context LLM service* [6] demonstrate that Tetris achieves up to  $4.35\times$  lower time-to-first-token (TTFT) under state-of-the-art systems' max sustainable loads, reduces median time-between-tokens (TBT) by up to 40.1%, and increases the max request capacity by up to 45%.

#### II. BACKGROUND AND MOTIVATION

## A. Transformer-based LLMs

Mainstream LLMs are built on transformer decoder layers [39], which contain an attention block and a feed-forward network (FFN) block. In the attention block, the inputs are projected to query, key, and value vectors, which interact with each other through self-attention. Then, the outputs of the attention block are processed by multi-layer perceptrons (MLPs) in the FFN block to produce the decoder layer outputs. After passing a stack of transformer layers, the final outputs can be used for downstream generative tasks.

LLM's generation procedure contains two stages: prefill and decoding. In the prefill stage, the LLM processes all tokens of the input prompt in parallel to produce the first output token. Then, moving on to the decoding stage, the LLM takes the token generated previously as input and predicts one new token per iteration, gradually building the full output sequence. Since self-attention requires each token to interact with all previous tokens' key/value vectors, these intermediate states are stored throughout LLM inference to avoid redundant computation, which is known as KV Cache [33].

