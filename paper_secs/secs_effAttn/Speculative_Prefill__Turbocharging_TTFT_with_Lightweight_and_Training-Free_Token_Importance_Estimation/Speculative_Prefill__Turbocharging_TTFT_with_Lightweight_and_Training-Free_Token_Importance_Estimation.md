## Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation

## Jingyu Liu <sup>1</sup> Beidi Chen <sup>2</sup> Ce Zhang <sup>1</sup>

## Abstract

Improving time-to-first-token (TTFT) is an essentially important objective in modern large language model (LLM) inference engines. Optimizing TTFT directly results in higher maximal QPS and meets the requirements of many critical applications. However, boosting TTFT is notoriously challenging since it is computebounded and the performance bottleneck shifts from the self-attention to the MLP part. We present SPECPREFILL[1](#page-0-0) , a training free framework that accelerates the inference TTFT for both long and medium context queries based on the following insight: LLMs are generalized enough to preserve the quality given only a *carefully chosen* subset of prompt tokens. At its core, SPECPREFILL leverages a lightweight model to speculate locally important tokens based on the context. These tokens, along with the necessary positional information, are then sent to the main model for processing. We evaluate SPECPRE-FILL with a diverse set of tasks, followed by a comprehensive benchmarking of performance improvement both in a real end-to-end setting and ablation studies. SPECPREFILL manages to serve Llama-3.1-405B-Instruct-FP8 with up to 7× maximal end-to-end QPS on real downstream tasks and 7.66× TTFT improvement.

## 1. Introduction

Large Language Models (LLMs) represent a transformative innovation in artificial intelligence, enabling machines to

*Proceedings of the* 42 nd *International Conference on Machine Learning*, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

<span id="page-0-0"></span><sup>1</sup>The code with experiment reproduction is available at [https:](https://github.com/anonymous/speculative_prefill) [//github.com/anonymous/speculative\\_prefill](https://github.com/anonymous/speculative_prefill).

understand and generate human-like languages [\(Bubeck](#page-9-0) [et al.,](#page-9-0) [2023;](#page-9-0) [Wei et al.,](#page-15-0) [2022;](#page-15-0) [Feng et al.,](#page-10-0) [2024\)](#page-10-0). Many SOTA models have been developed, such as GPT-4 [\(OpenAI](#page-12-0) [et al.,](#page-12-0) [2024\)](#page-12-0), the Llama family [\(Grattafiori et al.,](#page-10-1) [2024\)](#page-10-1), DeepSeek R1 [\(DeepSeek-AI et al.,](#page-9-1) [2025\)](#page-9-1), Mistral [\(Jiang](#page-11-0) [et al.,](#page-11-0) [2023a\)](#page-11-0), Gemini [\(Team et al.,](#page-13-0) [2024\)](#page-13-0), and Qwen2 [\(Yang](#page-16-0) [et al.,](#page-16-0) [2024\)](#page-16-0), to meet the increasing expectations of users. In order to broaden their real-world applications, one essential requirement is to build an efficient serving engine that can satisfy various requirements [\(Miao et al.,](#page-12-1) [2023;](#page-12-1) [Kwon et al.,](#page-11-1) [2023;](#page-11-1) [Zheng et al.,](#page-16-1) [2024;](#page-16-1) [Shoeybi et al.,](#page-13-1) [2020\)](#page-13-1).

There are several fundamental reasons why TTFT stands so pivotal: 1) many applications require a fast response time that directly influences how users perceive the responsiveness of the system and 2) more importantly, TTFT determines the scaling of maximal QPS an inference engine can support as shown in Figure [1.](#page-1-0) However, optimizing TTFT is an arduous task mostly because the prefill stage is largely compute-bounded and the computational bottleneck can change depending on the prompt length and batch size. For example, many works focus on improving the self-attention speed [\(Dao et al.,](#page-9-2) [2022;](#page-9-2) [Jiang et al.,](#page-11-2) [2024a\)](#page-11-2), but in reality, there is still a huge traffic of large-batch short to medium context queries where it is the MLP part that clogs the whole system. Despite achieving impressive results, prior works that target the prefill phase either require a post-training adaptation [\(Qiao et al.,](#page-12-2) [2024;](#page-12-2) [Horton et al.,](#page-11-3) [2024\)](#page-11-3) or scale less efficiently [\(Shi et al.,](#page-13-2) [2024\)](#page-13-2).

Inspired by those work, we found a key insight that LLMs can retain most of its performance when given only *a carefully chosen subset* of tokens from the prompt, and the model is able to adapt to that in a zero-shot manner. SPECPREFILL optimizes the TTFT by leveraging a secondary lightweight model to speculate locally important tokens. Only these tokens are sent later to the base model. It reduces the total FLOPS by a factor proportional to the percentage of token drop. SPECPREFILL does not require any fine-tuning and is ready to be deployed and scaled to larger models. We summarize our key contributions:

• We present a conceptually simple, effective, and noval framework called SPECPREFILL that significantly

<sup>1</sup>Department of Computer Science, The University of Chicago, Chicago, IL, USA <sup>2</sup>Department of Electrical and Computer Engineering, Carnegie Mellon University, Pittsburgh, PA, USA. Correspondence to: Jingyu Liu <jingyu6@uchicago.edu>, Ce Zhang <cez@uchicago.edu>.

![](_page_1_Figure_1.jpeg)

<span id="page-1-0"></span>Figure 1. Speculative Prefill QPS Improvement: In an end-to-end server-client setting with real world datasets, we benchmark the average query latency under a given fixed timeout when sending queries at a constant QPS. SpecPrefill significantly improves the maximum QPS supported by the vLLM server as well as the latency compared to not using it. When we reach low keep rate, we can even serve the 405B model with SpecPrefill to run more efficiently than the 70B model. As the base model size increases and keep rate drops, we can get  $7 \times$  end-to-end QPS boost while only occurring < 5% accuracy.

improve the prefill phase, hence the maximal QPS, of LLM inference without any fine-tuning or adaptation.

- We conducted comprehensive evaluations both on real and synthetic datasets to demonstrate its effectiveness and limitations, giving a full picture of the expected benefits when deployed to productions.
- We implemented our method on industry standard serving engines, and benchmark its performance in both an end-to-end fashion and ablation experiments. The end result is a system that can serve Llama-3.1-405B-Instruct-FP8 with up to 7× maximal QPS under the same system specification and 7.66× reduced TTFT while maintaining decent accuracy.
- Our method can be easily combined with techniques from quantization, KV eviction, and speculative decoding, making it an ideal add-on to existing engines.

### 2. Background

#### 2.1. Inference Bottlenecks

Improving LLM inference efficiency has been extensively studied in prior work (Miao et al., 2023; Yuan et al., 2024). We review works that focus on different aspects when dealing with real serving systems where the bottlenecks are quite different under various serving requirements (e.g. long context domains, latency sensitive applications, etc) (Kwon et al., 2023; Zheng et al., 2024).

LLM inference can be roughly divided into two major procedures, namely the prefill phase where the model computes

the KV necessary for producing the output based on the query and the decoding phase where the model predicts new token auto-regressively.

#### 2.2. Decoding Acceleration

The decoding phase is mostly memory-bounded, and therefore reducing the amount of data to move around will effectively help improve the latency. As a result, explicitly manipulating the KV cache has been extremely successful with many strategies: H2O (Zhang et al., 2023) and StreamingLLM (Xiao et al., 2024b) idenfitied key insights to KV dynamics which is used to evict less essential KV caches during decoding. CacheGen (Liu et al., 2024), Q-Hitter (Zhang et al., 2024b), and ShadowKV (Sun et al., 2024) apply efficient techniques to compress/quantize, store, and transmit KV caches to reduce memory overhead. Speculative decoding relies on the insight that hides the memory latency by concurrently verifying several speculated tokens from either a draft model or itself (Leviathan et al., 2023; Zhang et al., 2024a; Xia et al., 2024).

Despite being crucially important, decoding speed is not the only factor that influences the overall inference pipeline and we will review why *sometimes the prefill time optimization* is even more essential in many cases.

#### 2.3. Prefill Acceleration

The time-to-first-token (TTFT) is crucially important both from a user experience but also the system serving perspective (Sec 4.7). In many critical applications, the input token length can often eclipse that of the generation tokens (e.g. 10:1 ratio) in real traffic (Qiao et al., 2024). Unlike

the decoding phase, however, the prefill phase is usually compute-bounded and the cost for the MLP calculation and the communication of tensor parallelism quickly becomes a bottleneck.

Many prior works have explored ways to make selfattention faster: Flash-attention series [\(Dao et al.,](#page-9-2) [2022;](#page-9-2) [Dao,](#page-9-3) [2023\)](#page-9-3) compute the exact attention using carefully designed hardware-aware algorithms. Special (both static and dynamic) attention masks are designed for sparse calculation such as LongFormer [\(Beltagy et al.,](#page-9-4) [2020\)](#page-9-4), MInference [\(Jiang et al.,](#page-11-2) [2024a\)](#page-11-2), FlexPrefill [\(Lai et al.,](#page-11-5) [2025\)](#page-11-5), Hip Attention [\(Lee et al.,](#page-11-6) [2025\)](#page-11-6), Sample Attention [\(Zhu](#page-16-7) [et al.,](#page-16-7) [2024\)](#page-16-7), and Duo Attention [\(Xiao et al.,](#page-16-8) [2024a\)](#page-16-8). However, none of these directly make the MLP part faster like SPECPREFILL. SPECPREFILL achieves consistent efficiency improvements in various regimes because it skips parts of the attention + MLP calculation and the all-reduce overhead, which proves to be effective especially when the ratio of batch size sequence length is large [\(Xiong et al.,](#page-16-9) [2023\)](#page-16-9).

Orthogonal to techniques such as prompt compression/rewrite [\(Jiang et al.,](#page-11-7) [2023b;](#page-11-7) [2024b;](#page-11-8) [Li et al.,](#page-11-9) [2023\)](#page-11-9), layer dropping [\(Elhoushi et al.,](#page-9-5) [2024\)](#page-9-5), and weight quantization methods [\(Lin et al.,](#page-12-4) [2024\)](#page-12-4), we explore selecting *important* prompt tokens to skip the full forward computation. GemFilter [\(Shi et al.,](#page-13-2) [2024\)](#page-13-2) uses an extra pass to get a model's own middle layer attention information that decides on what tokens to keep for the real forward. Contrast to this, we apply a separate and cheaper model to speculate locally important tokens via token transferability, which can scale more efficiently than GemFilter. Concurrent to ours, SwiftKV [\(Qiao et al.,](#page-12-2) [2024\)](#page-12-2) learns to skip later layers by reusing the past layers' KV, which achieves up to 50% TTFT reduction (SPECPREFILL can reach up to 87% TTFT reduction). Unlike our zero-shot requirement, they require extra light-weight fine-tuning due to modified model behavior. It is worth noting that our method approaches the problem in a different way, which makes them complimentary to each other. Finally, akin to our motivation, KV Prediction [\(Horton et al.,](#page-11-3) [2024\)](#page-11-3) proposes to adapt a cheaper system (i.e. a learned auxiliary network and a KV predictor) to predict the KV cache of the base model, thus bypassing the original KV computation. We show that SPECPREFILL can accomplish better TTFT reduction than theirs, without introducing extra overhead when coupled with speculative decoding [\(Leviathan et al.,](#page-11-4) [2023\)](#page-11-4) while maintaining competitive quality.

## 3. Speculative Prefill

In this section, we present SPECPREFILL by first describing its high-level algorithm, followed by several design choices that mitigate various biases, and a detailed implementation account. Finally, we touch a bit on how to integrate

SPECPREFILL to speculative decoding, forming a full smallmodel-assisted inference paradigm.

#### 3.1. Overall Architecture

SPECPREFILL follows a conceptually simple architecture where a usually less expensive model is chosen as the speculator model that predicts contextually important tokens given a prompt. The speculated tokens, alone with the original position information, are then fed to the main model for processing. In the following section, we will discuss two central design choices in more details, namely the token estimation algorithm and the selection strategy. Note that SPECPREFILL can be seamlessly integrated with speculative decoding in which the small model can work both in the prefill stage for token selection and the decoding stage for drafting proposals, making our approach almost free to integrate and deploy.

#### <span id="page-2-1"></span>3.2. Token Importance Speculation

The goal here is to select which tokens are contextually important for a given query and send those along with necessary positional information for the main model. The procedure starts with calculating the attention scores from the speculator, which uses the last token's attention score w.r.t. the context as the surrogate for measuring token importance:

$$a_{ij} := \operatorname{Softmax}(Q_{M+j}K^T))_i, \forall 0 \le i < M, 0 \le j < N$$

where M is the context length, N is the number of lookahead steps, and aij is the attention score for the ith token in the prompt w.r.t. the jth decoded token, assuming we're looking at a particular layer.

We build on top of this by aggregating the scores over the whole speculator model (Section [3.2.2\)](#page-3-0) with potential lookahead (Section [3.2.1\)](#page-2-0) and select tokens based on chunks (Section [3.2.3\)](#page-3-1). The subset of chosen tokens with their original positional information (Section [3.2.4\)](#page-3-2) will then be used for the main model's inference.

### <span id="page-2-0"></span>3.2.1. MITIGATE POSITION BIAS VIA LOOK-AHEAD

Prior works have shown that there are many biases for attention scores, such as the sink phenomenon [\(Xiao et al.,](#page-16-4) [2024b\)](#page-16-4) (the first couple of tokens tend to have higher weights) and the proximity bias (tokens closer to the output tend to have higher weights [\(Lv et al.,](#page-12-5) [2024\)](#page-12-5)). To mitigate these issues, instead of relying on the attention score of the last token alone, we further decode the speculator by N steps and obtain the attention information from the new N tokens [\(Wan](#page-15-2) [et al.,](#page-15-2) [2024\)](#page-15-2). N here serves as a trade-off between bias and budget, which can substantially increase the performance for shorter context queries.

## <span id="page-3-0"></span>3.2.2. AGGREGATED ATTENTION SCORE AS TOKEN IMPORTANCE

Given the full attention scores of the speculator, we decide to use a max-mean aggregation strategy to map to scalar token importance. Formally, given an attention score tensor of shape [N, L, S, H] where N is the number of look-ahead tokens, L is the number of layers, S is the sequence length, and H is the number of heads, we take the maximum over H and L dimension to make salient tokens stand out, and average over N to account for fair token contribution.

## <span id="page-3-1"></span>3.2.3. DENOISE ATTENTION SCORES BY CHUNK SELECTION AND POOLING

It has also been observed in concurrent works (Lv et al., 2024) that tokens that are positioned nearby share similarity in importance. We take this insights to select tokens by chunks in order to reduce the variance of our token importance estimation. Specifically, we chunk the context contiguously and average the token score within each block, and then we select the Top-K blocks. In order to eliminate the artifacts of chunkation, we apply a 1D average pooling before this to smooth the cross block scores.

#### <span id="page-3-2"></span>3.2.4. RESTORATION OF POSITION IDS

Finally, when we select the subset of tokens based on our compute budget and query compressibility, we also need to restore the position information which are also sent to the main model. Basically, instead of using a contiguous position ids as before, we send a potentially non-continuously increasing position ids which are obtained from tokens' positions in the original context. In addition to that, we also need to explicitly set the decoding token position to the context length in case we dropped tokens before the first decoding token. An example is shown below with ten prompt tokens and three decoding tokens (bold):

```
Original Pos Ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Speculated Pos Ids: [0, 1, 3, 6, 7]
Decoding Pos Ids: [0, 1, 3, 6, 7, 10, 11, 12, ...]
```

where the **bold indices** are the decoding positions which are offset based on the original position information. We found this design choice to be crucially essential, especially for position-sensitive tasks such as synthetic tasks involving retrieval and counting.

#### 3.3. Implementation Details

We describe both the high-level procedure and the implementation details of SPECPREFILL in this section. In Algorithm 1, we list the high-level steps of conducting SPECPREFILL. Our implementation is based on creating a monkey patch on top of vLLM (Kwon et al., 2023) which only needs

a few line of code along with a configuration file to enable SPECPREFILL. The KV cache is not necessary if we do not need to look-ahead for our speculator, which can save lots of memory allocation. However, we do need to explicitly store the queries of the decoded tokens (including the last token of the input query) which we later retrieve to compute the attention score. Note that a specific mapping (e.g. slot mapping in vLLM) might be kept track of to retrieve the right data. For batched look-ahead, we only consider tokens that are valid by checking afterwards whether they are equal to the EOS tokens. Finally, we want to mention that despite being being a sequential implementation, we can actually split the process of speculation into a separate procedure and decouple from the inference of the main model by adding a new layer of scheduling, which we leave as a future work.

#### 3.4. Relation to Speculative Decoding

Speculative decoding has been proven to be extremely successful at accelerating the decoding TPS. Specprefill can be seamlessly combined with speculative decoding by sharing the same draft model. Since speculative decoding itself requires a full forward pass of the context, Specprefill will provide the necessary KV information required for subsequent decoding speculation, hence amortizing the overhead. This will open-up a huge space of possibilities, and lead to the first paradigm of an inference system that is fully aided by smaller speculators.

#### **Algorithm 1** Speculative Prefill

```
Require: Base model M, speculator S, look-ahead steps
    N, batch of mixed requests B, base model QKV cache
    C_b, speculator KV cache C_s
 1: B_p, B_d \leftarrow split\_prefill\_decode\_requests(B)
 2: {Section 3.2.1}
 3: for i = 1 to N do
       B'_p \leftarrow model\_forward(S, B_p, C_s, store\_q = True)
       B_p \leftarrow update\_requests(B_p, B_p')
       B_p \leftarrow check\_for\_eos(B_p)
 6:
 8: if is_tensor_paralleled () then
       tp\_gather\_qk(C_s)
10: end if
11: Q, K \leftarrow retrieve\_qk(B_p, C_s)
12: A \leftarrow compute\_attention\_score(Q, K)
13: {Section 3.2.2}
14: A \leftarrow aggregate\_attention\_score(A)
15: {Section 3.2.3}
16: T \leftarrow chunk\_select\_from\_smoothed\_attention (A)
17: {Section 3.2.4}
18: P \leftarrow restore\_pos\_ids(T, B_n)
```

19:  $B \leftarrow merge\_requests(T, P, B_p, B_d)$ 

<span id="page-3-3"></span>20: **Return**  $model\_forward$   $(M, B, C_b)$ 

## 4. Experiments

In this section, we start with our experiment setup for reproducibility, followed by categorizing prompt compressibility of different queries. We evaluate SPECPREFILL on downstream long context, synthetic context probing, and standard short tasks. Finally, we conclude with a comprehensive efficiency measurement of our system under the real end-to-end setting.

#### 4.1. Setup

We implement SPECPREFILL in vLLM that supports tensor parallelism with the same degree as the main model[2](#page-4-0) . Due to its token dropping nature, we focus on evaluating *generative* tasks in this section and include a comprehensive range of benchmarks to fully present its applicability and potential pitfalls. We run all of experiments using a tensor parallelism of 8 for both the speculator and the base model across either 8 NVIDIA H100s or H200s (full system specification in Appendix [D](#page-17-0) and guidance on reproducing results in Appendix [C\)](#page-17-1). We choose LLAMA-3.1-8B-INSTRUCT [\(Grattafiori et al.,](#page-10-1) [2024\)](#page-10-1) with BF16 precision as our speculator for a balance of efficiency and attention transferability and couple it with either Llama-3.1-70B-Instruct (BF16) or Llama-3.1-405B-Instruct-FP8[3](#page-4-1) (fully quantized FP8) as the base model. In terms of token keep rate, we use a fixed percentage (i.e. the ratio of chunks when we do chunk selection) for a given task. In practice, we might devise more adaptive strategy for how many tokens to keep based on the query compressibility discussed next, or delegate the decision to users based on their needs. We leave all these possibilities for prospective applications.

#### 4.2. Query Context Compressibility

We empirically found three types of queries during our evaluations based on the quality difference before and after applying SPECPREFILL:

- 1. *Information-dense queries*: These queries usually are short and information dense, which naturally makes token dropping less effective because there is no redundancy in the prompt.
- 2. *Compressible queries*: These queries are those that do not get degradation after removing a significant amount of tokens, often seen in long context tasks.
- 3. *Noisy queries*: These queries, perhaps surprisingly, get better results after dropping some "noisy" tokens.

We hypothesize the reason behind the improvement might be that SPECPREFILL helps remove noisy and distracting tokens in the prompt, hence projecting the prompt to the space where the main model performs better.

We will see examples in each categories in the following evaluations. It is worth noting that we used a fixed keep percentage for our evaluation and it can be tremendously helpful to automatically decide on the percentage based on the query, pushing the limit of SPECPREFILL, which we leave as a future work.

### 4.3. Baselines and SPECPREFILL Variants

We aim to showcase both the quality and efficiency of SPECPREFILL under a comprehensive set of applications. To do so, we compare *three* variants of SPECPREFILL against *four* baselines:

- *Baselines*: We compare SPECPREFILL against four different baselines: 1) Base Llama instruct model. 2) Sentence RAG: SPECPREFILL can be framed as a special case of retrieval-augmented (RAG) LLM with the granularity of tokens or blocks and the relevance metric controlled by the speculator's internal knowledge [\(Li et al.,](#page-11-10) [2024;](#page-11-10) [Gao et al.,](#page-10-2) [2024b;](#page-10-2) [Lewis](#page-11-11) [et al.,](#page-11-11) [2021\)](#page-11-11). Therefore, we implemented two simple sentence-level RAG baselines and report the better one as RAG-LLAMA. 3) LLMLingua [\(Jiang et al.,](#page-11-7) [2023b\)](#page-11-7): SPECPREFILL can also be seen as a context compression technique, and hence we test SPECPRE-FILL against a text-level compression method. 4) MInference [\(Jiang et al.,](#page-11-2) [2024a\)](#page-11-2): To understand the benefits of skipping the MLP part, we include a sparse attention optimization approach for completeness.
- SPECPREFILL: SPECPREFILL with raw attention scores and ignoring the techniques we discussed in Section [3.2.1](#page-2-0) and [3.2.3.](#page-3-1)
- SPECPREFILL *Full*: SPECPREFILL with all techniques but no look-ahead.
- SPECPREFILL *Full LAH*: SPECPREFILL with all techniques with 8-step look-ahead[4](#page-4-2) .

#### 4.4. Real Long Context Tasks: LongBench

We start with long context tasks using LongBench [\(Bai et al.,](#page-9-6) [2024\)](#page-9-6), which consists of six different categories focusing on various aspects of long context modeling abilities.

<span id="page-4-0"></span><sup>2</sup>We expose the API so that it only takes a few line of code to apply SPECPREFILL before initializing vLLM engines.

<span id="page-4-1"></span><sup>3</sup>[https://huggingface.co/neuralmagic/](https://huggingface.co/neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8) [Meta-Llama-3.1-405B-Instruct-FP8](https://huggingface.co/neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8).

<span id="page-4-2"></span><sup>4</sup>We empirically found that going beyond 16 look-ahead gives minimal performance gain.

![](_page_5_Figure_1.jpeg)

<span id="page-5-0"></span>Figure 2. LongBench Main Result on Llama 405B: In this figure, we showcase the effectiveness of SPECPREFILL on Long-Bench, which consists of six categories of long context downstream tasks. In each plot, the dash lines are the results of baseline Llama-3.1-405B-Instruct-FP8 for each subtask and we benchmark SPECPREFILL with increasing token keep rates. We observe different behaviors such as quality preservation, degradation, and improvement based on the task type.

In Figure 2, we report the main results on LongBench for Llama-3.1-405B-Instruct-FP8, whose length information is visualized in Appendix Figure 10. To compliment it, we also include the performance of Llama-3.1-70B-Instruct in Appendix Figure 8. We vary the token keep percentage starting from 10% to 90% and draw the baseline model quality using the dash lines for each subtask. To ablate the effect of our design choices, we compare SPECPREFILL and SPECPREFILL Full LAH with the 70B models in Appendix Figure 8, and SPECPREFILL Full and SPECPREFILL Full LAH with the 405B model in Figure 2.

As we can observe, for categories such as Single-Document QA, Multi-Document QA, Few-Shot Learning, SPECPRE-FILL can preserve most of the quality up to keeping only 10% tokens. For Summarization, we expect to see some degradation in performance as we drop more. Perhaps surprisingly, for the smaller 70B model, we can achieve better quality after we remove some tokens on tasks like Code Completion. As the model size increases, the quality gap

between applying SPECPREFILL or not becomes smaller, which indicates that bigger models adapt better with our speculated subset of tokens.

To ablate the effectiveness of techniques discussed in Section 3.2, we compare them separately in Figure 2 and 8 to avoid crowdedness. In both cases, we can see consistent improvement and the benefits of look-ahead are more consistent in shorter context tasks (more details in Sec 4.6).

Finally, we demonstrate the superiority of SPECPREFILL over three different baselines in terms of preserving the quality of the inference for the 70B model. In Table 1, we group tasks in LongBench into categories and list the results with varying degrees of compression rates. For RAG-LLAMA, we use the question to retrieve the relative information from the context (more detailed descriptions are given in Appendix B). For LLMLingua, we follow their official examples and only compress the context, leaving the question and template intact. With the prior knowledge of separated context and question, these two methods are eclipsed by

<span id="page-6-1"></span>Table 1. LongBench 70B Model Comparison: We compare different methods based on Llama-70B-Inst with varying compression rates on LongBench (grouped by task types). \* denotes models that not only require context-question separation but also have the true compression rates distinctive from the predefined ones. Among all tested methods, SPECPREFILL achieves superior average performance (underlined scores are the best under the comparable rate).

| Model       | Compression Rate    | Single-Doc QA | Multi-Doc QA | Sum   | Few-shot Learning | Code  | Synthetic | Avg   |
|-------------|---------------------|---------------|--------------|-------|-------------------|-------|-----------|-------|
| Baseline    | N/A                 | 50.57         | 53.11        | 25.84 | 66.93             | 52.33 | 72.50     | 53.55 |
|             | 10.38%              | 32.32         | 41.17        | 18.86 | 45.40             | 44.76 | 30.42     | 35.49 |
|             | 27.68%              | 38.43         | 47.41        | 21.42 | 50.53             | 45.80 | 35.50     | 39.85 |
| RAG*        | 45.64%              | 40.53         | 46.64        | 22.45 | 49.52             | 46.00 | 43.15     | 41.38 |
|             | 63.42%              | 41.40         | 47.43        | 23.30 | 52.21             | 46.19 | 47.22     | 42.96 |
|             | 82.22%              | 43.25         | 48.16        | 23.56 | 51.44             | 45.92 | 53.04     | 44.23 |
| LLMLingua*  | ∼10%                | 26.50         | 32.94        | 20.95 | 37.40             | 45.00 | 16.33     | 29.85 |
|             | ∼30%                | 38.83         | 44.02        | 23.37 | 42.23             | 47.27 | 37.00     | 38.79 |
|             | ∼50%                | 43.64         | 50.67        | 24.77 | 50.96             | 49.05 | 60.33     | 46.57 |
|             | ∼70%                | 45.90         | 52.88        | 25.44 | 59.77             | 51.48 | 68.50     | 50.66 |
|             | ∼90%                | 45.94         | 53.91        | 25.87 | 60.46             | 54.06 | 72.00     | 52.04 |
| MInference  | N/A (Section 4.7.2) | 50.46         | 53.23        | 25.83 | 66.36             | 52.48 | 69.00     | 52.89 |
| SPECPREFILL | 10%                 | 47.64         | 52.96        | 21.74 | 64.52             | 63.33 | 66.25     | 52.74 |
|             | 30%                 | 49.47         | 53.39        | 24.41 | 65.83             | 62.62 | 67.83     | 53.92 |
|             | 50%                 | 50.18         | 52.56        | 25.10 | 65.60             | 59.91 | 68.17     | 53.59 |
|             | 70%                 | 50.06         | 52.44        | 25.51 | 65.77             | 58.08 | 68.67     | 53.42 |
|             | 90%                 | 50.26         | 53.25        | 25.65 | 66.35             | 53.47 | 70.67     | 53.27 |

SPECPREFILL by a large margin under the same rate. For MInference, we use the official searched optimal pattern, and SPECPREFILL can reach 99.7% average score with only 10% keep rate and outperform it with larger keep rate. Since the exact token-level "keep rate" of a sparse attention kernel is not defined, we defer to Section [4.7.2](#page-7-1) for a more fair comparison between these two approaches. Overall, SPECPREFILL achieves impressive performance without any fine-tuning or input assumption, further supporting its effectiveness and flexibility.

#### 4.5. Synthetic Context Probing: RULER

In addition to LongBench, we also evaluate SPECPREFILL on a synthetic context probing task to see if SPECPREFILL can preserve effective context lengths. RULER [\(Hsieh et al.,](#page-11-12) [2024\)](#page-11-12) is a suite of synthetically created tasks with controllable lengths, which ranges from retrieval, multi-hop tracking, real QA datasets, and context aggregation tasks. In Table [2,](#page-7-2) we include the results for the 70B model with SPECPREFILL that keeps 10% context. As we can observe, SPECPREFILL preserves the quality despite only using one tenth of the tokens except for aggregation tasks, which we believe to fall into the category of information-dense queries that are not our main target application. Take CWE from aggregation category for example: CWE asks for the common words presented in the prompt, which becomes challenging to answer by token dropping. We hope to explore in the future ways of potentially rewriting the queries instead of directly dropping the tokens to mitigate this type of limitation [\(Jiang et al.,](#page-11-7) [2023b;](#page-11-7) [2024b\)](#page-11-8). Averaging scores without the aggregation task, we can see that in most context lengths,

SPECPREFILL even helps improve the quality[5](#page-6-2) , suggesting 1) that SPECPREFILL provides both efficiency and performance gains at the same time, and 2) the fact that there are lots of potential redundancy and noise in these synthetic tasks.

#### <span id="page-6-0"></span>4.6. Standard Short Tasks

Unlike prior works on prefill token dropping techniques [\(Lv](#page-12-5) [et al.,](#page-12-5) [2024;](#page-12-5) [Shi et al.,](#page-13-2) [2024\)](#page-13-2) that do not include regular short context task evaluation, we present a wide range of standard tasks to show the full spectrum of SPECPREFILL's performance and potential caveats. We select tasks spanning general knowledge (Generative MMLU [\(Hendrycks et al.,](#page-11-13) [2021\)](#page-11-13) and Instruction Following Evaluation [\(Zhou et al.,](#page-16-10) [2023\)](#page-16-10)), math (GSM8K 8 Shots [\(Cobbe et al.,](#page-9-7) [2021\)](#page-9-7)), coding (HumanEval [\(Chen et al.,](#page-9-8) [2021\)](#page-9-8) and MBPP [\(Austin et al.,](#page-9-9) [2021\)](#page-9-9)), and reasoning abilities (Arc Challenge [\(Clark et al.,](#page-9-10) [2018\)](#page-9-10) and GPQA 8 Shots [\(Rein et al.,](#page-12-6) [2023\)](#page-12-6)).

In Apendix Figure [6,](#page-18-0) we showcase the performance of Llama-3.1-70B-Instruct on these tasks. Nonsurprisingly, prompts from standard tasks without few shot examples are very information dense, making SPECPREFILL less effective with low token keep rate. However, for certain tasks (e.g. MBPP and GPQA), we do observe improved performance when dropping certain tokens. On average, SPECPREFILL can maintain and even surpass the baseline when choosing the right token keep rate.

<span id="page-6-2"></span><sup>5</sup> For 4k Multi-hop Tracking, since we only keep around 400 tokens, we might unintentionally ignore some essential information. But to keep experiment setup more consistent, we list the results here for clarity.

<span id="page-7-2"></span>Table 2. RULER Results on Llama 70B: We present results of SPECPREFILL with 10% token keep rate on the effective context probing suite RULER with varying context length. SPECPREFILL can preserve the performance of all except for aggregation tasks, which are *less compressible* due to the problem nature as each word in the prompt is important to reason about word frequency and commonality.

| Model Name                     | Task Length | Retrieval<br>Niah Variants | Multi-hop Tracking<br>Variable Checking | <b>QA</b><br>SQuAD & HotpotQA | Aggregation<br>CWE & FWE | Average<br>w/o Aggregation           |
|--------------------------------|-------------|----------------------------|-----------------------------------------|-------------------------------|--------------------------|--------------------------------------|
|                                | 4k          | 100.0                      | 100.0                                   | 76.9                          | 99.7                     | 92.3                                 |
| Llama-70B-Inst                 | 8K          | 99.9                       | 100.0                                   | 74.7                          | 98.0                     | 91.5                                 |
| Liama-/UB-inst                 | 16K         | 99.8                       | 100.0                                   | 72.0                          | 97.8                     | 90.6                                 |
|                                | 32K         | 99.6                       | 100.0                                   | 69.8                          | 96.9                     | 89.8                                 |
|                                | 64K         | 98.5                       | 99.9                                    | 65.1                          | 65.6                     | 87.9                                 |
|                                | 128K        | 76.5                       | 56.1                                    | 48.2                          | 41.3                     | 60.3                                 |
|                                | 4K          | 99.7                       | 89.6                                    | 75.2                          | 77.9                     | 88.2                                 |
|                                | 8K          | 99.6                       | 100.0                                   | 75.6                          | 79.7                     | 91.7                                 |
| Consporate with 100/ Keep Dete | 16K         | 99.5                       | 99.1                                    | 75.3                          | 78.5                     | 91.3                                 |
| SPECPREFILL with 10% Keep Rate | 32K         | 99.7                       | 100.0                                   | 72.6                          | 70.0                     | 90.8                                 |
|                                | 64K         | 99.5                       | 99.8                                    | 71.9                          | 54.9                     | 90.4                                 |
|                                | 128K        | 85.8                       | 55.6                                    | 55.3                          | 48.3                     | 91.7<br>91.3<br>90.8<br>90.4<br>65.6 |

#### <span id="page-7-0"></span>4.7. Efficiency Benchmarking

SPECPREFILL offers great improvement to TTFT, a speedup almost proportional to the percentage of tokens we drop from the speculator, with almost ignorable overhead as we increase the base model size. In this section, we benchmark both the 70B and 405B models under two settings: 1) understanding the average query latency and QPS dynamics with real downstream datasets, and 2) evaluating TTFT with varying sequence lengths on synthetic data. We used one node consisting of eight NVIDIA H200s for all experiments unless separately specified (full system specification is listed in Table 4 from the Appendix D).

