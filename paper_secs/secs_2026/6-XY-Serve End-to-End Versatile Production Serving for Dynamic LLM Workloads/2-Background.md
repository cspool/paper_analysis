# 2 Background

In this section we first characterize the dynamics of LLM inference and then, using Ascend NPU as a concrete example, present an architectural abstraction based on a tile-based programming model.

<span id="page-2-0"></span>![](_page_2_Figure_8.jpeg)

Figure 2. Dynamic Input/Output and Speculation Length.

#### 2.1 Dynamics of LLM Inference

The dynamics of LLMs manifests across multiple dimensions. First, input and output lengths vary widely across requests. Fig. 2(a) presents length distributions collected from widely-adopted open-source benchmarks (Conversation [9], Coding [8] API [10], and Agent [11]). These distributions span broad ranges that differ markedly across scenarios, so a single inference batch often contains both extremely long and very short sequences. Compounding this challenge, Prefix Reusing omits portions of inputs, further increasing length dynamism and unpredictability. Second, speculative decoding introduces an additional verify stage, converting token generation from one-token-per-step to multiple tokens per step. Fig. 2(b) shows a runtime trace from SpecServe [36], where the target model is Llama-3.1-70B-Instruct and the draft model is Llama-3.2-1B-Instruct. The trace reveals temporal fluctuations in speculation length, compounding overall dynamism. These optimizations collectively reshape attention masks into irregular trapezoidal patterns, as illustrated in Fig. 1(b). Within individual inferences, batched inputs of heterogeneous lengths and characteristics pose substantial optimization challenges.

