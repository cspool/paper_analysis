# PROMOE: Fast MoE-based LLM Serving using Proactive Caching

Xiaoniu Song<sup>1</sup> Zihang Zhong<sup>3,\*</sup> Rong Chen<sup>1,‡</sup> Haibo Chen<sup>1,2</sup>

<sup>1</sup>Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University <sup>2</sup>Key Laboratory of System Software (Chinese Academy of Sciences) <sup>3</sup>Zhejiang University

#### Abstract

The promising applications of large language models are often limited by the constrained GPU memory capacity available on edge devices. Mixture-of-Experts (MoE) models help address this issue by activating only a subset of the model's parameters during computation. This approach allows the unused parameters to be offloaded to host memory, thereby reducing the overall GPU memory demand. However, existing cache-based offloading solutions handle cache misses reactively, which significantly impacts system performance. In this paper, we introduce ProMoE, a novel proactive caching system that utilizes intermediate results to predict subsequent expert usage. By proactively fetching experts in advance, PROMoE eliminates passive cache misses, removes loading time from the critical path, and reduces the performance overhead associated with offloading. Our evaluations demonstrate that PROMoE achieves an average speedup of  $2.20\times$  (up to  $3.21\times$ ) and  $2.07\times$  (up to  $5.02\times$ ) in the prefill and decode stages, respectively, compared to existing offloading solutions.

### <span id="page-0-1"></span>1 Introduction

Large language models (LLMs) have revolutionized various fields, including natural language processing, content generation, and decision support [10, 29, 38, 44, 52]. Traditionally, these models have been deployed in data centers equipped with high-end GPUs. However, there is growing interest in running LLMs on consumer-grade platforms to enhance privacy and speed [12, 41, 50]. Despite this growing interest, significant challenges remain due to memory limitations. LLMs typically require substantial memory (often hundreds of gigabytes) [29, 44, 52], which exceeds the capacities of consumergrade GPUs, generally limited to around a dozen gigabytes. This limitation leads to serious performance issues, ultimately hindering the efficiency and adoption of LLMs on personal computers.

Mixture-of-Experts (MoE) [15, 17, 27, 28, 49] offers an opportunity to address the GPU memory constraints faced by LLMs by dividing the model into multiple experts and activating only a few during inference. This approach allows most experts to be offloaded to host memory, loading only the necessary ones into GPU memory. While this significantly reduces GPU memory requirements, *expert offloading* also introduces severe performance degradation up to 8.9× [30] due to limited PCIe bandwidth between host and GPU memory (32GB/s unidirectional on PCIe 4.0).

Recently, researchers have proposed caching frequently accessed experts in GPU memory to minimize offloading costs [18]. However, this caching approach handles missing experts in a **reactive** manner. Specifically, a cache miss is triggered passively when an expert is

<span id="page-0-0"></span>![](_page_0_Figure_12.jpeg)

Figure 1: A comparison of execution flow between reactive and proactive caching.

accessed during inference, leaving the expensive loading on the critical path (see Figure 1). For instance, when caching 50% of the experts in the deepseek-moe [4] model, the time spent on loading missing experts accounts for over 60% of the total inference time. Additionally, the inherent low skewness and poor locality of expert access patterns in MoE models, especially in modern decoder-only architectures, significantly limit the potential improvements that can be achieved through better caching policies.

In this paper, we propose ProMoE, a novel system to address the performance challenges associated with offloading in MoE-based LLMs through **proactive** caching, as shown in Figure 1. By actively predicting which experts will be needed and prefetching their parameters into a cache in GPU memory, ProMoE can take the time required for fetching missing experts off the critical path. This allows for better overlap with computation, enhancing overall performance and GPU utilization.

To achieve effective proactive caching, PROMOE addresses two main questions. First, given the dynamic nature of MoE models, PROMOE requires a predictive approach for prefetching. To evaluate the quality of a prediction method, PROMOE introduces a metric called GoodPred. This metric considers both the accuracy and efficiency of the predictions. To achieve a high GoodPred score, PROMOE proposes a learned predictor that prefetches experts in a stride manner. This learned predictor identifies correlations between intermediate results and expert selections , allowing for accurate predictions of experts while the stride prefetching technique perfectly hides prediction latency, ensuring high efficiency of prefetching.

Second, the processes of prefetching and inference can interfere with each other, leading to low utilization of the GPU, cache, and bandwidth for prefetching. Therefore, PROMOE needs to carefully coordinate these two processes to minimize interference. We observed that the required experts for each layer can be identified all at once, which creates opportunities to optimize prefetching and inference for better overlap. Based on this insight, PROMOE proposes several techniques to coordinate the execution of prefetching and inference processes, including chunked prefetching, early preemption, and reordered inference. These techniques eliminate passive cache misses and maximize the overlap between prefetching

<sup>‡</sup> Rong Chen is the corresponding author (rongchen@sjtu.edu.cn).

<sup>\*</sup> During internship at Shanghai Jiao Tong University.

<span id="page-1-0"></span>![](_page_1_Figure_0.jpeg)

Figure 2: (a) The execution flow of large language models (LLMs), (b) the architecture of a transformer layer in LLMs, and (c) the architecture of a Mixture-of-Experts (MoE) block that replaces FFN in a transformer layer.

<span id="page-1-1"></span>Table 1: MoE-based LLMs description. P, L, and E denote parameters, layers, and experts, respectively. Act. indicates the number of activated parameters or experts during the inference of a single token.

| MoE-based LLM               | #P    |       | #L | #E per L |      |
|-----------------------------|-------|-------|----|----------|------|
|                             | Total | Act.  |    | Total    | Act. |
| Deepseek-moe (DS-1) [4]     | 16.4B | 2.8B  | 28 | 64       | 6    |
| Deepseek-v2-lite (DS-2) [5] | 15.7B | 2.7B  | 27 | 64       | 6    |
| Qwen1.5-moe (QW-1) [7]      | 14.3B | 2.7B  | 24 | 60       | 4    |
| Qwen2-moe (QW-2) [8]        | 57.4B | 14.2B | 28 | 64       | 8    |
| Mixtral-8x7B (Mixt) [1]     | 46.7B | 12.9B | 32 | 8        | 2    |

and inference, thereby reducing inference latency and improving utilization.

To demonstrate the effectiveness of ProMoE in serving MoE-based LLMs on consumer-grade hardware, we integrated ProMoE into two widely used LLM frameworks: transformers and llama.cpp. Compared to hand-crafted caching baselines with state-of-the-art performance, ProMoE achieves an average speedup of 1.78× (up to 2.48×) in the prefill stage and 1.34× (up to 1.79×) in the decode stage. When compared to existing offloading methods available in open-source LLM frameworks, ProMoE achieves an average speedup of 2.20× (up to 3.21×) and 2.07× (up to 5.02×) for these two stages. The source code of ProMoE is publicly available at https://github.com/promoe-opensource/promoe.

### **Contributions**. We make the following contributions.

- (1) A new metric called "GoodPred" that can holistically evaluate various predictors in expert prefetching (§4).
- (2) A novel learned predictor, coupled with a stride mechanism, that achieves high accuracy while hiding prediction latency (§4).
- (3) A sophisticated proactive cache that eliminates passive cache misses by coordinating prefetching and inference (§5).
- (4) An implementation integrated into mainstream LLM frameworks (§6), along with an evaluation showing the efficacy and efficiency of ProMoE compared to state-of-the-art solutions (§7).

# 2 Background

### 2.1 Mixture-of-Experts (MoE) based LLMs

Large Language Models (LLMs) perform inference in two stages: prefill and decode, as illustrated in Figure 2(a). During the prefill stage, the model processes the user's input prompt in a single iteration. The tokens within the prompt are processed in parallel by the model, and the first token of the response is generated at the end of this iteration. In the decode stage, each iteration processes only one token generated from the previous iteration, producing the next token. These tokens are fed into the model sequentially and ultimately concatenated to form the complete response. Due to the differing computational scales between the two stages of LLM inference, their performance is typically measured separately. The performance of the prefill stage is usually quantified by the Time to First Token (TTFT), which represents the duration users wait for the LLM to process the prompt before it starts generating output. For the decode stage, performance is commonly measured using either Tokens Per Second (TPS) or Time Per Output Token (TPOT).

Large Language Models (LLMs) consist of a series of transformer layers. Each layer contains a self-attention block (self-attn) and a feed-forward network (FFN), as shown in Figure 2(b). These components process the input hidden states, add the results back to the inputs, and pass them to the next layer. Due to layer normalization, the outputs are numerically smaller than their inputs, leading to a slow change in hidden states across layers [33, 36]. Typically, the cosine similarity between hidden states of adjacent layers averages around 90%.

The Mixture-of-Experts (MoE) architecture enhances LLMs by expanding the FFN into multiple experts, as depicted in Figure 2(c). This approach increases the model's parameters while reducing overall computation, since only a subset of experts is activated during each forward pass. Specifically, each MoE block consists of a gate function and multiple experts. The gate function prioritizes which experts should process the current token. Each expert is structurally similar to the original FFN but contains fewer parameters. The output of the MoE block is a weighted average of the outputs from all activated experts.

In MoE-based LLMs, expert selection occurs independently for each token, as shown in Table 1. When processing multiple tokens

<span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

Figure 3: The (a) latency and (b) time breakdown of LRU caching under different cache rates in transformers with DS-1 model.

<span id="page-2-1"></span>![](_page_2_Figure_2.jpeg)

Figure 4: The (a) latency and (b) time breakdown of LRU caching under different cache rates in llama.cpp with QW-2 model.

simultaneously (e.g., when processing prompts or batching multiple requests), a larger portion of experts is activated, ranging from over 50% to nearly 100%, depending on the number of tokens.

#### 2.2 Caching MoE-based LLMs

In MoE-based LLMs, each token utilizes only a subset of experts. Most experts can be offloaded to CPU memory, with only the necessary experts loaded into GPU memory. This allows MoE-based LLMs to run on consumer-grade hardware with limited GPU memory. However, due to the limited PCIe bandwidth, directly offloading parameters to CPU memory can lead to high latency and low GPU utilization. For instance, when running the DS-1 model with 50% of experts offloaded to CPU memory, the TPOT is 67.9 ms, while fetching experts from host memory takes 58.1 ms, accounting for 85.6% of the total time. Each output token requires 2.67 GiB of expert parameters in FP16 precision, with 1.33 GiB needing to be transferred from CPU memory to GPU memory due to offloading. The achieved bandwidth is 23 GB/s, which matches the achievable bandwidth (23.9 GB/s in our bandwidth test) from host to GPU using PCIe 4.0x8.

To mitigate the performance issues caused by offloading, a traditional method is to cache frequently accessed experts in GPU memory. A common approach is to use LRU (Least Recently Used) or static caching to store these frequently accessed experts. For example, Mixtral-offloading [18] implements an LRU cache for the Mixtral model. Another example is CUDA's Unified Memory (UM), which leverages a paging mechanism to transfer data between the GPU and CPU on demand.

<span id="page-2-2"></span>![](_page_2_Figure_8.jpeg)

Figure 5: The (a) CDF of expert access frequency and (b) hit rate of LRU caching. The upper figures show results of traditional encoder-decoder models (switch-transformer [19], NLLB [13]), while the lower figures show results of modern decoder-only models listed in Table 1.

The major challenge of caching in MoE is its **reactive** nature when handling cache misses. When the inference process encounters an expert that is missing from GPU memory, the computation is blocked until the expert is fetched from host memory, resulting in high latency overhead in the critical path of inference.

We evaluated the performance of LRU caching in transformers [45] with the DS-1 (fp16) model, along with llama.cpp [22] using the QW-2 (int4) model. Figure 3 and 4 illustrate the inference latency and the time breakdown for both the prefill and decode stages. In the case of the DS-1 model, caching 50% of experts results in a blocking time of 60.4% on the critical path during the decode stage. The blocking time during the prefill stage is more severe, as more experts are accessed, leading to an 82.7% blocking time on the critical path. For llama.cpp, which achieves faster inference by eliminating the overhead of the Python interpreter, the proportion of blocking time is even greater. Caching 50% of experts results in 94.2% blocking time during the prefill stage and 79.0% during the decode stage.

Another factor that exacerbates the impact of blocking time on the critical path is the access frequencies of different experts in MoE-based LLMs, particularly modern decoder-only LLMs, which tend to be less skewed. Figure 5 shows the cumulative access frequency of experts and the hit rate of LRU caching. Traditional encoder-decoder MoE models like switch-transformer [19] (SWI) and NLLB [13] released in 2022 exhibit a power-law distribution in expert access frequencies, where a small number of experts are accessed more frequently than others. This high skewness leads to high hit rates and benefits both static and dynamic caching like LRU. In contrast, modern decoder-only MoE models exhibit a more uniform access pattern, as shown in the bottom of Figure 5. This

<span id="page-3-1"></span>![](_page_3_Figure_0.jpeg)

Figure 6: The architecture of ProMoE.

low skewness creates unique challenges for offloading and caching in modern MoE models.

This uniform access pattern can be attributed to the deliberate design of modern MoE models, which utilize various techniques during training to prevent any single expert from becoming a hotspot. This is crucial because uneven expert utilization can lead to inadequate training of certain experts, ultimately impacting the model's performance. This phenomenon is referred to as "routing collapse" [\[39\]](#page-13-7). To mitigate routing collapse, contemporary MoE models incorporate strategies such as Device-Limited Routing [\[17\]](#page-12-4) and Expert-Level Balance Loss [\[15\]](#page-12-3) during the training process. Consequently, the access frequencies of different experts tend to be more uniform during inference. Combined with the reactive handling of cache misses, the caching solution significantly degrades the critical path latency.

# 3 Overview of ProMoE

This paper presents ProMoE, a system that achieves low-latency inference for MoE-based LLMs on consumer-grade platforms. Pro-MoE addresses the reactive nature of existing solutions, which passively trigger data transfers on the critical path of inference, leading to high latency. To tackle this issue, ProMoE adopts a proactive caching approach. Instead of directly reducing data transfers between the CPU and GPU, proactive caching moves data transfers out of the critical path, allowing them to overlap with inference.

The architecture of ProMoE is illustrated in Figure [6.](#page-3-1) It consists of two main components: the predictor and the prefetcher. The predictor periodically predicts which experts will be selected. Based on these predictions, the prefetcher preloads experts into the GPU cache. During inference, the LLM inference engine accesses experts stored in the cache and triggers misses for any experts that are absent. Compared to existing solutions, most expert data transfers in ProMoE occur outside the critical path of inference, thus reducing latency and improving GPU utilization.

To achieve effective proactive caching, ProMoE must address the questions of "what to prefetch" and "how to prefetch" as mentioned in §[1.](#page-0-1) The predictor in ProMoE tackles the first question by making good predictions. To define what constitutes a good prediction, Pro-MoE proposes a GoodPred metric that considers both the accuracy and efficiency of the predictions. Based on this metric, ProMoE introduces a learned predictor that prefetches experts in a stride manner. This learned predictor memorizes the correlations between

intermediate results and expert selections to make accurate predictions of expert selections. Additionally, through stride prefetching, ProMoE overlaps the processes of prediction and prefetching to hide the latency of predictor.

ProMoE's prefetcher addresses the second question by carefully coordinating the prefetching and inference processes. Naive prefetching can lead to interference between these processes, resulting in suboptimal performance. ProMoE leverages the observation that the choice of experts for each layer becomes available all at once after the gating function. Based on this insight, ProMoE proposes three key techniques to effectively coordinate prefetching and inference: chunked prefetching, early preemption, and reordered inference. With these techniques working in concert, ProMoE can eliminate passive cache misses and maximize the overlap between prefetching and inference, thereby reducing inference latency.

# <span id="page-3-0"></span>4 GoodPred, Prediction, and Prefetching

The dynamic nature of MoE models necessitates the deployment of a predictor in ProMoE to make approximate predictions of experts for prefetching. To ensure effective prefetching, the predictor must meet two primary requirements: accuracy and efficiency. In this section, we first define a key metric called GoodPred, which combines these two aspects to evaluate the performance of a predictor. Subsequently, we introduce ProMoE's learned predictor and explain how it improves both accuracy and efficiency.

### 4.1 A New Prediction Metric: GoodPred

A good predictor requires both high accuracy and efficiency. Higher accuracy increases the likelihood that predicted experts will be utilized, while higher efficiency allows more time to load these predicted experts. These two goals must be pursued simultaneously, though these two goals might initially seem contradictory improving accuracy often requires more prediction time, which can reduce the time available for prefetching.

To assess the performance of the predictor, we define the Good-Pred metric as follows:

## GoodPred = Accuracy × FetchRate

GoodPred evaluates the effectiveness of the predictor in predicting experts for prefetching by considering both Accuracy and FetchRate. The Accuracy denotes the proportion of correctly predicted experts, while the FetchRate signifies the portion of predicted experts that can be prefetched in time before they are accessed during LLM inference. Thus, GoodPred measures the volume of correct experts that can be prefetched in a timely manner.

### 4.2 Existing Approaches

Recent research has proposed two main methods for predicting expert usage. Previous studies [\[28,](#page-12-6) [47\]](#page-13-8) introduced a token-based predictor that predicts expert usage based on input tokens, allowing for an iteration-wise prefetch pattern, as illustrated in Figure [7\(](#page-4-0)a). These studies suggest that the selection of experts in one iteration is closely related to the input token ID. This relationship can be intuitively explained: LLMs convert the input token ID into an embedding vector through a fixed mapping, and the computation in each iteration gradually adds contextual information to these embeddings. Consequently, the input token ID can be utilized to

<span id="page-4-0"></span>![](_page_4_Figure_0.jpeg)

Figure 7: Candidate prefetch manners.

predict the selection of experts across all layers within that iteration. Specifically, in the offline stage, a trace of input token IDs and their selected experts is collected. Then, during online inference, the predictor determines which experts to select for one iteration by identifying the most frequently used experts from this trace based on the input token ID.

By predicting experts for all layers before an iteration begins, the token-based predictor achieves optimal FetchRate, maximizing available time for prefetching. However, this approach suffers from low Accuracy. The iteration-wise pattern conducts prediction over a long distance, leading to decreased accuracy. Moreover, the input token ID lacks contextual information concerning the entire sequence. As shown in Figure 8, the average accuracy of the token-based predictor is only 58.3%. Despite delivering a high FetchRate through iteration-wise prefetching, the low Accuracy renders nearly half of this prefetching ineffective, resulting in a low GoodPred.

Another recent system [18] proposed a **skip-based** predictor that facilitates a **layer-wise** prefetch manner, as illustrated in Figure 7(b). This approach leverages the high similarity between inputs across layers in LLMs [33, 36]. It establishes a skip connection that transmits the input from i-th layer's MoE gate directly to the MoE gate in i + 1-th layer, thereby predicting the experts for i + 1-th layer at the point of i-th layer. For instance, in the DS-2 model, the cosine similarity between the consecutive layers' inputs is 91.7%. Thus, passing the input of the i-th layer to the i + 1-th layer's gate is likely to yield accurate predictions.

However, the skip-based predictor's accuracy remains limited.It depends on the similarity of inputs across different layers and the numerical stability of the gate function, which does not uniformly apply across all models. In Figure 8, the skip-based predictor achieves high accuracy with noticeable accuracy drop in the head and tail layers for the DS-1 model. However, the QW-2 model experiences a significant accuracy decline with an average accuracy of only 66.9%. This discrepancy arises because the gate function in the QW-2 model is sensitive to input variations, causing shifts in priority for expert selection even with slight input changes. Additionally, the layer-wise prefetch pattern of the skip-based predictor incurs higher prediction overhead, thus limiting prefetch efficiency.

<span id="page-4-1"></span>![](_page_4_Figure_6.jpeg)

Figure 8: The prediction accuracy of each layer in (a) DS-1 model and (b) OW-2 model.

# 4.3 Learning-based Predictor

To achieve high Accuracy, PROMOE introduces a learned predictor to conduct layer-wise prefetch. The main idea is to collect the correlation between layer inputs and expert selections across layers and memorize this correlation in predictors. The predictor then uses these correlations to make predictions. When paired with layer-wise prefetch, the learned predictor maintains high Accuracy.

PROMOE's learned predictor employs a small neural network (NN) to learn correlations between layer inputs and expert selections. This approach, which utilizes a small NN like multi-layer perceptrons (MLPs) as the predictor, has been effectively applied and validated in various system research contexts [24, 31, 36, 41]. These NNs are capable of learning complex correlations while providing fast predictions, which can be more challenging for traditional heuristic methods like nearest neighbor search. However, a significant drawback of NN-based methods is their lengthy training time. Fortunately, in the context of serving LLMs, the offline training is a one-time task for each LLM and is negligible compared to the extensive pre-training time required for LLMs [29].

The learned predictor in ProMoE operates in two phases: offline training and online prediction. In the offline phase, ProMoE trains a set of predictors by performing multiple iterations of LLM inference. This process collects the input for each layer and the corresponding output of the gate function. Based on these collected traces, ProMoE trains a set of predictors for each layer to learn and memorize the correlations between inputs and outputs.

To ensure the predictor's generalizability, PROMOE collects traces from the domain datasets used during either LLM training or inference. This approach ensures that the predictor aligns with the model across various conditions. Following standard practices, the collected traces are split into training and evaluation sets with a 9:1 ratio. The predictor is trained solely on the training set and evaluated only on the unseen data from the evaluation set.

In the online inference, the input for each layer is collected and fed into the corresponding predictor(s) to make predictions. The prediction output, similar to a gate's output, indicates the prefetch priority of experts in one layer. Based on this output, the predictor selects the same number of experts that the model would activate for one token (e.g. 6 for the DS-1 model in Table 1) and hands over these experts to the prefetcher for prefetching.

To assess the accuracy of different predictors, we evaluated them using the evaluation set of collected traces. As shown in Figure 8, PROMOE learned predictor maintains high Accuracy across both models, achieving an average accuracy of 84.7%. This improved accuracy enables PROMOE to accurately prefetch experts in a timely

manner, optimizing the use of the limited bandwidth between the CPU and GPU.

# 4.4 Stride Prefetching

To minimize the impact of the prediction on critical path latency, ProMoE executes the predictor on the CPU, allowing it to run concurrently with LLM inference. The latency of a single predictor on the CPU is about 200 microseconds. Compared to the millisecondlevel computation time of a single layer in LLMs, this latency can be hidden since the CPU-based prediction process operates in parallel with the LLM inference on the GPU. However, in layer-wise prefetching, the predictor's latency consumes available time for prefetching experts, resulting in a lower FetchRate.

To enhance the FetchRate, ProMoE introduces stride prefetching as shown in Figure [7\(](#page-4-0)c). Stride prefetching increases the prediction distance by 1, allowing prefetching to begin earlier than in layer-wise prefetching. Moreover, stride prefetching pipelines the prediction and prefetching processes, executing them simultaneously. In contrast to layer-wise prefetching, where prediction and prefetching are carried out sequentially, stride prefetching ensures that all available bandwidth between the CPU and GPU is fully utilized for prefetching. Consequently, this approach maximizes the FetchRate and provides a higher GoodPred.

While increasing the prediction distance may lead to a decrease in the predictor's Accuracy, practical observations reveal that the accuracy of ProMoE's learned predictor only declines by 5% during stride prefetching. Additionally, stride prefetching offers ample design space for more sophisticated predictors that may require additional time to generate predictions.

# <span id="page-5-0"></span>5 Coordination of Prefetching and Inference

The prefetcher in ProMoE is responsible for fetching experts into the GPU cache based on prediction results. It consists of a worker thread and a task queue. The worker thread retrieves prefetch tasks from the queue and copies the corresponding experts into the GPU's expert cache. The task queue maintains two priority levels: lowpriority speculative prefetch tasks provided by the predictor, and high-priority precise prefetch tasks triggered by cache misses during LLM inference. The worker thread always prioritizes executing high-priority tasks over low-priority ones.

To further enhance the coordination between expert prefetching and LLM inference, ProMoE proposes several optimizations: chunked prefetching, early preemption, and reordered inference. These optimizations aim to minimize interference and maximize the overlap between prefetching and inference, as illustrated in Figure [9.](#page-5-1)

# 5.1 Chunked Prefetching

When high-priority prefetch tasks are added to the queue, there is typically ongoing fetching of expert parameters from CPU to GPU. This fetching may stem from an incomplete prefetch task of the current layer or from a prefetch task of subsequent layers that has already begun. Due to the limitations of CUDA's asynchronous copy mechanism, an ongoing copy operation cannot be preempted midway. As a result, high-priority prefetch tasks must wait for the current copy operation to complete before they can start, introducing unnecessary latency into the critical path.

<span id="page-5-1"></span>![](_page_5_Figure_10.jpeg)

Figure 9: ProMoE coordinates prefetching with inference using a series of optimizations. Assume experts 1, 3, 4, 5 are prefetched in advance, and gate produces 1, 2, 4, 5, where expert 2 is not in cache.

To address this issue, ProMoE introduces chunk-based prefetching. The key idea is to split the parameters of each expert into multiple chunks. When the prefetcher identifies the predicted experts (from predictor), it divides their parameters into several chunks and adds them to the prefetch queue as low-priority tasks. Each task corresponds to one chunk of an expert's parameters, rather than the entire expert. This allows the worker thread to schedule low-priority tasks at a smaller granularity. When a high-priority prefetch task arises, the worker thread can quickly switch to it, encountering a maximum delay of just one chunk.

Figure [9](#page-5-1) illustrates an example of chunked prefetching. The cache miss for expert 2 is triggered after the execution of expert 1. Since the prefetcher is already working on a low-priority task, it must wait until this task completes before handling the high-priority task of expert 2. With chunked prefetching, the low-priority task is divided into three chunks. The cache miss is triggered while the prefetcher is working on the second chunk, allowing the highpriority task of expert 2 to start immediately after the second chunk is completed. In practice, we found that experts in MoE models share the same structure, consisting of three linear layers. Thus, ProMoE naturally splits each expert into three chunks, corresponding to these three linear layers. By implementing chunked prefetching, ProMoE reduces the delay in starting high-priority prefetch tasks, thereby improving critical path latency.

### 5.2 Early Preemption

Although ProMoE's predictor aims to maximize prediction accuracy, mispredictions are still unavoidable. This can result in necessary experts not being present in the GPU cache, triggering ondemand copying of missing experts during the critical path. Traditionally, these misses are detected and addressed only when the corresponding expert is accessed during inference, causing the inference process to be blocked while waiting for the missing expert parameters to be copied from CPU memory to GPU. This leads to under-utilization of the GPU and introduces high fetch latency in the critical path of inference execution.

To tackle this issue, ProMoE proposes early preemption. We observed that, in MoE models, the experts needed for the current layer are determined all at once when the gate function completes.

# <span id="page-6-0"></span>Algorithm 1 Prefetch Worker Thread

```
1: while True do
       task \leftarrow queue.pop()
2:
       if task.chunk = 0 then
3:
            evicted expert \leftarrow cache.replace with(task.expert)
4:
            evicted_expert.ready_chunk \leftarrow 0
5:
       end if
6:
       cache\_ptr \leftarrow cache.get(task.expert)
7:
       offset \leftarrow task.chunk \times chunk\_size
       copy(cache_ptr + offset, task.host_ptr + offset, chunk_size)
       task.expert.ready chunk ← task.chunk + 1
10:
11: end while
```

Instead of causing a cache miss each time an individual expert is accessed, the system can preempt the prefetch queue in advance when it knows which experts will be activated after the gate function. This allows the prefetching of any missing experts to begin much earlier, overlapping with the computation of the current layer. For example, as shown in Figure 9, early preemption triggers the cache miss for expert 2 immediately after the gate function completes, rather than waiting until the completion of expert 1. As a result, the high-priority task for expert 2 is scheduled by the prefetcher before the second chunk of the low-priority task is processed.

In practice, ProMoE implements early preemption by inserting a hook at the end of the gate function to obtain the list of required experts in advance. These experts are then prioritized as high-priority tasks and added to the prefetch queue, ensuring that the prefetch thread prioritizes these tasks. During this process, there may still be some low-priority speculative prefetch tasks for the same layer that have not yet completed. However, since the system has a precise list of the required experts, these low-priority tasks can be discarded. The prefetch thread simply clears any remaining low-priority speculative prefetch tasks for that layer, effectively achieving preemption.

During inference, when encountering an expert that is not in the cache, PROMOE no longer triggers a cache miss. Instead, it waits for the corresponding prefetch task to complete. As a result, all passive cache misses are transformed into proactive precise prefetching. This approach allows for earlier initiation of accurate prefetching, which increases the overlap between prefetching and computation, ultimately reducing latency on the critical path.

#### <span id="page-6-2"></span>5.3 Reordered Inference

In the inference process of LLMs, existing frameworks typically execute computations for different experts in the order of their IDs. This order often fails to fully utilize the cache status of experts, leading to unnecessary blocking and potential cache thrashing. Consider the example in Figure 9 where experts 1, 4, and 5 are cached, and expert 2 is missing. Since the computations are executed based on the order of expert ID, experts 4 and 5 must wait for the prefetch of expert 2 to complete before they can start. Consequently, the GPU remains underutilized while waiting for the prefetch of expert 2, even though experts 4 and 5 are already prefetched. More critically, the prefetching of the missing expert might evict other soon-to-be-accessed experts, causing cache thrashing. This issue is

# <span id="page-6-1"></span>Algorithm 2 Prefetcher Interface

```
1: function PushPredictedExperts(layer, experts)
       for e in experts do
          if e.ready_chunk > 0 then
3:
4:
              cache.hit(e)
 5
 6:
          for chunk ← e.ready_chunk to num_chunks-1 do
              queue.push(Task(layer, e, chunk, LOW))
7:
       end for
10: end function
   function PushPreciseExperts(layer, experts)
       queue.remove_low_pri_task_with_layer(layer)
       experts \leftarrow desc\_sort\_by\_ready\_chunk(experts)
13:
       for e in experts do
14:
          if e.ready_chunk > 0 then
              cache.hit(e)
16:
17:
          end if
          for chunk ← e.ready chunk to num chunks-1 do
18
              queue.push(Task(layer, e, chunk, HIGH))
19
          end for
20
       end for
       return experts
23: end function
```

particularly severe when dealing with a large number of experts sequentially, such as during the prefill stage of inference.

To address this issue, PROMOE proposes reordered inference, which alters the computation order of experts in a cache-aware manner. We observe that in MoE models, the computation order of experts is interchangeable. There is no dependency between the computations of different experts because their outputs are simply summed together. This property allows for adjusting the computation order based on the cache and prefetch status, making the inference process more cache-friendly.

Specifically, once the gate function completes, PROMOE adjusts the computation order accordingly. Experts already in the cache are prioritized first, followed by the experts currently being prefetched (if any), while experts whose prefetch has not yet begun are positioned last. Consider the example in Figure 9. When the gate produces experts 1, 2, 4, and 5, where expert 2 is missing, PROMOE changes the computation order to 1, 4, 5, and then 2. Therefore, the prefetching of expert 2 can be conducted in parallel with the computations of experts 4 and 5, further reducing the impact of prefetching on the critical path.

In practice, the reordering process occurs simultaneously with early preemption. After obtaining the list of experts to be accessed, ProMoE first reorders them as described above. Experts whose prefetching is not yet complete are managed through early preemption and added to the prefetch queue as high-priority tasks. The entire reordered sequence of experts is then returned to the inference framework for execution. This approach ensures that for experts with incomplete prefetches, both the prefetch threads and inference threads process them in the same order, effectively establishing a pipeline between computation and prefetching.

#### 5.4 Prefetcher Workflow

The prefetcher's workflow is summarized in Algorithms 1 and 2. Algorithm 1 outlines the prefetcher's worker thread, which continuously polls tasks from the queue and transfers expert parameters from host memory to GPU memory. Each task corresponds to a chunk of an expert's parameters, thereby implementing chunked prefetching.

The Predictor and LLM framework interact with the prefetcher through the APIs outlined in Algorithm 2. The Predictor enqueues predicted experts as low-priority tasks using the PushPredictedExperts function, while the LLM framework enqueues the actually required (precise) experts as high-priority tasks with the PushPreciseExperts function after completing the gate function.

When enqueuing high-priority tasks (precise experts), the system first clears any existing low-priority tasks from the queue (Line 12) to enable early preemption. The remaining precise experts are then reordered based on their current fetch status (Line 13). Subsequently, the inference framework executes the experts according to this new ordering (Line 22), thereby implementing reordered inference.

### <span id="page-7-0"></span>6 Implementation

ProMoE is implemented as an extension to LLM frameworks, comprising 6,600 lines of C++ code.

### 6.1 Cache Implementation

For simplicity, the cache component of ProMoE is implemented as a standard per-layer LRU cache. Both prefetching and inference trigger a cache access. When adding prefetch tasks, ProMoE leverages LRU by accessing experts that are already cached, thereby delaying their eviction. To reduce memory fragmentation, ProMoE pre-allocates the expert cache as a contiguous memory region.

# 6.2 System Integration

We have integrated ProMoE into two popular LLM frameworks: transformers and llama.cpp. To achieve this integration, we added hooks to capture input logits from the MoE layers and to reorder experts. We also implemented a dependency mechanism to ensure efficient prefetching and computation. Furthermore, ProMoE takes over the memory management for expert parameters. We did not integrate ProMoE with frameworks like vLLM and TGI due to their inadequate support for quantized MoE at the time of submission. Moreover, these frameworks primarily focus on batched inference for data centers and fuse the execution of multiple experts to enhance GPU utilization. However, this optimization assumes that all experts are ready before computation can begin, which is atypical on memory-constrained GPU platforms that ProMoE targets. It necessitates additional GPU memory for activated experts and hampers the overlap between expert loading and execution.

### 6.3 Training of Predictor

<span id="page-7-1"></span>Each layer's predictor in ProMoE is implemented as a two-layer multi-layer perceptron (MLP) with approximately 2 million parameters. The training of the learned predictor and the data collection for training takes less than 1–2 hours on a single GPU. This is a one-time offline task that can be parallelized across multiple GPUs. Given the extensive pre-training times of large language models (LLMs), we consider this time commitment acceptable.

#### 7 Evaluation

### 7.1 Experimental Setup

Hardware. The evaluation is conducted on a PC equipped with an NVIDIA RTX 4090 GPU (24 GB GDDR6X). The PC also features an Intel i9-14900K CPU and 128 GB of host DRAM. The GPU is connected to the CPU via PCIe 4.0, providing a unidirectional bandwidth of 32 GB/s. To simulate GPUs with varying memory capacities, we include an evaluation in §7.4 that adjusts the cache ratio to control memory occupancy.

Workload. We evaluated a broad range of MoE-based LLMs, as listed in Table 1. By default, we evaluate DS-1, DS-2, and QW-1 using FP16 precision, while QW-2 and Mixt are evaluated using INT4 precision. To further study the impact of model size, we also include an evaluation in §7.6 that varies the parameter size of the same model from FP16 to INT4. The evaluation utilizes the shareGPT dataset [3], which consists of user interactions with ChatGPT and serves as a representative example of real LLM services. We also conducted evaluations using the Alpaca dataset [43] and observed similar performance trends; results for this dataset are omitted due to space limitations. By default, we set the batch size to 1 to reflect edge deployment scenarios, and we include an evaluation in §7.5 that varies the batch size from 1 to 4.

**Baselines**. Our evaluation relies on two well-known codebases: Hugging Face transformers [23, 45] and llama.cpp [22]. Transformers supports a wide range of models and is easy to deploy, but it lacks optimal inference performance. We enhanced the efficiency of the MoE block by reducing CPU-GPU synchronization. Llama.cpp, which is written in C++, delivers state-of-the-art inference performance by eliminating overhead from the Python interpreter.

Both systems offer offloading baselines: transformers offloads only the parameters to the CPU (referred to as **TO**), while llama.cpp offloads both parameters and computations (referred to as **LO**). We improved TO by incorporating pinned memory and asynchronous copies. Additionally, we integrated PROMOE into both codebases and introduced three baselines: Unified Memory (**UM**), **static** cache, and **LRU** cache. These baselines, along with PROMOE, manage only expert parameters, while non-expert parameters consistently reside on the GPU. The UM baseline is optimized using cudaMemAdvise to enable instantaneous page invalidation without incurring the cost of swapping pages back to CPU memory. The static baseline caches a fixed set of experts and utilizes two additional expert buffers to load any missing experts.

**Metrics** We evaluate the performance of ProMoE and its baselines in the prefill and decode stages separately. The prefill stage performance is assessed by TTFT (Time To First Token), which reflects the latency in processing the user's prompt. The decode stage performance is measured using TPS (Tokens Per Second) and TPOT (Time Per Output Token), indicating the throughput and latency of the decoding process. We primarily report TPS as it is more intuitive and switch to TPOT for detailed breakdown analyses. The total latency for a single request can be expressed as  $Latency_{total} = TTFT + N \times TPOT$  (where N is output length). We measure the prefill and decode stages separately for two main reasons: (1) the significant variance in output lengths (ranging from tens to thousands of tokens) renders aggregated metrics unreliable

<span id="page-8-0"></span>![](_page_8_Figure_0.jpeg)

Figure 10: The overall performance of (a) prefill and (b) decode stage in transformers codebase.

for system comparisons, and (2) the prefill and decode stages exhibit distinct computational patterns (e.g. more experts are activated in the prefill stage).

#### 7.2 Overall Performance

Figure 10 shows the overall performance of the prefill and decode stages within the transformers codebase. In the prefill stage, Pro-MoE outperforms static and LRU baselines by an average of 1.42× (up to 1.61×) and 2.21× (up to 2.48×), respectively. The improvement of ProMoE primarily stem from its prefetching technique, which maximizes the overlap between loading parameters and computation. When comparing ProMoE with LRU, the greater improvement is attributed to the cache thrashing caused by LRU (see §5.3). In the prefill stage, nearly all experts are accessed since each token usually requires a different set of experts. As experts are accessed according to their IDs, LRU evicts a cached expert with a higher ID when it accesses a missing expert with a smaller ID first. The static cache avoids thrashing by fixing its cache set, while ProMoE intelligently reorders experts to minimize thrashing and reduce the blocking time caused by missing experts on the critical path.

In the decode stage, ProMoE outperforms the static and LRU baselines by an average of  $1.47\times$  (up to  $1.77\times$ ) and  $1.31\times$  (up to  $1.46\times$ ), respectively. LRU outperforms the static cache during the decode stage because cache thrashing occurs less frequently, and there is some reuse of experts across iterations. ProMoE excels over these baselines by keeping most copies of missing experts off the critical path through effective prefetching.

The TO (resp. UM) baseline consistently perform worse than the static (resp. LRU) baseline. Comparing to these baselines, ProMoE achieves a average speedup of 2.15× (up to 2.78×) in the prefill stage and 2.47× (up to 5.02×) in the decode stage. This performance gap arises because the static and LRU baselines can be seen as improved implementations of static and dynamic cache, respectively. In static cache, the TO baseline offloads non-expert parameters to the CPU, while the static baseline only offloads parameters of the experts. In dynamic cache, the UM baseline fetches parameters at the page level, which increases the volume of transferred data compared to

<span id="page-8-1"></span>![](_page_8_Figure_7.jpeg)

Figure 11: The overall performance of (a) prefill and (b) decode stage in llama.cpp codebase.

the LRU baseline. Therefore, in subsequent experiments, we mainly focus on comparing the static, LRU, and ProMoE.

Figure 11 shows the overall performance in the llama.cpp codebase. ProMoE surpasses the static and LRU baselines by an average of 1.36× (up to 1.75×) and 2.12× (up to 2.22×) in the prefill stage, and by 1.49× (up to 1.79×) and 1.09× (up to 1.17×) in the decode stage, respectively. The improvement in the llama.cpp codebase follows the same trend observed in transformers. However, it is less pronounced due to the removal of the Python interpreter overhead during inference, which provides fewer opportunities for ProMoE to prefetch experts.

As expected, the UM baseline consistently performs worse than the LRU baseline. The LO baseline in llama.cpp offloads both parameters and computations to the CPU, resulting in slower performance than that of the static baseline. Compared to these baselines, PROMOE achieves an average speedup of 2.25× (up to 3.21×) in the prefill stage and 1.66× (up to 2.07×) in the decode stage. However, when evaluating the Mixt model, the LO baseline is significantly faster and even surpasses PROMOE in the decode stage. This is because the Mixt model activates a larger ratio of experts (25%) for each token, increasing the cost of fetching parameters to the GPU compared with directly computing them on the CPU. We believe this does not undermine the significance of our work, as most recently released MoE-based LLMs typically activate a smaller ratio of experts (averaging 10%), and the TO baseline continues to show inferior performance across most cases.

# 7.3 Ablation Study

Figure 12 shows the performance of transformers with different optimizations enabled in ProMoE, starting from the LRU baseline. During the prefill stage, enabling prefetching shows minimal improvement and may even degrade performance. This is because nearly all experts are accessed, and prefetching alone merely replaces the cache set. Additionally, naive prefetching delays the handling of missing experts. The techniques of early preemption and reordered inference provide significant improvements, yielding speedups of  $1.27\times$  and  $2.39\times$  compared to the baseline, respectively.

<span id="page-9-2"></span>![](_page_9_Figure_0.jpeg)

Figure 12: The ablation study of (a) prefill and (b) decode stage in transformers codebase with different optimizations in PROMOE enabled. Base represents the LRU baseline, with prefetch, chunked-prefetch, early-preemption and reordered-inference applied.

<span id="page-9-3"></span>![](_page_9_Figure_2.jpeg)

Figure 13: The ablation study in llama.cpp codebase, following the same setup and naming convention as Figure 12.

In the decode stage, these techniques gradually enhance performance, resulting in a 1.35× increase over the baseline. Figure 13 presents an ablation study for the llama.cpp. The trends is similar to those observed in the transformers, except that in the prefill stage, most of the improvement is attributed to the reordered inference.

#### <span id="page-9-0"></span>7.4 Impact of Cache Rate

To examine the impact of GPU memory capacity on PROMoE's performance, we varied the cache rate to control the memory occupied by the expert cache. Figure 14 and 15 show the performance of prefill and decode stages of systems in the transformers codebase, with DS-1 and QW-2 models using different cache rates. During the prefill stage, LRU performs the worst due to cache thrashing, while PROMoE outperforms LRU on the DS-1 and QW-2 models by 1.72× (up to 2.36×) and 1.82× (up to 2.28×) on average, respectively. Compared to static caching, ProMoE achieves speedups of 1.22× and 1.39× on average in the prefill stages of these two models, respectively. The enhancement in the QW-2 model is more pronounced due to its increased computation during inference, allowing more opportunities for ProMoE to prefetch experts. Figure 15 also shows the breakdown of time spent loading parameters on the critical path. PROMOE reduces the loading time on the critical path from 69.68% to 30.96% in the QW-2 model as the cache rate increases, whereas the static cache still suffers from a reduction of only 77.44% to 56.04%. In the decode stage, PROMOE outperforms both static and LRU baselines by 1.60× and 1.29× on average, respectively. PROMoE decreases the loading time on the critical path to 25.61% and 29.20% for the DS-1 and QW-2 models, while LRU (the fast baseline) continues to endure loading times on the critical path of 45.52% and 50.89%, respectively.

We conducted similar experiments on the llama.cpp codebase, using layer-offloading (LO) included as a baseline. The results are

<span id="page-9-4"></span>![](_page_9_Figure_8.jpeg)

Figure 14: The (a) TTFT and (b) TPOT of systems in transformers codebase with DS-1 model using different cache rates.

<span id="page-9-5"></span>![](_page_9_Figure_10.jpeg)

Figure 15: The (a) TTFT and (b) TPOT of systems in transformers codebase with QW-2 model using different cache rates.

<span id="page-9-6"></span>![](_page_9_Figure_12.jpeg)

Figure 16: The (a) TTFT and (b) TPOT of systems in llama.cpp codebase with QW-2 model using different cache rates.

shown in Figure 16. In this case, the speedup of ProMoE over the fast baseline is reduced due to the faster inference speed of the llama.cpp codebase. ProMoE achieves performance improvements of 1.53× (resp. 1.14×) and 1.10× (resp. 1.27×) over LRU and static baselines on average during the prefill (resp. decode) stage, respectively. Notably, in the decode stage with a low cache rate, LO outperforms the other systems. Under low cache rates, the cache-based systems must fetch a significant number of experts through PCIe, while the limited computation makes offloading to the CPU more advantageous. As the cache rate increases, however, the cache-based systems quickly surpass LO.

# <span id="page-9-1"></span>7.5 Impact of Batch Size

We also evaluated the impact of batch size on the performance of ProMoE. Figure 17 and 18 show the throughput of systems in

<span id="page-10-1"></span>![](_page_10_Figure_0.jpeg)

Figure 17: The (a) prefill and (b) decode throughput of systems in llama.cpp codebase with DS-1 model as the batch size changes.

<span id="page-10-2"></span>![](_page_10_Figure_2.jpeg)

Figure 18: The (a) prefill and (b) decode throughput of systems in llama.cpp codebase with QW-2 model as the batch size changes.

<span id="page-10-3"></span>![](_page_10_Figure_4.jpeg)

Figure 19: The (a) TTFT and (b) TPOT of systems in llama.cpp codebase with QW-2 model as the batch size changes.

the llama.cpp codebase with DS-1 and QW-2 models as the batch size varies. During the prefill stage, throughput increases linearly with the batch size. This linear growth occurs because the time is primarily dominated by loading all experts, and the increased computation associated with a larger batch size is almost "free". This is supported by Figure 19(a), which shows the time breakdown of the prefill stage for the QW-2 model. As the batch size increases, the latency for one iteration in the prefill stage remains relatively stable. On average, PROMOE outperforms LRU and static baselines by 2.19× and 1.19×, respectively, in the prefill stage.

In the decode stage, the number of experts activated grows almost linearly with the batch size. This rapid increase in latency per iteration during the decode stage limits the improvement of throughput as the batch size increases. In this context, PROMOE outperforms both the LRU and static baselines by averages of 1.22× and 1.59×, respectively. The improvement of PROMOE over LRU grows progressively with increasing batch sizes. For instance, in the

<span id="page-10-4"></span>![](_page_10_Figure_8.jpeg)

Figure 20: The (a) prefill and (b) decode throughput of systems in transformers codebase with QW-2 model as the batch size changes.

<span id="page-10-5"></span>![](_page_10_Figure_10.jpeg)

Figure 21: The (a) TTFT and (b) TPOT of systems in transformers codebase with QW-2 model as the batch size changes.

<span id="page-10-6"></span>![](_page_10_Figure_12.jpeg)

Figure 22: The (a) TTFT and (b) TPOT of systems in llama.cpp codebase with DS-1 model using different bits per weight.

QW-2 model, the speedup of ProMoE over LRU is 1.16× when the batch size is 1 and increases to 1.34× when the batch size reaches 4. This improvement is attributed to cache thrashing that occurs as the batch size grows.

We further illustrate the impact of batch size in the transformers codebase in Figure 20 and 21. Here, ProMoE outperforms LRU and static baselines by averages of  $2.47\times(1.48\times)$  and  $1.54\times(1.87\times)$  in the prefill (decode) stage, respectively. The higher speedup is a result of longer computation times in the transformers codebase, which provides ProMoE with more opportunities to perform additional prefetches.

# <span id="page-10-0"></span>7.6 Impact of Model Size

To understand how model size affects PROMOE's performance, we varied the number of bits per weight (BPW) from 16 to 4 for the same model. The variance in BPW impacts the model's total memory footprint while keeping the amount of computation, measured in

<span id="page-11-0"></span>![](_page_11_Figure_0.jpeg)

Figure 23: The (a) TTFT and (b) TPOT of systems in llama.cpp codebase with QW-2 model using different bits per weight.

<span id="page-11-1"></span>![](_page_11_Figure_2.jpeg)

Figure 24: The (a) TTFT and (b) TPOT of systems in transformers codebase with DS-1 model using different bits per weight.

FLOPs (floating-point operations), constant. This variability can alter the relative speed of parameter loading and computation.

Figure 22 and 23 present the results for the DS-1 and QW-2 models within the llama.cpp codebase. In the DS-1 model, we maintained a fixed cache rate to keep a consistent ratio of parameters stored in the GPU. The GPU memory occupancy decreases with lower BPW values. As shown in Figure 22, the relative performance remains stable despite changes in BPW. PROMOE achieves an average speedup of  $2.05\times(1.21\times)$  and  $1.29\times(1.74\times)$  over LRU and static baselines during the prefill (decode) stage, respectively.

In the QW-2 model, we reduce the cache rate as BPW increases from the default 4-bit to 16-bit to ensure the model fits within our 24 GB GPU memory. The decreased cache rate and increased memory footprint per expert limit PROMOE's improvement. For example, at INT4, PROMOE outperforms LRU by 2.06× in the prefill stage while the speedup drops to 1.105× at FP16.

We also conducted similar experiments in the transformers codebase using the DS-1 model, as illustrated in Figure 24. The quantization was performed using the mainstream method GPTQ [21]. In this scenario, ProMoE effectively overlaps the prefetching of experts with computations as BPW decreases.

### 8 Related Work

Serving MoE-based LLMs with limited resources. Pre-gated MoE [26] modifies the computation flow of MoE models by providing gate results for the subsequent layer from the previous layer directly, allowing accurate layer-wise prefetching by determining the required experts in advance. SwapMoE [30] maintains a set of important experts in the GPU memory, using only these during

inference to prevent offloading overhead. In the background, it dynamically adjusts this set based on workload changes. However, these systems alter the original MoE model computation, inevitably affecting model accuracy. In contrast, ProMoE performs computations that are equivalent to the original model, accelerating the inference of MoE-based LLMs on edge devices without compromising accuracy.

Mixtral-offloading [18] implements an LRU cache for the Mixtral MoE model and introduces a skip-based prediction method to support expert prefetching. Brainstorm [14] designs a router abstraction to capture the dynamic aspects of models and proposes speculative loading and execution based on static skewness statistics. MoE-infinity [48] develops an Expert Activation Tracing mechanism for sequence-level prediction to facilitate prefetching, specifically designed for MoE-based encoder-decoder LLMs and aimed at throughput-oriented inference. In contrast, PROMOE utilizes a learned predictor with high GoodPred, focusing on latency-oriented inference for edge devices.

LLM serving on resource-constrained devices. Most modern frameworks [6, 22, 25, 32, 45] for serving LLMs provide basic offloading support that utilizes the CPU to handle parameters or computations, thereby reducing the GPU memory requirements. FlexGen [40] aggregates memory and computation resources from the GPU, CPU, and disk. It optimizes tensor storage and access patterns while also compressing weights and the attention cache. These frameworks mainly target general LLMs and emphasize throughput-oriented inference with large batch sizes. Model quantization [9, 21, 35] and pruning [20, 42] are techniques to reduce the memory requirements of LLMs. DejaVu [36] takes advantage of contextual sparsity in LLMs to lower inference costs with minimal impact on model quality. It employs a low-cost algorithm to predict input-dependent sparse subsets of attention heads and MLP parameters on-the-fly, which reduces the number of parameters needed during inference. Building on DejaVu, PowerInfer [41] utilizes the power-law distribution of neuron activations in LLMs, preloading frequently activated "hot" parameters onto the GPU while processing less active "cold" parameters on the CPU. Pro-MoE is orthogonal to these techniques and can be integrated with them to further minimize memory usage and enhance inference speed.

Generic LLM serving optimization. The rising demand for LLMs has prompted various system optimizations [2, 11, 16, 34, 37, 46] to improve their performance and efficiency. vLLM [32] introduces PagedAttention, which manages the key-value cache for LLM serving and allows for sharing the cache across requests. This improves batching efficiency and reduces memory fragmentation. Orca [51] proposes continuous and selective batching to optimize the performance of batched LLM serving. While these systems focus on enhancing batched LLM serving in the cloud environment, Pro-MoE is designed specifically for low-latency, single-request LLM inference on edge devices.

### 9 Conclusion

This paper presents ProMoE, a proactive caching system that enhances expert offloading for MoE-based LLMs. ProMoE leverages a learned predictor and carefully coordinates prefetching with inference. Our evaluation shows the efficacy and efficiency of ProMoE.

# References

- <span id="page-12-13"></span>[1] 2023. mistralai/Mixtral-8x7B-Instruct-v0.1 · Hugging Face. [https://huggingface.](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) [co/mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
- <span id="page-12-32"></span>[2] 2023. NVIDIA/TensorRT-LLM.<https://github.com/NVIDIA/TensorRT-LLM> original-date: 2023-08-16T17:14:27Z.
- <span id="page-12-21"></span>[3] 2024. anon8231489123/ShareGPT\_Vicuna\_unfiltered · Datasets at Hugging Face.
- <span id="page-12-9"></span>[4] 2024. deepseek-ai/deepseek-moe-16b-chat · Hugging Face. [https://huggingface.co/](https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat) [deepseek-ai/deepseek-moe-16b-chat](https://huggingface.co/deepseek-ai/deepseek-moe-16b-chat)
- <span id="page-12-10"></span>[5] 2024. deepseek-ai/DeepSeek-V2-Lite-Chat · Hugging Face. [https://huggingface.co/](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat) [deepseek-ai/DeepSeek-V2-Lite-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat)
- <span id="page-12-26"></span>[6] 2024. huggingface/text-generation-inference. [https://github.com/huggingface/](https://github.com/huggingface/text-generation-inference) [text-generation-inference](https://github.com/huggingface/text-generation-inference)
- <span id="page-12-11"></span>[7] 2024. Qwen/Qwen1.5-MoE-A2.7B-Chat · Hugging Face. [https://huggingface.co/](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat) [Qwen/Qwen1.5-MoE-A2.7B-Chat](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat)
- <span id="page-12-12"></span>[8] 2024. Qwen/Qwen2-57B-A14B-Instruct · Hugging Face. [https://huggingface.co/](https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct) [Qwen/Qwen2-57B-A14B-Instruct](https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct)
- <span id="page-12-29"></span>[9] Hicham Badri and Appu Shaji. 2023. Half-Quadratic Quantization of Large Machine Learning Models. [https://mobiusml.github.io/hqq\\_blog/](https://mobiusml.github.io/hqq_blog/)
- <span id="page-12-0"></span>[10] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems 33 (2020), 1877–1901.
- <span id="page-12-33"></span>[11] Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, and Tri Dao. 2023. Medusa: Simple Framework for Accelerating LLM Generation with Multiple Decoding Heads. [https://github.com/FasterDecoding/Medusa.](https://github.com/FasterDecoding/Medusa)
- <span id="page-12-2"></span>[12] Byung-Gon Chun, Sunghwan Ihm, Petros Maniatis, Mayur Naik, and Ashwin Patti. 2011. CloneCloud: elastic execution between mobile device and cloud. In Proceedings of the Sixth Conference on Computer Systems (Salzburg, Austria) (EuroSys '11). Association for Computing Machinery, New York, NY, USA, 301–314. <https://doi.org/10.1145/1966445.1966473>
- <span id="page-12-17"></span>[13] Marta R Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, et al. 2022. No language left behind: Scaling human-centered machine translation. arXiv preprint arXiv:2207.04672 (2022).
- <span id="page-12-25"></span>[14] Weihao Cui, Zhenhua Han, Lingji Ouyang, Yichuan Wang, Ningxin Zheng, Lingxiao Ma, Yuqing Yang, Fan Yang, Jilong Xue, Lili Qiu, Lidong Zhou, Quan Chen, Haisheng Tan, and Minyi Guo. 2023. Optimizing Dynamic Neural Networks with Brainstorm. 797–815. [https://www.usenix.org/conference/osdi23/presentation/](https://www.usenix.org/conference/osdi23/presentation/cui) [cui](https://www.usenix.org/conference/osdi23/presentation/cui)
- <span id="page-12-3"></span>[15] Damai Dai, Chengqi Deng, Chenggang Zhao, R. X. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y. K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang. 2024. DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models. <https://doi.org/10.48550/arXiv.2401.06066> arXiv[:2401.06066](https://arxiv.org/abs/2401.06066)
- <span id="page-12-34"></span>[16] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. <https://doi.org/10.48550/arXiv.2205.14135> arXiv[:2205.14135](https://arxiv.org/abs/2205.14135)
- <span id="page-12-4"></span>[17] DeepSeek-AI, Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Hao Yang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jin Chen, Jingyang Yuan, Junjie Qiu, Junxiao Song, Kai Dong, Kaige Gao, Kang Guan, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruizhe Pan, Runxin Xu, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou, Size Zheng, T. Wang, Tian Pei, Tian Yuan, Tianyu Sun, W. L. Xiao, Wangding Zeng, Wei An, Wen Liu, Wenfeng Liang, Wenjun Gao, Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaosha Chen, Xiaotao Nie, Xiaowen Sun, Xiaoxiang Wang, Xin Liu, Xin Xie, Xingkai Yu, Xinnan Song, Xinyi Zhou, Xinyu Yang, Xuan Lu, Xuecheng Su, Y. Wu, Y. K. Li, Y. X. Wei, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Li, Yaohui Wang, Yi Zheng, Yichao Zhang, Yiliang Xiong, Yilong Zhao, Ying He, Ying Tang, Yishi Piao, Yixin Dong, Yixuan Tan, Yiyuan Liu, Yongji Wang, Yongqiang Guo, Yuchen Zhu, Yuduan Wang, Yuheng Zou, Yukun Zha, Yunxian Ma, Yuting Yan, Yuxiang You, Yuxuan Liu, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhen Huang, Zhen Zhang, Zhenda Xie, Zhewen Hao, Zhihong Shao, Zhiniu Wen, Zhipeng Xu, Zhongyu Zhang, Zhuoshu Li, Zihan Wang, Zihui Gu, Zilin Li, and Ziwei Xie. 2024. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.<https://doi.org/10.48550/arXiv.2405.04434> arXiv[:2405.04434](https://arxiv.org/abs/2405.04434)
- <span id="page-12-8"></span>[18] Artyom Eliseev and Denis Mazur. 2023. Fast Inference of Mixture-of-Experts Language Models with Offloading. arXiv[:2312.17238](https://arxiv.org/abs/2312.17238) [cs.LG] [https://arxiv.org/](https://arxiv.org/abs/2312.17238) [abs/2312.17238](https://arxiv.org/abs/2312.17238)

- <span id="page-12-16"></span>[19] William Fedus, Barret Zoph, and Noam Shazeer. 2022. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research 23, 120 (2022), 1–39.
- <span id="page-12-31"></span>[20] Elias Frantar and Dan Alistarh. 2023. SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot. arXiv[:2301.00774](https://arxiv.org/abs/2301.00774) [http://arxiv.org/abs/2301.](http://arxiv.org/abs/2301.00774) [00774](http://arxiv.org/abs/2301.00774)
- <span id="page-12-23"></span>[21] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. 2023. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. <https://doi.org/10.48550/arXiv.2210.17323> arXiv[:2210.17323](https://arxiv.org/abs/2210.17323)
- <span id="page-12-18"></span>[22] Georgi Gerganov. 2024. ggerganov/llama.cpp. [https://github.com/ggerganov/](https://github.com/ggerganov/llama.cpp) [llama.cpp](https://github.com/ggerganov/llama.cpp) original-date: 2023-03-10T18:58:00Z.
- <span id="page-12-22"></span>[23] Sylvain Gugger, Lysandre Debut, Thomas Wolf, Philipp Schmid, Zachary Mueller, Sourab Mangrulkar, Marc Sun, and Benjamin Bossan. 2022. Accelerate: Training and inference at scale made simple, efficient and adaptable. [https://github.com/](https://github.com/huggingface/accelerate) [huggingface/accelerate.](https://github.com/huggingface/accelerate)
- <span id="page-12-19"></span>[24] Mingzhe Hao, Levent Toksoz, Nanqinqin Li, Edward Edberg Halim, Henry Hoffmann, and Haryadi S. Gunawi. 2020. {LinnOS}: Predictability on Unpredictable Flash Storage with a Light Neural Network. 173–190. [https://www.usenix.org/](https://www.usenix.org/conference/osdi20/presentation/hao) [conference/osdi20/presentation/hao](https://www.usenix.org/conference/osdi20/presentation/hao)
- <span id="page-12-27"></span>[25] Connor Holmes, Masahiro Tanaka, Michael Wyatt, Ammar Ahmad Awan, Jeff Rasley, Samyam Rajbhandari, Reza Yazdani Aminabadi, Heyang Qin, Arash Bakhtiari, Lev Kurilenko, and Yuxiong He. 2024. DeepSpeed-FastGen: Highthroughput Text Generation for LLMs via MII and DeepSpeed-Inference. [https:](https://doi.org/10.48550/arXiv.2401.08671) [//doi.org/10.48550/arXiv.2401.08671](https://doi.org/10.48550/arXiv.2401.08671) arXiv[:2401.08671](https://arxiv.org/abs/2401.08671)
- <span id="page-12-24"></span>[26] Ranggi Hwang, Jianyu Wei, Shijie Cao, Changho Hwang, Xiaohu Tang, Ting Cao, and Mao Yang. 2024. Pre-gated MoE: An Algorithm-System Co-Design for Fast and Scalable Mixture-of-Expert Inference. In 2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA). 1018–1031. [https:](https://doi.org/10.1109/ISCA59077.2024.00078) [//doi.org/10.1109/ISCA59077.2024.00078](https://doi.org/10.1109/ISCA59077.2024.00078)
- <span id="page-12-5"></span>[27] Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, and Geoffrey E. Hinton. 1991. Adaptive Mixtures of Local Experts. 3, 1 (1991), 79–87. [https://doi.org/10.](https://doi.org/10.1162/neco.1991.3.1.79) [1162/neco.1991.3.1.79](https://doi.org/10.1162/neco.1991.3.1.79) Conference Name: Neural Computation.
- <span id="page-12-6"></span>[28] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. 2024. Mixtral of experts. arXiv preprint arXiv:2401.04088 (2024).
- <span id="page-12-1"></span>[29] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling Laws for Neural Language Models. [https://doi.org/10.48550/arXiv.2001.](https://doi.org/10.48550/arXiv.2001.08361) [08361](https://doi.org/10.48550/arXiv.2001.08361) arXiv[:2001.08361](https://arxiv.org/abs/2001.08361)
- <span id="page-12-7"></span>[30] Rui Kong, Yuanchun Li, Qingtian Feng, Weijun Wang, Xiaozhou Ye, Ye Ouyang, Linghe Kong, and Yunxin Liu. 2024. SwapMoE: Serving Off-the-shelf MoE-based Large Language Models with Tunable Memory Budget. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 6710–6720.
- <span id="page-12-20"></span>[31] Tim Kraska, Alex Beutel, Ed H. Chi, Jeff Dean, and Neoklis Polyzotis. 2018. The Case for Learned Index Structures.<https://arxiv.org/abs/1712.01208>
- <span id="page-12-28"></span>[32] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient Memory Management for Large Language Model Serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles (New York, NY, USA, 2023-10-23) (SOSP '23). Association for Computing Machinery, 611–626. <https://doi.org/10.1145/3600006.3613165>
- <span id="page-12-14"></span>[33] Wonbeom Lee, Jungi Lee, Junghwan Seo, and Jaewoong Sim. 2024. InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24). USENIX Association, Santa Clara, CA, 155–172. <https://www.usenix.org/conference/osdi24/presentation/lee>
- <span id="page-12-35"></span>[34] Yaniv Leviathan, Matan Kalman, and Yossi Matias. 2023. Fast Inference from Transformers via Speculative Decoding. [https://doi.org/10.48550/arXiv.2211.](https://doi.org/10.48550/arXiv.2211.17192) [17192](https://doi.org/10.48550/arXiv.2211.17192) arXiv[:2211.17192](https://arxiv.org/abs/2211.17192)
- <span id="page-12-30"></span>[35] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. 2023. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. <https://doi.org/10.48550/arXiv.2306.00978> arXiv[:2306.00978](https://arxiv.org/abs/2306.00978)
- <span id="page-12-15"></span>[36] Zichang Liu, Jue Wang, Tri Dao, Tianyi Zhou, Binhang Yuan, Zhao Song, Anshumali Shrivastava, Ce Zhang, Yuandong Tian, Christopher Re, and Beidi Chen. 2023. Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time. In Proceedings of the 40th International Conference on Machine Learning (2023-07-03). PMLR, 22137–22176.<https://proceedings.mlr.press/v202/liu23am.html> ISSN: 2640-3498.
- <span id="page-12-36"></span>[37] Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, Chunan Shi, Zhuoming Chen, Daiyaan Arfeen, Reyna Abhyankar, and Zhihao Jia. 2024. SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification. In Proceedings of the 29th

- ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3 (La Jolla, CA, USA) (ASPLOS '24). Association for Computing Machinery, New York, NY, USA, 932–949. [https:](https://doi.org/10.1145/3620666.3651335) [//doi.org/10.1145/3620666.3651335](https://doi.org/10.1145/3620666.3651335)
- <span id="page-13-0"></span>[38] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2024. Training language models to follow instructions with human feedback. In Proceedings of the 36th International Conference on Neural Information Processing Systems (New Orleans, LA, USA) (NIPS '22). Curran Associates Inc., Red Hook, NY, USA, Article 2011, 15 pages.
- <span id="page-13-7"></span>[39] Noam M. Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc V. Le, Geoffrey E. Hinton, and Jeff Dean. 2017. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ArXiv abs/1701.06538 (2017). <https://api.semanticscholar.org/CorpusID:12462234>
- <span id="page-13-11"></span>[40] Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Daniel Y. Fu, Zhiqiang Xie, Beidi Chen, Clark Barrett, Joseph E. Gonzalez, Percy Liang, Christopher Ré, Ion Stoica, and Ce Zhang. 2023. FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU. [https:](https://doi.org/10.48550/arXiv.2303.06865) [//doi.org/10.48550/arXiv.2303.06865](https://doi.org/10.48550/arXiv.2303.06865) arXiv[:2303.06865](https://arxiv.org/abs/2303.06865)
- <span id="page-13-3"></span>[41] Yixin Song, Zeyu Mi, Haotong Xie, and Haibo Chen. 2023. PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU. [https://arxiv.org/abs/2312.](https://arxiv.org/abs/2312.12456v1) [12456v1](https://arxiv.org/abs/2312.12456v1)
- <span id="page-13-12"></span>[42] Mingjie Sun, Zhuang Liu, Anna Bair, and J. Zico Kolter. 2023. A Simple and Effective Pruning Approach for Large Language Models. arXiv[:2306.11695](https://arxiv.org/abs/2306.11695) <http://arxiv.org/abs/2306.11695>
- <span id="page-13-9"></span>[43] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford Alpaca: An Instruction-following LLaMA model. [https://github.com/tatsu-lab/stanford\\_](https://github.com/tatsu-lab/stanford_alpaca) [alpaca.](https://github.com/tatsu-lab/stanford_alpaca)
- <span id="page-13-1"></span>[44] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 (2023).
- <span id="page-13-6"></span>[45] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest,

- and Alexander M. Rush. 2020. Transformers: State-of-the-Art Natural Language Processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations. Association for Computational Linguistics, Online, 38–45.<https://www.aclweb.org/anthology/2020.emnlp-demos.6>
- <span id="page-13-13"></span>[46] Haojun Xia, Zhen Zheng, Yuchao Li, Donglin Zhuang, Zhongzhu Zhou, Xiafei Qiu, Yong Li, Wei Lin, and Shuaiwen Leon Song. 2023. Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity. arXiv[:2309.10285](https://arxiv.org/abs/2309.10285) [cs.DC]
- <span id="page-13-8"></span>[47] Fuzhao Xue, Zian Zheng, Yao Fu, Jinjie Ni, Zangwei Zheng, Wangchunshu Zhou, and Yang You. 2024. OpenMoE: An Early Effort on Open Mixture-of-Experts Language Models. In Forty-first International Conference on Machine Learning.
- <span id="page-13-10"></span>[48] Leyang Xue, Yao Fu, Zhan Lu, Luo Mai, and Mahesh Marina. 2024. MoE-Infinity: Offloading-Efficient MoE Model Serving. [https://doi.org/10.48550/arXiv.2401.](https://doi.org/10.48550/arXiv.2401.14361) [14361](https://doi.org/10.48550/arXiv.2401.14361) arXiv[:2401.14361](https://arxiv.org/abs/2401.14361)
- <span id="page-13-5"></span>[49] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. 2024. Qwen2 Technical Report.<https://doi.org/10.48550/arXiv.2407.10671> arXiv[:2407.10671](https://arxiv.org/abs/2407.10671) version: 4.
- <span id="page-13-4"></span>[50] Rongjie Yi, Liwei Guo, Shiyun Wei, Ao Zhou, Shangguang Wang, and Mengwei Xu. 2023. EdgeMoE: Fast On-Device Inference of MoE-based Large Language Models.<https://doi.org/10.48550/arXiv.2308.14352> arXiv[:2308.14352](https://arxiv.org/abs/2308.14352)
- <span id="page-13-14"></span>[51] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. 2022. Orca: A Distributed Serving System for Transformer-Based Generative Models. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22). USENIX Association, Carlsbad, CA, 521–538. <https://www.usenix.org/conference/osdi22/presentation/yu>
- <span id="page-13-2"></span>[52] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068 (2022).