## eMoE: Task-aware Memory Efficient Mixture-of-Experts-Based (MoE) Model Inference

Suraiya Tairin<sup>∗</sup> *,* Shohaib Mahmud<sup>∗</sup> *,* Haiying Shen *University of Virginia*

Anand Iyer *Georgia Institute of Technology*

## Abstract

In recent years, Mixture-of-Experts (MoE) has emerged as an effective approach for enhancing the capacity of deep neural network (DNN) with sub-linear computational costs. However, storing all experts on GPUs incurs significant memory overhead, increasing the monetary cost of MoE-based inference. To address this, we propose eMoE, a memoryefficient inference system for MoE-based large language models (LLMs) by leveraging our observations from experiment measurements. eMoE reduces memory usage by predicting and loading only the required experts based on recurrent patterns in expert routing. To reduce loading latency while maintaining accuracy, as we found using the same experts for subsequent prompts has minimal impact on perplexity, eMoE invokes the expert predictor every few prompts rather than for each prompt. In addition, it skips predictions for tasks less sensitive to routing accuracy. Finally, it has task-aware scheduling to minimize inference latency by considering Service Level Objectives (SLOs), task-specific output lengths, and expert loading latencies. Experimental results show that compared to existing systems, eMoE reduces memory consumption by up to 80% while maintaining accuracy and reduces inference latency by up to 17%. It also enables processing prompts 40× longer, batches 4.5× larger, and achieves 1.5× higher throughput.

## 1 Introduction

A recent trend in Deep learning (DL) has been increasing the model performance by increasing the number of model parameters [\[1,](#page-12-0) [2\]](#page-12-1). However, the linear relationship between the computational cost and the model size remains as a crucial impediment to the practical applications of the large models. To circumvent this bottleneck, sparse architectures [\[3–](#page-12-2)[8\]](#page-12-3) have been introduced. Among these, the Mixture-of-Experts (MoE) stands out as one of the most prominent sparse models. Its adoption has substantially amplified the scalability of DNN

models across various DL applications, encompassing natural language processing [\[9–](#page-12-4)[11\]](#page-12-5), computer vision [\[12\]](#page-12-6), speech recognition [\[13,](#page-12-7) [14\]](#page-12-8), and recommendation systems [\[15\]](#page-12-9).

An MoE layer comprises a gate and a collection of experts. The gate selects only a small number (e.g., 1 or 2) of experts for computation based on each input. Its sparse expert activation enables considerable expansion of model size by several orders of magnitude without a commensurate increase in computational requirements (FLOPs). However, the scaling of such models demands an excessively large amount of scarce GPU memory [\[16\]](#page-12-10). This directly corresponds to substantial monetary expenses when implementing MoE-based inference systems at scale. Our data analysis, detailed in [§2,](#page-1-0) reveals that an MoE-based model typically consumes 4x to 14x more memory compared to its dense counterpart. Due to the intricate dependence between model capacity and memory requirements, designing a memory-optimized MoE inference system is crucial for serving very large models. Existing state-of-the-art inference systems such as vLLM [\[16\]](#page-12-10), DeepSpeed-FastGEN [\[17\]](#page-12-11), Sarathi-Serve [\[18\]](#page-13-0), DistServe [\[19\]](#page-13-1) are designed for the traditional transformer architecture and thus unequipped to provide memory-efficient MoE inference.

In this work, we propose eMoE, a memory-efficient inference system for MoE-based large language models (LLMs). eMoE is designed by leveraging our observations from experiment measurements. eMoE integrates several components working in unison:

- Expert Prediction. We observe a recurring pattern in tokento-expert routing, where certain experts process more tokens than others within a given time period (i.e., popular experts). Based on this observation, we propose an on-demand loading of popular experts during inference to reduce memory footprint. By leveraging the prior distribution of expert selection, eMoE predicts and proactively loads the experts for future inputs.
- Periodic Expert Invocation. However, we found that timeefficient on-demand expert loading is challenging due to the increased inference latency. We observe that subsequent prompts are highly correlated, and using the same experts for

<sup>\*</sup>Equal contribution.

subsequent prompts does not significantly affect perplexity. To reduce expert loading overhead while maintaining accuracy, eMoE invokes the expert predictor every few prompts rather than for each prompt.

- Task-aware Expert Loading. Additionally, our data analysis reveals that an MoE-based model can produce the same output with inaccurate experts for certain tasks as it does with the correct experts. Leveraging this observation, eMoE skips predictions for tasks less sensitive to token-to-expert-routing accuracy.
- Task-aware Request Scheduling. Furthermore, leveraging the above observation with another observation that certain tasks generate significantly fewer tokens than others, eMoE uses profiled task-specific token generation length, task-aware expert loading latency and user-imposed inference latency Service Level Objective (SLO) in request scheduling with an objective to minimal impact on the end-to-end inference latency. It offers an advantage over existing request schedulers [\[16,](#page-12-10) [18–](#page-13-0)[20\]](#page-13-2) by considering expert loading latency.

Unlike previous methods that rely on continuous expert prefetching during inference [\[21–](#page-13-3)[23\]](#page-13-4), eMoE periodically loads and reuses experts, reducing the inference latency associated with these approaches. The limitations of these methods are discussed in [§2.2.](#page-2-0) Our real experimental evaluations show that compared to existing systems, eMoE reduces memory consumption by up to 80% while maintaining accuracy and reduces inference latency by up to 17%. It also enables processing prompts 40× longer, batches 4.5× larger, and achieves 1.5× higher throughput.

In summary, we make the following contributions:

- •We systematically analyze token-to-expert recurrence patterns and task-specific characteristics, such as tolerance for less accurate experts and varying output lengths. Our findings reveal that existing solutions for memory-efficient MoE inference are insufficient in addressing both memory optimization and inference latency effectively.
- •We propose eMoE, a memory-efficient inference system that uses expert prediction to load experts, incorporates periodic expert invocation, employs task-aware expert loading to minimize latency for less routing-sensitive tasks, and utilizes task-aware request scheduling to optimize both inference latency and memory consumption.
- •We evaluate eMoE against state-of-the-art transformer-based inference systems, demonstrating significant reductions in memory consumption while preserving accuracy and minimizing end-to-end inference latency.

## <span id="page-1-0"></span>2 Observation and Motivation

## <span id="page-1-2"></span>2.1 Experiment Settings

We conducted an analysis experiment on four popular publicly available MoE based LLMs: Mixtral-8x7B [\[24\]](#page-13-5), Mixtral-8x22B, TinyMixtral [\[24\]](#page-13-5) and OpenMoE [\[25\]](#page-13-6). We used various

Table 1: Dataset summary.

<span id="page-1-1"></span>

| Dataset           | Contents                | Task                     |  |
|-------------------|-------------------------|--------------------------|--|
| XSUM [26]         | News articles           | Summarize (SUM)          |  |
| Tweet             | Tweets<br>related<br>to | Classify tweet sentiment |  |
| sentiment [27]    | financial<br>markets,   | (CLSFY)                  |  |
|                   | stocks, and economic    |                          |  |
|                   | discussions             |                          |  |
| SQUAD [28]        | Context and question    | Question answer (QA)     |  |
|                   | pairs                   |                          |  |
| Headlines similar | Pairs of headlines on   | Compare semantic simi    |  |
| ity [29]          | same topics             | larity (COMP)            |  |
| Human-assistant   | Pairs of prompts and    | Conversation (CONV)      |  |
| conversation [30] | responses               |                          |  |
|                   |                         |                          |  |

popular natural language processing (NLP) tasks including text summarization, question answering (QA), semantic similarity comparison and semantic classification. Details of the datasets used for these tasks are summarized in Table [1.](#page-1-1) Following [\[20\]](#page-13-2) and [\[16\]](#page-12-10), we generated synthetic client request traces since there is no publicly available request trace for generative MoE-based language models. The generated trace follows the Poisson distribution. To generate a set of requests, first, we drew the number of requests from the Poisson distribution. Next, for each arrival of requests, we generate tasks following a multinomial distribution, where tasks have a uniform probability. We conducted our experiment on a computer equipped with Intel Xeon processor with 128GB host memory and four Nvidia A100 Tensor core GPU with 40GB device memory. We set the maximum number of generated tokens to 1000 for all the models to avoid out-of-memory errors. For accuracy measurement, we consider the output of an MoE-based model with all experts as the ground truth. We measured the semantic similarity between an output and ground truth as the accuracy using a pre-trained BERT model [\[2\]](#page-12-1).

#### 2.1.1 High Inference Latency with Dynamic Expert Loading During Inference

To reduce the memory requirements of MoE models, an alternative strategy involves keeping all experts on the CPU and dynamically transferring the router-selected experts to GPUs during inference, instead of pre-loading all experts onto the GPU. However, during inference, the communication between the CPU and GPUs for transferring the required experts introduces additional time overhead. To assess this trade-off, we conducted experiments comparing the inference time for the two cases. Figure [1\(](#page-2-1)a) shows that the inference time is significantly higher in dynamic loading than in static pre-loading. It is 3.2x higher for OpenMoE, and 5x higher for Mixtral-8x7B. Figure [1\(](#page-2-1)b) exhibits the corresponding memory consumption for each case. The memory consumption of dynamic loading is lower than the static pre-loading. It is 1.5x lower for OpenMoE, and 1.4x lower for Mixtral. The average number of active experts is around 67% and 78% for OpenMoE and Mixtral respectively. Further analysis reveals

that the average time to transfer experts from the CPU to the GPUs is  $\sim$ 4.431 seconds and  $\sim$ 12.744 seconds for OpenMoE and Mixtral, which is 2.2x and 3.6x of the inference time of OpenMoE and Mixtral, respectively. These findings highlight a critical trade-off: while dynamic loading reduces memory consumption, it significantly increases inference latency. As a result, instant loading of experts during inference is not a practical solution for memory-efficient MoE inference.

<span id="page-2-1"></span>![](_page_2_Figure_1.jpeg)

Figure 1: Inference time with memory consumption for dynamic expert loading during inference.

<span id="page-2-4"></span>**Observation 1** Instant loading of the required experts during inference is not a feasible solution to reduce the memory consumption of MoE models as it significantly increases inference latency.

<span id="page-2-2"></span>![](_page_2_Figure_4.jpeg)

Figure 2: Inference time with memory consumption for different approaches.

#### <span id="page-2-0"></span>2.2 Issues with Existing Approaches

Some existing approaches, like Pre-gatedMoE [21], MoE-Infinity [22], and EdgeMoE [23], reduce communication overhead by overlapping computation and data transfer, prefetching experts for the next layer during the current layer's computation into the GPU. To assess the impact of these techniques, we evaluated Pre-gatedMoE against dynamic loading, static pre-loading (all experts on GPU), and our proposed eMoE, which predicts, periodically loads, and reuses experts. Figure 2 shows that Pre-gatedMoE reduces latency by 1.2x-1.6x compared to dynamic loading but has 2.5x-3.5x higher latency than static pre-loading and eMoE. Real-time expert transfer between CPU and GPU can lead to memory and bandwidth contention, especially when the GPU is concurrently handling computations. This contention impacts both data transfer and processing speed, resulting in increased latency in Pre-gatedMoE. Additionally, it requires retraining the MoE

model with its gating mechanism, which is resource-intensive and time-consuming.

MoEInfinity and EdgeMoE both utilize expert prefetching to reduce memory usage but introduce complexity and overhead. MoEInfinity employs "group activation" to prefetch frequently co-activated experts but requires maintaining an expert activation matrix for each inference request, adding computational overhead. EdgeMoE is tailored for edge devices with limited GPU memory, making it less suitable for larger models or multi-GPU setups. Both methods increase inference latency and require either full MoE model training or creating additional data structures, complicating their use with larger models.

<span id="page-2-3"></span>![](_page_2_Figure_10.jpeg)

Figure 3: Expert activation transition patterns between consecutive layers.

#### 2.2.1 Repetitive Patterns in Expert Activation

To investigate the repetitive patterns in expert activation between consecutive layers in MoE model, we conducted experiments to track expert selection across layers. For each input, we recorded the pair of experts chosen at consecutive layers, compiling this data into a matrix where each entry at the position (i, j) represents the number of times Expert i at the current layer was followed by Expert j at the next layer. We use a heatmap to visualize these frequencies. Figure 3 presents the heatmap for summarization and question-answering tasks for Mixtral-8x7B. In the figure, darker colors (e.g., deep blue) indicate a higher frequency of transitions, while lighter colors (e.g., pale blue or white) indicate lower frequencies. The figure reveals strong, consistent patterns in expert activation between layers, suggesting an opportunity to predict future expert activations based on prior patterns. This insight highlights the potential for developing a predictive method to anticipate expert selection, which could significantly reduce memory usage.

#### 2.2.2 Inference Latency is Task Sensitive

The inference latency is directly proportional to the number of generated tokens. Therefore, the inference latency may vary depending on the task's characteristics. To test our hypothesis, we analyze a number of popular NLP tasks given in Table 1.

Figure 4 shows the Cumulative Distribution Frequency (CDF) of requests versus the number of generated tokens. The figure reveals variations in the lengths of generated outputs across different tasks. Specifically, the QA tasks yield the

<span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Figure 4: CDF of the number of generated tokens across different tasks

shortest output length, while the conversation tasks produce the longest output length. Notably, both models generate explanations in their generated texts for classification and comparison tasks, resulting in longer output lengths compared to the QA tasks. Additionally, the 90th percentile of the output length of the QA tasks is 17% and 26% smaller than that of the conversation tasks for the OpenMoE and Mixtral models, respectively. The rest of the tasks' 90th percentile output lengths follow this order: QA (Mixtral: 2.2 sec, OpenMoE: 1.8 sec) > Summarization (Mixtral: 1.8 sec, OpenMoE: 1.4 sec) > Semantic similarity measure (Mixtral: 1.4 sec, OpenMoE: 1.1 sec). The output length difference among the tasks is caused by their output characteristics. Therefore, we make the following observation

<span id="page-3-5"></span>**Observation 2** Different tasks exhibit different output length distributions. The length difference can be often as high as twice the output length of a certain task.

## <span id="page-3-4"></span>2.2.3 Tasks Show Varying Sensitivity To Tokens-toexpert Routing Accuracy

Existing work indicates that layers closer to the input in neural networks tend to learn general representations, while layers closer to the output specialize in task-specific representations [31,32]. Building upon this observation, we hypothesize that certain tasks may not require highly accurate representations in the layers closer to the input. This suggests that the inference system can prioritize experts selections for tasks that demand more accurate representation.

To ascertain the hypothesis, we conducted an analysis experiment. In this experiment, starting from the first MoE layer, we progressively made the MoE layers' routing inaccurate and measured the output accuracy. The results of this analysis experiment are shown in Figure 5, where the X-axis denotes the percentage of MoE layers where accurate token-to-expert routing was applied and Y-axis represents the similarity. We see that for the classification and compare tasks, both models maintain above 90% similarity with respect to the full model even when all tokens are routed incorrectly. We also note that open-ended tasks such as conversation and summarization

<span id="page-3-1"></span>![](_page_3_Figure_7.jpeg)

Figure 5: Accuracy with progressive applying of exact token-to-expert routing in MoE layers closer to output layer.

show low tolerance to inaccurate expert routing. Even with 75% accurate MoE layers, the similarity score falls below 80%. On the other hand, for the QA tasks which need to be more specific as the answer is in the context, the similarity score is above 80% for both models when only 50% of the MoE layers have accurate routing. For the summarization tasks, the accuracy falls below 80% when 50% of the MoE layers are accurate. We see that a task's tolerance to MoE layers' routing accuracy varies greatly. Therefore, we make the following observation:

<span id="page-3-3"></span>**Observation 3** Different tasks exhibit different tolerance to inaccurate MoE layers' expert routing. Tasks that are more open-ended such as conversation demand more accurate routing than tasks that are more specific such as semantic classification.

#### 3 System Design of eMoE

#### 3.1 Overview

<span id="page-3-2"></span>![](_page_3_Figure_13.jpeg)

Figure 6: Overview of eMoE.

Figure 6 illustrates the architecture of the eMoE inference system. The inference process starts with task type extraction. After extracting the task type, the prompts are scheduled by eMoE's task-aware request scheduler (①). The scheduler aims to minimize the inference latency by considering profiled task-specific token generation length, user-imposed latency SLO and expert loading latency. The scheduled requests are sent to the task-aware expert loading module (②). This module leverages tasks' sensitivity to token-to-expert routing accuracy and the expert prediction to selectively load experts to the inference engine to minimize expert loading latency. In addition to serving inference requests, the inference engine

also runs the expert prediction model based on recurrence patterns (③). To save the overhead for expert loading while maintaining accuracy, it periodically invokes expert prediction at regular intervals (④). Once expert loading is initiated, the inference engine starts serving the scheduled requests. In the following, we present each of these components.

#### 3.2 Expert Prediction

In the inference pipeline, we first identify the task type for each request for task-aware expert loading. Next, we predict the specific experts needed and load only predicted experts onto the GPU. We present task-type extraction and expert prediction methods in §3.2.1 and §3.2.2. The MoE inference process using the expert predictor is described in §3.3.

#### <span id="page-4-0"></span>3.2.1 Task Type Extraction

This task type extraction method identifies the task type associated with each individual user request. The method leverages the occurrence of certain task-specific keywords to determine a request's task type. A task is typically described by one or two sentences and task description for a task type is likely to have similar keywords. For example, a user may write "Summarize the following text" or "Write a synopsis for the following text" to request for a summarizing task. Upon receiving a user request, this method processes the input tokens and generates a potential task type as the output. Specifically, it searches for some profiled keywords in the sentences in the input. Each identified keyword is compared against the keywords from all task types. The task type that has the highest number of matches is assigned as the task type of the input. The method runs on an on-demand basis on CPU and therefore does not interfere with the inference system pipeline.

#### <span id="page-4-1"></span>3.2.2 Task-aware Expert Prediction

Observation 1 shows that instantly loading the required experts during inference in MoE models to reduce memory consumption increases inference latency. Using fewer experts than the model was trained with reduces memory consumption but compromises output quality [32]. Besides, the workload among experts varies dynamically during inference [12, 33]. An ideal solution should reduce memory consumption without harming the quality of model output or increasing inference latency. The most effective approach is to proactively determine the necessary experts for an inference request before processing it and load only those experts into the GPU's memory. Building on these insights, we propose an expert prediction method to reduce memory consumption while maintaining model output quality and inference latency.

We employ a machine learning (ML) based method to predict the experts. This predictor forecasts the future experts needed for an incoming request based on the prior distribution of experts. The experts are represented by numerical digits. For instance, if there are *m* MoE layers in an MoE model,

and if we denote the expert sequence for layer i as  $e_i$ , the sequence of the expert index for m layers can be represented as follows:  $e_1, e_2, ... e_i, ..., e_m$ . Each  $e_i$  represents a series of k indices for top-k routing. This expert prediction can be seen as a sequence prediction task, where elements in the sequence exhibit certain correlations or patterns. Typically, future elements in a sequence depend on the past elements. In sequence or series prediction, the output is a distribution of probabilities over possible outcomes, representing the relative frequency of each possible next element, rather than a single fixed value. The element with the highest probability is considered as the next element in the sequence.

Correlation in Expert Sequence. We analyze expert selection correlation and use cross-correlation [34] to measure the correlation between experts chosen in one layer and those chosen in the next layer. For a top-k routing, we consider the k chosen experts in layer i as a sequence  $e_i$  and the k chosen experts in layer i+1 as a sequence  $e_{i+1}$ , and compute the correlation between them. We compute the average cross-correlation of selected experts from one layer to the next for 1000 prompts using the settings in §2.1. Figures 7a and 7b show the correlation for each layer. For both Open-MoE and Mixtral-8x7B, we observe a significant correlation of around 0.50 between the experts selected by consecutive layers. This is because the gate network uses basic features such as parts of speech and word meaning to select experts, resulting in similar tokens being handled by similar experts across layers [33]. This consistent selection process leads to a high degree of correlation between the experts chosen in consecutive layers. We leverage this correlation to estimate the expert in the next layer using an ML model.

<span id="page-4-2"></span>![](_page_4_Figure_10.jpeg)

Figure 7: Correlation in expert sequence between consecutive layers.

We then calculate the cross-correlation to examine the relationship between experts selected for consecutive prompts. Figures 8a and 8b show the correlation coefficient between each prompt and the following prompt. In OpenMoE and Mixtral-8x7B, the correlations range from 0.4 to 0.6 and from 0.75 to 0.95, respectively, indicating a strong correlation in expert selection between prompts. There exists shared context or vocabulary across consecutive prompts. This shared information results in a high correlation in expert selection between consecutive prompts.

**Preparing Expert Sequence for Prediction.** Motivated by the above observations, we predict the future expert sequence based on past expert sequences. For an incoming prompt, be-

<span id="page-5-1"></span>![](_page_5_Figure_0.jpeg)

Figure 8: Correlation in expert sequence between consecutive prompts.

fore processing it, we must determine the expert sequence (i.e., the experts in each layer) which is the future expert sequence for all its tokens. For this future expert sequence, the past expert sequence would be the expert sequence of previous layers. We consider two cases to predict the expert sequence: (1) predicting future expert sequence for one layer at a time using the past layer's expert sequence (layer-by-layer prediction), denoted as eMoE-L, and (2) predicting future expert sequence for all the layers at once using the past prompt's expert distribution (all-layer prediction), denoted as eMoE-A. Suppose there is an incoming inference request  $r_1$ . For case (1), let the predicted expert of the *i*-th layer for prompt  $r_1$  be denoted as  $e_i^{r_1}$ . Then the prediction method f can be represented as follows:  $e_i^{r_1} = f(e_{i-1}^{r_1})$ . To predict the experts for the first layer, we use the experts of the first layer of the previous prompt.

For case (2), suppose there are two inference requests  $r_1$  and  $r_2$ . We want to predict the expert sequence of  $r_2$  using the expert selection of  $r_1$ . The experts selected by the router for all the layers for  $r_1$  are denoted as  $e_1^{r_1}, e_2^{r_1}, ..., e_m^{r_1}$ , where the model has m layers. Let's assume the predicted expert sequence for all layers of  $r_2$  is  $e_1^{r_2}, e_2^{r_2}, ..., e_m^{r_2}$ . Then, the prediction method f can be presented as follows:

$$e_1^{r_2}, e_2^{r_2}, ..., e_m^{r_2} = f(e_1^{r_1}, e_2^{r_1}, ..., e_m^{r_1}).$$
 (1)

Expert Predictor Selection. Selecting a suitable ML model for sequence prediction is crucial for effectively learning relationships between past and future sequences. We choose a transformer-based model given its ability to capture and maintain dependencies in very long sequences using the attention mechanism [35]. In contrast, other models for sequence prediction, such as LSTM [36], struggle to capture dependencies in long sequences that are far apart, as they process sequences sequentially and suffer from vanishing gradient problems in very long sequences [37]. Transformer-based models often perform comparably or better than traditional sequence prediction models [38]. We use the GPU to run our expert predictor to minimize latency, as it consumes very little GPU memory. Our empirical results in §5.3 show that the predictor typically uses only 0.24%-1.3% of the MoE model's memory. When loading predicted experts, we compare them with those already loaded in the GPU. New experts are identified and loaded into the GPU, while experts not in the prediction are moved to the CPU. Experts are loaded

according to a memory budget, prioritizing those with the higher workload, which is determined by the total number of tokens they receive. If the expert selected for a token is not on the GPU, the token is routed to the next top-k expert that is on the GPU.

### <span id="page-5-0"></span>3.3 Periodic Expert Invocation

Calling the expert predictor for each inference request adds latency to the inference time (shown in §5.7), which is undesirable. Therefore, calling the predictor should be strategized to avoid harming the inference latency performance of the MoE inference system. This means the expert predictor should be called opportunistically, not for every inference request. One straightforward approach is to invoke the expert predictor at regular or adaptive intervals, rather than for every inference request. The predicted experts can then be used for several consecutive requests without needing to re-invoke the expert predictor for each one. We assume that there is a level of stickiness or continuity among subsequent incoming inference requests. For example, in a conversation, there is often a logical flow or sequence to the topics being discussed. Therefore, subsequent inference requests are often related to the previous ones.

To observe the correlation between subsequent prompts, we conduct an experiment using the settings in §2.1. For each prompt, we compare its correlation [34] with the

<span id="page-5-2"></span>![](_page_5_Figure_10.jpeg)

Figure 9: Correlation in consecutive prompts.

subsequent prompt. Specifically, for a prompt, we consider pairs with the i-th subsequent prompt ( $i = 1, 2, \ldots, 20$ ) and measure the correlation of each pair. Figure 9 illustrates the average correlation between a prompt and each subsequent prompt. We observe that the correlation is significantly high, ranging from 0.48 to 0.55. This is due to the shared context or common vocabulary. When prompts are contextually related or use similar terms, their contents become more similar, leading to higher correlation. Words and phrases that frequently appear together or within the same context results in stronger correlations.

To investigate whether experts chosen for one prompt can be reused for subsequent prompts, we conducted an experiment using the settings described in §2.1. In this experiment, we used the same experts selected for a prompt for its next p prompts, with p values of 0, 20, 40, 60, and 80. p=0 means the same experts are not reused for any subsequent prompts. Figure 10 shows the average perplexity for different numbers of subsequent prompts. We observed that the average perplexity remains nearly the same for 20 and 40 prompts but starts increasing from 60 onwards. This is because the prompts are highly correlated, and using the same experts for some

subsequent prompts does not significantly impact perplexity.

<span id="page-6-0"></span>![](_page_6_Figure_1.jpeg)

Figure 10: Perplexity using same experts.

Motivated by the above observations, we consider that the expert predictor does not need to be invoked for every inference request. Once the expert predictor is called and the predicted experts are loaded, they can be reused for multiple subsequent inference requests in the future. Therefore, in our design, we consider calling the expert predictor at regular intervals (i.e., after a certain number of prompts). In our inference system, we maintain an index of the incoming requests as 0, 1, 2, and so on. Based on this index, we invoke the expert predictor after every p prompts. For each incoming request, the system first checks if it is the p-th prompt. If not, the system processes the prompt using the experts already loaded on the GPU without invoking the predictor. Otherwise, the system calls the expert predictor, loads the predicted experts onto the GPU, and then processes the prompt using these newly loaded experts. For the first prompt, the system loads all the experts onto the GPU, and subsequently, after every p prompts, it invokes the predictor and loads only the predicted experts.

### 3.4 Task-aware Expert Loading

Running MoE-based models with a lower number of experts loaded into the GPU memory than the model originally trained can degrade the quality of the model output. However, our Observation 3 indicates that certain tasks show greater sensitivity to token-to-expert routing accuracy while some tasks show little sensitivity to this accuracy. In the context of expert loading for a particular layer, we denote a task as either sensitive or insensitive. eMoE has task-aware expert loading, which only considers the predicted experts of the sensitive tasks during the expert loading to reduce the expert loading latency while incurring minimal compromise on the accuracy of the model output.

eMoE runs its task-aware expert loader each time one or more new input requests are scheduled. As detailed in § 3.2.2, the expert prediction module outputs the relative frequency with which tokens in a task type are likely to be routed to an expert. eMoE leverages this task type-based expert prediction to prioritize tasks that are more sensitive to token-to-expert routing accuracy. We define a task type's sensitivity to routing accuracy on an MoE layer basis. This sensitivity is evaluated through offline profiling, where, for each task type, we

progressively apply random token-to-expert routing, starting from the MoE layer closest to the input. We then record the accuracy drop as we progressively use less accurate routing. The details are in § 2.2.3.

The MoE layers with inaccurate routing that yield output with accuracy higher than a threshold (e.g., 85%) are marked as insensitive to the particular task. In each iteration of expert loading, eMoE first computes the expected number of tokens per expert per task type and assembles the computed data in a 3D array as shown in Figure 11. We obtain the expected number of tokens per ex-

<span id="page-6-1"></span>![](_page_6_Figure_9.jpeg)

Figure 11: eMoE maintains the expected number of tokens per expert for each task type.

pert with an objective to maximize the number of correct token-to-expert routing. We hypothesize that maximizing the number of correctly routed tokens per expert will ensure the least impact on the accuracy. The rationale behind such hypothesis is that if we reduce the error in token-to-expert routing for a given number of experts, the accuracy drop will decrease.

Let  $N_i$  be the expected number of tokens that will be routed to expert i in an arbitrary MoE layer for a given task type. Also, let  $s \in \{0,1\}$  be a binary variable to indicate whether a token is sensitive to token-to-expert routing accuracy of the task type to the particular MoE layer,  $f_i$  be predicted token-to-expert routing frequency, and T be the number of requests belonging to the given task type that are already running. We compute the value of  $N_i$  using the following equation:

<span id="page-6-2"></span>
$$N_i = \left(\sum_{j=0}^{T-1} W_j + T \cdot W_o\right) \cdot s \cdot f_i \tag{2}$$

where  $W_n$  denotes the number of input tokens in the *n*th request and  $W_o$  denotes the expected number of generated tokens

eMoE utilizes Equation (2) to determine the expected number of tokens per expert for a given set of requests that are to be scheduled and the requests that are already running by the inference engine. eMoE obtains the total expected number of tokens per expert by summing  $N_i$ s belonging to all task types. Next, eMoE sorts the experts at each MoE layer based on the expected number of tokens likely to be routed in descending order. Subsequently, eMoE picks the top L experts for every layer, where L is set based on the memory budget. In the MoE architecture where the number of experts across MoE layers is different, different L values are chosen for each layer. The values of L is set by the system user. Finally, eMoE identifies the experts that are already loaded into the inference

engine. The identified experts are then set to be loaded into the inference engine in the next dispatching of the inference requests.

## 3.5 Task-aware Request Scheduling

eMoE's request scheduler aims to minimize the impact on inference latency while ensuring compliance with the SLOs of running requests. Under such setting, the existing scheduling systems (e.g., [\[16–](#page-12-10)[19\]](#page-13-1)) which neglect expert loading cannot optimally meet the latency SLO in eMoE. Furthermore the existing systems do not exploit the Observations [2,](#page-3-5) [3](#page-3-3) which present some unique opportunities in terms of minimal impact on the inference latency. First, tasks' varying sensitivity to token-to-expert routing accuracy suggests that this phenomenon should be exploited to limit expert loading latency. Second, tasks have inherent latency requirements, leading to varying computational demands and differing impacts on the latency of running requests. This is because scheduling a request may significantly increase the latency of the existing running requests due to the expert loading and a large number of input tokens. Therefore, it will be beneficial to delay the incoming request if the request has relaxed latency SLO requirement and the existing running requests are likely to complete the token generation in short period of time. eMoE's request scheduler jointly considers user imposed latency SLOs, taskspecific profiled generation length, and task-specific profiled sensitivity to token-to-expert routing accuracy to minimize the end-to-end inference latency.

Scheduling a new batch of requests introduces an initial upfront latency cost due to a large increase of the number of tokens per MoE layer in the model. This increase in the number of tokens equals to the number of input tokens. The increase in input size results in higher computation latency and increased all-to-all communication among the experts in MoE layer and thereby increasing the latency of the running requests.

Let *W* be the number of input tokens in a new request to be scheduled, *G<sup>i</sup>* be the number of tokens to be generated by the *i*th request, ∆*E* be the expert loading latency due to the scheduling of new requests and *c* be the average expert computation and communication latency for a single input token. We use task-specific profiling to obtain *G<sup>i</sup>* . When a request is scheduled, *G<sup>i</sup>* is set to the average number of tokens generated by the task the request belongs to. After each iteration of token generation, *G<sup>i</sup>* is decremented. If *G<sup>i</sup>* reaches 0 but the request is not completed, *G<sup>i</sup>* is reset to a value proportional (e.g., 5%) to its initial value. We also use a profiled value for ∆*E* and *c*. Also, let *n<sup>i</sup>* be the number of existing running requests that will complete generation after the *i*th request. We can express the expected latency of the *i*th request *t<sup>i</sup>* due to scheduling of a new request as follows:

$$t_i = \Delta E + (W + n_i \cdot G_i) \cdot c + r_i \tag{3}$$

where *r<sup>i</sup>* denotes the run-time of request *i* so far. For a new

request, *r<sup>i</sup>* is set to zero and incremented as it runs on the inference engine.

Algorithm 1: eMoE Request Scheduling Algorithm

```
Input: Waiting queue Qw, Scheduled queue Qs, Maximum
        number of tokens Tmax
1 T ← Qs.length ;
2 Qw ← Sort Qw by SLO stringiness
3 for R ∈ Qw do
4 if R.inputTokens+T < Tmax then
5 if S.NewExpectedLatency < S.SLO,∀S ∈ Qs :
          S.ExpectedLatency < S.SLO then
6 Qs ← Qs ∪ {S} ;
7 T ← T +R.inputTokens
8 end
9 end
10 end
11 return Qs
```

eMoE adopts a greedy approach in its iterative scheduling algorithm. Algorithm [1](#page-7-0) shows the pseudocode of eMoE's scheduling algorithm. At each iteration, eMoE picks the request with the most stringent latency SLO and attempts to schedule the request (*line 2-6*). We measure a request's SLO stringiness by how quickly the first response token will be generated. The algorithm first checks if scheduling the request will exceed the total number of tokens that the inference engine can process. If the total number of tokens exceeds the limit, the request is kept in the waiting queue (*line 2*). Otherwise, the request is scheduled if it will not lead to SLO violations of the requests already scheduled (*line 3*). The request scheduler runs whenever a new request arrives or a running request completes token generation.

## 4 Implementation

We implemented eMoE using the Python programming language. The expert prediction model is implemented using Pytorch [\[39\]](#page-14-1). The expert prediction model runs on a separate process besides the inference engine. We utilize DeepSpeed-FastGen [\[17\]](#page-12-11) as the inference engine. The inference engine wraps Huggingface [\[40\]](#page-14-2) models for inference serving.

We extended the Huggingface model codes to augment functionality for expert loading. Specifically, we maintained Python multiprocessing lock for each MoE layer and wrapped the MoE layer's computation inside the lock. The lock is used by the expert loader process to synchronize the expert loading operation with the MoE layer computation. Furthermore, We wrapped the expert loading of an MoE layer with a CUDA event, which the corresponding MoE layer synchronizes with. This prevents an MoE layer from using stale model weights.

eMoE asynchronously loads experts in an MoE layer form host to device via *torch*.*Tensor*.*copy*\_(*non*\_*blocking* = *True*) to overlap the loading process with the computation by the non-expert layers. We condition the loading of experts in an MoE layer contingent upon the completion of the loading process from the previous layer. We made such design decision to prevent the saturation of the PCIe channel bandwidth.

## 5 Performance Evaluation

In this section, we first evaluate eMoE's end-to-end latency performance in [§5.1](#page-8-1) which demonstrate the effectiveness of eMoE. We evaluate eMoE's impact on accuracy in [§5.2.](#page-8-2) Then we demonstrate the performance of expert predictor in [§5.3.](#page-8-0) We then evaluate eMoE's inference performance with expert predictor ([§5.4\)](#page-9-0) and show the impacts of reducing memory consumption ([§5.5\)](#page-9-1). Furthermore, we evaluate the sensitivity of eMoE on different parameters [§5.6.](#page-11-1) We present the time overhead of our methods in [§5.7.](#page-11-0) Unless otherwise specified, we use the same experiment setting as detailed in [§2.1.](#page-1-2) Our key results include:

- eMoE reduces inference latency by up to 17%.
- eMoE reduces memory consumption up to 80% while maintaining accuracy. By optimizing memory usage, it enables processing prompts 40× longer, batches 4.5× larger, and achieves 1.5× higher throughput (tokens/second).

## <span id="page-8-1"></span>5.1 Evaluation on End-to-End Inference Latency

Figure [12](#page-9-2) shows the end-to-end inference latency of eMoE, vLLM, and DeepspeedFastGen for OpenMoE, Mixtral-8x7, TinyMixtral, and Mixtral-8x22B models. We compared with these two methods because they outperform other approaches. In the figure, the X-axis denotes the arrival rate of request per second and the Y-axis denotes the average inference latency. In the experiment, we varied the throughput in a step 0.1 request/second and generate tokens until the end of sequence (EOS) token. We only show the results for higher throughput for the interest of clarity. For Mixtral-8x22B model, the results include throughput upto 0.4, which is the maximum throughput supported by our experiment setup.

From the figure, we see that for all four models, eMoE outperforms DeepspeedFastGen and vLLM. Among the compared methods, vLLM performs the worst. DeepspeedFast-Gen's superior performance over vLLM is due to its kernel level optimizations tailored to MoE based models. eMoE also achieves superior performance compared to vLLM as eMoE is implemented on top of DeepspeedFastGen.

From the figure we see that, eMoE consistently outperforms DeepspeedFastGen for a given throughput. The highest improvement (17%) comes with the Mixtral-8x22B model which is the largest among the four in terms of the number of parameters. On the contrary, the smallest improvement (9%) comes from TinyMixtral model which is the smallest among the four. We see that with eMoE performance improvement is larger with the larger model. This is due to larger model being

compute heavy and memory heavy. eMoE runs with fewer experts which allows it to reduce the compute latency and global GPU memory read operations. Therefore, with larger model eMoE performs better.

## <span id="page-8-2"></span>5.2 Evaluation on Inference Accuracy

Figure [13](#page-9-3) shows the average accuracy achieved by the proposed task-aware expert loading and the task-agnostic expert loading methods. In the task-agnostic loading methods, the expert prediction model's output for all the tasks in Table [1](#page-1-1) was utilized in the expert loading. In the figure, the X-axis denotes different tasks and Y-axis represents the similarity of the model's output with the partially loaded expert to the model's output to the fully loaded experts. We refer the output of the latter as the ground truth. To measure the similarity with respect to ground truth, we pass the two outputs to a pretrained BERT [\[2\]](#page-12-1) to measure the similarity. From the figure, we see that for all the tasks, eMoE performs better if not the same as the task-agnostic method. We see that for text classification and text similarity measure tasks, both methods perform almost the same. Recall that these two tasks are less sensitive to the accuracy of expert routing as indicated in [§2.2.3](#page-3-4) . However, for conversation, question-answer, and text summarizing tasks, we see that eMoE achieves better accuracy than the task-agnostic method.

## <span id="page-8-0"></span>5.3 Performance of Expert Predictor

To find a suitable model for expert prediction, we conducted experiments on different variants of BERT [\[2\]](#page-12-1) and GPT-2 [\[41\]](#page-14-3) models from Huggingface [\[40,](#page-14-2) [42\]](#page-14-4) and compare their results. We collected the expert routing data from the OpenMoE and Mixtral models, and use 80% of the data for training and 30% for testing. The results are shown in Table [2.](#page-8-3) The precision, recall and F1-score for Mixtral are 0.712, 0.713, 0.705 and for OpenMoE are 0.696,0.689, 0.680. We observe that BERT-XLNet performs better than other models. XLNet [\[43\]](#page-14-5) learns bidirectional contexts by maximizing the likelihood over all possible permutations of the input sequence, capturing more complex relationships between elements in the sequence, which leads to higher accuracy than other models. Therefore, we chose XLNet for our system.

<span id="page-8-3"></span>Table 2: Performance of the expert predictor.

| Models     | # of            | Accuracy | Accuracy | Routing |
|------------|-----------------|----------|----------|---------|
|            | parameters eMoE |          | eMoE     | data    |
|            |                 | L        | A        |         |
| GPT-2      | 0.115B          | 50%      | 49%      | OpenMoE |
| BERT-base  | 0.108B          | 13%      | 12%      | OpenMoE |
| BERT-XLNet | 0.108B          | 70%      | 69%      | OpenMoE |
| GPT-2      | 0.115B          | 51%      | 51%      | Mixtral |
| BERT-base  | 0.108B          | 16%      | 16%      | Mixtral |
| BERT-XLNet | 0.108B          | 71%      | 71%      | Mixtral |

<span id="page-9-2"></span>![](_page_9_Figure_0.jpeg)

Figure 12: End-to-End latency performance evaluations.

<span id="page-9-3"></span>![](_page_9_Figure_2.jpeg)

Figure 13: Accuracy with different tasks.

