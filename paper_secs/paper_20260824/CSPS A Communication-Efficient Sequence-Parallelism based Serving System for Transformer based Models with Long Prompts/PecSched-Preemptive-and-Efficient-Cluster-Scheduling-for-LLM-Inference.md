# PecSched: Preemptive and Efficient Cluster Scheduling for LLM Inference

ZEYU ZHANG, University of Virginia, USA HAIYING SHEN, University of Virginia, USA

The scaling of transformer-based Large Language Models (LLMs) has significantly expanded their context lengths, enabling applications where inputs exceed 100K tokens. Our analysis of a recent Azure LLM inference trace reveals a highly skewed long-tail distribution of input lengths, with approximately 80% of inputs shorter than 2K tokens. Long inputs constitute only a small fraction. Existing cluster-level LLM scheduling strategies, including First-In-First-Out (FIFO), reservation-based, and priority-based approaches, primarily target shortinput requests with lengths below 2K and fail to address this heterogeneity, leading to inefficiencies such as head-of-line blocking, resource underutilization, and starvation of long-input requests. We propose PecSched, a Preemptive and Efficient Cluster SCHEDuling system for LLM inference. PecSched introduces the following key techniques: 1) preemptive scheduling that prioritizes short-input requests for their performance; 2) coordinated prefill-decode colocation and disaggregation, which reduces both the duration and frequency of preemptions; 3) fast Sequence Parallelism (SP) that minimizes the prefill time of long-input requests to further reduce the likelihood and frequency of preemptions. Evaluations based on Azure LLM inference trace show that, compared to state-of-the-art cluster-level LLM inference schedulers, PecSched reduces the 99th percentile queueing delay of short-input requests by up to 92% and improves their throughput by up to 595%, without significantly affecting the Job Completion Time (JCT) of long-input requests. We open-sourced our code.

Additional Key Words and Phrases: Job scheduling, LLM inference, Long sequence

### 1 Introduction

Transformer-based generative Large Language Models (LLMs) have enabled a wide range of applications. Early use cases such as conversational chat typically involved short-input requests [\[1\]](#page-17-0), usually under 2K tokens. As LLMs scale to billions or trillions of parameters [\[2](#page-17-1)[–7\]](#page-17-2), their contextual understanding and ability to process longer, more complex inputs have improved, pushing supported input lengths to 4K-1M tokens [\[8](#page-17-3)[–12\]](#page-17-4). This expansion has unlocked new applications such as book summarization [\[13–](#page-17-5)[15\]](#page-17-6), document classification [\[16,](#page-17-7) [17\]](#page-17-8), and code generation [\[18\]](#page-17-9), where inputs can exceed 100K tokens. To accelerate the processing of such long requests, existing work [\[19\]](#page-17-10) employs Sequence Parallelism (SP) based on ring attention [\[20\]](#page-17-11). Our analysis of the recently released Azure LLM inference trace [\[21,](#page-17-12) [22\]](#page-17-13) by Microsoft in [§3](#page-2-0) shows that the input length distribution in today's LLM inference clusters is a highly skewed long-tail distribution: about 80% of requests have input lengths below 2K. As the input length increases, the frequency drops. Compared to input lengths, output lengths are much smaller, implying that input lengths dominate the execution time. Yet, state-of-the-art cluster-level LLM scheduling strategies [\[19,](#page-17-10) [23–](#page-17-14)[27\]](#page-18-0) largely ignore this heterogeneity between short-input and long-input requests.

Cluster-level scheduling in current LLM inference systems generally falls into three categories: First-In-First-Out (FIFO) [\[19,](#page-17-10) [23\]](#page-17-14), reservation-based [\[24\]](#page-17-15), and priority-based [\[25–](#page-17-16)[27\]](#page-18-0) strategies. However, these works primarily target short-input requests with input lengths below 2K. In clusters with both short-input and long-input requests, FIFO [\[19,](#page-17-10) [23\]](#page-17-14) can cause head-of-line blocking, where long inputs delay short ones. Our measurements (in [§3\)](#page-2-0) show that long inputs can increase the 99th percentile queueing delay of short-input requests by up to 10.2×. Reservation-based strategies [\[24\]](#page-17-15) have to pre-allocate sufficient GPUs for long-input requests, limiting the resources available to short inputs and leading to queueing. Since long-input requests are rare, the reserved GPUs often sit idle. Our measurements in [§3](#page-2-0) show that, compared to FIFO, reservation strategies can increase

the 99th percentile delay of short-input requests by up to 1.94×. Priority-based strategies [\[25](#page-17-16)[–27\]](#page-18-0) can improve the throughput of short-input requests by assigning them higher priority, but this can cause the starvation of long-input requests. Our measurements in [§3](#page-2-0) show that over 92% of long-input requests receive no service under such strategies. FastServe [\[27\]](#page-18-0) raises the priority of long-waiting requests but does not specifically target long-input ones. Prioritizing such requests would reintroduce head-of-line blocking. Critically, due to the variability in input and output lengths, it is difficult to predict them ahead of time, making optimal scheduling inherently challenging.

We propose PecSched to address these limitations. PecSched is a Preemptive and Efficient Cluster SCHEDuling system for LLM inference. PecSched consists of the following key components.

Preemptive scheduling for long-input requests. Long inputs increase both the prefill phase (input processing to generate the first token) duration and the iteration time during the decode phase (generating subsequent output tokens). To ensure the prioritization of short-input requests, unlike prior work [\[25–](#page-17-16)[27\]](#page-18-0) that prioritizes based on output lengths, PecSched allows short-input requests to preempt long-input ones, reducing queueing time.

Coordinated prefill-decode colocation and disaggregation. Since output length is unpredictable at runtime, allowing short-input requests to preempt long-input ones may prolong the suspension time when the decode time of short-input requests is long. Long decode time of shortinput requests preceding a long-input request can also increase the queueing delay of the long-input request. To address this, we disaggregate the prefill and decode phases of short-input requests. Only the prefill phase is allowed to preempt long-input requests, and long-input requests only need to wait for short-input prefill to complete before execution. Given that prefill is compute-intensive and decode is memory-intensive, we colocate the decode of long-input requests with the prefill of short-input requests. This allows them to run concurrently and avoids suspensions during the decode phase of long input requests.

Fast SP for long request prefill. Long request prefill phases are more likely to be preempted if their durations are long. To minimize prefill time and reduce the frequency of preemptions, PecSched introduces fast SP. Fast SP adopts a hybrid SP strategy. Across nodes, fast SP employs ring attention [\[20\]](#page-17-11) to process sequence segments, offering scalable computation with low communication overhead. However, ring attention has low computational efficiency when the ring is long [\[28\]](#page-18-1). To improve computational efficiency and reduce latency, within a node, fast SP leverages highbandwidth interconnects (e.g., NVLink) and adopts SP variants that have higher communication volume but superior computational efficiency.

In summary, our work has the following contributions.

- •We analyze a recent Azure LLM inference trace [\[21\]](#page-17-12) and first observe that the distribution of input lengths is highly skewed and imbalanced: the vast majority of requests have short inputs, while long inputs constitute only a small fraction.
- •We propose PecSched, a preemptive and efficient cluster scheduling system for LLM inference. PecSched employs preemptive scheduling for long-input requests to address the head-of-line blocking problem. It adopts coordinated prefill-decode colocation and disaggregation to reduce both the duration and frequency of preemptions. Additionally, PecSched introduces fast SP to minimize long request prefill time, further reducing the likelihood and frequency of preemptions.
- Evaluations based on Azure LLM inference trace demonstrate that, compared to state-of-the-art cluster-level LLM inference schedulers, PecSched reduces the 99th percentile queueing delay of short requests by up to 92% and improves their throughput by up to 595%, without significantly affecting the Job Completion Time (JCT) of long requests.

We open-sourced the code of PecSched [\[29\]](#page-18-2).

#### 2 Background

### 2.1 Existing Cluster-Level LLM Scheduling Strategies

The cluster-level scheduling strategies adopted by current LLM inference systems can be categorized into three classes: FIFO [19, 23], reservation-based [24], and priority-based [25–27] scheduling.

- FIFO. FIFO-based LLM schedulers serve requests strictly in the order of their arrival.
- **Reservation.** Reservation-based scheduling (also referred to as isolation-based) allocates dedicated resources to different types of requests to eliminate resource interference between them.
- **Priority**. Priority-based scheduling assigns higher priorities to short-output requests by analyzing or predicting output length distributions, aiming to improve their throughput.

These strategies are primarily designed for requests with input lengths no more than 2K and do not consider the impact of long-input requests. We detail the limitations of these strategies in §3.2.

### <span id="page-2-3"></span>2.2 Sequence Parallelism for Long Requests

SP [19, 20] has been adopted in LLM inference to reduce latency and memory pressure for long-input requests by distributing computation and memory across multiple GPUs or nodes. Ring attention [20], the most widely used SP method, partitions a long input into segments processed by model replicas (ring attention nodes), which exchange KV data and compute full-sequence attention via online-softmax [30]. Tensor Parallelism (TP) can be applied within each node to accelerate segment processing [19]. This approach enables flexible scaling for long-input execution. However, its computational efficiency is low and degrades with increasing ring length, resulting in higher latency [28]. Moreover, ring attention is designed solely for long-input execution and overlooks scenarios where short-input and long-input requests coexist.

#### <span id="page-2-0"></span>3 Motivation

In this section, we provide a detailed analysis of the request length distribution in Microsoft's recently released Azure LLM inference trace [21, 22], and discuss the limitations of existing LLM scheduling strategies in this context, along with the challenges in addressing them. For brevity, we refer to short-input and long-input requests as **short requests and long requests**, respectively, in the rest of the paper when no ambiguity arises.

### <span id="page-2-2"></span>3.1 Request Length Distribution

In LLM inference clusters, most requests have short inputs, while long-input requests are rare. Output lengths also vary but remain relatively small. Based on Microsoft's Azure LLM inference trace [21, 22], Fig.1 shows a highly skewed long-tail distribution in both input and output lengths. Approximately 80% of requests have input lengths below 2K, with frequency decreasing as input length grows. This reflects typical usage patterns: short-input tasks (e.g., conversation) dominate, while long-input tasks (e.g., IR[31, 32], summarization [33]) are less common. Although output lengths are also long-tailed, they remain under 800 tokens, suggesting that input length has a greater execution-time impact. These observations underscore the need for scheduling mechanisms that account for the heterogeneity between short-input and long-input requests.

<span id="page-2-1"></span>> **[图片提取文字 (无描述)]:**
> بان آ ig 0.2 od 0.2 Proport 0.0 0.0 10002000300040005000600070008000 200 400 600 800 Output length Input length (a) Input distribution. (b) Output distribution.
![](_page_2_Figure_15.jpeg)

Fig. 1. Input and output length distributions of Azure LLM inference trace.

#### <span id="page-3-0"></span>3.2 Limitations in Current Scheduling Strategies

The current LLM inference schedulers [19, 23–26] are primarily designed for short requests with input lengths no more than 2K, and do not account for scenarios where short and long requests coexist within the same cluster.

FIFO. The key limitation of FIFO scheduling [19, 23] lies in the fact that long requests that require extensive GPU resources and prolonged execution times can block the execution of short requests, leading to the head-of-line blocking problem. This increases the queueing delay for short requests and degrades their throughput. We evaluate the FIFO-based scheduling strategy using the same setup described in §6.2. To study the impact of long requests on head-of-line blocking under a FIFO scheduling strategy, we conduct two experiments using FIFO. In one setting, we remove all long requests from the trace (the definition of long requests is provided in §6.2), and in the other, we retain all requests. We then evaluate the queueing delay and throughput of short requests under both settings. Fig. 2(a) shows the 1st, 25th, 50th, 75th, and 99th percentile normalized queueing delays of short requests with and without the presence of long requests. We observe that the 99th percentile queueing delay with long requests is 2.5× (Mistral-v0.3 7B), 2.78× (Phi-3 14B), 3.84× (Yi 34B), and 10.2× (Llama-3.1 70B) higher than the delay without long requests. These results indicate that the execution of long requests blocks short requests, significantly increasing their queueing delay. Moreover, as model size increases, the impact becomes more pronounced due to the longer execution time of long requests, resulting in more severe blocking. Fig. 2(b) presents the overall throughput of short requests, measured in Requests Per Second (RPS), under both the presence and absence of long requests. We find that when long requests are present, the throughput of short requests drops to only 0.64× (Mistral-v0.3 7B), 0.56× (Phi-3 14B), 0.39× (Yi 34B), and 0.19× (Llama-3.1 70B) of the throughput observed without long requests. This indicates that the execution of long requests blocks short requests, leading to a reduction in their throughput. Furthermore, the degradation in short request throughput becomes more severe as model size increases, for the same reason discussed earlier.

<span id="page-3-2"></span><span id="page-3-1"></span>> **[图片提取文字 (无描述)]:**
> w/o long w/ long w/o long w/ long oughput (RPS) Normalized neueing delay .0 Llama-3.1 70B € Mistral-v0.3 7 Phi-3 14B Mistral-v0.3 7Bhi-3 14B Llama-3.1 70B Yi 34B Yi 34B (a) Normalized queueing delay of short requests. (b) Throughput of short requests.
![](_page_3_Figure_5.jpeg)

Fig. 2. The normalized queueing delay and the throughput of short requests with and without long requests when using FIFO.

Since LLM input and output lengths are highly variable (§3.1) and difficult to predict accurately in advance, it becomes infeasible to proactively partition resources in an optimal way to alleviate head-of-line blocking. A simple solution is to allow short requests to preempt long requests. However, naive preemption can result in long requests being blocked for a prolonged time by short requests with long execution times. We provide a detailed discussion on how to design more effective preemption mechanisms in §3.3.

**Reservation.** In a cluster with many short requests and few long requests, reservation-based scheduling [24] leads to poor GPU utilization. Substantial resources must be reserved for rare long requests, but these resources often sit idle and cannot be reclaimed by short requests, resulting in significant inefficiency. This can simultaneously increase the queueing delay and decrease the throughput of short requests. We evaluate the reservation-based scheduling strategy using the same setup described in §6.2.

We begin by analyzing the GPU idle rate in the cluster under a reservation-based scheduling strategy. We define the GPU idle rate as:

GPU Idle Rate = 
$$\frac{\sum_{i} g_{i}^{I}}{\sum_{i} (g_{i}^{E} + g_{i}^{I})},$$
 (1)

where  $q_i^E$  and  $q_i^I$  are the total execution time and the total idle time of GPU i, respectively. Table 1 reports the GPU idle rates under the FIFO-based and the reservation-based scheduling strategies for different models. We observe that under the FIFO strategy, GPUs are rarely idle. In contrast, under the reservation-based strategy, GPU idle rates are significantly higher and increase with model size. This is because long requests require more GPU resources to be reserved than short requests, yet their arrival frequency is relatively low. As a result, a large portion of reserved GPU resources often remains unused, leading to high idle rates. When the model size increases, each long request demands even more GPU resources, necessitating more reservations. This further amplifies resource underutilization and results in higher GPU idle rates.

| GPU idle rate | Mistral-v0.3 7B | Phi-3 14B | Yi 34B | Llama-3.1 70B |
|---------------|-----------------|-----------|--------|---------------|
| FIFO          | 0.0004          | 0.00009   | 0.0005 | 0.00008       |
| Reservation   | 0.16            | 0.22      | 0.25   | 0.41          |

<span id="page-4-0"></span>Table 1. GPU idle rate for different models with FIFO and reservation-based strategies.

We further analyze the queueing delay and throughput (RPS) of short requests under both FIFO and reservation-based strategies. Fig. 3(a) shows the 1st, 25th, 50th, 75th, and 99th percentile normalized queueing delays of short requests. We observe that under the reservation-based strategy, the 99th percentile queueing delay is 1.2× (Mistral-v0.3 7B), 1.35× (Phi-3 14B), 1.8× (Yi 34B), and 1.94× (Llama-3.1 70B) compared to that under the FIFO strategy. This is because the GPU resources reserved for long requests cannot be used to serve short requests, limiting the available resources for them. As the model size increases, the queueing delay of short requests increases further. This is due to the fact that larger models require more GPU resources to be reserved for long requests, which further reduces the GPU resources available for short requests, resulting in longer queueing delays. Fig. 3(b) presents the overall throughput of short requests under both FIFO and reservation-based strategies. We observe that under the reservation-based strategy, the throughput of short requests drops to 0.49× (Mistral-v0.3 7B), 0.47× (Phi-3 14B), 0.46× (Yi 34B), and 0.44× (Llama-3.1 70B) of that under the FIFO strategy. The reason is the same as previously explained for Fig. 3(a).

<span id="page-4-1"></span>> **[图片提取文字 (无描述)]:**
> Reservation Reservation **FIFO FIFO** nalized delay Norr Mistral-v0.3 7Bhi-3 14B 0.00 Mistral-v0.3 7 Phi-3 14B Llama-3.1 70B Yi 34B Yi 34B Llam
![](_page_4_Figure_8.jpeg)

<span id="page-4-2"></span>(a) Normalized queueing delay of short requests.

(b) Throughput of short requests. Fig. 3. The normalized queueing delay and the throughput of short requests when using reservation.

The limitation of reservation-based scheduling can be mitigated by allowing short requests to be dispatched to the GPUs reserved for long requests when those resources are idle. However, this approach still suffers from the same head-of-line blocking issue as FIFO-based scheduling. When an long request is executing, many short requests may still be unable to access GPU resources and are forced to wait.

Priority. Priority-based scheduling strategies [25-27] assign higher priorities to short-output requests in order to improve their throughput. However, they do not handle long-input requests with a long-tail distribution. Simply assigning higher priorities to short-input requests has the following issue. In clusters where a large number of short-input requests coexist with a small number of long-input requests, the frequent arrival of short-input requests continuously occupies GPU resources. As a result, insufficient resources are left to serve long-input requests, leading to their starvation. We evaluate the priority-based scheduling strategy using the same setup described in [§6.2.](#page-11-0) Table [2](#page-5-1) shows the proportion of long requests that are starved under the priority-based strategy. The vast majority of long requests are starved and never get served. As the model size increases, the proportion of starved long requests also increases. This is because larger models have longer execution times, causing more short requests to accumulate in the queue, which further reduces the chances for long requests to be served.

| Mistral-v0.3 7B | Phi-3 14B | Yi 34B | Llama-3.1 70B |
|-----------------|-----------|--------|---------------|
| 92%             | 97%       | 100%   | 100%          |

<span id="page-5-1"></span>Table 2. The proportion of long requests that are starved under the priority-based strategy.

To avoid starvation of long requests, one potential solution is to promote their priorities [\[27\]](#page-18-0). However, this still introduces head-of-line blocking, as the execution of an long request can delay many short requests queued behind it.

### <span id="page-5-0"></span>3.3 Challenges

Head-of-line blocking. As discussed in [§3.2,](#page-3-0) the primary challenge in clusters serving both short and long requests lies in mitigating the head-of-line blocking introduced by the execution of long requests. Allowing short requests to preempt long requests can mitigate the head-of-line blocking problem. However, naively pausing the execution of long requests through preemption is inefficient for several reasons.

- First, the analysis of the Azure LLM inference trace in [§3.1](#page-2-2) reveals that the output lengths of requests are highly imbalanced. While the input length that determines the prefill time is known at the beginning of execution, the output length is unknown, making the decode time unknown at the beginning. Consequently, a preempted long request waits for the slowest short request to complete its decode phase, creating uncertainty in preemption duration and complicating scheduling optimization.
- Second, the execution of a request consists of a compute-intensive prefill phase and a memory-intensive decode phase. When a long request is at different phases of execution, how to design preemption strategies that account for its resource usage characteristics becomes a critical question for mitigating the impact of preempting long requests.

To address the first issue, we can disaggregate prefill and decode for short requests. Since the decode phase is memory-intensive and short requests have relatively small KV data, we can reserve a small portion of GPU resources exclusively for short request decode. This ensures that only the prefill phase of short requests preempts long requests. Since prefill time can be estimated from input length at the start of execution, we can leverage this to distribute the prefill workloads of short requests evenly across GPUs during preemption. This approach minimizes the preemption time for long requests by ensuring that short requests complete their prefill phases efficiently in a balanced way during preemption. Separating the prefill and decode phases of short requests provides an additional advantage. When a long request arrives, it only waits for the ongoing short requests to complete their prefill phases before starting execution, without needing to wait for their decode phases to finish.

To address the second issue, during the prefill phase of a long request, short request prefills can preempt its execution. As both request types have compute-intensive prefill phases, the long request's prefill is suspended by short request prefills during preemption. In the decode phase, a long request's KV data volume significantly exceeds that of short requests. To avoid high communication overhead from KV data migration, we keep the KV data of long requests on the prefill GPUs instead of separating the prefill and decode phases as done for short requests. Since decode is memory-intensive and prefill is compute-intensive, we can colocate the decode of long requests with the prefill of short requests, maximizing resource utilization without interrupting the long request's decode execution.

Reducing preemption of long requests. The longer the prefill time of a long request, the more frequently it will be preempted by short request prefills, leading to more preemptions. Table [3](#page-6-0) reports the total number of times the prefill phase of all long requests is preempted by short request prefill (i.e., the total number of suspensions). We observe that as model size increases, the number of preemptions experienced by long request prefill increases. This is because larger models result in longer prefill durations for long requests, making them more likely to be preempted by the prefill of short requests.

| Mistral-v0.3 7B | Phi-3 14B | Yi 34B  | Llama-3.1 70B |
|-----------------|-----------|---------|---------------|
| 167,394         | 205,947   | 278,504 | 379,305       |

Table 3. The total number of preemptions experienced by all long requests.

<span id="page-6-0"></span>To reduce the number of preemptions, it is necessary to shorten the prefill time of long requests, which requires optimizing SP used for their prefill.

As introduced in [§2.2,](#page-2-3) state-of-the-art LLM inference systems employ ring-attention-based SP to serve long requests. Each model replica functions as a ring attention node responsible for processing one segment of the input sequence, and TP can be applied within each replica to further accelerate segment processing. However, relying solely on ring attention and TP is insufficient to minimize prefill time. The computation efficiency of ring attention is low, and as the ring length increases, its efficiency degrades, leading to higher latency [\[28\]](#page-18-1). SP was originally proposed for training longcontext models [\[20,](#page-17-11) [34,](#page-18-8) [35\]](#page-18-9). Megatron [\[34\]](#page-18-8) introduced SP to enhance TP performance by reducing memory pressure and increasing the number of tokens that can be processed (thus improving throughput) without increasing communication overhead. Ulysses [\[35\]](#page-18-9) further extended SP to reduce the communication overhead in Megatron, lowering latency and improving throughput. However, Ulysses requires that model parameters remain intact (i.e., without TP) during linear projection computations involving model parameters. Compared to Megatron and Ulysses, ring attention [\[20\]](#page-17-11) is more easily scalable, as the number of model replicas can be flexibly adjusted to accommodate sequences of varying lengths, and is more suitable for cross-node communication [\[28\]](#page-18-1). However, despite Megatron and Ulysses having higher communication volume than ring attention, they achieve greater computational efficiency under sufficient bandwidth, resulting in higher throughput and lower latency [\[28\]](#page-18-1). Due to the lack of flexibility compared to ring attention, Megatron and Ulysses have not been adopted in inference systems where requests have varying lengths.

To minimize the prefill time of long requests in LLM inference, we can adopt a hybrid SP strategy that leverages the strengths of Megatron, Ulysses, and ring attention. Specifically, ring attention is used across nodes for handling sequence segments due to its scalability and communication efficiency, while within a node, where high-bandwidth interconnects (e.g., NVLink) are available, Megatron or Ulysses is employed to improve throughput and reduce latency. Megatron requires TP and is applicable only when intra-node model replicas use TP [\[34\]](#page-18-8). In contrast, Ulysses is used without TP and remains viable even when TP is enabled, by transferring model parameters to reconstruct full parameters for linear projections. In such cases, we can estimate both computation and communication costs to choose the lower-latency option between Megatron and Ulysses.

#### 4 Preliminaries

Before presenting our design, we first introduce some preliminaries on LLMs and SP initially introduced for training. Table 4 lists notations used in the paper.

<span id="page-7-0"></span>

| d | Model dimension size | $N_h$ | The number of heads | $N_l$ | The number of layers | $d_h$ | Head dimension size (= $d/N_h$ ) |
|---|----------------------|-------|---------------------|-------|----------------------|-------|----------------------------------|
| E | Token embeddings     | Q     | Query matrix        | K     | Key matrix           | V     | Value matrix                     |
| W | Parameter matrix     | s     | Sequence length     |       |                      |       |                                  |

Table 4. Notations used in the paper.

#### 4.1 LLM Foundations

<span id="page-7-1"></span>Fig. 4 shows the architecture of an LLM, which has multiple transformer layers stacked together. Each transformer layer mainly consists of an attention layer and a Multi-Layer Perception (MLP) layer. An attention layer consists of a QKV generation step, a multi-head self-attention computation, and a post-self-attention linear layer. An MLP block has a linear layer (scaling the model dimension size from d to 4d), a GeLU layer, and a second linear layer (from 4d back to d). The output of a transformer layer is the input of the next layer.

> **[图片提取文字 (无描述)]:**
> Transformer Layer Transformer Input + Word |
![](_page_7_Figure_8.jpeg)

Fig. 4. LLM architecture.

The attention layer in Fig. 4 includes the components below.

**QKV generation.** The attention layer takes a sequence embedding E as an input. QKV generation conducts the following operations to generate Q, K, and V of the sequence for head h:

<span id="page-7-2"></span>
$$Q^{h} = EW_{Q}^{h}, K^{h} = EW_{K}^{h}, V^{h} = EW_{V}^{h},$$
 (2)

where  $W_Q^h$ ,  $W_K^h$ , and  $W_V^h$  are the parameters for QKV generation in the attention head h.

**Self-attention.** The self-attention layer takes  $Q^h$ ,  $K^h$ , and  $V^h$  as inputs, and outputs

<span id="page-7-3"></span>
$$O^{h} = \operatorname{Softmax}\left(\frac{Q^{h}(K^{h})^{T}}{\sqrt{d_{h}}}\right)V^{h} = P^{h}V^{h},\tag{3}$$

where  $P_h$  is the attention probability for head h. The softmax function operates row-wise on the input matrix  $[a_{i,j}]$  as follows:

$$\frac{exp(a_{i,j})}{\sum_{k=1}^{t_i} exp(a_{i,k})},\tag{4}$$

where  $t_i$  is the index of the token on row i.

**Post-self-attention linear.** The post-self-attention linear layer takes  $O^h$  from all heads as the input, and it outputs

<span id="page-7-4"></span> $O_L = [O^1, O^2, ..., O^{N_h}]W_L = OW_L,$  (5)

where O is a concatenation of  $O^h$ , and  $W_L$  is the parameter of the post-self-attention linear layer.

### <span id="page-8-2"></span>Sequence Parallelism of Megatron and Ulysses

We introduce the SP architectures used by Megatron and Ulysses. Fig. 5(a) illustrates Megatron SP. Each GPU has a different sequence segment's token embeddings (aka, hidden states) as the input to attention. Each GPU executes Eq. (2) for the first projection to calculate the QKV data of its segment for all heads. Then, the heads are evenly split across GPUs for attention computation. For this purpose, the first all-to-all (A2A) communication sends different head partitions' QKV data from a GPU to their assigned GPUs. After the first A2A, each GPU has the QKV of the entire sequence for part of the heads, and then executes self-attention in Eq. (3). Next, it first gathers the head dimension and splits the sequence dimension of the self-attention output through the second A2A. Then, it conducts the second projection to get the final output via a linear matrix transformation as in Eq. (5).

Fig. 5(b) illustrates Ulysses SP. At the beginning, four GPU devices have their sequence segments. Before the first linear projection, due to the use of TP, each GPU must collect all segments through all-gather. Then, each GPU executes the first projection (Eq. (2)) to obtain the QKV data of the entire sequence of its assigned heads since each GPU is responsible for certain heads (i.e., one head in this example) in TP. Then, each GPU conducts the attention computation on its QKV data (Eq. (3)) and outputs the self-attention result for that head partition. The second linear projection in the post-self-attention linear layer conducts a linear matrix transformation between the self-attention output  $O^h$  and part i of the parameters of the linear layer  $W_I^i$ , generating  $O^hW_I^i$ , a  $L_{in}$ -by-d matrix. Next, using reduce-scatter,  $O^hW_L^i$  from all GPUs are added together to obtain the  $O_L$  in Eq. (5) and then split in sequence dimension into each GPU.

<span id="page-8-0"></span>> **[图片提取文字 (无描述)]:**
> Post-self-attention linear layer Post-self-attention **OKV** generation QKV generation Self-attention Self-attention linear layer Att.(Q, K, V) Att.(Q, K, V) Proj. ba GPU 0:1 Scatte K<sub>V</sub> Proj. Att.(Q, K, V) Proj. **GPU 1:** KV KV Att.(Q, K, V) All GPU 3:1 Proj. GPU 2:1 N<sub>V</sub> Att.(Q, K, V) Proj. GPU 4: I GPU 3:1 Self-attention Self-attention Linear projection Output Tokens' input Tokens' input Linear Tokens' Q, K, or Tokens' Q, K, or Linear Tokens' Q, K, or Self-attention Linear Output All tokens projection projection embedding projection V for each head V for each head output embedding V for each head output output (a) Megatron SP. (b) Ulvsses SP.
![](_page_8_Figure_5.jpeg)

<span id="page-8-1"></span>

Fig. 5. Megatron and Ulysses SP architectures.

### Design

Our system consists of the following main components. First, we enable preemption of long request prefill by short request prefill to mitigate head-of-line blocking. Second, we introduce coordinated prefill-decode colocation and disaggregation to disaggregate short request prefill and decode, and colocate long request decode with short request prefill to reduce the impact of preemptions on long requests. Third, we introduce fast SP for long request prefill by employing a hybrid SP strategy to accelerate execution and reduce the frequency of preemptions.

Fig. 6 provides an overview of our system architecture. Requests first arrive at the cluster-wide global queue and are dispatched by the scheduler (step ①). For short requests that can be handled by a single model replica, the scheduler first attempts to place them in the local queues of idle replicas not occupied by long request prefill and decode (step (2)). If no such replicas are available, it then considers colocating them with long request decode on GPUs with available compute capacity (steps (3) and (4)). If neither is feasible, the scheduler preempts long request prefill to make room for short request prefill execution (step §). Once the short request prefill completes, its KV data is migrated to a dedicated decode-only node for decode (step 6). Long requests, on the other hand, are scheduled across a sufficient number of model replicas, prioritizing those within the same node

to minimize communication overhead. The local queue length of a model replica is defined by the number of tokens in the queue [36]. When a request can be served by multiple valid combinations of model replicas, the combination with the smallest total local queue length is selected.

<span id="page-9-0"></span>> **[图片提取文字 (无描述)]:**
> Long-request prefill preemption & Colocation of long-request decode and short-request prefill execution with fast SP ........ Scheduler Requests 3 (3) (5) (4) Global queue Model Short-request Short-request Long-request prefill decode prefill replica Local queue of a ...... Communication Long-request Colocation within a long request/ Decode disaggregation for short requests model replica decode
![](_page_9_Figure_3.jpeg)

Fig. 6. Design overview.

#### 5.1 Preemption of Long-Request Prefill

While a long request prefill is executing, newly arrived short requests may accumulate. When these short requests need to preempt the ongoing prefill, the system pauses the long request prefill and saves all necessary data for future resumption. Since an LLM is composed of a stack of identical transformer layers, we categorize the data to be saved into two parts: data from completed layers and data from the currently executing layer.

For completed layers, only the generated KV data needs to be preserved for the future decode phase. For the layer currently being executed, we first check whether the KV data for that layer has been generated. If so, the KV data must be retained. We then identify the precise pause point in the prefill execution, which always falls between two kernel operations. A kernel operation may involve computations such as linear projections or self-attention. In such cases, we only need to store the intermediate data passed between these two operations, which is the output of the previous kernel or the input to the next. This intermediate data is usually the token embedding.

Since only one layer's intermediate data needs to be stored, the memory footprint of this data is small, usually less than 5% of the total size of all KV data. To resume execution, the system simply continues from the pause point using the stored intermediate data.

#### 5.2 Coordinated Prefill-Decode Colocation and Disaggregation

**Prefill-decode disaggregation for short requests.** Since short requests typically generate a small amount of KV data and their decode phase is computationally lightweight, we reserve only a small subset of model replicas to handle short request decode. The KV data produced during the prefill phase is transmitted to these decode-only nodes for decode. To further reduce KV transmission latency, we overlap prefill computation with KV transmission. As soon as the KV data for a transformer layer is generated, it is immediately sent to the decode replica, overlapping with the computation of the next layer's prefill.

Because short request prefill and decode are disaggregated, we only need to preempt long request prefill with short request prefill. Since the input length of a short request is known at the time of arrival, we can estimate its prefill latency. To ensure load balance across model replicas during preemption, we construct short request batches such that the total number of tokens per batch is balanced across replicas.

Colocation of long request decode with short request prefill. When a long request enters the decode phase, it typically processes only one input token at a time, resulting in underutilization of GPU compute resources. To improve utilization without suspending long request decode, we colocate short request prefill computations with long request decode. Since the KV data generated

during short request prefill is proactively transmitted to the decode node as soon as it is produced, rather than waiting for the entire prefill phase to complete, it does not consume additional GPU memory. This makes it feasible to colocate long request decode with short request prefill on the same GPU.

Fig. 7 illustrates an example of such colocation. The setup involves two GPUs. Req1 is a long request undergoing decode, containing only a single input token. Req2 and Req3 are short requests in their prefill phase, each with multiple input tokens. Req1 and Req2 are colocated on GPU 1, while Req3 is assigned to GPU 2. All input tokens are first processed through QKV generation as indicated in Eq. (2) to obtain their respective Q, K, and V data (step ①). Req1's Q is then copied from GPU 1 to GPU 2 for self-attention computation with the cached KV data (step ②). Since Req1 only has one token, the communication overhead is small. Meanwhile, the new KV data generated for Req1 on GPU 1 is concatenated with its existing KV cache (step ③). Each GPU then performs self-attention as indicated in Eq. (3) using the local Q and KV data (step ④), producing the output Q for each request. Req1's self-attention outputs from GPU 1 and GPU 2 ( $Q_1$  and  $Q_2$ ) are merged via an all-reduce operation to form the final output Q (step ⑤).

<span id="page-10-0"></span>> **[图片提取文字 (无描述)]:**
> Input token ಲ್ಟ ق (ك Input tokens GPU 2 ᇰᅩ Input tokens
![](_page_10_Figure_4.jpeg)

Fig. 7. An example of colocation of long decode with short prefill.

To avoid introducing additional latency to long request decode, the scheduler balances the total number of input tokens across GPUs and constrains the token count per GPU to a threshold that ensures no degradation in decode performance.

#### 5.3 Fast Sequence Parallelism for Long-Request Prefill

To further reduce the prefill time of long requests and decrease the frequency of preemption by short request prefill, we aim to improve the performance of SP during the prefill phase. Specifically, we leverage the high intra-node bandwidth to enhance inter-GPU collaboration, thereby improving token processing efficiency and reducing overall processing latency. Across nodes, we adopt ring-attention-based KV transmission for SP computation to minimize inter-node communication overhead while maintaining flexibility. Within each node, we employ a hybrid strategy combining Megatron SP and Ulysses SP, which are introduced in §4.2, and dynamically select the fastest approach based on input lengths to minimize the prefill time.

Fig. 8 illustrates an example of fast SP employing a hybrid SP strategy. Fast SP consists of two main stages: the attention stage and the MLP stage. During the attention stage, the system selects either Megatron SP or Ulysses SP within each node based on the actual sequence length to achieve the fastest prefill time. When a model replica uses TP (referred to as a TP region), Ulysses SP requires parameter transmission (step ①) to ensure that each GPU holds the complete parameters needed for computation. This transmission overhead is considered in the strategy selection process. Across nodes, computation is performed using ring attention. If Ulysses SP is used, the KV transmission for ring attention occurs across nodes (step ②); if Megatron SP is used, the KV transmission for ring attention occurs between TP regions (step ③). In the MLP stage, if TP is used without parameter transmission for reconstructing full parameters, the Megatron SP is adopted. Specifically, token embeddings are all-gathered before the MLP, and the MLP outputs are reduced-scattered afterward [34]. Alternatively, if parameters are transmitted (step ④) such that each GPU holds the complete parameters, MLP computation can be performed directly on each sequence segment without any token embedding communication [35].

<span id="page-11-1"></span>> **[图片提取文字 (无描述)]:**
> 2 3 (3) Manager excess excess and the same 3
![](_page_11_Figure_2.jpeg)

Fig. 8. An example of fast SP for prefill with a hybrid SP strategy.

We describe how to select the SP strategy within a node. Let the TP size be denoted by T, the number of GPUs in a node be G, and the sequence segment length processed by each GPU be s. Assume that Group Query Attention (GQA) [37] uses  $N_h$  query heads and  $N_h^{KV}$  key-value heads. In the attention stage, if Megatron SP is selected, the total communication volume in a node is 2sd(T-1)G for all-gather and reduce-scatter, and the total computation volume on a GPU is  $2sd(N_h+N_h^{KV})d_h/T+4(sT)^2d/T+2sd^2$  for QKV generation, self-attention, and post-self-attention linear layer; if Ulysses SP is selected, the total communication volume in a node is  $2s(N_h+N_h^{KV})d_h(G-1)+(d(N_h+N_h^{KV})d_h+d^2)G(T-1)/T$  for two A2A communications and parameter communications, and the total computation volume on a GPU is  $2sd(N_h+N_h^{KV})d_h+4(sG)^2d/G+2sd^2$  for QKV generation, self-attention, and post-self-attention linear layer. In the MLP stage, selecting Megatron SP results in communication volume 2sd(T-1)G in a node for all-gather and reduce-scatter, and computation volume  $16sd^2$  on a GPU for linear projections; selecting Ulysses SP results in communication volume  $8d^2(T-1)G/T$  in a node for parameter communications, and computation volume  $16sd^2$  on a GPU for linear projections. There are four possible combinations of SP strategies across the two stages. For each combination, we compute and estimate the total communication time and computation time, and select the one with the lowest overall latency.

#### 6 Performance Evaluation

#### 6.1 Implementation

We built our system on vLLM [23]. We extended the model code with our custom Distributed Attention class. We used the Triton [38] version of FlashAttention-2 [39] as the core attention but modified it to support the hybrid SP strategy. We modified the LLMEngine and Distributed GPUExecutor in vLLM to enable SP and launch Ray [40] workers for SP partitions. The communication backend is NVIDIA Collective Communications Library (NCCL) [41]. Each worker handling a sequence segment on a GPU runs as a Ray worker and communicates with each other via PyTorch's distributed package with the NCCL backend. The implementation of short request prefill-decode disaggregation was built upon the disaggregation framework in vLLM, but we further extended it by enabling overlap between KV transmission and prefill computation.

#### <span id="page-11-0"></span>6.2 Experiment Setup

In the evaluation, unless otherwise specified, we utilized the following settings.

**Testbed.** We employed four AWS p4de.24xlarge instances [42] located in four nodes. Each instance is equipped with 8 NVIDIA A100 GPUs (each with 80 GiB memory), 96 vCPUs, and 1152 GiB host memory, connected with a 400 Gbps network.

<span id="page-11-2"></span>**Models.** Table 5 lists the state-of-the-art models we used with their TP and Pipeline Parallelism (PP) size, following the setting in [43, 44].

| Mistral-v0.3 7B [45] |             |                   |             |
|----------------------|-------------|-------------------|-------------|
| Yi 34B [47]          | TP=4, no PP | Llama-3.1 70B [7] | TP=4, no PP |

Table 5. Model size and TP/PP size.

**Trace.** We used the Azure LLM inference trace [21] published by Microsoft Azure to generate requests. We adhered to the request arrival times recorded in the trace. We mimicked the input and

output length distributions observed in the trace. Since the maximum input length in the trace is only around 9K, while common long-input datasets such as IR [\[31,](#page-18-4) [32\]](#page-18-5) and book summarization [\[33\]](#page-18-6) contain inputs as long as 100K–500K tokens, we adjusted the trace to better reflect such long-input workloads. Given that the proportion of requests decreases with increasing input length, resulting in long-input requests constituting only a small fraction ([§3.1\)](#page-2-2), we classified requests with input lengths above the 95th percentile in the trace as long-input requests and replaced their input lengths with values randomly sampled from the range of 100K to 500K. All remaining requests were treated as short-input requests. As for output lengths, since the trace and the common long-input datasets such as IR and book summarization exhibit similar output lengths ranging from tens to a few hundred tokens, we directly mimic the output length distribution in the trace without modification. Comparison methods. We adopted the following three comparison methods.

- FIFO. We used vLLM [\[23\]](#page-17-14) as the comparison method that adopts a FIFO-based scheduling strategy.
- Reservation. We used Llumnix [\[24\]](#page-17-15) as the comparison method that adopts a reservationbased strategy. Specifically, it pre-allocates GPU resources capable of handling requests with input lengths of 500K tokens, dedicating them to serve long requests with input lengths between 100K and 500K. The remaining GPU resources are reserved exclusively for serving all other short requests.
- Priority. We used Past-Future [\[26\]](#page-18-7) as the comparison method that adopts a priority-based scheduling strategy. Requests with input lengths between 100K and 500K are assigned low priority, while all other short requests are given high priority.

For long requests, if a single model replica is insufficient to serve a request, SP with ring attention introduced in [§2.2](#page-2-3) is employed to partition the sequence and distribute the computation across multiple model replicas.

Scheduling. A cluster has a global queue, and each model replica in the cluster has a local queue. Each comparison method schedules all requests in the global queue based on its scheduling strategy. A selected request from the global queue is sent to a model replica that has the shortest local queue length [\[36\]](#page-18-10), defined by the number of tokens. Local queues adopt FIFO. If a request requires multiple model replicas, it first prioritizes using replicas within the same node. If those are insufficient, crossnode replicas are employed. When multiple combinations of replicas can satisfy the requirement, the combination with the smallest total local queue length is selected.

Short request decode. Based on the resource consumption characteristics of short request prefill and decode [\[36,](#page-18-10) [44\]](#page-18-18), we allocated 4, 4, 1, and 1 dedicated model replicas for short request decode when evaluating PecSched on Mistral-v0.3 7B, Phi-3 14B, Yi 34B, and Llama-3.1 70B, respectively.

## 6.3 Overall Performance

We measured the queueing delay and throughput of all short requests, as well as the JCT of long requests under each method. Fig. [9](#page-13-0) presents the 1st, 25th, 50th, 75th, and 99th percentile normalized queueing delays of all short requests. Across all models, the 99th percentile queueing delays of short requests under PecSched and Priority are similar. This is because, under Priority, short requests are prioritized and thus not delayed by long requests. In PecSched, short requests can preempt long request prefill, thereby avoiding delay. Compared to FIFO and Reservation, PecSched reduces the 99th percentile queueing delay of short requests by 58%–87% and 61%–92%, respectively, across all models. This improvement arises because long requests delay short ones under FIFO, and in Reservation, fewer GPUs are reserved for short requests than for long ones, causing a huge number of short requests to experience delayed service. The larger the size of the model, the greater the reduction in queueing delay achieved by PECSCHED, as longer execution times of long requests lead to more severe delays for short requests.

<span id="page-13-0"></span>> **[图片提取文字 (无描述)]:**
> **FIFO** Priority Reservation PecSched Mistral-v0.3 Phi-3 14B Llama-3.1 70B Yi 34B
![](_page_13_Figure_3.jpeg)

<span id="page-13-1"></span>> **[图片提取文字 (无描述)]:**
> FIFO Priority Throughput (RPS) Reservation PecSched Mistral-v0.3 7Bhi-3 14B Llama-3.1 70B Yi 34B
![](_page_13_Figure_4.jpeg)

<span id="page-13-2"></span>> **[图片提取文字 (无描述)]:**
> **FIFO** Priority PecSched Reservation ⊙ 30 Mistral-v0.3 7Bhi-3 14B Llama-3.1 70B Yi 34B
![](_page_13_Figure_5.jpeg)

Fig. 9. Normalized queueing delay of short requests.

Fig. 10. Throughput of short re- Fig. 11. Average JCT of long re-

Fig. 10 shows the throughput of all short requests. For all models, the throughput under PecSched is similar to that under Priority. Compared to FIFO and Reservation, PecSched improves short request throughput by 42%-318% and 193%-595%, respectively. The gains grow with model size, due to the same reasons explained for queueing delay.

Fig. 11 shows the average JCT of all long requests. The average JCT under Priority is unbounded, as over 90% of long requests are starved. In contrast, PecSched increases the average ICT of long requests by only 4%-7% and 6%-13% compared to FIFO and Reservation, respectively. This is because PecSched disaggregates the prefill and decode phases of short requests. Consequently, the queueing delay of long requests is only affected by the prefill phase of short requests, and the prefill of long requests can only be preempted by the prefill of short requests, not their decode. Moreover, colocating long request decode with short request prefill avoids interruptions to the former. The use of fast SP for long request prefill further reduces prefill time, thereby lowering both the probability and duration of preemptions. Since long requests have long JCTs, the slight increase in JCTs has a negligible impact on their overall performance.

#### **Ablation Study** 6.4

We test the variants of PecSched as follows to evaluate each individual method. 1) PecSched/PE is PecSched without PreEmption. Short request prefill must wait for the completion of long request prefill before execution. 2) PECSCHED/Dis is PECSCHED without Disaggregation of short request decode. 3) PecSched/CoL is PecSched without CoLocation of long request decode and short request prefill. Short request prefill also preempts long request decode. 4) PecSched/FSP is PECSCHED without Fast SP for long request prefill.

<span id="page-13-3"></span>> **[图片提取文字 (无描述)]:**
> PecSched /Dis /FSP /CoL Mistral-v0.3 7 hi-3 14B Yi 34B Llama-3.1 70B
![](_page_13_Figure_13.jpeg)

<span id="page-13-4"></span>> **[图片提取文字 (无描述)]:**
> PecSched /Dis /FSP Throughput (RPS) /CoL /PE Mistral-v0.3 7Bhi-3 14B Llama-3.1 70B Yi 34B
![](_page_13_Figure_14.jpeg)

<span id="page-13-5"></span>> **[图片提取文字 (无描述)]:**
> PecSched /Dis /CoL Mistral-v0.3 7Bhi-3 14B Llama-3.1 70B Yi 34B
![](_page_13_Figure_15.jpeg)

lay of short requests for individual quests for individual methods.

Fig. 12. Normalized queueing de- Fig. 13. Throughput of short re- Fig.

14. Average JCT of long requests for individual methods.

Fig. 12 shows the 1st, 25th, 50th, 75th, and 99th percentile normalized queueing delays of short requests under each individual method. The /PE method exhibits a 75%-376% higher 99th percentile queueing delay compared to PecSched, as it lacks preemption. Short requests must wait for long request prefill to complete before execution, thereby increasing their queueing delay. In contrast, other individual methods have similar 99th percentile queueing delays to PecSched, as they still incorporate preemption mechanisms that prevent long requests from blocking the execution of short ones. Fig. 13 shows the throughput of short requests under different individual methods. The /PE method achieves 21%–48% lower throughput compared to PecSched. Other individual methods exhibit throughput similar to PecSched. This is due to the same reasons discussed in the context of Fig. [12.](#page-13-3) Fig. [14](#page-13-5) presents the average JCT of long requests under different individual methods. The /PE method achieves 14%–18% lower average JCT compared to PecSched, as short request prefill does not preempt long request prefill, avoiding suspension and thereby reducing the JCT of long requests. In contrast, /Dis, /CoL, and /FSP exhibit 21%–29%, 23%–26%, and 39%–55% higher average JCTs than PecSched, respectively. This is because /Dis does not separate short request decode and prefill, allowing short request decode to preempt long request prefill and thus increasing its JCT. Under /CoL, short request prefill can preempt long request decode, also prolonging JCT. /FSP, on the other hand, increases the prefill time of long requests, which leads to a higher number and duration of preemptions by short requests.

<span id="page-14-0"></span>Table [6](#page-14-0) shows the total number of preemptions experienced by long requests under the individual methods that have the preemption mechanism. As shown, /Dis, /CoL, and /FSP all incur more preemptions than PecSched. For /Dis and /FSP, this is because the extended prefill phase increases the likelihood of being preempted. For /CoL, the increase is attributed to the fact that long request decode is also subject to preemption.

|          | Mistral-v0.3 7B | Phi-3 14B | Yi 34B  | Llama-3.1 70B |
|----------|-----------------|-----------|---------|---------------|
| PecSched | 94,057          | 116,290   | 139,247 | 170,914       |
| /Dis     | 108,552         | 133,784   | 156,891 | 203,628       |
| /CoL     | 130,925         | 167,607   | 209,834 | 261,720       |
| /FSP     | 167,394         | 205,947   | 278,504 | 379,305       |

Table 6. The total number of preemptions experienced by all long requests for individual methods.

### 6.5 Time Overhead

We evaluate the scheduling overheads for both short and long requests under PecSched. For short requests, the scheduling time includes both the scheduling decision time and the context-switching time incurred when preempting the prefill of long requests. For long requests, it consists of the scheduling decision time and the time spent selecting the SP strategy for fast SP. Table [7](#page-14-1) reports the 99th percentile of the scheduling time ratio to JCT for short and long requests. We observe that the ratio of scheduling time to JCT does not exceed 0.345%, indicating that the scheduling overhead is acceptable. As the model size or request length increases, this ratio decreases because the JCT increases while the scheduling overhead remains relatively constant, resulting in a lower overhead-to-JCT ratio.

|                | Mistral-v0.3 7B | Phi-3 14B | Yi 34B | Llama-3.1 70B |
|----------------|-----------------|-----------|--------|---------------|
| Short requests | 0.354%          | 0.282%    | 0.196% | 0.071%        |
| Long requests  | 0.183%          | 0.147%    | 0.055% | 0.019%        |

<span id="page-14-1"></span>Table 7. The 99th percentile ratio of scheduling time to JCT for long and short requests.

### 6.6 Scalability Test

We conduct a simulation-based study to evaluate the impact of cluster scale on the scheduling overhead of PecSched. Based on the input and output lengths of requests in the trace we used in [§6.2,](#page-11-0) we simulate their prefill and decode execution times. During the simulation, we vary the total number of GPUs in the cluster, while keeping each server node configured with 8 GPUs, consistent with [§6.2.](#page-11-0) For each GPU count setting, we generate requests following a Poisson distribution, with the request arrival rate (RPS) set to the cluster's maximum capacity based on the throughput data in Fig. [10.](#page-13-1) Fig. [15](#page-15-0) shows the 99th percentile of the scheduling time ratio to JCT under different GPU

counts. We observe that as the number of GPUs increases, the 99th percentile scheduling time ratio grows approximately linearly. Even when the cluster size reaches 8192 GPUs, the ratio remains below 5.2%, which is acceptable. The scheduling time ratio decreases as model size increases. This is because larger models lead to higher JCTs, while the number of model replicas that influence the scheduling search space becomes smaller, which in turn slightly reduces the scheduling decision time. For example, the scheduling time ratio for Llama-3.1 70B remains below 1.1% in the simulation. Given that modern clusters are typically dominated by large models, we expect the scheduling overhead of Pecsched to remain low in practice.

<span id="page-15-0"></span>> **[图片提取文字 (无描述)]:**
> Mistral-v0.3 7B ——Yi 34B The 99th percentile time ratio (%) 0 1 0 2 5 5 → Llama-3.1 70B Phi-3 14B 1000 2000 3000 4000 5000 6000 7000 8000 The number of GPUs
![](_page_15_Figure_3.jpeg)

Fig. 15. The 99th percentile ratio of scheduling time to JCT with different numbers of GPUs.

#### 7 Related Work

Cluster-level LLM inference scheduler. Current cluster-level LLM inference scheduling strategies can be broadly categorized into three classes: FIFO-based [19, 23], reservation-based [24], and priority-based [25–27] approaches. FIFO-based strategies serve requests in the order of their arrival. Reservation-based strategies allocate dedicated resources to different types of requests to avoid resource interference. Priority-based strategies assign higher priorities to short-output requests to improve their throughput. However, none of these approaches account for the coexistence of short-input and long-input requests within a cluster, which can lead to head-of-line blocking, low resource utilization, and starvation of long-input requests. Our system addresses these limitations. Inference execution. vLLM [23] uses Paged Attention to enable a non-contiguous KV cache, which reduces memory fragmentation and increases throughput. HuggingFace TGI [48] and NVIDIA TensorRT-LLM [49] have also implemented the non-contiguous KV cache. DistServe [44] and SplitWise [36] split prefill and decode to improve throughput. FastGen [50] and Sarathi-Serve [43] chunk a long prompt and batch chunks sequentially with token generation tasks. These works focus on improving inference execution rather than request scheduling within the cluster.

**Sequence parallelism for long request execution.** SP [20, 34, 35, 51] was initially proposed to train long-context models. Other recent studies [52–64] proposed transformer variants to handle long sequences for training, striking a balance between performance and accuracy. Recent work [19] introduced SP with ring attention [20] to LLM inference to handle long requests losslessly. However, it does not address the scheduling of short and long requests in a cluster. Our system handles this.

#### 8 Discussion and Limitations

Colocation of long request prefill with short request decode. We do not consider colocating long request prefill with short request decode because the long prefill time of long requests can delay a decode iteration of short requests. Additionally, the total decode time for short requests varies across different model replicas and is unpredictable, further complicating colocation.

**Limitations and future work.** Although PECSCHED mitigates preemption overhead for long requests, it does not fully eliminate it, as long request prefill phases may still be suspended. Our

future work will explore improved methods to boost short request throughput while further reducing long request JCT.

### 9 Conclusion

This paper presents PecSched, a preemptive and efficient cluster scheduling system for LLM inference. PecSched introduces preemptive scheduling, prefill-decode disaggregation of short requests, colocation of long request decode with short request prefill, and fast SP with a hybrid SP strategy for long request prefill. It improves the performance of short requests without significantly affecting the JCT of long requests. Both real experiments and large-scale simulations show the superior performance of PecSched in comparison with the state-of-the-art.

### References

- <span id="page-17-0"></span>[1] OpenChat ShareGPT4. https://huggingface.[co/datasets/openchat/openchat\\_sharegpt4\\_dataset,](https://huggingface.co/datasets/openchat/openchat_sharegpt4_dataset) 2024.
- <span id="page-17-1"></span>[2] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.
- [3] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc., 2020.
- [4] Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. arXiv, 2023.
- [5] Facebook opt models. [https://huggingface](https://huggingface.co/models?sort=trending&search=facebook+opt).co/models?sort=trending&search=facebook+opt, 2025.
- [6] Meta llama-2 models. [https://huggingface](https://huggingface.co/models?sort=trending&search=meta+Llama-2).co/models?sort=trending&search=meta+Llama-2, 2025.
- <span id="page-17-2"></span>[7] Meta Llama-3.1. [https://llama](https://llama.meta.com/).meta.com/, 2025.
- <span id="page-17-3"></span>[8] Aydar Bulatov, Yuri Kuratov, and Mikhail S. Burtsev. Scaling transformer to 1m tokens and beyond with rmt. ArXiv, abs/2304.11062, 2023.
- [9] Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with blockwise ringattention. arXiv preprint arXiv:2402.08268, 2024.
- [10] GLM-4-9B-1M. https://huggingface.[co/THUDM/glm-4-9b-chat-1m,](https://huggingface.co/THUDM/glm-4-9b-chat-1m) 2024.
- [11] InternLM2.5-7B-Chat-1M. https://huggingface.[co/internlm/internlm2\\_5-7b-chat-1m,](https://huggingface.co/internlm/internlm2_5-7b-chat-1m) 2024.
- <span id="page-17-4"></span>[12] Gemini. [https://gemini](https://gemini.google.com/).google.com/, 2024.
- <span id="page-17-5"></span>[13] Elozino Egonmwan and Yllias Chali. Transformer-based model for single documents neural summarization. In Proceedings of the 3rd Workshop on Neural Generation and Translation, pages 70–79, Hong Kong, November 2019. Association for Computational Linguistics.
- [14] Zi Gong, Cuiyun Gao, Yasheng Wang, Wenchao Gu, Yun Peng, and Zenglin Xu. Source code summarization with structural relative position guided transformer. In 2022 IEEE International Conference on Software Analysis, Evolution and Reengineering (SANER), pages 13–24, 2022.
- <span id="page-17-6"></span>[15] Haopeng Zhang, Xiao Liu, and Jiawei Zhang. HEGEL: Hypergraph transformer for long document summarization. arXiv, 2022.
- <span id="page-17-7"></span>[16] Ashutosh Adhikari, Achyudh Ram, Raphael Tang, and Jimmy Lin. Docbert: BERT for document classification. CoRR, abs/1904.08398, 2019.
- <span id="page-17-8"></span>[17] Xiang Dai, Ilias Chalkidis, Sune Darkner, and Desmond Elliott. Revisiting transformer-based models for long document classification. arXiv, 2022.
- <span id="page-17-9"></span>[18] GitHub Copilot. https://github.[com/features/copilot/,](https://github.com/features/copilot/) 2024.
- <span id="page-17-10"></span>[19] Bingyang Wu, Shengyu Liu, Yinmin Zhong, Peng Sun, Xuanzhe Liu, and Xin Jin. Loongserve: Efficiently serving longcontext large language models with elastic sequence parallelism. In Proceedings of the ACM SIGOPS 30th Symposium on Operating Systems Principles, SOSP '24, page 640–654, New York, NY, USA, 2024. Association for Computing Machinery.
- <span id="page-17-11"></span>[20] Hao Liu, Matei Zaharia, and Pieter Abbeel. Ring attention with blockwise transformers for near-infinite context. arXiv preprint arXiv:2310.01889, 2023.
- <span id="page-17-12"></span>[21] Azure LLM inference trace. https://github.[com/Azure/AzurePublicDataset/blob/master/](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md) [AzureLLMInferenceDataset2024](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md).md, 2024.
- <span id="page-17-13"></span>[22] Analysis of Azure LLM inference trace. https://github.[com/Azure/AzurePublicDataset/blob/master/analysis/](https://github.com/Azure/AzurePublicDataset/blob/master/analysis/AzureLLMInferenceDataset2024.ipynb) [AzureLLMInferenceDataset2024](https://github.com/Azure/AzurePublicDataset/blob/master/analysis/AzureLLMInferenceDataset2024.ipynb).ipynb, 2024.
- <span id="page-17-14"></span>[23] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, SOSP '23, page 611–626, New York, NY, USA, 2023. Association for Computing Machinery.
- <span id="page-17-15"></span>[24] Biao Sun, Ziming Huang, Hanyu Zhao, Wencong Xiao, Xinyi Zhang, Yong Li, and Wei Lin. Llumnix: Dynamic scheduling for large language model serving. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), pages 173–191, Santa Clara, CA, July 2024. USENIX Association.
- <span id="page-17-16"></span>[25] Hyungjun Oh, Kihong Kim, Jaemin Kim, Sungkyun Kim, Junyeol Lee, Du-seong Chang, and Jiwon Seo. Exegpt: Constraint-aware resource scheduling for llm inference. In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, ASPLOS '24, page 369–384, New York, NY, USA, 2024. Association for Computing Machinery.

- <span id="page-18-7"></span>[26] Ruihao Gong, Shihao Bai, Siyu Wu, Yunqian Fan, Zaijun Wang, Xiuhong Li, Hailong Yang, and Xianglong Liu. Past-future scheduler for llm serving under sla guarantees. ASPLOS '25, page 798–813, New York, NY, USA, 2025. Association for Computing Machinery.
- <span id="page-18-0"></span>[27] Bingyang Wu, Yinmin Zhong, Zili Zhang, Shengyu Liu, Fangyue Liu, Yuanhang Sun, Gang Huang, Xuanzhe Liu, and Xin Jin. Fast distributed inference serving for large language models. arXiv preprint arXiv:2305.05920, 2024.
- <span id="page-18-1"></span>[28] Jiarui Fang and Shangchun Zhao. Usp: A unified sequence parallelism approach for long context generative ai. arXiv preprint arXiv:2405.07719, 2024.
- <span id="page-18-2"></span>[29] The code of PecSched. [https://anonymous](https://anonymous.4open.science/r/PEACE).4open.science/r/PEACE, 2025.
- <span id="page-18-3"></span>[30] Maxim Milakov and Natalia Gimelshein. Online normalizer calculation for softmax. CoRR, abs/1805.02867, 2018.
- <span id="page-18-4"></span>[31] Sunhao Dai, Weihao Liu, Yuqi Zhou, Liang Pang, Rongju Ruan, Gang Wang, Zhenhua Dong, Jun Xu, and Ji-Rong Wen. Cocktail: A comprehensive information retrieval benchmark with llm-generated documents integration, 2024.
- <span id="page-18-5"></span>[32] Mo Li, Songyang Zhang, Yunxin Liu, and Kai Chen. Needlebench: Can llms do retrieval and reasoning in 1 million context window? arXiv preprint arXiv:2407.11963, 2024.
- <span id="page-18-6"></span>[33] BookCorpus. https://huggingface.[co/datasets/bookcorpus/bookcorpus,](https://huggingface.co/datasets/bookcorpus/bookcorpus) 2024.
- <span id="page-18-8"></span>[34] Vijay Anand Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. Reducing activation recomputation in large transformer models. Proceedings of Machine Learning and Systems, 5:341–353, 2023.
- <span id="page-18-9"></span>[35] Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He. Deepspeed ulysses: System optimizations for enabling training of extreme long sequence transformer models. arXiv, 2023.
- <span id="page-18-10"></span>[36] Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, and Ricardo Bianchini. Splitwise: Efficient generative llm inference using phase splitting. In 2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA), pages 118–132, 2024.
- <span id="page-18-11"></span>[37] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. GQA: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245, 2023.
- <span id="page-18-12"></span>[38] Triton. [https://triton-lang](https://triton-lang.org/main/index.html#).org/main/index.html#, 2024.
- <span id="page-18-13"></span>[39] Tri Dao. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691, 2023.
- <span id="page-18-14"></span>[40] Ray. [https://ray](https://ray.io).io, 2025.
- <span id="page-18-15"></span>[41] Nvidia collective communications library (NCCL). [https://developer](https://developer.nvidia.com/nccl).nvidia.com/nccl.
- <span id="page-18-16"></span>[42] Amazon EC2 P4 instances. https://aws.amazon.[com/ec2/instance-types/p4/.](https://aws.amazon.com/ec2/instance-types/p4/)
- <span id="page-18-17"></span>[43] Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav Gulavani, Alexey Tumanov, and Ramachandran Ramjee. Taming Throughput-Latency tradeoff in LLM inference with Sarathi-Serve. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), pages 117–134, Santa Clara, CA, July 2024. USENIX Association.
- <span id="page-18-18"></span>[44] Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang. DistServe: Disaggregating prefill and decoding for goodput-optimized large language model serving. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), pages 193–210, Santa Clara, CA, July 2024. USENIX Association.
- <span id="page-18-19"></span>[45] Mistral-v0.3. https://huggingface.[co/mistralai/Mistral-7B-Instruct-v0](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3).3, 2025.
- <span id="page-18-20"></span>[46] Microsoft Phi-3. https://huggingface.[co/microsoft/Phi-3-medium-128k-instruct,](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct) 2025.
- <span id="page-18-21"></span>[47] 01-ai model Yi. https://huggingface.[co/01-ai/Yi-34B-200K,](https://huggingface.co/01-ai/Yi-34B-200K) 2025.
- <span id="page-18-22"></span>[48] Hugging Face TGI. https://huggingface.[co/text-generation-inference,](https://huggingface.co/text-generation-inference) 2024.
- <span id="page-18-23"></span>[49] TensorRT-LLM. https://github.[com/NVIDIA/TensorRT-LLM,](https://github.com/NVIDIA/TensorRT-LLM) 2024.
- <span id="page-18-24"></span>[50] Connor Holmes, Masahiro Tanaka, Michael Wyatt, Ammar Ahmad Awan, Jeff Rasley, Samyam Rajbhandari, Reza Yazdani Aminabadi, Heyang Qin, Arash Bakhtiari, Lev Kurilenko, and Yuxiong He. Deepspeed-fastgen: High-throughput text generation for llms via mii and deepspeed-inference. arXiv preprint arXiv:2401.08671, 2024.
- <span id="page-18-25"></span>[51] Shenggui Li, Fuzhao Xue, Yongbin Li, and Yang You. Sequence parallelism: Long sequence training from system perspective. CoRR, abs/2105.13120, 2021.
- <span id="page-18-26"></span>[52] Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. CoRR, abs/2006.04768, 2020.
- [53] Genta Indra Winata, Samuel Cahyawijaya, Zhaojiang Lin, Zihan Liu, and Pascale Fung. Lightweight and efficient end-to-end speech recognition using low-rank transformer. In ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 6144–6148, 2020.
- [54] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. CoRR, abs/2006.16236, 2020.

- [55] Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamás Sarlós, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy J. Colwell, and Adrian Weller. Rethinking attention with performers. CoRR, abs/2009.14794, 2020.
- [56] Zhen Qin, XiaoDong Han, Weixuan Sun, Dongxu Li, Lingpeng Kong, Nick Barnes, and Yiran Zhong. The devil in linear transformer. arXiv, 2022.
- [57] Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, and Yee Whye Teh. Set transformer: A framework for attention-based permutation-invariant neural networks. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 3744–3753. PMLR, 09–15 Jun 2019.
- [58] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 4651–4664. PMLR, 18–24 Jul 2021.
- [59] Xuezhe Ma, Xiang Kong, Sinong Wang, Chunting Zhou, Jonathan May, Hao Ma, and Luke Zettlemoyer. Luna: Linear unified nested attention. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, volume 34, pages 2441–2453. Curran Associates, Inc., 2021.
- [60] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G. Carbonell, Quoc V. Le, and Ruslan Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context. CoRR, abs/1901.02860, 2019.
- [61] Aydar Bulatov, Yuri Kuratov, and Mikhail S. Burtsev. Scaling transformer to 1m tokens and beyond with rmt. arXiv, 2023.
- [62] Yuhuai Wu, Markus N. Rabe, DeLesley Hutchins, and Christian Szegedy. Memorizing transformers. arXiv, 2022.
- [63] Weizhi Wang, Li Dong, Hao Cheng, Xiaodong Liu, Xifeng Yan, Jianfeng Gao, and Furu Wei. Augmenting language models with long-term memory. arXiv, 2023.
- <span id="page-19-0"></span>[64] Jiayu Ding, Shuming Ma, Li Dong, Xingxing Zhang, Shaohan Huang, Wenhui Wang, Nanning Zheng, and Furu Wei. Longnet: Scaling transformers to 1,000,000,000 tokens. arXiv, 2023.