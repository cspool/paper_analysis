# <span id="page-4-0"></span>4.2 Service Time Predictor Design

The efficacy of our scheduling decisions hinges on accurate service time predictions for each stage of a request. The design of the predictor and the scheduling algorithm are orthogonal problems. To rigorously evaluate our scheduler's performance, isolated from potential prediction inaccuracies, we employ an idealized yet practical prediction methodology.

Compute Time Prediction. The total computation time,  $T_{\rm comp}(S_i^j)$ , is composed of prefill and decoding time. For the prefill stage, which is a deterministic function of input length, we build a data-driven performance model by offline-profiling the target hardware with various sequence lengths. For the decoding stage, predicting the generation length is a known challenge. Prior work has demonstrated high-accuracy predictors are feasible. Based on this, we adopt a **segment-level oracle** for the number of generated tokens  $(n_{\rm gen})$ , using the ground-truth value from our dataset. It is critical to note this oracle has no knowledge of future segments, thus preserving the online nature of the scheduling problem. The total predicted computation time is then:  $T_{\rm comp}(S_i^j) = f_{\rm prefill}(n_{\rm in}) + n_{\rm gen} \cdot {\rm avg\_decode\_latency\_per\_token}$ .

API Latency Prediction. API call durations are highly variable and depend on external factors. Static analysis is often intractable. However, we observe that APIs within the same functional category exhibit stable latency distributions. For example, math-related API calls average 9e-5 seconds, while image generation and chatbot APIs can take tens of seconds (e.g., means of 20.03s and 28.6s respectively). Leveraging this, we build a statistical model based on these categories. During scheduling, we extract the API category from the prompt and use the category's mean latency as the predicted value  $T_A(S_i^j)$ .

### <span id="page-4-1"></span>4.3 Stateful-MLFQ Scheduling Algorithm

4.3.1 Algorithm Design Overview. To operationalize the global optimization objective defined in Section 3.1, we design a state-aware multi-level feedback queue scheduling algorithm (Stateful-MLFQ), with its full logic presented in Algorithm 1. The core of this algorithm lies in its "Stateful" nature: it not only evaluates a segment's current characteristics (e.g., estimated service time) but also unifies the parent request's historical behavior (e.g., accumulated wait

time, past compute and I/O patterns) with future predictions into a single decision-making framework. It is a hierarchical, preemptive scheduling algorithm that aims to strike a balance between efficiency and fairness through macro and micro-level controls.

4.3.2 Macro-level Control: Event-driven Priority Migration. The algorithm's macro-level framework is built upon a multi-level feedback queue (MLFQ) structure, consisting of m queues  $Q_0, ..., Q_{m-1}$  with strict priorities.

We adopt Token Cost instead of a time slice as the migration threshold because the time slice is an unstable metric heavily influenced by batch composition in continuous batching. In contrast, Token Cost is a deterministic, intrinsic metric that solely measures the computational work a request has received. This stability makes priority migration decisions fairer and more robust.

The migration of requests between queues is event-driven, as defined in the event-handling functions of Algorithm 1.

- (1) **On Request Arrival:** All new requests are placed into the highest-priority queue  $Q_0$  to ensure a fast response.
- (2) On Segment Completion: After a segment finishes execution, the system dynamically adjusts its parent request's priority based on its behavior.
  - Demotion: If a segment's computational cost exceeds its queue's threshold, the parent request is identified as compute-intensive and demoted to the next lowerpriority queue.
  - **Promotion:** If a segment yields to an API call before exhausting its token cost quota, the parent request is identified as I/O-intensive and is promoted to a higher-priority queue. This policy aims to prioritize I/O-bound requests to minimize their impact on the total JCT.

4.3.3 Scheduling Cycle: Batch Building and Intra-Queue Sorting. In each scheduling cycle, our core scheduling function, BuildNextBatch, is invoked to determine the next batch of requests to execute.

The function's first step is to handle starvation (lines 1-6). It inspects all requests in the lowest-priority queue,  $Q_{m-1}$ , and if a request's response ratio exceeds a predefined aging threshold, it is preemptively promoted to the highest-priority queue,  $Q_0$ .

After handling starvation, the scheduler iterates from the highest-priority queue  $Q_0$  downwards (line 8) to find the first non-empty queue,  $Q_k$ . For all ready segments within this queue, the scheduler performs micro-level, intra-queue sorting (lines 11-14). We employ the Highest Response Ratio Next (HRRN) policy to calculate a score for each segment:

$$Score_{HRRN}(S_i^j) = \frac{W(R_i) + T_{\text{proc}}(S_i^j)}{T_{\text{proc}}(S_i^j)},$$
(4)

where  $W(R_i)$  is the accumulated waiting time of its parent request and  $T_{\mathrm{proc}}(S_i^j)$  is the estimated service time of the current segment. This mechanism behaves like Shortest Remaining Processing Time (SRPT) when waiting times are comparable, enhancing efficiency. As the waiting time of a long job accumulates, its score increases, ensuring intra-queue fairness.

Finally, after sorting candidate segments by their HRRN score, the scheduler packs them sequentially into the next batch until GPU memory capacity is reached (lines 15-20).

```
Input :: Number of queues; : Priority queues;  :
         Token thresholds; : Aging threshold.
 Output:The next execution batch .
1 function BuildNextBatch(,  , ):
     // 1. Starvation Prevention (Aging)
2 foreach request  ∈ −1 do
3 if
```

**Algorithm 1:** Stateful-MLFQ Scheduling

```
(.  + . )/.  >
       then
4 0 ← 0 ∪ {};
5 −1 ← −1 ∖ {};
   // 2. Batch Construction
6  ← ∅;
7 for  ← 0 to  − 1 do
8 if 
         is not empty then
9  ← GetAllReadySegments(
                          );
        // 3. Intra-queue sorting using HRRN
10 foreach segment  ∈  do
11 .    ←
           (. +. )/. ;
12  ← SortByHRRNScore(, DESC);
        // 4. Pack batch respecting memory
          constraints
13 foreach segment  ∈  do
14 if CanFitInMemory( ∪ {}) then
15  ←  ∪ {};
16 return ;
```

<span id="page-5-2"></span>4.3.4 Preemption Granularity. It is important to note that our algorithm's preemption occurs at the segment level. Once a batch of tasks begins execution, it runs to completion (i.e., until all segments in the batch either trigger an API call or generate a final response) without being interrupted at the iteration level. Preemption is realized in each new scheduling cycle: a newly arrived or promoted high-priority request can "preempt" the execution opportunity of a lower-priority request when the next batch is being constructed. This design avoids the prohibitive overhead of finegrained preemption and its associated KV cache swapping costs.

