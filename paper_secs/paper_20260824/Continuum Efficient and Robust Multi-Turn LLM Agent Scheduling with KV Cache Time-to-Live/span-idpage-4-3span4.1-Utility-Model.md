# <span id="page-4-3"></span>4.1 Utility Model

To set an effective TTL value (in seconds) for pinning a request's KV cache, Continuum must choose the value that best balances the benefit of potential reuse against its cost. Both the benefit and the cost are measured in units of time, since they ultimately translate into changes in the total job completion latency across all programs. Mathematically, given a request *r* and a TTL value τ, Continuum estimates Cost(τ,*r*) and Benefit(*r*) for pinning the KV cache of request *r* for τ. For simplicity, Benefit(*r*) assumes that the next request arrives within the TTL window. The case where TTL expires before the tool call returns is addressed in Sec. [4.2.](#page-5-0)

Cost Estimation. The cost of pinning a request's KV cache comes from the opportunity cost of occupying GPU memory

| Notation             | Description                                     |  |
|----------------------|-------------------------------------------------|--|
| τ                    | TTL                                             |  |
| MemUsage(r)          | GPU memory occupied by r                        |  |
| $\mathcal{M}$        | Average memory occupied by the seen requests    |  |
| CacheMissCost(r)     | Cost of reloading <i>r</i>                      |  |
| Prefill-Reload $(r)$ | Time for reconstructing KV cache in GPU         |  |
| OutofOrderCost(r)    | Cost of out-of-order for request r              |  |
| η                    | Memoryfulness factor of the workload            |  |
| $\mathcal{T}$        | Average waiting time                            |  |
| $\mathcal{P}(t,f)$   | Estimated finish-within-TTL probability for $f$ |  |

Table 3: Key notations in Continuum's cost model for a request r and its associated tool-call f.

that could otherwise be used to serve other requests:

$$\mathsf{Cost}(\tau,r) = \frac{\mathsf{MemUsage}(r)}{\mathcal{M}} \times \tau,$$

where  $\mathsf{MemUsage}(r)$  is the amount of GPU memory used by the KV cache of request r,  $\mathcal{M}$  is the average GPU memory footprint of active requests, and  $\tau$  is the TTL value.

The ratio  $\frac{\text{MemUsage}(r)}{\mathcal{M}}$  represents how many average requests are blocked when r is pinned. In other words, if pinning r occupies the same memory as k requests, then pinning r adds  $\tau$  latency to approximately k other requests. We assume that the waiting queue always contains enough requests for this blocking effect to occur when KV retention is necessary.

**Benefit Estimation.** The benefit of pinning a request's KV cache is realized when the request is re-issued within the TTL period, allowing it to avoid the overhead of reloading or prefilling the KV cache from *r*'s program while saving the per-turn queueing delay:

$$Benefit(r) = CacheMissCost(r) + OutofOrderCost(r)$$

Here, CacheMissCost(r) measures the cost of reloading or prefilling the KV cache for request r and OutofOrderCost(r) measures the expected queueing delay for the request due to waiting for other requests to free GPU memory. We use the sum of cost prevented as the benefit.

Similar to  $\mathsf{Cost}(\tau, r)$ , we can measure  $\mathsf{CacheMissCost}(r)$  by (1) the context reconstruct overhead  $\mathsf{Prefill}\text{-Reload}(r)$ ; and (2) the approximate number of requests will experience the additional latency overhead  $\frac{\mathsf{MemUsage}(r)}{\mathcal{M}}$ . The cost is formally defined as follows:

$$\mathsf{CacheMissCost}(r) = \frac{\mathsf{MemUsage}(r) \times \mathsf{Prefill-Reload}(r)}{\mathcal{M}}$$

Prefill-Reload(r) is the time cost for prefill or reloading depending on whether CPU offloading is turned on. This is based on a quick offline profiling described in Sec 5.2.

**Measuring the expected queuing delay:** As discussed in Sec. 3.2, retaining KV cache also eliminates the queueing delay that a returning program would experience if evicted—even when CPU offloading makes reload itself fast. This

OutofOrderCost component is the key term absent from prior retention policies such as InferCept [2], which only considers the reload cost. By modeling this term, Continuum can justify retaining KV cache even when reload is cheap, as long as the queueing delay savings outweigh the GPU memory occupation cost. Note that the queueing delay benefit is closely tied to the memoryfulness of the workload, *i.e.*, whether the number of remaining steps reduces predictably as the program progresses.

For example, if the number of requests issued by each program follows a geometric distribution, then the expected number of remaining requests is constant regardless of how many have already been served; in this case, pinning provides no benefit for the queueing delay since keeping the order does not accelerate finishing short jobs first. In contrast, if each program issues a fixed number of requests, then the TTL can eliminate the queueing cost by approximating Shortest Job First.

Let N be the total number of requests in a program and k the number of requests that have already been served. We define the following *memoryfulness factor* 

$$\eta = -Corr(k, N - k)$$

We can see this factor models the degree of memoryfulness in the workload well: when the workload is fully memoryless, we have that k is independent to N-k, leading to  $\eta=0$ . Conversely, when the workload is fully memoryful, *i.e.*, all programs have the same fixed number of requests, we have  $\operatorname{Corr}(k,N-k)=\operatorname{Corr}(k,-k)=-1$ , resulting in  $\eta=1$ . Note that, in some cases  $\eta$  may be less than zero (extremely longtail turn distribution), indicating an *anti-memoryful* pattern in which making progress on a program appears to reveal even more remaining work. We did not observe such patterns but Continuum is designed with such extreme workloads in mind: it would be preferable to serve each program only briefly and switch frequently to adapt to the long-tail turn distribution.

Now, we are ready to define the  $\mathsf{OutofOrderCost}(r)$  based on the  $\eta$  above. When  $\eta=1$ , the delay is exactly the waiting time when the program of r returns back to the waiting queue. To match this, we record the average waiting time per unit context size for the historical requests in this workload as  $\frac{T}{M}$ , where T is the average queueing delay for previous requests. In this case, the delay can be well measured by  $\frac{T}{M} \times \mathsf{MemUsage}(r)$ . Here, we consider  $\mathsf{MemUsage}(r)$  since large-context requests are harder to schedule (they must wait for enough contiguous memory to be freed). For the general cases, we define the out-of-order cost as follows:

$$\mathsf{OutofOrderCost}(r) = \frac{\mathcal{T}}{\mathcal{M}} \times \mathsf{MemUsage}(r) \times \eta.$$

## <span id="page-5-0"></span>4.2 Setting the TTL Value

In this part, we describe how Continuum sets the TTL value for KV cache based on the cost-benefit model above and historical tool-call information. As in Algorithm [1](#page-4-1) (line [12\)](#page-4-2), Continuum determines the optimal TTL value τ ∗ to maximize the expected net benefit of retaining the KV cache:

$$\mathbf{\tau}^* = \operatorname{argmax}_{\mathbf{\tau}} \, \boldsymbol{\mathcal{P}}(\mathbf{\tau}, f) \times \mathsf{Benefit}(r) - \mathsf{Cost}(\mathbf{\tau}, r), \quad \ (1)$$

where *P*(τ, *f*) estimates the probability that the tool call *f* completes within time τ. This formula captures the expected net benefit, in terms of total job latency, of retaining the KV cache of *r* for a duration of τ By eliminating the shared MemUsage(r) *M* , the formula above can be transformed to

$$\operatorname{argmax}_{\tau} \mathcal{P}(\tau, f) \times (\mathcal{T} \cdot \eta + \operatorname{Prefill-Reload}(r)) - \tau,$$
 (2)

indicating that we only need to additionally compute *T* and *P*(τ, *f*) in our implementation. *T* can be estimated as the sliding window average for queueing delay experienced by requests who was evicted. Since we cannot fully predict the duration of the next tool call, we estimate *P*(τ, *f*) using the empirical CDF derived from historical tool-call records *S*[ *f* ]. Specifically, we calculate it as the following:

$$\mathcal{P}(\tau, f) = \frac{1}{|S[f]|} \cdot \sum_{t \in S[f]} \mathbb{I}[t \le \tau]$$

, where I[·] is the indicator function. Finally, we solve Equation [\(2\)](#page-6-0) by enumerating all unique tool-call durations recorded in *S*[ *f* ] as candidates (including τ = 0) and selecting the one with the highest expected reward.

Cold-start Handling. When the number of historical records in *S*[ *f* ] is small, the empirical CDF estimation may be unreliable. In this case, we first try to use the global tool-call information to estimate *P*(τ, *f*any), which can be computed as ∑*t*∈*S* I[*t* ≤ τ]/|*S*|.

Moreover, at the very beginning of engine serving, even the global records might not be reliable. To address this, we design a minimal version of Continuum that uses a fixed TTL threshold *T*default, derived from the same cost model by assuming that the tool-call duration follows an exponential distribution with unit mean, *i.e.,* ToolCallDuration ∼ Exp(1); and the workload is fully memoryful, *i.e.,* η = 1. *T*default is then set to the optimal τ <sup>∗</sup> under this scenario.

In practice, we set a threshold *M* to decide whether to use fixed TTL, global records, or the fine-grained estimation above based on *S*[ *f* ]. That is, we use *T*default when |*S*| ≤ *K*; otherwise, we use the global records when |*S*[ *f* ]| ≤ *K*, and use the fine-grained TTL setting for the remaining cases. In our implementation, we set *K* = 100 and initialize *T* as zero.

Moreover, since agents are usually post-trained with the tools before production [\[12,](#page-12-8) [14,](#page-12-9) [50\]](#page-14-7), users can also obtain these cost-model statistics during training .

