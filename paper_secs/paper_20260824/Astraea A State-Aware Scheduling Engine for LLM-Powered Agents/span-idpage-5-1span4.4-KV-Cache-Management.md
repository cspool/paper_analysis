# <span id="page-5-1"></span>**4.4 KV Cache Management**

**<sup>17</sup> return** ;

The preemptive nature of our scheduling mechanism necessitates the preservation of intermediate states (KV cache) for all preempted yet incomplete requests. Without an effective management policy, GPU memory would become a critical bottleneck, limiting the scheduler's efficacy and potentially reintroducing the HoL blocking we aim to solve. To address this, Astraea integrates an adaptive KV cache management policy designed to dynamically balance single-request latency with overall system throughput. The

policy is governed by a high-watermark threshold for GPU memory usage, creating two distinct operational modes. In low-load scenarios (i.e., below the threshold), the system defaults to a **Preserve** policy for all I/O-bound requests to prioritize low latency.

Conversely, when memory pressure is high, the objective shifts to maximizing resource utilization by minimizing memory-time waste. The system evaluates the potential waste for three candidate policies: **Preserve**, **Discard**, and **Swap**, choosing the one with the minimum cost. Following the model proposed by Infercept [1], the waste () for each policy is estimated as:

$$W_{\text{preserve}} = T_{\text{api}} \cdot C_{\text{self}} \cdot M, \tag{5}$$

$$W_{\text{discard}} = T_{\text{recompute}} \cdot C_{\text{batch}} \cdot M, \tag{6}$$

$$W_{\text{swap}} = 2 \cdot T_{\text{swap}} \cdot C_{\text{batch}} \cdot M. \tag{7}$$

Here, api, , and are the predicted durations for the API call, KV cache recomputation, and swap I/O, respectively. is the token count of the request's own cache, ℎ is the total token count of other requests that could be batched if memory were freed, and is the memory required per token's KV cache.

The system then selects the optimal strategy that results in the least memory waste:

$$Strategy = \underset{s \in \{Preserve, Discard, Swap\}}{arg min} W_s.$$
 (8)

This adaptive policy ensures high resource utilization under contentious loads while maintaining responsiveness in uncongested scenarios.

