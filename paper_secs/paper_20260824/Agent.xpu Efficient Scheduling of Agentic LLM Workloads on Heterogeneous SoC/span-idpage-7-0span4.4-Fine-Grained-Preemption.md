# <span id="page-7-0"></span>4.4 Fine-Grained Preemption

Prioritized scheduling of reactive flows is critical for user experience. However, preemption mechanism tailored to agentic LLM workloads is absent on commodity hetero-SoCs. We exploit the unified memory and our double activation buffer design to achieve lightweight context switching. Considering fine-grained, short-lived prefill kernels and moderate reactive request frequency, we adopt lazy preemption at kernel or layer boundaries without user-perceptible delay, which avoids recompute overhead in kill-based approaches [\[26\]](#page-13-15).

- 1 Copy-Free Context Switching. An inference context comprises the KV cache, progress metadata (stage, current layer, finished kernels, generated tokens), and intermediate activations. Shared memory address eliminates the need to swap the KV cache when a reactive request arrives. Our double activation buffer separates proactive and reactive activations of prefill, enabling low-latency preemption and instant restoration without data copying.
- 2 Kernel-Level Prefill Preemption. During prefill, we employ wait-then-execute preemption at kernel boundaries. Each kernel is registered with a lightweight completion callback. Upon finishing, the callback efficiently probes the reactive queue; if non-empty, it signals the prefill event loop to immediately dequeue and decompose the reactive job into kernels, thereby preempting the current proactive task. Reactive tasks do not preempt each other and run sequentially like proactive ones. Because token-wise ops are chunked, kernel execution

time is tightly bounded. For an 8B model, chunked kernels (size 256) complete within 38 ms on NPU/iGPU, while MHA with sequence length 2048 completes within 97 ms. This ensures sub-100 ms preemption latency—nearly imperceptible to users with negligible TTFT impact.

3 Iteration-Level Decode Preemption. For decode, reactive requests wait until the current iteration completes, then immediately join the next batch. Proactive jobs may be evicted under adaptive batching policies (§[4.5\)](#page-8-0) to preserve reactive latency. Since iteration-level preemption is bounded by TPOT and the only per-request context is the localized KV cache and generated tokens, it avoids expensive swapping while still guaranteeing fast response.

### <span id="page-8-0"></span>4.5 Slack-Aware Piggybacking

To preserve reactive responsiveness while preventing starvation of proactive flows, Agent.xpu employs slack-aware piggybacking, where proactive work opportunistically fills pipeline slack in prefill and decode with negligible impact on reactive task latency.

- 1 Slack Identification. The scheduler detects two forms of slack during reactive-proactive coexistence: 1) *Compute slack*, arising from idle accelerator resources. Proactive prefill can overlap with the decode pipeline whenever no reactive prefill is pending. 2) *Bandwidth slack*, during memory-bound decode stage. Proactive requests can be co-batched with reactive ones under elaborate batching rules that bound the additional latency seen by reactive tasks, as 2 .
- 2 Adaptive Decode Batching. Agent.xpu implements *reactive-first* batching (Algorithm [2\)](#page-8-4) to govern the admission of proactive tasks into the next decode iteration. Since decode latency scales with both batch size and sequence length, blind mixing of requests can violate reactive latency targets. The scheduler updates the decode batch *Bdec* by unconditionally admitting all surviving and newly arrived reactive requests. Proactive requests are then opportunistically piggybacked into the remaining slots up to a threshold *Bcap* (set as 3). Crucially, when the candidate set exceeds capacity, Agent.xpu evicts "bottleneck" proactive requests with the longest sequence lengths to preserve TPOT of the entire batch.
- 3 Starvation Prevention. While reactive requests receive priority, the scheduler implements aging mechanisms to prevent indefinite postponement of proactive requests. Proactive requests that exceed a threshold age are promoted to prevent starvation. They obtain two privileges without suspending reactive execution: 1) the scheduler reallocates iGPU to manage the overdue proactive prefill, while retaining reactive prefill (except MHA) solely on NPU, and 2) after prefill, the aged request can immediately join the decode batch, bypassing the batch size cap for a reactive-first batch.

#### Algorithm 2 Reactive-First Adaptive Decode Batching

```
Require: Current batch Bdec, new arrivals Qreact ,Qproact , threshold Bcap
Ensure: Updated Bdec for the next decode iteration
  Rreact ← {r ∈ Bdec | r.type == REACTIVE} ∪PopAll(Qreact)
  Rproact ← {r ∈ Bdec | r.type == PROACTIVE} ∪PopAll(Qproact)
  Bdec ← Rreact ▷ Unconditionally admit reactive requests
  SortDescending(r ∈ Rproact ,key = Slen(r))
  for r ∈ Rproact do
     if |Bdec| < Bcap or Rreact == ∅ then
         Bdec ← Bdec ∪ {r} ▷ Piggyback proactive requests
     else
         PushBackToQueue(r,Qproact) ▷ Evict to prevent TPOT inflation
     end if
  end for
```

