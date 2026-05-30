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

