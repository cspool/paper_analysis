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

