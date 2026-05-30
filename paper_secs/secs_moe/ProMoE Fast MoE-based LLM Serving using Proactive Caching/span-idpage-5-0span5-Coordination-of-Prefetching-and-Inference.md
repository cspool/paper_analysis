# <span id="page-5-0"></span>5 Coordination of Prefetching and Inference

The prefetcher in ProMoE is responsible for fetching experts into the GPU cache based on prediction results. It consists of a worker thread and a task queue. The worker thread retrieves prefetch tasks from the queue and copies the corresponding experts into the GPU's expert cache. The task queue maintains two priority levels: lowpriority speculative prefetch tasks provided by the predictor, and high-priority precise prefetch tasks triggered by cache misses during LLM inference. The worker thread always prioritizes executing high-priority tasks over low-priority ones.

To further enhance the coordination between expert prefetching and LLM inference, ProMoE proposes several optimizations: chunked prefetching, early preemption, and reordered inference. These optimizations aim to minimize interference and maximize the overlap between prefetching and inference, as illustrated in Figure [9.](#page-5-1)

