# *B. GPU Sharing Opportunity*

An LLM instance's memory footprint primarily consists of model weights and KV-cache. While the weights are fixed, the KV-cache is dynamic with request concurrency and token length. To capture realistic memory usage, we sample token lengths from Azure LLM Trace [54]. Since it lacks multi-LLM invocation patterns, following ServerlessLLM [26], we fire requests based on Azure Serverless Trace [61].

Figure 9 shows the memory usage of the 7B and 13B model under real-world workloads on 4 A100-80GB GPUs. The label "P99, 7B" represents mapping the Llama-2-7B model to the top 1% most frequently invoked function in the Azure Trace. Since each instance occupies 1 GPU, a footprint exceeding 80GB implies that multiple instances are created.

For 7B and 13B LLMs, they need at least 14GB and 26GB of memory, respectively, corresponding to the model weights, regardless of the workload. Under the top 1% workload, memory footprint can peak at 169GB (7B) and 263GB (13B), due to bursts of over 128 concurrent requests (shown in Figure 12), necessitating exclusive use of GPUs. However, even under the top 1%, more than 50% of the time, memory footprint remains below 17GB (7B) and 43GB (13B).

Takeaway. One model's memory footprint remains low in most cases. Given that GPUs like A100 feature 80GB of memory, LLMs can be co-located under serverless workloads.

