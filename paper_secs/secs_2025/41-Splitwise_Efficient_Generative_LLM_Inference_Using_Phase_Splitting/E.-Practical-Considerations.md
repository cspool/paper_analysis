# E. Practical Considerations

**Accuracy impact.** Splitwise does not impact accuracy since it uses lossless KV-cache transfer and does not add any randomization. It executes inference with the same parameters and state as on a single machine.

**Scalability.** Since LLM requests are much longer than typical ML requests [37], [38], they incur lower scheduling overhead for similar cluster sizes. However, the CLS may become a scalability bottleneck for large clusters. Insights from prior work on partitioned or replicated scheduling could help improve scalability [27], [61], [72] and are orthogonal to Splitwise.

Reliability and fault tolerance. If the prompt or the token machine fail, Splitwise simply restarts requests from scratch, similar to today's LLM serving systems [44], [51]. Alternatively, Splitwise could checkpoint the KV-cache generated after prompt computation into an in-memory database. To recover, Splitwise can use this cache to skip prompt recomputation, and start right away with the token phase. The KV-cache could also be checkpointed periodically during the token phase. Designing safe and efficient failure recovery is out of scope for our paper.

#### V. METHODOLOGY

