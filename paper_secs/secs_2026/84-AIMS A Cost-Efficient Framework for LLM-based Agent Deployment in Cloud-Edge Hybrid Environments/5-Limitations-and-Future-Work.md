# 5 Limitations and Future Work

While AIMS demonstrates promising results in balancing cost-effectiveness and accuracy for AI agent deployment, several limitations warrant further research:

Application-specific accuracy requirements. In this paper, we assume a uniform similarity threshold across all applications. However, different applications may have varying accuracy requirements, which our approach does not currently account for. In future work, we aim to extend our framework to incorporate application-specific accuracy requirements, enabling a more tailored and effective decisionmaking process.

Cost model. Our current implementation uses a simplified cost model focusing primarily on LLM API costs. However, real-world deployments involve additional considerations. Future work should develop a more comprehensive model incorporating dynamic SLA-aware scheduling, resource utilization monitoring, network-aware decision making, and multi-tenant optimization.

Extensive profiling. The current system requires extensive profiling of each SLM and LLM pair, which can be timeconsuming and hard to retrain. Future work could explore more efficient profiling techniques to leverage information from previously profiled models.

Multi-model extension. While AIMS currently supports any compatible SLM-LLM pair, as demonstrated with Qwen-4B/GPT-5 and Gemma3 4B/Claude Sonnet 4, it could be extended to leverage multiple models simultaneously.

Making AIMS SLA-aware. The current AIMS design optimizes cloud cost reduction (by maximizing SLM usage) subject to an accuracy-retention target, and reports latency empirically rather than enforcing an explicit online budget. To make AIMS SLA-aware, a deployment could provide perrequest budgets (e.g., a maximum expected remote-token budget \$ and/or latency budget ). AIMS can then tighten routing thresholds or cap lookahead/convergence search depth to satisfy , while prioritizing cloud fallback for latestage or high-risk subtasks when the remaining budget is low. Implementing this requires lightweight runtime signals such as network RTT, device throughput, and queueing/resource utilization, enabling budget-aware threshold adaptation and network-aware routing.

