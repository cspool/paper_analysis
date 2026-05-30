# 10 Conclusion

We propose FineEP, a novel expert parallelism strategy to achieve fine-grained load balancing in MoE. FineEP dynamically balances GPU loads within every micro-batch through token scheduling. We primarily make two optimizations in FineEP: First, we formulate the token scheduling process as a linear programming problem, which can be solved efficiently. Second, we theoretically analyze the relationship between expert placement and load balancing capacity and develop two placement strategies for different training scenarios. Finally, we propose FineMoE, an efficient MoE training system based on FineEP. Our experimental evaluation demonstrates that FineMoE achieves significant performance improvements,

<span id="page-11-1"></span><sup>4</sup>By the time we wrote this paper, LPLB was still in the early research stage and had no evaluation results. As far as we know, LPLB is the only relevant work that schedules tokens for load balancing.

with up to 47.6% end-to-end speedup compared to Megatron-LM, and almost consistently maintains optimal load balance across GPUs.

