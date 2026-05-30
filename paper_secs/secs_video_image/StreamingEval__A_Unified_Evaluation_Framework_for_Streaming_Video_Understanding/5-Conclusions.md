# 5 Conclusions

We propose StreamingEval, a unified evaluation protocol for video question answering models under strictly streaming, online constraints. Beyond

accuracy, we emphasize deployability-oriented metrics, including time to first token (TTFT), MaxFPS, and a constrained memory bank, and introduce a resource-budget adapter to enable fair comparisons between online models and offline VideoLLMs. Using this protocol, we evaluate 12 representative models and show that being "online" does not necessarily translate into practical deployability; moreover, changes in the memory bank and input resolution lead to corresponding shifts in accuracy. We hope StreamingEval will facilitate reproducible, system-level evaluation and inspire future work to optimize for the end-to-end online user experience.

