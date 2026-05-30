# VIII. DISCUSSION

## A. Split Prefill and Decoding Node

Splitwise [43] proposed an LLM inference system by dividing nodes into prefill (prompt) and decoding (token) nodes. Based on the importance of high computing power for the prefill stage and memory bandwidth for the decoding stage, Splitwise suggested a cost-effective system by deploying suitable hardware across prefill and decoding nodes. This approach benefits in low tail latency of TBT compared to non-split systems, as no mixed stages are involved in token generation within the decoding nodes.

However, split systems show lower throughput than nonsplit systems due to the underutilization problems and the wasted memory capacity due to weight duplication (see Fig. 16). The utilization of prefill and decoding nodes varies across batch sizes, input, and output sequence lengths. One of the prefill or decoding nodes would suffer from low utilization unless it targets a specific scenario. For cloud service providers, managing separate devices for prefill and decoding stages and reconfiguring the system to prevent underutilization for each target scenario is highly burdensome. Further, the split approach incurs memory weight duplication, limiting batch size due to wasted memory capacity and thus degrading throughput compared to a non-split system.

