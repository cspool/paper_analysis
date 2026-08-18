# III. MOTIVATION

### A. Decode stage Overhead in xPU

Both the prefill and decode stages must be carefully considered in the design of LLM inference systems. However, once a system is optimized for the prefill stage—typically through a highly parallel architecture—the performance bottleneck shifts to the decode stage. Most commercial GPUs and NPUs fall into this category. Figure 3 illustrates the ratio of prefill and decode stage latencies according to token lengths on an Nvidia RTX 4090 [38]. As shown in the figure, even when the number of prefill tokens and decoding tokens is identical, the decode stage accounts for 98% of the end-to-end latency.

This imbalance is further exacerbated by the recent emergence of reasoning LLMs. Such models tend to generate significantly longer output sequences compared to conventional LLMs. Table I presents the context lengths produced on the GPQA-Diamond [50] benchmark by two models sharing the same base architecture. As shown, reasoning LLMs generate sequences that are approximately  $2.99\times$  to  $5.20\times$  longer than their non-reasoning counterparts. As a result, the decode

TABLE I
CONTEXT LENGTH COMPARISON OF VARIOUS LLMS

| Seqlen   | Llama3-8B Deepseek-R<br>Llama3-8B |      | Phi4-3.8B | Phi-3.8B-<br>Reasoning | Qwen3-4B | Qwen3-4B-<br>Thinking |
|----------|-----------------------------------|------|-----------|------------------------|----------|-----------------------|
| Prefill  | 222                               | 222  | 230       | 230                    | 232      | 232                   |
| Decoding | 1096                              | 5708 | 2739      | 8189                   | 1730     | 6348                  |

stage becomes substantially prolonged, further reinforcing its dominance in overall latency.

This outcome may change in cases of large-batch inference. Nevertheless, in the case of low-batch LLM inference acceleration, prioritizing performance improvement in the decode stage is more appropriate than enhancing computational capabilities.

# III. MOTIVATION

### A. Decode stage Overhead in xPU

Both the prefill and decode stages must be carefully considered in the design of LLM inference systems. However, once a system is optimized for the prefill stage—typically through a highly parallel architecture—the performance bottleneck shifts to the decode stage. Most commercial GPUs and NPUs fall into this category. Figure 3 illustrates the ratio of prefill and decode stage latencies according to token lengths on an Nvidia RTX 4090 [38]. As shown in the figure, even when the number of prefill tokens and decoding tokens is identical, the decode stage accounts for 98% of the end-to-end latency.

This imbalance is further exacerbated by the recent emergence of reasoning LLMs. Such models tend to generate significantly longer output sequences compared to conventional LLMs. Table I presents the context lengths produced on the GPQA-Diamond [50] benchmark by two models sharing the same base architecture. As shown, reasoning LLMs generate sequences that are approximately  $2.99\times$  to  $5.20\times$  longer than their non-reasoning counterparts. As a result, the decode

TABLE I
CONTEXT LENGTH COMPARISON OF VARIOUS LLMS

| Seqlen   | Llama3-8B Deepseek-R<br>Llama3-8B |      | Phi4-3.8B | Phi-3.8B-<br>Reasoning | Qwen3-4B | Qwen3-4B-<br>Thinking |
|----------|-----------------------------------|------|-----------|------------------------|----------|-----------------------|
| Prefill  | 222                               | 222  | 230       | 230                    | 232      | 232                   |
| Decoding | 1096                              | 5708 | 2739      | 8189                   | 1730     | 6348                  |

stage becomes substantially prolonged, further reinforcing its dominance in overall latency.

This outcome may change in cases of large-batch inference. Nevertheless, in the case of low-batch LLM inference acceleration, prioritizing performance improvement in the decode stage is more appropriate than enhancing computational capabilities.

