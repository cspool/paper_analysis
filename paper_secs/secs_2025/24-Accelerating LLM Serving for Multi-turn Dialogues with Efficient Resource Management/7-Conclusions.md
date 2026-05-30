# 7 Conclusions

This paper proposed FlashGen to accelerate multi-turn dialogues by efficiently utilizing the compute and memory resources of GPUs and the host hardware. We analyzed that state-of-the-art LLM frameworks are inefficient in serving multi-turn conversations and identified two sources of limiting performance. Our multi-level caching technique could preserve attention keys and values in GPU, CPU, and SSD so that it minimizes the recomputation phase for prior attention KVs in multi-turn scenarios. In addition to that, our request reordering technique could effectively utilize GPU memory, minimizing the waste of GPU memory capacity.

As the number of turns per session increases (i.e., long conversations), the context size for attention KVs increases proportionally, and the prompt length is further amplified. In line with this trend, we anticipate that FlashGen-Cache and FlashGen-Sched will become more important in handling these expanded contexts efficiently.

