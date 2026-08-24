# **7 Conclusion**

We introduced MEMENTO, a method that teaches language models to manage their own context by segmenting reasoning into blocks, compressing each into a dense memento, and masking completed blocks via sparse attention. Across three model families (Qwen3, Phi-4-reasoning, Olmo-3-7B-Think), MEMENTO reduces peak KV cache by 2–3× and KV AUC by up to 3.5×, translating to 1.75× higher serving throughput, while preserving strong reasoning accuracy: Qwen3-32B loses just 2.6 pp on AIME'26 and 3.5 pp averaged across five benchmark groups. The gap shrinks with scale (6.3 pp at 8B → 3.5 pp at 32B), and our initial CISPO RL result on Qwen3-8B recovers much of the remaining single-sample gap while retaining the KV savings.

A key finding is that mementos carry information from masked blocks through two complementary channels: the explicit summary text and the implicit KV representations computed while the block was still visible. Our KV ablation shows that removing this implicit channel degrades accuracy by 15 pp, distinguishing MEMENTO from methods that simply discard context after summarization.

Looking forward, we see two natural extensions: scaling the RL recipe to larger models, and applying MEMENTO to long-horizon agent tasks where agent steps form natural blocks and context windows are the primary bottleneck. We release OPENMEMENTOS (228K annotated reasoning traces) and our vLLM fork with native block masking support to facilitate further research.

