# *B. Disaggregated Prefill and Decode*

LLM inference pipelines alternate between prefill and decode stages. Prefill kernels are typically dense matrix multiplications that scale proportionally with frequency, whereas decode kernels are memory-bound and achieve marginal speedups as frequency increases. Applying a uniform frequency policy to both phases overprovisions decode. Figure 3 illustrates the different frequency-scaling behaviors of prefill and decode for Llama-3.1-8B and Qwen3-14B. Specifically, we observe that the hotter kernels of both prefill instances exhibit larger slowdowns as the frequency is reduced, whereas the hotter kernels in the decode phase are comparatively insensitive to frequency changes.

![](_page_3_Figure_8.jpeg)

Fig. 4: Throttling due to one application interferes with others.

Recent work [49] has highlighted that splitting prefill and decode phases and collocating them on a single device is practical, achieving high throughput and low latencies, without over-provisioning GPU resources. With spatial partitioning, one portion of the GPU can serve prefill while the remainder serves decode. This also reduces data movement, as the KV cache remains in one place, saving energy spent on transfers.

With disaggregated prefill and decode on a single device, spatial DVFS can be used to apply a higher frequency to the compute resources dedicated to prefill and a lower one to those used by decode. This indicates that spatially stacking multiple models is not necessary for spatial DVFS to be valuable; a single model with split prefill and decode is sufficient.

