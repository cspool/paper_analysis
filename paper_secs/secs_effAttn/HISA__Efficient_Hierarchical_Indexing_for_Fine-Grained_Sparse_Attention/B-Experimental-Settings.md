# **B** Experimental Settings

We detail the experimental settings for long-context evaluations in this section. All evaluations were conducted in a **zero-shot** setting.

#### **B.1** Long-context Benchmarks

We evaluated the long-context performance using the Needle In A Haystack (NIAH) test and the LongBench benchmark. We tested two models: **DeepSeek-V3.2** and **GLM-5**. Both models were deployed using the vLLM online serving framework with **FP8** precision.

**NIAH Settings** For the NIAH experiments, we utilized a customized evaluation codebase modified from the RULER<sup>2</sup> GitHub repository. We did not apply chat templates to either model to ensure a direct assessment of their raw retrieval capabilities.

**LongBench Settings** We evaluated LongBench using the lm-eval<sup>3</sup> framework. The configurations for LongBench varied slightly depending on the model characteristics:

• Chat Template Usage: DeepSeek-V3.2 was evaluated with its standard chat template. In contrast, GLM-5 was evaluated *without* a chat template. This decision was made because using the template triggered an extended thinking process that exceeded the maximum generation length and significantly slowed down inference. Furthermore, disabling the thinking process while keeping the template resulted in inferior performance compared to not using the template at all.

<span id="page-11-1"></span><sup>&</sup>lt;sup>2</sup>https://github.com/NVIDIA/RULER

<span id="page-11-2"></span><sup>&</sup>lt;sup>3</sup>https://github.com/EleutherAI/lm-evaluation-harness

• **Concurrency Settings:** The default number of concurrent requests (num concurrent) was set to 20. However, due to Out-Of-Memory (OOM) issues specific to GLM-5 on certain tasks, we adjusted the concurrency: longbench single was run with a concurrency of 1, and longbench summary was run with a concurrency of 2.

**Fairness of Comparison** We emphasize that although the specific settings (e.g., concurrency, chat template) differ across models and tasks to accommodate their unique characteristics and hardware constraints, we ensure that the settings are **strictly aligned** when comparing different methods within the same model and task combination. This guarantees a fair and rigorous comparison.