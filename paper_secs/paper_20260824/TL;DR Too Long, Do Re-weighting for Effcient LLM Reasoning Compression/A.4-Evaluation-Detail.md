# A.4 Evaluation Detail

We use the DeepSeek-R1-Distill model and apply a temperature setting of 0.7, which is the primary recommendation in QwQ-Preview, for evaluating all models. All datasets are restricted to an 8K context window for output generation. Meanwhile, considering the relatively small sizes of the AMC and AIME datasets, we sample 8 responses per question and compute the average.

### A.5 Evaluation Framework

We use *skythought-eval*[3](#page-17-3) as the framework, which supports accelerating long CoT reasoning evaluation with vLLM. The version of vLLM we use is 0.6.3.

<span id="page-17-3"></span><sup>3</sup><https://github.com/NovaSky-AI/SkyThought>

