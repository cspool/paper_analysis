# 5 CONCLUSIONS

We present an inference-time method for MoE LLMs that steers behavior by selectively activating or suppressing experts identified through activation differences in paired examples. This weightpreserving control improves grounding and safety, revealing that experts encode behavior-relevant signals beyond domain or lexical traits. Yet, the same mechanism exposes vulnerabilities: our attacks reveal exploitable unsafe experts and routing paths despite post-training alignment tuning. Future work includes expanding steering to more behaviors, enabling dynamic token-aware steering, and developing alignment methods that ensure all experts and routes are made safe and reliable.

<span id="page-8-1"></span><sup>5</sup> Following [Zheng et al.](#page-16-3) [\(2024\)](#page-16-3) we use GCG prompts optimized for LLaMA-2-Chat [\(Zou et al.,](#page-16-5) [2023\)](#page-16-5).

<span id="page-8-2"></span><sup>6</sup>Top 1 configuration based on [Jiang et al.](#page-11-11) [\(2024b\)](#page-11-11)

<span id="page-8-3"></span><sup>7</sup>AIM in Figure [3](#page-7-1) is applied on StrongREJECT, and in Table [2](#page-8-0) on 50 AdvBench prompts.

