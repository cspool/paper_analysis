# <span id="page-1-0"></span>**2. Challenges**

Accelerating long sequence generation is nevertheless a non-trivial task, even built upon prior success in speculative decoding (SD). In this section, we identify critical challenges encountered in accelerating ultra-long sequence generation.

**Challenge I: Frequent Model Reloading** One fundamental speed obstacle lies in the autoregressive (AR) generation scheme of LLM. For each token, the entire model must be loaded from GPU's storage unit to the computing unit [\(Yuan et al.,](#page-16-0) [2024\)](#page-16-0), which takes significantly more time than the relatively small amount of

computation performed (as shown in Table [2\)](#page-2-1). Consequently, the primary bottleneck in generation stems from I/O memory access rather than computation.

<span id="page-2-0"></span>Table 1. Experimental results of TriForce [\(Sun et al.,](#page-15-4) [2024a\)](#page-15-4) and MagicDec [\(Chen et al.,](#page-13-3) [2024a\)](#page-13-3) with default parameters on LLaMA3.1-8b. The Batch Size of MagicDec is set to 1.

| Method   | Gen. Len. | Draft Form                           | Speed Up     |
|----------|-----------|--------------------------------------|--------------|
| TriForce | 256       | Standalone Draft                     | 1.02         |
| MagicDec | 64        | Self-Speculation<br>Standalone Draft | 1.20<br>1.06 |

<span id="page-2-1"></span>Table 2. Taking NVIDIA A100 80G and LLaMA3.1-8b as example, *MAX* refers to the scenario with a maximum context window 128K. The calculation method is from [Yuan et al.](#page-16-0) [\(2024\)](#page-16-0).

| MEMORY                                           | COMPUTATION                                   |
|--------------------------------------------------|-----------------------------------------------|
| Bandwidth: 2.04e12 B/s<br>Model Weights: 15.0 GB | BF16: 312e12 FLOPS<br>MAX Operations: 83.9 GB |
| Loading Time: 7.4 ms                             | MAX Computing Time: 0.3 ms                    |

▷ *When generating ultra-long sequence, such as 100K tokens, the GPU must reload the model weights over 100,000 times. This repetitive process poses the challenge: How can we reduce the frequency of model reloading?*

**Challenge II: Prolonged Growing of KV Cache** Previous studies, such as TriForce [\(Sun et al.,](#page-15-4) [2024a\)](#page-15-4) and MagicDec [\(Chen et al.,](#page-13-3) [2024a\)](#page-13-3) have demonstrated that, a small KV cache budget can be used during the drafting phase to reduce the time increase caused by the loading enormous KV cache. While their one-time compression strategy at the prefill stage can handle scenarios with long prefixes and short outputs, it fails to address cases involving ultra-long outputs, as the growing size of KV cache would far exceed the allocated length budget.

▷ *To dynamically manage partial KV cache within limited budget during ultra-long sequence generation, the challenge lies in determining when and how to dynamically update the KV cache.*

**Challenge III: Repetitive Content Generation** The degeneration of AR in text generation tasks — characterized by output text that is bland, incoherent, or gets stuck in repetitive loops — is a widely studied challenge [\(Holtz](#page-13-4)[man et al.,](#page-13-4) [2020;](#page-13-4) [Nguyen et al.,](#page-14-5) [2024;](#page-14-5) [Hewitt et al.,](#page-13-5) [2022\)](#page-13-5). When generating sequences of considerable length, *e.g*., 100K, the model tends to produce repetitive sentences (Figure [5\)](#page-11-0).

▷ *Since our objective is lossless acceleration and repetition is an inherent problem in LLMs, eliminating this issue is not our focus. However, it is still essential and challenging to mitigate repetition patterns in ultra-long sequences.*

