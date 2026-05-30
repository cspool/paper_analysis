# Abstract

Mixed precision quantization has been adopted to accelerate large language models (LLMs) serving by leveraging high-throughput low-precision compute units in GPUs while preserving outliers in higher precision to maintain model accuracy. However, existing methods focus on mitigating single-dimensional channel-wise outliers, leading to model accuracy degradation when scaled to 4-bit precision.

In this paper, we present an algorithm-system co-design to effectively handle dual-dimensional outliers across both channel and token dimensions in LLMs. We introduce a novel rotation-based mixed precision quantization algorithm that suppresses and migrates channel-wise outliers to the token dimension. Based on this algorithm, we propose RoMeo, an efficient LLM serving system designed to overcome the unique system challenges posed by sparse computation pattern and dynamic outlier detection inherent in token-wise outlier handling. Extensive evaluations across various LLMs demonstrate that RoMeo improves quantized model accuracy by up to 5.17% compared to state-of-the-art methods QuaRot and MixQ, while maintaining efficiency comparable to uniform precision quantizations, achieving up to 2.10× end-to-end speedup over half-precision baseline. RoMeo is available at <https://github.com/thu-pacman/RoMeo>.

CCS Concepts: • Computing methodologies → Parallel computing methodologies; Natural language processing.

Keywords: Large Language Model, Mixed Precision Quantization, Algorithm-System Co-design

## ACM Reference Format:

Qihao Zhang, MingLiang Tang, Mingshu Zhai, Kinman Lei, and Jidong Zhai. 2026. RoMeo: Mitigating Dual-dimensional Outliers

![](_page_0_Picture_20.jpeg)

[This work is licensed under a Creative Commons Attribution 4.0 Interna](https://creativecommons.org/licenses/by/4.0)[tional License.](https://creativecommons.org/licenses/by/4.0)

PPoPP '26, Sydney, NSW, Australia © 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2310-0/2026/01 <https://doi.org/10.1145/3774934.3786419>

with Rotated Mixed Precision Quantization. In Proceedings of the 31st ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming (PPoPP '26), January 31 – February 4, 2026, Sydney, NSW, Australia. ACM, New York, NY, USA, [15](#page-14-0) pages. <https://doi.org/10.1145/3774934.3786419>

