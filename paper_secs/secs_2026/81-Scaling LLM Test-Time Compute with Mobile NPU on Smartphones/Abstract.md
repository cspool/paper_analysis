# Abstract

Deploying Large Language Models (LLMs) on mobile devices faces the challenge of insufficient performance in smaller models and excessive resource consumption in larger ones. This paper highlights that mobile Neural Processing Units (NPUs) have underutilized computational resources, particularly their matrix multiplication units, during typical LLM inference. To leverage this wasted compute capacity, we propose applying parallel test-time scaling techniques on mobile NPUs to enhance the performance of smaller LLMs. However, this approach confronts inherent NPU challenges, including inadequate hardware support for fine-grained quantization and low efficiency in general-purpose computations. To overcome these, we introduce two key techniques: a hardwareaware tile quantization scheme that aligns group quantization with NPU memory access patterns, and efficient LUTbased replacements for complex operations such as Softmax and dequantization. We design and implement an end-to-end inference system that leverages the NPU's compute capability to support test-time scaling on Qualcomm Snapdragon platforms. Experiments show our approach brings significant speedups: up to 19.0× for mixed-precision GEMM and 2.2× for Softmax. More importantly, we demonstrate that smaller models using test-time scaling can match or exceed the accuracy of larger models, achieving a new performance-cost Pareto frontier.

<sup>∗</sup>Corresponding authors.

![](_page_0_Picture_12.jpeg)

[This work is licensed under a Creative Commons Attribution 4.0 Interna](https://creativecommons.org/licenses/by/4.0/legalcode)[tional License.](https://creativecommons.org/licenses/by/4.0/legalcode)

EUROSYS '26, April 27–30, 2026, Edinburgh, Scotland Uk © 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2212-7/26/04 <https://doi.org/10.1145/3767295.3769382>

CCS Concepts: • Computer systems organization → Single instruction, multiple data; • Human-centered computing → Ubiquitous and mobile devices; • Computing methodologies → Natural language processing.

Keywords: Neural Processing Unit, mobile device, Large Language Model

### ACM Reference Format:

Zixu Hao, Jianyu Wei, Tuowei Wang, Minxing Huang, Huiqiang Jiang, Shiqi Jiang, Ting Cao, and Ju Ren. 2026. Scaling LLM Test-Time Compute with Mobile NPU on Smartphones. In European Conference on Computer Systems (EUROSYS '26), April 27–30, 2026, Edinburgh, Scotland Uk. ACM, New York, NY, USA, [16](#page-15-0) pages. [https:](https://doi.org/10.1145/3767295.3769382) [//doi.org/10.1145/3767295.3769382](https://doi.org/10.1145/3767295.3769382)

