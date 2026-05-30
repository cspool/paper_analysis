# Abstract

With the rapid expansion of large language model inference service users, cloud computing resource costs have become a critical challenge for service providers. Although utilizing end-device resources for auxiliary inference provides new possibilities to reduce cloud computing costs, existing solutions struggle to achieve an ideal balance across multi-task accuracy, end-to-end latency, and cloud computing costs.

We present TailorLLM, a task-level collaborative end-cloud inference solution based on low-rank fine-tuning for large language models. This framework comprises two core algorithms that support offline and online optimization, respectively: (i) To reduce transmission overhead while maintaining model performance, Resource-Friendly Low-Rank Adaptation (RFLoRA) decouples pre-trained parameters into cold and hot modules, reducing trainable parameters. (ii) To ensure coverage of users' common tasks, we introduce AdapterMgr, an imitation learning-based replacement strategy that enables near-optimal dynamic management of the on-device LoRA matrix library. Finally, we implemented the TailorLLM prototype system on NVIDIA 3090 and Tesla T4 servers and thoroughly evaluated it on public task datasets. Compared to a series of baselines, TailorLLM reduces cloud resource consumption by up to 69.8% and inference latency by up to 62% while maintaining high accuracy.

CCS Concepts: • Computing methodologies→Distributed computing methodologies; Natural language generation.

<sup>∗</sup>Corresponding Authors: Ziyi Wang (wangziyi0821@gmail.com) and Lanshan Zhang (zls326@sina.com).

![](_page_0_Picture_11.jpeg)

[This work is licensed under a Creative Commons Attribution 4.0 Interna](https://creativecommons.org/licenses/by/4.0)[tional License.](https://creativecommons.org/licenses/by/4.0)

EUROSYS '26, Edinburgh, Scotland Uk © 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2212-7/26/04 <https://doi.org/10.1145/3767295.3769346>

Keywords: Large Language Models, Edge Computing, Collaborative Inference, Low Rank Adaptation, Scheduling

#### ACM Reference Format:

Zian Wang, Ziyi Wang, Haonan Jin, Jie Xing, and Lanshan Zhang. 2026. TailorLLM: Collaborative End-Cloud Inference of Large and Small Language Models Based on Low-Rank Adaptation. In European Conference on Computer Systems (EUROSYS '26), April 27–30, 2026, Edinburgh, Scotland Uk. ACM, New York, NY, USA, [16](#page-15-0) pages. <https://doi.org/10.1145/3767295.3769346>

