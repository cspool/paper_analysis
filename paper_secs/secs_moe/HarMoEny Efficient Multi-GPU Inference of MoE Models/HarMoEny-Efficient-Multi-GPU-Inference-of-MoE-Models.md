# HarMoEny: Efficient Multi-GPU Inference of MoE Models

Zachary Doucet McGill Canada Rishi Sharma EPFL Switzerland Martijn de Vos EPFL Switzerland Rafael Pires EPFL Anne-Marie Kermarrec EPFL Oana Balmau McGill

Switzerland

## Abstract

Mixture-of-Experts (MoE) models offer computational efficiency during inference by activating only a subset of specialized experts for a given input. This enables efficient model scaling on multi-GPU systems that use expert parallelism without compromising performance. However, load imbalance among experts and GPUs introduces waiting times, which can significantly increase inference latency. To address this challenge, we propose HarMoEny, a novel solution to address MoE load imbalance through two simple techniques: (i) dynamic token redistribution to underutilized GPUs and (ii) asynchronous prefetching of experts from the system to GPU memory. These techniques achieve a near-perfect load balance among experts and GPUs and mitigate delays caused by overloaded GPUs. We implement HarMoEny and compare its latency and throughput with four MoE baselines using real-world and synthetic datasets. Under heavy load imbalance, HarMoEny increases throughput by 37%–70% and reduces time-to-first-token by 34%–41%, compared to the next-best baseline. Moreover, our ablation study demonstrates that HarMoEny's scheduling policy reduces the GPU idling time by up to 84% compared to the baseline policies.

Switzerland

