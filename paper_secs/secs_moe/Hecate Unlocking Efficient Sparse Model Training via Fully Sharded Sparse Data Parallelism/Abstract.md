# Abstract

Mixture-of-Experts (MoE) has emerged as a promising sparse paradigm for scaling up pre-trained models (PTMs) with remarkable cost-effectiveness. However, the dynamic nature of MoE leads to rapid fluctuations and imbalances in expert loads during training, resulting in significant straggler effects that hinder training performance when using expert parallelism (EP). Existing MoE training systems attempt to mitigate these effects through expert rearrangement strategies, but they face challenges in terms of memory efficiency and timeliness of rearrangement.

This paper proposes Fully Sharded Sparse Data Parallelism (FSSDP), an innovative approach that tackles the parallelization of MoE layers and potential straggler effects caused by imbalanced expert loads from a new perspective. FSSDP fully shards the parameters and optimizer states of MoE layers across devices and sparsely materializes MoE parameters from scratch in each iteration with two sparse collectives SparseAllGather and SparseReduceScatter.

We build Hecate, a high-performance MoE training system incorporating FSSDP to fully unlock its potential. Hecate introduces heterogeneous sharding, sparse materialization, and re-materialization techniques to flexibly construct efficient expert placements with low memory and communication overhead. Experiments reveal that Hecate achieves up to 3.54× speedup compared over state-of-the-art MoE training systems and consistently demonstrates improvements across model architectures and hardware environments.

