# SPPO: Efficient Long-sequence LLM Training via Adaptive Sequence Pipeline Parallel Offloading

Qiaoling Chen<sup>1,2,3</sup>, Shenggui Li<sup>2</sup>, Wei Gao<sup>3</sup>, Peng Sun<sup>3,4</sup>, Yonggang Wen<sup>2</sup>, Tianwei Zhang<sup>2</sup>

<sup>1</sup>S-Lab, NTU <sup>2</sup>Nanyang Technological University <sup>3</sup>Shanghai AI Laboratory <sup>4</sup>SenseTime

#### **Abstract**

In recent years, Large Language Models (LLMs) have exhibited remarkable capabilities, driving advancements in real-world applications. However, training LLMs on increasingly long input sequences imposes significant challenges due to high GPU memory and computational demands. Existing solutions face two key limitations: (1) memory reduction techniques, such as activation recomputation and CPU offloading, compromise training efficiency; (2) distributed parallelism strategies require excessive GPU resources, limiting the scalability of input sequence length.

To address these gaps, we propose Adaptive Sequence Pipeline Parallel Offloading (SPPO), a novel LLM training framework that optimizes memory and computational resource efficiency for long-sequence training. SPPO introduces adaptive offloading, leveraging sequence-aware offloading, and two-level activation management to reduce GPU memory consumption without degrading the training efficiency. Additionally, SPPO develops an adaptive pipeline scheduling approach with a heuristic solver and multiplexed sequence partitioning to improve computational resource efficiency. Experimental results demonstrate that SPPO achieves up to 3.38× throughput improvement over Megatron-LM and DeepSpeed, realizing efficient training of a 7B LLM with sequence lengths of up to 4M tokens on only 128 A100 GPUs.

