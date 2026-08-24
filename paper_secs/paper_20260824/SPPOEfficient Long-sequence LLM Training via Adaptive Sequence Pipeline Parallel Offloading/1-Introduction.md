# 1 Introduction

Large Language Models (LLMs) have demonstrated exceptional capabilities, revolutionizing a multitude of domains including coding [16, 51], image processing [41, 42], video stream analysis [52, 65], and scientific research [2, 3]. As illustrated in Figure 1, the growing length of contextual information in these applications necessitates the training of LLMs to support increasingly longer sequences, from 4K tokens [55, 57] to 32K [22, 38], 128K [1, 13], and even millions of tokens [4, 9, 36, 46]. Training LLMs with such long sequence lengths imposes significant demands on GPU memory and computational resources, as the activation computation and memory in training scales with the sequence length. For example, training a GPT model with the size of 7 billion and a sequence length of 4 million tokens demands approximately 16,384GB of activation memory and necessitates at least 128 NVIDIA H100 GPUs for 105 hours to process 1 billion tokens.

Therefore, many system optimizations have been proposed to improve the memory or computational resource efficiency of long-sequence LLM training. However, they are

<span id="page-0-0"></span>> **[图片提取文字 (无描述)]:**
> Megatron & DeepSpeed Ours 1M Tokens 70 Hours 55.2 Hours 5 Books 700K words OOM but same time process: 2M Tokens 1408K Tokens 55.4 Hours 2 Audio Books 80.5 Hours 20 hours 4M Tokens OOM but same time process: 2186K Tokens 56.2 Hours 2 Movies 105 Hours 4 hours
![](_page_0_Figure_9.jpeg)

**Figure 1.** Performance comparisons of SPPO and SOTA training systems on extremely long sequences of total 1B tokens, using 32, 64, and 128 GPUs, respectively.

not sufficient to achieve satisfactory performance due to the following two significant gaps.

- G1: Memory reduction techniques compromise the training efficiency. Two prominent memory reduction techniques have been widely adopted to alleviate memory overhead in LLM training: activation recomputation [6] and CPU offloading [43]. Activation recomputation reduces the memory footprint by recomputing the activations of certain layers during backward propagation instead of storing them. Despite the significant memory savings, the recomputing step can account for up to one-third of the gradient computation time. This overhead is particularly pronounced in long-sequence training, substantially prolonging the overall training time. CPU offloading relieves GPU memory pressure by transferring activations between GPU and CPU memory. Prior offloading techniques [11, 62] operate the entire sequence of activations and assume that the overhead of CPU offloading can be effectively hidden by overlapping it with GPU computation. However, as the sequence length increases, the overhead of CPU offloading grows linearly, causing substantial delays in GPU computation and markedly degrading training efficiency. Also, existing CPU offloading techniques keep the activations of at least one layer in GPU memory, which limits their scalability for longer sequences (discussed in Section 2.1). Consequently, offloading the entire sequence of activations represents a coarse-grained approach, rendering current CPU offloading methods inappropriate for long-sequence training.
- G2: Distributed parallelism techniques consume excessive GPU memory and resources. Many distributed parallelism strategies [15, 39, 56, 60] have been proposed to expedite long-sequence LLM training. However, their effectiveness relies heavily on the availability of substantial GPU

memory and resources. For instance, training a GPT model with 65 billion parameters and a sequence length of 4 million results in an activation memory footprint of 80TB, necessitating **over 1,024 NVIDIA H100 GPUs** to store activations and model weights. Such enormous resource requirements significantly hinder the scalability of sequence lengths in long-sequence training.

To address the substantial activation memory consumption of long sequences, prior works [20, 30, 33, 36] have adopted sequence partitioning, where long sequences are divided into multiple subsequences for processing. Building on them, we propose to perform CPU offloading and pipeline scheduling over the subsequences, instead of the entire sequences. We expect such adaptions could bring two potential advantages for long-sequence LLM training: (1) Subsequence-level activation offloading benefits the overlapping of GPU computation and CPU offloading. (2) Subsequence pipeline scheduling can reduce pipeline bubbles and improve training efficiency. To our best knowledge, CPU offloading and pipeline scheduling at the subsequence granularity has not been systematically studied before. However, applying them in practice still faces challenges due to their adoption of fixed offloading policy and pipeline schedule.

- Inefficient fixed offloading. Existing solutions commonly adopt fixed offloading policies. The length-based offloading policy [25, 61] overlooks the computational imbalance caused by later tokens requiring more computation in the attention layers. Meanwhile, the FLOPs-based offloading policy [33] achieves computational balance but at the cost of imbalanced activation memory consumption across subsequences, causing the CPU offloading overhead to outweigh GPU computation. Furthermore, fixed subsequence offloading introduces additional tensor dependencies among subsequences, resulting in unnecessary offloading overhead.
- *Inefficient fixed pipeline schedule*. The fixed pipeline schedule policy [12, 19, 39, 45] enforces a predetermined schedule for each subsequence, disregarding variations in memory and computational demands. More subsequences could increase kernel overhead and decrease throughput, while fewer subsequences could introduce high pipeline bubbles. Even with an optimized subsequence count N, pipeline bubbles remain inevitable. For example, with p=4 and N=16, the bubble ratio reaches 3/16, highlighting inefficiencies in fixed schedules, where the total computation time includes both bubble overhead and execution time.

In this paper, we propose Adaptive Sequence Pipeline Parallel Offloading (SPPO), a novel framework for long-sequence LLM training. It can fully exploit the potential benefits of sequence partitioning while overcoming the limitations of existing offloading policies and pipeline schedules. SPPO partitions long input sequences into multiple subsequences and innovatively customizes offloading and pipeline scheduling

<span id="page-1-0"></span>> **[图片提取文字 (无描述)]:**
> 2250 32 30 1979 2000 25 1500 16 989 1000 10 8 500 312 Gen 2 A100 H100 H100 B200 Gen 3 Gen 4 Gen 5 SXM SXM NVL (b) PCIe Transfer Speed (a) GPU BF16 Tensor Core **Performance Comparison** Comparison by Generation
![](_page_1_Figure_5.jpeg)

**Figure 2.** The evolution of GPU computing power and PCIe.

to optimize their memory and computational resource efficiency. First, SPPO designs an adaptive offloading approach to efficiently overlap the offloading of the activation of a subsequence with its computation. It consists of two key components. The first is the *sequence-aware offloading* policy to mitigate imbalanced memory allocation across subsequences. This policy adaptively computes the offload ratio to maximize the overlap between the CPU offloading for the  $(i-1)^{\rm th}$  subsequence and the computation of the  $i^{\rm th}$  subsequence. The second is the *two-level activation management* strategy that retains skeletal activations (i.e., those with high access frequency) in GPU memory, thereby preserving CPU-GPU bandwidth for activations with lower access frequency.

Second, we design an adaptive pipeline schedule to pipeline the computation of one subsequence in a given layer with the computation of the previous subsequence in the subsequent layer. To strike an optimal balance between GPU utilization and pipeline efficiency, we develop a *heuristic solver* to determine the ideal number of subsequences. Combined with the offloading ratio in adaptive offloading, this enables us to further improve both memory efficiency and training efficiency for long-sequence LLM training. Nevertheless, the *heuristic solver* is not always capable of eliminating resource bubbles in certain scenarios. To handle it, we introduce a *multiplexing sequence partitioning* to deliver a fine-grained partition of adjacent subsequences, further reducing the resource bubbles without sacrificing memory efficiency.

Our contributions are summarized as follows:

- We propose and implement the SPPO framework to partition long sequences into multiple subsequences. By tailoring dedicated offloading techniques and optimizing the pipeline schedule, SPPO significantly improves both memory and computational efficiency for long-sequence LLM training.
- We introduce an adaptive offloading approach, which comprises of sequence-aware offloading and two-level activation management, to maximize the overlap between CPU offloading and GPU computation.
- We design an adaptive pipeline schedule that incorporates a heuristic solver and multiplexed sequence partitioning, enhancing training efficiency without compromising the benefits from adaptive offloading.

• We evaluate SPP0 through extensive experiments, demonstrating up to a 3.38× throughput improvement over Megatron-LM and DeepSpeed. Moreover, SPP0 enables efficient training of a 7B LLM with sequence lengths of 1M, 2M, and 4M on only 32, 64, and 128 NVIDIA Ampere GPUs.

<span id="page-2-1"></span>

| В     | batch size           | PP       | pipeline parallel size |
|-------|----------------------|----------|------------------------|
| H     | hidden dimension     | SP       | sequence parallel size |
| S     | sequence length      | N        | #subsequences          |
| $M_m$ | memory of model      | $s_i$    | subsequence i          |
|       | memory of activation | $\alpha$ | offloading ratio       |
| BW    | bandwidth            | h        | hidden state           |

**Table 1.** Notations used in this paper.

