# CSKV: Training-Efficient Channel Shrinking for KV Cache in Long-Context Scenarios

Luning Wang1,<sup>2</sup> , Shiyao Li1,<sup>2</sup> , Xuefei Ning<sup>1</sup> , Zhihang Yuan<sup>2</sup> , Shengen Yan<sup>2</sup> , Guohao Dai3,<sup>2</sup> , Yu Wang<sup>1</sup>

> <sup>1</sup>Tsinghua University 2 Infinigence-AI <sup>3</sup>Shanghai Jiao Tong University

## Abstract

Large Language Models (LLMs) have been widely adopted to process long-context tasks. However, the large memory overhead of the key-value (KV) cache poses significant challenges in long-context scenarios. Existing training-free KV cache compression methods typically focus on quantization and token pruning, which have compression limits, and excessive sparsity can lead to severe performance degradation. Other methods design new architectures with le ss KV overhead but require significant training overhead. To address the above two drawbacks, we further explore the redundancy in the channel dimension and apply an architecturelevel design with minor training costs. Therefore, we introduce CSKV, a trainingefficient Channel Shrinking technique for KV cache compression: (1) We first analyze the singular value distribution of the KV cache, revealing significant redundancy and compression potential along the channel dimension. Based on this observation, we propose using low-rank decomposition for key and value layers and storing the low-dimension features. (2) To preserve model performance, we introduce a bi-branch KV cache, including a window-based full-precision KV cache and a low-precision compressed KV cache. (3) To reduce the training costs, we minimize the layer-wise reconstruction loss for the compressed KV cache instead of retraining the entire LLMs. Extensive experiments show that CSKV can reduce the memory overhead of the KV cache by 80% while maintaining the model's longcontext capability. Moreover, we show that our method can be seamlessly combined with quantization to further reduce the memory overhead, achieving a compression ratio of up to 95%. Code is available at <https://github.com/wln20/CSKV>.

## 1 Introduction

Large Language Models (LLMs) have been widely adopted in various natural language processing tasks, particularly those requiring long-context capabilities, such as document analysis and fact retrieval [\[6\]](#page-4-0). However, the key-value (KV) cache mechanism used in transformer-based LLMs poses significant efficiency challenges as its memory overhead grows linearly with the sequence length, often replacing the weights to be the memory bottleneck in long-context scenarios. For instance, processing a sequence with 200K tokens using LLaMA-2-7B [\[17\]](#page-4-1) results in a KV cache occupying around 100GB, compared to 14GB required for model weights. Compressing the KV cache by over 10× is necessary to fit such a sequence on a single NVIDIA RTX 4090 GPU with 24GB of memory.

Existing KV cache compression methods, mainly training-free techniques like token pruning [\[22,](#page-5-0) [12,](#page-4-2) [18,](#page-4-3) [8\]](#page-4-4) and quantization [\[10,](#page-4-5) [13,](#page-4-6) [11,](#page-4-7) [15\]](#page-4-8), struggle to maintain model performance at high compression ratios, particularly in long-context tasks. Alternatively, training-required techniques, such as MLA [\[3\]](#page-4-9)

and cache sharing [\[16,](#page-4-10) [2\]](#page-4-11), offer higher compression ratios but at the cost of significant retraining and are typically unable to be integrated with existing pre-trained models.

Inspired by MLA, we observe significant redundancy in the large channel dimensions of the KV cache, evidenced by the long-tailed distribution of singular values in the key and value caches (Details in Appendix). Experiments reveal that removing the smallest 50% of these singular values results in less than 1% average accuracy loss on the MMLU [\[5\]](#page-4-12) benchmark (from 0.458 to 0.449).

Given this redundancy, we propose CSKV, a training-efficient Channel Shrinking technique for the KV cache, designed to balances high compression ratios with low training costs. To sum up, we have the following contributions:

- To reduce the memory overhead of the KV cache while maintaining the performance, we design a bi-branch KV cache by preserving the recently used KV cache with original dimensions and reducing the dimension of the historical KV cache.
- To further improve the performance without significant training overhead, we propose an effective SVD-based initialization technique and train LLMs in a layer-wise manner by minimizing the reconstruction loss.
- Extensive experimental results demonstrate that our method can achieve an 80% KV cache compression ratio while maintaining the model's long-context capability. We further demonstrate that our method can be seamlessly combined with 4-bit quantization, showcasing its power in achieving a total compression ratio of 95%.

