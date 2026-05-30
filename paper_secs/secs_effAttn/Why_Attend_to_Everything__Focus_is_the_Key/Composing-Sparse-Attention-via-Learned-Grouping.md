# Composing Sparse Attention via Learned Grouping

Hengshuai Yao1,2 Xing Chen<sup>1</sup> Ahmed Murtadha<sup>1</sup> Jin Li<sup>1</sup> Yasin Abbasi Yadkori<sup>1</sup> Shuai Shao<sup>1</sup> Changling Liu<sup>1</sup> Guan Wang<sup>1</sup> Mingli Yuan<sup>1</sup> William Chen<sup>1</sup> Sen Song<sup>3</sup> <sup>1</sup>Sapient Intelligence <sup>2</sup>University of Alberta <sup>3</sup>Tsinghua University April 30, 2026

#### Abstract

Efficient attention methods reduce the O(n 2 ) cost of transformers, but existing approaches degrade perplexity, downstream accuracy, or both when retrofitted onto pretrained models. We introduce Focus, which instead learns which token pairs matter. A small set of learnable centroids (as few as 148K parameters) is added to each attention layer. These centroids act as gates, allowing only same-group token pairs to attend to each other at long range. Focus is composable with any pretrained model: only the centroids are trained; all original weights stay frozen.

Our experiments show that composing Focus onto pretrained models yields zero degradation on downstream benchmarks—from 124M to 70B parameters, across five attention architectures. Surprisingly, sparse attention surpasses full attention at 124M (30.3 vs 31.4 PPL) and matches it when trained from scratch at 7B (13.82 vs 13.89 PPL). Focus is also fast: top-k group membership yields 2× speedup with better quality than the pretrained model. With our FlashAttention decomposition, Focus reaches 8.6× speedup at 1M tokens with no custom kernels.

