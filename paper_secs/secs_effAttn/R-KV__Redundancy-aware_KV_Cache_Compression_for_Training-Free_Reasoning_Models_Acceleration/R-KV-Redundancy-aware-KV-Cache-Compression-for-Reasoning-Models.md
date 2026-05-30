# R-KV: Redundancy-aware KV Cache Compression for Reasoning Models

Zefan Cai<sup>1⊠</sup>, Wen Xiao<sup>2⊠</sup>, Hanshi Sun<sup>3</sup>, Cheng Luo<sup>4</sup>, Yikai Zhang<sup>1</sup>, Ke Wan<sup>5</sup>, Yucheng Li<sup>6</sup>, Yeyang Zhou<sup>5</sup>, Li-Wen Chang, Jiuxiang Gu, Zhen Dong<sup>7</sup>, Anima Anandkumar<sup>4</sup>, Abedelkadir Asi<sup>2</sup>, Junjie Hu<sup>1⊠</sup>

<sup>1</sup>University of Wisconsin - Madison <sup>2</sup>Microsoft <sup>3</sup>Carnegie Mellon University <sup>4</sup>California Institute of Technology <sup>5</sup>University of California - San Diego <sup>6</sup>University of Surrey <sup>7</sup>University of California - Berkeley https://zefan-cai.github.io/R-KV.page/ https://github.com/Zefan-Cai/R-KV

## Abstract

Reasoning models have demonstrated impressive performance in self-reflection and chain-of-thought reasoning. However, they often produce excessively long outputs, leading to prohibitively large key-value (KV) caches during inference. While chain-of-thought inference significantly improves performance on complex reasoning tasks, it can also lead to reasoning failures when deployed with existing KV cache compression approaches. To address this, we propose Redundancyaware KV Cache Compression for Reasoning models (R-KV), a novel method specifically targeting redundant tokens in reasoning models. Our method preserves nearly 100% of the full KV cache performance using only 10% of the KV cache, substantially outperforming existing KV cache baselines, which reaches only 60% of the performance. Remarkably, R-KV even achieves 105% of full KV cache performance with 16% of the KV cache. This KV-cache reduction also leads to a 90% memory saving and a 6.6× throughput over standard chain-ofthought reasoning inference. Experimental results show that R-KV consistently outperforms existing KV cache compression baselines across two mathematical reasoning datasets.

