# <span id="page-0-1"></span>QuantCache: Adaptive Importance-Guided Quantization with Hierarchical Latent and Layer Caching for Video Generation

Junyi Wu<sup>1\*</sup>, Zhiteng Li<sup>1\*</sup>, Zheng Hui<sup>2</sup>, Yulun Zhang<sup>1†</sup>, Linghe Kong<sup>1</sup>, Xiaokang Yang<sup>1</sup> <sup>1</sup>Shanghai Jiao Tong University, <sup>2</sup>MGTV, Shanhai Academy

<span id="page-0-0"></span>![](_page_0_Figure_3.jpeg)

Figure 1. QuantCache is a training-free acceleration framework with an end-to-end 6.72× speedup against Open-Sora [50]. Compared with ViDiT-Q [48] and AdaCache [15], QuantCache achieves superior quality scores, demonstrating the effectiveness of our method.

## **Abstract**

Recently, Diffusion Transformers (DiTs) have emerged as a dominant architecture in video generation, surpassing *U-Net-based models in terms of performance. However,* the enhanced capabilities of DiTs come with significant drawbacks, including increased computational and memory costs, which hinder their deployment on resourceconstrained devices. Current acceleration techniques, such as quantization and cache mechanism, offer limited speedup and are often applied in isolation, failing to fully address the complexities of DiT architectures. In this paper, we propose QuantCache, a novel training-free inference acceleration framework that jointly optimizes hierarchical latent caching, adaptive importance-guided quantization, and structural redundancy-aware pruning. QuantCache

<sup>\*</sup> Equal contribution.

<sup>†</sup> Corresponding author: Yulun Zhang, yulun100@gmail.com

<span id="page-1-0"></span>*achieves an end-to-end latency speedup of 6.72*× *on Open-Sora with minimal loss in generation quality. Extensive experiments across multiple video generation benchmarks demonstrate the effectiveness of our method, setting a new standard for efficient DiT inference. The code and models will be available at [https: // github. com/](https://github.com/JunyiWuCode/QuantCache) [JunyiWuCode/ QuantCache](https://github.com/JunyiWuCode/QuantCache) .*

