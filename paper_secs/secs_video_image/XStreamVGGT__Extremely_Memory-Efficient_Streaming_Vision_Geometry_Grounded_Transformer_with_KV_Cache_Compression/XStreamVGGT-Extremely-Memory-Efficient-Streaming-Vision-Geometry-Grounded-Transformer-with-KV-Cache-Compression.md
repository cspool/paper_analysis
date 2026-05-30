# XStreamVGGT: Extremely Memory-Efficient Streaming Vision Geometry Grounded Transformer with KV Cache Compression

Zunhai Su<sup>1\*</sup> Weihao Ye<sup>2\*</sup> Hansen Feng<sup>1</sup> Keyu Fan<sup>1</sup> Jing Zhang<sup>3</sup> Dahai Yu<sup>4</sup> Zhengwu Liu<sup>5</sup> Ngai Wong<sup>5</sup>

<sup>1</sup>Shenzhen International Graduate School, Tsinghua University
 <sup>2</sup>Institute of Artificial Intelligence, Xiamen University
 <sup>3</sup>China Star Optoelectronics Technology
 <sup>4</sup>TCL Corporate Research (HK) Co., Ltd.
 <sup>5</sup>Department of Electrical and Electronic Engineering, The University of Hong Kong

#### **Abstract**

Learning-based 3D visual geometry models have significantly advanced with the advent of large-scale transformers. Among these, StreamVGGT leverages framewise causal attention to deliver robust and efficient streaming 3D reconstruction. However, it suffers from unbounded growth in the Key-Value (KV) cache due to the massive influx of vision tokens from multi-image and long-video inputs, leading to increased memory consumption and inference latency as input frames accumulate. This ultimately limits its scalability for long-horizon applications. To address this gap, we propose XStreamVGGT, a tuning-free approach that seamlessly integrates pruning and quantization to systematically compress the KV cache, enabling extremely memory-efficient streaming inference. Specifically, redundant KVs generated from multi-frame inputs are initially pruned to conform to a fixed KV memory budget using an efficient token-importance identification mechanism that maintains full compatibility with high-performance attention kernels (e.g., FlashAttention). Additionally, leveraging the inherent distribution patterns of KV tensors, we apply dimension-adaptive KV quantization within the pruning pipeline to further minimize memory overhead while preserving numerical accuracy. Extensive evaluations show that XStreamVGGT achieves mostly negligible performance degradation while substantially reducing memory usage by 4.42× and accelerating inference by 5.48×, enabling practical and scalable streaming 3D applications. The code is available at https://github.com/ywh187/XStreamVGGT/.

