# <span id="page-6-0"></span>4.2 Computation Reduction

Quantizing weights using FC can significantly reduce computation complexities in models. By further quantizing activations and BN parameters to integers, the expensive floating-point multiplications and additions in convolutions can be replaced with simple bit-shift operations and integer additions. This can be realized with much faster software or hardware implementations, which directly translates to energy saving and much lower latencies in low-power devices. In Table 3, we evaluate the impact on accuracies by progressively applying FQ on weights, and integer quantizations on activations and batch normalization (BN) parameters. It is notable that the final fully quantized model achieve similar accuracies to LQ-Net.

<span id="page-7-1"></span>Table 3: Comparison of the original ResNet-18 with successive quantizations applied on weights, activations and BN parameters. Each row denotes added quantization on new components.

| Quantized                     | Top-1 | ∆     | Top-5 | ∆     |
|-------------------------------|-------|-------|-------|-------|
| Baseline                      | 68.94 | —     | 88.67 | —     |
| + Weights (5-bit FQ)          | 68.36 | -0.58 | 88.45 | -0.22 |
| + Activations (8-bit integer) | 67.89 | -1.05 | 88.08 | -0.59 |
| + BN (16-bit integer)         | 67.95 | -0.99 | 88.06 | -0.61 |

<span id="page-7-2"></span>Table 4: Computation resource estimates of custom accelerators for inference assuming the same compute throughput.

| Configuration                         | #Gates  | Ratio |
|---------------------------------------|---------|-------|
| ABC-Net (5 bases, or 5 bits)          | 806.1 M | 2.93× |
| LQ-Net (2 bits)                       | 314.4 M | 1.14× |
| Shift quantization (3 bits, unsigned) | 275.2 M | 1.00× |
| FQ (5 bits)                           | 275.6 M | 1.00× |
| FQ (5 bits) + Huffman                 | 276.4 M | 1.00× |

Figure [4](#page-8-1) shows an accelerator design of the dot-products used in the convolutional layers with recentralized quantization for inference. Using this, in Table [4](#page-7-2) we provide the logic usage required by the implementation to compute a convolution layer with 3 × 3 filters with a padding size of 1, which takes as input a 8 × 8 × 100 activation and produce a 8 × 8 × 100 tensor output. Additionally, we compare FQ to shift quantization, ABC-Net [\[15\]](#page-9-2) and LQ-Net [\[23\]](#page-10-4). The #Gates indicates the lower bound on the number of two-input logic gates required to implement the custom hardware accelerators for the convolution, assuming an unrolled architecture and the same throughput. Internally, a 5-bit FQ-based inference uses 3-bit unsigned shift quantized weights, with a minimal overhead for the added logic. Scaling constants *σ*<sup>−</sup> and *σ*<sup>+</sup> are equal and thus can be fused into *α<sup>l</sup>* . Perhaps most surprisingly, a 5-bit FQ has more quantization levels yet uses fewer logic gates, when compared to ABC-Net and LQ-Net implementing the same convolution but with different quantizations. Both ABC-Net and LQ-Net quantize each weight to *N* binary values, and compute *N* parallel binary convolutions for each binary weight. The *N* outputs are then accumulated for each pixel in the output feature map. In Table [4,](#page-7-2) they use *N* = 5 and 2 respectively. Even with the optimal compute pattern proposed by the two methods, there are at least *O*(*MN*) additional high-precision multiply-adds, where *M* is the number of parallel binary convolutions, and *N* is the number of output pixels. This overhead is much more significant when compared to other parts of compute in the convolution. As shown in Table [4,](#page-7-2) both have higher logic usage because of the substantial amount of high-precision multiply-adds. In contrast, FQ has only one final learnable layer-wise scaling multiplication that can be further optimized out as it can be fused into BN for inference. Despite having more quantization levels and a higher CR, and being more efficient in hardware resources, the fully quantized ResNet-18 in Table [3](#page-7-1) can still match the accuracy of a LQ-Net ResNet-18.[2](#page-7-3)

