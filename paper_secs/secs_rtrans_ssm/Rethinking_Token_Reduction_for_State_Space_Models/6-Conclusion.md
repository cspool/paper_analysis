# 6 Conclusion

In this paper, we introduced a unified post-training token reduction method for SSM architectures like Mamba. We addressed the limitations of existing token reduction techniques by combining token importance and similarity to create a fine-grained reduction strategy. Our method includes multiple design choices for effective intra-layer optimizations. Experiments show significant reductions in computational demands and peak memory usage, while maintaining competitive accuracy, outperforming baseline methods on benchmarks.

<span id="page-8-17"></span>![](_page_8_Figure_0.jpeg)

Figure 4: Comparison of the generation throughput between different FLOPS reduction ratios for Mamba-2.8B and Mamba-2-2.7B.

