# **Appendix**

## A Zero Token Count Progression across 256 Experts

When scaling to 256 experts, GatePro's advantages become even more pronounced across all network layers. The increased expert pool size creates greater challenges for efficient utilization, yet GatePro consistently demonstrates superior convergence rates compared to baseline configurations. In shallow layers such as Layer 0 and Layer 7, GatePro configurations (both with and without balance loss) achieve faster reduction in unused experts, with steeper decline curves that reach near-zero unused experts more rapidly than their baseline counterparts.

The benefits are particularly striking in deeper layers, where the complexity of expert specialization typically leads to slower activation patterns. In Layer 21 and Layer 28, GatePro maintains its acceleration advantage even with the expanded 256-expert pool, demonstrating that the competitive propagation mechanism scales effectively with increased expert capacity. Notably, the combination of GatePro with balance loss achieves the most rapid convergence across all layers, suggesting optimal synergy between diversity-driven competition and load balancing mechanisms.

These results with 256 experts validate that GatePro's effectiveness is not limited by expert pool size, but rather becomes more valuable as the number of available experts increases, addressing the growing challenge of efficient expert utilization in large-scale MoE architectures.

<span id="page-14-1"></span>![](_page_14_Figure_5.jpeg)

**Figure 6** Zero token count progression across different layers during training with 256 experts. The figure shows the comparison between different training configurations across six representative layers spanning the entire network depth: Baseline w/o balance (green), GatePro w/o balance (purple), GatePro with balance (blue), and Baseline with balance (orange).

