# B. Diverse Quantization Requirements

Quantization introduces a fundamental trade-off between computational efficiency and model fidelity. While moderatebit quantization (e.g., 8-bit or 6-bit) can often maintain nearlossless accuracy with techniques such as outlier smoothing and channel shifting [38], [42], [45], aggressive low-bit quantization (e.g., 4-bit or below) amplifies rounding errors and distribution mismatch, leading to noticeable accuracy degradation [5], [11]. Our empirical results reveal that quantization behavior varies widely across both model families and format choices. As shown in Figure 2, different LLMs exhibit distinct sensitivities to precision reduction: for instance, Llama-3-8B and Qwen3 [46] models experience noticeably larger degradation than Llama-2-7B or Mistral-7B [20] when moving from W8A8 to W4A4. Even within the same architecture family, models of different scales (e.g., Llama-2-7B vs. Llama-2-70B, or Qwen3-8B vs. Qwen3-14B) respond differently, indicating that model size and parameter distributions strongly influence robustness. These variations collectively underscore that no single quantization format can achieve the optimal efficiency-accuracy balance across all models and use cases. Therefore, LLM accelerators must support bit-width flexibility, enabling efficient execution of diverse quantization methods within a unified architecture.

<span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

Fig. 3: Comparison of performance scaling under a fixed area.

### C. Limitations of Existing Architectures

The growing diversity of quantization formats exposes key limitations in current accelerator designs. Existing approaches generally fall into two categories, yet both struggle to provide efficient support across multiple bit-widths.

**Fixed-Precision Accelerators.** Most existing accelerators [13], [19], [33], [50] are built around fixed-precision datapaths, offering only a small, predetermined set of supported formats (e.g., W4A16). These designs achieve high efficiency for their targeted precisions but cannot adapt to the heterogeneous formats. This makes them unsuitable for deployment scenarios where quantization settings vary across layers, models, or hardware constraints.

Bit-Composable Accelerators. A complementary line of work introduces bit-composable architectures, where small compute units can be fused to emulate larger bit-widths. Bit-Fusion [39] is a pioneering example, enabling reconfigurable integer Processing Elements (PEs) that dynamically assemble into wider operators. This approach has since inspired many designs targeting adaptive quantization and outlier processing [15], [16], [18], [22]. However, despite their flexibility, these architectures are fundamentally limited by the quadratic  $O(n^2)$  cost of multipliers. Because partial-product generation and accumulation grow with  $n^2$ , the achievable throughput of fused PEs decreases proportionally to  $1/n^2$  as precision increases. This "dynamic tax" allows good efficiency at 4-bit, but causes severe throughput collapse when switched to 8-bit or 16-bit modes. As shown in Figure 3a, these designs become prohibitively inefficient for higher-precision formats.

Consequently, the current accelerator landscape leaves a critical architectural gap: there is no hardware that can deliver both high performance and true bit-width scalability. This gap motivates the need for a new GEMM architecture that remains efficient across diverse precisions.

### III. MOTIVATION: LINEAR SCALABILITY WITH FPMA

We observe that the inefficiency of existing bit-flexible designs originates from their reliance on multipliers, whose cost grows quadratically with bit-width. With a more efficient scaling primitive, a bit-width-flexible architecture could achieve high efficiency. A promising direction is floating-point multiplication approximation (FPMA) [17], [26], which replaces floating-point multipliers with integer adders.

