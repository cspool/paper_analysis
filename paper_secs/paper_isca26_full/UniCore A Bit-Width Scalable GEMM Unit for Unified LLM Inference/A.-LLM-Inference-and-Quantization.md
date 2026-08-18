# A. LLM Inference and Quantization

Modern Large Language Models (LLMs) are composed of stacked transformer blocks [35], [40], [48], where masked self-attention is paired with large linear layers. The majority of computation and parameter storage comes from the linear layers in feed-forward networks and attention projections [27], [35], [40], [48]. These modules are built upon general matrix—matrix multiplication (GEMM) operations, which execute large-scale dense linear mappings between activations and weights. Prior studies, including AxCore [50] and FIGNA [19], show that GEMM contributes 69–99% of LLM inference time and remains the primary cost even within attention, especially during prefill.

Quantization improves LLM inference efficiency by representing high-precision tensors with low-bit values. In general,

<span id="page-1-0"></span>![](_page_1_Figure_10.jpeg)

Fig. 2: Accuracy degradation of various LLMs under different quantization precisions. The degree of accuracy loss differs significantly across models and bit-widths.

a tensor is mapped to a compact format by dividing its values by a scale factor and rounding them to the nearest representable code, which may be an integer (INT) or a floating-point variant (FP4/FP8) [30], [28], [49]. Modern LLMs predominantly adopt group quantization, where weights or activations are partitioned into small groups (e.g., 32–128 elements), and each group receives its own scale or floating-point parameters. This grouping allows the quantizer to adapt to local distribution variation—critical for enabling aggressive low-bit formats such as W4A4 and W4A8 [25], [49]. While weight-only quantization [12], [24] is critical for reducing memory footprint, joint weight and activation (W+A) quantization directly lowers both memory and computational cost by enabling low-bit matrix manipulation [3], [36].

