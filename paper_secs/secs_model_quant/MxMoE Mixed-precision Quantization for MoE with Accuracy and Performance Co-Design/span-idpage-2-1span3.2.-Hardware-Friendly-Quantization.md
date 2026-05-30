# <span id="page-2-1"></span>3.2. Hardware Friendly Quantization

The computational efficiency of different quantization schemes varies depending on the specific characteristics of the computation (Lin et al., 2024b). The effectiveness of these schemes is fundamentally determined by the arithmetic intensity (AI), defined as the ratio of FLOPs to memory access in bytes (Williams et al., 2009). Weight-only quantization mitigates memory bandwidth limitations by reducing data transfer, whereas weight-activation quantization leverages low-precision arithmetic units to accelerate compute-intensive operations (Frantar et al., 2024). Our

roofline analysis on the Nvidia RTX-4090 (Fig. 1b) identifies distinct performance regimes: for GEMM operations with shape [m,n,k] where  $n,k\gg m$ , the arithmetic intensity simplifies to  $\mathcal{A}=m$ . For example, our analysis shows that W4A16 outperforms W8A8 when  $\mathcal{A}<83$  and W2A16 outperformes W4A4 when  $\mathcal{A}<42$ .

In addition, we observe that MoE architectures exhibit significant computational heterogeneity. For instance, our evaluation of DeepSeekV2-Lite on the HumanEval-X dataset (Zheng et al., 2023) reveals that expert activation frequencies within individual MoE blocks vary by over  $10 \times (\text{Fig. 1b})$ . Considering W8A8 and W4A16, the computational heterogeneity implies that within a single MoE block, operations that are suited for W8A8 and W4A16 coexist simultaneously, as predicted by the roofline model. This characteristic, distinct from dense LLMs, suggests that by strategically combining quantization schemes across experts, we can potentially achieve better performance than uniform precision quantization.

From a hardware-friendly perspective, it is possible to select the most efficient quantization schemes based on computational characteristics. For example, W2A16 generally outperforms W4A16, and W4A4 outperforms W8A8. However, as discussed in Section 3.1, the allocation of bitwidth plays a critical role in model accuracy. Simply optimizing for performance may degrade model accuracy, while focusing exclusively on accuracy can result in suboptimal performance.

<span id="page-3-0"></span>![](_page_3_Figure_1.jpeg)

Figure 2. Comparison of the computation throughput of low-precision MoE block. W4 denotes 4-bit per-channel symmetric weight-only quantization, while W8A8 refers to 8-bit per-channel symmetric weight-activation quantization. The problem consists of 60 experts, each with a shape of [N,K]=[2816,2048] (from Qwen2\_MoE1.5), with each token activating 4 experts. The total number of input tokens is set to 512.

#### <span id="page-3-3"></span>3.3. Algorithm-System Co-Design and Challenges

The analysis presented raises a fundamental question: can we design a quantization scheme specifically tailored to MoE models that effectively balancing model accuracy and computational speed? Our findings suggest that heterogeneous quantization sensitivity at the linear-block level within MoE models significantly affects accuracy, while hardware resources determine the maximum achievable computational speed. Moreover, the variation in expert activation frequencies introduces divergent computational demands, which is crucial for identifying the optimal quantization strategy. Therefore, an effective mixed-precision quantization scheme must take into account three key factors: 1) parameter sensitivity, 2) expert activation frequencies, and 3) hardware characteristics.

For a given mixed-precision scheme, system-level support is necessary to translate theoretical performance improvements into actual wall-clock time reductions. While numerous works have optimized low-precision operators for dense LLMs (Zhao et al., 2024; Lin et al., 2024b), these approaches are often ill-suited for the MoE models. To demonstrate this, we leverage two widely used low-precision kernels: HQQ and VLLM-Marlin-MoE to build a MoE blocks with 4-bit weight, as shown in Fig. 2. HQQ, which does not fuse dequantization, significantly underperforms the fullprecision baseline. Marlin (Frantar et al., 2024) is a highly optimized W4A16 kernel achieves SOTA performance for W4A16 GEMM. The VLLM community (Kwon et al., 2023) adopts Marlin to build the VLLM-Marlin-MoE kernel. It sequentially invokes the Marlin kernel multiple times for each expert, which results in suboptimal GPU utilization. These shortcomings intensify when introducing mixed-precision configurations, as existing kernel designs lack the architectural flexibility to handle precision-heterogeneous expert computations efficiently.

We propose MxMoE to address above challenges. Mx-MoE tightly couple 1) a hardware-aware bitwidth allocation scheme that respects parameter sensitivity and activation patterns with 2) a specialized computation engine that eliminates kernel launch overhead and enables parallel mixed-precision expert execution.

## 4. Method

#### 4.1. Overview

<span id="page-3-1"></span>![](_page_3_Figure_10.jpeg)

Figure 3. Overview of MxMoE.

The workflow of MxMoE is illustrated in Fig. 3. First, MxMoE's allocator takes statistical data specific to the MoE model as input, navigating the interplay between parameter sensitivity, expert activation patterns, and hardware resources. It then optimizes within this multidimensional design space to identify the mixed-precision quantization scheme. Next, MxMoE generates a mixed-precision Group-GEMM kernel tailored to the identified scheme, efficiently orchestrating linear blocks with varying precision. During runtime, the automatically generated tile scheduler maps mixed-precision computation tasks to hardware in a load-balanced manner, fully parallelizing the MoE block.

We begin by formalizing the impact of quantization and expert activation frequency on model accuracy and execution performance, providing a comprehensive understanding of the design space for mixed-precision scheme. Subsequently, we present our solution and discuss the system-level support required for mixed-precision MoE blocks.

#### 4.2. Hardware-Aware Bitwidth Allocation

For a Given M-layer MoE model with parameter W and input X, the objective of bitwidth allocation in MxMoE is to minimize both the perturbation introduced by quantization and the total execution time of all MoE blocks in the model:

<span id="page-3-2"></span>
$$\min(\mathcal{L}(W,X) - \mathcal{L}(W_q, X_q))^r \cdot (\sum_{i=1}^{M} T_i)^{1-r}$$
 (3)

where r is a hyper-parameter balancing the trade-off between model accuracy loss and execution time. In this study, we adopt the setting presented in (Choukroun et al., 2019) which assumes a positive correlation between the change in the intermediate output of the quantized model and the final output. Therefore, minimizing the intermediate output loss leads to minimize the loss item in Eq.3. Furthermore, since the model is executed sequentially, minimizing the execution time of individual MoE blocks contributes directly to reducing the overall execution time item in Eq.3. Thus, the objective simplifies to minimizing the output loss L and execution time T of a single mixed-precision MoE block:

$$\min L^r \cdot T^{1-r} \tag{4}$$

To further detail the formulation, we decompose the terms L and T systematically.

#### <span id="page-4-0"></span>4.2.1. QUANTIZATION LOSS FORMULATION

Let S denote the set of hardware-supported quantization schemes (e.g., W6A6 is still unsupported by most existing hardware, while FP8 is supported on Nvidia RTX-4090 but not A100). For an MoE block comprising E experts, each containing N linear blocks (typically N=3 for modern architectures, corresponding to gate\_proj, up\_proj, and down\_proj). The composite loss aggregates individual quantization effects as:

$$L = \sum_{i=1}^{E} \sum_{j=1}^{N} \sum_{k=1}^{|S|} \Delta_{i,j,k} \cdot x_{i,j,k}$$
 (5)

where  $x_{i,j,k} \in \{0,1\}$  denotes the binary selection variable for applying the k-th quantization scheme to the j-th linear block in expert i. The perturbation coefficient  $\Delta_{i,j,k}$  quantifies the output distortion when using scheme k, computed via Euclidean distance between full-precision (O) and partially quantized (Ô) MoE block outputs:

<span id="page-4-2"></span>
$$\Delta = \left\| \hat{\mathbf{O}} - \mathbf{O} \right\|_{2} \tag{6}$$

For practical estimation, we employ a small calibration set (e.g., 128 samples from WikiText2) to compute  $\Delta_{i,j,k}$  values. Each linear block in expert i is sequentially quantized with scheme  $k \in S$ , with the corresponding output perturbation statistically estimated across calibration samples.

