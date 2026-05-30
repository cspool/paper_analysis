# 1 Introduction

With the rapid evolution of large language models (LLMs), the latest models now comprise tens to hundreds of billions of parameters [\[5,](#page-11-0) [9,](#page-12-0) [12,](#page-12-1) [44\]](#page-14-1), placing severe pressure on GPU memory capacity and serving efficiency.

Quantization [\[15,](#page-12-2) [16,](#page-12-3) [45,](#page-14-2) [50\]](#page-14-3) has emerged as a promising solution for serving these huge models. By compressing tensor representations from high-precision formats (e.g., FP16/BF16) to lower-precision types (e.g., INT8/INT4), it reduces memory footprint and enables the use of low-precision Tensor Core instructions, which offer substantially higher computational throughput on modern GPUs [\[3,](#page-11-1) [24,](#page-13-0) [42\]](#page-13-1).

To enable low bit-width quantization while preserving model accuracy, recent works have prioritized computation for outliers through mixed precision quantization [\[7,](#page-12-4) [10,](#page-12-5) [11,](#page-12-6) [25,](#page-13-2) [48\]](#page-14-4). These methods leverage the activation sparsity property that widely observed in LLMs [\[2,](#page-11-2) [33\]](#page-13-3), where a small fraction of outlier values significantly exceed others in activation. By adaptively allocating higher precision to these error-sensitive regions while maintaining lower precision for majority values, mixed precision quantization achieves improved accuracy without pronounced runtime overhead.

Reducing quantization bit-widths from 8-bit to 4-bit halves memory consumption and doubles computational efficiency for model serving on modern GPUs. Although effective at 8-bit precision, existing mixed precision quantization methods fail to preserve satisfactory accuracy under 4-bit precision due to their incomplete characterization of outliers. While current methods operate along the channel dimension to mitigate outliers in specific embedding positions, our empirical analysis reveals that substantial outliers still persist after channel-wise outliers removal. These remaining outliers do not exhibit channel-wise concentration but

are distributed in a token-wise manner. They cannot be adequately represented within 4-bit data types, resulting in significant quantization error in current channel-wise mixed precision quantization methods.

In this paper, we propose Rotated Token-wise Mixed Precision Quantization (RTMPQ) to address outliers in both channel and token dimensions, thereby improving model accuracy. Instead of directly applying mixed precision quantization to channel dimension, RTMPQ first employs Hadamard rotation to suppress channel-wise outliers. The rotation smooths irregularities across channels and migrates them to the token dimension, where they are subsequently resolved through token-wise mixed precision quantization.

However, efficiently implementing RTMPQ presents great system challenges, as RTMPQ employs mixed precision quantization to address token-wise outliers rather than channelwise ones. Two unique characteristics of token-wise outliers hinder the direct application of existing channel-wise mixed precision quantization techniques:

(1) Non-reduction Dimension Computation. Existing mixed precision methods depend on the mathematical property to decompose matrix multiplication along the reduction dimension, where channel-wise outliers reside. This property allows computations at different precisions to be densified and executed separately. In contrast, token-wise outliers correspond to the non-reduction dimension in matrix multiplication and cannot benefit from this optimization. Consequently, token-wise mixed precision computation must occur in a sparse pattern, introducing challenges for efficient task mapping on GPUs and creating fundamental incompatibilities with Tensor Core instruction requirements. (2) Unpredictable Outlier Distribution. Current mixed precision methods rely on static offline activation analysis to detect channel-wise outliers, which exhibit relatively stable patterns [\[7,](#page-12-4) [43\]](#page-14-5). In contrast, token-wise outliers lack such statistical regularity and follow a significantly more unpredictable distribution. This unpredictability stems from the fact that token-wise outliers arise from specific linguistic features of words or phrases within the input natural language sentences, making their occurrence difficult to forecast. Consequently, identifying these outliers requires dynamic online detection mechanisms, which could introduce nonnegligible runtime overhead.

To address these challenges, we propose RoMeo, a LLM serving system for efficient token-wise mixed precision quantization execution, enabling accurate and performant 4-bit quantization via our RTMPQ algorithm. The core idea is to reorganize dynamic token-wise outliers to facilitate efficient parallel execution with minimal overhead. RoMeo tackles the sparse memory layout of quantized mixed precision data through a lightweight permutation-free approach that restructures data into contiguous and unified precision blocks, enabling dense matrix computation. The

system further employs fine-grained asynchronous execution to parallelize non-dependent tasks in the quantization workflow, effectively hiding quantization operations and improving hardware utilization. Additionally, RoMeo implements highly optimized cross-precision multiplication kernels with software pipelining, alongside efficient fused kernels for online outlier detection and data packing.

RoMeo is evaluated for both accuracy and efficiency across a wide range of LLMs, compared against the uniform precision quantization baseline QuaRot [\[3\]](#page-11-1) and the channel-wise mixed precision quantization baseline MixQ [\[7\]](#page-12-4). Experimental results demonstrate that RoMeo achieves higher accuracy than QuaRot at low outlier ratios, outperforms MixQ under equivalent outlier constraints, and maintains computational efficiency comparable to QuaRot.

Our main contributions can be summarized as follows:

- We conduct an empirical analysis identifying dualdimensional outliers as the fundamental bottleneck in existing mixed precision quantization methods.
- We propose a novel quantization algorithm that addresses outliers in both channel and token dimensions through rotation-based smoothing and tokenwise mixed precision computation.
- We design an efficient permutation-free approach to handle token-wise outlier computation, combining specialized kernel implementation and asynchronous concurrent execution for optimal performance.
- Comprehensive evaluation demonstrates that our system achieves superior model accuracy preservation while delivering competitive speedup compared to state-of-the-art baselines.

## 2 Background

#### 2.1 Quantized Large Language Models

Model quantization reduces the precision of tensor representations from high-precision formats (e.g., FP32, FP16) to low-precision formats (e.g., INT8, INT4), thereby decreasing memory requirements and accelerating computation through specialized low-precision hardware units [\[3,](#page-11-1) [24,](#page-13-0) [43\]](#page-14-5). This paper focuses on joint weight-activation quantization for large language models, which applies quantization to both weights and activations and is widely adopted in data center model serving deployments [\[9\]](#page-12-0).

Quantized model inference introduces two additional processes: an online quantization process that compresses input activation to lower precision, and a dequantization process restores multiplication result to the original precision.

The quantization process typically computes the maximum absolute value of the input activation and scales it to fit the target precision value range. Formally, given input activation tensor and target integer bit-width , the quantized

<span id="page-2-0"></span>![](_page_2_Figure_2.jpeg)

Figure 1. The two dimensions of activation tensor and existing mixed precision quantization methods.

activation is computed as:

$$S_X = \frac{\max(|X|)}{2^{b-1} - 1}, \quad X_Q = round\left[\frac{X}{S_X}\right]$$
 (1)

where is a half-precision scaling factor and [·] denotes rounding to nearest integer.

The quantized activation then undergoes matrix multiplication with offline-quantized weight. The result is subsequently cast back to half-precision and dequantized with scaling factors:

$$Y = S_X \times (X_Q W_Q) \times S_W \tag{2}$$

In practice, quantizations are typically conducted at finer granularity (e.g., per-row for activations and per-column for weights) to better preserve quantized model accuracy. This approach transforms scaling factors into vectors rather than scalars, requiring broadcasted element-wise multiplication during the dequantization process.

#### 2.2 Mixed Precision Quantization

Prior works have established that activations in LLMs typically exhibit long-tailed distributions, wherein a small number of outlier values substantially exceed the magnitude of the majority [\[2,](#page-11-2) [33\]](#page-13-3). Mixed precision quantization leverages this inherent property to strategically assign higher precision formats to critical computational portions while employing lower precision elsewhere. Consequently, these methods achieve superior model accuracy preservation compared to uniform precision quantization.

Before introducing existing mixed precision quantization algorithms, we first clarify the two dimensions of the activation tensor. Input prompts in LLMs are embedded into activation tensors and propagated through the model. As illustrated in Figure [1a](#page-2-0), each input token corresponds to a row within the activation tensor, representing its embedding vector. The channel dimension refers to different positions along the hidden dimension of these embeddings.

Existing mixed precision quantization methods can be categorized based on the dimension along which precision is varied, forming two primary types: tensor-wise and channelwise mixed precision quantization.

Tensor-wise approaches [\[11,](#page-12-6) [39\]](#page-13-4), shown in Figure [1b](#page-2-0), employ coarse-grained mixed precision by assigning different precisions to distinct model modules. While computations in different modules are performed at varying precisions, the precision within each individual tensor remains uniform.

Channel-wise methods [\[7,](#page-12-4) [10,](#page-12-5) [25,](#page-13-2) [48\]](#page-14-4), depicted in Figure [1c](#page-2-0), provide finer granularity by operating on individual channels within tensors. For example, MixQ [\[7\]](#page-12-4) identifies outlier channels through per-channel maximum value measurements, quantizing normal channel values to INT8 while preserving outlier channels in FP16 to better maintain model accuracy. Although channel-wise methods effectively address significant outliers concentrated in specific channels, outliers distributed along the token dimension persist. These remaining token-wise outliers continue to degrade quantization effectiveness, particularly under aggressive 4-bit quantization schemes.

