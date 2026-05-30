# MICROMIX: EFFICIENT MIXED-PRECISION QUANTI-ZATION WITH MICROSCALING FORMATS FOR LARGE LANGUAGE MODELS

Wenyuan Liu<sup>1</sup> , Haoqian Meng<sup>1</sup> , Yilun Luo<sup>1</sup> , Yafei Zhao<sup>1</sup> , Peng Zhang1<sup>∗</sup> , Xindian Ma<sup>1</sup> <sup>1</sup> College of Intelligence and Computing, Tianjin University, Tianjin, China {lwy2020, typedef, lyl2023, zhaoyf, pzhang, xindianma } @tju.edu.cn

## ABSTRACT

Quantization significantly accelerates inference in large language models (LLMs) by replacing original high-precision matrices with low-precision counterparts. Recent advances in weight-activation quantization have primarily focused on mapping both weights and activations to the INT4 format. Although the new FP4 Tensor Cores in NVIDIA's Blackwell architecture offer up to 4× speedup over FP16, existing INT4-based kernels fail to fully exploit this capability due to mismatched data formats. To bridge this gap, we propose MicroMix, a co-designed mixed-precision quantization algorithm and GEMM kernel based on Microscaling (MX) data formats. Tailored for the Blackwell architecture, the MicroMix kernel supports arbitrary combinations of MXFP4, MXFP6, and MXFP8 channels, and produces BFloat16 outputs. To achieve a favorable trade-off between accuracy and efficiency for each linear layer, we introduce quantization thresholds that identify activation elements where lower-precision formats (MXFP4 or MXFP6) incur excessive quantization error. Our algorithm selectively allocates higher-precision channels to preserve accuracy while maintaining compute efficiency. On the Llama and Qwen model families, MicroMix achieves near-FP16 performance across diverse downstream tasks with an average precision of 5 bits. In particular, Qwen2.5-32B-Base, Coder and Math exhibit lossless accuracy on zero-shot, code generation, and mathematical reasoning benchmarks. In addition, on RTX 5070Ti laptop and RTX 5090 GPUs, our kernel achieves 2.29-3.38× acceleration compared to TensorRT-FP16. Our code is available at <https://github.com/lwy2020/MicroMix>.

### 1 INTRODUCTION

In recent years, large language models (LLMs) have demonstrated remarkable performance across a wide range of tasks [\(Vaswani et al., 2023;](#page-13-0) [Brown et al., 2020\)](#page-10-0). However, these capabilities come with substantial computational and energy costs. To mitigate this, quantization techniques replace highprecision matrix multiplications with more efficient low-bit alternatives [\(Yao et al., 2022;](#page-13-1) [Xiao et al.,](#page-13-2) [2024\)](#page-13-2), significantly improving LLM inference speed. Quantization techniques are broadly classified into weight-only and weight-activation approaches. Weight-only methods [\(Lin et al., 2024b;](#page-12-0) [Frantar](#page-11-0) [et al., 2023;](#page-11-0) [Yang et al., 2025\)](#page-13-3) have substantially mitigated the precision loss associated with 4-bit weights and 16-bit activations (W4A16). In parallel, weight-activation methods [\(Dettmers et al.,](#page-11-1) [2022;](#page-11-1) [Xiao et al., 2024\)](#page-13-2) suppress activation outliers effectively, enabling accurate 8-bit quantization of both weights and activations (W8A8). More recently, mixed-precision and rotation-based quantization algorithms [\(Ashkboos et al., 2024;](#page-10-1) [Zhao et al., 2024\)](#page-13-4) have pushed the frontier further to W4A4, achieving strong performance on downstream tasks.

Despite these advances, two key bottlenecks continue to restrict the kernel-level efficiency of INT4 based quantization: (1) The widely adopted group-wise integer quantization scheme requires dequantizing each integer group to floating-point values followed by partial summations. This procedure is executed on slower CUDA Cores, as INT8 Tensor Cores only support INT32 accumulation.

<sup>∗</sup>Corresponding Author: Peng Zhang

(2) NVIDIA's latest Blackwell architecture introduces FP4 Tensor Cores that offer up to 4× higher throughput than FP16 and 2× higher than FP8 or INT8. However, existing INT-based quantization kernels are incompatible with these new tensor cores and thus fail to leverage their full potential. As a result, significant room remains for optimizing quantization kernel throughput on the Blackwell architecture.

<span id="page-1-0"></span>![](_page_1_Figure_2.jpeg)

Figure 1: (a) MicroMix reorders channels and allocates different bit-widths accordingly. (b) The quantization thresholds T(4) and T(6) partition elements into three groups based on their quantization error magnitude. (c) MicroMix consistently achieves lower quantization error across all layers.

In this paper, we propose MicroMix, a mixed-precision quantization framework based on Microscaling (MX) data formats, featuring a co-designed algorithm and kernel. The key components of MicroMix are as follows:

- (1) Flexible bit-width ratios (4, 6, and 8 bits). To balance efficiency and accuracy, MicroMix assigns customized ratios of three precision levels to each linear layer. The quantization kernel supports multiple Microscaling formats (MXFP8, MXFP6, MXFP4) and arbitrary mixing ratios. By leveraging CUTLASS GEMM, we instantiate optimized matrix multiplication kernels tailored to specific data types and problem sizes. In addition, dequantization operations are deeply fused into MMA instructions, introducing negligible overhead on Blackwell Tensor Cores.
- (2) Low-error precision assignment strategy. We propose a bit allocation algorithm that adapts to input distributions from the perspective of quantization error. The key idea is to ensure that the quantization error of lower-bit formats remains below the upper bound of higher-precision formats. To this end, we define explicit quantization thresholds for MXFP4 and MXFP6: elements exceeding the threshold at a given bit-width are reassigned to higher-precision formats (see Figure [1\(](#page-1-0)b)). This formulation introduces explicit outlier thresholds for MXFP4 and MXFP6, addressing a limitation of prior work. As a result, MicroMix significantly reduces the quantization error induced by MXFP4, as shown in Figure [1\(](#page-1-0)c).
- (3) Efficient reorder-and-quantize operation. Since adjacent channels may be assigned different bit-widths, channels of the same precision need to be reordered into the same block. Without reordering, applying mixed-precision quantization directly results in irregular memory access and considerable overhead. To address this, MicroMix integrates the reordering step into the quantization kernel (Figure [1\(](#page-1-0)a)), enabling high-throughput quantization across heterogeneous precision levels with negligible additional latency.

We evaluate MicroMix on multiple downstream tasks, including zero-shot and few-shot learning, language modeling, code generation and mathematical reasoning. Across various Llama and Qwen models, MicroMix generally maintains at least 98% FP16 accuracy on zero-shot, code and math benchmarks, achieving comparable to or better than state-of-the-art baselines. In particular, MicroMix achieves near-FP16 performance on Qwen2.5-32B models (Base and Coder) with an average bits about 5.2. For efficiency analysis, we evaluate the MicroMix kernel on RTX 5070Ti laptop, RTX 5090 and RTX PRO 6000 GPUs. Compared with TensorRT-FP16, MicroMix achieves a kernel-level speedup of 2.45-2.93× on the RTX 5070Ti laptop and 2.29-3.38× on the RTX 5090. When integrated into the Transformer architecture, MicroMix achieves 1.98-2.02× higher end-toend compared to FP16. For end-to-end efficiency, MicroMix delivers at least 1.82 times higher decoding throughput than INT4 baselines on RTX PRO 6000.

#### 2 Preliminary and Motivations

#### 2.1 PRELIMINARY

Given an activation tensor X and a weight tensor W, quantization approximates the original matrix multiplication with a low-precision computation:

$$Y = XW \approx Q(X)Q(W) \cdot s_X s_W, \quad Q(X) = round(\frac{X}{s})$$
 (1)

where  $s_{\boldsymbol{X}}$  and  $s_{\boldsymbol{W}}$  are the scaling factors of  $\boldsymbol{X}$  and  $\boldsymbol{W}$  respectively.  $\forall X_j \in \boldsymbol{X}$ , the quantization error for  $X_j$  is defined as:

<span id="page-2-1"></span>
$$E(X_j) = |X_j - Q(X_j)| = |X_j - round(\frac{X_j}{s})s| = \gamma \cdot s$$
 (2)

where  $\gamma = |round(X_j/s) - X_j/s|$  is the rounding error. In Appendix C.1, we further analyze the relationship between quantization error and model accuracy. Empirically, we observe that model accuracy remains close to the FP16 baseline as long as the quantization error is constrained within a specific threshold. However, once the quantization error exceeds this threshold, accuracy degrades rapidly. For recent LLMs, INT8 quantization typically remains within the high-accuracy region, whereas INT4 often lies near the onset of significant accuracy degradation.

Microscaling data formats (MX) are advanced numerical formats designed for deep learning. The basic unit of MX is a block of size k, consisting of k scalar elements  $\{X_j\}_{j=1}^k$  and a single shared scaling factor s in E8M0 (Darvish Rouhani et al., 2023a). Recently, DeepSeek V3.1 (DeepSeek-AI, 2024) was trained using the UE8M0 FP8-scaled data format for both model weights and activations, ensuring compatibility with microscaling formats. Given a FP16 tensor  $\boldsymbol{X} \in \mathbb{R}^{L \times I}$ , quantization to MXFP8/MXFP6/MXFP4 first partitions  $\boldsymbol{X}$  into blocks of 32 elements  $\{\boldsymbol{X}_i\}_{i=1}^N, N = \frac{L \cdot I}{32}$ , then applies per-block symmetric quantization for  $\forall X_j \in \boldsymbol{X}_i$  as follows:

$$Q(X_j) = round(\frac{X_j}{s}), s = 2^{\lfloor \log_2(\max(|\mathbf{X}_i|)) \rfloor - b}$$
(3)

where  $round(\cdot)$  denotes rounding to the nearest MXFP value and the exponent bias b is format-specific (see Appendix B for more details).

#### 2.2 MOTIVATIONS

The primary motivations of this paper stem from addressing the limitations in current quantization methods and their corresponding kernels.

Motivation 1: Adaptive Mixed-precision Allocation for Diverse Activation Distributions. Existing mixed-precision quantization methods such as Atom (Zhao et al., 2024), employ a fixed number of high-precision channels across all layers. This uniform allocation fails to account for the heterogeneous activation distributions observed in different layers (see Figure 2). Specifically, layers with larger activation values across channels require more highprecision channels to reduce the quantization error. Consequently, directly applying current fixed-allocation mixed-precision algorithms to MX formats leads to a noticeable degradation in accuracy (see Table 11 in Appendix D.3). To overcome this, we propose a novel strategy that

<span id="page-2-0"></span>![](_page_2_Figure_15.jpeg)

Figure 2: Channel-wise mean values of three activation tensors from Llama3.1-8B, with outlier channels reordered to the end. Compared to prior methods, MicroMix assigns a larger portion of channels to higher-precision formats and applies layer-wise adaptive precision ratios across all linear layers.

flexibly allocates the number of 4, 6, and 8-bit channels per layer. This adaptive approach ensures that all linear layers consistently maintain low errors, thereby improving model accuracy.

Motivation 2: Leveraging FP4 Tensor Cores for Enhanced Kernel Efficiency. Current INT-based kernels, exemplified by Atom and QuaRot (Ashkboos et al., 2024), require dequantization on CUDA Cores because INT8 Tensor Cores only produce INT32 partial sums. The dequantization process on CUDA Cores limits the performance of these INT kernels (Lin et al., 2025b). In stark contrast, FP4 matrix multiplication allows for direct dequantization on FP4 Tensor Cores, leading to a significant improvement in computational efficiency. This fundamental advantage highlights the critical need for developing next-generation mixed-precision methods with kernels specifically designed for FP formats.

Motivation 3: Quantization Error Management through Adaptive Thresholding for Outliers. Quantization error, inherent to the format conversion between original activations X and their quantized representations Q(X), cannot be entirely eliminated. Therefore, it is critical to ensure this error remains within an acceptable bound. While prior works have introduced various techniques, such as smoothing, rotation, and clipping, to mitigate the impact of outliers in activations. There is still a notable gap in research concerning the precise threshold above which outliers should be constrained for MXFP4 and MXFP6. In this paper, we define specific quantization thresholds for MX. Elements exceeding these defined thresholds will be preferentially stored in higher bit-width, thereby effectively minimizing quantization error and maintaining high model fidelity.

### 3 METHOD

To address the accuracy degradation observed in INT4 quantized models on downstream tasks, prior work has explored various solutions. However, post-training quantization for Microscaling (MX) formats remains underexplored. Leveraging the inherent flexibility of multiple MX data formats, we propose MicroMix, a novel co-designed mixed-precision quantization algorithm and kernel.

#### 3.1 ALGORITHM

In MicroMix, the activation tensor channels are partitioned into three groups,  $G_4$ ,  $G_6$ , and  $G_8$ , which are quantized to MXFP4, MXFP6, and MXFP8, respectively. The corresponding weight channels are quantized to the same bit-width as their activation counterparts.

**Reducing Quantization Error through Permutation.** Due to the limited bit-width, the quantization error of MXFP4 or MXFP6 cannot, in general, be lower than that of INT8. Given a token  $X \in \mathbb{R}^I$ , our key idea is to constrain the quantization error of MXFP4 and MXFP6 such that it remains within the upper bound of the error introduced by INT8:

$$E(X)_{MXFP\{4,6\}} \le \overline{E}(X)_{INT8}, \quad \forall X \in G_{\{4,6\}}$$

$$\tag{4}$$

According to Equation 2, the reduction of quantization error in MX primarily depends on lowering the maximum value within each block of  $X_j$ . A straightforward approach is to group large values into the same blocks while keeping smaller values together. To achieve this, we introduce a permutation  $\sigma$  that rearranges the elements of X in ascending order:

$$\sigma: X \to \sigma(X) \tag{5}$$

**Defining Quantization Threshold for Partitioning.** After permutation, the next step is to determine the groups  $G_4$ ,  $G_6$ ,  $G_8$ . To accurately distinguish outliers from regular elements, we define the quantization threshold as follows:

**Definition 1.** Given a high-precision bit-width (e.g., 8-bit for recent LLMs) and a target bit-width n, the quantization threshold T(n) is defined as:

$$T(n) = 2^b \cdot \frac{2^{n-1}}{q_{max}} \cdot \overline{E}(X)_{INT8} \tag{6}$$

To maintain low quantization error at MXFP4 or MXFP6, the maximum allowable magnitude within group must satisfy:

$$\max(|G_n|) \le T(n) \tag{7}$$

Here, n denotes the number of bits, b is the exponent bias and  $q_{max}$  represents the maximum representable value in the target format. A detailed derivation of the quantization threshold is provided in Appendix C.2. Based on the thresholds, the groups  $G_4$ ,  $G_6$ , and  $G_8$  are defined as:

$$G_4 = \{X | X \le T(4)\} \quad G_6 = \{X | T(4) < X \le T(6)\}, \quad G_8 = \{X | T(6) < X\}$$
 (8)

We calculate the proportions  $p_4$ ,  $p_6$ , and  $p_8$  corresponding to the channel groups  $G_4$ ,  $G_6$ , and  $G_8$  for each linear layer in Llama3.1-8B. The results are shown in Figure 3. We summarize three key observations:

**Layer-wise Adaptivity**: The proportions vary dynamically across layers, reflecting the diverse input distributions in each activation. This demonstrates that the mixed-precision allocation is layer-specific rather than fixed globally.

**FP4 Dominance**: The proportion  $p_4$  consistently exceeds 50%, indicating that FP4 computations dominate the mixed-precision workflow. This dominance contributes significantly to the computational efficiency of the model.

**Cross-Dataset Stability**: The variations of  $p_4$ ,  $p_6$ , and  $p_8$  across different datasets and sampling strategies are minimal, suggesting that the mixed-precision assignment remains relatively stable.

<span id="page-4-0"></span>![](_page_4_Figure_7.jpeg)

Figure 3: Distribution statistics of  $p_4$  (E2M1),  $p_6$  (E3M2), and  $p_8$  across Llama3.1-8B. We evaluate 32 samples selected from WikiText2 (Merity et al., 2016) and the Pile dataset (Gao et al., 2020), covering batch sizes of 8, 16, 32, and 64, and sequence lengths of 512, 1024, 2048, and 4096. For each sample,  $p_4$ ,  $p_6$ , and  $p_8$  are computed over all linear layers. The figure reports the mean values and min-max ranges of  $p_4$ ,  $p_6$ , and  $p_8$  across all samples.

Offline Channel Assignment Strategy. Online evaluation of channel partitioning would introduce substantial runtime overhead. Instead, leveraging the observed stability of  $p_4$ ,  $p_6$ , and  $p_8$ , we precompute  $\{p_4^k, p_6^k, p_8^k, \sigma^k\}$  for the kth linear layer offline using calibration data. To allocate higher precision to more critical channels, we sort the activation channels according to their absolute mean values. Specifically, for the kth linear layer input tensor  $\boldsymbol{X}^k \in \mathbb{R}^{L \times I}$ , the channel-wise absolute mean vector  $\boldsymbol{M}^k \in \mathbb{R}^I$  is computed as:

$$\mathbf{M}^{k} = \left(\frac{1}{L} \sum_{i=1}^{L} |X_{:,1}^{k}|, \frac{1}{L} \sum_{i=1}^{L} |X_{:,2}^{k}|, \dots, \frac{1}{L} \sum_{i=1}^{L} |X_{:,I}^{k}|\right)$$
(9)

The permutation  $\sigma^k$  is obtained by sorting the elements of  $M^k$  in ascending order. Let  $p_4^k$ ,  $p_6^k$ , and  $p_8^k$  denote the proportions obtained for  $X^k$ ; the channel partitioning is then defined as:

$$G_4 = \sigma^k(X)_{:,:p_4^kI}, \quad G_6 = \sigma^k(X)_{:,p_4^kI:(p_4^k + p_6^k)I}, \quad G_8 = \sigma^k(X)_{:,(p_4^k + p_6^k)I:p_8^kI}$$
(10)

### 3.2 Kernel Design

Low-bit quantization offers significant performance improvements but presents considerable challenges in kernel design, especially for mixed-precision and fine-grained schemes like MicroMix. Recent advancements in GPU architectures, particularly the increased throughput of Tensor Cores for low-bit floating-point operations, combined with underlying support for block-scaled formats, have diminished the competitive advantage of traditional INT-type and GEMM quantization kernels. This simultaneously creates new opportunities for low-bit floating-point quantization.

**Mixed-precision Quantization.** Driven by the goal of deep algorithm-hardware integration, we have designed a mixed-precision, block-scaling quantization kernel for MicroMix. Complementing

this, we adopt MXFP-type GEMM kernels from CUTLASS, resulting in a kernel suite that delivers both excellent performance and high accuracy.

Fine-grained Block-scaled Data Formats. Quantization error is also influenced by the number of elements sharing a single scale factor. To fully harness the representational power of low-bit data types, fine-grained group quantization has become widely adopted and proven efficient in related works such as Atom and QuaRot. This used to be a tough trade-off between accuracy gains and dequantization overhead. However, the NVIDIA Blackwell architecture changes the game. Blackwell's mma instructions directly support new 4, 6, and 8-bit floating-point data types with integrated scale factors (known as MXFP formats), making block-scaled quantization a truly practical solution.

GEMM Kernel. As shown in Figure [4](#page-5-0) (a), the output matrix is divided into blocks within each GEMM kernel, with iterations for each block performed along the K dimension. After loading fragments of input matrices and their scale factors into Shared Memory or Tensor Memory, MMA instructions fused with dequantization operations are continuously executed on Tensor Cores. These operations accumulate FP32 partial sums into the BFloat16 result matrix. The process is highly decoupled, as matrices of specific data types invoke their corresponding GEMM kernels. This design allows for easy adjustment of data type categories and their ratios.

<span id="page-5-0"></span>![](_page_5_Figure_4.jpeg)

Figure 4: (a): The fused GEMM kernel of MicroMix. (b): The fused reorder-and-quantize operation. The quantization of weights is one-time cost and could be performed offline.

Quantization Kernel. Mixed-precision quantization often faces irregular memory access, leading to significant performance degradation. To tackle this, MicroMix adopts a strategy similar to Atom and RPTQ [\(Yuan et al., 2023\)](#page-13-5) by reordering channels to enable regular memory access. Our algorithm divides channels of activation into three distinct parts, to which we then apply block-wise scaling quantization in 32-element blocks. To ensure correct matrix multiplication, weights are correspondingly permuted to match the reordered activations before undergoing a similar three-part block-wise scaling quantization. Crucially, the reordering and quantization of activations must occur dynamically, while these processes for weights can be handled offline as a pre-processing step. To mitigate the overhead of dynamic reordering, we employ a kernel fusion technique (see Figure [4](#page-5-0) (b)), which combines the quantization and reordering operations into a single kernel. As shown in Figure [5,](#page-5-1) our fused kernel introduces little overhead compared to mixed-precision quantization only.

<span id="page-5-1"></span>![](_page_5_Figure_7.jpeg)

Figure 5: Comparison of the latency between single and fused operations with a batch size of 32.

### 4 EXPERIMENTS

### 4.1 EXPERIMENTAL SETUP

Quantization. MicroMix performs block-wise symmetric quantization with a block size of 32 for both weights and activations, using the E8M0 scaling format. The data formats are MXFP8 (E4M3), MXFP6 (E3M2), MXFP4 (E2M1) respectively. In Appendix [D.1,](#page-16-1) Table [9](#page-17-0) provides a summary of the quantized model information, including average using bits, offline calibration time and the size of quantized models.

**Baselines.** We compare MicroMix against four INT-based weight-activation quantization methods: Atom (Zhao et al., 2024), QUIK (Ashkboos et al., 2023), QuaRot (Ashkboos et al., 2024), FlatQuant (Sun et al., 2025) and one MX-based method, AMXFP4 (Lee et al., 2025). All baselines are reproduced on both Llama (Grattafiori et al., 2024) and Qwen (Qwen et al., 2025). Since MicroMix employs non-fixed bit-widths across linear layers, we additionally report the average bit-width per token element for all methods in Table 1. Implementation details are provided in Appendix D.1.

**Benchmarks.** For zero-shot evaluation, we use ARC\_C (Clark et al., 2018), Lambada (Paperno et al., 2016), Winogrande (Sakaguchi et al., 2019), BoolQ (Clark et al., 2019), and PIQA (Lourie et al., 2021). For five-shot accuracy, we adopt MMLU (Hendrycks et al., 2021a). WikiText2 (Merity et al., 2016) is used to evaluate perplexity (PPL). Additionally, we assess the Code and Math capabilities of the Qwen2.5 model series. Code benchmarks are Human-Eval (Chen et al., 2021) and MBPP (Austin et al., 2021), while Math benchmarks cover GSM8K (Cobbe et al., 2021), MMLU-STEM (Hendrycks et al., 2021a), CMATH (Wei et al., 2023) and MATH (Hendrycks et al., 2021b).

#### 4.2 MAIN RESULTS

Table 1 reports zero-shot and five-shot accuracy, along with WikiText-2 perplexity, for MicroMix and six baselines on Llama3.1-8B and Qwen2.5-32B. Across the five zero-shot benchmarks, MicroMix is the only quantization method that consistently preserves at least 98% of FP16 average accuracy on both models (Llama: 71.56 vs. 73.03; Qwen: 75.20 vs. 75.55). On the five-shot MMLU benchmark, MicroMix retains at least 96% of FP16 accuracy (Llama: 62.65 vs. 65.24; Qwen: 81.79 vs. 83.32), outperforming all competitors by  $\geq$ 1.32 points on Llama and  $\geq$ 0.27 points on Qwen. For WikiText2, MicroMix incurs only a marginal perplexity increase of 0.48 on Llama3.1-8B (6.72 vs. 6.24) and 0.46 on Qwen2.5-32B (5.56 vs. 5.02). The results of KV cache quantization is demonstrated in Table 10 of Appendix D.3.

<span id="page-6-0"></span>Table 1: Zero-shot and few-shot accuracy and perplexity of Llama3.1-8B and Qwen2.5-32B evaluated with lm-eval (Gao et al., 2024). "Avg. Bits" denotes the average bit-width per token element. INT6 is implemented using symmetric per-token quantization.

| Model       | Method    | Avg.  |       |       | 0-sho   | ot (†) |            |       | 5-shot (†) | PPL (↓)   |
|-------------|-----------|-------|-------|-------|---------|--------|------------|-------|------------|-----------|
| ivioue:     |           | Bits  | ARC_C | BoolQ | Lambada | PIQA   | Winogrande | Avg.  | MMLU       | WikiText2 |
|             | FP16      | 16.00 | 53.58 | 81.99 | 75.47   | 80.09  | 74.03      | 73.03 | 65.24      | 6.24      |
|             | QuaRot    | 4.12  | 46.42 | 76.24 | 68.48   | 77.91  | 70.96      | 68.00 | 55.23      | 6.98      |
|             | QUIK      | 5.95  | 44.62 | 77.09 | 73.28   | 77.09  | 69.19      | 68.05 | 56.65      | 7.29      |
| Llama3.1-8B | Atom      | 4.25  | 50.17 | 76.15 | 69.45   | 78.02  | 70.01      | 68.76 | 58.05      | 6.79      |
|             | FlatQuant | 4.19  | 51.54 | 78.87 | 73.32   | 79.16  | 71.98      | 70.97 | 61.33      | 6.95      |
|             | AMXFP4    | 5.00  | 43.77 | 74.80 | 71.12   | 75.19  | 66.85      | 66.34 | 53.79      | 7.49      |
|             | INT6      | 6.00  | 48.38 | 77.37 | 69.86   | 78.29  | 70.01      | 68.78 | 58.67      | 7.53      |
|             | MicroMix  | 5.51  | 50.26 | 81.13 | 74.13   | 80.14  | 72.14      | 71.56 | 62.65      | 6.72      |
|             | FP16      | 16.00 | 55.89 | 87.46 | 76.21   | 82.26  | 75.93      | 75.55 | 83.32      | 5.02      |
|             | QuaRot    | 4.12  | 53.08 | 84.77 | 74.89   | 80.96  | 73.14      | 73.36 | 79.39      | 5.86      |
|             | QUIK      | 6.21  | 52.90 | 85.87 | 74.21   | 80.36  | 71.82      | 73.03 | 78.89      | 5.92      |
| Qwen2.5-32B | Atom      | 4.21  | 54.78 | 86.54 | 75.92   | 81.45  | 73.48      | 74.43 | 79.54      | 5.89      |
| -           | FlatQuant | 4.71  | 56.23 | 86.30 | 75.41   | 81.50  | 74.19      | 74.72 | 81.52      | 5.74      |
|             | AMXFP4    | 5.00  | 51.54 | 87.09 | 75.24   | 80.85  | 73.51      | 73.64 | 79.96      | 5.85      |
|             | INT6      | 6.00  | 55.29 | 85.38 | 69.45   | 78.84  | 71.82      | 72.15 | 79.33      | 5.82      |
|             | MicroMix  | 5.22  | 56.66 | 87.13 | 77.37   | 80.65  | 74.19      | 75.20 | 81.79      | 5.56      |

Notably, higher average bit-width does not necessarily translate into higher accuracy. For instance, QUIK and INT6 employ more bits than MicroMix, yet provide limited performance gains.

<span id="page-6-1"></span>Table 2: Mixtral-8x7B-v0.1-Instruct performance comparison between FP16 and MicroMix.

|          | Arc_C | BoolQ | Lambada | PIQA  | Winogrande | Avg.  | Execution Time |
|----------|-------|-------|---------|-------|------------|-------|----------------|
| FP16     | 65.70 | 88.50 | 77.37   | 84.49 | 76.87      | 78.58 | 5min 18s       |
| MicroMix | 64.25 | 88.07 | 78.52   | 84.00 | 76.16      | 78.20 | 2min 03s       |

As shown in Table 2, MicroMix attains accuracy comparable to FP16 on Mixtral-8x7B-v0.1-Instruct, with an average score drop of only 0.38 points (78.58 to 78.20) and per-task differences

typically within ±1.5 points. Notably, this minor accuracy trade-off comes with a substantial runtime reduction, cutting execution time from 5m18s to 2m03s.

Math benchmarks. Table [3](#page-7-0) shows that MicroMix incurs an average accuracy drop of less than 4% compared to FP16, while retaining at least 98.4% of FP16 accuracy on GSM8K, MATH, and CMATH, with an average bit-width of 5.16.

<span id="page-7-0"></span>Table 3: Accuracy (↑) of Qwen2.5-Math-7B-Instruct on math benchmarks: GSM8K, MMLU-STEM, CMATH, and MATH. FP8 is implemented by vLLM [\(Kwon et al., 2023\)](#page-11-8).

| Model | Method          | GSM8K        | MATH         | MMLU-STEM    | CMATH        | Average      |
|-------|-----------------|--------------|--------------|--------------|--------------|--------------|
|       | FP16            | 95.8         | 83.7         | 77.8         | 91.5         | 87.2         |
| 7B    | FP8<br>MicroMix | 95.5<br>95.1 | 83.4<br>82.4 | 68.7<br>66.5 | 91.7<br>91.5 | 84.8<br>83.8 |

Code benchmarks. As reported in Table [4,](#page-7-1) MicroMix achieves accuracy comparable to or better than INT8 on the 14B (Avg. Bits: 5.54) and 32B (Avg. Bits: 5.18) models. Relative to FP16, the accuracy degradation remains within 1.5%.

<span id="page-7-1"></span>Table 4: Accuracy (↑) of Qwen2.5-Coder-{14B,32B}-Instruct on Code benchmarks: Human-Eval and MBPP. INT8 is implemented by Bitsandbytes [\(Dettmers et al., 2022\)](#page-11-1).

| Model | Method           | Human-Eval   | Human-Eval+  | MBPP         | MBPP+        |
|-------|------------------|--------------|--------------|--------------|--------------|
|       | FP16             | 87.8         | 84.1         | 81.0         | 69.2         |
| 14B   | INT8<br>MicroMix | 86.6<br>87.4 | 82.9<br>82.9 | 86.0<br>85.4 | 73.0<br>70.1 |
|       | FP16             | 88.4         | 84.1         | 84.5         | 70.9         |
| 32B   | INT8<br>MicroMix | 89.0<br>89.0 | 85.4<br>85.4 | 86.2<br>86.8 | 73.5<br>74.1 |

### 4.3 ABLATION STUDIES

In this section, we analyze the potential impact of different data formats and calibration datasets.

MXFP6 and MXFP8 Variants. We examine the impact of different MXFP6 (E2M3 and E3M2) and MXFP8 (E5M2 and E4M3) variants on zero-shot accuracy and perplexity. As shown in Table [5,](#page-7-2) all four configurations yield comparable results, suggesting that the specific exponent–mantissa tradeoff has only a minor effect on MicroMix. This robustness arises because the definition of quantization thresholds explicitly accounts for the influence of both exponents and mantissas on quantization error, thereby mitigating sensitivity to data format choices.

<span id="page-7-2"></span>Table 5: Zero-shot accuracy (↑) on Winogrande, Lambada, PIQA, and perplexity (↓) on WikiText2, using different exponent and mantissa bits for MXFP6 and MXFP8 on Llama3.1-8B. MXFP4 is E2M1 consistently.

| MXFP8 | MXFP6 | Winogrande | Lambada | PIQA<br>WikiText2 |
|-------|-------|------------|---------|-------------------|
| E5M2  | E3M2  | 72.53      | 73.36   | 80.25<br>6.84     |
|       | E2M3  | 72.69      | 72.83   | 80.14<br>6.81     |
| E4M3  | E3M2  | 72.14      | 74.13   | 6.72<br>80.14     |
|       | E2M3  | 71.51      | 74.07   | 80.09<br>6.73     |

Impact of Calibration Datasets. To assess the robustness of offline partitioning, we test different calibration datasets, including WikiText2, Pile and C4 [\(Raffel et al., 2019\)](#page-12-8). As shown in Table [12](#page-19-1) of Appendix [D.3,](#page-18-0) zero-shot and perplexity results remain stable across datasets with performance fluctuations within approximately 1%.

Time Breakdown by Component. We use the average values of p4, p6, and p<sup>8</sup> from Llama3.1- 8B to compute the runtime breakdown of reorder-and-quantize and GEMM relative to the total MicroMix kernel time, as shown in Table 6. Fused reorder-and-quantize operation takes less than 20% MicroMix kernel runtime across different lengths.

Table 6: The proportion of runtime for each part on RTX 5090.

<span id="page-8-0"></span>

| Part                 | Length = 128 | Length = 256 | Length = 512 | Length = 1024 | Length = 2048 | Length = 4096 |
|----------------------|--------------|--------------|--------------|---------------|---------------|---------------|
| Reorder-and-Quantize | 7.9%         | 9.7%         | 11.8%        | 14.3%         | 17.0%         | 16.9%         |
| GEMM                 | 92.1%        | 90.3%        | 88.2%        | 85.7%         | 83.0%         | 83.1%         |

#### 4.4 EFFICIENCY EVALUATION

In this section, we assess the efficiency of MicroMix from three perspectives: (1) single-kernel execution speed; (2) speedup of our custom kernel relative to CUTLASS; and (3) end-to-end performance in the prefill and decode stages. We evaluate MicroMix on three Blackwell architecture GPUs: RTX 5070Ti laptop, RTX 5090, and RTX PRO 6000, to examine its applicability across consumer and server GPUs.

**Kernel Efficiency.** We measure the latency of the MicroMix kernel across varying sequence lengths and hidden sizes. As baselines, we use strong TensorRT implementations: TensorRT FP8 (per tensor), W4A16 (per token), and FP16. Because MicroMix employs a nonfixed combination of 4-, 6-, and 8-bit channels, all kernel and transformer-block experiments report the minimum to maximum ranges alongside mean-value curves. As shown in Figure 6, MicroMix consistently outperforms TRT FP8 on both GPU platforms. On the RTX 5070Ti laptop, it achieves a 2.45 to 2.93× speedup over TRT FP16 and up to 1.45× over TRT FP8. On the RTX 5090, MicroMix delivers a 2.29 to 3.38× speedup over TRT FP16 and up to 1.74× over TRT FP8.

<span id="page-8-1"></span>![](_page_8_Figure_7.jpeg)

Figure 6: Computation latency of a single kernel with different lengths. "MicroMix-Range" denotes the latency span from the fastest to the slowest time.

**Performance of our customized kernel.** Table 7 reports the speedup of our customized GEMM kernel relative to CUTLASS. For small problem sizes with N = K = 4096, our kernels consistently outperform CUTLASS, achieving speedups of approximately 2.6 to  $4.0 \times$  for W4A4, 3.1 to  $5.0 \times$  for W6A6, and 1.2 to  $3.1 \times$  for W8A8 as M increases. The gains are most pronounced at moderate M values, for example M = 32, indicating improved utilization and kernel efficiency for low-precision formats, particularly W6A6.

<span id="page-8-2"></span>Table 7: Customized GEMM Kernel Speedup over CUTLASS on Small Problem Size (N=K=4096).

| M   |                   | W4A4                 |               |                   | W6A6                 |               |                   | W8A8                 |               |  |
|-----|-------------------|----------------------|---------------|-------------------|----------------------|---------------|-------------------|----------------------|---------------|--|
|     | CUTLASS<br>TFLOPS | Customized<br>TFLOPS | Speedup       | CUTLASS<br>TFLOPS | Customized<br>TFLOPS | Speedup       | CUTLASS<br>TFLOPS | Customized<br>TFLOPS | Speedup       |  |
| 1   | 1.04              | 2.72                 | 2.62×         | 0.65              | 2.04                 | 3.14×         | 1.02              | 1.63                 | 1.60×         |  |
| 2   | 2.07              | 5.45                 | $2.63 \times$ | 1.30              | 4.67                 | 3.59×         | 2.11              | 3.27                 | $1.55 \times$ |  |
| 4   | 4.16              | 10.91                | $2.62 \times$ | 2.62              | 8.19                 | 3.13×         | 4.16              | 6.55                 | 1.57×         |  |
| 8   | 8.09              | 26.17                | 3.23×         | 5.23              | 18.72                | 3.58×         | 8.46              | 16.37                | 1.93×         |  |
| 16  | 15.95             | 52.37                | 3.28×         | 10.47             | 43.68                | <b>4.17</b> × | 17.01             | 37.42                | $2.20 \times$ |  |
| 32  | 32.58             | 130.99               | $4.02 \times$ | 20.95             | 104.75               | $5.00 \times$ | 33.58             | 104.80               | 3.12×         |  |
| 64  | 66.82             | 209.50               | 3.14×         | 41.92             | 134.61               | $3.21 \times$ | 67.28             | 130.96               | 1.95×         |  |
| 128 | 130.91            | 260.90               | 1.99×         | 83.83             | 161.35               | 1.92×         | 134.82            | 160.90               | 1.19×         |  |

**Comparison of 4-bit baselines.** To demonstrate the end-to-end efficiency of MicroMix, we compare against two 4-bit baselines, Atom and QuaRot. As shown in Figure 7, MicroMix reduces prefill

<span id="page-9-0"></span>latency by about 85 percent compared to Atom and QuaRot on RTX PRO 6000. In the decode stage, MicroMix increases throughput by 1.82 to 3.02 times compared to Atom.

![](_page_9_Figure_2.jpeg)

Figure 7: Prefill latency (left) and decoding throughput (right) of three methods on RTX PRO 6000.

Comparison of FP16 and INT8 baselines. On RTX 5090, we further compare the prefill performance of MicroMix against two baselines: FP16 from HuggingFace and INT8 from Bitsandbytes. Figure [8](#page-9-1) reports the prefill latency and peak memory usage of Llama2-7B and Llama3.1-8B on the RTX 5090 with batch sizes {8, 12} and sequence length 2048. Compared with FP16, MicroMix reduces memory usage by 2.29–2.84× and latency by approximately 2.0× at batch size 8. Compared with INT8, MicroMix further reduces memory usage by 1.60–2.01× and latency by 1.80–1.84× at batch size 12.

<span id="page-9-1"></span>![](_page_9_Figure_5.jpeg)

Figure 8: Prefill latency and peak memory usage of MicroMix compared with FP16 and INT8.

### 5 CONCLUSION

In this paper, we present MicroMix, a co-designed mixed-precision quantization algorithm and kernel that supports MXFP4, MXFP6, and MXFP8 formats. Our algorithm introduces the quantization threshold to identify elements that incur excessive quantization error at the target bit width. We also propose an offline calibration strategy to determine the optimal channel assignments for each precision level on calibration dataset. To enable efficient inference, we design a matrix multiplication kernel that integrates three GEMM precisions and a fused reorder-and-quantize operation. MicroMix kernel achieves significant speedups over TensorRT baselines on both RTX 5070Ti laptop and RTX 5090 GPUs across various configurations. On the RTX PRO 6000, MicroMix consistently outperforms FP16, INT8 and INT4 baselines.

