# BitMoD: Bit-serial Mixture-of-Datatype LLM Acceleration

Yuzong Chen† , Ahmed F. AbouElhamayed† , Xilai Dai† , Yang Wang‡ , Marta Andronic§ , George A. Constantinides§ , and Mohamed S. Abdelfattah†

†Computer Systems Lab, Cornell University ‡Systems and Networking Research Group, Microsoft Research §Department of Electrical and Electronic Engineering, Imperial College London †{yc2367, afa55, xd44, mohamed}@cornell.edu ‡yang.wang92@microsoft.com §{marta.andronic18, g.constantinides}@imperial.ac.uk

*Abstract*—Large language models (LLMs) have demonstrated remarkable performance across various machine learning tasks. Yet the substantial memory footprint of LLMs significantly hinders their deployment. In this paper, we improve the accessibility of LLMs through BitMoD<sup>1</sup> , an algorithm-hardware co-design solution that enables efficient LLM acceleration at low weight precision. On the algorithm side, BitMoD introduces fine-grained data type adaptation that uses a different numerical data type to quantize a group of (e.g., 128) weights. Through the careful design of these new data types, BitMoD is able to quantize LLM weights to very low precision (e.g., 4 bits and 3 bits) while maintaining high accuracy. On the hardware side, BitMoD employs a bitserial processing element to easily support multiple numerical precisions and data types; our hardware design includes two key innovations: First, it employs a unified representation to process different weight data types, thus reducing the hardware cost. Second, it adopts a bit-serial dequantization unit to rescale the per-group partial sum with minimal hardware overhead. Our evaluation on six representative LLMs demonstrates that BitMoD significantly outperforms state-of-the-art LLM quantization and acceleration methods. For discriminative tasks, BitMoD can quantize LLM weights to 4-bit with < 0.5% accuracy loss on average. For generative tasks, BitMoD is able to quantize LLM weights to 3-bit while achieving better perplexity than prior LLM quantization scheme. Combining the superior model performance with an efficient accelerator design, BitMoD achieves an average of 1.69× and 1.48× speedups compared to prior LLM accelerators ANT and OliVe, respectively.

## I. INTRODUCTION

Large language models (LLMs) have achieved significant breakthroughs in natural language processing tasks [47], [57]. However, the growth of LLM size and complexity continues to outpace the scaling of compute performance and memory capacity in existing hardware platforms [22]. For example, the first generation of the GPT model, introduced in 2018, contains only 117 million parameters, while the second and third generations grew more than 10× and 1000×, respectively within two years [9]. This rapid increase in size necessitates significant memory capacity for model deployment, hindering their wide adoption, especially in edge scenarios with limited compute and memory resources. For instance, the state-of-theart (SOTA) open-source LLM family, Llama-3 [35], contains

<sup>1</sup>Code is available at: https://github.com/yc2367/BitMoD-HPCA-25

more than 8 billion parameters and requires more than 16GB of memory to store the model weights in 16-bit floatingpoint (FP16) format, which cannot fit in an edge GPU such as Jetson-TX2 with 8GB memory [37]. Therefore, designing novel LLM compression algorithms, together with accelerators co-designed for efficient deployment of the compressed models, presents a promising solution to enhancing the accessibility of LLMs on edge devices.

Quantization serves as one of the most hardware-efficient methods to mitigate the computation and memory demands of LLMs. Generally, there are two types of quantization mechanisms. The first one is quantization-aware training (QAT), where retraining is needed to update model weights and quantization parameters (e.g., scaling factors) [26], [31]. The second approach is post-training quantization (PTQ), which does not require retraining [10], [19], [20], [25], [30], [42], [52]. Although QAT can achieve more competitive accuracy than PTQ, the prohibitive cost of retraining LLMs makes it less practical. As a result, PTQ is commonly adopted in existing LLM quantization studies. While some PTQ works quantize both weights and activations into low precision [25], [42], [52], weight-only quantization can offer a better trade-off between model accuracy and hardware efficiency for edge deployment of LLMs, where weights dominate the memory footprint [10], [19], [20], [30]. However, existing weight-only quantization works on GPUs suffer from poor computational efficiency since GPUs lack dedicated hardware to perform multiplication between integer weight and floating-point activation. Consequently, these methods must first dequantize the weight to FP16 and rely on the floating-point pipeline for computation.

To achieve better computational efficiency for LLMs, a recent accelerator work, FIGNA [27], proposes a family of dedicated computing units for mixed-precision arithmetic between integer weights and floating-point activations. To further unleash the potential of quantization for improved hardware efficiency, several works have proposed algorithmhardware co-design solutions based on *custom* low-precision data types [25], [26], [38], [40]. The microscaling format (MX) [38], [40], assigns 8-bit metadata as the shared exponent to a group of low-precision weights. ANT [26] introduces a new data type that better adapts to the intra-tensor value distribution, thus reducing the quantization error. OliVe [25] proposes an outlier-victim-pair quantization mechanism, where an outlier value with a large magnitude is represented with an "Adaptive Biased Float" format and can be protected by pruning its adjacent victim value that has a small magnitude.

In this paper, we propose  $BitMoD^2$ , an algorithm-hardware co-design solution for efficient LLM acceleration at low weight precision. On the algorithm side, BitMoD exploits the per-group quantization [16], and modifies low-precision floating-point data types by repurposing the redundant zero value with a special value, which provides the ability to better adapt the data type itself to the numerical distribution of each weight group. Through careful choice of special values, BitMoD is able to quantize LLM weights to very low precision (e.g., 4-bit and 3-bit) with tiny encoding overhead while maintaining good model accuracy. On the hardware side, BitMoD employs the bit-serial computing paradigm with a unified representation for different low-precision data types to efficiently trade-off weight precision and hardware efficiency.

The main contributions of this paper are summarized below:

- 1) We propose *BitMoD*, a hardware-efficient PTQ solution for LLM acceleration. *BitMoD* introduces new data types that are tailored for per-group weight quantization at 4-bit and 3-bit precision with tiny encoding overhead.
- 2) We demonstrate that the proposed data types can be seamlessly integrated with other quantization optimization techniques, achieving better model perplexity than SOTA software-only LLM quantization works.
- 3) We propose an efficient accelerator design for *BitMoD*, which adopts a unified bit-serial representation for multiple low-precision data types. This effectively reduces the hardware cost to perform computation between low-precision weights and FP16 activations, and trades-off weight precision for improved hardware efficiency.
- 4) Our evaluation on six representative LLMs shows that on average, *BitMoD* achieves 2.2× speedup and 2.31× better energy efficiency compared to the baseline FP16 accelerator, *without* loss in accuracy. Compared to SOTA accelerators ANT and OliVe, *BitMoD* achieves an average speedup of 1.69× and 1.48×, respectively.

## II. BACKGROUND AND MOTIVATION

## A. Why Weight Quantization for LLMs?

To demonstrate the importance of LLM weight quantization for edge applications, we profile the total memory access footprint of weight and activation for four representative LLMs running both discriminative and generative tasks with a batch size of 1. For discriminative tasks, the LLM receives an input context and outputs a single token such as in sentiment analysis [46] and multiple-choice question answering [13]. For generative tasks, the LLM receive an input context and output multiple tokens. We set the input to output sequence length to 256:1 and 256:256 for discriminative and generative tasks,

<sup>2</sup>BitMoD stands for Bit-serial computation with Mixture of Data types.

![](_page_1_Figure_11.jpeg)

Fig. 1: Total memory access of weights and activations on discriminative tasks (with 256 input tokens and 1 output token) and generative tasks (with 256 input tokens and 256 generated tokens). Note the log scale on the y-axis. Note that the gap between weight and activation memory accesses increases for generative tasks at batch size 1 despite a much larger KV-cache than discriminative tasks. While prior work [44] has correctly reported a memory bottleneck caused by the KV-cache, this only occurs for 175B+ parameter models with a high batch size (e.g., 512) and a context lengths exceeding 512 tokens. This scenario is less relevant to our focus on low-batch edge LLM inference where the weights indeed dominate the total memory accesses.

respectively, catering for edge applications as suggested by Lin et al. [30]. As shown in Fig. 1, the LLM weights access consumes orders of magnitude larger memory than the activations access. Although discriminative tasks only need to output a single token (e.g., "A"/"B"/"C" for multiple-choice question answering), the weight tensor dimension of an LLM (e.g., 2048 for OPT-1.3B) is much larger than the input token length, leading to memory access dominated by weights. Moreover, generative tasks necessitate repeated weight fetching for every new output token, resulting in significantly higher memory access for LLM weights. Thus, weight quantization is more effective for deploying LLMs in edge scenario where the batch size is small and the input token length is typically short.

### B. Quantization Basics

One of the most popular quantization schemes is integer quantization, where a floating-point value is scaled and rounded to a low-precision integer. There are two widely used quantization modes – *symmetric* and *asymmetric*. Symmetric integer quantization can be expressed as follows:

$$\Delta = \frac{W_{f \max}}{2^{b-1}-1}; \ W_q = \mathrm{Round}\left(\frac{W_f}{\Delta}\right); \ W_{qf} = W_q \cdot \Delta \ (1)$$

where  $W_f$  is the original floating-point tensor,  $W_{f\text{max}}$  is the absolute maximum value, b is the quantized integer precision,  $\Delta$  is the scaling factor,  $W_q$  is the quantized integer value, and  $W_{qf}$  is the floating-point value after performing dequantization (i.e., re-scaling).

The symmetric quantization assumes that the minimum and maximum values of a tensor have the same absolute value (i.e., symmetric value range), but this is not always true. Hence, another popular mode of quantization is asymmetric quantization, which can be expressed as follows:

$$\begin{split} \Delta &= \frac{\text{Range}\left(W_f\right)}{2^b - 1} \, ; \quad z = \text{Round}\left(\frac{-W_{f\text{min}}}{\Delta}\right) \\ W_q &= \text{Round}\left(\frac{W_f}{\Delta}\right) + z \, ; \quad W_{qf} = (W_q - z) \cdot \Delta \end{split} \tag{2}$$

![](_page_2_Figure_0.jpeg)

Fig. 2: Maximum value and value range for different quantization granularity. Results are normalized to the standard deviation  $(\sigma)$  of the weight vector at the corresponding granularity, then averaged across all weight vectors. The per-group granularity has a group size of 128.

where  $W_{f\min}$  is the absolute minimum value of  $W_f$ , and z represents the zero-point of the quantized tensor.

#### C. Motivation

We analyze several techniques that are widely adopted in recent quantization studies, which motivates our proposed *BitMoD* framework. We mainly focus on weight quantization in our discussion.

Quantization Granularity Matters. Consider a floating-point weight tensor  $W_f^{K \times D}$ , where K represents the number of output channels and D is the channel size. There are three granularities to quantize the model weight: per-tensor, per-channel, and per-group. The per-tensor quantization uses the same scaling factor to quantize a whole weight tensor, while per-channel quantization divides the weight tensor along the output channel into K vectors, and quantizes every vector  $W_f^{1 \times D}$  independently. However, given the large tensor size and hidden dimension of LLMs, these two granularities still lead to large quantization error. Specifically, the quantization error of a dequantized weight in Eq. 1 can be expressed as:

$$\operatorname{Error}(W_{qf}) = \operatorname{ErrorRound}\left(\frac{W_f}{\Delta}\right) \cdot \Delta \tag{3}$$

where ErrorRound is the rounding error during quantization, which has been shown to have an expected value of 0.25 [30]. Therefore, the quantization error is proportional to the scaling factor  $\Delta$ , which is further proportional to the maximum value and range for symmetric (Eq. 1) and asymmetric (Eq. 2) quantization, respectively.

In order to further reduce the quantization error, recent LLM quantization studies adopt the *per-group* granularity [16], [19], [20], [30]. The per-group quantization further divides a weight channel  $W_f^{1\times D}$  into D/G groups, each with a group size of G. The group size introduces extra overhead to store the quantization parameters, i.e., scaling factor (and zero-point) for every group, and is usually set to 128 in SOTA quantization frameworks to balance accuracy and memory overhead [20], [30]. Fig. 2 demonstrates the benefits of per-group quantization by showing the maximum value and range in four representative LLMs at different granularity. The per-group granularity has the lowest maximum value and range, hence will have a lower quantization error compared to the other two granularity. Therefore, we focus on per-group quantization in this work.

TABLE I. Wikitext-2 perplexity (\$\psi\$) under different quantization granularity and 4-bit data types. "PC" and "PG" stand for per-channel and per-group, respectively. The group size is 128.

| Model       | OPT-1.3B |       | Phi   | -2B   | Llama-2-7B |      | Llama-2-13B |      |
|-------------|----------|-------|-------|-------|------------|------|-------------|------|
| Granularity | PC       | PG    | PC    | PG    | PC         | PG   | PC          | PG   |
| FP16        | 14.62    | 14.62 | 9.71  | 9.71  | 5.47       | 5.47 | 4.88        | 4.88 |
| INT4-Sym    | 36.05    | 16.04 | 13.03 | 11.15 | 12.92      | 5.84 | 5.47        | 5.07 |
| INT4-Asym   | 48.41    | 15.41 | 12.08 | 10.67 | 8.89       | 5.77 | 5.27        | 5.01 |
| FP4         | 16.07    | 14.99 | 11.24 | 10.68 | 8.07       | 5.77 | 5.15        | 5.05 |
| Flint       | 15.87    | 16.23 | 11.71 | 11.23 | 6.67       | 6.09 | 5.31        | 5.29 |

Quantization Data Type Matters. Numerous studies have proposed custom data types for quantization at the per-channel granularity [25], [26], [40]. We analyze the effects of adopting different data types for per-channel and per-group weight quantization. We explore four basic data types at 4-bit precision: integer with symmetric (INT4-Sym) and asymmetric (INT4-Asym) quantization, floating-point (FP4), and the Flint data type proposed by ANT [26]. Table I shows the resulting perplexity on the Wikitext-2 dataset [33]. We highlight two important observations. First, although Flint can achieve better perplexity at the per-channel granularity, it never outperforms other data types at the per-group granularity. Second, the per-group INT4-Asym and FP4 quantization achieve the best perplexity on some but not all studied LLMs, indicating that both asymmetry and FP data types are favorable for per-group quantization. The reason behind this is twofold. First, weight tensors typically exhibit Gaussian-like distribution that fits well to the floating-point data type [17], [51]. Second, while the effects of outliers are mitigated by pergroup quantization, a weight group can still contain outliers in an asymmetric pattern, being either solely positive or negative, as highlighted in previous studies [15], [16]. This characteristic benefits from asymmetric quantization.

The above observation motivates us to explore new quantization data types that can combine the benefits of asymmetry and FP formats to achieve better accuracy under per-group quantization. We notice that the basic FP data types have symmetric quantization values due to the inherent sign-magnitude binary representation that contains positive and negative zero values. Our key insight is that we can introduce additional asymmetry to FP by repurposing a redundant zero value with another special value. This approach provides us with two key benefits. First, it allows to fully utilize the limited quantization levels. Although the redundant zero value does not affect highprecision formats such as FP16, it constitutes a large fraction of quantization levels at low precision (e.g., 12.5\% at 3-bit precision). Second, we can tune the special value to make the extended FP data types better adapt to the per-group weight distribution, which we discuss in Section III-B.

Quantization Bit-width Matters. While prior LLM accelerators mainly rely on bit-parallel architectures that support 8-bit and 4-bit precision [25]–[27], recent studies have shown that 6-bit floating-point weights exhibit negligible accuracy loss across various LLM models and tasks [49], [51]. Motivated by this, we analyze the effects of using different 6-bit data types for per-group LLM weight quantization. We consider four data

TABLE II. Wikitext-2 and C4 perplexity (↓) under different 6-bit data types. We use per-group weight quantization with a group size of 128.

| Model     | OPT-1.3B |       |      | Phi-2B |      | Llama-2-7B | Llama-2-13B |      |
|-----------|----------|-------|------|--------|------|------------|-------------|------|
| Dataset   | Wiki     | C4    | Wiki | C4     | Wiki | C4         | Wiki        | C4   |
| FP16      | 14.62    | 14.72 | 9.71 | 12.74  | 5.47 | 6.97       | 4.88        | 6.47 |
| INT6-Sym  | 14.51    | 14.80 | 9.85 | 12.82  | 5.49 | 6.99       | 4.89        | 6.46 |
| INT6-Asym | 14.61    | 14.78 | 9.76 | 12.8   | 5.49 | 6.99       | 4.89        | 6.46 |
| FP6-E2M3  | 14.59    | 14.76 | 9.85 | 12.8   | 5.52 | 6.99       | 4.92        | 6.49 |
| FP6-E3M2  | 14.81    | 14.81 | 9.81 | 12.87  | 5.49 | 7.02       | 4.89        | 6.50 |

types: integer with symmetric (INT6-Sym) and asymmetric (INT6-Asym) quantization, floating-point with 2-bit exponent and 3-bit mantissa (FP6-E2M3), and floating-point with 3-bit exponent and 2-bit mantissa (FP6-E3M2). Table II compares the resulting perplexity of different quantization data types on Wikitext-2 [33] and C4 [18] datasets. On average, the studied 6-bit data types achieve similar and negligible perplexity loss compared to the FP16 baseline. For example, the average perplexity loss of INT6-Sym is less than 0.05, and its simple integer representation offers a promising solution to efficient LLM acceleration. Therefore, it is crucial for an accelerator to support diverse quantization bit-width to offer a better tradeoff between memory footprint and model accuracy.

A natural solution for accommodating variable precision is to adopt bit-serial architectures [3], [12], [28], [45]. However, existing bit-serial accelerators mainly target the integer data type, which causes significant accuracy loss at 3-bit precision as we will show in Section V-B. Furthermore, these accelerators cannot leverage per-group quantization for improved accuracy. This is because per-group quantization assigns different scaling factors for different groups, necessitating a floatingpoint unit with large area overhead to dynamically dequantize the partial sum after computing the dot-product for every group. Thus, an efficient dequantization mechanism with low hardware cost is desirable.

Algorithm-Hardware Co-Design Matters. Numerous frameworks have been proposed to accelerate LLM execution, as depicted in Table III. SOTA algorithmic solutions such as AWQ [30] quantize LLM weights to low-precision integer while preserving high accuracy. Nevertheless, AWQ is optimized for LLM acceleration on GPUs, which lack dedicated mixed-precision computing unit. As a result, it converts the low-precision weights to FP16 and relies on the GPU floatingpoint pipeline for computation, resulting in poor computational efficiency.

In contrast, ANT [26], OliVe [25], and FIGNA [27] propose efficient bit-parallel accelerators for quantized model acceleration. But their precision is limited to 8-bit and 4-bit, which restricts the ability to utilize other precision (e.g., 6 bit) for a better accuracy-efficiency trade-off. Moreover, their accelerators do not natively support per-group quantization, which requires a floating-point unit to dynamically dequantize the per-group partial sum on the fly. While Microscaling [40] accommodates diverse precision, it necessities a floating-point pipeline to handle the shared micro-exponent of a weight group, leading to higher energy consumption compared to

TABLE III. Comparison between *BitMoD* and SOTA co-design frameworks for LLM acceleration

| Framework         | Per-group<br>Quant? | Supported<br>Precision | Accuracy @<br>3-bit Weight | Hardware<br>Efficiency |
|-------------------|---------------------|------------------------|----------------------------|------------------------|
| AWQ [30]          | Yes                 | Limited                | High                       | Low                    |
| FIGNA [27]        | No                  | Limited                | Low                        | High                   |
| ANT [26]          | No                  | Limited                | Low                        | High                   |
| OliVe [25]        | No                  | Limited                | Medium                     | High                   |
| Microscaling [40] | Yes                 | Many                   | Low                        | Medium                 |
| BitMoD (Ours)     | Yes                 | Many                   | High                       | High                   |

other low-precision compute units. Furthermore, given the significant memory footprint of LLMs, it is desirable to explore sub-4-bit quantization while maintaining good model accuracy, which ANT, OliVe, and Microscaling do not address. As we will show in Section V-B, the custom quantization data types proposed by ANT, OliVe, and Microscaling fail to achieve better accuracy than the simple asymmetric integer quantization at 4-bit weight precision, and cause unacceptable accuracy loss at 3-bit weight precision under per-group quantization. The above limitation motivates us to propose an efficient LLM acceleration framework that supports a wide range of hardware-friendly bit-widths, while maintaining good accuracy at low weight precision.

#### III. BITMOD QUANTIZATION FRAMEWORK

In this section, we present the *BitMoD* quantization framework, which includes new data type families tailored for pergroup quantization at 3-bit and 4-bit precision. Section III-A describes our proposed data types that extend the basic floating-point data types at 3-bit and 4-bit precision. Section III-B presents an enhanced per-group LLM quantization strategy using the proposed data types. Section III-C describes the hardware-efficient per-group dequantization mechanism using integer scaling factors.

## *A. Asymmetric FP3 and FP4 Data Types*

The basic floating-point formats contain a redundant quantization level due to the sign-magnitude representation that has both +0 and −0. We propose to replace this redundant zero with another *special value* to fully utilize the available quantization levels and introduce additional asymmetry. We first use the basic FP3 format to derive our custom 3-bit data type, and then extend our idea to 4-bit precision.

FP3 Extension. The basic FP3 data type contains seven distinct values {0, ±1, ±2, ±4}. Our main idea is to extend FP3 and allows the redundant zero to be replaced by one of some pre-defined special values. Consequently, a weight group can be quantized by the basic FP3 data type together with a selected special value to minimize the quantization error. Ideally, the special values can have an arbitrary precision. But a high-precision (e.g., FP16) special value leads to more hardware overhead for computing, which offsets the efficiency of low-precision data types. Hence, we limit the special value to low-precision integers. Furthermore, given N as the number of allowed special values, an encoding overhead of logN 

![](_page_4_Figure_0.jpeg)

Fig. 3: Normalized weight quantization error (↓) with different special values (SV) for FP3. We use per-group quantization with a group size of 128. The special values ± 6 achieve the lowest overall quantization error, thus adopted in *BitMoD*.

bits is needed to specify which special value to be selected during computation. This selection also requires an N-to-1 mux in the hardware implementation. To balance the encoding overhead and hardware complexity, we set N = 4 which only requires 2-bit encoding per group.

The choice of special values will affect the resulting quantization error because it changes the set of available quantization values. As discussed in Section II-C, both asymmetry and floating-point data types are crucial for good accuracy under per-group quantization. Since the scaling factor and quantized values are ultimately determined by the absolute maximum value of a data type [32], we establish the set of special values based on two principles. First, some special values should fall inside the numerical range of FP3 to ensure that they do not alter its original absolute maximum (i.e., 4). This is advantageous for quantizing weight groups exhibiting symmetric, Gaussian-like distribution. Second, some special values could fall outside the numerical range of FP3 to introduce additional asymmetry, i.e., the absolute maximum and minimum quantization values of the extended FP3 are different. This can benefit weight groups that exhibit asymmetric distribution.

To satisfy the first property, the special values should be set to ± 3, which replace the redundant zero with + 3 and − 3, respectively. We call this new data type FP3-ER since it adds extra resolution (ER) within the range of FP3. To satisfy the second property, there are an infinite number of values that can fall outside the FP3 range. Therefore, we determine the two remaining special values that can minimize the quantization error. We further reduce the search space by restricting these two special values to have the same absolute value, which results in *balanced* asymmetry across all weight groups. This is desirable because, although an individual weight group may prefer asymmetric quantization, a whole weight tensor

TABLE IV. Our proposed extended resolution (ER) and extended asymmetry (EA) FP3 and FP4 data types.

| Dtype | Basic Values            | Extended Dtype | Special Value |
|-------|-------------------------|----------------|---------------|
|       | 0, ± 1, ± 2, ± 4<br>FP3 | FP3-ER         | −3 or +3      |
|       | FP3-EA                  | −6 or +6       |               |
|       | 0, ± 0.5, ± 1, ± 1.5    | FP4-ER         | −5 or +5      |
| FP4   | ± 2, ± 3, ± 4, ± 6      | FP4-EA         | −8 or +8      |

Algorithm 1: Fine-grained data type adaptation.

```
Input : Weight group: W; Quantization precision: p
  Output : Quantized weight group: Wqout ;
          Selected special value: vout
1 Func AdaptiveQuant( W, p ):
     // Get basic and special quantization
         values according to Table IV
2 basicValues = GetBasicValues( p )
3 specialValues = GetSpecialValues( p )
     // Search for the best special value
4 minError = +∞
5 for v in specialValues do
6 quantValues = basicValues ∪ v
7 Wq = NonLinearQuantize( W, quantValues)
8 newError = MeanSquareError( W, Wq )
9 if newError < minError then
10 minError = newError
11 Wqout = Wq
12 vout = v
13 return Wqout , vout
```

usually exhibits symmetric, Gaussian-like distribution [26], [55]. Fig. 3 shows the normalized per-group quantization error on six LLMs when adding different special values to FP3. We observe that adding asymmetry significantly reduces the quantization error. In addition, the special values ± 6 have the lowest quantization error on most LLMs except for OPT-1.3B, and are therefore adopted in *BitMoD*. We call the resulting new data type FP3-EA since it adds extra asymmetry (EA) to extend the range of FP3.

FP4 Extension. Similar to FP3-ER and FP3-EA, we add extra resolution and asymmetry to FP4. We conduct experiments to measure the effects of different FP4 special values on the resulting quantization error, which leads to the best FP4-ER and FP4-EA that have special values ± 5 and ± 8, respectively. Table IV summarizes the extended FP3 and FP4 data types. Note that although we have fixed the four special values given that they can minimize the quantization error for the diverse set of LLMs that we evaluate, the proposed *BitMoD* accelerator can flexibly accommodate other arbitrary special values that may perform well with different LLMs, which we discuss in Section IV-A.

