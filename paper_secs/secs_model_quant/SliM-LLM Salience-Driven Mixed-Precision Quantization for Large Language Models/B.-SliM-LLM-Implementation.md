# B. SliM-LLM Implementation

## B.1. Detailed Implementation

In this section, we present the specific implementation details of SliM-LLM, which utilizes GPTQ [\(Frantar et al.,](#page-8-4) [2022\)](#page-8-4) as its backbone for mixed-precision quantization and incorporates both SBA and SQC. SliM-LLM<sup>+</sup> is consistent with SliM-LLM in SBA computations but does not include the SQC component, instead retaining learnable weight clipping (LWC) approach in OmniQuant [\(Shao et al.,](#page-10-3) [2023\)](#page-10-3) for gradient optimization.

Algorithm [2](#page-12-2) primarily encompasses the core details of both SBA and SQC. In SBA, the importance of each group is determined by sorting the average salience of groups, followed by a bi-pointer search that increases the number of (N − 1)-bit and (N + 1)-bit groups to maintain their quantity equilibrium. The optimization function then utilizes the KL divergence from Eq. [\(4\)](#page-4-1) to determine the optimal mixed-precision ratio. SQC, on the other hand, enhances its information by amplifying the quantization error of unstructured weight groups. When the last two parameters, scale and zero point, in the fakequant(·) function are omitted, the default values from Eq. [\(1\)](#page-2-0) are used.

### <span id="page-11-0"></span>B.2. Mixed Bit Storage and Computing

We developed a framework for storage and inference deployment supporting mixed-precision quantization based on AutoGPTQ. The deployment process is as follows. After completing mixed-precision quantization with SliM-LLM, it outputs scales, zeros, and group-wise bit-widths generated during the quantization process to identify the quantization parameters and precision of each group in the Linear Projection weights. AutoGPTQ then packs the weights and zeros into integer-compressed representations (denoted by wˆ int and zˆint respectively) based on the precision of different groups, significantly reducing storage and operational bitwidth. After the quantized weights are packed, AutoGPTQ loads the model onto the GPU, where the mixed precision quantization kernel on the GPU performs dequantization on the weights and zeros of different groups and calculation with input activation, ultimately producing the final output.

In the mixed-precision deployment of AutoGPTQ, the weight memory layout is organized by group, with each group sharing the same precision, which is shown in Fig. [6.](#page-13-1) Within each group, elements with the same precision are packed as integers, eliminating the need for additional padding, which saves space. Given that the bit-widths of integers is a power of 2, this is compatible with group size that is also a power of 2. For instance, even with the oddbit such as 3-bit storage, integers can store these numbers without padding, as the commonly used group size is 128, a multiple of almost all definition of integer type. This ensures that elements within a group fully utilize the space provided by integers, without storing numbers of different precision within the same integer. zˆint follow the original logic of AutoGPTQ but are packed with a uniform precision along the channel direction for ease of use. Other tensors, like scales, remain in the same floating-point format to ensure the correctness of dequantization calculations.

To indicate the precision of each group, we also introduce an additional array to store bit-widths of each group, where each number is represented as a 2-bit value aggregated into integers, marking the quantization precision of each group for accurate reconstruction. We use cumulative calculations to determine the starting index of each group, ensuring correctness despite changes in wˆ int height and starting indices caused by varying precision. Using the above methods to store the quantized weights, zeros, and additional bit arrays effectively reduces memory usage during model storage and loading, thereby lowering the resource overhead required for model deployment.

Once the weights are packed, we follow the modified AutoGPTQ logic for GPU inference. The GPU processes and dequantizes the weights group by group for computation. During GPU computation, a thread dequantizes a segment of continuous memory data in one column of wˆ int and performs vector dot product calculations with the input activation shared within the block, accumulating the results in the corresponding result matrix. When threads form a logical block, the block handles the computation and reduction of a continuous channel region. We complete the linear layer computation by iterating through all logical blocks. Leveraging AutoGPTQ's initial logic and CUDA Warp's 32-thread units, we ensure similar code structure and data access logic for threads within each warp when group size is 128. This method was primarily conducted to validate feasibility os SliM-LLM, demonstrating that the mixed precision quantization with integer packing does not cause additional computational overhead, indicating the efficiency and accuracy advantage of SliM-LLM. In summary, by dividing weight into several structured groups with mixed precision and employing a reasonable GPU utilization strategy, Slim-LLM balances performance and efficiency.

#### Algorithm 1 Main Framework of SliM-LLM.

```
4: \mathcal{G}\{\cdot\} := SBA(\boldsymbol{w}, \boldsymbol{x}_F, \boldsymbol{H}^{in}, \beta, N)
func SliM-LLM(\boldsymbol{w}, \boldsymbol{x}_F, \beta, \lambda, N)
Input: \boldsymbol{w} \in \mathbb{R}^{n \times m} - FP16 weight
                                                                                                                                                                   5: for b = 0, \beta, 2\beta, ... do
                   \boldsymbol{x}_F \in \mathbb{R}^{t \times m} - calibration data
                                                                                                                                                                                \boldsymbol{w}^b\coloneqq \boldsymbol{w}_{:.b:b+\beta}
                   \beta - group size
                                                                                                                                                                                 g_b \coloneqq \mathcal{G}[b]
                                                                                                                                                                                \begin{aligned} \boldsymbol{w}_{s}^{b}, \boldsymbol{w}_{us}^{b} &\coloneqq \text{sal\_mask}(\boldsymbol{w}^{b}) \ \hat{\boldsymbol{w}}_{q}^{b} &\coloneqq \text{SQC}(\boldsymbol{w}_{s}^{b}, \boldsymbol{w}_{us}^{b}, g_{b}) \end{aligned}
                   \lambda - hessian regularizer
                    N - average bit-width
                                                                                                                                                                                 GPTQ-error compensation:
Output: \hat{\boldsymbol{w}}_q - quantized weight
                                                                                                                                                                10:
                                                                                                                                                                                 \boldsymbol{E}\coloneqq (\boldsymbol{w}_{:,b:b+\beta}-\hat{\boldsymbol{w}}_q^b)/\boldsymbol{H}_{bb:b+\beta b+\beta}^{\text{in}}
                                                                                                                                                                11:
  1: \boldsymbol{H}\coloneqq \frac{1}{P}\sum_{k=1}^{P} \boldsymbol{x}_F^{[k]} \boldsymbol{x}_F^{[k]T} hessian matrix
                                                                                                                                                                                 \boldsymbol{w}_{:,b+\beta:} \coloneqq \boldsymbol{w}_{:,b+\beta:} - \boldsymbol{E} \cdot \boldsymbol{H}_{b:b+\beta,b+\beta:}^{\text{in}}
  2: \boldsymbol{H}^{\text{in}} := \text{Cholesky}((\boldsymbol{H} + \lambda \boldsymbol{I})^{-1})
                                                                                                                                                                13: end for
                                                                                                                                                                14: return \hat{\boldsymbol{w}}_{a}
  3: \hat{\boldsymbol{w}}_q \coloneqq 0^{n \times m}
```

## Algorithm 2 Detailed functions in SliM-LLM.

```
func SBA(\boldsymbol{w}, \boldsymbol{x}_F, \boldsymbol{H}^{\text{in}}, \beta, N)
                                                                                                                                                func SQC(\boldsymbol{w}_{s}^{b}, \boldsymbol{w}_{us}^{b}, q_{b})
                                                                                                                                                   1: w_{\max} \coloneqq \max(\boldsymbol{w}_s^b \cup \boldsymbol{w}_{us}^b)
2: w_{\min} \coloneqq \min(\boldsymbol{w}_s^b \cup \boldsymbol{w}_{us}^b)
  1: \mathcal{G}\{\cdot\} := \{0\} // initialize group bit-width
  2: e := \inf //  bit-widths searching error
                                                                                                                                                   3: \lambda := 0.1
  3: p^* := 0 \text{ // number of } (N-1)\text{-bit and } (N+1)\text{-bit}
  4: l := N - 1 // \text{ lower bit-width}
                                                                                                                                                   4: n := 50
  5: h := N + 1 /\!/ higher bit-width
                                                                                                                                                   5: e := \inf // \text{ scale searching error}
  6: S\{\cdot\} := \operatorname{average}(\frac{\boldsymbol{w}^2}{[\boldsymbol{H}^{\text{in}}]_{\text{diag}}^2})
                                                                                                                                                   6: \Delta^* \in \mathbb{R}^{n \times 1} // per-channel scale
                                                                                                                                                   7: z^* \in \mathbb{R}^{n \times 1} // per-channel zero point
  7: for p = 1, 2, ..., \lceil \frac{m}{2\beta} \rceil do
                                                                                                                                                   8: for \tau \in [1 - \lambda, 1 + \lambda] with 2n slices do
           \hat{\boldsymbol{w}}_{l}^{b} \coloneqq \text{fakequant}(\boldsymbol{w}_{b \in \text{top k min(p)}}^{b}, l,)
                                                                                                                                                               \Delta \coloneqq \tau(w_{\text{max}} - w_{\text{min}})/(2^{g_s} - 1)
          \hat{\boldsymbol{w}}_h^b \coloneqq \operatorname{fakequant}(\boldsymbol{w}_{b \in \operatorname{top\_k\_max}(p)}^b, h, )
                                                                                                                                                               z := -\lfloor (\tau w_{\min})/\Delta \rceil
                                                                                                                                                 10:
             \hat{\boldsymbol{w}}_{N}^{b} \coloneqq \text{fakequant}(\boldsymbol{w}_{b \in \text{others}}^{b}, N,)
                                                                                                                                                               \hat{\boldsymbol{w}}_{s}^{b} \coloneqq \text{fakequant}(\boldsymbol{w}_{s}^{b}, g_{b}, \Delta, z)
10:
                                                                                                                                                               \hat{\boldsymbol{w}}_{us}^{b} \coloneqq \operatorname{fakequant}(\boldsymbol{w}_{us}^{b}, g_{b}, \Delta, z)
\mathcal{L}_{s} \coloneqq ||\boldsymbol{w}_{s}^{b} - \hat{\boldsymbol{w}}_{s}^{b}||_{1}^{2}
              \hat{\boldsymbol{w}}_q \coloneqq \hat{\boldsymbol{w}}_l^b \cup \hat{\boldsymbol{w}}_l^b \cup \hat{\boldsymbol{w}}_h^b
                                                                                                                                                 12:
11:
                                                                                                                                                 13:
              if \mathcal{D}_{kl} \left( \boldsymbol{x} \boldsymbol{w}^\top || \boldsymbol{x} \hat{\boldsymbol{w}}_q^\top \right) < e then
12:
                                                                                                                                                                \mathcal{L}_{us} \coloneqq || \boldsymbol{w}_{us}^b - \hat{\boldsymbol{w}}_{us}^b ||^2
                    e \coloneqq \mathcal{D}_{kl} \left( \boldsymbol{x} \boldsymbol{w}^{\top} \mid \boldsymbol{x} \hat{\boldsymbol{w}}_{q}^{\top} \right)
13:
                                                                                                                                                               if \mathcal{L}_s + \mathcal{L}_{us} < e then
                                                                                                                                                 15:
14:
                                                                                                                                                                    e := \mathcal{L}_s + \mathcal{L}_{us}
                                                                                                                                                 16:
               end if
15:
                                                                                                                                                                      z^* \coloneqq z
                                                                                                                                                 17:
16: end for
                                                                                                                                                                      \Delta^* := \Delta
                                                                                                                                                 18.
17: \mathcal{G}\{l\} := S\{\text{top k } \min(p^*) = l\}
                                                                                                                                                 19:
18: \mathcal{G}{h} := S{\text{top\_k\_max}(p^*) = h}
                                                                                                                                                20: end for
19: \mathcal{G}{N} := S{\text{middle\_k}(\left[\frac{m}{2}\right] - 2p^*)} = N
                                                                                                                                                21: \hat{\boldsymbol{w}}_q^b \coloneqq \text{fakequant}(\boldsymbol{w}^b, g_b, \Delta^*, z^*)
20: return \mathcal{G}\{\cdot\}
                                                                                                                                                 22: return \hat{\boldsymbol{w}}_{a}^{b}
```

