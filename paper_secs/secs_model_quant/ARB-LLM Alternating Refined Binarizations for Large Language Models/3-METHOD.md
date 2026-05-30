# 3 METHOD

Overview. As shown in Figure [4,](#page-3-0) to progressively align the distribution between binarized and full-precision weights in LLMs, we first propose a framework Alternating Refined Binarization (ARB) in Section [3.1.](#page-3-1) Based on ARB framework, we propose the Alternating Refined Binarization with calibration data (ARB-X) to enhance the usage of the calibration set, which is crucial for binary LLMs. Additionally, we introduce the Alternating Refined Binarization along row-column axes (ARB-RC) to address the column deviation challenge in LLM weights. These methods are detailed in Sections [3.2](#page-4-0) and [3.3,](#page-5-0) respectively. Finally, we discuss our refined strategy to combine salient column bitmap and group bitmap (CGB) in Section [3.4.](#page-5-1) Our final models, ARB-LLM<sup>X</sup> and ARB-LLMRC, are obtained by equipping ARB-X and ARB-RC with CGB respectively.

<span id="page-3-0"></span>![](_page_3_Figure_1.jpeg)

Figure 4: Overview of our ARB series. ARB: alternating refine mean, row scaling factor, and binarized matrix. ARB-X: introducing calibration data into the update of binarization parameters. ARB-RC: alternating refine row and column scaling factors.

## <span id="page-3-1"></span>3.1 ALTERNATING REFINED BINARIZATION (ARB)

We begin by discussing standard weight binarization in LLMs. For a full-precision weight  $\mathbf{W} \in \mathbb{R}^{n \times m}$ , we define the objective of binarization as (with dimension broadcasting omitted for simplicity)

<span id="page-3-2"></span>
$$\underset{\alpha, \mathbf{B}}{\operatorname{arg\,min}} ||\widetilde{\mathbf{W}} - \alpha \mathbf{B}||_F^2, \quad \text{where } \widetilde{\mathbf{W}} = \mathbf{W} - \mu, \, \mu = \frac{1}{m} \sum_{j=1}^m \mathbf{W}_{.j},$$
 (1)

where  $\alpha \in \mathbb{R}^n$  denotes the row-wise scaling factor, and  $\mathbf{B} \in \{+1, -1\}^{n \times m}$  is a binary matrix.

Since the mean of  $\mathbf{W}$  is not necessarily zero, a common practice is to apply a row-wise redistribution before binarization. After redistribution, the weights achieve a row-wise zero-mean distribution, which facilitates the binarization process. Under the objective of binarization (Equation (1)), the optimal solutions for  $\alpha$  and  $\mathbf{B}$  can be solved with  $\alpha = \frac{1}{m} \sum_{j=1}^m |\widetilde{\mathbf{W}}_{.j}|$  and  $\mathbf{B} = \mathrm{sign}(\widetilde{\mathbf{W}})$  respectively. Then we can define the quantization error  $\mathcal{L}_1$  after binarization as

$$\mathcal{L}_1 = ||\mathbf{W} - \widehat{\mathbf{W}}||_F^2, \quad \text{where } \widehat{\mathbf{W}} = \alpha \mathbf{B} + \mu.$$
 (2)

Moving forward, we aim to investigate how to reduce the quantization error  $\mathcal{L}_1$ . We first define the residual matrix as  $\mathbf{R} = \mathbf{W} - \widehat{\mathbf{W}}$ . In analyzing the residual matrix  $\mathbf{R}$ , we observe a distribution shift in  $\mathbf{R}$ , where the mean of  $\mathbf{R}$  is not always zero due to inevitable errors during the binarization process (see Figure 2). To address this, we introduce a correction term  $\delta_{\mu}$  to the original mean  $\mu$ , effectively mitigating the distribution shift. The refined mean is defined as follows:

<span id="page-3-3"></span>
$$\mu_{\text{refine}} = \mu + \delta_{\mu}, \quad \text{where } \delta_{\mu} = \frac{1}{m} \sum_{i=1}^{m} \mathbf{R}_{.j}.$$
 (3)

This is equivalent to taking the partial derivative of  $\mathcal{L}_1$  with respect to  $\mu$  and setting it to 0, as shown in Figure 4. Since  $\mu$  has been updated to  $\mu_{\text{refine}}$ , the original  $\alpha$  and  $\mathbf{B}$  are no longer optimal solutions for quantization error  $\mathcal{L}_1$ . To further minimize the quantization error, the optimal solutions for  $\alpha_{\text{refine}}$  and  $\mathbf{B}_{\text{refine}}$  can be obtained by setting  $\partial \mathcal{L}_1/\partial \alpha = 0$ , leading to the following expressions:

$$\alpha_{\text{refine}} = \frac{1}{m} \operatorname{diag}(\mathbf{B}^{\top}(\mathbf{W} - \mu_{\text{refine}})), \quad \mathbf{B}_{\text{refine}} = \operatorname{sign}(\mathbf{W} - \mu_{\text{refine}}).$$
 (4)

After refining  $\mu$ ,  $\alpha$ , and  $\mathbf{B}$ , we can obtain the  $\widehat{\mathbf{W}}_{\text{refine}}$  as  $\widehat{\mathbf{W}}_{\text{refine}} = \alpha_{\text{refine}} \cdot \mathbf{B}_{\text{refine}} + \mu_{\text{refine}}$ . We find that this parameter update strategy can be extended to an iterative algorithm.

In each iteration, we sequentially update  $\mu$ ,  $\alpha$ , and  $\mathbf{B}$  to ensure they are the optimal solutions under the current quantization error  $\mathcal{L}_1$ . The pseudocode is shown in Algorithm 1, which extends  $\mathbf{ARB}$  with group mask (a bitmap detailed in Section 3.4). Moreover, we theoretically analyze the quantization error during the  $\mathbf{ARB}$  process and derive a specific value for the reduced quantization error after  $\tau$  iterations, as stated in Theorem 1. The proof is provided in supplementary file.

## <span id="page-4-1"></span>Algorithm 1 First-Order Alternating Refined Binarization

```
func ARB^1(\mathbf{W}, \mathbf{M}, T)
                                                                                                                   func binary (\mathbf{W}, \mathbf{M})
                                                                                                                      1: \mu := \frac{1}{m} \sum_{i=1}^{m} (\mathbf{W} \odot \mathbf{M})_{.j}
Input: \mathbf{W} \in \mathbb{R}^{n \times m} - full-precision weight
                 \mathbf{M} \in \mathbb{R}^{n \times m} - group mask
                                                                                                                      2: \widetilde{\mathbf{W}} := \mathbf{W} - \mu
3: \alpha := \frac{1}{m} \sum_{j=1}^{m} |(\widetilde{\mathbf{W}} \odot \mathbf{M})_{.j}|
                 T - total iterations
Output: \mathbf{W} \in \mathbb{R}^{n \times m}
                                                                                                                      4: \mathbf{B} \coloneqq \operatorname{sign}(\mathbf{W} \odot \mathbf{M})
  1: \mathbf{W}, \alpha, \mathbf{B}, \mu \coloneqq \text{binary}(\mathbf{W}, \mathbf{M})
                                                                                                                      5: \widehat{\mathbf{W}} \coloneqq \alpha \cdot \mathbf{B} + \mu
  2: for iter = 1, 2, ..., T do
              \mathbf{R}\coloneqq \mathbf{W}-\widehat{\mathbf{W}} > residual matrix
                                                                                                                      6: return \widehat{\mathbf{W}}, \alpha, \mathbf{B}, \mu
              \delta_{\mu} := \sum_{i} (\mathbf{R} \odot \mathbf{M})_{.j}
                                                                                                                   func refine_alpha (\mathbf{B}, \mathbf{W}, \mathbf{M}, \mu)
              \mu \leftarrow \mu + \delta_{\mu}
                                                                    \triangleright refine mean
                                                                                                                     1: num := \sum_{j=1}^{m} (\mathbf{B}_{.j} \odot \mathbf{M}_{.j}) \cdot (\mathbf{W}_{.j} - \mu)

2: den := \sum_{j=1}^{m} (\mathbf{B}_{.j} \odot \mathbf{M}_{.j})^{2} + \epsilon \quad \triangleright \text{ avoid}
              \alpha \leftarrow \text{refine\_alpha}(\mathbf{B}, \mathbf{W}, \mathbf{M}, \mu)
               \mathbf{B} \leftarrow \operatorname{sign}(\mathbf{W} - \mu)
                                                                             ⊳ refine B
                                                                                                                              zero-division
  8:
               \widehat{\mathbf{W}} \leftarrow \alpha \mathbf{B} + \mu
                                                                                                                      3: \alpha \coloneqq \frac{num}{den}
  9: end for
                                                                                                                      4: return \alpha
10: return W
```

<span id="page-4-2"></span>**Theorem 1.** For any 
$$\tau \geq 0$$
, Algorithm 1 achieves a quantization error  $\mathcal{L}_1^{\tau}$  satisfying 
$$\mathcal{L}_1^{\tau} = \mathcal{L}_1^0 - m((\alpha^{\tau})^2 - (\alpha^0)^2 - (\mu^{\tau} - \mu^0)^2) \leq \mathcal{L}_1^0, \tag{5}$$

where  $\alpha^0$  and  $\mu^0$  denote the initial scaling factor and mean respectively,  $\alpha^{\tau}$ ,  $\mu^{\tau}$ , and  $\mathcal{L}_1^{\tau}$ represent the scaling factor, mean, and quantization error after the  $\tau$ -th iteration respectively.

To achieve better quantization precision, we extend ARB to second-order binarization and apply it to salient weights, following BiLLM (Huang et al., 2024). The second-order binarized matrix W is

$$\widehat{\mathbf{W}} = \alpha_1 \mathbf{B}_1 + \alpha_2 \mathbf{B}_2 + \mu, \tag{6}$$

where  $\mathbf{B}_1, \mathbf{B}_2 \in \{+1, -1\}^{n \times m}$  are binary matrices,  $\alpha_1, \alpha_2 \in \mathbb{R}^n$  are corresponding row-wise scaling factors, and  $\mu \in \mathbb{R}^n$  is the row-wise shifting factor. Based on the first-order **ARB**, we use Equation 3 to update  $\mu$ , then sequentially update  $\alpha_1$  and  $\alpha_2$  by setting  $\partial \mathcal{L}_1/\partial \alpha_1=0$  and  $\partial \mathcal{L}_1/\partial \alpha_2 = 0$  respectively, leading to the following formulas:

$$\tilde{\alpha}_1 = \frac{1}{m}\operatorname{diag}(\mathbf{B}_1^\top(\mathbf{W} - \mu_{\text{refine}} - \alpha_2\mathbf{B}_2)), \quad \tilde{\alpha}_2 = \frac{1}{m}\operatorname{diag}(\mathbf{B}_2^\top(\mathbf{W} - \mu_{\text{refine}} - \tilde{\alpha}_1\mathbf{B}_1)). \quad (7)$$
The final step is to update the binary matrices  $\mathbf{B}_1$  and  $\mathbf{B}_2$ . The objective of refining  $\mathbf{B}_1$  and  $\mathbf{B}_2$  is:

$$\widetilde{\mathbf{B}}_{1}, \widetilde{\mathbf{B}}_{2} = \underset{\mathbf{B}_{1}, \mathbf{B}_{2}}{\operatorname{arg \, min}} ||\mathbf{W} - \mu_{\text{refine}} - \tilde{\alpha}_{1} \mathbf{B}_{1} - \tilde{\alpha}_{2} \mathbf{B}_{2}||_{\ell 1}. \tag{8}$$

Since  $\mathbf{B}_1, \mathbf{B}_2 \in \{+1, -1\}^{n \times m}$ , there are only four possible combinations for  $(\tilde{\alpha}_1 \mathbf{B}_1 + \tilde{\alpha}_2 \mathbf{B}_2)$ . Thus, we construct a candidate vector  $\mathbf{V} = \{-\tilde{\alpha}_1 - \tilde{\alpha}_2, -\tilde{\alpha}_1 + \tilde{\alpha}_2, +\tilde{\alpha}_1 - \tilde{\alpha}_2, +\tilde{\alpha}_1 + \tilde{\alpha}_2\} \in \mathbb{R}^4$ , then use binary search to find the combination that is closest to  $(W - \mu_{refine})$ . The corresponding elements of  ${\bf B}_1$  and  ${\bf B}_2$  are then determined accordingly. Detailed pseudocode is provided in supplementary file.

## <span id="page-4-0"></span>3.2 ARB WITH CALIBRATION DATA (ARB-X)

Although the **ARB** algorithm can effectively reduce the quantization error  $\mathcal{L}_1$ , we observe that the weight matrix W operates in conjunction with the input data to produce the output. It means that  $\mathcal{L}_1$  alone does not fully capture the true impact of quantization. To address this issue, we introduce calibration data X and define a new quantization error  $\mathcal{L}_2$  as  $\mathcal{L}_2 = ||\mathbf{WX} - \mathbf{WX}||_F^2$ . Based on  $\mathcal{L}_2$ and the **ARB** algorithm, we propose an extended algorithm, naming **ARB-X**.

**Reformulation.** However, incorporating calibration data necessitates a large number of matrix multiplications when computing  $\mathcal{L}_2$ , substantially increasing computational overhead, and often making the combination of calibration data impractical. To address this issue, we reformulate the error computation by decoupling the calibration data and weight matrix as:

$$\mathcal{L}_2 = \langle \mathbf{S}, \mathbf{R}^{\mathsf{T}} \mathbf{R} \rangle_F = \text{Tr}(\mathbf{R} \mathbf{S} \mathbf{R}^{\mathsf{T}}), \text{ where } \mathbf{S} = \sum_b \mathbf{X}_b^T \mathbf{X}_b, \mathbf{R} = \mathbf{W} - \mu - \alpha \mathbf{B}.$$
 (9)

 $\mathbf{X} \in \mathbb{R}^{B \times L \times m}$  denotes the calibration data with batch size B, sequence length L, and embedding dimension m. By compressing the high-dimensional tensor X into a 2D matrix  $\mathbf{S} \in \mathbb{R}^{m \times m}$  and precomputing it, we can significantly reduce the computational overhead. To quantify the efficiency improvement of our reformulation, we define the speedup ratio  $\eta$ , which denotes the ratio between the time complexity of the original error computation and that of the revised method. We present the theoretical result in Theorem 2, with the proof provided in the supplementary file.

<span id="page-5-2"></span>**Theorem 2.** The speedup ratio  $\eta$  of the reformulation compared to the original method is

$$\eta \propto \frac{1}{k \cdot \left(\frac{1}{n \cdot T} + \frac{1}{B \cdot L}\right)},$$
(10)

where n is the hidden dimension of  $\mathbf{W}$ , k is the block size, and T is the number of iterations.

Typically, we set n to 4,096, B to 128, L to 2,048, T to 15, and k to 128. Under these circumstances,  $\eta$  is proportional to 389, meaning that the reformulated method is approximately 389× faster than the original one. Further details are provided in supplementary file.

**Parameter Update.** By combining the parameter updating strategy with the reformulated  $\mathcal{L}_2$ , we can derive the parameter update formulas for **ARB-X** by setting  $\partial \mathcal{L}_2/\partial \mu = 0$  and  $\partial \mathcal{L}_2/\partial \alpha = 0$ , which results in the sequential updates of  $\mu$  and  $\alpha$  respectively:

$$\mu = \frac{\mathbf{1}^{\top} \mathbf{S} (\mathbf{W} - \alpha \mathbf{B})^{\top}}{\mathbf{1}^{\top} \mathbf{S} \mathbf{1}}, \quad \alpha = \frac{\operatorname{diag} (\mathbf{B} \mathbf{S} (\mathbf{W} - \mu)^{\top})}{\operatorname{diag} (\mathbf{B} \mathbf{S} \mathbf{B}^{\top})}.$$
 (11)

More details are provided in supplementary file. It is worth noting that, during this process, the matrix  $\bf B$  is not updated. Since  $\bf B$  consists of discrete values (*i.e.*, +1 and -1), it is not possible to update  $\bf B$  directly by setting the partial derivative of  $\mathcal{L}_2$  with respect to  $\bf B$  to zero. The pseudocodes for the first-order and second-order  $\bf ARB-X$  are provided in supplementary file.

#### <span id="page-5-0"></span>3.3 ARB ALONG ROW-COLUMN AXES (ARB-RC)

Previous binarization methods use a row-wise scaling factor  $\alpha^r$  for weight binarization. However, our analyses of the numerical distribution of the weight matrix **W** in LLMs reveal significant deviations across columns, with some columns exhibiting notably larger values (Figure 3). As a result, using a single row-wise scaling factor may not effectively capture the distribution characteristics of LLM parameters. Additionally, the weight distribution shows a mean close to zero, making the redistribution to zero-mean less effective in LLM binarization.

To address this, we propose the **ARB-RC** algorithm, which introduces a column-wise scaling factor  $\alpha^c$  to better handle parameter variations across columns, while eliminating the redistribution parameter  $\mu$  to enhance compression in LLMs. The row-column binarization process is performed as follows:

$$\alpha^r = \frac{1}{m} \sum_{j=1}^m |\mathbf{W}_{.j}|, \quad \alpha^c = \frac{1}{n} \sum_{j=1}^n |\frac{\mathbf{W}_{j.}}{\alpha_j^r}|, \quad \mathbf{B} = \text{sign}(\mathbf{W}).$$
 (12)

Then, we can obtain the binarized matrix as  $\widehat{\mathbf{W}} = \alpha^r \alpha^c \mathbf{B}$ , where removing  $\mu$  while introducing  $\alpha^c$  reduces parameters but improves model performance. However, introducing  $\alpha^c$  without adopting an alternating parameter update strategy fails to improve performance and can even increase quantization error. Thus, it is necessary to combine  $\alpha^c$  with the discussed **ARB** algorithm. In this approach, we optimize the parameters using the quantization error  $\mathcal{L}_1$ . Although the quantization error  $\mathcal{L}_2$  is more aligned with real-world conditions, our analysis shows that incorporating **X** in the **ARB-RC** method results in parameter coupling, making optimization difficult (detailed in supplementary file). Thus, based on  $\mathcal{L}_1$ , we can update  $\alpha^r$  and  $\alpha^c$  by setting  $\partial \mathcal{L}_1/\partial \alpha^r = 0$  and  $\partial \mathcal{L}_1/\partial \alpha^c = 0$  respectively:

based on 
$$\mathcal{L}_1$$
, we can update  $\alpha^r$  and  $\alpha^c$  by setting  $\partial \mathcal{L}_1/\partial \alpha^r = 0$  and  $\partial \mathcal{L}_1/\partial \alpha^c = 0$  respectively:
$$\alpha^r = \frac{\operatorname{diag}(\mathbf{W}(\alpha^c \mathbf{B})^\top)}{\operatorname{diag}((\alpha^c \mathbf{B})(\alpha^c \mathbf{B})^\top)}, \quad \alpha^c = \frac{\operatorname{diag}(\mathbf{W}^\top(\alpha^r \mathbf{B}))}{\operatorname{diag}((\alpha^r \mathbf{B})^\top(\alpha^r \mathbf{B}))}. \tag{13}$$

The first-order and second-order pseudocodes of the **ARB-RC** are provided in supplementary file.

## <span id="page-5-1"></span>3.4 COLUMN-GROUP BITMAP (CGB)

Inspired by BiLLM (Huang et al., 2024), we partition the entire set of weights into salient and non-salient columns, and apply higher-bit representation, *i.e.*, second-order binarization, to the salient weights. However, different from BiLLM, we not only divide the non-salient weights into sparse and concentrated groups but also divide salient weights in a similar manner. This approach allows for more efficient use of both column bitmap and group bitmap, as shown in Figure 5.

To identify the sensitivity of weights, *i.e.*, salient weights, we follow well-established PTQ methods by utilizing the Hessian matrix as a standard criterion. The sensitivity is computed as  $s_i = w_i^2/[\mathbf{H}^{-1}]_{ii}^2$ ,

<span id="page-6-1"></span>

|                       | 1             | ,              |           |           |           |               |            |              |
|-----------------------|---------------|----------------|-----------|-----------|-----------|---------------|------------|--------------|
| Method                | Block<br>Size | Weight<br>Bits | 1.3B      | 2.7B      | 6.7B      | 13B           | 30B        | 66B          |
| Full Precision        | -             | 16.00          | 14.62     | 12.47     | 10.86     | 10.13         | 9.56       | 9.34         |
| RTN                   | -             | 3.00           | 13,337.38 | 15,594.72 | 5,797.32  | 3,357.01      | 1,566.00   | 6,126.09     |
| GPTQ                  | 128           | 3.00           | 16.45     | 13.61     | 11.31     | 10.47         | 9.71       | 10.55        |
| RTN                   | -             | 2.00           | 11,272.65 | 9,505.76  | 28,363.14 | 194,086.78    | 169,616.47 | 1,165,864.25 |
| GPTQ                  | 128           | 2.00           | 121.64    | 59.53     | 20.81     | 20.05         | 13.04      | 46.38        |
| RTN                   |               | 1.00           | 17,165.72 | 36,516.69 | 11,550.91 | 69,863,488.00 | 6,485.99   | 184,796.30   |
| GPTQ                  | 128           | 1.00           | 8,719.58  | 11,700.13 | 6,633.13  | 1,743,929.88  | 14,083.15  | 11,045.36    |
| PB-LLM                | 128           | 1.70           | 239.81    | 278.27    | 144.25    | 74.59         | 28.30      | 27.66        |
| BiLLM                 | 128           | 1.11           | 69.05     | 48.61     | 47.65     | 18.75         | 13.86      | 12.05        |
| ARB-LLM <sub>X</sub>  | 128           | 1.11           | 45.40     | 34.37     | 20.07     | 15.47         | 12.36      | 11.23        |
| ARB-LLM <sub>RC</sub> | 128           | 1.11           | 26.63     | 19.84     | 14.92     | 12.92         | 11.12      | 10.30        |

Table 1: Perplexity of RTN, GPTQ, PB-LLM, BiLLM, and our methods on **OPT** family. The columns represent the perplexity results on **WikiText2** datasets with different model sizes.

where H represents the Hessian matrix for each layer, and  $w_i$  represents the original value of each weight element. Weight columns with higher  $s_i$  are selected as salient columns, which are then marked using the salient column bitmap. For more details, please refer to Huang et al. (2024).

For non-salient columns, BiLLM further divides them into sparse and concentrated groups based on their magnitude, marking them using a group bitmap. Although this grouping strategy significantly reduces the quantization error, it can be further refined since some regions of the group bitmap are underutilized. As shown on the left side of Figure 5, the salient columns of the group bitmap remain unused. Thus, to better utilize the space of the group bitmap, we optimize the combination of the column bitmap and group bitmap. Specifically, we further categorize the salient weights into sparse and concentrated groups, which improve the quantization accuracy of salient weights without increasing bitmap stor- Figure 5: Comparison between BiLLM and our age. Our combination format is defined as follows: combination of column and group bitmaps.

<span id="page-6-0"></span>![](_page_6_Figure_5.jpeg)

$$\mathbf{G_s} = \mathbf{1}_n \mathbf{C_s}^{\top} \odot \mathbf{G}, \quad \mathbf{G_{ns}} = \mathbf{1}_n \mathbf{C_{ns}}^{\top} \odot \mathbf{G},$$
 (14)

where  $G_s$  and  $G_{ns}$  represent group bitmaps for salient and non-salient weights, respectively.  $C_s$ indicates the salient columns, while  $C_{ns} = \neg C_s$  indicates the non-salient columns. We extend the column bitmap along the row axis and then perform element-wise multiplication with the group bitmap to obtain the final partitions. Experiments demonstrate that our Column-Group Bitmap (CGB) further enhances the quantization performance when applied to ARB algorithms. Additionally, following BiLLM, we adopt the block-wise compensation (Frantar et al., 2023; Frantar & Alistarh, 2022) to mitigate quantization errors. For further details, please refer to their papers.

#### <span id="page-6-2"></span>EXPERIMENTS

#### 4.1 Setup

All the experiments are conducted with PyTorch (Paszke et al., 2019b) and Huggingface (Paszke et al., 2019a) on a single NVIDIA A800-80GB GPU. We implement 15 iterations for ARB-LLM<sub>x</sub> and ARB-LLM<sub>RC</sub> to ensure the convergence of binarization parameters. Following Frantar et al. (2023) and Huang et al. (2024), we use 128 samples from C4 (Raffel et al., 2020) dataset as calibration data.

**Models and Datasets.** We conduct extensive experiments on the LLaMA, LLaMA-2, and LLaMA-3 families (Touvron et al., 2023), the OPT family (Zhang et al., 2022), and instruction-tuned LLMs Vicuna (Chiang et al., 2023). To evaluate the effectiveness of our proposed ARB-LLM<sub>X</sub> (ARB-X + CGB) and ARB-LLM<sub>RC</sub> (ARB-RC + CGB), we measure the perplexity of LLM's outputs on WikiText2 (Merity et al., 2017), PTB (Marcus et al., 1994), as well as a part of the C4 (Raffel et al., 2020) data. Moreover, we also evaluate the accuracy for 7 zero-shot QA datasets: ARC-c (Clark et al., 2018), ARC-e (Clark et al., 2018), BoolQ (Clark et al., 2019), Hellaswag (Zellers et al., 2019), OBQA (Mihaylov et al., 2018), PIQA (Bisk et al., 2020), and Winogrande (Sakaguchi et al., 2020).

Table 2: Perplexity of RTN, GPTQ, PB-LLM, BiLLM, and our methods on LLaMA family. The columns represent the perplexity results on **WikiText2** dataset with different model sizes. N/A: LLaMA-2 lacks a 30B version, and LLaMA-3 lacks both 13B and 30B versions. \*: LLaMA has a 65B version, while both LLaMA-2 and LLaMA-3 have 70B versions.

<span id="page-7-0"></span>

| Model   | Method                                                | Block<br>Size | Weight<br>Bits | 7B/8B*       | 13B          | 30B       | 65B/70B*   |
|---------|-------------------------------------------------------|---------------|----------------|--------------|--------------|-----------|------------|
|         | Full Precision                                        | -             | 16.00          | 5.68         | 5.09         | 4.10      | 3.53       |
|         | RTN                                                   | -             | 3.00           | 25.54        | 11.40        | 14.89     | 10.59      |
|         | GPTQ                                                  | 128           | 3.00           | 8.63         | 5.67         | 4.87      | 4.17       |
|         | RTN                                                   | -             | 2.00           | 106,767.34   | 57,409.93    | 26,704.36 | 19,832.87  |
|         | GPTQ                                                  | 128           | 2.00           | 129.19       | 20.46        | 15.29     | 8.66       |
| LLaMA   | RTN                                                   |               | 1.00           | 168,388.00   | 1,412,020.25 | 14,681.76 | 65,253.24  |
|         | GPTQ                                                  | 128           | 1.00           | 164,471.78   | 131,505.41   | 10,339.15 | 20,986.16  |
|         | PB-LLM                                                | 128           | 1.70           | 82.76        | 44.93        | 23.72     | 12.81      |
|         | BiLLM                                                 | 128           | 1.09           | 49.79        | 14.58        | 9.90      | 8.37       |
|         | ARB-LLM <sub>X</sub>                                  | 128           | 1.09           | 21.81        | 11.20        | 8.66      | 7.27       |
|         | ARB-LLM <sub>RC</sub>                                 | 128           | 1.09           | 14.03        | 10.18        | 7.75      | 6.56       |
|         | Full Precision                                        | -             | 16.00          | 5.47         | 4.88         | N/A       | 3.32       |
|         | RTN                                                   | _             | 3.00           | 542.80       | 10.68        | N/A       | 7.53       |
|         | GPTQ                                                  | 128           | 3.00           | 6.44         | 5.46         | N/A       | 3.88       |
|         | RTN                                                   | -             | 2.00           | 17,788.94    | 51,145.61    | N/A       | 26,066.13  |
|         | GPTQ                                                  | 128           | 2.00           | 52.22        | 23.63        | N/A       | 8.18       |
| LLaMA-2 | RTN                                                   |               | 1.00           | 157,058.34   | 47,902.32    | N/A       | 160,389.91 |
|         | GPTQ                                                  | 128           | 1.00           | 59,758.69    | 22,926.54    | N/A       | 14,219.35  |
|         | PB-LLM                                                | 128           | 1.70           | 66.41        | 236.40       | N/A       | 28.37      |
|         | BiLLM                                                 | 128           | 1.08           | 32.31        | 21.35        | N/A       | 13.32      |
|         | ARB-LLM <sub>X</sub>                                  | 128           | 1.08           | 21.61        | 14.86        | N/A       | 7.88       |
|         | ARB-LLM <sub>RC</sub>                                 | 128           | 1.08           | 16.44        | 11.85        | N/A       | 6.16       |
|         | Full Precision                                        | -             | 16.00          | 6.14         | N/A          | N/A       | 2.86       |
|         | RTN                                                   | _             | 3.00           | 2,194.98     | N/A          | N/A       | 13,592.69  |
|         | GPTQ                                                  | 128           | 3.00           | 18.68        | N/A          | N/A       | 6.65       |
|         | RTN                                                   | _             | 2.00           | 1,335,816.13 | N/A          | N/A       | 481,927.66 |
|         | GPTQ                                                  | 128           | 2.00           | 1,480.43     | N/A          | N/A       | 82.23      |
| LLaMA-3 | RTN                                                   |               | 1.00           | 1,353,698.38 |              | N/A       | 375,658.34 |
|         | GPTQ                                                  | 128           | 1.00           | 1,121,260.50 | N/A          | N/A       | 130,516.50 |
|         | PB-LLM                                                | 128           | 1.70           | 73.08        | N/A          | N/A       | 22.96      |
|         | BiLLM                                                 | 128           | 1.06           | 55.80        | N/A          | N/A       | 66.30      |
|         | $\bar{A}\bar{R}\bar{B}$ - $\bar{L}\bar{L}\bar{M}_X^-$ | 128           | 1.06           | 31.98        | N/A          | N/A       | 14.15      |
|         | ARB-LLM <sub>RC</sub>                                 | 128           | 1.06           | 27.42        | N/A          | N/A       | 11.10      |

**Baselines.** We mainly compare our ARB series with BiLLM (Huang et al., 2024), the SOTA PTQ approach on binary LLMs. Other recent PTQ algorithms, such as RTN (round-to-nearest), GPTQ (Frantar et al., 2023), and PB-LLM (Shang et al., 2024) are also selected.

#### 4.2 Main Results

We follow BiLLM to report the average bitwidth of all methods, where our methods have the same bit-width as BiLLM. Table 1 presents the perplexity comparison of the OPT family across different model sizes. It can be observed

Table 3: Perplexity of GPTQ, PB-LLM, BiLLM, and our methods on **Vicuna** family. The columns represent the perplexity results on **WikiText2** datasets with different model sizes.

<span id="page-7-1"></span>

| Method                                        | Block<br>Size | Weight<br>Bits | 7B             | 13B            |
|-----------------------------------------------|---------------|----------------|----------------|----------------|
| Full Precision                                | -             | 16.00          | 6.34           | 5.57           |
| GPTQ                                          | 128           | 2.00           | 688.08         | 37.97          |
| PB-LLM                                        | 128           | 1.70           | 58.68          | 2,506.44       |
| BiLLM                                         | 128           | 1.08           | 39.36          | 43.39          |
| ARB-LLM <sub>X</sub><br>ARB-LLM <sub>RC</sub> | 128<br>128    | 1.08<br>1.08   | 22.79<br>17.60 | 13.76<br>13.38 |

that both ARB-LLM $_{\rm X}$  and ARB-LLM $_{\rm RC}$  significantly outperform SOTA BiLLM, and reduce the perplexity by up to **68.7%** without increasing weight bit-width. Table 2 presents the perplexity comparison on LLaMA1&2&3 families, which also suggests the superior performance of our ARB-LLM. It is noteworthy that ARB-LLM $_{\rm RC}$  outperforms RTN with 3-bit quantization on some models, such as the LLaMA1&3 families, LlaMA2-70B model, as well as OPT family. Similarly, ARB-LLM $_{\rm RC}$ 

Table 4: Ablation study on LLaMA-7B, where all ARB methods are equipped with CGB except for ablation (b). Results are measured by perplexity, with final results highlighted in **bold**.

(a) Effectiveness of two advanced variants

| (b) Effectiveness of CG. | tiveness of CGE |
|--------------------------|-----------------|
|--------------------------|-----------------|

<span id="page-8-1"></span>

| Method               | Calibration update | Row-column<br>update | WikiText2 ↓ | <b>C4</b> ↓ |
|----------------------|--------------------|----------------------|-------------|-------------|
| BiLLM                | -                  | -                    | 49.79       | 46.96       |
| ARB                  | ×                  | ×                    | 22.67       | 26.44       |
| ARB-LLM <sub>X</sub> | ✓                  | X                    | 21.81       | 22.73       |
| $ARB-LLM_{RC}$       | X                  | ✓                    | 14.03       | 17.92       |

<span id="page-8-2"></span>

| Method               | CGB | WikiText2 ↓ | <b>C4</b> ↓ |
|----------------------|-----|-------------|-------------|
| BiLLM                | -   | 49.79       | 46.96       |
| ARB-LLM <sub>X</sub> | ×   | 26.29       | 27.11       |
| $ARB-LLM_X$          | /   | 21.81       | 22.73       |
| $ARB-LLM_{RC}$       | X   | 15.85       | 19.42       |
| $ARB-LLM_{RC}$       | ✓   | 14.03       | 17.92       |

<span id="page-8-3"></span>(c) Study of decoupling column and group bitmaps

<span id="page-8-4"></span>(d) Study of ARB-LLM<sub>X</sub> calibration set size

| Method                | Column<br>bitmap | Group<br>bitmap | WikiText2 $\downarrow$ | <b>C4</b> ↓ |
|-----------------------|------------------|-----------------|------------------------|-------------|
| ARB-LLM <sub>RC</sub> | Х                | Х               | 10,942.45              | 11,032.93   |
| $ARB-LLM_{RC}$        | /                | X               | 369.20                 | 205.56      |
| $ARB-LLM_{RC}$        | X                | /               | 920.42                 | 572.69      |
| ARB-LLM <sub>RC</sub> | ✓                | ✓               | 14.03                  | 17.92       |

| Method               | Calibration set size | WikiText2 $\downarrow$ | <b>C4</b> ↓ |
|----------------------|----------------------|------------------------|-------------|
| BiLLM                | 128                  | 49.79                  | 46.96       |
| ARB-LLM <sub>X</sub> | 64                   | 24.79                  | 25.11       |
| $ARB-LLM_X$          | 128                  | 21.81                  | 22.73       |
| ARB-LLM <sub>X</sub> | 256                  | 21.88                  | 24.28       |

(e) Study of ARB-LLM iteration number

(f) Study of ARB-LLM group number

<span id="page-8-5"></span>

| Method               | #Iteration | WikiText2 ↓                  |
|----------------------|------------|------------------------------|
| BiLLM                | 0          | 49.79                        |
| ARB-LLM <sub>X</sub> | 1/3/15     | 22.59 / 21.12 / <b>21.81</b> |
| $ARB-LLM_{RC}$       | 1/3/15     | 15.23 / 14.34 / <b>14.03</b> |

<span id="page-8-6"></span>

| Method               | #Group | WikiText2 ↓          | C4 ↓                 |
|----------------------|--------|----------------------|----------------------|
| BiLLM                | 2      | 49.79                | 46.96                |
| ARB-LLM <sub>X</sub> | 2/4    | <b>21.81</b> / 6.55  | <b>22.73</b> / 8.56  |
| $ARB-LLM_{RC}$       | 2/4    | <b>14.03</b> / 12.77 | <b>17.92</b> / 16.06 |

<span id="page-8-0"></span>![](_page_8_Figure_14.jpeg)

Figure 6: Average accuracy of 7 zero-shot QA datasets on LLaMA1&2&3 families.

also surpasses GPTQ with 3-bit quantization on OPT-66B model. For *instruction-tuned* Vicuna comparison shown in Table 3, ARB-LLM $_{\rm X}$  and ARB-LLM $_{\rm RC}$  also show superior performance, surpassing SOTA binary PTQ method BiLLM for a large margin. Regarding average accuracy on QA datasets, ARB-LLM $_{\rm X}$  and ARB-LLM $_{\rm RC}$  both significantly outperform previous methods, as shown in Figure 6. More results are provided in the supplementary file.

## 4.3 ABLATION STUDY

**Effectiveness of Advanced Variants.** To validate the effectiveness of our advanced variants ARB-LLM $_{\rm X}$  and ARB-LLM $_{\rm RC}$ , we compare them with the vanilla ARB algorithm in Table 4a. First, we observe that the vanilla ARB already significantly outperforms BiLLM. Furthermore, by introducing either the calibration update or the row-column update to the binarization process, performance is further improved. This demonstrates that our advanced variants, ARB-LLM $_{\rm X}$  and ARB-LLM $_{\rm RC}$ , can further enhance the performance of binary LLMs based on ARB.

**Effectiveness of CGB.** To demonstrate the effectiveness of our column-group bitmap (CGB), we conduct an ablation study in Table 4b. In this study, the absence of CGB does not imply the exclusion of partitioning but rather the use of the partitioning strategy used by BiLLM. The results show that CGB further enhances the performance of both ARB-LLM $_{\rm X}$  and ARB-LLM $_{\rm RC}$ . Notably, even when using BiLLM's partitioning strategy, our methods significantly outperform BiLLM.

**Column Bitmap and Group Bitmap.** We use a column bitmap to differentiate between salient and non-salient weights, and a group bitmap to separate weights based on their magnitude. The combination of column and group bitmaps creates four distinct zones. As shown in Table 4c, we explore the effect of decoupling this combination by using either the column bitmap or the group bitmap individually. It is evident that using the column bitmap or group bitmap only will result in a significant performance drop. Omitting both column bitmap and group bitmap entirely (*i.e.*, #group=1), which reduces the method to naive binarization, leads to complete failure.

**Calibration Set Size.** Similar to other PTQ methods, our ARB-LLM requires a small calibration set of just 128 samples. We further incorporate the calibration data into the update of binarization parameters in ARB-LLM<sub>X</sub>. To explore the effect of calibration set size on performance, we compare results using different set sizes, as shown in Table 4d. It can be observed that using fewer calibration samples (*e.g.*, 64) results in a performance drop, while increasing the calibration set size from 128 to 256 yields similar results. This indicates that our ARB-LLM<sub>X</sub> requires only a small calibration set. Even with just 64 samples, ARB-LLM<sub>X</sub> significantly outperforms the baseline BiLLM.

**ARB Iteration Number.** We use 15 iterations for the main results (Table 1, Table 2, Table 3, and Figure 6), as all parameters have fully converged. To explore the impact of different iteration numbers, we compare results using 1, 3, and 15 iterations in Table 4e. As can be seen, regardless of the iteration number, the perplexity of ARB-LLM<sub>X</sub> and ARB-LLM<sub>RC</sub> significantly outperforms the baseline BiLLM. Increasing the iteration number further reduces perplexity, yet they can achieve superior results even with just one iteration. Additionally, we visualize the changes in the scaling factor  $\alpha$  throughout the alternating iterations to provide further insights in supplementary file.

**Group Number.** Following BiLLM (Huang et al., 2024), we introduce an additional bitmap for grouping weights, which has been demonstrated to enhance performance. To explore the impact of group size, we expand the group bitmap from a 1-bit to a 2-bit system, increasing the number of groups from 2 to 4. As shown in Table 4f, increasing the number of groups leads to better performance, especially for ARB-LLM<sub>X</sub>, which outperforms ARB-LLM<sub>RC</sub> with the same number of groups. Yet, this also results in extra storage (about 0.8 GB for LLaMA-7B). In contrast, using only one group (*i.e.*, the first row of Table 4c) results in total failure. Given the additional storage overhead, the 2-group configuration strikes a good balance between performance and memory efficiency.

## 4.4 TIME AND MEMORY ANALYSES

**Time Comparison.** As a binary PTQ framework, ARB-LLM eliminates the need for finetuning. The alternating algorithm requires more computation to align the distribution progressively, yet this overhead is acceptable. In Table 5, ARB-LLM<sub>RC</sub> with 15 iterations requires only 21 more minutes than BiLLM, while ARB-LLM<sub>RC</sub> (without CGB) regults in only 3 more minutes than BiLLM using just 1 iteration. The combination of CCR results in an increase of time worked due to

<span id="page-9-0"></span>Table 5: Time comparison between BiLLM and our ARB-LLM methods on LLaMA-7B.

| Method                | CGB | #Iter=1 | #Iter=3   | #Iter=15 |
|-----------------------|-----|---------|-----------|----------|
| BiLLM                 | -   | 45      | min (#Ite | er=0)    |
| ARB-LLM <sub>X</sub>  | X   | 52 min  | 59 min    | 70 min   |
| $ARB-LLM_X$           | ✓   | 72 min  | 78 min    | 88 min   |
| $ARB-LLM_{RC}$        | X   | 48 min  | 49 min    | 53 min   |
| ARB-LLM <sub>RC</sub> | ✓   | 67 min  | 68 min    | 76 min   |

CGB results in an increase of time overhead, due to the percentile search for optimal splitting.

**Memory Comparison.** Following PB-LLM and BiLLM, we present the memory usage with Raw bitmap / CSR compressed bitmap in Table 6. ARB-LLM<sub>RC</sub>, which replaces the row-wise mean with a column-wise scaling factor, achieves a higher compression ratio along with better performance. Although the refined column-group bitmap (CGB) strategy requires more memory due to more scaling factors, the combination of ARB-RC and CGB still results in lower storage requirements than BiLLM, while delivering outstanding performance. As shown in Table 6, ARB-LLM<sub>RC</sub> with or without CGB both require less storage

<span id="page-9-1"></span>Table 6: Memory (GB, Raw bitmap / CSR bitmap) comparison between FP16, PB-LLM, BiLLM, and our ARB-LLM methods.

| Method                | CGB  | LLaMA-7B    | LLaMA-13B   |
|-----------------------|------|-------------|-------------|
| FP16                  |      | 13.48       | 26.03       |
| PB-LLM                | -    | 2.91 / 2.21 | 5.33 / 3.96 |
| BiLLM                 | -    | 2.93 / 2.19 | 5.36 / 3.92 |
| ARB-LLM <sub>X</sub>  | _ X_ | 2.93 / 2.19 | 5.36 / 3.92 |
| $ARB-LLM_X$           | /    | 3.23 / 2.49 | 5.95 / 4.51 |
| $ARB-LLM_{RC}$        | ×    | 2.63 / 1.89 | 4.77 / 3.33 |
| ARB-LLM <sub>RC</sub> | ✓    | 2.83 / 2.09 | 5.17 / 3.73 |

than previous methods. The computation formulas can be found in supplementary file.

## 5 CONCLUSION

In this work, we propose ARB-LLM, a series of alternating refined binarization (ARB) methods for LLMs. Through the analyses of the distribution shift between binarized and full-precision weights, we propose an alternating refinement of binarization parameters to progressively align the weight distribution. Moreover, we extend the basic ARB by equipping the calibration data and scaling along row-column axes, resulting in ARB-X and ARB-RC respectively. Additionally, we propose a refined strategy to better combine the salient column bitmap and group bitmap. Our experiments on multiple open-source LLM families show that the final models ARB-LLM $_{\rm X}$  and ARB-LLM $_{\rm RC}$  can further push the performance boundary from the SOTA binary PTQ methods.

