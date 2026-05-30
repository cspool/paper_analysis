# 3 METHODOLOGY

In this section, we introduce AffineQuant, an approach that utilizes equivalent affine transformation for quantization. Compared to other methods, AffineQuant consistently maintains optimal mean square error throughout the optimization process. We also explore the reversibility of the affine transform matrix during optimization. To ensure stability, we propose a gradual masking approach based on the Levy–Desplanques theorem [\(Naimark & Zeheb, 1997\)](#page-10-5) to maintain the affine transform matrix as a strictly diagonally dominant matrix. Lastly, we analyze the inference efficiency of LLMs following the optimization performed by AffineQuant.

#### 3.1 AFFINEQUANT

When considering the concept of equivalent transformations from a physical perspective, we can draw analogies to certain operations. For instance, in SmoothQuant [\(Xiao et al., 2023\)](#page-11-2), we can analogize scale to scaling operations for vectors, while in Outlier Suppression+ [\(Wei et al., 2023\)](#page-11-3), we can analogize shift to translation operations for vectors. Similarly, rotations of vectors can also be classified as equivalent transformations.

We define the pseudo-quantization function as follows:

$$Q(x) = \Delta * \left( clamp\left( \left\lfloor \frac{x}{\Delta} \right\rceil + zp, 0, 2^n - 1 \right) - zp \right), \tag{1}$$

where ∆, zp and n are the quantization step-size, zero point and bits, respectively. ⌊·⌉ is the rounding operation. As depicted in Figure [1,](#page-1-0) AffineQuant involves left-multiplying the affine transform matrix A by weight matrix W to better align the weight distribution with the quantization function Q(·). Expanding the optimization space enables smaller quantization errors in the transformed weights, leading to a reduction in perplexity. Simultaneously, we right-multiply the inverse of the affine transform matrix A by the activation value X to maintain the invariance of the matrix multiplication output between activations and weights. For a single linear layer, AffineQuant formulates the following optimization problem:

<span id="page-3-0"></span>
$$\underset{A}{\operatorname{arg\,min}} \left\| XW - XA^{-1}\mathcal{Q}(AW) \right\|_F^2. \tag{2}$$

AffineQuant incorporates the essence of AWQ [\(Lin et al., 2023\)](#page-9-0) and SmoothQuant [\(Xiao et al., 2023\)](#page-11-2) when the main diagonal elements of the matrix A are computed from weight and activation statistics. It aligns with OmniQuant [\(Shao et al., 2023\)](#page-10-4) by exclusively updating the diagonal elements of A. The reordering matrices used in RPTQ [\(Yuan et al., 2023\)](#page-11-4) are a subset of the affine transformation matrix A when each row and column of A contains only one occurrence of the element 1. In summary, AffineQuant encompasses various previous equivalent quantization algorithms, thereby expanding the optimization possibilities for the weight distribution W.

In Figure [1,](#page-1-0) let the weight matrix W ∈ R <sup>2</sup>×<sup>2</sup> have 2 output channels and input channels. The scaling factor, translation factor, and affine transformation matrix are denoted as s, b, and A, respectively. We divide the weight matrix into 2 vectors {v1, v2} based on output channels. The scaling transform s<sup>i</sup> ∗ v<sup>i</sup> uniformly scales each element of v<sup>i</sup> . The translation transform v<sup>i</sup> + b<sup>i</sup> shifts v<sup>i</sup> along different axes. The affine transformation Av<sup>i</sup> allows for arbitrary repositioning of v<sup>i</sup> . However, the scaling and translation transformations are limited in their ability to map dimensions in v<sup>i</sup> to adjacent quantized fixed points. In contrast, the affine transformation guarantees convergence of all dimensions in a vector to the quantized fixed point. In other words, the affine transformation aligns the weight distribution with the noise introduced by the quantization function Q(x) in Equation [2,](#page-3-0) resulting in reduced quantization error. It is worth noting that normalizing the affine transformation matrix by rows A → s ′ A ′ , where each row of the matrix A ′ has a norm of 1, transforms A ′ into a standard rotation matrix. This rotation matrix rotates the output channels of the weights while preserving their magnitudes. The scaling factor s ′ performs scaling on the rotated vectors. Therefore, the affine transformation matrix A combines both scaling and rotation equivalent transformations and is orthogonal to the translation transformation.

The perplexity (ppl) exhibits an exponential relationship with the cross-entropy (CE) loss, which is positively correlated with the mean square error of the output activation before and after quantization, as demonstrated in [\(Nagel et al., 2020;](#page-10-8) [Li et al., 2021\)](#page-9-6). Hence, optimizing perplexity can be achieved by optimizing the mean square error before and after quantization. Specifically,

$$PPL \propto \mathcal{L}_{CE} \propto \left\| XW - XA^{-1}\mathcal{Q}(AW) \right\|_F^2,$$
 (3)

In large language models quantization, the optimization objective of AffineQuant is as follows,

<span id="page-3-1"></span>
$$\underset{A,\delta}{\operatorname{arg\,min}} \left\| f_i\left(X,W\right) - f_i\left(\left(X - \delta\right)A^{-1}, \mathcal{Q}\left(AW\right), b + \delta W\right) \right\|_F^2. \tag{4}$$

where f<sup>i</sup> is the i-th transformer block. (X − δ) A<sup>−</sup><sup>1</sup> , Q (AW), b + δW are the activation, weight and bias after the equivalent transformation, respectively. We combine the affine and translation transformations and use the mean square error of the transformer block output, both pre- and postquantization, as the optimization objective.

![](_page_4_Figure_1.jpeg)

<span id="page-4-0"></span>Figure 2: The gradual mask operates on the affine transformation matrix, gradually incorporating the elements of matrix A near the diagonal into the training process as training progresses.

Figure 3 illustrates the mean square error loss optimization for the last transformer block of LLaMA-7B and OPT-1.3B. Notably, AffineQuant exhibits a lower initial loss compared to OmniQuant (Shao et al., 2023) due to the superior performance of the affine transformation matrices in the preceding blocks. Additionally, AffineQuant demonstrates faster loss convergence and superior overall optimization performance in the last block compared to OmniQuant. These results reaffirm the significant potential of invertible matrix optimization. Figure 5 and 6 in Appendix A.4 presents a random sampling of multiple stability factors (*alpha*), which impact on quantization loss convergence. The data reveals a notable link between the last transformer block's quantization loss and the quantized model's performance. This implies AffineQuant's effectiveness in reducing quantization loss in Figure 3, thereby enhancing the model's quantization performance during optimization.

#### 3.2 REVERSIBILITY AND GRADUAL MASK

In the optimization process, it is necessary to invert the affine transformation matrix. However, we do not include any constraints in the objective function (Equation 4) to ensure the matrix remains full rank or well-conditioned. Therefore, how to keep the matrix invertible during the optimization process? To begin, let's define a strictly diagonally dominant matrix as follows:

**Definition 1** (Strictly Diagonally Dominant Matrix) A matrix A is considered strictly diagonally dominant if the absolute value of each diagonal element is greater than the sum of the absolute values of the remaining elements in the corresponding row. Specifically,

$$|a_{ii}| > \sum_{i \neq j} |a_{ij}|, \quad for \ all \ i.$$
 (5)

The Levy-Desplanques theorem (Naimark & Zeheb, 1997) establishes that all strictly diagonally dominant matrices are invertible. By initializing the affine transformation matrix with diagonal elements, we ensure it initially is strictly diagonally dominant. Although utilizing second-order momentum and lower learning rates in the optimizer can assist in satisfying the requirements of the Levy-Desplanques theorem, the optimization of large affine transform matrices still faces instability challenges as the model size increases.

To ensure that the affine transformation matrix remains strictly diagonally dominant during optimization, we introduce a gradual mask approach, as illustrated in Figure 2. At the start of each optimization block, we freeze all elements except for those on the main diagonal. As the optimization progresses, we gradually unfreeze the elements near the main diagonal. Eventually, all matrix elements become learnable for optimization. This freezing mechanism, referred to as the Gradual Mask (GM), is defined as follows:

<span id="page-4-1"></span>
$$GM_{ij} = \begin{cases} 1 & i = j, \\ \alpha & 0 < |i - j| \le \frac{e}{t} \times hidden \ size, \\ 0 & otherwise, \end{cases}$$
 (6)

![](_page_5_Figure_1.jpeg)

<span id="page-5-0"></span>Figure 3: Mean square error loss of the last transformer block of LLaMA-7b and OPT-1.3b. "w2a16" means 2-bit weight-only quantization. "w3a16g128" means 3-bit grouping 128 weight-only quantization. We optimize 40 and 20 epochs in the last block of LLaMA-7b and OPT-1.3b, respectively.

Where GMij is the i-th row, j-th column element of the mask matrix. t is the target epochs. e ∈ [1, t] is the current epochs. "hidden size" is the dimension of the affine transformation matrix. α is the stability factor. Within the attention module, we apply a gradual mask in each attention head. GM is a learning rate regulator that achieves its purpose by element-wise dot-producting with the matrix A. Specifically, the impact of the GM matrix on the optimization process can be divided into two aspects. Here, we present the optimization process for the matrix A after incorporating the GM.

Forward: 
$$A_e^* = A_e \circ GM_e,$$
 (7)

**Backward:** 
$$A_{e+1} = A_e + \eta \frac{\partial L}{\partial A_e^*} \frac{\partial A_e^*}{\partial A_e},$$
 (8)

<span id="page-5-1"></span>
$$= A_e + \eta G M_e \frac{\partial L}{\partial A_e^*}.$$
 (9)

Where ◦ is the Hadamard product. A<sup>e</sup> and GM<sup>e</sup> are the matrices A and Gradual Mask (GM) matrix in epoch e, respectively. η is the learning rate of matrix A. L is the optimization loss. The GM matrix effectively reduces the magnitude of non-principal diagonal elements in matrix A during forward propagation when the stability factor α is less than 1. This ensures the existence of a stable inverse matrix of A<sup>∗</sup> in the optimization process during epoch e, as per the Levy-Desplanques theorem. In backward propagation, GM affects the learning rate η, thereby suppressing the update rate of nonprimary diagonal elements in matrix A. Consequently, the impact of GM on η ensures that matrix A in epoch e + 1 maintains strictly diagonally dominant, satisfying the Levy-Desplanques theorem. Notably, as α approaches 0, the optimization process converges stably and becomes equivalent to OmniQuant [\(Shao et al., 2023\)](#page-10-4). Additionally, Appendix [A.2](#page-12-0) includes a theorem demonstrating that a sufficiently small stability factor α ensures the strictly diagonal dominance of matrix A during optimization.

The concept of gradual adaptation is also present in the post-training quantization of vision models. In Adaround [\(Nagel et al., 2020\)](#page-10-8), the gradient update of parameters is controlled using gradual powers of β in the soft quantization function. When β is sufficiently large, only values close to 0 or 1 are updated due to gradient limitations. As optimization progresses, the gradient of all rounded values is gradually released. However, it is important to note that the goals and approaches of the two methods are distinct. Adaround [\(Nagel et al., 2020\)](#page-10-8) employs the gradual power β to prevent fast convergence of the objective function, which can lead to suboptimal optimization. On the other hand, the gradual mask in AffineQuant ensures the strictly diagonally dominant property of the affine transformation matrix. Appendix [A.6](#page-13-2) showcases heat maps depicting different block affine transformation matrices at various epochs, demonstrating the effectiveness of the gradual mask approach in maintaining strictly diagonally dominant matrices.

<span id="page-6-1"></span>

| Table 1: Weight-only quantization PPL(↓) results on the OPT model WikiText2 dataset. |  |  |
|--------------------------------------------------------------------------------------|--|--|
|                                                                                      |  |  |

| Config    | Method                                                | 125M           | 1.3B           | 2.7B            | 6.7B           | 13B            | 30B            |
|-----------|-------------------------------------------------------|----------------|----------------|-----------------|----------------|----------------|----------------|
| FP16      | -                                                     | 27.65          | 14.63          | 12.47           | 10.86          | 10.12          | 9.56           |
|           | RTN                                                   | 1.2e3          | 1.3e4          | 1.6e4           | 6.5e3          | 4.6e3          | 1.5e3          |
|           | GPTQ (Frantar et al., 2022)<br>AWQ (Lin et al., 2023) | 53.05<br>69.43 | 21.17<br>28.01 | 16.83<br>263.10 | 15.09<br>15.13 | 11.73<br>20.09 | 10.30<br>35.74 |
| w3a16     | OmniQuant (Shao et al., 2023)                         | 35.66          | 16.68          | 13.80           | 11.65          | 10.87          | 10.00          |
|           | AffineQuant                                           | 30.56          | 15.94          | 13.15           | 11.44          | 10.76          | 9.98           |
|           | RTN                                                   | 51.22          | 119.00         | 297.98          | 23.54          | 46.03          | 18.80          |
|           | GPTQ (Frantar et al., 2022)                           | 39.24          | 16.47          | 13.69           | 11.65          | 10.35          | 9.73           |
| w3a16g128 | AWQ (Lin et al., 2023)                                | 36.74          | 16.32          | 13.58           | 11.41          | 10.68          | 9.85           |
|           | OmniQuant (Shao et al., 2023)                         | 32.25          | 15.72          | 13.18           | 11.27          | 10.47          | 9.79           |
|           | AffineQuant                                           | 30.21          | 15.61          | 12.98           | 11.18          | 10.51          | 9.81           |
|           | RTN                                                   | 37.28          | 48.17          | 16.92           | 12.10          | 11.32          | 10.97          |
|           | GPTQ (Frantar et al., 2022)                           | 31.43          | 15.56          | 12.82           | 11.41          | 10.31          | 9.63           |
| w4a16     | AWQ (Lin et al., 2023)                                | 32.28          | 15.49          | 12.93           | 11.30          | 10.39          | 9.77           |
|           | OmniQuant (Shao et al., 2023)                         | 29.45          | 15.04          | 12.76           | 11.03          | 10.30          | 9.65           |
|           | AffineQuant                                           | 28.39          | 14.92          | 12.64           | 10.96          | 10.26          | 9.65           |
|           | RTN                                                   | 30.47          | 15.29          | 13.02           | 11.15          | 10.30          | 9.94           |
|           | GPTQ (Frantar et al., 2022)                           | 29.81          | 14.89          | 12.52           | 10.93          | 10.17          | 9.58           |
| w4a16g128 | AWQ (Lin et al., 2023)                                | 29.15          | 14.94          | 12.74           | 10.93          | 10.21          | 9.59           |
|           | OmniQuant (Shao et al., 2023)                         | 28.86          | 14.88          | 12.65           | 10.96          | 10.20          | 9.62           |
|           | AffineQuant                                           | 28.33          | 14.79          | 12.58           | 10.92          | 10.19          | 9.62           |

<span id="page-6-0"></span>Table 2: AffineQuant and OmniQuant quantization performance of LLaMA-7B, 13B, 30B on six zero-shot datasets using 4/4 bit quantization.

|                   |                                    |       |       | PIQA ARC-e WinoGrande BoolQ ARC-c HellaSwag |       |       |       | Avg.  |
|-------------------|------------------------------------|-------|-------|---------------------------------------------|-------|-------|-------|-------|
|                   | Dataset                            | (↑)   | (↑)   | (↑)                                         | (↑)   | (↑)   | (↑)   | (↑)   |
|                   | FP16                               | 77.47 | 52.48 | 67.07                                       | 73.08 | 41.46 | 73.00 | 64.09 |
| LLaMA-7B<br>w4a4  | OmniQuant Shao et al. (2023) 66.15 |       | 45.20 | 53.43                                       | 63.51 | 31.14 | 56.44 | 52.65 |
|                   | AffineQuant                        | 69.37 | 42.55 | 55.33                                       | 63.73 | 31.91 | 57.65 | 53.42 |
|                   | FP16                               | 79.10 | 59.89 | 70.31                                       | 68.01 | 44.45 | 76.21 | 66.33 |
| LLaMA-13B<br>w4a4 | OmniQuant Shao et al. (2023) 69.69 |       | 47.39 | 55.80                                       | 62.84 | 33.10 | 58.96 | 54.37 |
|                   | AffineQuant                        | 66.32 | 43.90 | 54.70                                       | 64.10 | 29.61 | 56.88 | 52.58 |
|                   | FP16                               | 80.08 | 58.92 | 72.53                                       | 68.44 | 45.47 | 79.21 | 67.44 |
| LLaMA-30B<br>w4a4 | OmniQuant Shao et al. (2023) 71.21 |       | 49.45 | 59.19                                       | 65.33 | 34.47 | 64.65 | 56.63 |
|                   | AffineQuant                        | 70.84 | 49.41 | 58.64                                       | 70.12 | 37.12 | 65.53 | 58.61 |

#### 3.3 EFFICIENCY

Optimize Efficiency. PyTorch's linear algebra library [\(Paszke et al., 2019\)](#page-10-12) offers matrix inverse computations in both float and double precision. Consequently, we maintain the model's precision as either float or double throughout the optimization process. Furthermore, approximate computations of the matrix inverse may contain errors due to the numerical precision limitations of the computer. Therefore, we analyze memory consumption, optimization time, error magnitude, and the impact on model performance for both precision types in Section [4.3.](#page-7-0)

Inference Efficiency. In line with similar algorithms, we integrate the affine transformation matrix with other layers. Subsequently, we perform half-precision inference on the network. For all linear layers, we merge the affine transformation matrix with the weight and bias parameters. In addition, we only optimize the diagonal elements of the affine matrix after LayerNorm for weight-activation quantization. This allows us to merge the affine matrix with the LayerNorm weights and bias. Consequently, AffineQuant can be achieved without introducing any additional overhead to model inference. Tables [2](#page-6-0) and [3](#page-7-1) demonstrate AffineQuant's superior performance over other methods in zero-shot and PPL tasks, even without additional overhead, using 4/4-bit quantization.

| Table 3: Quantization performance of LLaMA1&2 on WikiText2 and C4 datasets using 4/4 bit |  |  |
|------------------------------------------------------------------------------------------|--|--|
| weight-activation quantization.                                                          |  |  |

<span id="page-7-1"></span>

| Datasets  | LLaMA1&2 Methods |                                      | 1-7B  | 1-13B | 1-30B        | 2-7B  | 2-13B |
|-----------|------------------|--------------------------------------|-------|-------|--------------|-------|-------|
|           | FP16             | -                                    | 5.68  | 5.09  | 4.10         | 5.47  | 4.88  |
|           |                  | SmoothQuant Xiao et al. (2023) 25.25 |       | 40.05 | 192.40 83.12 |       | 35.88 |
| WikiText2 | W4A4             | OmniQuant Shao et al. (2023)         | 11.26 | 10.87 | 10.33        | 14.26 | 12.30 |
|           |                  | AffineQuant                          | 10.28 | 10.32 | 9.35         | 12.69 | 11.45 |
|           | FP16             | -                                    | 7.08  | 6.61  | 5.98         | 6.97  | 6.46  |
|           |                  | SmoothQuant Xiao et al. (2023) 32.32 |       | 47.18 | 122.38 77.27 |       | 43.19 |
| C4        | W4A4             | OmniQuant Shao et al. (2023)         | 14.51 | 13.78 | 12.49        | 18.02 | 14.55 |
|           |                  | AffineQuant                          | 13.64 | 13.44 | 11.58        | 15.76 | 13.97 |

<span id="page-7-2"></span>Table 4: PPL, memory usage, optimization runtime, and merge error for the OPT model under three precision schemes. The "double" scheme maintains double precision for both the model and the transform matrix. The "float" scheme indicates that both the model and the transform matrix are in float precision. The "float-double" scheme denotes that the model is in float precision while the transform matrix is in double precision.

|                    |                |                      |                |                                                 | OPT-6.7B w4a16         |                            |  |  |
|--------------------|----------------|----------------------|----------------|-------------------------------------------------|------------------------|----------------------------|--|--|
|                    | PPL            |                      |                | PPL                                             |                        |                            |  |  |
| -                  | 24.60          | -                    | -              | 11.74                                           | -                      | -                          |  |  |
| 1.88e−16           | 42.43          | 7065.5Mb             | 1.19h          | 11.91                                           | 41414.3Mb              | 16.7h                      |  |  |
| 2.58e−3<br>3.48e−4 | 42.91<br>42.88 | 3586.6Mb<br>3663.6Mb | 0.78h<br>0.85h | 11.90<br>11.96                                  | 21188.9Mb<br>23189.5Mb | 8.65h<br>12.72h            |  |  |
|                    | Merge Error    |                      |                | OPT-125M w2a16g64<br>Memory Utilization Runtime |                        | Memory Utilization Runtime |  |  |

