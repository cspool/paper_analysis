# 3 METHOD

**Overview.** Fig. 2 illustrates the overall workflow of PT<sup>2</sup>-LLM. We first review the standard symmetric ternarization formulation and basic notations in Section 3.1. Building on this foundation, Section 3.2 introduces the Asymmetric Ternary Quantizer, which features two training-free stages: Iterative Ternary Fitting (ITF) and Activation-aware Grid Alignment (AGA). Section 3.3 then presents the Structural Similarity-based Reordering (SSR), demonstrating how column clustering by structural similarity can be effectively combined within the GPTQ framework.

## <span id="page-2-1"></span>3.1 PRELIMINARY

**Symmetric Ternarization.** Symmetric ternarization compresses full-precision weights into the ternary set  $\{-1,0,+1\}$  by minimizing the discrepancy between the original weight matrix and its

ternary approximation under an appropriate scaling:

$$\alpha^*, \mathbf{T}^* = \arg\min_{\alpha, \mathbf{T}} \|\mathbf{W} - \alpha \mathbf{T}\|_F^2,$$
 (1)

where  $\mathbf{W} \in \mathbb{R}^{n \times m}$  is the full-precision weight matrix,  $\alpha \in \mathbb{R}^{n \times 1}$  is a row-wise scaling factor, and  $\mathbf{T} \in \{-1,0,+1\}^{n \times m}$  is the ternary matrix. Since jointly optimizing  $\alpha$  and  $\mathbf{T}$  causes parameter coupling, TWN (Li et al., 2016) proposes a threshold-based solution. Specifically, for each element  $\mathbf{W}_{ij}$ , a row-wise threshold  $\Delta \in \mathbb{R}^{n \times 1}$  is used to determine the corresponding ternary value  $\mathbf{T}_{ij}$  as:

<span id="page-3-1"></span>
$$\mathbf{T}_{ij} = \begin{cases} 1, & \text{if } \mathbf{W}_{ij} > \Delta_i, \\ 0, & \text{if } |\mathbf{W}_{ij}| \le \Delta_i, \\ -1, & \text{if } \mathbf{W}_{ij} < -\Delta_i. \end{cases}$$
 (2)

Given a fixed threshold  $\Delta$ , the ternary matrix  $\mathbf{T}$  is deterministically defined, enabling a closed-form solution for the optimal scaling factor  $\alpha$ . Since directly optimizing  $\Delta$  is difficult in practice, TWN approximates  $\Delta$  based on assumed weight distributions. Assuming uniform or normal priors,  $\Delta$  is approximated by a scaled mean of absolute weights, and the optimal  $\alpha$  follows:

$$\Delta \approx \frac{0.75}{m} \sum_{j=1}^{m} |\mathbf{W}_{:,j}|, \quad \alpha = \frac{\sum_{j=1}^{m} \mathbf{T}_{:,j} \cdot \mathbf{W}_{:,j}}{\sum_{j=1}^{m} |\mathbf{T}_{:,j}|}.$$
 (3)

This approximation enables fast and training-free ternarization by decoupling  $\alpha$  and  $\mathbf{T}$ , providing a practical solution to ternary parameters initialization in PTQ settings.

#### <span id="page-3-0"></span>3.2 ASYMMETRIC TERNARY QUANTIZER

Asymmetric Ternary Initialization. Empirical observations reveal that the weight distributions in LLMs are not always symmetric, as many layers exhibit non-zero means. We provide visualizations in the supplementary file to further support this observation. While symmetric ternarization (as discussed in Section 3.1) performs well under QAT due to its ability to reshape the weight distribution through backpropagation, this assumption no longer holds in PTQ, where pre-trained weights remain fixed. To better capture the bias in pre-trained weights, we follow prior work (Chen et al., 2024) and adopt an asymmetric ternarization scheme by introducing a row-wise offset  $\mu \in \mathbb{R}^{n \times 1}$ , initialized as the mean of each row. The dequantized weight  $\widehat{\mathbf{W}}$  is then computed as:

$$\widehat{\mathbf{W}} = \alpha \mathbf{T} + \mu, \quad \mu = \frac{1}{m} \sum_{j=1}^{m} \mathbf{W}_{:,j}. \tag{4}$$

For the initialization of  $\alpha$  and the ternary matrix  $\mathbf{T}$ , we follow the same strategy described in Section 3.1, applying it to the centered weight matrix  $\widetilde{\mathbf{W}} = \mathbf{W} - \mu$  to remove bias:

$$\Delta \approx \frac{0.75}{m} \sum_{i=1}^{m} |\widetilde{\mathbf{W}}_{:,j}|, \quad \alpha = \frac{\sum_{j=1}^{m} \mathbf{T}_{:,j} \cdot \widetilde{\mathbf{W}}_{:,j}}{\sum_{j=1}^{m} |\mathbf{T}_{:,j}|}.$$
 (5)

**T** is still initialized using Eq. 2, with  $\Delta$  applied to  $\widetilde{\mathbf{W}}$ . This asymmetric initialization offers a stable and expressive foundation for post-training ternarization under non-zero-mean weight distributions.

Iterative Ternary Fitting. After initialization, we obtain the three key components of ternarization: the scaling factor  $\alpha$ , the shift parameter  $\mu$ , and the ternary matrix  $\mathbf{T}$ .  $\alpha$  and  $\mu$  together define a ternary grid with only three possible quantized values for each row i, namely  $\{-\alpha_i + \mu_i, \ \mu_i, \ \alpha_i + \mu_i\}$ . How to refine this ternary grid so that it better fits the underlying weight distribution is crucial for improving quantization quality. We first define the quantization error of weights  $\mathcal{E}_w$  as:

<span id="page-3-2"></span>
$$\mathcal{E}_w = \|\mathbf{W} - \widehat{\mathbf{W}}\|_F^2, \quad \text{where } \widehat{\mathbf{W}} = \alpha \mathbf{T} + \mu.$$
 (6)

Our current optimization objective is to minimize the quantization error  $\mathcal{E}_w$ , which is achieved by optimizing the ternarization parameters  $\alpha$ ,  $\mu$ , and  $\mathbf{T}$ . Since  $\alpha$  and  $\mu$  together determine the discrete grid values of ternarization, we refer to them as the ternary grid parameters. A well-constructed grid is essential to provide a reliable basis for subsequent optimization of the ternary matrix  $\mathbf{T}$ . Therefore, we first focus on establishing a high-quality ternary grid. By differentiating the quantization error  $\mathcal{E}_w$  with respect to  $\alpha_i$  and  $\mu_i$ , we obtain the following gradients:

$$\frac{\partial \mathcal{E}_w}{\partial \alpha_i} = 2 \left( \alpha_i \mathbf{t}_i + \mu_i \mathbf{1}^\top - \mathbf{w}_i \right) \mathbf{t}_i^\top, \quad \frac{\partial \mathcal{E}_w}{\partial \mu_i} = 2 \left( \alpha_i \mathbf{t}_i + \mu_i \mathbf{1}^\top - \mathbf{w}_i \right) \mathbf{1}, \tag{7}$$

where  $\mathbf{t}_i \in \mathbb{R}^{1 \times m}$  denotes the *i*-th row of  $\mathbf{T}$ , and  $\mathbf{w}_i \in \mathbb{R}^{1 \times m}$  is the corresponding row of  $\mathbf{W}$ .  $\mathbf{1} \in \mathbb{R}^{m \times 1}$  is a column vector of all ones. The parameters  $\alpha_i$  and  $\mu_i$  serve as the scaling factor and

shift associated with the i-th row, respectively. To obtain the optimal ternary grid parameters, we set the partial derivatives to zero and solve for  $\alpha_i$  and  $\mu_i$ . This leads to a system of linear equations for the optimal grid parameters  $\alpha_i$  and  $\mu_i$  (see supplementary file for detailed derivation):

$$\frac{\partial \mathcal{E}_w}{\partial \alpha_i} = 0, \quad \frac{\partial \mathcal{E}_w}{\partial \mu_i} = 0 \implies \begin{bmatrix} \mathbf{t}_i \mathbf{t}_i^\top & \mathbf{1}^\top \mathbf{t}_i^\top \\ \mathbf{t}_i \mathbf{1} & \mathbf{1}^\top \mathbf{1} \end{bmatrix} \begin{bmatrix} \alpha_i \\ \mu_i \end{bmatrix} = \begin{bmatrix} \mathbf{w}_i \mathbf{t}_i^\top \\ \mathbf{w}_i \mathbf{1} \end{bmatrix}, \tag{8}$$

which can be efficiently solved to obtain the optimal ternary grid for the i-th row. To enable efficient batched computation across rows, we further reformulate the optimal solutions of  $\alpha^*$  and  $\mu^*$  into a more compact vectorized form (see supplementary file for detailed derivation):

<span id="page-4-0"></span>
$$\alpha^* = \frac{m \cdot (\mathbf{W} \circ \mathbf{T})\mathbf{1} - (\mathbf{T}\mathbf{1}) \circ (\mathbf{W}\mathbf{1})}{m \cdot (\mathbf{T} \circ \mathbf{T})\mathbf{1} - (\mathbf{T}\mathbf{1})^2}, \quad \mu^* = \frac{(\mathbf{T} \circ \mathbf{T})\mathbf{1} \circ (\mathbf{W}\mathbf{1}) - (\mathbf{T}\mathbf{1}) \circ [(\mathbf{W} \circ \mathbf{T})\mathbf{1}]}{m \cdot (\mathbf{T} \circ \mathbf{T})\mathbf{1} - (\mathbf{T}\mathbf{1})^2}, \quad (9)$$

where  $\circ$  denotes element-wise multiplication, with all divisions also element-wise. m is the number of elements per row. This vectorized form enables parallel closed-form solutions across rows, ensuring optimal  $\alpha^*$  and  $\mu^*$  under fixed T and thus the best ternary grid at the current stage. After obtaining the current optimal ternary grid, we update T by mapping the full-precision weights onto it. Instead of a fixed threshold, which is rigid and often suboptimal for diverse weight distributions, we adopt a more flexible element-wise ternary rounding to minimize the quantization error  $\mathcal{E}_w$  (Eq. 6). Given  $\alpha^*$ and  $\mu^*$ , the optimal value of each entry  $\mathbf{T}_{ij}^*$  is determined by the following rule:

<span id="page-4-1"></span>
$$\mathbf{T}_{ij}^* = \arg\min_{t \in \{-1,0,1\}} |\mathbf{Z}_{ij} - t|, \quad \text{where } \mathbf{Z}_{ij} = \frac{\mathbf{W}_{ij} - \mu_i^*}{\alpha_i^*}.$$
 (10)

This guarantees that, under fixed  $\alpha^*$  and  $\mu^*$ , Algorithm 1 Pseudocode of the Asymmetric the updated  $T^*$  yields the minimal quantization error  $\mathcal{E}_w$ , making it the optimal ternary assignment for the current grid. We observe that obtaining the optimal ternary grid and the optimal ternary matrix naturally forms an iterative optimization scheme. By alternating between Eq. 9 and Eq. 10, the algorithm greedily reduces the quantization error  $\mathcal{E}_w$  at each step. Convergence is reached when the update in Eq. 10 no longer changes the ternary matrix T, indicating that the ternarized structure has stabilized. In practice, it converges within about 10 iterations.

Activation-aware Grid Alignment. While Iterative Ternary Fitting effectively minimizes the weight quantization error  $\mathcal{E}_w$ , the actual output of LLMs depends on the interaction between <span id="page-4-2"></span>Ternary Quantizer. See supp. file for details.

func ATQ(W, X) **Input:**  $\mathbf{W} \in \mathbb{R}^{n \times m}$  - weight matrix  $\mathbf{X} \in \mathbb{R}^{B \times L \times m}$  - calibration data Output:  $\widehat{\mathbf{W}} \in \mathbb{R}^{n \times m}$ 

1:  $\alpha, \mu, \mathbf{T} := \text{Ternary\_Init}(\mathbf{W})$ 2:  $\mathbf{T}_{prev} \coloneqq \mathbf{0}$ 3: while  $\mathbf{T} \neq \mathbf{T}_{prev}$  do 4:  $\mathbf{T}_{prev} \leftarrow \mathbf{T}$  $\alpha, \mu \leftarrow \text{Biuld\_Optimal\_Grid}(\mathbf{T}, \mathbf{W})$ 5:  $\mathbf{T} \leftarrow \text{Flexible\_Round}(\mathbf{W}, \alpha, \mu)$ 6: 7: end while

8:  $\alpha, \mu \leftarrow AGA(\mathbf{W}, \mathbf{T}, \mathbf{X})$ 9:  $\widehat{\mathbf{W}} \leftarrow \alpha \mathbf{T} + \mu$ 10: return  $\widehat{\mathbf{W}}$ 

weights and activations. To address this issue, we introduce the activation-aware output error  $\mathcal{E}_x$ :

$$\mathcal{E}_x = \|\mathbf{W}\mathbf{X} - \widehat{\mathbf{W}}\mathbf{X}\|_F^2, \quad \text{where } \widehat{\mathbf{W}} = \alpha \mathbf{T} + \mu.$$
 (11)

 $\mathcal{E}_x = \|\mathbf{W}\mathbf{X} - \widehat{\mathbf{W}}\mathbf{X}\|_F^2, \quad \text{where } \widehat{\mathbf{W}} = \alpha \mathbf{T} + \mu. \tag{11}$  Here,  $\mathbf{X} \in \mathbb{R}^{B \times L \times m}$  denotes the calibration data with batch size B, sequence length L, and embedding dimension m. This formulation directly couples quantization with the model outputs, ensuring that the optimization better reflects the real scenario. In line with the Iterative Ternary Fitting, we again differentiate  $\mathcal{E}_x$  with respect to  $\alpha_i$  and  $\mu_i$ , set the derivatives to zero, and obtain a system of equations that gives the optimal ternary grid under the current objective. As before, the solution is expressed row-wise for the i-th component

$$\frac{\partial \mathcal{E}_x}{\partial \alpha_i} = 0, \quad \frac{\partial \mathcal{E}_x}{\partial \mu_i} = 0 \implies \begin{bmatrix} \mathbf{t}_i \mathbf{C} \mathbf{t}_i^\top & \mathbf{1}^\top \mathbf{C} \mathbf{t}_i^\top \\ \mathbf{t}_i \mathbf{C} \mathbf{1} & \mathbf{1}^\top \mathbf{C} \mathbf{1} \end{bmatrix} \begin{bmatrix} \alpha_i \\ \mu_i \end{bmatrix} = \begin{bmatrix} \mathbf{w}_i \mathbf{C} \mathbf{t}_i \\ \mathbf{w}_i \mathbf{C} \mathbf{1} \end{bmatrix}, \tag{12}$$

where  $C = \sum_{b} \sum_{i} X_{bi} X_{bi}^{\dagger}$ . By solving this system, we obtain the closed-form solutions:

$$\alpha^* = \frac{d \cdot (\mathbf{W} \circ \mathbf{T})\mathbf{S}\mathbf{1} - \mathbf{v} \circ (\mathbf{W}\mathbf{S}\mathbf{1})}{d \cdot \mathbf{T}^2\mathbf{S}\mathbf{1} - \mathbf{v}^2}, \quad \mu^* = \frac{\mathbf{T}^2\mathbf{S}\mathbf{1} \circ (\mathbf{W}\mathbf{S}\mathbf{1}) - \mathbf{v} \circ [(\mathbf{W} \circ \mathbf{T})\mathbf{S}\mathbf{1}]}{d \cdot \mathbf{T}^2\mathbf{S}\mathbf{1} - \mathbf{v}^2}, \quad (13)$$

where  $d = \mathbf{1}^{\mathsf{T}} \mathbf{S} \mathbf{1}$  is a scalar, and  $\mathbf{v} = \mathbf{T} \mathbf{S} \mathbf{1}$ ;  $\mathbf{T}^2$  and  $\mathbf{v}^2$  denote element-wise squares. This activationaware alignment of grid parameters significantly improves the consistency between quantized and full-precision outputs. Ideally, we would update **T** to further reduce the output error  $\mathcal{E}_x$ , but unlike ITF, no optimal solution exists. Greedy search is possible, yet in practice we observe that updating T leads to severe overfitting on the calibration set. Therefore, we freeze **T** and update only  $(\alpha, \mu)$  once, which already yields accurate approximations. See supplementary file for details on overfitting.

<span id="page-5-1"></span>![](_page_5_Figure_1.jpeg)

Figure 3: Visualization of the proposed Asymmetric Ternary Quantizer (ATQ) and Structural Similarity-based Reordering (SSR) effects. Left: Quantization error E<sup>w</sup> across optimization steps during ATQ. Middle: Output error E<sup>x</sup> across optimization steps during ATQ. Right: After column reordering, the block-wise variance becomes smaller, showing a more compact weight distribution.

Overall ATQ Workflow. As shown in Algorithm [1,](#page-4-2) ATQ refines the ternary parameters via ITF for a more accurate T, and then applies AGA to further align the ternary grid parameters with the output, yielding the final quantized weights. In Fig. [3](#page-5-1) (left and middle), we plot quantization error E<sup>w</sup> and output error E<sup>x</sup> across steps. E<sup>w</sup> steadily decreases during ITF and slightly rises after the AGA update as the optimization objective shifts, while E<sup>x</sup> drops modestly during ITF and sharply after AGA. Overall, the results show ATQ reduces both quantization and output errors without retraining.

## <span id="page-5-0"></span>3.3 STRUCTURAL SIMILARITY-BASED REORDERING

Motivation. Following GPTQ [\(Frantar et al.,](#page-9-2) [2023\)](#page-9-2), our ternarization is blockwise: large weight matrices are split into fixed-size blocks and quantized independently. While this improves accuracy over quantizing the entire matrix at once, we still find that na¨ıve blockwise ternarization causes severe performance degradation. To investigate, we analyze the weight distributions and identify two key issues. (i) weights within a block often exhibit high variance, making ternarization too coarse and leading to large quantization error; (ii) many layers exhibit column-wise bias, where outlier columns distort the ternarization range and degrade fidelity.

Structure-Aware Column Clustering. To address these issues, we revisit *column reordering*, which in GPTQ [\(Frantar et al.,](#page-9-2) [2023\)](#page-9-2) is offered as an optional technique: while GPTQ can quantize weights in a fixed order, reordering columns by Hessian-derived importance has been shown to improve performance. Formally, reordering can be expressed using a permutation matrix P:

$$\mathbf{W}' = \mathbf{W}\mathbf{P}, \quad \mathbf{X}' = \mathbf{X}\mathbf{P}, \quad \mathbf{X}'\mathbf{W}'^{\top} = \mathbf{X}\mathbf{W}^{\top}. \tag{14}$$

Here P simply permutes columns, so the result of matrix multiplication remains unchanged. Since applying P is just index reordering rather than actual multiplications, the computational overhead during inference is negligible. Building on this formulation, we observe that the potential of reordering remains underexplored. Placing structurally similar and numerically close columns in the same block yields a more compact distribution, improving row-wise ternarization. Similarly, grouping outliers together prevents them from distorting normal columns—outliers among outliers cease to be outliers. To this end, we propose a simple yet effective structure-aware column clustering method. Specifically, we compute pairwise cosine similarities between weight columns to capture their structural similarity:

$$S_{ij} = \frac{\mathbf{W}_{:,i}^{\top} \mathbf{W}_{:,j}}{\|\mathbf{W}_{:,i}\|_{2} \|\mathbf{W}_{:,j}\|_{2}},$$
(15)

where W:,i denotes the i-th column of W. Based on the similarity matrix S, we cluster columns with aligned directions to form more homogeneous blocks for ternarization. As shown in Fig. [3](#page-5-1) (right), reordering reduces block variances, indicating more compact weight distributions within blocks.

Efficient Integration with GPTQ. GPTQ quantizes weights block by block, applying error compensation after each step. This inter-block dependency makes a one-time clustering-based reordering ineffective, while re-clustering after every update is too costly. To balance accuracy and efficiency, we adopt a lightweight strategy: after each update, we compute a mean reference from the residual and select the top-k similar columns, where k is the quantization block size:

$$\mathcal{B} = \text{Top-}k \left( \left\{ \frac{\mathbf{W}_{:,i}^{\top} \bar{\mathbf{w}}}{\|\mathbf{W}_{:,i}\|_2 \|\bar{\mathbf{w}}\|_2} \right\}_{i=k}^m \right), \quad \text{with } \bar{\mathbf{w}} = \frac{1}{m} \sum_{i=k}^m \mathbf{W}_{:,i}, \tag{16}$$

Here, w¯ is the mean vector of the remaining submatrix, and B contains the top-k most similar columns, forming the next quantization block. We term this lightweight strategy Structural Similarity-based Reordering (SSR), which retains the benefits of reordering with minimal overhead.

<span id="page-6-0"></span>Table 1: **Evaluation on Multiple LLM Backbones.** We report perplexity (PPL) on WikiText2 and C4, and accuracy (%) on seven zero-shot tasks. All quantized models use a block size of 128. Best and second-best results (excluding FP16) are marked in **bold** and underlined, respectively.

| Model    | Method               | #W   | Wiki2(↓)    | <b>C4</b> (↓) | PiQA       | Arc E                 | Arc C          | Hella.       | Wino.        | OBQA                  | BoolQ          | Avg(↑)       |
|----------|----------------------|------|-------------|---------------|------------|-----------------------|----------------|--------------|--------------|-----------------------|----------------|--------------|
| 11100001 | FP16                 | 16   | 5.68        | 7.34          | 78.67      | 75.34                 | 41.89          | 56.93        | 70.01        | 34.20                 | 75.05          | 61.73        |
|          | AWQ                  | 2    | 2.60e5      | 2.86e5        | 52.83      | 25.25                 | 22.44          | 25.27        | 49.88        | 14.00                 | 37.83          | 32.50        |
|          | GPTQ                 | 2    | 129.19      | 79.06         | 55.39      | 30.64                 | 19.62          | 27.36        | 48.30        | 14.40                 | 44.74          | 34.35        |
| L-7B     | OuIP                 | 2    | 29.74       | 33.74         | N/A        | N/A                   | N/A            | N/A          | N/A          | N/A                   | N/A            | N/A          |
|          | Slim-LLM             | 2    | 14.58       | 30.71         | 57.83      | 33.46                 | 25.09          | 36.70        | 52.64        | 16.40                 | 56.05          | 39.74        |
|          | PB-LLM               | 1.7  | 82.76       | 76.63         | 55.17      | 29.42                 | 18.94          | 27.68        | 47.83        | 12.20                 | 42.87          | 33.44        |
|          | PT <sup>2</sup> -LLM | 1.58 | 11.39       | 24.55         | 63.49      | 52.48                 | 24.32          | 34.04        | 59.12        | 18.40                 | 63.64          | 45.07        |
|          | FP16                 | 16   | 5.09        | 6.80          | 79.16      | 77.40                 | 46.42          | 59.91        | 72.69        | 33.20                 | 77.89          | 63.81        |
|          | AWQ                  | 2    | 2.76e5      | 2.30e5        | 53.37      | 26.22                 | 22.87          | 25.54        | 49.96        | 15.60                 | 62.17          | 36.53        |
| L-13B    | GPTQ                 | 2    | 20.46       | 18.97         | 61.97      | 41.67                 | 24.06          | 38.13        | 54.85        | 19.20                 | 47.13          | 41.00        |
|          | QuIP                 | 2    | 38.82       | 28.62         | N/A        | N/A                   | N/A            | N/A          | N/A          | N/A                   | N/A            | N/A          |
|          | Slim-LLM             | 2    | 9.12        | 19.59         | 66.59      | <u>55.22</u>          | <u>25.00</u>   | <u>38.25</u> | <u>62.83</u> | 24.40                 | 64.01          | <u>48.04</u> |
|          | PB-LLM               | 1.7  | 44.93       | 40.64         | 60.12      | 36.53                 | 19.28          | 30.78        | 50.67        | 14.60                 | 62.75          | 39.25        |
|          | PT <sup>2</sup> -LLM | 1.58 | 9.11        | 17.32         | 67.19      | 58.50                 | 26.19          | 39.17        | 63.54        | <u>22.20</u>          | <u>63.67</u>   | 48.64        |
|          | FP16                 | 16   | 3.53        | 5.81          | 81.28      | 81.36                 | 52.82          | 64.55        | 77.43        | 38.20                 | 84.83          | 68.64        |
|          | AWQ                  | 2    | 7.40e4      | 7.50e4        | 53.16      | 25.13                 | 22.27          | 25.48        | 49.80        | 11.20                 | 37.83          | 32.12        |
|          | GPTQ                 | 2    | 8.66        | 10.23         | 73.12      | 64.94                 | 35.07          | 47.95        | 63.93        | <u>26.60</u>          | 67.31          | 54.13        |
| L-65B    | QuIP                 | 2    | 7.83        | 13.99         | N/A        | N/A                   | N/A            | N/A          | N/A          | N/A                   | N/A            | N/A          |
|          | Slim-LLM             | 2    | 6.15        | 11.11         | 75.20      | 53.72                 | 35.10          | 45.91        | 70.21        | 25.80                 | 76.91          | 54.69        |
|          | PB-LLM               | 1.7  | 12.81       | 15.30         | 72.18      | 68.43                 | 35.18          | 46.35        | 70.16        | 26.40                 | 62.75          | 54.49        |
|          | PT <sup>2</sup> -LLM | 1.58 | 6.62        | 9.17          | 73.01      | 70.08                 | 35.58          | 46.52        | 70.48        | 28.00                 | 67.98          | 55.95        |
|          | FP16                 | 16   | 5.47        | 7.26          | 78.07      | 76.30                 | 43.34          | 57.16        | 68.98        | 31.40                 | 77.68          | 61.85        |
|          | AWQ                  | 2    | 2.22e5      | 1.70e5        | 52.39      | 25.00                 | 21.16          | 25.58        | 49.09        | 18.00                 | 37.83          | 32.72        |
|          | GPTQ                 | 2    | 52.22       | 35.27         | 58.05      | 33.16                 | 21.42          | 32.65        | 49.88        | 15.60                 | 55.47          | 38.03        |
| L2-7B    | QuIP                 | 2    | 39.73       | 31.94         | N/A        | N/A                   | N/A            | N/A          | N/A          | N/A                   | N/A            | N/A          |
|          | Slim-LLM             | 2    | 15.84       | 84.92         | 63.82      | 47.81                 | 23.38          | 33.76        | 56.91        | 17.80                 | <u>59.97</u>   | 43.35        |
|          | PB-LLM               | 1.7  | 66.41       | 80.69         | 53.59      | 27.82                 | 18.69          | 26.91        | 48.54        | 13.20                 | 41.25          | 32.86        |
|          | PT <sup>2</sup> -LLM | 1.58 | 11.56       | 24.38         | 62.95      | 47.01                 | 21.08          | 33.82        | 56.75        | 18.80                 | 62.91          | 43.33        |
|          | FP16                 | 16   | 3.32        | 5.71          | 82.15      | 82.79                 | 54.44          | 64.78        | 77.98        | 37.20                 | 83.76          | 69.01        |
|          | AWQ                  | 2    | 7.25e4      | 7.30e4        | 52.50      | 25.76                 | 22.35          | 25.33        | 49.49        | 14.20                 | 62.17          | 35.97        |
| L2-70B   | GPTQ                 | 2    | 8.18        | 19.55         | 72.52      | 62.67                 | 34.56          | 47.66        | 67.17        | 25.00                 | 66.76          | 53.76        |
| L2-70B   | QuIP<br>Slim-LLM     | 2 2  | N/A<br>6.28 | N/A<br>N/A    | N/A<br>N/A | N/A<br>N/A            | N/A<br>N/A     | N/A<br>N/A   | N/A<br>N/A   | N/A<br>N/A            | N/A<br>N/A     | N/A<br>N/A   |
|          | PB-LLM               | 1.7  | 28.37       | N/A<br>N/A    | N/A        | N/A<br>N/A            | N/A            | N/A<br>N/A   | N/A<br>N/A   | N/A<br>N/A            | N/A<br>N/A     | N/A<br>N/A   |
|          | PT <sup>2</sup> -LLM | 1.58 | 6.27        | 12.00         | 72.96      | 71.00                 | 37.71          | 46.17        | 71.35        | 25.60                 |                | 55.87        |
|          | FP16                 | 1.58 | 6.14        | 9.45          | 79.54      | 80.13                 | 50.34          | 60.13        | 73.40        | 34.60                 | 66.30<br>81.01 | 65.59        |
|          | AWQ                  | 2    | 1.70e5      | 2.10e5        | 52.72      | 24.16                 | 21.50          | 25.58        | 49.33        | 14.60                 | 62.17          | 35.72        |
|          | GPTQ                 |      | 1480.43     | 394.74        | 52.72      | 25.72                 | 21.50<br>21.59 | 26.72        | 49.33        | 13.60                 | 44.16          | 33.72        |
| L3-8B    | QuIP                 | 2 2  | 84.97       | 130.00        | N/A        | N/A                   | N/A            | N/A          | 49.17<br>N/A | N/A                   | N/A            | N/A          |
| L3-0D    | Slim-LLM             | 2    | 38.21       | 390.02        | 55.77      | 32.15                 | 19.11          | 27.83        | 48.78        | 13.20                 | 44.83          | 27.83        |
|          | PB-LLM               | 1.7  | 73.08       | 104.15        | 56.64      | 33.08                 | 17.15          | 27.98        | 51.07        | 12.40                 | 55.44          | 36.25        |
|          | PT <sup>2</sup> -LLM | 1.58 | 32.19       | 129.83        | 56.86      | 34.22                 | 18.43          | 30.36        | 53.28        | 13.80                 | 57.58          | 37.79        |
|          | FP16                 | 16   | 6.38        | 9.68          | 80.50      | 74.20                 | 44.11          | 54.27        | 74.59        | 35.00                 | 86.50          | 68.13        |
|          | AWQ                  | 2    | 2.68e7      | 2.18e7        | 53.00      | 24.60                 | 23.00          | 25.30        | 50.70        | 20.00                 | 46.20          | 34.69        |
|          | GPTQ                 | 2    | 37.90       | 74.50         | 56.31      | 34.64                 | 20.65          | 33.30        | 52.72        | 17.20                 | 46.33          | 37.31        |
| Qwen3    | OuIP                 | 2    | N/A         | N/A           | N/A        | N/A                   | N/A            | N/A          | N/A          | N/A                   | N/A            | N/A          |
| 14B-Base | Slim-LLM             | 2    | 22.85       | 68.38         | 61.83      | 52.54                 | 29.35          | 31.52        | 52.04        | 20.40                 | 61.20          | 44.13        |
|          | PB-LLM               | 2    | 2.89e4      | 2.44e4        | 54.08      | $\frac{52.51}{25.93}$ | 20.73          | 25.76        | 47.99        | $\frac{26.16}{15.00}$ | 38.04          | 32.50        |
|          | PT <sup>2</sup> -LLM | 1.58 | 16.48       | 68.13         | 62.95      | 53.03                 | 23.63          | 33.65        | 59.75        | 20.60                 | 62.17          | 45.11        |
|          | 1 - 1 - 1 - 1 - 1    | 1.00 | 10.10       | 00.20         | 52.50      | 20.00                 | 20.00          | 20.00        | 27           | -0.00                 | V              |              |

