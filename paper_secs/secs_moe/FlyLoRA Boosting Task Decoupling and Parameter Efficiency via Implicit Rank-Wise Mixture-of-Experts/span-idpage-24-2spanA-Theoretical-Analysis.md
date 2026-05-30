# <span id="page-24-2"></span>A Theoretical Analysis

#### <span id="page-24-0"></span>A.1 Distance Preserving Property for Sparse Random Projection

In this section, we prove that the fixed sparse random projection matrix A satisfies the distance-preserving property. This demonstrates that a fixed projection A can function like a hash router without an explicit router. Our result extends the well-known Johnson-Lindenstrauss Lemma [32]. First, [5, Corollary 3.5] provides the following bound:

<span id="page-24-3"></span>**Theorem A.1.** Let  $A \in \mathbb{R}^{r \times n}$  be a random matrix whose entries  $A_{ij}$  are sampled independently and randomly from a distribution that is symmetric around the origin with  $\mathbb{E}(A_{ij}^2) = \sigma^2 > 0$ .

1. Suppose  $C = \mathbb{E}(A_{ij}^4) < \infty$ . Then, for any  $\epsilon > 0$ ,

$$\mathbb{P}\left(\|\frac{1}{\sqrt{r}}\boldsymbol{A}\boldsymbol{x}\|^2 \le \sigma^2(1-\epsilon)\|\boldsymbol{x}\|^2\right) \le \exp\left(-\frac{(\epsilon^2 - \epsilon^3)r}{2(\frac{1}{\sigma^4}C + 1)}\right), \quad \text{for all } \boldsymbol{x} \in \mathbb{R}^n.$$

2. Suppose  $\exists L>0$  such that for any integer k>0,  $\mathbb{E}(\boldsymbol{A}_{ij}^{2k})\leq \sigma^{2k}\frac{(2k)!}{2^kk!}L^{2k}$ . Then, for any  $\epsilon>0$ ,

$$\mathbb{P}\left(\|\frac{1}{\sqrt{r}}\boldsymbol{A}\boldsymbol{x}\|^2 \ge \sigma^2(1+\epsilon)L^2\|\boldsymbol{x}\|^2\right) \le \exp\left(-(\epsilon^2 - \epsilon^3)\frac{r}{4}\right), \quad \textit{for all } \boldsymbol{x} \in \mathbb{R}^n.$$

According to the definition of A mentioned in Section 3.1, the second moment of  $A_{ij}$  satisfies  $\mathbb{E}(A_{ij}^2) = \frac{p}{nr^2} > 0$ . So, the 2k-th moment of  $A_{ij}$  is given by  $\mathbb{E}[A_{ij}^{2k}] = \frac{p}{n} \cdot \mathbb{E}[X_{ij}^{2k}] + \left(1 - \frac{p}{n}\right) \cdot 0 = \frac{p}{n} \cdot \mathbb{E}[X_{ij}^{2k}]$ , where  $X_{ij} \sim \mathcal{N}(0, \frac{1}{r^2})$ . From the property of Gaussian distribution, we derive  $\mathbb{E}[X_{ij}^{2k}] = (2k-1)!! \cdot \left(\frac{1}{r^2}\right)^{2k} = \frac{(2k)!}{2^k k!} \left(\frac{1}{r^2}\right)^{2k}$ , where !! denotes the double factorial. This leads to  $\mathbb{E}[A_{ij}^4] = \frac{3p}{nr^4}$ , which is clearly finite. To satisfy the inequality  $\mathbb{E}[A_{ij}^{2k}] = \frac{(2k)!}{2^k k!} \sigma^{2k} \leq \sigma^{2k} \frac{(2k)!}{2^k k!} L^{2k}$ , we require  $L \geq \log_{2k} \frac{p}{r}$ . Due to monotonicity, simply choosing  $L = \log_{2} \frac{p}{r}$  always satisfies the condition specified in Theorem A.1. Combining both bounds from Theorem A.1, we summarize the final result in Theorem A.2.

$$\mathbb{P}\left((1-\epsilon)\|\boldsymbol{x}-\boldsymbol{y}\|^{2} \leq \frac{1}{r\sigma^{2}}\|\boldsymbol{A}\boldsymbol{x}-\boldsymbol{A}\boldsymbol{y}\|^{2} \leq (1+\epsilon)\|\boldsymbol{x}-\boldsymbol{y}\|^{2}\right) = 1 - \mathbb{P}\left(\frac{1}{r}\|\boldsymbol{A}\boldsymbol{x}-\boldsymbol{A}\boldsymbol{y}\|^{2} < (1-\epsilon)\sigma^{2}\|\boldsymbol{x}-\boldsymbol{y}\|^{2}\right) -$$

$$\mathbb{P}\left(\frac{1}{r}\|\boldsymbol{A}\boldsymbol{x}-\boldsymbol{A}\boldsymbol{y}\|^{2} > (1+\epsilon)\sigma^{2}\|\boldsymbol{x}-\boldsymbol{y}\|^{2}\right), \tag{13}$$

<span id="page-24-4"></span>**Theorem A.2.** Given the matrix  $\mathbf{A} \in \mathbb{R}^{r \times n}$  with each entry i.i.d. from  $\mathcal{N}(0, \frac{1}{r^2})$ , and set to 0 otherwise, for any  $\epsilon > 0$ ,

$$\mathbb{P}\left((1-\epsilon)\|\boldsymbol{x}-\boldsymbol{y}\|^{2} \leq \frac{1}{r\sigma^{2}}\|\boldsymbol{A}\boldsymbol{x}-\boldsymbol{A}\boldsymbol{y}\|^{2} \leq (1+\epsilon)\|\boldsymbol{x}-\boldsymbol{y}\|^{2}\right) \geq 1-e^{-(\epsilon^{2}-\epsilon^{3})\frac{r}{4}}-e^{-\frac{(\epsilon^{2}-\epsilon^{3})r}{2(\frac{3p}{n}+1)}},$$

for any input embeddings  $x, y \in \mathbb{R}^n$ , where  $\sigma^2 = \frac{p}{nr^2}$ .

Since this construction does not differ significantly from our desired construction for A, which includes an additional constraint on non-zero entries per row and is more consistent with biological observations in the fly olfactory circuit, we use this more analytically tractable form as a surrogate to study the distance-preserving property. In practice, these two construction methods show negligible performance differences.

### <span id="page-24-1"></span>A.2 Top-k Activation Promotes Rank-Wise Decoupling

Let  $\Lambda \in \operatorname{diag}(\lambda_1, \dots, \lambda_r) \in \mathbb{R}^{r \times r}$  denote a binary mask where  $\lambda_i = 1$  if column i is activated by top-k. The relation of the masked gradient between top-k version  $(\frac{\partial \mathcal{L}}{\partial \tilde{B}})$  and the dense version  $(\frac{\partial \mathcal{L}}{\partial \tilde{B}})$  is:

$$\frac{\partial \mathcal{L}}{\partial \tilde{B}} = \frac{\partial \mathcal{L}}{\partial B} \Lambda \in \mathbb{R}^{m \times r}.$$
 (14)

Let  $\tilde{g}_i$  and  $g_i \in \mathbb{R}^m$  denote the gradient vectors for the *i*-th column of  $\frac{\partial \mathcal{L}}{\partial \tilde{B}}$  and  $\frac{\partial \mathcal{L}}{\partial B}$ , respectively. Their corresponding cross-column covariance are:

$$\Sigma_{(i,j)} = \mathbb{E}\left[\mathbf{g}_i^{\mathsf{T}} \mathbf{g}_j\right] - \mathbb{E}\left[\mathbf{g}_i\right]^{\mathsf{T}} \mathbb{E}\left[\mathbf{g}_j\right],\tag{15}$$

$$\tilde{\Sigma}_{(i,j)} = \mathbb{E}\left[\tilde{g}_i^{\top} \tilde{g}_j\right] - \mathbb{E}\left[\tilde{g}_i\right]^{\top} \mathbb{E}\left[\tilde{g}_j\right], \tag{16}$$

We can assume  $\mathbb{E}[\tilde{g}_i] = \mathbb{E}[\tilde{g}_i] = \mathbb{E}[g_i] = \mathbb{E}[g_i] = \mathbf{0}_m$ , these simplify to:

$$\Sigma_{(i,j)} = \mathbb{E}\left[\boldsymbol{g}_i^{\mathsf{T}} \boldsymbol{g}_j\right],\tag{17}$$

$$\tilde{\Sigma}_{(i,j)} = \mathbb{E}\left[\tilde{g}_i^{\top} \tilde{g}_j\right]. \tag{18}$$

Thus, the expected covariance depends only on the expected inner product of gradient vectors. From Assumption 3.2, the difference between  $\tilde{\Sigma}_{(i,j)}$ ,  $\Sigma_{(i,j)}$  is factored by the co-activation probability of columns i and j:

$$\mathbb{P}(\lambda_i = 1 \cap \lambda_j = 1) = \frac{\binom{r-2}{k-2}}{\binom{r}{k}} = \frac{k(k-1)}{r(r-1)} \approx \frac{k^2}{r^2}.$$
 (19)

This leads to the following theorem:

**Theorem A.3** (Covariance Reduction Under top-k). Let  $\Sigma$  and  $\Sigma$  denote the gradient covariance matrices with and without top-k activation. When r > k, the off-diagonal entries scale as:

$$\mathbb{E}[\tilde{\Sigma}_{(i,j)}] \approx \mathbb{E}[\Sigma_{(i,j)}] \cdot \frac{k^2}{r^2}, \quad \forall i \neq j.$$
 (20)

#### <span id="page-25-0"></span>A.3 Random Projection Induces Approximate Subspace Orthogonality

The orthogonality properties of random projections form the theoretical foundation for FlyLoRA's effectiveness in model merging. Consider two independent random projection matrices  $A_i, A_j \in \mathbb{R}^{r \times n}$  with entries distributed as specified in Section 3.1. The expectation of their product reveals:

$$\mathbb{E}[\boldsymbol{A}_{i}\boldsymbol{A}_{j}^{\top}] = \mathbb{E}\left[\sum_{k=1}^{n}(\boldsymbol{A}_{i})_{mk}(\boldsymbol{A}_{j})_{lk}\right] = \sum_{k=1}^{n}\mathbb{E}[(\boldsymbol{A}_{i})_{mk}]\mathbb{E}[(\boldsymbol{A}_{j})_{lk}] = \boldsymbol{0}_{r\times r}$$
(21)

This zero-expectation result follows from the independence and zero-mean property of the random matrices. To quantify how tightly the product concentrates around zero, we analyze its variance:

$$\operatorname{Var}\left((\boldsymbol{A}_{i}\boldsymbol{A}_{j}^{T})_{ml}\right) = \sum_{k=1}^{n} \left[\operatorname{Var}\left((\boldsymbol{A}_{i})_{mk}\right)\operatorname{Var}\left((\boldsymbol{A}_{j})_{lk}\right) + \operatorname{Var}\left((\boldsymbol{A}_{i})_{mk}\right)\left(\mathbb{E}(\boldsymbol{A}_{j})_{lk}\right)^{2} + \operatorname{Var}\left((\boldsymbol{A}_{j})_{lk}\right)\left(\mathbb{E}(\boldsymbol{A}_{i})_{mk}\right)^{2}\right]$$

$$= \sum_{k=1}^{n} \operatorname{Var}\left((\boldsymbol{A}_{i})_{mk}\right)\operatorname{Var}\left((\boldsymbol{A}_{j})_{lk}\right)$$

$$= n\sigma^{4} = \frac{p^{2}}{nr^{4}}$$
(22)

Applying Chebyshev's inequality, we bound the probability of large deviations for each entry:

$$\mathbb{P}\left(|(\boldsymbol{A}_{i}\boldsymbol{A}_{j}^{\top})_{ml}| \geq \epsilon\right) \leq \frac{\operatorname{Var}\left((\boldsymbol{A}_{i}\boldsymbol{A}_{j}^{\top})_{ml}\right)}{\epsilon^{2}} = \frac{p^{2}}{nr^{4}\epsilon^{2}}$$
(23)

The Frobenius norm characterization gives us:  $\|\boldsymbol{A}_{i}\boldsymbol{A}_{j}^{\top}\|_{F}^{2} = \sum_{m=1}^{r} \sum_{l=1}^{r} (\boldsymbol{A}_{i}\boldsymbol{A}_{j}^{\top})_{ml}^{2}$ . To analyze its probabilistic behavior, we employ the union bound principle: for any events  $E_{ij}$  defined as  $|(\boldsymbol{A}_{i}\boldsymbol{A}_{j}^{\top})_{ml}| \geq \epsilon$ , we have  $\mathbb{P}\left(\bigcup_{m,l} E_{ml}\right) \leq \sum_{m,l} \mathbb{P}(E_{ml})$ . Using this union bound over all  $r^{2}$  entries yields:

$$\mathbb{P}\left(\|\boldsymbol{A}_{i}\boldsymbol{A}_{j}^{\top}\|_{F} \geq \epsilon r\right) \leq \sum_{m=1}^{r} \sum_{l=1}^{r} \mathbb{P}\left(|(\boldsymbol{A}_{i}\boldsymbol{A}_{j}^{\top})_{ml}| \geq \epsilon\right) \leq r^{2} \cdot \frac{p^{2}}{nr^{4}\epsilon^{2}} = \frac{p^{2}}{nr^{2}\epsilon^{2}}.$$
 (24)

Since the spectral norm is bounded by the Frobenius norm ( $\|\boldsymbol{A}_i^{\top}\boldsymbol{A}_j\|_2 \leq \|\boldsymbol{A}_i^{\top}\boldsymbol{A}_j\|_F$ ), we conclude that when  $n \gg \frac{p^2}{r^2\epsilon^2}$ , the subspaces spanned by  $\boldsymbol{A}_i$  and  $\boldsymbol{A}_j$  are approximately orthogonal with high probability.

**Theorem A.4** (Approximate Subspace Orthogonality). For independent random matrices  $A_i$ ,  $A_j \in \mathbb{R}^{r \times n}$  with sparse Gaussian entries  $(\mathcal{N}(0, \frac{1}{r^2}) \text{ for } p < n \text{ randomly selected entries per row), the following holds:$ 

- 1. Exact mean orthogonality:  $\mathbb{E}[A_i A_i^{\top}] = \mathbf{0}_{r \times r}$
- 2. Polynomially decaying correlations:  $\mathbb{P}(\|A_iA_j^\top\|_2 \geq \epsilon r) \leq \frac{p^2}{nr^2\epsilon^2}$

We demonstrate the approximate orthogonality between distinct LoRA components  $B_i A_i$  and  $B_j A_j$  through Frobenius inner product analysis following [75]:

$$\langle \boldsymbol{B}_{j} \boldsymbol{A}_{j}, \boldsymbol{B}_{i} \boldsymbol{A}_{i} \rangle_{F} = \operatorname{tr} \left( (\boldsymbol{B}_{j} \boldsymbol{A}_{j})^{\top} (\boldsymbol{B}_{i} \boldsymbol{A}_{i}) \right)$$

$$= \operatorname{tr} \left( \boldsymbol{A}_{j}^{\top} \boldsymbol{B}_{j}^{\top} \boldsymbol{B}_{i} \boldsymbol{A}_{i} \right)$$

$$= \operatorname{tr} \left( \boldsymbol{B}_{j}^{\top} \boldsymbol{B}_{i} \boldsymbol{A}_{i} \boldsymbol{A}_{j}^{\top} \right)$$

$$\approx \operatorname{tr} \left( \boldsymbol{B}_{j}^{\top} \boldsymbol{B}_{i} \cdot \boldsymbol{0}_{r \times r} \right)$$

$$\approx 0$$
(25)

These analysis demonstrates that random projections naturally create nearly orthogonal subspaces, which helps prevent interference between different experts in FlyLoRA. The small residual correlation becomes negligible when input dimension  $n \to \infty$ , leading to the effectiveness of our sparse random projection approach for model merging.

Formally, considering the conventional LoRA merging scheme:

$$\mathbf{W}' = \mathbf{W}_0 + \sum_{i=1}^t w_i \mathbf{B}_i \mathbf{A}_i \tag{26}$$

When multiple FlyLoRA modules ( $B_iA_i$ ) are approximately orthogonal, the squared Frobenius norm of the merged weight matrix can be decomposed into the weighted sum of individual module norms.

$$\left\| \sum_{i=1}^{t} w_i \boldsymbol{B}_i \boldsymbol{A}_i \right\|_F^2 = \sum_{i=1}^{t} w_i^2 \|\boldsymbol{B}_i \boldsymbol{A}_i\|_F^2 + \sum_{i \neq j} w_i w_j \langle \boldsymbol{B}_i \boldsymbol{A}_i, \boldsymbol{B}_j \boldsymbol{A}_j \rangle_F$$
 (27)

$$\approx \sum_{i=1}^{t} w_i^2 \|\boldsymbol{B}_i \boldsymbol{A}_i\|_F^2, \tag{28}$$

which aligns with the "Weight disentanglement" property for task arithmetic [56]. We summarize these findings in the following corollary:

**Corollary A.5.** Let  $A_i, A_j \in \mathbb{R}^{r \times n}$  be fixed sparse random projections after initialization. Then for any learned matrices  $B_i A_i$  and  $B_j A_j$  the following properties hold:

1. Pairwise Orthogonality:

$$\langle \boldsymbol{B}_{i}\boldsymbol{A}_{i},\boldsymbol{B}_{i}\boldsymbol{A}_{i}\rangle_{F}\approx0$$
 for  $i\neq j$ 

2. Orthogonality's Outcome in Merging:

$$\left\| \sum_{i=1}^{t} w_i \boldsymbol{B}_i \boldsymbol{A}_i \right\|_F^2 \approx \sum_{i=1}^{t} w_i^2 \|\boldsymbol{B}_i \boldsymbol{A}_i\|_F^2$$

Through sparse random projections, FlyLoRA inherently constructs nearly orthogonal subspaces for parameter merging from different tasks.

#### <span id="page-26-0"></span>**B** Additional Results

### **B.1** Evaluation on Larger Models

We further conducted experiments using the Qwen-2.5-14B model, as shown in Tables 4 and 5, following the settings of Tables 1 and 2. The results show that FlyLoRA remains superior in accuracy

(for both single-task and multi-task settings) and is more parameter-efficient. We encountered no memory or convergence bottlenecks when training FlyLoRA on the 14B model, which consistently outperforms LoRA and Split-LoRA. This scalability confirms that our method's benefits extend effectively to larger model architectures without compromising training stability.

<span id="page-27-0"></span>Table 4: Performance Comparison of LoRA Variants in Single-task Evaluation using Qwen-2.5-14B. We evaluate various methods across four benchmarks: MMLU, ScienceQA, GSM8K (accuracy), and HumanEval (Pass@k), with all metrics reported in percentage (%). Param(%) indicates the percentage of activated trainable parameters relative to Full FT. The best results are highlighted in **bold**.

| Method                     | Param(%) | MMLU                         | ScienceQA                    | GSM8K                        | HumanEval        |
|----------------------------|----------|------------------------------|------------------------------|------------------------------|------------------|
| $LoRA_{(r=8)}$             | 0.23     | 56.74±0.56                   | $95.62{\scriptstyle\pm0.18}$ | 83.08±0.74                   | 51.69±1.48       |
| $LoRA_{(r=32)}$            | 0.93     | $59.35{\scriptstyle\pm0.79}$ | $97.05{\scriptstyle\pm0.22}$ | $85.31 \pm 0.25$             | $54.80 \pm 0.76$ |
| Split-LoRA $_{(4\times8)}$ | 0.29     | $58.26{\scriptstyle\pm1.13}$ | $96.85{\scriptstyle\pm0.35}$ | $84.88{\scriptstyle\pm0.51}$ | $54.65 \pm 0.59$ |
| FlyLoRA                    | 0.12     | $60.17{\scriptstyle\pm1.08}$ | 97.37±0.32                   | 85.96±0.89                   | 56.42±1.16       |

<span id="page-27-1"></span>Table 5: **Multi-task Performance Comparison using Qwen-2.5-14B.** We evaluate LoRA variants across MMLU, ScienceQA, GSM8K (accuracy), and HumanEval (Pass@k) benchmarks. The table shows the relative performance drop ( $\Delta$ %) before and after merging. The best results are highlighted in **bold**.

| Method                     | MMLU   | ScienceQA | GSM8K  | HumanEval |
|----------------------------|--------|-----------|--------|-----------|
| $LoRA_{(r=8)}$             | -13.75 | -25.20    | -11.43 | -18.60    |
| $LoRA_{(r=32)}$            | -8.91  | -20.45    | -7.62  | -16.34    |
| Split-LoRA $_{(4\times8)}$ | -7.48  | -21.97    | -6.05  | -14.87    |
| FlyLoRA                    | -4.35  | -17.89    | -2.18  | -11.72    |

### **B.2** More Baseline into Comparison

We further compare several strong and widely used baselines—AdaLoRA [95] (adaptive rank allocation), SoRA [16] (sparse adaptation), and HydraLoRA [76] (MoE-based)—using Qwen-2.5-7B, as shown in Tables 6 and 7. The results demonstrate that FlyLoRA remains superior in terms of accuracy and efficiency in both single-task learning and multi-task merging scenarios. Notably, FlyLoRA achieves this performance while maintaining a simpler training pipeline, as it avoids the additional hyperparameter tuning required by adaptive and sparse methods.

<span id="page-27-2"></span>Table 6: Performance Comparison with More LoRA Variants in Single-task Evaluation using Qwen-2.5-7B. We evaluate various methods across four benchmarks: MMLU, ScienceQA, GSM8K (accuracy), and HumanEval (Pass@k), with all metrics reported in percentage (%). Param(%) indicates the percentage of activated trainable parameters relative to Full FT. The best results are highlighted in **bold**.

| Method                       | Param(%) | MMLU                         | ScienceQA                    | GSM8K                        | HumanEval                    |
|------------------------------|----------|------------------------------|------------------------------|------------------------------|------------------------------|
| $AdaLoRA_{(r=8)}$            | 0.26     | $51.22{\scriptstyle\pm0.21}$ | $93.48 \pm 0.28$             | $77.65{\scriptstyle\pm0.14}$ | 47.96±1.34                   |
| $SoRA_{(r=8)}$               | 0.19     | $50.89{\scriptstyle\pm0.42}$ | $93.25 \pm 0.20$             | $78.46{\scriptstyle\pm0.82}$ | $47.83 \pm 0.94$             |
| HydraLoRA $_{(r=8,A=1,B=3)}$ | 0.52     | $53.05{\scriptstyle\pm0.16}$ | $94.69{\scriptstyle\pm0.34}$ | $79.31{\scriptstyle\pm0.49}$ | $52.98{\scriptstyle\pm1.57}$ |
| $FlyLoRA_{(k=8)}$            | 0.13     | 53.68±0.47                   | 95.55±0.18                   | 80.82±0.56                   | 54.34±2.13                   |

#### **B.3** Training Time and Memory Consumption

We further conduct experiments comparing the training time and memory consumption of the LoRA variants discussed in Section 4. The results, presented in Table 8, demonstrate that  $LoRA_{(r=32)}$  requires more training time and memory than  $LoRA_{(r=8)}$  due to its higher rank. Split- $LoRA_{(4\times8)}$  shows intermediate values between these two approaches. Notably, FlyLoRA achieves both the fastest training times (resulting from having the fewest activated parameters) and the lowest memory consumption (mainly attributed to its frozen matrix  $\boldsymbol{A}$  that significantly reduces memory for activation

<span id="page-28-0"></span>Table 7: Multi-task Performance Comparison with More Baselines using Qwen-2.5-7B. We evaluate LoRA variants across MMLU, ScienceQA, GSM8K (accuracy), and HumanEval (Pass@k) benchmarks. The table shows the relative performance drop ( $\Delta\%$ ) before and after merging. The best results are highlighted in **bold**.

| Method                      | MMLU   | ScienceQA | GSM8K | HumanEval |
|-----------------------------|--------|-----------|-------|-----------|
| $AdaLoRA_{(r=8)}$           | -10.90 | -33.15    | +4.52 | -25.46    |
| $SoRA_{(r=8)}$              | -9.45  | -34.81    | +4.05 | -26.50    |
| $HydraLoRA_{(r=8,A=1,B=3)}$ | -6.22  | -34.67    | +4.21 | -24.94    |
| $FlyLoRA_{(k=8)}$           | +6.55  | -23.77    | +4.80 | -21.23    |

values [94]). We also illustrate the theoretical memory consumption for these LoRA variants in Table 9, which tightly aligns with the experimental results. It's clear to see that under a fixed total rank r, a larger number of experts N causes Split-LoRA to require significantly more activated trainable parameters and memory consumption, degrading the efficiency of MoE-based LoRA methods. These results and analysis demonstrate the robustness of FlyLoRA's parameter efficiency compared to MoE-based LoRA methods.

<span id="page-28-1"></span>Table 8: Training Time (Hours) and Memory Usage (GB) of LoRA Variants on Different Datasets and Architectures. Comparison of  $LoRA_{(r=8)}$ ,  $LoRA_{(r=32)}$ , Split- $LoRA_{(4\times8)}$ , and FlyLoRA $_{(k=8)}$  fine-tuning Llama-3.1-8B and Qwen-2.5-7B on MMLU, ScienceQA, GSM8K, and CodeAlpaca-20k. The best results are highlighted in **bold**.

| Metric                                                       |                 | Llam            | a-3.1-8B        |                 | Qwen-2.5-7B     |                 |                 |                 |
|--------------------------------------------------------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| Wethe                                                        | MMLU            | ScienceQA       | GSM8K           | CodeAlpaca      | MMLU            | ScienceQA       | GSM8K           | CodeAlpaca      |
|                                                              | 4.79h<br>12.5GB | 5.30h<br>14.8GB | 0.52h<br>20.1GB | 1.85h<br>20.2GB | 4.45h<br>14.7GB | 5.07h<br>16.8GB | 0.54h<br>22.9GB | 1.75h<br>22.9GB |
|                                                              | 5.09h<br>13.2GB | 5.61h<br>15.3GB | 0.58h<br>20.7GB | 1.95h<br>20.7GB | 4.76h<br>15.4GB | 5.39h<br>17.3GB | 0.60h<br>23.4GB | 1.87h<br>23.4GB |
| Split-LoRA <sub>(4×8)</sub><br>Training Time<br>Memory Usage | 4.94h<br>12.8GB | 5.46h<br>15.0GB | 0.56h<br>20.4GB | 1.92h<br>20.4GB | 4.60h<br>15.0GB | 5.23h<br>17.1GB | 0.57h<br>23.2GB | 1.79h<br>23.2GB |
| FlyLoRA <sub>(k=8)</sub><br>Training Time<br>Memory Usage    | 4.73h<br>10.6GB | 5.23h<br>10.7GB | 0.51h<br>10.9GB | 1.82h<br>10.9GB | 4.39h<br>12.1GB | 4.99h<br>12.2GB | 0.52h<br>12.4GB | 1.70h<br>12.4GB |

<span id="page-28-2"></span>Table 9: Theoretical Memory Consumption Comparison of Different LoRA Variants for a Single Linear Layer. Param indicates the number of activated trainable parameters. Variables d, r, k, b, s, and N represent hidden dimension, total rank, activation rank, batch size, sequence length, and number of experts, respectively. We record memory usage for weights, gradients, optimizer states, and activations in bytes. These results are calculated under 16-bit mixed-precision training settings.

| Method     | Param    | Weight              | Gradient  | Optimizer   | Activation         |
|------------|----------|---------------------|-----------|-------------|--------------------|
| LoRA       | 2dr      | $2(d^2 + 2dr)$      | 4dr       | 24dr        | 2bsd + 2bsr        |
| Split-LoRA | 2dk + dN | $2(d^2 + 2dr + dN)$ | 4dk + 2dN | 24dk + 12dN | 2bsd + 2bsk + 2bsN |
| FlyLoRA    | dk       | $2(d^2 + 2dr)$      | 2dk       | 12dk        | 2bsk               |

#### **B.4** Multi-task Performance for Advanced Model Merging Techniques

Extending the results in Section 4.3, we also evaluate two advanced model merging techniques, TIES-MERGING [85] and DARE [87], for merging LoRA components from different domains. The results are listed in Tables 10 and 11, respectively. Overall, both TIES-MERGING and DARE outperform naive weight averaging by resolving parameter conflicts through intelligent selection (TIES' sign consensus and trimming) and selective rescaling (DARE's dropout-based redundancy elimination). Similar to weight averaging, these results demonstrate that FlyLoRA consistently surpasses  $LoRA_{(r=8)}$ ,  $LoRA_{(r=32)}$ , and Split- $LoRA_{(4\times8)}$  across all comparisons, highlighting the

<span id="page-29-0"></span>Table 10: Multi-task Performance Comparison Before and After Parameter Merging Using TIES-MERGING. We evaluate LoRA variants across MMLU, ScienceQA, GSM8K (accuracy), and HumanEval (Pass@k) benchmarks. The table shows performance before merging, after merging, and the relative performance drop ( $\Delta\%$ ). The best results are highlighted in **bold**.

| Model             | Method                                        | Merge Status | MMIII                        | CaianaaOA                    | CCMOV                          | ]                            | HumanEva                                                                                                       | l                            |
|-------------------|-----------------------------------------------|--------------|------------------------------|------------------------------|--------------------------------|------------------------------|----------------------------------------------------------------------------------------------------------------|------------------------------|
| Model             | Method                                        | Merge Status | MINILU                       | ScienceQA                    | GSMoK                          | Pass@1                       | Pass@5                                                                                                         | Pass@10                      |
|                   |                                               | Before       | 36.53±0.40                   | 91.39±0.55                   | 55.34±0.24                     | 29.13±0.56                   | 52.28±1.24                                                                                                     | 61.67±0.61                   |
|                   | $LoRA_{(r=8)}$                                | After        | $31.83 \pm 0.34$             | $35.96 \pm 1.46$             | $27.53 \pm 1.08$               | $18.45 \pm 1.47$             | $45.52 \pm 0.84$                                                                                               | $56.63 \pm 1.24$             |
|                   |                                               | $\Delta$ (%) | -4.70                        | -55.43                       | -27.81                         | -10.68                       | -6.76                                                                                                          | -5.04                        |
|                   |                                               | Before       | $38.93{\scriptstyle\pm1.04}$ | $94.01 \pm 0.17$             | $56.25{\scriptstyle\pm0.29}$   | $30.37{\scriptstyle\pm1.06}$ | $54.37{\scriptstyle\pm0.39}$                                                                                   | $64.02{\scriptstyle\pm0.94}$ |
|                   | $LoRA_{(r=32)}$                               | After        | $34.45 \pm 1.73$             | $36.63 \pm 0.79$             | $26.59 \pm 0.47$               | $20.98 \pm 1.13$             | $47.15 \pm 0.82$                                                                                               | $59.97 \pm 1.04$             |
| Llama-3.1-8B      | ,                                             | $\Delta$ (%) | -4.48                        | -57.38                       | -29.66                         | -9.39                        | -7.22                                                                                                          | -4.05                        |
| 21111111 211 02   |                                               | Before       | 38.44±0.69                   | 92.41±0.54                   | 55.65±0.47                     | 31.28±1.52                   | 54.16±1.12                                                                                                     | 63.94±0.89                   |
|                   | Split-LoRA <sub><math>(4\times8)</math></sub> | After        | $33.92 \pm 0.57$             | $40.81 \pm 1.34$             | $29.02{\scriptstyle\pm0.82}$   | $22.02{\scriptstyle\pm0.24}$ | $46.32{\scriptstyle\pm0.72}$                                                                                   | $59.72 \pm 0.25$             |
|                   |                                               | $\Delta$ (%) | -4.52                        | -51.60                       | -26.63                         | -9.26                        | -7.84                                                                                                          | -4.22                        |
|                   |                                               | Before       | 40.88±1.61                   | $94.15 \pm 0.36$             | 58.76±0.74                     | 36.88±1.91                   | 62.40±1.82                                                                                                     | $73.34 \pm 1.24$             |
|                   | $FlyLoRA_{(k=8)}$                             | After        | $39.03 \pm 1.31$             | $54.82 \pm 0.46$             | $39.12 \pm 1.02$               | $33.37 \pm 0.62$             | 24 46.32±0.72 59<br>-7.84  91 62.40±1.82 73<br>62 57.36±1.41 70<br>-5.04  54 78.89±0.36 85<br>24 68.96±1.22 81 | $70.35 \pm 0.39$             |
|                   |                                               | $\Delta$ (%) | -1.85                        | -39.33                       | -19.64                         | -3.51                        | -5.04                                                                                                          | -2.99                        |
|                   |                                               | Before       | $49.84{\scriptstyle\pm0.56}$ | $92.84{\scriptstyle\pm0.13}$ | $77.01{\scriptstyle\pm0.32}$   | $47.20{\scriptstyle\pm1.54}$ | $78.89{\scriptstyle\pm0.36}$                                                                                   | $85.94{\scriptstyle\pm0.64}$ |
|                   | $LoRA_{(r=8)}$                                | After        | $44.98{\scriptstyle\pm0.72}$ | $61.69 \pm 1.34$             | $81.89{\scriptstyle\pm1.04}$   | $23.37 \pm 1.24$             | $68.96{\scriptstyle\pm1.22}$                                                                                   | $81.25 \pm 0.35$             |
|                   |                                               | $\Delta$ (%) | -4.86                        | -31.15                       | +4.88                          | -23.83                       | $\begin{array}{cccccccccccccccccccccccccccccccccccc$                                                           | -4.69                        |
|                   |                                               | Before       | 52.07±0.31                   | $95.01 \pm 0.21$             | $79.23{\scriptstyle \pm 0.22}$ | 52.87±1.79                   | 81.67±1.14                                                                                                     | $87.80{\scriptstyle\pm0.72}$ |
|                   | $LoRA_{(r=32)}$                               | After        | $35.92 \pm 0.18$             | $58.02 \pm 0.27$             | $83.98{\scriptstyle\pm0.62}$   | $24.79 \pm 1.08$             | $67.50 \pm 1.13$                                                                                               | $80.35 \pm 1.46$             |
| Qwen-2.5-7B       |                                               | $\Delta$ (%) | -16.15                       | -36.99                       | +4.75                          | -28.08                       | -14.17                                                                                                         | -7.45                        |
| Q.,, e.i. 2.i. 72 |                                               | Before       | 50.68±1.06                   | $93.08{\scriptstyle\pm0.41}$ | $77.12 \pm 0.76$               | 48.65±1.18                   | $79.30{\scriptstyle\pm0.91}$                                                                                   | $86.05{\scriptstyle\pm0.44}$ |
|                   | Split-LoRA $_{(4\times8)}$                    | After        | $44.96 \pm 0.65$             | $60.59 \pm 0.26$             | $81.82 \pm 0.20$               | $23.53 \pm 0.74$             | $67.59 \pm 0.14$                                                                                               | $82.34 \pm 0.69$             |
|                   |                                               | $\Delta$ (%) | -5.72                        | -32.49                       | +4.70                          | -25.12                       | -11.71                                                                                                         | -3.71                        |
|                   |                                               | Before       | $53.68{\scriptstyle\pm0.47}$ | $95.55{\scriptstyle\pm0.18}$ | $80.82{\scriptstyle\pm0.56}$   | 54.34±2.13                   | $82.85{\scriptstyle\pm0.52}$                                                                                   | $89.63{\scriptstyle\pm0.55}$ |
|                   | $FlyLoRA_{(k=8)}$                             | After        | $60.51 \pm 0.37$             | $73.46 \pm 0.80$             | $86.24 \pm 0.31$               | $35.37 \pm 0.47$             | $76.05{\scriptstyle\pm1.25}$                                                                                   | $87.97 \pm 1.09$             |
|                   |                                               | $\Delta$ (%) | +6.83                        | -22.09                       | +5.42                          | -18.97                       | -6.80                                                                                                          | -1.66                        |

robustness of FlyLoRA's near-orthogonality property in reducing inter-task decoupling and further enhancing model merging performance.

We also include comparison with more advanced model merging techniques in Table 12. KnOTS [73] and L-LoRA [75] are both built upon  $LoRA_{(r=32)}$ . The results suggest that they achieve comparable performance to FlyLoRA, with each method excelling on different datasets. It is noteworthy that FlyLoRA is not a competitor to these methods; rather, they can be used in a plug-and-play manner with FlyLoRA to further improve performance after merging.

#### **B.5** Additional Ablation Studies on Load-Balancing Strategies

We compare experimental results using different load-balancing strategies. In Section 3.1, we employ an easy-to-implement loss-free balancing strategy. This loss-agnostic approach effectively achieves load balancing with negligible computational overhead and memory footprint. Simultaneously, other loss-controlled load-balancing strategies like [19] are widely used in MoE-like structures. A comparison of them is shown in Table 13. Our results show that different routing strategies achieve similar effects in performance. While FlyLoRA requires load-balancing strategies, it is not sensitive to specific methods.

### **B.6** Additional Ablation Studies on K-Selection Strategies

To evaluate the impact of activation selection, we analyze different K-selection approaches in our experiments. In neuroscience, the fly olfactory circuit implements a "winner-take-all" strategy, simulated through top-k selection based on activation values across dimensions. To validate top-k's effectiveness, we test random-k selection and full activation (without selection) as baselines. Results in Table 14 show that both top-k and random-k outperform full activation, confirming sparse activation mitigates intra-task interference. Crucially, top-k surpasses random-k because random selection cannot prioritize the most informative dimensions, leading to suboptimal performance.

<span id="page-30-0"></span>Table 11: Multi-task Performance Comparison Before and After Parameter Merging Using **DARE.** We evaluate LoRA variants across MMLU, ScienceQA, GSM8K (accuracy), and HumanEval (Pass@k) benchmarks. The table shows performance before merging, after merging, and the relative performance drop ( $\Delta$ %). The best results are highlighted in **bold**.

| Model          | Method                      | Merge Status  | MMLII                        | SajanaaOA        | CCMOR                        | ]                | HumanEva                     | l                |
|----------------|-----------------------------|---------------|------------------------------|------------------|------------------------------|------------------|------------------------------|------------------|
| Model          | Method                      | Wierge Status | MINILU                       | ScienceQA        | GSMOK                        | Pass@1           | Pass@5                       | Pass@10          |
|                |                             | Before        | 36.53±0.40                   | 91.39±0.55       | 55.34±0.24                   | 29.13±0.56       | 52.28±1.24                   | 61.67±0.61       |
|                | $LoRA_{(r=8)}$              | After         | $31.24 \pm 0.40$             | $34.37 \pm 1.25$ | $26.76{\scriptstyle\pm1.10}$ | $17.35 \pm 1.32$ | $45.49 \pm 0.96$             | $57.24 \pm 1.36$ |
|                | ` ′                         | $\Delta$ (%)  | -5.29                        | -57.02           | -28.58                       | -11.78           | -6.79                        | -4.43            |
|                |                             | Before        | 38.93±1.04                   | 94.01±0.17       | 56.25±0.29                   | 30.37±1.06       | 54.37±0.39                   | 64.02±0.94       |
|                | $LoRA_{(r=32)}$             | After         | $34.75 \pm 0.83$             | $37.56 \pm 0.59$ | $26.89{\scriptstyle\pm0.78}$ | $19.36 \pm 1.38$ | $46.67{\scriptstyle\pm0.86}$ | $59.85 \pm 0.67$ |
| Llama-3.1-8B   |                             | $\Delta$ (%)  | -4.18                        | -56.45           | -29.36                       | -11.01           | -7.70                        | -4.17            |
| Elama 3.1 ob   |                             | Before        | 38.44±0.69                   | 92.41±0.54       | 55.65±0.47                   | 31.28±1.52       | 54.16±1.12                   | 63.94±0.89       |
|                | Split-LoRA <sub>(4×8)</sub> | After         | $34.02{\scriptstyle\pm0.24}$ | $38.58 \pm 0.48$ | $28.63{\scriptstyle\pm0.72}$ | $23.52 \pm 1.43$ | $46.84{\scriptstyle\pm0.30}$ | $60.30 \pm 0.41$ |
|                | ,                           | $\Delta$ (%)  | -4.42                        | -53.83           | -27.02                       | -7.76            | -7.32                        | -3.64            |
|                | $FlyLoRA_{(k=8)}$           | Before        | 40.88±1.61                   | 94.15±0.36       | 58.76±0.74                   | 36.88±1.91       | 62.40±1.82                   | 73.34±1.24       |
|                |                             | After         | $39.37 \pm 1.02$             | $52.34 \pm 0.35$ | $38.20{\scriptstyle\pm1.52}$ | $33.34 \pm 0.83$ | $57.14 \pm 1.37$             | $70.24 \pm 0.42$ |
|                |                             | $\Delta$ (%)  | -1.51                        | -41.81           | -20.56                       | -3.54            | -5.26                        | -3.10            |
|                |                             | Before        | 49.84±0.56                   | 92.84±0.13       | 77.01±0.32                   | 47.20±1.54       | 78.89±0.36                   | 85.94±0.64       |
|                | $LoRA_{(r=8)}$              | After         | $45.20{\scriptstyle\pm0.40}$ | $61.39 \pm 1.32$ | $81.98{\scriptstyle\pm0.86}$ | $23.49 \pm 1.02$ | $69.04 \pm 0.17$             | $81.22 \pm 0.27$ |
|                | ` ′                         | $\Delta$ (%)  | -4.64                        | -31.45           | +4.97                        | -23.71           | -9.85                        | -4.72            |
|                |                             | Before        | 52.07±0.31                   | 95.01±0.21       | 79.23±0.22                   | 52.87±1.79       | 81.67±1.14                   | 87.80±0.72       |
|                | $LoRA_{(r=32)}$             | After         | $35.27 \pm 1.68$             | $56.79 \pm 1.34$ | $84.07 \pm 0.16$             | $24.35 \pm 1.07$ | $67.74 \pm 0.70$             | $80.12 \pm 0.32$ |
| Qwen-2.5-7B    |                             | $\Delta$ (%)  | -16.80                       | -38.22           | +4.80                        | -28.52           | -13.93                       | -7.68            |
| Q. (CH 2.5 / B |                             | Before        | 50.68±1.06                   | 93.08±0.41       | 77.12±0.76                   | 48.65±1.18       | 79.30±0.91                   | 86.05±0.44       |
|                | Split-LoRA <sub>(4×8)</sub> | After         | $43.56{\scriptstyle\pm0.84}$ | $62.15 \pm 0.26$ | $81.82 \pm 0.20$             | $23.79 \pm 0.52$ | $68.74 \pm 0.96$             | $81.53 \pm 1.07$ |
|                | ,                           | $\Delta$ (%)  | -7.12                        | -30.93           | +4.70                        | -24.86           | -10.56                       | -4.52            |
|                |                             | Before        | 53.68±0.47                   | 95.55±0.18       | 80.82±0.56                   | 54.34±2.13       | 82.85±0.52                   | 89.63±0.55       |
|                | $FlyLoRA_{(k=8)}$           | After         | $61.35{\scriptstyle\pm0.47}$ | $72.94 \pm 0.21$ | $86.34{\scriptstyle\pm0.36}$ | $34.47 \pm 0.44$ | $75.97 \pm 0.70$             | $87.64 \pm 1.04$ |
|                | . (0)                       | $\Delta$ (%)  | +7.67                        | -22.61           | +5.52                        | -19.87           | -6.88                        | -1.99            |

<span id="page-30-1"></span>Table 12: Multi-task Performance Comparison with More Advanced Merging Techniques using Qwen-2.5-7B. We evaluate different methods across MMLU, ScienceQA, GSM8K (accuracy), and HumanEval (Pass@k) benchmarks. The table shows the relative performance drop ( $\Delta$ %) before and after merging. The best results are highlighted in **bold**.

| Method         | MMLU   | ScienceQA | GSM8K | HumanEval |
|----------------|--------|-----------|-------|-----------|
| FlyLoRA        | +6.55  | -23.77    | +4.80 | -21.23    |
| KnOTS          | +10.76 | -26.85    | +4.68 | -23.37    |
| L-LoRA         | +4.51  | -22.48    | +4.74 | -20.85    |
| KnOTS+FlyLoRA  | +11.47 | -23.41    | +5.25 | -20.69    |
| L-LoRA+FlyLoRA | +7.65  | -21.42    | +5.02 | -19.85    |

These findings collectively demonstrate that biologically inspired top-k activation optimally balances efficiency and task-specific feature selection.

#### **B.7** Additional Ablation Studies on Matrix A initialization Schemes

We further compare three methods for generating the sparse projection A with  $\frac{p}{r}$  sparsity:

- 1. Gaussian (our default): Each non-zero entry is drawn from  $\mathcal{N}(0,\frac{1}{r^2})$ .
- 2. Rademacher (non-Gaussian): Each non-zero entry is  $\pm \frac{1}{r}$  with equal probability.
- 3. FJLT [3] (structured projection): A = PHD, where D is a random diagonal matrix with independent Rademacher variables on its diagonal, H is a normalized Hadamard matrix, and P enforces the  $\frac{p}{r}$  sparsity.
- 4. Two-Phase (briefly-learned): The non-zero entries of A are trainable for 5% of total steps as warm-up, then frozen for the remainder.

The results, shown in Table 15, indicate that almost all variants perform similarly. Non-Gaussian, structured, or briefly-learned initializations have little impact, except that the briefly-learned scheme

<span id="page-31-1"></span>Table 13: **Performance Comparison of Different Load-Balancing Strategies.** Evaluation on MMLU benchmark using Llama-3.1-8B.

| <b>Load-Balancing Strategy</b> | Accuracy (%)                 |
|--------------------------------|------------------------------|
| Loss-Free                      | $40.88{\scriptstyle\pm1.61}$ |
| Loss-Controlled                | $40.59 \pm 0.51$             |
| No Load-Balancing              | $37.56 \pm 2.87$             |

<span id="page-31-2"></span>Table 14: **Performance Comparison of Different K-Selection Strategies.** Evaluation on MMLU benchmark using Llama-3.1-8B.

| K-Selection Strategy | Accuracy (%)                  |
|----------------------|-------------------------------|
| top-k                | $40.88 {\scriptstyle\pm1.61}$ |
| random-k             | $40.02 \pm 0.26$              |
| full activation      | $39.40{\scriptstyle\pm1.14}$  |

shows a noticeable drop after merging. This demonstrates that FlyLoRA is robust to the choice of initialization scheme for matrix A, and that learning may break the approximate orthogonality of the random matrix, making it unsuitable.

#### B.8 Analysis of the Performance Gap Between Merged and Non-merged Scenarios

In Tables 2, 10, and 11, we can see that ScienceQA usually demonstrates a large performance drop after merging for all LoRA variants. Intuitively, the four tasks—general knowledge understanding (MMLU), scientific question answering (ScienceQA), mathematical reasoning (GSM8K), and code generation (HumanEval)—represent significantly different distributions, so merging their adapters is prone to substantial conflicts.

Empirically, following [73], we use centered kernel alignment (CKA) [33] to quantify the alignment between the output representations of each single-task adapter and the merged adapter. A higher CKA indicates better output alignment, which is likely the inherent reason, and therefore, results in a smaller accuracy drop after merging. Table 16 reports both CKA and accuracy drop ( $\Delta$ ) on Llama-3.1-8B. Since there is no apparent difference between LoRA $_{(r=8)}$ , LoRA $_{(r=32)}$ , and Split-LoRA $_{(4\times8)}$  in model merging, we use LoRA $_{(r=8)}$  as a representative to compare with FlyLoRA $_{(k=8)}$ . We observe that tasks with lower CKA (especially ScienceQA and GSM8K) suffer the largest accuracy drops. FlyLoRA consistently yields higher CKA than LoRA, which aligns with its consistently smaller  $\Delta$ . This micro-level analysis corroborates why FlyLoRA outperforms LoRA in heterogeneous-task merging.

