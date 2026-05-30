# <span id="page-14-0"></span>A QUANTIZATION ERROR DISTRIBUTION

We present the distribution of quantization errors and their relationship to outliers in the pre-trained weights, as discussed in Section [2.1,](#page-2-2) in Figure [7.](#page-14-1) Figure [7\(](#page-14-1)a) shows the overall error distribution, while Figure [7\(](#page-14-1)b) highlights the channel-wise similarity between quantization errors and pre-trained weights in the 14th layer of LLaMA-3.2-3B. During quantization, values are divided by the quantization scale, typically defined per group within each output channel, and then rounded to an integer and clamped within a range determined by the bit-width. Most quantization errors remain within this rounding range, but large-magnitude outliers are often clamped, leading to large errors. Because model accuracy is highly sensitive to outlier weights, their quantization errors can significantly degrade performance. In QA-PEFT, it is therefore crucial to mitigate such outlier-induced errors during initialization by adapting the weights, particularly for large-magnitude values originating from salient outliers.

<span id="page-14-1"></span>![](_page_14_Figure_2.jpeg)

Figure 7: (a) Weight quantization error distribution and (b) its channel-wise similarity to the pretrained weights in 14th layer Key projection of 4-bit quantized LLaMA-3.2-3B. In Figure (b), each pixel represents the ℓ<sup>2</sup> norm of weight quantization errors (left) and that of pre-trained weights (right) for each output channel ordered by channel index from top-left to bottom-right.

## B WHT-BASED ADAPTER (WHA)

#### <span id="page-15-0"></span>B.1 FT-BASED ADAPTER KERNELS

We describe a class of Fourier-related transform (FT) kernels employed in our adapters and prior studies in this section (Gao et al., 2024b; Du et al., 2025; Shen et al., 2025).

Walsh-Hadamard Transform (WHT). The Walsh-Hadamard Transform (WHT) matrix  $\boldsymbol{H}$  introduced in Equation 4 is constructed following conventions in prior works (Tseng et al., 2024; Ashkboos et al., 2024). For a dimension  $N=2^n$ , a WHT matrix  $\boldsymbol{H} \in \mathbb{R}^{N \times N}$  is defined recursively as the Kronecker product of smaller matrices:

$$\boldsymbol{H}_{2} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1\\ 1 & -1 \end{bmatrix}, \quad \boldsymbol{H}_{N} = \boldsymbol{H}_{2} \otimes \boldsymbol{H}_{2^{n-1}}, \tag{10}$$

where  $\otimes$  denotes the Kronecker product. For non-power-of-two dimensions, Hadamard matrices exist for certain values (Seberry & Yamada, 1992; Hedayat et al., 1999; Gerakoulis & Ghassemzadeh, 2004), which can be retrieved from Sloane (2004). More generally, for  $N=2^n \cdot m$ , where  $H_m$  is a known Hadamard matrix, the transform is defined as:

$$H_N = H_{2^n} \otimes H_m. \tag{11}$$

The rows of  $H_N$  form an orthogonal basis, known as Walsh-Hadamard bases, satisfying:

$$\boldsymbol{H}_{N}^{\top}\boldsymbol{H}_{N} = \boldsymbol{H}_{N}\boldsymbol{H}_{N}^{\top} = \boldsymbol{I}_{N}. \tag{12}$$

The matrix  $H_{2^n}$  can be computed in  $O(n \log n)$  time (Kunz, 1979). In practice,  $H_N$  can be precomputed once and cached for reuse across layers of the same size, incurring negligible cost in both computation and memory. To further accelerate computation, we employ the Fast Hadamard multiplication kernel from Dao-AILab (2024), which avoids explicit matrix construction by using a fused kernel of only additions and subtractions.

**Discrete Fourier Transform (DFT).** FourierFT (Gao et al., 2024b) was the first study of FT-based adapters and used the discrete Fourier transform (DFT). The transform kernel  $\boldsymbol{H} \in \mathbb{C}^{N \times N}$  is defined as:

$$H_{jk} = \frac{1}{\sqrt{N}} e^{-i\frac{2\pi jk}{N}} = \frac{1}{\sqrt{N}} \left\{ \cos\left(\frac{2\pi jk}{N}\right) - i\sin\left(\frac{2\pi jk}{N}\right) \right\}, \quad 0 \le j, k < N.$$
 (13)

Although effective, later works adopted real-valued FT variants to avoid the complex-domain nature of the DFT, since deep learning frameworks typically discard the imaginary components and compute only with the real values.

**Discrete Hartley Transform (DHT).** SSH (Shen et al., 2025) employs the discrete Hartley transform (DHT), a real-valued variant of the FT with kernel:

$$H_{jk} = \Re\left(\frac{1}{\sqrt{N}}e^{-i\frac{2\pi jk}{N}}\right) - \Im\left(\frac{1}{\sqrt{N}}e^{-i\frac{2\pi jk}{N}}\right) = \frac{1}{\sqrt{N}}\operatorname{cas}\left(\frac{2\pi jk}{N}\right), \quad 0 \le j, k < N, \tag{14}$$

where cas(x) = cos x + sin x.

**Discrete Cosine Transform (DCT).** LoCA (Du et al., 2025) employs another real-valued FT, the discrete cosine transform (DCT), whose kernel is:

$$H_{jk} = \begin{cases} \frac{1}{\sqrt{N}} & j = 0\\ \sqrt{\frac{2}{N}} \cos\left(\frac{\pi(2k+1)j}{2N}\right) & 0 < j < N \end{cases}, \ 0 \le k < N.$$
 (15)

### <span id="page-16-0"></span>B.2 RANK OF WHA

This section provides a detailed explanation of the full-rank property of WHA and its conditions, as discussed in Section 3.1 and illustrated in Figure 2(a). To preserve the expressiveness of a fine-tuned model under a limited parameter budget, it is critical to ensure high rank capacity in the weight update. Unlike low-rank adapters, which inherently restrict the parameter subspace, WHA is sparsely structured yet can retain high representational capacity by maintaining full rank. This also holds in typical sparse adapters, including FT-based adapters.

We build on theoretical insights from prior work on sparse random matrices Coja-Oghlan et al. (2020), which provides conditions under which such matrices are full rank. Specifically, consider a random sparse matrix  $F \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ , where each input and output channel has k and l non-zero entries on average. Then, F is full rank when  $k, l \geq 2$  as  $d_{\text{in}}, d_{\text{out}} \to \infty$ , and thus full rank with high probability. Following the notations in Coja-Oghlan et al. (2020), we derive the corresponding condition for our setting to guarantee full-rank behavior in WHA.

**Condition Function.** We define the probability generating functions for the distributions of random non-zero entries per column and per channel. Given that these distributions are degenerate, the generating functions and their derivatives are:

$$D(z) = z^k, \quad D'(z) = kz^{k-1}, \quad D'(1) = k,$$
 (16)

$$K(z) = z^{l}, \quad K'(z) = lz^{l-1}, \quad K'(1) = l,$$
 (17)

Then, the condition function  $\Phi(z)$  that determines the full rank condition is given by:

<span id="page-16-1"></span>
$$\Phi(z) = D\left(1 - \frac{K'(z)}{l}\right) - \frac{k}{l}\left[1 - K(z) - (1 - z)K'(z)\right]. \tag{18}$$

To ensure the full rank of the matrix A, the inequality must hold as:

<span id="page-16-2"></span>
$$\Phi(z) < \Phi(0), \quad \forall \, 0 < z \le 1, \tag{19}$$

Substituting the explicit forms for D(z), K(z), D'(z), K'(z) into Equation B.2 yields the right hand side as:

$$\Phi(z) = (1 - z^{l-1})^k - \frac{k}{l} + kz^{l-1} - \frac{k(l-1)}{l}z^l.$$
 (20)

As  $\Phi(z=0)=1-\frac{k}{l}$ , the condition in Equation B.2 finally simplifies to:

<span id="page-16-3"></span>
$$(1 - z^{l-1})^k + kz^{l-1} - \frac{k(l-1)}{l}z^l - 1 < 0, \quad 0 < z \le 1.$$
(21)

**Practical Considerations.** The inequality in Equation B.2 shows that the condition generally holds for integers  $k,l \geq 2$ . For the total number of parameters  $p = r(d_{\rm in} + d_{\rm out})$  with  $r \geq 2$ , we have  $k = p/d_{\rm in} > r$  and  $l = p/d_{\rm out} > r$  under random selection, thus satisfying the full-rank condition when  $d_{\rm in}$ ,  $d_{\rm out}$  are sufficiently large. Importantly, AdaAlloc's per-channel allocation with remainder assignment and temperature control guarantees at least two elements in every channel (i.e.,  $l \geq 2$ ), which meets the sufficient condition required for the full-rank property. In addition, although AdaAlloc selects coefficient indices within each channel based on the magnitude of  $\Delta W_Q H$ , the coefficient distribution of  $\Delta W_Q H$  under the WHT is close to a random normal distribution except for a small portion of outliers, as the correlations across input rows are nearly zero. Consequently, the selected index locations across input rows effectively behave like random choices. Empirically, parameter budgets corresponding to  $P(r \geq 4)$  ensure at least two elements per row (i.e.,  $k \geq 2$ ), even for linear layers with large output dimensions, which might otherwise receive few parameters per input row. Hence, the full-rank conditions hold, and the matrix F in QWHA is nearly full rank.

### <span id="page-17-0"></span>**B.3** ENERGY CONCENTRATION OF WHT

In this section, we quantify the energy concentration property of WHT discussed in Section 3.1, using Figure 2(b) and Figure 3(a).

**Distribution of Singular Values and Coefficients.** Figure 8 presents the distributions of singular values from SVD and of transform coefficients sorted by their squared magnitudes. Here, the area under each plot is equal to  $\|\Delta W_Q\|_F^2$  (details in the following paragraph). The distributions follow a Pareto-like behavior, where sharpness can be quantified using the hill index  $\eta$ . The Pareto hill index is a value which implies the heaviness of a tail. In fact, this is the reciprocal of the Pareto tail index, defined as the mean of the log ratios of consecutive order statistics from the top-k largest magnitudes. A smaller hill index  $\eta$  implies faster convergence of the cumulative distribution (Arnold, 1983). Hence, WHT exhibits the sharpest distribution, making it feasible to retain more information with fewer parameters P(r) when implemented in a sparse adapter, as shown in Figure 2.

<span id="page-17-1"></span>Singular Value and Coefficient Magnitude Distribution of  $\Delta W_Q$  in each Decomposition and Transforms

![](_page_17_Figure_4.jpeg)

Figure 8: Singular value and coefficient magnitude (squared) distributions with the Pareto hill index  $\eta$  in the 14<sup>th</sup>-layer Key projection of LLaMA-3.2-3B.

**Energy of Singular Values and Coefficients.** Throughout this work, we use the term *energy* to denote the squared  $\ell_2$  norm of the singular values in a decomposition or the spectral coefficients in a transform. We show that the total energy of both SVD and orthonormal transforms reduces to the Frobenius norm of the transformed matrix. (For example, in Figure 8, the area under each curve corresponds to  $\|\Delta W_Q\|_F^2$ .)

**Proposition 1.** Let  $M \in \mathbb{R}^{m \times n}$  be a matrix, and let its singular value decomposition (SVD) be  $M = U\Sigma V^{\top}$ , where  $U \in \mathbb{R}^{m \times m}$  and  $V \in \mathbb{R}^{n \times n}$  are orthonormal matrices, and  $\Sigma \in \mathbb{R}^{m \times n}$  is a diagonal matrix with entries  $\sigma_i$  on the diagonal for  $i = 1, \ldots, \min(m, n)$ . Define F = MH for an orthonormal matrix  $H \in \mathbb{R}^{n \times n}$ . Then, we have:

$$\|\bm{M}\|_F^2 = \sum_{i=1}^{\min(m,n)} \sigma_i^2 = \|\bm{F}\|_F^2.$$

#### Proof.

(i) Identity  $\|\mathbf{M}\|_F^2 = \sum_{i=1}^{\min(m,n)} \sigma_i^2$ .

Given  $M = U\Sigma V^{\top}$ , due to the orthonormality of  $U, M^{\top}M$  reduces to:

$$M^{\top}M = (U\Sigma V^{\top})^{\top}(U\Sigma V^{\top}) = V\Sigma^{\top}U^{\top}U\Sigma V^{\top} = V\Sigma^{\top}\Sigma V^{\top}.$$
 (22)

Therefore, by the cyclic property of trace and the orthonormality of V ,  $\|M\|_F^2$  reduces to:

$$\|\boldsymbol{M}\|_F^2 = \operatorname{tr}(\boldsymbol{M}^\top \boldsymbol{M}) = \operatorname{tr}(\boldsymbol{V} \boldsymbol{\Sigma}^\top \boldsymbol{\Sigma} \boldsymbol{V}^\top) = \operatorname{tr}\left(\boldsymbol{\Sigma}^\top \boldsymbol{\Sigma} (\boldsymbol{V}^\top \boldsymbol{V})\right) = \operatorname{tr}(\boldsymbol{\Sigma}^\top \boldsymbol{\Sigma}) = \|\boldsymbol{\Sigma}\|_F^2. \tag{23}$$

Since  $\Sigma^{\top}\Sigma$  is diagonal with entries  $\sigma_i^2$  for  $i=1,\ldots,\min(m,n)$ :

$$\|M\|_F^2 = \|\Sigma\|_F^2 = \sum_{i=1}^{\min(m,n)} \sigma_i^2.$$
 (24)

(ii) *Identity*  $\|M\|_F^2 = \|F\|_F^2$ .

Given F = MH, due to the orthonormality of H:

$$\mathbf{F}^{\mathsf{T}}\mathbf{F} = (\mathbf{M}\mathbf{H})^{\mathsf{T}}(\mathbf{M}\mathbf{H}) = \mathbf{H}^{\mathsf{T}}\mathbf{M}^{\mathsf{T}}\mathbf{M}\mathbf{H}.$$
 (25)

By the cyclic property of trace,  $||F||_F^2$  reduces to:

$$\|\boldsymbol{F}\|_F^2 = \operatorname{tr}\left((\boldsymbol{M}^\top \boldsymbol{M})(\boldsymbol{H}^\top \boldsymbol{H})\right) = \operatorname{tr}(\boldsymbol{M}^\top \boldsymbol{M}) = \|\boldsymbol{M}\|_F^2. \tag{26}$$

We note that this equivalence also applies to the coefficients defined with  $F' = H'\Delta WH$ , with  $H' \in \mathbb{R}^{m \times m}$ , such that  $\|M\|_F^2 = \|F\|_F^2 = \|F'\|_F^2$ .

**Outlier Reconstruction Ability of WHT.** Due to its energy concentration ability discussed above, WHT can reconstruct quantization errors with large outliers during initialization, which is critical for final model performance. Table 6 presents the numerical values corresponding to Figure 3(a), showing the proportion of outlier coefficients captured by each adapter.

<span id="page-18-0"></span>Table 6: Percentage of outlier coefficients captured by each adapter under a parameter budget of P(r=64) in the 14<sup>th</sup>-layer Key projection of LLaMA-3.2-3B. Higher is better.

| Adapter Type | Query | Key   | Value | Out   | Gate  | Up   | Down  | Average |
|--------------|-------|-------|-------|-------|-------|------|-------|---------|
| DCA          | 6.62  | 12.50 | 11.68 | 6.31  | 4.62  | 4.34 | 4.53  | 7.23    |
| DHA          | 18.82 | 32.29 | 21.98 | 13.19 | 14.14 | 8.00 | 11.01 | 17.06   |
| WHA          | 20.49 | 33.60 | 23.30 | 14.00 | 15.20 | 8.72 | 11.53 | 18.12   |

