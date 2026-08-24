# <span id="page-12-1"></span>A Quantitative Analysis of Vanilla and Long CoT Representations.

After discovering the distinct distributions of vanilla and long CoTs within LLMs, we further conduct a quantitative analysis of their representations. Specifically, we employ matrix-based entropy (Giraldo et al., 2015; Wei et al., 2024) to investigate the information content across different layers for both CoTs. Given the representations of n samples  $Z \in \mathbb{R}^{n*d}$ , the matrix-based entropy  $S_{\alpha}(\mathbf{Z})$  quantifies the diversity of features within the representations, as defined by the following equations:

$$\mathbf{K} = \mathbf{Z}\mathbf{Z}^{\top},\tag{8}$$

$$S_{\alpha}(\mathbf{Z}) = \frac{1}{1-\alpha} \log \left( \sum_{i=1}^{r} \left( \frac{\lambda_i(\mathbf{K})}{\operatorname{tr}(\mathbf{K})} \right)^{\alpha} \right), \quad (9)$$

where **K** is the Gram matrix of the representation **Z**,  $\lambda_i(\mathbf{K})$  represents the nonnegative eigenvalues of **K**, and  $r = \operatorname{rank}(\mathbf{K}) \leq \min(d, n)$ . Following Skean et al. (2025), we set  $\alpha = 1$  for simplicity.

The matrix-based entropy metrics for vanilla and long CoTs representations across different layers in Qwen2.5-7B-Instruct and Llama3.1-8B-Instruct are illustrated in Figure 10. We observe that the matrix-based entropy of long CoT is consistently higher than that of vanilla CoT, indicating that long CoT contains more diverse and less redundant features within the latent space. Additionally, we find that the entropy in the middle layers of the model is higher than in the final layer in both CoTs. This suggests that the middle layers are better at extracting diverse and complex features (Wang et al., 2025b), exhibiting powerful capabilities in reasoning tasks (El-Nouby et al., 2024; Fan et al., 2024).

<span id="page-12-5"></span>

| Domain                        | Math    | Physics | Chemistry | Biology |
|-------------------------------|---------|---------|-----------|---------|
| Average Tokens of Vanilla CoT | 400.98  | 365.45  | 356.59    | 347.73  |
| Average Tokens of Long CoT    | 2628.46 | 2094.35 | 1832.86   | 1607.29 |

Table 4: Statistics of the vanilla and long CoT examples.

