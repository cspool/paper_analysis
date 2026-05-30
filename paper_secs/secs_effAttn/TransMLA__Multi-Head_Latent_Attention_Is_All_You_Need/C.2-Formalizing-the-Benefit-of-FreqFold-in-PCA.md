# C.2 Formalizing the Benefit of FreqFold in PCA

The example above illustrates that FreqFold causes a re-grouping and concatenation of data segments prior to PCA. The benefit of this concatenation is explained by the following proposition. It states that performing PCA jointly on these concatenated segments (as FreqFold enables) is more effective at preserving variance (and thus minimizing loss) than the alternative of performing separate PCAs on the original, smaller segments and then notionally combining their outcomes.

Consider one such FreqFold merge: suppose M original RoPE frequency indices  $l_1, \ldots, l_M$  are deemed equivalent by FreqFold. Without FreqFold, each  $l_p$  would correspond to a dataset  $X_p$  (e.g., N samples of 2g-dimensional key segments). With FreqFold, these M datasets are concatenated into a single larger dataset  $X_{merged} = [X_1, X_2, \dots, X_M]$ , and PCA is applied to  $X_{merged}$ .

<span id="page-21-0"></span>**Proposition 2.** Let M distinct groups of key segments  $X_1, X_2, \ldots, X_M$  be identified. Each  $X_p \in$  $\mathbb{R}^{N \times d'}$  (where  $p \in \{1, \dots, M\}$ ) consists of N samples of d'-dimensional vectors. Assume data in each  $X_p$  is mean-centered. Let  $S_p = \frac{1}{N-1} X_p^T X_p \in \mathbb{R}^{d' \times d'}$  be its covariance matrix. FreqFold causes these M groups to be merged for a single PCA operation.

Define  $V_1 = \sum_{p=1}^M \lambda_{p,1}$ , where  $\lambda_{p,1}$  is the largest eigenvalue of  $S_p$ . This  $V_1$  represents the sum of variances if each of the M original groups  $X_p$  were individually reduced to its single most dominant

Let  $Z = [X_1, X_2, \dots, X_M] \in \mathbb{R}^{N \times (M \cdot d')}$  be the dataset formed by concatenating the features (columns) of these M groups. Let  $S_{concat} = \frac{1}{N-1}Z^TZ \in \mathbb{R}^{(M \cdot d') \times (M \cdot d')}$  be its covariance matrix. Define  $V_2 = \sum_{j=1}^M \mu_j$ , where  $\mu_1 \ge \mu_2 \ge \ldots \ge \mu_M$  are the M largest eigenvalues of  $S_{concat}$ . This  $V_2$  represents the variance captured if the concatenated data Z is reduced to M dimensions using

Then, the variance captured by the joint PCA on the FreqFold-merged data  $(V_2)$  is greater than or equal to the sum of variances from optimally reducing each original group to one dimension  $(V_1)$ :

$$V_2 > V_1$$

This proposition explains that FreqFold's strategy of enabling PCA over larger, concatenated segments (formed by merging data from RoPE frequencies deemed similar) is mathematically favored for variance preservation compared to separate, more fragmented PCAs.

#### C.3 Proof of Proposition 2

The objective is to prove that  $V_2 \ge V_1$ , using the notation from Proposition 2. The proof strategy is to construct a specific M-dimensional subspace for the concatenated data Z. We show that the variance captured by projecting Z onto this particular subspace equals  $V_1$ . Since the PCA procedure yielding  $V_2$  finds the optimal M-dimensional subspace maximizing captured variance,  $V_2$  must be at least  $V_1$ .

Let  $\lambda_{p,1}$  be the largest eigenvalue of  $S_p$  (covariance of  $X_p$ ), and  $w_{p,1} \in \mathbb{R}^{d'}$  be its corresponding eigenvector. So,  $S_p w_{p,1} = \lambda_{p,1} w_{p,1}$  and  $w_{p,1}^T w_{p,1} = 1$ . The variance  $\lambda_{p,1} = w_{p,1}^T S_p w_{p,1}$ .  $V_1 = \sum_{p=1}^{M} \lambda_{p,1}.$ 

For the concatenated data  $Z, V_2 = \sum_{j=1}^M \mu_j$ . By Ky Fan's theorem for matrix eigenvalues:  $V_2 = \max_{\substack{U \in \mathbb{R}^{(M-d') \times M} \\ U^T U = I_M}} \mathrm{Tr}(U^T S_{concat} U)$ 

$$V_2 = \max_{\substack{U \in \mathbb{R}^{(M \cdot d') \times M} \\ U^T U = I_M}} \operatorname{Tr}(U^T S_{concat} U)$$

where U's columns form an orthonormal basis for an M-dimensional subspace of  $\mathbb{R}^{M \cdot d'}$ .

Construct  $U^* = [\boldsymbol{u}_1^*, \dots, \boldsymbol{u}_M^*] \in \mathbb{R}^{(M \cdot d') \times M}$ . For  $p \in \{1, \dots, M\}$ , define  $\boldsymbol{u}_p^* \in \mathbb{R}^{M \cdot d'}$ :

$$\bm{u}_p^* = \begin{pmatrix} \bm{0}_{d'\times 1} \ \vdots \ \bm{w}_{p,1} & \text{(as the $p$-th block of size $d'$)} \ \vdots \ \bm{0}_{d'\times 1} \end{pmatrix}$$

The set  $\{u_1^*, \dots, u_M^*\}$  is orthonormal. The variance retained by projecting Z onto the subspace of  $U^*$  is:

$$\operatorname{Tr}((U^*)^T S_{concat} U^*) = \sum_{p=1}^{M} (\boldsymbol{u}_p^*)^T S_{concat} \boldsymbol{u}_p^*$$

Let  $S_{qr}$  be the (q,r)-th block of  $S_{concat}$ , where  $S_{qr} = \frac{1}{N-1}X_q^TX_r$ . Note  $S_{pp} = S_p$ . Each term  $(\boldsymbol{u}_p^*)^TS_{concat}\boldsymbol{u}_p^* = \boldsymbol{w}_{p,1}^TS_{pp}\boldsymbol{w}_{p,1} = \boldsymbol{w}_{p,1}^TS_p\boldsymbol{w}_{p,1} = \lambda_{p,1}$ . So,  $\mathrm{Tr}((U^*)^TS_{concat}U^*) = \sum_{p=1}^M \lambda_{p,1} = V_1$ . Since  $V_2$  is the maximum possible variance:

$$V_2 \ge \operatorname{Tr}((U^*)^T S_{concat} U^*) = V_1$$

Thus,  $V_2 \geq V_1$ . This proves Proposition 2.

