# <span id="page-13-1"></span>Algorithm 2 LOER (Zhang et al., 2024a)

**Require:** Pretrained weight W, target rank k, quantization function  $q(\cdot)$ , dequantization function  $dq(\cdot)$ , calibration dataset  $\mathbb{X} = \{ \boldsymbol{x}_i \in \mathbb{R}^m | i = 1, \dots, N \}$ 

- 1: Initialize vector  $s \leftarrow 0$
- 2: for sample x in X do

Calibration

 $\boldsymbol{s} \leftarrow \boldsymbol{s} + \operatorname{abs}(\boldsymbol{x})$ 

> Accumulate activation magnitude on each dimension

- 4: end for
- 5:  $S \leftarrow \frac{1}{N} \operatorname{diag}(s)$

 $\triangleright$  Construct a diagonal matrix S

- 6:  $W_q \leftarrow q(W)$
- 7:  $\mathbf{W} \leftarrow \operatorname{dq}(\mathbf{W}_q)$

> SVD on the scaled weight error

8:  $U, \Sigma, V^{T} \leftarrow \text{SVD}(S(W - \widetilde{W}))$ 9:  $A_k \leftarrow S^{-1}U_{:,:k}, B_k \leftarrow \Sigma_{:k,:k}V_{:k,:}^{T}$ 

 $\triangleright$  Rank-k approximation with un-scaling

#### <span id="page-13-2"></span>A.2 PROOF OF THEOREM 2

Here we present the full proof of QERA-approx. QERA-approx is an approximated solution to Problem 2 based on Assumption 1, which is suitable to initialize the low-rank terms in fine-tuning for lower computation complexity.

### **Proof of Theorem 2**

*Proof.* We continue at Equation (13). Since  $\mathbb{E}_{x \sim \mathbb{X}}$  is the expectation with respect to the input space, we move the expectation inside the summation of RHS of Equation (13).

$$\mathbb{E}_{\boldsymbol{y} \sim \mathbb{Y}}\{||\widetilde{\boldsymbol{y}} - \boldsymbol{y}||_{2}^{2}\} = \sum_{i=1}^{m} \sum_{j=1}^{m} \mathbb{E}_{\boldsymbol{x} \sim \mathbb{X}}\{x_{i}x_{j}\}\boldsymbol{p}_{i}\boldsymbol{p}_{j}^{T}$$
(25)

<span id="page-13-4"></span><span id="page-13-3"></span>Under Assumption 1,  $\mathbb{E}_{x \sim \mathbb{X}}\{x_i x_j\} = 0$  for  $i \neq j$ , the RHS of Equation (25) simplifies to:

$$\mathbb{E}_{\boldsymbol{y} \sim \mathbb{Y}}\{||\widetilde{\boldsymbol{y}} - \boldsymbol{y}||_{2}^{2}\} = \sum_{i=1}^{m} \mathbb{E}_{\boldsymbol{x} \sim \mathbb{X}}\{x_{i}^{2}\}\boldsymbol{p}_{i}\boldsymbol{p}_{i}^{T}$$
(26)

We can define diagonal matrix  $S = \operatorname{diag}(\sqrt{\mathbb{E}_{\boldsymbol{x} \sim \mathbb{X}}\{x_1^2\}}, \sqrt{\mathbb{E}_{\boldsymbol{x} \sim \mathbb{X}}\{x_2^2\}}, \dots, \sqrt{\mathbb{E}_{\boldsymbol{x} \sim \mathbb{X}}\{x_m^2\}})$  and rewrite the RHS of Equation (26) as:

$$\mathbb{E}_{\boldsymbol{y} \sim \mathbb{Y}}\{||\widetilde{\boldsymbol{y}} - \boldsymbol{y}||_{2}^{2}\} = \operatorname{Tr}(\boldsymbol{S}\boldsymbol{P}\boldsymbol{P}^{T}\boldsymbol{S}^{T}) = ||\boldsymbol{S}\boldsymbol{P}||_{F}^{2}$$
(27)

where  $Tr(\cdot)$  denotes the trace of a matrix.

Therefore, the objective of Problem 2 (Equation (8)) is equivalent to:

$$\arg\min_{\boldsymbol{C}_{k}} \mathbb{E}_{\boldsymbol{y} \sim \mathbb{Y}} \{ ||\widetilde{\boldsymbol{y}} - \boldsymbol{y}||_{2}^{2} \} = \arg\min_{\boldsymbol{C}_{k}} ||\boldsymbol{S}\boldsymbol{P}||_{F}^{2}$$

$$= \arg\min_{\boldsymbol{C}_{k}} ||\boldsymbol{S}(\widetilde{\boldsymbol{W}} + \boldsymbol{C}_{k} - \boldsymbol{W})||_{F}^{2}$$
(28)

If we assign  $Q = S(W - \widetilde{W})$  and  $Q_k = SC_k$ , the objective becomes:

$$\underset{\boldsymbol{Q}}{\operatorname{arg\,min}} ||\boldsymbol{Q}_k - \boldsymbol{Q}||_F^2 \tag{29}$$

Note that the invertible matrix S in  $Q_k$  does not change the rank of the matrix  $C_k$ . According to the Eckart-Young-Mirsky theorem, the optimal rank k approximation to Q is the truncated SVD of Q:

$$Q_k = U_{:,:k} \Sigma_{:k,:k} V_{:k,:}^T$$
(30)

where  $\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^T = \mathrm{SVD}(\boldsymbol{Q}) = \mathrm{SVD}\left(\boldsymbol{S}(\boldsymbol{W}-\widetilde{\boldsymbol{W}})\right)$ .

Finally, we get the optimal solution to the low-rank term  $C_k$ :

$$C_k = S^{-1}Q_k = S^{-1}U_{:,:k}\Sigma_{:k,:k}V_{:k,:}^T$$
(31)

### <span id="page-14-0"></span>A.3 CONNECTION AND DIFFERENCE BETWEEN CALDERA AND QERA

CALDERA (Saha et al., 2024) is the concurrent work close to QERA. Here we elaborate the connection and difference between CALDERA and QERA, and highlight the contributions of QERA.

CALDERA focuses on a different problem setup. Specifically, CALDERA focuses on the following problem:

$$\min_{\widetilde{\boldsymbol{W}}, \boldsymbol{A}_{k,n}, \boldsymbol{B}_{k,n}} ||\boldsymbol{X}\boldsymbol{W} - \boldsymbol{X}(\widetilde{\boldsymbol{W}} + \boldsymbol{A}_{q,k}\boldsymbol{B}_{q,k})||_F^2$$
(32)

where  $X \in \mathbb{R}^{b \times m}$  denotes a batch of calibration samples, and  $\widetilde{W}$ ,  $A_{k,q}$ , and  $B_{k,q}$  are all quantized variables to optimize. Note that this problem setup is different from QERA (Equation (9)):

$$\underset{\boldsymbol{C}_{k}}{\arg\min} \, \mathbb{E}_{\boldsymbol{x} \sim \mathbb{X}} \{ ||\boldsymbol{x}(\widetilde{\boldsymbol{W}} + \boldsymbol{C}_{k}) - \boldsymbol{x} \boldsymbol{W}||_{2}^{2} \}$$
(33)

<span id="page-14-1"></span>where only the low-rank high-precision  $C_k := A_k B_k$  is the variable to optimize, and the quantized weight  $\widetilde{W}$  is predefined given a quantization method.

Table 5: Notation Table for the Equivalence Derivation

| Notation                                              | Description                              | Comments            |
|-------------------------------------------------------|------------------------------------------|---------------------|
| $\overline{b}$                                        | Number of calibration samples (vectors)  |                     |
| m                                                     | Layer input feature size                 |                     |
| n                                                     | Layer output feature size                |                     |
| $\boldsymbol{X}$                                      | Calibration set                          | Shape: $b \times m$ |
| $\boldsymbol{x}$                                      | A sample in the calibration set          | Shape: $1 \times m$ |
| W                                                     | Original full-precision layer weights    | Shape: $m \times n$ |
| $\boldsymbol{Y}$                                      | Layer output matrix corresponding to $X$ | Shape: $b \times n$ |
| $\boldsymbol{y}$                                      | Layer output vector corresponding to $x$ | Shape: $1 \times n$ |
| k                                                     | Rank of the low-rank approximation       |                     |
| $\boldsymbol{C}_k$                                    | Approximated rank-k weight               | Shape: $m \times n$ |
| $\boldsymbol{U}, \boldsymbol{\Sigma}, \boldsymbol{V}$ | SVD decomposition of $X$                 | -                   |
| $\mathrm{SVD}_k(\cdot)$                               | Truncated rank-k SVD                     |                     |

We find that CALDERA's Lemma 4.2 is equivalent to Theorem 1 in QERA. Note that the proof of QERA-exact is different from Caldera's Lemma 4.2, though the final closed-form solution is

equivalent. Here we additionally show the derivation of the equivalence between QERA-exact and Caldera's Lemma 4.2 using the notation table in Table 5. For convenience, we remove the quantized weight term  $\widetilde{\boldsymbol{W}}$  from QERA (Problem 2 in Equation (9)), which does not change the proof. Now the problem becomes finding the optimal low-rank approximation of the weight matrix,  $\boldsymbol{C}_k$  that minimizes the layer output error.

First we note that the objective of QERA, Equation (9), is equivalent to CALDERA's Eq(5) scaled by a constant n:

QERA: 
$$\min_{C_k} E_{\mathbf{x}} \{ ||\mathbf{x}(C_k - W)||_2^2 \}$$
  
CALDERA:  $\min_{C_k} ||\mathbf{X}(C_k - W)||_F^2$  (34)

<span id="page-15-4"></span>Then we show that Theorem 1 (QERA-exact) is equal to Caldera's Lemma 4.2.

<span id="page-15-2"></span>QERA-exact : 
$$C_k = (R_{\mathbb{X}\mathbb{X}}^{\frac{1}{2}})^{-1} \cdot \text{SVD}_k(R_{\mathbb{X}\mathbb{X}}^{\frac{1}{2}}W)$$
 (35)

<span id="page-15-3"></span>CALDERA: 
$$C'_k = V \Sigma \cdot SVD_k(U^T Y)$$
 (36)

We firstly show that  $(\mathbf{R}_{\mathbb{XX}}^{\frac{1}{2}})^{-1}$  in Equation (35) equals to  $\mathbf{V}\Sigma$  in Equation (36) scaled by a constant  $\sqrt{b}$ :

$$R_{XX} = \frac{1}{b}(X^{T}X) = V\Sigma U^{T}U\Sigma V^{T} = V\Sigma^{2}V^{T}$$

$$R_{XX}^{\frac{1}{2}} = \frac{1}{\sqrt{b}}\Sigma V^{T}$$

$$(R_{XX}^{\frac{1}{2}})^{-1} = \sqrt{b}V\Sigma^{-1}$$
(37)

Then we show that  $R_{XX}^{\frac{1}{2}}W$  in Equation (35) equals to  $U^TY$  in Equation (36) scaled by the constant  $\frac{1}{\sqrt{h}}$ :

$$U^{T}Y = U^{T}XW = U^{T}U\Sigma V^{T}W = \Sigma V^{T}W = \sqrt{b}R_{XX}^{\frac{1}{2}}W$$

$$R_{XX}^{\frac{1}{2}}W = \frac{1}{\sqrt{b}}U^{T}Y$$
(38)

Therefore  $C_k$  equals to  $C'_k$ , and the two solutions are equivalent. Despite of the equivalence, we shortlist the differences between CALDERA and our work:

- Different problem setup (Equation (34)).
- We simplify QERA-exact and derive QERA-approx, which is a computationally-efficient
  approximated solution. Specifically, QERA-approx is more suitable for parameter-efficient
  fine-tuning than QERA-exact/CALDERA. Moreover, QERA-approx overcomes the pitfalls\nin existing methods and explains why previous heuristic methods like LQER work.
- The optimization objective is similar (vector form *vs* matrix form), and the final closed-form solution is equivalent, but the proof of QERA-exact is different from CALDERA.

## <span id="page-15-0"></span>A.4 DETAILED EXPERIMENT SETUP

We mainly use PyTorch, Transformers, PEFT, and Accelerate to implement QERA. We use SciPy's implementation of blocked Schur algorithm (Deadman et al., 2012) to calculate the matrix square root, which runs on CPUs. The evaluation is performed with lm-evaluation-harness, Evaluate, and AlpacaEval 2.0 (Dubois et al., 2024).

