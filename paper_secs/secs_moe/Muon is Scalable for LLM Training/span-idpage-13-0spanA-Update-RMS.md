# <span id="page-13-0"></span>A Update RMS

#### **Proof of Lemma** 1

*Proof.* Without loss of generality, consider the orthogonal matrices  $U \in \mathbb{R}^{n \times n}$  and  $V \in \mathbb{R}^{m \times m}$  where  $n \geq m \geq r$ . We will show that for  $X = U_{[:,:r]}V_{[:r,:]}$  (the update of the Muon has the same format), the RMS value is  $\sqrt{r/mn}$ . From the definition of matrix multiplication:

$$X_{i,j} = \sum_{k=1}^{r} U_{i,k} V_{k,j}$$

The RMS can be expressed as:

$$\begin{aligned} \text{RMS}(X)^2 &= \frac{1}{mn} \sum_{i=1}^n \sum_{j=1}^m \sum_{k=1}^r U_{i,k}^2 V_{k,j}^2 \\ &= \frac{1}{mn} \sum_{k=1}^r \left( \sum_{i=1}^n U_{i,k}^2 \right) \left( \sum_{j=1}^m V_{k,j}^2 \right) \\ &= \frac{1}{mn} \sum_{k=1}^r 1 \\ &= \frac{r}{mn} \end{aligned}$$

Therefore, RMS(X) =  $\sqrt{r/mn}$ . For the common case where the matrices are full-rank, r=m, yielding RMS(X) =  $\sqrt{1/n}$ .

Consistent Update RMS Across Muon and AdamW As discussed in 2.2, we'd like to match the update RMS between Muon and AdamW optimizers. This is validated by experiments on small-scale models. We set Muon's Update RMS in the range of [0.05, 0.1, 0.2, 0.4, 0.8] and AdamW as baseline. We reported the loss and representative weight matrix RMS at 2k steps (about 2B tokens) in the Table 8. From the results, we find that 0.2 RMS and 0.4 RMS performed similarly and much better than other settings. These findings are consistent with our empirical observation that AdamW's update RMS is in the range of  $0.2 \sim 0.4$ . We opted to control the update RMS of Muon to 0.2.

Table 8: Muon Update RMS Experiments

<span id="page-13-2"></span>

| Optimizer          | AdamW   | 0.05 RMS* | 0.1 RMS | 0.2 RMS | 0.4 RMS | 0.8 RMS |
|--------------------|---------|-----------|---------|---------|---------|---------|
| LM training loss   | 3.512   | 3.355     | 3.239   | 3.198   | 3.199   | 3.386   |
| LM validation loss | 3.679   | 3.503     | 3.374   | 3.325   | 3.314   | 3.543   |
| AttnQ weight RMS   | 1.01e-2 | 5.74e-3   | 8.44e-3 | 1.57e-2 | 2.95e-2 | 7.23e-2 |
| Mlp weight RMS     | 1.25e-2 | 8.01e-3   | 1.27e-2 | 2.35e-2 | 4.51e-2 | 8.73e-2 |

<sup>\*</sup>Except the first column, all other candidates are using Muon with controlled RMS.

