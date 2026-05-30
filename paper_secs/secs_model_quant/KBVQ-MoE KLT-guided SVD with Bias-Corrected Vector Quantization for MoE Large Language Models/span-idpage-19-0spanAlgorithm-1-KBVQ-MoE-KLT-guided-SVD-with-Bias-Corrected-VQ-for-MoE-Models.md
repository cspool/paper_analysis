# <span id="page-19-0"></span>Algorithm 1 KBVQ-MoE: KLT-guided SVD with Bias-Corrected VQ for MoE Models

**Input:** Expert weight matrices  $\{W^{(i)}\}_{i=1}^n$ , input activations X, codebook size K, sub-vector length

**Output:** Quantized weights  $\{\hat{W}^{(i)}\}_{i=1}^n$  with bias correction parameters (s,b)

```
Pre-Process: KLT-guided SVD (Redundancy Removal)
```

```
1: Compute input covariance: C_X = \frac{1}{R-1}X^{\top}X \in \mathbb{R}^{ic \times ic}
```

2: Eigen-decompose:  $C_X = U_{\text{KLT}}^{\top} \Lambda_{\text{KLT}} U_{\text{KLT}}$ 

3: Input-coherent basis:  $U_X = U_{\text{KLT}} \Lambda_{\text{KLT}}^{1/2}$ ;

4: **for** i = 1 **to** n **do** 

Project expert weights (right projection):  $\widetilde{W}^{(i)} = W^{(i)}U_X$ 

6: end for

7: Stack all experts (row-wise):  $\bar{W} = [\widetilde{W}^{(1)}; \cdots; \widetilde{W}^{(n)}] \in \mathbb{R}^{(n*oc) \times ic}$ 

 $\triangleright V \in \mathbb{R}^{(oc) \times n * oc}, U \in \mathbb{R}^{ic \times ic}$ 8: Product SVD:  $\overline{W} = (U\Sigma V^{\top})^T$ 

9: Select top-k:  $U_k = U_{[:,1:k]}$ ,  $V_k = V_{[:,1:k]}$ ,  $\Sigma_k = \Sigma_{1:k,1:k}$ 

10: Partition  $V_k$  by experts:  $V_k = [\Sigma_k V_k^{(1)}; \dots; \Sigma_k V_k^{(n)}]$ , each  $V_k^{(i)} \in \mathbb{R}^{oc \times k}$ 

11: **for** i = 1 **to** n **do** 

Define shared output loader:  $U_{\text{share}} \leftarrow U_X^{-1} U_k$ 12:

 $\triangleright ic \times k$ 

Shared part in projected basis:  $\widehat{W}_{\mathrm{share}}^{(i)} = (U_{\mathrm{share}}(V_k^{(i)})^T)^T$ 13:  $\triangleright oc \times ic$ 

Map back to original input space:  $W_{\mathrm{share}}^{(i)} = \widehat{W}_{\mathrm{share}}^{(i)}$ 14:

Special (residual) part:  $W_{\text{quant}}^{(i)} = W^{(i)} - W_{\text{share}}^{(i)}$ 15:

16: **end for** 

### **Quantization: Vector Quantization of Special Part**

17: **for** i=1 to n **do**18: Split  $W_{\text{quant}}^{(i)}$  into sub-vectors  $\{z\}$ 19: Initialize codebook via K-means++

20: Train VQ codebook  $C = \{c_1, \ldots, c_K\}$  by k-means

21: **for** each sub-vector z **do** 

Assign index:  $q = \arg\min_{j} ||z - c_{j}||^{2}$ 22:

23: Replace:  $z_q = c_q$ 

24:

Form quantized special part:  $W_{\text{quant,VO}}^{(i)}$ 25:

26: **end for** 

#### **Post-Process: Bias Correction**

Define quantized weights:  $\hat{W}^{(i)} = W_{\text{share}}^{(i)} + W_{\text{quant,VQ}}^{(i)}$ 

Estimate per-channel statistics from calibration data: 29:

 $\mu_{\nu}$ ,  $\sigma_{\nu}$  for original outputs  $y = W^{(i)}x$ 

 $\mu_{\hat{y}}, \sigma_{\hat{y}}$  for quantized outputs  $\hat{y} = \hat{W}^{(i)}x$ 

Compute correction parameters:

$$s_j = \frac{\sigma_{y_j}}{\sigma_{\hat{y}_j}} - 1, \quad b_j = \mu_{y_j} - (1 + s_j)\mu_{\hat{y}_j}$$

Corrected output:  $y_{corr} = (1+s) \odot (\hat{W}^{(i)}x) + b$ 31:

**32:** end for

30:

33: **return**  $U_{\text{share}}$ ,  $V_k$ , C, (s,b)

