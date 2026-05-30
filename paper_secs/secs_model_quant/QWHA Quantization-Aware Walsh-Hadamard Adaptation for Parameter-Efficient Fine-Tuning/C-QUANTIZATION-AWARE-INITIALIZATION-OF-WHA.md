# C QUANTIZATION-AWARE INITIALIZATION OF WHA

#### <span id="page-19-0"></span>C.1 OBJECTIVE FUNCTION

We reduce the original optimization objective in Equation 5 to the form in Equation 6, following the approach of Frantar et al. (2023); Deng et al. (2025). Using the notations from Section 3.2, the reduction proceeds as follows:

$$\|\Delta W_Q X - F H^{-1} X\|_F^2 = \|(\Delta W_Q - F H^{-1}) X\|_F^2$$
(27)

$$=\operatorname{tr}((\Delta \boldsymbol{W}_{Q}-\boldsymbol{F}\boldsymbol{H}^{-1})\boldsymbol{X}\boldsymbol{X}^{\top}(\Delta \boldsymbol{W}_{Q}-\boldsymbol{F}\boldsymbol{H}^{-1})^{\top}) \tag{28}$$

$$=\operatorname{tr}\left((\Delta \boldsymbol{W}_{Q}-\boldsymbol{F}\boldsymbol{H}^{-1})\boldsymbol{R}\boldsymbol{R}^{\top}(\Delta \boldsymbol{W}_{Q}-\boldsymbol{F}\boldsymbol{H}^{-1})^{\top}\right) \tag{29}$$

$$= \|(\Delta \boldsymbol{W}_Q - \boldsymbol{F} \boldsymbol{H}^{-1}) \boldsymbol{R}\|_F^2 \tag{30}$$

$$= \|\Delta \mathbf{W}_O \mathbf{R} - \mathbf{F} \mathbf{H}^{-1} \mathbf{R}\|_F^2, \tag{31}$$

where  $R = U\Sigma^{1/2} \in \mathbb{R}^{d_{\rm in} \times d_{\rm in}}$  is an invertible square root of the Hessian Gram matrix  $XX^{\top}$ . This term is obtained by applying the SVD  $XX^{\top} = U\Sigma U^{\top}$ , where  $\Sigma$  contains the eigenvalues on the diagonal and U is the matrix of orthonormal eigenvectors. Following Deng et al. (2025), we add a small regularization term  $\lambda = 0.0001 \cdot {\rm tr}(XX^{\top})/d_{\rm in}$  to the diagonal if R is not originally invertible. This reduction allows us to replace X with R, enabling efficient and effective calibration using multiple input data points. Rather than solving the optimization problem separately for each sample X, we can accumulate the contribution of activations via R and solve a single reduced problem.

#### <span id="page-19-1"></span>C.2 PARAMETER SELECTION STRATEGIES

We present the parameter selection patterns of each method discussed in Section 3.2. As shown in Figure 9, magnitude-based selection allocates parameters to a limited number of channels, while conventional methods such as SSH and LoCA incorporate random selection to avoid rank reduction. However, these approaches fail to reduce quantization error during initialization because the selected parameters are not optimal for error reconstruction. In contrast, AdaAlloc identifies the most important locations within each channel while preventing rank reduction through per-channel budgets, thereby providing the most effective initialization and fine-tuning.

<span id="page-19-2"></span>![](_page_19_Figure_11.jpeg)

Figure 9: Parameter selection patterns and two example zoomed-in results of each method in the 14<sup>th</sup>-layer Query projection of LLaMA-3.2-3B.

### <span id="page-20-0"></span>C.3 VALUE REFINEMENT

We present the layer output error after WHA initialization with and without value refinement in Table 7, as discussed in Figure 5 in Section 3.2. Without refinement of the selected coefficients in the initial dense solution matrix  $\Delta W_Q H$ , correlations among the columns are ignored, and the impact of sparsifying other columns cannot be considered, leading to suboptimal error reconstruction.

<span id="page-20-2"></span>Table 7: Layer output error ( $\ell_2$  norm, scaled by  $\times 10^3$ ) after initialization with and without value refinement in 4-bit quantized LLaMA-3.2-3B. Parameters are selected by AdaAlloc. 'None' denotes the error before initialization.

| Method         | Query | Key  | Value | Out  | Gate | Up    | Down  | Average |
|----------------|-------|------|-------|------|------|-------|-------|---------|
| None           | 13.84 | 0.54 | 28.08 | 4.66 | 1.88 | 25.76 | 21.36 | 7.21    |
| W/o Refinement | 11.39 | 0.62 | 26.21 | 4.39 | 2.11 | 24.33 | 20.97 | 7.06    |
| Refined        | 5.11  | 0.27 | 14.92 | 2.01 | 1.13 | 17.97 | 15.25 | 3.86    |

#### <span id="page-20-1"></span>C.4 CHANNEL-WISE PARAMETER SELECTION AND INITIALIZATION

We provide a detailed description of the formulation and solution of the sparse approximation problem underlying Algorithm 1, based on the notations in Section 3.2.

**Sparse Approximation Problem.** With a channel-wise breakdown of the objective in Equation 6, the goal is to initialize the *i*-th channel of the parameter matrix F, denoted  $F_{i,:}$ , given the perchannel parameter budget  $p_i$ . The objective is for  $F_{i,:}H^{-1}R$  to closely approximate the projected quantization error  $(\Delta W_Q)_{i,:}R$  in the  $\ell_2$  sense. As we constrain  $F_{i,:}$  to have exactly  $p_i$  non-zero elements, the term  $F_{i,:}H^{-1}R$  becomes a sparse linear combination of standard basis vectors:

$$F_{i,:}H^{-1}R = \sum_{k=0}^{p_i} F_{i,j_k}e^{(j_k)}H^{-1}R,$$
(32)

where  $e^{(j_k)}$  is the  $j_k$ -th standard basis vector. Since  $e^{(j_k)}H^{-1}R$  corresponds to the  $j_k$ -th channel of  $H^{-1}R$ , the problem reduces to selecting  $p_i$  rows from  $H^{-1}R$  that best approximate  $(\Delta W_Q)_{i,:}R$ .

Greedy Algorithm for Sparse Approximation. The problem generalizes to a standard sparse approximation problem: given a full set of basis vectors  $\beta = \{u_1, u_2, \dots, u_n\}$  with each  $u_i \in \mathbb{R}^d$ , we aim to select k vectors whose linear combination best approximates a target vector  $v \in \mathbb{R}^d$ . We represent the sparse coefficient vector as  $v = [x_1, x_2, \dots, x_k] \in \mathbb{R}^k$ , corresponding to the selected k basis vectors. Formally, we solve:

$$\min_{\boldsymbol{x}} \|\boldsymbol{v} - \boldsymbol{x}\boldsymbol{B}\|_2^2 \quad \text{subject to } \|\boldsymbol{x}\|_0 = k, \tag{33}$$

where  $B \in \mathbb{R}^{k \times d}$  is a submatrix formed from selected rows of the original basis. In our setting,  $B = H^{-1}R$ ,  $v = (\Delta W_O)_{i.:}R$ , and  $k = p_i$ .

Since this problem is NP-hard, we adopt a greedy approximation. We first compute  $x = vB^{-1} = \Delta W_Q H$ , which is in fact the non-sparse solution to the objective in Equation 6, and select the k entries of x with the largest magnitudes. Let the corresponding indices be  $i_1, i_2, \ldots, i_k$ , and define the selected basis  $B' = [u_{i_1}; \ldots; u_{i_k}]$ . We then solve a least-squares problem over the selected support:

$$\mathbf{x}^* = \operatorname{argmin}_{(x_{i_1}, \dots, x_{i_k})} \| [x_{i_1} \ x_{i_2} \ \dots \ x_{i_k}] \ \mathbf{B}' - \mathbf{v} \|_2^2 = \mathbf{v} \mathbf{B}'^{\top} (\mathbf{B}' \mathbf{B}'^{\top})^{-1}.$$
(34)

While this solution is numerically optimal when  $\boldsymbol{B}$  is orthogonal, we empirically demonstrate its effectiveness under general conditions. Combined with our AdaAlloc-based parameter allocation strategy, this initialization consistently yields high quantization error reconstruction ability while maintaining full rank capacity.

### D EXPERIMENTAL DETAILS AND ABLATIVE STUDY

#### <span id="page-21-0"></span>D.1 FINE-TUNING HYPERPARAMETERS

We follow the hyperparameter settings adapted from Gao et al. (2024b) and Deng et al. (2025). Training is performed using the AdamW optimizer (Loshchilov & Hutter, 2017). Table 8 reports the key settings, including minibatch size, weight decay, dropout ratio, learning rate scheduler, maximum sequence length, number of training epochs, warmup ratio, and the adapter scaling factor  $\alpha$ , which can be applied to adapters in Equation 2 through Equation 4. Due to an implementation detail in our codebase, the explicitly specified scaling factor is internally divided by the layer input dimension  $d_{\rm in}$ . As a consequence, the actual scaling applied during training is  $\alpha_{\rm effective} = \alpha_{\rm explicit}/d_{\rm in}$ . To match the effective scaling used in CLoQ (i.e.,  $\alpha_{\text{effective}} \approx 1.0$ ), we set the explicit scaling factor to  $\alpha_{\text{explicit}} = 4000$ , which is close to the typical input dimensions of our models (3072 for LLaMA-3.2-3B and 4096 for LLaMA-3.1-8B and Mistral-7B-v0.3). Under this implementation, the effective scaling becomes  $\alpha_{\text{effective}} = 4000/d_{\text{in}} \approx 1.0$ , ensuring consistent gradient scaling between low-rank. The learning rates for each combination of model, task, method, and bit-width are summarized in Table 9. We note that 128 sequences of length 2048, randomly sampled from the WikiText-2 (Merity et al., 2016) dataset, are used as a calibration set for quantization and adapter initialization, as these processes are robust to the choice of dataset (Frantar et al., 2023; Zhang et al., 2024; Deng et al., 2025). The total number of parameters for P(r = 64) is reported in Table 10, broken down by each projection, the layers containing these projections, and the entire model.

Table 8: Hyperparameter settings for Alpaca and GSM8K training

<span id="page-21-1"></span>

| Dataset             | A    | lpaca             | G    | SM8K              |
|---------------------|------|-------------------|------|-------------------|
| Method              | CLoQ | QWHA              | CLoQ | QWHA              |
| Optimizer           |      | Ada               | mW   |                   |
| Batch Size          |      | 6                 | 4    |                   |
| LR Scheduler        |      | cos               | sine |                   |
| Max Sequence Length |      | 5                 | 12   |                   |
| Epochs              |      | 3                 |      | 6                 |
| Warmup Ratio        |      | 0.1               | (    | 0.03              |
| Weight Decay        |      | 1                 |      | 0.1               |
| Dropout             | 0.1  | 0                 | 0.1  | 0                 |
| Scafe               | 1    | $4000$ / $d_{in}$ | 1    | $4000$ / $d_{in}$ |

<span id="page-21-2"></span>Table 9: Learning rate for each model and bit widths on Alpaca and GSM8K training.

| Model           | Bits | Al   | paca | GS   | M8K  |
|-----------------|------|------|------|------|------|
| 1,10401         | 2100 | CLoQ | QWHA | CLoQ | QWHA |
| Llama-3.1-8B    | 4    | 1e-5 | 3e-5 | 1e-4 | 5e-5 |
|                 | 3    | 1e-5 | 3e-5 | 1e-4 | 7e-5 |
|                 | 2    | 1e-5 | 2e-5 | 7e-5 | 5e-5 |
| Llama-3.2-3B    | 4    | 1e-4 | 3e-5 | 1e-4 | 7e-5 |
|                 | 3    | 1e-4 | 3e-5 | 1e-4 | 1e-4 |
|                 | 2    | 2e-4 | 5e-5 | 1e-4 | 2e-4 |
| Mistral-7B-v0.3 | 4    | 3e-5 | 5e-6 | 3e-5 | 2e-5 |
|                 | 3    | 2e-5 | 5e-6 | 3e-5 | 3e-5 |
|                 | 2    | 2e-5 | 7e-6 | 3e-5 | 3e-5 |

<span id="page-21-3"></span>Table 10: Total number of parameters at P(r = 64) for each projection, layer, and model.

| Model           | q_proj | k_proj | v_proj | o_proj | gate_proj | up_proj | down_proj | per-layer | per-model |
|-----------------|--------|--------|--------|--------|-----------|---------|-----------|-----------|-----------|
| LLaMA-3.1-8B    | 524288 | 327680 | 327680 | 524288 | 1179648   | 1179648 | 1179648   | 5242880   | 167772160 |
| LLaMA-3.2-3B    | 393216 | 262144 | 262144 | 393216 | 720896    | 720896  | 720896    | 3473408   | 97255424  |
| Mistral-7B-v0.3 | 524288 | 327680 | 327680 | 524288 | 1179648   | 1179648 | 1179648   | 5242880   | 167772160 |

