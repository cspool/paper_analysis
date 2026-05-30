# <span id="page-16-0"></span>C.1 PROOF OF OBSERVATION [1](#page-3-0)

Here, we show how the self-attention operation can be written as the gradient of an energy function. In particular, we define a scalar function that depends on the keys, queries, values and additionally on an auxiliary vector that we refer to as the *source* . The source is the parameter with respect to which

we compute the gradient of the scalar function to obtain the self-attention operation. We need the source in order to write down the generating function of the moments of the distribution above. It is also the variable with respect to which we can Taylor-expand the generating function and extract the moments as the coefficients of the monomials of  $\zeta$  appearing in the Taylor series. Explicitly, we want to find a function  $F(q, k, v, \zeta)$  such that:

$$\sum_{a=1}^{N} \operatorname{softmax}(q \cdot k_a) v_a = \frac{\partial F}{\partial \zeta} \bigg|_{\zeta=0}.$$
 (15)

This terminology is inspired by work on energy-based models in machine learning Beal (2003); LeCun et al. (2006); Song & Kingma (2021). A summary of variables and indices is provided in appendix G

We first show how the energy function is given by the cumulant-generating function associated to the distribution given by attention scores. Taking inspiration from statistical mechanics, where an analogous cumulant-generating function defines the Helmholtz Free energy (Landau & Lifshitz, 1958), we dub our cumulant-generating function the *energy function for self-attention*.

Let us focus on the case with a single query. As noted above, we leverage the fact that the attention operation can be seen as the computation of the expectation value of the vectors v in the distribution set by the attention scores z:

$$z = \langle v \rangle = \sum_{a=1}^{N} P_a v_a = \frac{\sum_{a=1}^{N} e^{q \cdot k_a^T} v_a}{\sum_{i=1}^{N} e^{q \cdot k_i^T}}.$$
 (16)

The probability density is given by:

$$P_{a} = \frac{e^{q \cdot k_{a}^{T}}}{\sum_{i=1}^{N} e^{q \cdot k_{i}^{T}}}.$$
 (17)

Typically, the denominator or normalization factor is identified with the so-called partition function:

$$Z = \sum_{a=1}^{N} e^{q \cdot k_a^T}.$$
 (18)

We can now compute the first moment of the probability distribution given above by introducing a source,  $\zeta \in \mathbb{R}^d$ . In our case, with  $\zeta$ , we can extend the partition function to the function:

$$Z(\zeta) = \sum_{a=1}^{N} e^{q \cdot k_a^T + \zeta \cdot v_a^T}.$$
 (19)

Now, we can compute any moment of the distribution as the *n*-th Taylor coefficient of  $Z(\zeta)$   $\forall A_1, A_2, \dots \in \{1, \dots, d_h\}$ :

$$\langle v_{A_1} \cdots v_{A_n} \rangle = \frac{1}{Z} \frac{\partial^n Z(\zeta)}{\partial \zeta_{A_1} \cdots \partial \zeta_{A_n}} \bigg|_{\zeta = 0}.$$
 (20)

In other words, we can write  $Z(\zeta)$  as:

$$Z(\zeta) = Z\left(1 + \langle v \rangle \zeta + \frac{1}{2!} \langle v_{A_1} v_{A_2} \rangle \zeta_{A_1} \zeta_{A_2} + \cdots\right)$$
 (21)

Therefore, the first moment can be written as:

$$\langle v \rangle = \frac{1}{Z} \frac{\partial Z}{\partial \zeta} \bigg|_{\zeta = 0},\tag{22}$$

which can be written as the gradient of the log of  $Z(\zeta)$ :

$$\langle v \rangle = \frac{\partial}{\partial \zeta} \log Z(\zeta) \bigg|_{\zeta=0}.$$
 (23)

This quantity is the generating function, a.k.a. the free energy:

$$F = \log \sum_{a} \exp\left(q \cdot k_a^T + \zeta \cdot v_a^T\right). \tag{24}$$

To compute causal self-attention, we introduce N sources  $\zeta^i$  each  $\in \mathbb{R}^d$  and take

$$F_{tot} = \sum_{i=1}^{N} F_i = \sum_{i=1}^{N} \log \sum_{a=1}^{i} \exp(q_i \cdot k_a^T + \zeta_i \cdot v_a^T).$$
 (25)

The truncation of the inner sum up to index i is due to causal masking.

Now, in order to compute the *i*-th element of causal self-attention, we differentiate with respect to  $\zeta_i$  and set it to zero:

$$\left. \frac{\partial F_{tot}}{\partial \zeta_{i,A}} \right|_{\zeta_i = 0, \forall i} = \frac{\sum_{a=1}^i \exp(q^i \cdot k_a^T) v_{a,A}}{\sum_{a=1}^i \exp(q^i \cdot k_a^T)}.$$
 (26)

The generalization to the multi-head attention case is straightforward. In this case, there is one key, query and value per head. For  $n_h$  total heads, the generating function takes the form:

$$F_{tot} = \sum_{i=1}^{N} \sum_{h=1}^{n_h} F^{i,h}, \tag{27}$$

where

$$F_{i,h} = \log \sum_{a=1}^{i} \exp \left( q_{i,h} \cdot k_{h,a}^{T} + \zeta_{h,i} \cdot v_{h,a}^{T} \right). \tag{28}$$

The output projection weight is included in the definition of  $v_i$  here, meaning that

$$v_{b,A} = x_{b,\bar{B}}(W_O W_V)_{A,\bar{B}} \tag{29}$$

where  $W_O \in \mathbb{R}^{d_h} \times \mathbb{R}^{d_{emb}}$  denotes a head size slice of the output projection weight and  $\bar{B} \in \{1, \cdots, d_h\}$  spans the intra-head indices. In the index notation above, the head indices are barred whereas the embedding space indices are unbarred. We proceed focusing on the single-head case, as it makes the presentation simpler, and the multi-head generalization is immediate. Note that we demonstrate that our energy function approach also can account for safe softmax in Appendix F

#### <span id="page-18-0"></span>C.2 BAYESIAN INTERPRETATION

The fact that it is possible to derive the self-attention operation as the minimization of an energy function implies that it is possible to provide a Bayesian gloss on self-attention by identifying a likelihood function and showing that we can obtain the forward pass of the attention block from computing the maximum a posteriori estimate of this likelihood.

In particular, we propose the following for the log-likelihood function:

$$\Gamma(\zeta, z) = \sum_{i=1}^{N} \sum_{A=1}^{d} \left( z_{i,A} \zeta_{i,A} - F(\zeta, x) \right). \tag{30}$$

We denote by x the input to the self-attention block from which we obtain q, k, v from multiplying it by the weights  $W_Q, W_K, W_V$  respectively. Let us minimize the above with respect to  $\zeta$  and z simultaneously:

$$\frac{\partial \Gamma}{\partial \zeta_{i,A}} = 0, \frac{\partial \Gamma}{\partial z_{i,A}} = 0. \tag{31}$$

These conditions written explicitly read

$$\zeta_{i,A*} = 0, \quad z_{i,A*} = \frac{\partial F}{\partial \zeta_{i,A}}.$$
(32)

Plugging the first condition into the second leads to the attention forward pass:

$$z_{i*,A} = \frac{\sum_{a=1}^{i} e^{q_i \cdot k_a^T} v_{a,A}}{\sum_{b=1}^{i} e^{q_i \cdot k_b^T}}.$$
 (33)

In all, this means we can obtain the gradient w.r.t. from MAP estimation of the following likelihood:

$$z_{i*,A}, \zeta_{i*,A} = \operatorname{argmax}_{\zeta,z} e^{-\Gamma(\zeta,z)}.$$
 (34)

Moreover, such a procedure enables us to identify the energy-based model associated with the self-attention function.

### <span id="page-19-1"></span>C.3 MORE PERFORMANCES RESULTS WITH A LLAMA TRANSFORMER MODEL

To extend our work in section [6.4,](#page-8-1) and to demonstrate that Tree Attention can be successfully applied to a range of hardware setups, we also experiment with running Llama3.2-1B on a dual NVIDIA RTX 4090 setup. The two 4090s are connected via PCIe networking. Even in this case, we observe a significant 4x speedup (growing to 5x at longer sequence lengths) of Tree Attention over Ring Attention for autoregressive decoding.

<span id="page-19-2"></span>Table 2: Average Decoding Time (in seconds) comparisons with prefill stage using the 1B Llama 3.2 model with Tree Attention (ours) and Ring Attention (SOTA) across various sequence lengths for 4090s. Average results and standard error (±) are computed using 10 trial runs.

| Sequence Length | Tree Attention | Ring Attention | Speedup |
|-----------------|----------------|----------------|---------|
|                 | Time (s)       | Time (s)       |         |
| 8000            | 0.34 ±<br>0.05 | 1.38 ±<br>0.07 | ×4      |
| 16000           | 0.58 ±<br>0.07 | 2.77 ±<br>0.04 | ×5      |
| 20000           | 0.74 ±<br>0.01 | 3.47 ±<br>0.04 | ×5      |
| 32000           | 1.01 ±<br>0.02 | 5.45 ±<br>0.03 | ×5      |

