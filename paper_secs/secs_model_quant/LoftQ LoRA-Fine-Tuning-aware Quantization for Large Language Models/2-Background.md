# 2 Background

### 2.1 Transformer Models

A transformer model contains a sequence of layers, where each layer consists of two sub-layers: a multi-head self-attention (MHA) and a fully connected feed forward network (FFN) [\(Vaswani et al.,](#page-16-4) [2017\)](#page-16-4). Given the input *X* ∈ R*n*×*<sup>d</sup>* , where *n* is the sequence length and *d* is the hidden dimension of the model, MHA computes the *h* attention heads in parallel:

$$\label{eq:MHA} \mathbf{MHA}(X) = \mathbf{Concat}(\mathbf{head}_1,...,\mathbf{head}_h)W_o,$$
 where  $\mathbf{head}_i = \mathbf{Softmax}(XW_{q_i}(XW_{k_i})^\top/\sqrt{d_h})XW_{v_i}$  for  $i=1,...,h,$ 

where *Wq<sup>i</sup> ,Wk<sup>i</sup> ,Wv<sup>i</sup>* <sup>∈</sup> <sup>R</sup>*d*×*d<sup>h</sup>* are query, key, and value matrices, *<sup>W</sup><sup>o</sup>* <sup>∈</sup> <sup>R</sup>*d*×*<sup>d</sup>* is the output matrix, and *d<sup>h</sup>* = *d/h*. FFN comprises two linear transformations and an activation function, and is defined as FFN(*X*) = *σ*(*XWf*<sup>1</sup> + *b*1)*Wf*<sup>2</sup> + *b*2*,* where *Wf*<sup>1</sup> <sup>∈</sup> <sup>R</sup>*d*×*d<sup>m</sup>* , *<sup>W</sup>f*<sup>2</sup> ∈ R*dm*×*<sup>d</sup>* , and *σ*(·) is the activation function. A residual connection is used and followed by layer normalization.

### 2.2 Quantization

Quantization. Given a high-precision number, e.g., such as 32-bit floating point number, *X* HP ∈ R, *N*-bit quantization encodes it to an integer *X* INT ∈ {0*,*1*,...,*2 *<sup>N</sup>* − 1}. This process can be expressed as

<span id="page-3-0"></span>
$$X^{\text{INT}} = \text{round}\left((2^N - 1)F\left(X^{\text{HP}}\right)\right),\tag{1}$$

where *F*(·): R 7→ [0*,*1] is a normalization function. Uniform quantization assumes *F*(*X*) = (*X* − *X*min)*/*(*X*max − *X*min). [Dettmers et al.](#page-14-0) [\(2023\)](#page-14-0) proposes 4-bit NormalFloat Quantization (NF4). It assumes *X* ∼ N (0*, σ*<sup>2</sup> ) and hence *F*(*X*) = Φ(*X/σ*), where Φ(·) is the cumulative distribution function of the standard normal distribution.

Dequantization. A lookup table T , where

$$\mathcal{T}[i] = F^{-1}\left(\frac{i}{2^N - 1}\right), i = 0, 1, ..., 2^N - 1,$$
(2)

is used to decode the integer *X* INT to its simulated high-precision counterpart *X* <sup>D</sup> ∈ R. Therefore, the dequantization can be expressed as

<span id="page-3-2"></span><span id="page-3-1"></span>
$$X^{D} = \mathcal{T}[X^{\text{INT}}]. \tag{3}$$

Simulated Quantization for Matrices. While it is possible to perform multiplication directly between quantized representations, it is common to apply simulated quantization for matrices [\(Bai](#page-13-1) [et al.,](#page-13-1) [2020;](#page-13-1) [Shen et al.,](#page-15-0) [2020\)](#page-15-0). There, quantized weight matrices are stored as encoded integers in memory, and are temporarily dequantized to simulated high-precision matrices by the lookup table when engaged in multiplication operations. In simulated quantization, it is only necessary to analyze the map from a high-precision matrix to a simulated high-precision matrix. We denote this end-to-end process by *<sup>q</sup><sup>N</sup>* (·): <sup>R</sup>*m*×*<sup>n</sup>* 7→ R *m*×*n N* , where R*<sup>N</sup>* : {T [*i*] ∈ R|0 ≤ *i <* 2 *<sup>N</sup>* }.

### <span id="page-4-0"></span>2.3 Low-Rank Adaptation

LoRA (Hu et al., 2021) updates two small weight matrices A and B that are attached to a frozen pre-trained weight matrix W. Hence, a linear transformation, Y = XW, is reformulated as

<span id="page-4-1"></span>
$$Y = XW + XAB^{\top},\tag{4}$$

where  $X \in \mathbb{R}^{n \times d_1}$ ,  $W \in \mathbb{R}^{d_1 \times d_2}$ ,  $A \in \mathbb{R}^{d_1 \times r}$ ,  $B \in \mathbb{R}^{d_2 \times r}$ , and  $r \ll \min\{d_1, d_2\}$ . Initially,

$$A \sim \mathcal{N}(0, \sigma^2), B = 0, \tag{5}$$

so as to align to the pre-trained weights. During the fine-tuning, W is fixed while A and B are updated by some SGD-type optimization method.

It is worth noting that if low-rank adapters A and B are attached to a quantized backbone  $Q = q_N(W)$  and are initialized by (5), the starting weight  $Q + AB^{\top}$  is no longer equal to the pretrained weight W due to the discrepancy introduced by the quantization.

### 3 Method

We propose **Lo**RA-Fine-Tuning-aware **Q**uantization (LoftQ), a quantization framework for LLMs. It alternatively applies quantization and low-rank approximation to approximate original pretrained weights. This quantization framework provides a promising initialization for LoRA fine-tuning, which alleviates the quantization discrepancy in QLoRA and improves generalization in downstream tasks significantly.

#### 3.1 LoRA-Aware Quantization

We use an N-bit quantized weight  $Q \in \mathbb{R}_N^{d_1 \times d_2}$  and low-rank approximations  $A \in \mathbb{R}^{d_1 \times r}$ ,  $B \in \mathbb{R}^{d_2 \times r}$  to approximate the original high-precision pre-trained weight  $W \in \mathbb{R}^{d_1 \times d_2}$  as the initialization of LoRA fine-tuning. Specifically, before fine-tuning, we initialize the network by minimizing the following objective:

<span id="page-4-2"></span>
$$\min_{Q,A,B} \left\| W - Q - AB^{\top} \right\|_{F},\tag{6}$$

where  $\|\cdot\|_F$  denotes the Frobenious norm. This objective in (6) takes LoRA fine-tuning into consideration by jointly optimizing the initial values of the quantized backbone Q and low-rank adapters A, B. Contrarily, practitioners typically convert the pre-trained weight W into a quantized weight Q outright, neglecting the subsequent LoRA fine-tuning process. This oversight leads to notable performance degradation in downstream tasks arising from the quantization discrepancy.

### 3.2 Alternating Optimization

We solve the minimization problem in (6) by alternating between quantization and singular value decomposition (SVD). To begin with, we set  $A_0$ , and  $B_0$  equal to 0.

**Quantization**. At the *t*-th step, we quantize the difference between the original pre-trained weight W and the low-rank approximation  $A_{t-1}B_{t-1}^{\top}$  from the last step to obtain the quantized weight  $Q_t$  by

$$Q_t = q_N(W - A_{t-1}B_{t-1}^{\mathsf{T}}), \tag{7}$$

where  $q_N(\cdot)$  maps a high-precision weight matrix to a quantized matrix.

We remark that our algorithm is compatible with different quantization functions  $q_N(\cdot)$ . We apply NF4 and the uniform quantization in Section 4 as examples. We also remark that  $Q_t$  is not an exact solution of the minimization in (6), given the fixed  $A_{t-1}B_{t-1}^{\top}$ , but it is an efficient approximation.

**SVD**. After obtaining the *t*-th quantized weight  $Q_t$ , SVD is applied to the residual of the quantization denoted by  $R_t = W - Q_t$  by

$$R_{t} = \sum_{i=1}^{d} \sigma_{t,i} u_{t,i} v_{t,i}^{\top}, \tag{8}$$

where  $d = \min\{d_1, d_2\}$ ,  $\sigma_{t,1} \ge \sigma_{t,2} \ge ... \ge \sigma_{t,d}$  are the singular values of  $R_t$ ,  $u_{t,i}$ 's and  $v_{t,i}$ 's are the associated left and right singular vectors of  $R_t$ . We then obtain a rank-r approximation of  $R_t$  by  $A_t B_t^{\mathsf{T}}$ , where

<span id="page-5-0"></span>
$$A_{t} = [\sqrt{\sigma_{t,1}} u_{t,1}, ..., \sqrt{\sigma_{t,r}} u_{t,r}],$$

$$B_{t} = [\sqrt{\sigma_{t,1}} v_{t,1}, ..., \sqrt{\sigma_{t,r}} v_{t,r}].$$
(9)

We summarize our method in Algorithm 1. It is worth noting that T = 1 is a special case where  $Q_1$  is the exact quantized weight obtained by QLoRA, and low-rank approximations  $A_1$ ,  $B_1$  are obtained by the SVD of the quantization residual  $W - Q_1$ . T = 1 is sufficient to mitigate the quantization discrepancy, and alternating optimization helps to find a closer initialization to the pre-trained weight W, which further improves the performance (see Section 3).

We remark that the computational cost of LoftQ is negligible because it is applied to individual weight matrices and therefore can be executed in parallel. We also remark one can apply LoftQ only once to a pre-trained model and reuse the initialization obtained by LoftQ for different downstream tasks.

#### 3.3 Applying to LoRA Fine-tuning

We store the  $Q_T \in \mathbb{R}_N^{d_1 \times d_2}$  obtained by LoftQ using an integer matrix M by (1) and a lookup table T by (2). We initialize the backbone with the integer matrix M and initialize the low-rank adapters with  $A_T, B_T$  obtained by LoftQ.

### <span id="page-6-1"></span>Algorithm 1 LoftQ

```
input Pre-trained weight W , target rank r, N-bit quantization function qN (·), alternating step T
```

- 1: Initialize *A*<sup>0</sup> ← 0*,B*<sup>0</sup> ← 0
- 2: for t = 1 to *T* do
- 3: Obtain quantized weight *Q<sup>t</sup>* ← *q<sup>N</sup>* (*W* − *At*−1*B* ⊤ *t*−1 )
- 4: Obtain low-rank approximation *A<sup>t</sup> ,B<sup>t</sup>* ← SVD(*W* − *Q<sup>t</sup>* ) by [\(9\)](#page-5-0)
- 5: end for

output *Q<sup>T</sup> ,A<sup>T</sup> ,B<sup>T</sup>*

During LoRA fine-tuning, we freeze the integer weight *M* and optimize the low-rank adapters with an efficient optimization algorithm, e.g., AdamW [\(Loshchilov and Hutter,](#page-15-3) [2017\)](#page-15-3). In forward propagation, the integer weight *M* is temporarily dequantized to the simulated high-precision weight *Q<sup>T</sup>* by its lookup table, as described in [\(3\)](#page-3-2). In back propagation, gradients and optimizer state are only related to low-rank adapters *A,B*, which reduces considerable training cost.

