# Attn优化和Acc设计

## KV Cache

**Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context**

**一般的self-Attn或Prefill阶段的Attn计算：**

X的每个列向量是1个token的embedding。

KQVproj是将Wproj和X进行MM，Wproj[D, D]的D-dim和X[D, N]的**D-dim乘累加**得到KQV[D, N]。

Attn分为：KQ计算得到S[Nq, Nk]，S进行softmax计算得到S‘[Nq, Nk]，SV计算得到C[D, Nk]。

KQ计算是KT[Nk, D]的D-dim和Q[D, Nq]和**D-dim乘累加**得到S[Nk, Nq]。

S[Nk, Nq]的每个行向量进行softmax计算，即**Nk-dim上乘累加**得到S‘[Nk, Nq]。

SV计算是V[D, Nk]的Nk-dim和S‘[Nk, Nq]的**Nk-dim上乘累加**得到C[D, Nq]。

KQ计算、softmax计算和SV计算的数据依赖正交，并行方式之间的依赖？

KV-Cache

。。。

## LLM动态块稀疏（top-K？）

ref：PISA: Supporting Dynamic Block Sparse Attention in Long-Context LLMs via Programmable and Decoupled PIM System

。。。

## LLM PIM（量化）

P3-LLM: An Integrated NPU-PIM Accelerator for LLM Inference Using Hybrid Numerical Formats

LLM推理

> **[图片提取文字 (image.png)]:**
> As depicted in Fig. 1(a), mainstream LLMs have a series of decoder layers in addition to an input embedding table and
> 
> A. Architecture of Large Language Models (LLMs)
> 
> an output language modeling (LM) head. During the prefilling stage of inference, the LLM receives a user prompt containing  $N_{\rm T}$  tokens, and converts it to an input matrix through an embedding table of size  $N_{\rm VOC} \times H$ , where  $N_{\rm VOC}$  is the vocabulary size and H is the hidden dimension size. The input matrix is processed by  $L\times$  decoder layers, followed by the LM head at the end to produce the first output token. During the decoding stage that LLM takes this output token as input
> 
> the decoding stage, the LLM takes this output token as input, and performs the same operation as in the prefilling stage to generate new tokens in an auto-regressive manner.
> 
> The decoder layer serves as the fundamental component in
> 
> The decoder layer serves as the fundamental component in LLMs, consisting of a self-attention module and a multi-layer perceptron (MLP). The self-attention module begins with three linear layers  $(W_Q, W_K, W_V)$  to generate query, key, and value
> 
> vectors, respectively. In recent LLM architectures [34], [69], [74],  $W_Q$  and  $W_K$  are usually followed by rotary position embedding (RoPE) [68], which encodes positional information into the query and key vectors through matrix rotation. The generated key and value vectors are also cached in off-chip memory for computation reuse during future decoding
> 
> generated key and value vectors are also cached in offchip memory for computation reuse during future decoding iterations, and are therefore referred to as the KV-cache. Then, each query and key-value vector is split into  $N_{\rm A}$  and  $N_{\rm KV}$ heads, respectively, where  $N_{\rm A}$  is the number of attention heads
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 1: Illustration of LLM architecture.
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%201.png)

> **[图片提取文字 (image.png)]:**
> calculate the attention-scores (P). The attention-scores are then multiplied with the value vectors  $(P \cdot V)$ , and the results are passed through a linear output-projection layer  $(W_O)$  to produce the attention output states. The MLP module contains three linear projection layers  $(W_{\text{gate}}, W_{\text{up}}, W_{\text{down}})$  to produce the MLP output states.
> 
> Considering the KV-cache needs to be stored for every token, its capacity can become significant for long-context
> 
> and  $N_{KV}$  is the number of key-value heads. For every attention
> 
> head, the query vectors are multiplied with the transposed
> 
> key vectors  $(Q \cdot K^{T})$ , followed by a softmax function to
> 
> token, its capacity can become significant for long-context scenarios [4]. To mitigate this storage overhead, recent LLMs have adopted the GQA mechanism. As shown in Fig. 1(b), conventional multi-head attention has the number of attention heads equal to the number of key-value heads, i.e.,  $N_{\rm A}=N_{\rm KV}$ . On the other hand, GQA partitions the  $N_{\rm A}$  attention heads into  $G=N_{\rm A}/N_{\rm KV}$  groups (two in this example), and different groups share the same key-value vectors, effectively reducing the KV-cache capacity by  $G\times$ .
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%202.png)

PIM LLM

> **[图片提取文字 (image.png)]:**
> higher internal bandwidth tailored for memory-bound operations [25], [39], [42], [43], [45], [63], [66]. Fig. 2 illustrates the architectures of two commercially available PIM devices: Samsung's HBM-PIM [43] and SK Hynix's Accelerator-in-Memory (AiM) [42]. The left part of Fig. 2 shows a PIM
> 
> channel consisting of 16 banks organized into 4 bank groups. One PIM compute unit (PCU) is placed near each DRAM bank to perform efficient GEMV operations by leveraging the abundant bank-level parallelism. Depending on area constraints, two banks may share the same PCU to amortize
> 
> the area overhead [43]. During LLM decoding, the DRAM bank transfers weights / KV-cache data to the PCU in 256-bit granularity (i.e.,  $16\times16$ -bit operands). Meanwhile, the input vector is sent from the host to either the PCU register file in HBM-PIM or the global buffer in AiM.
> 
> As shown in the right part of Fig. 2, HBM-PIM and AiM have different implementations of the PCU microarchitecture.
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%203.png)

> **[图片提取文字 (image.png)]:**
> In HBM-PIM, the PCU contains a 16-way single-instructionmultiple-data (SIMD) MAC unit, and allows to exploit input reuse during GEMV by multiplying the same input element with 16 weights. On the other hand, the PCU of AiM uses the brain floating-point (BF16) format for data representation, and adopts an adder-tree-based design to exploit output reuse during GEMV. Despite their simplicity, the PCUs incur considerable area overhead, ranging from 20% to 27% of the DRAM die area [42], [43], primarily because the DRAM process has roughly 10× lower transistor density and fewer metal layers for routing compared to CMOS at the same technology node [12]. This overhead significantly constrains the achievable compute throughput of PIM, restricting its performance benefits mainly to single-batch inference and
> 
> multi-head attention that do not exhibit data reuse.
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%204.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 2: Illustration of PIM architectures for LLM decoding acceleration.
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%205.png)

LLM量化

> **[图片提取文字 (image.png)]:**
> Quantization is a widely used technique for cost-effective LLM acceleration. Consider a group of operands X and a
> 
> C. LLM Quantization
> 
> list of quantization values Q, the quantized operand XQ and dequantized operand X can be calculated as follows:
> 
>  $\Delta = \frac{|\mathbf{X}|_{\text{max}}}{Q_{\text{max}}}; \ \mathbf{X}_{\mathbf{Q}} = \text{Round}\left(\frac{\mathbf{X}}{\Delta}, \mathbf{Q}\right); \ \widetilde{\mathbf{X}} = \mathbf{X}_{\mathbf{Q}} \cdot \Delta, \ (1)$ where  $\Delta$  is the scaling factor, and Round (x, Y) is a function
> 
> that rounds a value x to the closest value in a set Y. This rounding process inevitably introduces error between the
> 
> original and quantized operands. Numerous techniques have been proposed to reduce quantization error, such as mixed-
> 
> precision quantization and custom numerical formats.
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%206.png)

> **[图片提取文字 (image.png)]:**
> weight-only [16], [46] and KV-cache-only [26], [49] quantization have demonstrated near-lossless accuracy at 4-bit precision. To further alleviate the computation and memory demands of LLMs, several studies have explored weightactivation quantization [2], [14], [47], [67]. Meanwhile, recent literature in the architecture community has explored custom numerical formats that can better adapt to the tensor distribution of LLMs [7], [15], [20], [27], [30], [41], [48], [57], [64], [65], [73]. In addition to research efforts, custom quantization formats have been widely adopted by industry. For instance, the flagship GPUs of NVIDIA and AMD support the 8-bit floating-point format with two variants: 4-bit exponent 3bit mantissa (FP8-E4M3) and 5-bit exponent 2-bit mantissa
> 
> (FP8-E5M2) [57].
> 
> In mixed-precision domain, SoTA algorithmic solutions on
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%207.png)

## BP分解和K分解

ref： Learning Fast Algorithms for Linear Transforms Using **Butterfly Factorizations**

### **DFT的BP分解**

**单位1的N次幂根**是$w_N=e^{\frac{2 \pi }{N }\cdot { i}}$。

DFT矩阵矩阵$F_N$的第$k$行第$n$列是**旋转因子**$w_N^{- kn}=e^{-\frac{2 \pi \cdot kn }{N }\cdot { i}}$，$F_N$的第$k$行向量和向量$x$的内积$\sum_0^N w_N^{- kn} \cdot x_n$将$x_0$~$x_N$中属于频率$k$的成分提取并融合。**Butterfly矩阵**$B^{(N)}$和**Permutation矩阵**$P^{(N)}$是$F_N \cdot x$递归计算的矩阵表达。

$B_N$是旋转矩阵，**仅三对角线有非零复数**，每行$n$包含两个复数，其一是1，其二是**旋转因子**$w_N^{-n}$。

$P_N$是重排矩阵，**不一定是三对角阵**，将向量X中奇数项和偶数项分开，便于进行DFT的递归计算。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 1: Butterfly matrix for N=16. From left to right: single copy of  $B_{16}$ , blocks of  $B_8$ , blocks of  $B_4$ ,
> 
> blocks of  $B_2$ .
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0%E5%BC%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E7%B1%BB%E4%BC%BCGPU%E5%A4%9Awarp%E5%B9%B6%E8%A1%8C%EF%BC%89/image%2080.png)

> **[图片提取文字 (image.png)]:**
> Case study: DFT The Discrete Fourier Transform (DFT) transforms a complex input vector  $x = [x_0, ..., x_{N-1}]$  into a complex output vector  $X = [X_0, ..., X_{N-1}]$  by expressing the input in the basis of the complex exponentials:
> 
> $$X_k = \sum_{n=0}^{N-1} x_n e^{-\frac{2\pi i}{N}kn}, \quad k = 0, \dots, N-1, N = 2^m.$$
> 
> Let  $\omega_N := e^{2\pi i/N}$  denote a primitive N-th root of unity. The DFT can be expressed as matrix multiplication by the DFT matrix  $F_N \in \mathbb{C}^{N \times N}$ , where  $(F_N)_{kn} = \omega_N^{-kn}$ . The DFT of size N can be reduced to two DFTs of size N/2 on the even indices and the odd indices:
> 
> $$F_N x = \begin{bmatrix} F_{N/2} x_{\text{even}} + \Omega_{N/2} F_{N/2} x_{\text{odd}} \\ F_{N/2} x_{\text{even}} - \Omega_{N/2} F_{N/2} x_{\text{odd}} \end{bmatrix},$$
> 
> where  $x_{\text{even}} = [x_0, x_2, \dots, x_{N-2}], x_{\text{odd}} = [x_1, x_3, \dots, x_{N-1}],$  and  $\Omega_{N/2}$  is the diagonal matrix with entries  $1, \omega_N^{-1}, \dots, \omega_N^{-(N/2-1)}$ . This recursive structure yields the efficient recursive Cooley-Tukey Fast Fourier Transform (FFT) algorithm. This computation can be written as a matrix factorization
> 
> $$F_N = \begin{bmatrix} I_{N/2} & \Omega_{N/2} \\ I_{N/2} & -\Omega_{N/2} \end{bmatrix} \begin{bmatrix} F_{N/2} & 0 \\ 0 & F_{N/2} \end{bmatrix} \begin{bmatrix} \text{Sort the even} \\ \text{and odd indices} \end{bmatrix},$$
> 
> where  $I_{N/2}$  is the identity matrix, and the last factor is the permutation matrix  $P_N$  that separates the even and odd indices (e.g., mapping [0,1,2,3] to [0,2,1,3]) (see Figure 2). Unrolling the recursion, we obtain:
> 
> $$F_{N} = B_{N} \begin{bmatrix} F_{N/2} & 0 \\ 0 & F_{N/2} \end{bmatrix} P_{N}$$
> 
> $$= B_{N} \begin{bmatrix} B_{N/2} & 0 \\ 0 & B_{N/2} \end{bmatrix} \begin{bmatrix} F_{N/4} & 0 & 0 & 0 \\ 0 & F_{N/4} & 0 & 0 \\ 0 & 0 & F_{N/4} & 0 \\ 0 & 0 & 0 & F_{N/4} \end{bmatrix} \begin{bmatrix} P_{N/2} & 0 \\ 0 & P_{N/2} \end{bmatrix} P_{N}$$
> 
> $$= \cdots$$
> 
> $$= \begin{pmatrix} B_{N} \dots \begin{bmatrix} B_{2} & \dots & 0 \\ \vdots & \ddots & \vdots \\ 0 & B_{N} & \vdots \end{bmatrix} \begin{pmatrix} \begin{bmatrix} P_{2} & \dots & 0 \\ \vdots & \ddots & \vdots \\ 0 & B_{N} & \vdots \end{bmatrix} \dots P_{N} \end{pmatrix}.$$
> 
> (1)
> 
> The product of all the  $B_{N/2^k}$  matrices on the left is called a butterfly matrix, and each factor  $B_{N/2^k}$  is a  $2 \times 2$  block matrix of diagonal matrices called a butterfly factor. Figure  $\square$  illustrates the sparsity pattern of the structured butterfly factors. One can also combine the product of permutation matrices on the right to obtain a single permutation called the bit-reversal permutation, which sorts the indices by the reverse of their binary representation (e.g.  $[0, \ldots, 7] \rightarrow [0, 4, 2, 6, 1, 5, 3, 7]$ ).
> 
> Other transforms have similar recursive structure but differ in the entries of  $B_{N/2^k}$ , and in the permutation. For example, the DCT involves separating the even and the odd indices, and then reversing the second half (e.g.,  $[0,1,2,3] \rightarrow [0,2,1,3] \rightarrow [0,2,3,1]$ ).
> 
> Appendix A provides some examples of how important transforms, such as the DFT, DCT, Hadamard, and convolutions, can factor as similar products of sparse matrices.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0%E5%BC%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E7%B1%BB%E4%BC%BCGPU%E5%A4%9Awarp%E5%B9%B6%E8%A1%8C%EF%BC%89/image%2081.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 2: Three binary choices for constructing the permutation used at every step of the recursive process. One of 8 possible permutations can be constructed by multiplying a subset of these matrices in the presented order.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0%E5%BC%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E7%B1%BB%E4%BC%BCGPU%E5%A4%9Awarp%E5%B9%B6%E8%A1%8C%EF%BC%89/image%2082.png)

> **[图片提取文字 (image.png)]:**
> A butterfly parametrization Let  $x = [x_0, \dots, x_{N-1}]$  be an input vector [T] Let T be a linear transform of size N with matrix representation  $T_N \in \mathbb{F}^{N \times N}$ , where  $\mathbb{F} \in \{\mathbb{R}, \mathbb{C}\}$ . A general recursive structure is to separate the input vector into two halves by some permutation, apply the transform on each half, and combine the result in a linear manner by scaling by an diagonal matrix and adding the results. Written as a matrix factorization:
> 
> $$T_N = \begin{bmatrix} D_1 & D_2 \\ D_3 & D_4 \end{bmatrix} \begin{bmatrix} T_{N/2} & 0_{N/2 \times N/2} \\ 0_{N/2 \times N/2} & T_{N/2} \end{bmatrix} P_N,$$
> 
> where  $P_N$  is some permutation matrix and  $D_1, \ldots, D_4 \in \mathbb{F}^{N/2}$  are diagonal matrices. Inspired by the factors of the FFT, we call the matrix  $\begin{bmatrix} D_1 & D_2 \\ D_3 & D_4 \end{bmatrix}$  a butterfly factor, denoted by  $B_N$ . Unrolling the recursion as in equation (I) gives the factorization  $T_N = B^{(N)}P^{(N)}$ , where  $B^{(N)}$  is a butterfly matrix and  $P^{(N)}$  is a permutation that can be written as the product of  $\log_2(N)$  simpler block permutations. We also consider composing this module, hence learn either
> 
> $$T_N = B^{(N)} P^{(N)} \qquad T_N = B_2^{(N)} P_2^{(N)} B_1^{(N)} P_1^{(N)},$$
>  (2)
> 
> which we term the BP and the BPBP parametrization respectively. One dimensional convolutions (i.e. circulant matrices) are notably captured by BPBP, since they can be computed via an FFT, a component-wise product, then an inverse FFT (see Appendix A).
> 
> In the case of the FFT, as in Section 3.1 the entries of the butterfly factors are called twiddle factors, and the combined permutation  $P^{(N)}$  is called the bit-reversal permutation.
> 
> Learning a recursive permutation The butterfly blocks in the BP or BPBP parametrization have a fixed sparsity pattern and their parameters can be directly optimized. However, the transforms we are interested in capturing frequently require different permutations as part of the "divide" step, which form a set of discrete objects that we must consider. We will restrict to learning over permutations that have a simple structure often encountered in these algorithms: we assume that the distribution factors into  $\log_2 N$  steps following the  $\log_2 N$  recursive layers. At each step in the recursion, the permutation  $P_{N/2^k}$  is allowed to either keep the first half and second half intact or separate the even and the odd indices (e.g.,  $[0, 1, 2, 3] \rightarrow [0, 2, 1, 3]$ ). Then, it can choose to reverse the first half (e.g.,  $[0, 1] \rightarrow [1, 0]$ ) and can choose to reverse the second half (e.g.,  $[2, 3] \rightarrow [3, 2]$ ). Thus at each step, there are 3 binary choices and hence 8 possible permutations. These are illustrated in Figure 2 where  $P_N^a$  denotes the permutation matrix on N elements that separates the even and odd elements,  $P_N^b$  denotes the permutation matrix that reverses the first half, and  $P_N^c$  denotes the permutation matrix that reverses the first half, and  $P_N^c$  denotes the permutation matrix that reverses the second half.
> 
> Instead of searching over  $8^{\log_2 N}$  discrete permutations, we parameterize the permutation  $P^{(N)}$  as a categorical distribution of these  $8^{\log_2 N}$  permutations. The permutation  $P_{N/2^k}$  at step k is thus chosen as a convex combination of the 8 possible choices:
> 
> $$P_{N/2^k} = p_{cba} P_{N/2^k}^c P_{N/2^k}^b P_{N/2^k}^a + p_{cb} P_{N/2^k}^c P_{N/2^k}^b + \dots$$
> 
> This can be learned by representing this probability distribution  $\{p_{cba}, p_{cb}, \dots\}$  for example via logits and the softmax.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0%E5%BC%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E7%B1%BB%E4%BC%BCGPU%E5%A4%9Awarp%E5%B9%B6%E8%A1%8C%EF%BC%89/image%2083.png)

> **[图片提取文字 (image.png)]:**
> We make the further simplification that the probabilities  $p_{cba}$  factor into the three components; conceptually, that the choices of choosing  $P_{N/2^k}^c$ ,  $P_{N/2^k}^b$ ,  $P_{N/2^k}^a$  to be part of the product are independent of each other. This results in the representation  $P_{N/2^k} = (p_s P_{N/2^k}^s + (1 - p_s)I).$ s=c,b,aThus we learn the permutation  $P_{N/2^k}$  via equation (3) by optimizing over 3 logits  $\ell_a, \ell_b, \ell_c$  and setting  $p_s = \sigma(\ell_s)$ , where  $\sigma$  is the sigmoid function. To encourage the distribution over permutations to be peaked, one can add entropy regularization [15] or semantic loss 46. However, we found that these tricks are not necessary. For example, the learned transforms in Section 4.1 typically put weight at least 0.99 on a permutation.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0%E5%BC%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E7%B1%BB%E4%BC%BCGPU%E5%A4%9Awarp%E5%B9%B6%E8%A1%8C%EF%BC%89/image%2084.png)

### **线性变换的矩阵分解**（BP分解的例子）

似乎假设P已经完成，侧重一系列B矩阵的计算。

> **[图片提取文字 (image.png)]:**
> ## A Matrix Factorizations of Linear Transforms
> 
> Table 3 summarizes the transforms considered in Section 4.1. In general, they transform a (real or complex) vector  $x = [x_0, \ldots, x_{N-1}]$  into another (real or complex) vector  $X = [X_0, \ldots, X_{N-1}]$  by expressing the input signal in terms of another set of basis.
> 
> ## A.1 Discrete Cosine Transform (DCT) Matrix
> 
> The DCT (type II) of a vector  $x \in \mathbb{R}^N$  is defined as
> 
> $$X_k = \sum_{n=0}^{N-1} x_n \cos\left[\frac{\pi}{N}\left(n + \frac{1}{2}\right)k\right], \qquad k = 0, \dots, N-1.$$
> 
> As described in Makhoul [30], the DCT of x can be written in terms of the FFT of order N. To do this, we permute x into v by separating the even and odd indices and reversing the odd indices (e.g.  $[0,1,2,3] \rightarrow [0,2,3,1]$ ), taking the FFT of v to obtain V, and multiplying each  $V_k$  ( $k=0,\ldots,N-1$ ) by  $2e^{-\frac{i\pi k}{2N}}$  and taking the real part to get  $X_k$ .
> 
> Written in terms of matrix factorization:
> 
> $$DCT_N = \Re \operatorname{diag}\left(2e^{-\frac{i\pi k}{2N}}\right)F_N P',$$
> 
> where  $\Re$  takes the real part and P' is a permutation matrix (the permutation done at the beginning of the DCT). Recall that  $F_N$  has the form
> 
> $$F_N = B_N \begin{bmatrix} B_{N/2} & 0 \\ 0 & B_{N/2} \end{bmatrix} \dots \begin{vmatrix} B_2 & \dots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \dots & B_2 \end{vmatrix} P,$$
> 
> where P is the bit-reversal permutation matrix.  $\frac{\operatorname{diag}\left(2e^{-\frac{i\pi k}{2N}}\right)}{\operatorname{butterfly}}$  can be combined with  $B_N$  to form another butterfly factor  $B'_N$ . Thus the DCT has this factorization:
> 
> $$DCT_N = \Re B'_N \begin{bmatrix} B_{N/2} & 0 \\ 0 & B_{N/2} \end{bmatrix} \dots \begin{vmatrix} B_2 & \dots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \dots & B_2 \end{vmatrix} PP'.$$
> 
> This is a (BP)<sup>2</sup> factorization (with the additional final step of computing the real part) with the left BP performing the FFT and final scaling, the right butterfly matrix as the identity, and the right permutation matrix as the permutation at the beginning of the DCT.
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%208.png)

> **[图片提取文字 (image.png)]:**
> ## Discrete Sine Transform (DST) Matrix
> 
> The DST (type II) of a vector  $x \in \mathbb{R}^N$  is defined as
> 
> $$X_k = \sum_{n=0}^{N-1} x_n \sin\left[\frac{\pi}{N}\left(n + \frac{1}{2}\right)(k+1)\right], \qquad k = 0, \dots, N-1.$$
> 
> Just as with the DCT, we express the DST of x in terms of the FFT of order N. First, we permute x into v by separating the even and odd indices and reversing the odd indices (e.g.  $[0,1,2,3] \rightarrow [0,2,3,1]$ ), then negate those elements in the second half of v. We then multiply each  $v_k$  with  $e^{-\frac{i2\pi k}{N}}$ . Next, we take the FFT of v to obtain V. Finally multiply each  $V_k$   $(k=0,\ldots,N-1)$  by  $2ie^{-\frac{i\pi k}{2N}}$  and take the real part to get  $X_k$ .
> 
> Written in terms of matrix factorization:
> 
> $$DST_N = \Re \operatorname{diag}\left(2ie^{-\frac{i\pi k}{2N}}\right) F_N DP',$$
> 
> where  $\Re$  takes the real part, D is the matrix  $\begin{vmatrix} I_{N/2} \\ 0 \end{vmatrix} = 0$  diag  $\left(e^{-\frac{i2\pi k}{N}}\right)$  and P' is a permutation matrix (the permutation done at the beginning of the DST). Recall that  $F_N$  has the form
> 
> $$F_N = B_N \begin{bmatrix} B_{N/2} & 0 \\ 0 & B_{N/2} \end{bmatrix} \dots \begin{bmatrix} B_2 & \dots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \dots & B_2 \end{bmatrix} P,$$
> 
> where P is the bit-reversal permutation matrix. We may combine  $\operatorname{diag}\left(2ie^{-\frac{i\pi k}{2N}}\right)$  with  $B_N$  to obtain a new butterfly factor, which we call  $B'_N$ . Thus the DST has this factorization:
> 
> $$DST_N = \Re B_N' \begin{bmatrix} B_{N/2} & 0 \\ 0 & B_{N/2} \end{bmatrix} \dots \begin{bmatrix} B_2 & \dots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \dots & B_2 \end{bmatrix} PDP'.$$
> 
> Since D is a diagonal matrix and P is the bit-reversal permutation matrix, we have that PD = D'P where D' is the diagonal matrix obtained by permuting the diagonals of D. Hence
> 
> $$DST_N = \Re B_N' \begin{bmatrix} B_{N/2} & 0 \\ 0 & B_{N/2} \end{bmatrix} \dots \begin{bmatrix} B_2 & \dots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \dots & B_2 \end{bmatrix} D'PP'.$$
> 
> The diagonal matrix 
> $$D'$$
>  can be combined with the butterfly factor  $\begin{bmatrix} B_2 & \dots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \dots & B_2 \end{bmatrix}$  to yield another butterfly
> 
> factor of the same form. Therefore:
> 
> $$DST_N = \Re B_N' \begin{bmatrix} B_{N/2} & 0 \\ 0 & B_{N/2} \end{bmatrix} \dots \begin{vmatrix} B_2' & \dots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \dots & B_2' \end{vmatrix} PP'.$$
> 
> Hence, this factorization of the DST is a (BP)<sup>2</sup> factorization (with the additional final step of computing the real part) with the left BP performing the FFT and final scaling, the right butterfly matrix as the identity, and the right permutation matrix as the permutation at the beginning of the DST.
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%209.png)

BP分解发明于DFT的矩阵$F_N$递归算法，可应用于线性变换的矩阵$T_N$乘法分解，通过**BP或BPBP形式构造并训练线性层T**。

**Butterfly因子**$B_{N/2^k}$是大小为$N/2^k$的**三对角矩阵**。

**Butterfly因子块矩阵$B_{N/2^k}$**是由$2^{k}$个$B_{N/2^k}$块**按对角线堆叠**成的大小为N的矩阵。

**Butterfly矩阵**$B^{(N)}$是$k$从$0$到$(log_2^N-1)$的因子矩阵$B_{N/2^k}$**累乘**，即$log_2^N$个$B_{N/2^k}$**矩阵的矩阵乘法。**

> **[图片提取文字 (image.png)]:**
> where each  $\mathbf{D}_i$  is a  $\frac{k}{2} \times \frac{k}{2}$  diagonal matrix. We restrict k to be a power of 2. **Definition 2.2.** A butterfly factor matrix of size n with block size k (denoted as  $\mathbf{B}_k^{(n)}$ ) is a block diagonal matrix of  $\frac{n}{k}$  (possibly different) butterfly factors of size k:
> 
> **Definition 2.1.** A butterfly factor of size  $k \geq 2$  (denoted as  $\mathbf{B}_k$ ) is a matrix of the form  $\mathbf{B}_k = \begin{bmatrix} \mathbf{D}_1 & \mathbf{D}_2 \\ \mathbf{D}_2 & \mathbf{D}_4 \end{bmatrix}$ 
> 
> $$\mathbf{B}_k^{(n)}=\operatorname{diag}\left(\left[\mathbf{B}_k\right]_1,\left[\mathbf{B}_k\right]_2,\ldots,\left[\mathbf{B}_k\right]_{\frac{n}{k}}\right)$$
> 
> **Definition 2.3.** A butterfly matrix of size n (denoted as  $\mathbf{B}^{(n)}$ ) is a matrix that can be expressed as a product of butterfly factor matrices:  $\mathbf{B}^{(n)} = \mathbf{B}_n^{(n)} \mathbf{B}_{\frac{n}{2}}^{(n)} \dots \mathbf{B}_{2}^{(n)}$ . Equivalently, we may define  $\mathbf{B}^{(n)}$  recursively as a matrix that can be expressed in the following form:
> 
> $$\mathbf{B}^{(n)} = \mathbf{B}_n^{(n)} \begin{bmatrix} [\mathbf{B}^{(\frac{n}{2})}]_1 & 0 \ 0 & [\mathbf{B}^{(\frac{n}{2})}]_2 \end{bmatrix}$$
> 
> (Note that  $[\mathbf{B}^{(\frac{n}{2})}]_1$  and  $[\mathbf{B}^{(\frac{n}{2})}]_2$  may be different.)
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0%E5%BC%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E7%B1%BB%E4%BC%BCGPU%E5%A4%9Awarp%E5%B9%B6%E8%A1%8C%EF%BC%89/image%2079.png)

**Permutation矩阵**$P^{(N)}$是$k$从$0$到$(log_2^N-1)$的$P_{N/2^k}$**矩阵累乘**，即$log_2^N$个$P_{N/2^k}$**矩阵的矩阵乘法。**

### **BP分解的扩展——K分解**

ref：**Kaleidoscope**: An Eﬃcient, Learnable Representation For All Structured Linear Maps

相比一般FFT/LT中重排矩阵P的“不规整”，将**P矩阵分解成一系列三对角矩阵**的形式，便于计算。

**Butterfly矩阵**$B^{(N)}$的大小是N*N，是$k$从$0$到$(log_2^N-1)$的因子块矩阵$B_{N/2^k}$**累乘**。

$B^{(N)}B^{'(N)*}$定义为BP分解形式的矩阵乘法得到的矩阵（类似FFN矩阵$F_N$），依然存在P。

**$(B^{(N)}B^{'(N)*})^w$**是w个矩阵$B^{(N)}B^{'(N)*}$作乘法得到的矩阵，相比矩阵$B^{(N)}B^{'(N)*}$有更大的秩。

$(B^{(N)}B^{'(N)*})^w_e$是K矩阵，允许$B^{(N)}$的大小扩展到eN，则$(B^{(N)}B^{'(N)*})^w$是大小为eN的矩阵，而$(B^{(N)}B^{'(N)*})^w_e$只取$(B^{(N)}B^{'(N)*})^w$的左上角（1/e）。

**K矩阵**的大小是N*N，**$e$让$B^{(eN)}$**分解K矩阵后的保留更多全局信息，**$w$让$(B^{(eN)}B^{'(eN)*})^w$**分解K矩阵后保留更多原有秩（关联信息）。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 1: Visualization of the fixed sparsity pattern of the building blocks in  $\mathcal{BB}^*$ , in the case n = 16. The red and blue dots represent all the possible locations of the nonzero entries.
> 
> **Definition 2.4** (Kaleidoscope hierarchy, kaleidoscope matrices).
> 
> - Define  $\mathcal{B}$  as the set of all matrices that can be expressed in the form  $\mathbf{B}^{(n)}$  (for some n).
> - Define  $\mathcal{BB}^*$  as the set of matrices  $\mathbf{M}$  of the form  $\mathbf{M} = \mathbf{M}_1 \mathbf{M}_2^*$  for some  $\mathbf{M}_1, \mathbf{M}_2 \in \mathcal{B}$ .
> - Define  $(\mathcal{BB}^*)^w$  as the set of matrices  $\mathbf{M}$  that can be expressed as  $\mathbf{M} = \mathbf{M}_w \dots \mathbf{M}_2 \mathbf{M}_1$ , with each  $\mathbf{M}_i \in \mathcal{BB}^*$   $(1 \leq i \leq w)$ . (The notation w represents width.)
> - Define  $(\mathcal{BB}^*)_e^w$  as the set of  $n \times n$  matrices  $\mathbf{M}$  that can be expressed as  $\mathbf{M} = \mathbf{SES}^T$  for some  $en \times en$  matrix  $\mathbf{E} \in (\mathcal{BB}^*)^w$ , where  $\mathbf{S} \in \mathbb{F}^{n \times en} = \begin{bmatrix} \mathbf{I}_n & 0 & \dots & 0 \end{bmatrix}$  (i.e.  $\mathbf{M}$  is the upper-left corner of  $\mathbf{E}$ ). (The notation e represents **expansion** relative to n.)
> - M is a kaleidoscope matrix, abbreviated as K-matrix, if  $M \in (\mathcal{BB}^*)_e^w$  for some w and e.
> 
> The kaleidoscope hierarchy, or  $(\mathcal{BB}^*)$  hierarchy, refers to the families of matrices  $(\mathcal{BB}^*)_e^1 \subseteq (\mathcal{BB}^*)_e^2 \subseteq \dots$ , for a fixed expansion factor e. Each butterfly matrix can represent the identity matrix, so  $(\mathcal{BB}^*)_e^w \subseteq (\mathcal{BB}^*)_e^{w+1}$ . We show that the inclusion is proper in Appendix E. This hierarchy generalizes the  $\mathcal{BP}$  hierarchy proposed by Dao et al. (2019), as shown in Appendix J.
> 
> Efficiency in space and speed Each matrix in  $(\mathcal{BB}^*)_e^w$  is a product of 2w total butterfly matrices and transposes of butterfly matrices, each of which is in turn a product of  $\log(ne)$  factors with 2ne nonzeros (NNZ) each. Therefore, each matrix in  $(\mathcal{BB}^*)_e^w$  has  $4wne\log(ne)$  parameters and a matrix-vector multiplication algorithm of complexity  $O(wne\log ne)$  (by multiplying the vector with each sparse factor sequentially). We prove this more formally in Appendix E For the applications in Section w and w and w are small constants (up to 2), so those K-matrices have  $O(n\log n)$  parameters and runtime.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0%E5%BC%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E7%B1%BB%E4%BC%BCGPU%E5%A4%9Awarp%E5%B9%B6%E8%A1%8C%EF%BC%89/image%2085.png)

$n*n$**的矩阵$M$和向量$v$的乘积，s是运算逻辑门数，d是逻辑层次深度，则Mv运算时M可分解成$(B^{(O(s))}B^{'(O(s))*})^{O(d)}$，Mv的计算复杂度是$O(ds \cdot logs)$**：

$M(n*n)$可以被拆分成$(B^{(O(s))}B^{'(O(s))*})^{O(d)}$，即$2O(d)$个大小为$O(s) \cdot O(s)$的Butterfly矩阵$B^{(O(s))}$和其转置矩阵$B^{'(O(s))*}$累乘，每个Butterfly矩阵是$logO(s)$个大小为$O(s) \cdot O(s)$的因子块矩阵$B$累乘，每个因子块矩阵包含$2O(s)$个非零值$NNZ$。

对ProjKQV和FFN等线性层，**累加维度D决定逻辑门数s，逻辑层次深度d是常数（BP或BPBP分解）**，**N个D维tokens作Mv**计算，因此线性层(D * D)作BP分解计算的复杂度是**$O(ND \cdot logD)$**。

> **[图片提取文字 (image.png)]:**
> ## All low-depth structured matrices are in the kaleidoscope hierarchy
> 
> We now present our main theoretical result: the fact that general linear transformations, expressed as low-depth linear arithmetic circuits, are captured in the  $\mathcal{BB}^*$  hierarchy with low width. Arithmetic circuits are commonly used to formalize algebraic algorithmic complexity (Bürgisser et al.) 2013); we include a primer on this in Appendix M. The quantities of interest are the total number of gates in the circuit, representing the total number of steps required to perform the algorithm for a serial processor, and the depth, representing the minimum number of steps required for a parallel processor.
> 
> **Theorem 1.** Let  $\mathbf{M}$  be an  $n \times n$  matrix such that multiplication of  $\mathbf{M}$  times an arbitrary vector  $\mathbf{v}$  can be represented as a linear arithmetic circuit with s total gates and depth d. Then,  $\mathbf{M} \in (\mathcal{BB}^*)^{O(d)}_{O(\frac{s}{n})}$ .
> 
> The representation of such a matrix  $\mathbf{M}$  in the  $\mathcal{BB}^*$  hierarchy has  $O(ds \log s)$  parameters and yields a  $O(ds \log s)$  multiplication algorithm, compared to the O(s) parameters and runtime of the circuit representation. To the best of our knowledge, the most general classes of efficient matrices that have been studied ( $\overline{De}$  Sa et al.,  $\overline{2018}$ ) have depth d on the order of  $\log n$  or poly  $\log n$ . In these cases, the representation with K-matrices matches the best known bounds up to polylogarithmic factors.
> 
> The crux of the proof of Theorem I (shown in Appendix F) is the construction of an almost tight representation of any sparse matrix as a K-matrix (i.e. a product of butterfly matrices): specifically, we show that any  $n \times n$  sparse matrix with s nonzeros is in  $(\mathcal{BB}^*)_{O(1)}^{O(\lceil \frac{s}{n} \rceil)}$  (Theorem 3, Appendix I). We then leverage
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0%E5%BC%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E7%B1%BB%E4%BC%BCGPU%E5%A4%9Awarp%E5%B9%B6%E8%A1%8C%EF%BC%89/image%2086.png)

> **[图片提取文字 (image.png)]:**
> product width result of De Sa et al. (2018) referenced in Section 2.1) to complete the proof of Theorem 1 This intermediate result is also a novel characterization of sparse matrices. For a matrix with s NNZ, the kaleidoscope representation has  $O(s \log n)$  parameters and runtime, instead of the optimal O(s) parameters and runtime; so, we trade off an extra logarithmic factor in space and time for full differentiability (thanks to the fixed sparsity patterns in the representation). The intuition behind the result is as follows: a sparse matrix with s NNZ can be written as a sum of  $\lceil s/n \rceil$  matrices each with at most n NNZ. Any  $n \times n$  matrix with at most n NNZ, up to permuting the rows and columns, is a product of two butterfly matrices (Lemma  $\overline{1.1}$ ). Sorting networks (Knuth 1997) imply that permutation matrices are in  $(\mathcal{BB}^*)^{O(\log n)}$ , but we tighten the result to show that they are in fact in  $\mathcal{BB}^*$  (Theorem  $\mathbb{Z}$ , Appendix  $\mathbb{G}$ ). We thus obtain a kaleidoscope representation for each summand matrix with  $O(n \log n)$  parameters. By the addition closure property of the  $\mathcal{BB}^*$  hierarchy (Lemma [H.5]), each sparse matrix with s NNZ then has a kaleidoscope representation with
> 
>  $O(s \log n)$  parameters.
> 
> the expressivity result of products of sparse matrices to represent all arithmetic circuits (similar to the sparse
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0%E5%BC%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E7%B1%BB%E4%BC%BCGPU%E5%A4%9Awarp%E5%B9%B6%E8%A1%8C%EF%BC%89/image%2087.png)

> **[图片提取文字 (image.png)]:**
> ## Properties of the $\mathcal{BB}^*$ Hierarchy
> 
> Here, we justify why the definitions in Section 2.2 give rise to a hierarchy. We first make some basic observations about the parameterization.
> 
> **Observation E.1.** An  $n \times n$  matrix  $\mathbf{M} \in \mathcal{BB}^*$  has  $4n \log n$  parameters.
> 
> *Proof.* M can be expressed as a product of  $2 \log n$  butterfly factor matrices of size  $n \times n$ . Each of these factor matrices has 2 parameters per row, for a total of 2n parameters each. Hence, the total number of parameters is  $4n \log n$ .
> 
> **Observation E.2.** Let  $\mathbf{M}$  be an  $n \times n$  matrix in  $(\mathcal{BB}^*)_e^w$ . Then, given an arbitrary vector  $\mathbf{v}$  of length n, we can compute  $\mathbf{Mv}$  with  $O(wne\log(ne))$  field operations.
> 
> *Proof.* Since  $\mathbf{M} \in (\mathcal{BB}^*)_e^w$ , we can decompose it as  $\mathbf{SE}_1\mathbf{E}_2 \dots \mathbf{E}_w\mathbf{S}^T$ , where  $\mathbf{S}$  is as given in Definition 2.4, and each  $\mathbf{E}_i$  is an  $en \times en$  matrix in  $\mathcal{BB}^*$ . Therefore, to compute  $\mathbf{Mv}$ , we can use associativity of matrix multiplication to multiply the vector by one of these matrices at a time.
> 
> Since all of these factors are sparse, we use the naïve sparse matrix-vector multiplication algorithm (begin with a 0-vector and perform the corresponding multiplication and addition for each nonzero matrix entry). So (and thus  $S^T$ ) have n NNZ. Therefore, matrix-vector multiplication by S or  $S^T$  requires O(n) operations, which is dominated by the butterfly matrix-vector multiplication. Each  $E_i$  can be further decomposed into  $2\log(ne)$  matrices with at most 2ne non-zero entries each (by Observation E.1). Therefore, matrix vector multiplication by each  $E_i$  requires  $O(ne\log(ne))$ . Since there are w such  $E_i$ , we require a total of  $O(wne\log(ne))$  operations.
> 
> Now, we are ready to show that our definition of classes  $(\mathcal{BB}^*)_e^w$  forms a natural hierarchy. First, we must argue that all matrices are contained within the hierarchy.
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%2010.png)

> **[图片提取文字 (image.png)]:**
> ## Arithmetic Circuits in $\mathcal{BB}^*$ Hierarchy
> 
> In this appendix, we prove our main theoretical result, namely, our ability to capture general transformations, expressed as low-depth linear arithmetic circuits, in the  $\mathcal{BB}^*$  hierarchy. This result is recorded in Theorem 1
> 
> **Theorem 1.** Let  $\mathbf{M}$  be an  $n \times n$  matrix such that matrix-vector multiplication of  $\mathbf{M}$  times an arbitrary vector  $\mathbf{v}$  can be represented as a be a linear arithmetic circuit C comprised of s gates (including inputs) and having depth d. Then,  $\mathbf{M} \in (\mathcal{BB}^*)^{O(d)}_{O(\frac{s}{n})}$ .
> 
> To prove Theorem we make use of the following two theorems.
> 
> **Theorem 2.** Let **P** be an  $n \times n$  permutation matrix (with n a power of 2). Then  $\mathbf{P} \in \mathcal{BB}^*$ .
> 
> **Theorem 3.** Let **S** be an  $n \times n$  matrix of s NNZ. Then  $\mathbf{S} \in (\mathcal{BB}^*)_4^{4\lceil \frac{s}{n} \rceil}$ .
> 
> Theorem 2 is proven in Appendix G and Theorem 3 is proven in Appendix I
> 
> Proof of Theorem  $\square$ . We will represent C as a product of d matrices, each of size  $s' \times s'$ , where s' is the smallest power of 2 that is greater than or equal to s.
> 
> To introduce some notation, define  $w_1, \ldots w_d$  such that  $w_k$  represents the number of gates in the k'th layer of C (note that  $s = n + \sum_{k=1}^{d} w_k$ ). Also, define  $z_1, \ldots z_d$  such that  $z_1 = n$  and  $z_k = w_{k-1} + z_{k-1}$  ( $z_k$  is the number of gates that have already been used by the time we get to layer k).
> 
> Let  $g_i$  denote the *i*'th gate (and its output) of C ( $0 \le i < s$ ), defined such that:
> 
> $$g_i = \begin{cases} v_i & 0 \le i < n \\ \alpha_j g_{i_1} + \beta_i g_{i_2} & n \le i < s \end{cases}$$
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%2011.png)

## Butterfly Acc（BP-FFT+BSMM优化vanilla Attn）

**FNet: Mixing Tokens with Fourier Transforms**

**Adaptable Butterfly Accelerator for Attention-based NNs via Hardware and Algorithm Co-design**

### Butterfly Unit：BMM和FFT

使用BP分解方法计算BMM和FFT。

BU完成2个内积(BMV)，2个2元权重向量**复用相同2元输入分别作内积**：

2实数输入{in1, in2}，4实数权重{w1, w3, w2, w4}，2实数输出{in1*w1+in2*w3, in1*w2+in2*w4}，合计4次实数乘法和2次实数加法。

BU完成2个复数内积(FFT-BP)，2个2元复数权重**复用相同2元复数输入分别作内积**，2个2元权重向量分别是{1, $w_N^{- n}$}和{1, $w_N^{- (n+N/2)}$}，因为旋转因子$w_N^{- n}=e^{-\frac{2 \pi \cdot n }{N }\cdot { i}}$，有$w_N^{- (n+N/2)}=-w_N^{- n}$，只需要计算$w_N^{- n}$和2元输入的复数乘积，之后分别进行2个2元内积的加法：

2复数输入{in1-r, in1-i, in2-r, in2-i}，1复数权重$w_N^{- n}${w-r, w-i}，复数权重$w_N^{- n}${w-r, w-i}和复数输入{in2-r, in2-i}的复数乘积有4次实数乘法和2次实数加法，将{in1-r, in1-i}*1分别和上述乘积作加法和减法得到2复数输出{out1-r, out1-i, out2-r, out2-i}，总计4次实数乘法和2次实数加法和2次复数加法。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> (a) Hardware architecture of adaptable butterfly unit
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Figure 7. Microarchitecture and dataflow of the adaptable butterfly unit.
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%2012.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 2. The structure of a *Transformer*. Shortcut addition and layer normalization are omitted for simplicity.
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%2013.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 3: Improve transformer blocks using structured sparsity.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0%E5%BC%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E7%B1%BB%E4%BC%BCGPU%E5%A4%9Awarp%E5%B9%B6%E8%A1%8C%EF%BC%89/image%2053.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 5. Network structure of FABNet. (SC: shortcut addition, LN: layer normalization)
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%2014.png)

### Butterfly Acc：Butterfly Engine和Attn Engine

token的Batch并行是不同PE，每个PE只完成1个request的1个token的BMV。

那最开始的BMM-prompt呢？输入X的proj运算**使用MV完成**，因为BP分解的计算优化面向MV。

Prefill没有KV-Cache，Decode使用KV-Cache，计算都是MV运算，第二个并行维度是Batch，而不是N，因为不同request的token长度不一致，使用Batch更能利用硬件效率。

因此BP分解是对MV的计算优化。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 6. Hardware overview of the adaptable butterfly accelerator.
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%2015.png)

adaptive bs kernel是在BSMM中的BMM layer之间设计重排，避免Bank Conflict。

16*16的BM和16的V，x0和x8是row1，W的4个BM和向量X**从左向右作MV**，每个MV是1个stage。x是输入，**执行BP分解线性层的P矩阵部分，B矩阵部分则stage顺序相反**。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 8. Bank conflicts in column and row-major orders.
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%2016.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 9. Data layout and hardware design of S2P.
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%2017.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 10. An example of 16-input butterfly.
> 
> Data access
> 
> Starting index of each row in storage
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%2018.png)

overlapping策略

软硬件优化

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 11. Hardware design of *Index Coalescing* module.
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%2019.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 13. Different overlapping strategies.
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%2020.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Figure 12. Different address mapping strategies.
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%2021.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> ![](_page_0_Figure_2.jpeg)
> 
> Figure 15. Flow of the algorithm-hardware co-design process.
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%2022.png)

## BSMM/FFT MLX（量化的FFT+BMM优化vanilla Attn）

ref：MLX: Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures

**ref：FNet: Mixing Tokens with Fourier Transforms**

### Attn和BS-Attn的计算复杂度

FFT方法的复杂度写错了，应该是DNlogN。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 1: Tradeoffs among different implementations of transformer blocks.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0%E5%BC%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E7%B1%BB%E4%BC%BCGPU%E5%A4%9Awarp%E5%B9%B6%E8%A1%8C%EF%BC%89/image%2049.png)

FFN

方法1：FFN的递归计算，将矩阵FN进行BP分解成2logN个子矩阵，但访存太低效。

方法2：直接使用矩阵FN进行复数矩阵乘计算。

> **[图片提取文字 (image.png)]:**
> ## **Discrete Fourier Transform**
> 
> **Model** 
> 
> The Fourier Transform decomposes a function into
> 
> its constituent frequencies. Given a sequence  $\{x_n\}$ 
> 
> (DFT) is defined by the formula:
> 
> putation time to  $\mathcal{O}(N \log N)$ .
> 
> with  $n \in [0, N-1]$ , the discrete Fourier Transform
> 
>  $X_k = \sum_{n=0}^{N-1} x_n e^{-\frac{2\pi i}{N}nk}, \quad 0 \le k \le N-1.$  (1)
> 
> For each k, the DFT generates a new representation
> 
>  $X_k$  as a sum of all of the original input tokens  $x_n$ ,
> 
> with so-called "twiddle factors". There are two pri-
> 
> mary approaches to computing the DFT: the Fast
> 
> Fourier Transform (FFT) and matrix multiplication.
> 
> The standard FFT algorithm is the Cooley–Tukey
> 
> algorithm (Cooley and Tukey, 1965; Frigo and
> 
> Johnson, 2005), which recursively re-expresses the
> 
> DFT of a sequence of length  $N = N_1 N_2$  in terms
> 
> of  $N_1$  smaller DFTs of sizes  $N_2$  to reduce the com-
> 
> **DFT** matrix to the input sequence. The DFT matrix,
> 
> An alternative approach is to simply apply the
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%2023.png)

> **[图片提取文字 (image.png)]:**
> W, is a Vandermonde matrix for the roots of unity up to a normalization factor:  $W_{nk} = \left(e^{-\frac{2\pi i}{N}nk}/\sqrt{N}\right), \tag{2}$ 
> 
> where n, k = 0, ..., N - 1. This matrix multiplication is an  $\mathcal{O}(N^2)$  operation, which has higher asymptotic complexity than the FFT, but turns out to be faster for relatively shorter sequences on TPUs.
![image.png](Attn%E4%BC%98%E5%8C%96%E5%92%8CAcc%E8%AE%BE%E8%AE%A1/image%2024.png)

FFT-Attn用FFT压缩输入tokens，FFTKQV-Attn用FFT压缩K、Q、V后计算Attn。

FFT压缩：不同时间token（的能量/注意力分布）经过FFT得到不同频率token的能量（注意力）分布，其中**能量（注意力）最高的频率称为支配频率**。

浅层Attn的支配频率是**高频分量**，说明注意力在部分token处有较大值，即关注多个局部高频特征。深层Attn的支配频率是**低频分量**，说明少有局部关注，注意力在token上“平铺”，关注整体特征。

设fH=2^k是最大支配频率（相同能量的支配频率中选频率最大的），**2^k是保证L是2的幂次，让大小是L*L的矩阵W-FFT可以进行BP分解**，从而和BMM统一数据依赖模式。

将N个时间步划分成fH块，每块包含L=N/fH个token，则经过FFT后，**每块“平均拥有一个“支配频率中的token**，**去除重复多次的高频s部分**，进行信息的压缩，因为支配频率的tokens平均化后在每个L-size块中是低频部分。fH个L-size块[D, L]参与后续Attn计算。

分块BS-ProjKQV：先分块W-K/Q/V，后进行BP分解，和BS-ProjKQV。

分块BS-ProjKQV是粗粒度tiling（B），定义tile间的全局dataflow。FFTKQV-Attn是细粒度freq-BS pipeline，定义tile内的局部dataflow。

输入X[N, D]，Wproj[D, D]，**输入X[N, D]在D-dim按照B-size分块出{D/B}个X[N, B]**，X[N, B]作BMM得到KQV[N, B]的复杂度是$N*(D/B * BlogB + D/B * B)=NDlogB+ND$。因为D/B个X[1, B]和Wproj[B, B]作BMM(BlogB)，D/B个KQV[1, B]作向量累加(B)，N个X[1, B]和KQV[1, B]需要计算。

每个**KQV[N, B]在N-dim按照L-size分块出{N/L}个KQV[L, B]**，KQV[L, B]作FFT得到KQVfft[L, B]的复杂度是$D*(N/L * LlogL + N/L * L)=DNlogL + DN$。因为N/L个KQV[L, 1]和Wfft[L, L]作BMM(LlogL)，N/L个KQVfft[L, 1]作向量累加(L)，D个KQV[L, 1]和KQVfft[L, 1]需要计算。

输入X[N, D]作分块Proj和FFT的复杂度是$NDlogB + ND + DNlogL + DN = ND * (logB + logL + 2)$。

对X和W**在D-dim按照B-size分块而不降低性能的原因可能是BP分解的性质**。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 7: Our approach: hybridizing structured sparsity and FFT (Decompression is symmetric and omitted)
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0%E5%BC%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E7%B1%BB%E4%BC%BCGPU%E5%A4%9Awarp%E5%B9%B6%E8%A1%8C%EF%BC%89/image%2064.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 3: Improve transformer blocks using structured sparsity.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0%E5%BC%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E7%B1%BB%E4%BC%BCGPU%E5%A4%9Awarp%E5%B9%B6%E8%A1%8C%EF%BC%89/image%2053.png)

### MLX pipeline优化BS-Attn

每个PE的tag blk是一个layer的tile，写清楚每一个PE的指令，在探讨下面的pipeline

结合律：W的4个BM和X的向量，从右向左MV乘法。

不同层BSMV的pipeline：这个BMM的伪代码不对。

不同层BSMV的PE内pipeline：

MM算子：每个PE计算8*8的指令序列，PE之间设计数据流。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 3: Improve transformer blocks using structured sparsity.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0%E5%BC%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E7%B1%BB%E4%BC%BCGPU%E5%A4%9Awarp%E5%B9%B6%E8%A1%8C%EF%BC%89/image%2053.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> (a) Continuous BPMM Applied on a Vector (Lower half omitted)
> 
> matrix multiplications (BSMMs).
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0%E5%BC%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E7%B1%BB%E4%BC%BCGPU%E5%A4%9Awarp%E5%B9%B6%E8%A1%8C%EF%BC%89/image%2056.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 10: Allocating computing resources for BSMMs. (For clarity, batch-based SIMD and vertical hops for stride =4,8 are omitted.)
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0%E5%BC%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E7%B1%BB%E4%BC%BCGPU%E5%A4%9Awarp%E5%B9%B6%E8%A1%8C%EF%BC%89/image%2090.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> is vectorized
> 
> (a) Map a Single Layer to PE Mesh
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Fig. 11: Mapping a dense MM to MLX in multi-layer dataflow
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0%E5%BC%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E7%B1%BB%E4%BC%BCGPU%E5%A4%9Awarp%E5%B9%B6%E8%A1%8C%EF%BC%89/image%2091.png)

### BSMM的指令Mapping

**kernel参数**

N=1024，D=1024。

Proj：累加D，B-size=32的块，对D分块。

FFT：累加N，L-size=N，N/2，……1的块，对N分块，每层不同。

FFN：累加D，B-size=32的块，对D分块。

**PE指令Mapping**

对每种kernel，**16个PE并行P=16行的计算**，迭代4次有T=64行。

s=1、2、4、8、16、32、64、128、256、512。

i-base=0、16、32、48、64、。。。

3D cube graph，tag0优先tag1，tag1 优先tag2。

因子矩阵B0：2个PE计算1个因子b(=2)，16个PE计算8个因子b。

因子矩阵B1：4个PE计算1个因子b(=4)，16个PE计算4个因子b。

因子矩阵B2：8个PE计算1个因子b(=8)，16个PE计算2个因子b。

因子矩阵B3：16个PE计算1个因子b(=16)。

因子矩阵B4：32个PE计算1个因子b(=32)，16个PE计算0.5个因子b。

B0{s=1/b=2}，

B1{s=2/b=4}，

B2{s=4/b=8}，

B3{s=8/b=16}，

B4{s=16/b=32}，

B5{s=32/b=64}，

B6{s=64/b=128}，

B7{s=128/b=256}，

B8{s=256/b=512}，

B9{s=512/b=1024}

每个PE指令的index规则：基于base、tag-index决定指令。

PE-index被映射到x的i-base决定输出行index和其中一个输入行index i1（加粗）。

tag-index决定s和b（红色）。记i1处于b的倍数时为i1-base，当i1处于[i1-base, i1-base+b-1]时，第二个输入行index i2和i1距离s，让i2处于[i1-base, i1-base+b-1]。

PE**0(base=0/16…)**

(2): 1[**0**] x[**0**],x[1]

(4): 2[**0**] 1[**0**],1[2]

(8): 3[**0**] 2[**0**],2[4]

(16):4[**0**] 3[**0**],3[8]

(32):5[**0**] 4[**0**],4[16]

(32):5[16] 4[16],0]

………

(64):6[0] 5[0],32]

(64):6[16] 5[16],48]

(64):6[32] 5[32],0]

(64):6[48] 5[48],16]

PE**1(base=1/17…)**

(2): 1[**1**] x[**1**],x[0]

(4): 2[**1**] 1[**1**],1[3]

(8): 3[**1**] 2[**1**],2[5]

(16):4[1] 3[1],3[9]

(32):5[1] 4[1],4[17]

(32):5[17] 4[17],1]

………

(64):6[1] 5[1],33]

(64):6[17] 5[17],49]

(64):6[33] 5[33],1]

(64):6[49] 5[49],17]

PE**2(base=2/18…)**

(2): 1[**2**] x[**2**],x[3]

(4): 2[**2**] 1[**2**],1[0]

(8): 3[**2**] 2[**2**],2[6]

(16):4[2] 3[2],3[10]

(32):5[2] 4[2],4[18]

(32):5[18] 4[18],2]

………

(64):6[2] 5[2],34]

(64):6[18] 5[18],50]

(64):6[34] 5[34],2]

(64):6[50] 5[50],18]

PE**3(base=3/19…)**

(2): 1[**3**] x[**3**],x[2]

(4): 2[**3**] 1[**3**],1[1]

(8): 3[**3**] 2[**3**],2[7]

(16):4[3] 3[3],3[11]

(32):5[3] 4[3],4[19]

(32):5[19] 4[19],3]

………

(64):6[3] 5[3],35]

(64):6[19] 5[19],51]

(64):6[35] 5[35],3]

(64):6[51] 5[51],19]

PE**4(base=4/20…)**

(2): 1[**4**] x[**4**],x[5]

(4): 2[**4**] 1[**4**],1[6]

(8): 3[**4**] 2[**4**],2[0]

(16):4[4] 3[4],3[12]

(32):5[4] 4[4],4[20]

(32):5[20] 4[20],4]

(64):6[4] 5[4],36]

(64):6[20] 5[20],52]

(64):6[36] 5[36],4]

(64):6[52] 5[52],20]

PE**5(base=5/21…)**

(2): 1[**5**] x[**5**],x[4]

(4): 2[**5**] 1[**5**],1[7]

(8): 3[**5**] 2[**5**],2[1]

(16):4[5] 3[5],3[13]

(32):5[5] 4[5],4[21]

(32):5[21] 4[21],5]

(64):6[5] 5[5],37]

(64):6[21] 5[21],53]

(64):6[37] 5[37],5]

(64):6[53] 5[53],21]

PE**6(base=6/22…)**

(2): 1[**6**] x[**6**],x[7]

(4): 2[**6**] 1[**6**],1[4]

(8): 3[**6**] 2[**6**],2[2]

(16):4[6] 3[6],3[14]

(32):5[6] 4[6],4[22]

(32):5[22] 4[22],6]

(64):6[6] 5[6],38]

(64):6[22] 5[22],54]

(64):6[38] 5[38],6]

(64):6[54] 5[54],22]

PE**7(base=7/23…)**

(2): 1[**7**] x[**7**],x[6]

(4): 2[**7**] 1[**7**],1[5]

(8): 3[**7**] 2[**7**],2[3]

(16):4[7] 3[7],3[15]

(32):5[7] 4[7],4[23]

(32):5[23] 4[23],7]

(64):6[7] 5[7],39]

(64):6[23] 5[23],55]

(64):6[39] 5[39],7]

(64):6[55] 5[55],23]

PE**8(base=8/24…)**

(2): 1[**8**] x[**8**],x[9]

(4): 2[**8**] 1[**8**],1[10]

(8): 3[**8**] 2[**8**],2[12]

(16):4[8] 3[8],3[0]

(32):5[8] 4[8],4[24]

(32):5[24] 4[24],8]

(64):6[8] 5[8],40]

(64):6[24] 5[24],56]

(64):6[40] 5[40],8]

(64):6[56] 5[56],24]

PE**9(base=9/25…)**

(2): 1[**9**] x[**9**],x[8]

(4): 2[**9**] 1[**9**],1[11]

(8): 3[**9**] 2[**9**],2[13]

(16):4[9] 3[9],3[1]

(32):5[9] 4[9],4[25]

(32):5[25] 4[25],9]

(64):6[9] 5[9],41]

(64):6[25] 5[25],57]

(64):6[41] 5[41],9]

(64):6[57] 5[57],25]

PE**10(base=10/26..)**

(2): 1[**10**] x[**10**],x[11]

(4): 2[**10**] 1[**10**],1[8]

(8): 3[**10**] 2[**10**],2[14]

(16):4[10] 3[10],3[2]

(32):5[1**0**] 4[1**0**],26]

(32):5[26] 4[26],10]

(64):6[10] 5[10],42]

(64):6[26] 5[26],58]

(64):6[42] 5[42],10]

(64):6[58] 5[58],26]

PE**11(base=11/27…)**

(2): 1[**11**] x[**11**],x[10]

(4): 2[**11**] 1[**11**],1[9]

(8): 3[**11**] 2[**11**],2[15]

(16):4[11] 3[11],3[3]

(32):5[11] 4[11],27]

(32):5[27] 4[27],11]

(64):6[11] 5[11],43]

(64):6[27] 5[27],59]

(64):6[43] 5[43],11]

(64):6[59] 5[59],27]

PE**12(base=12/28..)**

(2): 1[**12**] x[**12**],x[13]

(4): 2[**12**] 1[**12**],1[14]

(8): 3[**12**] 2[**12**],2[8]

(16):4[12] 3[12],3[4]

(32):5[12] 4[12],28]

(32):5[28] 4[28],12]

(64):6[12] 5[12],44]

(64):6[28] 5[28],60]

(64):6[44] 5[44],12]

(64):6[60] 5[60],28]

PE**13(base=13/29..)**

(2): 1[**13**] x[**13**],x[12]

(4): 2[**13**] 1[**13**],1[15]

(8): 3[**13**] 2[**13**],2[9]

(16):4[13] 3[13],3[5]

(32):5[13] 4[13],29]

(32):5[29] 4[29],13]

(64):6[13] 5[13],45]

(64):6[29] 5[29],61]

(64):6[45] 5[45],13]

(64):6[61] 5[61],29]

PE**14(base=14/30..)**

(2): 1[**14**] x[**14**],x[15]

(4): 2[**14**] 1[**14**],1[12]

(8): 3[**14**] 2[**14**],2[10]

(16):4[14] 3[14],3[6]

(32):5[14] 4[14],30]

(32):5[30] 4[30],14]

(64):6[14] 5[14],46]

(64):6[30] 5[30],62]

(64):6[46] 5[46],14]

(64):6[62] 5[62],30]

PE**15(base=15/31..)**

(2): 1[**15**] x[**15**],x[14]

(4): 2[**15**] 1[**15**],1[13]

(8): 3[**15**] 2[**15**],2[11]

(16):4[15] 3[15],3[7]

(32):5[15] 4[15],31]

(32):5[31] 4[31],15]

(64):6[15] 5[15],47]

(64):6[31] 5[31],63]

(64):6[47] 5[47],15]

(64):6[63] 5[63],31]

**这个指令可以动态调整？**

### layout和IO优化

layout transformation

BSMM和FFT的layout倾向不同，BSMM在D-dim上乘累加，FFT和FFT-1在N-dim上乘累加。

BS locality、SIMD在PE-pipeline上temporal并行

**packing**

[D,N]的row major，最内层打包8个D-dim，即[D/8,N,8]，**每8个N-dim和8个D-dim是1个访问闭包**。

FFT在N-dim上累加，SIMD指令计算FFT的8个D-dim的结果。

BMM在D-dim上累加，SIMD指令计算BMM的8个N-dim的结果。

但是不同kernel的block参数的指令不同，依赖参数也不同，似乎需要具体分析？？？

**shuffle IO？**

closed-set locality：将计算依赖的数据“团”局部化存储。

不同radix表示不同因子矩阵，不同stage表示因子矩阵计算不同因子。

FFT和BSMM几何可拆分，将Block内部数据按照依赖距离重排（12.2→12.3），减少长距离Xfer。

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> ![](_page_0_Figure_1.jpeg)
> 
> Fig. 12: (a) Data footprint of BSMM; (b) Shuffling for a closed set of data footprint; (c) Optimize data layout for SIMD-friendly batching.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0%E5%BC%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E7%B1%BB%E4%BC%BCGPU%E5%A4%9Awarp%E5%B9%B6%E8%A1%8C%EF%BC%89/image%2092.png)

> **[图片提取文字 (image.png)]:**
> ![](_page_0_Figure_0.jpeg)
> 
> Fig. 3: Improve transformer blocks using structured sparsity.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0%E5%BC%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E7%B1%BB%E4%BC%BCGPU%E5%A4%9Awarp%E5%B9%B6%E8%A1%8C%EF%BC%89/image%2053.png)

> **[图片提取文字 (image.png)]:**
> Optimizing Data Layout. FFT and BSMM operate along orthogonal dimensions: FFT depends on the sequence axis N, while BSMM depends on the hidden axis D. This orthogonality allows a layout that simultaneously preserves butterfly locality and exposes SIMD parallelism. Under a naive row-major  $D \times N$  layout, SIMD accesses across D become stride-N, destroying locality in both operators. As shown in Fig. 12(b), we apply a lightweight packing step that tiles D into a 8wide entry and places it in the innermost dimension. This
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0%E5%BC%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E7%B1%BB%E4%BC%BCGPU%E5%A4%9Awarp%E5%B9%B6%E8%A1%8C%EF%BC%89/image%2093.png)

> **[图片提取文字 (image.png)]:**
> Closed-set Locality. In chunked FFT, each L-point segment remains confined to local data dependency determined by radix and stage. A k-layer BSMM shares the same structural property: each output depends only on inputs spaced at a block scale  $B=2^k$ , naturally partitioning an n-element vector into  $\frac{n}{2^k}$  disjoint closed sets of size  $2^k$  internally. As k grows, each closed set expands and sparsity increases, eventually inducing long-stride accesses (e.g., half-array strides when k=log n-1).
> 
> aligns SIMD along D while keeping butterfly dependencies
> 
> contiguous along N, producing tiles that match MLX's closed-
> 
> set blocks for continuous folded execution.
> 
> Therefore, intermediate results have to be pushed far across the scratchpad memory, as shown in 2 of Fig. [12]. As L or B grows, these sets enlarge and long-stride accesses emerge (e.g., half-array strides when  $k = \log n - 1$ ), breaking spatial locality (Fig. [12]). Our key observation is that the butterfly
> 
> dependency graph is always algebraically partitionable: FFT and BSMM can be reordered to strictly respect their closed sets. This reorganization reduces long-stride accesses to a small number of compact, on-chip-resident blocks (Fig. 123),
> 
> restoring bounded locality and providing the fixed execution
> 
> footprints required for multi-layer pipeline.
![image.png](%E5%A4%9A%E4%BB%BB%E5%8A%A1%E6%8A%A2%E5%8D%A0%E5%BC%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E7%B1%BB%E4%BC%BCGPU%E5%A4%9Awarp%E5%B9%B6%E8%A1%8C%EF%BC%89/image%2094.png)