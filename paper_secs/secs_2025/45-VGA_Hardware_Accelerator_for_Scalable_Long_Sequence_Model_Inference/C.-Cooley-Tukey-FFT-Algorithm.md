# *C. Cooley-Tukey FFT Algorithm*

The Discrete Fourier Transform (DFT) is a process that converts a vector ⃗u = (u0, u1, ..., uL−1) in the *time domain* to another length-L vector U⃗ = (U0, U1, ..., UL−1) in the *frequency domain*. This transformation is extensively used in diverse fields as it has many useful properties. However, since the computational complexity of transforming ⃗u to U⃗ is O(L 2 ), there are many algorithms that reduce it to O(Llog L). The most popular algorithm is the Radix-2 Cooley-Tukey algorithm which is used when L is a power of 2.

The Radix-2 algorithm is a divide-and-conquer algorithm that recursively divides the input into two sub-vectors of equal length. For example, the length-L vector ⃗u is divided into a vector consisting of the even-indexed elements ⃗e = (u0, u2, ..., uL−2) and a vector consisting of the odd-indexed elements ⃗o = (u1, u3, ..., uL−1). Then, from the DFT results E⃗ and O⃗ of the two sub-vectors, U⃗ can be constructed. Specifically, the relationship between two elements U<sup>i</sup> and Ui+L/<sup>2</sup> is described in Eq. (1).

The operation of computing U<sup>i</sup> and Ui+L/<sup>2</sup> from E<sup>i</sup> and Oi is commonly known as the *butterfly (BF) operation*, and the value z is a fixed complex number such that z <sup>L</sup> = 1. The powers of z, z i (i = 0, 1, ..., L − 1), are called *twiddle factors*. To summarize, the length-L vector ⃗u is repeatedly divided log<sup>2</sup> L times, and each pair (U<sup>i</sup> , Ui+L/2) of U⃗ is constructed through log<sup>2</sup> L butterfly operations. Fig. 2(a) shows an example when L = 4. Starting from the input vector on the leftmost column, each element of U⃗ on the rightmost column is acquired through two butterfly operations.

$$BF(E_i, O_i, z^{2i}) = \begin{cases} U_i = E_i + z^{2i} \cdot O_i \\ U_{i+L/2} = E_i - z^{2i} \cdot O_i \end{cases}$$

$$(i = 0, 1, ..., L/2 - 1)$$
(1)

The Cooley-Tukey algorithm has a generalized version applicable to arbitrary integers that can be expressed as a product of two natural numbers [17]. When L is equal to the product of L<sup>1</sup> and L2, the algorithm treats ⃗u as a 2D matrix with dimensions (L<sup>1</sup> × L2) in row-major form. An example of this is shown in Fig. 2(b) for the case of L<sup>1</sup> = 2

![](_page_3_Figure_0.jpeg)

Fig. 3. Interchangeability of self-attention and H3 layers. H3 models replace self-attention with H3 layers within transformer blocks. The red box denotes the target region of interest (ROI) for the accelerator.

and  $L_2=3$ , and it takes 6 steps to compute the FFT. Thirst, an input vector  $\vec{u}$  with a length of 6 is reshaped into a  $2\times 3$  matrix. Second, Fourier transform is conducted independently on each column. The results of each columnwise transformation is denoted as  $(c_0,c_3)$ ,  $(c_1,c_4)$  and  $(c_2,c_5)$ . Third, the result of the FFT is pointwise multiplied with a matrix of twiddle factors called *compensated twiddle factors* (CTFs). Fourth, another round of Fourier transforms are conducted independently on each row. Finally, the DFT of  $\vec{u}$ , denoted as  $\vec{U}$ , is obtained after reshaping into a 1D array.

However, the reduction in computational burden from the Cooley-Tukey algorithm does come at a cost. The Radix-2 algorithm introduces the need to permute the sequence every stage in order to conduct the butterfly operation. Furthermore, the result produced by the algorithm is not in the order of naive DFT, but rather in a shuffled order called *bit-reversal* order. A separate permutation has to be performed in order to recover the original order of DFT. These permutations are expressed as dotted lines in Fig. 2(a). The generalized version, on the other hand, requires both row-wise and column-wise access to the 2D data matrix, thus inevitably leading to inefficient access patterns in modern memory systems.

## D. State Space Model (SSM)-based Global Convolution

Many different approaches exist in global convolution models, and what mainly distinguishes each model is how it generates the filter. One of the most successful approaches builds upon SSMs, a class of models used in the fields of control theory and statistics to model a system that changes over time. A specific class of models called Linear Time Invariant (LTI) SSMs has to be used to generate the convolution filter  $\vec{\mathbf{K}}$  as in Eq. (2). The input vector  $\vec{u}$  and the output vector  $\vec{y}$  are both length-l real vectors, and the initial state vector  $\vec{x}_0$  is a length-m complex vector, where m is a parameter of the SSM.

$$\vec{\mathbf{K}} = (\mathbf{C}\mathbf{A}^{0}\mathbf{B}, \mathbf{C}\mathbf{A}^{1}\mathbf{B}, \cdots, \mathbf{C}\mathbf{A}^{l-1}\mathbf{B})$$

$$y_{i} = \mathbf{C}\mathbf{A}^{i}\vec{x}_{0} + (\vec{\mathbf{K}}*\vec{u})_{i} + \mathbf{D}\vec{u}_{i}$$

$$(\mathbf{A} \in \operatorname{diag}(\mathbb{C}^{m}), \mathbf{B} \in \mathbb{C}^{m \times 1}, \mathbf{C} \in \mathbb{C}^{1 \times m}, \mathbf{D} \in \mathbb{C}^{1 \times 1})$$
(2)

Generally, the parameter matrix A can be non-diagonal, but this leads to costly kernel generation, as computing the

elements  $\mathbf{C}\mathbf{A}^i\mathbf{B}$  of the filter  $\vec{\mathbf{K}}$  requires many matrix multiplications, making it difficult to train the model. Thus, the matrix  $\mathbf{A}$  is constrained to a complex diagonal matrix, as this diagonalization significantly reduces the computational complexity of computing  $\vec{\mathbf{K}}$  from  $O(lm^3)$  to O(lm).

H3 [18] is the state-of-the-art model that incorporates all the techniques explained previously. It exhibits lower perplexity in the WikiText103 [38] dataset compared to models with similar sizes such as GPT-2 and GPT-Neo. The H3 block (Fig. 3) has a similar structure to that of a transformer block mainly composed of an H3 layer and a Feed-Forward Network (FFN), with additional dropout, residual sum, and layer normalization layers in between. The overall structure of an H3 layer is similar to that of a self-attention layer, with the attention operation replaced by the global convolution. Specifically, as shown in Fig. 3, the input  $\vec{x}$  of dimensions  $l \times d$  is fed through 3 different fully connected layers to produce 3 equalsized matrices (Q, K, V). Afterwards, for each of the lengthl column vectors  $\vec{Q}_i$ ,  $\vec{K}_i$ ,  $\vec{V}_i$ , the following operations are conducted. First, a short 1-D convolution is conducted on  $\vec{K}_i$ , and the result is multiplied by  $\vec{V}_i$  in a pointwise manner. Then, an SSM-based global convolution operation (SSMConv) is performed, which is multiplied pointwise with  $\vec{Q_i}$  before the result is passed through the final fully connected layer.

#### E. State Passing

Global convolution models make use of FFTs to reduce the computational complexity of the convolution operation. However, the FFT algorithm reads and updates the entire sequence at each stage, making memory bandwidth a critical factor for its performance. To avoid a DRAM memory bandwidth bottleneck, optimized GPU kernels for FFT aggressively utilize the shared memory of SMs to fit the entire input sequence. The benefit of such optimizations becomes limited as the input sequence gets longer and the entire sequence no longer fits in SRAM, which inevitably generates additional DRAM accesses.

SSM-based models can resolve this problem as they can divide an input sequence  $\vec{u}$  of length  $N(=C\times L)$  into C chunks of length-L vectors  $\vec{u_c}$  (c=0,1,...,C-1), and convolution using FFT can be conducted separately on each chunk with the help of the *state passing algorithm*. As the chunk size can be selected freely to fit within the GPU SRAM capacity, this mechanism greatly reduces DRAM memory access during FFT. This is possible since the elements of  $\vec{K}$  have a recursive relation, all previous chunks can be summarized into a state vector  $\vec{x}_{c-1}$ , and its contribution can be computed separately from the convolution of the current chunk

$$\vec{\mathbf{K}} = \left(\mathbf{C}\mathbf{A}^{0}\mathbf{B}, \mathbf{C}\mathbf{A}^{1}\mathbf{B}, \cdots, \mathbf{C}\mathbf{A}^{L-1}\mathbf{B}\right)$$

$$\vec{x}_{c} = \mathbf{A}^{L}\vec{x}_{c-1} + \mathbf{M}_{ux}\vec{u}_{c}$$

$$\vec{y}_{c} = \mathbf{M}_{xy}\vec{x}_{c-1} + \vec{\mathbf{K}} * \vec{u}_{c} + \mathbf{D}\vec{u}_{c}$$
(3)

Eq. (3) and Fig. 4 illustrate two steps in state passing: 1) updating the state vector to  $\vec{x}_c$  using  $\vec{u}_c$  (State Update),

![](_page_4_Picture_0.jpeg)

Fig. 4. Detailed diagram of SSMConv with the state passing algorithm. (a) The state vector is passed from the previous State Passing (SP) block to the next SP block. (b) The SP block computation is composed of State Update, Output Projection, FFT convolution, and pointwise operations.

and 2) producing the output chunk  $\vec{y}_c$  by adding the FFT-based convolution output of the corresponding input chunk  $\vec{u}_c$  (FFTConv) to the projection of previous state vector  $\vec{x}_{c-1}$  (Output Projection).

 $\mathbf{M}_{xy}$  and  $\mathbf{M}_{ux}$  are matrices constructed from the parameters  $\mathbf{A}$ ,  $\mathbf{B}$  and  $\mathbf{C}$  as shown in Eq. (4) and Eq. (5). It is worth noting that the two matrices are Vandermonde matrices, which means all rows of  $\mathbf{M}_{xy}$  can be obtained by recursively multiplying the first row with a fixed vector consisting of diagonal elements of  $\mathbf{A}$ . Likewise, all columns of  $\mathbf{M}_{ux}$  can be computed by multiplying the first column with the same fixed vector. This property arises from the recurrent nature of the SSM.

$$\mathbf{M}_{xy} = \begin{pmatrix} \mathbf{C}_{0}\mathbf{A}_{0} & \mathbf{C}_{1}\mathbf{A}_{1} & \cdots & \mathbf{C}_{m-1}\mathbf{A}_{m-1} \\ \mathbf{C}_{0}\mathbf{A}_{0}^{2} & \mathbf{C}_{1}\mathbf{A}_{1}^{2} & \cdots & \mathbf{C}_{m-1}\mathbf{A}_{m-1}^{2} \\ \vdots & \vdots & \ddots & \vdots \\ \mathbf{C}_{0}\mathbf{A}_{0}^{L} & \mathbf{C}_{1}\mathbf{A}_{1}^{L} & \cdots & \mathbf{C}_{m-1}\mathbf{A}_{m-1}^{L} \end{pmatrix}$$

$$\mathbf{M}_{ux} = \begin{pmatrix} \mathbf{A}_{0}^{L-1}\mathbf{B}_{0} & \cdots & \mathbf{A}_{0}\mathbf{B}_{0} & \mathbf{B}_{0} \\ \mathbf{A}_{1}^{L-1}\mathbf{B}_{1} & \cdots & \mathbf{A}_{1}\mathbf{B}_{1} & \mathbf{B}_{1} \\ \vdots & \ddots & \vdots & \vdots \\ \mathbf{A}_{m-1}^{L-1}\mathbf{B}_{m-1} & \cdots & \mathbf{A}_{m-1}\mathbf{B}_{m-1} & \mathbf{B}_{m-1} \end{pmatrix}$$

$$(4)$$

