# Algorithm 1: Programmable Bootstrapping

```
Input: LWE ciphertext c = (a_1, \ldots, a_n, b) \in T_q^{n+1}
   Require: BSK, KSK, TP
   Output: LWE ciphertext c'' \in T_q^{n+1}
   // Modulus-Switching
\tilde{c} = (\tilde{a}_1, \dots, \tilde{a}_n, \tilde{b}) \leftarrow MS(c)
  ACC_0 \leftarrow \text{TP}
    // Blind Rotation
3 for i=1 to n do
        // Rotation
        ACC_{rotate} \leftarrow X^{\tilde{a}_i} \cdot ACC_{i-1} - ACC_{i-1};
            Decompose and FFT
        ACC_{fourier} \leftarrow Decompose\&FFT(ACC_{rotate});
        // MAC: Production of ACC_{fourier}
        // and pre-computed BSK
        ACC_{fourier} \leftarrow ACC_{fourier} \odot BSK;
            IFFT and Accumulation
        ACC_i = IFFT(ACC_{fourier}) + ACC_{i-1};
8 end
       Sample Extraction
   c' = (a'_1, \dots, a'_{kN}, b') \leftarrow \text{SE}(ACC_n)
   // Key Switch
10 c'' = (0, \dots, b') - \sum_{i=1}^{kN} \sum_{j=1}^{lk} (a'_i)_j \cdot \text{KSK}(i, j)
11 return c''
```

coefficients  $\tilde{a}_i$  to homomorphically rotate the test polynomial (TP) that encodes the target function  $f(\cdot)$ . At the beginning of this process, the accumulator  $\mathbf{ACC}_0$  is initialized to the test polynomial, i.e.,  $\mathbf{ACC}_0 = \mathrm{TP}$ , where TP is a **GLWE** ciphertext composed of (k+1) polynomials, each with N coefficients storing the full set of values of f(m).

Then, the rotation operation is applied to the previous accumulator  $\mathbf{ACC}_{i-1}$  as  $\mathbf{ACC}_i = \mathbf{X}^{\tilde{a}_i}\mathbf{ACC}_{i-1}$ , where  $\mathbf{X}^{\tilde{a}_i}$  denotes the rotation operator. Afterward, the result of  $\mathbf{X}^{\tilde{a}_i}\mathbf{ACC}_{i-1} - \mathbf{ACC}_{i-1}$  is decomposed into l base- $\beta$  digits through the Decompose procedure, which bit-slices and rounds each polynomial coefficient into l groups. This produces an intermediate ciphertext represented as a  $(k+1) \times l$  matrix of polynomials, upon which a matrix multiplication, known as the  $External\ Product$ , is performed.

The decomposed polynomial matrix of dimension  $1 \times (k+1)l$  is multiplied with the corresponding **Bootstrapping Key BSK**i of dimension  $(k+1)l \times (k+1)$ , generating a temporary ciphertext that is then integrated into **ACC**i. The Blind Rotation is performed iteratively for all n coefficients  $\tilde{a}_i$ , resulting in a final accumulator containing (k+1) polynomials. This accumulator is then processed by Sample Extraction, which extracts the  $0^{\text{th}}$  plaintext component from the GLWE ciphertext. This operation converts a GLWE ciphertext of shape  $(k+1) \times N$  into an LWE ciphertext of dimension kN+1. For the i-th component, Sample Extraction consists of a series of permutations defined in Eq. 1.

$$SE^{i}((A_{0}, A_{1}, A_{2}, ..., A_{n-1}, B))$$

$$= SE^{i}((a_{0,0}, a_{0,1}, ..., a_{0,N-1}), (a_{1,0}, a_{1,1}, ..., a_{1,N-1}), ..., (b_{0}, b_{1}, ..., b_{N-1}))$$

$$= ((a_{0,0}, ..., a_{0,i}, -a_{0,N-1}, ..., -a_{0,N-i-1}), (a_{1,0}, ..., a_{1,i}, -a_{1,N-1}, ..., -a_{1,N-i-1}), ..., (b_{i}))$$

$$(1)$$

![](_page_2_Picture_6.jpeg)

Fig. 1: Overview of GPU architecture.

The output of Sample Extraction is then processed by Keyswitch, which reduces the LWE ciphertext dimension from kN+1 to its original size of N+1. It first decomposes the ciphertext into  $l_k$  components, and then combines them with the corresponding key material according to Eq. 2.

$$c'' = (0, ..., b') - \sum_{i=1}^{kN} \sum_{j=1}^{l_k} (a'_i)_j \cdot KSK_{(i,j)}$$
 (2)

We note that Blind Rotation is the most computationally intensive stage of the PBS. This is primarily due to the large number of (I)FFT operations: the Decomposition is performed in the coefficient domain, whereas the polynomial multiplications in the  $External\ Product$  must be executed in the pointwise (Fourier) domain to reduce arithmetic cost. Furthermore, the BSK introduces substantial computational and memory overhead, as it contains  $(k+1)l\times(k+1)\times N\times n$  elements. To better characterize these impacts, this work conducts a detailed architectural analysis of PBS on commercial hardware and introduces several hardware-oriented algorithmic optimizations to improve computational efficiency.

