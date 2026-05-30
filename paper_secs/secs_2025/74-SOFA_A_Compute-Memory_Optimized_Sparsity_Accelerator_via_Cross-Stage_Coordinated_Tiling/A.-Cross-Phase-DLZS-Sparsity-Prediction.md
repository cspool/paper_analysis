# A. Cross-Phase DLZS Sparsity Prediction

Traditional dynamic sparsity entails predicting significant Q-K pairs, then utilizing these important Ks and Vs to execute computations. However, blindly generating unnecessary KV leads to wastage in computation and memory access. To this end, SOFA employs an on-demand computation strategy for KV. As shown in Fig. 7(a), On-demand means: only the required Ks and Vs are generated  $(\mathbf{K}_i = \mathbf{x}_i \mathbf{W}_k, \mathbf{V} = \mathbf{x}_i \mathbf{W}_v)$ , while trivial ones are not computed from the beginning. However, this requires the *pre-compute stage* first to estimate the  $\hat{\mathbf{K}}$ , then utilize it with  $\mathbf{Q}$  to predict  $\hat{\mathbf{A}}$ . Unfortunately, even utilizing low-precision matrix multiplication (e.g. halfprecision with MSBs only) results in considerable power consumption. Therefore, a power and memory-efficient prediction is imperative.

We propose a log-domain multiplication-free strategy, named differential leading zero summation (DLZS). Differential means: For multiplication, it only transforms one operand into the logarithmic domain using the leading zero encoder (LZE), to obtain its leading zero (LZ). Then, based on the LZ, it substitutes the costly multiplication with low-power shift operations on the other operand. Specifically, an INTtype number x can be mathematically expressed as Eq. (1a), where W stands for the bit-width, M represents the mantissa lying [0, 1], and LZ denotes the leading-zero count of x. Accordingly, the corresponding multiplication is derived as Eq. (1b) and approximated as Eq. (1c). Since the bit width Wis fixed for certain operands, we can directly operate  $LZ_{\nu}$  on x, to estimate the magnitude for the product of two numbers. Therefore, incorporating shifting and the sign bit, the results of multiplication can be predicted.

$$x = Sign \times M \times 2^{W - LZ},\tag{1a}$$

$$x = Sign \times M \times 2^{W-LZ},$$
 (1a)  
 
$$x \cdot y = XOR(S_x, S_y) M_x \cdot 2^{(W-LO_x)} M_y \cdot 2^{(W-LO_y)}$$
 (1b)

$$\approx \text{XOR}(S_x, S_y) M_x \cdot 2^{(W - LO_x + W - LO_y)}$$
 (1c)

Its workflow is depicted in Fig. 7(b). As the weights are pre-known and fixed during inference, we pre-convert the  $\mathbf{W}_k$  into LZ format and store it. Then, in the Key prediction phase (1.1), no LZE is required, as the weight  $W_k$  has been converted into LZ format. In the subsequent Attention prediction phase (1.2), to mitigate error accumulation, we convert  $\mathbf{Q}$  into the log domain instead of  $\mathbf{K}$ , then perform shifting and sum operations. Compared to the vanilla leading zero strategy (Fig. 7(c)), the proposed DLZS exhibits three Pros: a) Lower converter overhead; b) Higher accuracy; c) Less memory access.

