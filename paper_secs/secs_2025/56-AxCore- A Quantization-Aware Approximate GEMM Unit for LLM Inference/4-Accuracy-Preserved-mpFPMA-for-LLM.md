# 4 Accuracy-Preserved mpFPMA for LLM

#### 4.1 Extending FPMA to mpFPMA

To enable efficient mixed-precision general matrix-matrix multiplication (mpGEMM) for quantized LLM inference, we extend Floating-Point Multiplication Approximation (FPMA) to support operands with different precision levels. While the approximation formula remains structurally similar to conventional FPMA, the bit widths, fixed-point alignment, and bias correction must be carefully redesigned. For illustration, we denote  $r = a \times w_q$  as the multiplication between the input activation a in FP16 and the low-bit quantized weight  $w_q$  in FP4.

In mpFPMA, operands are first aligned to a common fixed-point representation to ensure correct addition. Since FP4 contains fewer mantissa bits than FP16, we left-shift (i.e., zero-pad) the FP4 operand's mantissa to match the resolution of FP16. The aligned value is expressed as:

$$Align(w_q) = w_q \ll (Mantissa_{FP16} - Mantissa_{FP4})$$
 (6)

This ensures the radix point aligns across both operands. However, due to differing exponent biases (e.g., 15 for FP16 vs. 1 for FP4 E2M1), a format-aware bias correction term  $B_1$  is needed:

$$B_1 = B_a + B_{w_q} - B_r (7)$$

where  $B_{\rm a}$ ,  $B_{w_q}$  and  $B_{\rm r}$  are the exponent biases of the activation, the quantized weight, and the result, respectively. For typical configurations where the activation and result are both in FP16, this simplifies to  $B_1=B_{w_q}$ . Combining the alignment and bias correction, the approximate result R of the mixed-precision product is:

$$R = A + Align(W_q) - B_1$$
 (8)

To illustrate, consider multiplying an FP4 (E2M1) weight encoded as "0\_01\_1" (representing 1.5) with an FP16 activation of 2. The aligned FP4 becomes "0\_00001\_1000000000", and the bias correction value  $B_1$  corresponds to 1. Adding the two and subtracting the bias yields a final result of 3, accurately approximating  $1.5 \times 2$ .

To improve the numerical fidelity of mpFPMA, especially under quantization noise and approximation error, we introduce a constant compensation term  $C_1$  (details in Section 4.3). The final mpFPMA expression becomes:

$$R = A + Align(W_a) - B_1 + C_1 \tag{9}$$

where R, A, and  $W_q$  are the binary approximation of the result, activations, and weights, respectively. This formulation allows Ax-Core to efficiently and accurately approximate mixed-precision multiplications using only integer additions.

