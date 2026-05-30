# <span id="page-6-2"></span>Results of AFPQ with FP formats

Additional results of RTN quantization using FP4/3 formats are shown in Table 7 and Table 8, respectively.

## <span id="page-6-3"></span>**Kernel implementation**

Currently, W-only quantization requires low-bit weights to be dequantized to FP16 during inference, and then calculations are performed with the FP16 activations. In our system implementation, we store two 4-bit quantized weights using one byte. During dequantization, we load the byte and recover it to two 4-bit weights.

For INT and FP formats, the conversion from 4-bit to FP16 values can be completed by algebraic computation. For NP formats, it can be realized

