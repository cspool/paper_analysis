# A Limitation

Due to our method's reliance on high-throughput 4-bit Tensor Core instructions to accelerate the Selection-Pass, it may lose its performance advantage on hardware that does not support efficient 4-bit matrix multiplication.

Moreover, our current implementation is limited to approximating attention weights using Int4 quantization. Additional adaptations would be needed to deploy our method on hardware that supports FP4 GEMM or LUT-based low-bit GEMM. We leave it as our future work.

