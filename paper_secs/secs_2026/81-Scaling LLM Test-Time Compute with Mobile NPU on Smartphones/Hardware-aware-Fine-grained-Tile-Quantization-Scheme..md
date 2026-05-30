# Hardware-aware Fine-grained Tile Quantization Scheme.

We propose a novel quantization layout that performs group quantization in fine-grained rectangular tiles, as opposed to conventional approaches that group along the accumulation axis. To align with the memory access patterns of both the matrix and vector units, we introduce an offline pipeline involving weight pre-quantization transformation, quantization, and post-quantization transformation. This enhances the continuity of runtime memory access and eliminates unnecessary computational overhead.

Efficient LUT-Based Computation. For more complex runtime operations, we leverage the vector unit's lookup table (LUT) instructions and generalized LUT mechanisms to replace intricate transformation logic. This approach accelerates key bottleneck operations in test-time scaling workloads, including dequantization within mixed-precision GEMM and the Softmax operation in Attention.

