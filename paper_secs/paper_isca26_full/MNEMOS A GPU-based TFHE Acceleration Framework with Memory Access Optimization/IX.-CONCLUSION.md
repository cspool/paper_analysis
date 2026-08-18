# IX. CONCLUSION

In this paper, we propose a high-performance GPU acceleration framework for TFHE's PBS to address its severe memorybound bottlenecks. Our key contributions include novel data reuse mechanisms and optimized Tensor Core based FFT execution. Specifically, we design a BSK tiling method for crossciphertext key reuse, a cross-iteration kernel fusion strategy for FFT/IFFT data sharing, and an optimized four-step FFT on Tensor Cores with on-the-fly Fourier matrix generation and a swizzled transposition scheme to reduce shared-memory traffic and bank conflicts. Under 128-bit security parameters, our framework achieves stable performance and outperforms the state-of-the-art ZAMA implementation, providing up to 3.01× PBS throughput and 1.96× speedup in real-world scenarios.

### ACKNOWLEDGMENTS

We thank the anonymous reviewers for their insightful comments and constructive suggestions. This work was supported by the National Key R&D Program of China under Grant Nos. 2023YFB4503200 and 2023YFB4503201, the Strategic Priority Research Program of the Chinese Academy of Sciences under Grant No. XDB0690100, and the National Natural Science Foundation of China under Grant No. 62502516.

