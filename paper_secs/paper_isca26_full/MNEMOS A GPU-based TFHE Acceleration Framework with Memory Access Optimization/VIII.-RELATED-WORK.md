# VIII. RELATED WORK

Hardware acceleration for Fully Homomorphic Encryption is an active area of research. For the TFHE scheme, various ASIC and FPGA accelerators have been proposed. ASIC-based efforts have introduced architectural innovations such as bootstrapping key unrolling[11], two-level ciphertext batching[27], and transform-domain reuse[28] to enhance throughput. More recently, the focus has expanded towards general-purpose acceleration, exemplified by [6], the first unified architecture designed to support both the CKKS and TFHE schemes, as well as the conversion between them.

Among FPGA-based TFHE accelerators, FPT [36] is particularly notable. It is the first design to employ compact fixedpoint arithmetic throughout the entire PBS process, and it introduces a method for determining the minimum bit-width required to preserve correctness. This result shows that highprecision floating-point arithmetic is not always necessary.

Beyond TFHE, a large body of work has explored FPGAand ASIC-based acceleration for BGV, BFV, and CKKS [9, 14, 29–32, 39]. In parallel, GPU acceleration for FHE has also been actively studied, mainly in the context of CKKS. For example, prior works have investigated the use of Tensor Cores to accelerate NTT [8, 10, 12], while others have focused on 32-bit RNS implementations and aggressive kernel fusion to improve performance [13]. These studies demonstrate the strong potential of GPUs for FHE acceleration, although they largely target arithmetic-oriented schemes.

More recently, several works have explored accelerating transforms such as FFT and NTT by mapping them to matrix multiplications on GPU Tensor Cores [7, 15]. However, these studies mainly rely on the native support of Tensor Cores for low-precision formats, such as half precision.

