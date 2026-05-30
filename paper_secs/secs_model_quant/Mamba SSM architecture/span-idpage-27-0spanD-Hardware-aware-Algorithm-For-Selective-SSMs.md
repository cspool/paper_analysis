# <span id="page-27-0"></span>D Hardware-aware Algorithm For Selective SSMs

Without input-dependent selectivity, SSMs can be efficiently implemented as a convolution (Dao, Fu, Saab, et al. [2023;](#page-17-2) Gu, Goel, and Ré [2022\)](#page-18-1), which leverages the fast Fourier transform (FFT) as primitive. With selectivity, SSMs are no-longer equivalent to convolution, but we leverage the parallel associative scan. While SSM scans are theoretically efficient (() FLOPs, scaling linear in ), training foundation models with selective SSMs requires them to be efficient on modern hardware (GPUs) as well. We describe how we use kernel fusion and recomputation to make SSM scan fast and memory-efficient. We evaluate the speed of our scan implementation compared to convolution and attention in Section [4.5,](#page-14-1) showing that it is up to 7× times faster than attention at sequence length 32K, and is as memory-efficient as the best attention implementation (FlashAttention).

Speed. On modern hardware accelerators (GPUs) most operations (except matrix multiply) are bounded by memorybandwidth (Dao, Fu, Ermon, et al. [2022;](#page-17-4) Ivanov et al. [2021;](#page-19-7) Williams, Waterman, and Patterson [2009\)](#page-21-10). This the case with our scan operation, and we use kernel fusion to reduce the amount of memory IOs, leading to significant speedup compared to a standard implementation.

The standard way to implement the scan algorithm in Section [3.2](#page-4-1) is to prepare the scan input , of size (, , , ) in GPU HBM (high-bandwidth memory, commonly referred to as GPU memory), call a parallel associative scan implementation to write the scan output of size (, , , ) to GPU HBM, then multiply that scan output with to produce an output of size (, , ). However, this requires the number of memory reads/writes on the order of (). We can instead fuse the discretization step, the scan, and the multiplication with into one kernel:

- 1. We read in ( + ) bytes of memory (Δ, , , ) from slow HBM to fast SRAM.
- 2. We discretize to produce , of size (, , , ) in SRAM.
- 3. We perform a parallel associative scan, yielding intermediate states of size (, , , ) in SRAM.
- 4. We multiply and sum with , producing outputs of size (, , ) and write it to HBM.

This way, we reduce IOs by a factor of () (the state dimension), which in practice speeds up the operation by 20-40 times (Section [4.5\)](#page-14-1).

For sequence length too long where we cannot fit the sequence in SRAM (which is much smaller than HBM), we split the sequences into chunks and perform the fused scan on each chunk. As long as we have the intermediate scan states, we can continue the scan with the next chunk.

Memory. We describe how we use the classical technique of recomputation to reduce the total amount of memory required to train selective SSM layers.

From the way we fuse the forward pass, we do not save the intermediate states of size (, , , ) to avoid memory blowup. However, these intermediate states are necessary for the backward pass to compute gradients. We instead recompute those intermediate states in the backward pass. Since the inputs Δ, , , and output gradient read from HBM to SRAM are of size ( + ), and the input gradients are also of size ( + ), recomputation avoids the cost of reading () elements from HBM. This means that recomputation of the SSM states in the backward pass speeds up the computation compared to storing them and reading them from HBM.

Beyond optimizing for the memory requirement of just the scan operation, we also use recomputation to optimize the memory requirement of the entire selective SSM block (input projection, convolution, activation, scan, output projection). In particular, we do not save intermediate activations that take a lot of memory but are fast to recompute (e.g. output of activation function or short convolution). As a result, the selective SSM layer has the same memory requirement as an

<span id="page-28-1"></span>Table 11: (**Induction heads**.) Models are trained on sequence length  $2^8 = 256$ , and tested on various sequence lengths of  $2^6 = 64$  up to  $2^{20} = 1048576$ .  $\checkmark$  denotes perfect generalization accuracy, while  $\checkmark$  denotes out of memory.

| Model    | Params |                | Test Accuracy (%) at Sequence Length |       |      |          |          |          |          |          |          |          |          |          |          |                 |
|----------|--------|----------------|--------------------------------------|-------|------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|-----------------|
|          |        | 2 <sup>6</sup> | 27                                   | 28    | 29   | $2^{10}$ | $2^{11}$ | $2^{12}$ | $2^{13}$ | $2^{14}$ | $2^{15}$ | $2^{16}$ | $2^{17}$ | $2^{18}$ | $2^{19}$ | 2 <sup>20</sup> |
| MHA-Abs  | 137K   | /              | 99.6                                 | 100.0 | 58.6 | 26.6     | 18.8     | 9.8      | 10.9     | 7.8      | Х        | Х        | Х        | Х        | Х        | ×               |
| MHA-RoPE | 137K   | 1              | 1                                    | 100.0 | 83.6 | 31.3     | 18.4     | 8.6      | 9.0      | 5.5      | X        | X        | X        | X        | X        | X               |
| MHA-xPos | 137K   | 1              | 1                                    | 100.0 | 99.6 | 67.6     | 25.4     | 7.0      | 9.0      | 7.8      | X        | X        | X        | X        | X        | X               |
| H3       | 153K   | 1              | 1                                    | 100.0 | 80.9 | 39.5     | 23.8     | 14.8     | 8.2      | 5.9      | 6.6      | 8.2      | 4.7      | 8.2      | 6.3      | 7.4             |
| Hyena    | 69M*   | 97.7           | 1                                    | 100.0 | ✓    | 44.1     | 12.5     | 6.6      | 5.1      | 7.0      | 5.9      | 6.6      | 6.6      | 5.9      | 6.3      | 9.8             |
| Mamba    | 74K    | 1              | 1                                    | 100.0 | 1    | 1        | 1        | 1        | 1        | ✓        | ✓        | /        | /        | ✓        | /        | 1               |

<sup>\*</sup> Most of the parameters are in learnable positional encodings.

optimized Transformer implementation with FlashAttention. In particular, each attention layer (FlashAttention) stores around 12 bytes of activations per token, an each MLP layer stores around 20 bytes of activations per token, for a total of 32 bytes ((assuming mixed-precision training in FP16 or BF16)). Each selective SSM stores around 16 bytes of activations per token. Hence two layers of selective SSMs have around the same activation memory as an attention layer and an MLP layer.

