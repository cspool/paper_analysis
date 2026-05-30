# **4 Implementation**

Comet consists of approximately 12k lines of C++ and CUDA code and 2k lines of Python. Comet provides a suite of user-friendly Python APIs and developers can seamlessly integrate the APIs into their frameworks. In production environment, Comet has been implemented in Megatron-LM for large-scale

<span id="page-7-1"></span>**Table 2** Configuration of MoE models used in experiments. The models are open-sourced on Hugging Face [\[9\]](#page-12-12). The meaning of symbols are explained in [Table 1.](#page-2-1)

|                | L  | E  | topk | N    | K     |
|----------------|----|----|------|------|-------|
| Mixtral 8x7B   | 32 | 8  | 2    | 4096 | 14336 |
| Qwen2-MoE-2.7B | 24 | 64 | 4    | 2048 | 1408  |
| Phi-3.5-MoE    | 32 | 16 | 2    | 4096 | 6400  |

MoE training. The source code will be available on GitHub.

**Optimized GEMM kernels for MoE.** Comet extensively utilizes the programming templates provided by CUTLASS to generate highly efficient GEMM kernels. Additionally, it incorporates various optimizations to minimize data movement overhead. For instance, in MoE layer 0, the row indices of the input matrix for GEMM operations must be accessed from global memory at each K iteration. By caching these row indices in registers, Comet significantly reduces the global memory access cost.

**NVSHMEM as communication library.** We employ NVSHMEM [\[24\]](#page-13-12) within kernels to support finegrained communication. NVSHMEM is a communication library designed for NVIDIA GPUs. It creates a global address space for data that spans the memory of multiple GPUs and can be accessed with finegrained GPU-initiated operations and CPU-initiated operations. Unlike NCCL [\[23\]](#page-13-13), which targets highlevel communication operations, NVSHMEM offers a more composable, low-level API that facilitates finer data access granularity within kernels.

