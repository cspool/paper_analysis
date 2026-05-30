# 1 Introduction

Scaling up the context length of Transformers (Vaswani et al., 2017) is a challenge, since the attention layer at their heart has runtime and memory requirements quadratic in the input sequence length. Ideally, we would like to go beyond the standard 2k sequence length limit to train models to understand books, high resolution images, and long-form videos. Just within the last year, there have been several language models with much longer context than before: GPT-4 (OpenAI, 2023) with context length 32k, MosaicML's MPT with context length 65k, and Anthropic's Claude with context length 100k. Emerging use cases such as long document querying and story writing have demonstrated a need for models with such long context.

To reduce the computational requirement of attention on such long context, there have been numerous methods proposed to approximate attention (Kitaev et al., 2020; Roy et al., 2021; Wang et al., 2020; Katharopoulos et al., 2020; Choromanski et al., 2020; Beltagy et al., 2020; Zaheer et al., 2020; Chen et al., 2021). Though these methods have seen some use cases, as far as we know, most large-scale training runs still use standard attention. Motivated by this, Dao et al. (2022) proposed to reorder the attention computation and leverages classical techniques (tiling, recomputation) to significantly speed it up and reduce memory usage from quadratic to linear in sequence length. This yields 2-4× wall-clock time speedup over optimized baselines, up to 10-20× memory saving, with no approximation, and as a result Flashattention has seen wide adoption in large-scale training and inference of Transformers.

However, context length increases even more, FLASHATTENTION is still not nearly as efficient as other primitives such as matrix-multiply (GEMM). In particular, while FLASHATTENTION is already  $2-4\times$  faster than a standard attention implementation, the forward pass only reaches 30-50% of the

theoretical maximum FLOPs/s of the device (Fig. 6), while the backward pass is even more challenging, reaching only 25-35% of maximum throughput on A100 GPU (Fig. 7). In contrast, optimized GEMM can reach up to 80-90% of the theoretical maximum device throughput. Through careful profiling, we observe that FLASHATTENTION still has suboptimal work partitioning between different thread blocks and warps on the GPU, causing either low-occupancy or unnecessary shared memory reads/writes.

Building on FLASHATTENTION, we propose FLASHATTENTION-2 with better parallelism and work partitioning to address these challenges.

- 1. In Section 3.1, we tweak the algorithms to reduce the number of non-matmul FLOPs while not changing the output. While the non-matmul FLOPs only account for a small fraction of the total FLOPs, they take longer to perform as GPUs have specialized units for matrix multiply, and as a result the matmul throughput can be up to 16× higher than non-matmul throughput. It is thus important to reduce non-matmul FLOPs and spend as much time as possible doing matmul FLOPs.
- 2. We propose to parallelize both the forward pass and backward pass along the sequence length dimension, in addition to the batch and number of heads dimension. This increases occupancy (utilization of GPU resources) in the case where the sequences are long (and hence batch size is often small).
- 3. Even within one block of attention computation, we partition the work between different warps of a thread block to reduce communication and shared memory reads/writes.

In Section 4, we empirically validate that FLASHATTENTION-2 yields significant speedup compared to even FLASHATTENTION. Benchmarks on different settings (with or without causal mask, different head dimensions) show that FLASHATTENTION-2 achieves around 2× speedup over FLASHATTENTION, reaching up to 73% of the theoretical max throughput in the forward pass, and up to 63% of the theoretical max throughput in the backward pass. During LLM inference, FLASHATTENTION-2's kernel is up to 7× faster than the attention kernel from FasterTransformer. When used end-to-end to train GPT-style models, we reach training speed of up to 225 TFLOPs/s per A100 GPU.

#### 2 Background

We provide some background on the performance characteristics and execution model of GPUs. We also describe the standard implementation of attention, as well as FLASHATTENTION.

#### 2.1 HARDWARE CHARACTERISTICS

**GPU** performance characteristics. The GPU consists of compute elements (e.g., floating point arithmetic units) and a memory hierarchy. Most modern GPUs contain specialized units to accelerate matrix multiply in low-precision (e.g., Tensor Cores on Nvidia GPUs for FP16/BF16 matrix multiply). The memory hierarchy comprise of high bandwidth memory (HBM), and on-chip SRAM (aka shared memory). As an example, the A100 GPU has 40-80GB of high bandwidth memory (HBM) with bandwidth 1.5-2.0TB/s and 192KB of on-chip SRAM per each of 108 streaming multiprocessors with bandwidth estimated around 19TB/s (Jia et al., 2018; Jia and Van Sandt, 2021). As the L2 cache is not directly controllable by the programmer, we focus on the HBM and SRAM for the purpose of this discussion.

**Execution Model.** GPUs have a massive number of threads to execute an operation (called a kernel). Threads are organized into thread blocks, which are scheduled to run on streaming multiprocessors (SMs). Within each thread blocks, threads are grouped into warps (a group of 32 threads). Threads within a warp can communicate by fast shuffle instructions or cooperate to perform matrix multiply. Warps within a thread block can communicate by reading from / writing to shared memory. Each kernel loads inputs from HBM to registers and SRAM, computes, then writes outputs to HBM.

#### <span id="page-1-1"></span>2.2 STANDARD ATTENTION IMPLEMENTATION

Given input sequences  $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d}$  where N is the sequence length and d is the head dimension, we want to compute the attention output  $\mathbf{O} \in \mathbb{R}^{N \times d}$ :

$$\mathbf{S} = \mathbf{O}\mathbf{K}^{\mathsf{T}} \in \mathbb{R}^{N \times N}$$
.  $\mathbf{P} = \operatorname{softmax}(\mathbf{S}) \in \mathbb{R}^{N \times N}$ .  $\mathbf{O} = \mathbf{P}\mathbf{V} \in \mathbb{R}^{N \times d}$ .

where softmax is applied row-wise. For multi-head attention (MHA), this same computation is performed in parallel across many heads, and parallel over the batch dimension (number of input sequences in a batch).

<span id="page-1-0"></span> $<sup>^1</sup>$ For clarity of exposition, we omit the scaling of  $\mathbf{Q}\mathbf{K}^{\top}$  (typically by 1/d), and optionally elementwise masking on  $\mathbf{S}$  and/or dropout applied to  $\mathbf{P}$ 

The backward pass of attention proceeds as follows. Let  $d\mathbf{O} \in \mathbb{R}^{N \times d}$  be the gradient of  $\mathbf{O}$  with respect to some loss function. Then by the chain rule (aka backpropagation):

$$\mathbf{dV} = \mathbf{P}^{\top} \mathbf{dO} \in \mathbb{R}^{N \times d} \qquad \mathbf{dP} = \mathbf{dOV}^{\top} \in \mathbb{R}^{N \times N}$$

$$\mathbf{dS} = \operatorname{dsoftmax}(\mathbf{dP}) \in \mathbb{R}^{N \times N} \qquad \mathbf{dQ} = \mathbf{dSK} \in \mathbb{R}^{N \times d} \qquad \mathbf{dK} = \mathbf{dS}^{\top} \mathbf{Q} \in \mathbb{R}^{N \times d}.$$

where dsoftmax is the gradient (backward pass) of softmax applied row-wise. One can work out that if  $p = \operatorname{softmax}(s)$  for some vector s and p, then with output gradient dp, the input gradient  $ds = (\operatorname{diag}(p) - pp^{\top})dp$ .

Standard attention implementations materialize the matrices S and P to HBM, which takes  $O(N^2)$  memory. Often  $N \gg d$  (typically N is on the order of 1k–8k and d is around 64–128). The standard attention implementation (1) calls the matrix multiply (GEMM) subroutine to multiply  $S = QK^T$ , writes the result to HBM, then (2) loads S from HBM to compute softmax and write the result P to HBM, and finally (3) calls GEMM to get O = PV. As most of the operations are bounded by memory bandwidth, the large number of memory accesses translates to slow wall-clock time. Moreover, the required memory is  $O(N^2)$  due to having to materialize S and S. Moreover, one has to save S0 for the backward pass to compute the gradients.

#### <span id="page-2-0"></span>2.3 FLASHATTENTION

To speed up attention on hardware accelerators such as GPU, (Dao et al., 2022) proposes an algorithm to reduce the memory reads/writes while maintaining the same output (without approximation).

#### 2.3.1 FORWARD PASS

FLASHATTENTION applies the classical technique of tiling to reduce memory IOs, by (1) loading blocks of inputs from HBM to SRAM, (2) computing attention with respect to that block, and then (3) updating the output without writing the large intermediate matrices S and P to HBM. As the softmax couples entire rows or blocks of row, online softmax (Milakov and Gimelshein, 2018; Rabe and Staats, 2021) can split the attention computation into blocks, and rescale the output of each block to finally get the right result (with no approximation). By significantly reducing the amount of memory reads/writes, FLASHATTENTION yields 2-4× wall-clock speedup over optimized baseline attention implementations.

We describe the online softmax technique (Milakov and Gimelshein, 2018) and how it is used in attention (Rabe and Staats, 2021). For simplicity, consider just one row block of the attention matrix S, of the form  $\begin{bmatrix} S^{(1)} & S^{(2)} \end{bmatrix}$  for some matrices  $S^{(1)}, S^{(2)} \in \mathbb{R}^{B_r \times B_c}$ , where  $B_r$  and  $B_c$  are the row and column block sizes. We want to compute softmax of this row block and multiply with the value, of the form  $\begin{bmatrix} V^{(1)} \\ V^{(2)} \end{bmatrix}$  for some matrices  $V^{(1)}, V^{(2)} \in \mathbb{R}^{B_c \times d}$ . Standard softmax would compute:

$$\begin{split} & m = \max(\operatorname{rowmax}(\mathbf{S}^{(1)}), \operatorname{rowmax}(\mathbf{S}^{(2)})) \in \mathbb{R}^{B_r} \qquad \ell = \operatorname{rowsum}(e^{\mathbf{S}^{(1)} - m}) + \operatorname{rowsum}(e^{\mathbf{S}^{(2)} - m}) \in \mathbb{R}^{B_r} \\ & \mathbf{P} = \begin{bmatrix} \mathbf{P}^{(1)} & \mathbf{P}^{(2)} \end{bmatrix} = \operatorname{diag}(\ell)^{-1} \begin{bmatrix} e^{\mathbf{S}^{(1)} - m} & e^{\mathbf{S}^{(2)} - m} \end{bmatrix} \in \mathbb{R}^{B_r \times 2B_c} \\ & \mathbf{O} = \begin{bmatrix} \mathbf{P}^{(1)} & \mathbf{P}^{(2)} \end{bmatrix} \begin{bmatrix} \mathbf{V}^{(1)} \\ \mathbf{V}^{(2)} \end{bmatrix} = \operatorname{diag}(\ell)^{-1} e^{\mathbf{S}^{(1)} - m} \mathbf{V}^{(1)} + e^{\mathbf{S}^{(2)} - m} \mathbf{V}^{(2)} \in \mathbb{R}^{B_r \times d}. \end{split}$$

Online softmax instead computes "local" softmax with respect to each block and rescale to get the right output at the end:

$$\begin{split} &m^{(1)} = \operatorname{rowmax}(\mathbf{S}^{(1)}) \in \mathbb{R}^{B_r} \qquad \ell^{(1)} = \operatorname{rowsum}(e^{\mathbf{S}^{(1)} - m^{(1)}}) \in \mathbb{R}^{B_r} \\ &\tilde{\mathbf{P}}^{(1)} = \operatorname{diag}(\ell^{(1)})^{-1} e^{\mathbf{S}^{(1)} - m^{(1)}} \in \mathbb{R}^{B_r \times B_c} \qquad \mathbf{O}^{(1)} = \tilde{\mathbf{P}}^{(1)} \mathbf{V}^{(1)} = \operatorname{diag}(\ell^{(1)})^{-1} e^{\mathbf{S}^{(1)} - m^{(1)}} \mathbf{V}^{(1)} \in \mathbb{R}^{B_r \times d} \\ &m^{(2)} = \max(m^{(1)}, \operatorname{rowmax}(\mathbf{S}^{(2)})) = m \\ &\ell^{(2)} = e^{m^{(1)} - m^{(2)}} \ell^{(1)} + \operatorname{rowsum}(e^{\mathbf{S}^{(2)} - m^{(2)}}) = \operatorname{rowsum}(e^{\mathbf{S}^{(1)} - m}) + \operatorname{rowsum}(e^{\mathbf{S}^{(2)} - m}) = \ell \\ &\tilde{\mathbf{P}}^{(2)} = \operatorname{diag}(\ell^{(2)})^{-1} e^{\mathbf{S}^{(2)} - m^{(2)}} \\ &\mathbf{O}^{(2)} = \operatorname{diag}(\ell^{(1)} / \ell^{(2)}) \mathbf{O}^{(1)} + \tilde{\mathbf{P}}^{(2)} \mathbf{V}^{(2)} = \operatorname{diag}(\ell^{(2)})^{-1} e^{s^{(1)} - m} \mathbf{V}^{(1)} + \operatorname{diag}(\ell^{(2)})^{-1} e^{s^{(2)} - m} \mathbf{V}^{(2)} = \mathbf{O}. \end{split}$$

We show how FLASHATTENTION uses online softmax to enable tiling (Fig. 1) to reduce memory reads/writes.

<span id="page-3-1"></span>![](_page_3_Figure_0.jpeg)

Figure 1: Diagram of how FLASHATTENTION forward pass is performed, when the key K is partitioned into two blocks and the value V is also partitioned into two blocks. By computing attention with respect to each block and rescaling the output, we get the right answer at the end, while avoiding expensive memory reads/writes of the intermediate matrices S and P. We simplify the diagram, omitting the step in softmax that subtracts each element by the row-wise max.

#### 2.3.2 BACKWARD PASS

In the backward pass, by re-computing the values of the attention matrices  $\bf S$  and  $\bf P$  once blocks of inputs  $\bf Q$ ,  $\bf K$ ,  $\bf V$  are already loaded to SRAM, FLASHATTENTION avoids having to store large intermediate values. By not having to save the large matrices  $\bf S$  and  $\bf P$  of size  $N\times N$ , FLASHATTENTION yields  $10\text{-}20\times$  memory saving depending on sequence length (memory required in linear in sequence length N instead of quadratic). The backward pass also achieves  $2\text{-}4\times$  wall-clock speedup due to reduce memory reads/writes.

The backward pass applies tiling to the equations in Section 2.2. Though the backward pass is simpler than the forward pass conceptually (there is no softmax rescaling), the implementation is significantly more involved. This is because there are more values to be kept in SRAM to perform 5 matrix multiples in the backward pass, compared to just 2 matrix multiples in the forward pass.

