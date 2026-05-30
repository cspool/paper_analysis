# *B. Residual Kernel*

A primary challenge in low-bit KV-cache design is supporting diverse quantization algorithms—especially differing scaling granularities (e.g., tensor-wise, channel-wise)—without sacrificing performance. Quantization involves reductions and element-wise operations to compute scale and zero-point, followed by bit-packing; during decoding these must run online, adding runtime overhead and risking misalignment with the rigid layouts expected by Tensor Cores. To address this, we design the *Residual Kernel* with two key optimizations:

(1) Partitioning KV cache based on residual block size. During prefill with context length L, we split the KV cache based on a Tensor Cores-aligned residual block size N<sup>r</sup> (see Eq. 1). The first N<sup>p</sup> = L − (L mod Nr) entries are quantized and packed into the low-bit KV cache using a fused quantization and packing operation. The remaining KV Tensor with size res\_len = L mod N<sup>r</sup> are stored in the half-precision residual KV cache. At each decode step, the newly generated K, V tensors are appended to the residual cache and used for attention computation. This cache grows incrementally until it reaches the residual block size Nr. Once per token generation, the Residual Kernel computes attention using the half-precision residual KV cache and optionally quantizes it (when res\_len = Nr) into packed format.

With this KV cache partitioning during decoding, we can naturally perform channel-wise quantization along the seq len and tensor-wise quantization along the hidden dimension within the residual block.

(2) Optimizing reduction with warp-level instructions. As shown in Fig. 7 (mid), once the half-precision KV data is computed, it remains in registers as Tensor Cores fragments—structured in the native interleaved layout used by mma operations. To efficiently compute the quantization parameters (scale and zero-point), we first perform thread-level reductions to obtain local min/max statistics within each group.

These local results are then aggregated across the warp using the PTX instruction <code>\_\_shfl\_xor\_sync</code>, enabling efficient warp-level reduction without shared memory. When the warp repetition factor  $W_n>1$ , we introduce a small shared memory buffer to coordinate the final reduction across warps.

After computing the quantization parameters, each thread performs in-register quantization and packs the low-bit values into INT16 format. This avoids extra memory movement and keeps data in a computation-ready state. To minimize overhead, both the scale and zero-point are stored in a compact half2 format, enabling efficient memory access and fused multiply-add during dequantization in the decode phase.

## C. Packing Kernel

Another challenge is the auxiliary low-bit metadata (scale and zero-point), which increases memory traffic, while dequantization still runs on CUDA cores. Without careful scheduling, this disrupts the load–compute pipeline and prevents overlap with Tensor Core operations. We therefore design a fine-grained asynchronous pipeline: CUDA cores handle dequantization, Tensor Cores execute matrix multiplications, and both are orchestrated to overlap with memory transfers through the GPU hierarchy—enabling efficient mixed-precision computation.

(1) Optimizing asynchronous data movement. From Global to Shared Memory, we follow FlashAttention [6] via block-wise tiling [32] and strategic recomputation. It processes input matrices  $Q \in \mathbb{R}^{T_m \times d}$ ,  $K, V \in \mathbb{R}^{T_n \times d}$  in tiles within shared memory, using block sizes  $T_m$  and  $T_n$ . The number of key-value tiles is  $C_n = \lceil L/T_n \rceil$ .

To efficiently manage quantization parameters, we introduce dedicated shared memory buffers for quantization parameter  $K_{pack}$  params  $(K_p)$  and  $V_{pack}$  param  $(V_p)$ , facilitating efficient tiling for memory copy. These buffers store scale and zeros in the half2 format, allowing them to be loaded in a single instruction.

The shape of  $K_p$  is determined by the quantization granularity setting, and the  $V_p$  follows a Tensor-wise layout:

- Channel-wise:  $(T_n/\text{group\_size}, d)$ .
- **Tensor-wise:**  $(T_n, d/\text{group\_size})$ .

To achieve optimal memory overlapping, all global-toshared memory transfers are executed asynchronously using the cp.async intrinsic, ensuring efficient pipeline execution, as shown in Fig. 7 (right). We optimize memory transactions using instructions with different caching strategies:

- cp.async.cg: Used for Q,  $K_{\text{pack}}$ , and  $V_{\text{pack}}$ , which cache only in global memory as they are not reused within the same kernel.
- cp.async.ca: Applied to  $K_p$  and  $V_p$ , ensuring smaller byte-level alignment for fine-grained memory access.

In Hopper architecture, we follow FA3, leveraging the tma.copy instruction for data loading. This facilitates warp-specialized scheduling, improving data locality and reducing memory latency across multiple warps.

From Shared Memory to Register, we use the PTX instruction ldmatrix to efficiently load  $K_{\rm pack}$ ,  $V_{\rm pack}$  and sAcc from shared memory into registers with the Tensor Cores tiling layout. To eliminate bank conflicts, we use a sizzling scheme [5] defined as:

$$col_{id} = row_{id} \oplus col_{id} \tag{2}$$

achieve bank conflict-free access. Additionally, we restructure the shared memory layout of  $K_p$  and  $V_p$  to further reduce bank conflict and maximize throughput efficiency.

(2) Asynchronous pipeline for overlapping CUDA Cores and Tensor Cores. To fully utilize both CUDA cores and Tensor Cores, we implement a register-level, asynchronous pipeline that overlaps computation with memory operations. In this pipeline, shared-memory loads via ldmatrix and dequantization (Dequant) run concurrently with Tensor Core matrix multiplications (mma) under the SM warp scheduler.

As shown in Fig. 7 (right), while the i-th slice is being processed by mma on Tensor Cores, the (i+1)-th slice is simultaneously loaded from shared memory (ldmatrix) and dequantized. This sustains a continuous producer—consumer flow, improving instruction throughput and maximizing utilization of both CUDA cores and Tensor Cores.

#### D. Latest Architectures Support

While the design presented thus far effectively targets pre-Hopper architectures (e.g., Ampere), newer generations introduce distinct hardware features that require tailored optimization strategies. Below, we detail how our approach adapts to leverage the specialized instructions and native data formats of the Hopper and Blackwell architectures.

- (1) Unlocking Hopper for warpgroup acceleration capabilities via smart uses of PTX-level instructions. Hopper Tensor Cores, increasingly introduce Warpgroup Matrix Multiply-Accumulate (wgmma) instruction. This instruction however imposes a key constraint: in a matrix multiplication C=AB, only A and C can be sourced from registers, while B must reside in shared memory. This presents a challenge for low-bit quantized data, as values are typically upconverted to FP16 in registers before computation. To resolve this, we leverage Hopper's STSM PTX instruction to store dequantized FP16 values in shared memory efficiently, accessible for wgmma\_SS operations. Remarkably, the asynchronous nature of WGMMA overlaps storage with computation, optimizing performance.
- (2) Accelerating Blackwell with native low-precision format. The Blackwell architecture introduces native support for low-precision tensor operations, eliminating the need for explicit dequantization. Consequently, the lop3-based register remapping described earlier is bypassed in favor of direct execution. We target Blackwell's low-precision mma instructions—specifically those supporting the micro-scaling

![](_page_8_Figure_0.jpeg)

Fig. 8: Kernel performance with mxfp4 on Blackwell architectures.

formats (e.g., mxfp4 / nvfp4)—to execute GEMM operations directly on packed 4-bit data. While these instructions enforce rigid layout constraints for both the packed values and their block-scaling factors, the layout transformation strategy proposed in Section IV-A is designed to be layout-agnostic. It automatically aligns the packed KV data with the hardware-mandated format, ensuring seamless integration with Black-well's native tensor pipelines.

#### VI. EVALUATION

In this section, we comprehensively evaluate BitDecoding against state-of-the-art approaches and systems. Our evaluation highlights the following key results:

- 1) BitDecoding outperforms FP16 FlashDecoding-v2 by significant margins across GPU generations, achieving speedups of up to 8.6× on Blackwell (using native MXFP4), 8.0× on Hopper, and 7.5× on Ada architectures, while surpassing the state-of-the-art low-bit system QServe by up to 4.3× (Section VI-A).
- 2) In end-to-end long-context inference, BitDecoding reduces single-batch latency by 3x (on LLaMA-3.1-8B with 128K context) and achieves over 4x higher serving throughput than QServe, demonstrating superior scalability in GQA settings where prior CUDA Core-only methods degrade (Section VI-B).
- 3) BitDecoding preserves near-FP16 accuracy while deriving significant performance gains from each system component, demonstrating only a 0.2% accuracy degradation with 4-bit quantization, while our ablation study confirms that every design module contributes to the overall speedup (Section VI-C).

## A. Kernels Performance Across GPU Architectures

**Kernels Settings.** Since different LLM serving scenarios require varying workloads and attention kernel designs, we evaluate performance under the following three representative settings:

- **Single:** A scenario where batch\_size = 1, representing inference for edge users with long context.
- **Batches:** A setting with a larger batch\_size, maintaining the same input length while applying simple padding.

![](_page_8_Figure_12.jpeg)

Fig. 9: Kernel performance on Hopper (H100).

• Page: A high-throughput scenario where a larger batch\_size is managed using the page management technique [15].

**Baselines.** We compare BitDecoding against several representative attention kernel implementations. For FP16 KV cache, we use FlashDecoding [6], [25]—a split-partitioned variant of FlashAttention optimized for long-context decoding—as our baseline for speedup normalization. For low-bit KV cache, we evaluate Kivi [18], a non-fused kernel supporting 4-bit and 2-bit quantization; Atom [37] and QServe [16], both fused-kernel implementations with CUDA Cores-only approach and supporting 4-bit cache with page management. Notably, Atom does not support GQA.

**Quantization Settings.** We evaluate BitDecoding under various quantization configurations, supporting 4-bit and 2-bit Key tensors with both Channel-wise (KC) and Tensor-wise (KT) schemes.

**Results on MXFP4 (RTX5090, RTX PRO 6000).** The Blackwell architecture provides native support for low-precision data formats, eliminating on-the-fly dequantization overhead while delivering very high compute throughput on low-bit operations. As shown in Fig. 8a, BitDecoding achieves remarkable performance, reaching up to  $8.6\times$  speedup in batched scenarios and over  $4.3\times$  in single-batch long-context decoding (128k), significantly outpacing the non-fused attention baseline. Similarly, Fig. 8b demonstrates that the RTX PRO 6000 attains substantial gains, peaking at  $6.5\times$  speedup with large batch sizes.

Results on Advanced Tensor Cores Acceleration (H100). Newer GPU architectures often introduce advanced compute

![](_page_9_Figure_0.jpeg)

Fig. 10: Kernel performance on RTX4090.

instructions that significantly accelerate kernel execution. As illustrated in Fig. 9, FlashDecoding-v3, optimized for Hopper Tensor Cores, delivers notable performance gains over its v2 counterpart. While BitDecoding-v2 reaches up to  $4.1\times$  speedup, the v3 implementation further boosts performance to  $8.0\times$ . This is enabled by BitDecoding's use of Hopper's wgmma and asynchronous memory instructions, ensuring high Tensor Cores utilization even in mixed-precision settings.

Results on Bandwidth-constrained GPU (RTX 4090). Leveraging low-precision data is critical for accelerating inference on bandwidth-constrained GPUs. As shown in Fig. 10, BitDecoding achieves roughly  $4\times$  (4-bit) and over  $7\times$  (2-bit) speedups over FlashDecoding-v2 in Single and Batches settings, gains that stem directly from alleviating DRAM bottlenecks via low-bit KV caching.

BitDecoding significantly outperforms baselines across all scenarios; unlike the non-fused KIVI, which relies on separate kernels and suffers severe degradation in GQA, BitDecoding's fully fused design maintains high efficiency. In Page settings, it surpasses fused CUDA-core baselines: for MHA, BitDecoding achieves over  $6\times$  speedup compared to QServe's  $3.5\times$ . Crucially, in compute-intensive GQA, it maintains a  $3\times$  speedup while QServe drops to  $1.4\times$ , confirming that leveraging Tensor Cores provides robust acceleration where CUDA-only approaches falter.

Results on High-Bandwidth GPU (A100). On architectures with high memory bandwidth like the A100, computation pressure becomes more pronounced, as performance bottlenecks shift from memory access to compute utilization—especially when kernel designs fail to fully exploit available compute resources. As shown in Fig. 11, both KIVI and QServe suffer from poor performance—KIVI due to its non-fused kernel design, and QServe due to underutilization of Tensor Cores—even performing worse than the FP16 baseline. In contrast, BitDecoding consistently outperforms all baselines

across workloads, achieving up to  $3\times$  speedup, thanks to its efficient utilization of Tensor Cores and fused execution pipeline. An interesting observation is that the performance gap between 4-bit and 2-bit variants narrows on A100, as the increased DRAM bandwidth reduces memory bottlenecks and shifts the performance balance toward compute-bound execution.

## B. Performance across LLMs Inference Systems

**Model settings.** We evaluate on a range of LLMs, including LLaMA-2-7B, LLaMA-3.1-8B, LLaMA-3.1-70B, Qwen3-8B, and Qwen3-14B. Among them, only LLaMA-2-7B adopts MHA, while the others use GQA. All models are run on a single A100 GPU, except LLaMA-3.1-70B, which is evaluated on 8×A100 GPUs.

**Quantization settings.** We choose channel-wise quantization for LLMs KV cache as it brings better accuracy and aligns with the Kivi.

Compared with Non-fused Attention. As illustrated in Fig. 12, in the Single setting, BitDecoding achieves up to 3.3× speedup at a 128K context length, where KV cache loading becomes the dominant bottleneck in LLMs inference. In contrast, Kivi suffers from limited scalability and encounters out-of-memory (OOM) failures at 128K due to the lack of block-tiling kernel support. For the Batches setting, BitDecoding significantly outperforms KIVI in throughput: BitDecoding-KC-4 and KC-2 reach up to 900 and 1200 tokens/s, respectively, while KIVI-4 and KIVI-2 peak below 700 tokens/s.

Compared with CUDA Cores-only fused Attention. We compare BitDecoding with Qserve for page-setting inference, as Qserve supports both MHA and GQA attention structures. The maximum throughput is evaluated under the largest batch sizes available within GPU memory. As illustrated in Fig. 13, Qserve achieves higher throughput than FlashDecoding-v2 on LLaMA-2-7B but suffers from degraded performance on

![](_page_10_Figure_0.jpeg)

Fig. 11: Kernel performance on A100.

![](_page_10_Figure_2.jpeg)

Fig. 12: Comparing Kivi with (a) end-to-end generation time and (b) decoding throughput.

![](_page_10_Figure_4.jpeg)

Fig. 13: Comparing Oserve with decoding throughput.

all other models due to inefficiencies in handling GQA. In contrast, BitDecoding consistently outperforms QServe across both LLaMA and Qwen architectures, under both single-GPU and multi-GPU settings, achieving more than  $2\times$  higher maximum throughput compared to QServe.

## C. Accuracy, Overhead and Performance Breakdown

Accuracy analysis. As shown in Table I, we evaluate throughput and accuracy across different bit widths. The 2-bit quantization reduces memory consumption significantly, enabling larger batch sizes and achieving a  $4.25\times$  higher throughput compared to FP16. Meanwhile, the 4-bit quantization achieves a  $2.98\times$  speedup while maintaining near full-precision accuracy with only a minimal 0.2% degradation. These results highlight the trade-off, with 4-bit quantization offering balance and 2-bit maximizing throughput at a slight accuracy cost.

**Half-precision Residual Kernel Overhead.** Half-precision residual KV Cache would introduce quite a small portion

TABLE I: Efficiency and accuracy tradeoff with low-bit KV cache. We use Llama-3.1-8B-Instruct with  $seq\_len = 32K$ , and evaluate average accuracy on longbench [3].

| KV Cache     | Throughput                         | Longbench Acc                  |
|--------------|------------------------------------|--------------------------------|
| FP16         | 49.25                              | 48.25                          |
| INT4<br>INT2 | 147.21 (+2.98x)<br>209.48 (+4.25x) | 48.16 (-0.2%)<br>47.38 (-2.7%) |

TABLE II: Latency (ms) comparison of quantization and packing during inference.

| Inference Phase | Marlin | Ladder | BitDecoding |
|-----------------|--------|--------|-------------|
| Prefill         | 58.02  | 4.79   | 0.0599      |
| Decode          | 0.41   | 0.65   | 0.008       |

TABLE III: Impact of cooperative softmax and warps on performance and validity.

| $W_n$ | Coop. Soft   | Latency (ms) | TCs Utilization (%) | Valid        |
|-------|--------------|--------------|---------------------|--------------|
| 1     | ×            | 3.746        | 10.91               | ✓            |
| 4     | ×            | 0.610        | 19.71               | ×            |
| 4     | $\checkmark$ | 0.613        | 19.66               | $\checkmark$ |

memory overhead as  $seq\_len >> N_r$ , while  $seq\_len$  would be more than 32K and  $N_r$  is always less than 256. The half-precision residual KV cache introduces only a slight runtime overhead due to an extra kernel launch, as shown in Fig. 14. Moreover, this overhead becomes increasingly negligible as the sequence length grows, since the residual portion constitutes a smaller fraction of the total KV cache.

Quantization and Packing Overhead. We evaluate the latency of quantization and packing under a sequence length of  $seq\_len=128K$ , comparing BitDecoding with Marlin [9] and Ladder [33]. As shown in Table II, the pre-transformation and packing step in previous mixed-precision computing methods introduce significant overhead, which cannot be ignored. Our kernel incurs minimal overhead after the Prefill phase, primarily due to kernel launch overhead. Moreover, during decoding, we achieves nearly negligible overhead, as it is fully fused into kernel computation.

**Dequantization Overhead.** Fig. 15a illustrates the high computational overhead of dequantization in Atom and QServe, consuming nearly half the kernel execution time. In

![](_page_11_Figure_0.jpeg)

Fig. 14: Runtime overhead of the residual KV cache.

![](_page_11_Figure_2.jpeg)

Fig. 15: Dequantization overhead analysis.

contrast, BitDecoding significantly reduces this overhead to less than 15% (4-bit) and 35% (2-bit), thanks to better Tensor Cores overlap.

A further microbenchmark comparing Atom and BitDecoding (Fig. 15b) reveals BitDecoding's superior memory throughput from effective Tensor Core usage. Conversely, Atom relies heavily on CUDA cores, increasing pressure on FMA and ALU operations.

Multi-warps Cooperative Softmax Overhead. Table III shows that increasing  $W_n$  improves Tensor Cores utilization and reduces latency, but breaks correctness without cooperative softmax. Enabling cooperative softmax restores correctness with only 0.5% overhead. Although it introduces shared memory access, the overhead is minimal since low-bit data reduces memory bandwidth pressure and shifts the kernel from memory-bound to compute-bound.

**BreakDown Analysis.** To further analyze the performance gains of BitDecoding, we decompose our optimizations in Fig. 16. Following [2], we use a continuous-packing baseline that quantizes and packs the KV cache at every generation step, which introduces substantial overhead and requires manual effort to maintain valid layouts. In contrast, our layout design automatically induces Tensor Core–compatible layouts for arbitrary low-bit formats, fully unlocking the compute potential of Tensor Cores. On top of this, the warp-parallelism strategy contributes significant additional speedups, while the pipeline optimizations further enhance end-to-end performance.

#### VII. RELATED WORKS

a) KV Cache Quantization Algorithms: KV cache quantization reduces memory usage in LLMs with long contexts

![](_page_11_Figure_10.jpeg)

Fig. 16: Breakdown of BitDecoding optimizations across architectural generations.

while maintaining performance. Recent works explore 4-bit, 2-bit, and even 1-bit KV cache quantization, aiming to push the limits of compression. Methods like KIVI [18], Gear [13], and KVQuant [12] use per-channel quantization to handle key-value outliers, while RotateKV [27] applies rotation to smooth channel-wise distributions. Although effective at higher compression ratios, these methods lack efficient system implementations, leading to suboptimal performance.

b) Mixed-precision Matrix Multiplication: Low-bit weight and low-bit KV cache in LLMs create a unique requirement for mixed-precision matrix multiplication (mpGEMM), where one input matrix is in lower precision (e.g., INT4/2/1) while the other matrix remains in higher precision (e.g., FP16/8). Optimized kernels like Ladder [33] and Marlin [9] improve performance via layout transformations and efficient dequantization. However, these methods require pre-packing and pre-transforming weights, limiting applicability to low-bit KV cache in autoregressive decoding.

c) System Implementation for Low-bit KV Cache: KIVI [31] uses Triton with separate kernels for low-bit KV Cache implementation. Atom [37] integrates quantization within the preceding linear layer, while QServe [16] fuses quantization directly into FlashAttention kernels. However, they both rely on GEMV operations with fused multiply—add (FMA) instructions, missing Tensor Core acceleration.

#### VIII. CONCLUSION

In this paper, we introduce BitDecoding, a GPU-optimized computing framework supporting low-bit KV cache decoding with Tensor Cores. We effectively resolve the layout mismatches imposed by rigid hardware patterns and propose finegrained optimizations to maximize computational utilization. Extensive evaluations demonstrate that BitDecoding achieves speedups of up to  $8.6\times$  on Blackwell,  $8.9\times$  on Hopper,  $7.5\times$  on Ada, and  $4.8\times$  on Ampere architectures compared to FP16 FlashDecoding-v2. Furthermore, on LLaMA-3.1-8B with a 128K sequence length, BitDecoding reduces singlebatch decoding latency by  $3\times$  and improves serving throughput by 4× over state-of-the-art methods. By providing a high-performance system foundation, BitDecoding opens new avenues for algorithm-system co-design—paving the way for efficient, near-lossless test-time scaling in next-generation long-context models.

## REFERENCES

- [1] J. Ainslie, J. Lee-Thorp, M. De Jong, Y. Zemlyanskiy, F. Lebron, and ´ S. Sanghai, "Gqa: Training generalized multi-query transformer models from multi-head checkpoints," *arXiv preprint arXiv:2305.13245*, 2023.
- [2] S. Ashkboos, A. Mohtashami, M. L. Croci, B. Li, P. Cameron, M. Jaggi, D. Alistarh, T. Hoefler, and J. Hensman, "Quarot: Outlier-free 4-bit inference in rotated llms," *Advances in Neural Information Processing Systems*, vol. 37, pp. 100 213–100 240, 2024.
- [3] Y. Bai, X. Lv, J. Zhang, H. Lyu, J. Tang, Z. Huang, Z. Du, X. Liu, A. Zeng, L. Hou, Y. Dong, J. Tang, and J. Li, "LongBench: A bilingual, multitask benchmark for long context understanding," in *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*. Bangkok, Thailand: Association for Computational Linguistics, Aug. 2024, pp. 3119–3137. [Online]. Available: https://aclanthology.org/2024.acl-long.172
- [4] Y. Chang, K. Lo, T. Goyal, and M. Iyyer, "Booookscore: A systematic exploration of book-length summarization in the era of llms," *arXiv preprint arXiv:2310.00785*, 2023.
- [5] N. Corporation, "Cutlass: Cuda templates for linear algebra subroutines and solvers," 2024, 3.6). [Online]. Available: https://github.com/ NVIDIA/cutlass
- [6] T. Dao, "FlashAttention-2: Faster attention with better parallelism and work partitioning," in *International Conference on Learning Representations (ICLR)*, 2024.
- [7] Y. Ding, L. L. Zhang, C. Zhang, Y. Xu, N. Shang, J. Xu, F. Yang, and M. Yang, "Longrope: Extending llm context window beyond 2 million tokens," *arXiv preprint arXiv:2402.13753*, 2024.
- [8] G. Fan, M. Zhang, F. Zheng, S. Fan, T. Zhou, X. Deng, W. Tang, L. Kong, Y. Song, and S. Yan, "Warpdrive: Gpu-based fully homomorphic encryption acceleration leveraging tensor and cuda cores," in *2025 IEEE International Symposium on High Performance Computer Architecture (HPCA)*. IEEE, 2025, pp. 1187–1200.
- [9] E. Frantar, R. L. Castro, J. Chen, T. Hoefler, and D. Alistarh, "Marlin: Mixed-precision auto-regressive parallel inference on large language models," *arXiv preprint arXiv:2408.11743*, 2024.
- [10] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan *et al.*, "The llama 3 herd of models," *arXiv preprint arXiv:2407.21783*, 2024.
- [11] D. Guo, D. Yang, H. Zhang, J. Song, R. Zhang, R. Xu, Q. Zhu, S. Ma, P. Wang, X. Bi *et al.*, "Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning," *arXiv preprint arXiv:2501.12948*, 2025.
- [12] C. Hooper, S. Kim, H. Mohammadzadeh, M. W. Mahoney, Y. S. Shao, K. Keutzer, and A. Gholami, "Kvquant: Towards 10 million context length llm inference with kv cache quantization," *arXiv preprint arXiv:2401.18079*, 2024.
- [13] H. Kang, Q. Zhang, S. Kundu, G. Jeong, Z. Liu, T. Krishna, and T. Zhao, "Gear: An efficient kv cache compression recipefor nearlossless generative inference of llm," *arXiv preprint arXiv:2403.05527*, 2024.
- [14] Y. J. Kim, R. Henry, R. Fahim, and H. H. Awadalla, "Who says elephants can't run: Bringing large scale moe models into cloud scale production," *arXiv preprint arXiv:2211.10017*, 2022.
- [15] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica, "Efficient memory management for large language model serving with pagedattention," in *Proceedings of the 29th ACM Symposium on Operating Systems Principles*, 2023. [Online]. Available: https://dl.acm.org/doi/10.1145/3600006.3613165
- [16] Y. Lin, H. Tang, S. Yang, Z. Zhang, G. Xiao, C. Gan, and S. Han, "Qserve: W4a8kv4 quantization and system co-design for efficient llm serving," *arXiv preprint arXiv:2405.04532*, 2024.
- [17] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan *et al.*, "Deepseek-v3 technical report," *arXiv preprint arXiv:2412.19437*, 2024.
- [18] Z. Liu, J. Yuan, H. Jin, S. Zhong, Z. Xu, V. Braverman, B. Chen, and X. Hu, "Kivi: A tuning-free asymmetric 2bit quantization for kv cache," *arXiv preprint arXiv:2402.02750*, 2024.
- [19] W. Luo, R. Fan, Z. Li, D. Du, Q. Wang, and X. Chu, "Benchmarking and dissecting the nvidia hopper gpu architecture," *arXiv preprint arXiv:2402.13499*, 2024.
- [20] NVIDIA and OpenAI, "OpenAI Triton on NVIDIA Blackwell Boosts AI Performance and Programmability," https://developer.nvidia.com/blog/openai-triton-on-nvidia-blackwell-

- boosts-ai-performance-and-programmability/, 2025, accessed: 2025- 12-01.
- [21] NVIDIA Corporation, "Nsight Compute Get Started," 2025, accessed: 2025-03-11. [Online]. Available: https://developer.nvidia.com/toolsoverview/nsight-compute/get-started
- [22] OpenAI, "Openai o3-mini," 2025, accessed: 2025-02-14. [Online]. Available: https://openai.com/index/openai-o3-mini/
- [23] B. Peng, J. Quesnelle, H. Fan, and E. Shippole, "Yarn: Efficient context window extension of large language models," *arXiv preprint arXiv:2309.00071*, 2023.
- [24] S. Sandokji, F. Essa, and M. Fadel, "A survey of techniques for warp scheduling in gpus," in *2015 IEEE Seventh International Conference on Intelligent Computing and Information Systems (ICICIS)*. IEEE, 2015, pp. 600–606.
- [25] J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani, and T. Dao, "Flashattention-3: Fast and accurate attention with asynchrony and lowprecision," *Advances in Neural Information Processing Systems*, vol. 37, pp. 68 658–68 685, 2024.
- [26] N. Shazeer, "Fast transformer decoding: One write-head is all you need," *arXiv preprint arXiv:1911.02150*, 2019.
- [27] Z. Su, Z. Chen, W. Shen, H. Wei, L. Li, H. Yu, and K. Yuan, "Rotatekv: Accurate and robust 2-bit kv cache quantization for llms via outlieraware adaptive rotations," *arXiv preprint arXiv:2501.16383*, 2025.
- [28] L. Sun, J. Jiang, C. Deng, X. Wu, H. Zhang, L. Chen, L. Ni, and J. Wang, "Gta: Grouped-head latent attention," *arXiv preprint arXiv:2506.17286*, 2025.
- [29] Q. Tao, W. Yu, and J. Zhou, "Asymkv: Enabling 1-bit quantization of kv cache with layer-wise asymmetric quantization configurations," *arXiv preprint arXiv:2410.13212*, 2024.
- [30] G. Team, P. Georgiev, V. I. Lei, R. Burnell, L. Bai, A. Gulati, G. Tanzer, D. Vincent, Z. Pan, S. Wang *et al.*, "Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context," *arXiv preprint arXiv:2403.05530*, 2024.
- [31] P. Tillet, H.-T. Kung, and D. Cox, "Triton: an intermediate language and compiler for tiled neural network computations," in *Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages*, 2019, pp. 10–19.
- [32] L. Wang, Y. Cheng, Y. Shi, Z. Tang, Z. Mo, W. Xie, L. Ma, Y. Xia, J. Xue, F. Yang *et al.*, "Tilelang: A composable tiled programming model for ai systems," *arXiv preprint arXiv:2504.17577*, 2025.
- [33] L. Wang, L. Ma, S. Cao, Q. Zhang, J. Xue, Y. Shi, N. Zheng, Z. Miao, F. Yang, T. Cao *et al.*, "Ladder: Enabling efficient {Low-Precision} deep learning computing through hardware-aware tensor transformation," in *18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24)*, 2024, pp. 307–323.
- [34] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv *et al.*, "Qwen3 technical report," *arXiv preprint arXiv:2505.09388*, 2025.
- [35] X. Yang, W. Wu, S. Feng, M. Wang, D. Wang, Y. Li, Q. Sun, Y. Zhang, X. Fu, and S. Poria, "Mm-bigbench: Evaluating multimodal models on multimodal content comprehension tasks," *arXiv preprint arXiv:2310.09036*, 2023.
- [36] T. Zhang, J. Yi, Z. Xu, and A. Shrivastava, "Kv cache is 1 bit per channel: Efficient large language model inference with coupled quantization," *Advances in Neural Information Processing Systems*, vol. 37, pp. 3304–3331, 2024.
- [37] Y. Zhao, C.-Y. Lin, K. Zhu, Z. Ye, L. Chen, S. Zheng, L. Ceze, A. Krishnamurthy, T. Chen, and B. Kasikci, "Atom: Low-bit quantization for efficient and accurate llm serving," *Proceedings of Machine Learning and Systems*, vol. 6, pp. 196–209, 2024.