# Algorithm V.3: Shared-memory GEMM microkernel

<span id="page-7-0"></span>1 Function GemmTile (A, B, C)

Each warp reads its fragment of the shared-memory weight tile A and accumulates into Cfor m, n, k in tile loops do  $C[m, n] += A[m, k] \cdot B[k, n] // \text{ weights}$  A[m, k] come from shared memory

by each thread block, since one decoded B fragment is reused across all M-blocks. At small batch sizes the tensor pipeline drains faster than the decoder can produce the next sub-tile and the kernel is decoder-bound; at large batch sizes matrix-multiply work dominates, the consumer never stalls, and tensor utilization recovers toward the uncompressed baseline. The fused kernel therefore achieves near-complete overlap in the high-batch regime and degrades gracefully to an unfused decode-then-GEMM schedule at low batch.

#### E. Global Memory Access Reduction

Consider a tiled GEMM  $C[M \times N] = A[M \times K] \cdot W[K \times N]$  with blocking  $(M_t, N_t, K_t)$  and a compute kernel that stages one W-tile of size  $K_t \times N_t$  into shared memory while iterating over the M dimension.

Without loss of generality, we assume the projection matrix  $W \in \mathbb{R}^{K \times N}$  does not fit in cache at all, so every time the kernel advances along the M-dimension it has to reload the relevant tiles of W from HBM. Since there are  $M/M_t$  tiles along the M-dimension, each element of W is therefore loaded once per  $M_t$ -tile, so the global-memory traffic for W becomes

$$V_B^{\text{uncompressed}} = \frac{M}{M_{\star}} \cdot KN.$$
 (1)

In contrast, with our on-the-fly ANS kernel, each element of  $\boldsymbol{W}$  is fetched once in compressed form and then reused from shared memory.

$$V_B^{\text{on-the-fly decomp}} = \left(\frac{M}{M_t} + \alpha - 1\right) \cdot KN. \tag{2}$$

Despite the fact that decompression throughput may not fully match the raw memory bandwidth of GPUs, our design can still sustain high GEMM throughput. By decoding tiles directly into shared memory and overlapping decompression with computation through carefully orchestrated pipeline scheduling, the system eliminates the repeated global-memory loads that dominate baseline execution. Moreover, the onthe-fly decompression stage introduces no additional global-memory pressure, ensuring that reductions in memory traffic translate directly into end-to-end performance gains.

#### VI. EVALUATION

We evaluate the proposed *on-the-fly decompression* design along three objectives:

1. Entropy characterization and compression efficiency.

Quantify the entropy of weights across representative

- LLMs, evaluate achievable compression ratios, and measure the gap between practical rANS-based coding and the Shannon bound.
- GPU system-level performance. Benchmark end-to-end inference performance with on-the-fly decompression under realistic GPU memory budgets, batch sizes, and sequence lengths.
- Microbenchmark of the proposed technique. Evaluate
  the efficiency and sensitivity of our design across different
  GEMM dimensions and under both prefill and decode
  settings.

#### A. Experimental Setup

**Evaluated LLM models.** To evaluate the effectiveness and generality of our approach, we conduct experiments across a diverse set of widely used open-source large language models spanning different scales and architectures. The evaluated models include Qwen-1.5B (Qwen2-1.5B), Mistral-7B (Mistral-7B-v0.3), and Qwen-14B (Qwen-14B), which represent commonly deployed dense transformer configurations. To further examine scalability to larger models, we include DeepSeek-67B (DeepSeek-LLM-67B) and Llama-3.1-405B. In addition, we evaluate mixture-of-experts architectures using Mixtral-8x22B (Mixtral-176B total parameters). These models collectively span parameter scales from 1.5B to 405B and cover both dense transformer architectures and modern MoE designs.

Numeric formats. Across these models, we evaluate a broad spectrum of widely adopted numeric formats, including bfloat16 (bf16), FP8-E5M2 (fp8), INT8, FP4-E2M1 (fp4), and INT4, as well as state-of-the-art group-quantized formats used in SmoothQuant [44] (sq8) and AWQ [29] (awq4). This range of representations allows us to evaluate the compressibility and runtime performance of our method across both standard floating-point formats and modern quantized weight representations used in LLM inference.

Hardware platforms. All compression-size and runtime performance experiments for our GPU kernels are conducted on two GPU servers to demonstrate the portability of our design across different GPU generations. The first server is equipped with eight NVIDIA A100 GPUs (80 GB HBM2e, 2 TB/s peak bandwidth). We further evaluate the performance on NVIDIA Hopper H200 GPUs. Experiments are implemented using PyTorch 2.5.1 and CUDA 12.1, with the CUTLASS-based GEMM baselines across platforms to ensure fair comparison.

## B. Entropy Characterization and Compression Bound

Figure 6 provides an overview of the effective bits-perweight achieved by our tile-level ANS compression compared against both the nominal storage formats and the Shannon entropy bound across the evaluated models. We have the following observations.

First, across all models and datatypes, including bf16, fp8, int8, group-quantized formats (sq8, awq4), and low-bit representations (fp4, int4), our ANS bitrates closely track the Shannon bound, typically within 0.01–0.05 bits. This

<span id="page-8-1"></span>

| Model                          | Sequence<br>Length | Variant              | Weight<br>Mem (GB) | KV Mem<br>(GB) | Total Mem (GB) | Max Batch                       | Throughput<br>(Token/s) | Median<br>TPOT (ms) |
|--------------------------------|--------------------|----------------------|--------------------|----------------|----------------|---------------------------------|-------------------------|---------------------|
| Qwen-14B<br>Budget: 80 GB      | 1024               | Uncompressed<br>Ours | 27.5<br>18.1       | 44.1<br>56.3   | 75<br>75       | 47<br><b>60</b> ( <b>1.3</b> ×) | 1131<br>1217            | 71<br>81            |
| (Single NVIDIA A100 GPU)       | 2048               | Uncompressed<br>Ours | 27.5<br>18.1       | 43.1<br>56.3   | 74<br>75       | 23<br><b>30</b> ( <b>1.3</b> ×) | 548<br>651              | 112<br>125          |
| Mixtral-176B<br>Budget: 320 GB | 1024               | Uncompressed<br>Ours | 261.9<br>163.7     | 26.3<br>124.6  | 304<br>304     | 20<br><b>95</b> (4.8×)          |                         |                     |
| (Four NVIDIA A100 GPU)         | 2048               | Uncompressed<br>Ours | 261.9<br>163.7     | 26.3<br>123.4  | 304<br>304     | 10<br><b>47</b> ( <b>4.7</b> ×) | 190<br>257              | 213<br>318          |

TABLE II: Illustration of memory footprint before and after lossless compression. Compression reduces the weight of memory, freeing capacity for larger KV-cache and enabling larger effective batch sizes and throughput at the same sequence length.

<span id="page-8-0"></span>![](_page_8_Figure_2.jpeg)

Fig. 6: Effective bit rates of tile-level ANS compression relative to Shannon entropy bounds.

near-perfect overlap indicates that the ANS encoder captures essentially all statistical redundancy in the weight distribution. The agreement is particularly tight for lower-bit formats such as int8, sq8, fp4, and int4, where the symbol alphabet is small and the empirical distribution is sharply peaked, allowing ANS to encode nearly at the theoretical optimum.

Second, the only consistent deviation appears for bf16, where the 16-bit symbol alphabet leads to a broader histogram and a larger frequency table. The finite-precision normalization required for 212-entry frequency tables introduces marginal overhead (roughly 0.1–0.2 bits), which is expected for high-cardinality distributions and well within the limits predicted by finite-precision ANS theory. Importantly, even in this worst case, our ANS bitrate remains within 1.1–1.5× of nominal

precision, significantly closer to the entropy limit than any existing lossless scheme.

Third, the figure highlights that nominal storage formats substantially over-allocate bits relative to the intrinsic information content of the weights. For example, bf16 weights often exhibit an effective entropy of only 10–12 bits, int8 typically compresses to 4–5 bits, and even aggressively quantized formats such as int4 retain only 0.6–1.0 bits of true entropy. These gaps are consistent across all model scales, from Qwen-1.5B to Llama-405B, demonstrating that the heavy-tailed, highly structured nature of transformer weight distributions persists uniformly across architectures and sizes.

# Algorithm V.3: Shared-memory GEMM microkernel

<span id="page-7-0"></span>1 Function GemmTile (A, B, C)

Each warp reads its fragment of the shared-memory weight tile A and accumulates into Cfor m, n, k in tile loops do  $C[m, n] += A[m, k] \cdot B[k, n] // \text{ weights}$  A[m, k] come from shared memory

by each thread block, since one decoded B fragment is reused across all M-blocks. At small batch sizes the tensor pipeline drains faster than the decoder can produce the next sub-tile and the kernel is decoder-bound; at large batch sizes matrix-multiply work dominates, the consumer never stalls, and tensor utilization recovers toward the uncompressed baseline. The fused kernel therefore achieves near-complete overlap in the high-batch regime and degrades gracefully to an unfused decode-then-GEMM schedule at low batch.

#### E. Global Memory Access Reduction

Consider a tiled GEMM  $C[M \times N] = A[M \times K] \cdot W[K \times N]$  with blocking  $(M_t, N_t, K_t)$  and a compute kernel that stages one W-tile of size  $K_t \times N_t$  into shared memory while iterating over the M dimension.

Without loss of generality, we assume the projection matrix  $W \in \mathbb{R}^{K \times N}$  does not fit in cache at all, so every time the kernel advances along the M-dimension it has to reload the relevant tiles of W from HBM. Since there are  $M/M_t$  tiles along the M-dimension, each element of W is therefore loaded once per  $M_t$ -tile, so the global-memory traffic for W becomes

$$V_B^{\text{uncompressed}} = \frac{M}{M_{\star}} \cdot KN.$$
 (1)

In contrast, with our on-the-fly ANS kernel, each element of  $\boldsymbol{W}$  is fetched once in compressed form and then reused from shared memory.

$$V_B^{\text{on-the-fly decomp}} = \left(\frac{M}{M_t} + \alpha - 1\right) \cdot KN. \tag{2}$$

Despite the fact that decompression throughput may not fully match the raw memory bandwidth of GPUs, our design can still sustain high GEMM throughput. By decoding tiles directly into shared memory and overlapping decompression with computation through carefully orchestrated pipeline scheduling, the system eliminates the repeated global-memory loads that dominate baseline execution. Moreover, the onthe-fly decompression stage introduces no additional global-memory pressure, ensuring that reductions in memory traffic translate directly into end-to-end performance gains.

#### VI. EVALUATION

We evaluate the proposed *on-the-fly decompression* design along three objectives:

1. Entropy characterization and compression efficiency.

Quantify the entropy of weights across representative

- LLMs, evaluate achievable compression ratios, and measure the gap between practical rANS-based coding and the Shannon bound.
- GPU system-level performance. Benchmark end-to-end inference performance with on-the-fly decompression under realistic GPU memory budgets, batch sizes, and sequence lengths.
- Microbenchmark of the proposed technique. Evaluate
  the efficiency and sensitivity of our design across different
  GEMM dimensions and under both prefill and decode
  settings.

#### A. Experimental Setup

**Evaluated LLM models.** To evaluate the effectiveness and generality of our approach, we conduct experiments across a diverse set of widely used open-source large language models spanning different scales and architectures. The evaluated models include Qwen-1.5B (Qwen2-1.5B), Mistral-7B (Mistral-7B-v0.3), and Qwen-14B (Qwen-14B), which represent commonly deployed dense transformer configurations. To further examine scalability to larger models, we include DeepSeek-67B (DeepSeek-LLM-67B) and Llama-3.1-405B. In addition, we evaluate mixture-of-experts architectures using Mixtral-8x22B (Mixtral-176B total parameters). These models collectively span parameter scales from 1.5B to 405B and cover both dense transformer architectures and modern MoE designs.

Numeric formats. Across these models, we evaluate a broad spectrum of widely adopted numeric formats, including bfloat16 (bf16), FP8-E5M2 (fp8), INT8, FP4-E2M1 (fp4), and INT4, as well as state-of-the-art group-quantized formats used in SmoothQuant [44] (sq8) and AWQ [29] (awq4). This range of representations allows us to evaluate the compressibility and runtime performance of our method across both standard floating-point formats and modern quantized weight representations used in LLM inference.

Hardware platforms. All compression-size and runtime performance experiments for our GPU kernels are conducted on two GPU servers to demonstrate the portability of our design across different GPU generations. The first server is equipped with eight NVIDIA A100 GPUs (80 GB HBM2e, 2 TB/s peak bandwidth). We further evaluate the performance on NVIDIA Hopper H200 GPUs. Experiments are implemented using PyTorch 2.5.1 and CUDA 12.1, with the CUTLASS-based GEMM baselines across platforms to ensure fair comparison.

## B. Entropy Characterization and Compression Bound

Figure 6 provides an overview of the effective bits-perweight achieved by our tile-level ANS compression compared against both the nominal storage formats and the Shannon entropy bound across the evaluated models. We have the following observations.

First, across all models and datatypes, including bf16, fp8, int8, group-quantized formats (sq8, awq4), and low-bit representations (fp4, int4), our ANS bitrates closely track the Shannon bound, typically within 0.01–0.05 bits. This

<span id="page-8-1"></span>

| Model                          | Sequence<br>Length | Variant              | Weight<br>Mem (GB) | KV Mem<br>(GB) | Total Mem (GB) | Max Batch                       | Throughput<br>(Token/s) | Median<br>TPOT (ms) |
|--------------------------------|--------------------|----------------------|--------------------|----------------|----------------|---------------------------------|-------------------------|---------------------|
| Qwen-14B<br>Budget: 80 GB      | 1024               | Uncompressed<br>Ours | 27.5<br>18.1       | 44.1<br>56.3   | 75<br>75       | 47<br><b>60</b> ( <b>1.3</b> ×) | 1131<br>1217            | 71<br>81            |
| (Single NVIDIA A100 GPU)       | 2048               | Uncompressed<br>Ours | 27.5<br>18.1       | 43.1<br>56.3   | 74<br>75       | 23<br><b>30</b> ( <b>1.3</b> ×) | 548<br>651              | 112<br>125          |
| Mixtral-176B<br>Budget: 320 GB | 1024               | Uncompressed<br>Ours | 261.9<br>163.7     | 26.3<br>124.6  | 304<br>304     | 20<br><b>95</b> (4.8×)          |                         |                     |
| (Four NVIDIA A100 GPU)         | 2048               | Uncompressed<br>Ours | 261.9<br>163.7     | 26.3<br>123.4  | 304<br>304     | 10<br><b>47</b> ( <b>4.7</b> ×) | 190<br>257              | 213<br>318          |

TABLE II: Illustration of memory footprint before and after lossless compression. Compression reduces the weight of memory, freeing capacity for larger KV-cache and enabling larger effective batch sizes and throughput at the same sequence length.

<span id="page-8-0"></span>![](_page_8_Figure_2.jpeg)

Fig. 6: Effective bit rates of tile-level ANS compression relative to Shannon entropy bounds.

near-perfect overlap indicates that the ANS encoder captures essentially all statistical redundancy in the weight distribution. The agreement is particularly tight for lower-bit formats such as int8, sq8, fp4, and int4, where the symbol alphabet is small and the empirical distribution is sharply peaked, allowing ANS to encode nearly at the theoretical optimum.

Second, the only consistent deviation appears for bf16, where the 16-bit symbol alphabet leads to a broader histogram and a larger frequency table. The finite-precision normalization required for 212-entry frequency tables introduces marginal overhead (roughly 0.1–0.2 bits), which is expected for high-cardinality distributions and well within the limits predicted by finite-precision ANS theory. Importantly, even in this worst case, our ANS bitrate remains within 1.1–1.5× of nominal

precision, significantly closer to the entropy limit than any existing lossless scheme.

Third, the figure highlights that nominal storage formats substantially over-allocate bits relative to the intrinsic information content of the weights. For example, bf16 weights often exhibit an effective entropy of only 10–12 bits, int8 typically compresses to 4–5 bits, and even aggressively quantized formats such as int4 retain only 0.6–1.0 bits of true entropy. These gaps are consistent across all model scales, from Qwen-1.5B to Llama-405B, demonstrating that the heavy-tailed, highly structured nature of transformer weight distributions persists uniformly across architectures and sizes.

