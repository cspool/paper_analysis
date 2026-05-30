# 1 Introduction

The transformative power of Large Language Models (LLMs) like GPT-4 [\[54\]](#page-15-0), LLaMA-3 [\[17\]](#page-14-0), and Qwen-3 [\[70\]](#page-16-1) is rooted in their massive scale [\[3,](#page-14-1) [36\]](#page-15-1), enabling a new paradigm of AI applications [\[6,](#page-14-2) [60,](#page-15-2) [74,](#page-16-2) [81\]](#page-16-3). However, this immense scale creates significant deployment challenges, making GPU memory capacity and bandwidth the primary bottlenecks for LLM serving, especially in resource-constrained environments.

Model compression offers a promising solution for efficient LLM deployment. Most existing approaches are lossy, reducing size by approximating model weights via

<span id="page-1-0"></span>![](_page_1_Figure_2.jpeg)

Figure 1. Execution time of lossless compression pipelines on NVIDIA L40S GPU with GateUp\_proj layers.

quantization (e.g., GPTQ [\[23\]](#page-14-3), AWQ [\[43\]](#page-15-3)) or pruning (e.g., SparseGPT [\[22\]](#page-14-4)). However, such approximations risk accuracy loss. For instance, aggressive 4-bit quantization (e.g., MXFP4) slashes accuracy from 56.0% to 36.2% on Live-CodeBench [\[44\]](#page-15-4), while even robust int8 quantization (GPTQint8) can cause up to 11.1% loss in long-context reasoning (NOCHA) [\[49\]](#page-15-5). These risks undermine reliability in safetycritical and user-facing settings, motivating approaches that guarantee bit-exact reproducibility and numerical integrity.

Lossless compression offers a compelling alternative by providing bit-exact model representation without accuracy loss. To date, its benefits have largely targeted storage and training workflows. For example, LMC [\[71\]](#page-16-4) and ZipNN [\[29\]](#page-15-6) employ Huffman [\[31\]](#page-15-7) to compress model checkpoints for efficient storage and distribution, while NeuZip [\[28\]](#page-15-8) and DietGPU [\[33\]](#page-15-9) mitigate memory and communication overhead during training. Although recent efforts, notably DFloat11 [\[85\]](#page-16-5), aim to extend these gains to inference, practical efficiency remains elusive. When integrated into serving pipelines, existing lossless techniques incur significant runtime overhead. As shown in Figure [1,](#page-1-0) the decoupled decompression step alone takes 1.56–3.44× the time of the core inference computation. This overhead forces an unpleasant tradeoff between memory efficiency and runtime efficiency.

We contend that this tradeoff is not fundamental but arises from a mismatch between conventional compression algorithms and modern GPU architectures. The issue manifests at two levels. At the kernel level, traditional entropy codecs (e.g., Huffman [\[31\]](#page-15-7) or ANS [\[18\]](#page-14-5)) produce variablelength bitstreams, whose decoding demands serialized, datadependent operations. These are ill-suited to the lockstep, parallel SIMT execution model of GPU warps, resulting in severe control-flow divergence and compute underutilization. At the system level, most frameworks employ a decoupled inference pipeline: weights are fully decompressed into a global-memory buffer before kernel consumption. This staged execution results in redundant, high-latency memory accesses, eroding compression-provided bandwidth savings and reducing arithmetic intensity during inference.

To rectify these fundamental algorithm-hardware mismatches, we present ZipServ[1](#page-1-1) , the first lossless compression framework co-designed for high-performance LLM inference on GPUs. Our key observation is that the exponent bits of BFloat16 weights in LLMs exhibit a highly skewed, low-entropy distribution in contemporary models. Exploiting this statistical redundancy, we propose Tensor-Core-Aware Triple Bitmap Encoding (TCA-TBE), a fixed-length, bitmapbased weight format tailored to GPU architectures. Unlike variable-length entropy codecs, TCA-TBE enables constanttime, parallel decoding using lightweight bitwise operations, thereby eliminating control-flow divergence and aligning with the GPU's SIMT execution model. Paired with TCA-TBE, ZipServ devises a fused decompression-GEMM kernel (ZipGEMM). Rather than decompressing weights into global memory as an intermediate step, ZipGEMM performs onthe-fly decoding, delivering compressed weights directly into the register files that feed Tensor Core matrix multiplication units. This "load-compressed, compute-decompressed" design eliminates intermediate buffers, reduces data movement, and maximizes the overlap between computation and memory access. By jointly addressing both the kernel-level and systemlevel mismatches, ZipServ transforms the theoretical storage savings of lossless compression into tangible performance gains on inference-optimized GPUs.

We demonstrate ZipServ's effectiveness through comprehensive benchmarking against state-of-the-art lossless approaches, including DietGPU [\[33\]](#page-15-9), vendor-optimized nvCOMP [\[53\]](#page-15-10), and the Huffman-based DFloat11 [\[85\]](#page-16-5). Compared to these baselines, which uniformly suffer significant runtime overhead, ZipServ consistently delivers substantial accelerations at both the kernel and system level on various inference-optimized GPUs, including RTX4090, L40S, and RTX5090. Our fused ZipGEMM achieves speedups of up to 2.21× over NVIDIA's cuBLAS, and up to 5.53× over DFloat11, the fastest lossless compression pipeline. These kernel-level improvements translate into an average 1.22× end-to-end speedup compared to leading systems like vLLM [\[39\]](#page-15-11). Our results demonstrate for the first time that when co-designed with hardware, lossless compression can provide both storage savings and substantial LLM inference acceleration.

The main contributions of this paper are as follows:

- We identify the fundamental mismatch between conventional entropy-based compression and GPU architectures, revealing both kernel- and system-level bottlenecks that hinder efficient inference.
- We propose TCA-TBE, a fixed-length, bitmap-based encoding tailored to SIMT execution and Tensor Core tiling, enabling constant-time, parallel decoding.
- We design ZipGEMM, a novel kernel that performs decompression on-the-fly directly into Tensor Core

<span id="page-1-1"></span><sup>1</sup>Publicly available at [https://github.com/HPMLL/ZipServ\\_ASPLOS26.git](https://github.com/HPMLL/ZipServ_ASPLOS26.git)

- registers, eliminating intermediate memory buffers and maximizing compute intensity.
- We present and evaluate ZIPSERV, a lossless compressed LLM inference framework that achieves end-to-end speedups across diverse LLMs and GPUs, constituting the first practical evidence that lossless compression can directly accelerate LLM serving.

## 2 Background

#### 2.1 Transformer-Based LLMs

Transformer-based LLMs [2, 17, 70] are composed of stacked layers of multi-head attention, feed-forward networks (FFNs), and normalization layers. During inference, computation proceeds autoregressively in two phases: prefill and decode. The prefill phase parallelizes computation over the input prompt, resulting in high arithmetic intensity due to large matrix multiplications operated over multiple tokens. On the contrary, the decode phase generates tokens one at a time, where matrix multiplications involve only a single token per batch element. The decode phase, hence, suffers from reduced compute utilization and greater sensitivity to memory bandwidth. In both phases, the dominant operation is dense matrix multiplication: Y = WX, where  $W \in \mathbb{R}^{M \times K}$ is a learned weight matrix and  $X \in \mathbb{R}^{K \times N}$  are activations, where M is the output dimension, K is the hidden dimension, and *N* is the number of tokens.

#### 2.2 BFloat16 Format

**BFloat16 (BF16)** [35] is a 16-bit floating-point format that has become the *de facto* precision standard for LLM inference, balancing memory efficiency with numerical robustness. It is natively supported by major hardware accelerators, including NVIDIA Tensor Cores [47], Google TPUs [34], and Intel AMX [37], and is widely adopted in production-scale models, including LLaMA-3 [17], Qwen [70], and Mistral [2]. A BF16 number consists of 1 sign bit, 8 exponent bits, and 7 mantissa bits. Its numerical value is computed as:

BF16(x) = 
$$(-1)^{\text{sign}} \times 2^{\text{exponent}-127} \times (1.\text{mantissa})$$
.

This layout preserves the full exponent range of IEEE FP32 (1-8-23) while reducing mantissa precision. Compared to FP16 (1-5-10), BF16 offers a wider dynamic range, reducing vulnerability to overflows and underflows in large models.

#### 2.3 GPU Architecture and Tensor Core Execution

Modern GPUs comprise multiple Streaming Multiprocessors (SMs), each with SIMT cores, Tensor Cores, registers, shared memory, and local caches. Threads are grouped into warps of 32, executing under the Single Instruction, Multiple Threads (SIMT) paradigm. Tensor Cores are specialized processors for high-throughput matrix multiplications. On recent NVIDIA architectures [50, 51], Tensor Cores support BF16 operands through the PTX-level

<span id="page-2-0"></span>![](_page_2_Figure_13.jpeg)

**Figure 2.** Exponent bit distribution in LLM weights.

mma . sync . m16n8k16 instruction, which performs fused matrix multiply-accumulate (FMA) operations across small matrix tiles. A typical BF16 Tensor Core operation can be expressed as:  $D_{\rm frag} = A_{\rm frag} \times B_{\rm frag} + C_{\rm frag}$ , where  $A_{\rm frag} \in \mathbb{R}^{16 \times 16}$ ,  $B_{\rm frag} \in \mathbb{R}^{16 \times 8}$ , and  $C_{\rm frag} \in \mathbb{R}^{16 \times 8}$  is the FP32 accumulator fragment. This operation is executed at the warp level, where a group of 32 threads collaborate to compute the matrix multiplication. The input and output fragments are distributed across the entire warp. Each thread holds a specific subset of fragment elements in its registers, and the complete fragment is formed collectively.

## <span id="page-2-2"></span>3 Gaps and Opportunities

Lossless compression enables *bit-exact* model representation but is rarely used for inference due to high runtime overheads stemming from a mismatch between traditional codecs and GPU architectures. This section quantifies compressibility in LLM weights and identifies key kernels and system-level bottlenecks that motivate our co-designed solution.

## <span id="page-2-1"></span>3.1 Compressibility of BF16 Weights

We analyzed the BF16 weights of leading LLMs, including Llama-3-8B-Instruct [17], Mistral-Small-24B-Instruct-2501 [2], and Qwen2.5-32B-Instruct [69], and observed remarkable redundancy in their 8-bit exponent fields. As shown in Figure 2, the exponent distributions are *highly skewed*: the **top-3** most frequent exponents account for more than 67% of all weights, and the **top-7** exponents cover over **95%** (e.g., 96.4% in LLaMA-3 and 97.4% in Mistral-24B). The information entropy of the exponent field is only **2.57–2.74 bits**, far below its 8-bit allocation, implying a theoretical lossless compression ratio of about 1.51× (16/10.6) for BF16 values. These findings are consistent with prior

<span id="page-3-0"></span>![](_page_3_Figure_2.jpeg)

**Figure 3.** Existing Lossless Compression for BF16 Weights. Illustrated with Huffman Encoding.

works [28, 29, 71, 83, 85]. We further scrutinized this redundancy across 3,875 weight matrices from four LLM families (Gemma-3, Mistral, Qwen2.5, and LLaMA-3.1), revealing a critical structural property: exponent contiguity. In 99.6% of these matrices, the top-7 most frequent exponents form a numerically contiguous sequence (i.e.,  $e^*, \ldots, e^* + 6$ ). Consequently, a simple contiguous window covers 97.1% of all weights on average, approaching the information-theoretic limit. In Appendix A, we prove that this is not coincidental but an intrinsic property of LLMs. This contiguity is the cornerstone of ZIPSERV. It obviates the need for complex, hardware-unfriendly variable-length codecs (e.g., Huffman) in favor of a **fixed-length**, base-plus-offset representation. This insight directly enables our Tensor-Core-Aware Triple Bitmap Encoding (TCA-TBE) and its implicit lookup mechanism described in §4.3.2.

## 3.2 Kernel-Level Architectural Mismatch

Existing methods exploit the exponent redundancy of BF16 weights by applying entropy coding to the exponent stream. For example, DFloat11 uses Huffman coding [85], while Diet-GPU employs Asymmetric Numeral Systems (ANS) [33]. As shown in Figure 3, these approaches produce a compressed bitstream with *variable-length* symbols depending on their statistical frequency. However, this bitstream must be decompressed *sequentially* to correctly recover each exponent, which fundamentally conflicts with the lockstep, massively parallel SIMT execution model of modern GPUs.

To illustrate this mismatch, we examine the three-stage decompression pipeline in DFloat11 [85]. Bitstream Partitioning. The bitstream is split into chunks for parallel thread processing. However, because variable-length symbols cross chunk boundaries, threads cannot operate independently but require additional metadata to locate valid symbol start points, introducing overhead and disrupting parallel execution. Symbol Extraction. Threads use hierarchical lookup tables (LUTs) for symbol decoding—a data-dependent operation. When warp threads encounter different symbol lengths, faster threads stall for slower ones, causing divergence and underutilization of GPU resources. Pointer Advancement. After symbol decoding, each thread advances its bit pointer by the symbol's length, which is only known

<span id="page-3-1"></span>![](_page_3_Figure_8.jpeg)

Figure 4. Existing lossless compression inference pipeline.

after the lookup completion. This inherently serializes the decoding loop and sacrifices opportunities for instruction-level parallelism. Our evaluation shows that on L40S GPUs, even highly optimized decompressors (e.g., ANS-based DietGPU and Huffman-based DFloat11) achieve only 43.7% and 76.5% of peak memory bandwidth, respectively. This inefficiency exposes a fundamental algorithm-hardware mismatch: entropy coding is inherently data-dependent, while efficient GPU execution desires regular, uniform parallelism.

## 3.3 Inefficiency of Decoupled Inference Pipeline

The architectural inefficiency of entropy-coded decoding is found not only at the kernel level, but also at the system pipeline level for LLM inference. In mainstream approaches, decompression is performed as a separate, decoupled preprocessing stage (see Figure 4): it materializes the entire decompressed weights in global memory first and then passes it to the compute kernels. This decoupled pipeline design leads to redundant data transfers, undermining the benefits of compression, particularly in bandwidth-constrained environments. We analytically quantify its inefficiency using the Roofline model, focusing on Compute Intensity (CI).

**Compute Intensity.** CI measures the number of floating-point operations (FLOPs) performed per byte read from global memory. For a typical BF16 GEMM operation  $Y_{M\times N} = W_{M\times K}X_{K\times N}$ , the compute intensity is:

$$CI_{GEMM} = \frac{MNK}{MK + KN + MN}. (1)$$

In the decoupled pipeline scenario, assuming an average compression ratio (CR) of 1.51 (§3.1), the CI becomes:

$$CI_{\mbox{Decoupled}} = \frac{2MNK}{MK\left(\frac{2}{\mbox{CR}} + 4\right) + 2(KN + MN)} \approx \frac{MNK}{2.66MK + KN + MN}. \eqno(2)$$

**Roofline Model Analysis.** Figure 5 illustrates the Roofline analysis on an NVIDIA RTX4090. During the decode stage, both the standard GEMM and the decoupled pipeline operate in the memory-bound regime, where performance scales linearly with CI. However, our analysis highlights a pronounced penalty for the decoupled approach: the additional memory traffic required to materialize intermediate decompressed weights significantly reduces CI. Specifically, for a

<span id="page-4-0"></span>![](_page_4_Figure_2.jpeg)

Figure 5. Roofline analysis.

weight matrix of size M = K = 4096, the decoupled pipeline exhibits a CI degradation of 62.3%, 62.2%, 62.0%, and 61.7% relative to standard GEMM for batch sizes of 8, 16, 32, and 64, respectively.

**ZIPSERV's Fused Design.** The inefficiency of decoupled pipelines arises directly from staging decompressed weights in global memory. ZIPSERV addresses this by introducing a fused decompression-GEMM kernel that directly fetches compressed weights from DRAM and decompresses them on-the-fly into register files, which immediately feed the Tensor Core. This approach effectively increases CI to

$$CI_{\text{ZIPSERV}} = \frac{2MNK}{MK \cdot \frac{2}{CR} + 2(KN + MN)} \approx \frac{MNK}{0.66MK + KN + MN}.$$
 (3)

Revisiting the Roofline model in Figure 5, ZIPSERV's fused execution ( $CI_{ZIPSERV}$ ) demonstrates a substantial improvement, achieving even higher CI (approximately 50%) than the uncompressed GEMM baseline. This benefit, most pronounced in memory-bound regimes, leads to linear speedups relative to the compression ratio, translating information-theoretic redundancy into wall-clock acceleration.

## 4 Design of ZIPSERV

Our earlier analysis identifies both kernel-level and systemlevel sources of inefficiency that hinder the decoding of lossless compression in LLM inference. In this section, we present ZIPSERV, a lossless compression system co-designed for storage efficiency and *fast*, *bit-exact* LLM inference.

## 4.1 Overview and Workflow

As illustrated in Figure 6, ZIPSERV consists of two main components: an *offline compressor*, which transforms BF16 model weights into a parallelization-friendly compressed representation, and an *online inference engine*, responsible for efficient decoding and computation at runtime.

**Offline Compressor.** At the core of the offline compressor is the *Tensor-Core-Aware Triple Bitmap Encoding* (TCA-TBE), a fixed-length, bitmap-based compression format designed to enable *parallel decoding* via GPU SIMT execution and Tensor Core–accelerated GEMM operations. As outlined in **Algorithm 1**, given a model, the compressor first profiles the exponent distribution of each layer's weights. Instead of

selecting arbitrary frequent exponents, it identifies a window of k numerically consecutive exponent values (typically k=7) that maximizes coverage of the weight distribution. The compressor records the value immediately preceding this range as the BaseExp (i.e.,  $\min(\text{range})-1$ ). Using this range, the compressor encodes the entire weight matrix into the TCA-TBE representation. Each  $8\times 8$  tile of weights is converted into three 64-bit bitmaps and two compact value buffers: one for high-frequency values falling within the selected exponent range (storing only the sign and mantissa relative to BaseExp), and another for outliers in full BF16 precision. The resulting compressed model is then loaded onto the GPU, ready for serving.

**Online Inference Engine.** The inference engine employs a stage-aware strategy that adapts the execution pipeline for the prefill and decode phases, all on the unified TCA-TBE format. During the compute-bound **prefill stage**, the engine performs decoupled execution: a dedicated decompression kernel decompresses the weights into global memory first, followed by the prefill computation. This approach allows high-throughput GEMM to effectively amortize the decompression overhead. In the memory-bound decode stage, the engine switches to a fused decompression-GEMM kernel (ZipGEMM). ZipGEMM enables a "load-compressed, computedecompressed" execution model, where weights are decompressed on-the-fly directly into Tensor Core registers. This eliminates redundant data transfers and maximizes compute intensity for each token generation. These two specialized execution paths deliver near-optimal inference performance.

#### 4.2 Tensor-Core-Aware Triple Bitmap Encoding

ZIPSERV is built on top of a novel Tensor-Core-Aware Triple Bitmap Encoding (TCA-TBE) scheme. It is designed to minimize the weight memory footprint while enabling efficient parallel decoding on GPUs. In contrast to existing variablelength bitstream-based entropy codecs, TCA-TBE employs a fixed-length, tile-structured representation that ensures constant-time, thread-local decompression. Its data layout is carefully aligned with Tensor Core tiling and register-level operand distribution, allowing the decompressed weights to be consumed directly by the mma. sync instruction. The core of TCA-TBE is a fixed-length 3-bit codeword assigned to each weight element, representing one of eight possible states (000-111). During offline compression, ZIPSERV profiles the exponent histogram of a weight matrix and identifies the top-7 most frequent exponent values and maps them to codewords 001-111. The special codeword 000 serves as a fallback, designating weights whose exponent falls outside the top-7, which are then stored in full precision.

The Choice of Codeword Length. We choose the 3-bit codeword because it achieves a near-optimal compression ratio by leveraging the highly skewed exponent distributions observed in contemporary LLMs. To quantify this design

<span id="page-5-0"></span>![](_page_5_Figure_2.jpeg)

Figure 6. Overview of ZIPSERV. ZIPSERV features an offline lossless compressor (left) and an online inference engine (right).

```
Algorithm 1 ZIPSERV Offline Compressor (TCA-TBE)
Input: Weight Matrix W, Tile Size T = 8 \times 8
Output: Bitmaps \mathcal{B}_{1..3}, High-Freq Buffer \mathcal{H}, Fallback Buffer
     \mathcal{L}, BaseExp e_{base}
  1: ▶ Phase I: Global Exponent Analysis
 2: Hist \leftarrow ComputeExponentHistogram(W)
 3: E_{top} \leftarrow \text{SelectTop7ConsecutiveExponents}(Hist)
 4: e_{base} ← min(E_{top}) − 1 \rightarrow Set base for implicit lookup
 5: ▶ Phase II: Tile Encoding
 6: for each tile t \in \mathcal{W} do
        Initialize local bitmaps b_1, b_2, b_3 \leftarrow 0
 7:
        for i = 0 to 63 do
 8:
 9:
           w \leftarrow t[i]; e \leftarrow w.exponent
           if e \in E_{top} then
 10:
                                   ▶ Compute 3-bit code c \in [1, 7]
              c \leftarrow e - e_{base}
11.
               b_1[i] \leftarrow c_0; \ b_2[i] \leftarrow c_1; \ b_3[i] \leftarrow c_2 \quad \triangleright \text{Set bits}
 12:
              \mathcal{H}.Push(Pack(w.sign, w.mantissa))
13:
 14:
                                       ▶ Store full precision fallback
               \mathcal{L}.Push(w)
15:
            end if
16:
        end for
17:
        Store b_1, b_2, b_3 to global \mathcal{B}_{1..3}
18:
```

choice, we calculate the expected per-element storage cost as:

19: end for

```
AverageBits(n) = r_n \cdot (n + 8) + (1 - r_n) \cdot (n + 16),
```

where n is the codeword length and  $r_n$  is the proportion of weights covered by the top  $2^n-1$  exponent values. As shown in §3.1,  $r_3 \approx 0.96$ , yielding an average of 11.3 bits per element, which approaches the theoretical lower bound (8+2.6=10.6 bits) and offers clear advantages over 2-bit (12.4 bits) and 4-bit (12.1 bits) codewords. Besides, the 3-bit encoding yields a compact 7-entry codebook, enabling decoding via a simple table lookup. This requires only a handful of bitwise operations per thread, which can be efficiently performed with warp-synchronous Tensor Core pipelines.

<span id="page-5-2"></span>![](_page_5_Figure_8.jpeg)

**Figure 7.** Tensor-Core-Aware Triple Bitmap Encoding. The  $4 \times 4$  FragTile shown is illustrative; the actual size is  $8 \times 8$ .

Decoupled Triple Bitmap Layout. To maximize decoding efficiency on SIMT architectures, TCA-TBE implements a *decoupled triple bitmap layout* rather than packing codewords into a dense bitstream. Conventional bitstreams are inefficient on GPUs because packing non-byte-aligned codes (e.g., 3-bit) forces codewords to span memory word boundaries. This necessitates complex logic for non-aligned accesses and introduces data-dependent branching, which in turn causes thread divergence that severely degrades SIMT throughput.

TCA-TBE avoids these bottlenecks by decomposing the 3-bit codewords for each 8 × 8 weight tile into three independent 64-bit bitmaps, with each bitmap representing a single bit-plane (Figure 7). This design enables two benefits. First, it guarantees coalesced memory accesses, as each bitmap is a contiguous 64-bit word, naturally aligned to native memory boundaries. Second, it enables branch-free decoding. All threads in a warp follow an identical execution path, aligning with the SIMT model on modern GPUs.

<span id="page-6-1"></span>![](_page_6_Figure_2.jpeg)

Figure 8. Data movement and instruction pipeline.

Hierarchical Tiling Design. TCA-TBE adopts a threelevel hierarchical tiling scheme that partitions the weight matrix according to the architectural granularity of modern GPUs. **1** FragTile (FT): The base unit is an 8×8 tile, matching the smallest operand fragment of Tensor Core instruction. 2 TensorCoreTile (TT): Each  $16 \times 16$  tile is composed of a  $2 \times 2$ grid of FragTiles. This size aligns with the operand dimensions (m=16, k=16) required by PTX-level Tensor Core mma instructions (mma.m16n8k16). 3 BlockTile (BT): At the coarsest level, a 64 × 64 tile aggregates multiple TensorCoreTiles and is processed cooperatively by a thread block. The FragTiles within a TensorCoreTile are stored in column-major order, mirroring the operand register layout (e.g., Ra0-Ra3) expected by Tensor Core instructions. This design eliminates the need for runtime coordinate transformation, reducing instruction overhead. Each 8 × 8 FragTile is encoded using five buffers. **1** Three 64-bit bitmaps, each representing one bit-plane of the 3-bit codewords. ② A PackedSignMantissa buffer, which holds the compact 8-bit representation (sign and mantissa) of weights whose exponents fall within the top-k frequent classes. **3** A FullValue buffer, which stores full-precision BF16 values for weights not covered by the exponent codebook. At the matrix level, TCA-TBE organizes these buffers into four contiguous global arrays, each nested according to the tiling hierarchy. In addition, an Offset array records the starting offset of each GroupTile within the PackedSignMantissa and FullValue arrays.

## 4.3 Fused ZipGEMM Kernel Design

TCA-TBE's SIMT-friendly design opens up new opportunities for high-throughput decoding. To achieve this, ZIPSERV fuses decompression and matrix multiplication into a single kernel, **ZipGEMM**, that fetches weights from global memory in a compact TCA-TBE format and decompresses them just-in-time during computation. ZipGEMM enables a *load-compressed*, *compute-decompressed* execution model, substantially reducing the memory bandwidth requirement for each token generation in the decode stage (see Figure 5).

**4.3.1 Kernel Workflow.** Figure 8 illustrates the workflow of the ZipGEMM kernel. Based on a split-K tiling architecture, each thread block iteratively processes the *K* dimension in chunks. In each iteration, the kernel proceeds through four coordinated stages. 1 Tile Loading. Threads cooperatively load the compressed weight tile and the corresponding activation tile from global memory into shared memory, with asynchronous and vectorized memory instructions (i.e., LDGSTS.128) to bypass the L1 cache and improve global memory bandwidth utilization. The PackedSignMantissa and FullValue arrays within each tile are padded offline to ensure 128-bit alignment. **②** Warp-Level Decoding. Each warp independently decompresses the compressed weight from shared memory. The decompressor reconstructs the original BF16 values in a layout compatible with Tensor Core consumption, utilizing lightweight ALU operations and avoiding shared memory round-trips. 3 Activation Register Transfer. The activation tile is moved from shared memory into registers using the LDSM. M88 instruction, which enables a warp to load a  $16 \times 16$  tile and arrange it in the layout required for Tensor Cores. **4** Tensor Core Computation. Once both decompressed weights and activations reside in registers, the warp performs Tensor Core mma instructions. The execution path closely mirrors the standard cuBLAS GEMM kernels, while operating directly on compressed representations and reducing global memory accesses.

<span id="page-6-0"></span>**4.3.2 Efficient Decompressor.** ZipGEMM incorporates an efficient Decompressor that enables thread-local reconstruction of compressed weights directly within the register file. The core principle of the Decompressor is that each thread independently decompresses the elements required for the proper Tensor Core fragment layout. Specifically, as shown in Figure 7, the fragment layout requires that thread i's .bf16x2 register (e.g., Ra0) holds the values at positions 2i and 2i + 1 within the  $8 \times 8$  tile, denoted as  $a_0$  and  $a_1$ respectively. Since each element is encoded in one of two states-either as a high-frequency fixed-length code or as a fallback full-precision value—and these states are distributed in an unstructured manner, the decompressor solves a sparse, non-uniform spatial reconstruction problem. Two challenges arise in this context. First, each thread must efficiently determine the state of its assigned element (compressed or fallback). Second, each thread should recover the original BF16 representation in a deterministic, SIMT-friendly manner. To this end, ZIPSERV's Decompressor is structured into three tightly integrated stages: spatial bitmap indicator, dynamic addressing, and fast exponent reassembly (see Figure 9 and Algorithm 2).

**Spatial Bitmap Indicator.** Each thread first determines the storage mode of its assigned elements by evaluating a spatial indicator mask. During offline compression, each 8×8 weight tile is encoded using three 64-bit bitmaps, where each

<span id="page-7-0"></span>![](_page_7_Figure_2.jpeg)

Figure 9. The Decompressor Design.

## <span id="page-7-1"></span>Algorithm 2 ZipGEMM Thread-Local Decompression

```
Input: Bitmaps \mathcal{B}_{1..3}, Buffers \mathcal{H}, \mathcal{L}, BaseExp e_{base}, LaneID l Output: Register pair R containing two BF16 values
```

```
1: ▶ Step 1: Spatial Indicator Construction
 2: \mathcal{M} \leftarrow \mathcal{B}_1 \vee \mathcal{B}_2 \vee \mathcal{B}_3
 3: ▶ Step 2: Parallel Element Decompression
 4: for k \in \{0, 1\} do
 5:
         p \leftarrow 2 \cdot l + k
                                                 \triangleright Global position in 8 \times 8 tile
          mask \leftarrow (1 \ll p) - 1
 6:
          idx_{\mathcal{H}} \leftarrow \text{Popc}(\mathcal{M} \& \textit{mask})
                                                                     ▶ Calculate index
 7:
          if (\mathcal{M} \gg p) \& 1 then
 8:
              ▶ Case A: High-Frequency Path
 9:
              val \leftarrow \mathcal{H}[\operatorname{start}_{\mathcal{H}} + idx_{\mathcal{H}}] \triangleright \operatorname{Fetch} \operatorname{Sign} + \operatorname{Mantissa}
10:
              ▶Reconstruct 3-bit code
11:
              c \leftarrow (\mathcal{B}_3[p] \ll 2) \vee (\mathcal{B}_2[p] \ll 1) \vee \mathcal{B}_1[p]
12:
              e \leftarrow e_{base} + c
                                                                    ▶ Implicit Lookup
13:
              w_k \leftarrow \text{MAKEBF16}(val.sign, e, val.mantissa)
14:
          else
15:
16:
              ▶ Case B: Fallback Path
              idx_{\mathcal{L}} \leftarrow p - idx_{\mathcal{H}} > Calculate index in fallback
17:
18:
              w_k \leftarrow \mathcal{L}[\operatorname{start}_{\mathcal{L}} + idx_{\mathcal{L}}]
19:
20: end for
21: R \leftarrow \text{PackRegister}(w_0, w_1)
22: return R
```

bitmap encodes a single bit of the 3-bit codeword. At runtime, the three bitmaps are combined using a warp-level bitwise OR to produce a single 64-bit indicator mask. Each bit in this mask specifies the storage mode of one element: 1 for compressed (high-frequency), 0 for fallback (uncompressed). Each thread determines its decoding path by inspecting the corresponding bits in this spatial indicator mask, which resides in registers. Specifically, for thread i, the bits at positions 2i (for  $a_0$ ) and 2i + 1 (for  $a_1$ ) indicate the state

of the two assigned elements. For instance, Thread 19 finds that bit 38 (2 × 19) is set, indicating its  $a_0$  element is stored in compressed form. It fetches the packed value from the high-frequency buffer and proceeds with exponent reassembly. In contrast, Thread 6 sees that bit 12 (2 × 6) is unset and simply loads its  $a_0$  directly from the fallback buffer. This bitwise decision process is lightweight, fully register–resident, and completes in constant time.

**Dynamic Addressing.** Once the storage mode is determined, each thread computes its read offset into the appropriate value buffer on-the-fly, without explicit per-element indices. This is achieved via a lightweight, warp-local prefix sum over the spatial indicator. For thread i, the offset is calculated by counting how many previous elements of the same storage type appear in bits [0, 2i-1] of the spatial indicator. Specifically, if the element is compressed (bit = 1), the offset equals the number of 1s; if uncompressed (bit = 0), it equals the number of 0s in that range. These counts are efficiently computed using GPU-native instructions such as \_\_popc() and \_\_shfl\_sync(). For example, Thread 6, encountering an unset bit at position 12, computes its fallback buffer offset by counting the number of 0s in bits [0, 11]. Thread 19, with bit 38 set, counts the number of 1s in bits [0, 37] to access the compressed buffer. This dynamic addressing mechanism transforms indexing into a deterministic, SIMT-friendly operation that aligns naturally with GPU execution patterns.

**Fast Exponent Reassembly via Implicit Lookup.** To further reduce the decoding overhead, ZIPSERV reconstructs exponents using an *implicit lookup* mechanism based on *arithmetic remapping*, avoiding table-based decoding. During offline compression, the top-7 most frequent exponent values are identified globally and assigned 3-bit codewords (001–111), ordered by increasing numerical value instead of frequency rank. A single global base exponent is recorded as base\_exp = min(top\_exponents)-1, which is shared by all

tiles. At runtime, each thread reconstructs the original exponent by adding the 3-bit codeword to the base exponent. This operation eliminates shared memory table lookups by using a single integer ALU instruction. The recovered exponent is then fused with the sign and mantissa fields to assemble a valid BF16 value. For example, Thread 19 observes that bit 38 in the spatial indicator is set and reconstructs the 3-bit codeword by reading the corresponding bits from the three bitmap planes, yielding 101 (5). With a global base exponent of 115, it recovers the original exponent as 115 + 5 = 120, then combines it with the sign and mantissa to form the final BF16 value. This arithmetic decoding process is fully SIMT-compatible, exploits the GPU's integer pipelines.

**Repacking into Tensor Core Fragments.** Each thread repacks the two reconstructed BF16 elements into a single bfloat162 register, matching the operand layout required by Tensor Core mma. sync instructions.

4.3.3 Fine-grained Software Pipeline. ZipGEMM uses a hierarchical two-level pipeline to overlap memory transfer, decompression, and computation, effectively hiding memory and decompression latency. At the coarse level, tile-wise double buffering overlaps global-to-shared memory transfers with computation; at the fine level, slice-wise interleaving overlaps shared-to-register movement and decompression with Tensor Core operations. This is implemented via two shared memory buffers for compressed weights (triple bitmaps, packed sign-mantissa, fallback values) and activations. Within each tile, computation is sliced along the K dimension (typically  $16 \times 16$  fragments) and processed using an interleaved load-decompress-compute pattern. While Tensor Cores execute matrix multiplication (mma) on slice i, ALU units concurrently load and decompress weights for slice i + 1 from shared memory into registers. This ensures a steady compute flow by hiding decompression and memory latency behind computation.

To coordinate the two pipeline levels, ZipGEMM uses a hierarchical barrier strategy for inter-tile and intra-warp synchronization. **Inter-tile synchronization:** cp.async.wait\_group<0>() and \_\_syncthreads() ensure all asynchronous transfers complete before switching buffers. This barrier is placed *after the final slice decompression but before the final slice* mma, allowing computation to proceed while the next tile is being loaded and decompressed, which maximizes overlap and minimizes stalls. **Intra-warp coordination:** Intra-warp operations are implicitly synchronized via the SIMT model, requiring no explicit barriers between load, decompress, and compute at the slice level.

## 4.4 Stage-Aware Inference Strategy

ZIPSERV uses the fused ZipGEMM kernel exclusively during the decode stage for accelerated token generation. For the compute-bound prefill stage, where large matrix dimensions ( $N = BS \times Seq\_len$ ) provide high arithmetic intensity,

![](_page_8_Figure_8.jpeg)

Figure 10. Hierarchical software pipeline design.

ZIPSERV falls back to a decoupled pipeline: an efficient decompression kernel first extracts the compressed weights to global memory, then performs high-throughput GEMM operations to amortize the decompression overhead (typically <4% as shown in §6.4). In both prefill and decode stages, the decompression kernel and ZipGEMM kernel share the same compressed format and per-thread decompression logic (§4.3.2), obviating the need for runtime format conversions.

## 5 Implementation

We implemented ZIPSERV as a high-performance, modular inference backend comprising approximately 3.5K lines of code. The core engine consists of about 2.5K lines of CUDA and C++, which implements the offline TCA-TBE compressor and the online ZipGEMM kernel. The kernel is compiled into a standalone shared library (. so) using nvcc, exposing C++ APIs for weight packing and kernel launching. The remaining 1.0K lines are Python glue code used to integrate ZIPSERV into **vLLM** [39]. We extended vLLM's model loader and linear execution modules to support the TCA-TBE format, utilizing PyBind11 to invoke our custom CUDA kernels.

#### 6 Evaluation

We evaluate the performance of ZIPSERV at two levels: the kernel level of the fused ZipGEMM and the standalone Decompression kernel (ZIPSERV-Decomp), and the end-to-end inference framework level. All experiments are conducted on two platforms. • A consumer-grade server equipped with 4× NVIDIA RTX4090 GPUs (Ada Lovelace, 24GB memory, Compute Capability 8.9), paired with an Intel Xeon Platinum 8352V CPU (144 cores, 512GB DDR4). 2 A datacenter platform with 4× NVIDIA L40S GPUs (Ada Lovelace, 48GB), paired with an Intel Xeon Gold 6230R CPU (104 cores, 512GB DDR4). We also evaluate ZipGEMM on the latest 3 RTX5090 GPU (Blackwell, 32GB, Compute Capability 12.0) to demonstrate forward compatibility. All code is compiled using GCC 11.3 and NVCC 12.4 (with NVCC 12.8 specifically for RTX5090). For kernel-level evaluation, we perform 100 warm-up iterations followed by 1,000 timed executions. For end-to-end evaluation, each configuration is run 10 times.

<span id="page-9-0"></span>![](_page_9_Figure_2.jpeg)

Figure 11. Kernel performance comparison on NVIDIA RTX4090 and L40S GPUs.

## 6.1 ZipGEMM Kernel Performance

**Datasets.** We benchmark the kernel-level performance on representative linear layers from state-of-the-art LLMs. The input shapes for kernel benchmarking are directly extracted from the real weight matrices of prominent LLM families, including LLaMA3.1 [17] (8B, 70B, and 405B), Qwen2.5 [69] (7B, 14B, 32B, and 72B), Gemma3 [68] (12B and 27B), and Mistral [2] (24B and 123B), covering a broad range of model scales and hidden dimensions.

Baselines. We compare ZipGEMM against four representative baselines: ① cuBLAS\_TC v12.4.5 [52], NVIDIA's official BF16 Tensor Core GEMM kernel; ② DietGPU [33], a popular open-source, GPU-native rANS codec for lossless decompression of floating-point weights; ③ nvCOMP (rANS) [53], NVIDIA's general-purpose asymmetric numeral systems-based decompression library; and ④ DFloat11 [85], a state-of-the-art Huffman-coded GPU decompression framework for LLM inference. Since nvCOMP lacks native BF16 support, we compress exponent bits as a bitstream via rANS and reconstruct BF16 values with a custom high-performance kernel. For DFloat11, whose compression code is unavailable, we benchmark full Transformer block decompression latency and linearly scale estimates for other matrix shapes.

Workloads. We profile all linear layers within a Transformer block, including the merged QKV projection (QKV\_proj), attention output projection (O\_proj), merged FFN gate and up projection (GateUp\_proj), and down projection (Down\_proj), along with the model's LM head layer. Benchmarks are conducted at batch sizes of 8, 16, and 32.

**Results.** We begin by evaluating the performance of our fused *ZipGEMM* kernel. Figure 11 shows the normalized

speedup relative to cuBLAS\_TC across all evaluated models and workloads. ZipGEMM consistently outperforms all baseline methods on both hardware platforms. On the RTX4090, ZipGEMM achieves an average speedup of 1.31× over cuBLAS\_TC, with a peak speedup of 1.71×. The advantage is even greater on the L40S, with an average speedup of 1.36× and a maximum of 2.21×. In contrast, other decoupled decompression methods introduce substantial overhead, resulting in significant slowdowns. Specifically, DietGPU, nvCOMP, and DFloat11 achieve average speedups of only  $0.17 \times /0.20 \times$ ,  $0.19 \times /0.23 \times$ , and  $0.28 \times /0.34 \times$  on RTX4090 and L40S, respectively. This indicates that the decoupled decompression processes incur overheads that exceed the computation time of the baseline GEMM. ZipGEMM stands out as the only implementation that can significantly surpass the efficient Tensor Core GEMM. These results highlight the effectiveness of ZipGEMM's fused decompressioncomputation approach, which efficiently transforms storage savings into tangible execution speedup.

We further conducted a layer-wise analysis (Figure 11(c)). ZipGEMM exhibits significant acceleration on most of the computationally intensive layers within a transformer block. For instance, within the LLaMA3.1 model family on the L40S, ZipGEMM achieves average speedups of 1.39× and 1.64× on the GateUp\_proj and Down\_proj layers, respectively. However, ZipGEMM may experience a slowdown when processing certain layers with small shapes; for example, on the L40S, its performance on the O\_proj layer of LLaMA3.1-8B is reduced to 0.79×. This is primarily because small layers require fine-grained parameter tuning (e.g., split-K configurations and precise tiling) to fully utilize hardware, which is beyond the scope of this work. Nevertheless, such layers account for only a small fraction of the total FLOPs within a Transformer block. ZipGEMM delivers robust block-level

<span id="page-10-0"></span>![](_page_10_Figure_2.jpeg)

Figure 12. Micro-level kernel performance analysis.

<span id="page-10-1"></span>![](_page_10_Figure_4.jpeg)

Figure 13. Standalone decompression kernel comparison.

speedups of 1.35× for LLaMA3.1-8B and 1.48× for LLaMA3.1- 405B on the L40S.

Micro-level Analysis. We profiled ZipGEMM with Nsight Compute (NCU) on an RTX4090 to identify the source of its speedup ( = 28672, = 4096 and = 32). As shown in Figure [12,](#page-10-0) the performance gain stems from a deliberate architectural trade-off: introducing a predictable ALU workload for on-the-fly decoding in exchange for a reduction in memory traffic. Figure [12\(](#page-10-0)a) quantifies this trade-off. The high volume of integer and logical instructions (LOP3, IADD, and POPC) reflects the computational cost of our core decoding steps. This workload is the price for a 29.3% drop in DRAM reads, a direct validation of the TCA-TBE format's efficiency. Crucially, the two-level software pipeline effectively hides the decoding latency by overlapping it with compute and memory operations. As a result, even with ALU utilization soaring to 66.0%, Tensor Core utilization is maintained at a remarkable 71.6% of the cuBLAS baseline, demonstrating that compute throughput is preserved (Figure [12\(](#page-10-0)b)). This high pipeline efficiency is enabled by our data layout. As seen in Figure [12\(](#page-10-0)c), shared memory bank conflicts are virtually eliminated (∼4.7K) compared to the millions incurred by methods like DietGPU. This conflict-free access is a prerequisite for our fine-grained pipeline, ensuring smooth data flow and maximizing SIMT throughput.

## 6.2 Decompression Kernel Performance

To further dissect decompression efficiency, we benchmark our standalone ZipServ-Decomp kernel. Figure [13](#page-10-1) presents the total decompression time for all weights in a full Transformer block of LLaMA3.1-8B and Mistral-24B. ZipServ-Decomp achieves average speedups of 2.14×, 1.83×, and

<span id="page-10-2"></span>![](_page_10_Figure_10.jpeg)

Figure 14. Cross-generation performance comparison.

<span id="page-10-3"></span>![](_page_10_Figure_12.jpeg)

Figure 15. ZipServ performance under different N settings.

1.10× over DietGPU, nvCOMP, and DFloat11, respectively. Although the TCA-TBE format was co-designed to support fused execution with matrix multiplication, its structure proves highly efficient for standalone decompression as well. This efficiency stems from its fixed-length, warp-aligned design, which eliminates control divergence and enables warpsynchronous per-thread decoding. In contrast, although existing baselines are explicitly optimized for decompression, they often rely on variable-length, entropy-coded formats. These lead to thread divergence, serialized bit parsing, and irregular memory access that degrade GPU efficiency.

## <span id="page-10-4"></span>6.3 Performance Across GPU Generations and Tiers

To establish forward compatibility, we benchmark ZipGEMM on the latest NVIDIA RTX5090 and compare it against toptier datacenter A100 and H800 using LLaMA3.1-8B and Mistral-24B GateUp\_proj layers at batch size 32. We first directly port ZipGEMM to the Blackwell-based RTX5090 without exploiting new features (e.g., Tensor Memory and asynchronous WMMA execution [\[32\]](#page-15-19)). As shown in Figure [14,](#page-10-2) ZipGEMM delivers substantial speedups over cuBLAS\_TC on RTX5090—1.34× for LLaMA3.1-8B and 1.87× for Mistral-24B—confirming the design to be forward-compatible. ZipGEMM also narrows the consumer–datacenter divide: on an RTX4090, ZipGEMM outperforms the standard cuBLAS\_TC on A100 with LLaMA3.1-8B (0.195 ms vs. 0.215 ms, 9.3% faster) and is only 2.7% slower on Mistral-24B (0.530 ms vs. 0.516 ms), effectively placing it in the same performance class. This trend intensifies on newer hardware. While a standard RTX5090 trails the H800 by 53.3% (LLaMA3.1-8B) and 125.7% (Mistral-24B), ZipGEMM reduces these deficits to 14.1% and 20.8%, respectively (Figure [14\(](#page-10-2)b)), approaching datacenter-level performance on consumer GPUs.

#### <span id="page-11-0"></span>6.4 Overhead Analysis

We analyze the system overhead from two perspectives: runtime inference overhead and offline preparation cost. **1** Runtime Overhead. Figure 15 quantifies the overhead of ZIPSERV during inference across different N settings  $(N = BS \times Seqlen)$ . In the decode stage (small N, typically 1– 128), the fused ZipGEMM kernel incurs no overhead. Instead, it consistently outperforms the cuBLAS\_TC baseline in these memory-bound regimes, with on-the-fly decompression fully hidden within the kernel execution. For the compute-bound prefill stage (large N, e.g., 8192), where ZipGEMM's onthe-fly decompression overhead outweighs its benefits from reduced memory access, ZIPSERV switches to a decoupled pipeline. The efficient decompression kernel first expands the compressed weights, followed by cuBLAS\_TC GEMMs. This incurs a limited overhead of only  $\sim 4\%/2\%$  of the GEMM time at N = 8192/16384. ② Offline Compression Cost. Bevond runtime performance, we also evaluate the one-time cost of preparing the model. Compressing the LLaMA-3.1-8B model takes approximately 2.5 minutes on a 16-core Intel Xeon 8352V CPU. Given that this is an offline operation performed only once prior to deployment, it does not impact the critical path of online serving and is negligible when amortized over the model's lifecycle.

#### 6.5 End-to-end Inference Performance

**Setup.** We evaluate the end-to-end inference performance of ZIPSERV on a range of representative models and hardware configurations: LLaMA3.1-8B on one RTX4090 GPU, Mistral-24B on two L40S GPUs, and LLaMA3.1-70B on four L40S GPUs with tensor parallelism. We benchmark using batch sizes of 8 and 32, with varied output sequence lengths of 128, 256, 512, 1024, and 2048 tokens to simulate different serving scenarios. We compare ZIPSERV against three leading baseline systems: • vLLM [39], a state-of-the-art LLM inference and serving framework; 2 Transformers [75], a widely adopted standard library; and 3 DFloat11 [85], representing state-of-the-art performance for lossless compression-based inference frameworks. We measure two key metrics: end-toend request latency (total time to generate the full output sequence) and throughput (output tokens per second). As shown in Figure 16, ZIPSERV consistently demonstrates superior performance across all tested configurations.

**Results.** For **latency**, on average, across all models and batch sizes, ZIPSERV reduces latency by 17.60%, 60.79%, and 82.13% compared to vLLM, Transformers, and DFloat11, respectively. For **throughput**, ZIPSERV provides average speedups of 1.22× over vLLM, 3.18× over Transformers, and 8.52× over DFloat11. The performance gains are pronounced for long-context generation, where the memory-bandwidth savings and computational efficiency of the fused ZipGEMM kernel in the decode phase become dominant. For instance, when generating 2048 output tokens with batch size of 32

<span id="page-11-1"></span>![](_page_11_Figure_7.jpeg)

**Figure 16.** End-to-end performance comparison.

using LLaMA3.1-8B, ZIPSERV achieves a throughput of 1105 tokens/sec, resulting in a 1.66× speedup over vLLM. We also analyzed the **memory consumption** during inference. For LLaMA3.1-8B, Mistral-24B, and LLaMA3.1-70B, ZIPSERV reduces the weight footprint of 14.96/43.92/131.56 GB down to 10.83 (72.4%)/31.30 (71.3%)/93.52 (71.1%) GB, respectively. The reduction in weight storage further enhances serving efficiency in two key ways. First, it enables the deployment of larger models on resource-constrained hardware. Second, the freed memory can be allocated to the KV cache, allowing memory managers like vLLM's PagedAttention [39] to support larger batch sizes and longer contexts, thereby converting static weight savings into dynamic throughput gains.

Breakdown Analysis. We further dissect the performance gains by analyzing the latency and memory composition of LLaMA-3.1-8B on an RTX4090, as detailed in Figure 17. In the baseline vLLM system (at sequence length 1024), GEMM operations dominate the runtime, consuming 24.99 ms (83.6% of total latency). ZIPSERV effectively alleviates this bottleneck: the fused ZipGEMM kernel, combined with residual dense GEMMs, reduces the total linear layer latency to 14.76 ms, a 1.69× improvement. Since Attention (3.02 ms) and other overheads (1.88 ms) remain constant, these kernel-level gains directly drive the end-to-end speedup. On the memory front, ZIPSERV compresses the static weights from 14.96 GB to 11.18 GB. This 3.78 GB saving is automatically repurposed by the memory manager to expand the KV cache capacity from 5.07 GB to 8.60 GB (a 1.70× increase), thereby enabling the higher throughput and longer context support observed in our end-to-end benchmarks.

## 7 Limitation and Discussion

ZIPSERV is designed for the increasingly important deployment scenario on resource-constrained consumer-grade and inference-optimized GPUs, where limited memory bandwidth makes lossless compression a powerful lever for efficiency. On such platforms, ZIPSERV consistently delivers substantial acceleration and memory savings. To stress-test performance under more bandwidth-relaxed conditions, we

<span id="page-12-0"></span>![](_page_12_Figure_2.jpeg)

Figure 17. Breakdown of end-to-end inference time and memory consumption.

<span id="page-12-1"></span>![](_page_12_Figure_4.jpeg)

Figure 18. Performance on training-oriented GPUs.

also benchmarked on training-oriented datacenter GPUs (A100, H800), where ZipGEMM may not always match the highly optimized cuBLAS baseline (Figure [18\)](#page-12-1). This reflects a hardware–software mismatch rather than an algorithmic limitation: abundant HBM (HBM2e/HBM3) alleviates the memory bottlenecks ZipServ is designed to mitigate, while lower core frequencies (e.g., 1410 MHz on A100 vs. 2520 MHz on RTX4090) make the intensive ALU workload harder to hide within the software pipeline. Nevertheless, ZipServ still provides best-in-class support for compressed inference. Our standalone decompression kernel outperforms state-ofthe-art by up to 2.64×, and ZipGEMM remains the fastest fused GEMM kernel. As shown in [§6.3,](#page-10-4) ZipServ also enables consumer-grade GPUs to close much of the gap with elite datacenter accelerators, offering a compelling cost-performance proposition for deployment on accessible hardware.

While ZipServ targets bit-exact inference, a comparison with lossy techniques is instructive. ZipGEMM was benchmarked against the Marlin W8A16 FP8 kernel on an RTX4090 GPU, using a representative weight shape (28672 × 4096) at batch size 32. Although ZipGEMM trails Marlin-W8A16 in latency (0.194 ms vs. 0.143 ms), the resulting 1.36× gap aligns closely with the ratio of effective bit-widths (∼11 bits vs. FP8). This indicates that our design reduces and hides the overhead of complex lossless decompression within the memory access latency. Furthermore, ZipServ is orthogonal to lossy methods and can be applied atop quantized weights to exploit residual redundancy, combining aggressive compression with enhanced performance [\[26\]](#page-14-7).

Three key directions are envisioned for extending ZipServ. First, the TCA-TBE format can be adapted for lossless KV

Cache compression, addressing the dominant memory bottleneck in long-context serving [\[45\]](#page-15-20). Second, although currently optimized for NVIDIA architectures, ZipGEMM can be adapted to other matrix accelerators, including Intel AMX [\[37\]](#page-15-15) and AMD Matrix Cores [\[61\]](#page-16-10). This extensibility is supported by the hardware-agnostic nature of the core design, as the integer arithmetic and population count instructions required for decompression are widely supported across modern instruction sets. Finally, ZipServ is applicable to broader system-level challenges, including efficient model checkpointing [\[65,](#page-16-11) [71\]](#page-16-4) and communication compression in distributed training [\[73,](#page-16-12) [84\]](#page-16-13).

