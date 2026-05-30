# 4 TINYCHAT: MAPPING AWQ ONTO EDGE PLATFORMS

AWQ can substantially reduce the size of LLMs. However, converting the theoretical memory savings from W4A16 (4-bit weight, 16-bit activation) quantization into measured speedup is non-trivial. Alternative W8A8 quantization methods, such as SmoothQuant [\(Xiao et al.,](#page-13-0) [2022\)](#page-13-0), maintain *the same* data precision for both storage and computation. This allows the dequantization procedure to be seamlessly integrated into the computation kernel's epilogue. On the other hand, W4A16 quantization employs *different* data types for memory access and computation. As a result, its dequantization must be incorporated into the primary computation loop for optimal performance, posing implementation challenges. To tackle this, we introduce TinyChat: a nimble system for AWQ model inference. It boasts a PyTorch frontend and a backend harnessing device-specific instruction sets (e.g., CUDA/PTX, Neon, AVX).

#### 4.1 Why AWQ Helps Accelerate On-Device LLMs

To understand the acceleration opportunities in quantized LLMs on the edge, we start by profiling the latency breakdown of LLaMA-7B [\(Touvron et al.,](#page-13-0) [2023a\)](#page-13-0) model on an RTX 4090 GPU. We adopt an inference batch size of 1, catering for edge use cases, and implement the model in FP16 with NVIDIA FasterTransformer.

Context *vs* generation latency. As in Figure 3(a), it takes 310 ms to generate 20 tokens, while summarizing a prompt

**Figure 4.** SIMD-aware weight packing for ARM NEON with 128-bit SIMD units. Original weights are reordered and packed to align with the bit width so that the weights can be unpacked into bytes at runtime using AND and shift bitwise operations with a 128-bit mask.

with 200 tokens only takes 10 ms. Consequently, the generation phase is substantially slower than the context stage, particularly for on-device interactive applications.

Generation stage is memory-bound. To accelerate the generation phase, we conduct a roofline analysis in Figure 3(b). The 4090 GPU has a peak computation throughput of 165 TFLOPS and a memory bandwidth of 1TB/s. Therefore, any workload with arithmetic intensity (the ratio of FLOPs to memory access) less than 165 is memory bounded on 4090 GPUs. Notably, when executed in FP16, the generation stage for on-device LLMs has arithmetic intensity≈1. This underscores the memory-bound nature of the workload. Since the FLOPs of a given model is fixed, the only way to improve the peak performance is to reduce the total amount of memory traffic. AWQ reduces the weight memory by four times.

Weight access dominates memory traffic. We therefore further break down the memory access for weight and activation in Figure 3(c). Clearly, weight access dominates the memory traffic for on-device LLMs. Quantizing the model weights to 4 bit integers will approximately increase the arithmetic intensity to 4 FLOPs/Byte, leading to a 4TFLOPS peak performance in Figure 3(b). Since weight-only quantization leads to a lower bit width for weights (and thus higher theoretical performance upper bound), it is natural for AWQ to follow this setting for on-device LLM applications.

#### 4.2 Deploy AWQ with TinyChat

To this end, we demonstrated that 4-bit weight quantization could lead to a  $4\times$  theoretical peak performance. We further design TinyChat to realize this speedup. On GPUs, we only focus on implementing essential components, including attention, layer normalization, and linear projection kernels. The flexible frontend allows easy customization and fast support for new models. TinyChat with 4-bit AWQ achieves more than  $3\times$  speedup compared with the Huggingface FP16 implementation across different families of LLMs on GPUs. On CPUs, we lower the entire computation graph to C++ to minimize overhead.

**On-the-fly weight dequantization.** For quantized layers, as the hardware does not provide multiplication instructions between INT4 and FP16, we need to dequantize the integers

to FP16 before performing matrix computation. We avoid writing dequantized weights into DRAM by fusing dequantization kernels with the matrix multplication kernel. Note that such fusion is adopted for both matrix-matrix (MM) and matrix-vector (MV) product kernels.

**SIMD-aware weight packing.** On-the-fly weight dequantization reduces intermediate DRAM access, but remains expensive. For instance, dequantizing a single 4-bit weight involves 1 shift, 1 bitwise AND, and 1 FMA scaling operations, while the dequantized weight undergoes only 1 FMA computation. This process is particularly costly on CPUs with SIMD architecture that favor vectorized instructions. To mitigate this, we suggest platform-specific weight packing tailored to the bitwidth of a device's SIMD units. Figure 4 demonstrates our strategy for ARM CPUs with 128-bit SIMD registers offering up to  $1.2 \times$  speedup. Here, each register holds 32 4-bit weights, sequenced as  $w_0, w_{16}, w_1, w_{17}, ..., w_{15}, w_{31}$ . This approach requires just three SIMD instructions to unpack all 32 weights, as opposed to 3 scalar instructions per weight in a conventional packing  $(w_0, w_1, ..., w_{31})$ . Generally, for  $2^n$ -bit SIMD registers, adjacent weights will have indices off by  $1/8 \times 2^n$ , since each register can hold  $1/8 \times 2^n$  8-bit integers. On GPUs, we found it more efficient to pack each 8 weights into  $w_{\{0,2,4,6,1,3,5,7\}}$  following (Kim et al., 2022).

**Kernel fusion.** We also extensively apply kernel fusion to optimize on-device LLM inference. For layer normalization, we fuse all operators (*e.g.* multiplication, division and square root) into a single kernel. For attention layers, we fuse QKV projections into a single kernel, and also perform on-the-fly positional embedding calculation. We also preallocate KV caches and perform cache updates within the attention kernel. Kernel fusion is particularly useful for models with inefficient forward pass implementations, such as Falcon (Penedo et al., 2023) and StarCoder (Li et al., 2023c). Notably, the computation time for each FP16 kernel is in the order of 0.01ms on the 4090 GPU, comparable to the GPU kernel launch overhead. Hence, reducing number of kernel calls through kernel fusion leads to direct speedups.

<span id="page-6-0"></span>

| PPL↓         |                              |                              | Llama-2                      |                              | LLaMA                        |                              |                              |                              |
|--------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|
|              |                              | 7B                           | 13B                          | 70B                          | 7B                           | 13B                          | 30B                          | 65B                          |
| FP16         | -                            | 5.47                         | 4.88                         | 3.32                         | 5.68                         | 5.09                         | 4.10                         | 3.53                         |
| INT3<br>g128 | RTN<br>GPTQ<br>GPTQ-R<br>AWQ | 6.66<br>6.43<br>6.42<br>6.24 | 5.52<br>5.48<br>5.41<br>5.32 | 3.98<br>3.88<br>3.86<br>3.74 | 7.01<br>8.81<br>6.53<br>6.35 | 5.88<br>5.66<br>5.64<br>5.52 | 4.88<br>4.88<br>4.74<br>4.61 | 4.24<br>4.17<br>4.21<br>3.95 |
| INT4<br>g128 | RTN<br>GPTQ<br>GPTQ-R<br>AWQ | 5.73<br>5.69<br>5.63<br>5.60 | 4.98<br>4.98<br>4.99<br>4.97 | 3.46<br>3.42<br>3.43<br>3.41 | 5.96<br>6.22<br>5.83<br>5.78 | 5.25<br>5.23<br>5.20<br>5.19 | 4.23<br>4.24<br>4.22<br>4.21 | 3.67<br>3.66<br>3.66<br>3.62 |

Table 4. AWQ improves over round-to-nearest quantization (RTN) for different model sizes and different bit-precisions. It consistently achieves better perplexity than GPTQ (w/ and w/o reordering) on LLaMA & Llama-2 models.

| Wikitext2 PPL↓ | Mixtral-8x7B | Mistral-7B |
|----------------|--------------|------------|
| FP16           | 5.94         | 4.14       |
| INT4-g128      | 6.05         | 4.30       |
| INT3-g128      | 6.52         | 4.83       |

Table 5. AWQ quantization results on Mistral-7B-Instructv0.2[\(Jiang et al.,](#page-12-0) [2023\)](#page-12-0) and Mixtral-8x7B-Instruct-v0.1 model [\(Jiang et al.,](#page-12-0) [2024\)](#page-12-0). The PPL result on wikitext shows that AWQ can achieve superior quantization performance on different model architectures including LLMs with GQA and Mixture-of-Experts (MoE) models.

