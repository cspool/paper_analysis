# FlashInfer employs a dynamic load-balanced scheduling framework to handle input dynamism effectively.

It separates compile-time tile size selection from runtime scheduling, offering lightweight APIs that adaptively manage scheduling with changing KV-Cache lengths during inference, while maintaining compatibility with CUDA-Graph's requirement for constant configurations [\(Gray,](#page-12-0) [2019;](#page-12-0) [Nguyen et al.,](#page-13-0) [2021\)](#page-13-0).

Figure 1 depicts our system design. We evaluated Flash-Infer' performance across standard LLM serving environments and innovative scenarios, including prefix sharing and speculative decoding. FlashInfer have been integrated with mainstream LLM serving engines, including vLLM [\(Kwon](#page-12-0) [et al.,](#page-12-0) [2023\)](#page-12-0), MLC-Engine [\(MLC Community,](#page-13-0) [2024;](#page-13-0) [Lai](#page-12-0) [et al.,](#page-12-0) [2023\)](#page-12-0), and SGLang [\(Zheng et al.,](#page-16-0) [2023b\)](#page-16-0), we assessed its impact on end-to-end latency and throughput improvements, showing significant enhancements on standard LLM serving benchmarks and novel applications such as longcontext inference and parallel generation.

Our contributions include:

- Introduction of flexible block-sparse and composable formats addressing KV-Cache storage heterogeneity for efficient memory management and access.
- Development of a customizable attention template accommodating diverse attention variants, ensuring highperformance execution via JIT compilation.
- Design of a dynamic scheduling framework managing input dynamism while remaining compatible with CUDAGraph, maximizing hardware utilization.
- Comprehensive evaluation demonstrating substantial improvements in kernel and end-to-end performance.

#### <span id="page-2-0"></span>2 BACKGROUND

#### 2.1 FlashAttention

FlashAttention (Dao et al., 2022) is an efficient algorithm for computing exact attention with reduced memory usage. During the forward pass, it employs the online-softmax trick (Milakov & Gimelshein, 2018), updating attention outputs on-the-fly using a constant amount of on-chip memory, thus avoiding materializing the attention matrix in GPU global memory. FlashAttention2&3 (Dao, 2023; Shah et al., 2024) improve performance by optimizing loop ordering and pipeline design for Ampere and Hopper GPUs. FlashInfer builds upon these advancements.

The operational intensity of FlashAttention is given by  $O\left(\frac{1}{1/l_{qo}+1/l_{kv}}\right)$ , where  $l_{qo}$  and  $l_{kv}$  are the query and keyvalue cache lengths, respectively. In LLM serving, the query length is either equal to (prefill) or smaller than (decode/incremental prefill) the key-value cache length, simplifying the operational intensity to  $O(l_{qo})$ . Techniques like batching (Yu et al., 2022) do not alter this operational intensity. Multi-Query Attention (MQA) (Shazeer, 2019) and Grouped Query Attention (GQA) (Ainslie et al., 2023) optimize the KV-Cache size by grouping queries and sharing the same KV-Cache entries. The ratio of the number of queries to the number of KV-Cache entries is denoted as the group size  $g = \frac{H_{qo}}{H_{kv}}$ , enhancing operational intensity to  $O(g \cdot l_{qo})$ .

### 2.2 Attention Composition

Block-Parallel Transformer (BPT) (Liu & Abbeel, 2023) demonstrates that attention outputs for the same query and different keys/values can be composed by preserving both the attention outputs and their scales. Let  ${\bf q}$  be a query, and let  ${\mathcal I}$  be an index set. We define the *attention scale* over  ${\mathcal I}$  via the log-sum-exp operation on the attention scores:

$$LSE(\mathcal{I}) = \log \sum_{i \in \mathcal{I}} \exp(\mathbf{q} \cdot \mathbf{k}_i)$$
 (1)

where  $\mathbf{k}_i$  is the *i*-th key vector. The corresponding *attention* output  $\mathbf{O}(\mathcal{I})$  is then

$$\mathbf{O}(\mathcal{I}) = \sum_{i \in \mathcal{I}} \frac{\exp(\mathbf{q} \cdot \mathbf{k}_i)}{\exp(\mathbf{LSE}(\mathcal{I}))} \cdot \mathbf{v}_i$$
 (2)

We define the *Attention State* for  $\mathcal{I}$  as the tuple of *attention output* and *attention scale*:  $\begin{bmatrix} \mathbf{O}(\mathcal{I}) \\ \mathbf{LSE}(\mathcal{I}) \end{bmatrix}$ . Crucially, the Attention State of  $\mathcal{I} \cup \mathcal{J}$  can be derived by composing the states of  $\mathcal{I}$  and  $\mathcal{J}$ . Specifically, introducing an operator  $\oplus$ :

$$\begin{split} \begin{bmatrix} \mathbf{O}(\mathcal{I} \cup \mathcal{J}) \\ \mathbf{LSE}(\mathcal{I} \cup \mathcal{J}) \end{bmatrix} &= \begin{bmatrix} \mathbf{O}(\mathcal{I}) \\ \mathbf{LSE}(\mathcal{I}) \end{bmatrix} \oplus \begin{bmatrix} \mathbf{O}(\mathcal{J}) \\ \mathbf{LSE}(\mathcal{J}) \end{bmatrix} \\ &= \begin{bmatrix} \frac{\exp(\mathbf{LSE}(\mathcal{I}))\mathbf{O}(\mathcal{I}) + \exp(\mathbf{LSE}(\mathcal{J}))\mathbf{O}(\mathcal{J})}{\exp(\mathbf{LSE}(\mathcal{I})) + \exp(\mathbf{LSE}(\mathcal{J}))} \\ \log(\exp(\mathbf{LSE}(\mathcal{I})) + \exp(\mathbf{LSE}(\mathcal{J}))) \end{bmatrix} \end{split}$$

Since  $\oplus$  is associative and commutative, multiple sets of attention states can be composed in any order. Ring-Attention (Liu et al., 2023) and Flash-Decoding (Dao et al., 2023) utilize this property to offload partial-attention computations, thereby reducing memory usage and improving hardware efficiency. In FlashInfer, the *Attention State* is adopted as the canonical output of an attention operation, and  $\oplus$  serves as the standard reduction operator (analogous to summation in GEMM) on these states.

## 2.3 Block/Vector Sparsity

Block Compressed Sparse Row (BSR) is a hardwareefficient sparse format that groups non-zero elements into contiguous matrices of size  $(b_r, b_c)$ , as opposed to the random scattering found in unstructured sparsity. This format offers several advantages over the standard Compressed Sparse Row (CSR) format. BSR improves register reuse efficiency (Im et al., 2004; Buluç et al., 2009) and demonstrates better compatibility with hardware matrix multiplication units on GPUs and NPUs (Narang et al., 2017; Gray et al., 2017). In addition, it provides the ability to skip empty blocks, reducing computational overhead. BSR's efficiency is particularly evident when subcomputations are aligned with hardware matrix multiplication instructions, such as NVIDIA's mma instructions. Traditionally, tensor core instructions operate on minimal dimensions of 16 (or larger for newer GPUs), leading most block-sparse kernels to use block sizes that are multiples of (16, 16). However, this approach is not always optimal for applications with fine-grained sparsity patterns (Wang et al., 2023). Many attention libraries restrict their block sizes to multiples of (128, 128) for block-sparse attention kernels.

Recent research (Chen et al., 2021; Li et al., 2022) has demonstrated that efficient utilization of the tensor core can be achieved with smaller block sizes, such as (16,1) for matrix B in GEMM, or (1,16) for matrix A (also known as vector-sparse). This is accomplished by first gathering rows/columns into contiguous shared memory and then applying dense tensor cores to these contiguous shared-memory data. This approach is particularly beneficial for applications with fine-grained sparsity patterns. FlashInfer builds upon these techniques to support blocks with arbitrary column sizes  $B_c$ , offering greater flexibility and efficiency in handling diverse sparsity patterns.

#### <span id="page-3-0"></span>3 DESIGN

In this section, we introduce the system design of FlashInfer. We begin by presenting the data structure employed in Flash-Infer and demonstrate how Block-Sparse Row (BSR) acts as a versatile abstraction for KV cache storage in attention kernels. Next, we discuss the FlashInfer compiler, which supports various attention variants, alongside a dynamic-aware runtime scheduler that facilitates load-balanced scheduling of attention kernels. Finally, we describe the user-level API designed for integrating FlashInfer with existing LLM serving systems.

