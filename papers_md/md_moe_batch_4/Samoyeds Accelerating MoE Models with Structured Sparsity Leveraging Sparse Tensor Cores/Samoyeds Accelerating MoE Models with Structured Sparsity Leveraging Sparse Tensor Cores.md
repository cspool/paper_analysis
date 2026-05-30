# Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores

Chenpeng Wu<sup>∗</sup> cpwu\_sjtu@sjtu.edu.cn Shanghai Jiao Tong University Shanghai, China

Qiqi Gu<sup>∗</sup> qiqi.gu@sjtu.edu.cn Shanghai Jiao Tong University Shanghai, China

Heng Shi† heng.shi@sjtu.edu.cn Shanghai Enflame Technology Co.Ltd; Shanghai Jiao Tong University Shanghai, China

Jianguo Yao† jianguo.yao@sjtu.edu.cn Shanghai Jiao Tong University Shanghai, China

# Abstract

The escalating size of Mixture-of-Experts (MoE) based Large Language Models (LLMs) presents significant computational and memory challenges, necessitating innovative solutions to enhance efficiency without compromising model accuracy. Structured sparsity emerges as a compelling strategy to address these challenges by leveraging the emerging sparse computing hardware. Prior works mainly focus on the sparsity in model parameters, neglecting the inherent sparse patterns in activations. This oversight can lead to additional computational costs associated with activations, potentially resulting in suboptimal performance.

This paper presents Samoyeds, an innovative acceleration system for MoE LLMs utilizing Sparse Tensor Cores (SpTCs). Samoyeds is the first to apply sparsity simultaneously to both activations and model parameters. It introduces a bespoke sparse data format tailored for MoE computation and develops a specialized sparse-sparse matrix multiplication kernel. Furthermore, Samoyeds incorporates systematic optimizations specifically designed for the execution of dual-side structured sparse MoE LLMs on SpTCs, further enhancing system performance. Evaluations show that Samoyeds outperforms SOTA works by up to 1.99× at the kernel level and 1.58× at the model level. Moreover, it enhances memory

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. EuroSys '25, March 30–April 3, 2025, Rotterdam, Netherlands

© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.

ACM ISBN 979-8-4007-1196-1/25/03 <https://doi.org/10.1145/3689031.3717455>

Haibing Guan hbguan@sjtu.edu.cn Shanghai Jiao Tong University Shanghai, China

efficiency, increasing maximum supported batch sizes by 4.41× on average. Additionally, Samoyeds surpasses existing SOTA structured sparse solutions in both model accuracy and hardware portability.

CCS Concepts: • Hardware → Emerging technologies; • Computing methodologies → Artificial intelligence; Parallel computing methodologies; • Mathematics of computing → Mathematical software performance.

Keywords: Structured Sparsity, Mixture-of-Experts, Large Language Model, Sparse Tensor Core, System Performance

#### ACM Reference Format:

Chenpeng Wu, Qiqi Gu, Heng Shi, Jianguo Yao, and Haibing Guan. 2025. Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores. In Twentieth European Conference on Computer Systems (EuroSys '25), March 30–April 3, 2025, Rotterdam, Netherlands. ACM, New York, NY, USA, [18](#page-17-0) pages. [https:](https://doi.org/10.1145/3689031.3717455) [//doi.org/10.1145/3689031.3717455](https://doi.org/10.1145/3689031.3717455)

# 1 Introduction

The emergence of ChatGPT has marked a pivotal milestone in the development of Large Language Models (LLMs), positioning them as a leading approach in deep learning. As LLMs evolve, there has been a notable escalation in both the scale of model parameters and computational demands[\[10,](#page-13-0) [22,](#page-14-0) [44\]](#page-15-0). Recent models, such as LLaMA3[\[54\]](#page-15-1), feature an impressive 400 billion parameters, a substantial increase from earlier models like BERT-base[\[21\]](#page-14-1), which had only 110 million parameters. This rapid evolution presents significant challenges for deploying these LLMs within existing AI infrastructure, particularly given the limitations imposed by the pace of hardware advancements.

Moreover, the adoption of novel architectures, particularly the Mixture-of-Experts (MoE) technique[\[35\]](#page-14-2), introduces additional complexity. The MoE layer, which consists of multiple experts, has been widely integrated into emerging LLMs due to its ability to enhance generalization[\[23,](#page-14-3) [33,](#page-14-4) [35\]](#page-14-2) and

<sup>∗</sup>Equal Contribution †Corresponding author

manage multi-modal tasks effectively[\[41,](#page-15-2) [53\]](#page-15-3). This architectural innovation introduces unique demands for storage, bandwidth, and computation resources. Addressing the increasing scale and the novel architecture of LLMs is critical for their efficient deployment on contemporary AI accelerators. This underscores the need for developing evolutionary systems and methodologies to alleviate these pressures while fully utilizing existing hardware.

To address these challenges, sparse computing[\[29,](#page-14-5) [34\]](#page-14-6) has emerged as a promising methodology to reduce memory footprint and computational costs by eliminating zero or least important elements. However, general sparsity, or unstructured sparsity[\[5,](#page-13-1) [26\]](#page-14-7), is inefficient to implement on modern AI accelerators, particularly GPUs. The core issue with unstructured sparsity lies in the irregular pattern of non-zero elements, which complicates scheduling and balancing in SIMD programming models. This irregularity significantly hinders the utilization of performance-critical features in modern GPUs, such as coalesced memory access and warp-level synchronization. To overcome these limitations, structured sparsity has been introduced[\[37,](#page-14-8) [39,](#page-15-4) [42\]](#page-15-5), effectively eliminating performance issues by representing data with regular sparse patterns. Also, this new paradigm of sparse computing has been supported by NVIDIA GPUs in its Sparse Tensor Cores (SpTCs) since the Ampere architecture, featuring a 2× peak performance boost compared to its dense counterparts[\[40\]](#page-15-6). This support makes structured sparsity a viable hardware-software co-design solution for tackling the performance challenges of LLMs.

Currently, structured sparsity is primarily utilized to encode deep learning model parameters, significantly reducing model storage and computational workloads. This approach is exemplified by methods like cuSPARSELt[\[6\]](#page-13-2) and VENOM[\[12\]](#page-13-3). However, this design overlooks a critical aspect of modern LLMs: the inherent sparse pattern in activations, primarily induced by the routing mechanism of MoE layers. Neglecting these features represents a missed opportunity for further optimizations in LLM applications. It is noteworthy that the underlying sparse pattern in activations is well-recognized and investigated in state-of-the-art (SOTA) works[\[9,](#page-13-4) [25\]](#page-14-9). However, none of these works effectively integrates this design principle with the capabilities of hardware.

To address these problems, we propose the Samoyeds system, which leverages sparsity in model parameters and activations simultaneously, without introducing extra overhead. To be specific, Samoyeds introduces a novel dual-side sparse data format, where one side represents the structured sparsity in model parameters, and the other side captures the dynamic sparsity that emerges during token routing in the MoE computation. By respecting the unique instruction requirements of SpTC and memory access pattern in dual-side structured sparse computation in MoE model, a specialized kernel execution scheme is proposed, which incorporates a

series of systematic optimizations, including tiling orchestration, data stationary management, data packing reorganization, and layout optimization.

In summary, this paper makes the following contributions:

- We revisit the underlying sparse patterns in largescale MoE LLMs and explore the potential optimization by harnessing the dual-side sparse pattern untouched by existing SOTA works. This insight identifies a hardware-software co-optimization opportunity by leveraging the structured sparse computing facilities supported by NVIDIA Sparse Tensor Cores.
- We introduce a novel sparse data format for MoE computation with more flexible sparsity configuration and more efficient memory access for the dual-side sparse pattern. A corresponding execution scheme of sparsesparse matrix multiplication kernels is proposed to improve hardware utilization.
- We implement several optimizations specifically tailored for this sparse format within the MoE execution, including hierarchical tiling, data stationary improvement, and data packing reorganization, collectively ensuring optimal computational efficiency.
- We extend Samoyeds kernel with various data layout configuration, meeting different requirements of matrix multiplication operands inside the MoE module, allowing seamless integration with existing MoE LLMs, and minimizing the overall memory I/O overhead.
- Experimental results show that the Samoyeds kernel outperforms SOTA sparse libraries, achieving up to 18.76× speedup over Sputnik (unstructured) and 1.99× over VENOM (structured). At the model level, Samoyeds surpasses the SOTA vLLM framework by up to 1.58× while increasing the average maximum batch size by 4.41×.

# 2 Background

### 2.1 Mixture-of-Experts (MoE)

<span id="page-1-0"></span>![](_page_1_Figure_15.jpeg)

Figure 1. MoE LLM Architecture.

The typical MoE-based models, as shown in Figure [1,](#page-1-0) consist of multiple Transformer Blocks[\[23,](#page-14-3) [35,](#page-14-2) [47,](#page-15-7) [57\]](#page-15-8). Each block includes attention, MoE and normalization layers. Within the MoE layer, a routing mechanism selects appropriate experts for each token. Tokens are then processed by Multilayer Perceptron (MLP) layers with multiple linear projections, called

<span id="page-2-0"></span>![](_page_2_Figure_2.jpeg)

Figure 2. Time Breakdown of MoE Models. Left: Without Flash-Attention; Right: With Flash-Attention.

<span id="page-2-1"></span>![](_page_2_Figure_4.jpeg)

Figure 3. Data Patterns in Different Sparse Formats. Blank cells represent sparse elements.

experts. The outputs of these experts are then propagated to the final output through a weighted sum.

Unlike traditional models that compute across all activations, MoE models enhance efficiency by selectively activating experts, enabling the computation of partial activations to be skipped[\[61\]](#page-15-9). The MoE architecture is promising as it improves model generality without increasing demands on storage and computation resources[\[23,](#page-14-3) [35,](#page-14-2) [48\]](#page-15-10). Research demonstrates that MoE models achieve competitive performance to larger dense models[\[45\]](#page-15-11).

Numerous studies, such as Flash-Attention[\[19,](#page-14-10) [20\]](#page-14-11) and KV cache[\[43\]](#page-15-12), have optimized the attention layer, however, the MoE layer has received less focus. The execution time breakdown for a transformer block is illustrated in Figure [2.](#page-2-0) Notably, the MoE layer, in most models, accounts for over half of the total processing time. With Flash-Attention enabled, the proportion of time consumed by the MoE layer exceeds 80% in these models. This phenomenon underscores the urgent need for optimizations of the MoE layer to enhance overall model efficiency.

## <span id="page-2-3"></span>2.2 Sparse Data Formats

Sparse computation employs different formats for data representation, providing a more efficient alternative to dense formats. These formats can be broadly categorized into two types: unstructured and structured, as illustrated in Figure [3.](#page-2-1) Unstructured formats such as Coordinate List (COO) and Compressed Sparse Row (CSR) organize non-zero elements without constraints of regular patterns. However,

<span id="page-2-2"></span>![](_page_2_Figure_11.jpeg)

Figure 4. 2:4 Sparse Encoding and Mapping for SpTC.

this flexibility complicates GPU processing as it impedes efficient parallel execution. Structured formats, which organize data into regular patterns like columns or diagonals, allow for efficient GPU processing but potentially lose critical features[\[26,](#page-14-7) [39,](#page-15-4) [42\]](#page-15-5).

N:M structured sparsity[\[31,](#page-14-12) [37,](#page-14-8) [52\]](#page-15-13) format imposes additional constraints on structured formats by requiring the retention of N units within every contiguous set of M units. These units can vary in granularity, ranging from individual elements to vectors or blocks, as illustrated in Figure [3.](#page-2-1) This format offers predictable patterns that facilitate efficient hardware implementation on accelerators, without compromising the flexibility and capability of feature representation. Given these advantages, our discussion will primarily focus on the N:M structured sparsity format.

## 2.3 Sparse Tensor Core (SpTC)

NVIDIA has introduced the third-generation Tensor Cores in its Ampere architecture GPUs, which can exploit the finegrained sparsity in model parameters[\[40\]](#page-15-6). The 2:4 sparse matrix multiplication operations are efficiently executed on SpTC, as illustrated in Figure [4.](#page-2-2) The original sparse matrix of size × is encoded into a non-zero data matrix and a 2-bits metadata matrix. The data matrix compresses all non-zero values into a dense format and the metadata matrix records the positions of non-zero elements in each contiguous set of 4 elements. NVIDIA's cuSPARSELt library provides kernels that support this format, enabling efficient compression and sparse operations[\[6\]](#page-13-2). Other GPU vendors also implement similar sparse arithmetic logic unit (ALU) in their products, such as the CDNA3 series GPUs (Instinct MI300) from AMD[\[3\]](#page-13-5). Additionally, modern deep learning frameworks such as PyTorch[\[2\]](#page-13-6) and compilers like LLVM[\[1\]](#page-13-7) now support specific data types, operators, and intermediate representations for 2:4 sparsity, further illustrating its widespread application. Besides existing kernels, programmers can utilize the SpTC in CUDA using the mma.sp instruction provided by Parallel Thread Execution (PTX) Instruction Set Architecture (ISA) since version 7.0, which provides the flexibility to customize kernels for various formats.

<span id="page-3-0"></span>![](_page_3_Figure_2.jpeg)

Figure 5. Redundancy in Data Flow of the MoE layer.

#### <span id="page-3-2"></span>3 Motivation

#### <span id="page-3-5"></span>3.1 Redundancy in Data Flow

In the MoE layer, input tokens I are routed and subsequently assigned to a subset of available experts. As illustrated in Figure 5, the initial input tensor is permuted into several new tensors, each corresponding to a certain expert. Specifically, the input tensor for expert  $E_i$  contains all the tokens that have been routed to  $E_i$ . The output of expert  $E_i$  is then propagated to a new tensor that matches the size of I, ensuring that the dimension of the MoE layer output aligns with the dimension of the original input.

During the input permutation phase, the creation of extra intermediate tensors necessitates additional memory management tasks, including *memory allocation* and *data movement*, thereby increasing the processing overhead. In the weighted un-permutation phase, the outputs of the experts are initially transferred from registers to the global memory on GPUs. Subsequently, the element-wise product operation reloads these outputs from global memory back into the registers. This *additional memory I/O* within GPUs introduces significant overheads, impacting overall performance efficiency. Therefore, these overheads necessitate a customized kernel capable of efficiently handling the sparsity in inputs.

## <span id="page-3-4"></span>3.2 Redundancy in Model Parameters

Recent research[13, 18, 24] has demonstrated redundancy in model parameters. In response, researchers have developed various sparse formats to accelerate the computing process by omitting the computation of redundant parameters.

With unstructured sparse formats like COO and CSR, individual parameters are removed based on their importance. Existing GPU kernels designed for these unstructured formats are primarily optimized for high-performance computing (HPC) applications, where the sparsity ratio often exceeds 95%[5, 14]. In contrast, sparsity ratios in LLMs typically fall between 50% and 90%[37]. Consequently, employing these unstructured formats in LLMs does not guarantee performance improvement, as the computational savings are often mitigated by the substantial overhead of decoding[40]. With structured sparse format, parameters in different patterns are removed by group. While this format provides significant performance improvements in computation, it also limits the expressiveness of the model and reduces accuracy.

<span id="page-3-1"></span>![](_page_3_Figure_11.jpeg)

Figure 6. Inefficient Memory Access when Input is Sparse. 

● illustrate the dense scenario. ● and ● lead to I/O amplification, while ● causes uncoalesced memory access.

Fortunately, the N:M structured sparse format provides both the benefits of these two formats. It has been demonstrated to have a negligible impact on the accuracy of LLMs with 50% or even larger sparse ratio[11, 15, 24, 40]. Meanwhile, hardware manufacturers like NVIDIA also provide hardware units (e.g. SpTC) to further accelerate the computation process. Therefore, to reduce the cost of computing redundant model parameters, kernels should be capable of handling this N:M structured sparse format.

# <span id="page-3-3"></span>3.3 Problem with Existing Solutions

To address the aforementioned redundancy in MoE execution, several solutions[9, 12, 25] have been proposed. However, these works either fail to explore the the potential of structured sparsity in model parameters, which can be accelerated by sparse ALU, or waste the memory bandwidth for dual-side sparse matrix multiplication.

Among them, Megablocks[25] provides a block sparse representation and a customized kernel, while vLLM[9] proposes a kernel that combines the computation processes of all experts into a single kernel to address data flow redundancies in MoE layers. However, these solutions overlook the opportunity to leverage structured sparsity in model weights. Moreover, their highly customized designs make it challenging to incorporate the structured sparsity efficiently.

Meanwhile, other research has developed kernels optimized for structured sparsity, delivering notable performance improvements over SOTA kernels for dense matrices or unstructured sparse matrices. However, kernels like BBS[11] and nmSPARSE[37] fail to utilize SpTC for further acceleration. Solutions such as cuSPARSELt[6] and DFSS[15] leverage SpTC but impose a sparse ratio limit of 50%. VENOM[12] allows for a flexible sparse ratio while utilizing SpTC with a V:N:M format, specifically optimized for sparse-dense matrix multiplication scenarios. As depicted in Figure 6, when encountering a sparse column in model weights, it skips the multiplication with the corresponding row in inputs. This approach maintains an efficient memory access pattern with coalesced memory access, as illustrated in **①**.

<span id="page-4-0"></span>![](_page_4_Figure_2.jpeg)

Figure 7. Samoyeds Dual-side Sparse Data Format.

However, challenges arise when both weight and input matrices are sparse. In such situations, as shown in Figure 6, the skipped row and the sparse column in inputs break the data into smaller tiles, adversely reducing performance. For instance, the data may be loaded into shared memory in formats ②, ③, and ③. Formats ② and ③ involve loading either sparse column data or skipped row data unnecessarily, leading to severe I/O amplification at high sparse ratios. Moreover, the data in format ④ are not contiguous in memory, leading to uncoalesced memory access and reducing GPU memory I/O bandwidth.

# 4 Design of Samoyeds

To address the limitations outlined in §3, Samoyeds introduces a new sparse data format and a dedicated kernel execution scheme. Furthermore, we adopt novel systematic optimizations, including tiling, data stationary, packing, and optimized layout, specifically tailored for efficient structured sparse MoE computation.

# <span id="page-4-2"></span>4.1 Dual-sided Sparse Data Format and Kernel Scheme

As discussed in §3.3, existing structured sparse data format can result in uncoalesced memory access or memory I/O amplification, thereby degrading overall kernel performance. Samoyeds firstly introduces a novel structured sparse format specifically tailored for sparse-sparse matrix multiplication in MoE computation, employing distinct sparsity patterns for weight and input matrices, as illustrated in Figure 7.

For weight matrices, the sparse pattern integrates 2:4 element-wise and vector-wise structured sparsity. Samoyeds employs the 2:4 pattern to align with the ISA requirements of the SpTC, leveraging the superior speedup of sparse ALU. Given that the 2:4 pattern enforces a fixed sparsity ratio of 50%, an additional sparse pattern is required to provide greater flexibility. The granularity of the added sparse pattern should be more coarse-grained to preserve the prior 2:4 element-wise sparse pattern. Prior studies[37] have proven block-wise sparsity is too coarse-grained to preserve model accuracy. Therefore, Samoyeds adopts vector-wise sparsity alongside the existing 2:4 sparsity, striking a balance between representation flexibility and accuracy preservation.

```
Algorithm 1: Samoyeds Kernel Scheme
   // Inputs in Samoyeds sparse format.
                                                     // Optimized Layout
   Input: A, Indices, Metadata, B, SEL
  Output: C
3 Init shared memory for A, Indices, B, SEL;
4 Init registers for A, Indices, Metadata, B, SEL, C;
5 Load All SEL from GMEM to SMEM;
   fetch \leftarrow 0;
7 for compute = 0 to \frac{k}{k_b} do
        Load Metadata from GMEM to Register;
                                                               // Packing
8
        while fetch < compute + num_{pipe} and fetch < \frac{k}{k_b} do
             // Fetch Stage
11
             Load I, A, B from GMEM to SMEM;
                                                                 // Tiling
12
             Commit group for pipeline;
             Increase fetch;
13
14
        end
        Wait group for pipeline;
15
        // Compute Stage
16
17
        Load I, A, B from SMEM to Register;
                                                                 // Tiling
        if compute \equiv 0 \pmod{\frac{V}{k_h}} then
18
19
             Shuffle C Register
                                                      // Data Stationary
20
        end
        Sparse MMA computations:
22 end
  Store C from Register to GMEM
                                                     // Optimized Layout
```

The weight sparse pattern is shown on the left of Figure 7. The original weight data is segmented into structured sparse blocks of size  $M \times V$ . Each vector within a block, termed as a Sub-Row, contains multiple SpTC units. Within each block, only N Sub-Rows are retained, while the others are pruned, where N depends on the target sparse ratio. The SpTC units within the selected Sub-Row are further pruned to conform to the sparse pattern supported by the hardware.

Based on the defined sparse pattern, the original weight is encoded into three components: data, indices, and metadata. The data matrix, with the shape of  $\frac{m}{M} \times \frac{k}{2}$ , serves as a compressed representation of the original sparse matrix of size  $m \times k$ . The compression ensures that elements are ordered sequentially, enabling GPU-friendly decoding during computation. The indices matrix, with size  $\frac{m}{M} \times \frac{k}{V}$ , captures the relative positions of the retained Sub-Rows within their respective blocks. Meanwhile, the metadata matrix, with size  $\frac{m}{M} \times \frac{k}{2}$ , details the sparse pattern for each SpTC unit. Notably, each item within the metadata matrix is encoded into 2-bits as required by the SpTC hardware specifications.

The input data sparse pattern, illustrated on the right of Figure 7, introduces a selection array (*SEL*) and encodes the columns in vector-wise sparsity way. Notably, this design naturally aligns with the sparsity pattern presented in token routing, ensuring mathematical equivalence with the original computation process.

To unleash the potential of the proposed sparse data format, Samoyeds customizes a GPU kernel to accelerate computation. The pseudo-code of the kernel is described in Algorithm 1. After receiving the encoded sparse input, the kernel begins with an initialization phase, followed by the fetch stage (line 10-13), where data is loaded from global

<span id="page-5-0"></span>![](_page_5_Figure_2.jpeg)

Figure 8. Tiling Strategy for GPU Memory Hierarchy.

memory to shared memory using the non-blocking *cp.async* instruction. This asynchronous copy operation is then committed in group. Next, in the compute stage (line 16-21), the kernel invokes the wait method for a specific group and synchronously loads data from shared memory into registers according to the predefined tiling size. Once all data are in place, the *mma.sp* instruction triggers SpTC to perform sparse computation. These two stages (fetch and compute) are efficiently overlapped using a pipeline mechanism. At the end of the kernel execution, the output matrix *C* is transferred from the registers back to global memory.

Notably, while our kernel scheme aligns with the typical execution flow of matrix multiplication kernel, adapting it to the unique Samoyeds data format and orchestrating various kernel optimization techniques—such as tiling, data stationary, packing, and optimized layout—is a non-trivial task. The following sections provide a detailed description of the optimizations implemented in Samoyeds to enhance performance.

#### 4.2 Enhanced Data Locality with Tiling

Tiling is a widely used technique in GPU kernel design that partitions data into equal-sized subsets to exploit data locality. By leveraging the multi-level memory hierarchy of GPUs, this approach significantly enhances memory access efficiency, reducing data-loading latency and increasing computational throughput. However, the Samoyeds data format encodes the original matrix into multiple matrices, fundamentally altering the data access patterns. Therefore, orchestrating an effective tiling strategy for these matrices becomes a non-trivial task.

As illustrated in Figure 8, we introduce a three-step tiling strategy to optimize Samoyeds kernel. Considering a  $m \times k \times n$  matrix multiply problem, denoted as  $C = A \times B$ . Notably, only  $len_d$  columns from matrix B are selected for computation, which are recorded in the selection array. For clarity, matrix A is presented in its original, non-encoded format. In step  $\mathbf{0}$ , each thread block is responsible for computing a tile of size  $m_b \times n_b$  in matrix C. During each iteration, data in the global memory is further partitioned along the k dimension. Each

<span id="page-5-1"></span>![](_page_5_Figure_9.jpeg)

<span id="page-5-2"></span>Figure 9. Data Stationary Optimization for Output Matrix.

thread block then loads segments of matrices A ( $m_b \times k_b$ ), B  $(k_b \times n_b)$  into shared memory. Note that although the tiling size for matrix B specifies  $n_b$  columns, the original layout in global memory includes more columns than  $n_b$  due to the omission of sparse columns. Only the columns identified by the mapping in the selection array are actually loaded, ensuring an efficient use of memory resources. In step 2, the output of matrix C is segmented and assigned to several warps, with each warp handling a tile size of  $m_w \times n_w$ . Given that threads in modern GPUs are organized into scheduling units called warps, each warp loads the corresponding sections of matrices A and B from shared memory into thread registers. In step **3**, these registers are further partitioned to align with the requirements of the SpTC hardware. The SpTC is then invoked to compute the output size of  $m_i \times n_i$  for acceleration. Moreover, since the metadata matrix, encoded in 2 bits, is relatively small, applying 3-step tiling leads to inefficient memory access on GPUs. Therefore, it skips the innermost tiling step and is loaded directly into registers. Similarly, the indices matrix is loaded using the 3-step tiling scheme but with a larger tiling size than the corresponding matrix A to enhance memory efficiency.

The selection of the tiling size requires careful consideration of multiple factors from various perspectives. First, hardware specifications impose constraints on the tiling size. Specifically, the values of  $(m_i, k_i, n_i)$  must comply with the requirements of the *mma.sp* instruction. Additionally, in the three-step tiling scheme, the amount of data loaded into shared memory and registers is bounded by available hardware resources. Second, while a larger tiling size improves data locality during kernel execution, a smaller tiling size increases parallelism by dividing the matrix into more execution units, thereby enhancing GPU hardware utilization. Consequently, a trade-off must be carefully balanced based on the specific problem size. Third, when integrated into MoE models, an excessively large subrow size V may degrade model accuracy. Additionally, the tiling size  $K_b$  for the reduction dimension K is constrained by V. Therefore, we employ a larger tile size in non-reduction dimensions (M and N) to improve data reuse while keeping  $K_b$  relatively small to avoid accuracy loss. Additionally, for models with more experts, the tiling size for N dimension should be reduced to avoid potential padding overhead or degraded hardware efficiency.

#### <span id="page-6-1"></span>4.3 Maximized Data Reuse with Data Stationary

Data stationary refers to the strategy of pinning data elements in faster memory hierarchy throughout the tiling loops. However, given that Samoyeds encodes data in a unique format, directly applying existing designs can result in suboptimal choices for stationary locations, potentially resulting in degraded kernel performance.

In matrix multiplication, the input matrices *A* and *B* are read-only. In contrast, the output matrix *C* requires both reading and writing in each tiling iteration. Thus, a well-established design approach[55] is keeping the input data in shared memory and maintaining the output data in registers.

However, when iterating on reduction along the k dimension, the Sub-Rows may span across different lines between sparse blocks, as illustrated in Figure 9(a). This presents a new challenge: when the tiling window shifts from one Sub-Row to another, the output of the SpTC must be remapped to different rows. Thus, when invoking the mma.sp instruction, it is crucial to carefully select the output registers based on the mappings recorded in the indices matrix to ensure the correctness of the computation. However, simply passing the indexed output to the instruction can result in the stationary location of output C falling back to local memory on GPU, as shown on the left of Figure 9(a), which will significantly reduce the kernel performance.

To address this challenge, we introduce additional intermediate registers  $C_{IR}$ , as shown on the right of Figure 9(b). All registers storing C are initialized to zero at the beginning of kernel execution. Stored C data is shuffled with  $C_{IR}$  according to the indices matrix every  $\frac{V}{k_b}$  iteration, which corresponds to the point when the tiling window shifts from one Sub-Row to another. This optimization minimizes frequent memory transfers between global memory and registers, thereby enhancing overall kernel performance with mathematical correctness.

Besides, Samoyeds adopts operator fusion to further enhance data reuse by improving cache utilization, eliminating intermediate results, increasing locality, and reducing data movement. This technique allows the succeeding operator to utilize the results of the preceding operator without roundtrips to global memory. Specifically, the activation function and its precedent operator are fused. Furthermore, the weighted accumulation operation, which contains a broadcast of scalar data and dot multiplication, is fused with matrix multiplication.

#### 4.4 Coalesced Memory Access with Data Packing

Data packing is a commonly used technique to enhance kernel performance by enabling coalesced memory access and maximizing memory bandwidth. However, the effectiveness of this technique is closely tied to the data representation

<span id="page-6-0"></span>

| Мар    | <b>Mapping:</b> $[row\_idx, col\_idx] \longrightarrow [\frac{col\_idx}{16} \times 8 + row\_idx \frac{\sqrt{8}}{8}, \frac{row\_idx}{8} \times 16 + col\_idx \frac{\sqrt{16}}{16}]$ |                       |                       |                       |                 |   |               |                 |                       |
|--------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------|-----------------------|-----------------------|-----------------|---|---------------|-----------------|-----------------------|
| Column | 03                                                                                                                                                                                | 47                    | 811                   | 1215                  | 1631            |   | Column<br>Row | 015             | 1631                  |
| 0      | T <sub>0</sub> [30]                                                                                                                                                               | T <sub>0</sub> [74]   | T <sub>0</sub> [118]  | T <sub>0</sub> [1512] | <del>-</del>    | - | ,0+           | T₀[150]         | T <sub>0</sub> [3116] |
| 1      | T <sub>4</sub>                                                                                                                                                                    | T <sub>4</sub>        | T <sub>4</sub>        | T <sub>4</sub>        | T <sub>5</sub>  | Ш | 1             | T <sub>4</sub>  | T <sub>4</sub>        |
|        |                                                                                                                                                                                   |                       |                       |                       |                 | Ш |               |                 |                       |
| 7      | T <sub>28</sub>                                                                                                                                                                   | T <sub>28</sub>       | T <sub>28</sub>       | T <sub>28</sub>       | T <sub>29</sub> | Ш | 7             | T <sub>28</sub> | T <sub>28</sub>       |
| 8      | T₀[1916]                                                                                                                                                                          | T <sub>0</sub> [2320] | T <sub>0</sub> [2724] | T <sub>0</sub> [3128] | T,              | - | 8             | Ť,              | T,:                   |
| 9      | T <sub>4</sub>                                                                                                                                                                    | T <sub>4</sub>        | T <sub>4</sub>        | T <sub>4</sub>        | T <sub>S</sub>  | Ш | 9             | T <sub>5</sub>  | T <sub>5</sub>        |
|        |                                                                                                                                                                                   |                       |                       |                       | 1               |   |               |                 |                       |
| 15     | T <sub>28</sub>                                                                                                                                                                   | T <sub>28</sub>       | T <sub>28</sub>       | T <sub>28</sub>       | T <sub>29</sub> |   | 15            | T <sub>29</sub> | T <sub>29</sub>       |
|        | (a) SPTC View                                                                                                                                                                     |                       |                       |                       |                 |   | (b)           | Memory V        | iew                   |

**Figure 10.** Packing Strategy with Reorganized Metadata.

format. Our newly proposed Samoyeds sparse format differs significantly from existing sparse formats, posing challenges for directly integrating existing optimizations into the Samoyeds kernel. Additionally, the data must be organized to align with the SpTC specifications, requiring customized packing designs. Samoyeds introduces a data packing strategy that optimally arranges the storage of data matrices *A* and *B*, as well as metadata.

According to the SpTC specification, the data of each thread is not contiguous in input matrices. For matrix A, data is packed in global memory according to the matrix format, and the transfer from global memory to shared memory conforms to the 128-bit memory transactions of modern GPUs. Compliance with the SpTC specification is achieved during transferring from shared memory to registers by invoking *ldmatrix* instructions. Additionally, data in shared memory is arranged with permutation to prevent bank conflicts. Considering matrix B, which retains row contiguity and exhibits column sparsity, we pack it with transposition in global memory. This transposition allows for contiguous memory access within rows and enables skipping over rows with zero values, optimizing memory bandwidth utilization through coalesced memory access. The approach for packing matrix *B* in shared memory and transferring data to registers mirrors that of matrix A.

However, for the metadata matrix, each element is encoded into the 2-bits format, incompatible with the specification of *ldmatrix* instruction. For efficient memory access, we propose a special packing format for the metadata matrix. Taking the mma.sp.m16n8k32 instruction for example, the metadata in bfloat type occupies a 32-bits register containing 16 2-bits vectors from the view of SpTC, as shown in Figure 10(a). We need to concatenate four consecutive metadata into a 16-bit metadata block and iterate twice to fill the 32-bits register. To avoid missing in the L2 cache, our proposed metadata packing for continuous access on device memory is shown in Figure 10(b). To be specific, the metadata is a 2-bits matrix with a size of 16×16. The element with location of [row\_idx, col\_idx] is mapped to the location of  $[row\_idx\%8 \times 2 + \frac{col\_idx}{8}, col\_idx\%8 + \frac{row\_idx}{8} \times 8]$ . Having got the new data mapping, the metadata loading for each thread is aligned to 32-bits, which is consistent with the 32-bits memory transactions of GPUs.

<span id="page-7-0"></span>![](_page_7_Figure_2.jpeg)

- (a) Output Sparsity in MoE Layer.
- <span id="page-7-1"></span>(b) Kernel Performance Improvement Optimized Output Layout.

Figure 11. Layout Optimization for Kernel Output.

## <span id="page-7-3"></span>4.5 Reduced Memory I/O with Optimized Layout

The data layout is crucial for optimizing GPU kernels, as an efficient layout reduces memory I/O and enhances computational performance. However, the constraints imposed by hardware instructions, combined with the sparsity inherent to the MoE computation, introduce additional challenges for data layout design. These limitations make existing solutions inefficient in such scenarios, necessitating specialized optimizations in the data layout.

Traditionally, the linear layer performs the operation xW, where x represents the input and W denotes the model weights. To align with SpTC hardware requirements, this computation is restructured to  $(W^T x^T)^T$ , where T denotes the matrix transpose operation. This transformation can significantly increase I/O volume across the memory hierarchy of GPUs. To mitigate this overhead, Samoyeds employs the three-step layout optimization for operands of kernels. Firstly, the transposition of the W matrix is performed during the offline model pruning phase, eliminating the memory I/O of transposition. Secondly, the input x for the MoE layer typically shows sparsity by row, corresponding to the token routing results, which allows for efficient contiguous storage in row-major format. Transposing this input prior to matrix multiplication would result in inefficient, scattered memory accesses, amplifying memory I/O. To address this, the transposition of the input is efficiently executed within the kernel as data transfers from global to shared memory, leveraging the hardware fast path feature of GPUs. Besides, the output transposition is integrated within the kernel, further minimizing memory I/O and boosting computational efficiency.

As shown in Figure 11(a), a typical expert layer consists of three linear layers: <code>gate\_proj</code>, <code>up\_proj</code>, and <code>down\_proj</code>. The token routing mechanism selects a subset of the input for each expert and aggregates the outputs from all experts with weighted accumulation (<code>Acc</code>). While the final output of the MoE module is dense after accumulation, the intermediate results within each expert exhibit a row-wise sparse pattern, with the sparsity ratio determined by the number of experts. This sparsity can lead to unnecessary memory transfers of zero values, causing performance degradation. To address

<span id="page-7-2"></span>**Table 1.** Hardware Support for Samoyeds on Prevalent GPUs.

|             |              | Mandatory     | Optional                    |                          |  |
|-------------|--------------|---------------|-----------------------------|--------------------------|--|
| GPU         | Architecture | Sparse<br>ALU | Asynchronous<br>Memory Copy | Collective<br>Load/Store |  |
| NVIDIA A100 | Ampere       | 1             | 1                           | 1                        |  |
| NVIDIA 4090 | Ada Lovelace | 1             | ✓                           | ✓                        |  |
| NVIDIA H100 | Hopper       | 1             | ✓                           | ✓                        |  |
| AMD W7900   | RDNA3        | X             | <b>X</b> *                  | <b>X</b> *               |  |
| AMD MI300   | CDNA3        | ✓             | <b>X</b> *                  | <b>X</b> *               |  |

this issue, Samoyeds adopts a compressed output layout that aligns with the input sparse pattern described in  $\S 2.2$ . This optimized layout eliminates redundant data transfers while preserving the mathematical equivalence of the computation. The performance gain achieved with this layout across varying input sparsity ratios is demonstrated in Figure 11(b). For typical MoE model configurations, this optimization speeds up the kernel by  $1.05\times$  on average for models with low input sparsity and  $2.66\times$  for models with higher sparsity.

## 5 Implementation

#### 5.1 Kernel Implementation

The Samoyeds kernels are implemented in CUDA, utilizing the SpTC hardware with inline assembly *mma.sp* instructions from NVIDIA PTX ISA. These kernels are compiled into a dynamic library via the NVIDIA CUDA Compiler (NVCC), making them accessible to other programs. Additionally, the compiled executable is exposed as a Python module through module registration with pybind11.

## 5.2 Compatibility with Different Hardware

The mandatory requirement of the Samoyeds kernel is sparse ALU, which is specifically designed for matrix multiplications involving a sparse matrix with 2:4 structured sparsity and a dense matrix. Beyond the computational units, memory efficiency can also influence kernel performance. On the one hand, the pipeline mechanism, foundational to this implementation, leverages asynchronous data movement (e.g. cp.async) and concurrent kernel execution to enhance throughput. On the other hand, the efficiency of kernel execution is further augmented by collective load and store (e.g. ldmatrix), which orchestrate multiple threads—such as those encapsulated by wrap in CUDA and wave in ROCm.

The Samoyeds kernel can be effectively adapted to other accelerators equipped with the aforementioned features. As demonstrated in Table 1, the Samoyeds kernel is compatible with most prevalent GPUs, including NVIDIA GPUs that incorporate Ampere architecture or more advanced versions[8], as well as AMD GPUs that are equipped with

CDNA3 architecture[3], which all satisfy the mandatory requirement for sparse ALU. However, the lack of native support for asynchronous memory copy and collective matrix load/store operations on AMD GPUs may lead to degradation in memory efficiency and increased development effort. Despite the applicability, the optimized kernel configuration (e.g. pipeline stages and tiling size) can be distinct on different hardware, based on their difference in resources, including the number of clusters/processors, cache line size, and shared memory capacity. The relationship between kernel configuration and hardware specification will be further explored in §6.6.

#### 6 Evaluation

The performance of Samoyeds system is evaluated on 3 distinct levels, including kernel-level performance improvements (§6.1), the enhancements achieved within the MoE layer (§6.2), the speedup and batch size benefits in end-to-end scenarios (§6.3). A breakdown analysis is provided in §6.4. Additionally, the accuracy of models pruned with Samoyeds sparse format are assessed in §6.5. Finally, the performance portability is examined in §6.6.

**Evaluation Setup.** The evaluation is conducted on the platforms equipped with Intel i7-12700 CPU, 16G×2 DDR5 memory, running Ubuntu 22.04LTS and installed with CUDA 12.1, cuSPARSELt 0.4.0, PyTorch 2.1.0, Transformers v4.40.0 and vLLM 0.4.0.post1. The GPU used in the evaluation is NVIDIA GeForce RTX 4070 Super (except in §6.6). The CPU frequency scaling is disabled in all experiments for fairness.

**Baselines.** *Kernel level:* The baselines for kernel level performance evaluation consist of several SOTA dense and sparse kernel libraries, including *cuBLAS*[4], *Sputnik*[26], *cuSPARSELt*[6] and *VENOM*[12]. *cuBLAS* and *cuSPARSELt* are black-box vendor-specific libraries provided by NVIDIA, which are manually written and optimized with expertise and present the peak performance on NVIDIA GPUs for dense and structured sparse operations, respectively. *Sputnik* is a leading open source library targeting accelerating the sparse operations in deep learning applications maintained by Google. *VENOM*, proposed in late 2023, accelerates sparse matrix multiplication with SpTC hardware, providing 1.38× speedup compared to the cuSPARSELt library.

Model level: We compare Samoyeds against three leading solutions: Transformers[56], the most popular framework for computing LLM models, with the latest released version (v4.40.0, on April 18, 2024); MegaBlocks[25], which is specifically designed and optimized for block-sparse operations in MoE computations, surpassing solutions like Tutel[32] and Megaron-LM[49]; and vLLM-DS[9, 17] which provides a SOTA fused kernel for the MoE process. It should be noticed that baseline vLLM-DS integrates the recent implementation and optimization of a fused kernel for MoE models (merged on March 2, 2024), achieving approximately a 2.8×

**Table 2.** Configurations of MoE Models.

<span id="page-8-1"></span>

| Model         | Expert | Hidden Size | Intermediate Size | Config Num |
|---------------|--------|-------------|-------------------|------------|
| Qwen2-MoE     | 60     | 1408        | 2048              | CFG#1      |
| DeepSeek-MoE  | 64     | 1408        | 2048              | CrG#1      |
| MiniCPM-MoE   | 8      | 2304        | 5760              | CFG#2      |
| OpenMoE-34B   | 32     | 3072        | 12288             | CFG#3      |
| Mixtral-8×7B  | 8      | 4096        | 14336             | CFG#4      |
| Mixtral-8×22B | 8      | 6144        | 16384             | CFG#5      |

<span id="page-8-2"></span>![](_page_8_Figure_10.jpeg)

**Figure 12.** Kernel Performance Comparison on Synthetic and Realistic Benchmarks. The synthetic benchmark covers 238 sizes; the realistic benchmark reflects typical model sizes.

speedup compared with previous non-fusion version[7]. To ensure fairness, all experiments in model level employs Flash-Attention2 in the decoder layer.

#### <span id="page-8-0"></span>6.1 Kernel Performance

<span id="page-8-3"></span>**6.1.1 Overall Kernel Performance.** The overall kernel-level performance of Samoyeds is evaluated on the synthetic benchmark to reveal the benefits of Samoyeds in wide application scenarios, and the realistic benchmark to present the performance benefits when applied to specific models. The synthetic benchmark consists of 238 distinct cases, with dimensions m, k, and n ranging from 256 to 16384. The cases in the realistic benchmark are extracted from MoE LLMs as detailed in Table 2.

The results are illustrated in Figure 12, demonstrating that Samoyeds consistently outperforms other baselines. For synthetic benchmark, shown on the left of Figure 12, compared to the SOTA sparse matrix library VENOM, Samoyeds kernel exhibits a speedup of up to 1.99×. Furthermore, the relative speedup over cuBLAS, cuSPARSELt, and Sputnik reaches as high as  $5.44\times$ ,  $3.18\times$ , and  $18.76\times$ , respectively. The results on the realistic benchmark are shown on the right of Figure 12. Samoyeds provides an average speedup of 2.33× (peaking at 2.49×) compared to the best baseline, VENOM. Compared to kernel libraries cuBLAS and cuSPARSELt, Samoyeds achieves average speedups of 3.95× and 4.29×, respectively. Despite being specifically optimized for sparse operations in deep learning, Sputnik still shows poor performance because it fails to leverage the structured pattern of the sparse data and hardware features. In contrast, Samoyeds significantly outperforms Sputnik, delivering an average speedup of 33.02×.

It is noteworthy that, under CFG#5, the overall throughput of both Samoyeds and VENOM is lower than that observed

<span id="page-9-1"></span>![](_page_9_Figure_2.jpeg)

Figure 13. Throughput Trend with Varying Operator Size.

with other configurations. This performance degradation is mainly caused by the skewness between dimensions m and n, with m being substantially larger. This imbalance leads to decreased memory efficiency during computation, a corner case that could be mitigated with a more sophisticated tiling implementation. It also adversely affects the throughput of other baseline kernels. Despite this, Samoyeds still offers a significant speed advantage over cuBLAS and cuSPARSELt under CFG#5, where VENOM does not, achieving a speedup of  $2.47\times$  relative to VENOM. These findings underscore the efficiency and practical applicability of Samoyeds in realworld computing scenarios, demonstrating its superiority over existing approaches in handling MoE computations.

<span id="page-9-2"></span>**6.1.2 Throughput with Different Operator Sizes.** We investigate the performance trends of Samoyeds kernel by profiling the throughput as matrix dimensions increase. The results, as illustrated in Figure 13, demonstrate that Samoyeds consistently outperforms all other baselines across nearly all matrix sizes. When compared to the best baseline, VENOM, Samoyeds achieves a speedup of up to  $2.77\times$  within a varying range of dimension m, and up to  $2.34\times$  and  $2.58\times$  for dimensions k and n, respectively.

As operand sizes scale up, the throughput of Samoyeds kernel increases significantly before reaching the peak performance. The reason for this increment varies with dimension. For dimension k, the initialization and write-back overhead of the kernel remains constant, allowing these costs to be amortized over a growing number of reduction computations. Consequently, the throughput of Samoyeds asymptotically approaches its maximum value as shown in Figure 13(b). For dimensions m and n, the number of warps per kernel increases as the operand sizes get larger. This increase allows the hardware warp scheduler to select from a larger pool of active warps when the current warp stalls, thereby optimizing resource utilization. This is reflected in Figure 13(a) & 13(c), where throughput scales linearly with dimensions m and n before it reaches a peak, benefiting from increased parallelism. These trends highlight the efficiency and scalability of Samoyeds across a wide range of matrix dimensions, underscoring its robust applicability in scenarios that require high computational throughput.

Notably, when dimensions *m* or *n* are set to 256, Samoyeds slightly underperforms compared to VENOM due to the

limited parallelism available at this kernel size, which can be addressed by implementing smaller kernel tiling sizes. Additionally, unlike other baselines presenting stable peak performance across varying operand sizes, fluctuations occur in Samoyeds kernels because our kernel has not been specially adapted for corner cases, indicating the potential for further refinement in Samoyeds kernels. Moreover, as shown in Figures 13(a) and 13(c), the performance exhibits a slight decline when the corresponding dimension size is 4096. This behavior can be explained by two factors: (1) as the size increases from 2048 to 4096, scheduling more warps on each SM leads to L1 cache eviction when switching between warps, which significantly reducing the cache hit rate by 76.38%, thereby degrading overall performance; (2) when the size further increases to 8192, the larger problem size enables better scheduling opportunities and amortizes the tail wave overhead, resulting in a 5.90% increase in active warps per scheduler and improved performance.

## <span id="page-9-0"></span>6.2 MoE Layer Performance

The current SOTA LLMs that employ the MoE technique can be categorized into two types. The first type utilizes multiple experts with identical routing priorities, where each token is dispatched to specific experts based on routing outcomes. The second type incorporates several isolated shared experts alongside individually routed experts. This setup enforces all tokens to be processed by all these shared experts in addition to their assigned routed experts. Therefore, it is crucial to demonstrate the effectiveness of Samoyeds under these two distinct routing types.

Figure 14 shows the speedup achieved on the MoE layer compared to the original Transformers solution. The configuration of the size and number of routed experts aligns with model configurations in Table 2, with the number of tokens set as 4096. The left of the figure displays results for MoE layers incorporating two isolated shared experts, while the right shows results for MoE layers without shared experts.

With shared experts enabled, Samoyeds achieves an average speedup of 1.46×, peaking at 1.73× compared to the most widely-used Transformers. Additionally, it outperforms MegaBlocks and the SOTA baseline vLLM-DS, providing a speedup of up to 1.66× and 1.53×, respectively. Without shared experts, Samoyeds achieves an average speedup of 1.45× and peaks at 1.68× compared to Transformers. Furthermore, Samoyeds surpasses other baselines in most models, including vLLM-DS.

Notably, the *NS* marker denotes *Not Supported*, as the OpenMoE-34B features a distinct activation function within the MLP layer, which is incompatible with the specialized kernels provided by MegaBlocks and vLLM-DS. Additionally, the OOM (Out-of-Memory) marker indicates that the corresponding solution failed to complete the computing process due to memory constraints. Samoyeds underperforms slightly when applied in the Mixtral-8×22B, due to

<span id="page-10-1"></span>![](_page_10_Figure_2.jpeg)

Figure 14. Execution Speedup for the MoE Layer. NS indicates not supported due to kernel incompatibility with OpenMoE-34B. OOM denotes out-of-memory errors preventing execution completion.

<span id="page-10-2"></span>![](_page_10_Figure_4.jpeg)

Figure 15. Speedup in End-to-end Latency of MoE Models.

the skewness in expert size, a tiling corner case previously discussed in [§6.1.1.](#page-8-3) This issue could be mitigated by further implementing different tiling sizes. In the same configuration, MegaBlocks suffers from significant slowdowns, and vLLM-DS encounters OOM errors, which highlights the robustness of Samoyeds in these conditions.

Moreover, in Qwen2-MoE and DeepSeek-MoE without shared experts, the speedup advantage of Samoyeds is less evident when compared to MegaBlocks and vLLM. This reduced performance can be attributed to two factors. Firstly, the smaller size of experts in these models results in lower parallelism, as discussed in [§6.1.2,](#page-9-2) leading to reduced computation speeds. Secondly, the number of tokens processed by each expert must be aligned with the tiling size of kernels. If this number does not align perfectly, the original input must be supplemented with empty tokens, creating extra computational work. Therefore, models with more experts will suffer severe padding overhead, considering the number of tokens each expert needs to process decreases. By simply implementing a smaller tiling size, the extra padding overhead can be saved, indicating further potential speedup improvements in Samoyeds.

# <span id="page-10-0"></span>6.3 End-to-End Model Evaluation

We benchmark the end-to-end performance of Samoyeds and the baselines on various real-world leading MoE LLMs shown in Table [2.](#page-8-1) To accommodate the memory capacity constraints of GPUs, we measure the performance of a single decoder layer. This is justified by two observations: (1) prevalent MoE LLMs are decoder-only, with decoder layers accounting for over 90% of the total execution time and (2) the decoder layers share similar architectures and sizes, leading to comparable execution times. Notably, the input

<span id="page-10-3"></span>![](_page_10_Figure_10.jpeg)

Figure 16. Throughput under Different Batch Sizes.

for Samoyeds and other baselines remains consistent, ensuring identical routing distributions and guaranteeing a fair comparison.

6.3.1 Overall Model Performance. We initially compare the overall performance of these models using Samoyeds and other baselines, with a default sequence length of 4096 and a batch size of 1. For the OpenMoE-34B, we adjust the sequence length to 2048 due to its maximum sequence length constraints. Additionally, for DeepSeek-MoE and Qwen2- MoE, we increase the batch size to 16 to leverage the larger number of experts within these models. MegaBlocks and vLLM-DS are not supported in OpenMoE-34B due to incompatibility. Meanwhile, they both fail to complete processing Mixtral-8×22B due to OOM errors.

As illustrated in Figure [15,](#page-10-2) Samoyeds significantly outperforms all competing baselines. In particular, Samoyeds achieves a remarkable speedup of up to 2.36× (1.42× on average) compared to Transformers. Additionally, it delivers speedup of up to 1.31× and 1.30× relative to MegaBlocks and the SOTA baseline vLLM-DS, respectively. These results highlight the effectiveness of our optimization strategies in enhancing performance.

6.3.2 Throughput with Different Batch Sizes. We explore the throughput of various models across different batch sizes, as illustrated in Figure [16.](#page-10-3) For models equipped with smaller expert configurations, including Qwen2-MoE and DeepSeek-MoE, we maintain a sequence length of 4096 per batch. Conversely, for other models featuring larger experts,

<span id="page-11-2"></span>**Table 3.** Maximum Batch Sizes for MoE Models.

| Model         | Transformers | MegaBlocks | vLLM-DS | Samoyeds | Boost over the<br>Best Baseline |
|---------------|--------------|------------|---------|----------|---------------------------------|
| MiniCPM-MoE   | 118          | 89         | 91      | 123      | 1.04×                           |
| OpenMoE-34B   | 3            | -          | -       | 56       | 18.67×                          |
| Mixtral-8×7B  | 62           | 36         | 36      | 86       | 1.38×                           |
| Mixtral-8×22B | 30           | 0          | 0       | 53       | 1.77×                           |
| Qwen2-MoE     | 35           | 28         | 28      | 44       | 1.26×                           |
| DeepSeek-MoE  | 22           | 21         | 21      | 52       | 2.36×                           |

we reduce the sequence length to 1024 per batch to provide a clearer insight into throughput trends with increasing batch sizes. OpenMoE-34B is not supported by MegaBlocks and vLLM due to incompatibility.

Our method, Samoyeds, shows superior throughput compared to other baselines across a variety of configurations and batch sizes. Specifically, Samoyeds achieves significant speedups over the best baseline in all models as batch size increases. The speedup of different models is up to 1.31×, 2.23×, 1.58×, 1.09×, 1.04×, and 1.11×, compared to the best baseline, respectively. Notably, the throughput using MegaBlocks and vLLM-DS shows minimal fluctuation along with batch size increasing, in contrast, the throughput using Samoyeds method increases significantly before reaching a stable peak. The underlying reason for these observations is the improved parallelism as discussed previously in §6.1.2.

Furthermore, as illustrated in Table 3, the maximum batch size supported by Samoyeds exceeds that of other methods. Compared to Transformers, Samoyeds supports a significantly wider range of batch sizes (4.41× larger on average). Interestingly, the boost in maximum batch size of OpenMoE-34B is exceptionally higher, likely due to its unique computation process compared to other models. Notably, although approaches like MegaBlocks and vLLM-DS can accelerate model execution over Transformers, the maximum batch size supported for these approaches significantly decreases. They even fail to complete computations for Mixtral-8×22B with the batch size set to 1. This finding highlights the superior efficiency of Samoyeds in memory utilization, which in turn enhances its ability to process more batches concurrently.

## <span id="page-11-0"></span>6.4 Breakdown Analysis

In this section, we break down the performance enhancements brought by Samoyeds. *Vanilla* represents the standard Transformers framework. Then the optimizations are enabled step by step as illustrated in Figure 17.

We first enable leveraging sparsity in model weights, denoted as Samoyeds+W, by utilizing the kernel for sparsedense matrix multiplication. The introduction of weight pruning (Samoyeds+W), discussed in §3.2, results in an average speedup of  $1.27\times$  over Vanilla, peaking at  $1.54\times$ .

<span id="page-11-3"></span>![](_page_11_Figure_10.jpeg)

**Figure 17.** Breakdown Analysis on Samoyeds Optimizations. Methods are denoted by abbreviation letters. *W*: weight sparsity, *I*: input sparsity, *L*: data layout, *S*: data stationary.

Next, we eliminate the redundancy in data flow, labeled as Samoyeds+WI, by adopting the sparse-sparse matrix multiplication kernel. By eliminating the input permutation overhead, discussed in §3.1, Samoyeds+WI enhances performance by 1.39× on average compared to the Vanilla method. This configuration also outperforms Samoyeds+W in all tested models, with speedups reaching up to 1.23×. Notably, models with more experts, such as Qwen2-MoE and DeepSeek-MoE, experience a greater performance benefit due to their amplified performance loss from input permutation.

Furthermore, we evaluate the benefits of reducing transposition overhead, denoted as Samoyeds+WIT. With this graph-level optimization as previously discussed in §4.5, Samoyeds+WIT improves performance by up to  $1.08\times$  on average compared to Samoyeds+WI.

Finally, we incorporate the data stationary optimization referred to as *Samoyeds+WITS*. Overall, the increased data reuse, as discussed in §4.3, delivers an average speedup of 1.04× over the *Vanilla* approach.

#### <span id="page-11-1"></span>6.5 Accuracy Assessment

In this section, we first prune the model using the proposed Samoyeds sparse format to evaluate model accuracy. The inference solutions proposed in Samoyeds are fully decoupled from the pruning process, enabling seamless integration with SOTA pruning method such as WoodFisher[50], which is based on second-order pruning and SparseGPT[24], which operates without gradient information. In our experiments, we use the WoodFisher method provided by the SparseML framework. Notably, WoodFisher incurs significantly higher memory usage during pruning compared to other methods. Therefore, we select the most representative models within the models that are feasible under a limited memory budget, including Bert, Tiny-LLaMA and Qwen2-1.5B. Moreover, as demonstrated in prior research[24, 40, 50], maintaining accuracy during pruning is more challenging for smaller models, making them a compelling choice for evaluating

<span id="page-12-1"></span>**Table 4.** F1 Score of Bert-Series Models pruned with different Samoyeds configurations on SQuAD 1.1 (higher is better).

| Model      | Dense | (1,2,16)           | (1,2,32)     | (4,8,32) | (8,16,32) |
|------------|-------|--------------------|--------------|----------|-----------|
| Bert-base  | 89.50 | <b>88.83</b> 88.26 | 88.48        | 88.57    | 88.60     |
| Bert-large | 88.86 |                    | <b>88.66</b> | 88.25    | 88.51     |

<span id="page-12-2"></span>**Table 5.** Perplexity of Models pruned into different formats on GSM8K (lower is better).

| Model      | Dense | Unstructured | VENOM | Samoyeds |
|------------|-------|--------------|-------|----------|
| Tiny-LLaMA | 1.72  | 1.94         | 1.95  | 1.82     |
| Qwen2      | 1.92  | 1.96         | 2.26  | 2.01     |

<span id="page-12-3"></span>![](_page_12_Figure_6.jpeg)

**Figure 18.** Performance with **Figure 19.** Performance Com-Direct Porting. parison with PIT.

the effectiveness of our approach. To ensure fairness, a uniform sparsity ratio of 75% is enforced across all methods, excluding the dense baseline.

First, we analyze model accuracy across different sparse configurations. The sparse format is denoted as (N,M,V), corresponding to the structured sparse granularity configuration introduced in Section 4.1. As shown in Table 4, the accuracy of BERT models remains stable under varying sparse configurations. On the SQuAD 1.1 task, the Samoyeds sparse format retains over 99.3% of the original accuracy on average. Additionally, the accuracy of models pruned with Samoyeds sparse format is comparable to that of dense models and those pruned with unstructured methods (magnitudebased)[27, 28]. As shown in Table 5, the increase in perplexity for the GSM8K text generation tasks is only 0.06 and 0.05 for the Tiny-LLaMA-1B and Owen2-1.5B models, respectively. Notably, models pruned with the Samoyeds sparse format outperform those pruned with the SOTA structured pruning method, VENOM[12], by 56% and 73%, respectively.

## <span id="page-12-0"></span>6.6 Performance Portability Analysis

In this section, we assess the performance portability of Samoyeds across various hardware platforms with similar micro-architectures, including NVIDIA 3090, 4070 Super, 4090 and A100 40G GPUs. We directly port the kernel implementation on 4070S to other hardware and evaluate the performance using the synthetic dataset from §6.1.1, which contains 238 distinct problem sizes. As shown in Figure 18,

<span id="page-12-4"></span>**Table 6.** Performance Portability under Suggested Adaptations. The results show the percentage of the synthetic set with improved, unchanged, or degraded performance after applying the adaptation.

| Portin<br>Targe |                          | Adaptation  | Per<br>Improved | f. Impact on Ca<br>Unchanged | ases<br>Degraded |
|-----------------|--------------------------|-------------|-----------------|------------------------------|------------------|
| A100            | SM↑<br>L2 Cache↓         | Tile Size ↓ | 55.9%           | 5.5%                         | 38.6%            |
| 3090            | Slower TC<br>Bandwidth ↑ | Stage Num ↑ | 39.1%           | 49.6%                        | 11.3%            |

Samoyeds maintains 65.2% of its relative speedup over cuS-PARSELt on average, with 41.0% retained in the worst-case scenario. In contrast, VENOM loses 95% relative speedup on A100, exhibiting almost no improvement compared to cuSPARSELt. This performance discrepancy stems from two key factors: (1) While vendor libraries (e.g., cuBLAS, cuS-PARSELt) employ hardware-specific kernel configurations across different GPUs, both VENOM and Samoyeds are primarily optimized for their native development platforms. This architectural specialization inevitably diminishes their performance gains when deployed on different hardware. (2) VENOM suffers from memory-computation imbalance when porting to A100 as this GPU is equipped with higher memory bandwidth but slower tensor cores, which increases pipeline stalls during execution. However, Samoyeds mitigates this imbalance through its sparse memory access paradigm, leading to better portability when porting to A100.

Additionally, we further explore the potential adaptation rules to improve the performance given different hardware configurations. Specifically, the tiling size hyper-parameter affects the utilization of streaming multiprocessors (SMs) in parallel and the L2 cache hit rate. Meanwhile, the tensor core processing speed and memory bandwidth can affect the overlapping of the pipeline stages. As illustrated in Table 6, we propose several suggested adaptations for porting to different devices and evaluate the performance with and without these adaptations using the synthetic set described in §6.1.1. For instance, A100 GPU features more SMs but has a smaller L2 cache compared to 4070S. To fully exploit the parallelism of A100 and improve the L2 cache hit rate, it is suggested to employ smaller tiling sizes, which can lead to a performance boost in 55.9% of the tested cases.

## 6.7 Comparison with Compiler for Dynamic Sparsity

In this section, we compare the performance of Samoyeds against the SOTA compiler-based solution, PIT[59], which is specifically designed to leverage the dynamic sparse pattern that emerges in the execution of LLMs. It aggregates multiple sparse micro-tiles into dense tiles with its *Permutation Invariant Transformation*, improving overall GPU utilization. Figure 19 illustrates the normalized speedup of the MoE layer with different batch sizes and expert numbers.

Samoyeds outperforms PIT by 1.15× to 1.27×, depending on the configuration.

It should be noted that while PIT claims theoretical support leveraging the sparsity pattern along three dimensions for matrix multiplication, its practical implementation is limited to permutation along one dimension. Furthermore, PIT does not integrate the SpTC hardware into its compiler to further leverage the sparse computing capability of hardware. Consequently, PIT can only exploit the sparsity pattern that dynamically emerges in activations, which makes Samoyeds naturally outperform PIT, as demonstrated in our evaluation.

# 7 Related Work

Leveraging sparsity in LLMs. To reduce computation costs in LLMs, recent works focus on leveraging sparse computation. For unstructured sparsity, cuSPARSE[\[5\]](#page-13-1) and Sputnik[\[26\]](#page-14-7) provide GPU-accelerated linear algebra subroutines optimized for deep learning. For structured sparsity, some studies [\[14,](#page-13-9) [39,](#page-15-4) [42,](#page-15-5) [60\]](#page-15-19) focus on leveraging sparsity in model parameters, while others[\[38,](#page-15-20) [51,](#page-15-21) [59\]](#page-15-18) explore the sparsity that dynamically emerges in activations during model execution. The N:M structured sparsity has gained attention for its benefits in boosting computation efficiency while preserving model accuracy[\[40\]](#page-15-6). Libraries such as DFSS[\[15\]](#page-14-15), nmSPARSE[\[37\]](#page-14-8) and cuSPARSELt[\[6\]](#page-13-2) optimize kernels leveraging SpTC hardware with 2:4 sparsity. Besides, VENOM[\[12\]](#page-13-3) extends the capabilities of SpTC by supporting flexible sparse ratios.

MoE optimizations. Prior works on MoE optimizations fall into three categories. The first involves designing routing algorithms to enhance model accuracy, such as employing hash layers[\[46\]](#page-15-22), random routing[\[62\]](#page-15-23) or reinforcement learning[\[16\]](#page-14-20). The second focuses on enhancing parallelism mechanisms, with approaches like addressing communication overhead in Lina[\[36\]](#page-14-21), employing dynamic shadowing in FasterMoE[\[30\]](#page-14-22) and automatically discovering optimal parallel strategies in SmartMoE[\[58\]](#page-15-24). The third optimizes MoE computation directly, like the introduction of block-sparse operations in MegaBlocks[\[25\]](#page-14-9) and the design of a faster kernel for MoE layers in DeepseekMoE[\[17\]](#page-14-17).

# 8 Conclusion

This paper presents Samoyeds, a novel acceleration system for MoE LLMs with software-hardware co-optimization. We introduce a new sparse format tailored to the dual-sided sparsity inherent in MoE LLMs and implement a bespoke sparsesparse multiplication kernel leveraging SpTC to eliminate redundant computation. Additionally, systematic optimizations specifically designed for this workload and memory access pattern are applied to the MoE execution flow, further enhancing overall efficiency. Evaluation results demonstrate that Samoyeds outperforms SOTA solutions in both computation speed and memory efficiency, while also providing superior model accuracy and hardware compatibility.

# 9 Acknowledgments

We thank all the anonymous reviewers and our shepherd, Dr. Qian Ge, for their insightful and detailed suggestions. This work was funded by the National Key Research & Development Program of China (No. 2022YFB4502002), the project from Wuxi Institute of Advanced Technology, NSFC (No. 62032008), STCSM (No. 23511100100), and Shanghai Science and Technology Development Funds (22QB1404600). This work is also supported by the Embedded Common Basic Software Technology Innovation Center. The corresponding authors are Heng Shi and Jianguo Yao.

# References

- <span id="page-13-7"></span>[1] 2023. LLVM 2:4 sparsity support. <https://reviews.llvm.org/D151775>.
- <span id="page-13-6"></span>[2] 2023. Pytorch Sparse semi-structured tensors. [https://pytorch.org/](https://pytorch.org/docs/2.1/sparse.html) [docs/2.1/sparse.html](https://pytorch.org/docs/2.1/sparse.html).
- <span id="page-13-5"></span>[3] 2024. "AMD Instinct MI300" Instruction Set Architecture: Reference Guide. [https://www.amd.com/content/dam/amd/en/documents/](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) [instinct-tech-docs/instruction-set-architectures/amd-instinct](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)[mi300-cdna3-instruction-set-architecture.pdf](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf).
- <span id="page-13-12"></span>[4] 2024. cuBLAS Docs. <https://docs.nvidia.com/cuda/cublas/index.html>.
- <span id="page-13-1"></span>[5] 2024. cuSPARSE Library. [https://docs.nvidia.com/cuda/cusparse/](https://docs.nvidia.com/cuda/cusparse/index) [index](https://docs.nvidia.com/cuda/cusparse/index).
- <span id="page-13-2"></span>[6] 2024. cuSPARSELt Library. <https://docs.nvidia.com/cuda/cusparselt/>.
- <span id="page-13-13"></span>[7] 2024. Fused MoE kernel support in vLLM. [https://github.com/vllm](https://github.com/vllm-project/vllm/pull/2453)[project/vllm/pull/2453](https://github.com/vllm-project/vllm/pull/2453).
- <span id="page-13-11"></span>[8] 2024. The programming guide to using PTX (Parallel Thread Execution) and ISA (Instruction Set Architecture). [https://docs.nvidia.com/](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) [cuda/parallel-thread-execution/index.html](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).
- <span id="page-13-4"></span>[9] 2024. vLLM Docs. <https://docs.vllm.ai/en/latest/>.
- <span id="page-13-0"></span>[10] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems 33 (2020), 1877–1901.
- <span id="page-13-10"></span>[11] Shijie Cao, Chen Zhang, Zhuliang Yao, Wencong Xiao, Lanshun Nie, De-chen Zhan, Yunxin Liu, Ming Wu, and Lintao Zhang. 2019. Efficient and Effective Sparse LSTM on FPGA with Bank-Balanced Sparsity. In Proceedings of the 2019 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays, FPGA 2019, Seaside, CA, USA, February 24-26, 2019, Kia Bazargan and Stephen Neuendorffer (Eds.). ACM, 63– 72. <https://doi.org/10.1145/3289602.3293898>
- <span id="page-13-3"></span>[12] Roberto L. Castro, Andrei Ivanov, Diego Andrade, Tal Ben-Nun, Basilio B. Fraguela, and Torsten Hoefler. 2023. VENOM: A Vectorized N: M Format for Unleashing the Power of Sparse Tensor Cores. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, SC 2023, Denver, CO, USA, November 12-17, 2023, Dorian Arnold, Rosa M. Badia, and Kathryn M. Mohror (Eds.). ACM, 72:1–72:14. <https://doi.org/10.1145/3581784.3607087>
- <span id="page-13-8"></span>[13] Jou-An Chen, Wei Niu, Bin Ren, Yanzhi Wang, and Xipeng Shen. 2023. Survey: Exploiting Data Redundancy for Optimization of Deep Learning. ACM Comput. Surv. 55, 10 (2023), 212:1–212:38. <https://doi.org/10.1145/3564663>
- <span id="page-13-9"></span>[14] Zhaodong Chen, Zheng Qu, Liu Liu, Yufei Ding, and Yuan Xie. 2021. Efficient tensor core-based GPU kernels for structured sparsity under reduced precision. In International Conference for High Performance Computing, Networking, Storage and Analysis, SC 2021, St. Louis, Missouri, USA, November 14-19, 2021, Bronis R. de Supinski, Mary W. Hall, and Todd Gamblin (Eds.). ACM, 78. [https://doi.org/10.1145/3458817.](https://doi.org/10.1145/3458817.3476182) [3476182](https://doi.org/10.1145/3458817.3476182)

- <span id="page-14-15"></span>[15] Zhaodong Chen, Zheng Qu, Yuying Quan, Liu Liu, Yufei Ding, and Yuan Xie. 2023. Dynamic N: M Fine-Grained Structured Sparse Attention Mechanism. In Proceedings of the 28th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming, PPoPP 2023, Montreal, QC, Canada, 25 February 2023 - 1 March 2023, Maryam Mehri Dehnavi, Milind Kulkarni, and Sriram Krishnamoorthy (Eds.). ACM, 369–379. <https://doi.org/10.1145/3572848.3577500>
- <span id="page-14-20"></span>[16] Aidan Clark, Diego de Las Casas, Aurelia Guy, Arthur Mensch, Michela Paganini, Jordan Hoffmann, Bogdan Damoc, Blake A. Hechtman, Trevor Cai, Sebastian Borgeaud, George van den Driessche, Eliza Rutherford, Tom Hennigan, Matthew J. Johnson, Albin Cassirer, Chris Jones, Elena Buchatskaya, David Budden, Laurent Sifre, Simon Osindero, Oriol Vinyals, Marc'Aurelio Ranzato, Jack W. Rae, Erich Elsen, Koray Kavukcuoglu, and Karen Simonyan. 2022. Unified Scaling Laws for Routed Language Models. In International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA (Proceedings of Machine Learning Research, Vol. 162), Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvári, Gang Niu, and Sivan Sabato (Eds.). PMLR, 4057–4086. [https://proceedings.mlr.press/v162/](https://proceedings.mlr.press/v162/clark22a.html) [clark22a.html](https://proceedings.mlr.press/v162/clark22a.html)
- <span id="page-14-17"></span>[17] Damai Dai, Chengqi Deng, Chenggang Zhao, R. X. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y. K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang. 2024. DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models. CoRR abs/2401.06066 (2024). <https://arxiv.org/abs/2401.06066>
- <span id="page-14-13"></span>[18] Fahim Dalvi, Hassan Sajjad, Nadir Durrani, and Yonatan Belinkov. 2020. Analyzing Redundancy in Pretrained Transformer Models. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (Eds.). Association for Computational Linguistics, 4908–4926. [https://doi.org/10.18653/](https://doi.org/10.18653/V1/2020.EMNLP-MAIN.398) [V1/2020.EMNLP-MAIN.398](https://doi.org/10.18653/V1/2020.EMNLP-MAIN.398)
- <span id="page-14-10"></span>[19] Tri Dao. 2023. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. CoRR abs/2307.08691 (2023). [https:](https://doi.org/10.48550/ARXIV.2307.08691) [//doi.org/10.48550/ARXIV.2307.08691](https://doi.org/10.48550/ARXIV.2307.08691) arXiv[:2307.08691](https://arxiv.org/abs/2307.08691)
- <span id="page-14-11"></span>[20] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (Eds.). [http://papers.nips.cc/paper\\_files/paper/2022/hash/](http://papers.nips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html) [67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html)
- <span id="page-14-1"></span>[21] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 (2018).
- <span id="page-14-0"></span>[22] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020).
- <span id="page-14-3"></span>[23] William Fedus, Barret Zoph, and Noam Shazeer. 2022. Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. J. Mach. Learn. Res. 23 (2022), 120:1–120:39. [http:](http://jmlr.org/papers/v23/21-0998.html) [//jmlr.org/papers/v23/21-0998.html](http://jmlr.org/papers/v23/21-0998.html)
- <span id="page-14-14"></span>[24] Elias Frantar and Dan Alistarh. 2023. SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA (Proceedings of Machine Learning Research, Vol. 202), Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (Eds.). PMLR, 10323–10337. <https://proceedings.mlr.press/v202/frantar23a.html>

- <span id="page-14-9"></span>[25] Trevor Gale, Deepak Narayanan, Cliff Young, and Matei Zaharia. 2023. MegaBlocks: Efficient Sparse Training with Mixture-of-Experts. Proceedings of Machine Learning and Systems 5 (2023).
- <span id="page-14-7"></span>[26] Trevor Gale, Matei Zaharia, Cliff Young, and Erich Elsen. 2020. Sparse GPU kernels for deep learning. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, SC 2020, Virtual Event / Atlanta, Georgia, USA, November 9-19, 2020, Christine Cuicchi, Irene Qualters, and William T. Kramer (Eds.). IEEE/ACM, 17. <https://doi.org/10.1109/SC41405.2020.00021>
- <span id="page-14-18"></span>[27] Masafumi Hagiwara. 1994. A simple and effective method for removal of hidden units and weights. Neurocomputing 6, 2 (1994), 207–218.
- <span id="page-14-19"></span>[28] Song Han, Jeff Pool, John Tran, and William Dally. 2015. Learning both Weights and Connections for Efficient Neural Network. In Advances in Neural Information Processing Systems, C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett (Eds.), Vol. 28. Curran Associates, Inc. [https://proceedings.neurips.cc/paper\\_files/paper/2015/file/](https://proceedings.neurips.cc/paper_files/paper/2015/file/ae0eb3eed39d2bcef4622b2499a05fe6-Paper.pdf) [ae0eb3eed39d2bcef4622b2499a05fe6-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2015/file/ae0eb3eed39d2bcef4622b2499a05fe6-Paper.pdf)
- <span id="page-14-5"></span>[29] Babak Hassibi, David G. Stork, and Gregory J. Wolff. 1993. Optimal Brain Surgeon and general network pruning. In Proceedings of International Conference on Neural Networks (ICNN'88), San Francisco, CA, USA, March 28 - April 1, 1993. IEEE, 293–299. [https:](https://doi.org/10.1109/ICNN.1993.298572) [//doi.org/10.1109/ICNN.1993.298572](https://doi.org/10.1109/ICNN.1993.298572)
- <span id="page-14-22"></span>[30] Jiaao He, Jidong Zhai, Tiago Antunes, Haojie Wang, Fuwen Luo, Shangfeng Shi, and Qin Li. 2022. Fastermoe: modeling and optimizing training of large-scale dynamic pre-trained models. In Proceedings of the 27th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming. 120–134.
- <span id="page-14-12"></span>[31] Connor Holmes, Minjia Zhang, Yuxiong He, and Bo Wu. 2021. NxM-Transformer: semi-structured sparsification for natural language understanding via ADMM. Advances in neural information processing systems 34 (2021), 1818–1830.
- <span id="page-14-16"></span>[32] Changho Hwang, Wei Cui, Yifan Xiong, Ziyue Yang, Ze Liu, Han Hu, Zilong Wang, Rafael Salas, Jithin Jose, Prabhat Ram, et al. 2023. Tutel: Adaptive mixture-of-experts at scale. Proceedings of Machine Learning and Systems 5 (2023).
- <span id="page-14-4"></span>[33] Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2024. Mixtral of Experts. arXiv[:2401.04088](https://arxiv.org/abs/2401.04088) [cs.LG]
- <span id="page-14-6"></span>[34] Yann LeCun, John S. Denker, and Sara A. Solla. 1989. Optimal Brain Damage. In Advances in Neural Information Processing Systems 2, [NIPS Conference, Denver, Colorado, USA, November 27-30, 1989], David S. Touretzky (Ed.). Morgan Kaufmann, 598–605. [http://papers.nips.cc/](http://papers.nips.cc/paper/250-optimal-brain-damage) [paper/250-optimal-brain-damage](http://papers.nips.cc/paper/250-optimal-brain-damage)
- <span id="page-14-2"></span>[35] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. 2021. GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net. [https://openreview.net/forum?id=](https://openreview.net/forum?id=qrwe7XHTmYb) [qrwe7XHTmYb](https://openreview.net/forum?id=qrwe7XHTmYb)
- <span id="page-14-21"></span>[36] Jiamin Li, Yimin Jiang, Yibo Zhu, Cong Wang, and Hong Xu. 2023. Accelerating distributed {MoE} training and inference with lina. In 2023 USENIX Annual Technical Conference (USENIX ATC 23). 945–959.
- <span id="page-14-8"></span>[37] Bin Lin, Ningxin Zheng, Lei Wang, Shijie Cao, Lingxiao Ma, Quanlu Zhang, Yi Zhu, Ting Cao, Jilong Xue, Yuqing Yang, and Fan Yang. 2023. Efficient GPU Kernels for N:M-SPARSE Weights in Deep Learning. In Sixth Conference on Machine Learning and Systems (MLSys'23). [https://www.microsoft.com/en-us/research/publication/](https://www.microsoft.com/en-us/research/publication/efficient-gpu-kernels-for-nm-sparse-weights-in-deep-learning/) [efficient-gpu-kernels-for-nm-sparse-weights-in-deep-learning/](https://www.microsoft.com/en-us/research/publication/efficient-gpu-kernels-for-nm-sparse-weights-in-deep-learning/)

- <span id="page-15-20"></span>[38] Zichang Liu, Jue Wang, Tri Dao, Tianyi Zhou, Binhang Yuan, Zhao Song, Anshumali Shrivastava, Ce Zhang, Yuandong Tian, Christopher Ré, and Beidi Chen. 2023. Deja Vu: contextual sparsity for efficient LLMs at inference time. In Proceedings of the 40th International Conference on Machine Learning (Honolulu, Hawaii, USA) (ICML'23). JMLR.org, Article 919, 40 pages.
- <span id="page-15-4"></span>[39] Huizi Mao, Song Han, Jeff Pool, Wenshuo Li, Xingyu Liu, Yu Wang, and William J Dally. 2017. Exploring the granularity of sparsity in convolutional neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 13–20.
- <span id="page-15-6"></span>[40] Asit Mishra, Jorge Albericio Latorre, Jeff Pool, Darko Stosic, Dusan Stosic, Ganesh Venkatesh, Chong Yu, and Paulius Micikevicius. 2021. Accelerating Sparse Deep Neural Networks. arXiv[:2104.08378](https://arxiv.org/abs/2104.08378) [cs.LG]
- <span id="page-15-2"></span>[41] Basil Mustafa, Carlos Riquelme, Joan Puigcerver, Rodolphe Jenatton, and Neil Houlsby. 2022. Multimodal contrastive learning with limoe: the language-image mixture of experts. Advances in Neural Information Processing Systems 35 (2022), 9564–9576.
- <span id="page-15-5"></span>[42] Sharan Narang, Erich Elsen, Gregory Diamos, and Shubho Sengupta. 2017. Exploring sparsity in recurrent neural networks. arXiv preprint arXiv:1704.05119 (2017).
- <span id="page-15-12"></span>[43] Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. 2023. Efficiently scaling transformer inference. Proceedings of Machine Learning and Systems 5 (2023).
- <span id="page-15-0"></span>[44] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research 21, 140 (2020), 1–67.
- <span id="page-15-11"></span>[45] Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, and Yuxiong He. 2022. DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale. In Proceedings of the 39th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 162), Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (Eds.). PMLR, 18332–18346. [https://proceedings.mlr.press/](https://proceedings.mlr.press/v162/rajbhandari22a.html) [v162/rajbhandari22a.html](https://proceedings.mlr.press/v162/rajbhandari22a.html)
- <span id="page-15-22"></span>[46] Stephen Roller, Sainbayar Sukhbaatar, Arthur Szlam, and Jason Weston. 2021. Hash Layers For Large Sparse Models. CoRR abs/2106.04426 (2021). arXiv[:2106.04426](https://arxiv.org/abs/2106.04426) <https://arxiv.org/abs/2106.04426>
- <span id="page-15-7"></span>[47] Omar Sanseviero, Lewis Tunstall, Philipp Schmid, Sourab Mangrulkar, Younes Belkada, and Pedro Cuenca. 2023. Mixture of Experts Explained. <https://huggingface.co/blog/moe>
- <span id="page-15-10"></span>[48] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc V. Le, Geoffrey E. Hinton, and Jeff Dean. 2017. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. Open-Review.net. <https://openreview.net/forum?id=B1ckMDqlg>
- <span id="page-15-16"></span>[49] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. 2019. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism. CoRR abs/1909.08053 (2019). arXiv[:1909.08053](https://arxiv.org/abs/1909.08053) [http://arxiv.org/abs/](http://arxiv.org/abs/1909.08053) [1909.08053](http://arxiv.org/abs/1909.08053)
- <span id="page-15-17"></span>[50] Sidak Pal Singh and Dan Alistarh. 2020. WoodFisher: Efficient Second-Order Approximation for Neural Network Compression. In Advances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (Eds.), Vol. 33. Curran Associates, Inc., 18098–18109. [https://proceedings.neurips.cc/paper\\_files/paper/](https://proceedings.neurips.cc/paper_files/paper/2020/file/d1ff1ec86b62cd5f3903ff19c3a326b2-Paper.pdf) [2020/file/d1ff1ec86b62cd5f3903ff19c3a326b2-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2020/file/d1ff1ec86b62cd5f3903ff19c3a326b2-Paper.pdf)
- <span id="page-15-21"></span>[51] Chenyang Song, Xu Han, Zhengyan Zhang, Shengding Hu, Xiyu Shi, Kuai Li, Chen Chen, Zhiyuan Liu, Guangli Li, Tao Yang, and Maosong Sun. 2024. ProSparse: Introducing and Enhancing Intrinsic Activation Sparsity within Large Language Models. arXiv[:2402.13516](https://arxiv.org/abs/2402.13516) [cs.LG]

- <https://arxiv.org/abs/2402.13516>
- <span id="page-15-13"></span>[52] Wei Sun, Aojun Zhou, Sander Stuijk, Rob Wijnhoven, Andrew O Nelson, Henk Corporaal, et al. 2021. DominoSearch: Find layer-wise finegrained N: M sparse schemes from dense neural networks. Advances in neural information processing systems 34 (2021), 20721–20732.
- <span id="page-15-3"></span>[53] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805 (2023).
- <span id="page-15-1"></span>[54] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 (2023).
- <span id="page-15-14"></span>[55] Field G. Van Zee and Robert A. van de Geijn. 2015. BLIS: A Framework for Rapidly Instantiating BLAS Functionality. ACM Trans. Math. Software 41, 3 (June 2015), 14:1–14:33. <https://doi.acm.org/10.1145/2764454>
- <span id="page-15-15"></span>[56] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. 2020. Transformers: State-of-the-Art Natural Language Processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations. Association for Computational Linguistics, Online, 38–45. [https://www.aclweb.org/anthology/](https://www.aclweb.org/anthology/2020.emnlp-demos.6) [2020.emnlp-demos.6](https://www.aclweb.org/anthology/2020.emnlp-demos.6)
- <span id="page-15-8"></span>[57] Fuzhao Xue, Zian Zheng, Yao Fu, Jinjie Ni, Zangwei Zheng, Wangchunshu Zhou, and Yang You. 2024. OpenMoE: An Early Effort on Open Mixture-of-Experts Language Models. arXiv[:2402.01739](https://arxiv.org/abs/2402.01739) [cs.CL]
- <span id="page-15-24"></span>[58] Mingshu Zhai, Jiaao He, Zixuan Ma, Zan Zong, Runqing Zhang, and Jidong Zhai. 2023. {SmartMoE}: Efficiently Training {Sparsely-Activated} Models through Combining Offline and Online Parallelization. In 2023 USENIX Annual Technical Conference (USENIX ATC 23). 961–975.
- <span id="page-15-18"></span>[59] Ningxin Zheng, Huiqiang Jiang, Quanlu Zhang, Zhenhua Han, Lingxiao Ma, Yuqing Yang, Fan Yang, Chengruidong Zhang, Lili Qiu, Mao Yang, and Lidong Zhou. 2023. PIT: Optimization of Dynamic Sparse Deep Learning Models via Permutation Invariant Transformation. In Proceedings of the 29th Symposium on Operating Systems Principles (Koblenz, Germany) (SOSP '23). Association for Computing Machinery, New York, NY, USA, 331–347. <https://doi.org/10.1145/3600006.3613139>
- <span id="page-15-19"></span>[60] Ningxin Zheng, Bin Lin, Quanlu Zhang, Lingxiao Ma, Yuqing Yang, Fan Yang, Yang Wang, Mao Yang, and Lidong Zhou. 2022. SparTA: Deep-Learning Model Sparsity via Tensor-with-Sparsity-Attribute. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22). USENIX Association, Carlsbad, CA, 213–232. [https://www.](https://www.usenix.org/conference/osdi22/presentation/zheng-ningxin) [usenix.org/conference/osdi22/presentation/zheng-ningxin](https://www.usenix.org/conference/osdi22/presentation/zheng-ningxin)
- <span id="page-15-9"></span>[61] Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Y. Zhao, Andrew M. Dai, Zhifeng Chen, Quoc V. Le, and James Laudon. 2022. Mixture-of-Experts with Expert Choice Routing. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (Eds.). [http://papers.nips.cc/paper\\_files/paper/2022/hash/](http://papers.nips.cc/paper_files/paper/2022/hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html) [2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html)
- <span id="page-15-23"></span>[62] Simiao Zuo, Xiaodong Liu, Jian Jiao, Young Jin Kim, Hany Hassan, Ruofei Zhang, Jianfeng Gao, and Tuo Zhao. 2022. Taming Sparsely Activated Transformer with Stochastic Experts. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net. [https://openreview.net/forum?id=](https://openreview.net/forum?id=B72HXs80q4) [B72HXs80q4](https://openreview.net/forum?id=B72HXs80q4)

# A Artifact Appendix

## A.1 Abstract

This artifact includes the source codes and experiments for replicating the evaluations in this paper.

## A.2 Description & Requirements

- A.2.1 How to access. All the source code and instructions can be accessed through the following platforms:
  - git: <https://github.com/guqiqi/Samoyeds.git>
  - zenodo: <https://doi.org/10.5281/zenodo.14880516>
  - docker image: kevinwu2017/samoyeds:1.0.0
- A.2.2 Hardware dependencies. GPUs with Sparse Tensor Core (such as NVIDIA GPUs with Ampere architecture or newer).
- A.2.3 Software dependencies. We recommend running the Samoyeds artifact on a Linux platform with Docker and an NVIDIA GPU driver supporting CUDA 11.4+. The artifact is pre-packaged in a Docker image.
- A.2.4 Benchmarks. The Samoyeds artifact requires several models and datasets, such as Bert, SQuAD 1.1, etc. All of these requirements will be automatically downloaded during runtime.

## A.3 Set-up

- 1 docker pull kevinwu2017/samoyeds:1.0.0
- 2 docker run −it −−gpus all −−name samoyeds−ae kevinwu2017/samoyeds:1.0.0

# A.4 Evaluation workflow

# A.4.1 Major Claims.

- Claim (C1): Samoyeds achieves an average speedup of 1.99× over baselines, as shown in Figure 12 and 13. This is proven by experiment (E1).
- Claim (C2): Samoyeds outperforms the baseline by 1.45× on average, as shown in Figure 14. This is validated by experiment (E2).
- Claim (C3): Samoyeds 1.42× improves overall model performance by 1.42× (Figure 15) and delivers superior throughput across different batch sizes (Figure 16). This is confirmed by experiment (E3).
- Claim (C4): Different optimizations of Samoyeds provide speedup according to our breakdown analysis (Figure 17). This can be reproduced with experiment (E4).
- Claim (C5): The optimization of Samoyeds does not affect the model accuracy, as shown in Table 4 and 5. This is verified by experiment (E5).
- Claim (C6): Samoyeds exhibits superior portability compared to baselines, as shown in Figure 18. This is proven by experiment (E6).

- A.4.2 Experiments. The hardware requirements for each experiment are as follows:
  - E1, E2, E3, and E6: These experiments can be conducted on a single GPU, such as the NVIDIA GeForce RTX 4070 Super used in our paper.
  - E4: This experiment involves post-training of models, which may require high-end GPUs such as the A100- 80G used in our paper.
  - E5: This experiment analyzes performance portability and requires multiple GPUs with different architectures (e.g., RTX 3090, RTX 4070 Super, RTX 4090, and A100, as used in our paper).
  - Experiment (E1): To reproduce the kernel level results (Figure 12, 13), execute:
  - 1 ./artifacts/kernel/synthetic\_scripts.sh
  - 2 ./artifacts/kernel/kernel\_model\_config\_scripts.sh

Figure 12 and 13 can be plotted with following files:

- 1 ./artifacts/kernel/figure12\_plot.ipynb
- 2 ./artifacts/kernel/figure13\_plot.ipynb
- Experiment (E2): To reproduce the MoE module level results (Figure 14), execute:
- 1 ./artifacts/MoE/figure14\_scripts.sh

Figure 14 can be plotted with following files:

- 1 ./artifacts/MoE/figure14\_plot.ipynb
- Experiment (E3): To reproduce the end-to-end level results (Figure 15, 16), execute:
- 1 ./artifacts/model/figure15\_scripts.sh
- 2 ./artifacts/model/figure16\_scripts.sh

Figure 15 and 16 can be plotted with following files:

- 1 ./artifacts/model/figure15\_plot.ipynb
- 2 ./artifacts/model/figure16\_plot.ipynb
- Experiment (E4): To reproduce the breakdown analysis results (Figure 17), execute:
- 1 ./artifacts/MoE/figure17\_scripts.sh

Figure 17 can be plotted with following files:

- 1 ./artifacts/MoE/figure17\_plot.ipynb
- Experiment (E5): We provide several scripts to reproduce the results of model accuracy (Table 4, 5). The following scripts require execution on high-memory GPUs or multi-GPU configurations. Specifically: (1) The script for collecting data in Table 4 is configured to utilize a cluster of 4 GPUs; (2) The scripts for collecting data in Table 5 must be run on an NVIDIA A100 80GB GPU to avoid Out-Of-Memory (OOM) errors. Lower-capacity GPUs may not have sufficient memory to handle these operations.
- 1 cd sparseml
- 2 # Table 4
- 3 bach benchmark/scripts/samoyeds\_gradual\_pair.sh
- 4 # Table 5
- 5 bash benchmark/scripts/samoyeds\_qwen2\_80G.sh
- 6 bash benchmark/scripts/samoyeds\_tiny\_llama\_80G.sh

- <span id="page-17-0"></span>The results are stored in the ./benchmark/output\_dir/ directory.
- Experiment (E6): To reproduce the performance portability results of Samoyeds (Figure 18), the following script need to run on multiple GPUs, including NVIDIA GeForce RTX 3070, NVIDIA GeForce RTX 4070 Super, NVIDIA GeForce RTX 4090, and NVIDIA A100.
- 1 ./artifacts/kernel/synthetic\_scripts.sh
  - Figure 18 can be reproduced by collecting results on different GPUs into ./artifacts/results/kernel/ folder. Figure 18 then can be plotted with following files:
- 1 ./artifacts/MoE/figure18\_plot.ipynb