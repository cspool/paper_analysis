2025 IEEE International Symposium on High Performance Computer Architecture (HPCA) 

# VQ-LLM: High-performance Code Generation for Vector Quantization Augmented LLM Inference 

Zihan Liu[1] _[,]_[2] , Xinhao Luo[1] _[,]_[2] , Junxian Guo[1] , Wentao Ni[1] , Yangjie Zhou[1] , Yue Guan[1] _[,]_[2] , Cong Guo[3] , Weihao Cui[1] _[,]_[4] , Yu Feng[1] , Minyi Guo[1] _[,]_[2] , Yuhao Zhu[5] , Minjia Zhang[6] , Chen Jin[7] , Jingwen Leng[1] _[,]_[2] _[,][∗]_ 

> 1 _Shanghai Jiao Tong University,_ 2 _Shanghai Qi Zhi Institute,_ 3 _Duke University,_ 4 _National University of Singapore_ 

> 5 _University of Rochester,_ 6 _University of Illinois Urbana-Champaign,_ 7 _Magik Compute_ 

_{_ altair.liu, lxh666, guojunxian, wennitao, yj zhou, bonboru, weihao, y-feng _}_ @sjtu.edu.cn, cong.guo@duke.edu guo-my@cs.sjtu.edu.cn, yzhu@rochester.edu, minjiaz@illinois.edu, chenj@magikcompute.ai, leng-jw@cs.sjtu.edu.cn 

_**Abstract**_ **—Vector quantization (VQ), which treats a vector as a compression unit, gains increasing research interests for its potential to accelerate large language models (LLMs). Compared to conventional element-wise quantization methods, VQ algorithms can compress weight and KV cache tensors in LLMs with a greater ratio while maintaining the high model accuracy. However, translating a VQ algorithm’s memory reduction into the actual latency improvement is challenging. We profile and analyze the current approach of integrating VQ into computation kernels and show that its major inefficiency lies in the poor access efficiency of codebooks in VQ algorithms and uncoordinated computation dataflow. Meanwhile, the diversity of VQ algorithms (e.g., different vector sizes and entry counts) and LLMs’ computation kernels (e.g matrix-matrix/vector multiplication and attention computation) makes it impractical to manually craft efficient kernel implementations for each specific case.** 

**In this work, we design and implement VQ-LLM, an efficient fused VQ kernel generation framework. We first introduce a software abstraction called codebook cache to optimize codebook access efficiency and support the integration of VQ with various computations. The codebook cache adaptively stores different entries across the GPU’s memory hierarchy, including off-chip global memory, on-chip shared memory, and registers. Centered around the codebook cache, we design an efficient computation engine that optimizes memory traffic during computations involving codebooks. This compute engine adopts the codebookcentric dataflow and fusion optimizations. Additionally, we provide adaptive heuristics to tailor parameter selection in our optimizations to diverse VQ configurations. Our optimizations achieve the latency reduction of 64.36% to 99.1% compared to existing open-source implementations. A final comparison with state-of-the-art element-wise quantization methods like AWQ and QoQ shows that our VQ-LLM is practically viable, achieving latencies close or even better latencies to those at equivalent bitwidths, potentially offering greater accuracy.** 

## I. INTRODUCTION 

With the great success of large language models (LLMs), neural networks are placing significant pressure on current hardware, especially memory systems [21], [27], [28], [33], [74], [75]. Quantization techniques become essential for deploying these large models [15], [18]–[20], [24], [30], [51], [62]. Quantization reduces the original IEEE-754 half format FP16 data to types with much narrower bit-widths, such as FP8 and INT4, decreasing the memory footprint significantly [1]. Researchers have developed numerous novel data 

_*Jingwen Leng is the corresponding author of this paper._ 

formats and algorithms, like MXFP and ANT, with varying scaling granularities to represent the original data using fewer bits [20], [49]. However, these techniques treat each data point as an independent element for compression, overlooking the potential information between elements. As a result, these methods typically reach a 4-bit limit; compressing to 2 bits or less leads to a substantial accuracy loss [12], [15], [56]. 

Under these scenarios, vector quantization (VQ) emerges as a pivotal technique to further reduce LLMs’ memory footprints [12], [56], [57], [67], [69]. The VQ methods compress a vector of multiple elements into a single element and enabling the capture of information across elements [57], [69]. Typically, this cross-element information is gathered through clustering, which involves applying a clustering algorithm to all vectors and using cluster centroids to represent nearby vectors [26], [37]. Furthermore, some researchers suggest iteratively processing the residuals between the original and quantized data to enhance reconstruction quality [32], [63]. For LLMs, VQ achieves higher accuracy at the same 4-bit level or maintains equivalent accuracy at 2-bits, and some approaches can compress the KV cache in LLMs to 1-bit [69]. 

Despite their appealing accuracy and compression ratios, VQ-augmented LLMs do not significantly enhance the model’s latency performance in practice. Our analysis in Sec. III indicates that existing VQ methods have substantially higher latency than conventional element-wise quantization methods, often performing worse than the original FP16 version. The inefficiencies primarily stem from how memory access and computation dataflow are managed when interacting with the codebooks in VQ methods. We have identified three key challenges that must be addressed to generate high-performance kernels integrating VQ with subsequent computations. 

The first challenge lies in the placement of VQ’s codebooks. We find that the common practice of storing all codebook entries in GPU shared memory increases shared memory usage, thereby reducing the number of thread blocks that can concurrently operate on each SM, which diminishes performance. Additionally, the number of codebook entries far exceeds the number of available shared memory banks, leading to significant bank conflicts. The second challenge involves coordinating the loading of codebooks and subsequent computation. There is excessive traffic in loading the codebook 

979-8-3315-0647-6/25/$31.00 ©2025 IEEE DOI 10.1109/HPCA61900.2025.00112 

1496 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:14:40 UTC from IEEE Xplore.  Restrictions apply. 

from global memory to shared memory, and in transferring codebook entries from shared memory to registers, which should be much lower in theory. The reasons include multiple thread blocks loading duplicate codebooks, and the requirement for threads to store data reconstructed via codebook entries (we refer to them as dequantized data throughout the paper) back to shared memory in a layout that differs from their dequantization for subsequent computations. The last challenge is that the diversity of VQ algorithms (e.g., different vector sizes and entry counts) and LLMs’ computation kernels (e.g matrix-matrix/vector multiplication and attention computation) makes it impractical to manually craft efficient kernel implementations for each specific case. 

To address the challenges, this work develops VQ-LLM, an automatic high-performance fused VQ kernel code generation framework. We begin by introducing a software abstraction called **codebook cache** , designed to optimize codebook access efficiency and facilitate the integration of VQ with various computations. This cache enables efficient codebook placement across the GPU’s memory hierarchy. We have identified that only a select few entries are accessed frequently. Therefore, rather than indiscriminately caching all entries in shared memory, we adopt a hierarchical approach: entries with low access frequency remain in global memory, while those accessed more frequently are cached in shared memory. To address inevitable bank conflicts, entries that are accessed extremely frequently are stored in thread-local registers, eliminating bank conflict issues. Furthermore, to mitigate negative impacts on computation (reduced concurrency), we utilize available slacks, which ensues no drop in resource utilization, to adaptively determine the optimal placement of entries. 

Centering the codebook cache, we design an efficient compute engine that optimizes memory traffic when computing with codebooks, and it consists of two novel techniques. The first called codebook-centric dataflow divides and parallelizes the original computation task in a way that minimizes the codebook switch overhead. It may split the reduction dimension of the original computation task, for which we adaptively determine the split factor to balance the global reduction. The second technique, codebook-centric hierarchical fusion, extends the default shared memory level fusion to support the additional register-level fusion. This mechanism leverages a GPU feature known as intra-warp data exchange [42] to rearrange the dequantized data into the required layout for subsequent computations directly in registers. We adaptively decide where to conduct the fusion based on profiled exchanging overhead and difference between layout of dequantized data and layout required by subsequent computation. 

Our evaluation shows that VQ-LLM achieves the latency reduction of 64.36% to 99.1% compared to compared to existing open-source implementations [13], [56]. We also perform extensive sensitivity to verify the effectiveness of each technique in our framework. A final comparison with stateof-the-art element-wise quantization methods like AWQ [30] and QoQ [31] shows that our VQ-LLM is practically viable, achieving latencies close or even better latencies to those at 

**==> picture [227 x 189] intentionally omitted <==**

**----- Start of picture text -----**<br>
Vector size<br>Residuals Quantization<br>2 0 2 3 1 1 0 3<br>Dequantization<br>+ + + +<br>Concat<br>y<br>#Entr<br>Codebooks<br>Codebooks<br>Entries<br>= = = =<br>**----- End of picture text -----**<br>


Fig. 1. Typical vector quantization pipeline. 

- equivalent bit-widths, potentially offering greater accuracy. We list our main contributions as follow: 

- To the best of our knowledge, we are the first to deeply dive into performance issues of vector quantization and make it practically feasible in LLM inference. 

- We deliver a detailed analysis and identify these issues are caused by inefficient codebook entries access and uncoordinated codebook loading and subsequent computation. 

- Based on the finding, we propose VQ-LLM to generate efficient fused VQ kernel implementation, it consists of codebook cache and codebook based compute engine, with configurable parameters and adaptive heuristics. 

- We compare VQ-LLM with open-sources implementations and element-wise quantization works, with detail speed-up breakdown analysis on proposed optimizations. 

## II. BACKGROUND AND RELATED WORKS 

This section first introduces the basic concept of vector quantization and its applications in quantizing large language models. It then provides a detailed analysis of serving vector quantized large language models with existing solutions. 

## _A. Vector Quantization (VQ)_ 

Compared to traditional quantization, vector quantization (VQ) treats the vector of multiple elements as a unit and uses trained quantization points organized into codebooks to quantize the vector into a single element, rather than in an element-wise manner as in traditional quantization. This technique is widely used in vector database, nearest neighbor search, etc. [29], [34] VQ has several configurable parameters, highlighted in Fig. 1, which allow it to be specified for product quantization (PQ), additive quantization (AQ), and hybrid quantization (PRQ) [4], [11], [17], [25]. Apart from 

1497 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:14:40 UTC from IEEE Xplore.  Restrictions apply. 

these, there are other techniques such as hash-based [23] and lattice-based methods [2]. However, these techniques either cannot reconstruct the original data or need to be used in conjunction with PQ, AQ, and PRQ. Therefore, we do not delve into these techniques as they do not influence the core findings and insights of this work. 

_**Typical VQ Pipeline.**_ We use the example in Fig. 1 to demonstrate the typical VQ pipeline, and numbers in _**(** ·_ _**)**_ represent the value of parameters in this example. We also summarize the VQ parameters in Tbl. I. First, the original 16dimensional vectors are split into four sets of _**vector size (4)**_ - dimensional sub-vectors. Next, we collect sub-vectors in one sub-space (or several sub-spaces, depending on algorithms) and conduct k-means clustering to group these sub-vectors into _**#Entry (4)**_ clusters. The original sub-vectors are then replaced with the index of their closest cluster centroids, using _**log**_ 2 _**#Entry (2)**_ bits. Next, we collect the differences between the original sub-vectors and their closest cluster centroids as the residuals. We then perform another round of k-means clustering and replace the residual sub-vectors with the index of the closest centroids of the new clusters. This process of residual quantization can be repeated, as determined by the _**Residual (2)**_ parameter. The quantization process is now complete, as shown in the upper part of Fig. 1. We then gather all the aforementioned cluster centroids and organize them into codebooks. In the following sections, we refer to these centroids as codebook entries. 

To reconstruct the original data, a dequantization process is required, as shown in the lower part of Fig. 1. For each residual, we use its quantized data to look up the corresponding codebooks and find the codebook entry indexed by the quantized data in each sub-space. We then gather the results from the same sub-spaces across different residuals, typically via element-wise accumulation. Finally, we concatenate the results from all sub-spaces. Throughout the entire process, _**vector size**_ , _**#Entry**_ , and _**Residual**_ are configurable. These configurations are annotated with **x,y,z** , in the format of **VQ<x,y,z>** . In this example, the configuration is **VQ<4,2,2>** . 

## _B. Large Language Models (LLMs)_ 

LLMs adopt the Transformer architecture [58], which is pivotal in processing and generating natural language in sequences of tokens. The core of the Transformer architecture is multi-head attention (MHA), designed to run several parallel attention processes, allowing the model to simultanesly focus on different types of information from a single input sequence. 

TABLE I 

PARAMETERS OF VQ ALGORITHMS 

|Item|Description|Value in<br>Sec. III|
|---|---|---|
|Vector size<br>#Entry<br>Residual|Number of elements to quantize at once<br>Number of quantization points (entries)<br>Number of times to quantize residual data|4<br>28<br>1|



**==> picture [214 x 139] intentionally omitted <==**

**----- Start of picture text -----**<br>
Element-wise VQ Element-wise VQ<br>6 2-bit 3-bit 4-bit 0.5 KVQuant 4-bit<br>4 AWQ QoQ<br>GPT-Q CQ<br>2 AWQ KVQuant<br>0 GPT-VQ 0<br>AQLM QuiP# 2-bit CQ<br>Quantization points Quantization points<br>2 2<br>Outlier<br>0 0<br>MSE=5.2e-3 MSE=3.2e-3<br>−2 −2<br>−2 0 2 −2 0 2<br>ΔPPL ΔPPL<br>**----- End of picture text -----**<br>


Fig. 2. (Upper) Accuracy of VQ and element-wise quantization, left is weight and right is KV cache quantization. (Lower) VQ (right) can better capture the distribution of data than element-wise quantization (left), with inter-dimensions information. 

Each head in MHA can be thought of as an independent attention layer with its own learnable parameters. Outputs of these heads are then concatenated and fed to subsequent operations. Mathematically, MHA can be described as follows: 

MultiHead( _Q, K, V_ ) = Concat(head1 _, . . . ,_ head _h_ ) _W[O] ,_ head _i_ = Attention( _Q_ = _HWi[Q][, K]_[=] _[ HW][ K] i[, V]_[=] _[ HW][ V] i_[)] _[,]_ Attention( _Q, K, V_ ) = softmax( _QK[T] /_ ~~�~~ _dk_ ) _V._ 

Here, _Wi[Q]_[,] _[W][ K] i_[,] _[W][ V] i_[,][and] _[W][ O]_[are][parameter][matrices][for] the _i_ -th head and the output projection, respectively. And _H_ is the hidden state. The _softmax_ function is applied over the keys to normalize their weights, ensuring that the output is a weighted sum of the values based on the input’s relevance. 

In the context of text generation, LLMs often first implement a prefill stage where the model processes existing tokens before generating new ones. This sets the initial state of the model’s memory and attention mechanisms, making the generation process more context-aware. Following this, the decode phase begins, during which the model generates one token at a time, updating its internal state based on both the newly generated token and the preceding context. To efficiently reuse previously computed token representations during the decode phase, a Key-Value (KV) cache mechanism is often utilized [46], [68], enhancing inference performance. 

## _C. VQ for LLM Acceleration_ 

VQ gains increasing interests for its great potential for compressing and accelerating LLMs. This is because LLMs are highly memory-bound [61], with many researchers identifying weights and KV-cache as the main bottlenecks, accounting for over 95% of the memory footprint [28]. To further compress the weights and KV-cache and reduce memory usage, VQ has come to the center of the stage with its superior compression ratio and reconstruction quality. Various newly proposed VQbased compression algorithms outperform SOTA elementwise quantization baselines in both weight-only compression (AWQ [30]) and KV-cache compression (KVQuant [24], QoQ [31]) under the same equivalent bitwidth [12], [56], 

1498 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:14:40 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [233 x 60] intentionally omitted <==**

**----- Start of picture text -----**<br>
FP16 KV Cache VQ attn KV Cache Quantized VQ attn KV Cache Quantized<br>attn GC SC<br>Codebook Codebook<br>@  Global Dequantize @  Shared Dequantize<br>Dequantized @  Shared Dequantized @  Shared<br>Q,K,V Attention Q,K,V Attention Q,K,V Attention<br>Output Output Output<br>**----- End of picture text -----**<br>


Fig. 3. Workflow of investigated VQ kernels. 

[57], [67], [69], as shown in the upper part of Fig. 2. Some can even achieve higher quality with fewer equivalent bits. The underlying reason is depicted in the lower part of Fig. 2. With cross-dimension information, VQ can better capture the distribution characteristics of the data, resulting in lower reconstruction error. In contrast, traditional quantization relies on the Cartesian product of quantization points between dimensions and cannot represent some outliers well. 

While converting the reduced memory footprint to actual speed-up is challenging due to the need for efficient kernels that take quantized data and codebooks as inputs, dequantize them, and perform computations. Unfortunately, existing algorithms only provide kernels with high latency, making them impractical for use [12], [56], as verified in Sec. VII. In the VQ pipeline, dequantization is the main bottleneck in the context of LLMs. This is because quantization can be done offline (for weights) or asynchronously with tiny overhead (for KV cache, also discussed in Sec. VII). However, dequantization is required every time before a computation since the quantized data store codebook indices and cannot be directly operated on. Therefore, this paper focuses on developing efficient fused dequantization-computation kernels. 

In the next section, we will analyze the inefficiencies of existing and vanilla optimized fused dequantization-computation kernel. As mentioned before, the core difference between VQ and element-wise quantization is the use of vectorized codebooks, and we primiaily focus on them in our analysis. 

Noted that we target NVIDIA GPUs in this paper, althouth GPUs from other vendors share similar concepts [3], [39], [40], [54]. A GPU compute kernel launches thousands of threads, organized into thread blocks within a grid. Each thread block is dispatched to a Streaming Multiprocessor (SM), which may handle multiple thread blocks to overlap instructions [70]. Threads access three memory hierachies: registers (local to each thread), shared memory (local to the thread block), and global memory (accessible by all threads). 

## III. MOTIVATION 

In this section, we analyze the inefficiencies of current VQ implementation centering how codebooks are placed and utilized. We first outline our setup for a micro-benchmarkbased investigation in Fig. 3 and then analyze it in detail. 

## _A. Investigation Setup_ 

We evaluate an attention kernel from Llama-7B [55] with 32 heads and head dimension of 128 on an RTX 4090 GPU [44]. We investigate three implementations of vector quantized (VQ) KV cache with the configuration **VQ<4,8,1>** that follows 

**==> picture [232 x 52] intentionally omitted <==**

**----- Start of picture text -----**<br>
FP16-attn 5<br>2 4 FP16-attn 2.52×10 [6]<br>3<br>1 2<br>1<br>0 0<br>SM Shared Shared Global-> Shared-><br>Utilization Usage Bank Shared Reg<br>Conflict Traffic Traffic<br>VQ-attnGCVQ-attnSC<br>Ratio<br>Relative Relative<br>performance<br>**----- End of picture text -----**<br>


Fig. 4. (left) Latency of **VQ-attn-GC** and **VQ-attn-SC** relative to **FP16-attn** . (right) Relative performance counters of **VQ-attn-SC** . 

CQ-2 [69]. As illustrated in Fig. 3, the first **FP16-attn** version implements Flash Decoding [10] from the FlashAttention library [7], [9]. We implement the **VQ-attn-GC** version ourselves following the original paper [12], [56], [57], [69] due to the lack of open-source kernels. **VQ-attn-GC** receives the VQ quantized KV cache and its codebooks, dequantizes them to FP16 precision, and performs the subsequent attention computation, with codebooks stored in _global memory_ . Given the long access latency of global memory, we propose and implement another optimized version that stores codebooks in _shared memory_ and hence is labelled as **VQ-attn-SC** , with the rest of the process mirroring that of **VQ-attn-GC** . Here we only analyze attention kernel thus KV cache compression, while these observations can also be generalized to GeMM/GeMV and weight compression. 

## _B. Inefficiency Analysis_ 

Since the attention (decoding) process is highly memorybound, using **VQ<4,8,1>** , which compresses the KV cache to 1/8, should significantly enhance its performance. However, as depicted on the left of Fig. 4, both VQ versions underperform the FP16 baseline. We also observe that the shared-memory-based codebook version, **VQ-attn-SC** , outperforms the global-memory-based version, **VQ-attn-GC** , demonstrating the effectiveness of utilizing shared memory for codebooks. Although shared memory and the GPU L1 cache share the same physical space, the hardware-managed L1 cache fails to capture the temporal locality of codebook entries. This is because the size and irregular access pattern of the entries does not align with the cache line size and prefetch width (128 bytes [41]) of the L1 cache. According to our profiling results, **VQ-attn-GC** achieves only a 12.45% L1 cache hit rate, indicating significant wasted capacity in the L1 cache. Consequently, we default to the **VQ-attn-SC** version to investigate its sources of inefficiencies. 

_**Inefficient Codebook Access.**_ Fig. 4 (right) compares the various performance counters of the **VQ-attn-SC** version and the FP16 version. We first observe an over 30% drop in compute (SM) utilization in the **VQ-attn-SC** version (1 _[st]_ bar). This decline is attributed to the VQ’s significantly increasing shared memory footprint (2 _[nd]_ bar), which reduces the number of thread blocks that can run concurrently on each SM, leading to decreased performance. Additionally, we note high bank conflicts (3 _[rd]_ bar), indicative of highly serialized access to shared memory. Eliminating these bank conflicts is challenging for several reasons. First, the number of codebook entries vastly exceeds the number of shared memory 

1499 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:14:40 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [214 x 102] intentionally omitted <==**

**----- Start of picture text -----**<br>
Channel Channel<br>FP16-attn<br>Query Output<br>Inner product Global reduce<br>Block 0 Block 0<br>Block 1 Block 1<br>K Cache V Cache<br>Block 2 Block 2<br>Block 3 Block 3<br>K Codebook VQ-attn-SC V Codebook<br>Tokens Tokens<br>Dequantize Dequantize<br>Global reduce<br>**----- End of picture text -----**<br>


Fig. 5. Dataflow of **FP16-attn** (inner box) and **VQ-attn-SC** (outer box). 

banks, e.g., 256 entries versus 32 banks, and their accesses are random during the VQ dequantization process, precluding the use of common static reordering or padding solutions for coalesced accesses [41]. It is possible to reorder entries or threads at runtime, which can introduce extra complexity and overhead. Second, a single codebook entry can occupy multiple banks in VQ, exacerbating the difficulty of mitigating bank 

**Takeaway 1** Storing codebooks in fast on-chip buffers like shared memory is necessary, but not trivial. 

_**Uncoordinated Codebook Load and Compute.**_ The 4 _[th]_ bar in Fig. 4 (right) indicates that the traffic from off-chip global to on-chip shared memory is higher for the VQ version than for the FP16 version. This is counterintuitive since VQ is expected to significantly reduce global memory access. The cause of this unexpected off-chip traffic is that integrating VQ into the original compute kernel results in uncoordinated and duplicated loads of codebooks. 

The inner box of Fig. 5 shows the original FlashDecoding’s dataflow [10], which parallelizes the computation of different tokens and computes the local softmax in global memory. When integrating the VQ codebooks to this computation dataflow, computing every four channels for a token needs to switch to a different codebook, following the VQ algorithm of CQ-2 [69]. Consequently, thread blocks handling different tokens end up accessing and loading identical codebooks as they process data across all channels, as shown in the outer box of Fig. 5. This results in significant duplicated off-chip memory traffic, and this challenge is also presented in the integration of VQ with GeMM kernels. For GPTVQ-2 [57], 

TABLE II 

VQ ALGORITHM AND THEIR CONFIGURATIONS 

|Algorithm|Compression<br>Ratio against FP16|Vector<br>Size|#Entry|Residual|
|---|---|---|---|---|
|**QuiP#-4**<br>**AQLM-3**<br>**GPTVQ-2**<br>**CQ-4**<br>**CQ-2**|25%<br>18.75%<br>12.5%<br>25%<br>12.5%|8<br>8<br>4<br>2<br>4|65536*<br>4096<br>256<br>256<br>256|2<br>2<br>1<br>1<br>1|
|**Confgs.**||2_1,2,..._|2_1,2,..._|1,2,...|



_*QuiP# utilize a lattice-based codebook, though it has 65536 entries, it only need to look up from 256 of them every dequantization with bit operations._ 

**==> picture [178 x 43] intentionally omitted <==**

**----- Start of picture text -----**<br>
Tid 0 Dequant<br>Tid 1 Dequant<br>Tid 0 Compute<br>Tid 1 Compute K cache V cache<br>**----- End of picture text -----**<br>


Fig. 6. Layout of dequantized data and required layout of following computation of KV cache in attention (decoding). 

every (256, 256) tile of the weight matrix shares a codebook, while the task is spliced into ( _·_ , 128) tiles on weight matrix, and every two thread blocks access and load a same codebook. 

Besides the increased off-chip global memory traffic, we also observe a significant rise in on-chip shared memory to register traffic in the **VQ-attn-SC** version, as shown in the last bar of Fig. 4 (right). Ideally, this traffic should remain the same to the **FP16-attn** version, given that the computation precision and the volume of data involved in the computation remain unchanged. The unusual Shared _→_ Reg traffic stems from a mismatch between the layout of dequantized data and the layout required by the computation. 

As illustrated in Fig. 6, one thread dequantizes a row of four elements at a time for the KV cache following the CQ-2 algorithm [69]. It then stores these four elements in thread-local registers. However, the computation requires a column-wise weighted accumulation on the V cache, and the four dequantized elements by the thread do not match the data needed for subsequent computations. Consequently, the dequantized data in local registers must be stored back into shared memory, allowing the correct threads to access them. Notice that as depicted in the figure, the K cache does not introduce such a shared memory round-trip since its row-wise accumulation process aligns with the dequantization process. 

**Takeaway 2** Integrating and fusing VQ algorithms into LLM’s kernels requires a careful coordination between the codebook load and the fused kernel’s compute dataflow. 

## _C. Additional Complexity of VQ Diversity_ 

Our above analysis primarily focuses on a specific VQ configuration for the FlashDecoding kernel. In fact, we have surveyed state-of-the-art methods of using VQs to accelerate LLMs and found considerable diversity, as listed in Tbl. II. These varied configurations add complexity when generating high-performance fused computation kernels. Moreover, different algorithms choose to train a codebook with different parts of tensor which further push up this complexity. For instance, QuiP# [56] can avoid duplicated Global _→_ Shared traffic as it train one codebook with the entire weight tensor, yet it may increase bank conflicts and cause layout mismatches with its vector size 8. Conversely, CQ-4 [69] is able to reduce bank conflicts and layout issues with its vector size 2, but it may lead to significantly duplicated Global _→_ Shared traffic since it train different codebooks with different channels. 

On the other hand, there are various computations associated with VQ algorithms, such as **VQ-gemm** and **VQ-gemv** for weight-only quantization, and **VQ-attn** for KV cache quantization, as previously mentioned. The combination of 

1500 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:14:40 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [190 x 101] intentionally omitted <==**

**----- Start of picture text -----**<br>
Codebook-Based Compute Engine<br>6.1 Codebook Centric Dataflow<br>5 Codebook Cache<br>Global Shared Register<br>Cold Medium Hot<br>Codebook Reorder & Update<br>6.2 Codebook Centric Hierachical Fusion<br>Adaptive Heuristics<br>**----- End of picture text -----**<br>


Fig. 7. VQ-LLM design overview. 

VQ algorithm diversity and multiple subsequent computation patterns makes it impractical to manually craft efficient kernel implementations for each specific case. 

**Takeaway 3** An adaptive solution is necessary to achieve optimal performance across a variety of VQ algorithms and their subsequent computations. 

**==> picture [211 x 46] intentionally omitted <==**

**----- Start of picture text -----**<br>
48<br>μ+3σ<br>32<br>16 μ+0σ<br>26 1911<br>0<br>0 1024 2048 3072 4096<br>Codebook entry ID<br>Access frequency<br>**----- End of picture text -----**<br>


Fig. 8. Codebook entries access frequency of one thread block in VQ-GeMM kernel, with **VQ<8,12,2>** (AQLM-3). 

overhead and difference between layout of dequantized data and layout required by subsequent computation. 

Our VQ-LLM framework comprises a set of CUDA templates that employ a codebook-centric dataflow and fusion scheme, along with a set of adaptive heuristics. These templates include both algorithm-specific and hardware-related parameters. To generate a specific VQ-augmented compute kernel, we supply the configuration of the algorithm and target GPU to the corresponding compute kernel template. VQ-LLM then automatically selects the optimal parameters based on the template specifications and heuristics. 

## IV. VQ-LLM OVERVIEW 

From the analysis in the previous section, we identify three key challenges in utilizing VQ to accelerate LLM inference: i) efficient codebook entry access, ii) coordinated codebook loading and subsequent computation, and iii) significant diversity in VQ algorithms and subsequent computation patterns. 

To address these challenges, we design and implement VQLLM, an automatic high-performance code generation framework in Fig. 7. We introduce a software abstraction called codebook cache to optimize codebook access efficiency and support the integration of VQ with various computations. The codebook cache **adaptively** stores different entries across the GPU’s memory hierarchy, including off-chip global memory, on-chip shared memory, and registers. It does so by leveraging the offline-profiled characteristics of codebook entry access, such as cold, medium, and hot. 

The codebook cache also enables seamless integration with the subsequent computations. Centered around the codebook cache, we design an efficient computation engine that optimizes memory traffic during computations involving codebooks, incorporating two core techniques. The first technique, called codebook-centric dataflow, divides and parallelizes the original computation task in a way that minimizes the codebook switch overhead. It may split the reduction dimension of the original computation task, for which we **adaptively** determine the split factor to balance the global reduction. With this dataflow, we eliminate the excessive off-chip memory traffic caused by redundant codebook loads from different thread blocks in current VQ implementations. 

The second technique employed by our compute engine, named codebook-centric hierarchical fusion, extends the default shared memory level fusion to support the additional register-level fusion. This mechanism leverages a GPU feature known as intra-warp data exchange [42] to rearrange the dequantized data into the required layout for subsequent computations directly in registers. And we **adaptively** decide where to conduct the fusion based on profiled exchanging 

## V. CODEBOOK CACHE 

We first present the design intuition of codebook cache, and then implementation details. Finally, we describe the user interface that can be utilized by subsequent computations. 

## _A. Design Rationale_ 

As Sec. III explains, naively placing the entire codebook in the shared memory results in suboptimal performance due to two issues: i) increased shared memory usage and ii) significant bank conflicts. To address these issues, we propose storing different entries at various memory levels based on their access frequencies. Specifically, we can store rarely accessed entries in off-chip global memory to conserve shared memory usage, and store the most frequently used entries in the thread local registers to eliminate bank conflicts. 

We find that different entries in a codebook indeed demonstrate varying levels of ‘hotness’ in terms of access frequency. Fig. 8 illustrates such an example of AQLM-3, and results of other algorithms will be shown in Sec. VII. Over half of the codebook entries are accessed less frequently than the average, indicating that placing them in shared memory yields little benefit. There are 26 hot entries that are accessed more frequently than _µ_ +3 _σ_ (mean plus three standard deviations), suggesting that they are more susceptible to inevitable bank conflicts. This observation forms the foundation of our codebook cache design, the details of which we introduce next. 

## _B. Implementation_ 

Typically, the implementation of a cache relies on tag array [59] or lookup table [36], which could incur additional latency and storage overhead. In our codebook cache implementation, we adopt a reorder-based static mapping mechanism that is extremely lightweight and configurable, which means there is also no complex eviction policy. 

In our implementation, we first sort and reorder the codebook entries by their access frequency in the descending order. 

1501 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:14:40 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [235 x 34] intentionally omitted <==**

**----- Start of picture text -----**<br>
Block ID Entry ID<br>**----- End of picture text -----**<br>


Fig. 9. Entries hot and cold of different parts of tensor. 

This is done at the profiling-based offline phase, which ensures that the index of the most frequent entry is 0, and the index of the least frequent entry is the maximum value. All the quantized data would use these new indices. Next, we establish two boundaries: _nreg_ and _nshared_ . We allocate the first _nreg_ entries to thread local registers and the subsequent entries up to _nshared_ in shared memory. We store any remaining entries in global memory. During runtime dequantization, addressing codebook entries involves simple index comparisons, we locate entries in registers if the index _<nreg_ , in shared memory if _nreg≤_ index _<nshared_ , and in global memory if the index _≥nshared_ . 

In this implementation, we conduct frequency-based reordering at the tensor level, although different parts of a tensor might have different frequently accessed entries. Fig. 9 presents data to support our choice, where the y-axis represents different parts of the tensor (i.e., different thread blocks), and x-axis indicates the access frequencies of different codebook entries of a thread block. White color indicates frequently accessed entries, and the opposite for darker shades. We observe many vertical white lines, suggesting that these entries are consistently accessed across different tensor parts. This observation supports the rationale for globally determining the most frequently accessed entries. 

_**Adaptivity.**_ The shared memory and register resources of our codebook cache can be adjusted using two parameters: _nreg_ and _nshared_ . As mentioned in Sec. III, these resources are limited on GPUs, and excessive usage can decrease the occupancy of thread blocks. We employ a heuristic-based method that adapts their allocation to subsequent computations. Initially, we identify slack in the use of both recources. This concept is illustrated in Fig. 10, where we assign varying amounts of shared memory and registers to two computation kernels, highlighting the most performant configuration with a circle marker. Resource slack, depicted as the blue shaded area in Fig. 10, is a space of resource that we can occupy without hurting concurrency and GPU utilization. The existence of these slacks is due to the GPU’s resource partitioning and scheduling [52], which we will not explore further due to space constraints. It is important to note that different computations exhibit varying slacks, which can also be derived by offline profiling. We determine _nreg_ and _nshared_ by dividing the available slacks by the size of a single codebook entry. 

## _C. User Interface_ 

We provide and explain the following APIs for users to utilize our codebook cache, henceforth abbreviated as CB. 

_CBcached, nreg,shared ←_ **Load** ( _CB, Slack_ ) _Entry ←_ **Access** ( _CBcached, nreg,shared, CB, Index_ ) _CB ←_ **Switch** ( _New CB Pointer_ ) 

**==> picture [184 x 76] intentionally omitted <==**

**----- Start of picture text -----**<br>
Computation resource consumption<br>1<br>0<br>1<br>0<br>0 64 128 192 0 32 64 96<br>B32 registers (#) Shared memory (KB)<br>OP A<br>OP B<br>GPU occupancy<br>**----- End of picture text -----**<br>


Fig. 10. Computation kernel resource consumption and corresponding occupancy of the hardware. The blue region is the resource slacks we can use without influencing the performance. 

**==> picture [224 x 89] intentionally omitted <==**

**----- Start of picture text -----**<br>
+<br>K cache (1 head) Reduce V cache (1 head)<br>Partial Attn mapSoftmax<br>TB 0 TB 1 TB 2 TB.. TB 0 TB 1 TB 2 TB..<br>Tokens Tokens<br>Codebook 0 Codebook 1 Codebook … Codebook 31 Codebook 0 Codebook 1 Codebook … Codebook 31<br>**----- End of picture text -----**<br>


Fig. 11. Example of codebook centric dataflow for attention (decode) computation following CQ configuration. 

The first API is **Load** , which loads codebooks stored in global memory into the cache. It accepts the codebooks and memory slack, returning the codebooks cached across the memory hierarchy along with two access boundaries. The second API is **Access** , allowing users to access specific entries during the dequantization process. It accepts cached and global memorystored codebooks along with indices to locate entries. It also uses two boundaries to determine where to locate entries. Additionally, while we configure these boundaries with preset heuristics, users can still overwrite them. 

The last API is **Switch** , useful when algorithms train different codebooks for different parts of a tensor, as in GPTVQ2 [57]. This API facilitates the switch to new codebooks based on the specific tensor section being processed by the user. 

## VI. CODEBOOK-BASED COMPUTE ENGINE 

Based on the above codebook cache, we design an efficient compute engine to optimize the excessive codebook-related traffic when using VQ in the subsequent computation. We first introduce two core techniques employed by our computation engine: codebook-based dataflow and codebook-based hierarchical fusion. We then detail the combined usage of the entire computation engine along with the codebook cache. 

## _A. Codebook Centric Dataflow_ 

We start by explaining the intuition of our design. Subsequently, we detail our implementation. 

_1) Design Rationale:_ To fully leverage the parallel computation resources of GPUs, researchers employ tiling to divide and parallelize computation tasks [6], [43], [73]. Under the VQ scenario, naive parallelization introduces excessive traffic due to conflicts between the codebook switch axes and the task reduction axes, as discussed in Sec. III. We address this issue with a new codebook-centric dataflow illustrated in Fig. 11, 

1502 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:14:40 UTC from IEEE Xplore.  Restrictions apply. 

which employs the same settings as Fig. 5 in Sec. III. In this codebook-centric dataflow, we partition and parallelize the task across every four channels, i.e., every codebook, ensuring that each thread block only needs to load one codebook, thus eliminating any need for duplicated codebooks or switches. Instead of globally reducing the local softmax of different tokens as in FlashDecoding [10], we now require global accumulation of partial inner-products. 

_2) Implementation:_ We now formally define our design for the codebook-based dataflow. We first identify the axes where reduction occurs and where codebooks need to be switched, as indicated in Tbl. III. Subsequently, we split and parallelize the computation along the codebook switch axes. Finally, for those axes that traditionally perform temporal accumulation but are now parallelized (intersecting with the codebook switch axes and annotated with colors), we perform an explicit global reduction to ensure accurate results. 

_**Adaptivity.**_ To balance the overhead of global reduction in our dataflow, we utilize a split factor to control the extent of task parallelization along the codebook switch axes. A larger split factor results in fewer duplicated codebooks but necessitates more global reductions, and vice versa. With the objective of minimizing overhead, we adaptively determine the split factor based on the size of the tensor that needs reduction and the traffic associated with duplicated codebooks. 

**==> picture [163 x 29] intentionally omitted <==**

Since these two variables exhibit opposing trends with respect to the split factor, we can achieve a minimum by equating them according to the Mean Value Theorem [48]. 

## _B. Codebook Centric Hierarchical Fusion_ 

Similarly, we begin with a concrete example to illustrate our new fusion scheme. Subsequently, we formally abstract the hierarchical fusion algorithm and detail our implementation. 

_1) Design Rationale:_ The baseline method described in Sec. III employs shared-memory-level fusion, which combines VQ dequantization and the subsequent computation kernel by transferring data through shared memory. It leads to excessive 

## TABLE III 

REDUCE AND CODEBOOK SWITCH AXES OF COMPUTATIONS 

|GeMM<br>GeMV|All axes|Reduce<br>axes|Codebook<br>switching axes|
|---|---|---|---|
|Weight|**M,N,R**|**M,R**|**R**: AQLM,QuiP#<br>**M,N**: GPT-VQ|
|_R: Residual, M,N: M rows, N columns_||||
|Attention|All axes|Reduce<br>axes|Codebook<br>switching axes|
|K Cache<br>V Cache|**B,H,T,C**<br>**B,H,T,C**|**C**<br>**T**|**H,C**: CQ<br>**H,C**: CQ|



_B: Batch, H: Head, T: Token, C: Channel_ 

**==> picture [228 x 98] intentionally omitted <==**

**----- Start of picture text -----**<br>
Quantized Thread id Register array index<br>0 2 0 1 2 3 0 1 2 3 0 0 2 3 0 1 2 2<br>reg[tid^1]=shfl(reg[tid^1],1) reg[tid^2]=shfl(reg[tid^2],2)<br>1 3 0 1 2 3 0 1 2 3 1 1 2 3 0 1 3 3<br>(16,2) (16,2) (16,16) (16,16)<br>mma.sync<br>0 0 0 0 2 2 2 2 0 0 0 3 2 1 2 2<br>reg[tid^3]=shfl(reg[tid^3],3)<br>1 1 1 1 3 3 3 3 1 1 2 1 0 3 3 3<br>(16,16) (16,16)<br>ThreadMapping Dequantize Shuffle<br>Shuffle<br>Matched!Compute Shuffle<br>Warp tile<br>**----- End of picture text -----**<br>


Fig. 12. Intra-warp data exchange based on shuffle API example, eight elements are dequantized one time per thread, while following computation requires one thread hold only two elements (mma instructions). 

traffic between shared memory and registers, as previously explained. Alternatively, we utilize a modern GPU feature that facilitates register-level data exchange [42], effectively bypassing shared memory with following API: 

**==> picture [188 x 10] intentionally omitted <==**

This API exchanges the _reg_ of the calling thread ( _idsrc_ ) with _reg_ of the thread whose _iddst⊕offset_ = _idsrc_ **in place** ( _⊕_ : _xor_ ). Note that this instruction is commonly used to enhance the efficiency of collective communication and result reduction [27], [72]. However, we are the first to apply it to accelerate VQ-compressed LLMs. 

We illustrate the application of this API for register-level fusion through an example that fuses **VQ<8,...>** with GeMM. In Fig. 12, the layout of the dequantized data is 8 (i.e., VQ vector size), while the layout required by the mma instruction is 2. We initially map the dequantization threads in a specialized manner, as depicted in the figure, to ensure that all data exchanges are confined to four threads, which we subsequently refer to as a mini-warp. Within this mini-warp, we execute three exchange (shfl) operations as follows: 

**==> picture [170 x 9] intentionally omitted <==**

**==> picture [170 x 9] intentionally omitted <==**

**==> picture [170 x 9] intentionally omitted <==**

Note that both the array index and thread ID can be represented using the xor operation. After these shuffle operations, the data held by each thread’s register aligns precisely with the requirements of the mma computation instruction. 

_**Thread Mapping.**_ Our approach necessitates a specialized thread mapping within a warp for dequantization, as the naive sequential mapping requires a complex exchange pattern. Consider the sequential mapping with the mma instruction in Fig. 12, data[8,0:8] (blue color) is dequantized by thread 16 but is required by threads 0-3. However, the data held by threads 0-3 is not needed by thread 16 but rather by threads 0-7. This results in a complex data exchange path where ultimately all threads are implicated. Meanwhile, it requires additional registers as the exchange happens in place. To circumvent this, we predetermine the thread mapping offline, based on the layout of the dequantized data and the layout required by the computation, with details described as follows. 

1503 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:14:40 UTC from IEEE Xplore.  Restrictions apply. 

**Algorithm 1** Intra-warp data exchange based on shuffling 

|**Input:** _data, iter, layoutdequant,compute_|**Input:** _data, iter, layoutdequant,compute_|||
|---|---|---|---|
|**Output:** _data_||||
|1: **function** THREAD<br>MAPPING(_data, layoutdequant,compute_)||||
|2:|**for** _item ∈data_ **do**|||
|3:|_item.tidcompute,dequant ←GetTid_(_item, _|_layoutcompute,dequant_)||
|4:|_mini_<br>_warps ←_[]|||
|5:|**for** _dequant_<br>_thread ∈warp_ **do**|||
|6:<br>7:|_mw ←_[_data.tidcompute_ **_for_** _data.tiddequant_**=**_dequant_<br>**if** _mw /∈mini_<br>_warps_ **then**||_thread_]|
|8:|_mini_<br>_warps_[_mw_]_←_[]|||
|9:|_mini_<br>_warps_[_mw_]_.append_(_dequant_|_thread_)||
|10:|**for** _mw ∈mini_<br>_warps_ **do**|||
|11:|_mini_<br>_warps_[_mw_][_i_]_←mw_[_i_]**_// Thread mapping _**||**_we need_**|
|12: **function** REG<br>FUSION(_data, iter_)||||
|13:|**for** _off_ **in** [1_, iters_) **do** **_// intersected_** 0 **_no shuffe needed_**|||
|14:|_data_[_tid_**ˆ**_off_]_←shflxor_(_data_[_tid_**ˆ**_off_]_,off_)|||
|15:|**return** _reg_|||



_2) Implementation:_ We outline our algorithm in Alg. 1. To determine the thread mapping, we first find the association between each element in terms of dequantization and computation (lines 2-3). Subsequently, for each thread, we identify all threads that require its dequantized data, grouping these threads into a mini-warp (lines 4-6). We then construct miniwarps for all threads (lines 7-9). In the previous example, threads 0, 1, 16, and 17 possess identical data [0 _,_ 1 _,_ 2 _,_ 3] and thus form a mini-warp. Finally, we remap all threads by miniwarps (lines 10-11); for instance, we assign threads 2 and 3 to dequantize the data initially handled by threads 16 and 17. This process is executed offline to ensure proper thread mapping in runtime dequantization, enabling the implementation of register-level fusion via the shuffle API (lines 12-15). 

_**Adaptivity.**_ Clearly, a larger discrepancy between the dequantization layout and the required layout of the computation kernel increases the need for shuffling. Consequently, we propose conducting hierarchical fusion adapted to the vector size of the codebook entry. Profiling results indicate that the latency of shared memory access is nearly five times that of register access combined with shuffling. Therefore, for quantized tensors requiring fewer than five shuffle operations, we implement register-level fusion. For other tensors, we maintain the conventional shared memory-level fusion. 

## _C. Overall Workflow_ 

Our compute engine adopts a template-based design in Alg. 2 to generate final fused kernels. First at the offline phase, based on the VQ configuration and targeted computation, we determine shared/register budgets, split factors, required number of shuffles, and the corresponding thread mapping for our proposed optimizations (lines 2-8). 

Subsequently, we launch the codebook-centric dataflow computation (line 9) via the **Parallel_For** function that binds following sub-tasks to parallel thread blocks. Its two parameters represent the task splitting axes and the split factor, respectively. Within each parallelized task, we first load the codebook into the codebook cache (lines 10-12), followed by dequantization using the provided APIs in Sec. V (lines 

13-14). Notice now threads are mapped to quantized data following **Thread_Mapping** determined offline, for minimum shuffle if applicable. After dequantization, we perform codebook-centric hierarchical fusion (lines 15-18) using the **Reg_Fusion** and **Shared_Fusion** function. Both functions accept dequantized data, with the former requiring a counter _nshuffle_ to indicate the number of required shuffle operations and latter requiring the source-destination layout to initialiate correct shared memory accesses. Once the data is in the proper layout, we proceed with computation (lines 19-20). Finally, we perform a global reduction if necessary (line 21) via the **Reduce** function, where the first parameter specifies the partial result to be reduced and the second determines the axes along which the global reduction is conducted. 

**Algorithm 2** Complete VQ-aware computation template **Input:** _quantized, codebook, compute op_ **Output:** _output_ 

|1:|**function** KERNEL<br>TEMPLATE||||
|---|---|---|---|---|
|2:|_All, Reduce ←compute_<br>_op.all_<br>_axes, reduce_<br>_axes_||||
|3:|_layoutsrc,dst ←codebook.vector_<br>_size, compute_<br>_op.required_|||_size_|
|4:|_Budget ←Free shared and reg to preserve occupancy_||||
|5:|_factor ←Value to make TraffcReduce=TraffcCodebook_||||
|6:|_nshuffe ←layoutsrc/layoutdst_||||
|7:|**if** _nshuffe ≤thresshuffe_(= 5) **then**||||
|8:|**Thread_Mapping**(_compute_<br>_op.warp_<br>_tile, layoutsrc,dst_)||||
|9:|**Parallel_For**(_codebook.switch_<br>_axes, factor_)||||
|10:|**if** _required by algorithm_ **then**||||
|11:|_CB ←_**Switch**(_New codebook ptr_)||||
|12:|_CBcached, boundry ←_**Load**(_CB, Budget_)||||
|13:|**for** _id_ **in** _quantized_<br>_data_ **do**||||
|14:|_data ←_**Access**(_CBcached, boundry, CB, id_)||||
|15:|**if** _nshuffe ≤thresshuffe_ **then**||||
|16:|_data ←_**Reg_Fusion**(_data, nshuffe_)||||
|17:|**Else**||||
|18:|_data ←_**Shared_Fusion**(_data, layoutsrc,dst_)||||
|19:|**for** _temporal_<br>_iteration_ **on** _All −codebook.switch_|_axes_||**do**|
|20:|_partial ←compute_<br>_op_(_data, temporal_<br>_iteration_)||||
|21:|_output ←_**Reduce**(_partial, Reduce ∩codebook.switch_||_axes_)||
|22:|**Return** _output_||||



## VII. EVALUATION 

In this section, we evaluate the effectiveness of proposed optimizations in VQ-LLM through comprehensive experiments. We first present overall speedup results for various VQbased computation kernels over existing approaches. Then, we provide a detailed breakdown analysis of the proposed optimizations. Next, we compare our work with FP16 kernels and several element-wise quantization works to show its viability for accelerating LLMs. Finally, we performed a comprehensive end-to-end evaluation, analyzing both the overall speedup and accuracy across various GPUs. 

## _A. Experimental Setup_ 

In this study, we conduct a comprehensive evaluation at both the individual kernel and end-to-end model levels. The evaluations were performed on an NVIDIA RTX 4090 24GB GPU [44]. For the end-to-end evaluation, we included a Tesla A40 GPU [39] to explore the potential of VQ-LLM with lower bandwidth.. The evaluated computation kernels 

1504 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:14:40 UTC from IEEE Xplore.  Restrictions apply. 

TABLE IV 

BREAK DOWN ANALYSIS CONFIGURATION 

|ID<br>Optimization<br>Category<br>Description|ID<br>Optimization<br>Category<br>Description|ID<br>Optimization<br>Category<br>Description|
|---|---|---|
|GC|No|Naive implementation|
|SC|Greedy|Cache all entries in shared memory|
|O1<br>O2|Hierarchical<br>Buffer|+ Shared memory level caching (medium entries)<br>+ Register level caching (hot entries)|
|O3<br>O4|Compute<br>Engine|+ Codebook centric datafow<br>+ Codebook centric hierarchical fusion|



include various VQ-augmented GeMM, GeMV and FlashDecoding [10]. The evaluated VQ configurations are listed in Tbl. II, including QuiP#-4, AQLM-3, GPTVQ-2 and CQ2/4, the suffix number represent the equivalent bit-width. The first two kernels adopt weight quantization and the last one adopts KV cache quantization. For the kernel-level evaluation, we set the shape for these kernels following the Llama-7B and Llama-65B [55] models. These kernels run on a single GPU, while large model serving like Llama-65B typically uses multiple GPUs with Tensor Parallel (TP) strategy [35], [47], [71]. The required adjustments to our framework include final results gathering for Attention and partial results concatenation/reduction for GeMM/GeMV [38]. These are usually conducted via communication library like NCCL [45], and we identify this distributed scenario an orthogonal topic and leave it to the future work. 

Tbl. IV lists various baselines and VQ-LLM optimizations used in our evaluation. For the baselines, we use **GC** and **SC** method explained in Sec. III that stores the codebook in global memory and shared memory, respectively. For the results, we report the latency reduction against **GC** . We also decompose the optimizations used in VQ-LLM into four levels ( **O1** - **O4** ), with each explained in Tbl. IV. We also compare VQ-LLM with SOTA element-wise quantization methods under the same equivalent 4-bit width, including AWQ [30] for GeMM/GeMV and QoQ for Attention [31], all integrated into qServe [31]. For FP16, we use cutlass [43] and flash-attn [8]. 

In practice, LLM inference involves various operators beyond GeMM/GeMV and Attention, such as RMSNorm [66], SiLU [14], and RoPE [53], etc. Therefore, it is crucial to evaluate the end-to-end speedup that accounts for all operators. For the end-to-end evaluation, we set a batch size of 16 and a sequence length of 1024, measuring the total latency for generating 256 tokens. We also assess accuracy using the arcchallenge task [5], applying the QuiP#-4 and CQ-4 algorithms for quantizing the weights and KV-Cache, respectively. To obtain the final accuracy results, we integrate these algorithms into the LMEval framework [16]. 

## _B. Overall Speedup_ 

As shown in Fig. 13, VQ-LLM reduces the latency by an average of 46.13% (53.73% at most), corresponding to a speedup of 1.9 _×_ (2.2 _×_ ) (BS _x_ indicates the batch size of _x_ ). 

**==> picture [227 x 56] intentionally omitted <==**

**----- Start of picture text -----**<br>
80%<br>GeMM GeMV BS1 GeMV BS16 1k  Attention  4k Mean<br>40%<br>0%<br>7B<br>QuiP#-4AQLM-3GPTVQ-2QuiP#-4AQLM-3GPTVQ-2QuiP#-4AQLM-3GPTVQ-2CQ-2 BS1CQ-2 BS8CQ-2 BS1CQ-2 BS8 65B<br>Latency reduced<br>**----- End of picture text -----**<br>


Fig. 13. Overall latency reduction of best perform version against unoptimized version for various VQ configurations. 

For Attention (Decode), 1k and 4k means sequence length of 1024 and 4096, respectively. 

Although VQ-LLM achieves significant speedup values for both GeMM and GeMV kernels, we observe a counterintuitive discrepancy that our optimizations achieve a relatively high speedup value for GeMM kernels compared to GeMV kernels. In other words, the quality of VQ algorithm integration is more critical to the compute-bound kernels (e.g., GeMM) than to the memory-bound kernels (e.g., GeMV). The reason is the former benefit less from reduced memory footprint while suffer more from extra operation (dequantization) [60], leading to significant performance degradation of unoptimized implementation. Meanwhile, we also observe an opposite trend for AQLM-3 between GeMM and GeMV. This AQ configuration has an unaligned 12-bit storage format, which necessitates additional unpacking and decoding logic and requires a more careful optimization for the integration. 

We observe that our speedup values for GeMV kernels remain consistent regardless of batch size, whereas they increase with batch size for attention kernels. This is because different input samples share the same weight tensor but have distinct KV caches. Since the GeMV kernel corresponds to weight quantization and the attention kernel to KV cache quantization, the former only requires loading the VQ-compressed weight tensor once, while the latter loads VQ-compressed KV cache tensors multiple times. Consequently, our optimizations are more effective for the attention kernel with large batch sizes. 

Moreover, Llama-65B achieves almost identical speedup to Llama-7B, except in the Attention (Decode) scenario with a 1k sequence length and a single batch. This identical speedup occurs because the operators in the larger model can be trivially assembled using those from the smaller ones. We can readily double the launched thread blocks when we double the hidden dimension, demonstrating the good scalability of our optimizations. The sole exception arises because, in Llama7B, the baseline cannot fully utilize the hardware due to an insufficient number of thread blocks for a 1k sequence length single batch. In contrast, for Llama-65B, the baseline fully occupies the hardware, resulting in better performance and reducing the relative speedup of our system. 

## _C. Speedup Breakdown_ 

We first analyze the speedup breakdown of GeMM and GeMV, as depicted in Fig. 14. Tbl. V enumerates several factors that influence optimization effects, facilitating our analysis. For QuiP#-4, **SC** and **O1** perform identically due to the small size of its codebook (i.e., 2 KB in Tbl. V). 

1505 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:14:40 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [232 x 65] intentionally omitted <==**

**----- Start of picture text -----**<br>
BS1 BS16<br>GC SC O1 O2 O3 O4 GC SC O1 O2 O3 O4 GC SC O1 O2 O3 O4<br>QuiP#-4 AQLM-3 GPTVQ-2<br>2<br>2 1<br>1<br>1<br>0 0 0<br>150 200<br>10050 100 50<br>0 0 0<br>Latency3us)(10<br>(us)<br>Latency<br>**----- End of picture text -----**<br>


**==> picture [231 x 40] intentionally omitted <==**

**----- Start of picture text -----**<br>
1 CQ-2<br>0<br>1k BS1 1k BS8 4k BS1 4k BS8<br>40 150 60 400<br>20 100 30 200<br>0 050 0 0<br>1k BS11k BS84k BS14k BS8<br>Latency (us) Relative latency<br>GC SC O1 O2 O3 O4 GC SC O1 O2 O3 O4 GC SC O1 O2 O3 O4 GC SC O1 O2 O3 O4<br>**----- End of picture text -----**<br>


Fig. 15. (left) Breakdown of optimizations of CQ-2 for Attention (Decode). (right) Relative latency of CQ-4 against CQ-2. 

Fig. 14. Breakdown of optimizations for GeMM (upper) and GeMV (lower). 

area have a lesser impact on performance. 

TABLE V 

FACTORS THAT INFLUENCE THE EFFECT OF OPTIMIZATIONS 

|Item|QuiP#-4|AQLM-3|GPTVQ-2|CQ-2|
|---|---|---|---|---|
|Codebook/block<br>#Entry freq_> µ_+3_σ_<br>Output size/block<br>#Shuffe|2 KB<br>128 KB<br>32 KB<br>1-3<br>15-30<br>_<_1<br>32 KB/_<_1 KB*<br>3/7*<br>3/7*<br>1/3|||64 KB<br>_<_1<br>1-4 KB<br>3|



_*GeMM/GeMV_ 

AQLM-3 and GPTVQ-2 exhibit noticeable improvements, attributed to their larger codebooks. Additionally, for GeMV, **SC** has a significantly negative impact on AQLM-3, due to its large codebook (i.e., 128 KB in Tbl. V), which restricts the parallelization of memory-bound computations. 

**O2** delivers the most improvement in AQLM-3; we find that frequencies of 15-30 entries exceed _µ_ +3 _σ_ , and **O2** ’s registerlevel caching optimization effectively reduces bank conflicts when accessing these entries. Conversely, the remaining two configurations QuiP#-4 and GPTVQ-2 exhibit far fewer entries exceeding _µ_ +3 _σ_ , indicating the less optimization opportunity of register-level caching and hence marginal improvements. 

**O3** affects GeMM and GeMV differently. In GeMM, **O3** introduces negative effects due to a large output size. Furthermore, multiple residuals in QuiP#-4 configuration lead to redundant computations for **O3** , causing significantly increased latency in GeMM. In contrast, for AQLM-3, its misaligned 12-bit indices result in costly unpacking and decoding. It leads to low compute pipeline utilization, and hence is more tolerant to redundant computations. In GeMV, the output size is much smaller and the computation is lighter compared to GeMM. The smaller output size results in minimal global reduction overhead, and the lighter computation introduces less computational overhead than in GeMM. These factors make **O3** more advantageous for GeMV. 

**O4** significantly enhances GeMM’s performance. This improvement primarily stems from GeMM’s utilization of mma instructions, which require a layout of 2 and can be satisfied through one to three shuffling instructions. Additionally, **O4** conserves a substantial amount of shared memory, which is crucial as GeMM typically consumes a large shared memory, thus yielding a significant positive impact. Conversely, GeMV requires element-wise reduction, resulting in QuiP#-4 and AQLM-3, with a vector size of 8, requiring a greater number of shuffling instructions. This leads to a slowdown in these configurations. However, for GPTVQ-2 with a vector size of 4, a slight improvement is still observed. Furthermore, since GeMV typically uses minimal shared memory, savings in this 

For Attention (decode), VQ-LLM achieves similar improvements with various sequence lengths and batches. **SC** significantly reduces performance due to CQ’s large codebook, necessitating the use of **O1** for achieving high performance. **O2** offers only a slight improvement because few entries are accessed very frequently, mirroring situations in QuiP#4 and GPTVQ-2. **O3** significantly enhances performance by eliminating considerable duplicated traffic in the original computation dataflow. **O4** provides a minor improvement, for reasons similar to those for **O4** in GeMV. Additionally, we illustrate the latency of CQ-4 relative to CQ-2 in the right part of Fig. 15. Our proposed optimizations achieve a similar speedup to CQ-2, so we omit the detailed results to save space. 

## _D. FP16 and Element-wise Quantization Comparison_ 

We now compare the latency of our optimized VQ kernels against FP16 and element-wise quantization works. Under the same equivalent bit width, the latency of kernels with the element-wise quantization is the theoretical upper bound of VQ kernels if using the same computation dataflow. As such, this comparison further verifies the effectiveness of our work. 

As shown in Fig. 16, at 4-bit encoding, our work achieves latencies comparable to (1.01 _×_ for Attention (Decode)), or even lower than (0.88 _×_ /0.96 _×_ for GeMV/GeMM), those of AWQ [30] and QoQ [31]. This reduction in latency likely results from our co-designed computational dataflow. These results suggest that our implementation is as viable as AWQ and QoQ, and therefore comparable to qServe [31]. Moreover, VQ kernels can deliver better accuracies at the same bit-width. The open-source implementations of QuiP# [56] and AQLM [12] are impractical for real-world applications, exhibiting 2.83 _×_ to 114.4 _×_ latencies. Our work successfully translates theoretical algorithmic improvements into practical applications. 

We would like to explain that in Fig. 16, while both our approach and element-wise quantization methods outperform the cutlass-FP16 baseline in GeMV and Attention kernels, both underperform relative to the cutlass-FP16 baseline in GeMM kernels. This underperformance is due to the complex tiling 

**==> picture [225 x 83] intentionally omitted <==**

**----- Start of picture text -----**<br>
AWQ-4bit QoQ-4bit (All from qServe) *: Open source impl.<br>10<br>114× GeMM GeMV Attention<br>BS16 BS1 1k<br>1<br>0.1<br>Fig. 16. Latency comparing to element-wise quantization works.<br>QuiP#-4*AQLM-4*cutlass-16QuiP#-4GPTVQ-2 QuiP#-4*AQLM-4*cutlass-16QuiP#-4GPTVQ-2 Flash-16CQ-4CQ-2<br>Relative latency<br>**----- End of picture text -----**<br>


1506 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:14:40 UTC from IEEE Xplore.  Restrictions apply. 

**==> picture [230 x 66] intentionally omitted <==**

**----- Start of picture text -----**<br>
3<br>2 46% FP16 VQ-LLM (4 bit)<br>1 FP16 45%<br>0<br>44%<br>qServe (4 bit)<br>1 2<br>qServe (4 bit)VQ-LLM (4 bit)VQ-LLM (2 bit)VQ-LLM (4 bit)@Tesla A40 E2E Speedup<br>E2E<br>Speedup<br>Accuracy<br>arc-challenge<br>**----- End of picture text -----**<br>


**==> picture [232 x 63] intentionally omitted <==**

**----- Start of picture text -----**<br>
Seq_len = 1k Seq_len = 2k Seq_len = 4k<br>01234 BS1 BS8<br>Flash Flash Flash Flash<br>DecodingPaged FlashDecoding AttentionPaged FlashAttention DecodingPaged FlashDecoding AttentionPaged FlashAttention<br>to VQ-LLM<br>Relative latency<br>**----- End of picture text -----**<br>


Fig. 18. Relative latency of various attention baselines against our best perform implementation of CQ-4. 

Fig. 17. (left) Overall speedup against FP16 and (right) accuracy of arcchallenge of SOTA element-wise quantization (qServe) and VQ-LLM. 

strategy employed by cutlass-FP16 GeMM, which could incorporate our method. However, we do not pursue this integration for two reasons. First, accelerating individual GeMM kernels offers minimal overall speedup for LLM inference, as these kernels are used in the prefilling stage (Sec. II-B). The decoding stage, which dominates LLM inference execution time, has a greater impact on performance [64], as confirmed by our end-to-end evaluation results in the next subsection. Second, modifying the cutlass code requires significant engineering effort due to its intricate, template-based kernel design [22], [76]. Therefore, we leave this integration for future work. 

## _E. End-to-End (E2E) Result_ 

We present the end-to-end LLM inference improvements of various quantization methods compared to the FP16 baseline in Fig. 17 (left). In the equivalent 4-bit setting, VQ-LLM achieves a speedup comparable to the state-of-the-art elementwise quantization method, qServe [31], with both providing approximately a 2.2 _×_ improvement over the FP16 baseline. Additionally, VQ-LLM surpasses qServe in accuracy by about 2.5% on the arc-challenge task [5], as shown in Fig. 17 (right). This result demonstrates VQ-LLM’s effectiveness in accelerating LLM inference. he RMSNorm, SiLU, and RoPE operators together account for roughly 10% and 20% of total latency in the FP16 and 4-bit quantized versions, respectively. We also observe a greater speedup with a 2-bit compression ratio, further highlighting the potential of VQ, as previous research suggests that 2-bit quantization can maintain practical accuracy. Additionally, we evaluate the performance of VQLLM in a 4-bit setting on the Tesla A40 GPU, which provides 67% of the memory bandwidth of the RTX 4090 [39]. Interestingly, the Tesla A40 demonstrates a greater speedup than the RTX 4090, suggesting that VQ-LLM is more effective in bandwidth-constrained environments. In summary, VQ-LLM offers improved accuracy with comparable latency to elementwise quantization, and vice versa. In terms of memory usage, the FP16 baseline consumes over 22 GB, whereas qServe-4 and VQ-LLM-4 use less than 6 GB of GPU memory, aligning closely with theoretical estimates [65]. 

## _F. Additional Discussion_ 

_**Different Types of Attention.**_ The aforementioned details about Attention (Decode) are based on using Flash Decoding [10] as our baseline dataflow. We also evaluate the speedup of our work against various attention baselines, including Flash 

Attention, Paged Flash Attention and Paged Flash Decoding [7]–[9], [50]. As illustrated in Fig. 18, our work surpasses all these baselines, primarily due to a significantly reduced KV cache memory footprint enabled by CQ-4. We achieved a 66.4% latency reduction compared to the best-performing FP16 baseline, with a 75% reduction in memory footprint, under the conditions of eight batches and a sequence length of 4096. This indicates an effective transfer from theoretical benefit to practical application. Additionally, our work scales effectively with increases in sequence length and batch size. 

_**Quantization Overhead.**_ For weight compression, no runtime quantization overhead is introduced. In KV cache compression, the runtime overhead of on-the-fly quantization for the new key and value of a new token in the decode phase is negligible ( _<_ 1 _µs_ ). During the prefill phase, quantizing the keys and values of all prompt tokens introduces less than a 10% overhead compared to linear projections. However, the subsequent computation does not immediately require the quantized KV cache, rendering these overheads negligible. 

## VIII. CONCLUSIONS 

In this work, we proposed VQ-LLM, an optimized code generation framework for vector quantization (VQ), consisting of codebook cache and codebook based compute engine. With which we achieve 46.13% latency reduction on average over unoptimized version and up-to 99% over open source implementations. For codebook cache, we propose a hierachical placement strategy to preserve hardware utilization and reduce bank conflict. For compute engine, we propose codebook centric dataflow and fusion scheme to reduce excessive off-chip and on-chip traffic. All proposed optimizations are configured adaptively via several heuristics. Finally we demostrate effectiveness and viability of VQ-LLM comparing to un-optimized implementations and element-wise quantization works. 

## IX. ACKNOWLEDGEMENTS 

This work was supported by the National Key R&D Program of China under Grant 2022YFB4501400, the National Natural Science Foundation of China (NSFC) grant (62222210, U21B2017 and 62072297). This work was also supported by Shanghai Qi Zhi Institute Innovation Program SQZ202316. We would like to thank the anonymous reviewers for their constructive feedback and comments to improve this work. Special thanks to Vega Jiang for continuous help and support. Any opinions, findings and conclusions in this paper are those of the authors only and do not necessarily reflect the views of our sponsors. 

1507 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:14:40 UTC from IEEE Xplore.  Restrictions apply. 

## REFERENCES 

- [1] “Ieee standard for floating-point arithmetic,” _IEEE Std 754-2019 (Revision of IEEE 754-2008)_ , pp. 1–84, 2019. 

- [2] E. Agrell and B. Allen, “On the best lattice quantizers,” _IEEE Trans. Inf. Theory_ , vol. 69, no. 12, pp. 7650–7658, 2023. 

- [3] AMD, “Amd cdna architecture: The all-new amd gpu architecture for the modern era of hpc & ai,” https://www.amd.com/content/dam/amd/en/documents/instinct-businessdocs/white-papers/amd-cdna-white-paper.pdf, 2020. 

- [4] A. Babenko and V. S. Lempitsky, “Additive quantization for extreme vector compression,” in _2014 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2014, Columbus, OH, USA, June 23-28, 2014_ . IEEE Computer Society, 2014, pp. 931–938. 

- [5] S. Bhakthavatsalam, D. Khashabi, T. Khot, B. D. Mishra, K. Richardson, A. Sabharwal, C. Schoenick, O. Tafjord, and P. Clark, “Think you have solved direct-answer question answering? try arc-da, the direct-answer AI2 reasoning challenge,” _CoRR_ , vol. abs/2102.03315, 2021. 

- [6] T. Chen, T. Moreau, Z. Jiang, L. Zheng, E. Q. Yan, H. Shen, M. Cowan, L. Wang, Y. Hu, L. Ceze, C. Guestrin, and A. Krishnamurthy, “TVM: an automated end-to-end optimizing compiler for deep learning,” in _13th USENIX Symposium on Operating Systems Design and Implementation, OSDI 2018, Carlsbad, CA, USA, October 8-10, 2018_ . USENIX Association, 2018, pp. 578–594. [Online]. Available: https://www.usenix.org/conference/osdi18/presentation/chen 

- [7] T. Dao, “Flashattention-2: Faster attention with better parallelism and work partitioning,” _CoRR_ , vol. abs/2307.08691, 2023. 

- [8] ——, “flash-attention,” 2024. [Online]. Available: https://github.com/ 

- [9] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. R´e, “Flashattention: Fast and memory-efficient exact attention with io-awareness,” in _Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022_ , 2022. 

- [10] T. Dao, D. Haziza, F. Massa, and G. Sizov, “Flash-decoding for long-context inference,” 2023. [Online]. Available: https://crfm.stanford. edu/2023/10/12/flashdecoding.html 

- [11] M. DouzeIR, “Faiss the index factory,” 2024. [Online]. Available: https://github.com/facebookresearch/faiss/wiki/The-index-factory 

- [12] V. Egiazarian, A. Panferov, D. Kuznedelev, E. Frantar, A. Babenko, and D. Alistarh, “Extreme compression of large language models via additive quantization,” _CoRR_ , vol. abs/2401.06118, 2024. 

- [13] ——, “Official pytorch repository for extreme compression of large language models via additive quantization,” 2024. [Online]. Available: https://github.com/vahe1994/AQLM 

- [14] S. Elfwing, E. Uchibe, and K. Doya, “Sigmoid-weighted linear units for neural network function approximation in reinforcement learning,” _Neural Networks_ , vol. 107, pp. 3–11, 2018. 

- [15] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, “GPTQ: accurate post-training quantization for generative pre-trained transformers,” _CoRR_ , vol. abs/2210.17323, 2022. 

- [16] L. Gao, J. Tow, B. Abbasi, S. Biderman, S. Black, A. DiPofi, C. Foster, L. Golding, J. Hsu, A. Le Noac’h, H. Li, K. McDonell, N. Muennighoff, C. Ociepa, J. Phang, L. Reynolds, H. Schoelkopf, A. Skowron, L. Sutawika, E. Tang, A. Thite, B. Wang, K. Wang, and A. Zou, “A framework for few-shot language model evaluation,” 07 2024. 

- [17] T. Ge, K. He, Q. Ke, and J. Sun, “Optimized product quantization,” _IEEE Trans. Pattern Anal. Mach. Intell._ , vol. 36, no. 4, pp. 744–755, 2014. 

- [18] C. Guo, Y. Qiu, J. Leng, X. Gao, C. Zhang, Y. Liu, F. Yang, Y. Zhu, and M. Guo, “Squant: On-the-fly data-free quantization via diagonal hessian approximation,” in _The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022_ . OpenReview.net, 2022. 

- [19] C. Guo, J. Tang, W. Hu, J. Leng, C. Zhang, F. Yang, Y. Liu, M. Guo, and Y. Zhu, “Olive: Accelerating large language models via hardwarefriendly outlier-victim pair quantization,” in _Proceedings of the 50th Annual International Symposium on Computer Architecture, ISCA 2023, Orlando, FL, USA, June 17-21, 2023_ . ACM, 2023, pp. 3:1–3:15. 

- [20] C. Guo, C. Zhang, J. Leng, Z. Liu, F. Yang, Y. Liu, M. Guo, and Y. Zhu, “ANT: exploiting adaptive numerical data type for low-bit deep neural network quantization,” in _55th IEEE/ACM International Symposium on Microarchitecture, MICRO 2022, Chicago, IL, USA, October 1-5, 2022_ . IEEE, 2022, pp. 1414–1433. 

- [21] C. Guo, R. Zhang, J. Xu, J. Leng, Z. Liu, Z. Huang, M. Guo, H. Wu, S. Zhao, J. Zhao, and K. Zhang, “Gmlake: Efficient and transparent GPU memory defragmentation for large-scale DNN training with virtual memory stitching,” in _Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, ASPLOS 2024, La Jolla, CA, USA, 27 April 2024- 1 May 2024_ . ACM, 2024, pp. 450–466. 

- [22] B. Hagedorn, B. Fan, H. Chen, C. Cecka, M. Garland, and V. Grover, “Graphene: An IR for optimized tensor computations on gpus,” in _Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3, ASPLOS 2023, Vancouver, BC, Canada, March 25-29, 2023_ . ACM, 2023, pp. 302–313. 

- [23] P. Haghani, S. Michel, P. Cudr´e-Mauroux, and K. Aberer, “LSH at large - distributed KNN search in high dimensions,” in _11th International Workshop on the Web and Databases, WebDB 2008, Vancouver, BC, Canada, June 13, 2008_ , 2008. 

- [24] C. Hooper, S. Kim, H. Mohammadzadeh, M. W. Mahoney, Y. S. Shao, K. Keutzer, and A. Gholami, “Kvquant: Towards 10 million context length LLM inference with KV cache quantization,” _CoRR_ , vol. abs/2401.18079, 2024. 

- [25] H. J´egou, M. Douze, and C. Schmid, “Product quantization for nearest neighbor search,” _IEEE Trans. Pattern Anal. Mach. Intell._ , vol. 33, no. 1, pp. 117–128, 2011. 

- [26] B. K¨ovesi, J. Boucher, and S. Saoudi, “Stochastic k-means algorithm for vector quantization,” _Pattern Recognit. Lett._ , vol. 22, no. 6/7, pp. 603–610, 2001. 

- [27] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. Yu, J. Gonzalez, H. Zhang, and I. Stoica, “vllm: Easy, fast, and cheap llm serving with pagedattention,” 2024. [Online]. Available: https://blog.vllm.ai/2023/06/20/vllm.html 

- [28] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang, and I. Stoica, “Efficient memory management for large language model serving with pagedattention,” in _Proceedings of the 29th Symposium on Operating Systems Principles, SOSP 2023, Koblenz, Germany, October 23-26, 2023_ . ACM, 2023, pp. 611–626. 

- [29] Y. Lee, H. Choi, S. Min, H. Lee, S. Beak, D. Jeong, J. W. Lee, and T. J. Ham, “ANNA: specialized architecture for approximate nearest neighbor search,” in _IEEE International Symposium on High-Performance Computer Architecture, HPCA 2022, Seoul, South Korea, April 2-6, 2022_ . IEEE, 2022, pp. 169–183. 

- [30] J. Lin, J. Tang, H. Tang, S. Yang, W. Chen, W. Wang, G. Xiao, X. Dang, C. Gan, and S. Han, “AWQ: activation-aware weight quantization for on-device LLM compression and acceleration,” in _Proceedings of the Seventh Annual Conference on Machine Learning and Systems, MLSys 2024, Santa Clara, CA, USA, May 13-16, 2024_ . mlsys.org, 2024. 

- [31] Y. Lin, H. Tang, S. Yang, Z. Zhang, G. Xiao, C. Gan, and S. Han, “Qserve: W4A8KV4 quantization and system co-design for efficient LLM serving,” _CoRR_ , vol. abs/2405.04532, 2024. 

- [32] S. Liu, H. Lu, and J. Shao, “Improved residual vector quantization for high-dimensional approximate nearest neighbor search,” _CoRR_ , vol. abs/1509.05195, 2015. 

- [33] Z. Liu, J. Leng, Z. Zhang, Q. Chen, C. Li, and M. Guo, “VELTAIR: towards high-performance multi-tenant deep learning services via adaptive compilation and scheduling,” in _ASPLOS ’22: 27th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Lausanne, Switzerland, 28 February 2022 - 4 March 2022_ . ACM, 2022, pp. 388–401. 

- [34] Z. Liu, W. Ni, J. Leng, Y. Feng, C. Guo, Q. Chen, C. Li, M. Guo, and Y. Zhu, “JUNO: optimizing high-dimensional approximate nearest neighbour search with sparsity-aware algorithm and ray-tracing core mapping,” in _Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, ASPLOS 2024, La Jolla, CA, USA, 27 April 20241 May 2024_ . ACM, 2024, pp. 549–565. 

- [35] G. Lu, R. Chen, Y. Wang, Y. Zhou, R. Zhang, Z. Hu, Y. Miao, Z. Cai, L. Li, J. Leng, and M. Guo, “Distsim: A performance model of largescale hybrid distributed DNN training,” in _Proceedings of the 20th ACM International Conference on Computing Frontiers, CF 2023, Bologna, Italy, May 9-11, 2023_ . ACM, 2023, pp. 112–122. 

- [36] Z. Ma, Y. Tan, H. Jiang, Z. Yan, D. Liu, X. Chen, Q. Zhuge, E. H. Sha, and C. Wang, “Unified-tp: A unified TLB and page table cache structure for efficient address translation,” in _38th IEEE International Conference on Computer Design, ICCD 2020, Hartford, CT, USA, October 18-21, 2020_ . IEEE, 2020, pp. 255–262. 

1508 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:14:40 UTC from IEEE Xplore.  Restrictions apply. 

- [37] E. Mata, S. Bandeira, P. S. G. de Mattos Neto, W. T. A. Lopes, and F. Madeiro, “Accelerating families of _Fuzzy K-Means_ algorithms for vector quantization codebook design,” _Sensors_ , vol. 16, no. 11, 2016. 

- [38] D. Narayanan, M. Shoeybi, J. Casper, P. LeGresley, M. Patwary, V. Korthikanti, D. Vainbrand, P. Kashinkunti, J. Bernauer, B. Catanzaro, A. Phanishayee, and M. Zaharia, “Efficient large-scale language model training on GPU clusters using megatron-lm,” in _International Conference for High Performance Computing, Networking, Storage and Analysis, SC 2021, St. Louis, Missouri, USA, November 14-19, 2021_ . ACM, 2021, p. 58. 

- [39] NVIDIA, “Nvidia ampere ga102 gpu architecture,” https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpuarchitecture-whitepaper-v2.pdf, 2021. 

- [40] ——, “Nvidia h100 tensor core gpu architecture,” https://www.advancedclustering.com/wpcontent/uploads/2022/03/gtc22-whitepaper-hopper.pdf, 2022. 

- [41] ——, “Cuda c++ programming guide,” 2024. [Online]. Available: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html 

- [42] ——, “Cuda c++ programming guide: Shared memory,” 2024. [Online]. Available: https://docs.nvidia.com/cuda/ cuda-c-programming-guide/index.html#shared-memory-5-x 

- [43] ——, “Cuda templates for linear algebra subroutines,” 2024. [Online]. Available: https://github.com/NVIDIA/cutlass 

- [44] ——, “Nvidia ada craft the engineering marvel of the rtx 4090.” 2024. [Online]. Available: https://images.nvidia.com/aem-dam/Solutions/ geforce/ada/ada-lovelace-architecture/nvidia-ada-gpu-craft.pdf 

- [45] ——, “Nvidia collective communications library (nccl),” 2024. [Online]. Available: https://developer.nvidia.com/nccl 

- [46] OpenAI, “GPT-4 technical report,” _CoRR_ , vol. abs/2303.08774, 2023. 

- [47] S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He, “Zero: memory optimizations toward training trillion parameter models,” in _Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, SC 2020, Virtual Event / Atlanta, Georgia, USA, November 9-19, 2020_ . IEEE/ACM, 2020, p. 20. 

- [48] K. A. Ross, _Elementary Analysis: The Theory of Calculus_ . Springer New York, NY, 2013. 

- [49] B. D. Rouhani, R. Zhao, A. More, M. Hall, A. Khodamoradi _et al._ , “Microscaling data formats for deep learning,” _CoRR_ , vol. abs/2310.10537, 2023. 

- [50] J. Shah, G. Bikshandi, Y. Zhang, V. Thakkar, P. Ramani, and T. Dao, “Flashattention-3: Fast and accurate attention with asynchrony and lowprecision,” _CoRR_ , vol. abs/2407.08608, 2024. 

- [51] W. Shao, M. Chen, Z. Zhang, P. Xu, L. Zhao, Z. Li, K. Zhang, P. Gao, Y. Qiao, and P. Luo, “Omniquant: Omnidirectionally calibrated quantization for large language models,” _CoRR_ , vol. abs/2308.13137, 2023. 

- [52] G. Shobaki, A. Kerbow, and S. Mekhanoshin, “Optimizing occupancy and ILP on the GPU using a combinatorial approach,” in _CGO ’20: 18th ACM/IEEE International Symposium on Code Generation and Optimization, San Diego, CA, USA, February, 2020_ . ACM, 2020. 

- [53] J. Su, M. H. M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu, “Roformer: Enhanced transformer with rotary position embedding,” _Neurocomputing_ , vol. 568, p. 127063, 2024. 

- [54] M. Thread, “Mtt s4000: Empower large model ai with no limits,” https://en.mthreads.com/product/S4000, 2024. 

- [55] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M. Lachaux _et al._ , “Llama: Open and efficient foundation language models,” _CoRR_ , vol. abs/2302.13971, 2023. 

- [56] A. Tseng, J. Chee, Q. Sun, V. Kuleshov, and C. D. Sa, “Quip#: Even better LLM quantization with hadamard incoherence and lattice codebooks,” _CoRR_ , vol. abs/2402.04396, 2024. 

- [57] M. van Baalen, A. Kuzmin, M. Nagel, P. Couperus, C. Bastoul, E. Mahurin, T. Blankevoort, and P. N. Whatmough, “GPTVQ: the blessing of dimensionality for LLM quantization,” _CoRR_ , vol. abs/2402.15319, 2024. 

- [58] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” in _Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA_ , 2017, pp. 5998–6008. 

- [59] H. Wang, T. Sun, and Q. Yang, “CAT - caching address tags: A technique for reducing area cost of on-chip caches,” in _Proceedings of the 22nd Annual International Symposium on Computer Architecture, ISCA ’95, Santa Margherita Ligure, Italy, June 22-24, 1995_ . ACM, 1995. 

- [60] S. Williams, A. Waterman, and D. A. Patterson, “Roofline: an insightful visual performance model for multicore architectures,” _Commun. ACM_ , vol. 52, no. 4, pp. 65–76, 2009. 

- [61] C. Wolters, X. Yang, U. Schlichtmann, and T. Suzumura, “Memory is all you need: An overview of compute-in-memory architectures for accelerating large language model inference,” _CoRR_ , vol. abs/2406.08413, 2024. 

- [62] G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and S. Han, “Smoothquant: Accurate and efficient post-training quantization for large language models,” in _International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA_ , ser. Proceedings of Machine Learning Research, vol. 202. PMLR, 2023. 

- [63] D. Yang, S. Liu, R. Huang, J. Tian, C. Weng, and Y. Zou, “Hifi-codec: Group-residual vector quantization for high fidelity audio codec,” _CoRR_ , vol. abs/2305.02765, 2023. 

- [64] L. Yu and J. Li, “Stateful large language model serving with pensieve,” _CoRR_ , vol. abs/2312.05516, 2023. 

- [65] Z. Yuan, Y. Shang, Y. Zhou, Z. Dong, C. Xue, B. Wu, Z. Li, Q. Gu, Y. J. Lee, Y. Yan, B. Chen, G. Sun, and K. Keutzer, “Llm inference unveiled: Survey and roofline model insights,” 2024. 

- [66] B. Zhang and R. Sennrich, “Root mean square layer normalization,” in _Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada_ , 2019. 

- [67] H. Zhang, X. Ji, Y. Chen, F. Fu, X. Miao, X. Nie, W. Chen, and B. Cui, “Pqcache: Product quantization-based kvcache for long context llm inference,” _CoRR_ , vol. abs/2407.12820, 2024. 

- [68] S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. T. Diab, X. Li, X. V. Lin, T. Mihaylov, M. Ott, S. Shleifer, K. Shuster, D. Simig, P. S. Koura, A. Sridhar, T. Wang, and L. Zettlemoyer, “OPT: open pre-trained transformer language models,” _CoRR_ , vol. abs/2205.01068, 2022. 

- [69] T. Zhang, J. Yi, Z. Xu, and A. Shrivastava, “KV cache is 1 bit per channel: Efficient large language model inference with coupled quantization,” _CoRR_ , vol. abs/2405.03917, 2024. 

- [70] H. Zhao, W. Cui, Q. Chen, Y. Zhang, Y. Lu, C. Li, J. Leng, and M. Guo, “Tacker: Tensor-cuda core kernel fusion for improving the GPU utilization while ensuring qos,” in _IEEE International Symposium on High-Performance Computer Architecture, HPCA 2022, Seoul, South Korea, April 2-6, 2022_ . IEEE, 2022, pp. 800–813. 

- [71] L. Zheng, Z. Li, H. Zhang, Y. Zhuang, Z. Chen, Y. Huang, Y. Wang, Y. Xu, D. Zhuo, E. P. Xing, J. E. Gonzalez, and I. Stoica, “Alpa: Automating inter- and intra-operator parallelism for distributed deep learning,” in _16th USENIX Symposium on Operating Systems Design and Implementation, OSDI 2022, Carlsbad, CA, USA, July 11-13, 2022_ . USENIX Association, 2022, pp. 559–578. [Online]. Available: https://www.usenix.org/conference/osdi22/presentation/zheng-lianmin 

- [72] L. Zheng, L. Yin, Z. Xie, J. Huang, C. Sun, C. H. Yu, S. Cao, C. Kozyrakis, I. Stoica, J. E. Gonzalez, C. W. Barrett, and Y. Sheng, “Efficiently programming large language models using sglang,” _CoRR_ , vol. abs/2312.07104, 2023. 

- [73] S. Zheng, S. Chen, S. Gao, L. Jia, G. Sun, R. Wang, and Y. Liang, “Tileflow: A framework for modeling fusion dataflow via tree-based analysis,” in _Proceedings of the 56th Annual IEEE/ACM International Symposium on Microarchitecture, MICRO 2023, Toronto, ON, Canada, 28 October 2023 - 1 November 2023_ . ACM, 2023, pp. 1271–1288. 

- [74] S. Zheng, S. Chen, P. Song, R. Chen, X. Li, S. Yan, D. Lin, J. Leng, and Y. Liang, “Chimera: An analytical optimizing framework for effective compute-intensive operators fusion,” in _IEEE International Symposium on High-Performance Computer Architecture, HPCA 2023, Montreal, QC, Canada, February 25 - March 1, 2023_ . IEEE, 2023. 

- [75] Y. Zhou, J. Leng, Y. Song, S. Lu, M. Wang, C. Li, M. Guo, W. Shen, Y. Li, W. Lin, X. Liu, and H. Wu, “ugrapher: High-performance graph operator computation via unified abstraction for graph neural networks,” in _Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, ASPLOS 2023, Vancouver, BC, Canada, March 2529, 2023_ . ACM, 2023, pp. 878–891. 

- [76] Y. Zhou, M. Yang, C. Guo, J. Leng, Y. Liang, Q. Chen, M. Guo, and Y. Zhu, “Characterizing and demystifying the implicit convolution algorithm on commercial matrix-multiplication accelerators,” in _IEEE International Symposium on Workload Characterization, IISWC 2021, Storrs, CT, USA, November 7-9, 2021_ . IEEE, 2021, pp. 214–225. 

1509 

Authorized licensed use limited to: BEIHANG UNIVERSITY. Downloaded on May 14,2026 at 06:14:40 UTC from IEEE Xplore.  Restrictions apply. 

