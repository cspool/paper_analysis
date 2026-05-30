# LASP-2: Rethinking Sequence Parallelism for Linear Attention and Its Hybrid

Weigao Sun $^1$  Disen Lan $^{12}$  Yiran Zhong $^1$  Xiaoye Qu $^1$  Yu Cheng $^3$ 

## **Abstract**

Linear sequence modeling approaches, such as linear attention, provide advantages like linear-time training and constant-memory inference over sequence lengths. However, existing sequence parallelism (SP) methods are either not optimized for the right-product-first feature of linear attention or use a ring-style communication strategy, which results in lower computation parallelism, limits their scalability for longer sequences in distributed systems. In this paper, we introduce LASP-2, a new SP method to enhance both communication and computation parallelism when training linear attention transformer models with very-long input sequences. Compared to previous work LASP, LASP-2 rethinks the minimal communication requirement for SP on linear attention layers, reorganizes the whole communication-computation workflow of LASP. In this way, only one single AllGather collective communication is needed on intermediate memory states, whose sizes are independent of the sequence length, leading to significant improvements of both communication and computation parallelism, as well as their overlap. Additionally, we extend LASP-2 to LASP-2H by applying similar communication redesign to standard attention modules, offering an efficient SP solution for hybrid models that blend linear and standard attention layers. Our evaluation on a Linear-Llama3 model, a variant of Llama3 with linear attention replacing standard attention, demonstrates the effectiveness of LASP-2 and LASP-2H. Specifically, LASP-2 achieves training speed improvements of 15.2% over LASP and 36.6% over Ring Attention, with a sequence length of 2048K across 64 GPUs. The Code is released as a part of: https://github.com/ OpenSparseLLMs/Linear-MoE.

Copyright 2025 by the author(s).

## 1. Introduction

Transformer, originally introduced by Vaswani et al. (Vaswani et al., 2017), has become the backbone of modern models across a wide range of domains, including language, vision, audio, video, graphs, and even time-series data (Achiam et al., 2023; Team, 2023; Qu et al., 2024). Although the Transformer dates back to 2017, its adaptability and robustness have made it indispensable for a variety of tasks. Central to its success is the self-attention mechanism, which is highly effective for sequence modeling, but has quadratic complexity (w.r.t. sequence length), leading to significant computational costs during training. However, the ability to handle long-context sequences is crucial for large model applications, not only for language tasks but also for multi-modal tasks, where sequences naturally tend to be long (Xue et al., 2024). FlashAttention series (Dao et al., 2022; Dao, 2023; Shah et al., 2024) have provided substantial advancements in scaling attention to handle longer sequences by optimizing the CUDA-level computations for better hardware utilization. However, the theoretical complexity of FlashAttention remains quadratic. Moreover, the need to maintain the KV cache presents further difficulties in managing memory as the sequence length extends (Qin et al., 2024c). As a result, long-sequence processing in Transformer models continues to be a complex and resource-intensive problem.

Recently, numerous variations of attention have been proposed, primarily aimed at addressing its quadratic computational and memory complexity, as well as the growing size of the KV cache (Peng et al., 2023; 2024). One promising approach line is linear attention (Katharopoulos et al., 2020), which replaces the exponential kernel in softmax attention with a simpler dot product between key and query vectors. This shift allows linear attention to be structured as a linear recurrent neural network (RNN) with matrixvalued hidden states, thereby eliminating the need for a KV cache. In consequence, it supports constant-memory inference and reduces training complexity from quadratic to linear (Yang et al., 2023). A parallel line of research focuses on State Space Models (SSMs), such as Mamba (Gu & Dao, 2023) and Mamba 2 (Dao & Gu, 2024), which draw upon concepts from control theory. Both linear attention and SSMs share a common recurrent formulation, expressed as  $\mathbf{M}_s = \mathbf{M}_{s-1} + \widehat{\mathbf{M}}_s$ , where  $\widehat{\mathbf{M}}_s$  represents the incre-

<sup>&</sup>lt;sup>1</sup>Shanghai AI Laboratory <sup>2</sup>South China University of Technology <sup>3</sup>The Chinese University of Hong Kong. Work was done during Disen Lan's internship at Shanghai AI Laboratory. Correspondence to: Yu Cheng <chengyu@cse.cuhk.edu.hk>.

mental memory state of the s-th token [\(Yang et al.,](#page-11-3) [2024\)](#page-11-3). However, despite these advantages, they tend to perform poorly on recall-intensive tasks, such as in-context learning (e.g., five-shot MMLU [\(Hendrycks et al.,](#page-9-6) [2020\)](#page-9-6), Phone-book lookup [\(Jelassi et al.,](#page-9-7) [2024\)](#page-9-7), Needle In A Haystack [\(Briakou](#page-9-8) [et al.,](#page-9-8) [2023\)](#page-9-8)) and long-context reasoning. Empirical research [\(Lieber et al.,](#page-9-9) [2024;](#page-9-9) [Ren et al.,](#page-10-6) [2024;](#page-10-6) [Waleffe et al.,](#page-11-4) [2024;](#page-11-4) [Li et al.,](#page-9-10) [2025\)](#page-9-10) has shown that models relying solely on linear sequence modeling struggle to excel in these domains. However, a hybrid architecture combining linear sequence modeling layers with standard transformer layers has been demonstrated to significantly enhance model performance on tasks that are recall-intensive.

Sequence Parallelism (SP) techniques [\(Korthikanti et al.,](#page-9-11) [2022;](#page-9-11) [Jacobs et al.,](#page-9-12) [2023;](#page-9-12) [Liu et al.,](#page-9-13) [2023\)](#page-9-13) are commonly employed to partition long sequences into smaller subsequences, allowing them to be processed across multiple GPUs in parallel. Despite the advantages offered by SP for handling large sequences, current SP methods do not fully exploit the right-product-first feature of linear attention, which can lead to inefficiencies in parallelism and communication. LASP [\(Sun et al.,](#page-10-7) [2024a\)](#page-10-7) (referred to as LASP-1) introduced a SP approach specifically tailored for linear attention, that uses a point-to-point (P2P) communication strategy. In this method, intermediate states are transferred across GPUs in a ring-style pattern within the distributed world. However, although such P2P ring-style communication offers certain benefits, part of its computation has to be executed sequentially, which leads low computation parallelism. In addition, too many small P2P operators make the overlapping of communication and computation difficult.

In this paper, we introduce LASP-2 by rethinking the minimal communication requirement involved in SP of linear attention. Specifically, we innovatively reorganize the whole computation and communication workflow with an optimized execution mechanism. In this way, only a single allgather collective communication is needed in the forward or backward of each iteration. These bring both communication and computation efficiency improvements: 1) the size of intermediate memory state tensor the all-gather operator works on is independent of the sequence length, making the communication burden insignificant in the context of long sequences. The communication parallelism and accessibility to overlap with computation are notably improved. 2) the refactored workflow improves both communication and computation parallelism over multiple devices. Additionally, we separately present LASP-2 with and without masking for autoregressive and bidirectional tasks, respectively, as the presence of a mask matrix significantly impacts the design criterion of LASP-2. To extend LASP-2 to hybrid models with both linear and standard attention layers, we introduce LASP-2H. This extension employs the same allgather-based communication for standard attention layers,

with a similar designing philosophy on linear attention. We conduct experiments with up to sequence length of 2048K to verify the efficiency advantages of LASP-2 and LASP-2H.

Our main contributions can be summarized as follows:

- We rethink the communication design for the current SP on linear attention, reorganize its whole communication & computation workflow with an optimized execution mechanism. This involves using a single AllGather collective communication on intermediate memory states, whose sizes are independent of sequence length. The resulted LASP-2 improves both communication and computation parallelism for SP on linear attention, thus significantly enhances efficiency.
- We extend LASP-2 to LASP-2H, offering an efficient SP solution for hybrid models that blend both linear and standard attention layers, employing an unified all-gather-based communication design.
- We construct a series of Linear-Llama3 models, including both purely linear and hybrid versions. Extensive experimental results on these models with up to a sequence length of 2048K, validate the efficiency improvement and performance of LASP-2 and LASP-2H.

# 2. Preliminary

Notation In this paper, we ensure the consistent use of notations to enhance clarity. Table [1](#page-2-0) provides a complete list of all the symbols utilized throughout, including indices, constants, vectors, and matrices. Vectors and matrices are represented in boldface. For simplicity, we have omitted the dimensions related to batch size and number of heads in tensor shapes.

Linear Attention The term "attention" generally refers to a computation that assigns scores to pairs of positions within a sequence, enabling each element to "attend" to others. The most widely used and significant variant of this mechanism is softmax self-attention, which is central to standard transformer models [\(Vaswani et al.,](#page-11-0) [2017\)](#page-11-0). During training, with an assumption of a single attention head for simplicity, softmax self-attention computes as follows:

$$\mathbf{Q}, \mathbf{K}, \mathbf{V} = \mathbf{X}\mathbf{W}_{Q}, \mathbf{X}\mathbf{W}_{K}, \mathbf{X}\mathbf{W}_{V},$$

$$\mathbf{O} = \text{Softmax}(\mathbf{Q}\mathbf{K}^{\top})\mathbf{V}.$$
(1)

The mechanism of pairwise comparisons (induced by materializing QK<sup>⊤</sup>) leads to the characteristic quadratic training cost of softmax self-attention. Recently, Linear Attention [\(Katharopoulos et al.,](#page-9-3) [2020;](#page-9-3) [Shen et al.,](#page-10-8) [2024;](#page-10-8) [Qin et al.,](#page-10-9) [2024a\)](#page-10-9) has gained attention as a potential alternative to softmax self-attention, with two key distinctions. First, it removes the Softmax(·) operation, incorporating it into a kernel feature map. Second, it leverages the associativity of matrix multiplication to reformulate (QK<sup>⊤</sup>)V = Q(K<sup>⊤</sup>V).

| Indices   |                        | Operations                                                                   |                            |
|-----------|------------------------|------------------------------------------------------------------------------|----------------------------|
| i         | Any indices            | · (or omitted)                                                               | Matrix multiplication      |
| s         | Index of current token | ·                                                                            | Hadamard multiplication    |
| t         | Index of chunk         | Vectors and Matrices                                                         | -                          |
| Constants |                        | $\mathbf{x},\mathbf{o} \in \mathbb{R}^{1 \times d}$                          | Input and output vectors   |
| d         | Hidden dimension       | $\mathbf{q},\mathbf{k},\mathbf{v}\in\mathbb{R}^{1\times d}$                  | Query, key, value vectors  |
| W         | World size             | $\mathbf{X},\mathbf{O} \in \mathbb{R}^{N \times d}$                          | Input and output matrices  |
| N         | Sequence length        | $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$                  | Query, key, value matrices |
| T         | Total number of chunks | $\mathbf{M} \in \mathbb{R}^{d \times d}$                                     | Memory state matrix        |
| C         | Chunk length           | $\mathbf{W}_{O}, \mathbf{W}_{K}, \mathbf{W}_{V} \in \mathbb{R}^{d \times d}$ | Weight matrices            |

<span id="page-2-0"></span>Table 1: Notations. Indices, operations, constants, vectors and matrices used in the paper.

These adjustments reduce both the computation and memory complexity of attention calculation from  $O(N^2d)$  to  $O(Nd^2)$ . This technique is often referred to as the right-product kernel trick because it prioritizes the multiplication on the right side first.

During inference, both softmax self-attention and linear attention handle a single token at each iteration. Given the s-th token  $\mathbf{x}_s \in \mathbb{R}^{1 \times d}$ , softmax self-attention computes requiring the storage of an expanding set of keys  $\{\mathbf{k}_1, \cdots, \mathbf{k}_s\}$  and values  $\{\mathbf{v}_1, \cdots, \mathbf{v}_s\}$  i.e., the "KV cache", which leads to a significant memory burden when dealing with long input sequences. In linear attention, researchers have experimented with using various nonlinear kernels to replace the  $\exp(\cdot)$  function in Eq. 2.

<span id="page-2-1"></span>
$$\mathbf{q}_{s}, \mathbf{k}_{s}, \mathbf{v}_{s} = \mathbf{x}_{s} \mathbf{W}_{Q}, \mathbf{x}_{s} \mathbf{W}_{K}, \mathbf{x}_{s} \mathbf{W}_{V},$$

$$\mathbf{o}_{s} = \frac{\sum_{i=1}^{s} \exp(\mathbf{q}_{s} \mathbf{k}_{i}^{\top}) \mathbf{v}_{i}}{\sum_{i=1}^{s} \exp(\mathbf{q}_{s} \mathbf{k}_{i}^{\top})}.$$
(2)

However, recent studies (Sun et al., 2023; Yang et al., 2023; Qin et al., 2024c) have found that employing a linear kernel (i.e., using the identity function) without a normalizing denominator works effectively in practice. This results in an unnormalized linear attention form as below:

$$\mathbf{o}_s = \sum_{i=1}^s \mathbf{q}_s(\mathbf{k}_i^{\top} \mathbf{v}_i) = \mathbf{q}_s \sum_{i=1}^s (\mathbf{k}_i^{\top} \mathbf{v}_i) = \mathbf{q}_s \mathbf{M}_s, \quad (3)$$

where  $\mathbf{M}_s = \sum_{i=1}^s \mathbf{k}_i^{\top} \mathbf{v}_i$  is the prefix sum of  $\mathbf{k}_i^{\top} \mathbf{v}_i$  from i=1 to s, which is also known as the memory state in linear attention. This reformulation leads to a recurrent structure for linear attention, resembling the behavior of RNNs as

$$\mathbf{M}_s = \mathbf{M}_{s-1} + \mathbf{k}_s^{\mathsf{T}} \mathbf{v}_s, \quad \mathbf{o}_s = \mathbf{q}_s \mathbf{M}_s. \tag{4}$$

### 3. Method

## 3.1. LASP-2 without Masking

SP methods work by dividing long input sequences into several smaller chunks, which are then distributed across multiple computational devices. Each device independently processes the queries, keys, and values for its assigned chunk in parallel. To complete the attention computation for the entire sequence, necessary communication steps are performed to either gather the results from all devices or exchange information between them. LASP (Sun et al., 2024a) was introduced as a sequence parallelism technique designed specifically for the linear attention module.

Let us consider a distributed computing setup where there are W devices, and the input sequence is divided into T chunks, referred to as the sequence parallel size. In the usual case, T is evenly divisible by W, and we often assume W=T. It means each chunk is assigned to a single device, ensuring that every chunk is processed in parallel across the distributed system. This scenario exemplifies pure sequence parallelism. Additionally, in Sec.A.4.1, we will explore cases where  $W \neq T$ , representing a hybrid approach that combines sequence parallelism with data parallelism.

In LASP-2, the input sequence  $\mathbf{X}$  is divided into T smaller chunks, represented as  $[\mathbf{X}_t]_1^T$ , and each chunk is distributed across the devices in the distributed system. For each chunk  $\mathbf{X}_t$ , its corresponding query, key, value, and the linear attention memory state can be computed in parallel across all chunks. This parallel computation is carried out as follows:

$$\mathbf{Q}_{t}, \mathbf{K}_{t}, \mathbf{V}_{t} = \mathbf{X}_{t} \mathbf{W}_{Q}, \mathbf{X}_{t} \mathbf{W}_{K}, \mathbf{X}_{t} \mathbf{W}_{V},$$
$$\mathbf{M}_{t} = \mathbf{K}_{t}^{\mathsf{T}} \mathbf{V}_{t}.$$
 (5)

By performing this concurrent computation for each chunk, LASP-2 efficiently handles long input sequences in a distributed setting. The query  $\mathbf{Q}_t$ , key  $\mathbf{K}_t$ , value  $\mathbf{V}_t$ , and the memory state  $\mathbf{M}_t$  are calculated individually for every chunk of the sequence, ensuring that no single device is overburdened with processing the entire sequence at once. This distributed approach facilitates better memory management and computational efficiency, especially when dealing with extremely long sequences. Thus, LASP-2 leverages the power of sequence partitioning to optimize the calculation of linear attention in a distributed framework.

Notably, in LASP-2, only a single all-gather collective communication operation is required during the forward pass.

### <span id="page-3-0"></span>Algorithm 1 LASP-2 w/o Masking

- 1: Input: input sequence X, distributed world size W, sequence parallel size T=W.
- 2: Distribute  $\mathbf{X} = [\mathbf{X}_t]_1^T$ .
- 3: for chunk  $t \in \{1, \cdots, T\}$  on ranks  $\{1, \cdots, W\}$  in parallel do
- 4: Calculate  $\mathbf{Q}_t = \mathbf{X}_t \mathbf{W}_Q$ ,  $\mathbf{K}_t = \mathbf{X}_t \mathbf{W}_K$ ,  $\mathbf{V}_t = \mathbf{X}_t \mathbf{W}_V$ .
- 5: Compute  $\mathbf{M}_t = \mathbf{K}_t^{\top} \mathbf{V}_t$ .
- 6: Communicate

$$[\mathbf{M}_t]_1^T = \text{AllGather}([\mathbf{M}_t]_1^T).$$

- 7: Compute  $\mathbf{M}_{1:T} = \text{Sum}([\mathbf{M}_t]_1^T)$ .
- 8: Compute  $\mathbf{O}_t = \mathbf{Q}_t \mathbf{M}_{1:T}$ .
- 9: end for
- 10: return  $\mathbf{O} = [\mathbf{O}_t]_1^T$ .

This all-gather operation acts on the memory states  $[\mathbf{M}_t]_1^T$  associated with each sequence chunk, ensuring that every device in the system has access to the complete set of memory states  $[\mathbf{M}_t]_1^T$ . Once the memory states from all chunks have been gathered, they are concurrently accumulated on all devices to compute the memory state corresponding to the entire input sequence. This process is expressed as follows:

$$[\mathbf{M}_t]_1^T = \texttt{AllGather}([\mathbf{M}_t]_1^T), \\ \mathbf{M}_{1:T} = \texttt{Sum}([\mathbf{M}_t]_1^T).$$
 (6)

Finally, the linear attention output corresponding to the local query  $\mathbf{Q}_t$  can be computed as:

$$\mathbf{O}_t = \mathbf{Q}_t \mathbf{M}_{1:T}.$$

Importantly, the accumulation step  $Sum([\mathbf{M}_t]_1^T)$  can be efficiently performed in a recursive manner, by adding each memory state sequentially as  $\mathbf{M}_{1:t-1} + \mathbf{M}_t$ . This eliminates the need to repeatedly calculate the sum of the memory states from earlier chunks, improving the efficiency of the computation. To further optimize performance, we cache the accumulated result  $\mathbf{M}_{1:T}$  in high-bandwidth memory (HBM). This caching strategy speeds up the backward pass by avoiding redundant recalculations of  $\mathbf{M}_{1:T}$ , which is necessary for computing gradients. This approach is akin to the concept of activation checkpointing, where intermediate activations are saved to avoid recomputation.

It is important to point out that each memory state  $\mathbf{M}_t$  has dimensions of  $d \times d$ , which means the communication cost for the all-gather operation is independent of the sequence or chunk length. Instead, the cost scales linearly with the number of devices involved in the SP communication group. For clarity, we provide a summary of the LASP-2 method, without considering the attention mask, in Algorithm 1. During the backward pass, a similar all-gather communication operation on the gradients of memory states  $\mathbf{dM}_t$  is required. The details of this backward pass without masking, can be found in Algorithm 3 in Appendix A.1 for further reference.

### <span id="page-3-2"></span>Algorithm 2 LASP-2 w/ Masking

- 1: **Input:** input sequence X, distributed world size W, sequence parallel size T = W.
- 2: Distribute  $\mathbf{X} = [\mathbf{X}_t]_1^T$ .
- 3: Initialize mask matrix  $\Psi$ , where  $\Psi_{ij} = 1$  if  $i \geq j$  and  $\Psi_{ij} = -\infty$  if i < j.
- 4: for chunk  $t \in \{1, \dots, T\}$  on ranks  $\{1, \dots, W\}$  in parallel do
- 5: Calculate  $\mathbf{Q}_t = \mathbf{X}_t \mathbf{W}_Q$ ,  $\mathbf{K}_t = \mathbf{X}_t \mathbf{W}_K$ ,  $\mathbf{V}_t = \mathbf{X}_t \mathbf{W}_V$ .
- 6: Compute  $\mathbf{M}_t = (\mathbf{K}_t)^{\top} \mathbf{V}_t$ .
- 7: Communicate  $[\mathbf{M}_t]_1^T = \text{AllGather}([\mathbf{M}_t]_1^T)$ .
- 8: Compute  $\mathbf{O}_{\mathrm{t,intra}} = [(\mathbf{Q}_t \mathbf{K}_t^{\top}) \odot \mathbf{\Psi}] \mathbf{V}_t$ .
- 9: Compute prefix sum  $\mathbf{M}_{1:t-1} = \text{PrefixSum}([\mathbf{M}_t]_1^{t-1})$ .
- 10: Compute  $\mathbf{O}_{t, \text{inter}} = \mathbf{Q}_t \mathbf{M}_{1:t-1}$ .
- 11: Compute  $O_t = O_{t,intra} + O_{t,inter}$ .
- 12: **end for**
- <span id="page-3-1"></span>13: return  $\mathbf{O} = [\mathbf{O}_t]_1^T$ .

![](_page_3_Figure_33.jpeg)

Figure 1: Computation Decomposition in LASP-2 with masking. Colored chunks represent inter-chunks.

## 3.2. LASP-2 with Masking

In autoregressive tasks, the mask matrix  $\Psi \in \{-\infty, 1\}^{N \times N}$  is typically a lower triangular matrix, where  $\Psi_{ij} = 1$  for  $i \geq j$  and  $\Psi_{ij} = -\infty$  when i < j. This structure enforces a causal constraint during computation. Specifically, when calculating  $\mathbf{O} = \operatorname{Softmax}(\mathbf{Q}\mathbf{K}^{\top} \odot \Psi)\mathbf{V}$ , it becomes impossible to leverage the associative property of matrix multiplication to reduce the computational complexity from quadratic to linear in a parallel form.

To address this challenge in linear attention with a causal mask, we adopt the approach of computation decomposition, as proposed in earlier work (Yang et al., 2023; Sun et al., 2024a). Figure 1 provides an illustration that highlights the difference between intra-chunk and inter-chunk computations in linear attention. Inter-chunk calculations, which have no dependencies on other chunks across devices, can be treated as if they have no causal mask. As a result, these computations can be parallelized across all devices in the distributed setup. In contrast, intra-chunk calculations account for the influence of previous chunks (1 to (t-1)) on

the t-th chunk. These intra-chunk operations are affected by the mask matrix, and therefore, require specialized handling to respect the causal constraints.

For linear attention computation on intra-chunks, given the query, key, and value matrices Qt, Kt, and V<sup>t</sup> corresponding to the chunk Xt, the output is computed as

$$\mathbf{O}_{t,\text{intra}} = [(\mathbf{Q}_t \mathbf{K}_t^{\top}) \odot \mathbf{\Psi}] \mathbf{V}_t, \tag{7}$$

This formulation adheres to the standard left-product matrix multiplication. Although the computation can be executed in parallel across devices, it retains the quadratic complexity commonly associated with traditional attention mechanisms during training. This limitation arises from the element-wise masking operation (⊙Ψ), which enforces causal constraints within the chunk, preventing the use of optimizations that would reduce the computational cost to linear.

For linear attention computation across inter-chunks, we follow a similar approach as the procedure outlined for LASP-2 without masking. First, the memory states for each chunk are computed concurrently across different devices as M<sup>t</sup> = K<sup>⊤</sup> <sup>t</sup> Vt. These memory states, corresponding to each chunk, are initially distributed across separate devices. To synchronize the results, an AllGather collective communication operation is performed. This step ensures that all devices hold the memory states for all chunks, enabling further parallel processing. Once the memory states have been gathered, we proceed with a concurrent PrefixSum operation across all devices. This operation accumulates the memory states from the 1st chunk up to the (t−1)-th chunk, effectively building the necessary intermediate states. This can be expressed as:

$$\begin{aligned} [\mathbf{M}_t]_1^T &= \texttt{AllGather}([\mathbf{M}_t]_1^T), \\ \mathbf{M}_{1:t-1} &= \texttt{PrefixSum}([\mathbf{M}_t]_1^{t-1}). \end{aligned} \tag{8}$$

The PrefixSum operation can be optimized by implementing it recursively, utilizing cached memory states stored on the HBM. Specifically, the accumulation of memory states is computed as:

$$\mathbf{M}_{1:t-1} = \mathbf{M}_{1:t-2} + \mathbf{M}_{t-1}. \tag{9}$$

By caching M1:t−1, the backward pass computation is facilitated since this cached value is a necessary activation for gradient calculations. This approach not only speeds up the backward pass but also reduces the computational load, as the cached memory state eliminates the need for repeated re-computation.

Following the calculation of the memory states, the outputs corresponding to the inter-chunks and the final output for the t-th token can be derived with ease. The overall output for the t-th token is obtained by summing both the intra-chunk and inter-chunk outputs.

$$\mathbf{O}_{t,\text{inter}} = \mathbf{Q}_t \mathbf{M}_{1:t-1}, \quad \mathbf{O}_t = \mathbf{O}_{t,\text{intra}} + \mathbf{O}_{t,\text{inter}}.$$
 (10)

We provide the complete algorithm for LASP-2 with masking in Algorithm [2,](#page-3-2) and its backward pass in Algorithm [4](#page-12-3) in Appendix [A.1.](#page-12-2) Note that, in Algorithm [2,](#page-3-2) the communication operation in line 7 (in magenta), along with the computation of Ot,intra in line 8 (in cyan), can be overlapped by executing them on separate threads. This concurrent execution helps improve overall efficiency, as it allows for the overlap of communication and computation.

### 3.3. LASP-1 vs LASP-2

LASP-2, as well as its previous version LASP-1, both aim on efficient SP on linear attention. Although, in theory, LASP-1 and LASP-2 share similarity on communicating the KV activation (d × d), whose size is independent of the sequence or chunk length. They have fundamental distinctions where the key differences lie in their communication manners and the computational order reorganization, as elaborated as below:

- LASP-1 utilizes a ring-style P2P communication, which needs to launch many send & receive operators between devices, to sequentially transfer the KV activation one-by-one among the devices. This makes the communication process relatively slow and hard to adequately overlap with intra-chunk computations.
- While LASP-2 uses a single AllGather collective communication operator to exchange KV activation concurrently among all decices. This offers practical advantages: (1) Only one well-optimized collective communication operator needs to be launched, and the exchange of KV activation on all devices can be finished concurrently all at once; (2) the collective communication can be more easily overlapped with computations. Like in LASP-2 with masking, the AllGather communication is able to overlap with the intra-chunk output computations. And, in addition, LASP-2 reorganizes the whole computation order to make the AllGather based communication strategy feasible and efficiency.

We also write down the Algorithms of LASP-1 (with and without masking) in identical mathematical symbols in Appendix [A.2](#page-12-4) for convenience to compare with LASP-2 on their algorithmic differences.

## 3.4. Theoretical Cost Analysis

For better understanding the superiorities of LASP-2, we provide a theoretical cost analysis of both LASP-1 and LASP-2. We consider the pure SP scenario, i.e., the distributed world size is W, and an input sequence with a

<span id="page-5-0"></span>![](_page_5_Figure_1.jpeg)

Figure 2: **Visualization of LASP-2H on Linear Attention and Standard Attention hybrid model.** We exemplify LASP-2H on the hybrid layers of linear attention and standard attention modules with both TP and SP (both have a dimension of 2). The communication operations colored in yellow and green are for TP and SP, respectively. AG/RS: all-gather in forward and reduce-scatter in backward, and vice versa. AG/No: all-gather in forward and no-op in backward, and vice versa. Note that the SP communication operations for linear attention operate on the memory state  $\mathbf{M}_t \in \mathbb{R}^{d \times d}$ , while for standard attention, they operate on states  $\mathbf{K}_t$ ,  $\mathbf{V}_t \in \mathbb{R}^{C \times d}$ .

length of N is partitioned into T=W chunks, thus all devices in this world need to involve into the communication. Below B denotes batch size, H represents number of heads.

Communication traffic in each communication step: LASP-1:  $BHd^2$ , LASP-2:  $BHd^2$ . This is because both LASP-1 and LASP-2 transfer linear attention memory states (not keys and values) among devices. The memory state corresponding to each chunk (located at each device) has a tensor shape of [B, H, d, d]. Thus in each communication step, their communication traffic are both  $BHd^2$ .

For a Linear-Llama3-1B model with B=16, H=16 and d=2048, each memory state will has  $BHd^2\approx 1.07$ B parameters, which takes around 2.14GB memory in FP16. For a Linear-Llama3-8B model with B=16, H=32 and d=4096, each memory state has  $BHd^2\approx 8.59$ B parameters, which takes around 17.18GB memory in FP16.

Number of communication steps in each iteration: LASP-1: 2(W-1), LASP-2: 2. This depends on the different communication manners of these two algorithms. During the forward of an iteration, LASP-2 launches a single all-gather operation to gather all memory states  $\mathbf{M}_t$  to all devices, i.e.,  $[\mathbf{M}_t]_1^T = \texttt{AllGather}([\mathbf{M}_t]_1^T)$ . This collective operation is concurrently executed on all devices. While in backward, another all-gather is performed on the gradients of  $\mathbf{M}_t$ , i.e.,  $[\mathbf{dM}_t]_1^T = \texttt{AllGather}([\mathbf{dM}_t]_1^T)$ . Thus in each iteration, LASP-2 has 2 communication steps. While LASP-1 uses a pair of send & receive operation to sequentially exchange the memory state from one device to another device. During forward, device i sends its memory state to device i+1, and device i+1 receives the memory

state from device i, and so on. Computations of  $\mathbf{O}_{\mathrm{t,inter}}$ ,  $\mathbf{O}_t$  and updates of  $\mathbf{M}_t$  are followed behind each receive operation on that device. Thus in the process of forward, LASP-1 has W-1 communication steps. In the backward, this process is repeated reversely from the last device to device 0. Thus in each iteration, LASP-1 have totally 2(W-1) communication steps.

Given that both LASP-1 and LASP-2 perform a total of I iterations, their communication traffic models can be expressed as follows: LASP-1:  $2(W-1)IBHd^2$  and LASP-2:  $2IBHd^2$ . Ideally, the communication traffic of LASP-2 would be reduced by a factor of W-1 compared to LASP-1. However, the actual communication cost depends on practical factors like communication bandwidth, which is typically faster within nodes and slower across nodes, and communication stability. As a result, the benefits of LASP-2 become more evident in clusters with slower interconnects, and vice versa. It is important to note that this cost model only accounts for communication, excluding computation or data-loading. In practice, communication represents a smaller portion of the total cost, thus the overall training speedup achieved by LASP-2 is less than W-1times. LASP-2 performs best in scenarios involving long sequences, large clusters, slow communication links, and efficient data-loading and computation.

#### 3.5. Hybrid Model Sequence Parallelism

The hybrid model, which combines linear transformer layers with standard transformer layers that utilize softmax self-attention, has been demonstrated to effectively enhance long-context capabilities, particularly in tasks such as re-

call and retrieval. To optimize SP in such hybrid models, we propose an extended version of LASP-2, referred to as LASP-2H. This approach introduces a comprehensive solution by incorporating SP into both the linear attention and standard attention modules. The structure of LASP-2H is illustrated in Fig. [2.](#page-5-0)

On Linear Attention Module. As outlined in Algorithm [1](#page-3-0) and Algorithm [2,](#page-3-2) LASP-2H handles linear attention modules by performing a single all-gather communication operation on the memory state M<sup>t</sup> ∈ R d×d . The communication complexity remains independent of both sequence or chunk length, and only scales linearly with the SP size T, making this method efficient in distributed clusters.

On Standard Attention Module. Context Parallelism (CP) is a SP technique in Megatron-LM that divides network inputs and all activations along the sequence dimension. This approach is specifically tailored for standard softmax attention. While traditional CP implementations in Megatron-LM rely on overlapping communication and computation in a ring-like structure [\(Liu et al.,](#page-9-13) [2023\)](#page-9-13), our LASP-2H adopts a different method, following the best practice in Llama3 [\(Dubey et al.,](#page-9-14) [2024\)](#page-9-14). Instead of the ring-style strategy, LASP-2H employs AllGather-based communication on standard attention, where the K<sup>t</sup> and V<sup>t</sup> tensors are first gathered across devices, after which the attention output is computed locally for the Q<sup>t</sup> tensor chunk. Although the all-gather communication has a higher latency compared to ring-based methods, it provides greater ease and flexibility in handling various types of attention masks, such as document-level masks. This flexibility is particularly beneficial in scenarios where different attention patterns are needed. Additionally, the all-gather latency is minimized because the K<sup>t</sup> and V<sup>t</sup> tensors are significantly smaller than the Q<sup>t</sup> tensor, especially when using Grouped Query Attention (GQA) [\(Ainslie et al.,](#page-9-15) [2023\)](#page-9-15). As a result, the time complexity of computing the attention output far exceeds the complexity of all-gather operation. We present the description of AllGather-based Context Parallelism in Algorithm [7](#page-14-0) in Appendix [A.3.](#page-12-5)

# 4. Experiments

We conducted an empirical evaluation of LASP-2 by applying it to a model based on Llama3 [\(Dubey et al.,](#page-9-14) [2024\)](#page-9-14). We replaced the standard softmax attention with various linear attention modules, including the original basic linear attention [\(Katharopoulos et al.,](#page-9-3) [2020\)](#page-9-3), Lightning Attention [\(Qin](#page-10-11) [et al.,](#page-10-11) [2024b\)](#page-10-11), Retention [\(Sun et al.,](#page-10-10) [2023\)](#page-10-10), Gated Linear Attention (GLA) [\(Yang et al.,](#page-11-2) [2023\)](#page-11-2), Based [\(Arora et al.,](#page-9-16) [2024\)](#page-9-16), and Rebased [\(Aksenov et al.,](#page-9-17) [2024\)](#page-9-17). This modified model, termed Linear-Llama3, comprises 16 (linear transformer) layers, with a total of 1B parameters. Additionally, we created a hybrid model by retaining transformer layers with

standard softmax attention at every fourth layer of Linear-Llama3, forming a 1/4 hybrid architecture. All experiments were conducted on the SlimPajama dataset [\(Soboleva et al.,](#page-10-12) [2023\)](#page-10-12), utilizing the Llama3 tokenizer [\(Dubey et al.,](#page-9-14) [2024\)](#page-9-14). The full dataset contains 627B tokens, but for our experiments, we used a 50B tokens subset derived from the first chunk of the training split. The experiments were performed using GPT-style autoregressive language modeling tasks with attention masks, as this setup mirrors many practical scenarios where such tasks are commonly applied. Note that the primary focus of these experiments is to assess the training efficiency of LASP-2 when handling very-long input sequences. Training a large language model with optimal long-context capabilities falls outside the scope of this study.

Besides the following results, we have provided more additional experiment results in Appendix [A.5.](#page-14-1)

### 4.1. Experimental Setup

Hardware and Software. Our experiments were conducted on a configuration of up to 16 DGX-A100 servers, each equipped with 8 A100 GPUs. The GPUs are connected through NVSwitch, offering an inter-GPU bandwidth of 600 GBps. The experiments were implemented using PyTorch 2.3.1, with support from CUDA 12.1, cuDNN 8.9.2, and NCCL 2.20.5. The algorithm was developed on top of NVIDIA's Megatron-Core 0.9.0 [\(Shoeybi et al.,](#page-10-13) [2019\)](#page-10-13). We use Triton 2.3.1 [\(Tillet et al.,](#page-11-5) [2019\)](#page-11-5) to accelerate the linear attention computation on GPU, and take FlashAttention-2 [\(Dao,](#page-9-2) [2023\)](#page-9-2) as the standard attention implementation. When implement other SP methods (e.g., Ring Attentoin, Megatron-SP) on linear attention instances for the purpose of comparison, we do not incorporate the right-product kernel trick. We maintain the use of each method's original communication primitives and computational manners as they originally proposed for standard attention.

Hyperparameters. For training the Linear-Llama3 model, we employed a cosine learning rate schedule with a linear warm-up phase [\(Sun et al.,](#page-10-14) [2024b\)](#page-10-14). The minimum learning rate was set to 1e −6 . We applied gradient clipping with a value of 1.0 and weight decay at a rate of 0.1. The Adam optimizer [\(Kingma & Ba,](#page-9-18) [2014\)](#page-9-18) was used, configured with β<sup>1</sup> = 0.9 and β<sup>2</sup> = 0.95 [\(Zhang et al.,](#page-11-6) [2019;](#page-11-6) [Zhou et al.,](#page-11-7) [2020\)](#page-11-7). Additionally, the dropout rate in both attention and hidden layers was set to 0 [\(Tang et al.,](#page-10-15) [2023\)](#page-10-15).

### 4.2. Speed

To assess the speed performance of our proposed LASP-2, we conducted a comparison against existing SP methods, including Megatron-SP [\(Korthikanti et al.,](#page-9-11) [2022\)](#page-9-11), Ring Attention [\(Liu et al.,](#page-9-13) [2023\)](#page-9-13), and LASP-1 [\(Sun et al.,](#page-10-7) [2024a\)](#page-10-7). As depicted in Fig. [3,](#page-7-0) LASP-2 demonstrated superior throughput, particularly when sequence lengths exceeded 64K. This

<span id="page-7-2"></span>Table 2: **Convergence Performance Results.** All experiments used 8 A100 GPUs, sequence length of 16K, and batch size of 8, trained on 50B tokens from the SlimPajama corpus.

| Model         | SP Method      | Attention Module                                  | Pure Model |         | 1/4 Hybrid Model |       |
|---------------|----------------|---------------------------------------------------|------------|---------|------------------|-------|
|               |                |                                                   | Thpt       | Loss    | Thpt             | Loss  |
| Llama3        | Ring Attention | Standard Attention                                | 16549.5    | 2.759   | \                | \     |
|               | LASP-2(H)      | Basic Linear Attention                            | 17834.3    | 2.892   | 17394.7          | 2.824 |
|               |                | Lightning Attention                               | 17926.1    | 2.862   | 17384.2          | 2.758 |
| Linear-Llama3 |                | Retention                                         | 17859.6    | 2.867   | 17352.5          | 2.759 |
| Linear-Liamas | LA3P-2(П)      | ASP-2(H) GLA 17785.3 2.845<br>Based 17946.1 2.754 | 2.845      | 17273.2 | 2.754            |       |
|               |                |                                                   | 17462.5    | 2.751   |                  |       |
|               |                | Rebased                                           | 17896.2    | 2.845   | 17284.5          | 2.787 |

<span id="page-7-0"></span>![](_page_7_Figure_3.jpeg)

Figure 3: **Speed Comparison (tokens/s).** Experiments were carried out on a pure Linear-Llama3-1B model, utilizing the basic linear attention module. A total of 64 A100 GPUs were employed, and the SP size T was also set to 64. To accommodate very-long sequence lengths, such as 2048K, the batch size was kept fixed at 1 throughout this experiment.

performance advantage became increasingly prominent as sequence lengths grew longer. Specifically, at a sequence length of 512K, LASP-2 outperformed Ring Attention by 17.8% and surpassed LASP-1 by 7.3%. This advantage became even more pronounced at a sequence length of 2048K, where LASP-2 achieved throughput gains of 36.6% over Ring Attention and 15.2% over LASP-1.

## 4.3. Scalability

We assessed the scalability of LASP-2 in terms of both GPU memory usage and throughput by adjusting the sequence length and the number of GPUs. The results were displayed in Figure 4. LASP-2 demonstrated the ability to scale linearly with the input sequence length by increasing the number of GPUs. For instance, while maintaining the same memory cost per GPU, using 8 GPUs allowed training on sequences up to 128K in length, whereas 128 GPUs (16  $\times$  8 GPUs) enabled training on sequences as long as 2048K (16  $\times$  128K). Additionally, we observed that increasing both sequence length and device numbers results in higher throughput, indicating improved communication efficiency and linear scalability. More detailed quantitative scalability outcomes are provided in Table 6 in Appendix A.5.

<span id="page-7-1"></span>![](_page_7_Figure_8.jpeg)

![](_page_7_Figure_9.jpeg)

Figure 4: **Scalability Results.** Experiments were conducted on a pure Linear-Llama3-1B model using the Basic Linear Attention module. SP size T was always equal to number of GPUs. Batch size was fixed as 1 to accommodate very-long sequence lengths, e.g., 2048K. The sign " $\times$ " with a dotted line represented occurring an Out of Memory (OOM).

#### 4.4. Convergence Performance

We conducted additional experiments to assess the pretraining convergence performance of LASP-2 on Llama-3 with various attention modules, including standard softmax attention, basic linear attention, Lightning Attention, Retention, GLA, Based, Rebased, and their 1/4 hybrid models. All experiments were performed on the SlimPajama corpus (Soboleva et al., 2023), using 50B tokens, a sequence length of 16K, and a global batch size of 8, using 8 A100 GPUs. The results, as shown in Table 2, indicated that for pure Linear-Llama3 models with different linear attention modules, LASP-2 achieved comparable, though slightly higher, loss values while maintaining superior throughput. On the 1/4 hybrid Linear-Llama3 model, the loss results were generally better than those of the pure linear models, with Lightning Attention, Retention, and GLA even attaining equivalent or lower loss values compared to the baseline. The Based attention module shows strong throughput and

loss performance, since its original design uses a mix of (Taylor) linear attention and sliding window attention. The 1/4 hybrid model striked a balance between throughput and convergence performance, performing competitively when compared to both the baseline and its pure linear version.

### 4.5. Related Work

## 4.5.1. LINEAR SEQUENCE MODELING

Linear Attention. Vanilla linear attention [\(Katharopoulos](#page-9-3) [et al.,](#page-9-3) [2020\)](#page-9-3) introduces the use of kernel methods as a replacement for the Softmax attention [\(Vaswani et al.,](#page-11-0) [2017\)](#page-11-0), thereby reducing the computational complexity to linear in sequence length. Following this, several variants of linear attention have been proposed. TransNormerLLM [\(Qin](#page-10-16) [et al.,](#page-10-16) [2023b;](#page-10-16)[a\)](#page-10-17) proposes Lightning Attention, a refined linear attention mechanism that accelerates processing by optimizing IO interactions. Lightning Attention-2 [\(Qin et al.,](#page-10-11) [2024b\)](#page-10-11) further realizes the theoretical advantages of linear attention by separately handling inter- and intra-block computations. RetNet [\(Sun et al.,](#page-10-10) [2023\)](#page-10-10) introduces a retention mechanism that combines recurrence with attention, benefiting from both parallel training and linear inference. Gated Linear Attention (GLA) [\(Yang et al.,](#page-11-2) [2023\)](#page-11-2) incorporates a data-independent gating mechanism into the linear attention framework, and presents an efficient algorithm for training. DeltaNet [\(Schlag et al.,](#page-10-18) [2021\)](#page-10-18) and its parallelized version [\(Yang et al.,](#page-11-3) [2024\)](#page-11-3) use a delta rule-like update to enhance linear attention performance in long-context scenarios. Finally, Gated Slot Attention (GSA) [\(Zhang et al.,](#page-11-8) [2024\)](#page-11-8), inspired by GLA, introduces a gated linear attention mechanism with bounded-memory slot control to further improve efficiency.

State Space Modeling. The SSM serves as a powerful framework for representing the behavior of sequences within dynamic systems, and it has shown considerable promise in the realm of linear sequence modeling. Mamba [\(Gu &](#page-9-4) [Dao,](#page-9-4) [2023\)](#page-9-4) incorporates a mechanism for selecting states, thereby facilitating the scaling of linear sequence lengths. This architecture has been further enhanced in Mamba-2 [\(Dao & Gu,](#page-9-5) [2024\)](#page-9-5), where the introduction of the state space duality (SSD) framework optimizes its performance.

Linear RNN. Traditional RNNs face significant challenges in handling long-context sequence modeling, primarily due to their inherent sequence dependency during training, which prevents them from fully capitalizing on scaling laws [\(Sun et al.,](#page-10-10) [2023\)](#page-10-10). To address these limitations, RWKV [\(Peng et al.,](#page-10-4) [2023;](#page-10-4) [2024\)](#page-10-5) was introduced as a linear RNN-based large language model that aims to efficiently manage long-term dependencies. Additionally, HGRN [\(Qin](#page-10-19) [et al.,](#page-10-19) [2024e\)](#page-10-19) highlights the critical role of data-dependent decay mechanisms in enhancing linear RNN performance, demonstrating how adjustments to decay parameters can improve learning in long-context tasks. An enhanced version, HGRN2 [\(Qin et al.,](#page-10-20) [2024d\)](#page-10-20), expands on this approach by incorporating a state expansion mechanism that utilizes outer product operations, which allows for greater scalability and improved modeling capabilities over extended sequences. Both RWKV and HGRN series seek to overcome weaknesses of RNNs for efficient long-sequence modeling.

## 4.5.2. SEQUENCE PARALLELISM

SP [\(Li et al.,](#page-9-19) [2022\)](#page-9-19) is a distributed technology designed for training language models more efficiently, which is implemented by dividing a long sequence into multiple shorter subsequences and processing these subsequences in parallel on multiple computing devices. Existing SP methods [\(Kor](#page-9-11)[thikanti et al.,](#page-9-11) [2022;](#page-9-11) [Jacobs et al.,](#page-9-12) [2023\)](#page-9-12) whose parallelism degree cannot exceed the number of attention heads, which limits their scalability. Ring Attention [\(Liu et al.,](#page-9-13) [2023\)](#page-9-13) is proposed to address high memory cost in long sequence modeling by distributing subsequences across different devices and overlapping the communication of KV blocks. LASP [\(Sun et al.,](#page-10-7) [2024a\)](#page-10-7) proposes a new linear attentiontailored SP strategy based on GPU friendly implementation by utilizing a P2P ring-style communication strategy, but still lacks of optimizations for hybrid model architecture.

# 5. Conclusion

This paper presents LASP-2, a new SP method that addresses the inefficiencies of existing SP approaches for linear sequence modeling. By redesigning the whole algorithm workflow and leveraging a single all-gather communication strategy, LASP-2 significantly enhances both the communication and computation parallelism, and enables easier communication-computation overlapping, comparing with preceding work LASP-1. Our results demonstrate that LASP-2 offers significant improvements in speed and scalability, especially in the context of very-long sequence length. Furthermore, the extension to LASP-2H enables efficient SP in hybrid models that integrate both linear and standard attention modules, both utilize an unified all-gatherbased communication primitive. Experimental evaluations on the Linear-Llama3 models validate these advancements, with LASP-2 outperforming previous methods like LASP-1 and Ring Attention by substantial margins, particularly at extreme sequence lengths. These findings confirm the practical utility of LASP-2 for large-scale distributed systems, making it a promising approach for future applications in long-sequence linear transformer models.

# Impact Statement

This work represents a notable advancement in artificial intelligence and machine learning, particularly in improving the efficiency and scalability of linear attention-based models. LASP-2 enables the processing of much longer sequences compared to existing methods while significantly accelerating computation, making it highly beneficial for tasks like natural language understanding, genomic sequence analysis, and time-series forecasting. However, the enhanced capabilities and efficiency introduced by LASP-2 also raise ethical and societal considerations, such as the potential for misuse in generating persuasive but misleading content or in surveillance applications. Nevertheless, the contributions of LASP-2 to reducing computational overhead and energy consumption in training large models may also bring positive environmental impacts.

# References

- <span id="page-9-0"></span>Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. GPT-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023.
- <span id="page-9-15"></span>Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., and Sanghai, S. GQA: Training generalized multi-query transformer models from multi-head checkpoints. *arXiv preprint arXiv:2305.13245*, 2023.
- <span id="page-9-17"></span>Aksenov, Y., Balagansky, N., Vaina, S. M. L. C., Shaposhnikov, B., Gorbatovski, A., and Gavrilov, D. Linear transformers with learnable kernel functions are better incontext models. *arXiv preprint arXiv:2402.10644*, 2024.
- <span id="page-9-16"></span>Arora, S., Eyuboglu, S., Zhang, M., Timalsina, A., Alberti, S., Dylan Zinsley, J. Z., Rudra, A., and Ré, C. Simple linear attention language models balance the recallthroughput tradeoff. *arXiv preprint arXiv:2402.18668*, 2024.
- <span id="page-9-8"></span>Briakou, E., Cherry, C., and Foster, G. Searching for needles in a haystack: On the role of incidental bilingualism in palm's translation capability. *arXiv preprint arXiv:2305.10266*, 2023.
- <span id="page-9-2"></span>Dao, T. Flashattention-2: Faster attention with better parallelism and work partitioning. *arXiv preprint arXiv:2307.08691*, 2023.
- <span id="page-9-5"></span>Dao, T. and Gu, A. Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. *arXiv preprint arXiv:2405.21060*, 2024.
- <span id="page-9-1"></span>Dao, T., Fu, D., Ermon, S., Rudra, A., and Ré, C. Flashattention: Fast and memory-efficient exact attention with io-awareness. *Advances in Neural Information Processing Systems*, 35:16344–16359, 2022.

- <span id="page-9-20"></span>Ding, H., Wang, Z., Paolini, G., Kumar, V., Deoras, A., Roth, D., and Soatto, S. Fewer truncations improve language modeling. *arXiv preprint arXiv:2404.10830*, 2024.
- <span id="page-9-14"></span>Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al. The Llama 3 herd of models. *arXiv preprint arXiv:2407.21783*, 2024.
- <span id="page-9-4"></span>Gu, A. and Dao, T. Mamba: Linear-time sequence modeling with selective state spaces. *arXiv preprint arXiv:2312.00752*, 2023.
- <span id="page-9-6"></span>Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. *arXiv preprint arXiv:2009.03300*, 2020.
- <span id="page-9-12"></span>Jacobs, S. A., Tanaka, M., Zhang, C., Zhang, M., Song, S. L., Rajbhandari, S., and He, Y. Deepspeed Ulysses: System optimizations for enabling training of extreme long sequence transformer models, 2023.
- <span id="page-9-7"></span>Jelassi, S., Brandfonbrener, D., Kakade, S. M., and Malach, E. Repeat after me: Transformers are better than state space models at copying. *arXiv preprint arXiv:2402.01032*, 2024.
- <span id="page-9-3"></span>Katharopoulos, A., Vyas, A., Pappas, N., and Fleuret, F. Transformers are RNNs: Fast autoregressive transformers with linear attention. In *International Conference on Machine Learning*, pp. 5156–5165. PMLR, 2020.
- <span id="page-9-18"></span>Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*, 2014.
- <span id="page-9-11"></span>Korthikanti, V., Casper, J., Lym, S., McAfee, L., Andersch, M., Shoeybi, M., and Catanzaro, B. Reducing activation recomputation in large transformer models, 2022.
- <span id="page-9-10"></span>Li, A., Gong, B., Yang, B., Shan, B., Liu, C., Zhu, C., Zhang, C., Guo, C., Chen, D., Li, D., et al. Minimax-01: Scaling foundation models with lightning attention. *arXiv preprint arXiv:2501.08313*, 2025.
- <span id="page-9-19"></span>Li, S., Xue, F., Baranwal, C., Li, Y., and You, Y. Sequence parallelism: Long sequence training from system perspective, 2022.
- <span id="page-9-9"></span>Lieber, O., Lenz, B., Bata, H., Cohen, G., Osin, J., Dalmedigos, I., Safahi, E., Meirom, S., Belinkov, Y., Shalev-Shwartz, S., et al. Jamba: A hybrid transformer-mamba language model. *arXiv preprint arXiv:2403.19887*, 2024.
- <span id="page-9-13"></span>Liu, H., Zaharia, M., and Abbeel, P. Ring attention with blockwise transformers for near-infinite context, 2023.

- <span id="page-10-4"></span>Peng, B., Alcaide, E., Anthony, Q., Albalak, A., Arcadinho, S., Biderman, S., Cao, H., Cheng, X., Chung, M., Derczynski, L., Du, X., Grella, M., Gv, K., He, X., Hou, H., Kazienko, P., Kocon, J., Kong, J., Koptyra, B., Lau, H., Lin, J., Mantri, K. S. I., Mom, F., Saito, A., Song, G., Tang, X., Wind, J., Wo´zniak, S., Zhang, Z., Zhou, Q., Zhu, J., and Zhu, R.-J. RWKV: Reinventing RNNs for the transformer era. In Bouamor, H., Pino, J., and Bali, K. (eds.), *Findings of the Association for Computational Linguistics: EMNLP 2023*, pp. 14048–14077, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp. 936. URL [https://aclanthology.org/2023.](https://aclanthology.org/2023.findings-emnlp.936) [findings-emnlp.936](https://aclanthology.org/2023.findings-emnlp.936).
- <span id="page-10-5"></span>Peng, B., Goldstein, D., Anthony, Q., Albalak, A., Alcaide, E., Biderman, S., Cheah, E., Ferdinan, T., Hou, H., Kazienko, P., et al. Eagle and Finch: RWKV with matrixvalued states and dynamic recurrence. *arXiv preprint arXiv:2404.05892*, 2024.
- <span id="page-10-22"></span>Pouransari, H., Li, C.-L., Chang, J.-H. R., Vasu, P. K. A., Koc, C., Shankar, V., and Tuzel, O. Dataset decomposition: Faster llm training with variable sequence length curriculum. *arXiv preprint arXiv:2405.13226*, 2024.
- <span id="page-10-17"></span>Qin, Z., Li, D., Sun, W., Sun, W., Shen, X., Han, X., Wei, Y., Lv, B., Luo, X., Qiao, Y., et al. TransNormerLLM: A faster and better large language model with improved transnormer. 2023a.
- <span id="page-10-16"></span>Qin, Z., Li, D., Sun, W., Sun, W., Shen, X., Han, X., Wei, Y., Lv, B., Yuan, F., Luo, X., et al. Scaling transnormer to 175 billion parameters. *arXiv preprint arXiv:2307.14995*, 2023b.
- <span id="page-10-9"></span>Qin, Z., Shen, X., Li, D., Sun, W., Birchfield, S., Hartley, R., and Zhong, Y. Unlocking the secrets of linear complexity sequence model from a unified perspective. *arXiv preprint arXiv:2405.17383*, 2024a.
- <span id="page-10-11"></span>Qin, Z., Sun, W., Li, D., Shen, X., Sun, W., and Zhong, Y. Lightning Attention-2: A free lunch for handling unlimited sequence lengths in large language models. *arXiv preprint arXiv:2401.04658*, 2024b.
- <span id="page-10-3"></span>Qin, Z., Sun, W., Li, D., Shen, X., Sun, W., and Zhong, Y. Various lengths, constant speed: Efficient language modeling with lightning attention. *arXiv preprint arXiv:2405.17381*, 2024c.
- <span id="page-10-20"></span>Qin, Z., Yang, S., Sun, W., Shen, X., Li, D., Sun, W., and Zhong, Y. HGRN2: Gated linear rnns with state expansion. *arXiv preprint arXiv:2404.07904*, 2024d.
- <span id="page-10-19"></span>Qin, Z., Yang, S., and Zhong, Y. Hierarchically gated recurrent neural network for sequence modeling. *Advances in Neural Information Processing Systems*, 36, 2024e.

- <span id="page-10-1"></span>Qu, X., Dong, D., Hu, X., Zhu, T., Sun, W., and Cheng, Y. LLaMA-MoE v2: Exploring sparsity of llama from perspective of mixture-of-experts with post-training. *arXiv preprint arXiv:2411.15708*, 2024.
- <span id="page-10-21"></span>Rajbhandari, S., Rasley, J., Ruwase, O., and He, Y. Zero: Memory optimizations toward training trillion parameter models, 2020.
- <span id="page-10-6"></span>Ren, L., Liu, Y., Lu, Y., Shen, Y., Liang, C., and Chen, W. Samba: Simple hybrid state space models for efficient unlimited context language modeling. *arXiv preprint arXiv:2406.07522*, 2024.
- <span id="page-10-18"></span>Schlag, I., Irie, K., and Schmidhuber, J. Linear transformers are secretly fast weight programmers. In *International Conference on Machine Learning*, 2021.
- <span id="page-10-2"></span>Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., and Dao, T. Flashattention-3: Fast and accurate attention with asynchrony and low-precision. *arXiv preprint arXiv:2407.08608*, 2024.
- <span id="page-10-8"></span>Shen, X., Li, D., Leng, R., Qin, Z., Sun, W., and Zhong, Y. Scaling laws for linear complexity language models. *arXiv preprint arXiv:2406.16690*, 2024.
- <span id="page-10-13"></span>Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., and Catanzaro, B. Megatron-LM: Training multibillion parameter language models using model parallelism. *arXiv preprint arXiv:1909.08053*, 2019.
- <span id="page-10-12"></span>Soboleva, D., Al-Khateeb, F., Myers, R., Steeves, J. R., Hestness, J., and Dey, N. SlimPajama: A 627B token cleaned and deduplicated version of RedPajama, 2023. URL [https://huggingface.co/](https://huggingface.co/datasets/cerebras/SlimPajama-627B) [datasets/cerebras/SlimPajama-627B](https://huggingface.co/datasets/cerebras/SlimPajama-627B).
- <span id="page-10-7"></span>Sun, W., Qin, Z., Li, D., Shen, X., Qiao, Y., and Zhong, Y. Linear attention sequence parallelism. *arXiv preprint arXiv:2404.02882*, 2024a.
- <span id="page-10-14"></span>Sun, W., Qin, Z., Sun, W., Li, S., Li, D., Shen, X., Qiao, Y., and Zhong, Y. CO2: Efficient distributed training with full communication-computation overlap. *arXiv preprint arXiv:2401.16265*, 2024b.
- <span id="page-10-10"></span>Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., Xue, J., Wang, J., and Wei, F. Retentive network: A successor to transformer for large language models. *arXiv preprint arXiv:2307.08621*, 2023.
- <span id="page-10-15"></span>Tang, X., Sun, W., Hu, S., Sun, Y., and Guo, Y. MS-Net: A multi-path sparse model for motion prediction in multiscenes. *IEEE Robotics and Automation Letters*, 2023.
- <span id="page-10-0"></span>Team, I. InternLM: A multilingual language model with progressively enhanced capabilities, 2023.

- <span id="page-11-5"></span>Tillet, P., Kung, H.-T., and Cox, D. D. Triton: an intermediate language and compiler for tiled neural network computations. *Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages*, 2019.
- <span id="page-11-0"></span>Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. *Advances in neural information processing systems*, 30, 2017.
- <span id="page-11-4"></span>Waleffe, R., Byeon, W., Riach, D., Norick, B., Korthikanti, V., Dao, T., Gu, A., Hatamizadeh, A., Singh, S., Narayanan, D., et al. An empirical study of mambabased language models. *arXiv preprint arXiv:2406.07887*, 2024.
- <span id="page-11-1"></span>Xue, F., Chen, Y., Li, D., Hu, Q., Zhu, L., Li, X., Fang, Y., Tang, H., Yang, S., Liu, Z., et al. LongVILA: Scaling long-context visual language models for long videos. *arXiv preprint arXiv:2408.10188*, 2024.
- <span id="page-11-2"></span>Yang, S., Wang, B., Shen, Y., Panda, R., and Kim, Y. Gated linear attention transformers with hardware-efficient training. *arXiv preprint arXiv:2312.06635*, 2023.
- <span id="page-11-3"></span>Yang, S., Wang, B., Zhang, Y., Shen, Y., and Kim, Y. Parallelizing linear transformers with the delta rule over sequence length. *arXiv preprint arXiv:2406.06484*, 2024.
- <span id="page-11-10"></span>Zeng, J., Li, M., Wu, Z., Liu, J., Liu, Y., Yu, D., and Ma, Y. Boosting distributed training performance of the unpadded bert model. *arXiv preprint arXiv:2208.08124*, 2022.
- <span id="page-11-11"></span>Zhai, Y., Jiang, C., Wang, L., Jia, X., Zhang, S., Chen, Z., Liu, X., and Zhu, Y. ByteTransformer: A highperformance transformer boosted for variable-length inputs. In *2023 IEEE International Parallel and Distributed Processing Symposium (IPDPS)*, pp. 344–355. IEEE, 2023.
- <span id="page-11-6"></span>Zhang, H.-T., Sun, W., Li, Y., Fu, D., and Yuan, Y. A fast optimal power flow algorithm using powerball method. *IEEE Transactions on Industrial Informatics*, 16(11): 6993–7003, 2019.
- <span id="page-11-8"></span>Zhang, Y., Yang, S., Zhu, R., Zhang, Y., Cui, L., Wang, Y., Wang, B., Freda Shi, Bailin Wang, W. B., Zhou, P., and Fu, G. Gated slot attention for efficient linear-time sequence modeling. *arXiv preprint arXiv:2409.07146*, 2024.
- <span id="page-11-9"></span>Zhao, Y., Gu, A., Varma, R., Luo, L., Huang, C.-C., Xu, M., Wright, L., Shojanazeri, H., Ott, M., Shleifer, S., et al. Pytorch FSDP: experiences on scaling fully sharded data parallel. *arXiv preprint arXiv:2304.11277*, 2023.

<span id="page-11-7"></span>Zhou, B., Liu, J., Sun, W., Chen, R., Tomlin, C. J., and Yuan, Y. pbSGD: Powered stochastic gradient descent methods for accelerated non-convex optimization. In *IJCAI*, pp. 3258–3266, 2020.

## A. Appendix

### <span id="page-12-2"></span>A.1. LASP-2 Algorithms (Backward Pass)

See Algorithm 3 and Algorithm 4.

## <span id="page-12-1"></span>Algorithm 3 LASP-2 w/o Masking (Backward Pass)

```
1: Input: distributed world size W, sequence parallel size T=W, \mathbf{Q}_t, \mathbf{K}_t, \mathbf{V}_t, \mathbf{O}_t, \mathbf{dO}_t \in \mathbb{R}^{C \times d} for chunk t \in \{1, \cdots, T\}.
 2: for chunk t \in \{1, \dots, T\} on ranks \{1, \dots, W\} in parallel do
 3:
          Compute d\mathbf{M}_t = (\mathbf{Q}_t)^{\top} d\mathbf{O}_t.
 4:
          Communicate [\mathbf{dM}]_1^T = \text{AllGather}([\mathbf{dM}]_1^T).
 5:
          Compute d\mathbf{M}_{1:T} = \text{Sum}([d\mathbf{M}]_{t+1}^T).
          Compute d\mathbf{Q}_{t} = d\mathbf{O}_{t}\mathbf{M}_{1:T}^{\top}.
 6:
 7:
          Compute \mathbf{dK}_{t} = \mathbf{V}_{t} \mathbf{dM}_{1:T}^{\top}.
          Compute dV_t = K_t dM_{1:T}.
 8:
 9: end for
10: return \mathbf{dQ} = [\mathbf{dQ}_t]_1^T, \mathbf{dK} = [\mathbf{dK}_t]_1^T, \mathbf{dV} = [\mathbf{dV}_t]_1^T.
```

### <span id="page-12-3"></span>Algorithm 4 LASP-2 w/ Masking (Backward Pass)

```
1: Input: distributed world size W, sequence parallel size T = W, \mathbf{Q}_t, \mathbf{K}_t, \mathbf{V}_t, \mathbf{O}_t, \mathbf{dO}_t \in \mathbb{R}^{C \times d} for chunk t \in \{1, \dots, T\}.
 2: for chunk t \in \{1, \dots, T\} on ranks \{1, \dots, W\} in parallel do
           Compute d\mathbf{M}_t = (\mathbf{Q}_t)^{\top} d\mathbf{O}_t.
 3:
 4:
           Communicate [\mathbf{dM}]_1^T = \text{AllGather}([\mathbf{dM}]_1^T).
           Compute d\mathbf{Q}_{t,\text{intra}} = [(d\mathbf{O}_t \mathbf{V}_t^{\top}) \odot \boldsymbol{\Psi}] \mathbf{K}_t.
 5:
           Compute \mathbf{dK}_{t,\text{intra}} = [(\mathbf{dO}_t \mathbf{V}_t^{\top}) \odot \mathbf{\Psi}]^{\top} \mathbf{Q}_t.
 6:
 7:
           Compute d\mathbf{V}_{t,\text{intra}} = [(\mathbf{Q}_t \mathbf{K}_t^{\top}) \odot \mathbf{\Psi}]^{\top} d\mathbf{O}_t.
           Compute \mathbf{dQ}_{t.inter} = \mathbf{dO}_t \mathbf{M}_{1:t-1}^{\top}.
 8:
 9:
           Compute suffix sum d\mathbf{M}_{t+1:T} = \text{SuffixSum}([d\mathbf{M}]_{t+1}^T).
10:
            Compute \mathbf{dK}_{t,\text{inter}} = \mathbf{V}_t \mathbf{dM}_{t+1:T}^{\top}.
11:
            Compute dV_{t,inter} = K_t dM_{t+1:T}.
            Combine intra- and inter-chunk parts of dQ_t, dK_t, dV_t
12:
                                    \mathbf{dQ}_t = \mathbf{dQ}_{t,\mathrm{intra}} + \mathbf{dQ}_{t,\mathrm{inter}}, \quad \mathbf{dK}_t = \mathbf{dK}_{t,\mathrm{intra}} + \mathbf{dK}_{t,\mathrm{inter}}, \quad \mathbf{dV}_t = \mathbf{dV}_{t,\mathrm{intra}} + \mathbf{dV}_{t,\mathrm{inter}}.
13: end for
14: return d\mathbf{Q} = [d\mathbf{Q}_t]_1^T, d\mathbf{K} = [d\mathbf{K}_t]_1^T, d\mathbf{V} = [d\mathbf{V}_t]_1^T.
```

## <span id="page-12-4"></span>A.2. LASP-1 Algorithms

See Algorithm 5 and Algorithm 6.

### <span id="page-12-5"></span>A.3. AllGather-based Context Parallelism

See Algorithm 7.

#### A.4. Compatibility

### <span id="page-12-0"></span>A.4.1. HYBRID PARALLELISM

LASP-2 enables the selection of a sequence parallel size that is smaller and divisible by the distributed world size. This setup splits the input data along both the batch and sequence dimensions, a parallelization strategy known as data-sequence hybrid parallelism. The ZeRO-series optimizers (Rajbhandari et al., 2020) and FSDP (Zhao et al., 2023) offer methods for distributing model states such as optimizer states, gradients, and model parameters across all GPUs in the distributed system. As these techniques are variants of data parallelism, they integrate seamlessly with LASP. Their primary objective of minimizing the memory footprint of model states complements LASP-2's specific focus on reducing activation memory on each GPU, making the training of large-scale models that handle long sequence lengths significantly more manageable.

### <span id="page-13-0"></span>Algorithm 5 LASP-1 w/o Masking

```
1: Input: input sequence X, distributed world size W, sequence parallel size T = W.
 2: Distribute input \mathbf{X} = [\mathbf{X}_t]_1^T.
 3: for chunk t \in \{1, \dots, T\} at rank i \in \{1, \dots, W\} in parallel do
         Compute \mathbf{Q}_t = \mathbf{X}_t \mathbf{W}_Q, \mathbf{K}_t = \mathbf{X}_t \mathbf{W}_K, \mathbf{V}_t = \mathbf{X}_t \mathbf{W}_V.
 4:
         Compute \mathbf{M}_t = \mathbf{K}_t^{\top} \mathbf{V}_t.
 5:
 6: end for
 7: for chunk t \in \{1, \dots, T\} at rank i \in \{1, \dots, W\} sequentially do
         Recv activation \mathbf{M}_{t-1} from rank (i-1). Save \mathbf{M}_{t-1} in memory for backward computation.
 8:
 9:
         Compute \mathbf{O}_t = \mathbf{Q}_t \mathbf{M}_{t-1}.
10:
         Update \mathbf{M}_t = \mathbf{M}_{t-1} + \mathbf{K}_t^{\top} \mathbf{V}_t.
         Send activation M_t to rank (i + 1).
11:
12: end for
13: return O = [\mathbf{O}_t] with t \in \{1, \dots, T\}.
```

### <span id="page-13-1"></span>Algorithm 6 LASP-1 w/ Masking

```
1: Input: input sequence X, distributed world size W, sequence parallel size T = W.
 2: Distribute input \mathbf{X} = [\mathbf{X}_t]_1^T.
 3: Initialize mask matrix \Psi, where \Psi_{ij} = 1 if i \geq j, and \Psi_{ij} = -\infty if i < j.
 4: for chunk t \in \{1, \dots, T\} at rank i \in \{1, \dots, W\} in parallel do
         Compute \mathbf{Q}_t = \mathbf{X}_t \mathbf{W}_Q, \mathbf{K}_t = \mathbf{X}_t \mathbf{W}_K, \mathbf{V}_t = \mathbf{X}_t \mathbf{W}_V.
 5:
         Compute \mathbf{M}_t = (\mathbf{K}_t)^{\top} \mathbf{V}_t.
 6:
         Compute \mathbf{O}_{\mathrm{t,intra}} = [(\mathbf{Q}_t \mathbf{K}_t^{\top}) \odot \mathbf{\Psi}] \mathbf{V}_t.
 7:
 8: end for
 9: for chunk t \in \{1, \dots, T\} at rank i \in \{1, \dots, W\} sequentially do
         Recv activation \mathbf{M}_{t-1} from rank (i-1). Save \mathbf{M}_{t-1} in memory for backward computation.
10:
11:
         Compute \mathbf{O}_{t,\text{inter}} = \mathbf{Q}_t \mathbf{M}_{t-1}.
         Compute O_t = O_{t,intra} + O_{t,inter}.
12:
13:
         Update \mathbf{M}_t = \mathbf{M}_{t-1} + \mathbf{K}_t^{\top} \mathbf{V}_t.
         Send activation M_t to rank (i + 1).
14:
15: end for
16: return O = [\mathbf{O}_t] with t \in \{1, \dots, T\}.
```

LASP-2 also offers support for both tensor parallelism (TP) and pipeline parallelism (PP). In the case of TP, its integration with LASP-2 is straightforward and efficient. Linear attention layers apply TP to break down matrix operations across both intra-chunk and inter-chunk computations. At the same time, the MLP layers are processed as usual under TP, without any modification. When LASP-2 is paired with PP, instead of using traditional micro-batches, it substitutes them with sub-sequences extracted from the mini-batch. One key difference from standard PP is that each device locally and specifically stores the intermediate states,  $\mathbf{M}_t$  during the forward pass and  $\mathbf{dM}_t$  during the backward pass without communicating these states to other devices.

#### A.4.2. VARIABLE LENGTH

During pretraining, the batch typically contains sequences of uniform length. However, when finetuning or during inference, the model might encounter input sequences of varying lengths. A straightforward solution to address this is to right-pad all sequences in a batch to match the length of the longest sequence. Unfortunately, this method can be inefficient, especially when the lengths differ significantly across sequences. For standard transformers, more sophisticated approaches have been developed to handle this challenge. These include techniques like load-balancing across GPUs without padding (Zeng et al., 2022; Zhai et al., 2023) or packing multiple sequences into a single batch and adjusting the attention mask accordingly (Ding et al., 2024; Pouransari et al., 2024). LASP-2 can manage variable sequence lengths efficiently by treating the entire batch as a single long sequence, streamlining the process without requiring padding.

### <span id="page-14-0"></span>Algorithm 7 AllGather-based Context Parallelism

- 1: **Input:** input sequence **X**, distributed world size W, sequence parallel size T = W.
- 2: Distribute  $\mathbf{X} = [\mathbf{X}_t]_1^T$ .
- 3: for chunk  $t \in \{1, \dots, T\}$  on ranks  $\{1, \dots, W\}$  in parallel do
- 4: Calculate  $\mathbf{Q}_t = \mathbf{X}_t \mathbf{W}_Q$ ,  $\mathbf{K}_t = \mathbf{X}_t \mathbf{W}_K$ ,  $\mathbf{V}_t = \mathbf{X}_t \mathbf{W}_V$ .
- 5: Communicate  $[\mathbf{K}_t]_1^T = \text{AllGather}([\mathbf{K}_t]_1^T)$  and  $[\mathbf{V}_t]_1^T = \text{AllGather}([\mathbf{V}_t]_1^T)$ .
- 6: Concatenate  $\mathbf{K} = \text{Concat}([\mathbf{K}_t]_1^T)$  and  $\mathbf{V} = \text{Concat}([\mathbf{V}_t]_1^T)$ .
- 7: Compute  $\mathbf{O}_t = \text{Softmax}(\mathbf{Q}_t \mathbf{K}^\top / \sqrt{d}) \mathbf{V}$ .
- 8: end for
- 9: **return O** =  $[\mathbf{O}_t]_1^T$ .

### <span id="page-14-1"></span>A.5. Additional Experiment Results

#### A.5.1. BIDIRECTIONAL LANGUAGE MODELING TASK

To evaluate on the bidirectional language modeling task, we take RoBERTa as the base model and replace its standard attention modules with Basic Linear Attention, train it on 4 A100 GPUs for 50K iterations with a total input sequence length of 2048. As the results shown in Table 3, LASP-2 with Basic Linear Attention is able to reach an approximate convergence performance with Ring Attention on the standard attention based model.

<span id="page-14-2"></span>Table 3: Convergence Performance on Bidirectional Language Modeling Task. Both training and validation loss values are reported.

| Model                                        | Training Loss | Validation Loss |
|----------------------------------------------|---------------|-----------------|
| RoBERTa Baseline (Ring Attention)            | 1.815         | 1.957           |
| RoBERTa with Basic Linear Attention (LASP-2) | 1.813         | 1.957           |

#### A.5.2. ABLATION STUDY ON HYBRID RATIO

We provide ablation results on the hybrid ratio of hybrid models. Let "L" denotes linear Transformer layers and "N" denotes normal Transformer layers. The hybrid models evaluated here have architectures of: 0 Hybrid: "LLLL LLLL LLLL LLLL"; 1/8 Hybrid: "LLLL LLLL LLLL LLLL"; 1/4 Hybrid: "LLLN LLLN LLLN"; 1/2 Hybrid: "LNLN LNLN LNLN LNLN LNLN LNLN LNLN". Comparing with the Llama3-1B baseline using standard attention, whose loss value is 2.759, it shows that higher hybrid ratios tend to lead better convergence performance, but sometimes, a moderate hybrid ratio may reach a better result.

Table 4: **Ablation Study on Hybrid Ratio in Hybrid Models.** Loss values are reported in the Table. Note that pure linear models use LASP-2, while hybrid models use LASP-2H.

| <b>Linear Sequence Modeling Module</b> | 0 Hybrid (Pure Linear Model) | 1/8 Hybrid | 1/4 Hybrid | 1/2 Hybrid |
|----------------------------------------|------------------------------|------------|------------|------------|
| Basic Linear Attention                 | 2.892                        | 2.826      | 2.824      | 2.775      |
| Lightning Attention                    | 2.848                        | 2.756      | 2.750      | 2.742      |
| Retention                              | 2.855                        | 2.757      | 2.758      | 2.748      |
| GLA                                    | 2.845                        | 2.751      | 2.754      | 2.753      |

## A.5.3. ABLATION STUDY ON VARYING SIZES OF GATHERING

We have conducted ablation study on varying sizes of gathering memory states. Considering a batch size of 1, in the Linear-Llama3-1B model (with 16 heads and hidden dimension of 2048), the tensor shape of each memory state is [1, 16, 2048, 2048]. We use 64 GPUs and a sequence length of 1024K, repeat each test 10 times and report their mean values. We change the split size of gathering memory states and present the LASP-2 throughput results in Table 5. It can be seen that smaller split size (i.e., more number of splits) tends to lead lightly slower throughput. The results show that the utilization of all-gather operation is not the only reason of efficiency enhancement. The communication manner as well as

the computational workflow reorganization plays an important role.

<span id="page-15-0"></span>Table 5: Throughput Results (tokens/sec) on Varying Split Sizes of Gathering. Linear-Llama3-1B model (with 16 heads and hidden dimension of 2048) is used.

| Split Size of Gathering | 2048   | 512    | 128    | 32     |
|-------------------------|--------|--------|--------|--------|
| Number of Splits        | 1      | 4      | 16     | 64     |
| Throughput              | 486183 | 486166 | 486169 | 486158 |

## A.5.4. QUANTITATIVE SCALABILITY RESULTS

See Table [6](#page-16-0) in next page.

<span id="page-16-0"></span>Table 6: Quantitative Scalability Results of LASP-2 on Throughput (tokens/sec) and Memory Usage Per GPU (GB). Experiments are performed on Linear-Llama3-1B, scaling sequence length from 2K to 4096K.

| Sequence Length | Number of GPUs | Throughput | Memory Usage Per GPU |
|-----------------|----------------|------------|----------------------|
|                 | 16             | 1254       | 25.6                 |
|                 | 32             | 1209       | 25.6                 |
| 2K              | 64             | 1285       | 25.6                 |
|                 | 128            | 1205       | 25.6                 |
|                 | 16             | 2478       | 25.6                 |
|                 | 32             | 2446       | 25.6                 |
| 4K              | 64             | 2327       | 25.6                 |
|                 | 128            | 2344       | 25.6                 |
|                 | 16             | 4835       | 25.6                 |
| 8K              | 32             | 4784       | 25.6                 |
|                 | 64             | 4693       | 25.6                 |
|                 | 128            | 4678       | 25.6                 |
|                 | 16             | 9530       | 25.6                 |
| 16K             | 32             | 9494       | 25.6                 |
|                 | 64             | 9305       | 25.6                 |
|                 | 128            | 9313       | 25.6                 |
|                 | 16             | 18105      | 28.7                 |
| 32K             | 32             | 17755      | 25.6                 |
|                 | 64             | 17835      | 25.6                 |
|                 | 128            | 17807      | 25.6                 |
|                 | 16             | 35507      | 33.8                 |
| 64K             | 32             | 34240      | 28.7                 |
|                 | 64             | 34118      | 25.6                 |
|                 | 128            | 33344      | 25.6                 |
|                 | 16             | 68406      | 40.2                 |
| 128K            | 32             | 68545      | 33.8                 |
|                 | 64             | 67344      | 28.7                 |
|                 | 128            | 66811      | 25.6                 |
|                 | 16             | 135635     | 57.8                 |
| 256K            | 32             | 132605     | 40.2                 |
|                 | 64             | 130215     | 33.8                 |
|                 | 128            | 131550     | 28.7                 |
|                 | 16             | OOM        | OOM                  |
| 512K            | 32             | 250586     | 57.8                 |
|                 | 64             | 245353     | 40.2                 |
|                 | 128            | 233442     | 33.8                 |
|                 | 16             | OOM        | OOM                  |
| 1024K           | 32             | OOM        | OOM                  |
|                 | 64             | 442221     | 57.8                 |
|                 | 128            | 416465     | 40.2                 |
|                 | 16             | OOM        | OOM                  |
| 2048K           | 32             | OOM        | OOM                  |
|                 | 64             | OOM        | OOM                  |
|                 | 128            | 769030     | 57.8                 |
|                 | 16             | OOM        | OOM                  |
| 4096K           | 32             | OOM        | OOM                  |
|                 | 64             | OOM        | OOM                  |
|                 | 128            | OOM        | OOM                  |