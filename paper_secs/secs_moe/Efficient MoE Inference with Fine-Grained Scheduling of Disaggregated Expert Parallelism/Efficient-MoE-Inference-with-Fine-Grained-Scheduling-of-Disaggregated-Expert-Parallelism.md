# Efficient MoE Inference with Fine-Grained Scheduling of Disaggregated Expert Parallelism

Xinglin Pan<sup>1</sup>, Shaohuai Shi<sup>2</sup>, Wenxiang Lin<sup>2</sup>, Yuxin Wang<sup>3</sup>, Zhenheng Tang<sup>4</sup>, Wei Wang<sup>4</sup>, Xiaowen Chu<sup>1,4</sup>

<sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou), China
<sup>2</sup>Harbin Institute of Technology, Shenzhen, China
<sup>3</sup>Hong Kong Baptist University, Hong Kong SAR
<sup>4</sup>The Hong Kong University of Science and Technology, Hong Kong SAR

#### **Abstract**

The mixture-of-experts (MoE) architecture is commonly employed in contemporary large language models (LLMs) due to its advantage of scaling model size with a sublinear increase in computational demand. Nevertheless, the inference of MoE models demands substantial memory, making it memory-intensive in attention layers due to the necessity of accessing key-value (KV) caches and in expert layers, utilizing only a limited number of experts. Recent studies attempt to utilize disaggregated expert parallelism (DEP) to distribute attention and experts to two dedicated GPU groups, the attention group (AG) and the expert group (EG), to improve inference efficiency. However, the existing DEP has limited support for modern MoE models with shared experts, and it under-explores task scheduling in both GPU groups, which have complex communication and computation tasks, leading to suboptimal inference performance. To address these issues, we propose FinDEP, a finegrained task scheduling algorithm for DEP with maximal task overlap to improve the inference throughput of MoE models. FinDEP integrates our three proposed key innovations: 1) partitioning intensive computation and communication tasks to multiple smaller tasks in both AG and EG to enable fine-grained task pipelining w/ or w/o shared experts, 2) formulating an optimization problem to the fine-grained task scheduling that should support different task partition granularity and ordering, and 3) developing an efficient solution to the optimization problem which contains a huge solution space to derive the near-optimal task schedule of DEP. Experiments are conducted on four types of GPU systems with two representative MoE backbones, DeepSeek-V2 and Qwen3-MoE. Experimental results show that FinDEP improves inference throughput by up to 1.61× over state-of-the-art methods. Notably, on a 32-GPU system, FinDEP still achieves a significant speedup of up to 1.24x, demonstrating its efficiency at large scales.

#### **CCS** Concepts

 $\bullet$  Computing methodologies  $\to$  Distributed artificial intelligence.

#### Keywords

Mixture-of-Expert, disaggregated expert parallelism, ping-pong parallelism, throughput

#### 1 Introduction

Large language models (LLMs) are scaling rapidly, and so are their computational costs. For instance, models like Falcon [1]

<span id="page-0-1"></span>![](_page_0_Figure_12.jpeg)

Figure 1: A typical structure of an MoE model. MLA refers to Multi-Head Latent Attention [5], while MHA denotes Multi-Head Attention [30]. The "Shared" block indicates one shared expert or several shared experts, which may be optional depending on the MoE configuration.

with 180 billion parameters and Llama-3.1 [27] with 405 billion parameters exemplify this trend. Mixture-of-Experts (MoE) architectures [12, 15, 24] address this challenge by activating only a subset of the model's expert components for each input. This makes it possible to build much larger models without making training or inference more expensive. Recent MoE-based LLMs, such as DeepSeek-V3 [6] and Qwen3-MoE [28], show that this design can create highly capable models that are still fast and cheap to use. As a result, MoE has become a key technique for building future LLMs in a way that balances power and efficiency.

Despite the advantages, running inference on large MoE models remains challenging [2, 6, 13, 34, 37] due to its extensive memory requirement to hold all experts in the MoE layers and key-value (KV) caches in the attention layers. As a result, distributing the MoE model across multiple GPUs has been a common practice [6, 15, 36] for efficient inference through expert parallelism (EP), which assigns experts across GPUs.

Recent research [17, 29, 36] suggests distributing attention layers and expert layers onto distinct GPUs through disaggregated expert parallelism (DEP)<sup>1</sup>, due to the different computational and memory access patterns of attention layers and expert layers. This approach enables the modules to scale independently while optimizing the use of various hardware capabilities. In DEP, a multi-GPU system is divided into two groups: the attention group (AG), responsible for storing all attention layers, and the expert group (EG), which holds all non-shared experts. It is important to mention that in certain MoE models like DeepSeek-V3 [6], shared experts within the MoE layer are often placed in the AG as they need to be processed by all input tokens. The dependency between attention and expert layers is substantial, as each attention layer's output serves as the input

<span id="page-0-0"></span><sup>&</sup>lt;sup>1</sup>Also referred to as Attention-FFN Disaggregation (AFD); we adopt the term DEP following [36].

for the subsequent expert layer, which then outputs to another attention layer as shown in Fig. 1. Consequently, DEP necessitates bidirectional communication: from AG to EG (A2E) and the reverse (E2A). Data dependencies and communication overhead easily lead to the GPU computational resources idle, thereby limiting inference efficiency.

Existing optimizations try to alleviate the GPU idle duration of DEP via 1) overlapping computation and communication tasks to reduce the communication time with the ping-pong pipeline (PP-Pipe) algorithm proposed in MegaScale-Infer [36] or 2) offloading communication tasks to CPU resources to enable overlaps between CPU communications and GPU computations in StepMesh [29]. These techniques enable only coarse-level task scheduling by dividing a mini-batch into several micro-batches. As a result, different tasks from these micro-batches can be executed in a pipeline fashion, but this does not sufficiently hide A2E/E2A communications, leading to suboptimal inference efficiency. Moreover, certain cutting-edge MoE models such as DeepSeek series [4–6] introduce shared experts within the MoE layer, which are required to compute for every input token, similar to the attention layer, leading to increased GPU idle time.

In this paper, we propose FinDEP, a fine-grained task scheduling framework for MoE inference with DEP to address the above two efficiency problems by three key innovations. (1) We partition timeconsuming tasks including computations in EG, communications in A2E and E2A, and computations in AG into smaller tasks by splitting each task's input tensor into several segments (denoted as r). This partitioning of the tensor creates r smaller tasks per original task, allowing for dynamic scheduling aimed at improving the throughput for MoE models, regardless of whether they have shared experts. (2) Intuitively, increasing r allows greater parallelization for enhanced overlapping. However, this also increases the launch overheads associated with executing tasks, such as kernel dispatch on GPUs and communication startup costs. Thus, a balance must be built between the advantages of overlapping and the execution overheads. Consequently, we construct performance models for computation tasks in AG and EG and their A2E/E2A communication tasks. Using these models, we establish an optimization problem to characterize the DEP inference time with fine-grained task scheduling, including task ordering and tensor partition granularity. (3) We develop an efficient algorithm to find the near-optimal solution to the formulated optimization problem with a polynomial time complexity, thus avoiding the very time-consuming brute-force search on the huge solution space.

We conduct extensive experiments on four GPU systems with two representative MoE model backbones, DeepSeek-V2 (with shared experts) and Qwen3-MoE (without shared experts). Experimental results show that our FinDEP achieves speedups of upto 1.61× over the best-configured PPPipe algorithm in MegaScale-Infer. Furthermore, on the 32-GPU system, FinDEP consistently provides a speedup of up to 1.24×. Beyond peak throughput, we also confirm the computational efficiency of our fine-grained task scheduling solver. Our solver is highly efficient, taking less than one second to compute the near-optimal configuration. This minimal overhead enables real-time adaptation to dynamic workloads, which is crucial for maximizing throughput in online serving environments with dynamically varying sequence lengths and batch sizes.

Table 1: Notations.

<span id="page-1-0"></span>

| Name    | Description                                           |
|---------|-------------------------------------------------------|
| P       | # of GPUs in the cluster.                             |
| ag      | Size of attention group (AG).                         |
| eg      | Size of expert group (EG).                            |
| $m_a$   | # of samples per micro-batch per GPU in AG.           |
| $m_e$   | # of tokens per micro-batch per expert.               |
| S       | Sequence length of each sample.                       |
| E       | Total number of global experts.                       |
| T       | Total number of layers.                               |
| M       | Embedding size for each token.                        |
| H       | Hidden size of the feed-forward layer within experts. |
| $top_k$ | # of experts activated per token.                     |
| $r_1$   | Pipeline degree of the AG.                            |
| $r_2$   | Fine-grained pipeline degree of the EG.               |

<span id="page-1-1"></span>![](_page_1_Figure_7.jpeg)

Figure 2: An illustration of DEP. GPUs are partitioned into two groups: AG and EG. AG handles the attention and shared expert computation, while EG handles experts computation.

#### 2 Background and Motivations

This section provides an overview of background concepts, followed by a summary of the motivations for this research. For clarity, Table 1 offers a summary of the frequently used notations throughout the paper.

#### 2.1 MoE Layer

MoE models replace each dense feed-forward network (FFN) in transformers with sparsely activated FFNs (or experts) by the MoE layer, as shown in Fig. 1. Each token is routed to k experts via a gating function: the gate computes routing scores over all experts, applies a softmax function, and selects the top-k experts for each token [31]. The input is partitioned accordingly, with each expert processing only its assigned tokens. Some implementations include a shared expert that processes all tokens [5, 6, 22], while it is optional in some implementations like Qwen3-MoE [28]. The layer output of the MoE layer is the aggregated contributions from selected experts (and the shared expert if present).

### 2.2 Disaggregated Expert Parallelism and Ping-pong Pipeline

Disaggregated Expert Parallelism. Disaggregated Expert Parallelism (DEP) is a novel parallelization strategy specifically tailored for the high-throughput, low-latency inference of large MoE-based models. Its foundational principle is the physical separation and independent allocation of core model components across distinct GPU groups. This partitioning divides the available hardware into two dedicated functional units: the Attention Group (AG) and the Expert Group (EG) as shown in Fig. [2.](#page-1-1) The AG is dedicated to storing and processing the standard components of the Transformer block, including the Self-Attention layers and the Shared Expert (if present), (i.e., components that are densely activated across all tokens). Conversely, the EG houses the entire set of sparse MoE experts, distributed across its constituent devices. This structural disaggregation enables to independently scale the computational resources for each module based on specific memory and computational bottlenecks, a key advantage over monolithic parallel approaches.

A key architectural benefit of DEP is the elimination of intragroup communication overhead. Within the AG, parameters are fully replicated, allowing each device to operate independently without costly collective operations (e.g., All-Reduce). Similarly, within the EG, the inherent sparsity of the token-to-expert routing ensures that an activated expert's computation is confined to a single GPU. This confinement prevents the necessity of communication between expert devices.

The necessary collective communication occurs solely between the two groups through two defined communication phases: 1) Attention-to-Expert (A2E), where tokens processed by the AG are routed to the appropriate expert(s) in the EG, and 2) Expert-to-Attention (E2A), where the expert outputs are gathered and returned to the AG for subsequent layers. This disaggregation enables independent scaling of computational resources for each module and the development of tailored parallel strategies [\[17,](#page-11-14) [36\]](#page-11-13). The sequential execution of MoE inference with DEP is illustrated in Fig. [3\(](#page-2-0)a). Due to the data dependency between modules, the EG remains idle until the AG completes its forward pass and dispatches tokens via A2E communication. Conversely, the AG must wait idly for the EG to finish processing and return results via E2A before it can proceed to the next layer. While this disaggregation offers significant flexibility, this sequential handoff leads to significant device idle time in a naive implementation, as computational resources in one group are consistently underutilized while waiting for the other group to fulfill its part of the pipeline.

Ping-Pong Pipeline Parallelism. To rigorously address the device idle time inherent in the sequential dependency between the Attention Group and the Expert Group, the Ping-Pong Pipeline Parallelism (PPPipe) algorithm [\[36\]](#page-11-13) serves as a specialized microbatch scheduling strategy that enables the concurrent utilization of both AG and EG as shown in Fig. [3b.](#page-2-0) Specifically, PPPipe divides the input mini-batch into <sup>1</sup> micro-batches (e.g., <sup>1</sup> = 2 in Fig. [3b\)](#page-2-0) to allow GPUs in EG to begin computations without waiting for the full output of the mini-batch. Thus, in PPPipe, AG and EG computation tasks can be executed in parallel, and the communication tasks can

<span id="page-2-0"></span>![](_page_2_Figure_6.jpeg)

Figure 3: Timeline of naive DEP, PPPipe, and our FinDEP.

also be overlapped with the computation tasks, thus improving the inference throughput.

#### 2.3 Motivations

While PPPipe in MegaScale-Infer [\[36\]](#page-11-13) allows AG and EG tasks to be pipelined to reduce the GPU idle time, it is still suboptimal due to the following limitations.

Computation tasks of the shared expert are not well scheduled. In PPPipe [\[36\]](#page-11-13), it assumes that there is no shared expert in AG, which does not support recent MoE models like DeepSeek-V3. Built atop PPPipe, one can support including the shared expert by regarding it as a part of attention, since both attention and the shared expert should process all input tokens. As shown in Fig. [3b,](#page-2-0) A2E can only begin after the completion of the shared expert computation. However, the computation of experts in AE has no data dependency with the shared expert; thus, they can also overlap. This means the computation tasks of the shared expert should be well-scheduled to achieve better efficiency.

Micro-batch level pipelining is insufficient to overlap tasks fully. Existing DEP implementations, including PPPipe, overlook potential performance gains from overlapping communication and computation between attention and expert modules. While existing solutions focus on overlapping attention and expert computations, they underestimate additional benefits from overlapping A2E or E2A communication with expert computation. This overlap allows the GPU-resident expert module to begin computation earlier, potentially improving utilization and throughput [\[25\]](#page-11-19). We show an example in Fig. [3d,](#page-2-0) where dividing the expert into two micro-batches reduces end-to-end execution time. However, introducing pipelining can also incur kernel launch overhead, which in some cases may increase rather than reduce expert computation time, thereby

worsening bottlenecks. To address this trade-off, a modern and adaptive pipelining degree is required to balance early computation benefits and kernel launch costs. Balancing overlap benefits and launch costs is crucial for further exploiting throughput potential.

Huge search space to find an optimal schedule. The integration of shared experts and fine-grained pipeline settings significantly expands the search space for optimal configurations. Consequently, this expansion stems from numerous interdependent design choices, including pipeline degrees, microbatch sizes, and task orders. Similarly, decisions regarding the fine-grained degree and micro-size of experts, alongside the configurations of AG and EG, further compound this complexity. Due to such entanglement, brute-force enumeration becomes impractical. Therefore, an adaptive and efficient algorithm is necessary to explore the design space to find the optimal solution efficiently.

In this paper, we aim to address the above three issues by proposing FinDEP to partition tensors for fine-grained task scheduling with the support of shared experts. Thus, we split the attention input for each GPU along the batch dimension to enable a microbatch level pipeline. The number of pipelines is denoted by  $r_1$ , and the micro-batch size per GPU is denoted by  $m_a$ . Since there are no data dependencies, the shared expert and A2E of each micro-batch can run in parallel, as shown in Fig. 3c. Other task orders are discussed in the next section. Unlike the attention part, which involves interactions between tokens (i.e., intra-sequence), the expert part processes samples token by token. Based on this, we can further partition along the token dimension. The pipeline degree is denoted by  $r_2$ , and  $m_e$  represents the token processed by each expert. An example is shown in Fig. 3d. A primary challenge within FinDEP is to define the optimal problem (§3) and derive the optimal solution (§4), which we will present in the next two sections.

#### <span id="page-3-0"></span>3 Problem Formulation

The inference time of an MoE model under disaggregated expert parallel decomposes into three primary components: the computation time for the expert feed-forward networks, the computation time for the attention layers, and the communication overhead for transferring activations between the attention and expert groups. We formulate each component as a function of its workload and the underlying hardware characteristics.

#### 3.1 Execution Time Formulation

First, for GEMM, we denote the time function as  $t_{gm}(x, F)$ , where  $x = m \times k \times n$  represents the total FLOPs required for multiplying two matrices  $A \in \mathbb{R}^{m \times k}$  and  $B \in \mathbb{R}^{k \times n}$  on a GPU with peak floating-point performance F.

The second part of the computation involves the self-attention mechanism. The time function for the attention computation is denoted as  $t_{attn}(y, F)$ , where y represents the total workload for self-attention, and F is the GPU's peak performance again. In this case, the workload is defined based on the dimensions of the query (Q), key (K), and value (V) matrices. These matrices have the following shapes:  $Q, K \in \mathbb{R}^{N_h \times B \times S \times D_k}$ , and  $V \in \mathbb{R}^{N_h \times B \times S \times D_v}$ , where  $N_h$  is the number of attention heads, B is the batch size, S is the sequence length,  $D_k$  is the dimensionality of the key, and  $D_v$  is the dimensionality of the value. The core computational burden comes

from two GEMMs: the computation of the attention scores via  $QK^{\top}$ , which has a complexity of  $N_hBS^2D_k$ , and the computation of the attention-weighted values Attention( $QK^{\top}$ )V, which has a complexity of  $N_hBS^2D_v$ . Therefore, the total workload for self-attention is  $y = N_hBS^2(D_k + D_v)$ .

Third,  $t_c(z, eg, ag)$  denotes the communication time required for data transfer between GPUs. More specifically, it measures the time taken for ag GPUs to send messages to eg other GPUs. Here, z represents the communication workload per machine, while eg and ag refer to the expert and attention group sizes, respectively.

For any given hardware and group configuration, the GPU performance F and group sizes (eg, ag) are constant. This stability allows us to simplify the time functions, specifically  $t_{gm}(x)$ ,  $t_{attn}(y)$ , and  $t_{a2e}(z)$ , for convenience. These simplified functions provide a foundation for modeling the end-to-end performance of MoE systems.

**The Attention Part.** The attention component consists of a sequence of computational operations that include both GEMMs and attention. To illustrate this process, we consider the standard Multi-Head Attention (MHA) layer as an example. In the t-th transformer layer, the input comprises hidden states  $\mathbf{h}_t \in \mathbb{R}^{m_a \times S \times M}$ . The total forward pass time for this layer, denoted  $t_a(m_a)$ , is a function of these dimensions and can be decomposed into the cumulative runtime of several GEMM and self-attention operations:

<span id="page-3-1"></span>
$$t_a(m_a) = 2t_{gm}(m_a SMn_h d_k) + 2t_{gm}(m_a SMn_h d_v)$$
  
+ 
$$t_{attn}(m_a S^2 n_h (d_k + d_v)).$$
 (1)

The coefficients 2 and 2 in the  $t_{gm}$  terms account for the four linear projections required by the MHA operation: Q and K projections, and V and Output (O) projections. Notably, other attention variants like MLA [5] can also be modeled using similar formulations involving  $t_{attn}$  and  $t_{gm}$ , enabling unified analysis across various attention designs.

The Shared Expert Part. The Shared Expert computation follows a structure similar to the attention layer, consisting of three primary linear projections: the gating projection, the up-projection, and the down-projection. For each expert i ( $1 \le i \le N_{shared}$ ), the gating and up-projections are represented by  $W_i^{gate}$  and  $W_i^U$ , respectively, with dimensions  $W_i^{gate}$ ,  $W_i^U \in \mathbb{R}^{H \times M}$ , while the down-projection is given by  $W_i^D \in \mathbb{R}^{M \times H}$ .

Each device in AG performs the shared expert transformations locally. The gating operation computes  $\mathbf{z}_{t,i}^{gate} = W_i^{gate} \mathbf{h}_t$ , the upprojection computes  $\mathbf{z}_{t,i}^u = W_i^U \mathbf{h}_t$ , and the down-projection computes  $\mathbf{z}_{t,i}^d = W_i^D \mathrm{Swish}(\mathbf{z}_{t,i}^{gate} \otimes \mathbf{z}_{t,i}^u)$  [23], where  $\mathrm{Swish}(x) = \frac{x}{1+e^{-x}}$ . These operations result in outputs with dimensions  $\mathbf{z}_{t,i}^{gate}$ ,  $\mathbf{z}_{t,i}^u \in \mathbb{R}^{m_a \times S \times H}$  and  $\mathbf{z}_{t,i}^d \in \mathbb{R}^{m_a \times S \times M}$ , each taking  $t_{gm}(m_a S M H)$  time. The total computation time for the Shared Expert across  $N_{shared}$  expert layers is the sum of all layers:

<span id="page-3-2"></span>
$$t_s(m_a) = 3N_{shared}t_{qm}(m_aSMH). \tag{2}$$

**The MoE Part.** The MoE layer employs conditional computation through a set of feed-forward networks, known as experts. The total number of E experts is distributed across eg devices. Each device is responsible for computing E/eg distinct experts. Each device in the expert group receives tokens, represented as  $\mathbf{h}_t' \in \mathbb{R}^{(E/eg) \times m_e \times M}$ , which are then partitioned along the first dimension into E/eg slices:

 $\mathbf{h}'_{t,1}, \mathbf{h}'_{t,2}, \dots, \mathbf{h}'_{t,E/eg}$ . Each slice,  $\mathbf{h}'_{t,i} \in \mathbb{R}^{m_e \times M}$ , is assigned to the corresponding local expert i. Here,  $m_e$  denotes the number of tokens processed by a single expert. For each expert i, the computation involves a feed-forward network with weights including the upprojection  $W_i^U \in \mathbb{R}^{H \times M}$ , the gating projection  $W_i^{gate} \in \mathbb{R}^{H \times M}$ , and the down-projection  $W_i^D \in \mathbb{R}^{M \times H}$ , all of which reside on the assigned device. The total computation time for each device is given by:

<span id="page-4-5"></span>
$$t_e(m_e) = 3(E/eg)t_{qm}(m_eMH). \tag{3}$$

**A2E and E2A communication.** DEP employs two distinct communication operations: Attention-to-Expert (A2E) and Expert-to-Attention (E2A). We denote their respective communication times as  $t_{a2e}$  and  $t_{e2a}$ . Since the communication workload is  $z = E/eg \times m_e \times M$ . We have:

<span id="page-4-7"></span>
$$t_{a2e}(m_e) = t_c(m_e EM/eq). \tag{4}$$

Due to the symmetric nature of communication in dual-workload topologies like PCIe or NVLink [10], where data transfer occurs in different directions simultaneously, the time taken for A2E equals that for E2A, i.e.,  $t_{a2e}(m_e) = t_{e2a}(m_e)$ .

#### 3.2 Optimization Problem Formulation

For a given layer t, we define the timestamps that capture the start times of major computational and communication stages. Let  $\tau_a^{(t,i)}$  represent the start time of the i-th attention segment within the  $r_1$  pipeline, and  $\tau_s^{(t,i)}$  the start time of the corresponding Shared Expert computation. The Expert computation within each pipeline segment is also divided into  $r_2$  parts, and we denote the start time of the expert processing for the i-th  $r_1$  slice and j-th  $r_2$  token group as  $\tau_e^{(t,i,j)}$ . The communication timestamps are defined as  $\tau_{a2e}^{(i,j)}$  and  $\tau_{e2a}^{(i,j)}$ , indicating the start times of the A2E and E2A communication phases, respectively.

Based on the above execution time formulations, we derive a set of timing constraints between key scheduling timestamps:  $\tau_a^{(t,i)}$ ,  $\tau_s^{(t,i,j)}$ ,  $\tau_{e^{t}}^{(t,i,j)}$ ,  $\tau_{e^{t}}^{(t,i,j)}$ , and  $\tau_{e2a}^{(t,i,j)}$ . These constraints describe how different stages of computation and communication must be ordered to avoid conflicts and ensure data dependencies are satisfied. All constraints are represented as

<span id="page-4-1"></span>
$$\begin{cases} \tau_{s}^{(t',i')}, \tau_{a}^{(t',i')} \notin [\tau_{a}^{(t,i)}, \tau_{a}^{(t,i)} + t_{a}(m_{a})) \\ \tau_{s}^{(t',i')}, \tau_{a}^{(t',i')} \notin [\tau_{s}^{(t,i)}, \tau_{s}^{(t,i)} + t_{s}(m_{a})) \\ \tau_{a2e}^{(t',i')}, \notin [\tau_{a2e}^{(t,i,j)}, \tau_{a2e}^{(t,i,j)} + t_{a2e}(m_{e})) \\ \tau_{e2a}^{(t',i',j')}, \notin [\tau_{e2a}^{(t,i,j)}, \tau_{e2a}^{(t,i,j)} + t_{e2a}(m_{e})) \\ \tau_{e}^{(t',i',j')} \notin [\tau_{e}^{(t,i,j)}, \tau_{e}^{(t,i,j)} + t_{e}(m_{e})) \\ \tau_{s}^{(t',i',j')} \notin [\tau_{e}^{(t,i,j)}, \tau_{e}^{(t,i,j)} + t_{e}(m_{e})) \\ \tau_{s}^{(t,i)}, \tau_{a2e}^{(t,i,j)} \ge \tau_{a}^{(t,i)} + t_{a}(m_{a}) \\ \tau_{e2a}^{(t,i,j)} \ge \tau_{a2e}^{(t,i,j)} + t_{e}(m_{e}) \\ \tau_{a}^{(t+1,i)} \ge \max(\tau_{e2a}^{(t,i,j)} + t_{e2a}(m_{e}), \tau_{s}^{(t,i)} + t_{s}(m_{a})) \\ m_{s}, r_{s}, F = m_{s}, a_{s}, t_{s}, t_{s}, r_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T_{s}, t_{s} \le T$$

The first five rules prevent different stages from using the same hardware at the same time. This avoids resource conflicts. Rules 6 to 9 ensure that each stage starts only after the previous one finishes in the same micro-batch. The final rule ensures that all data is processed accurately without any loss.

Our goal is to maximize the throughput of the disaggregated MoE pipeline by jointly optimizing the pipeline degrees and token partition sizes. This leads to the following optimization formulation:

<span id="page-4-2"></span>
$$\max_{\substack{r_1, m_a \\ r_2, m_e}} \frac{r_1 \cdot m_a \cdot ag}{\max(\tau_s^{(T, r_1)} + t_s(m_a), \ \tau_{e2a}^{(T, r_1, r_2)} + t_{e2a}(m_e))}$$
s. t. constraints in Eq. (5).

For any fixed choice of  $r_1, m_a, r_2, m_e$ , the remaining task is to assign start times  $\tau$  that satisfy the constraints in Eq. (5) and minimize the makespan  $\max \left( \tau_s^{(T,r_1)} + t_s(m_a), \ \tau_{e2a}^{(T,r_1,r_2)} + t_{e2a}(m_e) \right)$ . This scheduling subproblem is a variant of the job-shop problem: each operation (attention, shared, A2E, expert, E2A) runs on a dedicated machine (resources) and the operations of each micro-batch follow the precedence graph implied by rules 6–9. It is well known that job-shop scheduling is NP-hard even with three machines. Our model involves four distinct resources (e.g., AG, EG, A2E, and E2A), therefore the subproblem is NP-hard. Consequently, the overall problem, which additionally optimizes over the integer parameters  $r_1, m_a, r_2, m_e$ , is also NP-hard, because a polynomial-time algorithm for the overall problem would yield a polynomial-time solution for the NP-hard subproblem by fixing those parameters appropriately.

#### <span id="page-4-0"></span>4 Solution

To solve the above problem, we need to explicitly determine the communication and computation times. Thus, we need to model the performance for a given communication or computation operation, so that we can predict their execution time with different sizes of input. In this section, we first build simple yet effective performance models for attention, GEMM computation, and A2E/E2A communication, then we derive the near-optimal solution to the problem of minimizing Eq. 6.

#### 4.1 Performance Models

**Performance model of computation.** Following [16, 20, 25], we use a linear model (with bias) to represent computation. The model includes an intercept term  $(\alpha_{gm})$  to account for fixed overheads, such as kernel launches and memory management, and a scaling factor  $(\beta_{gm})$  to capture the increase in computational cost as the input size grows. The model is expressed as:

<span id="page-4-4"></span><span id="page-4-3"></span>
$$t_{gm}(x) = \alpha_{gm} + \beta_{gm}x. \tag{7}$$

$$t_{attn}(y) = \alpha_{attn} + \beta_{attn}y. \tag{8}$$

**Performance model of communication.** For both A2E and E2A operations, the communication time can also be accurately described using a single  $\alpha$ - $\beta$  model. These operations are essentially reverse processes that share identical communication structures, allowing for a unified linear model. We define the communication time as:

<span id="page-4-6"></span>
$$t_c(z) = \alpha_c + \beta_c z,\tag{9}$$

where z represents the input data size (bytes of elements communicated),  $\alpha_c$  is the network startup time (overhead), and  $\beta_c$  is the transmission time per byte, which is influenced by factors such as network bandwidth.

**Performance models of different layers.** By substituting Eq. 7 and Eq. 8 into Eq. 1, we derive a simplified linear model for the performance of the MHA layer, expressed as:  $t_a(m_a) = \alpha_a + \beta_a m_a$ , where the new coefficients are defined as follows:

$$\alpha_a := 4\alpha_{am} + \alpha_{attn} \tag{10}$$

and

$$\beta_a := \beta_{gm} (2SMn_h d_k + 2SMn_h d_v)$$

$$+ \beta_{attn} S^2 n_h (d_k + d_v).$$
(11)

For clarity and analytical tractability, we absorb all terms that do not vary with  $m_a$  into the constants  $\alpha_a$  and  $\beta_a$ . This yields a linear performance model that captures the contribution of  $m_a$  to computation time.

Similarly, by substituting Eq. 7 into Eq. 2, we derive a simplified linear model for the performance of the shared expert layer, expressed as  $t_s(m_a) = \alpha_s + \beta_s m_a$ , where the new coefficients are defined as follows:  $\alpha_s := 3N_{shared}\alpha_{qm}$  and  $\beta_s := 3N_{shared}\beta_{gm}SMH$ .

Building on this methodology, the MoE layer's performance is modeled. Substituting Eq. 7 into Eq. 3 and Eq. 9 into Eq. 4, and absorbing all terms not varying with  $m_e$  into constants, we express its performance as  $t_e(m_e) = \alpha_e + \beta_e m_e$ . Here,  $\alpha_e := (E/eg)\alpha_{gm}$  and  $\beta_e := (E/eg)\beta_{gm}(MH)$ . For A2E and E2A, the model is  $t_{a2e}(m_e) = \alpha_{a2e} + \beta_{a2e}m_e$ , where  $\alpha_{a2e} := \alpha_c$  and  $\beta_{a2e} := \beta_c EM/eg$ . These linear models provide a tractable framework for analyzing and predicting the computational overhead of each architectural component, laying the groundwork for subsequent performance optimization. Although streamlined, this model effectively captures the dominant performance determinants of startup latency and workload-dependent scaling, and its fidelity is empirically validated in (§5.2).

#### 4.2 Determine Task Order, $m_a$ , $r_1$ , $m_e$ , and $r_2$ .

**Determine the order of Attention and Shared Expert.** We investigate the optimal execution order of attention and Shared Expert operations in AG by evaluating two primary scheduling strategies. The number of possible non-illness computing orders in a layer is given by  $C(r_1 + r_1 - 1, r_1) = \frac{(2r_1 - 1)!}{(r_1!)((r_1 - 1)!)}$ , which is cumbersome to verify one by one. However, we can observe that the advantages of more efficient computing are: (a) it allows for the earliest possible start of A2E communication, which helps utilize EG without idle time, and (b) it enables the use of AG (Attention Gate) without idle time.

We focus on the most representative strategies and explain why they are effective. The first, AASS (Attention-All, Shared-All), processes all attention segments within the same layer before proceeding to all Shared Expert segments. The second, ASAS (Attention-Shared-Alternating-Sequential), alternates between attention and Shared Expert operations.

As illustrated in Fig. 4, each schedule presents distinct advantages. The *AASS* approach enables earlier initiation of A2E communication and expert computation, as evident when comparing Fig. 4a and Fig. 4b. Conversely, *ASAS* improves GPU utilization by interleaving Shared Expert segments during periods in which attention-ready signals are pending, as shown in Fig. 4c and Fig. 4d.

To determine the better strategy, we independently identify the best-performing configuration for both AASS and ASAS. We then

<span id="page-5-0"></span>![](_page_5_Figure_13.jpeg)

![](_page_5_Figure_14.jpeg)

![](_page_5_Figure_15.jpeg)

![](_page_5_Figure_16.jpeg)

(d) An example illustrating the advantages of ASAS.

Figure 4: Comparative examples highlighting the advantages and limitations of AASS and ASAS scheduling strategies.

compare their performance outcomes to select the superior scheduling policy.

**Determine**  $m_a$ . For illustrative purposes, we focus on optimizing the *ASAS* scheduling strategy. The same methodology can be straightforwardly applied to *AASS*.

Firstly, our optimization focuses on  $r_1$  and  $m_a$ . Given a fixed execution order, we can iteratively compute the key timing variables:  $\tau_a^{(t,i)}$ ,  $\tau_s^{(t,i)}$ ,  $\tau_e^{(t,i,j)}$ ,  $\tau_{a2e}^{(t,i,j)}$ , and  $\tau_{e2a}^{(t,i,j)}$ .

We first examine the timing relationships within the 0-th layer. Since the derivation for each variable follows a similar pattern, we use  $\tau_{e2a}^{(0,i,j)}$  as an illustrative example. This timestamp depends on the completion of i pipeline chunks and additionally on the completion of j fine-grained pipeline steps. Its value therefore decomposes into three components: an initial latency, the cumulative delay from the  $r_1$  pipeline, and the cumulative delay from the fine-grained  $r_2$  pipeline, visualized in Fig. 5.

<span id="page-6-0"></span>
$$\begin{split} \tau(\boxed{\uparrow_0^{(i,j)}}) &= i * \underbrace{\max(\boxed{A} \quad S}, r_2 * \max(\boxed{E}, \boxed{\downarrow}))}_{F: \text{ Pipeline time}} \\ &+ j * \max(\boxed{E}, \boxed{\downarrow}) + \underbrace{\boxed{A} \quad \bot E}_{Y: \text{ Fine-grained pipeline time}} \end{split}$$

Figure 5: Diagram of the 0-th layer start timestamp  $\tau_{e2a}^{(0,i,j)}$  , decomposed into three components: pipeline time, fine-grained pipeline time, and initial latency.

Proceeding similarly for all variables, we obtain the complete set of timing expressions for the 0-th layer:

$$\begin{cases} \tau_a^{(0,i)} = i \cdot X(m_a) \\ \tau_s^{(0,i)} = i \cdot X(m_a) + t_a(m_a) \\ \tau_{a2e}^{(0,i,j)} = t_a(m_a) + i \cdot F(m_a, m_e) + j \cdot t_{a2e}(m_e) \\ \tau_e^{(0,i,j)} = t_a(m_a) + t_{a2e}(m_e) + i \cdot F(m_a, m_e) + j \cdot Y(m_e) \\ \tau_{e2a}^{(0,i,j)} = t_a(m_a) + t_{a2e}(m_e) + t_e(m_e) \\ + i \cdot F(m_a, m_e) + j \cdot Y(m_e) \end{cases}$$

where  $X(m_a) = t_a(m_a) + t_s(m_a)$ ,  $Y(m_e) = \max(t_e(m_e), t_{a2e}(m_e))$ ,

and  $F(m_a, m_e) = max(X(m_a), r_2 \cdot Y(m_e))$ . For the t-th layer, the timing variables  $\tau_a^{(t,i)}$ ,  $\tau_s^{(t,i)}$ ,  $\tau_e^{(t,i,j)}$ ,  $\tau_{a2e}^{(t,i,j)}$ , and  $\tau_{e2a}^{(t,i,j)}$  can be derived based on the corresponding variables from the (t-1)-th layer. Specifically, each of them is computed by adding an offset term to their respective (t-1)-th counterparts. This offset is given by:  $\max(G(m_a, m_e), r_1 \cdot F(m_a, m_e))$ , where

$$G(m_a, m_e) = t_a(m_a) + t_{a2e}(m_e) + t_e(m_e) + t_{a2e}(m_e) + (r_2 - 1) \cdot Y(m_e).$$
(12)

Here,  $E(m_a, m_e)$  represents the time required to ensure that the GPUs in AG are idle and ready for the next attention segment, while  $r_1 \cdot F(m_a, m_e)$  denotes the time required for the output of the expert computation on the 0-th chunk to be sent back.

We can simplify the optimal objective defined in Eq. 6 as follows:

$$\max_{\substack{r_1, m_a \\ r_2, m_e}} \frac{r_1 \cdot m_a}{(T-1) \max(G(m_a, m_e), r_1 F(m_a, m_e))} \cdot (13)$$

$$+ \max(X(m_a), G(m_a, m_e)) + (r_2 - 1) Y(m_e)$$

$$+ (r_1 - 1) F(m_a, m_e)$$

To accelerate the search process, we first identify a crucial property: for a fixed value of  $r_1$ , the objective function defined in Eq. 13 increases monotonically with respect to  $m_a$ . To establish this, we employ a two-step proof. First, we demonstrate that for any given pair  $(r_1, r_2)$ , the objective function in Eq. 13 is monotonically increasing with respect to  $m_a$ . The detailed proof of this claim is presented below.

<span id="page-6-2"></span>Theorem 1. Given pair  $(r_1, r_2)$ , the objective function in Eq. 13 is monotonically increasing with respect to  $m_a$ .

PROOF. To analyze the behavior of the objective function concerning  $m_a$ , we first establish a direct relationship between  $m_e$  and  $m_a$ . From the constraint  $m_a \cdot ag \cdot top_k \cdot S = m_e \cdot r_2 \cdot E$ , we can express

 $m_e$  as a linear function of  $m_a$ . We have  $m_e = k \cdot m_a$ , where the constant  $k = \frac{ag \cdot top_k \cdot S}{r_2 \cdot E}$ .

The component functions  $X(m_a)$ ,  $Y(m_e)$ , and  $E(m_a, m_e)$  are defined as sums and maximums of the base linear performance models  $t_a(m_a)$ ,  $t_s(m_a)$ , and  $t_e(m_e)$ . By substituting  $m_e = k \cdot m_a$ , each of these components becomes a linear or piecewise linear function of  $m_a$ . Specifically, the denominator of the objective function is constructed from additions and max operations on these functions. Since the sum of linear functions is linear, and the maximum of linear functions is piecewise linear and convex, the entire denominator is a positive, piecewise linear, and convex function of  $m_a$ .

Therefore, the objective function takes the form of  $\frac{r_1 m_a}{D(m_a)}$ where  $D(m_a)$  is the piecewise linear denominator. Within any linear segment of  $D(m_a)$ , the objective function can be written as  $\frac{r_1m_a}{\alpha_{total}+\beta_{total}m_a}$ , where  $\alpha_{total}$  and  $\beta_{total}$  are positive constants aggregated from the underlying  $\alpha$  and  $\beta$  parameters of the performance models. To demonstrate its monotonic nature, we can rewrite the previous equation as  $\frac{\alpha_{total}}{\frac{\alpha_{total}}{m_a} + \beta_{total}}$ . As  $m_a$  increases, the term  $\frac{\alpha_{total}}{m_a}$ decreases. This causes the denominator of the overall expression to decrease, which in turn increases the value of the function. Thus, the objective function is monotonically increasing concerning  $m_a$ across each linear segment, and therefore, it is monotonically increasing for all  $m_a > 0$ .

Next, we extend the result to show that for a fixed value of  $r_1$ , the objective function in Eq. 13 increases monotonically with respect to  $m_a$ . This follows directly from Theorem 1. The detailed proof of this generalized claim is provided below.

<span id="page-6-3"></span>THEOREM 2. Given  $r_1$ , the objective function in Eq. 13 is monotonically increasing with respect to  $m_a$ .

Proof. Consider any arbitrary value  $m_a$  and the corresponding pair  $(r_1, r_2^*)$ , where

$$r_2^* = \arg\max_{r_2} \frac{r_1 \cdot m_a}{\max(\tau_s^{(T,r_1)} + t_s(m_a), \ \tau_{e2a}^{(T,r_1,r_2)} + t_{e2a}(m_e))},$$

<span id="page-6-1"></span>according to Theorem 1, for any  $m'_a > m_a$ , the following inequality holds:

$$\frac{r_1 \cdot m'_a}{\max(\tau_s^{(T,r_1)} + t_s(m'_a), \ \tau_{e2a}^{(T,r_1,r_2^*)} + t_{e2a}(m'_e))} > \max_{r_2} \frac{r_1 \cdot m_a}{\max(\tau_s^{(T,r_1)} + t_s(m_a), \ \tau_{e2a}^{(T,r_1,r_2)} + t_{e2a}(m_e))}$$

It implies that

$$\begin{split} &\frac{r_1 \cdot m_a'}{\max(\tau_s^{(T,r_1)} + t_s(m_a'), \ \tau_{e2a}^{(T,r_1,r_2^*)} + t_{e2a}(m_e'))} \\ \leq &\max_{r_2} \frac{r_1 \cdot m_a'}{\max(\tau_s^{(T,r_1)} + t_s(m_a'), \ \tau_{e2a}^{(T,r_1,r_2)} + t_{e2a}(m_e'))}. \end{split}$$

Consequently, the objective function increases monotonically as  $m_a$  increases, which completes the proof.

**Determine**  $r_1$ . We now turn our attention to analyzing the behavior of the objective function with respect to the parameter  $r_1$ . Specifically, we aim to demonstrate that, for a fixed value of

<span id="page-7-2"></span>![](_page_7_Figure_1.jpeg)

Figure 6: The pipeline of FinDEP, which consists of an offline planning phase and an online adaptive phase.

 $m_a$ , the objective function defined in Eq. 13 is monotonically nondecreasing with respect to  $r_1$ . To establish this result, we adopt a two-step proof strategy analogous to that used in prior analysis.

The first step involves showing that, for any fixed pair  $(m_a, r_2)$ , the objective function increases or remains constant as  $r_1$  increases. The second step mirrors the approach in Theorem 2. However, due to space constraints, we omit this proof from the paper. In what follows, we focus on formally proving the first step.

THEOREM 3. Given  $(m_a, r_2)$ , the objective function in Eq. 13 is monotonically non-decreasing with respect to  $r_1$ .

PROOF. To analyze monotonicity with respect to  $r_1$ , observe that the objective function takes the form  $\frac{r_1m_a}{D(r_1)}$ , where the denominator  $D(r_1)$  is piecewise linear in  $r_1$ . When expressed as  $D(r_1) = Br_1 + C$  in each linear segment (B > 0), monotonicity depends critically on the sign of the constant term C. We demonstrate  $C \ge 0$ , where

$$C = \max(X(m_a), G(m_a, m_e) + (r_2 - 1)Y(m_e)) - F(m_a, m_e).$$
 (14)

First, from the inequality  $E(m_a, m_e) \ge Y(m_e)$ , we derive:

$$G(m_a, m_e) + (r_2 - 1)Y(m_e) \ge r_2 Y(m_e).$$
 (15)

Consequently, we have:

$$\max(X(m_a), G(m_a, m_e) + (r_2 - 1)Y(m_e)) \ge \max(X(m_a), r_2Y(m_e)). \tag{16}$$

Since  $F(m_a, m_e) = \max(X(m_a), r_2Y(m_e))$ , it follows directly that  $C \ge 0$ . With  $A = m_a > 0$ , B > 0, and  $C \ge 0$ , the objective function becomes  $\frac{Ar_1}{Br_1+C}$ . Its derivative is:

$$\frac{d}{dr_1}(\frac{Ar_1}{Br_1+C}) = \frac{A(Br_1+C) - Ar_1B}{(Br_1+C)^2} = \frac{AC}{(Br_1+C)^2} \ge 0,$$

since A > 0,  $C \ge 0$ , and the denominator is positive. Therefore, the objective function is monotonically non-decreasing in  $r_1$ .

**Determine**  $r_2$  and  $m_e$ . The final parameters to verify are  $r_2$  and  $m_e$ . Given  $m_a$  and  $r_1$ , we have only one free variable, as the other is constrained by the relation:  $m_a \cdot ag \cdot top_k \cdot S/E = m_e \cdot r_2$ . To simplify, we express  $m_e(1/r_2) = (m_a ag top_k S)/(E \cdot r_2) = k'/r_2$  thereby reducing the problem to solving for  $r_2$  alone. Fortunately, the objective function is convex with respect to  $1/r_2$ .

Theorem 4. Given  $r_1$  and  $m_a$ , the objective function in Eq. 13 is convex with respect to  $1/r_2$ .

PROOF. To optimize the objective function in Eq. 13, we express it as the following equivalent form:

$$\min_{r_2} ((T-1) \max(G(m_a, m_e), r_1 F(m_a, m_e))) + \max(X(m_a), G(m_a, m_e)) + (r_2 - 1) Y(m_e) + (r_1 - 1) F(m_a, m_e).$$
(17)

#### <span id="page-7-1"></span>Algorithm 1 FinDEP Configuration Search

```
Input: P, ag, eg, \alpha_*, \beta_*, B, S, H, M, N_{\text{shared}}, E, top_k, T
Output: best_config = (m_a, r_1, m_e, r_2, order)
  1: best_tps ← 0
  2: best_config ← ∅
  s: r'_1 \leftarrow 0
                                                                          ▶ Previous r<sub>1</sub>
  4: for m_a = \infty downto 1 do
           r_1 \leftarrow \text{getMaxR1}(ag, eg, m_a, P, B, S, H, M, N_{\text{shared}}, E, top_k, T)
     ▶ Memory-constrained
           if r_1 == 0 or r_1 == r'_1 then
                continue
                                           ▶ Skip non-Pareto-optimal (m_a, r_1)
           for order \in \{ASAS, AASS\} do \triangleright Evaluate both execution
     orders
                r_2^*, tps \leftarrow Solve(min<sub>r_2</sub> Eq. 17) \triangleright Returns optimizer and
     optimal value
               m_e \leftarrow \frac{m_a \cdot ag \cdot top_k \cdot S}{r_2^* \cdot E}
 10:
                if tps > best_tps then
 11:
 12
                     best\_tps \leftarrow tps
                     best_config \leftarrow (m_a, r_1, m_e, r_2^*, \text{ order})
 14:
           r_1' \leftarrow r_1
 15: return best_config
```

We aim to prove the convexity of this objective function. Specifically, we need to verify the convexity of the term  $(r_2 - 1)Y(m_e)$  and the product  $r_2(m_e)Y(m_e)$  within  $F(m_a, m_e)$ . The performance models  $t_e(1/r_2) = \alpha_e + \beta_e k'/r_2$  and  $t_{a2e}(1/r_2) = \alpha_{a2e} + \beta_{a2e} k'/r_2$  are linear functions of  $1/r_2$ . Since their coefficients are positive, these functions are convex and monotonically increasing. We define  $Y(1/r_2) = \max(t_e(1/r_2), t_{a2e}(1/r_2))$ . The maximum of linear functions is piecewise linear and convex, and since both are increasing,  $Y(1/r_2)$  is non-decreasing, preserving convexity.

Next, the product  $r_2 \cdot Y(m_e)$  is the maximum of terms of the form  $\alpha r_2 + k'\beta$ , which are convex for  $r_e > 0$ . Hence,  $r_2 \cdot Y(m_e)$  is convex.

The function  $G(m_a, m_e)$  includes terms like  $t_e(1/r_2)$ ,  $t_{a2e}(1/r_2)$ , and  $(r_2-1)Y(1/r_2)$ . The linear terms are convex, and  $(r_2-1)Y(m_e)$ , which is a maximum of convex functions, is also convex. Thus,  $G(m_a, m_e)$  is convex.

<span id="page-7-0"></span>The objective function is the sum of three terms:  $(r_1 - 1)F(m_a, m_e)$ , which is convex since  $r_1 \ge 1$  and  $F(m_a, m_e)$  is convex;  $(T-1)\max(G(m_a, m_e), r_1F(m_a, m_e))$ , which is convex since both  $G(m_a, m_e)$  and  $r_1F(m_a, m_e)$  are convex, and  $T \ge 1$ ; and  $\max(X(m_a), G(m_a, m_e))$ , which is convex because it is the maximum of a constant and a convex function. Since the sum of convex functions is convex, the entire objective function is convex with respect to  $1/r_2$ .

#### 4.3 Algorithm

Based on the previous analysis, we propose an efficient algorithm to find the near-optimal configuration for  $r_1$ ,  $m_a$ ,  $r_2$ , and  $m_e$ , as shown in Algorithm 1. Given a computing order, the algorithm provides the optimal configuration for that order, focusing on maximizing inference throughput. Specifically, we focus on the Pareto frontier of  $(m_a, r_1)$  under memory constraints, respecting the monotonicity of  $m_a$  and  $r_1$ . The algorithm iterates over  $m_a$  in descending order and calculates the maximum allowable  $r_1$  based on memory limits. We skip configurations with the same  $r_1$  as the previous iteration to avoid redundancy.

For each unique  $(m_a, r_1)$  pair, the algorithm evaluates two execution orders: ASAS and AASS. For each order, we solve a convex optimization problem to find the optimal  $r_2$  that maximizes the objective in Eq. 13. Then, we calculate  $m_e$  as:  $m_e = m_a \cdot ag \cdot top_k \cdot S/(r_2 \cdot E)$ . The configuration with the highest throughput is returned. This approach efficiently explores the search space and eliminates suboptimal configurations. With the near-optimal solution derived from Algorithm 1, we obtain the fine-grained task schedule in FinDEP for MoE inference.

**Complexity Analysis** Our complexity analysis is divided into two steps: first, determining the number of possible  $(r_1, m_a)$  positions on the Pareto frontier, and second, analyzing the time spent on convex optimization. Since the memory constraint is  $r_1 \cdot m_a \leq M$ , where M is the largest micro-batch size that the GPU can hold, the number of distinct values of  $m_a = \lfloor \frac{M}{r_1} \rfloor$  corresponds to the number of divisors of M, denoted as d(M). The number of divisors grows at most as  $O(\sqrt{M})$ . Since convex optimization is performed for a single parameter  $r_2$ , the solver operates quickly. Assuming constant optimization time, denoted as C, the overall complexity is  $O(C \cdot d(M))$ . Given the fast nature of the solver, the inference time is almost unaffected by this process(§5).

**Online Pipeline** Fig. 6 illustrates our system pipeline, which is bifurcated into offline and online phases. The offline phase handles initialization: we first select the serving model (e.g., DeepSeek or Qwen) and determine the sizes of the Attention Group and Expert Group (*ag*, *eg*). Subsequently, we utilize an offline performance model to collect the necessary model coefficients and hardware parameters, which serve as inputs for the optimization solver.

The online phase addresses runtime adaptation. As input data shapes are unknown prior to request arrival, configuration decisions must be made in real-time. Upon data arrival, the system executes the lightweight Algorithm 1 to rapidly derive the optimal configuration ( $m_a$ ,  $r_1$ ,  $m_e$ ,  $r_2$ , order). This approach allows FinDEP to dynamically adapt to varying workloads, achieving superior speedup ratios compared to static settings.

#### <span id="page-8-1"></span>5 Evaluation

#### 5.1 Testbeds

Our experiments leverage four distinct hardware testbeds. Testbed A uses a single node with eight NVIDIA A6000 GPUs, while Testbed B is configured with eight NVIDIA A10 GPUs. Testbed C also employs a single node, equipped with eight NVIDIA H20 GPUs, and Testbed D scales this configuration across four nodes, each containing eight H20 GPUs. Further details regarding the server configuration can be found in Table 2. Our software environment runs

<span id="page-8-2"></span>![](_page_8_Figure_10.jpeg)

Figure 7: Performance models for GEMM, Attention, and communication. Markers represent measured values, while the lines correspond to predicted values with estimated parameters. (a)  $\alpha_{gm}=0.17$  and  $\beta_{gm}=8.59\times10^{-11}$ .  $\alpha_{attn}=0.15$  and  $\beta_{attn}=1.54\times10^{-11}$ . (b)  $(\alpha_{a2e},\beta_{a2e})$  take the following values:  $(0.10,9.61\times10^{-7})$  for (eg=7,ag=1),  $(0.01,1.28\times10^{-6})$  for (eg=6,ag=2), and  $(0.37,2.55\times10^{-6})$  for (eg=4,ag=4).

on Ubuntu 22.04, with Python 3.10, CUDA 11.3, PyTorch 2.4, and NCCL 2.27.5. We implement attention using FlashInfer 0.3.0 [32].We implement Attention-to-Expert and Expert-to-Attention transfer atop NCCL.

#### <span id="page-8-0"></span>5.2 Verification of Performance Models

We conduct micro-benchmarks to determine the values of  $\alpha_{gm}$ ,  $\beta_{gm}$ ,  $\alpha_{attn}$ ,  $\beta_{attn}$ ,  $\alpha_{a2e}$  and  $\beta_{a2e}$  before solving the algorithm. For the GEMM component, we test a range of matrix configurations across all matrix sizes encountered in the MLA. This comprehensive testing ensures our model can effectively handle varying configurations. The results, shown in Fig. 7a, yield an  $R^2$  value of 0.997132.

For the communication component, we separately compute  $\alpha_{a2e}$  and  $\beta_{a2e}$  for different ag and eg settings, as these parameters are interdependent. Since the time for Expert-to-Attention matches that for Attention-to-Expert, we do not need to rerun the microbenchmark for the former case. The results, shown in Fig. 7b, yield  $R^2$  values of 0.999986, 0.999911, and 0.994018, indicating a strong fit. This demonstrates that simple linear models can accurately predict execution time, consistent with findings in prior work on performance modeling [16, 20, 25].

We run 30 trials per data point: 10 for warm-up and 20 for statistics. The full micro-benchmark, including Attention, GEMM, and communication steps, takes under 2 minutes.

