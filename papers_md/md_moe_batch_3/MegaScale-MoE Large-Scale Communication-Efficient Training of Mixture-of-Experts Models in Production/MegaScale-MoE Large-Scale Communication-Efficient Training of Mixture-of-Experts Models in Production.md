# MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production

Chao Jin\*§, Ziheng Jiang\*†, Zhihao Bai<sup>†</sup>, Zheng Zhong<sup>†</sup>, Juncai Liu<sup>†</sup>, Xiang Li<sup>†</sup>, Ningxin Zheng<sup>†</sup>, Xi Wang<sup>†</sup>, Cong Xie<sup>†</sup>, Qi Huang<sup>†</sup>, Wen Heng<sup>†</sup>, Yiyuan Ma<sup>†</sup>, Wenlei Bao<sup>†</sup>, Size Zheng<sup>†</sup>, Yanghua Peng<sup>†</sup>, Haibin Lin<sup>†</sup>, Xuanzhe Liu<sup>§</sup>, Xin Jin<sup>§</sup>, Xin Liu<sup>†</sup>

§School of Computer Science, Peking University †ByteDance Seed

#### **Abstract**

We present MegaScale-MoE, a production system tailored for the efficient training of large-scale mixture-of-experts (MoE) models. MoE emerges as a promising architecture to scale large language models (LLMs) to unprecedented sizes, thereby enhancing model performance. However, existing MoE training systems experience a degradation in training efficiency, exacerbated by the escalating scale of MoE models and the continuous evolution of hardware.

Recognizing the pivotal role of efficient communication in enhancing MoE training, MegaScale-MoE customizes communication-efficient parallelism strategies for attention and FFNs in each MoE layer and adopts a holistic approach to overlap communication with computation at both inter- and intra-operator levels. Additionally, MegaScale-MoE applies communication compression with adjusted communication patterns to lower precision, further improving training efficiency. When training a 352B MoE model on 1,440 NVIDIA Hopper GPUs, MegaScale-MoE achieves a training throughput of 1.41M tokens/s, improving the efficiency by 1.88× compared to Megatron-LM. We share our operational experience in accelerating MoE training and hope that by offering our insights in system design, this work will motivate future research in MoE systems.

CCS Concepts: • Computer systems organization  $\rightarrow$  Cloud computing; • Computing methodologies  $\rightarrow$  Machine learning; • Networks  $\rightarrow$  Data center networks.

*Keywords:* Mixture-of-experts, distributed training, computation-communication overlap

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. EUROSYS '26, April 27–30, 2026, Edinburgh, Scotland Uk

@ 2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.

ACM ISBN 979-8-4007-2212-7/26/04...\$15.00 https://doi.org/10.1145/3767295.3769325

#### **ACM Reference Format:**

Chao Jin, Ziheng Jiang, Zhihao Bai, Zheng Zhong, Juncai Liu, Xiang Li, Ningxin Zheng, Xi Wang, Cong Xie, Qi Huang, Wen Heng, Yiyuan Ma, Wenlei Bao, Size Zheng, Yanghua Peng, Haibin Lin, Xuanzhe Liu, Xin Jin, Xin Liu. 2025. MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production. In *EuroSys '26, April 27–30, 2026, Edinburgh, UK*. In ACM, New York, NY, USA, 17 pages. https://doi.org/10.1145/3767295.3769325

#### 1 Introduction

As the size of Large Language Models (LLMs) [7, 18, 49] grow, so does the scale of their training regimes. The escalation in training scale has made efficiency improvements not just desirable but crucial [19]. As a company building AI products for billions of users, we remain committed to training LLMs with hundreds of billions of parameters on thousands of GPUs. Consequently, even marginal gains in training efficiency can significantly reduce computational resource consumption and training time, directly influencing the feasibility and sustainability of developing state-of-the-art LLMs.

Within the landscape of LLM architectures, Mixture-of-Experts (MoE) models stand out for their sparse activation [7, 10, 18, 46], which dynamically routes input tokens to a selected set of specialized network components, known as *experts*, rather than to all parameters. This design leads to sub-linear scaling of FLOPs required as the model size increases, thereby significantly reducing the computational cost. Recent industrial advancements [2, 3, 9, 27, 40] have demonstrated the potential of MoE models, achieving an order-of-magnitude reduction in training cost compared to dense models with equivalent model quality.

Despite the lower training costs of MoE models, we observe a critical performance bottleneck during training from a systems perspective—communication. For instance, when training an internal model on NVIDIA Hopper GPUs, communication accounts for 43.6% of the total time during the forward pass and 32% over the entire training process. Two primary factors contribute to this bottleneck. First, MoE models inherently introduce more communication overhead.

<sup>\*</sup>Equal contribution.

<span id="page-1-0"></span>![](_page_1_Figure_2.jpeg)

Figure 1. Evolution of NVIDIA GPUs.

Compared to dense model training, MoE model training requires distribution across more GPUs for model parallelism due to its larger parameter size. Second, enabling sparse computation requires two extra all-to-all communications in both the forward and backward passes to dispatch and aggregate tokens, respectively, which hinders ongoing computation.

Moreover, as hardware advances, the imbalance between computation and communication becomes increasingly pronounced, with communication overhead growing more dominant. Alongside improvements in model architectures, hardware capabilities have evolved rapidly, with GPUs achieving significantly higher processing speeds (Figure 1). Concurrently, reductions in training precision have been adopted to enhance efficient and cost-effective training [27, 38]. These trends lead to a scenario where the raw computation time decreases, making the relative impact of communication overhead a more critical bottleneck. For instance, simply extending existing tensor parallelism to multi-node setups has been observed to push communication overhead beyond 50% in certain cases. As a result, optimizing communication is essential for sustaining and improving the scalability of MoE model training, particularly in distributed environments where frequent data synchronization across multiple GPUs is required.

In this paper, we present the design, implementation, and operational experience of MegaScale-MoE, a production system optimized for efficient large-scale MoE training. By meticulously addressing the communication bottleneck, MegaScale-MoE strives to push the boundaries of MoE training, achieving significant improvements in performance and efficiency. Based on the insight that the key architectural distinctions between MoE and dense models are intra-layer, which is the primary source of the communication overhead, MegaScale-MoE confines each MoE layer to within a single node, utilizing high-bandwidth NVLink. Our analysis (§3) and evaluation (§6) show that despite the cross-node expert parallelism common in existing systems [15, 27], our approach effectively scales MoE training to models of several hundred billion parameters on thousands of GPUs.

Specifically, MegaScale-MoE addresses the communication problem in MoE training from three key aspects. First, MegaScale-MoE reduces the communication volume by customizing parallelism strategies for the attention and FFN modules in each MoE layer. We compare the parallelism

strategies in existing LLM training frameworks, comprehensively considering their impact on large-scale training, including the communication volume and whether communication can be effectively overlapped (i.e., whether it lies on the critical path). Based on this analysis, we select the optimal combination of parallelism strategies for MoE training.

Second, MegaScale-MoE fully overlaps communication with computation at the operator level. MegaScale-MoE partitions the forward and backward passes of each MoE layer into distinct computation and communication operators. For inter-operator overlap, MegaScale-MoE employs a holistic scheduling strategy that carefully reorders communication and computation operators during both forward and backward propagation, hiding communication within independent computations. This approach also optimizes GPU memory usage. MegaScale-MoE utilizes selective activation rematerialization, retaining only a subset of activations in GPU memory during the forward pass, and recomputing or recommunicating to obtain the required activations during the backward pass. With this holistic scheduling, MegaScale-MoE effectively hides the rematerialization overhead, achieving comparable performance while storing only half of the activations.

To overlap communication on the critical paths, MegaScale-MoE employs a fine-grained approach that splits communication into tiles and aligns with the GPU compute pattern, fusing these tile-level communications into the compute kernels. For MoE models with token dispatch, MegaScale-MoE fuses an efficient local scatter operation into the kernel and reorganizes the computation tasks along the scattered dimension to mitigate communication bottlenecks from multiple data sources. This fine-grained overlap occurs within each node, leveraging the high-bandwidth connectivity between GPUs.

Third, MegaScale-MoE leverages communication compression to further enhance MoE training efficiency. Specifically, for widely-used BF16 mixed-precision training, MegaScale-MoE reduces the inter-node parameter synchronization precision from FP32 to BF16, halving the associated overhead. In FP8 training, MegaScale-MoE replaces BF16 reduce-scatter with FP8 communication, incorporating tailored quantization strategies and FP32 reduction to decrease communication volume while preserving convergence stability.

MegaScale-MoE is deployed in our datacenters to train MoE models for our products. Compared to the state-of-the-art open-source LLM training framework, Megatron-LM [48], MegaScale-MoE achieves up to 1.88× higher MFU (Model FLOPs Utilization) when training a 352B MoE model on 1,440 NVIDIA Hopper GPUs. With comprehensive communication optimizations, MegaScale-MoE powers large-scale training in our production, efficiently scaling to trillions of parameters and thousands of GPUs while saving millions of GPU hours.

<span id="page-2-1"></span>![](_page_2_Figure_2.jpeg)

Figure 2. Mixture-of-Experts (MoE) layer.

# 2 Background

# 2.1 Mixture-of-Experts for Transformer

The Mixture of Experts (MoE) mechanism is an advanced approach designed to boost the performance of Transformer [\[51\]](#page-15-5) models, which are increasingly pivotal in the realm of LLMs [\[2,](#page-13-0) [7,](#page-14-0) [18,](#page-14-1) [27\]](#page-14-5). It extends the Transformer architecture by integrating multiple expert networks within the feed-forward network (FFN) component. As illustrated in Figure [2,](#page-2-1) MoE models dynamically route input tokens to the most relevant experts based on their characteristics. This routing is managed by a trainable gating mechanism that selects the best-suited experts for each token. This architectural innovation enables MoE models to scale in capacity without a proportional increase in inference costs, as only a subset of experts is activated for each input.

#### 2.2 Large-scale LLM Training

Training large language models at scale on tens of thousands of GPUs is a complex system engineering challenge that requires multiple systems techniques. To distribute the training workload, a combination of parallelism strategies such as data, tensor, and pipeline parallelism is necessary [\[19,](#page-14-2) [43,](#page-15-6) [48\]](#page-15-4), as each approach has limitations that prevent relying on a single method for effective scaling.

Data parallelism uniformly distributes the training data across all devices, with each device replicating the model parameters and optimizer states. To synchronize the parameters after each training iteration, data parallelism performs an all-reduce communication operation. Zero Redundancy Optimizer (ZeRO) [\[41\]](#page-15-7) improves over data parallelism by distributing model states across all participating devices. ZeRO unfolds across three progressive stages, each designed to increasingly conserve memory, though this comes with the trade-off of elevated communication.

Tensor parallelism distributes compute-intensive tensor operations over multiple devices, enabling parallel computation and significantly accelerating the training process. The specific partitioning strategy and the dependencies among operators within the model dictate that tensor parallelism may necessitate gathering split inputs (all-gather) or merging outputs (reduce-scatter). In LLM training, operators like

<span id="page-2-2"></span>![](_page_2_Figure_11.jpeg)

Figure 3. Different parallelism strategies for self-attention. "TP" denotes partitioning along the dimension of hidden size, while "SP" denotes partitioning along the dimension of sequence length.

LayerNorm and Dropout, though less compute-intensive, require substantial activation memory. To tackle this problem, a variant of tensor parallelism known as sequence parallelism [\[20\]](#page-14-7) is proposed, which partitions these operators along the dimension of sequence length. For long-context training, several works [\[1,](#page-13-2) [16,](#page-14-8) [48\]](#page-15-4) apply sequence parallelism or tensor parallelism to different operators in self-attention. Figure [3](#page-2-2) illustrates the mainstream parallelism strategies for attention, namely tensor, sequence, and context parallelism (TP, SP, and CP), which we analyze in [§3.1.](#page-3-0)

Pipeline parallelism enhances efficiency by dividing model layers into stages that are processed on different devices, enabling pipelined execution. Each batch is split into several micro-batches for this purpose. To minimize pipeline bubbles, various scheduling strategies have been developed, e.g., GPipe [\[14\]](#page-14-9), PipeDream 1F1B [\[33\]](#page-14-10) and Interleaved 1F1B [\[34\]](#page-14-11), etc. Megatron-LM adopts Interleaved 1F1B pipeline scheduling, further dividing each stage on one device into multiple virtual stages to reduce the pipeline bubble rate.

Expert parallelism is tailored for training MoE models by distributing experts across multiple devices, alleviating memory pressure and enabling parallel processing. To efficiently assign tokens to the appropriate experts and retrieve their outputs, all-to-all communication is typically employed.

# <span id="page-2-0"></span>3 Communication-Efficient Parallelism

With the rise of MoE models and the evolution of hardware compute capabilities, communication overhead has become increasingly critical in MoE training in production. In this section, we delve into the parallelism strategies employed to reduce communication volume and meet other training requirements, such as high GEMM (General Matrix Multiplication) efficiency.

Figure [4](#page-3-1) shows the design space of parallelism strategies for large-scale MoE training, excluding the outermost data

<span id="page-3-1"></span>![](_page_3_Figure_2.jpeg)

**Figure 4.** Design space for large-scale MoE training.

parallelism. We start with inter-node parallelism. Expert parallelism alleviates memory pressure from MoE models' large parameter size by distributing experts across nodes but incurs per-layer cross-node communication, harming training efficiency. Similarly, tensor parallelism's high communication overhead makes it more efficient to limit TP to a single node. Following prior work [19], we adopt pipeline parallelism to distribute model parameters, reduce communication, and overlap communication of different micro-batches.

Prior large-scale MoE training systems, such as Megatron-LM [48] and DeepSpeed-MoE [40], incorporate tensor parallelism to scale up training by partitioning the model parameters within the node. However, in our practice, we observe two issues with this approach: (1) TP partitions the expert dimension, which negatively impacts GEMM efficiency; and (2) TP introduces significant communication overhead, which remains constant as the parallelism size increases, eventually causing communication to exceed computation on modern hardware.

To address these issues, we tailor parallelism strategies for MoE model components. For feed-forward networks (i.e., experts), we replace tensor parallelism with expert parallelism and use custom communication modes optimized for varying top-k and expert sizes, ensuring communication overhead stays lower than tensor parallelism. For other components, we apply sequence parallelism, partitioning along the sequence dimension instead of the batch dimension, allowing scaling without increasing global batch size. This also reduces communication on critical paths compared to tensor parallelism. The additional memory and DP communication overhead remain manageable due to the parameter asymmetry across components. We detail the rationale and analysis of this intra-node parallelism strategy in the following sections. Table 1 lists the key symbols.

#### <span id="page-3-0"></span>3.1 Sequence Parallelism for Attention

Due to the inherent parallelizability of the expert components in MoE models, most prior work on MoE training [22, 40] focuses on optimizing expert parallelism, while data parallelism (DP) is typically applied to the non-MoE components such as attention. However, when scaling up

<span id="page-3-2"></span>

| Symbol         | Description                                                                |
|----------------|----------------------------------------------------------------------------|
| $\overline{b}$ | micro-batch size                                                           |
| S              | sequence length                                                            |
| h              | hidden dimension size                                                      |
| n              | model parallelism (TP, SP, or EP) size                                     |
| m              | the ratio between the number of query heads and<br>that of key-value heads |
| k              | number of experts that each token is routed to                             |

**Table 1.** Description of symbols.

MoE training, this approach proves insufficient due to the  $n\times$  activation memory consumption. This issue arises because DP splits the batch dimension both across and within nodes. Compared to other intra-node parallelism strategies shown in Figure 4, applying DP to attention forces each GPU within a node to process one micro-batch simultaneously, increasing the activation size by  $8\times$ , which often results in out-of-memory issues.

To enable scalable MoE training, implementing intra-node parallelism for the attention module is crucial. Tensor parallelism (TP) is commonly employed to parallelize attention operations within nodes. However, it introduces inevitable communication costs due to all-gathering and reduce-scattering activations along the critical path. With the increasing gap between computational FLOPs and communication bandwidth, we find that the TP communication overhead can even surpass the computation time of self-attention. This communication-dominated bottleneck limits the ability to overlap communication and computation, ultimately reducing training efficiency.

We adopt sequence parallelism (SP), as proposed in DeepSpeed-Ulysses [16], to scale MoE training and effectively reduce communication along the critical path. SP is commonly used in long-context training to address memory challenges associated with long inputs. We find it also works well in large-scale MoE training. First, it significantly reduces communication overhead compared to TP, especially when using grouped-query attention [4]. Second, while it introduces some parameter redundancy and increased communication overhead during parameter synchronization, the unique characteristics of MoE models make these trade-offs manageable and acceptable.

**Communication efficiency.** When utilizing TP, the communication volume in attention is

<span id="page-3-4"></span><span id="page-3-3"></span>
$$2bsh(n-1)/n. (1)$$

With SP, the communication volume decreases to

$$2bsh(n-1)/n \times (2+2/m)/n,$$
 (2)

where *m* represents the ratio between the number of query heads and that of key-value heads. Assuming the model is trained on an NVIDIA Hopper GPU workstation with an NVLink domain of size 8, the communication latency for sequence parallel attention can be reduced to about one-fourth of that required by tensor-parallel attention.

<span id="page-4-0"></span>![](_page_4_Figure_2.jpeg)

**Figure 5.** Hierarchical communication for parameter synchronization in SP attention.

**Data communication & memory overhead.** A notable difference between SP and TP attention is how parameters are distributed across devices: TP shards the attention weights, while SP replicates them. This raises the concern about the potential increase in communication overhead for synchronizing gradients and parameters. Counterintuitively, given the intra- and inter-node bandwidth asymmetry and the adoption of hierarchical communication operations in modern communication libraries [35] as shown in Figure 5 and analyzed in Appendix A.1, although SP attention requires synchronization of  $n \times$  more parameters compared to TP attention, the difference in communication overhead is minimal in practical scenarios.

On the other hand, the additional GPU memory consumption introduced by SP attention is minimal in MoE training. For large-scale MoE models with tens to hundreds of experts, the majority of GPU memory is consumed by the expert parameters. Our experiments, detailed in §6.2, confirm that the extra parameter synchronization and memory overhead of SP attention remain manageable.

Balanced vs. imbalanced. In addition to the Ulysses-style SP attention, we also explored other forms, including context parallelism (CP) [1], which partitions all activations along the sequence dimension. CP attention, however, faces workload imbalance due to causal masking in attention, as each token only attends to previous tokens. To mitigate this, we attempted the zigzag strategy by grouping the head and tail partitions of the sequence on the same GPU, although achieving perfect balance remains challenging. Consequently, in large-scale training, the entire training process is often constrained by the most imbalanced data batch. Moreover, this imbalance disturbs the training pipeline, thereby reducing overall training efficiency.

<span id="page-4-1"></span>![](_page_4_Figure_7.jpeg)

<span id="page-4-2"></span>**Figure 6.** Communication-efficient expert parallelism. *e* represents the number of tokens routed to the worker.

![](_page_4_Figure_9.jpeg)

**Figure 7.** Comparison of AG, RS, and A2A for token dispatch.

# 3.2 Expert Parallelism for Feed-forward Network

In the choice of parallelism strategies for the feed-forward network component, expert parallelism (EP) consistently outperforms tensor parallelism. TP partitions the hidden dimension of each expert, reducing GEMM efficiency, whereas EP maintains full expert computation on each device. Theoretically, the communication cost for EP is

$$2k/n \times bsh(n-1)/n, \tag{3}$$

while for TP it is

<span id="page-4-4"></span><span id="page-4-3"></span>
$$2bsh(n-1)/n. (4)$$

Although their relative efficiency depends on the ratio k/n, we design an adaptive communication strategy for different top-k values to minimize the communication volume of EP.

**Efficient communication pattern.** Figure 6 compares the typical EP implementation with MegaScale-MoE's approach. The standard EP implementation requires two all-to-all communications for token dispatch and aggregation. Additionally, a scatter operation may be required before sending and after receiving tokens to ensure that tokens assigned to the same expert reside in a contiguous memory space.

When the top-k value exceeds n, we replace traditional all-to-all communication with all-gather and reduce-scatter. First, an all-gather operation collects tokens from all workers. Then, a local scatter operation discards unneeded tokens, retaining only those required by the experts on the current worker. After expert computation, the tokens are assembled into a complete tensor. This approach enables a gather operation before communication, followed by a reduce-scatter to produce the final result, ensuring that EP's communication overhead remains equal to or lower than TP's.

In practical training, all-to-all communication is less efficient than all-gather and reduce-scatter, as it requires

<span id="page-5-0"></span>![](_page_5_Figure_2.jpeg)

(a) Forward pass of a MoE layer.

![](_page_5_Figure_4.jpeg)

(b) Backward pass snippet with activation rematerialization.

Figure 8. Selective activation rematerialization.

each worker to communicate with all others, whereas all-gather and reduce-scatter follow a ring-based communication pattern with only neighboring workers. As shown in Figure 7, the communication time for these three operations in Mixtral-8×7B reveals that when top-k > 6, the all-gather-based EP implementation is more efficient.

Efficient operators. Instead of using torch. scatter\_add and torch. gather for tensor scattering and gathering like Megatron-LM, we develop efficient scatter and gather operators directly using CUDA. Based on the token routing results, we pre-calculate the mapping from each row of the input tensor (representing a token) to the corresponding row in the output tensor. The scatter and gather operators then perform data transfers efficiently according to this mapping.

Load balance. A well-known challenge in MoE model training is load balancing across experts [22, 26]. To address this, we use auxiliary loss and token dropping to balance the workload across GPUs within each node. Similar to DeepSeek-V2 [26], we treat the experts placed on the same GPU as a group and calculate the balance loss and computational capacity for each device rather than for each individual expert.

# 4 Communication-computation Overlap

After optimizing parallelism strategies to minimize communication volume, we further reduce the communication overhead to nearly zero using comprehensive communication-computation overlapping techniques. Training large models

<span id="page-5-1"></span>

| Activation              | Shape                    | Obtained From                                |
|-------------------------|--------------------------|----------------------------------------------|
| hidden                  | [b, s/n, h]              | # Input                                      |
| ln1_out                 | [b, s/n, h]              | # RMSNorm(hidden)                            |
| qkv                     | [b, $s/n$ , $h(1+2/m)$ ] | <pre># MatMul(ln1_out, qkv_weight)</pre>     |
| q_rope                  | [b, s/n, h]              | <pre># RopeEmbedding(q)</pre>                |
| k_rope                  | [b, s/n, h/m]            | <pre># RopeEmbedding(k)</pre>                |
| qkv_a2a                 | [b, s, $h(1+2/m)/n$ ]    | <pre># All-to-All(q_rope, k_rope, v)</pre>   |
| attn                    | [b, s, h/n]              | <pre># SelfAttention(qkv_a2a)</pre>          |
| attn_a2a                | [b, s/n, h]              | # All-to-All(attn)                           |
| attn_out                | [b, s/n, h]              | <pre># MatMul(attn_a2a, out_weight)</pre>    |
| ln2_in                  | [b, s/n, h]              | <pre># Add(hidden, attn_out)</pre>           |
| ln2_out                 | [b, s/n, h]              | <pre># RMSNorm(ln2_in)</pre>                 |
| ln2_out_ag              | [b, s, h]                | # All-Gather(ln2_out)                        |
| ffn_in                  | [b*s*k/n, h]             | # Scatter(ln2_out_ag)                        |
| fc1_out                 |                          | <pre># GroupedGEMM(ffn_in, fc1_weight)</pre> |
| fc3_out                 | [b*s*k/n, fh]            | <pre># GroupedGEMM(ffn_in, fc3_weight)</pre> |
| fc2_in                  | [b*s*k/n, fh]            | # SiLU(fc1_out, fc3_out)                     |
| fc2_out                 | [b*s*k/n, h]             | <pre># GroupedGEMM(fc2_in, fc2_weight)</pre> |
| fc2_out_rs              | [b, s, h]                | # Gather(fc2_out)                            |
| ffn_out                 | [b, s/n, h]              | <pre># Reduce-Scatter(fc2_out_rs)</pre>      |
| <pre>hidden(next)</pre> | [b, s/n, h]              | <pre># Add(ln2_in, ffn_out)</pre>            |

Figure 9. Activation shapes in rematerialization.

involves integrating various techniques, which increases the complexity of communication overlap. For instance, at any given moment, the device might concurrently handle computation and communication kernels, overlap PP and DP communications, and manage data transfers between the device and host. Existing frameworks like Megatron-LM assemble attention and FFN modules into MoE layers and rely on the torch.autograd package for backward propagation, which limits the flexibility of communication overlap. In contrast, MegaScale-MoE decomposes the attention and FFN modules of each MoE layer into operators that run as GPU kernels, enabling fine-grained communication overlap through flexible scheduling.

# 4.1 Inter-operator Overlap

We overlap communication operators with independent computation operators by executing them asynchronously on different CUDA streams. To achieve optimal performance during the training process, we adopt a specifically hand-tailored, holistic scheduling strategy.

Holistic scheduling. From the caller's perspective, we implement a unified macro module to execute the entire MoE layer's forward and backward passes, thereby expanding our scheduling flexibility. For instance, during the backward pass, various communication operators can be overlapped with dependency-free computations, such as activation recomputation, to improve efficiency. From the runtime perspective, a key challenge is efficiently managing concurrent communication tasks by resolving resource conflicts to prevent blocking and maximize throughput. This requires careful coordination, such as determining the number of SMs allocated to each communication operator, to minimize interference and optimize overall throughput.

**Selective activation rematerialization.** The holistic scheduling strategy also helps reduce memory usage without compromising training speed. Compared to dense models with equivalent computational requirements, MoE models exert significantly higher memory pressure during training due to

<span id="page-6-0"></span>![](_page_6_Figure_2.jpeg)

Figure 10. Fine-grained intra-operator communication-computation overlap.

their parameter count being several times larger. In addition to employing ZeRO optimizations [41] to eliminate redundant optimizer states across DP groups, we further optimize memory usage through selective activation rematerialization. This approach reduces activation memory requirements by re-performing computation and communication operators that can be overlapped with other necessary operators.

Figure 8a illustrates the forward pass of a Mixtral [18] MoE layer and highlights key activations produced during this process. MegaScale-MoE strategically retains activations that are computationally expensive to recompute, while recalculating others generated by memory-intensive operations or communication operations. This minimizes dependencies on backward computation, enabling rematerialization operations to overlap with other computations and communications, avoiding delays in the critical path. For example, as shown in Figure 8b, the backward pass of the GroupedGEMM operator for FC2 requires the activation fc2\_in and the gradient of fc2\_out (denoted as Δfc2\_out) as inputs. MegaScale-MoE recomputes fc2\_in and overlaps this operator with gradient communication (i.e., all-gather for Δffn\_out). Similarly, ffn\_in is obtained through re-performing RMSNorm and allgather, with these operators hidden within the preceding communication and the FC2 GroupedGEMM, respectively. MegaScale-MoE also places the weighted sum of ffn\_out immediately after the SwiGLU [45] activation function to eliminate the need to store ffn\_out. This reordering ensures computational consistency by avoiding operators that cross non-linear boundaries.

Figure 9 illustrates the shapes of the key activations produced during forward propagation, with the highlighted activations retained for backward propagation. Let the model parallelism size within one MoE layer be n and the intermediate hidden size of one expert be fh. The total activation of a single MoE layer is

$$(2n + 2k + 3kf + 12 + 5/m)bsh/n$$
,

which we have reduced to

$$(2kf + 4 + 2/m)bsh/n$$
.

MegaScale-MoE reduces the activation memory by  $\sim 50\%$  while maintaining the same training speed.

# 4.2 Intra-operator Overlap

Although inter-operator overlap effectively hides communication latency, squeezing all bubbles in the execution timeline remains non-trivial—especially in the forward pass, where no rematerialization or gradient computation operators exist to overlap with communication. Some forward operators directly depend on communication, such as token dispatch for expert computation, making overlap impossible unless another micro-batch is introduced, which increases memory pressure.

A widely adopted solution [19, 50, 52] is to decompose operators into smaller parallel ones to enable pipelining by executing them on separate CUDA streams. However, this approach introduces non-negligible overhead: (*i*) complex stream control, involving host interference and causing random bubbles due to the non-deterministic feature of CPU control; (*ii*) imperfect tail computation, increasing overall computation latency.

To address the above issues, we adopt intra-operator overlap to parallelize communication and computation operators with direct dependencies. The core idea is to fuse these operators and break down the workloads into tiles. Following prior work [5, 17, 53, 56], we implement barriers in device memory between communication and computation operators. These barriers enable fine-grained tile-level notifications and remove the need for host interference, further improving training performance. We implement two types of kernels, overlapping with GEMMs and overlapping with MoE GroupedGEMMs, for the attention and FFN modules, respectively.

Overlapping with GEMMs. We first introduce the intraoperator communication-computation overlap for GEMM kernels. Specifically, we implement all-to-all(A2A)+GEMM and GEMM+A2A kernels for Output and QKV Projections in SP attention, respectively, where X+Y means Y executed after X. Figure [10](#page-6-0) shows the data flow and overlapping pattern in A2A+GEMM. The GEMM on local data and communication for remote data starts simultaneously. We leverage dedicated GPU copy engines for data transfer, ensuring that all SMs (streaming multiprocessors) are fully utilized for computation. Once a remote data tile arrives at local memory, a signal notifies the GEMM kernel to continue its computation on the arrived tile. For GEMM+A2A, the all-to-all operation is fused into the GEMM kernel. Each tile of GEMM computation ends with a remote data transfer that writes the output data tile to remote ranks. We also implement all-gather+GEMM and GEMM+reduce-scatter kernels for tensor parallelism, which are similar to A2A+GEMM and GEMM+A2A.

For A2A+GEMM and GEMM+A2A, we allocate a small number of SMs for communication as all-to-all is more complex than all-gather and reduce-scatter. The number of SMs for communication is tuned to make communication and computation exhibit similar latency. Moreover, multiple ranks may simultaneously read from or write to the same device, potentially causing contention in NVLink. To mitigate this, we apply swizzling [\[5,](#page-13-4) [53,](#page-15-11) [56\]](#page-15-12) to reorder tile communication and computation so that the arrival of communication tiles aligns with the pace of computation tiles.

Overlapping with GroupedGEMMs For expert parallelism with token dispatch and combine, we aim to overlap communication with GroupedGEMMs. We implement two types of overlapping kernels: all-gather+scatter+GroupedGEMM and GroupedGEMM+gather+reduce-scatter. Unlike the overlapping techniques for GEMM kernels, MoE GroupedGEMMs require token shuffling (scatter/gather). As a result, each computation tile may depend on tokens from multiple ranks. To effectively overlap computation with communication, we sort the token order to minimize the number of dependent ranks for each computation tile. Additionally, since each tile has its own dependencies, the signal control for each tile varies depending on the MoE routing, which is determined dynamically.

In detail, for AG+scatter+GroupedGEMM, we reorder tokens along the sequence dimension based on their routed expert index. Then, for each expert, we sort the routed tokens according to their source rank index. Finally, we slice the sorted sequence into blocks and perform GroupedGEMM using a sequence of computation tiles. Specifically, as shown in Figure [10c](#page-6-0), we fuse the local scatter into the kernel by selecting rows of input data based on the index mapping. The GroupedGEMM computation for each expert is divided into tiles, with each tile depending on only a subset or even a single source rank. This reduces the overall waiting time for

<span id="page-7-0"></span>![](_page_7_Figure_6.jpeg)

Figure 11. DP communication compression.

each computation block, avoids redundant loading of expert parameters, and improves the overlap between computation and communication tiles.

# <span id="page-7-1"></span>5 Communication Compression

We further reduce communication overhead by applying communication compression. To maintain convergence stability, mixed-precision training frameworks typically transfer tensors awaiting reduction in higher precision, such as FP32, to ensure more accurate accumulation. A common example of this is gradient reduce-scatter in data parallelism.

DP communication compression. As MoE model parameters increase, so does the communication overhead for parameter and gradient synchronization in data parallelism. Prior work has explored gradient compression to mitigate this cost. In our BF16 mixed-precision training, we carefully apply FP32-to-BF16 precision reduction for gradient synchronization, balancing efficiency and convergence stability.

Specifically, as shown in Figure [11,](#page-7-0) we retain the main gradients in FP32 during local gradient accumulation in pipeline parallelism. After each model stage completes accumulation, instead of relying solely on reduce-scatter for gradient synchronization, we cast gradients to BF16 and perform all-to-all communication within the data parallel group to gather the required gradient shards, which are then locally aggregated in FP32. Our results show that this approach introduces negligible precision loss compared to directly performing reduce-scatter with FP32, while reducing gradient communication overhead by 50%.

This approach minimizes risk for two key reasons. First, it performs a one-time conversion of accumulated gradients to BF16 during communication, while the local gradient accumulation is maintained in FP32 precision. Second, instead of using ring-style reduce for BF16 gradient communication, it employs all-to-all communication, with the final reduction computed using FP32 summation. This design prevents precision loss that could arise from repeated accumulation of BF16 values in ring-based reductions.

We observe that casting large gradients and performing all-to-all communication increases peak memory consumption, potentially causing out-of-memory errors. To mitigate this, we develop a memory-efficient operator that in-places BF16 gradients into half of the FP32 input buffer while using

<span id="page-8-2"></span>

| Ī | Name          | #layers | h    | #heads | m  | $h_{ffn}$ | #experts | top-k |
|---|---------------|---------|------|--------|----|-----------|----------|-------|
|   | Internal-352B | 60      | 4096 | 32     | 4  | 14336     | 32       | 3     |
|   | Mixtral-8×7B  | 32      | 4096 | 32     | 4  | 14336     | 8        | 2     |
|   | Mixtral-8×22B | 56      | 6144 | 48     | 6  | 16384     | 8        | 2     |
| I | Hunyuan-Large | 64      | 6400 | 80     | 10 | 18304     | 16       | 1     |
|   | Phi-3.5-MoE   | 32      | 4096 | 32     | 4  | 6400      | 16       | 2     |
|   | DeepSeekMoE   | 28      | 2048 | 16     | 1  | 1408      | 64       | 6     |
|   |               |         |      |        |    |           |          |       |

Table 2. Model configurations in evaluation.

the remaining half as the output buffer for BF16 all-to-all communication, preventing peak memory growth.

Communication compression for FP8 training. In lowprecision FP8 training, the proportion of communication time increases due to reduced computation time. To mitigate communication overhead, we explore compressing communication volume using FP8 precision with appropriate quantization techniques. Currently, we apply communication compression in FP8 MoE training with tensor parallelism, focusing on reduction scenarios prone to overflow or underflow. For example, we adopt the E4M3 format (4-bit exponent and 3-bit mantissa) for all tensors. Similar to DP reduce-scatter compression, we replace BF16 TP reduce-scatter with FP8 all-to-all in forward propagation and perform reduction in FP32 precision. In the corresponding backward propagation, we apply FP8 all-gather for gradients. Notably, simply reducing precision leads to loss misalignment with BF16 training. To mitigate this, we apply per-token activation quantization for forward communication and per-channel quantization for backward communication. In backward propagation, we further group quantization along the token dimension using a small group size (e.g., 128).

#### <span id="page-8-0"></span>6 Evaluation

In this section, we present a comprehensive evaluation of MegaScale-MoE, covering overall training performance (§6.1), ablation studies of MegaScale-MoE's key optimizations (§6.2), and the effectiveness of the precision-communication co-design (§6.3). Table 2 lists the configurations of the MoE models used in our evaluation, detailing hidden size (h), FFN intermediate size ( $h_{ffn}$ ), number of experts, and top-k values. The evaluation is conducted on NVIDIA H800 GPUs unless otherwise specified, with the specifications provided in Table 4.

# <span id="page-8-1"></span>**6.1** Training Performance

MegaScale-MoE is built on top of Megatron-LM [48], a state-of-the-art open-source LLM training system that supports 3D parallelism strategies and is continuously updated to incorporate the latest optimizations from the community. Our evaluation uses the Megatron-LM on GitHub [32] with commit hash f1f03922, selected for its stability at the commencement of our experiments months ago. For fair comparison, we use the same global batch size for Megatron-LM

<span id="page-8-3"></span>

|               | #GPUs | Iteration | Throughput               | Training Time for |  |
|---------------|-------|-----------|--------------------------|-------------------|--|
| System        |       | Time (s)  | (tokens/s)               | 1T Tokens (days)  |  |
|               | 240   | 39.94     | 151.1k                   | 76.61             |  |
|               | 480   | 19.56     | 301.1k                   | 38.38             |  |
| Megatron-LM   | 720   | 13.70     | 430.5k                   | 26.88             |  |
|               | 960   | 10.82     | 550.2k                   | 21.23             |  |
|               | 1440  | 7.90      | 746.6k                   | 15.50             |  |
|               | 240   | 21.61     | 272.9k ( <b>1.81</b> ×)  | 42.41             |  |
|               | 480   | 11.83     | 498.6k ( <b>1.65</b> ×)  | 23.21             |  |
| MegaScale-MoE | 720   | 7.97      | 740.1k ( <b>1.72</b> ×)  | 15.64             |  |
|               | 960   | 6.12      | 963.8k ( <b>1.77</b> ×)  | 12.01             |  |
|               | 1440  | 4.19      | 1407.7k ( <b>1.88</b> ×) | 8.22              |  |

**Table 3.** Strong-scaling training performance for the 352B MoE model with NVIDIA H800 GPUs. The number in parentheses in the throughput column represents the speedup of MegaScale-MoE compared to Megatron-LM.

<span id="page-8-4"></span>![](_page_8_Figure_12.jpeg)

**Figure 12.** Weak-scaling training performance for the 352B MoE model with NVIDIA H800 GPUs.

and MegaScale-MoE and choose the optimal parallelism configurations for the two systems, respectively. Specifically, MegaScale-MoE employs SP attention and EP within each node, while Megatron-LM adopts TP within each node, with both systems configured with a PP size of 15. We tune the configuration of Megatron-LM to meet its requirement of a uniform TP size across all components. As discussed in §3.1, for Megatron-LM, a TP size of 1 leads to a prohibitive 8× activation memory (addressable only with slow recomputation via gradient checkpointing), while a TP size of 8 forces EP to operate across nodes, incurring more communication costs than PP. Notably, both systems in the evaluation enable the communication-computation overlap techniques from MegaScale [19] for data and pipeline parallelism. Therefore, the communication overhead mainly comes from intra-node model parallelism, e.g. TP, SP and EP. Sequence length is 8,192 and vocabulary size is 65,536.

Scalability. Table 3 compares the strong-scaling training performance of Megatron-LM and MegaScale-MoE on the 352B MoE model. We scale the number of GPUs while keeping the global batch size fixed at 720. Across all settings, MegaScale-MoE achieves 1.65–1.88× speedups over Megatron-LM. As the number of GPUs increases, the MFU (Model FLOPs Utilization) of MegaScale-MoE declines from 32.48% to 27.89%. This is expected, as the batch size is fixed and the number of micro-batches for each pipeline decreases with more GPUs, leading to more bubbles.

Figure 12 presents the weak-scaling training performance of Megatron-LM and MegaScale-MoE on the same model.

<span id="page-9-2"></span>![](_page_9_Figure_2.jpeg)

**Figure 13.** Performance breakdown of training Mixtral-8×7B on different GPUs.

<span id="page-9-1"></span>

| GPU  | Compute Cap-     | Memory Spec. |            | NVLink     |
|------|------------------|--------------|------------|------------|
| GFU  | ability (TFLOPS) | Cap. (GB)    | Bw. (TB/s) | Bw. (GB/s) |
| H800 | 989              | 80           | 3.4        | 400        |
| A100 | 312              | 80           | 2.0        | 600        |
| H20  | 148              | 96           | 4.0        | 900        |

**Table 4.** Specifications of different NVIDIA GPUs.

We scale the global batch size from 360 to 1,080 in proportion to the number of GPUs (from 480 to 1,440). MegaScale-MoE achieves a 1.74-1.79× training throughput compared to Megatron-LM. As the scale increases, Megatron-LM's throughput degrades by 2.74% due to increased communication overhead. In contrast, MegaScale-MoE exhibits near-linear scalability, with its throughput declining by only 0.2%, benefiting from comprehensive communication-computation overlap.

Performance breakdown on different GPUs. We conduct a deep dive into MegaScale-MoE to further understand the performance of training a MoE model in production environments. We train Mixtral-8×7B on 32 NVIDIA H800, H20, and A100 GPUs, respectively. The specifications of GPUs we used are listed in Table 4. We set the DP size as four, the TP size as eight for Megatron-LM, and the SP and EP size as eight for MegaScale-MoE. As shown in Figure 13b, across the four kinds of GPUs, MegaScale-MoE consistently outperforms Megatron-LM by up to 1.58× in MFU. Figure 13a demonstrates the iteration time breakdown of Megatron-LM and MegaScale-MoE. Exposed communication time represents the communication time that is not overlapped with computation operations. FlashAttention and GEMMs are the operations we count when calculating MFU. The performance gain primarily results from MegaScale-MoE's communicationefficient parallelism strategies and fine-grained overlapped communication.

Note that the MFU value decreases as GPU compute capability increases. This is because, unlike dense models, MoE models involve many memory-intensive operations like routing, local scatter, and gather, which remain time-consuming since memory bandwidth does not scale as quickly as compute capabilities. Additionally, GEMM efficiency declines with increasing compute capability, as it also relies on memory loading, constrained by memory bandwidth.

<span id="page-9-3"></span>

| Idx | Method                          | Normalized<br>Throughput | Δ    |
|-----|---------------------------------|--------------------------|------|
| 1   | baseline                        | 1                        |      |
| 2   | (1) with SP+EP                  | 1.13                     | +13% |
| 3   | (2) with inter-operator overlap | 1.22                     | +9%  |
| 4   | (3) with intra-operator overlap | 1.28                     | +6%  |

**Table 5.** Throughput improvement breakdown when training the 352B MoE model with 240 NVIDIA H800 GPUs and batch size is 720.

<span id="page-9-4"></span>![](_page_9_Figure_11.jpeg)

Figure 14. Parallelism efficiency for different models.

<span id="page-9-5"></span>![](_page_9_Figure_13.jpeg)

**Figure 15.** Parameter synchronization time under SP and TP attention.

#### <span id="page-9-0"></span>6.2 Ablation Study

We evaluate the effectiveness of the optimization techniques of MegaScale-MoE. First, we conduct an experiment about systematic breakdown by incrementally enabling each technique to isolate its contribution to the overall performance. Table 5 shows the throughput improvement breakdown with different optimizations when training the 352B MoE model on 240 GPUs with a global batch size of 720. The baseline is a version of MegaScale-MoE that adopts TP for both attention and FFNs and disables communication-computation overlap. First, by applying communication-efficient strategies-namely, SP for attention and EP for experts-we achieve an initial 13% throughput improvement over this baseline. We then target the primary bottleneck in large-scale MoE training: communication overhead. Our inter-operator and intra-operator overlap methods effectively hide these costs, further accelerating training by an additional 9% and 6%, respectively.

Following the systematic breakdown, we perform ablation studies on each component, varying a single setting at a time while keeping all others constant, to gain deeper insights into its behavior.

**Parallelism strategy.** We compare the training efficiency under various intra-node parallelism strategies using a single

<span id="page-10-0"></span>![](_page_10_Figure_2.jpeg)

**Figure 16.** Overlapped communication-computation time vs. non-overlapped time of each layer. M1-M6 represent the six models listed from top to bottom in Table 2; A2A, AG, and RS refer to all-to-all, all-gather, and reduce-scatter, respectively.

<span id="page-10-1"></span>![](_page_10_Figure_4.jpeg)

**Figure 17.** Ablation study of selective activation rematerialization (SAR).

node with eight NVIDIA H800-SXM GPUs. We denote parallelism strategies as X+Y, where X represents the parallelism strategy for attention, and Y corresponds to that for experts. The available parallelism strategies for attention include TP and our SP, whereas for experts, the choices are TP and EP. To isolate the performance benefits of optimized parallelism, we disable other system optimizations.

We measure the training MFU of one internal and five open-source MoE models with diverse model configurations as listed in Table 2. The global batch size is set to 32, and we adjust the number of layers for each model to fit within the GPU memory. Figure 14 shows that MegaScale-MoE's parallelism strategy, SP+EP, consistently outperforms the other three parallelism strategies, achieving 14.9%-32.9% higher MFU compared to TP+TP. The performance gains are attributed to two main factors. First, as discussed in §3, SP and EP effectively reduce the communication volume compared to TP, thereby decreasing communication overhead. Second, TP partitions the FFN module along the intermediate size dimension, which results in lower GEMM efficiency.

To provide a more comprehensive evaluation of the parallelism strategy, we also report the additional overhead introduced by the replicated attention parameters in SP. In terms of memory usage, SP incurs a 1.2%–5.4% higher memory footprint compared to TP, requiring 1.7%–8.1% more memory to store parameters, gradients, and optimizer states across all seven models. This overhead is manageable considering the significant performance gains achieved by SP.

For the parameter synchronization time, we follow largescale training setups and set the size of the TP or SP to 8, effectively parallelizing each layer within a single node. The

<span id="page-10-2"></span>![](_page_10_Figure_10.jpeg)

**Figure 18.** The training loss curve of MegaScale-MoE with DP communication compression.

attention parameter size on each GPU is varied from 384 MB to 1536 MB, while the FFN parameter size is fixed at 10 GB per GPU, reflecting typical real-world training setups. We run MegaScale-MoE with SP and TP attention, using 4 and 8 DP groups, which correspond to a total of 32 and 64 GPUs, respectively. Figure 15 shows that the synchronization times for SP and TP attention are consistently comparable, differing by only 0.3%–3.1%. This aligns with our hypothesis that SP and TP would exhibit similar performance characteristics in DP communication latency.

Intra-operator commmunication overlap. We then measure the duration of four key communication and the corresponding computation operators in the forward pass: (i) QKV Projection paired with all-to-all, (ii) all-to-all with Output Projection, (iii) all-gather with scatter and GroupedGEMM, and (iv) GroupedGEMM with gather and reduce-scatter, as depicted in Figure 8. Figure 16 demonstrates that across all six models, MegaScale-MoE achieves a 1.2–4.7× reduction in the combined time of communication and computation operators compared to the baseline lacking fine-grained overlap. And MegaScale-MoE reduces the training iteration time by 7.1%-12.9% due to intra-operator communication-computation overlap.

Selective activation rematerailization. We compare MegaScale-MoE to a baseline that disables selective activation rematerialization (No SAR), which stores all activations in GPU memory during training. We evaluate both methods by training Mixtral-8×7B and Mixtral-8×2B on 128 NVIDIA H800 GPUs. Figure 17 shows the memory usage breakdown and the training MFU. Compared to No SAR, MegaScale-MoE reduces activation memory consumption by 45.5% and 57.2% for the two models, respectively, resulting in overall

<span id="page-11-1"></span>![](_page_11_Figure_2.jpeg)

**Figure 19.** The loss curve of MegaScale-MoE in FP8 and BF16.

memory reductions of 21.3% and 35%, while maintaining the training performance difference within 0.5%.

Data parallelism communication compression. We validate the effectiveness of our communication compression technique by training a 7B MoE model using BF16 all-to-all DP communication and FP32 reduce-scatter communication, as described in §5. Figure 18 illustrates the training loss curves, which are nearly identical. This optimization compresses only the accumulated gradients of the batch and performs conversions between BF16 and FP32 exclusively during communication, introducing minimal risk.

# <span id="page-11-0"></span>6.3 Model Convergence

We evaluate model convergence with MegaScale-MoE. Figure 19 demonstrates the loss curves of training a 35B MoE model from scratch and continuing training a 176B MoE model from a checkpoint, with results shown for both BF16 and FP8 precision. MegaScale-MoE ensures stable convergence and consistent training loss across BF16 and FP8 formats.

# 7 Experience

In this section, we describe our deployment and operational experience of MegaScale-MoE.

Deployment experience. MegaScale-MoE has been deployed in our production environment and is responsible for the majority of large-scale MoE training tasks within our company. It enables the training of models with trillions of parameters, supports single training jobs scaling beyond 10,000 GPUs, with individual training tasks running for several months. By combining the aforementioned techniques, MegaScale-MoE minimizes idle communication time and optimizes memory usage in MoE training without compromising model performance, ultimately saving millions of GPU hours in large-scale MoE training. Figure 20 shows the model convergence from a real production job, which trains a proprietary MoE model with 200B parameters, 20B activated for each token. This job uses over 10,000 GPUs and lasts for months. The loss continues to converge with a stable training process.

<span id="page-11-2"></span>![](_page_11_Figure_11.jpeg)

**Figure 20.** The normalized training loss curve of a real production job on more than 10,000 GPUs for months, training a MoE model with 20B activated and 200B total parameters on multi-trillion tokens. Different colors indicate training restarts.

**FP8 training.** We have made extensive efforts to maintain the convergence stability of FP8 training. For example, we observe that the SwiGLU operator significantly expands the numerical range. To address this, we replace per-tensor quantization with higher-precision per-token quantization  $(1 \times h)$ . Additionally, since multiplying SwiGLU with the gating weight further amplifies the dynamic numerical range, we shift the gating weight multiplication back to after the FC2 output, reducing quantization errors.

Beyond ensuring training convergence, we introduce additional engineering optimizations. Existing FP8 training implementations [25, 50] store model parameters in BF16, requiring frequent FP8 conversion for GEMM computations, adding casting and transpose overhead. To address this, we use a multi-precision optimizer to store model parameters directly in FP8, while keeping main parameters in FP32 with separate buffers for different data types. This lowers memory consumption and halves parameter all-gather communication in data parallelism.

Scale up. When training MoE models, an intriguing engineering question arises: can we indefinitely scale the training size by increasing model parameters without raising computational load? This approach is impractical in tensor parallelism, as scaling up the model necessitates a higher TP degree to accommodate additional parameters. While increased TP reduces per-GPU computation, the communication overhead remains constant, as shown in Formula 1 and 4, leading to progressively longer communication times and reduced training efficiency. In other words, TP has inherent scalability limitations and often relies on high-speed intra-node links to mitigate communication delays.

In contrast, when scaling training with SP and EP, the communication volume decreases as the parallel size n increases, as shown in Formula 2 and 3. This implies that, in theory, this parallelism strategy can scale to significantly larger sizes. However, in practical hierarchical infrastructures, a critical challenge emerges: can this approach maintain training efficiency when scaling beyond the NVLink domain, where bandwidth drops to RDMA levels?

Formally, for a SwiGLU structure incorporating a MoE mechanism, the ratio between computation time and communication time is defined as:

$$comm\_time = \frac{2k \times bsh(n-1)/n/n}{bandwidth},$$
 (5)

comp\_time = 
$$\frac{3k \times bsh \times h_{ffn}/n}{peak}$$
. (6)

$$R = \frac{\text{comp\_time}}{\text{comm\_time}} \tag{7}$$

$$= 3/2 \times h_{ffn} \times \frac{bandwidth}{peak} \times n/(n-1)$$
 (8)

$$\approx 3/2 \times h_{ffn} \times \frac{bandwidth}{peak} \tag{9}$$

To sustain training efficiency, the FFN's computation time must exceed the communication time, ensuring effective overlap of communication overhead. Therefore, our goal is to maintain > 1, leading to two key insights:

- The value of is independent of the number of experts, top-, hidden dimension, parallelism size, or input size, providing flexibility in selecting algorithm parameters.
- is solely determined by the expert's intermediate dimension, computational peak, and communication bandwidth. Consequently, on fixed hardware, as long as the expert dimension is sufficiently large, the MoE model can be scaled while maintaining training efficiency from an engineering perspective.

Holistic vs. automatic. We have invested substantial engineering efforts in inter-operator communicationcomputation overlap, including determining operator execution order, concurrency of communication and computation, and SM allocation for communication. These manual interventions provide deeper insights into training dynamics, enabling targeted optimizations. As training progresses and experience accumulates, we seek to automate operator scheduling within the search space to optimize the training process at a fine-grained level and achieve optimal performance. We leave automatic optimization for future work.

MoE vs. dense model training. In our continued efforts to optimize MoE model training, we have identified several critical distinctions from the training of dense models. In a dense Transformer layer, optimization efforts are concentrated on self-attention and GEMMs. The former is often accelerated by techniques like FlashAttention [\[8\]](#page-14-18), while the latter, as a dense computation, generally achieves high utilization on the GPU's parallel processing units. In contrast, as shown in Figure [13a](#page-9-2), the combined runtime of attention and GroupedGEMM accounts for only about one-third of a layer's execution time. The remainder is consumed by communication and other operators. While MegaScale-MoE effectively addresses the communication overhead, we observe that the computational operators in MoE models, which are

inherently more complex than their dense counterparts, also introduce performance degradation. Specifically, they are a primary source of stragglers for three main reasons:

First, the intermediate dimension of each expert is smaller than the FFN layer in a dense model. To efficiently process computations for multiple experts concurrently, GroupedGEMM employs a single CUDA kernel for numerous small matrix multiplications. The resource usage of this kernel—including shared memory, L1 cache, and number of threads—is finely controlled via cuFuncSetAttribute. This granular control, however, can introduce synchronization delays. Second, due to the imbalanced number of tokens routed to each expert, the inputs and outputs for GroupedGEMM are dynamically shaped tensors. The frequent allocation and deallocation of these tensors exacerbate GPU memory fragmentation. Third, the MoE gating mechanism involves a multitude of small operators for tasks like calculating routing scores and communicating routing decisions. Jitter in CPU performance can delay the launch of these kernels to the point where the launch latency exceeds their actual execution time on the GPU, creating pipeline bubbles.

# 8 Related Work

Large model training. LLM research has led to the development of scalable, efficient, and robust training techniques [\[19,](#page-14-2) [43,](#page-15-6) [48,](#page-15-4) [54\]](#page-15-13) to meet the substantial computational demands of these models. DeepSpeed [\[43\]](#page-15-6) features the Zero Redundancy Optimizer (ZeRO) [\[41,](#page-15-7) [42,](#page-15-14) [44\]](#page-15-15), which shards model parameters, gradients, and optimizer states across participating GPUs in data parallelism, enabling the scaling of LLMs with manageable memory consumption. Megatron-LM [\[48\]](#page-15-4) focuses on intra-layer model parallelism techniques, partitioning the parameters and computation of each layer. Pipeline parallelism assigns the parameters and computation of a contiguous subset of layers to each GPU[\[14,](#page-14-9) [33\]](#page-14-10), breaks a batch into micro-batches, and processes the microbatches in a pipelined fashion. MegaScale [\[19\]](#page-14-2) shows how combining tensor, pipeline, and data parallelism can be an efficient strategy to train large multi-billion parameter models at unprecedented scale.

Mixture-of-Expert training. To address the computational challenges of training advanced neural networks, the machine learning field has increasingly adopted Mixture-of-Experts architectures. Subsequently, a number of deep learning frameworks have been proposed for training or running inference on MoEs on multi-GPU clusters. DeepSpeed-MoE [\[40\]](#page-15-2) significantly reduces training costs through model architecture designs and compression techniques. Hetu-MoE [\[36\]](#page-14-19) utilizes a hierarchical all-to-all communication strategy to achieve performance speedup. SE-MoE [\[47\]](#page-15-16) distinguishes itself by focusing on scalable and efficient training with heterogeneous resources like CPU memory and SSDs. FasterMoE [\[13\]](#page-14-20) introduces a comprehensive suite of optimizations such as dynamic shadowing, fine-grained scheduling, and congestion-avoiding expert selection strategies. Janus [\[30\]](#page-14-21) proposes a data-centric paradigm shift for MoE models, aiming to lower communication demands and boost training efficiency. Tutel [\[15\]](#page-14-6) offers a dynamic solution for MoE models, employing adaptive parallelism and pipelining. However, its dynamic parallelism switching and hierarchical all-to-all can cause significant overheads for models with hundreds of billions of parameters. To avoid such overhead, latest MoE training systems [\[26,](#page-14-14) [27\]](#page-14-5) use auxiliary loss or routing bias for load balancing and limit cross-node token dispatch. By mapping each MoE layer to intra-node, MegaScale-MoE eliminates cross-node token dispatch.

Recently, DeepSeek-V3 [\[27\]](#page-14-5) introduced two key optimizations for training production-scale MoE models: DeepEP, for high-performance cross-node all-to-all communication, and DualPipe, for overlapping communication with computation. Due to the relatively low cross-node InfiniBand bandwidth, DeepEP limits the token dispatch to a maximum of 4 nodes to maintain a constant cross-node communication volume, restricting its routing flexibility. In contrast, MegaScale-MoE places each MoE layer intra-node to ensure efficient routing to any top-k experts. DualPipe leverages pipeline parallelism for communication-computation overlap across different micro-batches, which requires storing 2× the model parameters. In contrast, MegaScale-MoE's overlap occurs within a single micro-batch's forward or backward pass, incurring no additional memory overhead and remaining compatible with systems both with and without pipeline parallelism.

Long-context training. While Megatron-LM [\[20,](#page-14-7) [48\]](#page-15-4) opts to partition only specific operations along the sequence dimension, various methods of sequence parallelism [\[11,](#page-14-22) [21,](#page-14-23) [23,](#page-14-24) [29\]](#page-14-25) have been explored for training models requiring long contexts. The Blockwise Parallel Transformer [\[28\]](#page-14-26) method implements blockwise computation of self-attention and the fusion of FFNs based on online softmax calculations. Ring Attention [\[23,](#page-14-24) [29\]](#page-14-25) introduces a ring-style communication mechanism integrated with self-attention calculations, facilitating the exchange of key and value chunks. We adopt the all-to-all style of SP attention from DeepSpeed Ulysses [\[16\]](#page-14-8), which partitions attention by heads rather than sequence length, due to its reduced communication volume and balanced computation pattern.

Communication-computation overlap. Several frameworks [\[12,](#page-14-27) [24,](#page-14-28) [31,](#page-14-29) [39,](#page-15-17) [55\]](#page-15-18) focus on overlapping communication with computation in distributed deep learning training with a single parallelism strategy. Some compiler-style work [\[17,](#page-14-15) [37,](#page-14-30) [52\]](#page-15-10) provides fine-grained overlap among kernels, but excessive partitioning of GEMM kernels can result

in low GPU utilization. Centauri [\[6\]](#page-14-31) enhances communication overlap for LLM training with 3D parallelism by communication partitioning and hierarchical scheduling. Similar to Centauri, our inter-operator communication overlap hides communication within independent computation by reordering operators. We further conceal communication on critical paths through intra-operator overlap, without compromising GPU utilization.

# 9 Conclusion

In this paper, we offer an in-depth look at the design, implementation, and deployment of MegaScale-MoE, a production-grade system built to efficiently train MoE models. MegaScale-MoE exploits communication-efficient approaches, including parallelism strategies with lower communication volume, inter- and intra-operator communicationcomputation overlap, and communication compression with adjusted communication patterns to unleash the compute capabilities of high-performance GPUs. MegaScale-MoE achieves 1.41M tokens/s in throughput when training a 352B MoE model on 1,440 NVIDIA Hopper GPUs, a 1.88× improvement over Megatron-LM. By sharing our insights on accelerating large-scale MoE training, we hope our work will inspire future research.

Acknowledgements. We thank our shepherd, Cheng Li, and the anonymous reviewers for their valuable feedback and suggestions. This work was supported in part by the National Key Research and Development Program of China under Grant 2022YFB4500700, the Scientific Research Innovation Capability Support Project for Young Faculty under Grant ZYGXQNJSKYCXNLZCXM-I1, the Fundamental Research Funds for the Central Universities, Peking University, and the National Natural Science Foundation of China under Grant 62172008 and Grant 62325201. Xin Jin and Xin Liu are the corresponding authors. Chao Jin, Xuanzhe Liu, and Xin Jin are also with the Key Laboratory of High Confidence Software Technologies (Peking University), Ministry of Education.

# References

- <span id="page-13-2"></span>[1] 2025. Context parallelism in Megatron-LM. (2025). https://docs.nvidia.[com/megatron-core/developer-guide/latest/api](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html)[guide/context\\_parallel](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html).html.
- <span id="page-13-0"></span>[2] 2025. Introducing DBRX: A New State-of-the-Art Open LLM. (2025). https://www.databricks.[com/blog/introducing-dbrx-new](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)[state-art-open-llm](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)
- <span id="page-13-1"></span>[3] 2025. Open Release of Grok-1. (2025). https://x.[ai/blog/grok-os](https://x.ai/blog/grok-os)
- <span id="page-13-3"></span>[4] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. 2023. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245 (2023).
- <span id="page-13-4"></span>[5] Li-Wen Chang, Wenlei Bao, Qi Hou, Chengquan Jiang, Ningxin Zheng, Yinmin Zhong, Xuanrun Zhang, Zuquan Song, Chengji Yao, Ziheng Jiang, et al. 2024. FLUX: fast software-based communication overlap on gpus through kernel fusion. arXiv preprint arXiv:2406.06858 (2024).

- <span id="page-14-31"></span>[6] Chang Chen, Xiuhong Li, Qianchao Zhu, Jiangfei Duan, Peng Sun, Xingcheng Zhang, and Chao Yang. 2024. Centauri: Enabling Efficient Scheduling for Communication-Computation Overlap in Large Model Training via Communication Partitioning. In ACM ASPLOS.
- <span id="page-14-0"></span>[7] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2023. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research (2023).
- <span id="page-14-18"></span>[8] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. Flashattention: Fast and memory-efficient exact attention with io-awareness. Neural Information Processing Systems (2022).
- <span id="page-14-4"></span>[9] Nan Du, Yanping Huang, Andrew M Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, Barret Zoph, Liam Fedus, Maarten P Bosma, Zongwei Zhou, Tao Wang, Emma Wang, Kellie Webster, Marie Pellat, Kevin Robinson, Kathleen Meier-Hellstern, Toju Duke, Lucas Dixon, Kun Zhang, Quoc Le, Yonghui Wu, Zhifeng Chen, and Claire Cui. 2022. GLaM: Efficient Scaling of Language Models with Mixture-of-Experts. In International Conference on Machine Learning (ICML).
- <span id="page-14-3"></span>[10] William Fedus, Barret Zoph, and Noam Shazeer. 2022. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research (2022).
- <span id="page-14-22"></span>[11] Diandian Gu, Peng Sun, Qinghao Hu, Ting Huang, Xun Chen, Yingtong Xiong, Guoteng Wang, Qiaoling Chen, Shangchun Zhao, Jiarui Fang, et al. 2024. Loongtrain: Efficient training of long-sequence llms with head-context parallelism. arXiv preprint arXiv:2406.18485 (2024).
- <span id="page-14-27"></span>[12] Sayed Hadi Hashemi, Sangeetha Abdu Jyothi, and Roy Campbell. 2019. Tictac: Accelerating distributed deep learning with communication scheduling. Proceedings of Machine Learning and Systems (2019).
- <span id="page-14-20"></span>[13] Jiaao He, Jidong Zhai, Tiago Antunes, Haojie Wang, Fuwen Luo, Shangfeng Shi, and Qin Li. 2022. Fastermoe: modeling and optimizing training of large-scale dynamic pre-trained models. In ACM PPoPP.
- <span id="page-14-9"></span>[14] Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Dehao Chen, Mia Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V Le, Yonghui Wu, et al. 2019. Gpipe: Efficient training of giant neural networks using pipeline parallelism. Neural Information Processing Systems (2019).
- <span id="page-14-6"></span>[15] Changho Hwang, Wei Cui, Yifan Xiong, Ziyue Yang, Ze Liu, Han Hu, Zilong Wang, Rafael Salas, Jithin Jose, Prabhat Ram, et al. 2023. Tutel: Adaptive mixture-of-experts at scale. Proceedings of Machine Learning and Systems (2023).
- <span id="page-14-8"></span>[16] Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He. 2023. Deepspeed ulysses: System optimizations for enabling training of extreme long sequence transformer models. arXiv preprint arXiv:2309.14509 (2023).
- <span id="page-14-15"></span>[17] Abhinav Jangda, Jun Huang, Guodong Liu, Amir Hossein Nodehi Sabet, Saeed Maleki, Youshan Miao, Madanlal Musuvathi, Todd Mytkowicz, and Olli Saarikivi. 2022. Breaking the computation and communication abstraction barrier in distributed machine learning workloads. In ACM ASPLOS.
- <span id="page-14-1"></span>[18] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. 2024. Mixtral of experts. arXiv preprint arXiv:2401.04088 (2024).
- <span id="page-14-2"></span>[19] Ziheng Jiang, Haibin Lin, Yinmin Zhong, Qi Huang, Yangrui Chen, Zhi Zhang, Yanghua Peng, Xiang Li, Cong Xie, Shibiao Nong, Yulu Jia, Sun He, Hongmin Chen, Zhihao Bai, Qi Hou, Shipeng Yan, Ding Zhou, Yiyao Sheng, Zhuo Jiang, Haohan Xu, Haoran Wei, Zhang Zhang, Pengfei Nie, Leqi Zou, Sida Zhao, Liang Xiang, Zherui Liu, Zhe Li, Xiaoying Jia, Jianxi Ye, Xin Jin, and Xin Liu. 2024. MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs. In USENIX NSDI.

- <span id="page-14-7"></span>[20] Vijay Anand Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. 2023. Reducing activation recomputation in large transformer models. Proceedings of Machine Learning and Systems (2023).
- <span id="page-14-23"></span>[21] Dacheng Li, Rulin Shao, Anze Xie, Eric P. Xing, Xuezhe Ma, Ion Stoica, Joseph E. Gonzalez, and Hao Zhang. 2024. DISTFLASHATTN: Distributed Memory-efficient Attention for Long-context LLMs Training. arxiv preprint arXiv:2310.03294 (2024).
- <span id="page-14-12"></span>[22] Jiamin Li, Yimin Jiang, Yibo Zhu, Cong Wang, and Hong Xu. 2023. Accelerating distributed {MoE} training and inference with lina. In USENIX ATC.
- <span id="page-14-24"></span>[23] Shenggui Li, Fuzhao Xue, Chaitanya Baranwal, Yongbin Li, and Yang You. 2021. Sequence parallelism: Long sequence training from system perspective. arXiv preprint arXiv:2105.13120 (2021).
- <span id="page-14-28"></span>[24] Shen Li, Yanli Zhao, Rohan Varma, Omkar Salpekar, Pieter Noordhuis, Teng Li, Adam Paszke, Jeff Smith, Brian Vaughan, Pritam Damania, et al. 2020. Pytorch distributed: Experiences on accelerating data parallel training. arXiv preprint arXiv:2006.15704 (2020).
- <span id="page-14-17"></span>[25] Wanchao Liang, Tianyu Liu, Less Wright, Will Constable, Andrew Gu, Chien-Chin Huang, Iris Zhang, Wei Feng, Howard Huang, Junjie Wang, et al. 2024. TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training. arXiv preprint arXiv:2410.06511 (2024).
- <span id="page-14-14"></span>[26] Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, et al. 2024. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434 (2024).
- <span id="page-14-5"></span>[27] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. 2024. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437 (2024).
- <span id="page-14-26"></span>[28] Hao Liu and Pieter Abbeel. 2024. Blockwise Parallel Transformers for Large Context Models. Neural Information Processing Systems (2024).
- <span id="page-14-25"></span>[29] Hao Liu, Matei Zaharia, and Pieter Abbeel. 2023. Ring attention with blockwise transformers for near-infinite context. arXiv preprint arXiv:2310.01889 (2023).
- <span id="page-14-21"></span>[30] Juncai Liu, Jessie Hui Wang, and Yimin Jiang. 2023. Janus: A unified distributed training framework for sparse mixture-of-experts models. In ACM SIGCOMM.
- <span id="page-14-29"></span>[31] Kshiteej Mahajan, Ching-Hsiang Chu, Srinivas Sridharan, and Aditya Akella. 2023. Better Together: Jointly Optimizing ML Collective Scheduling and Execution Planning using {SYNDICATE}. In USENIX NSDI.
- <span id="page-14-16"></span>[32] Megatron-LM 2025. GPU optimized techniques for training transformer models at-scale. (2025). https://github.[com/NVIDIA/Megatron-](https://github.com/NVIDIA/Megatron-LM)[LM](https://github.com/NVIDIA/Megatron-LM).
- <span id="page-14-10"></span>[33] Deepak Narayanan, Aaron Harlap, Amar Phanishayee, Vivek Seshadri, Nikhil R Devanur, Gregory R Ganger, Phillip B Gibbons, and Matei Zaharia. 2019. PipeDream: generalized pipeline parallelism for DNN training. In ACM SOSP.
- <span id="page-14-11"></span>[34] Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, et al. 2021. Efficient large-scale language model training on gpu clusters using megatronlm. In International Conference for High Performance Computing, Networking, Storage and Analysis.
- <span id="page-14-13"></span>[35] NCCL 2025. Optimized primitives for inter-GPU communication. (2025). https://github.[com/NVIDIA/nccl](https://github.com/NVIDIA/nccl).
- <span id="page-14-19"></span>[36] Xiaonan Nie, Pinxue Zhao, Xupeng Miao, Tong Zhao, and Bin Cui. 2022. HetuMoE: An efficient trillion-scale mixture-of-expert distributed training system. arXiv preprint arXiv:2203.14685 (2022).
- <span id="page-14-30"></span>[37] Suchita Pati, Shaizeen Aga, Mahzabeen Islam, Nuwan Jayasena, and Matthew D. Sinclair. 2024. T3: Transparent Tracking & Triggering for Fine-grained Overlap of Compute & Collectives. In ACM ASPLOS.

- <span id="page-15-3"></span>[38] Houwen Peng, Kan Wu, Yixuan Wei, Guoshuai Zhao, Yuxiang Yang, Ze Liu, Yifan Xiong, Ziyue Yang, Bolin Ni, Jingcheng Hu, et al. 2023. Fp8 lm: Training fp8 large language models. arXiv preprint arXiv:2310.18313 (2023).
- <span id="page-15-17"></span>[39] Yanghua Peng, Yibo Zhu, Yangrui Chen, Yixin Bao, Bairen Yi, Chang Lan, Chuan Wu, and Chuanxiong Guo. 2019. A generic communication scheduler for distributed DNN training acceleration. In ACM SOSP.
- <span id="page-15-2"></span>[40] Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, and Yuxiong He. 2022. DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale. In International Conference on Machine Learning (ICML).
- <span id="page-15-7"></span>[41] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. 2020. Zero: Memory optimizations toward training trillion parameter models. In International Conference for High Performance Computing, Networking, Storage and Analysis.
- <span id="page-15-14"></span>[42] Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley, Shaden Smith, and Yuxiong He. 2021. Zero-infinity: Breaking the gpu memory wall for extreme scale deep learning. In International Conference for High Performance Computing, Networking, Storage and Analysis.
- <span id="page-15-6"></span>[43] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. 2020. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In ACM SIGKDD.
- <span id="page-15-15"></span>[44] Jie Ren, Samyam Rajbhandari, Reza Yazdani Aminabadi, Olatunji Ruwase, Shuangyan Yang, Minjia Zhang, Dong Li, and Yuxiong He. 2021. Zero-offload: Democratizing billion-scale model training. In USENIX ATC.
- <span id="page-15-8"></span>[45] Noam Shazeer. 2020. Glu variants improve transformer. arXiv preprint arXiv:2002.05202 (2020).
- <span id="page-15-1"></span>[46] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. 2017. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538 (2017).
- <span id="page-15-16"></span>[47] Liang Shen, Zhihua Wu, WeiBao Gong, Hongxiang Hao, Yangfan Bai, HuaChao Wu, Xinxuan Wu, Jiang Bian, Haoyi Xiong, Dianhai Yu, et al. 2022. Se-moe: A scalable and efficient mixture-of-experts distributed training and inference system. arXiv preprint arXiv:2205.10034 (2022).
- <span id="page-15-4"></span>[48] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. 2019. Megatron-lm: Training multibillion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053 (2019).
- <span id="page-15-0"></span>[49] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 (2023).
- <span id="page-15-9"></span>[50] TransformerEngine 2025. A library for accelerating Transformer models on NVIDIA GPUs, including using 8-bit floating point (FP8) precision on Hopper and Ada GPUs, to provide better performance with lower memory utilization in both training and inference. (2025). https://github.[com/NVIDIA/TransformerEngine](https://github.com/NVIDIA/TransformerEngine).
- <span id="page-15-5"></span>[51] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Neural Information Processing Systems (2017).
- <span id="page-15-10"></span>[52] Shibo Wang, Jinliang Wei, Amit Sabne, Andy Davis, Berkin Ilbeyi, Blake Hechtman, Dehao Chen, Karthik Srinivasa Murthy, Marcello Maggioni, Qiao Zhang, et al. 2022. Overlap communication with dependent computation via decomposition in large deep learning models. In ACM ASPLOS.
- <span id="page-15-11"></span>[53] Shulai Zhang, Ningxin Zheng, Haibin Lin, Ziheng Jiang, Wenlei Bao, Chengquan Jiang, Qi Hou, Weihao Cui, Size Zheng, Li-Wen Chang, et al. 2025. Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts. arXiv preprint arXiv:2502.19811 (2025).

- <span id="page-15-13"></span>[54] Zili Zhang, Yinmin Zhong, Yimin Jiang, Hanpeng Hu, Jianjian Sun, Zheng Ge, Yibo Zhu, Daxin Jiang, and Xin Jin. 2025. Disttrain: Addressing model and data heterogeneity with disaggregated training for multimodal large language models. In ACM SIGCOMM.
- <span id="page-15-18"></span>[55] Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, Alban Desmaison, Can Balioglu, Pritam Damania, Bernard Nguyen, Geeta Chauhan, Yuchen Hao, Ajit Mathews, and Shen Li. 2023. PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel. Proceedings of the VLDB Endowment (2023).
- <span id="page-15-12"></span>[56] Size Zheng, Jin Fang, Xuegui Zheng, Qi Hou, Wenlei Bao, Ningxin Zheng, Ziheng Jiang, Dongyang Wang, Jianxi Ye, Haibin Lin, et al. 2025. Tilelink: Generating efficient compute-communication overlapping kernels using tile-centric primitives. arXiv preprint arXiv:2503.20313 (2025).

# A Appendix

# <span id="page-16-0"></span>A.1 Hierarchical Communication for Parameter Synchronization

Let the full attention weights size be P, the dimension of model parallelism (TP or SP) be n, and the data parallel size be d. Typically, GPUs for model parallelism are located on the same node, requiring intra-node communication, whereas data parallelism spans across nodes, requiring inter-node communication. Consider a data parallelism group containing d devices, each holding the identical partition of the parameter.

For parameter synchronization in TP attention, communication involves data of size P/n across d devices in two primary steps in LLM training:

- inter-node reduce-scatter operation, where the data size is P/n, on d devices.
- inter-node all-gather operation, where the data size is P/n, on d devices.

leading to primarily inter-node communication, with a communication volume of 2P/n(d-1)/d.

With SP attention, the parameter synchronization involves the entire data of size P across  $n \times d$  devices. Considering the discrepancy between intra-node and inter-node network bandwidth, this process can be implemented by four-step hierarchical communication, where the replicated parameters are first reduced within a node and then reduced across nodes, before being distributed back to each device. Figure 5a illustrates a hierarchical communication example where n=3 and d=2. The detailed steps are as follows.

- intra-node reduce-scatter operation, where the data size is *P*. on *n* devices.
- inter-node reduce-scatter operation, where the data size is P/n, on d devices.
- inter-node all-gather operation, where the data size is *P/n*, on *d* devices.
- intra-node all-gather operation, where the data size is *P*, on *n* devices.

The inter-node communication volume in SP attention remains at 2P/n(d-1)/d, with additional intra-node volume of 2P(n-1)/n.

Moreover, due to the distinct resources for intra-node and inter-node communications, these steps can be segmented into small chunks and pipelined to efficiently hide each other as shown in Figure 5b. The ratio of inter-node communication latency and intra-node communication latency is

$$\frac{1}{n} \times \frac{\text{intra-node bandwidth}}{\text{inter-node bandwidth}} \times \frac{n(d-1)}{d(n-1)}$$
 (10)

Consider a typical training scenario involving an H100 SXM machine, where the NVLink bandwidth is 450 GB/s, and the inter-device NIC communication bandwidth is 50 GB/s. In this context, the latency of inter-node communication can easily surpass that of intra-node communication. This implies that the communication within a node can overshadow that between nodes. Consequently, in such scenarios, the synchronization of gradients and parameters with SP attention is, in fact, consistent with TP attention.