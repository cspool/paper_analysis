## <span id="page-0-0"></span>Sub-MoE: Efficient Mixture-of-Expert LLMs Compression via Subspace Expert Merging

**Lujun Li**<sup>1†</sup>, **Qiyuan Zhu**<sup>1†</sup>, **Jiacheng Wang**<sup>2</sup>, **Wei Li**<sup>3</sup>, **Hao Gu**<sup>1</sup>, **Sirui Han**<sup>1\*</sup>, **Yike Guo**<sup>1\*</sup>

<sup>1</sup>Hong Kong University of Science and Technology, <sup>2</sup>Xi'an Jiaotong University,

<sup>3</sup>University of Birmingham

{lliee,qzhuat,siruihan,yikeguo}@ust.hk \*

#### **Abstract**

Mixture of Experts (MoE) LLMs face significant obstacles due to their massive parameter scale, which imposes memory, storage, and deployment challenges. Although recent expert merging methods promise greater efficiency by consolidating multiple experts, they are fundamentally hindered by parameter conflicts arising from expert specialization. In this paper, we present Sub-MoE, a novel MoE compression framework via Subspace Expert Merging. Our key insight is to perform joint Singular Value Decomposition (SVD) on concatenated expert weights, reducing conflicting parameters by extracting shared U-matrices while enabling effective merging of the expert-specific V components. Specifically, Sub-MoE consists of two innovative phases: (1) Adaptive Expert Clustering, which groups functionally coherent experts via K-means clustering based on cosine similarity of expert outputs; and (2) Subspace Expert Merging, which first enforces Experts Union Decomposition to derive the shared U-matrix across experts in the same group, then pursues frequency-based merging for individual V-matrices, and finalizes expert reconstruction using the merged V-matrix. In this way, we align and fuse experts in a shared subspace, and can be extended with intra-expert compression for further inference optimization. Extensive experiments on Mixtral, DeepSeek, and Qwen-1.5|3 MoE LLMs demonstrate that our Sub-MoE significantly outperforms existing expert pruning and merging methods. Notably, our Sub-MoE maintains 96%186%of original performance with 25%|50% expert reduction on Mixtral-8×7B in zeroshot benchmarks. Code will be released at https://github.com/lliai/MoERazor.

## 1 Introduction

The Mixture of Experts (MoE) architecture has emerged as a pivotal advancement in Large Language Models (LLMs) [31], demonstrated by recent models like DeepSeek-R1 [7] and Qwen3-MoE [45]. At its core, MoE consists of expert networks and a gating mechanism that dynamically routes each input to the most relevant experts. MoE sparsely activates only a small subset of experts, significantly reducing computational costs while scaling model size. However, MoE LLMs also introduce challenges from their large parameter count, including substantial memory/storage requirements and inference latency that complicate deployment on resource-constrained devices [36, 21, 25]. Additionally, distributed implementations face communication [19] bottlenecks when synchronizing experts across multiple nodes, impacting real-time performance [34, 8].

To overcome these issues, researchers are developing fundamental yet important expert reduction approaches that can be broadly categorized into two primary approaches: expert pruning and expert merging. Expert pruning methods remove underperforming experts through regularization (e.g.,

<sup>\*\*</sup>Corresponding authors, † equal contribution.

<span id="page-1-2"></span><span id="page-1-0"></span>![](_page_1_Picture_0.jpeg)

Figure 1: Overview of our Sub-MoE framework. The process consists of two main stages: (1) Adaptive Expert Clustering, which groups similar experts via K-means clustering with steps: Clusters Assignment, Means Update, and Convergence; and (2) Subspace Expert Merging, which aligns and combines experts via Experts Union SVD, V-Matrix Merging. and Expert Reconstruction.

SEER-MoE [30]) or search-based techniques (e.g., NAEE [28], MoE-I<sup>2</sup> [46]). While these approaches effectively reduce parameter counts, they fundamentally discard portions of the model's learned knowledge, resulting in performance degradation that necessitates resource-intensive finetuning to recover. Expert merging techniques (e.g., MC-SMoE [23], HC-SMoE [3], and EEP [27]) propose a promising alternative by preserving knowledge through the consolidation of multiple experts. However, current merging approaches encounter a critical limitation that undermines their effectiveness: the parameter conflict problem. This fundamental challenge arises from the core design principle of MoE architectures, where routing mechanisms deliberately create specialized experts with divergent parameter spaces by training them on distinct input distributions. The recent study [11] shows that Mixtral-8×7B demonstrates this divergence, showing inter-expert similarities typically ranging between 0.1~0.3. When conventional merging operations are applied to such dissimilar experts, catastrophic parameter conflicts emerge that compromise the specialized capabilities of the original experts and significantly degrade overall model performance. Existing merging approaches employ simplistic aggregation functions that cannot effectively reconcile these divergent parameter spaces and often require computationally expensive post-merging operations (e.g.,  $\bar{D}^2$ -MoE [11]), undermining the efficiency gains. This motivates our core research question:

# (RQ) How can we reduce parameter conflicts among diverse experts and enhance the effectiveness of expert merging?

To answer the question, we present Sub-MoE, a novel expert merging framework rooted in subspacebased decomposition and alignment. Our approach leverages Singular Value Decomposition (SVD) to transform the concatenated weight matrices of multiple experts into a shared low-dimensional subspace, represented by a common orthogonal basis U, singular values  $\Sigma$ , and individual projections  $V^T$ . By performing the merging operation solely on the  $V^T$  component—while preserving alignment to the shared U—we exploit intrinsic correlations among experts, thereby minimizing conflicting parameters and retaining specialized knowledge. As illustrated in Figure 1, our Sub-MoE framework consists of two synergistic stages: Adaptive Expert Clustering and Subspace Expert Merging. In the first stage, we perform adaptive expert grouping via K-means clustering based on output similarities of experts, ensuring that merging is performed on functionally coherent groups. In addition, we jointly cluster multi-layer experts under a target overall compression ratio and adaptively determine the layer-wise grouping numbers. In the second stage, we concatenate expert weights from the same group and enforce co-decomposition to obtain the shared U-matrix across experts and expert-specific  $V^T$ -matrixs. Figure 2 demonstrates that this subspace-sharing process can align the output of the various experts. For the remaining unmerged components, we further introduce the frequencybased merging strategy that weights expert contributions according to their activation patterns. This approach gives greater influence to frequently activated experts while still preserving capabilities from experts. Finally, the merged weight matrix is reconstructed as  $U\Sigma[V_{\text{merged}}]^{T}$ . Additionally, we extend Sub-MoE to Sub-MoE† with MoE-specific activation-aware truncated SVD for intra-expert

<span id="page-1-1"></span><sup>&</sup>lt;sup>2</sup>The singular values  $\Sigma$  is multiplied in the shared U matrix during the joint SVD process in implementation.

<span id="page-2-1"></span><span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

Figure 2: Cosine similarity of output of original expert (left) and subspace aligned matrices (right) on Mixtral-8×7B. Figure 3: Mean accuracy of MoE compressors on zero-shot tasks.

compression for greater parameter efficiency. We incorporate input activation statistics by weighting expert parameters with the whitening matrix of hidden activations, further stabilizing performance at high compression levels.

We conduct comprehensive experiments on Mixtral 8x7B [\[18\]](#page-10-8), Qwen3-235B-A22B [\[45\]](#page-11-0) Qwen1.5- MoE-A2.7B [\[38\]](#page-11-4) and DeepSeekMoE-16B-Base [\[6\]](#page-9-4). As shown in Figure [3,](#page-2-0) our proposed Sub-MoE method consistently outperforms existing expert reduction techniques. With Mixtral-8×7B, Sub-MoE maintains 94% and 87% of accuracy using only 75% or 50% of experts, surpassing HC-SMoE[\[3\]](#page-9-2) by 13.7%. For Qwen3 MoE, Sub-MoE maintains 83% of accuracy with half the experts, while HC-SMoE drops to 55%. Similarly, on DeepSeek-MoE-16B, our method preserves 86% of performance with half the experts, outperforming HC-SMoE by 6.5%. These results affirm the effectiveness and generalizability of our approach across diverse MoE architectures and downstream tasks, positioning Sub-MoE as a principled and scalable solution to expert merging for next-generation MoE LLMs.

## 2 Related Work

MoE Compression. To improve the efficiency of MoE LLMs, researchers have developed numerous system-level optimizations (*e.g.*, expert parallel [\[2\]](#page-9-5) and offloading [\[44\]](#page-11-5)) and model-level techniques (quantization [\[15,](#page-10-9) [9\]](#page-9-6) and compression [\[33\]](#page-11-6)). Among them, expert reduction methods primarily focus on removing redundant experts to achieve optimal efficiency-performance tradeoffs. For optimizationbased expert pruning, SEER-MoE [\[30\]](#page-10-4) removes non-professional experts through regularization-based fine-tuning. In search-based expert pruning approaches, NAEE [\[28\]](#page-10-5) trims unimportant experts by minimizing pruning error, while MoE-I 2 [\[46\]](#page-11-3) employs genetic search strategies. In sharp contrast to these pruning approaches, our Sub-MoE explores the merging paradigm that requires neither searching nor fine-tuning. Other kind methods use weight or hybrid compression for MoE. MoE-Pruner [\[43\]](#page-11-7) prunes weights based on activations and router logits, STUN [\[20\]](#page-10-10) combines structured and unstructured pruning and D<sup>2</sup> -MoE [\[11\]](#page-9-3) introduces delta compensation. MoE-Compression [\[12\]](#page-9-7) provides compressors evaluations. Different from these, we highlight that Sub-MoE mainly addresses expert merging rather than weight compression.

Expert Merging methods [\[22,](#page-10-11) [48\]](#page-11-8) fuse multiple experts into a single one through weighted summation or averaging. For instance, MC-SMoE [\[23\]](#page-10-6) merges experts with similar routing policies, HC-SMoE [\[3\]](#page-9-2) utilizes hierarchical clustering to merge experts in a task-agnostic manner, and EEP [\[27\]](#page-10-7) optimizes fusion matrices through evolutionary search algorithms. However, these methods typically employ original weight merging techniques [\[42\]](#page-11-9), which achieve success primarily when handling models with high similarity, such as fine-tuned variants of the same base model [\[17\]](#page-10-12). When applied to MoE models with low-similarity experts, these methods generally fail due to significant parameter conflicts during the merging process. Our Sub-MoE distinctly differs from previous expert mergers by addressing this low-similarity issue through expert decomposition into subspaces and enforcing matrices alignment. Thanks to this subspace alignment approach, we can effectively fuse different MoE LLMs without requiring additional training, searching, or complex weight operations.

#### <span id="page-3-1"></span>3 Methodology

Our Sub-MoE framework consists of two synergistic stages: (1) **Adaptive Expert Clustering** stage that clusters similar experts, (2) **Subspace Expert Merging** stage that includes union decomposition, frequency-based V-Matrix fusion and reconstruction. The overall process is illustrated in Figure 1.

#### 3.1 Recap of MoE Architecture

The fundamental principle of MoE models is to dynamically route input data to specialized expert networks. Consider an input token  $x \in \mathbb{R}^d$ , a set of expert modules  $\{E_1, E_2, ..., E_n\}$ , and a router network R. The output y of an MoE layer is computed as:

$$y = \sum_{i=1}^{n} G_i(x) \cdot E_i(x), \quad E(x) = (\sigma(x \cdot W_{gate}) \odot (x \cdot W_{up})) \cdot W_{down}$$
 (1)

where  $G_i(x)$  represents the routing score for expert i, and  $E_i(x)$  denotes its output. Each expert typically implements a feed-forward layer with weight matrices  $\{W_{\rm up}, W_{\rm gate}W_{\rm down}\}$ , and  $\sigma$  activation (e.g., SiLU function). The router R employs a top-k strategy with softmax normalization, activating only the most relevant experts for each input token and thereby enhancing computational efficiency.

#### 3.2 Adaptive Expert Clustering

A critical challenge in compressing MoE models is identifying which experts can be effectively merged with minimal information loss. Rather than relying on architectural heuristics or arbitrary grouping strategies, we propose a data-driven approach that captures the functional similarity between experts. Our key insight is that experts processing similar input patterns in comparable ways are more amenable to merging. To implement this intuition, we first collect a representative set of input tokens  $\mathcal{X} = \{x_1, x_2, ..., x_m\}$  from the target domain. For each expert  $E_i$ , we compute output vectors across this input set, yielding output collections  $\mathcal{Y}_i = \{E_i(x_1), E_i(x_2), ..., E_i(x_m)\}$  that characterize the expert's functional behavior. We then quantify the functional similarity between experts using the average cosine similarity of their outputs:

<span id="page-3-0"></span>
$$Sim(E_i, E_j) = \frac{1}{m} \sum_{l=1}^{m} \frac{E_i(x_l) \cdot E_j(x_l)}{||E_i(x_l)|| \cdot ||E_j(x_l)||},$$
(2)

This similarity metric captures how consistently two experts respond to the same inputs, regardless of their internal parameter representations. Experts with high similarity scores are likely to serve overlapping functions within the model and thus become strong candidates for merging.

Based on this similarity measure, we employ K-means clustering to organize the experts into k coherent groups. This process consists of four key steps:

- 1. Means Initialization: Initial cluster centroids  $C = \{C_1, C_2, ..., C_k\}$  are established through advanced seeding method (i.e., k-means++ [16]) to ensure diverse starting points across the expert functional space.
- **2. Clusters Assignment:** Each expert  $E_j$  is assigned to the nearest cluster centroid based on the similarity metric in Equation 2, forming expert groups  $Q_i$  that share functional characteristics.
- 3. Means Update: Cluster centroids are recalculated as the mean of all experts assigned to that cluster:  $C_i = \frac{1}{|Q_i|} \sum_{E_j \in Q_i} \mathcal{Y}_j$ .
- **4. Convergence:** Steps 2 and 3 are repeated until cluster assignments stabilize or maximum iterations are reached, minimizing the objective function:

$$J = \sum_{i=1}^{k} \sum_{E_j \in Q_i} ||\mathcal{Y}_j - C_i||^2,$$
(3)

where  $Q_i$  represents the set of experts assigned to cluster i. This data-driven approach discovers inherent functional relationships between experts that might not be apparent from architecture alone.

**Multi-layer Adaptive Allocation:** Unlike traditional manners that impose uniform reduction across all MoE layers, we introduce a multi-layer adaptive allocation that optimizes the numbers of groups

on a per-layer basis. We recognize that different layers within a model exhibit varying degrees of functional redundancy and specialization. By jointly clustering experts on multiple MoE layers while maintaining a target overall compression ratio, our automated clustering process dynamically adjusts clustering centers and determines the optimal number of clusters for each layer without manual intervention. Layers with higher expert similarity naturally form fewer, more cohesive clusters, while those with more diverse patterns maintain more clusters to preserve their specialized capabilities.

#### 3.3 Subspace Expert Merging

**Problems in Vanilla Expert Merging.** The central challenge in merging expert networks lies in their different parametric representations. Even when experts serve similar functions, their internal parameters often operate in distinct representation spaces, making direct merging problematic and leading to performance degradation. Given n expert weight matrices  $W^{(1)}, W^{(2)}, ..., W^{(n)} \in \mathbb{R}^{O \times I}$ , conventional merging methods apply operations directly:

$$W_{\text{merged}} = \sum_{i=1}^{n} \alpha_i W^{(i)}, \tag{4}$$

where  $\alpha_i$  are weight coefficients. This approach often leads to parameter conflicts because each  $W^{(i)}$  operates in its own representation space.

**Subspace Alignment via Experts Union Decomposition.** We address this challenge by transforming experts into a common subspace before merging. For each expert group identified in the clustering step, we concatenate their weight matrices vertically and apply SVD:

$$SVD\left([W^{(1)}; W^{(2)}; \dots; W^{(n)}]\right) = U\Sigma[V^{(1)}; V^{(2)}; \dots; V^{(n)}]^T,$$
(5)

where  $U \in \mathbb{R}^{O \times r}$  contains left singular vectors, which form an orthonormal basis for the input space,  $\Sigma \in \mathbb{R}^{r \times r}$  is a diagonal matrix of singular values, and  $V \in \mathbb{R}^{r \times nI}$  contains right singular vectors, which can be partitioned into n blocks, each corresponding to an expert.

**Frequency-based** *V***-Matrix Merging.** Our method introduces a simple yet effective merging approach that respects the usage patterns of experts in real-world scenarios. We observe that not all experts contribute equally to model outputs—some experts specialize in handling common patterns while others focus on rare cases. Incorporating this frequency information helps preserve the model's capabilities across diverse inputs. For each expert *i*, we calculate its sampling frequency based on actual router activations:

$$f(V_i) = \frac{\sum_{x \in \mathcal{X}} \mathbb{I}[i \in \text{TopK}(G(x), k)]}{|\mathcal{X}|},$$
(6)

where  $\mathcal{X}$  represents the set of input tokens,  $\mathbb{I}[\cdot]$  is the indicator function that equals 1 when expert i is among the top-k experts selected by the routing mechanism for input x, and 0 otherwise. This frequency metric captures how often each expert is activated across a representative dataset. We then compute the merged V matrix in each group as a frequency-weighted average:

$$V_{\text{merged}} = \frac{\sum_{i \in Q} f(V_i) \cdot V_i}{\sum_{i \in Q} f(V_i)}$$
 (7)

This frequency-based merging effectively integrates expert information according to their practical utilization patterns, giving greater weight to frequently activated experts while still preserving capabilities from experts.

**Expert Reconstruction.** The final merged expert weights are constructed as:

$$W_{\text{merged}} = U\Sigma[V_{\text{merged}}]^T \tag{8}$$

By construction, all experts within a cluster are merged into a single set of parameters  $W_{\text{merged}}$ , which is reconstructed using the shared orthogonal basis U, singular values  $\Sigma$ , and the frequency-weighted merged right singular vectors  $V_{\text{merged}}$ . This process effectively aligns the original experts to a common subspace and compresses them into one representative expert. Through this three-stage process,

<span id="page-5-0"></span>Sub-MoE achieves effective expert reduction while maintaining performance by operating in a shared subspace, minimizing parameter conflicts, and preserving the essential characteristics of each expert.

Understanding of Subspace Expert Merging. The foundation of our subspace merging approach can be understood through the lens of manifold learning [39, 14, 26] and representation alignment [35, 37]. Each expert can be viewed as operating on a different manifold in parameter space [40], but these manifolds often share underlying structures due to the similar functionality of the experts [11]. By applying SVD to the union of experts, we discover a common coordinate system (defined by the left singular vectors U) that captures the shared functional subspace across all experts.

The singular values in  $\Sigma$  quantify the importance of each dimension in this common subspace, while the right singular vectors in V represent how each expert maps to these dimensions. By sharing the U matrix across all experts within a cluster, we ensure that they operate in the same subspace, making their outputs more compatible and reducing interference when merging. The frequency-weighted merging of the V matrices further ensures that the common subspace is biased toward preserving the functionality of the most frequently used experts, which typically process the most important patterns in the data. Mathematically, this process can also be understood as finding a low-rank approximation of the original expert collection that minimizes the difference between the outputs of the original experts and their reconstructed versions on input data x, subject to the constraint that they share a common input transformation U. This optimization can be written as:

$$\min_{U,\Sigma,V} \left\| \sum_{i \in Q} G_i(x) W^{(i)} x - U \Sigma V^T x \right\|_2 \tag{9}$$

This formulation reveals why our approach is more effective than direct averaging: we are projecting each expert onto a common subspace before merging, which preserves their essential functionality on real input data while eliminating conflicting parameter representations.

#### 3.4 Sub-MoE† for Intra-Expert Compression

Sub-MoE reduces expert counts without changes to intra-expert sizes. To further improve the compression ratio for resource-constrained scenarios, we present extended Sub-MoE $\dagger$  to reduce the size of U,V by truncating before reconstruction. Beyond previous dense LLM SVD techniques [41], our Sub-MoE $\dagger$  employs MoE-specific activation-aware truncating SVD.

For expert weight matrix  $W_i$ , we first obtain the activation weighted matrix  $S_i$  by measuring the correlation of input activations  $X_i$ .  $S_i$  effectively preserves salience weights and reduces decomposition errors [41]. Then, we re-weight each expert's weight matrix as  $W_i' = W_i S_i$ . Next, we concatenate the re-weighted weight matrices from all experts in the same group and apply union decomposition:

$$SVD\left([W'^{(1)}; W'^{(2)}; \dots; W'^{(n)}]\right) = U'\Sigma'[V'^{(1)}; V'^{(2)}; \dots; V'^{(n)}]^T$$
(10)

For experts in cluster Q, we compute the frequency-weighted merged vector with de-whitening:

$$V_{\text{merged}} = \frac{\sum_{i \in Q} f(V_i) \cdot V'^{(i)} S_i^{-1}}{\sum_{i \in Q} f(V_i)}$$
(11)

After truncating the smallest singular values in  $\Sigma'$  to control the compression ratio, the final merged expert weight is given by:

$$W_{\text{merged}}^{\text{trunc}} = U' \cdot \text{Trunc.}(\Sigma') \cdot V_{\text{merged}}$$
(12)

This process enables fine-grained control over compression while minimizing information loss, as the activation weighted matrix S enables a direct mapping between singular values and compression loss [41]. By combining expert clustering, subspace alignment, intra-expert compression, and frequency-aware merging, our framework provides a comprehensive solution for MoE model compression across multiple levels of redundancy.

### 4 Experiments

In this section, we present a comprehensive evaluation and ablation studies of our Sub-MoE across multiple tasks. All experiments are conducted on eight NVIDIA H800 GPUs.

<span id="page-6-1"></span><span id="page-6-0"></span>Table 1: Comparisons of expert prune/merge methods in multiple MoE LLMs. We report perplexity (lower is better↓) on language modeling tasks and accuracy (higher is better↑) on reasoning tasks.

|         | · · · · · · · · · · · · · · · · · · · | 00          |         | 0       |          |           | <i>-</i> | 0       |      | 17 - |      | - 0    |          |
|---------|---------------------------------------|-------------|---------|---------|----------|-----------|----------|---------|------|------|------|--------|----------|
| Expert  | Method                                | WikiText-2↓ | PTB↓    | C4↓     | ARC-c    | ARC-e     | BoolQ    | HellaS. | MMLU | OBQA | RTE  | WinoG. | Average↑ |
|         |                                       |             |         |         | Mixtra   | 1-8×7B    |          |         |      |      |      |        |          |
| Num=8   | Original                              | 3.98        | 14.79   | 7.33    | 0.56     | 0.84      | 0.85     | 0.65    | 0.67 | 0.35 | 0.71 | 0.76   | 0.67     |
| Num=6   | Frequency-prune                       | 6.22        | 18.00   | 9.94    | 0.48     | 0.78      | 0.78     | 0.57    | 0.47 | 0.32 | 0.55 | 0.75   | 0.59     |
|         | Output-prune                          | 6.17        | 18.28   | 9.63    | 0.47     | 0.77      | 0.75     | 0.58    | 0.46 | 0.30 | 0.60 | 0.75   | 0.58     |
|         | MC-SMoE                               | 58.11       | 173.51  | 98.86   | 0.29     | 0.60      | 0.59     | 0.43    | 0.25 | 0.20 | 0.53 | 0.60   | 0.44     |
|         | HC-SMoE                               | 5.92        | 18.70   | 9.49    | 0.45     | 0.73      | 0.83     | 0.57    | 0.56 | 0.29 | 0.69 | 0.75   | 0.61     |
|         | Sub-MoE (Ours)                        | 5.16        | 18.58   | 8.54    | 0.49     | 0.80      | 0.86     | 0.62    | 0.59 | 0.32 | 0.65 | 0.75   | 0.64     |
|         | Frequency-prune                       | 17.45       | 79.43   | 22.40   | 0.22     | 0.39      | 0.60     | 0.36    | 0.24 | 0.14 | 0.53 | 0.53   | 0.38     |
|         | Output-prune                          | 15.40       | 81.96   | 20.08   | 0.21     | 0.39      | 0.63     | 0.38    | 0.24 | 0.16 | 0.54 | 0.56   | 0.39     |
| Num=4   | MC-SMoE                               | 854.05      | 1204.41 | 1408.10 | 0.21     | 0.28      | 0.52     | 0.28    | 0.25 | 0.11 | 0.50 | 0.52   | 0.33     |
|         | HC-SMoE                               | 9.88        | 34.13   | 16.78   | 0.32     | 0.61      | 0.75     | 0.49    | 0.39 | 0.26 | 0.61 | 0.67   | 0.51     |
|         | Sub-MoE (Ours)                        | 6.97        | 26.88   | 10.64   | 0.45     | 0.75      | 0.84     | 0.57    | 0.48 | 0.29 | 0.57 | 0.72   | 0.58     |
|         |                                       |             |         | Qwe     | n1.5-MoI | E-A2.7B-C | Chat     |         |      |      |      |        |          |
| Num=60  | Original                              | 8.12        | 12.97   | 11.62   | 0.40     | 0.71      | 0.81     | 0.59    | 0.60 | 0.31 | 0.74 | 0.66   | 0.60     |
|         | Frequency-prune                       | 11.44       | 15.40   | 14.24   | 0.33     | 0.57      | 0.77     | 0.55    | 0.43 | 0.29 | 0.73 | 0.65   | 0.54     |
|         | Output-prune                          | 11.09       | 18.00   | 16.65   | 0.34     | 0.59      | 0.71     | 0.52    | 0.48 | 0.27 | 0.66 | 0.59   | 0.52     |
| Num=45  | MC-SMoE                               | 12.76       | 17.45   | 16.39   | 0.37     | 0.65      | 0.76     | 0.53    | 0.38 | 0.25 | 0.78 | 0.67   | 0.55     |
|         | HC-SMoE                               | 11.62       | 16.39   | 15.40   | 0.34     | 0.66      | 0.75     | 0.53    | 0.50 | 0.28 | 0.70 | 0.61   | 0.55     |
|         | Sub-MoE (Ours)                        | 9.48        | 14.84   | 13.16   | 0.37     | 0.69      | 0.80     | 0.56    | 0.53 | 0.30 | 0.76 | 0.66   | 0.58     |
|         | Frequency-prune                       | 32.09       | 42.52   | 39.94   | 0.26     | 0.41      | 0.62     | 0.39    | 0.25 | 0.20 | 0.55 | 0.57   | 0.40     |
|         | Output-prune                          | 38.71       | 36.94   | 42.52   | 0.27     | 0.51      | 0.64     | 0.40    | 0.33 | 0.19 | 0.55 | 0.54   | 0.43     |
| Num=30  | MC-SMoE                               | 586.98      | 1865.43 | 2889.24 | 0.19     | 0.33      | 0.57     | 0.29    | 0.23 | 0.18 | 0.45 | 0.52   | 0.34     |
|         | HC-SMoE                               | 25.60       | 38.18   | 48.94   | 0.25     | 0.50      | 0.64     | 0.33    | 0.35 | 0.19 | 0.50 | 0.57   | 0.42     |
|         | Sub-MoE (Ours)                        | 17.51       | 29.00   | 25.28   | 0.32     | 0.58      | 0.51     | 0.46    | 0.38 | 0.25 | 0.57 | 0.58   | 0.46     |
|         |                                       |             |         |         | Qwen3-3  | 0B-A3B    |          |         |      |      |      |        |          |
| Num=128 | Original                              | 8.64        | 15.40   | 14.47   | 0.53     | 0.80      | 0.89     | 0.60    | 0.78 | 0.35 | 0.83 | 0.71   | 0.69     |
|         | HC-SMoE                               | 18.86       | 31.11   | 29.68   | 0.35     | 0.64      | 0.82     | 0.40    | 0.55 | 0.22 | 0.73 | 0.61   | 0.54     |
| Num=96  | Sub-MoE (Ours)                        | 13.59       | 23.48   | 21.38   | 0.44     | 0.70      | 0.86     | 0.47    | 0.65 | 0.25 | 0.76 | 0.66   | 0.60     |
|         | HC-SMoE                               | 72.33       | 162.99  | 148.41  | 0.23     | 0.44      | 0.63     | 0.29    | 0.30 | 0.13 | 0.50 | 0.50   | 0.38     |
| Num=64  | Sub-MoE (Ours)                        | 21.05       | 43.19   | 36.37   | 0.40     | 0.68      | 0.84     | 0.41    | 0.56 | 0.23 | 0.77 | 0.63   | 0.57     |
|         |                                       |             |         | D       | eepSeek- | MoE-16E   | }        |         |      |      |      |        |          |
| Num=64  | Original                              | 6.51        | 9.72    | 10.15   | 0.44     | 0.76      | 0.72     | 0.58    | 0.38 | 0.33 | 0.62 | 0.70   | 0.57     |
|         | HC-SMoE                               | 9.13        | 12.19   | 13.45   | 0.39     | 0.71      | 0.72     | 0.52    | 0.30 | 0.30 | 0.64 | 0.70   | 0.53     |
| Num=48  | Sub-MoE (Ours)                        | 8.48        | 11.29   | 12.60   | 0.40     | 0.72      | 0.73     | 0.54    | 0.32 | 0.27 | 0.66 | 0.70   | 0.55     |
|         | HC-SMoE                               | 15.34       | 21.07   | 23.30   | 0.31     | 0.60      | 0.69     | 0.43    | 0.24 | 0.20 | 0.57 | 0.64   | 0.46     |
| Num=32  | Sub-MoE (Ours)                        | 13.71       | 18.35   | 20.70   | 0.32     | 0.63      | 0.68     | 0.44    | 0.25 | 0.22 | 0.65 | 0.65   | 0.49     |
|         |                                       |             |         |         |          |           |          |         |      |      |      |        |          |

#### 4.1 Experimental Setups

**Models and Datasets.** We conduct experiments on 4 MoE LLMs: Mixtral 8x7B [18], Qwen3-235B-A22B [45] Qwen1.5-MoE-A2.7B [38] and DeepSeekMoE-16B-Base [6]. For Mixtral 8x7B, reducing experts from 8 to 4 decreases the model size from 46.7B to 24.2B parameters and reduces computational requirements from 2989 to 1546 GFLOPs. Similarly, for Qwen1.5-MoE, reducing experts from 60 to 30 results in a 43% reduction in model size (from 14.3B to 8.1B parameters). To evaluate our method comprehensively, we use two types of metrics: (1) perplexity on standard language modeling benchmarks including WikiText-2, PTB, and C4, and (2) accuracy on eight diverse reasoning and understanding tasks [10] like ARC [5], BoolQ [4], HellaSwag [47], MMLU [13], OBOA [29], RTE [1], and WinoG. [32].

**Implementation Details.** For our method, we use a calibration dataset of 128 samples, each containing 2,048 tokens sampled from WikiText-2, unless otherwise specified. In our subspace alignment process, we apply expert grouping based on functional similarity using the expert output metric and K-means clustering as our default configuration. For expert merging, we employ frequency-based V matrix merging, which weighs by their activation frequency in the calibration data. We provide reproduced results of expert pruning methods, frequency-prune, output-prune based on frequency in MoE-Compression [12] and expert merge methods (*i.e.*, MC-SMoE [23], HC-SMoE [3]).

#### 4.2 Performance Comparisons

Table 1 presents comprehensive comparisons of our Sub-MoE method against baseline approaches across four different MoE language models with varying degrees of expert reduction. The results demonstrate the consistent superiority of our proposed method across all evaluated models and compression ratios. For Mixtral-8×7B, when compressing from 8 to 6 experts, Sub-MoE achieves significantly better perplexity scores on WikiText-2 (5.16), PTB (18.58), and C4 (8.54) compared to pruning-based methods and other merging approaches. Notably, when reducing to just 4 experts (50% compression), our approach maintains impressive performance with an average accuracy of

<span id="page-7-2"></span><span id="page-7-0"></span>Table 2: Performance of Sub-MoE and MC-SMoE under extra intra-expert compression ratios. Runtime denotes runtime throughput (Tokens/sec) on 8x H800 GPUs.

| Model        | Ratio | Runtime | Method                     | WikiText-2↓             | PTB↓                    | C4↓                     | ARC-c               | ARC-e               | BoolQ               | HellaS.             | MMLU                | OBQA                | RTE                 | WinoG.              | Average↑            |
|--------------|-------|---------|----------------------------|-------------------------|-------------------------|-------------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|
| Mixtral 8x7B | 0     | 87.7    | Original                   | 3.98                    | 12.99                   | 6.78                    | 0.56                | 0.84                | 0.85                | 0.65                | 0.67                | 0.35                | 0.71                | 0.76                | 0.67                |
| Mixtral 6x7B | 10%   | 93.1    | MC-SMoE<br>Sub-MoE† (Ours) | 9.05<br><b>6.50</b>     | 65.86<br><b>40.33</b>   | 25.79<br>13.43          | 0.35<br><b>0.44</b> | 0.66<br><b>0.75</b> | 0.62<br><b>0.78</b> | 0.43<br><b>0.52</b> | 0.39<br><b>0.52</b> | 0.24<br><b>0.31</b> | 0.53<br><b>0.62</b> | 0.66<br><b>0.72</b> | 0.48<br><b>0.58</b> |
|              | 20%   | 104.7   | MC-SMoE<br>Sub-MoE† (Ours) | 12.96<br><b>7.97</b>    | 115.58<br><b>63.37</b>  | 49.71<br><b>20.48</b>   | 0.26<br><b>0.38</b> | 0.54<br><b>0.70</b> | 0.62<br><b>0.67</b> | 0.36<br><b>0.46</b> | 0.33<br><b>0.43</b> | 0.18<br><b>0.28</b> | 0.53<br><b>0.58</b> | 0.60<br><b>0.68</b> | 0.43<br><b>0.52</b> |
|              | 30%   | 120.9   | MC-SMoE<br>Sub-MoE† (Ours) | 50.49<br>11.22          | 314.19<br><b>106.72</b> | 135.13<br>38.16         | 0.20<br><b>0.29</b> | 0.42<br><b>0.60</b> | 0.39<br><b>0.63</b> | 0.34<br><b>0.38</b> | 0.26<br><b>0.33</b> | 0.16<br><b>0.22</b> | 0.53<br><b>0.53</b> | 0.54<br><b>0.61</b> | 0.35<br><b>0.45</b> |
|              | 10%   | 95.3    | MC-SMoE<br>Sub-MoE† (Ours) | 708.03<br><b>8.60</b>   | 1595.59<br><b>54.27</b> | 1204.41<br><b>16.15</b> | 0.21<br><b>0.40</b> | 0.26<br><b>0.70</b> | 0.40<br><b>0.72</b> | 0.27<br><b>0.48</b> | 0.27<br><b>0.41</b> | 0.11<br><b>0.28</b> | 0.50<br><b>0.56</b> | 0.49<br><b>0.69</b> | 0.31<br><b>0.53</b> |
| Mixtral 4x7B | 20%   | 108.2   | MC-SMoE<br>Sub-MoE† (Ours) | 730.51<br>10.23         | 1698.49<br><b>83.93</b> | 1322.79<br>23.71        | 0.21<br><b>0.34</b> | 0.25<br><b>0.65</b> | 0.40<br><b>0.65</b> | 0.27<br><b>0.43</b> | 0.26<br><b>0.37</b> | 0.12<br><b>0.23</b> | 0.52<br><b>0.53</b> | 0.50<br><b>0.65</b> | 0.32<br><b>0.48</b> |
|              | 30%   | 122.7   | MC-SMoE<br>Sub-MoE† (Ours) | 2113.81<br><b>14.82</b> | 2630.68<br>147.42       | 2471.30<br><b>47.70</b> | 0.22<br><b>0.26</b> | 0.27<br><b>0.55</b> | 0.44<br><b>0.62</b> | 0.26<br><b>0.36</b> | 0.25<br><b>0.29</b> | 0.12<br><b>0.19</b> | 0.47<br><b>0.53</b> | 0.51<br><b>0.61</b> | 0.32<br><b>0.43</b> |

<span id="page-7-1"></span>Table 3: Ablation on our (A) Expert Clustering and (B) Subspace Merging for Mixtral  $8x7B \rightarrow 6x7B$ .

| Settings          | Options                 | WikiText-2↓ | PTB↓         | C4↓         | ARC-c       | ARC-e       | BoolQ       | HellaS.     | MMLU        | OBQA        | RTE         | WinoG.      | Average↑    |
|-------------------|-------------------------|-------------|--------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
|                   |                         |             | (A) Ada      | ptive Ex    | pert Clus   | tering Set  | tings       |             |             |             |             |             |             |
| Clustering Layer  | Sub-MoE (1-Layer)       | 5.47        | 23.77        | 9.51        | 0.47        | 0.79        | 0.83        | 0.59        | 0.55        | 0.31        | 0.64        | 0.75        | 0.62        |
|                   | Sub-MoE (2-Layer)       | <b>5.16</b> | <b>18.58</b> | <b>8.54</b> | <b>0.49</b> | <b>0.80</b> | <b>0.86</b> | <b>0.62</b> | <b>0.59</b> | <b>0.32</b> | <b>0.65</b> | <b>0.75</b> | <b>0.64</b> |
|                   | Sub-MoE (3-Layer)       | 7.02        | 49.99        | 7.22        | 0.38        | 0.74        | 0.65        | 0.48        | 0.46        | 0.24        | 0.56        | 0.71        | 0.53        |
| Similarity Metric | Sub-MoE (Router-logits) | 5.65        | 16.59        | 9.17        | 0.48        | 0.76        | 0.82        | 0.61        | 0.59        | 0.31        | 0.64        | 0.74        | 0.62        |
|                   | Sub-MoE (Weight)        | 5.41        | 22.11        | 8.97        | 0.49        | 0.78        | 0.86        | 0.63        | 0.60        | 0.33        | 0.69        | 0.72        | 0.63        |
|                   | Sub-MoE (Expert output) | <b>5.16</b> | <b>18.58</b> | <b>8.54</b> | <b>0.49</b> | <b>0.80</b> | <b>0.86</b> | <b>0.62</b> | <b>0.59</b> | <b>0.32</b> | <b>0.65</b> | <b>0.75</b> | <b>0.64</b> |
| Clustering Alg.   | Sub-MoE (Random)        | 6.19        | 18.70        | 9.87        | 0.50        | 0.75        | 0.80        | 0.57        | 0.56        | 0.28        | 0.61        | 0.71        | 0.60        |
|                   | Sub-MoE (Hierarchical)  | 5.46        | 19.30        | 9.01        | 0.50        | 0.73        | 0.82        | 0.61        | 0.62        | 0.33        | 0.69        | 0.71        | 0.63        |
|                   | Sub-MoE (K-means)       | <b>5.16</b> | <b>18.58</b> | <b>8.54</b> | <b>0.49</b> | <b>0.80</b> | <b>0.86</b> | <b>0.62</b> | <b>0.59</b> | <b>0.32</b> | <b>0.65</b> | <b>0.75</b> | <b>0.64</b> |
|                   |                         |             | (B) Sub      | ospace E    | xpert Me    | rging Sett  | ings        |             |             |             |             |             |             |
| U-Sharing         | Sub-MoE (Vanilla SVD)   | 7.02        | 23.91        | 10.67       | 0.42        | 0.70        | 0.61        | 0.59        | 0.63        | 0.28        | 0.72        | 0.69        | 0.58        |
|                   | Sub-MoE (Union SVD)     | <b>5.16</b> | <b>18.58</b> | <b>8.54</b> | <b>0.49</b> | <b>0.80</b> | <b>0.86</b> | <b>0.62</b> | <b>0.59</b> | <b>0.32</b> | <b>0.65</b> | <b>0.75</b> | <b>0.64</b> |
| V-Merging         | Sub-MoE (Drop)          | 5.53        | 19.77        | 9.05        | 0.50        | 0.80        | 0.84        | 0.59        | 0.59        | 0.32        | 0.61        | 0.71        | 0.61        |
|                   | Sub-MoE (Average)       | 5.31        | 18.63        | 8.88        | 0.50        | 0.81        | 0.85        | 0.61        | 0.59        | 0.31        | 0.64        | 0.74        | 0.62        |
|                   | Sub-MoE (Frequency)     | <b>5.16</b> | <b>18.58</b> | <b>8.54</b> | <b>0.49</b> | <b>0.80</b> | <b>0.86</b> | <b>0.62</b> | <b>0.59</b> | <b>0.32</b> | <b>0.65</b> | <b>0.75</b> | <b>0.64</b> |

0.58 across reasoning tasks, substantially outperforming the next best method HC-SMoE (0.51) and far surpassing pruning-based approaches that struggle to exceed 0.39 average accuracy. The performance gap becomes even more pronounced with the Qwen1.5-MoE-A2.7B-Chat model, where compressing from 60 to 45 experts shows our Sub-MoE maintaining near-original performance (0.58 vs. 0.60) while other methods show significant degradation. When examining larger models like Qwen3-30B-A3B with 128 experts, Sub-MoE demonstrates remarkable resilience even at 50% compression (64 experts), maintaining 0.57 average accuracy while HC-SMoE drops dramatically to 0.38. This pattern repeats with DeepSeek-MoE-16B, where our approach consistently preserves more of the original model's capabilities across both language modeling and reasoning tasks. The substantial performance advantage of Sub-MoE becomes increasingly evident as the compression ratio increases, highlighting the effectiveness of our subspace alignment approach in preserving expert functionality compared to traditional merging or pruning techniques.

#### 4.3 Effect of Intra-Expert Compression

Table 2 compares Sub-MoE† against MC-SMoE under various intra-expert compression ratios. Our method consistently outperforms MC-SMoE across all settings, with dramatic differences at higher compression rates. When compressing Mixtral 4x7B with a 10% ratio, Sub-MoE† maintains reasonable perplexity scores (8.60, 54.27, 16.15 on WikiText-2, PTB, and C4), while MC-SMoE suffers catastrophic degradation. This performance gap widens as compression increases, demonstrating Sub-MoE†\*s robustness to parameter reduction. As shown in Figure 4 (Left), Sub-MoE† with fine-tuning (Sub-MoE†+FT) is able to additionally recover the accuracy significantly compared to more compressors, achieving gains of 4-6% over the base Sub-MoE across benchmarks. Our method obtains a stabilizing gain across diverse reasoning tasks, outperforming the other competitor ( $D^2$ -MoE [11]) by 6% on ARC-e, 5% on WinoGrande, and 6% on the challenging ARC-c dataset, demonstrating robust generalization capabilities even after substantial compression.

#### 4.4 Ablation Study

**Ablation on Core Components:** Table 3 presents a comprehensive ablation study on key components of our Sub-MoE. For the Clustering component (A), we investigate three critical design choices:

(1) Multi-layer configuration in Adaptive Allocation impacts performance, with 2-layer clustering (grouping 8×2=16 experts) yielding lowest perplexity and highest average accuracy. This balanced

<span id="page-8-1"></span><span id="page-8-0"></span>![](_page_8_Figure_0.jpeg)

Figure 4: Left: Comparison of Sub-MoE† against SVD-LLM [\[41\]](#page-11-14), NAEE [\[28\]](#page-10-5), MoE-I<sup>2</sup> [\[46\]](#page-11-3), MoE-SVD [\[24\]](#page-10-16), D<sup>2</sup> -MoE [\[11\]](#page-9-3) on 20% compressed Mixtral 6x7B. Middle: Effect of calibration sample size on perplexity. Right: Trade-off between expert count, memory usage, and accuracy for Mixtral MoE.

approach provides sufficient flexibility for identifying functional relationships while maintaining manageable cluster sizes. In contrast, 1-layer clustering limits the diversity of potential merge candidates, while 3-layer clustering creates overly complex groupings that lead to accuracy drops.

(2) Similarity Metric comparison reveals that while router-logits and weight-based similarity measures perform reasonably well, our expert output similarity metric achieves the best overall balance between language modeling and reasoning tasks (0.64 mean accuracy). (3) Clustering Algorithm analysis shows that K-means consistently delivers optimal results compared to random grouping or hierarchical clustering, particularly on language modeling tasks, though hierarchical clustering achieves comparable performance on reasoning tasks.

For Merging component (B), we examine two key aspects:

- (5) U-Sharing strategy comparison demonstrates that our union SVD approach substantially outperforms vanilla SVD across all metrics (8.54 vs. 10.67 on C4; 0.64 vs. 0.58 average accuracy), with particularly notable improvements on BoolQ (0.86 vs. 0.61). This confirms the effectiveness of our approach in finding a common representational space that preserves expert functionality.
- (6) V -Merging strategy experiments show that our frequency-based approach consistently outperforms both dropping the least significant components and simple averaging. The frequency-weighted approach maintains better overall performance, demonstrating the importance of respecting expert utilization patterns when merging. These ablation results empirically validate our design choices and demonstrate that each component of Sub-MoE contributes meaningfully to its overall effectiveness.

Impact of Calibration Size . Figure [4](#page-8-0) (Middle) shows how calibration sample size affects model performance. Increasing samples from 32 to 128 substantially reduces perplexity on WikiText-2 (8.69→5.16) and C4 (23.89→8.54), while further increases yield minimal gains.

Memory and Runtime Analysis. As shown in Figure [4](#page-8-0) (Right), reducing Mixtral-8×7B from 8 to 6 experts decreases memory by 24% with only a 6% accuracy drop, while compression to 4 experts achieves optimal efficiency with 48% memory reduction and 13% accuracy decline. Compression below 4 experts causes disproportionate performance degradation, indicating a practical lower bound for maintaining capabilities. For runtime, our Sub-MoE† can achieve 1.1∼1.3× throughput speedup (from 87.7 to 120.9 tokens/second in Table [2\)](#page-7-0) by compressing the weights of activated experts.

## 5 Conclusions

In this paper, we present Sub-MoE, a new expert merging framework that addresses parameter conflicts in MoE LLM compression through subspace alignment. By decomposing concatenated experts via SVD, our approach extracts shared U-matrices while enabling the effective merging of expert-specific V components. Our two-phase, Adaptive Expert Clustering and Subspace Expert Merging, identifies functionally similar experts and combines them with minimal information loss. Extensive experiments on Mixtral, DeepSeek, and Qwen MoE LLMs reveal that our approach consistently outperforms state-of-the-art pruning and merging baselines, achieving higher compression ratios with minimal loss in model efficacy. This superior performance stems from our ability to minimize parameter conflicts by operating in a common representation space and weighting expert contributions based on their activation patterns. Our approach offers immediate practical benefits for

deploying MoE models on resource-constrained devices while opening promising research directions, including applying subspace alignment to other models and developing more sophisticated merging strategies.

Limitations. Following most MoE compressions which rely on calibration datasets, our method needs calibration data during clustering and merging. We will explore data-free ways in future work.

## References

- <span id="page-9-13"></span>[1] Luisa Bentivogli, Peter Clark, Ido Dagan, and Danilo Giampiccolo. The fifth pascal recognizing textual entailment challenge. *TAC*, 7(8):1, 2009. [7](#page-6-1)
- <span id="page-9-5"></span>[2] Weilin Cai, Juyong Jiang, Le Qin, Junwei Cui, Sunghun Kim, and Jiayi Huang. Shortcut-connected expert parallelism for accelerating mixture-of-experts. *arXiv preprint arXiv:2404.05019*, 2024. [3](#page-2-1)
- <span id="page-9-2"></span>[3] I Chen, Hsu-Shen Liu, Wei-Fang Sun, Chen-Hao Chao, Yen-Chang Hsu, Chun-Yi Lee, et al. Retraining-free merging of sparse mixture-of-experts via hierarchical clustering. *arXiv preprint arXiv:2410.08589*, 2024. [2,](#page-1-2) [3,](#page-2-1) [7](#page-6-1)
- <span id="page-9-11"></span>[4] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 2924–2936. Association for Computational Linguistics, June 2019. [7](#page-6-1)
- <span id="page-9-10"></span>[5] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. *arXiv preprint arXiv:1803.05457*, 2018. [7](#page-6-1)
- <span id="page-9-4"></span>[6] Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y Wu, et al. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models. *arXiv preprint arXiv:2401.06066*, 2024. [3,](#page-2-1) [7](#page-6-1)
- <span id="page-9-0"></span>[7] DeepSeek-AI et al. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model, 2024. [1](#page-0-0)
- <span id="page-9-1"></span>[8] Peijie Dong, Lujun Li, Zhenheng Tang, Xiang Liu, Xinglin Pan, Qiang Wang, and Xiaowen Chu. Pruner-zero: Evolving symbolic pruning metric from scratch for large language models. In *ICML*, 2024. [1](#page-0-0)
- <span id="page-9-6"></span>[9] Peijie Dong, Lujun Li, Yuedong Zhong, Dayou Du, Ruibo Fan, Yuhan Chen, Zhenheng Tang, Qiang Wang, Wei Xue, Yike Guo, et al. Stbllm: Breaking the 1-bit barrier with structured binary llms. In *ICLR*, 2025. [3](#page-2-1)
- <span id="page-9-9"></span>[10] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, et al. A framework for few-shot language model evaluation, 2024. [7](#page-6-1)
- <span id="page-9-3"></span>[11] Hao Gu, Wei Li, Lujun Li, Zhu Qiyuan, Mark Lee, Shengjie Sun, Wei Xue, and Yike Guo. Delta decompression for moe-based LLMs compression. In *Forty-second International Conference on Machine Learning*, 2025. [2,](#page-1-2) [3,](#page-2-1) [6,](#page-5-0) [8,](#page-7-2) [9](#page-8-1)
- <span id="page-9-7"></span>[12] Shwai He, Daize Dong, Liang Ding, and Ang Li. Demystifying the compression of mixture-ofexperts through a unified framework. *arXiv preprint arXiv:2406.02500*, 2024. [3,](#page-2-1) [7](#page-6-1)
- <span id="page-9-12"></span>[13] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding, 2021. [7](#page-6-1)
- <span id="page-9-8"></span>[14] Alexander Holiday, Mahdi Kooshkbaghi, Juan M Bello-Rivas, C William Gear, Antonios Zagaris, and Ioannis G Kevrekidis. Manifold learning for parameter reduction. *Journal of computational physics*, 392:419–431, 2019. [6](#page-5-0)

- <span id="page-10-9"></span>[15] Wei Huang, Yue Liao, Jianhui Liu, Ruifei He, Haoru Tan, Shiming Zhang, Hongsheng Li, Si Liu, and XIAOJUAN QI. Mixture compressor for mixture-of-experts LLMs gains more. In *The Thirteenth International Conference on Learning Representations*, 2025. [3](#page-2-1)
- <span id="page-10-13"></span>[16] Abiodun M. Ikotun, Absalom E. Ezugwu, Laith Abualigah, Belal Abuhaija, and Jia Heming. K-means clustering algorithms: A comprehensive review, variants analysis, and advances in the era of big data. *Information Sciences*, 622:178–210, 2023. [4](#page-3-1)
- <span id="page-10-12"></span>[17] Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wilson. Averaging weights leads to wider optima and better generalization. *UAI*, 2018. [3](#page-2-1)
- <span id="page-10-8"></span>[18] Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, et al. Mixtral of experts, 2024. [3,](#page-2-1) [7](#page-6-1)
- <span id="page-10-3"></span>[19] Chenyu Jiang, Ye Tian, Zhen Jia, Shuai Zheng, Chuan Wu, and Yida Wang. Lancet: Accelerating Mixture-of-Experts Training via Whole Graph Computation-Communication Overlapping. *arXiv preprint arXiv:2404.19429*, 2024. [1](#page-0-0)
- <span id="page-10-10"></span>[20] Jaeseong Lee, Aurick Qiao, Daniel F Campos, Zhewei Yao, Yuxiong He, et al. Stun: Structuredthen-unstructured pruning for scalable moe pruning. *arXiv preprint arXiv:2409.06211*, 2024. [3](#page-2-1)
- <span id="page-10-1"></span>[21] Lujun Li, Peijie, Zhenheng Tang, Xiang Liu, Qiang Wang, Wenhan Luo, Wei Xue, Qifeng Liu, Xiaowen Chu, and Yike Guo. Discovering sparsity allocation for layer-wise pruning of large language models. In *NeuIPS*, 2024. [1](#page-0-0)
- <span id="page-10-11"></span>[22] Margaret Li, Suchin Gururangan, Tim Dettmers, Mike Lewis, Tim Althoff, Noah A Smith, and Luke Zettlemoyer. Branch-train-merge: Embarrassingly parallel training of expert language models. 2022. [3](#page-2-1)
- <span id="page-10-6"></span>[23] Pingzhi Li, Zhenyu Zhang, Prateek Yadav, Yi-Lin Sung, Yu Cheng, Mohit Bansal, and Tianlong Chen. Merge, then compress: Demystify efficient SMoe with hints from its routing policy. 2024. [2,](#page-1-2) [3,](#page-2-1) [7](#page-6-1)
- <span id="page-10-16"></span>[24] Wei Li, Lujun Li, You-Liang Huang, Mark G. Lee, Shengjie Sun, Wei Xue, and Yike Guo. Structured mixture-of-experts LLMs compression via singular value decomposition. In *ICML*, 2025. [9](#page-8-1)
- <span id="page-10-2"></span>[25] Wei Li, Lujun Li, Mark Lee, and Shengjie Sun. Als: Adaptive layer sparsity for large language models via activation correlation assessment. In *NeuIPS*, 2024. [1](#page-0-0)
- <span id="page-10-14"></span>[26] Deyuan Liu, Zhanyue Qin, Hairu Wang, Zhao Yang, Zecheng Wang, Fangying Rong, Qingbin Liu, Yanchao Hao, Xi Chen, Cunhang Fan, et al. Pruning via merging: Compressing llms via manifold alignment based layer merging. *arXiv preprint arXiv:2406.16330*, 2024. [6](#page-5-0)
- <span id="page-10-7"></span>[27] Enshu Liu, Junyi Zhu, Zinan Lin, Xuefei Ning, Matthew B Blaschko, Shengen Yan, Guohao Dai, Huazhong Yang, and Yu Wang. Efficient expert pruning for sparse mixture-of-experts language models: Enhancing performance and reducing inference costs. *arXiv preprint arXiv:2407.00945*, 2024. [2,](#page-1-2) [3](#page-2-1)
- <span id="page-10-5"></span>[28] Xudong Lu, Qi Liu, Yuhui Xu, Aojun Zhou, Siyuan Huang, Bo Zhang, Junchi Yan, and Hongsheng Li. Not all experts are equal: Efficient expert pruning and skipping for mixture-ofexperts large language models. In *ACL*, 2024. [2,](#page-1-2) [3,](#page-2-1) [9](#page-8-1)
- <span id="page-10-15"></span>[29] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In *EMNLP*, 2018. [7](#page-6-1)
- <span id="page-10-4"></span>[30] Alexandre Muzio, Alex Sun, and Churan He. Seer-moe: Sparse expert efficiency through regularization for mixture-of-experts, 2024. [2,](#page-1-2) [3](#page-2-1)
- <span id="page-10-0"></span>[31] Nam V Nguyen, Thong T Doan, Luong Tran, Van Nguyen, and Quang Pham. Libmoe: A library for comprehensive benchmarking mixture of experts in large language models. *arXiv preprint arXiv:2411.00918*, 2024. [1](#page-0-0)

- <span id="page-11-16"></span>[32] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. 2019. [7](#page-6-1)
- <span id="page-11-6"></span>[33] Soumajyoti Sarkar, Leonard Lausen, Volkan Cevher, Sheng Zha, Thomas Brox, and George Karypis. Revisiting smoe language models by evaluating inefficiencies with task specific expert pruning. *arXiv preprint arXiv:2409.01483*, 2024. [3](#page-2-1)
- <span id="page-11-2"></span>[34] Liang Shen, Zhihua Wu, WeiBao Gong, Hongxiang Hao, Yangfan Bai, HuaChao Wu, Xinxuan Wu, Jiang Bian, Haoyi Xiong, Dianhai Yu, et al. Se-moe: A scalable and efficient mixtureof-experts distributed training and inference system. *arXiv preprint arXiv:2205.10034*, 2022. [1](#page-0-0)
- <span id="page-11-11"></span>[35] Sidak Pal Singh and Martin Jaggi. Model fusion via optimal transport. *NeurIPS*, 2020. [6](#page-5-0)
- <span id="page-11-1"></span>[36] Yixin Song, Zeyu Mi, Haotong Xie, and Haibo Chen. Powerinfer: Fast large language model serving with a consumer-grade gpu, 2023. [1](#page-0-0)
- <span id="page-11-12"></span>[37] Norman Tatro, Pin-Yu Chen, Payel Das, Igor Melnyk, Prasanna Sattigeri, and Rongjie Lai. Optimizing mode connectivity via neuron alignment. *NeurIPS*, 2020. [6](#page-5-0)
- <span id="page-11-4"></span>[38] Qwen Team. Qwen1.5-moe: Matching 7b model performance with 1/3 activated parameters", February 2024. [3,](#page-2-1) [7](#page-6-1)
- <span id="page-11-10"></span>[39] Joshua B Tenenbaum, Vin de Silva, and John C Langford. A global geometric framework for nonlinear dimensionality reduction. *science*, 290(5500):2319–2323, 2000. [6](#page-5-0)
- <span id="page-11-13"></span>[40] David Vander Mijnsbrugge, Femke Ongenae, and Sofie Van Hoecke. Parameter efficient neural networks with singular value decomposed kernels. *IEEE Transactions on Neural Networks and Learning Systems*, 2021. [6](#page-5-0)
- <span id="page-11-14"></span>[41] Xin Wang, Yu Zheng, Zhongwei Wan, and Mi Zhang. Svd-llm: Truncation-aware singular value decomposition for large language model compression. *arXiv preprint arXiv:2403.07378*, 2024. [6,](#page-5-0) [9](#page-8-1)
- <span id="page-11-9"></span>[42] Mitchell Wortsman, Gabriel Ilharco, Samir Ya Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, et al. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. In *International Conference on Machine Learning*, pages 23965–23998. PMLR, 2022. [3](#page-2-1)
- <span id="page-11-7"></span>[43] Yanyue Xie, Zhi Zhang, Ding Zhou, Cong Xie, Ziang Song, Xin Liu, Yanzhi Wang, Xue Lin, and An Xu. Moe-pruner: Pruning mixture-of-experts large language model using the hints from its router. *arXiv preprint arXiv:2410.12013*, 2024. [3](#page-2-1)
- <span id="page-11-5"></span>[44] Leyang Xue, Yao Fu, Zhan Lu, Luo Mai, and Mahesh Marina. Moe-infinity: Activation-aware expert offloading for efficient moe serving, 2024. [3](#page-2-1)
- <span id="page-11-0"></span>[45] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. *arXiv preprint arXiv:2412.15115*, 2024. [1,](#page-0-0) [3,](#page-2-1) [7](#page-6-1)
- <span id="page-11-3"></span>[46] Cheng Yang, Yang Sui, Jinqi Xiao, Lingyi Huang, Yu Gong, Yuanlin Duan, Wenqi Jia, Miao Yin, Yu Cheng, and Bo Yuan. Moe-i2: Compressing mixture of experts models through inter-expert pruning and intra-expert low-rank decomposition. *arXiv preprint arXiv:2411.01016*, 2024. [2,](#page-1-2) [3,](#page-2-1) [9](#page-8-1)
- <span id="page-11-15"></span>[47] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, 2019. [7](#page-6-1)
- <span id="page-11-8"></span>[48] Hao Zhao, Zihan Qiu, Huijia Wu, Zili Wang, Zhaofeng He, and Jie Fu. HyperMoE: Towards better mixture of experts via transferring among experts. August 2024. [3](#page-2-1)