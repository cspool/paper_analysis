# 1 Introduction

The Transformer architecture has significantly advanced the capabilities of Large Language Models (LLMs), showcasing exceptional performance across a wide range of tasks due to its effective attention mechanism. This mechanism excels in precise sequence modeling and leverages the parallel processing capabilities of modern GPUs during training. However, the self-attention component scales quadratically with sequence length, leading to substantial computational demands that pose challenges for both training and inference.

To mitigate these issues, researchers have explored alternatives such as linear Transformers (Katharopoulos et al., 2020a), which replace traditional softmax-based attention with kernelized dot-product-based linear attention, substantially reducing memory requirements during inference by reframing as a linear RNN with matrix-valued states. While early versions of linear Transformers underperformed in language modeling tasks compared to standard Transformers, recent enhancements—such as incorporating data-dependent gating mechanisms akin to those in LSTMs, exemplified by models like GLA (Yang et al., 2024a) and Mamba2 (Dao & Gu, 2024a)—have shown promising improvements. However, challenges persist in managing information over long sequences, particularly for in-context retrieval tasks where traditional Transformers maintain their advantage (Arora et al., 2023a; 2024a; Jelassi et al., 2024; Wen et al., 2024; Akyürek et al., 2024).

This phenomenon is not surprising: linear Transformers can be interpreted as implementing an outer-product-based key-value association memory, reminiscent of tensor product representation (Smolensky, 1990). However, the number of orthogonal key-value pairs they can store is *bounded* by the model's dimensionality. When the sequence length exceeds this dimension, "memory collisions" become inevitable, hindering exact retrieval (Schlag et al., 2021a).

Mamba2 addresses this limitation by introducing a simple gated update rule,  $S_t = \alpha_t S_{t-1} + v_t k_t^{\mathsf{T}}$ , which uniformly decays all key-value associations at each time step by a dynamic ratio,  $\alpha_t \in (0, 1)$ .

<sup>\*</sup>Equation contribution. Work done during SY's internship at NVIDIA.

However, this approach does not account for the varying importance of different key-value associations, potentially leading to inefficient memory utilization. If the model needs to forget a specific key-value association, all key-value associations are equally forgotten, making the process less targeted and efficient.

In contrast, the linear Transformer with the delta rule (Widrow et al., 1960), known as DeltaNet (Schlag et al., 2021a; Yang et al., 2024b), selectively updates memory by (softly) replacing an old key-value pair with the incoming one in a sequential manner. This method has demonstrated impressive performance in synthetic benchmarks for in-context retrieval. However, since this process only modifies a single key-value pair at a time, the model lacks the ability to rapidly clear outdated or irrelevant information, especially during context switches where previous data needs to be erased. Consequently, DeltaNet has been found to perform moderately on real-world tasks (Yang et al., 2024b), likely due to the absence of a robust memory-clearing mechanism.

Recognizing the complementary advantages of the gated update rule and the delta rule in memory management, we propose the gated delta rule, a simple and intuitive mechanism that combines both approaches. This unified rule enables flexible memory control: it can promptly clear memory by setting  $\alpha_t \to 0$ , while selectively updating specific content without affecting other information by setting  $\alpha_t \to 1$  (effectively switching to the pure delta rule).

The remaining challenge lies in implementing the gated delta rule in a hardware-efficient manner. Building upon Yang et al. (2024b)'s efficient algorithm that parallelizes the delta rule computation using the WY representation (Bischof & Loan, 1985), we carefully extend their approach to incorporate the gating terms. Our extension preserves the benefits of chunkwise parallelism (Hua et al., 2022b; Sun et al., 2023a; Yang et al., 2024a;b), enabling hardware-efficient training.

Our resulting architecture, Gated DeltaNet, consistently outperforms both Mamba2 and DeltaNet across a comprehensive suite of benchmarks, including language modeling, commonsense reasoning, in-context retrieval, length extrapolation, and long-context understanding. Building on these results, we also develop hybrid architectures that strategically combine Gated DeltaNet layers with sliding window attention or Mamba2 layers, further enhancing both training efficiency and model performance.

## **PRELIMINARY**

#### 2.1 Mamba2: Linear Attention with Decay

It is known that the linear transformer (Katharopoulos et al., 2020b) can be formulated as the following linear recurrence when excluding normalization and query/key activations:

$$\mathbf{S}_t = \mathbf{S}_{t-1} + v_t k_t^\intercal \in \mathbb{R}^{d_v \times d_k}, \qquad \qquad o_t = \mathbf{S}_t q_t \in \mathbb{R}^{d_v}$$

 $\mathbf{S}_t = \mathbf{S}_{t-1} + \boldsymbol{v}_t \boldsymbol{k}_t^\mathsf{T} \in \mathbb{R}^{d_v \times d_k}, \qquad \boldsymbol{o}_t = \mathbf{S}_t \boldsymbol{q}_t \in \mathbb{R}^{d_v}$  where  $d_k$  and  $d_v$  represent the (head) dimensions for query/key and value, respectively. By expanding the recurrence, we can express it in both vector form (left) and matrix form (right) as follows:

$$\boldsymbol{o}_t = \sum_{i=1}^t (\boldsymbol{v}_i \boldsymbol{k}_i^\intercal) \boldsymbol{q}_t = \sum_{i=1}^t \boldsymbol{v}_i (\boldsymbol{k}_i^\intercal \boldsymbol{q}_t) \in \mathbb{R}^{d_v}, \qquad \mathbf{O} = (\mathbf{Q} \mathbf{K}^\intercal \odot \mathbf{M}) \mathbf{V} \in \mathbb{R}^{L \times d_v}$$

where L is the sequence length, and  $\mathbf{M} \in \mathbb{R}^{L \times L}$  is the causal mask defined by  $\mathbf{M}_{ij} = 0$  when i < j, and 1 otherwise.

However, this vanilla linear attention underperforms Transformers in language modeling by a large margin. To address this, it is common to add a decay term to forget historical information. Here we take Mamba2 (Dao & Gu, 2024a) as an example, which can be represented by the following linear recurrence (up to specific parameterization):

$$\mathbf{S}_t = \mathbf{\alpha}_t \mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^\mathsf{T}, \qquad \mathbf{o}_t = \mathbf{S}_t \mathbf{q}_t$$

where  $\alpha_t \in (0,1)$  is a data-dependent scalar-valued decay term that varies with t. Define the cumulative decay product  $\gamma_j = \prod_{i=1}^j \alpha_i$ , and by expanding the recurrence, we can express the result in both a vector form (left) and a matrix parallel form (right):

$$\boldsymbol{o}_t = \sum_{i=1}^t \left(\frac{\gamma_t}{\gamma_i} \boldsymbol{v}_i \boldsymbol{k}_i^\intercal \right) \boldsymbol{q}_t = \sum_{i=1}^t \boldsymbol{v}_i \left(\frac{\gamma_t}{\gamma_i} \boldsymbol{k}_i^\intercal \boldsymbol{q}_t \right), \qquad \mathbf{O} = \left( \left( \mathbf{Q} \mathbf{K}^\intercal \right) \odot \mathbf{\Gamma} \right) \mathbf{V}$$

Here,  $\Gamma \in \mathbb{R}^{L \times L}$  is a decay-aware causal mask where  $\Gamma_{ij} = \frac{\gamma_i}{\gamma_j}$  if  $i \geq j$  and  $\Gamma_{ij} = 0$  otherwise. The equivalence between these parallel and recurrent forms is also referred to as the state space duality (SSD) described in Dao & Gu (2024a). This recurrence structure appears in several other architectures including Gated RFA (Peng et al., 2021), xLSTM (Beck et al., 2024), and Gated RetNet (Sun et al., 2024b). When  $\gamma_t$  is data-independent, the formulation reduces to RetNet (Sun et al., 2023a) and Lightning-Attention (Qin et al., 2024a). Furthermore, if  $\gamma_t$  is extended to be matrix-valued rather than scalar-valued, efficient training algorithms remain possible when parameterized with an outer-product structure, as demonstrated by Yang et al. (2024a) and used by Yang et al. (2024a); Peng et al. (2024); Qin et al. (2024b); Zhang et al. (2024); Chou et al. (2024); He et al. (2025); Lu et al. (2025).

Chunkwise training However, both the recurrent and parallel forms are not ideal for efficient training (Hua et al., 2022b; Yang et al., 2024a), which motivates the use of the chunkwise parallel form (Hua et al., 2022b; Sun et al., 2023a) for hardware-efficient, linear-time training, as introduced below. To summarize, the chunkwise parallel form splits inputs and outputs into several chunks of size C, and computes outputs for each chunk based on the final state of the previous chunk and the query/key/value blocks of the current chunk. Following the notation of Sun et al. (2023b); Yang et al. (2024a;b), we take the query block, q, as an example. We denote  $\mathbf{Q}_{[t]} := q_{tC+1:(t+1)C+1}$  as the query block for chunk t, and  $q_{[t]}^r := q_{tC+r}$  as the r-th query within chunk t. The initial state of chunk t is defined as  $\mathbf{S}_{[t]} := \mathbf{S}_{[t]}^0 = \mathbf{S}_{[t]}^C$ . By partially expanding the recurrence, we have

chunk 
$$t$$
 is defined as  $\mathbf{S}_{[t]} := \mathbf{S}_{[t]}^0 = \mathbf{S}_{[t-1]}^C$ . By partially expanding the recurrence, we have 
$$\mathbf{S}_{[t]}^r = \mathbf{S}_{[t]} + \sum_{i=1}^r \boldsymbol{v}_{[t]}^i \boldsymbol{k}_{[t]}^{i\intercal} \in \mathbb{R}^{d_v \times d_k}, \qquad \boldsymbol{o}_{[t]}^r = \mathbf{S}_{[t]}^r \boldsymbol{q}_{[t]}^r = \mathbf{S}_{[t]} \boldsymbol{q}_{[t]}^r + \sum_{i=1}^r \boldsymbol{v}_{[t]}^i \left(\boldsymbol{k}_{[t]}^{i\intercal} \boldsymbol{q}_{[t]}^r\right) \in \mathbb{R}^{d_v}$$

Equivalently, in matrix form:

$$\mathbf{S}_{[t+1]} = \mathbf{S}_{[t]} + \mathbf{V}_{[t]} \mathbf{K}_{[t]}^\intercal \in \mathbb{R}^{d_v \times d_k}, \qquad \mathbf{O}_{[t]} = \mathbf{Q}_{[t]} \mathbf{S}_{[t]}^\intercal + \left(\mathbf{Q}_{[t]} \mathbf{K}_{[t]}^\intercal \odot \mathbf{M}\right) \mathbf{V}_{[t]} \in \mathbb{R}^{C \times d_v}$$

where  $\mathbf{M} \in \mathbb{R}^{C \times C}$  is the causal mask. The above equations are rich in matrix multiplications (matmuls), allowing for tensor-core-based hardware optimization. This chunkwise algorithm could be easily extended to linear attention with decay:

$$\mathbf{S}_{[t+1]} = \overrightarrow{\mathbf{S}_{[t]}} + \mathbf{V}_{[t]}^{\mathsf{T}} \overrightarrow{\mathbf{K}_{[t]}} \in \mathbb{R}^{d_v \times d_k}, \quad \mathbf{O}_{[t]} = \overleftarrow{\mathbf{Q}_{[t]}} \mathbf{S}_{[t]}^{\mathsf{T}} + \left( \mathbf{Q}_{[t]} \mathbf{K}_{[t]}^{\mathsf{T}} \odot \Gamma_{[t]} \right) \mathbf{V}_{[t]} \in \mathbb{R}^{C \times d_v} \quad (1)$$

where  $(\Gamma_{[t]})_{ij} = \frac{\gamma_{[t]}^i}{\gamma_{[t]}^j}, \gamma_{[t]}^j = \prod_{j=tC+1}^{tC+j} \alpha_j$ . Here we use the left arrow  $(\stackrel{\leftarrow}{\cdot})$  or the right arrow  $(\stackrel{\rightarrow}{\cdot})$  to denote a variable decaying to the first position and the last position of each chunk, respectively,

$$\begin{aligned} & \overrightarrow{q_{[t]}^r} = \gamma_{[t]}^r q_{[t]}^r & \text{decaying each vector to the first position of chunk } t \\ & \overrightarrow{k_{[t]}^r} = \frac{\gamma_{[t]}^C}{\gamma_{[t]}^r} k_{[t]}^r & \text{decaying each vector to the last position of chunk } t \\ & \overrightarrow{\mathbf{S}_{[t]}} = \gamma_{[t]}^C \mathbf{S}_{[t]} & \text{decaying the state matrix over the entire chunk } t \end{aligned}$$

<span id="page-2-3"></span><span id="page-2-2"></span>(2)

and likewise for other variables (e.g.,  $\overrightarrow{v}$ ). The SSD decomposition algorithm introduced in Mamba2 is largely equivalent to this chunkwise algorithm. For a more generalized approach, Yang et al. (2024a) proposed an extended chunkwise algorithm for linear attention that incorporates fine-grained decay mechanisms.

