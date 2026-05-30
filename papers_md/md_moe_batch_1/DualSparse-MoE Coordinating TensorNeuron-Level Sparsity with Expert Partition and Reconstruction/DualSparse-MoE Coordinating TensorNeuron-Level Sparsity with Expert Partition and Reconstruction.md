# DualSparse-MoE: Coordinating Tensor/Neuron-Level Sparsity with Expert Partition and Reconstruction

# Weilin Cai

The Hong Kong University of Science and Technology (Guangzhou) Guangzhou, China wcai738@connect.hkust-gz.edu.cn

# Junwei Cui

The Hong Kong University of Science and Technology (Guangzhou) Guangzhou, China jcui382@connect.hkust-gz.edu.cn

# Le Qin

The Hong Kong University of Science and Technology (Guangzhou) Guangzhou, China lqin674@connect.hkust-gz.edu.cn

# Ang Li

University of Maryland, College Park Maryland, USA angliece@umd.edu

# Shwai He

University of Maryland, College Park Maryland, USA shwaihe@umd.edu

# Jiayi Huang<sup>∗</sup>

The Hong Kong University of Science and Technology (Guangzhou) Guangzhou, China hjy@hkust-gz.edu.cn

# Abstract

Mixture of Experts (MoE) has become a mainstream architecture for building Large Language Models (LLMs) by reducing per-token computation while enabling model scaling. It can be viewed as partitioning a large Feed-Forward Network (FFN) at the tensor level into fine-grained sub-FFNs, or experts, and activating only a sparse subset for each input. While this sparsity improves efficiency, MoE still faces substantial challenges due to their massive computational scale and unpredictable activation patterns.

To enable efficient MoE deployment, we identify dual sparsity at the tensor and neuron levels in pre-trained MoE modules as a key factor for both accuracy and efficiency. Unlike prior work that increases tensor-level sparsity through finergrained expert design during pre-training, we introduce posttraining expert partitioning to induce such sparsity without retraining. This preserves the mathematical consistency of model transformations and enhances both efficiency and accuracy in subsequent fine-tuning and inference. Building upon this, we propose DualSparse-MoE, an inference system that integrates dynamic tensor-level computation dropping with static neuron-level reconstruction to deliver significant efficiency gains with minimal accuracy loss.

Experimental results show that enforcing an approximate 25% drop rate with our approach reduces average accuracy by only 0.08%—0.28% across three prevailing MoE models, while nearly all degrees of computation dropping consistently yield proportional computational speedups. Furthermore, incorporating load-imbalance awareness into expert parallelism achieves a 1.41× MoE module speedup with just 0.5% average accuracy degradation.

# 1 Introduction

Recently, the Mixture of Experts (MoE) architecture [\[12,](#page-11-0) [26,](#page-12-0) [39,](#page-12-1) [47\]](#page-13-0) has emerged as the mainstream design for Large Language Models (LLMs) [\[7,](#page-11-1) [21,](#page-12-2) [28,](#page-12-3) [35\]](#page-12-4), primarily due to

its superior trade-off between computational efficiency and model quality. This trade-off is achieved through the tensorlevel sparsity inherent in the MoE architecture, which can be viewed as partitioning a large Feed-Forward Network (FFN) into fine-grained sub-FFNs (termed experts) and selectively activates only a subset of these experts for processing each input token. By reducing computational workload compared to dense activation, existing hardware and systems can accommodate scaling model parameters to unprecedented sizes, thereby enabling higher levels of intelligence [\[29,](#page-12-5) [53,](#page-13-1) [54,](#page-13-2) [56\]](#page-13-3).

Building upon the inherent tensor-level sparsity of the MoE architecture, existing research has developed specialized methods to enhance the deployment of MoE models across various scenarios. For instance, Expert Parallelism (EP) and its systematic optimizations [\[6,](#page-11-2) [17,](#page-12-6) [29,](#page-12-5) [39,](#page-12-1) [49\]](#page-13-4) have been proposed to enable efficient distributed training and inference of MoE models; MoE compression techniques [\[8,](#page-11-3) [22,](#page-12-7) [23,](#page-12-8) [32\]](#page-12-9), motivated by observations of unbalanced expert selection, have been introduced to facilitate MoE model inference on edge devices. However, MoE models continue to present significant challenges for current machine learning systems, primarily due to their unprecedented model size and unpredictable sparse activation patterns.

To facilitate efficient post-training deployment of MoE models, we begin by analyzing the activation patterns within the MoE modules. As shown in Figure [1,](#page-1-0) we observe pronounced imbalance in activation at both the tensor and neuron levels. We refer to this pattern as dual sparsity, wherein the output of each FFN neuron is modulated by the product of its gating score and activation value. We further identify dual sparsity as a key factor affecting both the efficiency and accuracy of MoE models, evident in two aspects: (1) Tensor-level sparsity—prior studies [\[3,](#page-11-4) [12,](#page-11-0) [18,](#page-12-10) [33\]](#page-12-11) have shown that pre-training models with increased tensor-level sparsity through more granular expert design can improve model accuracy, but may also increase gate routing overhead and

1

<sup>∗</sup>Corresponding author.

lower compute intensity, leading to reduced GPU utilization; (2) Neuron-level sparsity—research on dense models has revealed a trade-off between accuracy and efficiency by omitting neurons with zero activation (as in ReLU) [27, 31] or negligible activation values (as in SwiGLU) [58].

The tensor-level sparsity configuration (expert granularity) established during pre-training may not be optimal for deployment. To address this, we propose expert partition methods (complete and partial transformations) to promote additional sparsity in the post-training phase. These methods preserve the mathematical consistency of model transformations while improving deployment efficiency. Specifically, complete transformation converts a pre-trained MoE model into one with finer-grained experts, thereby enhancing model quality during fine-tuning. In contrast, partial transformation targets efficiency improvements, notably by enabling our proposed Soft Expert-Tensor Parallelism (S-ETP), which provides benefits across diverse scenarios.

Moreover, we present DualSparse-MoE, an inference system that improves efficiency in a training-free manner while minimizing accuracy loss. DualSparse-MoE incorporates three key strategies: (1) Static expert partition and reconstruction, which divides neurons in each expert into major and minor sub-experts based on importance profiling from calibration samples; (2) Dynamic token-expert computation dropping, which selectively skips computations through dual thresholding of normalized gating scores, applying lower and upper thresholds to major and minor sub-experts, respectively; (3) Load-aware thresholding, which dynamically adjusts thresholds according to workload imbalance across devices, thereby reducing the drop rate and preserving accuracy while achieving the same speedup benefits in expert-parallel (EP) deployment.

In our experiments, we apply complete transformation to partition the Mixtral-8×7B model from 8 experts into 32 finer-grained experts, reducing fine-tuning loss and improving downstream accuracy by 0.59%. Moreover, partial transformation enables S-ETP, which improves EP communication efficiency in both real-world and simulated environments. DualSparse-MoE system demonstrates significant advantages: enforcing a approximate 25% drop rate reduces average benchmark accuracy by only 0.08%-0.28% across three MoE models. Notably, nearly all drop rates translate directly into proportional computation reduction and speedup, a result difficult to achieve with other sparsity-based acceleration techniques. With load-aware thresholding in EP, our method achieves a 1.41× MoE module speedup with only 0.5% average accuracy loss. Unlike previous selection-aware MoE compression methods designed for edge deployment, which often suffer from significant accuracy degradation and limited generalization, our approach is tailored for distributed server-side inference, where acceleration with minimal accuracy loss is essential.

In summary, our contributions are as follows:

<span id="page-1-0"></span>![](_page_1_Figure_6.jpeg)

**Figure 1.** Visualization of accumulated absolute activation values for each neuron across 64 SwiGLU FFN experts in a single MoE layer OLMoE [35] model during inference, highlighting tensor-level sparsity (y-axis) and neuron-level sparsity (x-axis) inherent to MoE architectures.

- We identify dual sparsity in MoE architectures at the tensor and neuron levels, and demonstrate their pivotal role in balancing accuracy and efficiency.
- We propose expert partition methods—complete and partial transformations—to induce tensor-level sparsity during the post-training phase, achieving both accuracy and efficiency.
- We design DualSparse-MoE, an inference system that integrates dynamic tensor-level computation dropping with static neuron-level reconstruction, improving efficiency with minimal accuracy loss.
- We conduct extensive experiments to demonstrate the effectiveness of our approach in enabling more efficient post-training deployment of MoE models.

#### 2 Background

#### <span id="page-1-2"></span>2.1 Sparsity in Mixture-of-Experts Architecture

By visualizing MoE activation patterns during inference with the pre-trained OLMoE model [35], as shown in Figure 1, we observe that the output of MoE module is governed by dual sparse at both the tensor and neuron levels. Specifically, color variations across rows (y-axis) reflect tensor-level sparsity, while color differences among points within each row (xaxis) capture neuron-level sparsity.

<span id="page-1-3"></span>**2.1.1 Tensor-Level Sparsity.** MoE [14, 26, 47] is a neural network architecture that dynamically selects experts to process each token. An MoE layer comprises E expert networks alongside a gating network G. The gating network, typically a linear network with a softmax activation function, calculates a selection probability (gating score s) for each expert, defined as:

<span id="page-1-1"></span>
$$\mathbf{s} = G(\mathbf{x}) = \text{Softmax}(\mathbf{x} \cdot \mathbf{W}_a),\tag{1}$$

where  $\mathbf{W}_g \in \mathbb{R}^{d_{model} \times E}$  is the weight matrix of the linear gating network. Based on the gating scores, the Top-K gating

method is commonly used to route each input token to a subset of experts for computation, defined as:

$$g_e(\mathbf{x}) = \begin{cases} \mathbf{s}_i & \text{if } i \in \text{TopK}(\mathbf{s}, K), \\ 0 & \text{otherwise,} \end{cases}$$
 (2)

where  $g_e(\mathbf{x})$  denotes the gating score for expert e. Each input token is processed by the K experts with the highest gating scores, and the MoE output is a weighted sum of the outputs of the selected experts:

$$\mathbf{y} = \sum_{e=1}^{E} g_e(\mathbf{x}) \cdot f_e(\mathbf{x}), \tag{3}$$

where  $f_e(\mathbf{x})$  denotes the output of expert e. Using the prevalent SwiGLU [46] feed-forward network (FFN) expert as an example, the expert output  $f(\mathbf{x})$  is formulated as:

<span id="page-2-2"></span>
$$f(\mathbf{x}) = (\operatorname{Swish}(\mathbf{x} \cdot \mathbf{W}_1) \odot (\mathbf{x} \cdot \mathbf{W}_3)) \cdot \mathbf{W}_2, \tag{4}$$

where  $\mathbf{W}_1, \mathbf{W}_3 \in \mathbb{R}^{d_{model} \times d_{ffn}}$  and  $\mathbf{W}_2 \in \mathbb{R}^{d_{ffn} \times d_{model}}$  denote FFN weights, and the Swish activation function [42] is used. The three linear transformations associated with  $\mathbf{W}_1, \mathbf{W}_3, \mathbf{W}_2$  are commonly referred to as the gate, up, and down projections, respectively.

Recent studies [3, 12, 18, 33] have systematically demonstrated that configuring experts at finer granularity—while maintaining a fixed per-token computational budget—can substantially reduce pre-training loss. For instance, a MoE model with 32 experts of intermediate size 1024 and Top-8 selection outperforms a MoE model with 8 experts of intermediate size 4096 and Top-2 selection. However, further increasing the number of experts may reduce computational efficiency due to lower compute intensity and increased gating overhead. These findings highlight a key trade-off: finer tensor-level sparsity can improve accuracy, but overly fine granularity may hurt efficiency.

Importantly, existing approaches determine expert granularity only before pre-training, making it difficult to adapt pre-trained MoE models to finer-grained structure for higher tensor sparsity. To address this limitation, we propose expert partition methods that restructure experts during post-training. Our approach preserves mathematical consistency while improving both accuracy and efficiency in deployment.

2.1.2 Neuron-Level Sparsity. In addition to the tensor-level sparsity inherent to the MoE architectures, prior research has investigated computation dropping and parameter pruning upon weight sparsity [13, 15, 52] and activation sparsity [51, 60] in dense FFN. However, these approaches face several challenges: (1) LLMs exhibit low tolerance to high drop or prune rates, with significant accuracy degradation as these rates increase; (2) Highly fine-grained sparsity at low drop rates is difficult to translate to real speedups due to limited sparsity support in hardware and kernel designs; and (3) Most methods focus primarily on the ReLU activation function [4], which naturally produce zeros, and

<span id="page-2-0"></span>![](_page_2_Figure_11.jpeg)

**Figure 2.** An example of a state-of-the-art 5-D hybrid parallel strategy for MoE model deployment. For simplicity, the illustration omits that EP can be extended cross DP groups. Our contributions mainly focus on improving the MoE part.

therefore cannot be directly applied to modern LLMs employing SwiGLU activations [46]. In this work, we identify the activation sparsity present in MoE FFN experts as neuron-level sparsity and propose a framework that coordinates it with tensor-level sparsity. By jointly leveraging these two complementary forms of sparsity, we address the aforementioned challenges and improve both algorithmic accuracy and system efficiency.

#### <span id="page-2-1"></span>2.2 Hybrid Parallelism for MoE Model Deployment

Scaling the training and inference of MoE models across distributed devices requires an effective hybrid parallelism strategy. In this work, we adopt one of the state-of-the-art hybrid parallelism strategies [2], as illustrated in Figure 2, to demonstrate the deployment pattern. This approach integrates Data Parallelism (DP) [40, 41, 44], Pipeline Parallelism (PP) [20, 36], Expert Parallelism (EP) [14, 26, 49], Tensor Parallelism (TP) [37, 48, 50], and Context Parallelism (CP) [1, 24]. Recent studies further suggest decoupling attention and MoE modules to enable more efficient resource allocation [30, 61]. Although specific implementations differ—for instance, EP may be realized using either AlltoAll or AllGather—this 5-D parallelism strategy is broadly representative and offers a general reference for both training and inference.

Importantly, tensor-level sparsity directly influences the choice and configuration of parallel strategies. In MoE layers, the number of experts and their intermediate sizes primarily affect EP and TP, since these parallel strategies handle the distribution of computational and memory loads and rely on communication patterns such as "AlltoAll+AllGather" and "ReduceScatter+AlltoAll." Therefore, the parallel strategy must comprehensively balance FLOPs utilization, communication overhead, GPU memory capacity, and other relevant factors.

![](_page_3_Figure_1.jpeg)

**Figure 3.** Illustration of expert partition methods, demonstrated by transforming a pre-trained 2-expert MoE model into a finer-grained 4-expert model. (a) Original MoE layer in the pre-trained model. (b) Complete transformation, which involves repeating the gating network weights, partitioning expert neurons, and scaling the down-projection weights  $\mathbf{W}_2$ . (c) Partial transformation, which involves partitioning expert neurons, repeating gating scores, and remapping expert indices.

#### 3 Expert Partition

We propose two expert partition methods: complete transformation, which restructures the model to increase tensor-level sparsity after pre-training, and partial transformation, which enables compute efficiency optimization.

#### 3.1 Complete Transformation

Complete transformation partitions each expert of a pretrained MoE model into P finer-grained experts (e.g. P=2, as shown in Figure 3(b)). This approach allows the transformed MoE model to integrate seamlessly with existing MoE frameworks, functioning identically to the original model. Specifically, the transformation involves three steps: (1) Repeat the gating network weights P times and adjust the Top-K selection to Top- $(K \times P)$ ; (2) Evenly partition the original experts' neurons into P finer-grained experts; (3) Scale the down-projection weight  $\mathbf{W}_2$  of each partitioned expert by a factor of P.

Next, we provide a formal derivation to demonstrate that this transformation ensures mathematical consistency. According to Equation (1) in Section 2.1, MoE module first employs  $\mathbf{W}_g = [h_1, h_2, \dots, h_E] \in \mathbb{R}^{d_{model} \times E}$  to compute the gating logits  $\mathbf{l}$ , where each  $h_i$  is a vector of dimension  $d_{model}$ . Given an input token  $\mathbf{x}_i$ , its gating logits are computed as:

$$\mathbf{l} = \mathbf{x}_i \cdot \mathbf{W}_a = [l_1, l_2, \dots, l_E]. \tag{5}$$

These gating logits are passed through a softmax function to obtain the gating scores  $\mathbf{s} = [s_1, s_2, \dots, s_E]$ , where the gating score  $s_e$  for the expert e is calculated as follows:

$$s_e = \frac{\exp(l_e)}{\sum_{i=1}^{E} \exp(l_i)}.$$
 (6)

In the complete transformation, each vector  $h_e$  in  $\mathbf{W}_g$  is repeated P times to construct the new gating weight matrix

<span id="page-3-1"></span><span id="page-3-0"></span> $\mathbf{W}_{a}^{P} \in \mathbb{R}^{d_{model} \times (E \times P)}$ , defined as:

$$\mathbf{W}_g = [h_{1,1}, h_{1,2}, \dots, h_{1,P}, h_{2,1}, h_{2,2}, \dots, h_{2,P}, \dots, h_{E,1}, h_{E,2}, \dots, h_{E,P}].$$
(7)

where  $h_{e,p}$  denotes the p-th copy of the e-th original expertspecific vector. Accordingly, the new gating logits  $\mathbf{l}^P$  for an input token  $\mathbf{x}_i$ , obtained via  $\mathbf{W}_a^P$ , can be expressed as:

$$\mathbf{l}^{P} = \mathbf{x}_{i} \cdot \mathbf{W}_{g}^{P} = [l_{1,1}, l_{1,2}, \dots, l_{1,P}, l_{2,1}, l_{2,2}, \dots, l_{2,P}, \dots, l_{E,1}, l_{E,2}, \dots, l_{E,P}],$$
(8)

where  $l_{e,1} = l_{e,2} = \dots = l_{e,P}$  due to the repeated vectors  $h_{e,1} = h_{e,2} = \dots = h_{e,P}$  in  $\mathbf{W}_g^P$ . Given the extended gating logits  $\mathbf{l}^P$ , the gating score for each finer-grained expert  $s_{e,p}$  is calculated as:

$$s_{e,p} = \frac{\exp(l_{e,p})}{\sum_{i=1}^{E} \sum_{l=1}^{P} \exp(l_{i,k})} = \frac{1}{P} \cdot \frac{\exp(l_{e})}{\sum_{i=1}^{E} \exp(l_{i})}.$$
 (9)

Since all P finer-grained experts partitioned from the same original expert share identical gating scores, they are activated together under the Top- $(K \times P)$  selection mechanism. Moreover, the sum of their outputs equals the original expert output, as shown below:

$$f_e(\mathbf{x}_i) = \sum_{p=1}^{P} f_{e,p}(\mathbf{x}_i), \tag{10}$$

which is analogous to the effect of tensor parallelism. Consequently, the output  $\mathbf{y}_i^P$  of the partitioned MoE module for an input token  $\mathbf{x}_i$  can be formulated as:

$$\mathbf{y}_{i}^{P} = \sum_{e=1}^{E} \sum_{p=1}^{P} \frac{1}{P} \cdot \frac{\exp(l_{e,p})}{\sum_{j=1}^{E} \sum_{k=1}^{P} \exp(l_{j,k})} \cdot f_{e,p}(\mathbf{x}_{i})$$

$$= \frac{1}{P} \sum_{e=1}^{E} \cdot \frac{\exp(l_{e})}{\sum_{j=1}^{E} \exp(l_{j})} \cdot \sum_{p=1}^{P} f_{e,p}(\mathbf{x}_{i}) = \frac{\mathbf{y}_{i}}{P}.$$
(11)

<span id="page-4-0"></span>![](_page_4_Figure_1.jpeg)

**Figure 4.** Fine-tuning loss curves for Mixtral-8×7B [21] models under different configurations, including the original model (activating top-2 out of 8 experts) and models completely transformed with partitioned experts (activating top-4 out of 16 experts and top-8 out of 32 experts).

This derivation shows that the complete transformation preserves the overall MoE output by a scaling factor of P.

To ensure that  $\mathbf{y}_i^P$  is equivalent to the original output  $\mathbf{y}_i$ , the result must be scaled by a factor of P. There are two ways to achieve this: (1) multiplying the gating scores by P, or (2) scaling the expert weight  $\mathbf{W}_2$  of the down-projection by P. To preserve the original model structure without modifying the existing framework, we choose to scale the expert weights for complete transformation. For example, in Figure 3(b), the down-projection weights  $\mathbf{W}_2$  of each partitioned expert is multiplied by 2, as P=2.

Model Quality Improvements During Fine-Tuning. As discussed in Section 2.1.1, prior work has demonstrated that increasing tensor-level sparsity during pre-trainingby configuring finer-grained experts-can enhance model quality. Our proposed expert partitioning method (complete transformation) effectively promotes such sparsity for pretrained MoE models, improving model performance during fine-tuning. As shown in Figure 4, models with partitioned experts exhibit significantly lower loss curves compared to the original Mixtral-8×7B [21] model; moreover, finergrained experts yield further loss reduction. However, increasing the number of partitions beyond a certain point offers only marginal improvements. Experimental results on the downstream tasks, presented in Table 1 and discussed in Section 5.2, further confirm the model quality gains attributable to our approach.

#### 3.2 Partial Transformation

Partial transformation offers an alternative approach for partitioning experts in pre-trained MoE models, as illustrated in Figure 3(c). In contrast to complete transformation, partial transformation preserves the original gating network and only modifies the results of the Top-K selection through two operations: (1) repeating the gating scores and (2) remapping the expert indices. Specifically, the original gating scores of the selected expert  $[s_1, s_2, \ldots, s_K]$  are repeated P times, resulting in  $[s_1, s_2, \ldots, s_K]^P$ . The corresponding expert indices

<span id="page-4-2"></span>![](_page_4_Figure_8.jpeg)

<span id="page-4-1"></span>**Figure 5.** Communication patterns in (a) Expert-Tensor Parallelism and (b) Soft Expert-Tensor Parallelism.

 $I = [i_1, i_2, ..., i_K]$  are remapped as follows:

$$\mathbf{I}^{P} = [i_{1}P, i_{2}P, \dots, i_{K}P, i_{1}P+1, i_{2}P+1, \dots, i_{K}P+1, \dots, i_{1}P+P-1, i_{2}P+P-1, \dots, i_{K}P+P-1],$$
(12)

where each original expert is partitioned and placed contiguously, maintaining its relative position. Each expert is evenly split into P finer-grained experts without scaling the down-projection weight. This is because the product of the partitioned experts' outputs and the repeated gating scores reproduces the original MoE outputs, formulated as follows:

$$\mathbf{y}_{i}^{P} = \sum_{e=1}^{E} \cdot \frac{\exp(l_{e})}{\sum_{j=1}^{E} \exp(l_{j})} \cdot \sum_{p=1}^{P} f_{e,p}(\mathbf{x}_{i}) = \mathbf{y}_{i}.$$
 (13)

While partial transformation requires additional modifications to the existing MoE framework, it reduces computational overhead compared to the extended gating network required by complete transformation, despite the gating network constituting only a minor portion of the overall MoE layer's cost. Moreover, partial transformation maintains the original gating network parameters, enabling mathematically consistent reverse transformation and focusing solely on system efficiency. Consequently, we apply partial transformation to the Soft Expert-Tensor Parallelism introduced in Section 3.3, as well as to the DualSparse-MoE inference system described in Section 4.2.

#### <span id="page-5-0"></span>3.3 Soft Expert-Tensor Parallelism (S-ETP)

As discussed in Section 2.2, EP and TP are employed to scale MoE deployment across distributed devices. In this context, applying TP to partition expert weights within EP is commonly referred to as Expert-Tensor Parallelism (ETP) [30, 49].

In contrast, we propose Soft Expert-Tensor Parallelism (S-ETP), which enables tensor-level partitioning of expert weight through an algorithmic approach rather than relying solely on system-level implementation. Specifically, S-ETP integrates expert partition (partial transformation) with EP to achieve the same functionality as ETP.

S-ETP offers the following advantages: (1) Reduced Framework Complexity. ETP often requires additional control mechanisms and framework modifications. In contrast, S-ETP addresses these challenges from an algorithmic rather than a system perspective, requiring only EP implementations, and thereby simplifying system optimization efforts. (2) Optimized Communication Patterns. S-ETP uses only the AlltoAll operation (Figure 5(b)) to achieve the same effects as the "AlltoAll+AllGather" and "ReduceScatter+AlltoAll" patterns used in ETP (Figure 5(a)). This approach reduces kernel launches and synchronization overhead, improving interconnect link utilization in both training and inference.

In addition to optimizing scenarios that traditionally require ETP, our expert partition approach also benefits cases that require scaling up EP. Specifically, our method enables the deployment of a larger number of experts, thereby involving more EP devices and enhancing scalability. Furthermore, the aforementioned advantages are also applicable to models restructured using complete expert transformation.

#### 4 DualSparse-MoE Inference System

Given the correlation between tensor-level sparsity and the superior accuracy-efficiency trade-off offered by MoE architectures, we analyze the behavior of the gating mechanism—arguably the most critical component of MoE—to identify exploitable features for inference acceleration.

Previous studies have observed imbalances in expert activation and have attempted to reduce computational cost or compress model size by skipping or pruning rarely activated experts [8, 22, 23, 32]. However, this approach often results in significant accuracy degradation and poor generalization across tasks, primarily due to the loss of dynamic tensor-level sparsity. As shown in Figure 6(a), expert activation patterns are highly dynamic across different benchmarks and input samples. We argue that diminishing the dynamic nature of tensor-level sparsity can harm model quality.

In contrast, our investigation reveals a relatively stable phenomenon within the gating mechanism: the distribution of gating scores for activated token-expert pairs. Figure 6(b) shows that across four tasks, the distribution of gating scores is remarkably similar, with most scores falling within the

<span id="page-5-2"></span><span id="page-5-1"></span>![](_page_5_Figure_10.jpeg)

<span id="page-5-3"></span>**Figure 6.** Distributions of (a) expert selection, (b) gating scores, and (c) normalized gating scores observed during OL-MoE model inference across four distinct benchmark tasks.

<span id="page-5-4"></span>![](_page_5_Figure_12.jpeg)

**Figure 7.** Benchmark accuracy and token-expert computation drop rate for OLMoE model inference using different threshold values of 1T-Drop. "Stars" indicate the threshold that achieves the maximum accuracy for each benchmark.

ranges of 0-0.05 and 0.05-0.1, and progressively fewer scores in higher ranges. Further normalization of the gating scores, as illustrated in Figure 6(c), produces a flatter distribution while preserving the consistent pattern across various tasks.

Building on this insight, we propose the DualSparse-MoE inference system, which selectively drops token-expert computations to enhance efficiency while minimizing accuracy loss. By exploiting the consistent characteristics of gating

<span id="page-6-1"></span>![](_page_6_Figure_1.jpeg)

**Figure 8.** Overview of the proposed dual-threshold token-expert computation dropping approach (2T-Drop) and its enhancement through load-aware thresholding under EP, in the context of deploying pre-trained MoE models for inference.

score distributions, our approach maintains generalizability and robust performance across a wide range of cases.

# <span id="page-6-2"></span>4.1 Token-Expert Dropping via Thresholding of Normalized Gating Scores

Since each token-expert computation is weighted by its corresponding gating score, a lower score indicates a lesser contribution to the final result. In the extreme case where the gating score is zero, the computation has no effect.

To improve efficiency, we propose an operation termed "1T-Drop", which selectively drops token-expert computations whose normalized gating scores fall below a specified threshold ( $T^1_{drop}$ ). Specifically, for each input token at each MoE layer, we normalize the gating scores of the Top-K activated experts and only retain experts whose normalized gating scores exceeding the threshold. The output of each token-expert computation remains weighted by its original gating score. It is worth noting that for some MoE models [29, 56] already normalize the gating scores of activated experts, this additional normalization step is unnecessary.

Interestingly, our empirical results show that applying a low threshold (approximately 0.05) for dropping computations can even improve accuracy, as illustrated in Figure 7. Across all benchmarks, the highest accuracy is achieved when some token-expert computations are dropped, suggesting that computations with very low gating scores may negatively impact overall performance. However, as the threshold increases further, thereby dropping more computations, the accuracy decreases across benchmarks. Note that the accuracy sensitivity to the drop operation also varies by task. For instance, GSM8K [11], a benchmark evaluating mathematical reasoning ability, exhibits the most pronounced accuracy decline as the drop rate increases.

#### <span id="page-6-0"></span>4.2 Dual-Threshold Token-Expert Dropping with Expert Partition and Reconstruction

Directly dropping token-expert computations based on a single threshold of normalized gating scores (1T-Drop) can lead to accuracy degradation, particularly at higher thresholds. Motivated by our observed dual sparsity in MoE models (Figure 1), we propose a dual-threshold token-expert dropping strategy, referred to as "2T-Drop". It coordinates both tensor-level and neuron-level sparsity to alleviate accuracy degradation while preserving computational savings and efficiency gains. As illustrated in Figure 8, 2T-Drop consists of the following three key operations:

- (a) Expert Partition. We employ the expert partition (partial transformation) method to enhance tensor-level sparsity, enabling finer-grained and thus more flexible combinations of token-expert computations dropping at the tensor level.
- (b) Expert Reconstruction. To exploit neuron-level sparsity within each expert, we perform neuron importance profiling on calibration samples. Neurons are then reorganized to reconstruct a major sub-expert comprising neurons of higher importance and a minor sub-expert comprising those of lower importance. Note that in our implementation, expert partitioning and reconstruction are executed as a unified process: all neurons in an original expert are first profiled and then reorganized into two separate sub-experts, one major and one minor. We employ this static approach to leverage neuron-level sparsity and avoid the challenges of dynamically identifying neuron activations for runtime dropping.

Furthermore, we experiment with various neuron importance profiling methods within SwiGLU experts: (1) accumulated gate value

$$Importance = \sum Swish(\mathbf{x} \cdot \mathbf{W}_1^{neuron}), \tag{14}$$

| Model                      | E-Activ./Total | $T_{drop}^1$ | Drop Rate | ARC-C | BoolQ | GSM8K | HellaSwag | MMLU  | OBQA  | PIQA  | RTE   | WinoGrande | AVG.(↑)      |
|----------------------------|----------------|--------------|-----------|-------|-------|-------|-----------|-------|-------|-------|-------|------------|--------------|
|                            | 2/8            | -            | 0         | 59.47 | 85.14 | 58.07 | 84.05     | 67.13 | 47.00 | 83.79 | 70.40 | 76.56      | 70.18        |
| Mixtral-8×7B               | $4/16 \ (P=2)$ | -            | 0         | 59.56 | 85.32 | 58.30 | 84.02     | 67.05 | 47.20 | 83.41 | 70.76 | 76.01      | 70.18        |
|                            | $8/32 \ (P=4)$ | -            | 0         | 59.47 | 85.26 | 58.07 | 83.99     | 67.22 | 46.80 | 83.46 | 70.76 | 76.72      | <u>70.19</u> |
| Fine-Tuned<br>Mixtral-8×7B | 2/8            | -            | 0         | 60.58 | 87.06 | 60.73 | 82.99     | 64.92 | 46.20 | 83.62 | 71.84 | 76.87      | 70.53        |
|                            | $4/16 \ (P=2)$ | -            | 0         | 59.56 | 87.06 | 62.85 | 82.96     | 65.65 | 47.00 | 83.3  | 72.92 | 76.48      | 70.86        |
| Mixuai-ox/D                | $8/32 \ (P=4)$ | -            | 0         | 60.67 | 87.55 | 62.85 | 83.06     | 65.10 | 47.60 | 83.46 | 72.92 | 76.87      | 71.12        |
| Fine-Tuned                 | 2/8            | 0.30         | 20.3%     | 59.39 | 87.06 | 61.84 | 82.72     | 64.26 | 46.40 | 82.86 | 71.48 | 76.64      | 70.29        |
| Mixtral-8×7B               | $4/16 \ (P=2)$ | 0.15         | 21.0%     | 59.64 | 87.00 | 63.46 | 82.58     | 64.75 | 46.40 | 83.13 | 73.65 | 76.24      | 70.76        |
| Threshold Drop             | 8/32 (P = 4)   | 0.08         | 23.9%     | 59.73 | 87.31 | 62.85 | 82.75     | 64.76 | 47.00 | 83.03 | 74.01 | 76.48      | 70.88        |

<span id="page-7-0"></span>**Table 1.** Comparison of downstream accuracy between the original Mixtral-8×7B model and its expert-partitioned variant.

(2) accumulated absolute gate value

$$Importance = \sum \left| Swish(\mathbf{x} \cdot \mathbf{W}_{1}^{neuron}) \right|, \tag{15}$$

(3) accumulated gate-up value

$$Importance = \sum (Swish(\mathbf{x} \cdot \mathbf{W}_1^{neuron}) \odot (\mathbf{x} \cdot \mathbf{W}_3^{neuron})), (16)$$

(4) accumulated absolute gate-up value

$$Importance = \sum \left| Swish(\mathbf{x} \cdot \mathbf{W}_1^{neuron}) \odot (\mathbf{x} \cdot \mathbf{W}_3^{neuron}) \right|. (17)$$

Here,  $\mathbf{W}_1^{\text{neuron}}$  and  $\mathbf{W}_3^{\text{neuron}}$  denote each neuron's  $\mathbf{W}_1$  and  $\mathbf{W}_3$  weights, following the formulation of the SwiGLU expert in Equation (4). Empirically, we observe that different models exhibit varying affinities for different profiling methods, highlighting the need to empirically determine the optimal configuration for each specific model.

In addition, we have considered partitioning experts into more granular based on neuron importance, which may yield higher accuracy. However, since this approach could reduce computational intensity and lead to low GPU utilization, we choose to partition and reconstruct each original expert into only two sub-experts.

(c) Dual-Threshold Drop. Building on the reconstructed minor and major experts, we propose the dual-threshold drop (2T-Drop) method. This approach applies token-expert computation dropping using a higher threshold-minor ( $T_{minor}^2$ ) for minor sub-experts and a lower threshold-major ( $T_{major}^2$ ) for major sub-experts. Specifically, original experts with gating scores above  $T_{minor}^2$  are fully engaged in computation, while those with gating scores below  $T_{major}^2$  are entirely dropped—similar to the 1T-Drop method. Uniquely, experts with gating scores between  $T_{minor}^2$  and  $T_{major}^2$  compute only the major half of their neurons. Based on our empirical experiments, we select dual thresholds of  $T_{major}^2 = T_{drop}^1 - 0.01$  and  $T_{minor}^2 = T_{drop}^1 + 0.01$ , which preserve a similar drop rate while achieving higher inference accuracy.

Given that our approach affects the computation granularity of token-expert grouped-GEMM and introduces additional control operations in the gating function, we optimize the corresponding Triton kernel to enhance efficiency.

#### 4.3 Load-Aware Thresholding in Expert Parallelism

Load imbalance among distributed devices is a major factor limiting efficiency in MoE model inference with expert parallelism. Since the overall MoE computation is blocked by the device with the heaviest computational load, simply dropping computations uniformly across devices can unnecessarily degrade accuracy on devices with lighter workloads.

To address this, we propose a load-aware thresholding mechanism that dynamically adjusts token-expert dropping based on the load of each device. This approach enables the system to adaptively balance computation across devices while maintaining high accuracy.

As shown in Figure 8(d), we employ a step-down thresholding strategy: devices with higher workloads apply higher thresholds, dropping more token-expert computations, while devices with lighter workloads use lower thresholds. To minimize control overhead in distributed environments, we calculate the ratio of the actual load to the ideal balanced load for each device. If this ratio exceeds 1, the threshold is set to a predefined maximum value; if it is below 1, the threshold is proportionally reduced according to the deviation from 1. This method ensures that all devices drop computation as little as possible, while maintaining their workload at or below that of the originally most-loaded device. By incorporating load-aware thresholding in expert parallelism, our approach achieves higher inference accuracy while maintaining the same level of acceleration.

#### 5 Evaluation

#### 5.1 Experimental Setup

To evaluate the efficacy of our proposed methods, we conduct experiments on a server equipped with 8 Nvidia H20 GPUs. Specifically, we utilize EleutherAI's LM-Evaluation-Harness [16] to assess model quality, reporting either accuracy or normalized accuracy for each benchmark, as applicable. Our evaluation tasks include zero-shot evaluations on the ARC-C [10], BoolQ [9], HellaSwag [57], MMLU [19], OBQA [34], PIQA [5], RTE [55], and WinoGrande [45] benchmarks, as well as 5-shot evaluation on GSM8K [11]. We utilize the Tulu-3-sft-mixture dataset [25] for our fine-tuning experiments.

<span id="page-8-1"></span>![](_page_8_Figure_1.jpeg)

![](_page_8_Figure_2.jpeg)

Figure 9. Comparison of communication bandwidth across different input sizes using ETP and S-ETP. In real-world tests (a), "E2T4" denotes a configuration with EP=2 and TP=4, while "E4T2" denotes a configuration with EP=4 and TP=2. In simulation (b), NVL72 [\[38\]](#page-12-27) is configured with EP=9 and TP=8, whereas CloudMatrix384 (CM384) [\[62\]](#page-13-17) is configured with EP=48 and TP=8.

Furthermore, we implement our proposed DualSparse-MoE inference system and evaluate its acceleration effectiveness upon the SGLang framework [\[59\]](#page-13-18), which supports efficient distributed inference for prevailing MoE models such as Mixtral [\[21\]](#page-12-2), OLMoE [\[35\]](#page-12-4), and DeepSeek [\[28\]](#page-12-3).

Additionally, we perform small-scale real-world tests using the PyTorch Distributed framework with the NCCL backend, as well as large-scale simulations using the ASTRA-SIM simulator [\[43\]](#page-13-19) to evaluate the communication optimization achieved by the Soft Expert-Tensor Parallelism (S-ETP).

#### <span id="page-8-0"></span>5.2 Evaluation of Expert Partition

We conduct experiments to substantiate the benefits of promoting tensor-level sparsity during the post-training phase, using our proposed expert partition methods.

<span id="page-8-2"></span>5.2.1 Model Quality Gains during Fine-tuning. We apply the expert partition (complete transformation) to the Mixtral-8×7B model, partitioning its original 8 experts into 16 ( = 2) and 32 ( = 4) finer-grained experts. As shown in Table [1,](#page-7-0) the partitioned models demonstrate the same downstream accuracy, with only negligible fluctuations. This consistency is attributed to the mathematical equivalence maintained by the partitioning process, although minor variations may arise due to floating-point precision errors. While models with partitioned experts exhibit significantly lower fine-tuning loss curves in Figure [4,](#page-4-0) these partitioned models also achieve higher downstream accuracy after fine-tuning. Notably, even when applying a 1T-Drop with a 23.9% drop rate to the partitioned model ( = 4), this model still achieves a higher average downstream accuracy of 70.88% than the 70.53% accuracy attained by the fine-tuned original model.

# 5.2.2 Efficiency Improvements Achieved via S-ETP. In Figure [9,](#page-8-1) our proposed S-ETP method exhibits significant improvements in communication bandwidth compared to existing ETP approach. The bandwidth is measured by dividing the input size per device by the total communication time. In a real-world test configuration of EP=4 and TP=2 on

an 8×H20 node, S-ETP achieves a bandwidth improvement ranging from 3.0% to 29.9%. When configured with EP=2 and TP=4, the improvement ranges from 9.2% to 15.2%.

Furthermore, the benefits of S-ETP are particularly evident in systems equipped with fully peer-to-peer high-bandwidth interconnections, such as NVL72 [\[38\]](#page-12-27) and CloudMatrix384 [\[62\]](#page-13-17). These systems feature homogeneous network architectures, eliminating the substantial disparities typically observed between inter-node and intra-node bandwidth. Our simulations in these environments reveal improvements of 10.2% to 80.4% on NVL72 and 9.9% to 28.3% on CloudMatrix384.

#### 5.3 Evaluation of DualSparse-MoE Inference System

Given that DualSparse-MoE inference system is proposed to enhance efficiency in a training-free manner, with minimal impact on accuracy, we evaluate it on both accuracy and efficiency perspectives in the following subsections.

5.3.1 Impact on Accuracy. As shown in Table [2,](#page-9-0) applying 1T-Drop for MoE computation dropping on the evaluated models leads to a relatively significant accuracy degradation, while applying 2T-Drop with only expert partition results in a similar level of accuracy loss. In contrast, when expert partition is combined with the reconstruction of major and minor sub-experts, 2T-Drop substantially minimizes the accuracy loss at the same drop rate. Specifically, imposing an approximate 25% drop rate yields only a 0.08% reduction in average accuracy for Mixtral, 0.28% for OLMoE, and 0.18% for DeepSeek. In particular, since the DeepSeek-V2-Lite-Chat model utilizes the shared expert architecture, its drop rate is calculated as the ratio of dropped routed expert computations to the total routed and shared expert computations.

Moreover, the drop methods exceed the accuracy of the baseline with no drop in some tasks. This phenomenon may be attributed to the same factors discussed in Section [4.1,](#page-6-2) where applying an appropriate threshold for dropping can even enhance accuracy. Furthermore, the accuracy impact of different drop methods appears to vary across tasks.

<span id="page-9-0"></span>

| <b>Table 2.</b> Comparison of downstream accuracy across different drop methods evaluated on three models. Note that setting                        |
|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| $T_{major}^2 = T_{minor}^2$ is equivalent to using 1T-Drop with $T_{drop}^1$ . 2T (partition) denotes the 2T-Drop without neuron-level reconstruct. |

| Model                                   | Drop Method      | $T_{major}^2$ | T <sub>minor</sub> | Drop Rate | ARC-C | BoolQ | GSM8K | HellaSwag | MMLU  | OBQA  | PIQA  | RTE   | WinoGrande | AVG.(↑)      |
|-----------------------------------------|------------------|---------------|--------------------|-----------|-------|-------|-------|-----------|-------|-------|-------|-------|------------|--------------|
| Fine-Tuned Mixtral-8×7B $(8/32, P = 4)$ | No Drop          | -             | -                  | 0         | 60.67 | 87.55 | 62.85 | 83.06     | 65.10 | 47.60 | 83.46 | 72.92 | 76.87      | 71.12        |
|                                         | 1T-Drop          | 0.08          | 0.08               | 23.9%     | 59.73 | 87.31 | 62.85 | 82.75     | 64.76 | 47.00 | 83.03 | 74.01 | 76.48      | 70.88        |
|                                         | 2T (Partition)   | 0.07          | 0.09               | 24.0%     | 59.47 | 87.25 | 63.15 | 82.90     | 64.62 | 47.00 | 83.51 | 74.37 | 75.85      | 70.90        |
| , , ,                                   | 2T (Reconstruct) | 0.07          | 0.09               | 24.0%     | 58.79 | 87.40 | 63.61 | 82.26     | 64.78 | 47.60 | 82.86 | 74.73 | 77.03      | 71.04        |
|                                         | No Drop          | -             | -                  | 0         | 49.40 | 76.64 | 67.85 | 80.70     | 52.86 | 47.60 | 78.18 | 72.20 | 67.72      | <u>65.91</u> |
| OLMoE-Instruct                          | 1T-Drop          | 0.08          | 0.08               | 21.7%     | 48.46 | 77.40 | 64.67 | 80.11     | 52.23 | 45.60 | 78.35 | 72.20 | 66.85      | 65.10        |
|                                         | 2T (Partition)   | 0.07          | 0.09               | 22.0%     | 48.12 | 76.70 | 68.54 | 80.04     | 52.81 | 44.80 | 77.26 | 71.12 | 66.61      | 65.11        |
|                                         | 2T (Reconstruct) | 0.07          | 0.09               | 22.0%     | 50.00 | 77.00 | 67.22 | 80.13     | 52.45 | 47.80 | 79.38 | 71.12 | 65.59      | 65.63        |
| DeepSeek-V2-Lite-Chat                   | No Drop          | -             | -                  | 0         | 53.92 | 82.91 | 65.05 | 80.81     | 56.91 | 45.20 | 81.12 | 72.56 | 71.98      | 67.83        |
|                                         | 1T-Drop          | 0.12          | 0.12               | 27.0%     | 51.79 | 82.91 | 63.61 | 80.30     | 55.18 | 44.40 | 81.61 | 74.37 | 71.27      | 67.27        |
|                                         | 2T (Partition)   | 0.11          | 0.13               | 26.9%     | 52.13 | 83.43 | 63.53 | 80.23     | 55.13 | 45.80 | 80.96 | 74.01 | 71.11      | 67.37        |
|                                         | 2T (Reconstruct) | 0.11          | 0.13               | 26.9%     | 52.47 | 82.94 | 64.37 | 80.37     | 55.58 | 44.80 | 81.39 | 74.73 | 72.22      | 67.65        |

![](_page_9_Figure_3.jpeg)

**Figure 10.** Comparison of the actual speedups achieved by 1T-Drop and 2T-Drop with the drop rates reported in Table 2. Specifically, Mixtral is deployed with TP=8 on an 8×H20 node, OLMoE is deployed on a single H20 GPU, and DeepSeek is deployed with EP=8 on an 8×H20 node.

Additionally, we conduct experiments on the fine-tuned and expert-partitioned Mixtral model, as introduced in Section 5.2.1, to demonstrate its compatibility with our proposed model transformation and inference acceleration techniques.

5.3.2 Efficiency Improvement. Given the comparable MoE computation drop rates achieved by various drop methods, as shown in Table 2, we evaluate their actual speedup across different models and deployment strategies. To demonstrate the broad applicability of our approaches, we conduct experiments using diverse deployment strategies, including single GPU and multi-GPU setups with TP or EP. These experiments are conducted on 2,000 random prompts with input lengths set to 500 and output lengths set to 100. The proposed drop methods are applied to achieve the drop rates shown in Table 2. The results demonstrate that our methods consistently yield speedups across various deployment configurations, attributable to the reduction in computation volume achieved by our approach.

Notably, the observed MoE computation drop rates of 22% to 27% can be effectively translated into actual speedups

<span id="page-9-1"></span>![](_page_9_Figure_8.jpeg)

**Figure 11.** Comparison of speedup and accuracy among 1T-Drop, 2T-Drop, and 2T-Drop with load-aware thresholding for DeepSeek-V2-Lite-Chat model inference on an  $8\times$ H20 node with EP=8.  $T^1$  represents the threshold applied in 1T-Drop, while  $T^2$  denotes the thresholds utilized in 2T-Drop.

for the MoE module, ranging from 1.17 to 1.23, and end-toend speedups of 1.07 to 1.12. This is primarily because our methods perform dropping at the tensor level, making them well-suited for existing computing devices. In contrast, current sparsity-based acceleration methods struggle to achieve meaningful speedups at such low drop rates, as they require specialized hardware and custom kernels. Furthermore, by employing optimized computing kernels, we achieve comparable speeds for 2T-Drop and 1T-Drop, even though 2T-Drop performs finer-grained computation drops than 1T-Drop.

# **5.3.3 Improvement with Load-Aware Thresholding.** As illustrated in Figure 11, increasing the drop threshold results in higher acceleration but leads to a reduction in accuracy. Since the accuracy of the reasoning task GSM8K [11] is particularly sensitive to the computation drop rate, we report both the accuracy on GSM8K and the average accuracy across downstream tasks presented in Table 2.

It is evident that 2T-Drop achieves higher accuracy compared to 1T-Drop, and that load-aware thresholding enhances

<span id="page-10-0"></span>![](_page_10_Figure_1.jpeg)

**Figure 12.** Drop rates across different layers of the OLMoE-Instruct model as a function of varying thresholds. The proportions in legend represent the overall drop rate of all layers.

the accuracy of 2T-Drop significantly. Specifically, with the integration of load-aware thresholding, 2T-Drop achieves a 1.41× speedup for the MoE module and a 1.13× end-to-end speedup, while incurring only a 0.5% average accuracy loss. During the practical deployment of DualSpare-MoE inference, the drop threshold can be dynamically adjusted to meet specific requirements for accuracy or throughput.

Given that load-aware thresholding requires dynamic modification of the drop threshold upon device workload, we analyze the relationship between the threshold value and the computation drop rate for tested models. In Figure 12, which presents this relationship for the OLMoE-Instruct model, the drop rate does not change linearly with increasing threshold values. It indicates the need for a tailored mapping between threshold and drop rate. Furthermore, the drop rate varies across different layers, suggesting potential for further exploration of per-layer thresholding strategies in future work.

5.3.4 Analysis of Neuron Importance Profiling. Given the four profiling methods introduced in Section 4.2, we select the accumulated absolute gate value as the neuron importance metric for the Mixtral and OLMoE models, and the accumulated absolute gate-up value for the DeepSeek model. Using DeepSeek-V2-Lite-Chat as an example, the average accuracy for each profiling method is as follows: 67.17 for accumulated gate, 67.29 for accumulated absolute gate, 66.79 for accumulated gate-up, and 67.65 for accumulated absolute gate-up. These results indicate that profiling methods based on absolute values are more effective in assessing neuron importance, probably because they prevent positive and negative contributions from canceling each other out.

As shown in Figure 13, low-load experts exhibit numerous negative accumulated gate values, whereas such values are

<span id="page-10-1"></span>![](_page_10_Figure_7.jpeg)

**Figure 13.** Comparison of neuron importance values derived from four profiling methods for expert 15 (high-load) and expert 21 (low-load) in layer 20 of DeepSeek-V2-Lite-Chat.

uncommon in high-load experts. This observation suggests a potential interconnection between tensor-level and neuron-level activation, underscoring the dual sparsity characteristic of MoE architectures. In contrast, accumulated gate-up values display similar distributions between low-load and high-load experts, which may account for the superior performance of this method in profiling the DeepSeek model.

Additionally, the selection of calibration samples also impacts the effectiveness of neuron importance profiling. In our experiments, we perform profiling on test models using the MMLU dataset [19], which demonstrates strong generalization across all evaluation benchmarks. In conclusion, neuron importance profiling methods can be further explored with different models and in various scenarios in future research.

#### 5.4 Advancements over Related Work

Previous research has explored methods for exploiting sparsity to accelerate inference or achieve model compression. Therefore, we compare our proposed method with several prior approaches, including: Efficient Expert Skipping (EES) [32], which dynamically skips expert computation for acceleration, and Efficient Expert Pruning (EEP) [32], which permanently removes unimportant experts for compression.

EES skips the expert computation associated with the second-highest score in the Top-2 selection if the second score is less than  $\beta$  times the first score, where  $\beta$  is determined by the median ratio of the second score to the first score across calibration samples. Since both EES and our proposed method are designed to reduce runtime FLOPs for

**Table 3.** Comparison of our method with existing work on Mixtral-8×7B-Instruct model inference. "GSM8K Acc. Variation" denotes the percentage change in accuracy relative to the original model. EEP [32] is evaluated under two configurations, pruning 2 (r = 6) and 4 (r = 4) out of 8 experts.

| Method                | Memory | Speedup | GSM8K Acc. Variation (†) |
|-----------------------|--------|---------|--------------------------|
| 2T-Drop (Partition)   | -      | 1.08×   | +0.5%                    |
| 2T-Drop (Reconstruct) | -      | 1.08×   | +1.2%                    |
| EES                   | -      | 1.05×   | -2.4%                    |
| EEP $(r = 6)$         | -24%   | 1.20×   | -8.0%                    |
| EEP(r = 6) + EES      | -24%   | 1.28×   | -14.9%                   |
| EEP $(r=4)$           | -48%   | 1.28×   | -25.9%                   |
| EEP(r = 4) + EES      | -48%   | 1.33×   | -36.4%                   |

inference acceleration, a fair comparison reveals that our method achieves superior accuracy ( $\pm 1.2\%$  vs.  $\pm 2.4\%$ ) and greater speedup ( $\pm 1.08 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times 1.05 \times$ 

Although it is not strictly fair to compare dynamic computation reduction methods with model compression techniques, such a comparison remains informative. Notably, static expert pruning tends to cause substantial accuracy loss relative to dynamic computation reduction, highlighting the importance of maintaining dynamic activation within models. This loss is also pronounced in weight pruning methods such as Wanda [52], resulting in a substantial GSM8K accuracy reduction of 50.7% under the 2:4 sparsity pattern. Our proposed 2T-Drop (reconstruct) method combines dynamic tensor-level dropping with static neuron-level weight differentiation, thereby achieving enhanced performance.

Furthermore, our comparative experiments do not incorporate load-aware thresholding, as previous works have largely overlooked distributed MoE inference with EP and have not addressed its load imbalance characteristics. It is also important to note the distinction between deployment scenarios: While edge devices prioritize model compression to accommodate limited device capacity, server-side deployments typically utilize distributed inference, where the primary focus is on maximizing accuracy and throughput rather than minimizing memory usage.

#### 6 Conclusion

While MoE architecture offers an excellent balance between accuracy and efficiency for LLMs through its inherent tensor-level sparsity, deploying MoE models remains challenging in current machine learning systems. To facilitate efficient post-training deployment of MoE models, we explore the dual-sparse activation patterns present at both the tensor and neuron levels. Specifically, we introduce expert partition to enhance tensor-level sparsity during the post-training phase, thereby improving both accuracy and efficiency. Moreover, we present the DualSparse-MoE inference system, integrating dynamic tensor-level computation drop with static

neuron-level reconstruct to accelerate inference in a trainingfree manner. Experimental results show that our methods enhance efficiency while maintaining high accuracy.

#### References

- <span id="page-11-9"></span> 2025. context\_parallel package - NVIDIA Docs - docs.nvidia.com. https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context\_parallel.html#context-parallelism-overview.
- <span id="page-11-8"></span> [2] 2025. GitHub - NVIDIA/Megatron-LM: Ongoing research training transformer models at scale — github.com. https://github.com/ NVIDIA/Megatron-LM.
- <span id="page-11-4"></span>[3] Samira Abnar, Harshay Shah, Dan Busbridge, Alaaeldin El-Nouby, Joshua M Susskind, and Vimal Thilak. [n. d.]. Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models. In Forty-second International Conference on Machine Learning.
- <span id="page-11-7"></span>[4] Abien Fred Agarap. 2018. Deep learning using rectified linear units (relu). arXiv preprint arXiv:1803.08375 (2018).
- <span id="page-11-13"></span>[5] Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. 2020. PIQA: Reasoning about Physical Commonsense in Natural Language. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 34. 7432–7439.
- <span id="page-11-2"></span>[6] Weilin Cai, Juyong Jiang, Le Qin, junweicui, Sunghun Kim, and Jiayi Huang. 2025. Shortcut-connected Expert Parallelism for Accelerating Mixture of Experts. In Forty-second International Conference on Machine Learning.
- <span id="page-11-1"></span> Weilin Cai, Juyong Jiang, Fan Wang, Jing Tang, Sunghun Kim, and Jiayi Huang. 2024. A Survey on Mixture of Experts. arXiv:2407.06204v1 https://arxiv.org/abs/2407.06204v1
- <span id="page-11-3"></span>[8] Yuanteng Chen, Yuantian Shao, Peisong Wang, and Jian Cheng. 2025. EAC-MoE: Expert-Selection Aware Compressor for Mixtureof-Experts Large Language Models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Vienna, Austria, 12942–12963.
- <span id="page-11-12"></span>[9] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019. BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). 2924–2936.
- <span id="page-11-11"></span>[10] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabhar-wal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. arXiv preprint arXiv:1803.05457 (2018).
- <span id="page-11-10"></span>[11] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168 (2021).
- <span id="page-11-0"></span>[12] Damai Dai, Chengqi Deng, Chenggang Zhao, Rx Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y Wu, et al. 2024. DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 1280–1297.
- <span id="page-11-6"></span>[13] Ruibo Fan, Xiangrui Yu, Peijie Dong, Zeyu Li, Gu Gong, Qiang Wang, Wei Wang, and Xiaowen Chu. 2025. Spinfer: Leveraging low-level sparsity for efficient large language model inference on gpus. In Proceedings of the Twentieth European Conference on Computer Systems. 243–260.
- <span id="page-11-5"></span>[14] William Fedus, Barret Zoph, and Noam Shazeer. 2022. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research* 23, 120 (2022), 1–39.

- <span id="page-12-15"></span>[15] Elias Frantar and Dan Alistarh. 2023. Sparsegpt: Massive language models can be accurately pruned in one-shot. In International conference on machine learning. PMLR, 10323–10337.
- <span id="page-12-23"></span>[16] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac'h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. 2024. The Language Model Evaluation Harness. doi:[10.5281/zenodo.12608602](https://doi.org/10.5281/zenodo.12608602)
- <span id="page-12-6"></span>[17] Jiaao He, Jidong Zhai, Tiago Antunes, Haojie Wang, Fuwen Luo, Shangfeng Shi, and Qin Li. 2022. Fastermoe: modeling and optimizing training of large-scale dynamic pre-trained models. In Proceedings of the 27th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming. 120–134.
- <span id="page-12-10"></span>[18] Xu Owen He. 2024. Mixture of a million experts. arXiv preprint arXiv:2407.04153 (2024).
- <span id="page-12-24"></span>[19] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring Massive Multitask Language Understanding. In International Conference on Learning Representations.
- <span id="page-12-18"></span>[20] Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Dehao Chen, Mia Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V Le, Yonghui Wu, et al. 2019. GPipe: Efficient Training of Giant Neural Networks Using Pipeline Parallelism. Advances in Neural Information Processing Systems 32 (2019).
- <span id="page-12-2"></span>[21] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. 2024. Mixtral of experts. arXiv preprint arXiv:2401.04088 (2024).
- <span id="page-12-7"></span>[22] Young Jin Kim, Ammar Ahmad Awan, Alexandre Muzio, Andres Felipe Cruz Salinas, Liyang Lu, Amr Hendy, Samyam Rajbhandari, Yuxiong He, and Hany Hassan Awadalla. 2021. Scalable and efficient moe training for multitask multilingual models. arXiv preprint arXiv:2109.10465 (2021).
- <span id="page-12-8"></span>[23] Yeskendir Koishekenov, Alexandre Bérard, and Vassilina Nikoulina. 2023. Memory-efficient NLLB-200: Language-specific Expert Pruning of a Massively Multilingual Machine Translation Model. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 3567–3585.
- <span id="page-12-21"></span>[24] Vijay Anand Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. 2023. Reducing activation recomputation in large transformer models. Proceedings of Machine Learning and Systems 5 (2023), 341– 353.
- <span id="page-12-26"></span>[25] Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. 2024. Tulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124 (2024).
- <span id="page-12-0"></span>[26] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. 2021. GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding. In International Conference on Learning Representations.
- <span id="page-12-12"></span>[27] Zonglin Li, Chong You, Srinadh Bhojanapalli, Daliang Li, Ankit Singh Rawat, Sashank J. Reddi, Ke Ye, Felix Chern, Felix Yu, Ruiqi Guo, and Sanjiv Kumar. 2023. The Lazy Neuron Phenomenon: On Emergence of Activation Sparsity in Transformers. In The Eleventh International Conference on Learning Representations.
- <span id="page-12-3"></span>[28] Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, et al. 2024. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434 (2024).

- <span id="page-12-5"></span>[29] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. 2024. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437 (2024).
- <span id="page-12-22"></span>[30] Dennis Liu, Zijie Yan, Xin Yao, Tong Liu, Vijay Korthikanti, Evan Wu, Shiqing Fan, Gao Deng, Hongxiao Bai, Jianbin Chang, et al. 2025. MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core. arXiv preprint arXiv:2504.14960 (2025).
- <span id="page-12-13"></span>[31] Zichang Liu, Jue Wang, Tri Dao, Tianyi Zhou, Binhang Yuan, Zhao Song, Anshumali Shrivastava, Ce Zhang, Yuandong Tian, Christopher Re, et al. 2023. Deja vu: Contextual sparsity for efficient llms at inference time. In International Conference on Machine Learning. PMLR, 22137–22176.
- <span id="page-12-9"></span>[32] Xudong Lu, Qi Liu, Yuhui Xu, Aojun Zhou, Siyuan Huang, Bo Zhang, Junchi Yan, and Hongsheng Li. 2024. Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 6159– 6172.
- <span id="page-12-11"></span>[33] Jan Ludziejewski, Jakub Krajewski, Kamil Adamczewski, Maciej Pióro, Michał Krutul, Szymon Antoniak, Kamil Ciebiera, Krystian Król, Tomasz Odrzygóźdź, Piotr Sankowski, et al. 2024. Scaling laws for fine-grained mixture of experts. In Forty-first International Conference on Machine Learning.
- <span id="page-12-25"></span>[34] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. 2381–2391.
- <span id="page-12-4"></span>[35] Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Evan Pete Walsh, Oyvind Tafjord, Nathan Lambert, et al. 2025. OLMoE: Open Mixture-of-Experts Language Models. In The Thirteenth International Conference on Learning Representations.
- <span id="page-12-19"></span>[36] Deepak Narayanan, Aaron Harlap, Amar Phanishayee, Vivek Seshadri, Nikhil R Devanur, Gregory R Ganger, Phillip B Gibbons, and Matei Zaharia. 2019. PipeDream: Generalized Pipeline Parallelism for DNN Training. In Proceedings of the 27th ACM Symposium on Operating Systems Principles. 1–15.
- <span id="page-12-20"></span>[37] Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, et al. 2021. Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis. 1–15.
- <span id="page-12-27"></span>[38] Nvidia. 2025. NVIDIA GB200 NVL72. [https://www.nvidia.com/en](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)[us/data-center/gb200-nvl72/](https://www.nvidia.com/en-us/data-center/gb200-nvl72/).
- <span id="page-12-1"></span>[39] Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, and Yuxiong He. 2022. DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale. In International Conference on Machine Learning. PMLR, 18332–18346.
- <span id="page-12-16"></span>[40] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. 2020. Zero: Memory Optimizations Toward Training Trillion Parameter Models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis. IEEE, 1–16.
- <span id="page-12-17"></span>[41] Samyam Rajbhandari, Olatunji Ruwase, Jeff Rasley, Shaden Smith, and Yuxiong He. 2021. Zero-Infinity: Breaking The GPU Memory Wall for Extreme Scale Deep Learning. In Proceedings of The International Conference for High Performance Computing, Networking, Storage and Analysis. 1–14.
- <span id="page-12-14"></span>[42] Prajit Ramachandran, Barret Zoph, and Quoc V Le. 2017. Searching for activation functions. arXiv preprint arXiv:1710.05941 (2017).

- <span id="page-13-19"></span>[43] Saeed Rashidi, Srinivas Sridharan, Sudarshan Srinivasan, and Tushar Krishna. 2020. ASTRA-Sim: Enabling SW/HW Co-Design Exploration for Distributed DL Training Platforms. In 2020 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS). IEEE, 81–92.
- <span id="page-13-10"></span>[44] Jie Ren, Samyam Rajbhandari, Reza Yazdani Aminabadi, Olatunji Ruwase, Shuangyan Yang, Minjia Zhang, Dong Li, and Yuxiong He. 2021. Zero-Offload: Democratizing Billion-Scale Model Training. In 2021 USENIX Annual Technical Conference (USENIX ATC 21). 551–564.
- <span id="page-13-16"></span>[45] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021. WinoGrande: An Adversarial Winograd Schema Challenge at Scale. Commun. ACM 64, 9 (2021), 99–106.
- <span id="page-13-6"></span>[46] Noam Shazeer. 2020. Glu variants improve transformer. arXiv preprint arXiv:2002.05202 (2020).
- <span id="page-13-0"></span>[47] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc V. Le, Geoffrey E. Hinton, and Jeff Dean. 2017. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. In International Conference on Learning Representations.
- <span id="page-13-11"></span>[48] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. 2019. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism. arXiv preprint arXiv:1909.08053 (2019).
- <span id="page-13-4"></span>[49] Siddharth Singh, Olatunji Ruwase, Ammar Ahmad Awan, Samyam Rajbhandari, Yuxiong He, and Abhinav Bhatele. 2023. A Hybrid Tensor-Expert-Data Parallelism Approach to Optimize Mixture-of-Experts Training. In Proceedings of the 37th International Conference on Supercomputing. 203–214.
- <span id="page-13-12"></span>[50] Shaden Smith, Mostofa Patwary, Brandon Norick, Patrick LeGresley, Samyam Rajbhandari, Jared Casper, Zhun Liu, Shrimai Prabhumoye, George Zerveas, Vijay Korthikanti, et al. 2022. Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model. arXiv preprint arXiv:2201.11990 (2022).
- <span id="page-13-8"></span>[51] Yixin Song, Zeyu Mi, Haotong Xie, and Haibo Chen. 2024. Powerinfer: Fast large language model serving with a consumer-grade gpu. In Proceedings of the ACM SIGOPS 30th Symposium on Operating Systems Principles. 590–606.
- <span id="page-13-7"></span>[52] Mingjie Sun, Zhuang Liu, Anna Bair, and J Zico Kolter. 2024. A Simple and Effective Pruning Approach for Large Language Models. In The Twelfth International Conference on Learning Representations.
- <span id="page-13-1"></span>[53] GLM-4.5 Team. 2025. GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models. arXiv[:2508.06471](https://arxiv.org/abs/2508.06471)
- <span id="page-13-2"></span>[54] Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, et al. 2025. Kimi K2: Open Agentic Intelligence. arXiv preprint arXiv:2507.20534 (2025).
- <span id="page-13-15"></span>[55] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. 2018. GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding. In International Conference on Learning Representations.
- <span id="page-13-3"></span>[56] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025).
- <span id="page-13-14"></span>[57] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. HellaSwag: Can a Machine Really Finish Your Sentence?. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics.
- <span id="page-13-5"></span>[58] Zhengyan Zhang, Yixin Song, Guanghui Yu, Xu Han, Yankai Lin, Chaojun Xiao, Chenyang Song, Zhiyuan Liu, Zeyu Mi, and Maosong Sun. 2024. ReLU<sup>2</sup> Wins: Discovering Efficient Activation Functions for Sparse LLMs. arXiv preprint arXiv:2402.03804 (2024).
- <span id="page-13-18"></span>[59] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Livia Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. 2024. Sglang: Efficient execution of structured language model programs. Advances in neural information processing

- systems 37 (2024), 62557–62583.
- <span id="page-13-9"></span>[60] Ningxin Zheng, Huiqiang Jiang, Quanlu Zhang, Zhenhua Han, Lingxiao Ma, Yuqing Yang, Fan Yang, Chengruidong Zhang, Lili Qiu, Mao Yang, et al. 2023. Pit: Optimization of dynamic sparse deep learning models via permutation invariant transformation. In Proceedings of the 29th Symposium on Operating Systems Principles. 331–347.
- <span id="page-13-13"></span>[61] Ruidong Zhu, Ziheng Jiang, Chao Jin, Peng Wu, Cesar A Stuardo, Dongyang Wang, Xinlei Zhang, Huaping Zhou, Haoran Wei, Yang Cheng, et al. 2025. MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism. arXiv preprint arXiv:2504.02263 (2025).
- <span id="page-13-17"></span>[62] Pengfei Zuo, Huimin Lin, Junbo Deng, Nan Zou, Xingkun Yang, Yingyu Diao, Weifeng Gao, Ke Xu, Zhangyu Chen, Shirui Lu, et al. 2025. Serving Large Language Models on Huawei CloudMatrix384. arXiv preprint arXiv:2506.12708 (2025).