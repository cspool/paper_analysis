# Expert-Token Resonance MoE: Bidirectional Routing with Efficiency Affinity-Driven Active Selection

Jing Li\*, Zhijie Sun\*, Dachao Lin\*, Xuan He, Binfan Zheng, Yi Lin, Rongqian Zhao, Xin Chen

<sup>1</sup>Huawei Technologies Co., Ltd

#### Abstract

Mixture-of-Experts (MoE) architectures enable efficient scaling of large language models by activating only a subset of parameters per input. However, existing MoE models suffer from two critical limitations: (1) inefficient token-toexpert routing that causes excessive communication overhead, and (2) expert homogenization that leads to redundant computations. Current approaches address these challenges separately, failing to achieve simultaneous improvements in both training efficiency and model performance. We present Expert-Token Resonance (ETR), a theoreticallygrounded bidirectional routing mechanism that fundamentally reimagines expert-token interactions in MoE architectures. Our key insight is that optimal routing requires adaptive coordination between token-choice routing (TCR) during early training phases and expert-choice routing (ECR) in later stages. We prove that this dynamic approach maximizes training success rate—the probability of correct token-expert assignments—while reducing the expert capacity lower bound by up to 40%. ETR incorporates three technical innovations: (1) an affinity-based routing architecture using Grouped Average Pooling (GrAP) that reduces computational complexity from O(d²) to O(d²/D) while maintaining orthogonality to prevent expert homogenization; (2) a bidirectional selection mechanism that enables both tokens and experts to actively participate in the routing process based on cosine similarity scores; and (3) an adaptive capacity strategy that dynamically adjusts expert bounds based on training progress, eliminating communication bubbles in All-to-All operations. Extensive experiments on Ascend NPU clusters demonstrate that ETR achieves 5.4%-46.6% improvements in end-to-end training efficiency compared to baseline MoE implementations, with 9.7%-14.5% performance gains across GDAD, GPQA, HumanEval, and TeleQnA benchmarks. These results establish ETR as the efficient MoE routing approach to deliver substantial improvements in both computational efficiency and model quality, enabling practical deployment of larger sparse models previously constrained by communication bottlenecks.

# Introduction

Large language models (LLMs) have demonstrated remarkable capabilities in understanding complex semantic relationships and generating coherent text (Zhao et al. 2023). However, as model parameters scale to billions or even trillions, the computational and communication costs grow prohibitively (Jiang et al. 2024b). The Mixture-of-Experts (MoE) architecture offers a promising solution by activating only a subset of model parameters for each input, enabling efficient model scaling without proportional increases in computational demands (Lepikhin et al. 2020; Fedus, Zoph, and Shazeer 2022). Recent MoE-based LLMs, such as DeepSeek-V3 (Liu et al. 2024a) and Mixtral (Jiang et al. 2024a), have achieved state-of-the-art performance across various benchmarks.

Despite these successes, MoE models face two fundamental challenges that limit their practical deployment and effectiveness. First, the routing efficiency problem: current routing mechanisms suffer from suboptimal token-toexpert assignments, leading to significant computational waste through excessive padding in All-to-All communications and underutilized expert capacity. Second, the expert homogenization problem: existing routing strategies often fail to maintain expert specialization, resulting in redundant computations where multiple experts learn similar representations. These challenges are particularly pronounced during early training phases when routing networks lack sophisticated assignment capabilities.

Previous approaches have attempted to address these issues separately. Load balancing techniques (Zhou et al. 2022b; Dai et al. 2022) focus on distributing tokens evenly across experts but often sacrifice routing quality. Expert specialization methods (Xie et al. 2024; Li et al. 2024) aim to maintain diversity but typically incur additional computational overhead. Critically, no existing work has demonstrated how to simultaneously improve both training efficiency and model performance through a unified routing mechanism.

This paper introduces Expert-Token Resonance (ETR), a novel bidirectional routing strategy that fundamentally rethinks how tokens and experts interact in MoE architectures. Our key insight is that optimal routing requires not only tokens selecting appropriate experts (token-choice routing, TCR) but also experts actively selecting their most relevant tokens (expert-choice routing, ECR). We theoretically prove that TCR achieves higher training success rates during early training when routing networks need refinement, while ECR excels in later stages when expert specialization becomes critical. By adaptively combining both paradigms,

<sup>\*</sup>These authors contributed equally.

ETR achieves what previous methods could not: simultaneous improvements in training efficiency and model performance.

Specifically, our contributions are threefold:

- 1. A theoretically-grounded bidirectional routing mechanism that dynamically balances token-choice and expert-choice routing based on training progress. We prove that this approach maximizes the training success rate—the probability of correct token-expert assignments—throughout the entire training process while reducing the required expert capacity lower bound by up to 40% compared to conventional methods.
- 2. An efficient affinity-based routing architecture leveraging Grouped Average Pooling (GrAP) layers that reduces computational complexity from  $O(d^2)$  to  $O(d^2/D)$  compared to traditional MLP routers, where d is the hidden dimension and D is the grouping factor. The orthogonality properties of GrAP naturally prevent expert homogenization while enabling precise affinity computations through cosine similarity.
- 3. Comprehensive empirical validation on Ascend NPU clusters demonstrating that ETR improves end-to-end training efficiency by 5.4%-46.6% compared to baseline MoE implementations and 2.9%-13.3% compared to state-of-the-art LocMoE, while simultaneously enhancing downstream task performance by 9.7%-14.5% and 1.7%-4.1%, respectively.

The significance of this work extends beyond incremental improvements. By addressing the fundamental trade-off between routing quality and computational efficiency, ETR enables practical deployment of larger MoE models that were previously constrained by communication bottlenecks. Our theoretical analysis provides new insights into the dynamics of expert-token interactions, opening avenues for future research in sparse model architectures.

#### **Related Work**

#### **Routing Mechanisms**

Routing design fundamentally determines MoE performance. Token-choice routing (TCR) (Shazeer et al. 2017a) allows tokens to select experts but suffers from load imbalance. Expert-choice routing (ECR) (Zhou et al. 2022a) ensures balanced loads by having experts select tokens, risking the loss of critical tokens. Recent dynamic routing strategies (Huang et al. 2024) adapt expert allocation to input complexity but remain unidirectional—either tokens choose experts or vice versa. This fundamental limitation prevents simultaneous optimization of routing quality and load balance.

#### **Expert Specialization**

Expert homogenization critically limits MoE performance. Prior work addressed this through fine-grained expert segmentation (Dai et al. 2024), competitive mechanisms (Pham et al. 2024), and theoretical analyses linking data clustering to specialization (Chen et al. 2022). However, these approaches require auxiliary losses or complex training procedures.

![](_page_1_Picture_11.jpeg)

Figure 1: The illustrative diagram of GrAP.

#### **System Optimization**

System-level MoE optimizations focus on hardware utilization through block-sparse operations (Gale et al. 2022) and communication reduction via expert sharding (Balmau et al. 2025) or sequence migration (Chen et al. 2024). While these approaches improve specific bottlenecks, they overlook communication bubbles in All-to-All operations—a critical inefficiency.

#### **Positioning of This Work**

Recent surveys (Cai et al. 2024) reveal that existing MoE methods typically prioritize either routing quality or computational efficiency. Our ETR framework addresses previously overlooked communication inefficiencies and training bottleneck, suggesting potential for more efficient MoE deployment at scale.

#### Method

In this section, we present the efficient routing mechanism, and our adaptive bidirectional selection mechanism is detailed. Then, for traditional drop-and-pad strategies, a dynamic token distribution analysis module that optimizes the lower bounds of expert capacity are displayed. Moreover, we also describe the loss for expert load balancing.

#### Model Architecture.

**Backbone.** The MoE architecture, based on the Transformer framework, efficiently scales up model size with low computational overhead, benefiting from two primary structures: a sparse gating network for routing tokens and expert networks for processing specific token categories.

We consider the supervised classification for brevity where the training samples are  $\{(\boldsymbol{x}^{(i)}, y_i)\}_{i=1}^N \sim \mathcal{D}$ . Each training sample  $\boldsymbol{x}^\top = (\boldsymbol{x}_1^\top, \dots, \boldsymbol{x}_s^\top) \in \mathcal{R}^{sd}$  has s tokens with token feature  $\boldsymbol{x}_i \in \mathcal{R}^d, \forall i \in [s]$ , and label  $y \in \mathcal{N}^+$ . The objective is to learn the map of  $\boldsymbol{x}$  to the corresponding y. The general MoE structure are formulated as

$$MoE(\boldsymbol{x}) = \sum_{t=1}^{s} \sum_{i=1}^{n} G_i(\boldsymbol{x}_t) \cdot E_i(\boldsymbol{x}_t), \tag{1}$$

where n is the number of experts,  $G(\boldsymbol{x}_t) \colon \mathcal{R}^d \to \mathcal{R}^n$  is the gating weight vector of experts which maps the tokens of  $\boldsymbol{x}_t$  into the coresponding experts with weights, e.g.,  $G_i(\boldsymbol{x}) = \operatorname{Softmax}(\boldsymbol{W}\boldsymbol{x} + \boldsymbol{\epsilon})$  where the softmax is applied to each row, and  $E_i(\boldsymbol{x}_t) \colon \mathcal{R}^d \to \mathcal{R}$  is the i-th expert network, see (Liu et al. 2024b) for current different router methods. Generally,  $n \ll s$ , which saves much computation compared to the dense structure.

Cost-Efficient Sparse Expert-Token Affinity.  $W_{\rm aff}$  denotes the expert-token affinity matrix. After processing through the GrAP routing layer, tokens generate a diagonal sparse matrix as shown. Compared to the dense matrix produced by traditional routing layers, this reduces the parameter count to 1/D of the original, significantly decreasing the computational overhead of the expert routing layer.

As shown in Figure 1, GrAP performs average pooling with group segmented by the number of experts. With GrAP as the layer of feature extraction, the formulation of  $\boldsymbol{W}_{\rm aff}$  is as followed:

$$\mathbf{W}_{\text{aff}} = \begin{pmatrix} \mathbf{w}_1 & 0 & \cdots & 0 \\ 0 & \mathbf{w}_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \mathbf{w}_n \end{pmatrix}$$
 (2)

$$\mathbf{w}_i = \frac{n}{d} \cdot \mathbf{1} \left\{ \frac{i \cdot d}{n} \le j < (i+1) \frac{d}{n} \right\} \quad 0 \le j < d \quad (3)$$

The expert-token affinity matrix is employed as the gating weight to calculate the affinity score between each expert and token. We define the affinity score of t-th token and i-th expert as the cosine similarity between vectors  $x_t$  and  $w_i$ :

$$\delta_{ti} = \cos(\boldsymbol{x}_t, \boldsymbol{w}_i) := \boldsymbol{x}_t^{\top} \boldsymbol{w}_i / (\|\boldsymbol{x}_t\| \cdot \|\boldsymbol{w}_i\|) \tag{4}$$

The affinity score intuitively reflect how closely the two inputs are associated. From a perspective of semantic, the affinity scores derived from affinity metrics consisting of orthogonal vectors represent the degree of association between each token and various experts, as shown in Figure 2. Therefore, we leverage the affinity score as the principle of our affinity-driven active selection routing mechanism.

![](_page_2_Figure_8.jpeg)

Figure 2: The illustration of affinity score.

**Routing Strategy.** We consider our affinity-driven active selection routing as a hybrid of TCR (Clark et al. 2022; Zhou et al. 2022b) and ECR. As the name suggested, TCR lets

![](_page_2_Figure_11.jpeg)

Figure 3: The architecture of the gate network along with the hybrid TCR + ECR router.

each token choose its *top-scored* experts, and ECR lets each expert choose its *top-scored* tokens. Specifically, we use the result of the expert-token affinity metrics as the affinity score between tokens and experts. In conventional TCR routing strategy, the tokens are simply route to their Top-1 expert. In our hybird **TCR+ECR** routing strategy, experts also select tokens for processing from assigned tokens according to affinity scores:

$$\left(\tilde{E}_{t1}, \dots, \tilde{E}_{t\ell}\right) = \text{Top-}\ell\left(\left\{\delta_{t1}, \dots, \delta_{tn}\right\}\right),$$

$$\tilde{I}_{tk} \in [n], \forall t \in [s], k \in [\ell].$$
(5)

and then the expert to choose its Top- $\ell$  tokens where  $\ell$  is determined by a threshold of the sum of affinity scores:

$$(I_{1i}, \dots, I_{Ci}) = \text{Bottom-}C\left(\left\{t \in [s] : \exists j \in [\ell], \tilde{I}_{tj} = i\right\}\right),$$

$$I_{ki} \in [s] \cup \text{None}, \forall i \in [n], k \in [C].$$
(6

Such bidirectional selection mechanism motivates each expert to receive a certain number of tokens with the highest affinity score to itself, thereby achieving a resonance effect. The resonance effect can help mitigate the homogenization in MoE.

**Locality Loss.** Feed-forward network (FFN) layers are commonly employed in expert networks, allowing each expert to learn independently as a separate neural network, thus preventing interference between samples. This mechanism leads to a severe load imbalance, as experts frequently selected in the early stages are more likely to be chosen in later stages. To mitigate this skewness in token allocation, the auxiliary loss (Shazeer et al. 2017b) has been proposed. Building upon the auxiliary loss, our work introduces a loss bias term based on data locality, represented as  $L_{\rm loc} = \mu {\rm KL}(D_{\rm c}||D_{\rm l}) = -\mu \int D_{\rm c}(x) \ln[\frac{D_{\rm l}(x)}{D_{\rm c}(x)}] {\rm d}x$ , i.e., the

Kullback-Leibler (KL) divergence of the current distribution  $D_{\rm c}(x)$  and the fully localized distribution  $D_{\rm l}(x)$ . This loss term serves as a soft constraint, encouraging tokens to be sent to experts residing on the same node, thereby mitigating the substantial overhead incurred by partial inter-node communication.

#### **Training Strategy**

Token Distribution Dynamics under Expert Routing. The proposed hybrid TCR+ECR bidirectional routing mechanism operates through a two-stage process: tokens are initially assigned to experts based on the similarity between their feature fragments and the corresponding gating weights, followed by each expert selecting the Top- $\ell$  most relevant tokens through a scoring mechanism. This hybrid approach dynamically determines the number of tokens processed by each expert via an adaptive threshold, thereby ensuring the retention of class-discriminative tokens while optimizing computational efficiency.

**Dynamic Lower Bound Module for Expert Capacity in ETR** To explain the motivation of our method, we show some theoretical insights in this section.

**Assumption 1** (data assumption). Each input  $x \in \mathbb{R}^{sd}$  with s tokens is comprised of one class-discriminative pattern  $o_1, \ldots, o_n \in \mathbb{R}^d$ , with each decides the label in [n], and s-1 class-irrelevant patterns  $r \sim \mathcal{N}$  for certain distribution  $\mathcal{N}$ . For example,  $x = (r_1, r_2, o_1, r_3, \ldots, r_{s-1})$  has label 1, where  $r_i \stackrel{i.i.d.}{\sim} \mathcal{N}, \forall i \in [s-1]$ .

Based on Assumption 1, Chowdhury et al. (2023) demonstrated that the training of MoE go through two phases:

**Phase 1: Router training** This process ensures that each expert only receives the class-discriminative tokens related to the specific class.

**Phase 2: Expert training** This process is designed to establish each expert's ability to handle and solve problems.

To quantitatively measure the difference between TCR and ECR, we define **training success rate** of input motivated by the training process of MoE.

**Definition 2** (training success rate). We say the input  $x \in \mathbb{R}^{sd}$  with s tokens succeed in training if the class-discriminative pattern in x, e.g.,  $o_i$  is correctly dispatched to i-th expert. We further define **training success rate** as the probability that the input succeed in training.

Furthermore, to show the quantitative comparison of TCR and ECR in training success rate, we need following asssumptions and notations of token patterns.

**Assumption 3** (class-discriminative). We assume the location and feature of class-discriminative pattern is uniformly distribute in [s] and [n], i.e.,

$$i \sim \text{Unif}([s]), \boldsymbol{x}_i \sim \text{Unif}(\{\boldsymbol{o}_1, \dots, \boldsymbol{o}_n\}).$$
 (7)

We also assume that  $\forall i \in [n], \mathbf{o}_i$  should be sent to the i-th expert, and define the true positive probability in token choice setting is no worse than the uniform dispatch as below

$$\mathcal{P}(\delta_{o_i,i} \ge \delta_{x_i,i}, \forall j \in [s]) = p_i \ge 1/n, \forall i \in [n].$$
 (8)

**Assumption 4** (class-irrelevant). The distribution of class-irrelevant patterns is isotropy, i.e.,

$$\mathcal{P}(\boldsymbol{r} \sim \mathcal{N}, \delta_{\boldsymbol{r},i} \ge \delta_{\boldsymbol{x}_i,i}, \forall j \in [s]) = 1/n, \forall i \in [n]. \tag{9}$$

And we define the false positive probability in expert choice setting as

$$\mathcal{P}(\mathbf{r} \sim \mathcal{N}, \delta_{\mathbf{r},i} \ge \delta_{\mathbf{o}_{i},i}) = q_{i}, \forall i \in [n], \tag{10}$$

which measures the possibility that expert i chooses the wrong token r instead of the correct token  $o_i$ .

**Theorem 5.** Under Assumptions 3 and 4, the training success rate of TCR in each sample x is

$$\mathcal{P}(TCR \ succeed) = \Theta\left(C \sum_{i=1}^{n} p_i / s\right), \tag{11}$$

and the training success rate of ECR is  $\forall i \in [n]$ ,

$$\mathcal{P}(ECR \ succeed) \begin{cases} \leq \frac{1}{n} \sum_{i=1}^{n} e^{-\frac{(s-1)q_i}{8}}, & C \leq (s-1)q_i/2, \\ \geq 1 - e^{-3C/16}, & C \geq 2sq_i. \end{cases}$$
(12)

**Corollary 6.** In practice, For constant number of experts (Jiang et al. 2024a), i.e.,  $n = \Theta(1)$ , and C < s to save computation cost. We have the following lower bound for capacity C to ensure high training success rate:

- 1. Suppose  $q_i = \Theta(1)$ . Then TCR is much better than ECR, and we only need  $C = \Theta(s)$ .
- 2. Suppose  $\forall i \in [n], sq_i \leq C^*$  for some  $C^* > 0$ . Then ECR is much better than TCR, and we only need  $C \geq 2C^*$ .

**Remark 7.** Under the assumptions of orthogonal gating weights and uniform data distribution, we establish that the lower bound of expert capacity is given by  $C_{\min} = \frac{1}{n} \exp\{d\delta_{\max}^2/(2-\delta_{\max}^2)\}$ , where the capacity is intrinsically linked to the angular relationship between gating weights and tokens.

Building on Theorem 5 and the evolution of feature distributions during training, we demonstrate the optimality of transitioning from TCR to ECR. In early training stages, when class-irrelevant tokens exhibit near-isotropic distribution with  $q_i = \Theta(1)$ , TCR achieves superior training success rates of C/s compared to ECR's exponentially decaying rate of  $e^{-s}$ , necessitating a larger expert capacity of  $C = \Theta(s)$ . As training progresses and experts develop discriminative capabilities, the distribution shifts such that  $q_i \ll 1$  or  $sq_i \leq C^*$  for some constant  $C^* > 0$ , at which point ECR approaches unit success rate while TCR remains bounded by C/s, enabling efficient operation with reduced capacity  $C = \Theta(1)$  when  $C \geq 2C^*$ .

#### **Communication Optimization**

To mitigate the serial execution bottlenecks inherent in LLM training, we implement Communication Over Computation (CoC) optimization, which transforms sequentially dependent matrix multiplication and collective communication operations in parallel linear layers into unified, fine-grained kernels. By leveraging MTE's remote memory access capabilities, CoC fuses computation and communication primitives, enabling pipeline-parallel execution that overlaps previously serialized operations.

# Experiments

### Experimental Setup

We implement our approach using Mixtral 8×7B, a 46.7Bparameter model with Group Query Attention (GQA) and 32 sparse MoE blocks. Each block contains 8 experts, with tokens routed to their top-2 selections. To accommodate long-context applications, we extend the sequence length to 32,768 tokens. Our experiments span three cluster scales with tailored parallelization strategies: 32 NPUs (TP=4, PP=4, DP=2, EP=2), 64 NPUs (TP=8, PP=4, DP=2, EP=2), and 256 NPUs (TP=8, PP=8, DP=4, EP=2), maintaining a global batch size of 128 throughout. Additional experimental details are provided in the Appendix.

### Efficiency Promotion and Memory Footprint Reduction

We consistently employ Top-1 routing to align implementation with our theoretical framework. The baseline model uses constrained expert capacity rather than groupedGEMM, preventing token dropout with a capacity factor of 1.1. LocMoE incorporates distributional uniformity and estimates expert capacity via a theoreticallyderived lower bound formula from the initial batch, maintaining this value throughout training. Our approach ("Loc-MoE+" in figures) constrains score sum ranges, processes hidden states, and dynamically calculates expert capacity. The subsequent analysis examines training efficiency, convergence, and memory utilization across multiple Ascend cluster configurations.

![](_page_4_Figure_5.jpeg)

Figure 4: The time consumption during training iterations with different schemes and cluster sizes.

Figure 4 presents the training overhead analysis across the initial 1000 training iterations. To ensure measurement stability and exclude initialization artifacts, we commence time profiling from the fifth iteration onward. The baseline model exhibits consistent temporal performance throughout the evaluation period. In contrast, LocMoE demonstrates a marginal decrease in execution time as training progresses, with this trend being particularly pronounced in the 32N and 64N configurations. This observation corroborates our hypothesis that locality-aware optimization achieves optimal efficiency when the number of experts meets or exceeds the number of computational nodes. Our proposed method introduces a modest computational overhead relative to Loc-MoE, attributable to the token rearrangement mechanism. However, this overhead diminishes progressively as token representations converge during training. Specifically, the convergence of token features leads to a reduction in the number of tokens requiring rearrangement, resulting in stabilized computational costs in later training phases. Empirically, our approach achieves a reduction in total training time ranging from 2.9% to 13.3% compared to LocMoE, and from 5.4% to 46.6% relative to the baseline configuration.

Figure 5 depicts the temporal distribution of computational phases across training. We sample performance metrics at ten equidistant intervals throughout training, capturing computation, communication, overlap, and idle time. While profiling introduces minor overhead, this methodology provides robust insights into system behavior.

Both LocMoE and our proposed method demonstrate reduced latency across all components, with computational overhead exhibiting more substantial improvements than communication costs. This efficiency gain follows a clear pattern: as cluster size expands, the computationcommunication overlap ratio decreases, accompanied by diminishing returns in computational speedup. This trend reflects the inherent scalability challenges in distributed MoE architectures.

![](_page_4_Figure_11.jpeg)

Figure 5: The average composition of computation, communication, overlap, and idle with different schemes and cluster sizes.

Figure 6 validates that these efficiency improvements preserve model quality. All methods exhibit comparable convergence trajectories, confirming that our optimization strategy maintains training stability while delivering performance gains. The perplexity curves demonstrate that accelerated training does not compromise the fundamental learning dynamics of the model.

![](_page_4_Figure_14.jpeg)

Figure 6: The perplexity during training iterations with different schemes.

Figure 7 presents the operator-level computational profiling across different hardware components. The system leverages AI CORE for matrix multiplication and convolution operations, AI VECTOR CORE for parallelized vector computations, MIX AIC for heterogeneous operator fusion, and AI CPU for specialized AI instruction execution. Our token selection strategy yields substantial performance gains: the FFN MatMul operator achieves a 17× speedup compared to the baseline and 2.6× improvement over LocMoE. This optimization translates to a 2.8× reduction in cumulative Mat-Mul execution time and a 2.6× decrease in Cube computational load. The rearrangement-associated operators (TopK and IndexPutV2) exhibit marginal overhead increases, representing an acceptable trade-off for the significant computational savings achieved through selective token processing.

![](_page_5_Figure_1.jpeg)

Figure 7: The distribution of time consumption for operators.

Figure 8 analyzes memory consumption patterns during stable training phases, based on 100,000 memory profiling samples per device. Our approach demonstrates substantial memory efficiency gains, achieving 4.57-16.27% reduction compared to the baseline and 2.86-10.5% reduction compared to LocMoE. The memory optimization exhibits scaledependent characteristics: larger clusters show reduced computational overhead proportions and correspondingly narrower memory usage differentials. Furthermore, our method effectively eliminates transient memory spikes and reduces short-term memory fluctuations, contributing to more predictable and stable resource utilization throughout training.

![](_page_5_Figure_4.jpeg)

Figure 8: print recorded in one acquisition cycle with different schemes and cluster sizes.

### Expert Homogenization and Load Distribution Analysis

The Calinski-Harabasz (CH) index (Lima and Cruz 2020) measurements reveal that bidirectional affinity selection significantly enhances token clustering quality in MoE architectures:

$$CH = \frac{\sum_{i=1}^{k} n_i ||c_i - c||^2}{\sum_{i=1}^{k} \sum_{x \in C_i} ||x - c_i||^2}$$
(13)

While baseline and single-mechanism approaches achieve comparable improvements, the integrated LocMoE+ method demonstrates superior performance, as shown in Figure 9. Combining token-to-expert and expert-to-token selection mechanisms creates synergistic effects that accelerate natural clustering tendencies during training. Our bidirectional approach establishes a positive feedback loop between token routing and expert affinity, fundamentally enhancing expert utilization efficiency through improved feature organization and specialization.

![](_page_5_Figure_11.jpeg)

Figure 9: The Calinski-Harabasz index across training steps.

Figure 10 reveals fundamental differences in how various loss functions affect token distribution across experts. The baseline approach suffers from severe load imbalance, with certain experts becoming overloaded while others remain idle. The auxiliary loss method provides marginal improvements through regularization, yet distribution remains significantly skewed. The locality loss demonstrates transformative effects by incorporating architectural topology into the optimization objective, achieving balanced token allocation across all experts through KL divergence constraints that simultaneously minimize inter-node communication and prevent expert collapse.

![](_page_5_Figure_14.jpeg)

Figure 10: The distribution of tokens assigned to experts with different loss function.

Figure 11 presents the cumulative distribution function (CDF) and empirical cumulative distribution function (ECDF) analysis across these routing methods throughout the training progression. The locality loss approach presents distinctly optimal characteristics across both CDF and ECDF measurements, maintaining consistently high performance levels throughout the training process with remarkable stability during later training phases. The sustained performance across different probability and load thresholds indicates that incorporating expert-token affinity into the routing objective creates robust optimization dynamics that preserve both routing quality and load distribution efficiency. These findings underscore the effectiveness of ETR in addressing the inherent challenges of expert-token assignment optimization, providing a principled foundation for scalable sparse model.

![](_page_6_Figure_1.jpeg)

Figure 11: The CDF and ECDF of different schemes.

# The Performance of Downstream Tasks

The GDAD benchmark, comprising three distinct evaluation tasks, serves as the primary assessment framework for domain task capabilities, while TeleQnA provides specialized validation for telecommunications knowledge applications. Our integrated LocMoE+ approach consistently outperforms both baseline and single-mechanism variants across all evaluation metrics (see Table 1), achieving notable gains through bidirectional selection mechanisms. The consistent improvements observed across both general domain tasks and specialized telecommunications applications confirm that the synergistic combination of bidirectional selection mechanisms creates substantial performance advantages while validating the robustness and generalizability of our proposed architecture modifications.

Table 1: Domain performance promotion obtained by our approach on different datasets.

|              |      | GDAD                             |  |           |      |
|--------------|------|----------------------------------|--|-----------|------|
|              |      | GDAD-1 GDAD-2 GDAD-3 Avg TeleQnA |  |           |      |
| Baseline     | 47.8 | 43.0                             |  | 65.4 52.8 | 62.1 |
| LocMoE (TCR) | 55.5 | 47.6                             |  | 71.1 59.0 | 67.6 |
| LocMoE (ECR) | 45.8 | 45.6                             |  | 62.8 56.3 | 61.8 |
| LocMoE+      | 57.4 | 49.9                             |  | 74.5 61.5 | 68.8 |

Table 2 presents general performance evaluation results across three widely-recognized benchmarks—MMLU (Hendrycks et al. 2021) for comprehensive knowledge assessment, GPQA (Rein et al. 2023) for advanced reasoning capabilities, and HumanEval (Chen et al. 2021) for code generation proficiency—revealing distinct performance characteristics of our proposed methods. The results demonstrate that our bidirectional LocMoE+ approach achieves superior performance in reasoning and coding tasks while maintaining competitive general knowledge capabilities, with individual constraint routing mechanisms exhibiting complementary strengths across different evaluation dimensions. While the baseline maintains slight advantage in MMLU, the integrated LocMoE+ approach demonstrates that bidirectional selection mechanisms create meaningful improvements in task-specific capabilities without substantial degradation in general knowledge retention, suggesting that our architectural modifications enhance model specialization for complex reasoning and generation tasks while preserving foundational knowledge capabilities.

Table 2: General performance comparison of different MoE methods

| Method       | MMLU | GPQA | HumanEval |
|--------------|------|------|-----------|
| Baseline     | 71.8 | 29.2 | 40.2      |
| LocMoE (TCR) | 68.4 | 30.3 | 52.8      |
| LocMoE (ECR) | 45.8 | 32.5 | 57.6      |
| LocMoE+      | 70.4 | 33.5 | 67.8      |

To enhance conversational capabilities and downstream task adaptability, we conducted supervised fine-tuning on the pre-trained models. As shown in Figure 12, our approach demonstrates substantial improvements across multiple evaluation dimensions within the General and Domainspecific Assessment Dataset (GDAD). The method achieves an average improvement of approximately 20.1% across 16 sub-capabilities of Domain Task Capability compared to the baseline, with particularly notable gains in rewriting and summary capabilities. In the Domain Competency Exam assessments, our approach shows an average improvement of 16% relative to the baseline, with IP Training in digital communications demonstrating the most significant advancement. Among the 18 sub-capabilities of General Ability, the method exhibits an improvement of about 13.9% relative to the baseline, with planning capabilities showing the highest enhancement at 26.8%.

![](_page_6_Figure_12.jpeg)

Figure 12: The performance on three categories of GDAD.

# Conclusion

In this paper, we propose ETR, a fundamentally new approach to MoE routing that solves the longstanding trade-off between computational efficiency and model performance through theoretically-grounded bidirectional selection mechanisms. By dynamically coordinating tokenchoice and expert-choice routing based on training progress, ETR achieves simultaneous improvements in both training efficiency and downstream task quality. The substantial performance gains demonstrated across diverse benchmarks, combined with significant reductions in computational overhead, establish ETR as a critical advancement for practical deployment of large-scale sparse models. Our theoretical contributions provide new insights into expert-token dynamics that extend beyond incremental optimizations, opening pathways for next-generation MoE architectures.

# References

- Balmau, O.; Kermarrec, A.-M.; Pires, R.; Santo, A. L. E.; de Vos, M.; and Vujasinovic, M. 2025. Accelerating MoE Model Inference with Expert Sharding. arXiv:2503.08467.
- Cai, W.; Jiang, J.; Wang, F.; Tang, J.; Kim, S.; and Huang, J. 2024. A Survey on Mixture of Experts in Large Language Models. arXiv:2407.06204.
- Chen, F.; Li, P.; Hong, Z.; Su, Z.; and Guo, S. 2024. Communication-Efficient Sparsely-Activated Model Training via Sequence Migration and Token Condensation. arXiv:2411.15419.
- Chen, M.; Tworek, J.; Jun, H.; Yuan, Q.; de Oliveira Pinto, H. P.; Kaplan, J.; Edwards, H.; Burda, Y.; Joseph, N.; Brockman, G.; Ray, A.; Puri, R.; Krueger, G.; Petrov, M.; Khlaaf, H.; Sastry, G.; Mishkin, P.; Chan, B.; Gray, S.; Ryder, N.; Pavlov, M.; Power, A.; Kaiser, L.; Bavarian, M.; Winter, C.; Tillet, P.; Such, F. P.; Cummings, D.; Plappert, M.; Chantzis, F.; Barnes, E.; Herbert-Voss, A.; Guss, W. H.; Nichol, A.; Paino, A.; Tezak, N.; Tang, J.; Babuschkin, I.; Balaji, S.; Jain, S.; Saunders, W.; Hesse, C.; Carr, A. N.; Leike, J.; Achiam, J.; Misra, V.; Morikawa, E.; Radford, A.; Knight, M.; Brundage, M.; Murati, M.; Mayer, K.; Welinder, P.; Mc-Grew, B.; Amodei, D.; McCandlish, S.; Sutskever, I.; and Zaremba, W. 2021. Evaluating Large Language Models Trained on Code. arXiv:2107.03374.
- Chen, Z.; Deng, Y.; Wu, Y.; Gu, Q.; and Li, Y. 2022. Towards Understanding Mixture of Experts in Deep Learning. arXiv:2208.02813.
- Chowdhury, M. N. R.; Zhang, S.; Wang, M.; Liu, S.; and Chen, P.-Y. 2023. Patch-level routing in mixture-of-experts is provably sample-efficient for convolutional neural networks. In *International Conference on Machine Learning*, 6074–6114. PMLR.
- Clark, A.; de Las Casas, D.; Guy, A.; Mensch, A.; Paganini, M.; Hoffmann, J.; Damoc, B.; Hechtman, B.; Cai, T.; Borgeaud, S.; et al. 2022. Unified scaling laws for routed language models. In *International conference on machine learning*, 4057–4086. PMLR.
- Dai, D.; Deng, C.; Zhao, C.; Xu, R. X.; Gao, H.; Chen, D.; Li, J.; Zeng, W.; Yu, X.; Wu, Y.; Xie, Z.; Li, Y. K.; Huang, P.; Luo, F.; Ruan, C.; Sui, Z.; and Liang, W. 2024. DeepSeek-MoE: Towards Ultimate Expert Specialization in Mixtureof-Experts Language Models. arXiv:2401.06066.
- Dai, D.; Dong, L.; Ma, S.; Zheng, B.; Sui, Z.; Chang, B.; and Wei, F. 2022. Stablemoe: Stable routing strategy for mixture of experts. *arXiv preprint arXiv:2204.08396*.
- Fedus, W.; Zoph, B.; and Shazeer, N. 2022. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23(120): 1–39.
- Gale, T.; Narayanan, D.; Young, C.; and Zaharia, M. 2022. MegaBlocks: Efficient Sparse Training with Mixture-of-Experts. arXiv:2211.15841.
- Hendrycks, D.; Burns, C.; Basart, S.; Zou, A.; Mazeika, M.; Song, D.; and Steinhardt, J. 2021. Measuring Massive Multitask Language Understanding. *Proceedings of the International Conference on Learning Representations (ICLR)*.

- Huang, Q.; An, Z.; Zhuang, N.; Tao, M.; Zhang, C.; Jin, Y.; Xu, K.; Xu, K.; Chen, L.; Huang, S.; and Feng, Y. 2024. Harder Tasks Need More Experts: Dynamic Routing in MoE Models. arXiv:2403.07652.
- Jiang, A. Q.; Sablayrolles, A.; Roux, A.; Mensch, A.; Savary, B.; Bamford, C.; Chaplot, D. S.; Casas, D. d. l.; Hanna, E. B.; Bressand, F.; et al. 2024a. Mixtral of experts. *arXiv preprint arXiv:2401.04088*.
- Jiang, Z.; Lin, H.; Zhong, Y.; Huang, Q.; Chen, Y.; Zhang, Z.; Peng, Y.; Li, X.; Xie, C.; Nong, S.; et al. 2024b. {MegaScale}: Scaling Large Language Model Training to More Than 10,000 {GPUs}. In *21st USENIX Symposium on Networked Systems Design and Implementation (NSDI 24)*, 745–760.
- Lepikhin, D.; Lee, H.; Xu, Y.; Chen, D.; Firat, O.; Huang, Y.; Krikun, M.; Shazeer, N.; and Chen, Z. 2020. Gshard: Scaling giant models with conditional computation and automatic sharding. *arXiv preprint arXiv:2006.16668*.
- Li, J.; Sun, Z.; He, X.; Zeng, L.; Lin, Y.; Li, E.; Zheng, B.; Zhao, R.; and Chen, X. 2024. Locmoe: A low-overhead moe for large language model training. *arXiv preprint arXiv:2401.13920*.
- Lima, S. P.; and Cruz, M. D. 2020. A genetic algorithm using Calinski-Harabasz index for automatic clustering problem. *Revista Brasileira de Computac¸ao Aplicada ˜* , 12(3): 97–106.
- Liu, A.; Feng, B.; Xue, B.; Wang, B.; Wu, B.; Lu, C.; Zhao, C.; Deng, C.; Zhang, C.; Ruan, C.; et al. 2024a. Deepseekv3 technical report. *arXiv preprint arXiv:2412.19437*.
- Liu, T.; Blondel, M.; Riquelme, C.; and Puigcerver, J. 2024b. Routers in Vision Mixture of Experts: An Empirical Study. *arXiv preprint arXiv:2401.15969*.
- Maatouk, A.; Ayed, F.; Piovesan, N.; De Domenico, A.; Debbah, M.; and Luo, Z.-Q. 2023. Teleqna: A benchmark dataset to assess large language models telecommunications knowledge. *arXiv preprint arXiv:2310.15051*.
- Pham, Q.; Do, G.; Nguyen, H.; Nguyen, T.; Liu, C.; Sartipi, M.; Nguyen, B. T.; Ramasamy, S.; Li, X.; Hoi, S.; and Ho, N. 2024. CompeteSMoE – Effective Training of Sparse Mixture of Experts via Competition. arXiv:2402.02526.
- Rein, D.; Hou, B. L.; Stickland, A. C.; Petty, J.; Pang, R. Y.; Dirani, J.; Michael, J.; and Bowman, S. R. 2023. Gpqa: A graduate-level google-proof q&a benchmark. *arXiv preprint arXiv:2311.12022*.
- Shazeer, N.; Mirhoseini, A.; Maziarz, K.; Davis, A.; Le, Q.; Hinton, G.; and Dean, J. 2017a. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. arXiv:1701.06538.
- Shazeer, N.; Mirhoseini, A.; Maziarz, K.; Davis, A.; Le, Q.; Hinton, G.; and Dean, J. 2017b. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *arXiv preprint arXiv:1701.06538*.
- Xie, Z.; Zhang, Y.; Zhuang, C.; Shi, Q.; Liu, Z.; Gu, J.; and Zhang, G. 2024. MoDE: A Mixture-of-Experts Model with Mutual Distillation among the Experts. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 38, 16067–16075.

Zhao, W. X.; Zhou, K.; Li, J.; Tang, T.; Wang, X.; Hou, Y.; Min, Y.; Zhang, B.; Zhang, J.; Dong, Z.; et al. 2023. A survey of large language models. *arXiv preprint arXiv:2303.18223*.

Zhou, Y.; Lei, T.; Liu, H.; Du, N.; Huang, Y.; Zhao, V.; Dai, A.; Chen, Z.; Le, Q.; and Laudon, J. 2022a. Mixture-of-Experts with Expert Choice Routing. arXiv:2202.09368.

Zhou, Y.; Lei, T.; Liu, H.; Du, N.; Huang, Y.; Zhao, V.; Dai, A. M.; Le, Q. V.; Laudon, J.; et al. 2022b. Mixtureof-experts with expert choice routing. *Advances in Neural Information Processing Systems*, 35: 7103–7114.

# Appendix

### **Missing Proof**

# **Auxiurary Results**

**Lemma 8.** Let  $X_1, \ldots, X_n$  be n independent random variables with

$$\mathcal{P}(X_i = 1) = p_i, \mathcal{P}(X_i = 0) = 1 - p_i.$$
 (14)

We consider the sum  $X = \sum_{i=1}^{n} X_i$ , with expectation  $\mathcal{E}(X) = \sum_{i=1}^{n} p_i$ . Then we have

(Lower tail) 
$$\mathcal{P}(\mathbf{X} \leq \mathcal{E}\mathbf{X} - \lambda) \leq e^{-\frac{\lambda^2}{2\mathcal{E}\mathbf{X}}},$$
 (15)  
(Upper tail)  $\mathcal{P}(\mathbf{X} \geq \mathcal{E}\mathbf{X} + \lambda) \leq e^{-\frac{\lambda^2}{2(\mathcal{E}\mathbf{X} + \lambda/3)}}.$ 

#### **Proof of Theorem 5**

Proof. 1) For the TCR, denote

$$s_i = |\{t < k : \boldsymbol{x}_t \text{ sent to expert } i, \boldsymbol{x}_k = \boldsymbol{o}_i\}|, \forall i \in [n]$$
 (16)

as the top class-irrelevant token number candidated to the i-th expert before the valid token. Then by Assumption 4, each class-irrelevant token uniformly gives to any expert, leading to  $s_i|(\boldsymbol{x}_k=\boldsymbol{o}_i)\sim\mathcal{B}(k-1,1/n)$  (Binomial distribution), i.e.,  $\forall t\in[k-1]$ ,

$$\mathcal{P}(s_i = t | \boldsymbol{x}_k = \boldsymbol{o}_i) = {k-1 \choose t} \cdot \left(\frac{1}{n}\right)^t \left(1 - \frac{1}{n}\right)^{k-1-t}.$$
(17)

Then we could derive that

 $\mathcal{P}(\boldsymbol{x} \text{ succeed in training})$ 

$$= \sum_{i=1}^{n} \mathcal{P}(\boldsymbol{o}_{i} \text{ sent to expert } i | \boldsymbol{o}_{i} \text{ is in } \boldsymbol{x}) \cdot \mathcal{P}(\boldsymbol{o}_{i} \text{ is in } \boldsymbol{x})$$

$$= \frac{1}{ns} \sum_{i=1}^{n} \sum_{k=1}^{s} p_{i} \mathcal{P}(s_{i} < C | \boldsymbol{x}_{k} = \boldsymbol{o}_{i})$$

$$= \frac{1}{ns} \sum_{i=1}^{n} p_{i} \left( C + \sum_{k=C+1}^{s} \mathcal{P}(s_{i} < C | \boldsymbol{x}_{k} = \boldsymbol{o}_{i}) \right).$$

Note that  $\mathcal{E}s_i = (k-1)/n$ . When  $k \geq 2nC$ , by lower tail bound in Lemma 8, we get

$$\mathcal{P}(s_i < C | \boldsymbol{x}_k = \boldsymbol{o}_i) \le e^{-\frac{(k-1-n(C-1))^2}{2(k-1)n}} \le e^{-\frac{k-1}{8n}}.$$
 (18)

Hence, we get the upper bound that

 $\mathcal{P}(x \text{ succeed in training})$ 

$$\stackrel{0}{\leq} \frac{1}{ns} \sum_{i=1}^{n} \sum_{k=1}^{s} p_{i} \mathcal{P}(s_{i} < C | \boldsymbol{x}_{k} = \boldsymbol{o}_{i})$$

$$= \frac{1}{ns} \sum_{i=1}^{n} p_{i} \left( 2nC + \sum_{k=2nC+1}^{s} \mathcal{P}(s_{i} < C | \boldsymbol{x}_{k} = \boldsymbol{o}_{i}) \right)$$

$$\leq \frac{1}{ns} \sum_{i=1}^{n} p_{i} \left( 2nC + \sum_{k=2nC}^{s-1} e^{-\frac{k}{8n}} \right)$$

$$\leq \frac{1}{ns} \sum_{i=1}^{n} p_{i} \left( 2nC + \frac{e^{-\frac{C}{4}}}{1 - e^{-\frac{1}{8n}}} \right)$$

$$\stackrel{(i)}{\leq} \frac{1}{ns} \sum_{i=1}^{n} p_{i} \left( 2nC + (8n+1)e^{-\frac{C}{4}} \right) \leq \frac{10C \sum_{i=1}^{n} p_{i}}{s},$$

where (i) uses the inequality that  $e^{-t} \leq 1/(1+t), \forall t \geq 0$ . Moreover, for  $1+\frac{nC}{4} \leq k \leq 1+\frac{nC}{2}$ , i.e.,  $2(k-1) \leq nC \leq 4(k-1)$ , by upper tail bound in Lemma 8, we get

$$\mathcal{P}(s_i < C | \boldsymbol{x}_k = \boldsymbol{o}_i) = 1 - \mathcal{P}(s_i \ge C | \boldsymbol{x}_k = \boldsymbol{o}_i)$$
$$\ge 1 - e^{-\frac{3(nC - k + 1)^2}{2n[2(k-1) + nC]}} \ge 1 - e^{-\frac{k-1}{4n}}.$$

Hence, we get the lower bound that

 $\mathcal{P}(\boldsymbol{x} \text{ succeed in training})$ 

$$\stackrel{2}{\geq} \frac{1}{ns} \sum_{i=1}^{n} \sum_{k=1}^{s} p_{i} \mathcal{P}(s_{i} < C | \boldsymbol{x}_{k} = \boldsymbol{o}_{i})$$

$$= \frac{1}{ns} \sum_{i=1}^{n} p_{i} \left( \sum_{k=\lceil 1+nC/4 \rceil}^{\lfloor 1+nC/2 \rfloor} \mathcal{P}(s_{i} < C | \boldsymbol{x}_{k} = \boldsymbol{o}_{i}) \right)$$

$$\stackrel{2}{\geq} \frac{1}{ns} \sum_{i=1}^{n} p_{i} \left( \frac{nC}{4} - 1 - \sum_{k=\lceil 1+nC/4 \rceil}^{\lfloor 1+nC/2 \rfloor} e^{-\frac{k-1}{4n}} \right)$$

$$\stackrel{2}{\geq} \frac{1}{ns} \sum_{i=1}^{n} p_{i} \left( \frac{nC}{4} - 1 - \frac{e^{-\frac{C}{16}}}{1 - e^{-\frac{1}{4n}}} \right)$$

$$\stackrel{(i)}{\geq} \frac{1}{ns} \sum_{i=1}^{n} p_{i} \left( \frac{nC}{4} - 2 - (4n+1)e^{-\frac{C}{16}} \right) \stackrel{2}{\geq} \frac{C \sum_{i=1}^{n} p_{i}}{5s},$$

where (i) uses the inequality that  $e^{-t} \leq 1/(1+t), \forall t \geq 0$ , and the final inequality needs  $C \geq 48$ , which can be satisfied in common experiments. Combining the upper and lower bounds, we obtain the desired result.

2) For the ECR, denote  $s_i$  as the class-irrelevant token number with the score larger than  $o_i$  for i-th expert. By Assumption 4, we derive that  $s_i \sim \mathcal{B}(s-1,q_i), \forall i \in [n]$ .

 $\mathcal{P}(\boldsymbol{x} \text{ succeed in training})$ 

$$= \sum_{i=1}^{n} \mathcal{P}(\text{expert } i \text{ choose } \boldsymbol{o}_{i} | \boldsymbol{o}_{i} \text{ is in } \boldsymbol{x}) \mathcal{P}(\boldsymbol{o}_{i} \text{ is in } \boldsymbol{x})$$

$$= \frac{1}{n} \sum_{i=1}^{n} \mathcal{P}(s_{i} \leq C - 1, s_{i} \sim \mathcal{B}(s - 1, q_{i}))$$

If  $C-1 \leq (s-1)q_i/2$ , by lower tail bound in Lemma 8 with  $\lambda=(s-1)q_i-(C-1)<\mathcal{E}s_i$ , we obtain that

$$\mathcal{P}(s_i < C - 1) < e^{-\frac{(s-1)q_i}{2} \left(1 - \frac{C - 1}{(s-1)q_i}\right)^2} < e^{-\frac{(s-1)q_i}{8}}. \quad (19)$$

If  $C \ge 2(s-1)q_i$ , by upper tail bound in Lemma 8 with  $\lambda = C - (s-1)q_i > 0$ , we obtain that

$$\mathcal{P}(s_i \le C - 1) = 1 - \mathcal{P}(s_i \ge C)$$

$$\ge 1 - e^{-\frac{[C - (s-1)q_i]^2}{2(C + 2(s-1)q_i)/3}} \ge 1 - e^{-\frac{3C}{16}}.$$

Hence, we conclude Eq.(10).

#### **Token Feature Distribution**

We also validate the feature distribution before and after MoE training shown in Figure 13. We can see before training, all 8192 tokens in one training sample are nearly orthogonal with correlation coefficient near zero, which verifies the isotropy distribution assumption in the first bullet of Remark 7. After training, the token features are nearly aligned

![](_page_10_Figure_0.jpeg)

Figure 13: The correlation matrix of one training sample feature before (left) and after (right) training.

with correlation coefficien large than 0.8. We can also observe that neighbouring tokens share similar features, and clear block feature behavior, meaning that the token features are relatively separated and the number of tokens in each cluster is bounded, which somehow matches the distribution assumption in the second bullet of Remark 7.

# Experimental Setup

### Datasets for Pre-training and Fine-Tuning

The dataset used in this paper is a self-constructed dataset that integrates knowledge from multiple domains, including wireless, data communication, and cloud-core technologies. It comprises Chinese, English, and bilingual corpora. The corpora are parsed from various internal technical documents, such as iCase, blogs, Wiki, and feature documents. Taking iCase as an example, iCase is a case record of problem localization and handling processes, containing code, instructions, and corresponding logs. In addition, the abovementioned domain-specific knowledge corpora are mixed with general corpora in a ratio of 1:5. The general corpora are collected from hundreds of websites, including online novels, cooking guides, movie reviews, and more. After cleaning, deduplication, and review operations, the dataset is thoroughly shuffled. A total of 4.19 billion tokens is sampled as the experimental pre-training dataset. To evaluate downstream tasks, this paper also adopt hybrid sft data items to fine-tune the pre-trained model. The dataset comprises 762,321 general question-answer pairs and 11,048 domain-specific question-answer pairs, with a general-todomain ratio of 68:1. The general characteristics encompass multi-tasking, mathematical ability, coding ability, logical reasoning, multi-turn dialogue, knowledge reasoning, language understanding, text generation, multi-tasking, FunctionCall, CoT, MRC summarization, refusal to answer, Chinese, and English. The domain-specific characteristics include domain knowledge understanding, RAG, Function-Call, information extraction, multi-turn dialogue, reading comprehension, paraphrasing, and intent recognition.

The pre-training data comprises 300B tokens in total, with 150B tokens from the ICT domain and 150B tokens sampled from general data. The sampling ratios are shown in Table 3. For SFT data, we employed a two-stage training: the first stage primarily enhances the model's logical reasoning capabilities such as multi-task capability, mathematics, puzzlesolving, complex logic, etc. The total scale of samples is approximately two million, while the second stage focuses on improving instruction-following abilities, tool function call, sentiment, security, etc. The total scale of samples in stage2 is about three million.

#### Experimental Environment

The experiments are conducted on a cluster composed of Ascend 910B3 NPUs, divided into three groups: 32 NPUs (hereinafter referred to as 32N, and so on), 64N, and 256N. The 910B3 series NPU contains 20 AI cores with a main frequency of 1.8GHz and a theoretical computing power of 313T under fp16 precision. The physical High Bandwidth Memory (HBM) of the 910B3 NPU is 64G, with an HBM frequency of 1.6GHz and an HBM bandwidth of 1.6T. Every 8 NPUs are mounted on the same Atlas 800T A2 server, which internally adopts a fullmesh networking scheme, meaning that any two NPUs are interconnected.

#### Evaluation Metrics and Datasets

To evaluate model performance, this paper designs a comprehensive metric called the General and Domain-specific Assessment Dataset (GDAD), which consists of three evaluation systems: domain task capability, domain capability certification exam, and general capability. Among them, the domain task capability includes a total of 16 categories and 2,657 questions, such as domain logical reasoning; the domain capability certification exam includes a total of 13 categories and 13,968 questions, such as data communication; and the general capability includes a total of 18 categories and 1,435 questions, such as programming ability. The questions include objective and subjective questions in Chinese, English, and bilingual formats. For subjective questions, the cosine similarity between the model output and the standard answer is used as the score. In addition, this paper also employs GPQA (Rein et al. 2023) and TeleQnA (Maatouk et al. 2023) to evaluate the model's Chinese language capability.

Table 3: Data sources and sampling ratios of general pre-training data.

| Primary Category       | Secondary Category | Tertiary Source                  | Sampling ratio |
|------------------------|--------------------|----------------------------------|----------------|
| General English        | Webpages           | Reasoning steplist               | 25%            |
|                        |                    | Model rewrite                    | 100%           |
|                        | Books & Papers     | book3                            | 25%            |
|                        |                    | bookcorpus                       | 100%           |
|                        |                    | all libgen books                 | 20%            |
|                        |                    | all libgen scihub                | 10%            |
|                        |                    | RedPajama arxiv                  | 25%            |
|                        |                    | arxiv latex2Markdown cleaned     | 25%            |
|                        |                    | wiki                             | 100%           |
|                        | WebText            | stackexchange cleaned            | 20%            |
|                        |                    | cosmopedia v2                    | 15%            |
| General Chinese        | Webpages           | aigc dataset                     | 15%            |
|                        | Book               | all book deduped                 | 10%            |
|                        |                    | zh book CommonData               | 10%            |
|                        |                    | zh general STEM                  | 80%            |
|                        |                    | all zhiwang                      | 20%            |
|                        | WebText            | baike MBAzhiku sougou ye zhiarge | 50%            |
|                        |                    | baike sougou baidu kuaidong      | 50%            |
|                        |                    | wiki                             | 10%            |
|                        |                    | zhihu caigou merged cleaned      | 10%            |
| High-density Knowledge | Q&A                | quiz data                        | 100%           |
|                        | Collection         | density knowledge                | 100%           |
|                        |                    | collection updated               | 100%           |
|                        |                    | english question and answer      | 100%           |
|                        |                    | annealing                        | 100%           |
|                        | Code               | code python edu high quality     | 30%            |
| Code                   | Forum              | CSDN                             | 20%            |
|                        |                    | Ultra textbooks                  | 100%           |