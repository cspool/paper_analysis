## Unveiling Hidden Collaboration within Mixture-of-Experts in Large Language Models

Yuanbo Tang<sup>1</sup> , Yan Tang<sup>2</sup> , Naifan Zhang<sup>1</sup> , Meixuan Chen<sup>1</sup> , Yang Li1,\*

- <sup>1</sup> Tsinghua University International Campus Phase I, Nanshan District, Shenzhen
- <sup>2</sup> College of Software, Northeastern University, Shenyang, Liaoning Province

#### Abstract

Mixture-of-Experts based large language models (MoE LLMs) have shown significant promise in multitask adaptability by dynamically routing inputs to specialized experts. Despite their success, the collaborative mechanisms among experts are still not well understood, limiting both the interpretability and optimization of these models. In this paper, we focus on two critical issues: (1) identifying expert collaboration patterns, and (2) optimizing MoE LLMs through expert pruning. To address the first issue, we propose a hierarchical sparse dictionary learning (HSDL) method that uncovers the collaboration patterns among experts. For the second issue, we introduce the Contribution-Aware Expert Pruning (CAEP) algorithm, which effectively prunes low-contribution experts. Our extensive experiments demonstrate that expert collaboration patterns are closely linked to specific input types and exhibit semantic significance across various tasks. Moreover, pruning experiments show that our approach improves overall performance by 2.5% on average, outperforming existing methods. These findings offer valuable insights into enhancing the efficiency and interpretability of MoE LLMs, offering a clearer understanding of expert interactions and improving model optimization.

#### 1 Introduction

In recent years, the MoE LLMs have gained significant attention as a computationally efficient framework, demonstrating exceptional representational power for large-scale machine learning tasks [\(Jiang](#page-8-0) [et al.,](#page-8-0) [2024;](#page-8-0) [Fedus et al.,](#page-8-1) [2022\)](#page-8-1). By leveraging a dynamic routing mechanism, MoE enables the collaborative operation of specialized "Experts", each designed to process complex input data. Compared to traditional architectures, MoE LLMs offer more flexible and adaptive knowledge representations while reducing computational costs, making them

<span id="page-0-0"></span>![](_page_0_Figure_10.jpeg)

Figure 1: In MoE LLMs, a group of experts often collaborate to analyze a certain type of tokens, and they are not necessarily in the same layer.

well-suited for resource-intensive situations [\(Cai](#page-8-2) [et al.,](#page-8-2) [2024\)](#page-8-2).

Existing research on understanding the working mechanism of MoE LLMs has largely focused on analyzing the behavior of the router, which governs expert selection [\(Lo et al.,](#page-8-3) [2024\)](#page-8-3). For instance, some studies highlight the influence of output norms on expert selection [\(Lo et al.,](#page-8-3) [2024\)](#page-8-3), while others reveal that token IDs play a significant role in routing decisions [\(Jiang et al.,](#page-8-0) [2024;](#page-8-0) [Xue](#page-9-0) [et al.,](#page-9-0) [2024;](#page-9-0) [Dai et al.,](#page-8-4) [2024\)](#page-8-4). These efforts have provided valuable insights into how MoE LLMs allocate tasks to specialized experts, enhancing multitask adaptability.

Despite the widespread success of MoE LLMs, several key challenges remain underexplored. One of the main challenges is understanding the collaborative mechanisms among the experts within the network. While MoE LLMs generate final outputs by combining the predictions of multiple experts, how these experts cooperate to produce the outputs is still not well understood. Figure [1](#page-0-0) conceptualizes the notion of cross-layer expert collaboration coordinated groups of experts across distinct layers that exhibit synchronized activation to implement specific functional modules. This phenomenon is empirically validated in operational MoE networks. Figure [2](#page-1-0) illustrates a representative case of strong

<sup>\*</sup> Corresponding author: [tori2011@gmail.com](mailto:tori2011@gmail.com)

co-activation patterns between Expert 21 in Layer 5 and Expert 3 in Layer 6. Comprehending these collaboration patterns is essential, as it directly influences knowledge sharing, model interpretability, performance, and optimization. Another key challenge lies in the high model complexity of MoE LLMs, which presents significant challenges in terms of deployment, limiting their scalability for large-scale applications [\(Lu et al.,](#page-8-5) [2024;](#page-8-5) [He et al.,](#page-8-6) [2024\)](#page-8-6).

Therefore, this study aims to investigate and reveal the collaboration patterns between experts in MoE LLMs, and utilize these patterns to enhance model efficiency and performance. The core questions we address include: (1) Are there consistent collaboration patterns among experts, and what do they reveal about the tasks implicitly learned in MOE LLMs? (2) Can these collaboration patterns be leveraged to compress MoE LLMs?

To address the two key questions, we begin by extracting the expert activation matrix, which serves as the foundation for further analysis. For the first question, we apply a novel hierarchical sparse dictionary learning (HSDL) approach to uncover collaboration structures within the expert activation data. Building on these insights, we then investigate expert pruning through the Contribution-Aware Expert Pruning (CAEP) algorithm, which identifies and removes low-contribution experts. This process reduces model redundancy, alleviating storage pressure while preserving or even enhancing performance. The entire pipeline, as outlined in Figure [3,](#page-2-0) comprises three key components: (1) Expert Activation Data Collection, (2) MoE Collaboration Pattern Mining, and (3) Expert Pruning Based on Expert Collaboration Pattern.

In our experimental evaluation, we tested several representative MoE architectures, including the DeepSeek model, on the MMLU-pro dataset, which contains 2,812 samples across five chosen domains: mathematics, computer science, physics, law, and psychology. Our analysis of the learned dictionaries revealed domain-specific expert collaboration patterns with distinct semantic significance. Building on these insights, we conducted pruning experiments using the CAEP method, which demonstrated that pruning experts based on these patterns effectively reduces the number of experts while maintaining or even improving performance. Our method outperforms baselines with an average improvement of 2.5%, and in the best case, pruning 50% of experts results in only a 5.7% performance

<span id="page-1-0"></span>![](_page_1_Figure_4.jpeg)

Figure 2: Here (x, y) refers to the y-th expert in x-th layer. By selecting any two experts from the MoE, we can calculate the probability of their co-activation. It can be observed that Expert 21 from the layer 5 and Expert 3 from the layer6 frequently activate simultaneously, forming an expert collaboration pattern.

drop for specific tasks.

Our contribution can be summarized as follows:

- We explore and uncover the latent collaboration patterns among experts in MoE LLMs. We propose hierarchical sparse dictionary learning (HSDL) and reveal how experts interact and cooperate, which provides new insights into the collaborative mechanisms that drive the performance of MoE LLMs.
- We propose the Contribution-Aware Expert Pruning (CAEP) algorithm, which optimizes model efficiency by pruning low-contribution experts without sacrificing performance. Our experiments show that CAEP maintains competitive performance while significantly reducing the number of experts, effectively balancing pruning and performance retention.

## 2 Literature Review

#### 2.1 Analysis of Routing in MoE Networks

The analysis of router behavior in MoE networks focuses on understanding how the model selects experts based on input features, which is key for optimizing performance. For instance, Lo et al. found that routers typically select experts with larger output norms [\(Lo et al.,](#page-8-3) [2024\)](#page-8-3), while other studies suggest that router choices are more related to token IDs than to expert fields [\(Jiang et al.,](#page-8-0) [2024;](#page-8-0) [Xue et al.,](#page-9-0) [2024;](#page-9-0) [Dai et al.,](#page-8-4) [2024\)](#page-8-4). While these approaches offer valuable insights, they often treat

<span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

Figure 3: Overview of Our Study's Pipeline.

experts as independent entities, overlooking the collaboration patterns between them.

2.2 Expert Pruning in MoE

Expert pruning reduces storage consumption in MoE networks by removing less impactful experts. Current strategies include: (1) discarding experts with low activation frequencies based on router decisions (Muzio et al., 2024), (2) identifying experts with minimal output influence using |x-f(x)| differences (Lu et al., 2024; He et al., 2024), and (3) merging experts by calculating weight similarities (Li et al., 2023; Zhang et al., 2024). However, these methods often treat experts independently or focus on merging similar groups, without exploring diverse expert combinations with distinct roles.

#### 2.3 Sparse Dictionary Learning

Sparse dictionary learning is a well-established method in representation learning and dimensionality reduction (Yang et al., 2010; Wright et al., 2009). It constructs a dictionary of features that enables sparse representation of data, facilitating efficient encoding of high-dimensional information (Tang et al., 2023; Chen et al., 2013). This approach has proven effective in various applications, such as image processing and signal recovery, where it helps capture essential features while reducing noise (Hou et al., 2021, 2020). Recently, companies like OpenAI, Google, and Anthropic have applied sparse dictionary learning to understand large language models' mechanisms (Rajamanoharan et al., 2024; Gao et al.). Despite its success in other areas, sparse dictionary learning

has been underutilized in explanatory research on MoE networks.

#### 3 Extraction of Expert Activation Matrix

In MoE LLMs, the activation weights of the experts reflect the intensity of their responses to the input data, thereby elucidating the collaborative patterns among them. Furthermore, these activation data provide a foundational basis for optimizing pruning strategies, which in turn contribute to enhanced computational and storage efficiency. Consequently, the extraction and analysis of activation weights are critical steps in the effective exploration of collaboration patterns and the implementation of pruning techniques.

Given an MoE LLM with m layers and n experts, and an input dataset S containing  $N_s$  samples, we extract the expert activation data to construct a two-dimensional activation tensor  $V \in \mathbb{R}^{N_s \times (m \times n)}$ , where each element  $v_{i,j,k}$  represents the activation weight of the k-th expert in the j-th layer for the i-th sample. This activation weight quantifies the intensity of the expert's response to the input sample, with values constrained within the range [0,1].

To aggregate the activation data of each sample into a sentence-level representation, we sum the activation values of all tokens within a sample, thereby obtaining the sentence-level activation value for each layer. Let  $\alpha(i)_{t,j,k}$  denote the routing allocation of the t-th token in sample  $S_i$  to the k-th expert in the j-th layer. The sentence-level

activation value is then computed as:

$$v_{i,j,k} = \sum_{t=1}^{T} \alpha(i)_{t,j,k}.$$
 (1)

where T represents the sequence length. Finally, by transposing and accumulating these activation data, we construct the expert activation matrix X, which serves as the input to the subsequent analysis of collaboration patterns among experts.

## 4 MoE Collaboration Pattern Mining

In this section, we propose a novel **Hierarchical Sparse Dictionary Learning (HSDL)** approach to uncover collaboration patterns among experts in MoE LLMs through hierarchical decomposition. Furthermore, We evaluate its effectiveness on the MMLU-pro dataset, validating the method by comparing it to exhaustive search techniques and exploring domain-specific expert interactions, demonstrating its versatility and efficiency in capturing complex MoE dynamics.

#### 4.1 Problem Definition

The objective of this task is to extract the collaboration patterns among experts in MoE LLMs. Given a dataset  $S = \{s_1, s_2, \ldots, s_{N_s}\}$  comprising  $N_s$  samples, we construct an expert activation matrix  $X \in \mathbb{R}^{N_e \times N_s}$ , where  $N_e$  denotes the total number of experts. By employing sparse dictionary learning techniques to decompose X, we obtain a dictionary matrix  $D \in \mathbb{R}^{N_e \times N_p}$  and a sparse coding matrix  $R \in \mathbb{R}^{N_p \times N_s}$ , with  $N_p$  representing the predefined dictionary capacity. Our goal is to decompose the expert activation matrix  $\mathbf{X}$  into a dictionary matrix  $\mathbf{D}$  and a sparse coding matrix  $\mathbf{R}$ , which can be expressed as follows:

$$X \approx D \cdot R.$$
 (2)

Here, the dictionary matrix D encodes the collaboration patterns among experts, while the sparse coding matrix R determines how these patterns combine to reconstruct X.

## **4.2** Hierarchical Sparse Dictionary Learning for Expert Collaboration Patterns Mining

Sparse dictionary learning is an effective unsupervised method for uncovering latent structures in data through sparse representations. By modeling data as a linear combination of dictionary atoms, it reveals expert collaboration patterns in

<span id="page-3-0"></span>![](_page_3_Picture_11.jpeg)

Figure 4: Hierarchical Sparse Dictionary Learning.

MoE LLMs. However, a single-layer approach fails to capture complex patterns across varying granularities. To address this, we propose the HSDL approach, which recursively decomposes the dictionary matrix, capturing collaboration patterns from coarse to fine granularity, thus revealing multi-layered expert interactions.

We extend the original single-layer structure decomposition into a hierarchical structure by recursively decomposing the dictionary matrix at each layer k into finer subpatterns represented by  $D_{k+1}$ , formulated as:

$$D_k \approx D_{k+1} \cdot R_{k+1}. \tag{3}$$

Figure 4 illustrates the hierarchical structure of Sparse Dictionary Learning, showing how the multi-layered expert collaboration is modeled across different layers.

Furthermore, we introduce three key constraints to optimize the multi-layer dictionary learning process:

(1) **Sparsity Constraint**: This ensures that the sparse coding matrix  $R_k$  at each layer remains sparse, preventing certain dictionary elements from dominating. Specifically,  $R_{k,i,:}$  denotes the sparse coding of the i-th data point at layer k. This constraint is defined as:

$$L_{\text{sparse}} = ||R_{k,i,:}||_{\infty}. \tag{4}$$

(2) Inter-Layer Consistency Constraint: This controls the influence of dictionary learning across layers. The matrix  $R_{k,j}$  represents the contribution of the j-th dictionary atom at layer k. The formula is:

$$L_{\text{hier}} = \sum_{j} \|R_{k+1,j}\|_{1} \cdot \|R_{k,j}\|_{1}/N.$$
 (5)

(3) Reconstruction Error Term: This ensures that the relationships between dictionaries at successive layers are consistently learned. The reconstruction error is defined as:

$$L_{\text{rec}} = \sum_{j} \|D_{k,j} - (D_{k+1}R_{k+1})_{j}\|_{1} \cdot \|R_{k,j}\|_{1}/N.$$
(6)

These three constraints collectively guide the optimization of both the hierarchical dictionary and sparse coding matrices. The overall loss function is formulated as:

$$L_{\text{total}} = L_{\text{sparse}} + \lambda_1 L_{\text{hier}} + \lambda_2 L_{\text{rec}}, \qquad (7)$$

where λ<sup>1</sup> and λ<sup>2</sup> are hyperparameters that control the respective losses. By minimizing this loss function, we optimize both the dictionary matrix D<sup>k</sup> and the sparse coding matrix R<sup>k</sup> at each layer, effectively capturing the multi-level structure of expert collaboration.

## 4.3 Experimental Analysis of Expert Collaboration Patterns

In this subsection, we aim to explore how the collaboration patterns among experts in MoE-based LLMs reflect the tasks implicitly learned by the model, thereby contributing to a deeper understanding of its functioning. We present a detailed analysis of the expert collaboration patterns identified through our hierarchical sparse dictionary learning method. To investigate these patterns and their semantic implications, we conduct a series of experiments using the MMLU-pro dataset.

#### 4.3.1 Experimental Setup

We use the phi-moe model and apply our HSDL method to 2,812 samples from the MMLU-pro dataset, covering five domains: mathematics, computer science, physics, law, and psychology.

### 4.3.2 Prompt Interpretation using Expert Collaboration Pattern

To explore how expert collaboration patterns in MoE LLMs reflect the model's understanding of tasks, we conduct a detailed analysis using the hierarchical dictionary learning method. Specifically, we aim to understand how different experts collaborate to handle specific aspects of a problem.

To achieve this, we designed a semantic annotation scheme for input sentences to interpret the semantics of expert collaboration patterns derived from HSDL. We color words processed by the

same dictionary atoms (i.e., expert collaboration patterns) with the same color. This color-coding scheme facilitates the observation of both the performance and interrelationships of the expert collaboration patterns. We analyze the input samples using the dictionary atoms obtained through HSDL, with one such analysis shown in Figure [5.](#page-4-0)

<span id="page-4-0"></span>![](_page_4_Figure_13.jpeg)

Figure 5: Hierarchical Semantic Annotation of Dictionary Elements on MMLU.

Results and Discussion. We find that the hierarchical semantic annotation of expert collaboration patterns reveals how MoE LLMs understand and process different tasks within a problem. As shown in Figure [5,](#page-4-0) in the upper left corner, we can observe that: Expert collaboration patterns in higher-layer and lower-layer dictionaries demonstrate a hierarchical semantic relationship, which becomes increasingly fine-grained as layer increases. The lower left corner of the figure displays this from a semantic perspective, where the top layer captures broad categories such as "Date, symbol, and mathematical calculation," while deeper layers break these down into more detailed components like "Mathematical calculation" or "Key verbs."

These findings provide a direct answer to our central question on expert collaboration patterns in MoE LLMs. The hierarchical decomposition offers a more detailed understanding of the model's internal processes, shedding light on how tasks are learned and executed. This approach could evolve into a tool for visualizing MoE LLMs behavior, enhancing interpretability and supporting optimization for domain-specific applications.

## 4.3.3 Comparison with Exhaustive Search Results

To investigate whether the top dictionary elements correspond to the most frequent expert combinations, we compared the dictionary's expert collaboration patterns with those from an exhaustive search

method. Due to the high computational cost of considering larger combinations, we limited the analysis to pairs and triplets.

To quantify the coverage of the most frequent expert combinations in our dictionary, we define Ntop as the number of dictionary items in the top k% of the traversal pattern, and Ntotal as the total number of dictionary items. The coverage is then calculated using the following formula:

Top 
$$k\%$$
 Coverage =  $\frac{N_{\text{top}}}{N_{\text{total}}}$ . (8)

<span id="page-5-0"></span>![](_page_5_Figure_3.jpeg)

Figure 6: Comparison of overlap with the results of the exhaustive method.

Results and Discussion. As shown in Figure [6,](#page-5-0) the collaboration patterns identified by our method predominantly align with the most frequent expert combinations found during the exhaustive search. Specifically, 60% of the patterns identified by our method correspond to the top 10% of the most frequent expert combinations, indicating that our method efficiently identifies the most prevalent collaboration patterns.

While our method focuses on the most frequent expert combinations, it also captures some lowfrequency patterns. These less frequent combinations, though less common, are critical for capturing the diversity of expert interactions, which enhances the model's ability to tackle a wider range of tasks. This highlights the importance of considering both high- and low-frequency expert combinations in shaping the performance and versatility of MoE LLMs.

## 4.3.4 Domain-Specific Expert Collaboration Patterns

In this experiment, our goal is to explore how expert collaboration patterns vary across different domains and to understand the domain-specific nature of expert interactions within MoE LLMs. Specifically, we aim to examine the activation frequencies

of experts for inputs from various fields, including mathematics, computer science, physics, law, and psychology, to uncover potential domain-related patterns.

we analyzed the frequency distribution of activated experts during the model processing for inputs from different domains and calculated the cosine similarity between the distributions of each domain, resulting in a confusion matrix.

<span id="page-5-1"></span>![](_page_5_Figure_11.jpeg)

Figure 7: The distribution of expert selection frequencies during inputs from different fields.

Results and Discussion. Figure [7](#page-5-1) shows the expert selection frequency distribution across domains. We can observe that for inputs from different fields, the distribution of expert activation frequencies in the MoE LLM varies. For semantically similar domains, such as mathematics, physics, and computer science indicated by the orange dashed box, their distributions are closer to each other. In contrast, the distributions of expert activation frequencies are more different for domains with greater semantic differences, such as mathematics and law. This suggests that expert collaboration is more specialized within specific domains, reflecting domain-specific interactions in MoE LLMs.

These findings indicate that experts in MoE LLMs exhibit domain preferences, adjusting expert selection based on the input domain's characteristics to optimize performance for domainspecific tasks. Understanding these patterns can enhance the model's efficiency and its ability to handle specialized tasks.

## 5 Expert Pruning Based on Expert Collaboration Patterns

In this section, we present the CAEP method, which utilizes expert collaboration patterns to reduce the number of experts in an MoE LLM while preserving performance. We first introduce the pruning algorithm and then demonstrate its effectiveness through two types of experiments: (1) General Tasks Evaluation, where we compare CAEP with baseline methods on diverse tasks, and (2) Domain-Specific Evaluation, where we assess its ability to retain domain-relevant capabilities after pruning.

#### 5.1 Pruning algorithm

We propose the **Contribution-Aware Expert Pruning** (**CAEP**) algorithm. The algorithm aims to produce a mask vector that incorporates our retention strategy, given a specific pruning ratio k. In this mask vector, experts corresponding to positions with a value of 1 are retained, while those with a value of 0 are discarded. This pruning process is achieved by progressively discarding less significant dictionary atoms, guided by the contribution scores derived from R. The CAEP algorithm proceeds as follows (Algorithm 1):

- Calculation and Ranking: Calculate the contribution scores for each expert by the sparse representation matrix R and the dictionary matrix D, obtaining the total contribution and sorting it in descending order.
- Initial Threshold Mask: Determine the score based on the predefined threshold ratio and generate the initial binary mask, marking the experts whose contribution scores are above.
- Iterative Pruning: Before reaching the target pruning ratio, repeatedly identify the least used patterns and remove them from the dictionary and the sparse representation while updating the contribution scores and the mask, until only the desired ratio of experts remains.

# 5.2 Experiments on General and Domain-Specific Tasks

We conduct a series of experiments to evaluate the effectiveness of our proposed pruning method, CAEP. We perform experiments on both general tasks and domain-specific tasks. The goal is to assess how well the pruned model retains its capabilities across a variety of tasks, while optimizing performance retention in specific domains. The dataset and specific configurations used in this part of the experiment can be found in Appendix B.

#### 5.2.1 Experiments on General Tasks

The goal of this experiment is to evaluate how well the pruned model retains its performance across

#### <span id="page-6-0"></span>**Algorithm 1** Expert Pruning Strategy

```
Require: Dictionary matrix \mathbf{D} \in \mathbb{R}^{N_e \times N_p}
  1: Sparse representation matrix \mathbf{R} \in \mathbb{R}^{N_p \times N_s}
  2: Threshold ratio k_1 \in (0,1)
 3: Target pruning ratio k_2 \in (0,1)
Ensure: Pruned expert mask \mathbf{m} \in \{0, 1\}^{N_e}
 4: \mathbf{R}_{\text{sum}} \leftarrow \sum_{j=1}^{N_s} \mathbf{R}_{:,j}

  5: \mathbf{D}_{sum} \leftarrow \mathbf{D} \cdot \mathbf{R}_{sum}^{\top}

 6: \mathbf{e} \leftarrow \sum_{i=1}^{N_p} \mathbf{D}_{\text{sum},i}

  7: Sort e in descending order: e_{\text{sorted}}
  8: f \leftarrow \mathbf{e}_{\text{sorted}}[\lceil k_1 \cdot N_e \rceil]
                                                        \triangleright Threshold at k_1-quantile
  9: \mathbf{m} \leftarrow \mathbf{1}_{\mathbf{e} \geq j}
                                                                 ▶ Initial binary mask
10: while \|\vec{\mathbf{m}}\|_0 > (1 - k_2) \cdot N_e do
11:
             i^* \leftarrow \arg\min_i \mathbf{R}_{\text{sum}}(i)
                                                          ▶ Find least used pattern
             Remove column i^* from D and row i^* from R
12:
             Recompute \mathbf{R}_{sum}, \mathbf{D}_{sum}, \mathbf{e}
13:
             Update \mathbf{m} \leftarrow \mathbf{1}_{\mathbf{e} > f}
14:

⊳ Adapt mask

15. end while
               return m
```

a broad set of general tasks. We compare CAEP with baseline pruning methods to analyze the tradeoff between reducing the number of experts and maintaining task performance.

Comparison with Other Expert Pruning Baselines. We compare CAEP to two baseline pruning strategies: (1) Routing Score-Based Pruning (Muzio et al., 2024): Retains experts with higher averaged routing scores. (2) Behavior-based Pruning (Zhang et al., 2024): Remove experts with minimal impact on the output.

Results and Discussion. Figure 8 and Table 1 show that CAEP-pruned models maintain competitive performance, outperforming random and baseline methods with an average score of 0.612. Notably, CAEP retains higher performance after pruning 25% of the experts, especially on tasks like OBQA and RTE. This is further supported by Figure 8, where CAEP shows a lower accuracy drop across multiple tasks, including HellaSwag and PIQA, even with a high pruning ratio.

Through the analysis of the experimental results, we found that CAEP effectively retains performance across a broad set of general tasks while significantly reducing the number of experts. This demonstrates that CAEP successfully balances pruning and performance retention, optimizing computational efficiency while minimizing performance loss.

#### 5.2.2 Experiments on Domain-Specific Tasks

In this experiment, we focus on investigating how expert collaboration patterns differ across various domains and how these differences reflect the domain-specific interactions within MoE

Table 1: Performance evaluation of different expert pruning methods with 25% experts dropped.

<span id="page-7-1"></span>

| Model    | Method      | AVG↑   | OBQA↑ | ARC-C↑ | HellaSwag↑ | WinoGrande↑ | RTE↑  | PIQA↑ |
|----------|-------------|--------|-------|--------|------------|-------------|-------|-------|
| DeepSeek | Random      | 0.500  | 0.363 | 0.564  | 0.485      | 0.568       | 0.641 | 0.381 |
|          | SEER-MoE    | 0.5872 | 0.420 | 0.672  | 0.665      | 0.617       | 0.755 | 0.394 |
|          | GEM         | 0.5870 | 0.422 | 0.67   | 0.658      | 0.649       | 0.739 | 0.384 |
|          | CAEP (Ours) | 0.612  | 0.473 | 0.693  | 0.691      | 0.635       | 0.757 | 0.424 |

<span id="page-7-0"></span>![](_page_7_Figure_2.jpeg)

Figure 8: Performance of CAEP on benchmark tasks with varying expert pruning drop ratios.

LLMs. Our objective is to analyze the activation frequencies of experts for inputs from five fields—mathematics, computer science, physics, law, and psychology—in order to identify domaindependent patterns in expert selection.For each domain, we prune 50% of the experts using CAEP on Phi. This setup enables us to assess whether the pruned model retains superior performance in a specific domain at the expense of others.

Performance Evaluation Metric. To assess the impact of pruning, we focus primarily on the relative changes in performance. The metric is computed as:

$$\frac{Acc_{\text{pruned}} - Acc_{\text{no-pruned}}}{Acc_{\text{no-pruned}}}.$$
 (9)

A higher value indicates better retention of domainspecific capabilities, with the ideal result being maximized diagonal elements, showing that each pruned model retains domain-specific expertise.

Results and Discussion. Figure [9](#page-7-2) shows the accuracy degradation after pruning for different domains, presented as a heatmap. The color scale indicates the percentage of accuracy drop, where darker blue shades represent larger losses. From the figure, we observe that pruning for domains like law and psychology leads to the most significant accuracy drops, particularly when the target domain is law. In contrast, pruning for the "physics" or "psychology" domains results in relatively smaller accuracy drops, suggesting a less severe impact on performance.

We find that this variation in pruning impact,

<span id="page-7-2"></span>![](_page_7_Figure_10.jpeg)

Figure 9: Performance degradation accuracy after pruning for specific domain

depending on both the target and benchmark domains, reveals an uneven distribution of domainspecific knowledge across the model. Some domains rely more heavily on specialized expertise, while others are more flexible in terms of expert collaboration. These findings suggest that pruning strategies should account for the varying importance of domain-specific knowledge, allowing for more efficient expert retention and minimizing unnecessary performance degradation in MoE LLMs.

#### 6 Conclusion

This paper addresses a key gap in MoE LLMs, where existing research has largely overlooked the collaboration patterns among experts, both within the same layer and across layers. By applying hierarchical sparse dictionary learning, we uncover dominant expert collaboration patterns and develop a pruning strategy to enhance MoE LLMs' efficiency. Our experiments demonstrate that this approach not only improves accuracy but also significantly boosts model compression and inference efficiency compared to existing methods. This work provides valuable insights into expert interactions and offers a novel way to optimize MoE LLMs for both performance and scalability.

### References

- <span id="page-8-18"></span>Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. 2020. Piqa: Reasoning about physical commonsense in natural language. In *Proceedings of the AAAI conference on artificial intelligence*, volume 34, pages 7432–7439.
- <span id="page-8-2"></span>Weilin Cai, Juyong Jiang, Fan Wang, Jing Tang, Sunghun Kim, and Jiayi Huang. 2024. [A Survey](http://arxiv.org/abs/2407.06204) [on Mixture of Experts.](http://arxiv.org/abs/2407.06204) *arXiv preprint*.
- <span id="page-8-9"></span>Chen Chen, Hao Su, Qixing Huang, Lin Zhang, and Leonidas Guibas. 2013. [Pathlet learning for com](https://doi.org/10.1145/2525314.2525443)[pressing and planning trajectories.](https://doi.org/10.1145/2525314.2525443) In *Proceedings of the 21st ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems*, pages 392–395, Orlando Florida. ACM.
- <span id="page-8-16"></span>Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019. [BoolQ: Exploring the surprising](https://doi.org/10.18653/v1/N19-1300) [difficulty of natural yes/no questions.](https://doi.org/10.18653/v1/N19-1300) In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 2924–2936, Minneapolis, Minnesota. Association for Computational Linguistics.
- <span id="page-8-15"></span>Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. [Think you have solved question](https://arxiv.org/abs/1803.05457) [answering? try arc, the ai2 reasoning challenge.](https://arxiv.org/abs/1803.05457) *Preprint*, arXiv:1803.05457.
- <span id="page-8-4"></span>Damai Dai, Chengqi Deng, Chenggang Zhao, R. X. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y. K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang. 2024. [DeepSeekMoE: Towards](https://doi.org/10.48550/arXiv.2401.06066) [Ultimate Expert Specialization in Mixture-of-Experts](https://doi.org/10.48550/arXiv.2401.06066) [Language Models.](https://doi.org/10.48550/arXiv.2401.06066) *arXiv preprint*.
- <span id="page-8-1"></span>William Fedus, Barret Zoph, and Noam Shazeer. 2022. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23(120):1–39.
- <span id="page-8-13"></span>Leo Gao, Gabriel Goh, and Ilya Sutskever. Scaling and evaluating sparse autoencoders.
- <span id="page-8-20"></span>Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac'h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, d Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. 2023. [A framework for few-shot language model](https://doi.org/10.5281/zenodo.10256836) [evaluation.](https://doi.org/10.5281/zenodo.10256836)
- <span id="page-8-6"></span>Shwai He, Daize Dong, Liang Ding, and Ang Li. 2024. [Demystifying the Compression of Mixture](https://doi.org/10.48550/arXiv.2406.02500)[of-Experts Through a Unified Framework.](https://doi.org/10.48550/arXiv.2406.02500) *arXiv preprint*.

- <span id="page-8-14"></span>Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. [Measuring massive multitask language under](https://openreview.net/forum?id=d7KBjmI3GmQ)[standing.](https://openreview.net/forum?id=d7KBjmI3GmQ) In *International Conference on Learning Representations*.
- <span id="page-8-11"></span>Zhi Hou, Xiaojiang Peng, Yu Qiao, and Dacheng Tao. 2020. [Visual Compositional Learning for Human-](https://doi.org/10.1007/978-3-030-58555-6_35)[Object Interaction Detection.](https://doi.org/10.1007/978-3-030-58555-6_35) In *Computer Vision – ECCV 2020*, pages 584–600, Cham. Springer International Publishing.
- <span id="page-8-10"></span>Zhi Hou, Baosheng Yu, Yu Qiao, Xiaojiang Peng, and Dacheng Tao. 2021. [Detecting Human-Object Inter](https://openaccess.thecvf.com/content/CVPR2021/html/Hou_Detecting_Human-Object_Interaction_via_Fabricated_Compositional_Learning_CVPR_2021_paper.html)[action via Fabricated Compositional Learning.](https://openaccess.thecvf.com/content/CVPR2021/html/Hou_Detecting_Human-Object_Interaction_via_Fabricated_Compositional_Learning_CVPR_2021_paper.html) pages 14646–14655.
- <span id="page-8-0"></span>Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2024. [Mix](https://doi.org/10.48550/arXiv.2401.04088)[tral of Experts.](https://doi.org/10.48550/arXiv.2401.04088) *arXiv preprint*.
- <span id="page-8-8"></span>Pingzhi Li, Zhenyu Zhang, Prateek Yadav, Yi-Lin Sung, Yu Cheng, Mohit Bansal, and Tianlong Chen. 2023. Merge, then compress: Demystify efficient smoe with hints from its routing policy. *CoRR*.
- <span id="page-8-3"></span>Ka Man Lo, Zeyu Huang, Zihan Qiu, Zili Wang, and Jie Fu. 2024. [A Closer Look into Mixture-of-Experts in](https://doi.org/10.48550/arXiv.2406.18219) [Large Language Models.](https://doi.org/10.48550/arXiv.2406.18219) *arXiv preprint*.
- <span id="page-8-5"></span>Xudong Lu, Qi Liu, Yuhui Xu, Aojun Zhou, Siyuan Huang, Bo Zhang, Junchi Yan, and Hongsheng Li. 2024. [Not all experts are equal: Efficient expert](https://doi.org/10.48550/arXiv.2402.14800) [pruning and skipping for mixture-of-experts large](https://doi.org/10.48550/arXiv.2402.14800) [language models.](https://doi.org/10.48550/arXiv.2402.14800) *CoRR*, abs/2402.14800.
- <span id="page-8-17"></span>Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a suit of armor conduct electricity? a new dataset for open book question answering. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 2381–2391.
- <span id="page-8-7"></span>Alexandre Muzio, Alex Sun, and Churan He. 2024. [SEER-MoE: Sparse Expert Efficiency through Regu](https://doi.org/10.48550/arXiv.2404.05089)[larization for Mixture-of-Experts.](https://doi.org/10.48550/arXiv.2404.05089) *arXiv preprint*.
- <span id="page-8-12"></span>Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Tom Lieberum, Vikrant Varma, János Kramár, Rohin Shah, and Neel Nanda. 2024. [Improving Dic](https://doi.org/10.48550/arXiv.2404.16014)[tionary Learning with Gated Sparse Autoencoders.](https://doi.org/10.48550/arXiv.2404.16014) *arXiv preprint*.
- <span id="page-8-19"></span>Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021. Winogrande: An adversarial winograd schema challenge at scale. *Communications of the ACM*, 64(9):99–106.

<span id="page-9-4"></span>Yuanbo Tang, Zhiyuan Peng, and Yang Li. 2023. [Ex](https://doi.org/10.1145/3589132.3625607)[plainable Trajectory Representation through Dictio](https://doi.org/10.1145/3589132.3625607)[nary Learning.](https://doi.org/10.1145/3589132.3625607) In *Proceedings of the 31st ACM International Conference on Advances in Geographic Information Systems*, SIGSPATIAL '23, pages 1–4, New York, NY, USA. Association for Computing Machinery.

<span id="page-9-6"></span>Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. 2019. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In the Proceedings of ICLR.

<span id="page-9-3"></span>J. Wright, A. Y. Yang, A. Ganesh, S. S. Sastry, and Yi Ma. 2009. [Robust Face Recognition via Sparse](https://doi.org/10.1109/TPAMI.2008.79) [Representation.](https://doi.org/10.1109/TPAMI.2008.79) *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2(31):210–227.

<span id="page-9-0"></span>Fuzhao Xue, Zian Zheng, Yao Fu, Jinjie Ni, Zangwei Zheng, Wangchunshu Zhou, and Yang You. 2024. [Openmoe: An early effort on open mixture-of](https://openreview.net/forum?id=1YDeZU8Lt5)[experts language models.](https://openreview.net/forum?id=1YDeZU8Lt5) In *Forty-first International Conference on Machine Learning*.

<span id="page-9-2"></span>Jianchao Yang, John Wright, Thomas S. Huang, and Yi Ma. 2010. [Image super-resolution via sparse rep](https://doi.org/10.1109/TIP.2010.2050625)[resentation.](https://doi.org/10.1109/TIP.2010.2050625) *IEEE transactions on image processing: a publication of the IEEE Signal Processing Society*, 19(11):2861–2873.

<span id="page-9-5"></span>Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. [HellaSwag: Can a ma](https://doi.org/10.18653/v1/P19-1472)[chine really finish your sentence?](https://doi.org/10.18653/v1/P19-1472) In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pages 4791–4800, Florence, Italy. Association for Computational Linguistics.

<span id="page-9-1"></span>Zeliang Zhang, Xiaodong Liu, Hao Cheng, Chenliang Xu, and Jianfeng Gao. 2024. [Diversifying the Ex](https://doi.org/10.48550/arXiv.2407.09590)[pert Knowledge for Task-Agnostic Pruning in Sparse](https://doi.org/10.48550/arXiv.2407.09590) [Mixture-of-Experts.](https://doi.org/10.48550/arXiv.2407.09590) *arXiv preprint*.

#### Appendix

## A Limitations and Future Work

The entire work operates under the assumption that the allocation result provided by the router is the most optimal. However, this may only reflect one aspect of the model's behavior. By considering both router information and weight data in a more comprehensive way, we could gain a deeper and more complete understanding. Additionally, there has been limited analysis from the perspective of combinatorial learning, which might offer useful insights into the task selection process. Moreover, the labeling of mined patterns has primarily been done manually up until now. In the future, we aim to explore automating this process, such as using large language models to identify and summarize common tokens associated with specific expert collaboration patterns.

#### B Pruning Effect Calculation

For the DeepSeek-MoE-16B model, considering the significant impact of shared experts on the model, we only prune the normal experts during the pruning operation. Through calculations, we estimate the parameter counts of various parts of DeepSeek-MoE-16B as follows: word embeddings 0.2B, attention mechanism 0.4B, gate and shared experts 0.9B, routing network of MoE 14.7B, and output layer 0.2B. Therefore, for this model, our conclusion is that the total parameters after pruning with a pruning ratio of k% can be calculated as:

New Total Parameters = 
$$(16.4 - 14.7 \times k\%)$$
 B (10)

#### C Pruning Experiment Setup

In section 5, following the setup in [\(He et al.,](#page-8-6) [2024\)](#page-8-6), we implement our pruning method on the MMLU [\(Hendrycks et al.,](#page-8-14) [2021\)](#page-8-14) dataset, using 128 samples with an input sequence length of 2,048 tokens. All pruning experiments are conducted on the DeepSeek-MoE-16B model, where only normal experts are pruned, preserving shared experts due to their importance. Model performance is evaluated using the LM-Harness benchmark, which includes a range of tasks: ARC-C [\(Clark et al.,](#page-8-15) [2018\)](#page-8-15), BoolQ [\(Clark et al.,](#page-8-16) [2019\)](#page-8-16), HellaSwag [\(Zellers](#page-9-5) [et al.,](#page-9-5) [2019\)](#page-9-5), MMLU , OBQA [\(Mihaylov et al.,](#page-8-17) [2018\)](#page-8-17), PIQA [\(Bisk et al.,](#page-8-18) [2020\)](#page-8-18), RTE [\(Wang et al.,](#page-9-6)

[2019\)](#page-9-6), and WinoGrande [\(Sakaguchi et al.,](#page-8-19) [2021\)](#page-8-19). The evaluation is carried out using the EleutherAI LM Harness framework [\(Gao et al.,](#page-8-20) [2023\)](#page-8-20), and we report normalized zero-shot accuracy for each task.