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

