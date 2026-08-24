# 3 Method

In this section, we first introduce the bidirectional importance score of each local thinking step, which is employed in the subsequent A\* search process for thought compression.

#### <span id="page-2-1"></span>3.1 Step-Level Bidirectional Importance Score

As mentioned in Section [2,](#page-2-0) it is prohibitively costly to explore all potential subsets of the long CoT. We therefore propose an importance score to improve sampling efficiency for thought selection. A related approach is LongLLMLingua [\(Jiang et al., 2024\)](#page-10-8), which employs conditional probabilities to estimate the question-aware importance of each token in a long document. In complex tasks that require long CoTs, the importance of an intermediate thought is not only related to the question but also to the final solution. Relevance to the solution may provide important information for identifying useful thoughts. We therefore propose a bidirectional importance score that considers both the question and the final solution when estimating the importance of each thinking step.

Specifically, we assess importance at two levels, including the attention and model levels. At the attention level, we use attention weights to represent the importance of x for y:

$$ATTN(\mathbf{y}|\mathbf{x}) = \frac{1}{H|\mathbf{y}||\mathbf{x}|} \sum_{h=1}^{H} \sum_{j=1}^{|\mathbf{y}|} \sum_{i=1}^{|\mathbf{x}|} a_h(y_j, x_i), \qquad (2)$$

where  $a_h(y_j, x_i)$  denotes the attention score of the query  $y_j$  to the key  $x_i$  for the h-th head. H represents the number of heads. A higher importance score indicates a more significant effect of x on y. At the model level, we use the negative log-likelihood to assess the importance score of x for y:

$$NLL(\mathbf{y}|\mathbf{x}) = -\frac{1}{|\mathbf{y}|} \sum_{j=1}^{|\mathbf{y}|} \log P(y_j|\mathbf{x}, \mathbf{y}_{< j}).$$
(3)

To assess the contribution of each thinking step  $\mathbf{t}^{(n)}$  in the overall thought t, we analyze its influence on both the question  $\mathbf{q}$  and the solution  $\mathbf{s}$ . Each thought is concatenated with the question and the solution, forming the sequences  $\langle \mathbf{t}^{(n)}, \mathbf{q} \rangle$  and  $\langle \mathbf{t}^{(n)}, \mathbf{s} \rangle$ , respectively. Subsequently, we utilize a compact language model, specifically GPT-2², to quantify attention scores and NLL values. The selection of a smaller model enhances the efficiency of this importance estimation procedure.

Specifically, the bidirectional importance score of  $t^{(n)}$  is denoted as

<span id="page-3-3"></span>
$$BIS\left(\mathbf{t}^{(n)}\right) = \frac{(1-\alpha)\operatorname{ATTN}\left(\mathbf{q}|\mathbf{t}^{(n)}\right) + \alpha\operatorname{ATTN}\left(\mathbf{s}|\mathbf{t}^{(n)}\right)}{(1-\alpha)\operatorname{NLL}\left(\mathbf{q}|\mathbf{t}^{(n)}\right) + \alpha\operatorname{NLL}\left(\mathbf{s}|\mathbf{t}^{(n)}\right)},\tag{4}$$

where  $\alpha$  is a hyper-parameter to control the relative weighting of relevance to the question versus the solution. Figure 3 presents the distribution of BIS for a sequence of thought steps, where the thought steps are divided by "\n\n". The figure illustrates that not all steps hold high importance concerning both the question and the solution. Only a limited subset offers significant contributions. Subsequently, these BIS values will be leveraged in conjunction with the A\* search algorithm to determine the pruned thought t'.

> **[图片提取文字 (无描述)]:**
> SIS 50 53 Thinking Steps
![](_page_3_Figure_8.jpeg)

<span id="page-3-1"></span>Figure 3: Distribution of BIS values for individual thinking steps in Long CoT.

#### <span id="page-3-2"></span>3.2 Path-Level A\* Search

Our objective is to extract an alternative thought trajectory  $\mathbf{t}'$  from an original trajectory  $\mathbf{t}$ . For a trajectory  $\mathbf{t}$  with N thinking steps, the  $2^N$  candidate trajectories render exhaustive exploration computationally intractable for extended  $\mathbf{t}$ . We therefore employ the  $A^*$  search algorithm for efficient traversal of this space. Following an initialization phase, the algorithm iteratively conducts verification and exploration. In the k-th iteration, a verification model  $\mathcal V$  ascertains if the current candidate path  $\mathbf{t}'_k$  can lead to the correct solution. During exploration, each path  $\mathbf{t}'_k$  is evaluated using cost functions. The algorithm is detailed in the following subsections.

#### 3.2.1 Overview

To enhance search efficiency, the thinking steps within the original long CoT  $\mathbf{t} = \{\mathbf{t}^{(1)}, \mathbf{t}^{(2)}, \dots, \mathbf{t}^{(N)}\}$  are first sorted in descending order based on their BIS values, yielding

<span id="page-3-0"></span><sup>&</sup>lt;sup>2</sup>https://huggingface.co/openai-community/gpt2.

 $\mathbf{t}_{\mathrm{sort}} = \{\mathbf{t}^{(n_1)}, \mathbf{t}^{(n_2)}, \ldots, \mathbf{t}^{(n_N)}\}$ . Subsequently, the A\* search algorithm iteratively expands a search tree, denoted as  $\mathcal{T}$ , according to defined cost functions. In this tree, each node corresponds to a span centered on a specific thinking step, encompassing its immediately preceding and succeeding steps. Formally, a node associated with the thinking step  $\mathbf{t}^{(n)}$  is represented as  $\mathbf{r}^{(n)} = \langle \mathbf{t}^{(n-1)}, \mathbf{t}^{(n)}, \mathbf{t}^{(n+1)} \rangle$ . This approach aims to mitigate the adverse effects of fragmented information that can arise from thought segmentation.

**Initialization** By leveraging the bidirectional importance estimation mechanism detailed in Section 3.1, A\*-Thought identifies a logical starting point within the potentially redundant thinking trajectory, thereby enhancing both efficiency and performance. Initially, a thought queue, denoted as  $\mathcal{Q}$ , is constructed using the sorted thought sequence  $\mathbf{t}_{sort}$ . Subsequently, the first thought is dequeued from  $\mathcal{Q}$  to form the root node of the search tree  $\mathcal{T}$ . This selection ensures the implementation of a best-first sampling strategy throughout the subsequent search iterations.

**Verification** To assess the efficacy of the current thinking path, denoted as  $\mathbf{t}_k'$ , which encompasses the thinking spans from the root node to the current active leaf node within the search tree  $\mathcal{T}$ , a validation model  $\mathcal{V}$  is introduced. This model is employed to determine whether the current path  $\mathbf{t}_k'$  successfully leads to the solution s:

$$\mathcal{V}\left(\mathbf{q} + \mathbf{t}_{k}^{\prime}\right) \begin{cases} \neq \mathbf{s}, & \text{expand } \mathcal{T} \\ = \mathbf{s}, & \text{return } \mathbf{t}^{\prime} = \mathbf{t}_{k}^{\prime} \end{cases}$$
 (5)

It has been observed that verification tends to be ineffective for extremely short thought sequences, thereby offering limited guidance for the search process. Consequently, a lower boundary, denoted as  $k_{\min}$ , is established for verification. Verification is performed exclusively when the depth of the search tree, k, satisfies the condition  $k \ge k_{\min}$ .

**Exploration** If the current active leaf node does not pass verification, the first W thoughts are dequeued from  $\mathcal Q$  to function as next-level leaf nodes, denoted as  $\{\mathbf r_1,\dots,\mathbf r_W\}$ . Each of these nodes is then appended to the current thinking path  $\mathbf t_k'$  to construct a set of candidate thinking paths. we assign a cost function  $f(\cdot)$  to each candidate thinking path, where

$$f(\mathbf{t}_{k}' + \mathbf{r}_{w}) = g(\mathbf{t}_{k}' + \mathbf{r}_{w}) + h(\mathbf{t}_{k}' + \mathbf{r}_{w}). \tag{6}$$

The design of our cost function,  $f(\cdot)$ , is informed by the A\* search algorithm. Specifically,  $g(\cdot)$  denotes the cumulative cost incurred from the root node to the current node. Concurrently,  $h(\cdot)$  functions as a heuristic, providing an estimate of the prospective cost from the current node to the target solution. We select the node that with the minimal cost as the new active leaf node:

$$\hat{\mathbf{r}}_w = \operatorname*{argmin}_{w \in \{1, \dots, W\}} f(\mathbf{t}'_k + \mathbf{r}_w). \tag{7}$$

The newly formed active thinking path,  $\mathbf{t}'_{k+1} = \langle \mathbf{t}'_k, \hat{\mathbf{r}}_w \rangle$ , subsequently proceeds to the next iteration of the process. Figure 2 shows an example for the search process. To prevent an excessively deep search tree, an upper bound,  $k_{\max}$ , is imposed on its depth. The search process is terminated, and  $\mathbf{t}$  is directly returned when the current depth, k, reaches or exceeds this limit, i.e.,  $k \geq k_{\max}$ . The resulting compact responses can be leveraged to distill LRMs, fostering enhanced thinking efficiency.

#### 3.2.2 Design of Cost Functions

To identify an effective and compact thought, denoted as  $\mathbf{t}'$ , the quality of each intermediate thought  $\mathbf{t}'_k$  is assessed from two perspectives: (1) the quality of the current intermediate thought, which is quantified by  $g(\cdot)$ ; and (2) the estimated future cost associated with extending the current intermediate thought  $\mathbf{t}'_k$  to the final thought sequence  $\mathbf{t}'$ .

**Current Cost Function** The function  $g(\cdot)$  measures the quality of the current intermediate thought  $t'_k$ . A verification model leveraging its reasoning capabilities is employed to estimate this quality:

<span id="page-4-0"></span>
$$g(\mathbf{t}_{k}') = -\frac{\beta}{|\mathbf{t}_{k}'|} \log P_{\mathcal{V}}(\mathbf{t}_{k}'|\mathbf{q}), \qquad (8)$$

where  $\beta$  is the weight controlling the effect of the current cost.

**Future Cost Function** The function h(n) estimates the cost from the current node to the goal, thereby influencing the efficiency of the search path. A higher estimated future cost suggests that a more extensive sequence of future thoughts will be required to reach the final solution from the current state. To quantify this, we employ the conditional self-information of the correct solution s, given the current thought  $\mathbf{t}'_{L}$  and the input question  $\mathbf{q}$ . Formally, it can be represented as:

$$h(\mathbf{t}_k') = \mathcal{I}(\mathbf{s}|\mathbf{q}, \mathbf{t}_k'). \tag{9}$$

Larger values of the conditional self-information  $\mathcal{I}(\cdot)$  indicate a lower likelihood of generating the solution s. We quantify  $\mathcal{I}(\cdot)$  as follows:

$$\mathcal{I}(\mathbf{s}|\mathbf{q}, \mathbf{t}'_k) = -\frac{1}{|\mathbf{s}|} \log P_{\mathcal{V}}(\mathbf{s}|\mathbf{q}, \mathbf{t}'_k). \tag{10}$$

In particular, A\*-Thought enhances reasoning efficiency by compressing the thought trajectory. This is achieved through the systematic reduction of redundant steps, thereby streamlining the path from the initial query to the final solution. Such targeted compression significantly improves the performance of LRMs, enabling them to deliver robust outcomes across diverse budgets.

