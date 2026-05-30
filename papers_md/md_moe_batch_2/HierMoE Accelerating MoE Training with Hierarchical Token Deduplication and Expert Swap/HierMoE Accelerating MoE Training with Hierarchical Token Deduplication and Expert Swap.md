# HierMoE: Accelerating MoE Training with Hierarchical Token Deduplication and Expert Swap

Wenxiang Lin<sup>†</sup>, Xinglin Pan<sup>‡</sup>, Lin Zhang<sup>§</sup>, Shaohuai Shi<sup>†\*</sup>, Xuan Wang<sup>†</sup>, Xiaowen Chu<sup>‡§</sup>

<sup>†</sup>School of Computer Science and Technology, Harbin Institute of Technology, Shenzhen

<sup>‡</sup>Data Science and Analytics Thrust, The Hong Kong University of Science and Technology (Guangzhou)

§Department of Computer Science and Engineering, The Hong Kong University of Science and Technology wenxianglin@stu.hit.edu.cn, xpan413@connect.hkust-gz.edu.cn, lzhangbv@connect.ust.hk

shaohuais@hit.edu.cn, wangxuan@cs.hitsz.edu.cn, xwchu@ust.hk

Abstract—The sparsely activated mixture-of-experts (MoE) transformer has become a common architecture for large language models (LLMs) due to its sparsity, which requires fewer computational demands while easily scaling the model size. In MoE models, each MoE layer requires to dynamically choose tokens to activate particular experts for computation while the activated experts may not be located in the same device or GPU as the token. However, this leads to substantial communication and load imbalances across all GPUs, which obstructs the scalability of distributed systems within a GPU cluster. To this end, we introduce HierMoE to accelerate the training of MoE models by two topology-aware techniques: 1) token deduplication to reduce the communication traffic, and 2) expert swap to balance the workloads among all GPUs. To enable the above two proposed approaches to be more general, we build theoretical models aimed at achieving the best token duplication and expert swap strategy under different model configurations and hardware environments. We implement our prototype HierMoE system atop Megatron-LM and conduct experiments on a 32-GPU cluster with DeepSeek-V3 and Qwen3-30B-A3B models. Experimental results show that our HierMoE achieves  $1.55\times$  to  $3.32\times$  faster communication and delivers  $1.18\times$  to  $1.27\times$  faster end-to-end training compared to state-of-the-art MoE training systems, Tutel-2DH, SmartMoE, and Megatron-LM.

Index Terms—Distributed Deep Learning; Mixture-of-Experts; Expert Parallelism; Token Deduplication; Expert Swap.

#### I. INTRODUCTION

The mixture-of-experts (MoE) architecture with sparse activation has gained significant research interest in large language models (LLMs) [1]-[7]. It provides an effective solution for model size scaling, where the computational requirement grows sub-linearly with increasing number of parameters. The MoE architecture incorporates the MoE layer, which comprises multiple feed-forward networks (FFNs), known as experts, substituting the dense feed-forward layer while activating only a subset of these experts for each input token [2,7]. A trainable routing function, generally a small neural network utilizing a softmax mechanism, is employed to dynamically select which experts should be trained for each input token [2]. This architecture allows the model size to expand to nearly Etimes (where E represents the number of experts per MoE layer) that of a standard dense model, yet the computational demand remains comparatively stable.

\*Corresponding author.

<span id="page-0-0"></span>![](_page_0_Picture_8.jpeg)

(a) Two-level hierarchy

(b) Four-level hierarchy

Fig. 1: Two commonly used hierarchical topologies.

However, training MoE LLMs typically requires expert parallelism (EP) [1.8] to place different experts on different GPUs since a single GPU has limited memory to hold all experts. Due to the dynamic nature of dispatching input tokens to experts that are located in different GPUs, EP introduces significant communications, which are implemented by the AlltoAll collective, easily limiting the scalability of the distributed training system. Recent research [2,7]-[12] suggests that communication overheads of the AlltoAll operation constitute 30-60% of the overall training time in GPU/TPU clusters. Some studies are trying to address the communication problem through 1) algorithmic optimization [13]-[17] by using better routing functions to balance the workload of experts, and 2) system-level optimization by designing more communication-efficient AlltoAll collective algorithms [8,18]— [20] and adaptive task scheduling to overlap communication tasks and computation tasks [8,10]-[12,21]-[31]. Since the process of algorithmic optimization can significantly impact model convergence, this study concentrates on system-level optimization that does not compromise model accuracy.

Specifically, AlltoAll requires each GPU to exchange data with all the other GPUs, so its performance is highly affected by the network topology of GPUs (i.e., the hierarchical connection between GPUs as shown in Fig. 1). That is, a low-bandwidth link may significantly slow down the overall communication performance. For example, in the four-level hierarchical topology as shown in Fig. 1b (Inter-node

through InfiniBand, Intra-node through NVLink, PCIe, and QPI), InfiniBand or QPI would easily limit the communication performance of AlltoAll. In larger-scale clusters, GPU nodes should be connected across switches, which introduces higher levels of the topology [32]–[35]. Existing related optimizations include 1) hierarchical AlltoAll algorithms like two dimensional hierarchical (2DH) AlltoAll in Tutel (Tutel-2DH) [8], PipeA2A in ScheMoE [11], and dedicated kernels for Nvidia Hopper architecture in DeepSeek-V3 [7] to better utilize Intra-node and Inter-node network bandwidths, and 2) expert placement algorithms to balance the communication workloads of different GPUs like SmartMoE [23]. These methods underestimate the impacts of the hierarchical structure of GPU connection and have not explored the full hierarchical structure to optimize AlltoAll communication, thus achieving suboptimal training performance.

To this end, we propose HierMoE to fully utilize the hierarchical structure to optimize token distribution and expert migration among GPUs, achieving minimal AlltoAll communication time in MoE model training. HierMoE incorporates three innovative strategies: 1) conducting theoretical research on the links between hierarchical dimensions and the redundant transfer challenge to design a hierarchical token deduplication AlltoAll algorithm aimed at decreasing data transfer redundancy among varying hierarchical dimensions, 2) designing a hierarchical expert swap mechanism to balance the communication workloads of different GPUs aimed at further improving the AlltoAll communication efficiency, and 3) devising theoretical frameworks that render the token deduplication and expert swap strategy broadly applicable and practical for varying models. We implement our HierMoE atop the widely-used LLM training system Megatron-LM<sup>1</sup>, and conduct extensive experiments on a 32-GPU cluster using representative real-world MoE models, including DeepSeek-V3 [7] and Qwen3-30B-A3B [4]. Experimental results show that HierMoE improves the AlltoAll communication efficiency by  $1.55\times$  to  $3.32\times$  and achieves  $1.18\times$  to  $1.27\times$  faster endto-end training over the state-of-the-art MoE training systems Tutel-2DH [8], SmartMoE [23] and Megatron-LM.

TABLE I: Notations.

<span id="page-1-1"></span>

| Notation       | tion Description                                          |  |  |  |  |
|----------------|-----------------------------------------------------------|--|--|--|--|
| $\overline{M}$ | Embedding dimension of a token                            |  |  |  |  |
| K              | Number of experts selected for each token                 |  |  |  |  |
| G              | Number of workers (or GPUs) in the cluster                |  |  |  |  |
| D              | Number of hierarchical dimensions                         |  |  |  |  |
| U[i]           | Number of experts group when                              |  |  |  |  |
|                | performing Inter-level-i AlltoAll                         |  |  |  |  |
| E              | Total number of experts                                   |  |  |  |  |
| $t_d$          | Time of d-dimensional deduplication hierarchical AlltoAll |  |  |  |  |

# II. PRELIMINARIES AND MOTIVATIONS

In this section, we present some preliminaries of MoE training and our motivations. For convenience, we provide a summary of the commonly used notation in Table I.

<span id="page-1-2"></span>![](_page_1_Figure_7.jpeg)

(b) Expert parallelism

Fig. 2: A typical structure of the MoE model.

# A. MoE Layer

Generally, an MoE model follows the architecture of a dense model by replacing its dense feed-forward layers with MoE layers as shown in Fig. 2a. An MoE layer consists of two primary elements: a gating function and a collection of trainable feed-forward neural networks (FFNs) called experts. At each training iteration, the input tokens are distributed to selected experts according to the gating function. The gating function also uses a small trainable FFN followed by a softmax layer and top-K selection to determine which experts should process which tokens.

# B. Expert Parallelism

Since a single GPU has limited memory to accommodate all experts, the expert parallelism (EP) [1,2,8] is required to place different experts at each layer to different GPUs during training. Let E indicate the number of experts at each MoE layer and G the number of GPUs in the cluster. EP would place E/G experts every MoE layer on each GPU. Together with data parallelism (DP) [36,37], which is a de facto training paradigm in distributed training of LLMs, the input data in each device is different. Therefore, the token distribution generated by the gating function requires an AlltoAll collective method to dispatch tokens to particular experts (referred to as AlltoAll Dispatch), illustrated as "Dispatch" shown in Fig. 2b. It ensures that data is routed to the correct experts for processing. After dispatch, the tokens undergo processing by their designated experts. The results from the experts then need to be processed by subsequent layers (e.g., Attention) of the MoE layer, requiring another AlltoAll operation to merge the expert outputs (termed as AlltoAll Combine) illustrated as "Combine" shown in Fig. 2b. The two AlltoAll operations at every MoE layer generally introduce significant communication overheads, which limit the scaling efficiency of training systems. Modern systems like Tutel [8] and DeepSeed-MoE [18] try to use the two-dimensional hierarchical AlltoAll (2DH-AlltoAll) algorithm [8,18] that is a dedicated design for the hierarchical topology of GPUs.

<span id="page-1-0"></span><sup>&</sup>lt;sup>1</sup>https://github.com/NVIDIA/Megatron-LM/

<span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

(a) Tokens assigned to each expert (b) Tokens assigned to each group

![](_page_2_Figure_2.jpeg)

(c) Tokens assigned to each group after deduplication (d) Tokens assigned to each group after expert swap

Fig. 3: The illustration of tokens assignment with different configurations. Different colors represent different tokens.

<span id="page-2-1"></span>TABLE II: Token duplication rates with different K and R.

|    |     | K   |     |     |  |  |  |  |  |
|----|-----|-----|-----|-----|--|--|--|--|--|
| R  | 2   | 4   | 6   | 8   |  |  |  |  |  |
| 32 | 2%  | 4%  | 7%  | 9%  |  |  |  |  |  |
| 16 | 3%  | 9%  | 14% | 18% |  |  |  |  |  |
| 8  | 6%  | 17% | 27% | 34% |  |  |  |  |  |
| 4  | 12% | 32% | 46% | 55% |  |  |  |  |  |

# *C. Motivations*

*1) Duplicate tokens with the hierarchical topology:* Since each GPU would hold E/G experts per MoE layer when the number of experts is larger than the number of GPUs (i.e., E/G > 1), tokens are required to be *redundantly* transmitted to multiple selected experts that are located on the same GPU to exploit the AlltoAll collective. As illustrated in Fig. [3a,](#page-2-0) each expert is assigned particular tokens, and a single token requires multiple experts (i.e., K in the top-K selection). Assume that each GPU (or group) holds two experts as shown in Fig [3b.](#page-2-0) Every group would have duplicate tokens, which results in redundant communication in the AlltoAll operation. *Thus, eliminating this duplication can reduce the communication traffic as shown in Fig. [3c](#page-2-0)*.

Moreover, the duplication rate is highly affected by the number of groups (say R) and the number of selected experts (i.e., K) per token. We conduct preliminary experiments with different R and K to measure the duplication rate at each group as shown in Table [II.](#page-2-1) The results indicate that lower R (The hierarchical topology can divide experts into different groups.) and higher K (which is very common in modern MoE models like DeepSeek-V3, Qwen-MoE, etc.) would result in a higher duplication rate. *Thus, how to eliminate the duplicated tokens by considering* K*,* R*, and the GPU topology becomes more challenging.*

*2) Unbalanced routing workloads with the hierarchical topology:* Since the selected experts for each token are determined by the routing function, it is easy to cause imbalanced workloads for each expert, which results in increased commu-

<span id="page-2-2"></span>![](_page_2_Figure_11.jpeg)

Fig. 4: Four types of hierarchical AlltoAll with different dimensions. The example has two nodes with eight GPUs per

node. We use "..." to omit some GPUs and nodes.

nication traffic [\[2,](#page-9-2)[7,](#page-9-1)[23\]](#page-9-12). Existing solutions like FlexMoE [\[38\]](#page-10-5) and SmartMoE [\[23\]](#page-9-12) dynamically adjust expert placement during training to balance token distribution across GPUs, *but they neither account for token deduplication nor adapt to the hierarchical topology of GPUs.* If the token duplications have been overlooked, simply swapping experts to balance the workload could result in a higher communication overhead. For example, as shown in Fig. [3d,](#page-2-0) we swap expert 1 and expert 3 such that the workload of each group is more balanced, but its communication traffic becomes higher than that of Fig. [3c.](#page-2-0)

*Therefore, it requires a new expert swap strategy taking into account token deduplication and hierarchical bandwidth constraints to achieve higher training performance.*

# III. HIERARCHICAL TOKEN DEDUPLICATION

# *A. Hierarchical Deduplication AlltoAll*

To better utilize the hierarchical topology for token transferring in the MoE layer, we design a multi-dimensional AlltoAll algorithm with token deduplication, called HierD-AlltoAll. To make our design general to existing AlltoAll algorithms, existing standard AlltoAll and 2DH-AlltoAll algorithms can be seen as particular cases. Specifically, for the standard AlltoAll algorithm, it can be denoted as a onedimensional algorithm as it does not consider any topology, as shown in Fig. [4a.](#page-2-2) Similarly, the 2DH-AlltoAll algorithm is a two-dimensional algorithm that is dedicated for twodimensional hierarchical topology as shown in Fig. [4b.](#page-2-2) For

<span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Fig. 5: An illustration of token deduplication in hierarchical AlltoAll with 2 nodes and 16 GPUs.

<span id="page-3-2"></span>![](_page_3_Figure_2.jpeg)

Fig. 6: An illustration of experts and GPUs index for Inter-Node/Intra-Node AlltoAll.

intricate topologies with more than two layers of hierarchy, we arrange the GPUs into groups, ensuring that the number of groups aligns with the hierarchy levels. For example, for the four levels of hierarchy, like the common case where each node has NVLink, PCIe, and QPI connections as shown in Fig. 1b, we organize GPUs into four groups and do a four-dimensional AlltoAll. As shown in Fig. 4d, the first level (Inter-level-1) simultaneously invokes 8 Inter-node AlltoAll operations, each pair of GPUs communicates with each other through IB. The second level (Inter-level-2) simultaneously invokes 8 Inter-QPI AlltoAll, each of which only has two GPUs. Similarly, the third level (Inter-level-3) performs Inter-NVLink AlltoAll, and the fourth level (Intra-level-3) invokes Intra-NVLink AlltoAll to complete the functionality of the original AlltoAll. In general, a (d)-dimensional hierarchical AlltoAll is composed of Inter-level-1, Inter-level-2, up to Inter-level-(d-1) AlltoAll followed by an Intra-level-(d-1) AlltoAll. As for an (d+1)-dimensional AlltoAll algorithm, we further split the Intra-level-(d-1) AlltoAll into Inter-level-(d) and Intra-level-(d) AlltoAll.

According to the hierarchical AlltoAll algorithm, we design token deduplication strategy for minimizing the overall time of communication and propose Hier-AlltoAll. Let *D* denote the number of dimensions for the hierarchical structure. As the high-dimensional hierarchical topology can also perform low-dimensional hierarchical AlltoAll, we refer to the dedupli-

cation version of *D* kinds of dimensional hierarchical AlltoAll shown in Fig. 4 as HD1-AlltoAll, HD2-AlltoAll, and up to HD*D*-AlltoAll, respectively.

We take HD1-AlltoAll and HD2-AlltoAll as an example to demonstrate the initiative effect of the hierarchical deduplication. As illustrated in Fig. 5, the HD2-AlltoAll (shown on the right) with deduplicated tokens in both dimensions requires only two tokens transferred to another node using Inter-node communication. In contrast, the HD1-AlltoAll (depicted on the left) necessitates the dispatch of four tokens. The Internode AlltoAll from HD2-AlltoAll redistributes experts from 16 groups by GPUs into 2 groups by nodes, leading to increased duplicated tokens in each group. By removing these duplications, we improve Inter-node communication traffic over IB, which has low bandwidth. But HD2-AlltoAll requires two more tokens transferred to other GPUs using Intra-node communication. Similarly, using HD3-AlltoAll can lower the communication traffic of Inter-QPI communication through QPI compared to HD2-AlltoAll, but increase the communication volume of Intra-QPI communication. Notably, the experts group number of Inter-QPI AlltoAll is bigger than that of Inter-Node AlltoAll, as it further splits the experts group by QPI. According to the Table. II, high top-K selection and low experts group results in high duplication rates and vice versa. Token deduplication benefits might diminish when employing HD3-AlltoAll rather than HD2-AlltoAll. The same applies to HD4-AlltoAll.

Therefore, the dimension of the hierarchical AlltoAll is not necessarily larger. We need to formulate the performance model of different dimensional AlltoAll to determine the optimal dimension, ensuring communicational overhead reduction.

#### B. Performance Model

We model the time cost of the standard AlltoAll communication (also HD1-AlltoAll) via linear models [22] as follows (will verify in §V-B):

<span id="page-3-3"></span>
$$t_1 = \alpha_{a2a} + n_{a2a} \cdot \beta_{a2a},\tag{1}$$

where  $n_{a2a}$  represents the volume of the communication message,  $\alpha_{a2a}$  denotes the startup time and  $\beta_{a2a}$  represents the time per byte transmitted.  $\alpha$  and  $\beta$  parameters of Inter/Intralevel-(i) AlltoAll are represented as  $\alpha_{a2a}^{\text{Inter/Intra}(i)}$  and  $\beta_{a2a}^{\text{Inter/Intra}(i)}$  respectively.

Unlike  $\beta_{a2a}$  and  $\alpha_{a2a}$  associated with the cluster and determined during initialization,  $n_{a2a}$  is related to the dynamic routing results of the MoE layer. We further model  $n_{a2a}$  as the product of the number of GPUs in the AlltoAll operation and the number of tokens sent to each GPU.

<span id="page-3-1"></span>
$$n_{a2a} = G \cdot \max(p) \cdot M \cdot v, \tag{2}$$

where G denotes the number of GPUs in the cluster,  $p \in \mathbb{R}^G$  represents the duplicate-free number of tokens assigned to each expert group (the number of groups is the same as that of GPUs in HD1-AlltoAll), M denotes the embedding dimension size of each token and v denotes the bytes of one embedding

dimension. To ensure that all tokens are dispatched, we use max(p) to represent the number of tokens sent to each GPU.

For HDd-AlltoAll where d > 1, as shown in Fig. 4, a (d)-dimensional hierarchical AlltoAll is composed of Inter-level-1, Inter-level-2, up to Inter-level-(d-1) AlltoAll followed by an Intra-level-(d-1) AlltoAll. We thus formulate the time cost of (d)-dimensional AlltoAll as

<span id="page-4-3"></span>
$$t_{d} = \sum_{i=1}^{d-1} (n_{a2a}^{\text{Inter}(i)} \cdot \beta_{a2a}^{\text{Inter}(i)} + \alpha_{a2a}^{\text{Inter}(i)}) + n_{a2a}^{\text{Intra}(d-1)} \cdot \beta_{a2a}^{\text{Intra}(d-1)} + \alpha_{a2a}^{\text{Intra}(d-1)},$$
(3)

where  $1 < d \leq D$ . Similar to Eq. (2), we also use the product of the number of GPUs in an Inter-level-(i) AlltoAll and the number of tokens sent to each GPU to represent  $n_{a2a}^{\mathrm{Inter}(i)}$ . Notably, input tokens of Inter-level-(i) AlltoAll and the group number of experts U[i] are different from each other. So we distinguish the number of tokens assigned to each expert group for Inter-level-(i) AlltoAll as  $p_{a2a}^{\mathrm{Inter}(i)} \in \mathbb{R}^{U[i]}$ . Similarly, we use  $\max(p_{a2a}^{\mathrm{Inter}(i)})$  to represent the number of tokens sent to each GPU during Inter-level-(i) AlltoAll.

Notably, U[i] in  $U \in \mathbb{R}^D$  denotes the group number of experts when performing Inter-level-(i) AlltoAll. Taking the topology shown in Fig. 1b as the example, Inter-level-1 (Inter-Node) AlltoAll divides experts into four groups by nodes so U[1]=4 (also illustrated in Fig. 6). Inter-level-2 (Inter-QPI) AlltoAll further splits experts in each node into two parts by QPI so U[2]=8. Inter-level-3 (Inter-NVLink) AlltoAll divides experts in each QPI group into two parts so U[3]=16. Specially, we set U[0]=1.

Additionally, through our numerical analysis, we find that  $\frac{U[i]}{U[i-1]}$  can signify the number of GPUs involved in an Interlevel-(i) AlltoAll while  $\frac{G}{U[d-1]}$  can represent the GPUs count participating in an Intra-level-(d-1) AlltoAll. Interestingly, the number of GPUs used in an Inter/Intra-level AlltoAll differs from that of expert groups. Because both Inter-level-(i+1) and Intra-level-(i+1) AlltoAll take place within the GPUs group of an Intra-level-(i) AlltoAll as they are derived from it. For instance, Inter/Intra-QPI AlltoAll occurs within a node without interfacing with GPUs from other nodes. Consequently, Interlevel-(i) AlltoAll will first divide experts into U[i] groups to count duplicate-free tokens and then select corresponding U[i] groups to dispatch tokens. Then we can derive that

<span id="page-4-0"></span>
$$n_{a2a}^{\text{Inter}(i)} = \frac{U[i]}{U[i-1]} \cdot \max(p_{a2a}^{\text{Inter}(i)}) \cdot M \cdot v. \tag{4}$$

Similarly, we use  $\max(p_{a2a}^{\operatorname{Intra}(d-1)})$  to represent the number of tokens sent to each GPU during Intra-level-(d-1) AlltoAll. Specially, the expert group count is the same as the number of GPUs for all Intra-level AlltoAll. Intra-level-(d-1) AlltoAll will first divide experts into G groups and then select corresponding  $\frac{G}{U[d-1]}$  groups to dispatch, given that the index of experts in GPUs selected by Intra-level AlltoAll is always contiguous as shown in Fig. 6. So we have

<span id="page-4-1"></span>
$$n_{a2a}^{\text{Intra}(d\text{-}1)} = \frac{G}{U[d-1]} \cdot \max(p_{a2a}^{\text{Intra}(d\text{-}1)}) \cdot M \cdot v. \tag{5}$$

#### C. Problem Formulation and Solution

Based on the above performance models for different dimensional hierarchical deduplication AlltoAll, we can derive the problem of determining the optimal dimension  $d^*$  as

<span id="page-4-2"></span>
$$d^* = \begin{cases} 1, & t_1 < \min_{1 < d \le D} (t_d) \\ \arg\min_{1 < d \le D} (t_d), & else \end{cases}$$
 (6)

All parameters in Eq. (2), Eq. (4), and Eq. (5) are cluster-related and can be pre-initialized, except p,  $p_{a2a}^{\mathrm{Intra}(i)}$ , and  $p_{a2a}^{\mathrm{Intra}(d-1)}$ , which need to be calculate by the MoE layer's routing results. To formulate the relationship, we use  $p_{a2a}^{(l,g)} \in \mathbb{R}^g$ , which denotes the duplicate-free number of tokens assigned to g expert groups of Inter-level-(l) or Intra-level-(l-1) AlltoAll, to generally represent p (i.e.,  $p_{a2a}^{(1,G)}$ ),  $p_{a2a}^{\mathrm{Intra}(i)}$  (i.e.,  $p_{a2a}^{(i,U[i])}$ ) and  $p_{a2a}^{\mathrm{Intra}(d-1)}$  (i.e.,  $p_{a2a}^{(d,G)}$ ). Notably, as shown in Fig. 4, input tokens of Inter-level-(d) AlltoAll are the same as that of Intra-level-(d-1) AlltoAll so we can use  $p_{a2a}^{(d,G)}$  to represent  $p_{a2a}^{\mathrm{Intra}(d-1)}$ . Let  $\mathcal{I}_{route}^{(l,E)} \in \mathbb{R}^{T'[l] \times E}$  represent the routing result mask for input tokens of Inter-level-(l) AlltoAll with the datatype of boolean, T'[l] being the number of input tokens of Inter-level-(l) AlltoAll. And  $\mathcal{I}_{route}^{(l,E)}[i,j]$  represents wether the i-th token select j-th expert. Then we can formulate  $p_{a2a}^{(l,g)}[j]$  by

$$\mathcal{I}_{route}^{(l,g)}[i,j] = \bigvee_{j_{1}=(j-1)\frac{E}{g}+1} \mathcal{I}_{route}^{(l,E)}[i,j_{1}], 
p_{a2a}^{(l,g)}[j] = \sum_{i} \mathbb{I}(\mathcal{I}_{route}^{(l,g)})[i,j],$$
(7)

where  $\bigvee$  denotes the bitwise OR operation, allowing for the elimination of deduplication tokens, and  $\mathcal{I}_{route}^{(l,g)}[i,j]$  represents whether the i-th token selects the j-th expert group. Denote T as the total number of tokens for the MoE layer, the time complexity to calculate p,  $p_{a2a}^{\mathrm{Inter}(i)}$  and  $p_{a2a}^{\mathrm{Intra}(d-1)}$  is  $O(D \cdot T \cdot K)$ . Then, we proceed by examining each possible value of d to determine the optimal dimension. HierD-AlltoAll refers to hierarchical deduplication AlltoAll with this optimal  $d^*$ .

#### D. Algorithm

According to the above solution, we derive the algorithm to determine the optimal dimension of HierD-AlltoAll for any given MoE layer as shown in Algorithm 1. The input including the embedding size of the token M, the routing result mask  $\mathcal{I}_{route}^{(1,E)}$ , the number of GPUs G, the number of experts E, the number of dimensions for the hierarchical structure in the cluster D, the expert number of group U for each Inter-level AlltoAll and cluster parameters  $\beta_{a2a}, \alpha_{a2a}, \beta_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname{Inter}(l)}, \alpha_{a2a}^{\operatorname$ 

# <span id="page-5-0"></span>Algorithm 1 Find the Optimal Dimension for HierD-AlltoAll

Input: 
$$\mathcal{I}_{route}^{(1,E)}, U, M, G, E, D, \beta_{a2a}, \alpha_{a2a},$$
1:  $\beta_{a2a}^{\text{Inter}(l)}, \alpha_{a2a}^{\text{Inter}(l)}, \beta_{a2a}^{\text{Inter}(l)}, \alpha_{a2a}^{\text{Inter}(l)}, \alpha_{a2a}^{\text{Inter}(l)}, 0 < l < D$ 
Output: Optimal dimension  $d^*$ 
2:  $m \leftarrow E/G$ 
3:  $\mathcal{I}_{route}^{(1,G)}[i,j] \leftarrow \bigvee_{j_1=(j-1)m+1}^{j \cdot m} \mathcal{I}_{route}^{(1,E)}[i,j_1], \quad 1 \leq j \leq G$ 
4:  $p[j] \leftarrow \sum_i \mathbb{I}(\mathcal{I}_{route}^{(1,G)}[i,j])$ 
5: for  $0 < k < D$  do
6:  $m \leftarrow E/U[k]$ 
7:  $\mathcal{I}_{route}^{(k,U[k])}[i,j] \leftarrow \bigvee_{j_1=(j-1)m+1}^{j \cdot m} \mathcal{I}_{route}^{(k,E)}[i,j_1], 1 \leq j \leq U[k]$ 
8:  $p_{a2a}^{(k,U[k])}[j] \leftarrow \sum_i \mathbb{I}(\mathcal{I}_{route}^{(k,U[k])}[i,j])$ 
9:  $\mathcal{I}_{route}^{(k+1,E)} \leftarrow process(\mathcal{I}_{route}^{(k,E)}) \succ \text{monitoring the change after performing Inter-level-}(k) \text{ communication}}$ 
10:  $p_{a2a}^{(k+1,G)}[j] \leftarrow \sum_i \mathbb{I}(\mathcal{I}_{a2a}^{(k+1,E)}[i,j])$ 
11: end for
12:  $d^* \leftarrow \text{Eq.}$  (6)
13: return  $d^*$ 

#### IV. HIERARCHICAL EXPERT SWAP

Our HierD-AlltoAll addresses the token duplication problem, but the workloads of different GPUs may still be imbalanced. Earlier methods, such as SmartMoE [23], swap experts by counting allocated tokens without considering duplicated tokens to determine the distribution across GPUs, which is incompatible with our proposed HierD-AlltoAll. To address this, we introduce a hierarchical expert swap strategy (HierD-ES) tailored for our HierD-AlltoAll communication that counts duplicate-free tokens assigned to each hierarchical group. Specifically, in HierD-ES, the key idea is to swap the positions of two experts during training, and the two experts are iteratively chosen to minimize communication overhead with the time model of Eq. (3). The main challenge is to formulate the optimization problem and develop the optimal solution with minimal overhead.

#### A. Problem Formulation and Solution

HierD-ES needs to count the duplicate-free tokens assigned to each hierarchical group after swapping two experts. So we combine  $t_1$  and  $t_d$  on Eq. (1) and Eq. (3) and extend to  $\mathcal{Q}_d \in \mathbb{R}^{E \times E}$ . Each element  $\mathcal{Q}_d[r,c]$  represents the estimated time cost of d-dimensional hierarchical deduplication AlltoAll after swapping the positions of r-th and c-th experts. And we formulate  $\mathcal{Q}_d$  by  $\mathcal{N}_{a2a}^{\text{Inter-}i} \in \mathbb{R}^{E \times E}$  and  $\mathcal{N}_{a2a}^{\text{Intra-}(d-1)} \in \mathbb{R}^{E \times E}$ . Each element  $\mathcal{N}_{a2a}^{\text{Inter-}i}[r,c]$  and  $\mathcal{N}_{a2a}^{\text{Intra-}(d-1)}[r,c]$  are communication bytes for Inter-level-(i) and Intra-level-(d-1) AlltoAll after swapping the positions of r-th and c-th experts.

<span id="page-5-2"></span>
$$Q_{d}[r,c] = \sum_{i=1}^{d-1} \left( \mathcal{N}_{a2a}^{\text{Inter}(i)}[r,c] \cdot \beta_{a2a}^{\text{Inter}(i)} + \alpha_{a2a}^{\text{Inter}(i)} \right) + \mathcal{N}_{a2a}^{\text{Intra}(d-1)}[r,c] \cdot \beta_{a2a}^{\text{Intra}(d-1)} + \alpha_{a2a}^{\text{Intra}(d-1)},$$

$$0 < d < D$$
(8)

<span id="page-5-1"></span>![](_page_5_Figure_7.jpeg)

Fig. 7: An illustration of our strategies to swap experts. We count the duplicate-free tokens assigned to each group after swapping any two experts and select the expert pair that minimizes the communication overhead.

Specially, we set  $\alpha_{a2a}^{\text{Intra}(0)}=\alpha_{a2a}$  and  $\beta_{a2a}^{\text{Intra}(0)}=\beta_{a2a}$  to cover one dimensional AlltoAll. And similar to Eq. (2), Eq. (4) and Eq. (5), we can formulate  $\mathcal{N}_{a2a}^{\text{Intra-}i}[r,c]$  and  $\mathcal{N}_{a2a}^{\text{Intra-}(d-1)}[r,c]$  as the product among the number of GPUs in the corresponding AlltoAll, the number of tokens sent to each GPU, the embedding dimension of a token M and bytes per dimension v.

<span id="page-5-3"></span>
$$\begin{cases} \mathcal{N}_{a2a}^{\mathsf{Inter}(i)}[r,c] = \frac{U[i]}{U[i-1]} \cdot \max(\mathcal{Z}_{a2a}^{\mathsf{Inter}(i)}[r,c,:]) \cdot M \cdot v, \\ \mathcal{N}_{a2a}^{\mathsf{Intra}(d-1)}[r,c] = \frac{G}{U[d-1]} \cdot \max(\mathcal{Z}_{a2a}^{\mathsf{Intra}(d-1)}[r,c,:]) \cdot M \cdot v. \end{cases}$$
(9)

 $\frac{U[i]}{U[i-1]} \text{ and } \frac{G}{U[d-1]} \text{ are the number of GPUs involved in an Inter-level-}(i) AlltoAll and Intra-level-}(d-1) AlltoAll which have been discussed on Eq. (4) and Eq. (5). We use <math display="block">\max(\mathcal{Z}_{a2a}^{\text{Inter-}i}[r,c,:]) \text{ and } \max(\mathcal{Z}_{a2a}^{\text{Intra-}(d-1)}[r,c,:]) \text{ to represent the number of tokens sent to each GPU in Inter-level-}(i) AlltoAll and Intra-level-}(d-1) AlltoAll after swapping <math>r$ -th and c-th experts. And each element  $\mathcal{Z}_{a2a}^{\text{Inter-}i}[r,c,k]$  of  $\mathcal{Z}_{a2a}^{\text{Inter-}i}\in\mathbb{R}^{E\times E\times U[i]}$  denotes the duplicate-free number of tokens assigned to k-th expert group of size U[i] after swapping the positions of r-th and c-th experts given that Inter-level-}(i) AlltoAll will first categorize experts into U[i] groups to count tokens assigned to each group and then select corresponding  $\frac{U[i]}{U[i-1]}$  groups to dispatch tokens. Similarly, each element  $\mathcal{Z}_{a2a}^{\text{Intra-}(d-1)}[r,c,k]$  in  $\mathcal{Z}_{a2a}^{\text{Intra-}(d-1)}\in\mathbb{R}^{E\times E\times G}$  denotes the number assigned to k-th expert group of size G after swapping the positions of r-th and c-th experts before Intra-level-}(d-1) AlltoAll.

<span id="page-6-0"></span>![](_page_6_Figure_0.jpeg)

Fig. 8: Four cases after swapping two experts with the one selected by the token while the other is not. An orange core indicates swapped experts, while an underline signifies experts selected by the token.

Taking the configuration shown in Fig. 7 as the example where experts number and groups number are four and two, we use  $\mathcal{Z}$  to show the duplicate-free number of tokens to two groups after swapping any two experts. After swapping E1 with E3, the duplicate-free number of tokens to two groups is both four, so  $\mathcal{Z}[1,3,:] = \mathcal{Z}[3,1,:] = [4,4]$ . Similarly,  $\mathcal{Z}[2,4,:] = \mathcal{Z}[4,2,:] = [4,4]$ ,  $\mathcal{Z}[2,3,:] = \mathcal{Z}[3,2,:] = [3,2]$  and  $\mathcal{Z}[1,4,:] = \mathcal{Z}[4,1,:] = [2,3]$ .

However, directly calculating  $\mathcal{Z}^{\text{Inter-}i}_{a2a}$  and  $\mathcal{Z}^{\text{Intra-}(d-1)}_{a2a}$  is expensive, which requires a time complexity of  $O(D \cdot T \cdot K \cdot E^2)$  as the experts number can be large (256 in DeepSeek-V3 [7] and 2048 in Switch [15]), where T is the total number of tokens for an MoE layer.

To reduce the complexity, we design a strategy to calculate  $Z_{a2a}^{\text{Inter-}i}$  and  $Z_{a2a}^{\text{Intra-}(d-1)}$ . Taking the configuration shown in Fig. 8 as the example where experts number and groups number are eight and four, we use Z to count the duplicatefree number of tokens to four groups after swapping any two experts. When a token arrives, it will select K experts. If both swapped experts A and B are either selected or not by the token, the token number to each group remains unchanged, just as if there were no swapping. If one expert A is selected while the other B is not, there are four possible cases illustrated in Fig. 8. In the first and second cases, if the group of the not selected expert B has no selected experts (termed as "Group2"), we must raise the "Group2" count after the swap. In the first case, if the group of the selected expert A has at least two selected experts (illustrated as "Group1"), no adjustment is needed for the "Group1" count. However, in the second case, with only one selected expert in the group (shown as "Group4"), the "Group4" count must be decreased. In the third and fourth cases, if the group of the not selected expert Bhas selected experts (termed as "Group3"), the "Group3" count remains unchanged. For the group of the selected expert A, no change is required if there are at least two selected experts (illustrated as "Group1"), as in the third case, but the value should be decreased if there is only one selected expert (shown as "Group4"), as in the fourth case. Therefore, we initially assign  $\mathcal{Z}_{a2a}^{\text{Inter-}i}$  and  $\mathcal{Z}_{a2a}^{\text{Intra-}(d-1)}$  to the value without swapping

<span id="page-6-1"></span>TABLE III: The server configurations in our testbed.

| Name    | Configuration                                 |
|---------|-----------------------------------------------|
| CPU     | Dual Intel(R) Xeon(R) Platinum 8358 @ 2.60GHz |
| GPU     | 8x Nvidia RTX A6000-48G @1.46GHz              |
| Memory  | 512GB DDR4                                    |
| NVlink  | 112.5GB/s (4x)                                |
| PCIe    | 4.0 (x16)                                     |
| Network | Mellanox MT28908 @ 200Gb/s                    |

and then adjust them across all cases to obtain the final value. The time complexity is reduced to  $O(D \cdot T \cdot K \cdot E)$ .

Then, we follow Eq. (6) to get the final estimation matrix  $Q^* = Q_{d^*}$  that represents the estimated time matrix of our HierD-AlltoAll after swapping the positions of any two experts.

**Theorem 1.** Given an MoE layer running on a cluster with expert parallelism using HierD-AlltoAll for communication, we can reduce the communication overhead by swapping the position of two experts. To achieve minimal communication time, the expert pair  $(r^*, c^*)$  should satisfy

$$(r^*, c^*) = \arg\min \mathcal{Q}^*[r, c]. \tag{10}$$

*Proof.* As discussed in Eq. (8) and Eq. (9), we have covered all cases for swapping the position of two experts. Therefore, the optimal expert pair  $(r^*, c^*)$  is identified by evaluating all cases to find the one minimizing communication time, i.e.,  $\arg\min \mathcal{Q}^*[r,c]$ , which completes the proof.

To improve the landscape of  $Q_d$ , we choose a smoother max function [39] to avoid abrupt changes in values as follows

<span id="page-6-2"></span>smooth-max
$$(x, \gamma) = \max(x) \cdot \left(\sum_{i=1}^{n} \left(\frac{x[i]}{\max(x)}\right)^{\gamma}\right)^{1/\gamma}, (11)$$

where  $\gamma$  is a parameter to control the smoothness of the function. We set  $\gamma=10$  by default (will verify in §V-E).

#### V. EVALUATION

#### A. Experimental Settings

**Testbeds.** Experiments are carried out on a 32-GPU cluster comprising four interconnected nodes, each of which is equipped with eight Nvidia A6000 GPUs. The details of the server configuration are shown in Table III. The software environments are Ubuntu-20.04, CUDA-12.1, PyTorch-2.1.2 and NCCL-2.18.5.

**Baselines.** We implement our HierMoE atop the prominent Megatron-LM training system, which supports various MoE models such as, DeepSeek and Qwen. We compare our HierMoE with three representative baselines Megatron-LM, SmartMoE and Tutel with 2DH-AlltoAll (Tutel-2DH).

**Real-World MoE Models.** To assess the end-to-end training performance on real-world MoE models, we exploit two commonly used MoE models based on DeepSeek-V3 and Qwen3-30B-A3B. Due to the GPU memory constraints of our testbed, we configure the hidden dimension and model dimension to be half of the original DeepSeek-V3 with 6 layers. For Qwen3-30B-A3B, we use 32 layers. For other

<span id="page-7-2"></span>![](_page_7_Figure_0.jpeg)

(a) Inter-level-1 and standard (b) Intra-level-1 and Inter-AlltoAll. level-2.

![](_page_7_Figure_2.jpeg)

(c) Intra-level-2 and Inter- (d) Intra-level-3 AlltoAl level-3.

Fig. 9: Performance models. Markers are measured values and lines are predicted values with estimated parameters. (a)  $\alpha_{a2a}^{\rm inter(1)}=4.97\times 10^{-1},~\beta_{a2a}^{\rm inter(1)}=5.29\times 10^{-7},~\alpha_{a2a}=7.22\times 10^{-1},~\beta_{a2a}=5.70\times 10^{-7}.$  (b)  $\alpha_{a2a}^{\rm inter(2)}=3.01\times 10^{-1},~\beta_{a2a}^{\rm inter(2)}=1.17\times 10^{-7},~\alpha_{a2a}^{\rm intra(1)}=5.71\times 10^{-1},~\beta_{a2a}^{\rm intra(1)}=1.27\times 10^{-7}.$  (c)  $\alpha_{a2a}^{\rm inter(3)}=1.49\times 10^{-1},~\beta_{a2a}^{\rm inter(3)}=2.06\times 10^{-8},~\alpha_{a2a}^{\rm intra(2)}=1.14\times 10^{-1},~\beta_{a2a}^{\rm intra(2)}=2.63\times 10^{-8}.$  (d)  $\alpha_{a2a}^{\rm intra(3)}=2.04\times 10^{-1},~\beta_{a2a}^{\rm intra(3)}=1.64\times 10^{-8}.$ 

configurations on the end-to-end experiments, we set micro batch size to 1, sequence length to 1024, the EP degree to 32, the same as the number of GPUs.

# <span id="page-7-0"></span>B. Verification of Performance Models

We require the input parameters that are related to the cluster for the performance models of AlltoAll communication. We measure the elapsed time with a range of sizes for seven types of AlltoAll communication to fit the performance models in Eq. (1) and Eq. (3) using micro-benchmark tools. In particular, we utilize the NCCL collective communication primitives along with nccl-tests<sup>2</sup> to evaluate communication durations across diverse message sizes. As shown in Fig. 9, our linear models with intercept terms (i.e., startup time) can well fit the measured performance. Specifically, the  $r^2$ for the communication tasks are as follows: standard Allto All: 0.999997, Inter-level-1 Allto All: 0.999991, Intra-level-1 AlltoAll: 0.998922, Inter-level-2 AlltoAll: 0.998682, Intralevel-2 AlltoAll: 0.999051, Inter-level-3 AlltoAll: 0.999031, Intra-level-3 AlltoAll: 0.997245. The total time required for communication in the performance models is under 300 seconds. Fitting through the least squares method takes under 10 milliseconds. When dealing with a new GPU cluster, it only needs to estimate the parameters one time using micro-

<span id="page-7-3"></span>![](_page_7_Figure_9.jpeg)

Fig. 10: The end-to-end speedup  $(\times)$  of HierMoE, HD2-MoE and HD2-MoE-Smart over Megatron-LM on DeepSeek-V3 and Qwen3-30B-A3B.

<span id="page-7-4"></span>![](_page_7_Figure_11.jpeg)

Fig. 11: The AlltoAll communication speedup (×) of Tutel-2DH, HD2-MoE, HD2-MoE-Smart, HD-MoE and HierMoE over Megatron-LM on DeepSeek-V3 and Qwen3-30B-A3B.

benchmarks prior to model training, without impacting the training efficiency.

#### C. End-to-end Training Time Comparison

To evaluate the effectiveness of HierMoE, we compare HierMoE with Megatron-LM and SmartMoE on DeepSeek-V3 and Qwen3-30B-A3B models. For better comparison, we further perform experiments on an additional schedule, HD2-MoE, which only implements the two-dimensional hierarchical deduplication as shown in Fig. 4b. We also integrate our HD2-MoE with SmartMoE (termed as HD2-MoE-Smart). The experimental results are shown in Fig. 10, which indicates that HierMoE achieves speedups of  $1.18\times$  to  $1.27\times$  compared to Megatron-LM. Additionally, HD2-MoE-Smart performs even worse than HD2-MoE, which validates that careful expert swap strategies are required on our HierD-AlltoAll. Compared to HD2-MoE, HierMoE can still achieve speedups of  $1.10\times$  to  $1.15\times$ , validating the improvement of our HierD-AlltoAll and HierD-ES.

#### D. AlltoAll Communication Time Comparison

To further evaluate the effectiveness of HierMoE on AlltoAll communication. We compare the AlltoAll time of HierMoE with that of Megatron-LM, Tutel-2DH, HD2-MoE, HD2-MoESmart and HD-MoE (HierMoE w/o HierD-ES) on Deepseek-V3 and Qwen3-30B-A3B as shown in Fig. 11. The experimental results reveal that our HierMoE provides  $1.55 \times$  to  $1.64 \times$  speedups over HD2-MoE-Smart,  $1.99 \times$  to  $2.72 \times$  speedups

<span id="page-7-1"></span><sup>&</sup>lt;sup>2</sup>https://github.com/NVIDIA/nccl-tests

<span id="page-8-1"></span>![](_page_8_Figure_0.jpeg)

Fig. 12: The curve's smoothness comparing the time cost of AlltoAll for HierMoE and Megatron as iterations rise at the first layer of Qwen3-30B-A3B.

<span id="page-8-2"></span>![](_page_8_Figure_2.jpeg)

Fig. 13: The time cost of AlltoAll for different configurations on 4 nodes and 1 node.

over Megatron-LM, and 2.34× to 3.32× over Tutel-2DH, showing the effectiveness of our approach. It can be seen that Tutel-2DH performs worse than Megatron-LM, whereas our HD2-MoE achieves 1.26× to 1.76× speedups over Megatron-LM. Furthermore, HD2-MoE-Smart is less effective than HD2-MoE, illustrating the limitations of SmartMoE. Furthermore, HD-MoE achieves a speedup of 1.37× to 1.45× compared to HD2-MoE, demonstrating the efficacy of our HierD-AlltoAll. In addition, HierMoE boosts the performance by 2.55× to 2.72× on DeepSeek-V3 and 1.72× to 1.99× on Qwen3-30B-A3B by implementing HierD-ES atop HierD-AlltoAll, further highlighting the importance of HierD-ES.

Furthermore, we assess the iteration time during the training iterations as shown in Fig. [12.](#page-8-1) It is seen that our HierMoE is much more stable than that of Megatron-LM.

# *E. Ablation Study*

Impacts by K, E, and G. We evaluate the performance with configured different K (the number of top experts se-

<span id="page-8-0"></span>TABLE IV: The AlltoAll communication speedup (×) of HD2-MoE, HD-MoE and HierMoE over Megatron-LM with varied K, E, and G.

| Method  | K    |      |      | E    |      |      | G    |      |      |
|---------|------|------|------|------|------|------|------|------|------|
|         | 6    | 8    | 10   | 64   | 128  | 256  | 8    | 16   | 32   |
| HD2-MoE | 0.95 | 1.26 | 1.30 | 1.12 | 1.26 | 1.18 | 1.24 | 1.71 | 1.26 |
| HD-MoE  | 1.37 | 1.72 | 1.82 | 1.57 | 1.72 | 1.61 | 2.36 | 2.50 | 1.72 |
| HierMoE | 1.56 | 1.99 | 2.10 | 1.83 | 1.99 | 1.84 | 2.86 | 2.62 | 1.99 |

lected for each token), E (the number of experts per MoE layer), and G (the total number of GPUs of the cluster) and measure the speedups of our proposed methods over Megatron-LM as shown in Table [IV.](#page-8-0) Results indicate that HierMoE achieves speedups ranging from 1.83× to 1.99× with different E. And the speedup of HierMoE improves from 1.56× to 2.10× as K increases, highlighting a worsened token duplication issue with higher K. As G rises, the speedup of HierMoE drops. At G = 8, without internode communication, its speedup distinctly differs from the others. For G ∈ {16, 32}, the increase in nodes leads to reduced duplicate rates at the first hierarchical level, resulting in a decrease in HD2-MoE's speedup. Nevertheless, HD-MoE achieves a comparable speedup with HD2-MoE with 1.38× when G = 16 and 1.37× when G = 32, demonstrating the validity of our approach in determining the dimensions for HierD-AlltoAll. Meanwhile, compared to HD2-MoE, the speedup of HierMoE improves from 1.53× to 1.58× with increasing G from 16 to 32, highlighting the effectiveness of our HierD-ES strategy.

Performance with different dimensions. We assess the AlltoAll time cost for nine configurations to determine the influence of varying dimensions on 4 nodes and 1 node, as depicted in Fig. [13.](#page-8-2) H1-MoE, H2-MoE, H3-MoE, and H4-MoE denote the MoE layer using hierarchical AlltoAll without deduplication, whereas HD1-MoE, HD2-MoE, HD3- MoE, and HD4-MoE represent the MoE layer with hierarchical deduplication AlltoAll. Notably, experiments on 1 node only have 3 dimensions. HD-MoE corresponds to the MoE layer employing our HierD-AlltoAll. Results indicate that while hierarchical AlltoAll does not reduce communication overhead, our deduplication approach does, and our HierD-AlltoAll optimally selects the dimension.

Performance with different kinds of max functions. We also conduct a set of experiments over three kinds of max function on Eq. [\(9\)](#page-5-3), including a smooth max function on Eq. [\(11\)](#page-6-2), a standard max function and a Log-Sum-Exp function (lnP i (exp x[i])). Results show that the standard max function, smooth max function and Log-Sum-Exp achieve speed up HierMoE over HD-MoE by 1.13×, 1.17× and 1.16×. Smooth max functions improve little to the performance. We thus simply choose the best one. We also evaluate HierMoE against HD-MoE by varying γ within [5, 7, 9, 11, 13, 15, 17, 19] to assess sensitivity to the max function's smoothness in Eq. [\(11\)](#page-6-2). Results indicate a speed up of HierMoE over HD-MoE between 1.16× and 1.17×, suggesting low sensitivity to γ.

Performance with varied expert placements updating frequency. In practice, swapping two experts takes just 1% of the total end-to-end time. We find that HierMoE achieves 1.17×, 1.17×, 1.15×, and 1.13× faster than HD-MoE with an HierD-ES update frequency of every 1, 2, 4, and 8 iterations, respectively. Higher frequencies are seen to have better performance, so we choose to update HierD-ES every iteration.

# VI. CONCLUSION

In this work, we present HierMoE, a novel MoE training approach that substantially reduces communication overhead through three key innovations: 1) a hierarchical All-to-All mechanism with token deduplication that eliminates redundant transfers across hierarchical levels, 2) a hierarchical expert placement strategy to align diverse efficiencies across different hierarchical levels in our proposed All-to-All, and 3) theoretical models to evaluate and enhance the hierarchical deduplication All-to-All and expert replacement strategy. Implemented atop Megatron-LM, our HierMoE demonstrates significant performance gains with extensive experiments conducted on a 32-GPU cluster using DeepSeek-V3 and Qwen3-30B-A3B models, achieving 1.55× to 3.32× faster AlltoAll communication compared to state-of-the-art systems like Tutel-2DH, SmartMoE and Megatron-LM, while delivering 1.18× to 1.27× faster end-to-end training time.

# ACKNOWLEDGMENTS

The research was supported in part by the National Natural Science Foundation of China (NSFC) under Grant No. 62302123, Grant No. 62272122, and Grant No. 62376073, Guangdong Provincial Key Laboratory of Novel Security Intelligence Technologies under Grant 2022B1212010005, the Guangzhou Municipal Joint Funding Project with Universities and Enterprises under Grant No. 2024A03J0616, Guangzhou Municipality Big Data Intelligence Key Lab (2023A03J0012), Shenzhen Science and Technology Program under Grant No. KJZD20240903104103005, Grant No. KJZD20230923114213027 and Grant No. KJZD20230923115113026, the Colleges and Universities Stable Support Project of Shenzhen, China (No. GXWD20220817164856008 and No. GXW-D20220811173149002), Hong Kong RGC CRF grants under contracts C7004-22G and C6015-23G.

# REFERENCES

- <span id="page-9-0"></span>[1] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, "Outrageously large neural networks: The sparsely-gated mixture-of-experts layer," in *International Conference on Learning Representations*, 2016.
- <span id="page-9-2"></span>[2] D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, and Z. Chen, "Gshard: Scaling giant models with conditional computation and automatic sharding," in *International Conference on Learning Representations*, 2020.
- [3] A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. d. l. Casas, E. B. Hanna, F. Bressand *et al.*, "Mixtral of experts," *arXiv preprint arXiv:2401.04088*, 2024.
- <span id="page-9-13"></span>[4] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv *et al.*, "Qwen3 technical report," *arXiv preprint arXiv:2505.09388*, 2025.
- [5] DeepSeek-AI, "Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model," 2024.
- [6] Y. Shen, Z. Guo, T. Cai, and Z. Qin, "Jetmoe: Reaching llama2 performance with 0.1m dollars," *CoRR*, vol. abs/2404.07413, 2024.
- <span id="page-9-1"></span>[7] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan *et al.*, "Deepseek-v3 technical report," *arXiv preprint arXiv:2412.19437*, 2024.
- <span id="page-9-3"></span>[8] C. Hwang, W. Cui, Y. Xiong, Z. Yang, Z. Liu, H. Hu, Z. Wang, R. Salas, J. Jose, P. Ram *et al.*, "Tutel: Adaptive mixture-of-experts at scale," *Proceedings of Machine Learning and Systems*, vol. 5, 2023.

- [9] R. Liu, Y. J. Kim, A. Muzio, and H. Hassan, "Gating dropout: Communication-efficient regularization for sparsely activated transformers," in *International Conference on Machine Learning*. PMLR, 2022, pp. 13 782–13 792.
- <span id="page-9-9"></span>[10] J. Li, Y. Jiang, Y. Zhu, C. Wang, and H. Xu, "Accelerating distributed {MoE} training and inference with lina," in *USENIX Annual Technical Conference*, 2023, pp. 945–959.
- <span id="page-9-11"></span>[11] S. Shi, X. Pan, Q. Wang, C. Liu, X. Ren, Z. Hu, Y. Yang, B. Li, and X. Chu, "Schemoe: An extensible mixture-of-experts distributed training system with tasks scheduling," in *Proceedings of the Nineteenth European Conference on Computer Systems*, 2024, pp. 236–249.
- <span id="page-9-4"></span>[12] X. Pan, W. Lin, L. Zhang, S. Shi, Z. Tang, R. Wang, B. Li, and X. Chu, "Fsmoe: A flexible and scalable training system for sparse mixtureof-experts models," in *Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1*, 2025, pp. 524–539.
- <span id="page-9-5"></span>[13] M. Lewis, S. Bhosale, T. Dettmers, N. Goyal, and L. Zettlemoyer, "BASE layers: Simplifying training of large, sparse models," in *International Conference on Machine Learning*. PMLR, 2021, pp. 6265–6274.
- [14] Y. Zhou, T. Lei, H. Liu, N. Du, Y. Huang, V. Zhao, A. Dai, Z. Chen, Q. Le, and J. Laudon, "Mixture-of-experts with expert choice routing," *arXiv preprint arXiv:2202.09368*, 2022.
- <span id="page-9-15"></span>[15] W. Fedus, B. Zoph, and N. Shazeer, "Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity," *The Journal of Machine Learning Research*, vol. 23, no. 1, pp. 5232–5270, 2022.
- [16] Z. Chi, L. Dong, S. Huang, D. Dai, S. Ma, B. Patra, S. Singhal, P. Bajaj, X. Song, X.-L. Mao *et al.*, "On the representation collapse of sparse mixture of experts," *Advances in Neural Information Processing Systems*, vol. 35, pp. 34 600–34 613, 2022.
- <span id="page-9-6"></span>[17] J. Puigcerver, C. Riquelme, B. Mustafa, and N. Houlsby, "From sparse to soft mixtures of experts," *arXiv preprint arXiv:2308.00951*, 2023.
- <span id="page-9-7"></span>[18] S. Rajbhandari, C. Li, Z. Yao, M. Zhang, R. Y. Aminabadi, A. A. Awan, J. Rasley, and Y. He, "Deepspeed-moe: Advancing mixture-ofexperts inference and training to power next-generation ai scale," in *International Conference on Machine Learning*. PMLR, 2022, pp. 18 332–18 346.
- [19] R. Y. Aminabadi, S. Rajbhandari, A. A. Awan, C. Li, D. Li, E. Zheng, O. Ruwase, S. Smith, M. Zhang, J. Rasley *et al.*, "Deepspeed-inference: enabling efficient inference of transformer models at unprecedented scale," in *International Conference for High Performance Computing, Networking, Storage and Analysis*. IEEE, 2022, pp. 1–15.
- <span id="page-9-8"></span>[20] Z. Ma, J. He, J. Qiu, H. Cao, Y. Wang, Z. Sun, L. Zheng, H. Wang, S. Tang, T. Zheng *et al.*, "Bagualu: targeting brain scale pretrained models with over 37 million cores," in *Proceedings of the 27th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming*, 2022, pp. 192–204.
- <span id="page-9-10"></span>[21] J. He, J. Zhai, T. Antunes, H. Wang, F. Luo, S. Shi, and Q. Li, "Faster-MoE: modeling and optimizing training of large-scale dynamic pretrained models," in *Proceedings of the 27th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming*, 2022, pp. 120–134.
- <span id="page-9-14"></span>[22] S. Shi, X. Pan, X. Chu, and B. Li, "PipeMoE: Accelerating mixtureof-experts through adaptive pipelining," in *IEEE INFOCOM 2023-IEEE Conference on Computer Communications*, 2023.
- <span id="page-9-12"></span>[23] M. Zhai, J. He, Z. Ma, Z. Zong, R. Zhang, and J. Zhai, "SmartMoE: Efficiently training Sparsely-Activated models through combining offline and online parallelization," in *USENIX Annual Technical Conference*, 2023, pp. 961–975.
- [24] J. Liu, J. H. Wang, and Y. Jiang, "Janus: A unified distributed training framework for sparse mixture-of-experts models," in *Proceedings of the ACM SIGCOMM 2023 Conference*, 2023, pp. 486–498.
- [25] X. Pan, W. Lin, S. Shi, X. Chu, W. Sun, and B. Li, "Parm: Efficient training of large sparsely-activated models with dedicated schedules," in *IEEE INFOCOM 2024-IEEE Conference on Computer Communications*, 2024.
- [26] C. Chen, X. Li, Q. Zhu, J. Duan, P. Sun, X. Zhang, and C. Yang, "Centauri: Enabling efficient scheduling for communication-computation overlap in large model training via communication partitioning," in *Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3*, 2024, pp. 178–191.
- [27] C. Jiang, Y. Tian, Z. Jia, C. Wu, Y. Wang, and S. Zheng, "Lancet: Accelerating mixture-of-experts training by overlapping weight gradient computation and all-to-all communication," *Proceedings of Machine Learning and Systems*, vol. 6, pp. 74–86, 2024.

- [28] N. Wang, W. Lin, L. Zhang, S. Shi, R. Zhou, and B. Li, "Spmoe: Expediting mixture-of-experts training with optimized pipelining planning," in *IEEE INFOCOM 2025-IEEE Conference on Computer Communications*. IEEE, 2025, pp. 1–10.
- [29] X. Pan, R. Wang, W. Lin, S. Shi, and X. Chu, "Mitigating contention in stream multiprocessors for pipelined mixture of experts: An sm-aware scheduling approac," in *2025 IEEE 45th International Conference on Distributed Computing Systems (ICDCS)*. IEEE, 2025.
- [30] W. Lin, X. Pan, S. Shi, X. Wang, B. Li, and X. Chu, "Mast: Efficient training of mixture-of-experts transformers with task pipelining and ordering," in *2025 IEEE 45th International Conference on Distributed Computing Systems (ICDCS)*. IEEE, 2025.
- <span id="page-10-0"></span>[31] W. Lin, X. Pan, S. Shi, X. Wang, and X. Chu, "Scheinfer: Efficient inference of large language models with task scheduling on moderate gpus." in *European Conference on Parallel Processing*. Springer, 2025.
- <span id="page-10-1"></span>[32] M. Alizadeh, A. Greenberg, D. A. Maltz, J. Padhye, P. Patel, B. Prabhakar, S. Sengupta, and M. Sridharan, "Data center tcp (dctcp)," in *Proceedings of the ACM SIGCOMM 2010 Conference*, 2010, pp. 63–74.
- [33] M. Al-Fares, A. Loukissas, and A. Vahdat, "A scalable, commodity data center network architecture," *ACM SIGCOMM computer communication review*, vol. 38, no. 4, pp. 63–74, 2008.
- [34] W. Wang, M. Khazraee, Z. Zhong, M. Ghobadi, Z. Jia, D. Mudigere, Y. Zhang, and A. Kewitsch, "{TopoOpt}: Co-optimizing network topology and parallelization strategy for distributed training jobs," in *20th USENIX Symposium on Networked Systems Design and Implementation (NSDI 23)*, 2023, pp. 739–767.
- <span id="page-10-2"></span>[35] C. Zhao, C. Deng, C. Ruan, D. Dai, H. Gao, J. Li, L. Zhang, P. Huang, S. Zhou, S. Ma *et al.*, "Insights into deepseek-v3: Scaling challenges and reflections on hardware for ai architectures," in *Proceedings of the 52nd Annual International Symposium on Computer Architecture*, 2025, pp. 1731–1745.
- <span id="page-10-3"></span>[36] J. Dean, G. Corrado, R. Monga, K. Chen, M. Devin, M. Mao, M. Ranzato, A. Senior, P. Tucker, K. Yang *et al.*, "Large scale distributed deep networks," *Advances in neural information processing systems*, vol. 25, 2012.
- <span id="page-10-4"></span>[37] M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, and B. Catanzaro, "Megatron-lm: Training multi-billion parameter language models using model parallelism," 2020. [Online]. Available: <https://arxiv.org/abs/1909.08053>
- <span id="page-10-5"></span>[38] X. Nie, X. Miao, Z. Wang, Z. Yang, J. Xue, L. Ma, G. Cao, and B. Cui, "Flexmoe: Scaling large-scale sparse pre-trained model training via dynamic device placement," *Proceedings of the ACM on Management of Data*, vol. 1, no. 1, pp. 1–19, 2023.
- <span id="page-10-6"></span>[39] Z. Zhou, Q. Zhang, and A. M.-C. So, "\ell 1, p-norm regularization: Error bounds and convergence rate analysis of first-order methods," in *International conference on machine learning*. PMLR, 2015, pp. 1501– 1510.