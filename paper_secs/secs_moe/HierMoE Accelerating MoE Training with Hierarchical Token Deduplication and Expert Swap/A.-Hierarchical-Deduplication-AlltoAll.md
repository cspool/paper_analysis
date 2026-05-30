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

