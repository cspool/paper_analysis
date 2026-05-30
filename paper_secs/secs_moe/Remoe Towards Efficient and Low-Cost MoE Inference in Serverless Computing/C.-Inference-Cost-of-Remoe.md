# C. Inference Cost of Remoe

Consistent with prior work [19], we mainly consider the cost of memory usage. We divide the total cost into two parts: the main model cost and remote experts cost.

1) Main Model Cost: The cost of the main model can be calculated as:

as:  

$$C^{loc} = (PT + GT)[c^g M^g + c^e \sum_{v=0}^{V} w_v m_v]$$
 (6)

where  $c^c$  is the cost of using 1MB of  $\overset{v=1}{\text{CPU}}$  memory for 1 second, and  $M^g$  is the total GPU memory occupied by the main model, which can be expressed as:

where 
$$a_l$$
 is the data size of the kv-cache for a single token

in layer l. Kv-cache technique [20] prevents the model from re-computing transformer matrices for previous tokens. Consequently, the term  $(N^{in} + N^{out})(D + \sum_{l} a_{l})$  represents the total memory occupied by the token embeddings and the entire kv-cache, while  $\sum_{l} \mu(f_{l})$  is the memory occupied by the nonexpert modules.

2) Remote Experts Cost: The cost associated with the remote experts can be divided into prefilling cost  $PC^{rem}$ , and the decoding cost  $GC^{rem}$ . Therefore, the total cost for the remote experts is expressed as  $C^{rem} = PC^{rem} + GC^{rem}$ .

**Prefilling Cost**. The cost of remote experts during prefilling

is calculated as follows: 
$$PC^{rem} = c^c \sum_{l=1}^L \sum_{v=1}^{V^e} y_{l,v} m_v \sum_{j=1}^{z_l} ZT_{l,j} \tag{8}$$

where the cost of each replica is the product of its memory  $y_{l,v}m_v$  and runtime  $ZT_{i,j}$ .

Decoding Cost. The decoding cost of remote experts,  $GC^{rem}$ , is calculated as:

$$GC^{rem} = c^{c} \sum_{i=N^{in}+1}^{N^{in}+N^{out}} \sum_{l=1}^{L} \sum_{v=1}^{V^{e}} y_{l,v} m_{v} \sum_{k=1}^{K_{l}} x_{l,k} s_{l,k,i}$$

$$\cdot (GT_{l,k}^{rem} + 2D/B + t_{l}^{rem})$$
(9)

where the cost is also the product of its memory  $y_{l,v}m_v$  and runtime  $(GT_{l,k}^{rem} + 2D/B + t_l^{rem})$ .

#### D. Problem Formulation

<span id="page-3-9"></span>The objective is to minimize the total model cost while satisfying SLOs, which is defined as follows:

$$\min_{x,y,z,w} C^{loc} + C^{rem} \tag{10a}$$

s.t. 
$$T^{ttft} \leqslant TTFT$$
, (10b)

$$T^{tpot} \leqslant TPOT,$$
 (10c)

<span id="page-3-3"></span><span id="page-3-2"></span><span id="page-3-1"></span><span id="page-3-0"></span>
$$\sum_{v=1}^{V^e} y_{l,v} = 1, \tag{10d}$$

$$\sum_{k=1}^{K_l} x_{l,k} (\mu(e_{l,k}) + DN_{l,k}^{pre}) \leqslant \sum_{v=1}^{V^e} y_{l,v} m_v,$$
 (10e)

$$\sum_{k=1}^{L} \sum_{k=1}^{K_l} (1 - x_{l,k}) \mu(e_{l,k}) + DN^{out} \leq \sum_{v=1}^{V} w_v m_v, \quad (10f)$$

<span id="page-3-5"></span><span id="page-3-4"></span>
$$\sum_{e_{l,k} \in \mathcal{R}_{l,j}} N_{l,k}^{pre} D \leq U^{payload}, \tag{10g}$$

<span id="page-3-6"></span>
$$x_{l,k}, y_{l,v}, w_v \in \{0, 1\}, \quad \forall l, k, v,$$
 (10h)

<span id="page-3-7"></span>
$$z_l \leqslant z^{max}, z_l \in \mathbb{Z}^+, \quad \forall l.$$
 (10i)

Thereinto, Constraint (10b) and (10c) guarantee the TTFT and TPOT. Constraint (10d) ensures that the remote experts at each layer can only be assigned a single memory specification. Constraint (10e) ensures that the allocated memory for remote experts at each layer is sufficient to hold both the model weights and the data for the tokens they process. Similarly, Constraint (10f) ensures that the memory allocated to the main model is sufficient for its weights and all tokens. Constraint (10g) guarantees that the data transferred to a single replica does not exceed the payload size,  $U^{payload}$ . Finally, Constraints (10h) and (10i) define the domains of the decision variables, ensuring the number of expert replicas does not exceed a maximum limit,  $z^{max}$ .

Challenge. In the model described above, unpredictable tokens and complex solutions are two key challenges. In fact, the variable  $s_{l,k,i}$  is unknown a priori. Even if all token routing paths were known, the optimization objective remains difficult to solve since it involves products of the decision variables. The situation places the original problem in the category of Nonlinear Programming, which is known to be NP-hard [21].

#### IV. REMOE DESIGN

## A. Main Idea

To address the challenges previously discussed, we design a system for the MoE inference in serverless, named *Remoe*. When a request arrives, *Remoe* executes the following steps:

- i. Activation Prediction. The arriving request is first processed by the pre-processing layer and Remoe gets the input tokens. Then Remoe invokes SPS algorithm to predict the expert activation matrix for the new request. In the offline phase, Remoe builds a multi-fork clustering tree based on historical data. Soft Cosine Similarity (SCS) is used to measure the semantic similarity between prompts and build the tree.
- ii. Resource Pre-allocation. Upon request arrival, Remoe employs the MMP algorithm to pre-allocate resources. To satisfy TTFT and TPOT constraints, MMP determines the optimal remote expert ratio b by estimating the worst-case remote load—a process justified by a proven upper bound. Based on this ratio, it assigns the memory allocation  $w_v$  and initiates the main model's cold start.
- *iii. Remote experts Selection.* Afterwards, *Remoe* will calculate the expected utility of all experts based on the predicted matrix and set all low-utility experts as remote.
- iv. Memory Optimization. To reduce the complexity, we construct a new correlation function for  $y_{l,v}$  based on their characteristics and fit it, which reformulates the problem. Then, Remoe uses Lagrangian duality to solve the problem and the subsequent convexity analysis proves that the optimal solution can be found within the feasible region.
- v. Multi-replicas Inference. We formulate the multi-replica inference during prefilling as a Multiway Number Partitioning Problem. Remoe employs the LPT algorithm to solve it, and the resulting upper bound dictates the necessary number of remote expert replicas  $z_l$ .

#### <span id="page-4-2"></span>B. Activation Distribution Prediction

For incoming requests, after the pre-processing layer, *Remoe* predicts subsequent expert activation based on the semantic information of input tokens.

<span id="page-4-0"></span>![](_page_4_Figure_10.jpeg)

Fig. 3: Semantic similarity and expert activation distribution Fig. 3 compares the semantic similarities and the Jensen-Shannon (JS) Divergence of expert activation distributions between 1 test sample and 15 training samples from LMSYS-Chat-1M dataset [22], fed into GPT2-MoE (Sec. V-A). Note that JS Divergence is a typical probability distribution similarity comparing method [23]. Obviously, semantic similarity positively correlates with expert activation similarity, enabling its use as a proxy for expert activation comparison. Promptlevel expert activation prediction is detailed below.

Semantic Similarity Comparison. We compute the semantic similarity between two prompts,  $\zeta_1$  and  $\zeta_2$ , using SCS

[24]. This involves normalizing and concatenating their token embedding matrices, then multiplying by the transpose to yield a symmetric token similarity matrix  $\mathbb{C}_{\zeta_1,\zeta_2}$ . We also construct two alignment vectors,  $\mathbb{V}_1$  and  $\mathbb{V}_2$ , to mark token ownership per prompt via binary indicators (1: belonging; 0: otherwise).  $\mathbb{V}_1$  and  $\mathbb{V}_2$  are column vectors. Thus SCS between semantic embeddings of  $\zeta_1$  and  $\zeta_2$  is calculated below:

embeddings of 
$$\zeta_1$$
 and  $\zeta_2$  is calculated below:  

$$SCS_{1,2} = \frac{\mathbb{V}_1^T \mathbb{C}_{\zeta_1,\zeta_2} \mathbb{V}_2}{\sqrt{\mathbb{V}_1^T \mathbb{C}_{\zeta_1,\zeta_2} \mathbb{V}_1} \cdot \sqrt{\mathbb{V}_2^T \mathbb{C}_{\zeta_1,\zeta_2} \mathbb{V}_2} + \sigma}, \quad (11)$$

where  $\sigma$  is an extremely small value used to prevent division by zero. Because  $\mathbb{C}_{\zeta_1,\zeta_2}$  is a Gram matrix, which is positive semi-definite,  $\mathbb{V}_j^T\mathbb{C}_{\zeta_1,\zeta_2}\mathbb{V}_j$  is non-negative.

**Semantically Similar Prompts Searching.** We efficiently search semantically similar prompts for a new one based on the multi-fork clustering tree.

Pairwise semantic similarities for all historical prompts are precomputed. During tree construction, any node (cluster) with more than  $\beta$  prompts is recursively partitioned. The partition is based on a customized k-medoids clustering algorithm using prompt-level semantic similarity as distance metric, where roulette wheel sampling-based centroid initialization and subcluster-level centroid updating are conducted.

We set  $\beta > \alpha$  to augment tree retrieval with local brute-force searching. For a new prompt, the tree is traversed to a leaf by successively selecting the semantically closest subcluster centroid. If there are enough prompts in the leaf, top- $\alpha$  semantically similar ones are returned; otherwise, we turn to the leaf's siblings for supplement.

