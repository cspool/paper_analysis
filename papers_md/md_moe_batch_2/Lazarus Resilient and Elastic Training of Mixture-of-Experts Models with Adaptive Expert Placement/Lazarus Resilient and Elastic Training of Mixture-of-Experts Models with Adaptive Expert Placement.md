# Lazarus: Resilient and Elastic Training of Mixture-of-Experts Models

Yongji Wu1,<sup>∗</sup> Wenjie Qu2,<sup>∗</sup> Xueshen Liu3,<sup>∗</sup> Tianyang Tao<sup>2</sup> Yifan Qiao<sup>1</sup> Zhuang Wang<sup>4</sup> Wei Bai<sup>5</sup> Yuan Tian<sup>6</sup> Jiaheng Zhang<sup>2</sup> Z. Morley Mao<sup>3</sup> Matthew Lentz<sup>7</sup> Danyang Zhuo<sup>7</sup> Ion Stoica<sup>1</sup> <sup>1</sup>UC Berkeley <sup>2</sup>NUS <sup>3</sup>UMich <sup>4</sup>AWS <sup>5</sup>NVIDIA <sup>6</sup>UCLA <sup>7</sup>Duke

# Abstract

Sparsely-activated Mixture-of-Experts (MoE) architecture has increasingly been adopted to further scale large language models (LLMs). However, frequent failures still pose significant challenges as training scales. The cost of even a single failure is significant, as all GPUs need to idle wait until the failure is resolved, potentially losing considerable training progress as training has to restart from checkpoints. This problem is exacerbated by the growing use of spot instances on public clouds for model training, which despite offering substantial cost savings, introduce frequent preemptions—essentially failures that regularly occur throughout the training process. Existing solutions for efficient faulttolerant training either lack elasticity or rely on building resiliency into pipeline parallelism, which cannot be applied to MoE models due to the expert parallelism strategy adopted by the MoE architecture.

We present Lazarus, a system for resilient and elastic training of MoE models. Lazarus adaptively allocates expert replicas to address the inherent imbalance in expert workload and speeds up training, while a provably optimal expert placement algorithm is developed to maximize the probability of recovery upon failures. Through adaptive expert placement and a flexible token dispatcher, Lazarus can also fully utilize all available nodes after failures, leaving no GPU idle. Our evaluation shows that Lazarus outperforms existing MoE training systems by up to 5.7x under frequent node failures and 3.4x on a real spot instance trace.

### 1 Introduction

The advent of large language models (LLMs) has demonstrated ever-increasing capabilities with the rapid growth in both model sizes and training datasets. Recently, the sparselyactivated Mixture-of-Experts (MoE) models have been increasingly adopted by the community to further scale model parameters [\[14,](#page-13-0) [21,](#page-13-1) [23,](#page-13-2) [30\]](#page-13-3). Training state-of-the-art MoE

models is becoming resource-intensive. For instance, it takes over 32K H100 GPUs to train the 2T Llama 4 model [\[23\]](#page-13-2).

The likelihood and frequency of failures significantly increase as the scale and duration of training increase. Meta projects that the mean time to failure (MTTF) is as little as 14 minutes for a cluster with 128K GPUs [\[16\]](#page-13-4). Even a single failure is costly, as all GPUs are idle until the failure is resolved and failed nodes are replaced. It is reported that failures can slow the training progress by up to 43% [\[24\]](#page-13-5). In addition, most cloud providers offer preemptible (spot) instances that can be leveraged for training LLMs with minimized monetary cost [\[4,](#page-12-0) [35\]](#page-13-6), as they offer cost savings of up to 90% compared to on-demand instances. Preemptions, which are essentially failures, can happen as frequently as every 5~10 minutes [\[35\]](#page-13-6).

Existing solutions for LLM training with quick failure recovery can be categorized into two classes: checkpointing optimizations or pipeline-parallelism based elastic training. The first line of work [\[2,](#page-12-1) [38,](#page-14-0) [39\]](#page-14-1) reduces checkpointing overhead by either using CPU memory of neighboring nodes to periodically checkpoint model states, or relying on stale states which compromises correctness [\[2\]](#page-12-1). They also lack elasticity and have to wait for replacement nodes of the failed ones to recover from failure and continue training, which may not be available for hours to days until failed nodes are repaired [\[9\]](#page-13-7). Especially for training on spot instances, such new node availability cannot be taken for granted.

The second line of works builds resiliency and elasticity into pipeline parallelism by taking advantage of its configurability in stages-nodes mapping [\[4,](#page-12-0) [13,](#page-13-8) [35\]](#page-13-6). In particular, they can continue training upon failures without requiring additional nodes. However, these approaches do not apply to MoE models, as the distributed training of MoE models depends on a different parallelism strategy: expert parallelism (EP) [\[17\]](#page-13-9). EP distributes experts across multiple GPUs (and nodes) and uses all-to-all communication to dispatch input tokens to GPUs with corresponding experts.

In this paper, we present Lazarus, a system for resilient and elastic training of MoE models. Lazarus achieves highthroughput training accompanied by a high failure recovery probability without restarting from checkpoints. Upon

1

<sup>∗</sup>Yongji Wu, Wenjie Qu and Xueshen Liu are co-first authors of this work.

<span id="page-1-0"></span>![](_page_1_Figure_0.jpeg)

Figure 1: MoE architecture utilizes expert parallelism for distributed training, yet it also suffers from imbalanced workload due to the dynamic nature of gate networks.

failures, Lazarus quickly reconfigures the training job and utilizes all remaining GPUs (regardless of how many nodes fail).

Our insight is that adaptively adjusting the number of replicas (GPUs) assigned to each expert and their placement enables elastic training while improving resiliency against failures. Due to the dynamic nature of its architecture, MoE models suffer from dynamic and imbalanced workload [\[10,](#page-13-10) [27,](#page-13-11) [40\]](#page-14-2). Tokens are routed to experts based on the decisions of trainable gate networks. Some experts have more tokens routed to than others. Traditional EP partitions experts into equal-sized chunks, and each is assigned to the same number of GPUs. In contrast, Lazarus allocates more replicas to popular experts and flexibly assigns them using all available GPUs. Such flexible expert allocation not only results in performance boosts but also leads to better elasticity. As long as a single replica for each expert remains available, training can continue to progress with all remaining nodes utilized; traditional EP requires using a multiple of EP size GPUs, which can induce significant performance degradation even for minor failures.

There are three key challenges Lazarus must address. First, we need an expert allocation and placement algorithm that takes account of the imbalanced workload, to speed up expert computation while ensuring a high probability of successful recovery. Second, with our asymmetrical expert placements in the cluster, how do we efficiently dispatch tokens to GPUs with corresponding experts and balance their loads? Third, how do we quickly re-instantiate lost expert replicas and efficiently migrate the cluster to a new placement plan in response to failures?

To address these challenges, we propose a strategy for allocating expert replicas based on the load distribution, while maintaining a fault-tolerant threshold to guarantee failure recovery when a small number of nodes fail. We design a provably optimal algorithm for placing these replicas to maximize the recovery probabilities under arbitrary node failures. We develop a CUDA kernel that dispatches tokens in parallel

<span id="page-1-1"></span>![](_page_1_Figure_6.jpeg)

Figure 2: Expert loads on a 16 experts model (GPT-L in [§6.1\)](#page-7-0). The distribution varies during training and across layers.

with a flexible all-to-all that minimizes inter-GPU communication. During migration, Lazarus utilizes a greedy strategy to reduce state transfers for efficient reconfiguration.

We implement Lazarus in PyTorch. We evaluate Lazarus across MoE models of different scales with both controlled failures and spot instance traces. Our results show that Lazarus outperforms checkpointing-based DeepSpeed MoE [\[31\]](#page-13-12), a widely adopted system for training MoE models, by up to 2.3x under infrequent failures (40 mins MTBF) and 5.7x under a high failure frequency (5 mins MTBF), while our evaluation on a real spot instance trace demonstrates a performance improvement of 3.4x.

In this paper, we make the following contributions:

- To the best of our knowledge, Lazarus is the first system for resilient and elastic training of MoE models that enables both quick recovery from failures and full utilization of all available (remaining) GPUs.
- We design a provably optimal algorithm for determining expert placement that maximizes recovery probability in response to uniformly random node failures.
- We implement and evaluate Lazarus with MoE models of different scales under a variety of scenarios.

#### 2 Background and Motivation

#### 2.1 MoE Models and Expert Parallelism

Mixture-of-Experts architecture has been recently applied to scale LLMs due to its high cost-efficiency, which replaces the dense feed-forward network (FFN) in a transformer block. MoE employs multiple parallel FFNs called experts. In each MoE layer, a trainable gate network routes each token to only the top- experts. As experts are sparsely activated, MoE enables scaling model parameters without an increase of the per-token computational cost.

As the size of an MoE model is dominated by the weights of the experts, expert parallelism (EP) [\[17\]](#page-13-9) has been proposed and has become the de facto approach to train large-scale MoE models. In expert parallel training, the experts of each layer are split into equal-sized chunks and allocated across multiple GPUs similar to tensor parallelism, while the input samples are distributed along the batch dimension similar to data parallelism. The number of GPUs required to split the experts is called the EP size and such a set of GPUs forms an EP group. For instance, in [Figure 1,](#page-1-0) there are 4 experts and each GPU accommodates 2 experts, therefore it has a EP size of 2. EP can be used in conjunction with other types of parallelism like data and tensor parallelism.

As each GPU in an EP group only holds a subset of experts, all-to-all communication is used to dispatch the input tokens to the GPUs with corresponding experts that the gate network routes to. The computation of the experts are performed on the owning GPUs and the results are sent back to the original GPUs with a second all-to-all (combine).

The most distinctive feature of expert parallelism is the dynamic nature of gate networks. The distribution of tokens routed to each expert can be highly unbalanced depends on the input data. We plot the evolution of expert loads from a training trace [\[40\]](#page-14-2) in [Figure 2.](#page-1-1) We observe that the load of experts is highly skewed, with up to 87% tokens routed to 2 most popular experts. The load distribution also varies at different layers and training iterations.

The skewed expert loads in MoE training directly translates to imbalance in expert computation. GPUs holding more popular experts takes much longer time to compute due to large amount of tokens dispatched to them, while other GPUs are idling. Previous works [\[8,](#page-13-13) [10,](#page-13-10) [27,](#page-13-11) [40\]](#page-14-2) addresses this challenge by dynamically adjusting parallelism strategies on a cluster with a fixed number of GPUs. They do not apply in an elastic environment with changing device membership.

In addition to the problem of imbalanced workload, traditional EP also utilizes a multiple of EP size GPUs, which may leave some of GPUs idle upon a failure. The waste of GPUs only grows with increasing number of experts, as more GPUs are needed for a single EP group, i.e., larger EP size.

#### 2.2 Fault-Tolerant and Elastic Training

A growing research effort has been made in resilient training in recent years, due to the fact that both the frequencies and costs of failures increase as the scale and duration of training increase. It is reported during the two-months training of OPT 175B, around 100+ failures were encountered [\[41\]](#page-14-3), wasting over 178,000 GPU hours. The cost of even one failure is significant, as all the GPUs must wait idle until the failure is resolved and failed nodes are repaired, which could take hours to days depending on the nature of failures [\[9\]](#page-13-7). To minimize the GPU idling and the resulting economic loss, a training system must be designed with resiliency in terms of it can quickly recover from failures, and elasticity in terms that it can efficiently utilize currently available GPU resources to continue training. Such systems also enable one

<span id="page-2-0"></span>![](_page_2_Figure_7.jpeg)

Figure 3: System architecture of Lazarus.

to leverage preemptible instances on public clouds to train LLMs with significant cost savings [\[4,](#page-12-0) [35\]](#page-13-6).

Existing training solutions with quick failure recovery capability can be divided into two categories: checkpointing optimizations and elastic training using pipeline parallelism. Checkpointing based solutions focus on reducing the overhead in both saving checkpoints and restarting [\[2,](#page-12-1) [6,](#page-13-14) [37](#page-14-4)[–39\]](#page-14-1). In particular, in-memory based checkpointing [\[38,](#page-14-0) [39\]](#page-14-1) has been proposed to store model states in the CPU memory of other nodes in addition to persistent storage, while MoC-System [\[2\]](#page-12-1), an MoE specific checkpointing solution, compromises correctness by using stale states. However, they lack elasticity as they have to wait until replacements of failed nodes are available to resume training.

To support both elastic and fault tolerant training without the overhead of checkpointing and restarting, recent attempts [\[4,](#page-12-0) [13,](#page-13-8) [35\]](#page-13-6) have been made in building resiliency into pipeline parallelism due to its configurability. However, they fail to apply to MoE models. As the model states of a single MoE layer can exceed the GPU memory capacity, they are generally trained in conjunction with expert parallelism, requiring resiliency for expert states distributed across GPUs.

In summary, existing systems for fault-tolerant and elastic training fail to adapt to MoE models. Lazarus targets MoE training, utilizing adaptive expert allocation and placement to address expert parallelism's inelastic nature while handling the imbalanced expert load distribution caused by the dynamic gate networks.

#### 3 System Overview

Lazarus is a resilient and elastic system for training MoE models. Lazarus speeds-up training by adaptively allocating expert replicas based on the dynamic expert load distribution using all available GPUs, while our fault-tolerant expert placement strategy maximizes Lazarus's recovery probability even under simultaneous failures of multiple nodes.

The architecture of Lazarus is shown in Figure 3. Lazarus consists of three main components: a centralized controller that manages a GPU cluster, an agent process on each GPU node that spins up worker processes with Lazarus runtime. The controller runs persistently on a (CPU-only) node and it communicates with each Lazarus agent, monitors the cluster and detects node failures and replenishment. A scheduler in the controller allocates expert replicas and computes a fault-tolerant placement plan for all GPU nodes that maximizes the recovery probability (§4.1). The placement is sent to each Lazarus agent to configure the workers. Based on the placement plan, Lazarus runtime fills up each layer with corresponding experts assigned to it. Unlike vanilla expert parallelism where all experts are equally replicated, Lazarus assigns more replicas and more GPUs to the heavily loaded experts. As the expert placement becomes asymmetric, Lazarus runtime also contains a CUDA kernel based dispatcher (§4.2) to efficiently dispatch tokens to GPUs with corresponding experts and balance their loads.

Upon detection of failures, the controller recomputes an expert placement plan using all remaining nodes and minimizes the number of replicas migrated. Once Lazarus runtime receives the new plan relayed by Lazarus agent, it dynamically reconfigures the parallelism setups and retrieves missing model states from other nodes (§4.3). To handle dynamics in workloads, Lazarus agent also periodically collects the expert load distribution (routing history of gate networks) from Lazarus runtime. The load distribution is communicated to the load monitor on the controller, which then rebalances the expert allocation and placement.

#### 4 Design

# <span id="page-3-0"></span>4.1 Adaptive Expert Allocation and Placement

Lazarus considers that each GPU can hold a certain number of replicas limited by its GPU memory, similar to traditional EP. Through assigning more replicas to popular experts, Lazarus can speed up training by giving them more computation resources. Note that we allow multiple replicas of the same expert assigned to a single GPU, which indicates more tokens (of the specific expert) can be processed by that GPU compared to assigning a single replica.

Yet, there is an inherent trade-off between speeding up computation and fault resiliency. On the one hand, if a less

<span id="page-3-1"></span>![](_page_3_Figure_8.jpeg)

Figure 4: Fault resiliency depends on how expert replicas are placed. With the same replica allocation of 4 experts and 4 replica slots per node, placement plan A and B differ in recovery probability under 3 node failures.

popular expert is assigned with only a single replica, then as long as the GPU (node) holding that replica fails, Lazarus cannot recover due to the loss of expert state. On the other hand, a balanced allocation of replicas improves fault resiliency; however, it degenerates to traditional EP and defeats the goal of addressing expert load imbalance.

Moreover, given an expert replica allocation, the placement of these replicas determines the probability of failure recovery. For instance, if all replicas of an expert are all placed on GPUs in a single node, the loss of that node would lead to an unrecoverable failure. Hence, when allocating and placing expert replicas, Lazarus should not only consider the imbalanced workload to speed up computation, but also take account of the impact on fault tolerance.

We divide the problem into two phases and separately consider allocation and placement. In the first phase, we design an expert allocation strategy that balances between the workload's expert distribution and fault tolerance. In the second phase, we design an expert placement algorithm which is theoretically optimal, maximizing the recovery probability given a fixed expert allocation. In this way, our allocation and placement plan strikes a balance between the two goals. **Expert allocation.** For ease of presentation, we show how expert replicas are allocated and placed on each node. If a node has multiple GPUs, Lazarus simply distributes the assigned replicas among all GPUs on that node, as we consider failures at the node level. We denote the number of nodes as N, the number of experts as E, the number of replicas each node can hold as c, the total number of tokens routed to expert e as  $t_e$ , the number of replicas assigned for expert e as  $r_e$ . To speed-up computation, we want the ratio of replicas assigned to each expert match the ratio of tokens routed to that expert, namely  $\frac{r_e}{\sum_{e'} r_{e'}} \approx \frac{t_e}{\sum_{e'} t_{e'}}$ . Furthermore, for better fault tolerance, we define a fault-tolerant threshold f, where Lazarus guarantees a 100% probability of failure recovery as long as fewer than f nodes fail simultaneously. Hence, each expert is assigned at least f replicas.

Assume that the experts are sorted by the number of routed tokens  $(t_e)$  in ascending order. We iteratively compute the number of replicas  $r_e$  assigned to each expert e as follows:

$$r_e = \max\{\lfloor \frac{t_e}{\sum_{e'=e}^{E} t_{e'}} \cdot (N \cdot c - \sum_{e'=1}^{e-1} r_{e'}) \rfloor, f\}$$
 (1)

Our assignment strategy ensures that  $\sum_e r_e = N \cdot c, r_e \geq f, r_e \geq r_{e-1}$ .  $\frac{r_e}{\sum_{e'} r_{e'}} \approx \frac{t_e}{\sum_{e'} t_{e'}}$  is also satisfied in most cases for training speed-up under the imbalanced workload. As  $r_e \geq f$ , Lazarus guarantees recovery under failures of a small number (< f) of nodes.

**Expert placement.** However, when the number of failed nodes  $\geq f$ , the probability of failure recovery differs between different placement plans. Figure 4 shows an example of 4 experts and 5 nodes. Assume that 3 nodes fail simultaneously. In plan A, the probability of recovery is  $\frac{5}{10}$ , as recovery is possible only if the alive nodes are (1, 2), (1, 3), (1, 4), (2, 4), (2, 5), while there are 10 possible cases. In plan B, however, the probability of recovery is much higher at  $\frac{7}{10}$ .

Placement solution for an easier case. We first consider a simpler case where  $E \leq c$ , which we can easily derive an optimal placement strategy inspired by the previous example. The strategy is that for the first  $min(r_1, N)$  nodes we place the first (least popular) expert, for the first  $min(r_2, N)$  nodes we place the second expert, and so on. For the vacant positions, we uniformly place the experts that still have replicas left. In this way, denote the set of nodes that have the e-th expert as  $S_e$ . This strategy satisfies  $S_1 \subset S_2 \cdots \subset S_E$ . Thus, the recovery probability is equal to the probability that the first expert belongs to an alive node (i.e., any of the first  $r_1$ nodes are alive). Furthermore, the first expert belonging to an alive node is a necessary condition of failure recovery. Thus, for any placement plan, the recovery probability is upper bounded by the probability of any of the first  $r_1$  nodes is alive. Since there are only  $r_1$  replicas for the first expert, it can span across at most  $r_1$  different nodes. Therefore, in the case of  $E \leq c$ , this placement strategy achieves the upper bound of the recovery probability of all placement plans, guaranteeing its optimality.

The above strategy relies on a core principle: maximize the nodes overlapped between the experts. Take the first and second expert as an example; if we overlap all the replicas of the first experts with the second expert's replicas on the same nodes, the two experts' states will be lost only when all of the first expert's replicas are lost. However, if some of the first expert's replicas are not overlapped with the second

expert's, the two experts cannot be recovered when either all of the first's replicas are lost or all of the second's are lost. Placement solution for the more difficult case. When E > c, the optimal strategy becomes more complicated. The previously introduced maximum overlap principle cannot be directly applied due to the infeasibility of overlapping all experts when E > c. To address this issue, we partition both the experts and nodes into  $\lceil \frac{E}{c} \rceil$  groups. We also modify the second principle into maximizing the overlap of experts in each group. Furthermore, we constrain the expert partitions to be consecutive, i.e.,  $1, \dots, c$  are in the first group,  $c+1, \cdots, 2c$  forms the second group, and so on. For the nodes, we divide the first  $\min\{N, \sum_{i=1}^{\lceil \frac{E}{c} \rceil} r_{c*(i-1)+1}\}$  nodes into  $\lceil \frac{E}{c} \rceil$  groups. The first group has  $r_1$  nodes, the second has  $r_{c+1}$  nodes, and so on, while the last group has  $\min\{N-\sum_{i=1}^{\lceil\frac{E}{c}\rceil-1}r_{c*(i-1)+1},r_{c*(\lceil\frac{E}{c}\rceil-1)+1}\}$  nodes. For group iin the first  $\lceil \frac{E}{c} \rceil - 1$  groups, each node contains one replica of expert c \* (i - 1) + 1, ..., c \* i. For the last group, each node contains one replica of expert  $c * (\lceil \frac{E}{c} \rceil - 1) + 1, \dots, E$ . For the vacant slots, we uniformly place the experts that still have replicas left to place. Our strategy satisfies  $S_{c*(i-1)+1} \subset$  $S_{c*(i-1)+2} \cdots \subset S_{c*i}$  for different i, which intuitively maximizes the node overlap of experts in each group. The recovery of the experts in the i-th group hence only requires one node in  $S_{c*(i-1)+1}$  to be alive, where we define the expert c \* (i - 1) + 1 as the representative of group i. The complete recovery is equivalent to that one replica of each group's representative still remains. Our maximum rank overlap (MRO) placement plan maximizes recovery probability under uniformly random node failures for any given replica number r. Concretely, we have Theorem 1. Its proof can be found in the supplementary material.

<span id="page-4-0"></span>THEOREM 1. For any MRO plan T and R, given the number of replicas  $r_e$  for each expert e, T maximizes the recovery probability  $\Pr(\bigcup_{a \in A} Col_a = [E])$ , where [E] is the set of experts,  $Col_a$  is the set of replicas assigned to node a, A is a uniformly sampled set of R nodes that remain alive.

The two core insights of our method are to partition experts into different groups based on their popularity, and maximize the overlap across experts in the same group. Here, we offer some intuition for analyzing the optimality of our method.

Denote the failed node number as k; there are  $\binom{n}{k}$  different combinations of node failure cases. The failure analysis of different placement plans would be much more explicit by visualizing a graph formed this way: It is a bipartite graph, where one set has E vertices, each representing an expert. The other set has  $\binom{n}{k}$  vertices, each corresponding to a case of node failure. Each placement strategy can be analyzed by constructing such a bipartite graph: for every expert in the

<span id="page-5-1"></span>![](_page_5_Figure_0.jpeg)

Figure 5: Lazarus minimizes the failure probability by minimizing the number of vertices representing node failures that have incident edges. Here we consider 3 node failures. Comparing Case I and II, when expert overlap on nodes is not maximized, there are more unique failure patterns. Comparing Case I and III, swapping any expert also leads to more failure patterns.

placement, an edge is created between that expert and every node failure combination that renders the expert unrecoverable.

Essentially, our strategy achieves optimality by "putting all the eggs in one basket." The failure probability of a given placement strategy can be counted by the proportion of the second set of vertices (failure combinations) that have incident edges. Our strategy minimizes the number of the second sets of vertices that have incident edges. This is achieved by forcing experts to share the same failure combinations as much as possible, through maximizing the overlap between popular experts. For each non-representative expert, the set of failure combination vertices it connects to is a subset of the vertices the representative expert connects to. For different placement plans, the number of edges is the same. By letting more experts fail under a smaller set of failure cases, we reduce the number of failure nodes that have incident edges, thus improving recovery probability.

When the overlap between experts is not maximized, as in case II of Figure 5, expert *B* creates unique failure combinations (1, 3, 4), (1, 3, 5), expert *D* creates a unique failure combination (2, 4, 5). Therefore, the failure probability rises to  $\frac{7}{10}$ , compared to the optimal of  $\frac{4}{10}$ .

In addition, case III in Figure 5 offers an example of the optimality of our group partition strategy. If any swap is conducted in two expert groups, where the second expert group has a more popular representative, the total number of failed combinations connected by the two groups' representatives increases, because the new representatives are less popular than the old representatives.

We note that the expert load distribution can be different across layers, hence we compute an expert replica allocation and placement plan independently for each layer. As the load distribution also shifts during training according to the workload, Lazarus also periodically rebalances the expert allocation and updates the placement plan.

Now, we have developed the strategy to assign expert replicas to each node (GPUs). Next, we explore under such asymmetric replica placements, how Lazarus efficiently dispatches tokens to GPUs with replicas of routed experts.

#### <span id="page-5-0"></span>4.2 Flexible Token Dispatcher

In traditional expert parallelism, each token can be simply dispatched to the GPU that owns the corresponding expert, as there is only a single replica for each expert within a particular EP group. Concretely, an all-to-all is performed with all ranks (GPUs) in the EP group sending and receiving the same number of tokens, which is dynamically set to the maximum number routed to a single expert to prevent token dropping, while unused slots are padded [10, 31].

With Lazarus's adaptive expert placement, there are varying numbers of replicas assigned for each expert on different sets of GPUs. Therefore, each rank must decide which rank with the routed expert's replica to dispatch a token to. The fact that multiple replicas can be assigned to the same rank (indicating more tokens should be dispatched to it), combined with the difference in expert routing on different ranks, a challenge emerges — how can we efficiently dispatch the tokens to all GPUs with the routed experts while balancing the load? If tokens are poorly dispatched, some ranks could

# Algorithm 1: Token dispatch algorithm.

**Input**: N: Number of GPUs; i: Current GPU rank; h:

 $R_{e,j}$ : Number of replicas for expert e

Activation of input tokens to the MoE block;

```
assigned to rank j; T_{e,j}: Number of tokens
                 routed to expert e at rank j;
    Output: h': Shuffled inputs for all-to-all dispatch; s<sub>i</sub>:
                 Number of tokens to dispatch to rank j
 1 for e \leftarrow 0 to E in parallel do
          r_e \leftarrow \sum_{i} R_{e,j} // total #replicas for expert e
          t_e \leftarrow \sum_i T_{e,i} // total #tokens routed to expert e
 3
         p_e \leftarrow t_e/r_e // #tokens each replica should handle
         for j \leftarrow 0 to N in parallel do
 5
               P_{e,j} \leftarrow c_e R_{e,j} // \text{ #tokens rank } j \text{ can process}
 6
               P_{e,j} \leftarrow P_{e,j} - \min(P_{e,j}, T_{e,j})
             // rank j's local tokens are prioritized
          D_{e,i} \leftarrow c_e R_{e,i} - P_{e,i} // locally processed #tokens
 8
         for j \leftarrow 0 to N, j \neq i in parallel do
 9
            D_{e,j} \leftarrow (T_{e,i} - D_{e,i}) \frac{P_{e,j}}{\sum_{k \neq j} P_{e,k}}
10
             // distribute remaining tokens to other ranks
11 for j \leftarrow 0 to N in parallel do
         s_i \leftarrow \sum_{e'} D_{e',j} // \text{ #tokens dispatched to rank } j
12
          for e \leftarrow 0 to E in parallel do
13
               start \leftarrow \sum_{0..j-1} s_{j'} + \sum_{0..e-1} D_{e',j}
14
               end \leftarrow \sum_{0..j-1} s_{j'} + \sum_{0..e} D_{e',j}
15
              h'[start..end] \leftarrow (\sum_{j'=0}^{j-1} D_{e,j'})-th to
16
                (\sum_{j'=0}^{j} D_{e,j'})-th tokens in h that routed to e
17 return h', s
```

<span id="page-6-7"></span><span id="page-6-6"></span><span id="page-6-5"></span><span id="page-6-4"></span><span id="page-6-1"></span>receive significantly more tokens than others, hence defeating the purpose of our adaptive expert allocation. Moreover, the padded all-to-all is no longer viable in our case where a token can be dispatched to any rank (instead of within an EP group), as padding would dominate the communication.

To address these issues, we design a flexible token dispatcher that efficiently dispatches each token to a particular GPU and balances the number of tokens routed to each GPU. With the dispatch schedule computed, Lazarus performs a flexible all-to-all without padding. Algorithm 1 shows the workflow of the token dispatcher, which is implemented in a CUDA kernel to process all experts and target ranks in parallel. The basic idea behind Algorithm 1 is that each replica of an expert should compute around the same number of tokens, and each rank should utilize its local processing "capacity" before dispatching remaining tokens to other ranks.

Before computing the dispatch schedule, an all-gather is first performed to collect how many tokens are routed to each expert from all ranks, i.e.,  $T_{e,j}$ .  $T_{e,j}$  is collected so that the token dispatcher can better balance the load to each

GPU based on the expert routing distribution of all tokens from all ranks, instead of using only locally computed tokens. In addition, since collective communication operations require synchronization of all participant ranks,  $T_{e,j}$  is also necessary in computing how many tokens a rank should receive from each of the other ranks. Since only E integers are collected from each rank, this extra all-gather imposes negligible overhead, as demonstrated in §6.5.1. The number of replicas allocated to each GPU  $R_{e,j}$  from the placement plan is also passed to the token dispatcher.

After  $T_{e,j}$  is collected, each rank i independently computes how many tokens it dispatches to each of all N ranks, for all E experts. First, for each expert e, the number of tokens each replica should process is computed in line 4 by evenly distributing all  $t_e$  tokens routed to e onto all  $r_e$  replicas. The processing capacity of each rank j can then be computed by multiplying  $p_e$  with the number of replicas of e that j is assigned (line 6). This capacity will be prioritized towards tokens computed locally on j. After the remaining capacities of all ranks are computed, rank i dispatches the remaining  $(T_{e,i} - D_{e,i})$  tokens that are beyond i's local processing capacity. The number of tokens  $D_{e,j}$  to dispatch to each rank for e is calculated based on their residual capacities (line 10).

Since the all-to-all collective operates on a continuous buffer, the token dispatcher has to reshuffle the input activations h to the MoE block, so that tokens routed to the same expert and dispatched to the same rank are grouped together. The total tokens  $s_j$  to dispatch to rank j across all experts is computed in line 12. In lines 13-16, these tokens are sorted by their routed experts and placed consecutively in h'. The reshuffled activations h' are then used in the dispatch all-to-all collective, with  $s_j$  tokens sent to each rank j. The token dispatcher also computes how many tokens to receive from each rank j in the all-to-all in a similar fashion.

At this point, Lazarus has the ability to adaptively assign expert replicas and dynamically dispatches tokens among replicas of routed experts. Next, we discuss how Lazarus efficiently migrates to a new configuration upon failures.

#### <span id="page-6-0"></span>4.3 Efficient Reconfiguration

As discussed in §4.1, if at least a single replica of each expert still remains after failures, Lazarus can recover without restarting from checkpoints. However, the remaining expert replicas' distribution could deviate from the desired allocation computed for the workload, and their placement may be prone to subsequent failures. Therefore, Lazarus must reallocate expert replicas and efficiently migrate to a new placement. Such migration is also required when Lazarus rebalances the expert allocation and when new nodes join.

The ordering of nodes in the placement plan is not enforced in the placement algorithm, as long as each node in the plan maps to a physical node in the cluster. However, when migrating from an old placement plan, such a mapping becomes relevant. It directly determines how many experts' states a node needs to retrieve from other nodes, as only newly assigned ones not in the old placement plan have to be fetched. To reduce the number of replicas to shuffle during migration, hence the communication, Lazarus applies a greedy algorithm that iteratively maps a physical node to a node in the new placement plan that the number of newly assigned experts is minimized.

After the node mapping is determined, Lazarus schedules the transfers of expert states. Each node fetches missing states for the newly assigned experts from other nodes that own them. If multiple nodes require the states of the same expert, Lazarus distributes their state transfers among all owning nodes, to minimize the overall migration time.

#### 5 Implementation

Lazarus is implemented in 4K LoC in Python and 500 LoC in CUDA, building on top of PyTorch [\[12\]](#page-13-15) (v2.3) and using components from DeepSpeed [\[32\]](#page-13-16) (v0.13).

Lazarus controller and agents. We implement the controller and agents using Python's asynchronous framework. New agents register with the controller, using a TCP socket for communication. The controller maintains a global view of node availability, where agents periodically send heartbeats for it to detect failures. Upon failures or scaling up with newly arrived nodes, the controller computes an updated expert placement plan, which is sent to each agent and relayed to the worker process that uses Lazarus runtime. The agents also periodically collect expert routing history from each worker and send it to the controller for expert rebalancing. Lazarus runtime. Based on the controller's configuration, our runtime sets up NCCL [\[26\]](#page-13-17) communication groups for expert and non-expert gradients all-reduce, as well as all-toall in expert computation. We implement data parallelism and expert parallelism with our adaptive expert placement; however, Lazarus can be extended to combine with pipeline parallelism using techniques like Oobleck [\[13\]](#page-13-8), which are orthogonal and complementary to ours. Upon failures, enqueued NCCL operations time out and the model states are not updated on the failed step, while a new configuration is received from the agent via a listener thread. Batched NCCL send/recv primitives are used to transfer states during migration. For scaling up and rebalancing, Lazarus performs reconfiguration lazily, only after the current training step is finished.

<span id="page-7-1"></span>Table 1: Configurations of models used in the evaluation.

|              | GPT-S | GPT-M | GPT-L |
|--------------|-------|-------|-------|
| # Layers     | 12    | 12    | 12    |
| Feature dim. | 768   | 1024  | 1024  |
| # Experts    | 8     | 12    | 16    |
| # Params     | 521M  | 1.3B  | 1.7B  |

#### 6 Evaluation

#### <span id="page-7-0"></span>6.1 Setups

Testbed. We have five servers in our testbed, each with 2 NVIDIA RTX 3090 GPUs and a 100 Gbps Mellanox ConnectX-5 NIC connected to a single 100 Gbps Mellanox SN2100 switch. Due to limited resources, we treat each GPU as a separate node to emulate a cluster of 10 GPUs. To store checkpoints, we deploy a NFS server on a separate machine, which is connected to the GPU servers via 10 Gbps NICs.

Baselines. As there is no existing system to support resilient and elastic training of MoE models, we compare Lazarus against a checkpoint-based baseline using DeepSpeed MoE (DS) [\[31\]](#page-13-12), a widely adopted MoE training system with both system-side and model design-side optimization. To evaluate Lazarus's adaptive expert placement algorithm and flexible token dispatcher, we also build a fault tolerant baseline based on DeepSpeed MoE, utilizing efficient reconfiguration module from Lazarus runtime. We denote this baseline as DS(FT). Similar to Lazarus, if a complete replica of all experts still exists upon failures, it reconfigures the workers (reassigns EP groups) and retrieves required model (expert) states from owning nodes.

Workloads. Based on the widely used GPT-2 architecture, we adopt three MoE models of varying sizes and number of experts, listed in [Table 1.](#page-7-1) We use a per-GPU batch size of 4 and a sequence length of 1024 following GPT-2's setup [\[5\]](#page-13-18). For all evaluations, we use Wikitext-2 dataset [\[25\]](#page-13-19), top-1 gate, and FP16 precision for training.

For reproducibility, we use the routing history trace from SmartMoE [\[40\]](#page-14-2) artifact to emulate gate networks' routing decisions. We use the loads of the top experts at each layer to construct a routing trace for each of the models we evaluate. We set the number of expert replica slots for each GPU to 6, which is the upper limit based on available GPU memory. With DS's traditional expert parallelism, GPT-M can fully utilize all slots, while GPT-S and GPT-L can only use 4, as the multiple of slots per GPU and EP size must equal to the number of experts. GPT-S and GPT-M can utilize an EP size of 2, hence DS and DS(FT) fully utilize all 10 nodes in the cluster, while with 16 experts and an EP size of 4, they can only utilize 8 nodes on GPT-L. We set the checkpoint

interval to every 50 steps for DS and every 250 steps for DS(FT), unless mentioned otherwise. We set the minimum replicas per expert ( ) to 2 for Lazarus so that recovery is guaranteed under common single node failure scenarios. Lazarus rebalances expert replica allocation every 200 steps.

## <span id="page-8-1"></span>6.2 Controlled Single Node Failures

We first evaluate the performance in a more common case where a single node fails at a time. We consider both high failure frequency and low frequency scenarios, where we randomly choose a node to fail every 5 or 40 minutes, until only half of the nodes remain. The same set of nodes is selected to fail in each run for fair comparison. The results are shown in [Figure 6](#page-9-0) and [Figure 7.](#page-9-1) The throughput is smoothed over a short time window for visibility. The fluctuation in Lazarus's throughput is caused by the reconfiguration after node failures and the periodic rebalance of expert allocations, while the fluctuation in DS and DS(FT) is caused by checkpointing, restarting and reconfiguration (only for DS(FT)). To reduce the overhead of checkpointing for DS and DS(FT) in the low failure frequency (40 minutes) setting, we increase their checkpoint intervals by 4x to 200 steps and 1000 steps, respectively. We also note that using such low checkpoint frequency would prevent DS from making any effective progress under high failure frequency (5 minutes).

From [Figure 6](#page-9-0) with a failure frequency of 5 minutes, we observe that over the 30 minutes duration of training, Lazarus finished a total of 2926 and 1996 steps, trained 2.8x and 5.7x samples on GPT-S and GPT-L, compared with DS. The performance gains significantly increase on GPT-L, as the checkpointing and restarting overhead grows with model sizes. Moreover, as in the GPT-L setting (EP size is 4), 4 nodes are required to hold a complete replica of all experts for DS and DS(FT), they can only utilize either 4 or 8 nodes, while they can utilize all 10 nodes at the start for GPT-S and GPT-M.

Lazarus also outperforms DS(FT) by 1.4x and 2.8x on GPT-S and GPT-L. On the smaller GPT-S, there are a large number of replicas for each expert (5 replicas initially), hence DS(FT) can recover in each failure. However, as the number of experts and EP size increases on GPT-L, DS(FT) has to restart from checkpoints after failures of both EP groups.

When the failure is infrequent as shown in [Figure 7,](#page-9-1) the performance difference between Lazarus and DS decreases as the overhead of checkpointing and restarting decreases. Still, Lazarus outperforms DS by 1.6x and 2.3x on GPT-S and GPT-L. As the overhead of DS decreases, DS and DS(FT) have similar performance in this case.

We also observe that Lazarus's throughput tends to monotonically decrease as the number of nodes decreases, as Lazarus can fully utilize all remaining nodes for training.

<span id="page-8-0"></span>

|                    | GPT-S    |           | GPT-L    |           |
|--------------------|----------|-----------|----------|-----------|
|                    | step 200 | step 4000 | step 200 | step 4000 |
| # Lost nodes       | 2        | 3         | 4        | 5         |
| Reconfig time (s)  | 21.3     | 34.1      | 18.2     | 19.7      |
| # Experts transfer | 11       | 52        | 160      | 55        |
| Transfer time (s)  | 2.3      | 3.0       | 7.6      | 7.8       |

Table 2: [Multi-node failures]: Recovery overhead of Lazarus under multiple node failures on sampled cases.

While for DS and DS(FT), the throughput experiences steep drops since they can only utilize a multiple of EP size nodes. We note that the throughput of Lazarus increases in the last 40 minutes in [Figure 7.](#page-9-1) This is because Lazarus no longer enforces a minimum of 2 replicas for each expert for fault tolerance, as there are not enough slots with 5 nodes left.

Lazarus still outperforms DS by a great margin, even when both of them fully utilize all 10. For instance, for GPT-M, during the first 40 minutes in [Figure 7](#page-9-1) when no node fails, Lazarus has a throughput of 45 samples/sec during effective computation (factoring out checkpoint and rebalance overheads), while DS only reaches 34 samples/sec.

From [Figure 6,](#page-9-0) we also observe that compared to DS, DS(FT) sometimes has higher throughput during effective computation. For instance, for GPT-M, DS(FT) outperforms DS by 1.6x during the 5~10 minutes window, when they all fully utilize the remaining 8 nodes. This is mainly caused by the highly imbalanced expert loads during the early periods of training, while DS(FT) progresses much faster without the overhead of checkpoint and restarting. When we increase the checkpoint intervals by 4x for both baselines in [Figure 7,](#page-9-1) together with the lower failure frequency, such divergence disappears. Instead, for GPT-S and GPT-M, DS(FT) is slower than DS during the last 80 minutes. In these two cases, DS(FT) always resumes training by reconfiguring currently used nodes that are still alive. It does not use previously dropped nodes (due to exceeding EP size of 2), while DS attempts to utilize all nodes it can when restarting.

Overall, checkpointing and restarting overhead becomes increasingly significant with larger models and higher failure frequency. Comparing with DS(FT) which shares Lazarus's efficient reconfiguration runtime, Lazarus's adaptive expert placement improves both training throughput and resiliency.

## 6.3 Controlled Multi Node Failures

Next, we study how well Lazarus handles simultaneous failures of multiple nodes. Whether Lazarus can recover from such failures depends on both the expert allocation (i.e., how

<span id="page-9-0"></span>![](_page_9_Figure_0.jpeg)

Figure 6: [Single node failure]: Throughput and total trained samples with a single node fails every 5 minutes. DS refers to checkpointing-based DeepSpeed MoE; DS(FT) is a fault tolerant version we build using components from Lazarus's runtime.

<span id="page-9-1"></span>![](_page_9_Figure_2.jpeg)

Figure 7: [Single node failure]: Throughput and total trained samples with a single node fails every 40 minutes.

<span id="page-9-2"></span>![](_page_9_Figure_4.jpeg)

Figure 8: [Multi-node failures]: Recovery probabilities using different expert placement strategies.

many replicas are assigned to each expert) and expert placement, as well as which concrete set of nodes fail. The allocation and placement changes as the expert load distribution varies over the duration of training, and it is also different for different layers. Hence, we evaluate Lazarus's system overhead of recovery by sampling several cases for GPT-S and GPT-L at different training steps, while we evaluate Lazarus's placement algorithm by computing the recovery probability for a model at a given training step. The recovery probability can be computed by enumerating all possible

combinations of failed nodes, since the way experts are allocated and placed only depends on the expert load at the particular step.

The recovery overhead for sampled cases is shown in Table 2, where 2 to 5 nodes are selected to fail at training step 200 and 4000. We report the total number of experts replicas that need to be transferred between nodes and the time spent on the state transfers. The weights and optimizer states of each is 63MB for GPT-S and 112MB for GPT-L. We find that the overhead of state transfers is negligible. This low overhead is mainly contributed by the fact that required states can be fetched from other nodes instead of the much slower remote storage, and Lazarus balances the point to point send/recv operations among all owning ranks of an expert's states. We also report the total reconfiguration time, from failure occurrence to training resumption, where state transfers only constitute a small portion. Throughout our entire evaluation, we find that each reconfiguration event takes 20~40 seconds. It takes 10~20 seconds for enqueued NCCL kernels to time out and 5~15 seconds for reconfiguring NCCL's communication groups. We also observe that the placement plan's computation takes less than 100ms.

To demonstrate the effectiveness of Lazarus's fault-tolerant expert placement algorithms, we compare it with two baselines: a spread placement strategy which distributes each

<span id="page-10-1"></span>![](_page_10_Figure_0.jpeg)

Figure 9: [Spot instance]: Throughput changes in spot instance environment.

expert's replicas across different nodes in a round-robin fashion, and a compact strategy that packs an expert's replicas on a minimum number of nodes. The recovery probabilities with respect to the number of nodes failed are illustrated in Figure 8. We find that Lazarus's placement algorithm greatly outperforms both baselines. For instance, for GPT-L at step 200, Lazarus has a 41% recovery probability with 4 node failures, compared to 12% of spread placement. We also observe that on the smaller GPT-S when expert loads are relatively more balanced at later step 4000, compact placement achieves limited recovery capability with 1 or 2 node failures. However, it completely fails to recover in any failure scenario on the larger GPT-L with 16 experts.

#### <span id="page-10-3"></span>6.4 Spot Instance Trace

We also borrow a real spot instance node availability trace from Bamboo [35] to evaluate Lazarus under both failures and scaling-up. The trace includes both preemption events and node additions. We replay a representative 80 minutes segment of the availability trace collected on AWS EC2 P3 instances. As the original trace is collected on a 32 nodes cluster, we cap the maximum number of nodes to 10 in our testbed setup. To handle rare cases where recovery is not possible due to too many nodes failing at the same time, we also apply periodic checkpointing for Lazarus. We set the checkpoint interval to every 250 steps, the same as DS(FT), for fair comparison. For node addition events, all compared methods waited for 2 minutes to accumulate sufficient nodes before scaling up, to avoid frequent reconfiguration or restarting. The results are shown in Figure 9.

Over the 80 minutes duration, Lazarus trained 2.3x and 3.4x samples on GPT-S and GPT-L, compared with DS. Lazarus outperforms DS(FT) by 1.2x and 1.8x on GPT-S and GPT-L. We also note that Lazarus's throughput changes proportionally to the number of nodes available, as Lazarus wastes no node, while DS and DS(FT) are limited by EP sizes.

Due to the overhead of checkpointing and restarting, DS trained 51% and 48% fewer samples than DS(FT). DS(FT) can always recover from failures for GPT-S and GPT-M, as

<span id="page-10-2"></span>![](_page_10_Figure_7.jpeg)

Figure 10: [Ablation Study]: Single layer throughput and recovery probabilities under different expert load ratios.

it evenly allocates up to 5 replicas to all experts at a cost of reduced throughput. For GPT-L, however, when there is fewer than 8 nodes, DS(FT) cannot utilize more than 4 nodes for redundancy. It has to restart from the checkpoint each time, leading to 3~5 minutes of lost progress.

We observe that only in a single preemption event when 4 nodes are lost at 34 minutes, Lazarus has to restart from checkpoint. Note that in the original trace, only a maximum of 19% nodes failed at a time.

#### 6.5 Ablation Study

<span id="page-10-0"></span>6.5.1 Impacts of Expert Load Imbalance. To study how the expert load imbalance in workloads affects both Lazarus's performance and fault resiliency, we build a single MoE layer with 8 experts and a feature dimension of 1024. We construct workloads with different expert load ratios. We show the layer forward throughput in Figure 10a. Here, a load ratio of 4:1 indicates that 4x more tokens are routed to one of the experts than if all experts are evenly routed to.

We observe that Lazarus's throughput remains constant as the load ratio changes, attributed to Lazarus's adaptive expert allocation based on expert load distribution. DS's throughput, however, dramatically decreases as the workload becomes more skewed. When the workload is perfectly balanced (1:1), Lazarus suffers a small overhead due to its token dispatcher.

<span id="page-11-0"></span>![](_page_11_Figure_0.jpeg)

Figure 11: [Ablation Study]: Running time breakdown of GPT-S and GPT-L on the spot instance trace.

We also evaluate the effectiveness of Lazarus's expert placement algorithm in fault tolerance as the load distribution changes. [Figure 10b](#page-10-2) shows the recovery probability of Lazarus with varying numbers of failed nodes on 2:1 and 4:1 load ratios, compared with the spread placement strategy. We observe that the recovery probability decreases with more imbalanced workload, as less popular experts are assigned fewer replicas. Still, Lazarus's placement algorithm is much more effective than spread placement, while our previous evaluation demonstrates the increased throughput is worth the effort of skewed expert allocation.

6.5.2 Running Time Breakdown. We breakdown the running time on the spot instance trace from [§6.4](#page-10-3) in [Figure 11.](#page-11-0) Both Lazarus and DS(FT) have much more time spent in effective computation, benefiting from efficient reconfiguration module in Lazarus runtime, while over half of the time is spent on checkpointing and restarting (fallbacks) for DS. The reconfiguration and rebalance overhead of Lazarus is much smaller than restarting, accepting for less than 10%. We also find that DS(FT) can recover in all cases on GPT-S, yet it suffers 27% restarting overhead on GPT-L. Despite similar effective time, Lazarus outperforms DS(FT) by 1.8x in terms of total trained samples, contributed by our adaptive expert allocation and flexible token dispatcher.

# <span id="page-11-1"></span>6.6 Comparison with Other MoE Training Systems

While many MoE training optimizations are orthogonal and can be directly applied to Lazarus to speed up training, some modify the token dispatch logic and are nontrivial to integrate. Still, Lazarus significantly improves end-to-end training performance with fault tolerance and elasticity.

Here we study how Lazarus compares to Tutel [\[10\]](#page-13-10), a MoE training system that implements state-of-the-art kernel optimizations for dispatch and combine operations under traditional expert parallelism. We also build a fault tolerant variant of Tutel using Lazarus's runtime reconfiguration module, which we denote as Tutel(FT). To demonstrate that Lazarus

adapts to different hardware setups, we set up a testbed on AWS using 16 g5.2xlarge instances. Each instance has an NVIDIA A10G GPU, while the instances are connected via a 10 Gbps TCP network. To store checkpoints, we employ AWS EFS that provides shared file systems. Due to the limited network bandwidth, we also employ the widely-used gradient accumulation technique [\[34\]](#page-13-20) with an accumulation step of 20 to avoid frequent gradient synchronization. We keep other settings the same as in [§6.2.](#page-8-1)

We report in [Figure 12](#page-12-2) the training performance under a random node failure of every 5 minutes. Lazarus outperforms Tutel(FT) and Tutel by 1.8x and 2.1x for GPT-S, and by 3.9x and 4.0x for GPT-L. We also observe that with more nodes in the cluster, Tutel(FT) can leave many unused, as it cannot reuse idle nodes that are previously dropped without restarting, same as DS(FT) in [§6.2.](#page-8-1) In the supplementary material, we provide the results under an ideal case where subsequent failed nodes are the unused ones by Tutel and Tutel(FT).

#### 6.7 Simulation

We evaluate how Lazarus scales to larger models and larger clusters via simulations. We follow Bamboo [\[35\]](#page-13-6) to setup our simulation, using a constant node preemption probability while varying new node allocation probabilities per simulation hour. We simulate the training of the DeepSeek V3 model [\[21\]](#page-13-1), where each node has 8 H200 GPUs and 8 400 Gbps NICs. We use the performance model from [\[1\]](#page-12-3). We consider a mixture of data, tensor and expert parallelism, where tensor parallelism is used within each node for non-expert components, while each GPU holds 4 expert replicas for both traditional expert parallelism and Lazarus. We sample an expert routing trace for DeepSeek-V3 using ShareGPT dataset [\[42\]](#page-14-5). We run 10 trials for each setting.

In [Figure 13a,](#page-12-4) we show the job completion time (JCT) for training 5M samples with 512 GPUs (64 nodes) under different failure probabilities, where the JCT is normalized to Lazarus. With a low failure probability of 0.01, Lazarus outperforms DS(FT) and DS by 2.9x and 3.9x, while it increases to 3.6x and 14.5x under a failure probability of 0.15. In [Fig](#page-12-4)[ure 13b,](#page-12-4) we show the JCT under a failure probability of 0.05 with different numbers of GPUs. With 256 GPUs, Lazarus speeds up over DS(FT) and DS by 2.9x and 4.5x, while with 1K GPUs, Lazarus' speed-up increases to 3.5x and 10.6x. We also observed that our expert placement algorithm in [§4.1](#page-3-0) takes less than 1 sec on a single CPU core for 1K GPUs.

## 7 Related Work

MoE training systems Extensive studies have focused on optimizing MoE training. A series of works [\[15,](#page-13-21) [18,](#page-13-22) [22,](#page-13-23) [28,](#page-13-24) [31,](#page-13-12) [33\]](#page-13-25) optimize the all-to-all communication performance.

<span id="page-12-2"></span>![](_page_12_Figure_0.jpeg)

Figure 12: [Comparison with Tutel]: Throughput and total trained samples with a single node fails every 5 minutes.

<span id="page-12-4"></span>![](_page_12_Figure_2.jpeg)

Figure 13: [Simulation]: Simulated training performance of DeepSeek V3, under different failure probabilities with 512 GPUs and with different #GPUs under 5% failure probability. Error bars represent 95% confidence intervals.

Another line of works design different MoE algorithms and architectures [3, 19, 28, 31, 44–46]. Various system optimizations have been proposed to deal with the imbalanced workload. For example, Tutel [10] and SmartMoE [40] propose dynamic parallelism switching; FasterMoE [8] and FlexMoE [27] also utilize the idea of expert replication. However, these works all focus on speeding up training on a fixed-sized cluster, while Lazarus considers an elastic environment where resiliency and quick reconfiguration is crucial. Many of these optimizations can also be integrated to Lazarus.

Fault-tolerant and elastic training. Early efforts in elastic training focus on small models trained with pure data parallelism. TorchElastic [36] restarts a job upon node membership changes. Elastically allocating resources among multiple jobs have also been explored in [7, 11, 20, 29, 43]. However, they do not work for modern LLMs which are frequently well beyond a single GPU's memory capacity. To enable resilient training of large models, many checkpointing optimization techniques have been proposed [2, 6, 37–39]. Yet, they lack elasticity and require replacement nodes to resume training. We note that Gemini [39] designs a strategy for placing checkpoints in CPU memory across machines to maximize

recovery probability. However, it assumes each GPU's check-point has the same number of replicas, hence does not apply to our expert placement problem, where different experts have different number of replicas. Systems supporting both resilient and elastic training of LLMs [4, 13, 35] are all based on pipeline parallelism, utilizing its flexibility in stage-device mapping. These works are complementary to Lazarus, where we target expert parallelism introduced in MoE.

#### 8 Conclusion

This paper presents Lazarus, the first system for resilient and elastic distributed training of MoE models. Lazarus adaptively allocates replicas based on the expert routing distribution of the workload to speed-up training. With a proven optimal expert placement strategy, Lazarus maximizes the probability of failure recovery. Upon failures, Lazarus efficiently migrates to a new expert placement plan with all remaining GPUs fully utilized. Our results show that Lazarus outperforms state-of-the-art checkpointing based MoE training systems by up to 5.7x under frequent node failures and 3.4x on a real spot instance trace. We will open source Lazarus.

#### References

- <span id="page-12-3"></span> 2025. DeepSeek-V3/R1 Performance Simulator. https://github.com/ zartbot/shallowsim.
- <span id="page-12-1"></span>[2] Weilin Cai, Le Qin, and Jiayi Huang. 2025. MoC-System: Efficient Fault Tolerance for Sparse Mixture-of-Experts Model Training. In Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2. 655-671.
- <span id="page-12-5"></span>[3] Zewen Chi, Li Dong, Shaohan Huang, Damai Dai, Shuming Ma, Barun Patra, Saksham Singhal, Payal Bajaj, Xia Song, Xian-Ling Mao, et al. 2022. On the representation collapse of sparse mixture of experts. Advances in Neural Information Processing Systems 35 (2022), 34600–34613.
- <span id="page-12-0"></span>[4] Jiangfei Duan, Ziang Song, Xupeng Miao, Xiaoli Xi, Dahua Lin, Harry Xu, Minjia Zhang, and Zhihao Jia. 2024. Parcae: Proactive, {Liveput-Optimized} {DNN} Training on Preemptible Instances. In 21st USENIX Symposium on Networked Systems Design and Implementation (NSDI 24). 1121–1139.

- <span id="page-13-18"></span>[5] William Fedus, Barret Zoph, and Noam Shazeer. 2022. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research 23, 120 (2022), 1–39.
- <span id="page-13-14"></span>[6] Swapnil Gandhi and Christos Kozyrakis. 2024. MoEtion: Efficient and Reliable Checkpointing for Mixture-of-Experts Models at Scale. arXiv preprint arXiv:2412.15411 (2024).
- <span id="page-13-28"></span>[7] Diandian Gu, Yihao Zhao, Yinmin Zhong, Yifan Xiong, Zhenhua Han, Peng Cheng, Fan Yang, Gang Huang, Xin Jin, and Xuanzhe Liu. 2023. ElasticFlow: An elastic serverless training platform for distributed deep learning. In Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2. 266–280.
- <span id="page-13-13"></span>[8] Jiaao He, Jidong Zhai, Tiago Antunes, Haojie Wang, Fuwen Luo, Shangfeng Shi, and Qin Li. 2022. Fastermoe: modeling and optimizing training of large-scale dynamic pre-trained models. In Proceedings of the 27th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming. 120–134.
- <span id="page-13-7"></span>[9] Tao He, Xue Li, Zhibin Wang, Kun Qian, Jingbo Xu, Wenyuan Yu, and Jingren Zhou. 2023. Unicron: Economizing self-healing llm training at scale. arXiv preprint arXiv:2401.00134 (2023).
- <span id="page-13-10"></span>[10] Changho Hwang, Wei Cui, Yifan Xiong, Ziyue Yang, Ze Liu, Han Hu, Zilong Wang, Rafael Salas, Jithin Jose, Prabhat Ram, et al. 2023. Tutel: Adaptive mixture-of-experts at scale. *Proceedings of Machine Learning and Systems* 5 (2023).
- <span id="page-13-29"></span>[11] Changho Hwang, Taehyun Kim, Sunghyun Kim, Jinwoo Shin, and KyoungSoo Park. 2021. Elastic resource sharing for distributed deep learning. In 18th USENIX Symposium on Networked Systems Design and Implementation (NSDI 21). 721–739.
- <span id="page-13-15"></span>[12] Sagar Imambi, Kolla Bhanu Prakash, and GR Kanagachidambaresan. 2021. PyTorch. Programming with TensorFlow: Solution for Edge Computing Applications (2021), 87–104.
- <span id="page-13-8"></span>[13] Insu Jang, Zhenning Yang, Zhen Zhang, Xin Jin, and Mosharaf Chowdhury. 2023. Oobleck: Resilient distributed training of large models using pipeline templates. In Proceedings of the 29th Symposium on Operating Systems Principles. 382–395.
- <span id="page-13-0"></span>[14] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. 2024. Mixtral of experts. arXiv preprint arXiv:2401.04088 (2024).
- <span id="page-13-21"></span>[15] Chenyu Jiang, Ye Tian, Zhen Jia, Shuai Zheng, Chuan Wu, and Yida Wang. 2024. Lancet: Accelerating Mixture-of-Experts Training via Whole Graph Computation-Communication Overlapping. arXiv preprint arXiv:2404.19429 (2024).
- <span id="page-13-4"></span>[16] Apostolos Kokolis, Michael Kuchnik, John Hoffman, Adithya Kumar, Parth Malani, Faye Ma, Zachary DeVito, Shubho Sengupta, Kalyan Saladi, and Carole-Jean Wu. 2025. Revisiting Reliability in Large-Scale Machine Learning Research Clusters. In 2025 IEEE International Symposium on High Performance Computer Architecture (HPCA). IEEE, 1259–1274.
- <span id="page-13-9"></span>[17] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. 2020. Gshard: Scaling giant models with conditional computation and automatic sharding. arXiv preprint arXiv:2006.16668 (2020).
- <span id="page-13-22"></span>[18] Jiamin Li, Yimin Jiang, Yibo Zhu, Cong Wang, and Hong Xu. 2023. Accelerating distributed {MoE} training and inference with lina. In 2023 USENIX Annual Technical Conference (USENIX ATC 23). 945–959.
- <span id="page-13-26"></span>[19] Jing Li, Zhijie Sun, Xuan He, Li Zeng, Yi Lin, Entong Li, Binfan Zheng, Rongqian Zhao, and Xin Chen. 2024. Locmoe: A low-overhead moe for large language model training. arXiv preprint arXiv:2401.13920 (2024).

- <span id="page-13-30"></span>[20] Jiamin Li, Hong Xu, Yibo Zhu, Zherui Liu, Chuanxiong Guo, and Cong Wang. 2022. Aryl: An elastic cluster scheduler for deep learning. arXiv preprint arXiv:2202.07896 (2022).
- <span id="page-13-1"></span>[21] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. 2024. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437 (2024).
- <span id="page-13-23"></span>[22] Juncai Liu, Jessie Hui Wang, and Yimin Jiang. 2023. Janus: A unified distributed training framework for sparse mixture-of-experts models. In Proceedings of the ACM SIGCOMM 2023 Conference. 486–498.
- <span id="page-13-2"></span>[23] llama4 2024. Meta Llama 4. https://ai.meta.com/blog/llama-4-multimodal-intelligence/.
- <span id="page-13-5"></span>[24] Kiwan Maeng, Shivam Bharuka, Isabel Gao, Mark Jeffrey, Vikram Saraph, Bor-Yiing Su, Caroline Trippel, Jiyan Yang, Mike Rabbat, Brandon Lucia, et al. 2021. Understanding and improving failure tolerant training for deep learning recommendation with partial recovery. Proceedings of Machine Learning and Systems 3 (2021), 637–651.
- <span id="page-13-19"></span>[25] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016. Pointer sentinel mixture models. arXiv preprint arXiv:1609.07843 (2016).
- <span id="page-13-17"></span>[26] nccl 2024. The NVIDIA Collective Communication Library (NCCL). https://developer.nvidia.com/nccl.
- <span id="page-13-11"></span>[27] Xiaonan Nie, Xupeng Miao, Zilong Wang, Zichao Yang, Jilong Xue, Lingxiao Ma, Gang Cao, and Bin Cui. 2023. Flexmoe: Scaling largescale sparse pre-trained model training via dynamic device placement. Proceedings of the ACM on Management of Data 1, 1 (2023), 1–19.
- <span id="page-13-24"></span>[28] Xiaonan Nie, Pinxue Zhao, Xupeng Miao, Tong Zhao, and Bin Cui. 2022. HetuMoE: An efficient trillion-scale mixture-of-expert distributed training system. arXiv preprint arXiv:2203.14685 (2022).
- <span id="page-13-31"></span>[29] Aurick Qiao, Sang Keun Choe, Suhas Jayaram Subramanya, Willie Neiswanger, Qirong Ho, Hao Zhang, Gregory R Ganger, and Eric P Xing. 2021. Pollux: Co-adaptive cluster scheduling for goodputoptimized deep learning. In 15th {USENIX} Symposium on Operating Systems Design and Implementation ({OSDI} 21).
- <span id="page-13-3"></span>[30] qwen3 2024. Alibaba Qwen3. https://github.com/QwenLM/Qwen3.
- <span id="page-13-12"></span>[31] Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, and Yuxiong He. 2022. Deepspeed-moe: Advancing mixture-of-experts inference and training to power next-generation ai scale. In *International* conference on machine learning. PMLR, 18332–18346.
- <span id="page-13-16"></span>[32] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. 2020. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 3505–3506.
- <span id="page-13-25"></span>[33] Shaohuai Shi, Xinglin Pan, Qiang Wang, Chengjian Liu, Xiaozhe Ren, Zhongzhe Hu, Yu Yang, Bo Li, and Xiaowen Chu. 2024. ScheMoE: An Extensible Mixture-of-Experts Distributed Training System with Tasks Scheduling. In Proceedings of the Nineteenth European Conference on Computer Systems. 236–249.
- <span id="page-13-20"></span>[34] Shaden Smith, Mostofa Patwary, Brandon Norick, Patrick LeGresley, Samyam Rajbhandari, Jared Casper, Zhun Liu, Shrimai Prabhumoye, George Zerveas, Vijay Korthikanti, et al. 2022. Using deepspeed and megatron to train megatron-turing nlg 530b, a large-scale generative language model. arXiv preprint arXiv:2201.11990 (2022).
- <span id="page-13-6"></span>[35] John Thorpe, Pengzhan Zhao, Jonathan Eyolfson, Yifan Qiao, Zhihao Jia, Minjia Zhang, Ravi Netravali, and Guoqing Harry Xu. 2023. Bamboo: Making preemptible instances resilient for affordable training of large {DNNs}. In 20th USENIX Symposium on Networked Systems Design and Implementation (NSDI 23). 497–513.
- <span id="page-13-27"></span>[36] torchelastic 2024. TorchElastic. https://pytorch.org/docs/stable/ distributed.elastic.html.

- <span id="page-14-4"></span>[37] Borui Wan, Mingji Han, Yiyao Sheng, Yanghua Peng, Haibin Lin, Mofan Zhang, Zhichao Lai, Menghan Yu, Junda Zhang, Zuquan Song, et al. 2024. ByteCheckpoint: A Unified Checkpointing System for Large Foundation Model Development. arXiv preprint arXiv:2407.20143 (2024).
- <span id="page-14-0"></span>[38] Yuxin Wang, Shaohuai Shi, Xin He, Zhenheng Tang, Xinglin Pan, Yang Zheng, Xiaoyu Wu, Amelie Chi Zhou, Bingsheng He, and Xiaowen Chu. 2023. Reliable and Efficient In-Memory Fault Tolerance of Large Language Model Pretraining. arXiv preprint arXiv:2310.12670 (2023).
- <span id="page-14-1"></span>[39] Zhuang Wang, Zhen Jia, Shuai Zheng, Zhen Zhang, Xinwei Fu, TS Eugene Ng, and Yida Wang. 2023. Gemini: Fast failure recovery in distributed training with in-memory checkpoints. In Proceedings of the 29th Symposium on Operating Systems Principles. 364–381.
- <span id="page-14-2"></span>[40] Mingshu Zhai, Jiaao He, Zixuan Ma, Zan Zong, Runqing Zhang, and Jidong Zhai. 2023. {SmartMoE}: Efficiently Training {Sparsely-Activated} Models through Combining Offline and Online Parallelization. In 2023 USENIX Annual Technical Conference (USENIX ATC 23). 961–975.
- <span id="page-14-3"></span>[41] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068 (2022).
- <span id="page-14-5"></span>[42] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric. P Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-asa-judge with MT-Bench and Chatbot Arena. arXiv:2306.05685 [cs.CL]
- <span id="page-14-8"></span>[43] Pengfei Zheng, Rui Pan, Tarannum Khan, Shivaram Venkataraman, and Aditya Akella. 2023. Shockwave: Fair and efficient cluster scheduling for dynamic adaptation in machine learning. In 20th USENIX Symposium on Networked Systems Design and Implementation (NSDI 23), 703–723.
- <span id="page-14-6"></span>[44] Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew M Dai, Quoc V Le, James Laudon, et al. 2022. Mixtureof-experts with expert choice routing. Advances in Neural Information Processing Systems 35 (2022), 7103–7114.
- [45] Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, and William Fedus. 2022. St-moe: Designing stable and transferable sparse expert models. arXiv preprint arXiv:2202.08906 (2022).
- <span id="page-14-7"></span>[46] Simiao Zuo, Xiaodong Liu, Jian Jiao, Young Jin Kim, Hany Hassan, Ruofei Zhang, Tuo Zhao, and Jianfeng Gao. 2021. Taming sparsely activated transformer with stochastic experts. arXiv preprint arXiv:2110.04260 (2021).

#### A Supplementary Material

# A.1 Training Performance under Ideal Failure Patterns for Tutel and Tutel(FT)

In Figure 14 we present the training performance under a single node failure every 5 minutes. We ensure that subsequent failed nodes are nodes that are previously dropped by Tutel and Tutel(FT), as the total number of nodes is not a multiple of EP size. We keep other settings the same as in §6.6.

In this case, after initial failures at 5 minutes, Tutel and Tutel(FT) essentially only encounter a failure every 10 minutes for GPT-S and GPT-M, due to an EP size of 2; every 20 minutes for GPT-L, due to an EP size of 4. In terms of total trained samples, Lazarus outperforms Tutel(FT) by 1.4x for GPT-S and by 2.6x for GPT-L.

# A.2 Proof of Optimality of the MRO Placement Plan

Recall the setting of our placement problem, we have N nodes, E experts, each node can hold c expert replicas. The i-th expert has  $r_i$  replicas. Assume there are R nodes alive simultaneously, we want to find a placement plan that maximizes the probability of recovering all the experts when the R alive nodes are sampled uniformly. We denote [k] as the set of  $\{1,2,\cdots,k\}$ . We use integer matrix  $T\in\mathbb{N}^{c\times N}$  to denote the placement plan,  $T_{ij}$  represents the expert placed at node j's i-th slot. T satisfies the following properties:

$$T_{ij} \in [E], \forall i \in [c], j \in [N]$$

$$r_k = \sum_{i=1}^c \sum_{j=1}^N \mathbb{1}_{T_{ij} = k}, \forall k \in [E]$$
(2)

Without loss of generality, we assume r is sorted in the ascending order,  $r_1 \le r_2 \le \cdots \le r_m$ . Let  $Col_j$  denote the set composed of elements in the j-th column of T(removing duplicates),  $j = 1, \cdots, N$ . Let A be the set of R random columns that are alive, A is uniformly sampled. Our goal is:

$$\max \Pr(\bigcup_{a \in A} Col_a = [E])$$
 (3)

Theorem 1. The **maximum rank** overlap placement plan (MRO plan) is defined as follows: [N] could be partitioned into  $\lceil \frac{E}{c} \rceil$  disjoint subsets:  $|S_i| = r_{1+(i-1)*c}, i \in \lceil \frac{E}{c} \rceil - 1 \rceil, |S_{\lceil \frac{E}{c} \rceil}| = \min\{N - \sum_{j=1}^{\lceil \frac{E}{c} \rceil - 1} r_{1+(\lceil \frac{E}{c} \rceil - 1)*c}, r_{1+(\lceil \frac{E}{c} \rceil - 1)*c}\}$ , such that, for  $\forall i \in \lceil \frac{E}{c} \rceil \rceil, j \in S_i, \{1+(i-1)*c, \cdots, \min\{i*c, E\}\} \subseteq Col_j$ . We prove that any MRO plan T maximizes  $\Pr(\bigcup_{a \in A} Col_a = \lfloor E \rfloor)$ .

PROOF. We first consider the simple case of  $E \le c$ .

Under this case, if  $N \le r_1 + R - 1$ , by Pigeonhole principle, apparently we have  $\Pr(\bigcup_{a \in A} Col_a = [E]) = 1$  for any MRO plan.

Otherwise  $N \le r_1 + R - 1$ , then  $|S_1| = r_1$ . For any placement plan T, the probability of recovering all experts is upper bounded by the probability of recovering expert 1:

$$\Pr(\bigcup_{a \in A} Col_a = [E]) \le \Pr(1 \in \bigcup_{a \in A} Col_a) \tag{4}$$

For any placement plan T, the probability of recovering expert 1 satisfies:

$$\Pr(1 \in \bigcup_{a \in A} Col_a) \le 1 - \frac{\binom{N - r_1}{R}}{\binom{N}{R}} \tag{5}$$

For any MRO plan, by definition, we have:

<span id="page-15-1"></span><span id="page-15-0"></span>
$$\{1, \cdots, E\} \subseteq Col_j, j \in S_1$$
 (6)

Therefore,

$$\Pr(1 \in \bigcup_{a \in A} Col_a) \ge \Pr(\bigcup_{a \in A} Col_a = [E]) \ge 1 - \frac{\binom{N - r_1}{R}}{\binom{N}{R}}$$
 (7)

Combining Inequality 5 and Inequality 7, we have: for  $E \le c$ , any MRO plan maximizes  $\Pr(\bigcup_{a \in A} Col_a = [E])$  and thus is optimal.

To prove the case of E > c, we first define two functions  $P_T(\cdot,\cdot,\cdot)$  and  $P_s(\cdot,\cdot,\cdot)$ .  $P_T$  is defined as:

<span id="page-15-2"></span>
$$P_T(M, n, r) = \Pr(\bigcup_{a \in A} Col_a \supseteq M)$$
(8)

where matrix  $T \in \mathbb{N}^{c \times n}$ , A is r columns randomly sampled from n columns , M is a subset of [E].  $P_T$  is used to illustrate the probability of recovering the subset M from a sub-matrix T.

For set M, we define M[j] as j-th smallest element in set M.  $P_s$  is defined as:

$$P_s(M, n, r) = \Pr(r \text{ samples cover the first } \lceil \frac{|M|}{c} \rceil \text{ segments of vector } v)$$
(9)

where vector v has length n, with consecutively  $\lceil \frac{|M|}{c} \rceil$  segments, the i-th segment has length  $L_{M,i} = r_{M[1+(i-1)*c]}$ ,  $i=1,\cdots,\lceil \frac{|M|}{c} \rceil -1, L_{M,\lceil \frac{|M|}{c} \rceil} = \min\{n-\sum_{j=1}^{\lceil \frac{|M|}{c} \rceil -1} L_{M,j}, r_{\lceil \frac{|M|}{c} \rceil}\}$ .  $P_s$  is defined to illustrate the recover probability of MRO plans.

We prove the optimality of MRO plan when E > c by mathematical induction. We first have the following assumption:

<span id="page-16-0"></span>![](_page_16_Figure_0.jpeg)

Figure 14: [Comparison with Tutel]: Throughput and total trained samples with a single node fails every 5 minutes, where subsequent failed nodes are unused nodes for Tutel and Tutel(FT).

<span id="page-16-2"></span>Assumption 1.  $\forall m' < E, \forall n', r', \forall \text{ set } M', |M'| = m',$ 

$$\max_{T} P_{T}(M', n', r') = P_{s}(M', n', r')$$
 (10)

We want to prove that for  $\forall |M| = E, \forall N, R$ ,

$$\max_{T} P_T(M, N, R) = P_s(M, N, R)$$
(11)

Proving Equation 11 indicates that any MRO plan achieves optimal recover probability across all different T.

We first consider the case of |M| > c. First if R = 1, |M| > c, for  $\forall T$ ,  $P_T(M, N, R) = 0$ ,  $P_s(M, N, R) = 0$ , the claim trivially satisfies.

When R > 1, |M| > c, for  $\forall T$ , we can transform T to T' by reordering the columns to let the columns containing 1 be the first consecutive columns. And  $\forall T$  we have:

$$P_T(M, N, R) = P_{T'}(M, N, R)$$
 (12)

Let A' as the set of R columns randomly sampled on T',  $S_t$  be the set of different values of column t of matrix T', C is the largest column ID of T' that contains 1. By conditioning on t, we have:

$$P_{T'}(M, N, R) = \sum_{t=1}^{C} \Pr(\min A' = t) \Pr(\bigcup_{a \in A' \setminus \{t\}} Col_a \supseteq M \setminus S_t | \min A' = t)$$
(13)

If we consider T'' as the sub-table of T' composed of its last N-t rows, we have:

$$\Pr(\bigcup_{a \in A' \setminus \{t\}} Col_a \supseteq M \setminus S_t | \min A' = t) \le \max_{T''} P_{T''}(M \setminus S_t, N - t, R - 1)$$
(14)

By Assumption 1, due to  $S_t \neq \emptyset$ , we have:

$$\max_{T''} P_{T''}(M \setminus S_t, N - t, R - 1) = P_s(M \setminus S_t, N - t, R - 1)$$
(15)

Recall Equation 13, we have:

$$P_{T'}(M, N, R) \le \sum_{t=1}^{r_{M[1]}} \Pr(\min A' = t) P_s(M \setminus S_t, N - t, R - 1)$$
(16)

<span id="page-16-1"></span>To upper bound  $P_{T'}(M, N, R)$ , we have to upper bound  $P_s(M \setminus S_t, N - t, R - 1)$ . We first prove the following proposition:

<span id="page-16-5"></span>PROPOSITION 1. Denote  $Min_cM$  as the smallest c elements of M. For  $\forall M$ , we have:

<span id="page-16-4"></span>
$$Min_c M = \arg\max_{S_t} P_s(M \setminus S_t, N - t, R - 1)$$
 (17)

It is apparent that removing elements from the recover target set results in an increase of  $P_s$ . Therefore, if  $|S_t| < c$ ,  $\forall s \neq S_t$ ,

$$P_s(M \setminus (S_t \cup s), N - t, R - 1) \ge P_s(M \setminus S_t, N - t, R - 1)$$
(18)

Therefore the set  $S_t$  that maximizes  $P_s(M \setminus S_t, N-t, R-1)$  must have c cardinality.

<span id="page-16-3"></span>Consider  $|S_t| = c$ . If  $S_t$  is not the smallest c elements of M, we substitute an element in  $S_t$  with a smaller element obtaining  $S'_t$ ,  $|S'_t| = c$ . By the property of rankings, we have,

$$L_{M \setminus S'_t, i} \ge L_{M \setminus S_t, i}, \forall i$$
 (19)

Therefore,  $\forall S'_t$  obtained by this way,

$$P_s(M \setminus S'_t, N - t, R - 1) \ge P_s(M \setminus S_t, N - t, R - 1)$$
 (20)

We recursively apply this substitution and obtains  $Min_c M$ , therefore, for  $\forall S_t$ , we have:

$$P_s(M \setminus \mathsf{Min}_c M, N - t, R - 1) \ge P_s(M \setminus S_t, N - t, R - 1)$$
(21)

Thus finishes the proof of the proposition. This proposition tells us that  $S_t = \text{Min}_c M$  maximizes  $P_s(M \setminus S_t, N-t, R-1)$ . By Equation 16 and Proposition 1, we have,

$$P_{T}(M, N, R) \le \sum_{t=1}^{r_{M[1]}} \Pr(\min A' = t) P_{s}(M \setminus \text{Min}_{c}M, N - t, R - 1)$$
(22)

For  $P_s(M, N, R)$ , consider the left most sample should fall on the first segment, and the other R-1 samples should cover the set M', where M' satisfies the j-th segment of M' has equal length with the j+1-th segment of M for  $\forall j$ . Therefore  $M' = \{M[1+c], \cdots, M[|M|]\}$ .

$$P_{s}(M, N, R) = \sum_{t=1}^{r_{M[1]}} \Pr(\min A' = t) P_{s}(M', N - t, R - 1)$$

$$= \sum_{t=1}^{r_{M[1]}} \Pr(\min A' = t) P_{s}(\{M[1 + c], \dots, M[|M|]\}, N - t, R - 1)$$

$$= \sum_{t=1}^{r_{M[1]}} \Pr(\min A' = t) P_{s}(M \setminus \text{Min}_{c}M, N - t, R - 1)$$
(23)

Substituting Equation 23 into Inequality 22, we have:

<span id="page-17-2"></span>
$$P_T(M, N, R) \le P_s(M, N, R) \tag{24}$$

Now we have proven that  $P_s$  is an upper bound of  $P_T$ . Next, we prove that if T is a MRO plan, Inequality 24 can actually achieve equal. For  $\forall$  MRO plan  $T^*$ , we have:

$$\bigcup_{a \in A} Col_a = [E] \iff A \text{ covers } S_i, \forall i \in \{1, \cdots, \lceil \frac{E}{c} \rceil\}$$
 (25)

For  $\forall$  MRO plan  $T^*$ , we can reorder the columns so that for each column set  $S_i$ , all columns in  $S_i$  are consecutive. We denote the reordered MRO plan as T', and the randomly sampled columns on T' as A'.

$$\Pr(\bigcup_{a \in A'} Col_a = [E])$$

=Pr(A' covers segment with length  $|S_i|, \forall i \in \{1, \dots, \lceil \frac{m}{c} \rceil \}$ ) = $P_s(M, N, R)$ 

(26)

Therefore for  $T^*$  which is a MRO plan, by the definition of  $P_T$  in Equation 8, we have:

<span id="page-17-3"></span>
$$P_{T^*}(M, N, R) = P_s(M, N, R)$$
 (27)

Equation 27 indicates that  $\exists$ MRO plan  $T^*$ ,  $P_{T^*}(M, N, R) = P_s(M, N, R)$ , hence we prove that, under Assumption 1, Equation 11 holds when E > c.

Assumption 1 trivially holds due to the optimality of MRO plan when  $E \le c$ .

<span id="page-17-1"></span>By mathematical reduction, for  $\forall E, \forall |M| = E, \forall N, R$ , we have,

$$\max_{T} P_T(M, N, R) = P_s(M, N, R)$$
 (28)

Furthermore, for  $\forall$  MRO plan  $T^*$  we have:

<span id="page-17-0"></span>
$$P_{T^*}([E], N, R) = \max_{T} \Pr(\bigcup_{a \in A} Col_a = [E])$$
(29)

П