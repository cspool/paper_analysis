# FineMoE: Fine-grained Load Balancing for Mixture-of-Experts with Token Scheduling

Chenqi Zhao Peking University

Wenfei Wu Peking University

Yuchen Xu

Peking University

Linhai Song Institute of Computing Technology, Chinese Academy of Sciences

Yitao Yuan Peking University

# Abstract

Mixture-of-Experts (MoE) has emerged as a promising approach to scale up deep learning models due to its significant reduction in computational resources. However, the dynamic nature of MoE leads to load imbalance among experts, severely impacting training efficiency. While previous research has attempted to address the load balancing challenge, existing solutions either compromise model accuracy or introduce additional system overhead. As a result, they fail to achieve efficient and fine-grained load balancing, which is crucial to optimizing training efficiency.

We propose FineEP, a novel parallelization strategy to achieve fine-grained load balancing in MoE systems. FineEP is capable of achieving complete load balancing in every micro-batch through efficient token scheduling across GPUs. Furthermore, we propose FineMoE, an efficient distributed MoE training system with FineEP's load balancing capabilities. Our experimental results demonstrate that FineMoE improves the end-to-end training throughput by up to 47.6% compared with the state-of-the-art system, and almost consistently achieves complete load balance among GPUs.

# 1 Introduction

In recent years, transformer-based large models have fueled significant advancements across various domains, including natural language processing [\[6,](#page-12-0) [43,](#page-15-0) [53\]](#page-15-1), computer vision [\[31,](#page-14-0) [47\]](#page-15-2), and multimodality [\[3,](#page-12-1) [32\]](#page-14-1). These achievements underscore that scaling up model size is a straightforward yet fundamental approach to enhancing model performance. Nevertheless, deploying large models necessitates substantial computational resources. To tackle this, Mixture-of-Experts (MoE) proposes a sparsely-activated architecture [\[49\]](#page-15-3), enabling the scaling up of model capacity with only a sub-linear increase in computational cost. Nowadays, MoE has gained widespread adoption in state-of-the-art models, such as GPT-5 [\[37\]](#page-14-2), Llama 4 [\[33\]](#page-14-3), Gemini 3 [\[14\]](#page-13-0), Grok-4 [\[59\]](#page-15-4), Claude

4 [\[1\]](#page-12-2), Qwen3 [\[61\]](#page-15-5), and DeepSeek-V3 [\[9\]](#page-13-1).

However, the sparse nature of MoE can indeed introduce inefficiencies during training. MoE divides a model layer into multiple *experts* and routes each input token to its *top-K* most suitable experts at runtime. This dynamic routing can result in significant load imbalances among experts [\[18,](#page-13-2) [25\]](#page-14-4). Furthermore, recent distributed training frameworks commonly employ *expert parallelism* (EP) to distribute experts across multiple GPUs [\[11\]](#page-13-3). While EP reduces memory consumption per GPU, it introduces a *straggler* problem: All GPUs must wait for the most heavily loaded GPU in the EP group to complete computation before proceeding, resulting in wasted computational resources and reduced system throughput. Therefore, the load imbalance issue poses a significant challenge in efficiently training MoE models.

To address the load imbalance issue of MoE, many works propose *algorithmic solutions* by modifying the token-toexpert routing algorithm. These approaches incorporate loadbalancing loss [\[10,](#page-13-4) [11,](#page-13-3) [54\]](#page-15-6), route tokens to less suitable experts [\[28,](#page-14-5) [66\]](#page-16-0), or simply drop excess tokens to alleviate the burden on heavily loaded experts [\[11,](#page-13-3) [25,](#page-14-4) [28\]](#page-14-5). However, these modifications to model logic can degrade model accuracy and potentially impact model convergence [\[15,](#page-13-5) [55,](#page-15-7) [60\]](#page-15-8). As a result, they introduce a trade-off between model accuracy and system efficiency [\[42,](#page-14-6) [52\]](#page-15-9).

Other works propose *systematic solutions* for load balancing in MoE without compromising the model accuracy. A series of works balance GPU loads through *expert scheduling*, adjusting the expert-to-GPU mapping [\[36,](#page-14-7) [41,](#page-14-8) [51,](#page-15-10) [55,](#page-15-7) [56,](#page-15-11) [63\]](#page-16-1). For instance, FlexMoE [\[36\]](#page-14-7) adaptively replicates experts based on their popularity, distributing loads of more popular experts to more GPUs. However, these solutions cannot effectively achieve ideal load balance, which should satisfy two fundamental requirements:

R1: Maximizing system efficiency requires fine-grained load balance among all GPUs. Giant model sizes demand numerous GPUs for training [\[9,](#page-13-1) [52\]](#page-15-9), making even minor load imbalances costly in terms of wasted GPU time. The significant expense necessitates a fine granularity for load balancing, hopefully equalizing all GPU loads. However, existing expert scheduling solutions typically treat each expert replica as a scheduling unit for communication efficiency [\[36,](#page-14-7) [41,](#page-14-8) [55,](#page-15-7) [63\]](#page-16-1). Since scheduling at the expert granularity results in a limited, discrete scheduling space, they fail to achieve optimal load balance across a vast, near-continuous space of possible load distributions.

R2: Dynamic expert load distributions require microbatch-level adaptation at runtime. Data heterogeneity across micro-batches causes severe variations in expert loads, necessitating dynamic load balancing to optimize GPU utilization consistently. Existing expert scheduling solutions involve migrating expert parameters to align with the changing load distributions [\[36,](#page-14-7) [55,](#page-15-7) [63\]](#page-16-1). However, the substantial size of these parameters presents a significant challenge to making frequent adjustments seamlessly.

To meet these requirements, we adopt a novel systematic approach by *scheduling input tokens* instead of experts, which has two advantages: (R1) First, token scheduling allows a precise control over per-GPU workloads (relative to token counts), enabling fine-grained load balancing. (R2) Second, token scheduling leverages existing all-to-all communication in EP, incurring minimal communication overhead. This allows us to achieve dynamic load balancing within each micro-batch without compromising the entire system efficiency. Furthermore, we can combine fine-grained token scheduling and coarse-grained expert scheduling, improving overall load balancing capability.

However, performing token scheduling in existing EP frameworks faces two challenges. (C1) Current frameworks dispatch tokens within an EP group, which contains exactly one replica of each expert [\[46,](#page-15-12)[50\]](#page-15-13). As a result, the GPU loads are fixed by expert loads, which are determined by the routing algorithm. To tackle this, we schedule tokens among expert replicas across *multiple EP groups*, balancing GPU loads while keeping expert loads. (C2) Identical expert placement across EP groups severely limits the scheduling space (see Figure [3b\)](#page-3-0). Since replicas of a popular expert have identical EP ranks in all EP groups, naïve token scheduling cannot migrate their loads to other EP ranks, resulting in only local load balance. Although some studies adopt non-identical expert placement [\[36\]](#page-14-7), the relationship between expert placement and token scheduling remains uninvestigated. To tackle this, we theoretically analyze the influence of expert placement on token scheduling with *graph theory* ([§6.1\)](#page-6-0). Based on the analysis, we tailor the *expert placement* to expand the scheduling space and achieve global load balance.

We propose FineEP, a novel EP strategy that leverages token scheduling for fine-grained load balancing across GPUs. FineEP optimizes load balancing from two perspectives: (1) From a short-term perspective, FineEP schedules tokens within every micro-batch using *linear programming* to minimize the maximum GPU loads ([§5.1\)](#page-4-0). To optimize system efficiency, FineEP reduces communication overhead through *locality-aware routing* ([§5.2\)](#page-5-0) and minimizes scheduling latency through *overlapping* ([§5.4\)](#page-5-1). (2) From a long-term perspective, FineEP tailors *expert placement* to optimize its token scheduling performance and mitigate coarse-grained imbalances. Based on our theoretical analysis, FineEP incorporates multiple expert placement strategies ([§6.2-](#page-6-1)[6.3\)](#page-7-0) and an *adaptive replacement* mechanism ([§6.4\)](#page-7-1). These optimizations enable FineEP to maintain complete load balance under dynamic and highly skewed workloads.

Based on FineEP, we design FineMoE, an efficient distributed MoE training system. We implement FineMoE upon Megatron-LM [\[50\]](#page-15-13) and evaluate its load balancing performance. Experimental results show that FineMoE achieves up to 47.6% improvement on end-to-end throughput compared with Megatron-LM and maintains complete load balance even under highly imbalanced workloads.

Briefly, we make the following contributions:

- We propose FineEP to achieve dynamic and fine-grained load balancing among GPUs with token scheduling ([§4\)](#page-3-1).
- We formulate the load balancing problem into a linear programming problem and solve it efficiently ([§5\)](#page-4-1).
- We propose several expert placement strategies and an adaptive replacement mechanism to optimize FineEP's load balancing capacity ([§6\)](#page-6-2).
- We implement FineMoE, an efficient distributed MoE training system based on FineEP. We conduct comprehensive experiments to demonstrate that FineMoE exhibits superior system efficiency and load balancing capability ([§7\)](#page-8-0).

# 2 Background

# 2.1 Transformer and MoE

Transformer is the state-of-the-art architecture for deep learning models [\[53\]](#page-15-1). A transformer model consists of multiple sequential layers, each comprising an attention layer and a feed-forward network (FFN). Figure [1a](#page-2-0) illustrates the architecture of a standard (dense) transformer layer.

Figure [1b](#page-2-0) illustrates an MoE transformer layer [\[11\]](#page-13-3). MoE replaces the FFN with multiple *expert* FFNs, coordinated by a *gate* network. After the attention computation, the gate network assigns each token to its *top-K* most suitable experts. Then, these selected experts process the token and aggregate their results to produce the final result.

# 2.2 Expert Parallelism and Expert Data Parallelism

The increasing scale of deep learning models has necessitated the development of various parallelization strategies for dis-

<span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

(c) An MoE transformer layer with expert parallelism.

Figure 1: An example of transformer, MoE, and expert parallelism.

tributed training across multiple computational devices. We introduce some of them as follows.

Data parallelism (DP) is a fundamental parallelization strategy that partitions input data across devices to enhance training throughput. In the forward pass, each DP rank (e.g., a device) maintains a complete copy of the model parameters while processing distinct *micro-batches* of data independently. In the backward pass, all devices in a DP group aggregate their gradients for parameter synchronization. Recent advances, such as ZeRO [45], have further optimized DP's memory efficiency by eliminating memory redundancies across devices.

Tensor parallelism (TP) addresses per-device memory constraints by partitioning model parameter tensors across devices. Each TP rank processes complete inputs using its allocated slice of model parameters, with results aggregated within the TP group to complete the computation.

Pipeline parallelism (PP) addresses memory constraints by partitioning model layers [20, 34]. While PP introduces lower communication overhead compared with TP, it wastes some computational resources due to pipeline bubbles [20, 40].

Expert parallelism (EP) is a widely employed technique in distributed MoE systems [11]. EP is a specialized hybrid parallelization strategy combining elements of DP and TP. By exploiting the inherent sparsity of MoE, EP benefits from both DP's training efficiency and TP's memory efficiency. As shown in Figure 1c, EP partitions the FFN layer by distributing distinct experts across multiple devices. At runtime, the attention layer and the gate network utilize conventional DP to process different tokens on separate devices. Next, devices within the EP group perform an all-to-all communication operation to dispatch tokens towards their designated expert devices. Then, each device processes its received tokens with its local experts. Finally, the devices perform another all-to-all operation to return tokens to their original devices.

Expert data parallelism (EDP) is a specialized form of

<span id="page-2-1"></span>![](_page_2_Figure_9.jpeg)

Figure 2: Expert load distribution of GPT 32×1.3B layer 20 in some training iterations.

DP emerging from EP. In configurations where the DP degree (number of DP ranks) exceeds the EP degree, systems must employ EDP across multiple EP groups within a DP group. Each EDP group consists of devices sharing the same EP rank across different EP groups. Devices within an EDP group maintain replicated instances of identical experts while processing distinct tokens from different EP groups. The synchronization of parameters and gradients among these expert replicas follows conventional DP mechanisms, ensuring consistency across the distributed system.

#### 2.3 Load Imbalances in MoE

The dynamic token-to-expert assignment in MoE systems inherently results in significant load imbalances across experts, presenting a critical performance bottleneck for distributed training. We conduct an experiment to trace the expert load distribution in a training process. As illustrated in the left part of Figure 2, the expert load distribution exhibits substantial variability and skewness, particularly during the initial training iterations.

The imbalance issue severely impacts resource utilization and training throughput in MoE systems with EP. In current MoE implementations, the FFN computation time of a GPU is approximately proportional to the total number of tokens assigned to the experts on this GPU [13]. Since GPUs within an EP group must synchronize through all-to-all communication both before and after FFN computation, all GPUs must wait until the slowest GPU completes computation before proceeding. Consequently, the system's overall throughput is bottlenecked by the most heavily loaded device, namely the *straggler*. This strict synchronization requirement emphasizes the necessity of effective load balancing strategies to optimize MoE training efficiency.

Moreover, the straggler effect continuously degrades training efficiency in *every micro-batch*. As shown in the right part of Figure 2, **expert load distribution fluctuates significantly between consecutive micro-batches**. This dynamic nature of load imbalances necessitates fine-grained load balancing at the micro-batch level to maintain optimal training efficiency.

<span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

(a) Vanilla EP. The DP degree is 4, and the EP degree is 2.

![](_page_3_Figure_2.jpeg)

(b) Merging two EP groups. Afterward, GPU loads within EDP groups  $\{0,2\}$  and  $\{1,3\}$  are equal.

![](_page_3_Figure_4.jpeg)

(c) FineEP, shuffling expert placement and scheduling replica loads. Afterward, all GPU loads are equal.

Figure 3: Converting EP to FineEP. The shape and color of a symbol indicate the source GPU and the assigned expert of a token. The bottom curves indicate EDP groups.

#### 3 Challenges in Token Scheduling

We aim to achieve fine-grained load balancing for MoE systems through *token scheduling* within every micro-batch. However, this approach poses two fundamental challenges:

**Challenge 1:** Vanilla EP restricts token dispatching within each EP group, providing no space for scheduling.

To perform token scheduling for load balancing, we must first determine the *scheduling space*. Specifically, when a token is assigned to an expert, we must be able to compute it on one of *multiple* GPUs.

Unfortunately, such scheduling space is unattainable in the vanilla EP paradigm. As shown in Figure 3a, vanilla EP restricts token dispatching within the scope of an EP group, where each EP group contains exactly one replica of each expert's parameters. When a token is assigned to an expert, it must be computed on the GPU hosting that expert's replica in its EP group. Consequently, the token-to-GPU mapping is fixed by the token-to-expert mapping, eliminating any space for computation scheduling.

**Solution 1:** Our key observation is that we can find scheduling space through **expert data parallelism**. We call the set of

<span id="page-3-2"></span>![](_page_3_Figure_13.jpeg)

Figure 4: FineMoE architecture.

GPUs that host replicas of expert *e* as the EDP group of expert *e*. Since all replicas of an expert maintain identical parameters, a token can be computed equivalently with any replica in its designated expert's EDP group. This observation inspires us to merge multiple EP groups and schedule tokens across their experts' EDP groups, as shown in Figure 3b.

**Challenge 2:** *Identical expert placement across EP groups results in constrained scheduling space.* 

While this approach improves token distribution, it falls short of optimal load balancing. The fundamental limitation lies in the restricted scheduling space—load balancing occurs only within individual EDP groups. In the current EP paradigm, each EP group maintains identical expert placement. Consequently, the EDP groups of different experts are either completely disjoint or identical (e.g., in Figure 3b, the EDP groups of expert 0,1 are both GPU {0,2}, while the EDP groups of expert 2,3 are both GPU {1,3}). This constraint indicates that we can at most equalize workloads within each EDP group, while significant load imbalances may persist across different EDP groups.

**Solution 2:** To expand the scheduling space, we **shuffle** the expert placement within the merged EP groups. This operation creates intersecting EDP groups across different experts, substantially enlarging the scheduling space. As illustrated in Figure 3c, this approach can potentially achieve complete load balance across all GPUs.

#### <span id="page-3-1"></span>4 FineMoE Overview

We introduce a novel expert parallelism strategy, namely FineEP. FineEP optimizes load balancing across GPUs within *every micro-batch* through *token scheduling* across *EDP groups*.

Based on FineEP, we propose FineMoE, an efficient distributed MoE training system. Figure 4 illustrates the architecture of FineMoE. Following PP, each device hosts multiple model layers. Each MoE layer consists of a *token dispatcher* and multiple local expert replicas. Expert replica placement is coordinated by a global *placement manager* residing on device 0. We briefly illustrate the workflow of FineMoE as

follows.

**Prerequisites.** FineEP obtains scheduling space from intersecting EDP groups, which require two conditions: the DP degree must exceed the EP degree, and each GPU must host multiple expert replicas. We find that these requirements are satisfied in many open-sourced MoE models [9,52]. For example, DeepSeek-V3's pre-training configuration employs 64-way EP and 128-way DP, with each GPU hosting 8 experts [9].

**Configuration.** FineEP introduces an integer parameter d, where  $1 < d \le \frac{\mathrm{DP\_degree}}{\mathrm{EP\_degree}}$ . FineEP merges every d EP groups into a FineEP group. In the following sections, we consider the simplest case where  $d = \frac{\mathrm{DP\_degree}}{\mathrm{EP\_degree}}$ , i.e., the entire DP group forms a single FineEP group.

**Initialization.** During model initialization, the placement manager generates expert placements within all FineEP groups and broadcasts the placements to all devices. The placement strategies are detailed in §6.

**Runtime.** During model training, FineEP balances GPU workloads by scheduling tokens within the FineEP group. Specifically, after the gate networks assign tokens to specific experts, the dispatchers route tokens to particular expert replicas within their respective EDP groups. The scheduling algorithm is detailed in §5.

After scheduling, the dispatchers initiate an all-to-all (dispatch) operation within the FineEP group, sending tokens to their routed expert replicas. Then, each GPU executes FFN computation to process received tokens using its local expert replicas. Afterward, a second all-to-all (combine) operation returns processed tokens to their source GPUs. These operations follow the same patterns as vanilla EP, but occur in a communication group d times larger.

An example. Surprisingly, FineEP can almost always achieve complete load balance among GPUs, unless the expert load distribution is extremely skewed (as shown in §7.3). We can intuitively learn how FineEP achieves such good performance from the example in Figure 3c. Since expert 3 is heavily loaded, it places 8 tokens on GPU 3, preventing expert 0 from placing any token on the same GPU. Fortunately, expert 0 can place all its 4 tokens on GPU 0. Similarly, expert 2 also "shifts" some load from GPU 2 to GPU 1, while expert 1 shifts some load from GPU 1 to GPU 0. Consequently, the computational burden of heavily loaded experts is effectively distributed across the entire FineEP group.

#### <span id="page-4-1"></span>5 Token Scheduling

Token scheduling is the core of FineEP for fine-grained load balancing. The scheduling algorithm runs in two steps: The first step is to distribute expert loads among their replicas, balancing GPU loads (§5.1). The second step is to route tokens to specific expert replicas, enforcing the calculated replica loads (§5.2). Then, we discuss where and when to

Table 1: List of Symbols and Notations.

| Symbol              | Description                                            |
|---------------------|--------------------------------------------------------|
| E                   | set of all experts                                     |
| $G_{FineEP}$        | set of GPUs in the concerned FineEP group              |
| d                   | a parameter = $\frac{ G_{FineEP} }{\text{EP\_degree}}$ |
| $G_{EDP}^{e}$       | EDP group of expert <i>e</i>                           |
| load <sub>e</sub>   | total load of expert e                                 |
| $x_e^g$             | replica load of expert e on GPU g                      |
| in put <sub>e</sub> | input load of expert e from GPU g                      |
| m                   | optimal objective value of LPP 1                       |

launch the scheduling in §5.3 and §5.4. Finally, we discuss some additional scheduling optimizations in Appendix A.

#### <span id="page-4-0"></span>5.1 Determining Replica Loads

We aim to balance GPU loads through optimal distribution of expert loads among their replicas. We formulate the load balancing problem as an optimization problem.

- *Notations*. Let E represent the set of all experts,  $G_{FineEP}$  represent the set of GPUs in the concerned FineEP group, and  $G_{EDP}^e$  represent the set of GPUs in the EDP group of expert e. Since we currently focus on a single FineEP group, we assume that  $G_{EDP}^e \subseteq G_{FineEP}, \forall e \in E$ .
- *Variables*. The variables are  $\{x_e^g: e \in E, g \in G_{EDP}^e\}$ , where  $x_e^g$  is the replica load of expert e on GPU g.
- Constraints. Let  $load_e(e \in E)$  denote the total load (number of tokens) of expert e in the concerned FineEP group. A valid distribution must ensure that each expert distributes its total load across its replicas.
- *Objective*. Due to synchronization requirements for all-to-all communication before and after expert computation, the GPU with the highest load becomes the performance bottleneck. Therefore, we should minimize the maximum GPU load across the FineEP group.

We formulate the optimization problem as follows.

<span id="page-4-2"></span>
$$\begin{aligned} & \text{minimize} \max_{g \in G_{FineEP}} \left\{ \sum_{e \in E: g \in G_{EDP}^e} x_e^g \right\}, \\ & \text{subject to} \sum_{g \in G_{EDP}^e} x_e^g = load_e, \quad \forall e \in E, \\ & x_e^g \geq 0, \quad \forall e \in E, g \in G_{EDP}^e. \end{aligned}$$

Problem 1 is a *linear programming problem* (LPP), which can be efficiently solved in polynomial time relative to the number of GPUs and experts. The number of variables is O(|E|d), and the number of constraints is  $O(|E| + |G_{FineEP}|)$ . Given its modest scale, we solve this LPP using a single CPU

thread with the HiGHs solver [21]. GPU acceleration or multithreading would not yield significant performance benefits for this scale of optimization.

Notably, across different micro-batches, while the constraint matrix (determined by expert placement  $G_{EDP}^{e}$ ) remains the same, only the constraint bounds ( $load_{e}$ ) vary. This property enables the *warm-start* of the LPP solving by reusing the immediate states of the previous solution, significantly reducing optimization overhead.

## <span id="page-5-0"></span>5.2 Routing Tokens to Replicas

Our next step is routing tokens to specific expert replicas, enforcing the calculated replica loads  $(x_e^g)$ . Specifically, we determine this *token-to-replica* mapping using a sequential routing strategy: First, we arrange tokens assigned to expert e from all GPUs in sequence, along with an ordered list of expert e's replicas. Then, we iterate through these tokens, routing each to the first replica that has not yet reached its allocated load  $x_e^g$ .

To enhance efficiency, we can manipulate token ranges rather than individual tokens. Let  $input_e^g$  denote the input load of expert e from GPU g, i.e., the number of tokens on GPU g assigned to expert e ( $\sum_{g \in G_{FineEP}} input_e^g = load_e$ ). The pseudo code of token routing is shown in Algorithm 1, Lines 10-16. **Locality-aware routing.** Previous research has underscored that all-to-all communication introduces significant overhead to MoE systems [18, 22, 27]. To reduce the all-to-all communication volume, we can leverage data locality during token routing. Specifically, when GPU g holds a replica of expert e, we prioritize routing tokens from GPU g to its local expert replica before considering remote replicas. The pseudo code is shown in Algorithm 1, Lines 4-9.

Communication-aware scheduling. Furthermore, we can consider communication overhead during scheduling. Specifically, we can reformulate the optimization problem to minimize the overall maximum execution time, incorporating both computation and communication. Moreover, we can model the heterogeneity between intra-node and inter-node communication. Due to space limits, the modified optimization problem is detailed in Appendix A.1.

#### <span id="page-5-2"></span>**5.3** Distributed Scheduling across Devices

As described in §5.1 and §5.2, the scheduling algorithm requires global load information ( $input_e^g$ ) across the entire FineEP group to generate a token dispatching plan. This raises the question of where to place the scheduler.

We consider two candidate locations for the scheduler: *centralized* on one device or *distributed* across all devices. A centralized scheduler needs to gather load information from all devices, perform the scheduling, and scatter the results. Alternatively, distributed schedulers require all devices to perform an all-gather operation to collect global load information

Algorithm 1: Routing tokens to expert replicas.

```
Input: \{input_e^g\}, \{x_e^g\}
 1 \{remain\_input_e^g\} = \{input_e^g\}
   \{remain\_x_e^g\} = \{x_e^g\}
3 for e \in E do
       // First, route local tokens to local replicas.
5
        for g \in G_{EDP}^e do
            y = \min(remain\_input_e^g, remain\_x_e^g)
 6
             route the next y tokens of expert e from GPU g
 7
              to the replica on GPU g
             remain\_input_e^g \leftarrow remain\_input_e^g - y
 8
            remain x_e^g \leftarrow remain x_e^g - y
 9
        // Then, route global tokens to global replicas.
10
        for g ∈ G_{FineEP} do
11
            for g' \in G_{EDP}^e do
12
                 y = \min(remain\_input_e^g, remain\_x_e^{g'})
13
                 route the next y tokens of expert e from
14
                  GPU g to the replica on GPU g'
                 remain\_input_e^g \leftarrow remain\_input_e^g - y
15
                 remain x_e^{g'} \leftarrow remain x_e^{g'} - y
```

and execute the scheduling algorithm independently. <sup>1</sup>

We choose the distributed approach for better scalability because it requires only one communication operation compared to two operations in the centralized approach. Since the load information is small, the primary performance factor is latency rather than throughput, making fewer communication operations advantageous.

#### <span id="page-5-1"></span>5.4 Overlapping Scheduling to Hide Latency

Since we need to execute token scheduling in every microbatch, minimizing the scheduling overhead is significant for system efficiency. Therefore, we reduce the scheduling overhead by *overlapping* scheduling with other operations.

As described in §4, the scheduling executes immediately after the gate network and before the all-to-all communication. We observe that existing distributed training frameworks usually perform some operations during this period. For instance, Megatron-LM [50] executes a token permutation operation to replicate tokens by top-K times and sort tokens by expert indices before all-to-all dispatching. Therefore, we can overlap the scheduling on CPUs with the permutation on GPUs. For frameworks without suitable overlapping operations, we propose a *pipelining* mechanism to hide scheduling latency, which is detailed in Appendix A.2.

<span id="page-5-4"></span><sup>&</sup>lt;sup>1</sup>The distributed scheduling approach maintains consistency because FineEP's scheduling algorithm is deterministic.

<span id="page-6-3"></span>![](_page_6_Figure_0.jpeg)

Figure 5: The graph abstraction of an example expert placement. Color bars indicate edges (experts) between vertices (GPUs).  $G_{max}$  contains GPU 0, 3. Expert 0 is entirely in  $G_{max}$ . Experts 1,3 partially intersect with  $G_{max}$  and cannot distribute any load within  $G_{max}$ .

### <span id="page-6-2"></span>6 Expert Placement

Expert placement is fundamental to the load balancing performance of token scheduling in FineEP. Figure 6a shows the relationship between expert placement and token scheduling in MicroMoE. Specifically, according to LPP 1, the expert placement determines the EDP groups ( $G_{EDP}^e$ ), which are the key components of the constraint matrix and thus significantly affect the optimization result. Although our experiments demonstrate that some simple placement methods, such as random shuffling, usually provide acceptable results (§7.3), identifying optimal placement strategies is still crucial for maximizing system performance. In the following sections, we first analyze "What is an optimal expert placement?" in §6.1 and then discuss "How to construct an optimal expert placement?" in §6.2 and §6.3. Finally, we propose an adaptive replacement mechanism in §6.4.

#### <span id="page-6-0"></span>6.1 Analysis of Optimal Expert Placement

**Analysis of the LPP solution.** Since the optimal expert placement should minimize the objective value of LPP 1, we begin by analyzing the solution of this problem. Let *m* denote the optimal objective value of LPP 1, which represents the minimized maximum load across all GPUs.

*Lemma*. There exists an optimal solution to LPP 1 that satisfies: Let  $G_{max}$  denote the set of GPUs with load m in this solution. An expert distributes no load within  $G_{max}$  unless its EDP group is entirely in  $G_{max}$ .

*Proof.* Among all optimal solutions, we choose the solution with the minimal  $|G_{max}|$  (number of GPUs with load m). Consider experts whose EDP groups partially intersect with  $G_{max}$ . Suppose that the EDP group of expert e consists of two GPUs  $g_0 \in G_{max}$  and  $g_1 \notin G_{max}$ . If expert e did distribute any load on GPU  $g_0$ , it could move a tiny amount of load from  $g_0$  to  $g_1$ . Afterward, the loads of both  $g_0$  and  $g_1$  would be less than m, yielding a new optimal solution with smaller  $|G_{max}|$ . The new solution would contradict our choice of the solution with the minimal  $|G_{max}|$ . Therefore, expert e cannot distribute any

load on GPU  $g_0$ .

Figure 5 shows an example of the above lemma. From this lemma, we can derive an equation by calculating the total loads within  $G_{max}$  in two ways:

<span id="page-6-4"></span>
$$m \cdot |G_{max}| = \sum_{e \in E: G_{EDP}^e \subseteq G_{max}} load_e \tag{2}$$

From Equation 2, we can derive an equation to calculate m by enumerating every possible  $G_{max}^2$ :

<span id="page-6-6"></span>
$$m = \max_{G_{max} \subseteq G_{FineEP}} \left\{ \frac{1}{|G_{max}|} \sum_{e \in E: G_{FDP}^e \subseteq G_{max}} load_e \right\}$$
(3)

Equation 3 provides a mathematical approach to determine the optimal objective value of LPP 1 without actually executing the LPP solving.

**Graph abstraction of expert placement.** Furthermore, we formulate expert placement with graph theory. Let each GPU  $g \in G_{FineEP}$  represent a vertex, and each expert  $e \in E$  represent a hyperedge connecting all GPUs in  $G_{EDP}^e$ . Hence, an expert placement can be represented by an undirected hypergraph denoted by  $\mathcal{G}(G_{FineEP}, \{G_{EDP}^e : e \in E\})$ . (When d=2, the hypergraph is a conventional graph. For simplicity, we omit the prefix "hyper" in the rest of this paper.) Figure 5 shows an example of the graph and the optimal solution corresponding to an expert placement.

Now, we can explain Equation 3 from the perspective of graph theory: We assign the weight of the edge (expert) e as  $load_e$ . We define the density of a weighted graph as the sum of all edge weights divided by the number of vertices<sup>3</sup>. Consider the subgraph in G induced by  $G_{max}$ . The edges in the induced subgraph represent the experts whose EDP groups are entirely in  $G_{max}$  ( $\{e \in E : G_{EDP}^e \subseteq G_{max}\}$ ). According to Equation 3, the optimal objective value m is the maximum density across all induced subgraphs of graph G. Since the goal of expert placement is to minimize m, we can characterize the optimal expert placement as follows:

**Property of the optimal expert placement.** The optimal expert placement is the graph whose maximum induced subgraph density is minimal.

Having understood the property of the optimal expert placement, our next challenge is how to find such a placement. We distinguish between two scenarios based on our knowledge of the expert load *load*<sub>e</sub> in §6.2 and §6.3.

# <span id="page-6-1"></span>**6.2** Symmetric Placement without Expert Loads

If we have no prior knowledge of the real expert load distribution, we can construct *symmetric placements*, treating all

<span id="page-6-7"></span><span id="page-6-5"></span><sup>&</sup>lt;sup>2</sup>We omit the derivation from Equation 2 to Equation 3.

<sup>&</sup>lt;sup>3</sup>This definition may differ from conventional graph density definitions in the literature.

<span id="page-7-2"></span>![](_page_7_Figure_0.jpeg)

(a) Data flow and analysis flow of MicroMoE.

![](_page_7_Figure_2.jpeg)

(b) Data flow of expert scheduling systems.

Figure 6: Differences between MicroMoE and expert scheduling systems.

experts equally. Symmetric placements provide conservative and general load balancing capability in terms of unknown load distributions. In such scenarios, we can assume that all expert loads follow an independent and identically distributed (i.i.d.) pattern. Thereby, the problem becomes: Given the number of vertices and edges as well as the identical distribution of edge weights, how to construct a graph that minimizes the expectation of the maximum induced subgraph density?

We recognize the challenge of the above problem due to the vast space of possible graphs and distributions. Nevertheless, we propose a near-optimal symmetric placement strategy for many practical configurations using *Cayley graphs* [57]. Our intuition is that the inherent symmetry of Cayley graphs ensures a balanced distribution of edges across vertices, preventing some induced subgraphs from having significantly larger density than others. Since Cayley graphs involve complex group theory, we illustrate our construction method in Appendix B.

#### <span id="page-7-0"></span>**6.3** Asymmetric Placement with Expert Loads

If we know real expert load distributions in advance, we can construct *asymmetric placements* tailored to them. Unlike symmetric scenarios, we can vary both replica counts and replica locations across different experts for better load balancing, similar to previous works [36,55,63].

We adopt an empirical strategy to construct a near-optimal asymmetric expert placement in two steps: (1) First, we determine the number of replicas for each expert with a *greedy* algorithm: We maintain a heap of experts sorted by load-per-replica, and iteratively allocate remaining replicas to the expert with the maximum load-per-replica. (2) Second, we determine the placement of expert replicas across GPUs with *Monte Carlo sampling*: We randomly generate many placement graphs, and choose the one whose maximum induced subgraph density is minimal.

#### <span id="page-7-1"></span>6.4 Adaptive Replacement

Based on symmetric and asymmetric placements, we further propose an *adaptive replacement* (AR) mechanism for FineMoE to optimize performance under dynamic expert loads.

Relationship between token scheduling and adaptive replacement. The adaptive replacement mechanism complements the token scheduling in §5 by addressing different levels of load balancing. Token scheduling performs *transient*, *fine-grained* load balancing through per-micro-batch token arrangement, while adaptive replacement handles *longterm*, *coarse-grained* load imbalances through periodic expert arrangement.

Specifically, for moderately imbalanced workloads, token scheduling sufficiently maintains complete balance with static, symmetric placements (as shown in §7.3). For highly skewed workloads, FineMoE adopts asymmetric placements to mitigate coarse-grained imbalances before using token scheduling for fine-grained optimization. Since asymmetric placements require real-time expert loads, FineMoE adopts adaptive replacement to monitor expert load distributions and adjust placements when significant distributional shifts are detected. Implementation of adaptive replacement. We implement the adaptive replacement mechanism in FineMoE using the placement manager (according to Figure 4). (1) During model initialization, the placement manager initializes the model states of all devices using the symmetric placement strategy, providing conservative and general load balancing capabilities. (2) During training, the placement manager monitors expert load information within each micro-batch in the background. (3) For every few iterations, the placement manager predicts future load distributions using historical data with time series analysis techniques, such as moving averages [7]. Then, it evaluates the performance of current placements on future distributions using Equation 3. If the future performance drops below a specific threshold, the placement manager generates new optimal asymmetric placements and reinitializes global model states accordingly.

Difference between FineMoE's adaptive replacement and expert scheduling solutions. The system implementation of FineMoE's adaptive replacement is similar to existing expert scheduling solutions, such as FlexMoE [36] and Smart-MoE [63]. Nevertheless, their design goals and algorithms are fundamentally different:

Design goals: In systems like FlexMoE, changing expert placement is the only means for load balancing, as shown in Figure 6b. However, in FineMoE, the primary weapon is token scheduling, while adaptive replacement is a further optimization to token scheduling, as shown in Figure 6a. Even with static placement, FineMoE can still achieve good load balancing performance at micro-batch granularity.

Algorithms: Different design goals yield distinct placement algorithms. In existing expert scheduling systems,

<span id="page-8-1"></span>![](_page_8_Figure_0.jpeg)

Figure 7: End-to-end speedup of different systems compared with Megatron-LM.

an expert's replicas typically have identical loads (i.e., replica\_load=expert\_load/replica\_count). Therefore, they can leverage greedy or dynamic programming algorithms accordingly [36,63]. In contrast, replica loads in FineMoE are determined by linear programming. Therefore, FineMoE requires the graph theory in §6.1 to guide the placement strategy.

#### <span id="page-8-0"></span>7 Evaluation

#### 7.1 Experimental Setup

**Testbed.** Our testbed consists of 4 nodes, each equipped with 8 NVIDIA H100 80GB SXM GPUs connected via 900 GBps NVLink. Nodes are interconnected using two 400 Gbps Infiniband NICs per node.

**Models.** We use GPT [2] and Mixtral [23] models for evaluation. We use GPT 32×1.3B to represent an MoE model converted from a 1.3B dense GPT model with 32 experts. We pretrain these models with the Wikipedia dataset [12]. A detailed list of model hyperparameters is provided in Appendix C.

**Baselines.** We compare FineMoE with four baseline systems.

- Megatron-LM [50] is a popular distributed training framework for large language models (LLMs). It supports various parallelism strategies, as well as state-of-the-art optimizations [5].
- SmartMoE [63] balances GPU loads by adjusting the expert placement within EP groups.
- FlexMoE [36] achieves load balancing by dynamically adjusting replica counts based on expert loads. FlexMoE places expert replicas across the entire DP group, similar to FineMoE's asymmetric placement (§6.3).
- DeepSpeed [46] is a high-performance distributed framework for both LLM training and inference. We enable ZeRO-1 [45] optimization in DeepSpeed (currently, ZeRO-2 does not support PP, and Tutel [22] does not support top-K>1).

We compare two variants of FineMoE: "FineMoE (w/o AR)" uses static, symmetric placement (§6.2), while "FineMoE" uses adaptive, asymmetric placement (§6.3-6.4).

**Implementation.** We implement FineMoE upon Megatron-LM [50]. FineMoE provides a model wrapper similar to Pytorch's Distributed Data Parallel (DDP) [29], enabling users to benefit from FineEP's fine-grained load balancing capabilities within their training jobs. We modify Megatron-LM, including its MoELayer and DDP with Python, and implement the token scheduling algorithm in FineEP with C++.

For fair comparison, we also implement SmartMoE and FlexMoE in Megatron-LM, as SmartMoE's repository is outdated (last commit in 2023) [63], and FlexMoE is not open-sourced [36].

**Parallelization configurations.** Due to the limited inter-node network bandwidth in our testbed, we only employ PP for inter-node parallelism. Specifically, we set the PP degree to the number of nodes used, and the DP degree to 8. We set the EP degree to 4, resulting in 2 EP groups per DP group. We set the parameter *d* in FineEP to 2, resulting in a single FineEP group per DP group. We disable TP due to its high communication overhead.

Other configurations. We use a small auxiliary loss (listed in the appendix) to prevent extreme load imbalance from degrading model accuracy. We enable the distributed optimizers in Megatron-LM, which resembles DeepSpeed's ZeRO-1 [45]. We disable the token dropping mechanism introduced by GShard [25]. We use BF16 precision.

#### <span id="page-8-2"></span>7.2 End-to-end Performance

Figure 7 shows the end-to-end performance of all systems, varying models and number of GPUs. DeepSpeed exhibits poor performance with 16 or 32 experts. This is because DeepSpeed always adopts a padding mechanism, padding the load of each expert to the maximum expert load [25]. This mechanism wastes significant time and memory when expert loads are highly imbalanced. With as few as 8 experts, the inefficiency of padding becomes less significant, allowing DeepSpeed to outperform Megatron-LM due to its system-level optimizations.

Comparing SmartMoE and FineMoE (w/o AR), where experts have uniform replica counts, FineMoE (w/o AR) ex-

<span id="page-9-1"></span>![](_page_9_Figure_0.jpeg)

Figure 8: Max GPU load normalized by average GPU load, varying skewness of expert loads. DP\_degree=8, num\_experts=32.

hibits superior performance, thanks to FineEP's fine-grained token scheduling. While SmartMoE attempts to optimize performance by changing expert placement for load balancing among GPUs, it sometimes performs worse than vanilla Megatron-LM. This is because SmartMoE optimizes expert placement based on long-term expert load distributions. However, expert loads are highly dynamic during training, and SmartMoE's long-term optimal placement may be suboptimal for individual micro-batches.

Comparing FlexMoE and FineMoE (with AR), where experts have varied replica counts, FineMoE exhibits superior performance, thanks to its fine-grained, per-micro-batch token scheduling.

In conclusion, **compared with Megatron-LM, FineMoE improves the end-to-end throughput by up to 47.6%, with an average improvement of 36.9%.** The average performance improvement of FineMoE surpasses FlexMoE, the second-best system, by 13.9%. Note that this is already the upper bound of performance improvement attainable through load balancing, since MicroMoE already achieves complete balance (detailed in § 7.3).

#### <span id="page-9-0"></span>7.3 Load Balancing Capability

We evaluate the load balancing capabilities of SmartMoE, FlexMoE, and FineMoE with skewed expert loads. We generate expert loads following a Zipfian distribution with skewness s, where the probability of a token being assigned to the i-th most loaded expert is proportional to  $i^{-s}$ .

Figure 8 shows the load balancing performance of different systems across varied skewness s. SmartMoE's maximum GPU load increases as load skewness increases. While Flex-MoE maintains relatively balanced GPU loads by adjusting expert replica counts, it falls short of achieving optimal load balance due to its lack of fine-grained dynamicity. FineMoE (random) represents FineMoE with pure random placement, which performs slightly worse than FineMoE (w/o AR) with symmetric placement. FineMoE (w/o AR) achieves perfect load balance when s < 1 thanks to FineEP's fine-grained to-

<span id="page-9-2"></span>![](_page_9_Figure_8.jpeg)

Figure 9: Execution time breakdown of an MoE layer. DP\_degree=8, num\_experts=32, micro\_batch\_size=8, sequence\_length=2048, topK=2, hidden\_size=4096, skewness *s*=1.

ken scheduling. Nonetheless, its performance degrades when s>1 as uniform replica counts are insufficient for severe imbalances. FineMoE with asymmetric placements can always achieve complete load balance, due to the combination of both coarse-grained expert replacement and fine-grained token scheduling. Overall, **FineMoE exhibits the best load balancing capability among all systems and consistently achieves complete load balance.** 

#### <span id="page-9-3"></span>7.4 Execution Time Breakdown

Figure 9 shows the execution time breakdown of an MoE layer across different systems. We omit DeepSpeed due to its poor performance. For all remaining systems, the primary bottleneck is expert computation time. FineMoE achieves the shortest computation time by maintaining perfect load balance (with either symmetric or asymmetric placement, as shown in Figure 8).

Specifically, the dispatch time consists of two primary components: (1) Preparation time, which includes the all-gather operation of expert load information and the scheduling of FineEP. While FineMoE introduces additional overhead in dispatch time due to token scheduling operations, we effectively minimize this impact through overlapping with other operations in Megatron-LM. (2) All-to-all communication time. Each all-to-all communication in dispatch and combine requires approximately 1.3 ms in Megatron-LM.

#### 7.5 Inter-node Communication with DeepEP

We evaluate the dispatch time of FineEP and vanilla EP for inter-node communication. We additionally integrate FineEP with DeepEP [9], a high-performance all-to-all communication backend. Currently, Megatron-LM [50] supports both NCCL [19] (by default) and DeepEP for all-to-all communication.

**Experimental considerations.** Due to testbed limitations, two important experimental considerations should be noted: (1) Our testbed consists of 8 GPUs but only 2 NICs per node,

<span id="page-10-0"></span>![](_page_10_Figure_0.jpeg)

Figure 10: Dispatch time of FineEP and EP Figure 11: Scheduling time for FineEP, Figure 12: Migration time for adaptive rewith DeepEP and NCCL, varying number of varying number of experts and GPUs. placement of FineMoE. GPUs.

resulting in limited inter-node network bandwidth. Therefore, we avoid employing EP or FineEP across multiple nodes in §7. However, since this section focuses on evaluating the performance of different communication backends, we expand the communication group to multiple nodes. Consequently, the all-to-all time for inter-node communication is significantly higher than the intra-node communication. (2) §7 focuses on system performance, so we compare FineEP using 8 GPUs per group with EP using 4 GPUs per group (d=2). However, this section focuses on communication performance, so we compare FineEP and EP using the same group size.

Overhead of inter-node communication. Notably, since FineEP expands the all-to-all group size by a factor of d, FineEP may convert some inter-node communication into inter-node, leading to extra overhead. However, this overhead is minimal in two typical scenarios: (1) When  $d \times \text{EP\_degree} \le \#$  GPUs per node, the all-to-all communication in FineEP remains entirely intra-node. (2) When EP\_degree is super large (e.g., 64 in DeepSeek-v3 [9]), nearly all communication is inherently inter-node. Consequently, FineEP incurs negligible overheads in both single-node and massive-node scenarios. Furthermore, our communication-aware scheduling mechanism can jointly optimize the time of both communication and computation, as shown in Appendix C.2.

Results. Figure 16 shows the dispatch time comparison between FineEP and EP using both DeepEP and NCCL, varying number of GPUs. We use the same setting as in §7.4, except for the all-to-all group size. DeepEP exhibits better performance than NCCL due to its superior implementation. When using NCCL, FineEP requires less time than EP, thanks to the locality-aware routing in §5.2. However, when using DeepEP, FineEP requires more time than EP due to data format incompatibilities between DeepEP and Megatron-LM. Consequently, Megatron-LM needs to pre-process the data for DeepEP, while FineEP incurs a higher pre-processing overhead than EP. We believe that FineMoE will yield lower communication overheads on other practical testbeds (e.g., with one NIC per GPU).

## <span id="page-10-1"></span>7.6 Overhead Analysis

**Scheduling Overhead.** We evaluate the scheduling overhead of FineEP, including the LPP solving time and token routing time. Our evaluation reveals that the LPP solving time is the dominant factor, which scales with the number of experts and GPUs. As shown in Figure 11, the scheduling overhead remains remarkably low, with a minimum time of approximately 100 us. Even with 64 GPUs and 256 experts, the scheduling time remains below 1 ms. This minimal overhead per micro-batch enables FineEP to maintain high training throughput while providing load balancing benefits. Additionally, we evaluate the performance of pipelining to hide the scheduling latency, which results are shown in Appendix C.3. **Replacement overhead.** The adaptive replacement strategy in FineMoE necessitates model re-initialization to transition to new configurations. Although replacement is beneficial for load balancing, model re-initialization causes temporary suspension of training. Our evaluation highlights two key components in the replacement overhead: the migration time of expert parameters and their optimizer states. As shown in Figure 12, the total migration time typically spans hundreds of milliseconds across different model configurations.

The above results emphasize the importance of carefully selecting the expert replacement frequency to optimize the trade-off between per-micro-batch training efficiency and overall replacement overhead. In practice, we recommend tuning the replacement interval to 50 iterations during the beginning phase of training, which adds less than 1% overhead to the entire system. One may increase this interval to several hundred iterations or even make no replacement when workloads become less volatile, as shown in Figure 2.

# 7.7 Ablation Study

We conducted an ablation study to evaluate three optimizations in FineMoE: (1) warm solving in §5.1, (2) locality-aware routing in §5.2, and (3) overlapping in §5.4. All these optimizations aim to reduce the dispatch time. As shown in Figure 13, both warm solving and overlapping reduce scheduling

<span id="page-11-0"></span>![](_page_11_Figure_0.jpeg)

Figure 13: Ablation study of dispatch time, with the same setting as Figure [9.](#page-9-2)

time, while locality-aware routing reduces all-to-all communication time. With the combination of optimizations, FineMoE introduces only 0.4 ms additional dispatch time compared to vanilla Megatron-LM. This modest overhead is substantially outweighed by the reduced computation time, ensuring the overall system efficiency of FineMoE.

# 8 Related Work

MoE training systems. Switch Transformer [\[11\]](#page-13-3) pioneers expert parallelism for transformer-based models. FairSeq [\[38\]](#page-14-13) and FastMoE [\[17\]](#page-13-14) provide PyTorch [\[39\]](#page-14-14) plugins to optimize MoE training. DeepSpeed-MoE [\[44\]](#page-15-16) combines EP with various optimizations, such as ZeRO [\[45\]](#page-15-14) for large-scale MoE training. Tutel [\[22\]](#page-13-9) dynamically adapts parallelism strategies and pipeline degrees to handle dynamic workloads of MoE. MoE load balancing. Many studies propose algorithmic solutions to address the load imbalance issue of MoE. Shazeer et al. [\[49\]](#page-15-3) introduce a load-balancing loss, which is further refined by subsequent works [\[4,](#page-12-5)[11,](#page-13-3)[16,](#page-13-15)[28,](#page-14-5)[54](#page-15-6)[,58\]](#page-15-17). GShard [\[25\]](#page-14-4) introduces the concept of expert capacity to limit the number of tokens assigned to each expert, which is adopted by later works [\[11,](#page-13-3) [22,](#page-13-9) [28\]](#page-14-5). Some studies propose alternative gating algorithms in place of the original top-K gating strategy [\[26,](#page-14-15) [35,](#page-14-16) [48,](#page-15-18) [66\]](#page-16-0). As a systematic approach, FineMoE is compatible with all these algorithmic solutions.

Many studies offer systematic solutions for load balancing in MoE. FasterMoE [\[18\]](#page-13-2) models the time consumption of MoE and enables broadcasting the most popular expert to all devices. MoESys [\[62\]](#page-16-2) dynamically adjusts the number of training nodes to ensure load balance in multi-task scenarios. SmartMoE [\[63\]](#page-16-1) focuses on optimizing expert placement. FlexMoE [\[36\]](#page-14-7) adapts the number of replicas of each expert based on their popularity. Several other works [\[41,](#page-14-8) [51,](#page-15-10) [55,](#page-15-7) [56\]](#page-15-11) embrace similar approaches to FlexMoE, and further reduce the adjustment overhead by integrating expert migration with ZeRO's communication operations [\[45\]](#page-15-14). While these solutions typically balance loads at a per-iteration, expert-level granularity, FineMoE is capable of achieving optimal load

balance at a per-micro-batch granularity.

LPLB [\[8\]](#page-13-16) is a recent work that leverages linearprogramming-based token scheduling for MoE load balancing, similar to MicroMoE. However, it lacks comprehensive analysis and experimental evaluation. [4](#page-11-1) In contrast, this paper theoretically analyzes the relationship between expert placement and token scheduling, proposes multiple placement strategies, and incorporates key optimizations such as locality-aware routing. We believe that our work provides clearer insights and practical guidance for designing nextgeneration MoE systems.

Janus [\[30\]](#page-14-17) proposes a data-centric EP paradigm that avoids the inherent load imbalance issue in conventional expertcentric paradigms. However, Janus requires migrating all expert parameters within every micro-batch, which incurs substantial communication overhead, especially when dealing with large batch numbers and model scales.

# 9 Discussion

Scalability. Scaling FineMoE to large clusters and increased expert counts introduces new challenges to the system. When scaling up, FineMoE benefits from an expanded scheduling space, which enhances the load balancing capability of FineEP. However, this larger scheduling space increases computational overhead in scheduling. To compromise between load balancing capability and system efficiency, we can organize GPUs and experts into groups and perform scheduling at the group level, similar to prior works [\[9,](#page-13-1) [25,](#page-14-4) [63\]](#page-16-1).

FSDP. We currently implement FineMoE based on Megatron-LM's DDP [\[50\]](#page-15-13). Megatron-LM also supports Fully Sharded Data Parallel (FSDP) [\[65\]](#page-16-3), which resembles DeepSpeed's ZeRO-3 [\[45\]](#page-15-14), sharding model parameters and gradients for memory saving. We plan to integrate FineMoE with FSDP in future work.

# 10 Conclusion

We propose FineEP, a novel expert parallelism strategy to achieve fine-grained load balancing in MoE. FineEP dynamically balances GPU loads within every micro-batch through token scheduling. We primarily make two optimizations in FineEP: First, we formulate the token scheduling process as a linear programming problem, which can be solved efficiently. Second, we theoretically analyze the relationship between expert placement and load balancing capacity and develop two placement strategies for different training scenarios. Finally, we propose FineMoE, an efficient MoE training system based on FineEP. Our experimental evaluation demonstrates that FineMoE achieves significant performance improvements,

<span id="page-11-1"></span><sup>4</sup>By the time we wrote this paper, LPLB was still in the early research stage and had no evaluation results. As far as we know, LPLB is the only relevant work that schedules tokens for load balancing.

with up to 47.6% end-to-end speedup compared to Megatron-LM, and almost consistently maintains optimal load balance across GPUs.

# References

- <span id="page-12-2"></span>[1] Anthropic. Introducing claude 4. [https://www.](https://www.anthropic.com/news/claude-4) [anthropic.com/news/claude-4](https://www.anthropic.com/news/claude-4), 2025.
- <span id="page-12-3"></span>[2] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In *Proceedings of the 34th International Conference on Neural Information Processing Systems (NIPS '20)*, 2020.
- <span id="page-12-1"></span>[3] Bing Cao, Yiming Sun, Pengfei Zhu, and Qinghua Hu. Multi-modal gated mixture of local-to-global experts for dynamic image fusion. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV '23)*, 2023.
- <span id="page-12-5"></span>[4] Chang Chen, Min Li, Zhihua Wu, Dianhai Yu, and Chao Yang. Ta-moe: Topology-aware large scale mixture-ofexpert training. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, *Advances in Neural Information Processing Systems (NIPS '22)*. Curran Associates, Inc., 2022.
- <span id="page-12-4"></span>[5] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. <https://arxiv.org/abs/1604.06174>, 2016.
- <span id="page-12-0"></span>[6] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sashank Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and

- Noah Fiedel. Palm: scaling language modeling with pathways. *J. Mach. Learn. Res.*, 2023.
- <span id="page-13-10"></span>[7] Peizhuang Cong, Aomufei Yuan, Shimao Chen, Yuxuan Tian, Bowen Ye, and Tong Yang. Prediction is all moe needs: Expert load distribution goes from fluctuating to stabilizing. <https://arxiv.org/abs/2404.16914>, 2024.
- <span id="page-13-16"></span>[8] DeepSeek. Linear-programming-based load balancer. <https://github.com/deepseek-ai/LPLB>, 2025.
- <span id="page-13-1"></span>[9] DeepSeek-AI. Deepseek-v3 technical report. [https:](https://arxiv.org/abs/2412.19437) [//arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437), 2024.
- <span id="page-13-4"></span>[10] Nan Du, Yanping Huang, Andrew M Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, Barret Zoph, Liam Fedus, Maarten P Bosma, Zongwei Zhou, Tao Wang, Emma Wang, Kellie Webster, Marie Pellat, Kevin Robinson, Kathleen Meier-Hellstern, Toju Duke, Lucas Dixon, Kun Zhang, Quoc Le, Yonghui Wu, Zhifeng Chen, and Claire Cui. GLaM: Efficient scaling of language models with mixture-of-experts. In *Proceedings of the 39th International Conference on Machine Learning (ICML '22)*, 2022.
- <span id="page-13-3"></span>[11] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: scaling to trillion parameter models with simple and efficient sparsity. *J. Mach. Learn. Res.*, 2022.
- <span id="page-13-12"></span>[12] Wikimedia Foundation. Wikimedia downloads. [https:]( https://dumps.wikimedia.org/) [//dumps.wikimedia.org/]( https://dumps.wikimedia.org/), 2025.
- <span id="page-13-7"></span>[13] Trevor Gale, Deepak Narayanan, Cliff Young, and Matei Zaharia. Megablocks: Efficient sparse training with mixture-of-experts. *Proceedings of Machine Learning and Systems (MLSys '23)*, 2023.
- <span id="page-13-0"></span>[14] Google. Gemini 3: Our most intelligent ai model that brings any idea to life. [https://deepmind.google/](https://deepmind.google/models/gemini/) [models/gemini/](https://deepmind.google/models/gemini/), 2025.
- <span id="page-13-5"></span>[15] Hongcan Guo, Haolang Lu, Guoshun Nan, Bolun Chu, Jialin Zhuang, Yuan Yang, Wenhao Che, Sicong Leng, Qimei Cui, and Xudong Jiang. Advancing expert specialization for better moe. [https://arxiv.org/abs/](https://arxiv.org/abs/2505.22323) [2505.22323](https://arxiv.org/abs/2505.22323), 2025.
- <span id="page-13-15"></span>[16] Hussein Hazimeh, Zhe Zhao, Aakanksha Chowdhery, Maheswaran Sathiamoorthy, Yihua Chen, Rahul Mazumder, Lichan Hong, and Ed Chi. Dselect-k: Differentiable selection in the mixture of experts with applications to multi-task learning. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, *Advances in Neural Information Processing Systems (NIPS '21)*. Curran Associates, Inc., 2021.

- <span id="page-13-14"></span>[17] Jiaao He, Jiezhong Qiu, Aohan Zeng, Zhilin Yang, Jidong Zhai, and Jie Tang. Fastmoe: A fast mixtureof-expert training system. [https://arxiv.org/abs/](https://arxiv.org/abs/2103.13262) [2103.13262](https://arxiv.org/abs/2103.13262), 2021.
- <span id="page-13-2"></span>[18] Jiaao He, Jidong Zhai, Tiago Antunes, Haojie Wang, Fuwen Luo, Shangfeng Shi, and Qin Li. Fastermoe: modeling and optimizing training of large-scale dynamic pre-trained models. In *Proceedings of the 27th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP '22)*, 2022.
- <span id="page-13-13"></span>[19] Zhiyi Hu, Siyuan Shen, Tommaso Bonato, Sylvain Jeaugey, Cedell Alexander, Eric Spada, James Dinan, Jeff Hammond, and Torsten Hoefler. Demystifying nccl: An in-depth analysis of gpu communication protocols and algorithms. [https://arxiv.org/abs/2507.](https://arxiv.org/abs/2507.04786) [04786](https://arxiv.org/abs/2507.04786), 2025.
- <span id="page-13-6"></span>[20] Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Dehao Chen, Mia Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V Le, Yonghui Wu, and zhifeng Chen. Gpipe: Efficient training of giant neural networks using pipeline parallelism. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, *Advances in Neural Information Processing Systems*. Curran Associates, Inc., 2019.
- <span id="page-13-8"></span>[21] Q. Huangfu and J. A. J. Hall. Parallelizing the dual revised simplex method. *Mathematical Programming Computation*, 2018.
- <span id="page-13-9"></span>[22] Changho Hwang, Wei Cui, Yifan Xiong, Ziyue Yang, Ze Liu, Han Hu, Zilong Wang, Rafael Salas, Jithin Jose, Prabhat Ram, HoYuen Chau, Peng Cheng, Fan Yang, Mao Yang, and Yongqiang Xiong. Tutel: Adaptive mixture-of-experts at scale. In D. Song, M. Carbin, and T. Chen, editors, *Proceedings of Machine Learning and Systems (MLSys '23)*. Curan, 2023.
- <span id="page-13-11"></span>[23] Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mixtral of experts. [https://arxiv.org/abs/2401.](https://arxiv.org/abs/2401.04088) [04088](https://arxiv.org/abs/2401.04088), 2024.
- <span id="page-13-17"></span>[24] Vijay Anand Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. Reducing activation

- recomputation in large transformer models. In *Proceedings of Machine Learning and Systems (MLSys '23)*, 2023.
- <span id="page-14-4"></span>[25] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. [https://arxiv.org/abs/2006.](https://arxiv.org/abs/2006.16668) [16668](https://arxiv.org/abs/2006.16668), 2020.
- <span id="page-14-15"></span>[26] Mike Lewis, Shruti Bhosale, Tim Dettmers, Naman Goyal, and Luke Zettlemoyer. Base layers: Simplifying training of large, sparse models. In *International Conference on Machine Learning (ICML '21)*. PMLR, 2021.
- <span id="page-14-11"></span>[27] Jiamin Li, Yimin Jiang, Yibo Zhu, Cong Wang, and Hong Xu. Accelerating distributed MoE training and inference with lina. In *2023 USENIX Annual Technical Conference (USENIX ATC '23)*. USENIX Association, July 2023.
- <span id="page-14-5"></span>[28] Jing Li, Zhijie Sun, Xuan He, Li Zeng, Yi Lin, Entong Li, Binfan Zheng, Rongqian Zhao, and Xin Chen. Locmoe: A low-overhead moe for large language model training. <https://arxiv.org/abs/2401.13920>, 2024.
- <span id="page-14-12"></span>[29] Shen Li, Yanli Zhao, Rohan Varma, Omkar Salpekar, Pieter Noordhuis, Teng Li, Adam Paszke, Jeff Smith, Brian Vaughan, Pritam Damania, and Soumith Chintala. Pytorch distributed: experiences on accelerating data parallel training. *Proc. VLDB Endow.*, 2020.
- <span id="page-14-17"></span>[30] Juncai Liu, Jessie Hui Wang, and Yimin Jiang. Janus: A unified distributed training framework for sparse mixture-of-experts models. In *Proceedings of the ACM SIGCOMM 2023 Conference (SIGCOMM '23)*, New York, NY, USA, 2023. Association for Computing Machinery.
- <span id="page-14-0"></span>[31] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV '21)*, October 2021.
- <span id="page-14-1"></span>[32] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. *ViLBERT: pretraining task-agnostic visiolinguistic representations for vision-and-language tasks*. 2019.
- <span id="page-14-3"></span>[33] Meta. The llama 4 herd: The beginning of a new era of natively multimodal ai innovation. [https://ai.meta.](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) [com/blog/llama-4-multimodal-intelligence/](https://ai.meta.com/blog/llama-4-multimodal-intelligence/), 2025.

- <span id="page-14-9"></span>[34] Deepak Narayanan, Aaron Harlap, Amar Phanishayee, Vivek Seshadri, Nikhil R. Devanur, Gregory R. Ganger, Phillip B. Gibbons, and Matei Zaharia. Pipedream: generalized pipeline parallelism for dnn training. In *Proceedings of the 27th ACM Symposium on Operating Systems Principles (SOSP '19)*. Association for Computing Machinery, 2019.
- <span id="page-14-16"></span>[35] Xiaonan Nie, Qibin Liu, Fangcheng Fu, Shenhan Zhu, Xupeng Miao, Xiaoyang Li, Yang Zhang, Shouda Liu, and Bin Cui. Lsh-moe: Communication-efficient moe training via locality-sensitive hashing. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, *Advances in Neural Information Processing Systems (NIPS '24)*. Curran Associates, Inc., 2024.
- <span id="page-14-7"></span>[36] Xiaonan Nie, Xupeng Miao, Zilong Wang, Zichao Yang, Jilong Xue, Lingxiao Ma, Gang Cao, and Bin Cui. Flexmoe: Scaling large-scale sparse pre-trained model training via dynamic device placement. *Proc. ACM Manag. Data*, 2023.
- <span id="page-14-2"></span>[37] OpenAI. Gpt-5 is here. <https://openai.com/gpt-5>, 2025.
- <span id="page-14-13"></span>[38] Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. fairseq: A fast, extensible toolkit for sequence modeling. In *Proceedings of NAACL-HLT 2019: Demonstrations*, 2019.
- <span id="page-14-14"></span>[39] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. 2019.
- <span id="page-14-10"></span>[40] Penghui Qi, Xinyi Wan, Guangxing Huang, and Min Lin. Zero bubble pipeline parallelism. [https://arxiv.](https://arxiv.org/abs/2401.10241) [org/abs/2401.10241](https://arxiv.org/abs/2401.10241), 2023.
- <span id="page-14-8"></span>[41] Yuhao Qing, Guichao Zhu, Fanxin Li, Lintian Lei, Zekai Sun, Xiuxian Guan, Shixiong Zhao, Xusheng Chen, Dong Huang, Sen Wang, and Heming Cui. Hecate: Unlocking efficient sparse model training via fully sharded sparse data parallelism. [https://arxiv.org/abs/](https://arxiv.org/abs/2502.02581) [2502.02581](https://arxiv.org/abs/2502.02581), 2025.
- <span id="page-14-6"></span>[42] Zihan Qiu, Zeyu Huang, Bo Zheng, Kaiyue Wen, Zekun Wang, Rui Men, Ivan Titov, Dayiheng Liu, Jingren Zhou, and Junyang Lin. Demons in the detail: On implementing load balancing loss for training specialized mixture-of-expert models. [https://arxiv.org/abs/](https://arxiv.org/abs/2501.11873) [2501.11873](https://arxiv.org/abs/2501.11873), 2025.

- <span id="page-15-0"></span>[43] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019.
- <span id="page-15-16"></span>[44] Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, and Yuxiong He. DeepSpeed-MoE: Advancing mixture-of-experts inference and training to power next-generation AI scale. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, *Proceedings of the 39th International Conference on Machine Learning (ICML '22)*, 2022.
- <span id="page-15-14"></span>[45] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In *International Conference for High Performance Computing, Networking, Storage and Analysis (SC '20)*, 2020.
- <span id="page-15-12"></span>[46] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20)*. Association for Computing Machinery, 2020.
- <span id="page-15-2"></span>[47] Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, André Susano Pinto, Daniel Keysers, and Neil Houlsby. Scaling vision with sparse mixture of experts. In *Proceedings of the 35th International Conference on Neural Information Processing Systems (NIPS '21)*, 2021.
- <span id="page-15-18"></span>[48] Stephen Roller, Sainbayar Sukhbaatar, arthur szlam, and Jason Weston. Hash layers for large sparse models. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, editors, *Advances in Neural Information Processing Systems (NIPS '21)*. Curran Associates, Inc., 2021.
- <span id="page-15-3"></span>[49] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. [https://arxiv.org/abs/](https://arxiv.org/abs/1701.06538) [1701.06538](https://arxiv.org/abs/1701.06538), 2017.
- <span id="page-15-13"></span>[50] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. [https://arxiv.org/](https://arxiv.org/abs/1909.08053) [abs/1909.08053](https://arxiv.org/abs/1909.08053), 2020.
- <span id="page-15-10"></span>[51] Athinagoras Skiadopoulos, Mark Zhao, Swapnil Gandhi, Thomas Norrie, Shrijeet Mukherjee, and Christos Kozyrakis. Accelerating mixture-of-experts training

- with adaptive expert replication. [https://arxiv.org/](https://arxiv.org/abs/2504.19925) [abs/2504.19925](https://arxiv.org/abs/2504.19925), 2025.
- <span id="page-15-9"></span>[52] Yehui Tang, Xiaosong Li, Fangcheng Liu, Wei Guo, Hang Zhou, Yaoyuan Wang, Kai Han, Xianzhi Yu, Jinpeng Li, Hui Zang, Fei Mi, Xiaojun Meng, Zhicheng Liu, Hanting Chen, Binfan Zheng, Can Chen, Youliang Yan, Ruiming Tang, Peifeng Qin, Xinghao Chen, Dacheng Tao, and Yunhe Wang. Pangu pro moe: Mixture of grouped experts for efficient sparsity. [https://arxiv.](https://arxiv.org/abs/2505.21411) [org/abs/2505.21411](https://arxiv.org/abs/2505.21411), 2025.
- <span id="page-15-1"></span>[53] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In *Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS'17)*, 2017.
- <span id="page-15-6"></span>[54] Lean Wang, Huazuo Gao, Chenggang Zhao, Xu Sun, and Damai Dai. Auxiliary-loss-free load balancing strategy for mixture-of-experts. [https://arxiv.org/abs/](https://arxiv.org/abs/2408.15664) [2408.15664](https://arxiv.org/abs/2408.15664), 2024.
- <span id="page-15-7"></span>[55] Wei Wang, Zhiquan Lai, Shengwei Li, Weijie Liu, Keshi Ge, Yujie Liu, Ao Shen, and Dongsheng Li. Prophet: Fine-grained load balancing for parallel training of largescale moe models. In *2023 IEEE International Conference on Cluster Computing (CLUSTER '23)*, 2023.
- <span id="page-15-11"></span>[56] Wei Wang, Zhiquan Lai, Shengwei Li, Weijie Liu, Keshi Ge, Ao Shen, Huayou Su, and Dongsheng Li. Proprophet: A systematic load balancing method for efficient parallel training of large-scale moe models. [https:](https://arxiv.org/abs/2411.10003) [//arxiv.org/abs/2411.10003](https://arxiv.org/abs/2411.10003), 2024.
- <span id="page-15-15"></span>[57] Wikipedia. Cayley graph. [https://en.wikipedia.](https://en.wikipedia.org/wiki/Cayley_graph) [org/wiki/Cayley\\_graph](https://en.wikipedia.org/wiki/Cayley_graph), 2025.
- <span id="page-15-17"></span>[58] Shaohua Wu, Jiangang Luo, Xi Chen, Lingjun Li, Xudong Zhao, Tong Yu, Chao Wang, Yue Wang, Fei Wang, Weixu Qiao, Houbo He, Zeru Zhang, Zeyu Sun, Junxiong Mao, and Chong Shen. Yuan 2.0-m32: Mixture of experts with attention router. [https://arxiv.](https://arxiv.org/abs/2405.17976) [org/abs/2405.17976](https://arxiv.org/abs/2405.17976), 2024.
- <span id="page-15-4"></span>[59] xAI. Grok 4. <https://x.ai/news/grok-4>, 2025.
- <span id="page-15-8"></span>[60] Fuzhao Xue, Zian Zheng, Yao Fu, Jinjie Ni, Zangwei Zheng, Wangchunshu Zhou, and Yang You. Openmoe: An early effort on open mixture-of-experts language models. [https://arxiv.org/abs/2402.](https://arxiv.org/abs/2402.01739) [01739](https://arxiv.org/abs/2402.01739), 2024.
- <span id="page-15-5"></span>[61] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei,

Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. https://arxiv.org/abs/2505.09388, 2025.

- <span id="page-16-2"></span>[62] Dianhai Yu, Liang Shen, Hongxiang Hao, Weibao Gong, Huachao Wu, Jiang Bian, Lirong Dai, and Haoyi Xiong. Moesys: A distributed and efficient mixture-of-experts training and inference system for internet services. *IEEE Transactions on Services Computing*, 2024.
- <span id="page-16-1"></span>[63] Mingshu Zhai, Jiaao He, Zixuan Ma, Zan Zong, Runqing Zhang, and Jidong Zhai. SmartMoE: Efficiently training Sparsely-Activated models through combining offline and online parallelization. In *2023 USENIX Annual Technical Conference (USENIX ATC '23)*, 2023.
- <span id="page-16-4"></span>[64] Shulai Zhang, Ningxin Zheng, Haibin Lin, Ziheng Jiang, Wenlei Bao, Chengquan Jiang, Qi Hou, Weihao Cui, Size Zheng, Li-Wen Chang, Quan Chen, and Xin Liu. COMET: Fine-grained computation-communication overlapping for mixture-of-experts. In Eighth Conference on Machine Learning and Systems (MLSys '25), 2025.
- <span id="page-16-3"></span>[65] Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, Alban Desmaison, Can Balioglu, Pritam Damania, Bernard Nguyen, Geeta Chauhan, Yuchen Hao, Ajit Mathews, and Shen Li. Pytorch fsdp: Experiences on scaling fully sharded data parallel. *Proc. VLDB Endow.*, 2023.
- <span id="page-16-0"></span>[66] Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew M Dai, zhifeng Chen, Quoc V Le, and James Laudon. Mixture-of-experts with expert choice routing. In Advances in Neural Information Processing Systems (NIPS '22), 2022.

#### A Optimizations to FineEP

#### <span id="page-16-6"></span>A.1 Communication-Aware Scheduling

Many studies demonstrate that the all-to-all communication also takes a significant amount of time in MoE layers [18, 22, 27]. However, the optimization problem 1 only considers the computation time. Therefore, we can further consider the communication time during scheduling.

In many existing frameworks [46, 50], the time of an MoE layer is mainly determined by the communication time + the computation time (unless using some communication-computation overlapping techniques, such as Comet [64] and DualPipe [9]). Therefore, we introduce an additional objective to minimize the maximum communication volume on a GPU. According to Algorithm 1, Line 6, the local data volume of GPU g can be formulated as  $local_g = \sum \min(x_e^g, input_e^g)$ . For each GPU, we consider its all-

 $e \in E: g \in G_{EDP}^e$  to-all communication volume as the greater value between its send volume and receive volume. The send/receive volume of GPU g can be calculated from  $x_e^g$ ,  $input_e^g$ , and  $local_g$ . Putting them together, we derive the new optimization problem:

<span id="page-16-5"></span>minimize  $comp + \alpha \cdot comm$ , subject to  $comp = \max_{g \in G_{FineEP}} \left\{ \sum_{e \in E: g \in G_{EDP}^e} x_e^g \right\}$ ,  $comm = \max_{g \in G_{FineEP}} \left\{ \max(send_g, recv_g) \right\}$ ,  $send_g = \left( \sum_{e \in E: g \in G_{EDP}^e} input_e^g \right) - local_g$ ,  $recv_g = \left( \sum_{e \in E: g \in G_{EDP}^e} x_e^g \right) - local_g$ ,  $local_g = \sum_{e \in E: g \in G_{EDP}^e} \min(x_e^g, input_e^g)$ ,  $\sum_{g \in G_{EDP}^e} x_e^g = load_e, \quad \forall e \in E$ ,  $x_e^g > 0$ .

 $\alpha$  is a constant parameter reflecting the weight of the communication time in the optimization problem. In practice, we can set  $\alpha$  to the ratio of expert computation throughput to all-to-all communication throughput. The optimization problem 4 is still an LPP. We omit the derivation of converting problem 4 to a standard LPP format.

**Topology-aware scheduling.** We can further consider the impact of network topology on communication. In a typical network topology, intra-node communication (e.g., through NVLink) is many times faster than inter-node communication (e.g., through Infiniband). Therefore, we can consider this difference during scheduling for better communication efficiency.

In the LPP 4, we can split the communication time into intra-node and inter-node. Intra-node communication has a lower weight  $(\alpha_1)$  than inter-node communication  $(\alpha_2)$ . Additionally, we modify the routing mechanism as follows: First, route local tokens to local replicas within the same GPUs. Second, route tokens to replicas on other GPUs in the same nodes. Third, route tokens to other global GPUs.

**Overhead.** While communication-aware scheduling minimizes communication volume, it increases computational

complexity due to additional parameters and constraints in the LPP (comparing LPP 4 and 1). Therefore, we should carefully consider when to enable the communication-aware scheduling, performing a trade-off between the LPP solving time and the communication time.

#### <span id="page-17-3"></span>**A.2** Pipelining FineEP

<span id="page-17-1"></span>![](_page_17_Figure_2.jpeg)

(b) ThicEr with pipenning

Figure 14: Pipelining FineEP.

Pipelining offers an additional approach to hide the scheduling overhead of FineEP, different from the overlapping technique in §5.4. Pipelining is particularly useful in scenarios where it is impossible to fully hide the scheduling time through overlapping. For example, some high-performance frameworks, such as DeepEP [9], have no intermediate operations to overlap between the gate network and all-to-all communication; when dealing with large-scale deployments with numerous GPUs and experts, the scheduling can incur substantial overhead (as illustrated in §7.6). With pipelining, we can overlap the scheduling of some tokens with the all-to-all communication of other tokens. This approach provides more flexibility in latency hiding.

We find many design choices to perform pipelining for FineEP and illustrate one of them as follows. We observe that FineEP can achieve perfect load balance in most circumstances (as shown in §7.3). Our intuition is that even if the expert load becomes more imbalanced, FineEP can still achieve perfect balance. Inspired by this observation, we can split the tokens into two parts: We apply EP to the former part<sup>5</sup>, and apply FineEP to the latter part. Moreover, the optimization problem 1 should consider the computational workloads in both EP and FineEP. In this way, we can overlap the scheduling of the latter part with the all-to-all communication of the former part, as shown in Figure 14.

**Overhead.** Pipelining FineEP introduces some additional system overhead. For example, splitting one all-to-all communication operation into two increases synchronization time and GPU kernel launching time. Therefore, we recommend

<span id="page-17-2"></span>![](_page_17_Figure_9.jpeg)

Figure 15: Examples of Cayley graphs.

using FineEP with pipelining in scenarios with substantial scheduling time but minimal system overhead.

# **B** Expert Placement using Cayley Graphs

#### **B.1** Cayley Graphs

In many practical applications, we can construct near-optimal symmetric expert placements in FineEP using Cayley graphs. The inherent symmetry of Cayley graphs makes them well-suited for constructing optimal expert placements.

A Cayley graph is constructed from a group A and its generating set S. Each element in the group  $a \in A$  is assigned as a vertex. For every  $a \in A$  and  $s \in S$ , there is an edge from the vertex a to the vertex as.

We assume the FineEP parameter d=2, so the hypergraph is a conventional graph. We observe that in practical applications, the quantities of GPUs and experts are usually powers of two. Let the number of GPUs be  $2^p$ , and the number of experts per GPU be  $2^q$ . Consequently, the number of vertices is  $2^p$ , the degree of each vertex is  $2^q$ , and the number of edges is  $2^{p+q-1}$ . In practice, we heuristicly construct many Cayley graphs for different (p,q)s.

#### **B.2** Examples of Cayley Graphs

We illustrate some example constructions as follows.

**Example 1:** 8 vertices, 8 edges.

We have p = 3, q = 1. The group is  $(\mathbb{Z}_8, +)$ , and the generating set is  $\{1, -1\}$ . The constructed graph is a cycle.

Example 2: 16 vertices, 32 edges.

<span id="page-17-0"></span><sup>&</sup>lt;sup>5</sup>Since we have already changed expert placement, this EP is somehow different from typical EP and more like FlexMoE.

We have p = 4, q = 2. The group is  $(\mathbb{Z}_4 \times \mathbb{Z}_4, +)$ , and the generating set is  $\{(0,1), (0,-1), (1,0), (-1,0)\}$ . This graph is a  $4 \times 4$  toroidal grid graph, as shown in Figure 15a.

#### Example 3: 8 vertices, 16 edges.

We have p = 3, q = 2. The group is  $(\mathbb{Z}_2 \times \mathbb{Z}_4, +)$ , and the generating set is  $\{(0,1), (0,-1), (1,1), (1,-1)\}$ . The constructed graph is shown in Figure 15b, which is isomorphic to the complete bipartite graph  $K_{4,4}$ .

This construction satisfies a good property:  $\forall i \in 1,...,8$ , the maximum edge counts among all induced subgraphs with exactly i vertices is minimal.

#### Example 4: 8 vertices, 32 edges.

Note that a complete graph with 8 vertices has  $C_8^2 = 28$  edges. Since complete graphs are certainly optimal, we can first generate a complete graph and then add the remaining 32-28=4 edges. For the remaining 4 edges, we can simply create an edge between every vertex pair (0, 1), (2, 3), (4, 5), (6, 7) without using Cayley theory.

This method is generalizable to scenarios with more edges: We can first generate multiple complete graphs and then allocate the remaining edges. Note that the number of edges is a power of 2, the number of vertices is  $2^p$ , and the number of edges in a complete graph is  $\frac{2^p(2^p-1)}{2^p}$ . Consequently, the number of remaining edges must still be a power of 2, ranging from  $2^{p-1}$  to  $2^{2p-2}$ .

#### **B.3** Synchronization Consistency

Different EDP groups across experts can lead to a consistency issue during parameter and gradient synchronization. Specifically, the synchronization for different experts occurs in different EDP groups, which may incur deadlocks. To prevent deadlocks, we add a consistency restriction in expert placement: All replicas of an expert must have identical local expert indices. For example, in Figure 3c, the replicas of expert 2 are the first local replicas of both GPU 1 and GPU 2; the colors of edges in Figure 15 also indicate the local expert indices. Since DDP executes parameter synchronization following the order of local parameters (and gradient synchronization following the reverse order) [29], deadlocks are effectively avoided.

#### **C** Supplementary Experiments

## **C.1** Detailed Experimental Settings

This section provides detailed configurations for our experiments in §7. Table 2 lists the detailed hyperparameters for models used in §7.2.

Activation recomputation is a technique to reduce memory footprint by avoiding recording activations in the forward pass and recomputing them in the backward pass [5]. Furthermore, selective activation recomputation enables recomputing a subset of model modules to perform a fine-grained trade-off

<span id="page-18-0"></span>![](_page_18_Figure_13.jpeg)

Figure 16: Dispatch time of FineEP and EP with DeepEP and NCCL, varying number of GPUs.

between computation efficiency and memory [24]. We enable selective recomputation in Megatron-LM to recompute only the MoE FFN, avoiding the out-of-memory (OOM) issue while maintaining relatively high throughput. Since Deep-Speed currently does not support selective recomputation, we recompute the whole layer in DeepSpeed. Furthermore, we find that we can adjust the granularity of selective recomputation at runtime. When the expert loads are highly imbalanced, we can recompute the whole MoE layer for better memory efficiency, avoiding OOM. Otherwise, we can recompute only the MoE FFN for better computation efficiency. For fair comparison, we do not adjust the recomputation granularity during evaluation.

#### **C.2** Evaluation of DeepEP

We evaluate the dispatch time of FineEP and vanilla EP with DeepEP [9], a high-performance all-to-all communication backend. Megatron-LM [50] currently supports both NCCL [19] (by default) and DeepEP for all-to-all communication. We additionally implement FineEP with DeepEP.

Before analyzing the experimental results, two important experimental considerations should be noted: (1) Our testbed consists of 8 GPUs and only 2 NICs per node, resulting in limited inter-node network bandwidth. Therefore, we avoid employing EP or FineEP across multiple nodes in §7. However, in this section, we focus on the performance of different communication backends, so we expand the communication group into multiple nodes. Consequently, the all-to-all time for inter-node communication is significantly longer than the intra-node communication. (2) In §7, we focus on the system performance, so we compare FineEP using 8 GPUs per group with EP using 4 GPUs per group (d=2). However, in this section, we focus on the communication performance, so we compare FineEP and EP using the same group size.

Figure 16 shows the dispatch time comparison between

<span id="page-19-0"></span>

| Model                      | GPT 32×1.3B | GPT 16×3.2B | GPT 8×6.7B | Mixtral 16×2B | Mixtral 8×7B |
|----------------------------|-------------|-------------|------------|---------------|--------------|
| # layers                   | 24          | 16          | 32         | 32            | 32           |
| # attention heads          | 16          | 32          | 32         | 32            | 32           |
| hidden size                | 2048        | 4096        | 4096       | 2048          | 4096         |
| FFN hidden size            | 8192        | 16384       | 16384      | 8192          | 14336        |
| sequence length            | 2048        | 2048        | 2048       | 4096          | 4096         |
| # experts                  | 32          | 16          | 8          | 16            | 8            |
| top-K                      | 2           | 2           | 2          | 2             | 2            |
| micro batch size           | 4           | 2           | 2          | 2             | 1            |
| global batch size          | 512         | 512         | 512        | 256           | 256          |
| learning rate              | 1e-5        | 2e-6        | 1e-6       | 1e-5          | 1e-6         |
| load-balancing loss coeff. | 1e-4        | 1e-4        | 1e-4       | 1e-4          | 5e-4         |
| # GPUs                     | 16          | 16          | 32         | 16            | 32           |
| PP degree                  | 2           | 2           | 4          | 2             | 4            |
| EP degree                  | 4           | 4           | 4          | 4             | 4            |

<span id="page-19-1"></span>![](_page_19_Figure_2.jpeg)

Figure 17: Execution time breakdown of an MoE layer with FineEP, varying the levels of communication-aware scheduling.

FineEP and EP using both DeepEP and NCCL, varying number of GPUs. We use the same setting as in [§7.4,](#page-9-3) except for the all-to-all group size. DeepEP exhibits better performance than NCCL due to its high-performance all-to-all implementation. When using NCCL, FineEP requires less time than EP thanks to the locality-aware routing in [§5.2.](#page-5-0) However, when using DeepEP, FineEP requires more time than EP due to the data format incompatibilities between DeepEP and Megatron-LM. Consequently, Megatron-LM needs to pre-process the data for DeepEP, while FineEP incurs a higher pre-processing overhead than EP.

# <span id="page-19-2"></span>C.3 Evaluation of Communication-Aware Scheduling

We evaluate the performance of the communication-aware scheduling in Appendix [A.1.](#page-16-6) The communication-aware scheduling considers two levels of locality in token dispatching: GPU-level (intra-node) locality and node-level (internode) locality. We set α<sup>1</sup> = 0.1,α<sup>2</sup> = 1.0 as the weights of intra-node and inter-node communication in Problem [4.](#page-16-5) We use DeepEP as the communication backend due to its superior performance and reduced system overhead compared to NCCL. We believe that the dispatch time of DeepEP provides a more accurate reflection of the communication volume. For other parameters, we use 16 GPUs, 32 experts, hidden\_size=2048, sequence\_length=4, micro\_batch\_size=4. We use randomly generated tokens as input.

We compare the execution time of an MoE layer while enabling/disabling the GPU-level/node-level locality in the communication-aware scheduling. As shown in Figure [17,](#page-19-1) the overall execution time decreases as we consider more levels of locality during scheduling.

# C.4 Evaluation of FineEP with Pipelining

We evaluate the performance of FineEP with pipelining in Appendix [A.2.](#page-17-3) We enable the communication-aware scheduling and DeepEP. We use 8 GPUs and 128 experts. Other parameters are the same as Appendix [C.3.](#page-19-2)

We compare the dispatch time with varying ratios of data in FineEP. Figure [18](#page-20-0) demonstrates that pipelining can reduce the dispatch time by overlapping FineEP preparation with EP allto-all communication. However, the dispatch time increases as the FineEP ratio increases. This is because the EP all-toall time decreases and becomes insufficient to fully hide the FineEP scheduling time.

<span id="page-20-0"></span>![](_page_20_Figure_0.jpeg)

Figure 18: Dispatch time breakdown with pipelining, vary the ratios of data in FineEP (1.0 indicates no pipelining).