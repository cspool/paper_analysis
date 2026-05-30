# HD-MoE: Hybrid and Dynamic Parallelism for Mixture-of-Expert LLMs with 3D Near-Memory Processing

Haochen Huang1,2† , Shuzhang Zhong1,2,3,4† , Zhe Zhang3,<sup>4</sup> , Shuangchen Li3,<sup>4</sup> Dimin Niu3,<sup>4</sup> , Hongzhong Zheng3,<sup>4</sup> , Runsheng Wang2,5,<sup>6</sup> , Meng Li1,2,6<sup>∗</sup> 1 Institute for Artificial Intelligence, Peking University, Beijing, China 2 School of Integrated Circuits, Peking University, Beijing, China <sup>3</sup>DAMO Academy, Alibaba Group, Beijing, China <sup>4</sup>Hupan Lab, Hangzhou, China 5 Institute of Electronic Design Automation, Peking University, Wuxi, China <sup>6</sup>Beijing Advanced Innovation Center for Integrated Circuits, Beijing, China

*Abstract*—Large Language Models (LLMs) with Mixture-of-Expert (MoE) architectures achieve superior model performance with reduced computation costs, but at the cost of high memory capacity and bandwidth requirements. Near-Memory Processing (NMP) accelerators that stack memory directly on the compute through hybrid bonding have demonstrated high bandwidth with high energy efficiency, becoming a promising architecture for MoE models. However, as NMP accelerators comprise distributed memory and computation, how to map the MoE computation directly determines the LLM inference efficiency. Existing parallel mapping strategies, including Tensor Parallelism (TP) and Expert Parallelism (EP), suffer from either high communication costs or unbalanced computation utilization, leading to inferior efficiency. The dynamic routing mechanism of MoE LLMs further aggravates the efficiency challenges. Therefore, in this paper, we propose HD-MoE to automatically optimize the MoE parallel computation across an NMP accelerator. HD-MoE features an offline automatic hybrid parallel mapping algorithm and an online dynamic scheduling strategy to reduce the communication costs while maximizing the computation utilization. With extensive experimental results, we demonstrate that HD-MoE achieves a speedup ranging from 1.1× to 1.8× over TP, 1.1× to 1.5× over EP, and 1.0× to 1.4× over the baseline Hybrid TP-EP with Compute-Balanced parallelism strategies.

*Index Terms*—Automated Deployment, Mixture-of-Experts, 3D Near-Memory Processing

# I. INTRODUCTION

Mixture-of-Experts (MoE) has become a widely adopted architecture for Large Language Models (LLMs) [\[1,](#page-8-0) [2\]](#page-8-1). By selectively activating only a small subset of experts, MoE significantly reduces the computational demands while maintaining model capacity. However, the sparse activation mechanism exacerbates the memorybound problem, particularly on edge devices with limited memory bandwidth and small batch sizes.

The recent emerging 3D Near-Memory Processing (NMP) architectures seem to be a promising solution for memory-bound problems [\[3,](#page-8-2) [4\]](#page-8-3). 3D NMP vertically stacks DRAM dies directly on top of logic dies using high-bandwidth interconnects. In contrast to conventional von Neumann architectures, the vertical stacking of 3D NMP allows multiple memory banks to be accessed independently and in parallel, enabling fine-grained, high-throughput data access. This makes 3D NMP particularly suitable for MoE inference workloads.

This work was supported in part by NSFC under Grant 62495102 and Grant 92464104, in part by the National Key Research and Development Program under Grant 2024YFB4505004, in part by Beijing Municipal Science and Technology Program under Grant Z241100004224015, and in part by 111 Project under Grant B18001.

†Equal Contribution <sup>∗</sup>Corresponding author: meng.li@pku.edu.cn

While MoE's bandwidth efficiency makes it suitable for 3D NMP deployment, the architectural shift from GPU-style shared memory to distributed NoC-based designs introduces new mapping challenges. The distributed nature of 3D NMP, with its bank-local memory organization, requires careful co-design of expert parallelism and communication routing strategies to maintain performance. As illustrated in Figure [1,](#page-1-0) current approaches employ either Tensor Parallelism [\[5\]](#page-8-4) (TP) or Expert Parallelism [\[6\]](#page-8-5) (EP): TP distributes each expert's parameter tensor across banks while EP assigns complete experts to different banks. This presents a fundamental trade-off: TP achieves better workload balance but incurs substantial all-reduce communication overhead, whereas EP minimizes communication but suffers from workload imbalance due to varying expert utilization.

Previous works on GPU clusters have explored combining EP with replication of frequently activated experts to achieve both workload balance and low communication overhead. This method has been adopted by DeepSeek-AI to deploy its DeepSeek-R1 model [\[1\]](#page-8-0). However, this approach is impractical for 3D NMP due to its limited memory capacity. Furthermore, the dynamic and imbalanced nature of expert activation patterns significantly complicates mapping and scheduling decisions, requiring more sophisticated optimization strategies tailored to the constraints of 3D NMP architectures.

To address the challenge of dynamic expert activation, several studies focusing on offloading scenarios have investigated dynamic scheduling of experts [\[7](#page-8-6)[–10\]](#page-8-7). In these scenarios, experts are stored in secondary storage, with on-demand loading becoming the primary bottleneck. These studies demonstrate that MoE models often exhibit high activation similarity between adjacent layers, which can be exploited for prefetching to alleviate the on-demand loading overhead.

In light of these challenges and opportunities, we propose HD-MoE, a hybrid and dynamic parallelism framework designed for MoE inference on 3D NMP architectures. Reducing latency on distributed systems requires both balancing computation utilization and minimizing communication cost, while also addressing memory limitations. To achieve this, HD-MoE adopts a hybrid parallelism approach, as illustrated in Figure [1\(](#page-1-0)c). Experts with low activation frequency are mapped using Expert Parallelism to minimize communication overhead, while high-frequency experts utilize Tensor Parallelism to maximize computational resource utilization. Additionally, HD-MoE incorporates an online dynamic expert placement strategy to mitigate the impact of the dynamic activation pattern. The related codes can be accessed at [https://github.com/angerybob/HD-MoE.](https://github.com/angerybob/HD-MoE)

The key contributions of HD-MoE are summarized as follows:

• Performance Analytical Model. We develop a unified per-

<span id="page-1-0"></span>![](_page_1_Figure_0.jpeg)

Fig. 1: Expert Deployment and Computation Timeline in (a) Tensor Parallel, (b) Expert Parallel, and (c) Hybrid parallel.

<span id="page-1-1"></span>![](_page_1_Figure_2.jpeg)

Fig. 2: MoE structure and two parallel strategies

formance analysis framework applicable to diverse hardware configurations and parallelism strategies.

- Automated Hybrid Parallelism. We propose an efficient placement strategy searching method that combines TP and EP to optimize computation and communication overheads.
- Dynamic Placement. We introduce a dynamic expert placement strategy, which adjusts expert deployment in real-time based on the inference workload, ensuring optimal performance even in different inference scenarios.
- We conduct extensive experiments to validate our approach, demonstrating significant improvements in both TBT latency and speedup compared to baseline methods.

#### II. BACKGROUND

# A. Mixture-of-Experts Models

Mixture-of-Experts (MoE) improves scalability by activating a small subset of specialized subnetworks per input, reducing computation and memory costs. A gating mechanism selects experts per token, enabling efficient large-scale and multi-task learning. Recent MoE models such as Mixtral [2], Qwen [11], and DeepSeek [1, 12] have shown significant performance and scalability gains.

MoE optimization has been explored across inference and training. Offloading methods target hybrid CPU-GPU deployment for throughput efficiency [7, 13]; serving-oriented works improve expert scheduling and placement [14–16]; and training strategies integrate data, tensor, and expert parallelism to enhance scalability [6, 17–19]. These approaches highlight the need for adaptable solutions under diverse constraints in memory, computation, and communication.

# B. Near-Memory Processing Architectures

Traditional von Neumann architectures, including GPUs, face increasing limitations from the memory wall, particularly in sparse, bandwidth-intensive models like MoEs. Processing-in-Memory (PIM)

<span id="page-1-2"></span>TABLE I: Comparison of MoE Deployment Scenarios on Different Hardware Architectures.

|         | Scenario       | Memory      | Capacity | Parallelism  |
|---------|----------------|-------------|----------|--------------|
| GPU     | Edge Inference | Shared      | Low      | TP           |
| Cluster | Cloud Serving  | Distributed | High     | EP+Replicate |
| 3D NMP  | Edge Inference | Distributed | Low      | Hybrid       |

alleviates memory latency by colocating computation with data [20], but its low compute density, due to logic-storage contention, limits scalability. In contrast, 3D Near-Memory Processing (NMP) vertically integrates DRAM and logic dies to provide high bandwidth and moderate compute density, making it more suitable for bandwidth-bound workloads.

An example is **Hybrid Bonding DRAM** [21–24], which employs fine-pitch vertical interconnects for high-bandwidth links between memory and logic, supporting large-scale sparse models efficiently.

## C. Distributed Inference Strategies

Distributed inference on 3D NMP architectures requires efficient parallelism to utilize compute and memory resources, with Tensor Parallelism (TP) and Expert Parallelism (EP) as two primary strategies. As shown in Fig. 2b, TP splits expert parameters across processing elements (PEs), achieving balanced computation but incurring costly all-reduce communication that scales with batch size. EP (Fig. 2c) assigns entire experts to separate PEs, reducing communication per token but causing load imbalance and irregular all-to-all traffic due to sparse, dynamic expert activation.

Fig. 2 compares the two strategies. Our approach combines them in a hybrid framework to optimize MoE inference on 3D NMP systems, efficiently balancing compute and communication overhead.

## III. MOTIVATION

Challenge 1: MoE Deployment Challenges Unique to 3D NMP Architectures. 3D NMP architectures offer high bandwidth but lack shared memory across compute units, resulting in a distributed execution environment distinct from traditional GPUs or clusters.

In single-GPU systems, global memory and shared L2 cache enable low-latency communication among SMs, making Tensor Parallelism (TP) effective for balancing workloads. In contrast, cluster deployments favor Expert Parallelism (EP) to reduce inter-node communication, often replicating hot experts to improve load balance with minimal memory overhead—especially feasible when KV cache dominates memory use.

However, these strategies are ill-suited for 3D NMP. As shown in Table I, TP incurs prohibitive communication overhead due to limited NoC bandwidth, while EP with replication exceeds the memory budget. This calls for a hybrid parallelism strategy tailored to 3D NMP constraints, balancing communication cost, workload skew, and tight storage capacity for scalable MoE inference.

Challenge 2: Imbalanced and Dynamic Expert Activation of MoE. The expert selection frequency in MoE is inherently uneven, leading to significant workload imbalance. For instance, as shown in Fig. 3(a), nearly half of the tokens in certain layers of Qwen2 converge to a single expert, while others remain underutilized [25]. This imbalance creates both computational and communication inefficiencies. Overloaded experts become performance bottlenecks, while underutilized experts waste computing resources. Furthermore, the skewed workload triggers inefficient all-to-all communication patterns in 2D mesh networks, increasing latency and reducing

<span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

Fig. 3: (a) Expert Activation Frequency, (b) Expert Activation Overlap Between Adjacent Iterations, (c) Expert Routing Affinity, The value at position (i,j) in the figure represents the conditional probability of expert j being activated, given that expert i is activated, (d) Prediction Accuracy of Expert Activation.

effective bandwidth. Together, these effects significantly degrade system performance.

Dynamic expert activation exacerbates these challenges. As illustrated in Figure 3(b), expert selection exhibits high variability across iterations, with some experts experiencing heavy utilization in one iteration and minimal usage in the next. Thus, an efficient MoE deployment must address both static imbalance and dynamic variability to optimize compute and communication efficiency.

Opportunity 1: Activation Dependency in Expert Groups. Despite MoE activation imbalance, experts show strong co-activation affinity. For example, when expert A is activated, expert B is more likely to be activated, forming a distinct co-activation pattern (Fig. 3(c)). This pattern enables optimization by co-locating frequently co-activated experts in 3D NMP, reducing data movement, optimizing communication, and balancing device loads. The key challenge is leveraging activation dependencies and 3D NMP architecture features to address load imbalance and communication latency, improving MoE inference performance.

**Opportunity 2: Expert Activation Prediction.** MoE models exhibit strong activation similarity between adjacent layers due to residual connections, enabling accurate prediction of future expert activations. As shown in Figure 3(d), the gate functions of subsequent layers achieve high prediction accuracy, providing a promising approach to mitigate the challenges of dynamic activation patterns. By leveraging these predictions, we can proactively optimize resource allocation, reduce communication overhead, and improve load balancing in distributed systems.

# IV. HD-MoE DESIGN

# A. Overview

Fig. 4 provides an overview of the HD-MoE framework, which consists of an offline mapping phase and an online inference phase. The offline phase involves automated hybrid expert mapping on 3D NMP with Node Balance (Sec. IV-C1) and Link Balance (Sec. IV-C2) optimization techniques. During online inference, dynamic scheduling is employed to predict computation load, prioritize experts (Sec. IV-D1), and pre-broadcast the expert with highest load (Sec. IV-D2),

<span id="page-2-1"></span>![](_page_2_Figure_9.jpeg)

Fig. 4: Overview of HD-MoE

ensuring efficient resource utilization without inducing additional communication overhead (Sec. IV-D3).

## B. Performance Analytical Model

We first build a performance analytical model to estimate the total inference cost, including computation and communication estimation.

1) Computation Overhead Modeling: The computation time  $t_{\rm comp}$  is determined by the maximum load across all computing nodes, reflecting expert utilization imbalance. For each node c, the load depends on the placement matrix  $P_{ic}$ , where  $P_{ic}$  denotes the proportion that expert i is assigned to node c. The explanation of other parameters is listed in the table II. The formula is defined as:

$$t_{\text{comp}} = \max_{c} \left\{ \frac{\sum_{i=0}^{E-1} P_{ic} f_i B \cdot 2h \cdot IS}{\text{comp}} \right\}$$
 (1)

Here,  $2h \cdot IS$  is the computation volume per token,  $f_iB$  denotes the number of tokens that activate expert i, and  $P_{ic}f_iB$  is the effective number of tokens that node c needs to process.

We emphasize that the variables  $P_{ic}$  are continuous rather than binary, allowing our hybrid placement strategy to partially distribute expert computation across multiple nodes (similar to Tensor Parallelism). This design improves deployment flexibility, alleviates hotspots, and enhances compute balance.

However, this introduces a trade-off with communication overhead. Splitting experts across nodes requires data transfers, which can increase both transfer volume and communication irregularity. Balancing computation load and communication overhead is crucial for optimal MoE performance on 3D PNM architectures.

<span id="page-2-2"></span>2) Communication Overhead Modeling: We adopt a discrete-event simulation framework to model irregular all-to-all communications in 2D mesh architectures, extracting the total schedule time as communication overhead.

**Discrete-Event Simulation for Accurate Latency Estimation.** We build a discrete-event simulator to model the latency of irregular all-to-all communication in 2D mesh networks, which mainly consists of the following parts:

TABLE II: Explanation of Parameter Meanings.

<span id="page-3-1"></span>

| Parameter                 | Meaning                                     |  |  |
|---------------------------|---------------------------------------------|--|--|
| $t_{\rm comp}$            | Computation time                            |  |  |
| $t_{\rm comm}$            | Communication time                          |  |  |
| $\hat{t}_{\mathrm{comm}}$ | Linear Approximation for communication time |  |  |
| c                         | Index of computing nodes                    |  |  |
| $\overline{E}$            | Total number of experts                     |  |  |
| $\overline{e}$            | Number of experts activated per token       |  |  |
| $P_{ic}$                  | Proportion of expert i placed on node c     |  |  |
| $f_i$                     | Activation frequency of expert i            |  |  |
| B                         | Batch size                                  |  |  |
| h                         | Hidden dimension                            |  |  |
| IS                        | MoE intermediate size                       |  |  |
| D                         | Number of nodes                             |  |  |
| comp                      | Computational throughput of a node          |  |  |
| BW                        | NoC link bandwidth                          |  |  |
| G                         | Expert groups                               |  |  |
| $\overline{g}$            | Index of expert groups                      |  |  |
| $f_g$                     | Co-activation frequency of expert group $g$ |  |  |

- Communication Task Generation: For each token's activated expert group, we first map experts to their physical nodes (src) and identify a target node (dst) for aggregation (e.g., randomly choose a node that is activated by this token).
- XY Routing Path Calculation: We use a cached XY routing algorithm to generate minimal-hop paths (Manhattan distance) between source and destination nodes.
- Event Scheduling and Link Management: Communication tasks are split into chunks and scheduled in a priority queue. Transmission time is calculated based on available bandwidth and link occupancy. A link schedule dictionary tracks link availability, ensuring tasks are scheduled promptly and avoiding collisions.

**Linear Approximation for Optimization.** To enable efficient deployment optimization via linear programming (LP), we approximate the communication latency using a node-traffic model:

$$\hat{t}_{\text{comm}} = \frac{4}{\text{BW}} \max_{c} \left\{ \sum_{g \in G} \left( \prod_{i \in g} \lceil P_{ic} \rceil \right) f_g B h \right\}$$
 (2)

Here,  $\lceil P_{ic} \rceil$  indicates whether expert i is activated on node c, and  $\prod_{i \in g} \lceil P_{ic} \rceil$  checks if any expert in group g is placed on node c. If any expert in group g is placed, the data volume for transfer is  $4f_gBh$ , assuming FP32 representation. The total communication volume that node c needs to send is given by  $\sum_{g \in G} \left(\prod_{i \in g} \lceil P_{ic} \rceil\right) 4f_gBh$ .

We empirically validate the accuracy of the proposed linear approximation model. By comparing the estimated communication latency  $\hat{t}_{\text{comm}}$  with the latency values obtained from simulation  $t_{\text{comm}}$ , we observe a strong linear correlation between the two. The fitting results are shown in Fig. 5.

As a result, the relationship can be approximated as:

$$t_{\text{comm}} = \gamma \hat{t}_{\text{comm}} \tag{3}$$

where  $\gamma$  is a scaling coefficient determined through linear regression. Experimental results show that the coefficient of determination  $(R^2)$  consistently exceeds 0.9 across various scenarios, confirming the model's reliability in estimating communication overhead for LP.

In structured communication patterns like ring-based all-reduce [26], commonly used in TP training and inference, the communication algorithm is regular and deterministic. In such cases, total

<span id="page-3-2"></span>![](_page_3_Figure_13.jpeg)

Fig. 5: The fitting results illustrating the relationship between schedule-based communication mechanisms and node communication patterns ( $R^2 = 0.96$ ).

<span id="page-3-3"></span>TABLE III: Comparison between the analytical modeling and ASTRA-sim-based simulation results for Ring-AllReduce.

| Latency | Bandwidth | Analytical Results | Simulation Results |
|---------|-----------|--------------------|--------------------|
| 0.1 us  | 25 Gb/s   | 673 us             | 668 us             |
| 5 us    | 25 Gb/s   | 751 us             | 879 us             |
| 0       | 20 Gb/s   | 671 us             | 692 us             |

communication time nearly equals per-node communication time due to the algorithm's balanced nature.

$$t_{\rm comm} \approx \hat{t}_{\rm comm} \approx \frac{4Bh}{\rm BW}$$
 (4)

This implies that  $\gamma=1$  for ring all-reduce, demonstrating that our linear communication model is not only accurate for irregular all-to-all traffic, but also directly applicable to structured communication paradigms such as tensor parallel all-reduce.

We validate our analytical model by comparing its results with the widely used distributed deep learning simulator ASTRA-sim [27] for the ring all-reduce operation. As shown in the table III, the latency predictions from the analytical model closely match the simulation results, demonstrating strong alignment between the two. This confirms that our analytical model provides a reliable estimate of latency and can be effectively used for performance prediction and optimization in similar distributed systems.

#### C. Optimal Placement Strategy Searching

We propose a two-stage **Node-Link Balance Co-optimization** strategy for efficiently deploying MoE models on 3D NMP architectures. The placement problem is divided into a logical optimization stage that balances computational load and reduces communication volume, and a physical mapping stage to minimize link-level congestion. This separation of logical workload and physical interconnects simplifies the placement problem, as detailed in the following stages.

<span id="page-3-0"></span>1) Node Balance Optimization via Linear Programming: In the first stage of our co-optimization framework, we focus on balancing computational and communication workloads across logical compute clusters, abstracting away their physical topology. Given the large number of variables and the combinatorial nature of the expert placement problem, manual tuning becomes infeasible. Therefore, we adopt a linear programming (LP) formulation to enable automated and scalable optimization across diverse hardware configurations and inference scenarios. The LP model simultaneously considers computation bottlenecks and approximated communication costs using the linear estimator  $\hat{t}_{\text{comm}}$  derived earlier. The LP Optimization Problem is modeled as follows:

The notations of the LP formulation are defined in Table II. Among them, the continuous variables  $P_{ic} \in [0, 1]$  represent the proportion

of expert i's workload assigned to cluster c. The binary variables  $Z_{ic} \in \{0,1\}$  indicates whether expert i is active on cluster c.

Then we define some constraints to guarantee the legal mapping and efficiently search for the optimal allocation strategy.

$$Z_{ic} \ge P_{ic}, \quad \forall (i, c)$$
 (5)

$$\sum_{c} P_{ic} = 1, \quad \forall i$$
 (6)

$$t_{\text{comp}} \ge \frac{\sum_{i=0}^{E-1} P_{ic} f_i B \cdot 2h \cdot IS}{\text{comp}}, \quad \forall c$$
 (7)

$$t_{\text{comm}} \ge \frac{4Bh}{\text{BW}} \sum_{g \in G} f_g \left( \sum_{i \in g} Z_{ic} \right), \quad \forall c$$
 (8)

$$0 < \sum_{i}^{E-1} P_{ic} f_i \le \left(\frac{1}{R_{CC}} + 1\right) \frac{e}{D}, \quad \forall c$$
 (9)

$$R_{CC} = \frac{t_{\text{comp}}}{t_{\text{TP,comm}}} = \frac{BW \cdot IS \cdot e}{2D \cdot \text{comp}}$$
 (10)

Constraints 5 and 6 handle expert placement and workload assignment across nodes. Constraints 7 and 8 ensure balanced computation and communication times, minimizing bottlenecks.

Constraints 9 and 10 restrict node workload to avoid imbalance, with an upper bound set by the theoretical compute + communication time of TP inference. These constraints prune suboptimal placements early, reducing search space complexity and improving solver convergence without sacrificing optimality.

Finally, we can represent and minimize the node-level inference overhead, defined as a combination of computation time and communication time:

$$\min t_{\text{node\_overhead}}$$
 (11)

$$t_{\text{node\_overhead}} = t_{\text{comp}} + 2t_{\text{comm}} = t_{\text{comp}} + 2\gamma \hat{t}_{\text{comm}}$$
 (12)

Here,  $\gamma$  is the scaling coefficient empirically derived from simulation (Sec. IV-B2), and the factor 2 accounts for the cost of all-to-all dispatch and all-to-all gather, which constitute a pair of symmetric communication operations.

This LP formulation enables a globally coordinated placement strategy that balances computation and communication, providing a foundation for the second stage of physical mapping.

<span id="page-4-0"></span>2) Link Balancing via Bayesian Optimization: In this stage, the logical clusters are mapped to physical nodes on the 2D mesh network. The objective here is to minimize link congestion and improve communication tail latency. We adopt Bayesian Optimization to search for low-congestion mapping strategies, as it is well-suited for problems with expensive evaluations and relatively smooth objective functions—e.g., swapping nearby clusters causes only minor changes in communication cost. This enables efficient exploration of the mapping space with minimal simulation overhead.

Figure 6 illustrates example placements under four parallelism strategies, highlighting the trade-offs between computation balance and communication overhead in MoE inference. MoE communication can be categorized into Intra-Expert Communication, where a single expert is split across nodes and requires result aggregation (as in Tensor Parallelism), and Inter-Expert Communication, where multiple experts activated by the same token need to exchange and aggregate results. In (a), TP achieves balanced computation by splitting all experts across nodes, but incurs heavy Intra-Expert communication as each token triggers all-reduce operations. In (b), EP avoids Intra-Expert communication, with only sparse Inter-Expert communication—e.g., between E1 (Expert 1) and E3 via Node 4. However,

<span id="page-4-8"></span><span id="page-4-7"></span><span id="page-4-6"></span><span id="page-4-5"></span><span id="page-4-4"></span><span id="page-4-3"></span><span id="page-4-2"></span>![](_page_4_Picture_17.jpeg)

Fig. 6: (a) TP: balanced computation but heavy communication; (b) EP: light communication but imbalanced computation, (c) Hybrid parallel with node balance: balanced computation but irregular communication; (d) Hybrid parallel with node-link balance: balanced computation and uncongested, regular communication.

tokens T1, T2, and T3 all activate E3, leading to overload on Node 3 and poor resource utilization. In (c), Node Balance optimization redistributes part of E3 to Node 4 to balance computation. Yet, this split introduces Intra-Expert communication between Nodes 3 and 4, causing link congestion along the  $3\rightarrow4$  path while leaving other links underutilized. In (d), Node-Link Balance adjusts the physical mapping (e.g., swapping Node 2 and Node 3), allowing the synchronization between Nodes 3 and 4 to be routed via  $3\rightarrow1\rightarrow4$  and  $3\rightarrow2\rightarrow4$ , which alleviates link congestion and achieves both balanced computation and regular, uncongested communication.

#### D. Dynamic Placement Strategy

We propose a runtime-adjustable deployment strategy for dynamic expert routing in MoE inference, consisting of three key components: congestion-aware expert prediction, cost-optimal broadcasting, and communication-efficient token routing.

<span id="page-4-1"></span>1) Priority Detection and Computation Prediction: We leverage temporal locality in expert activations to predict computation hotspots in the next layer. For each expert i on node c, we define a priority score that estimates its future compute cost:

$$prio_{ic} = \frac{2P_{ic}\hat{f}_i \cdot IS}{comp} \tag{13}$$

Here,  $\hat{f}_i$  is the predicted activation frequency of expert i. The expert with the highest priority on the most congested node is selected for

pre-broadcast, repeating this process for a limited number of iterations based on the previous layer's inference latency.

- <span id="page-5-0"></span>2) Optimal Broadcast Chunk Size: Broadcasting an expert involves splitting it into chunks of size c, with the following trade-offs:
  - Larger chunks reduce hops, lowering latency.
  - Smaller chunks reduce per-hop traffic but increase transmission delays.

The traditional  $\alpha$ - $\beta$  communication model can clearly describe this kind of trade-off:

$$latency = \alpha (2\sqrt{D} + \frac{h \cdot IS}{c})$$
 (14)

$$bandwidth = \beta(h \cdot IS + 2c\sqrt{D})$$
 (15)

$$t_{\text{pre b}} = \text{latency} + \text{bandwidth}$$
 (16)

Here, k is the number of pre-broadcast iterations allowed within the runtime window. The model yields a lower bound:

$$t_{\text{pre\_b}} \ge h \cdot IS \cdot \beta k + 2\alpha \sqrt{D} + 2\sqrt{2\sqrt{D}\beta k\alpha h \cdot IS}$$
 (17)

This bound is tight when the chunk size c is selected optimally as:

$$c = \sqrt{\frac{\alpha h \cdot IS}{2\beta k \sqrt{D}}} \tag{18}$$

This provides a solution for the most efficient pre-broadcast under a given runtime window constraint.

<span id="page-5-1"></span>3) Communication-Efficient Dispatch: After expert broadcasting, each token can be routed to any node holding a copy of its activated experts. To avoid incurring additional communication overhead, we restrict the routing candidates to nodes where the routed experts are already present. Among these candidates, the token is dispatched to the node with the lowest current compute load, minimizing workload imbalance without introducing extra data movement.

An example of dynamic expert scheduling is illustrated in Fig. 7. Subfigure (a) shows the static deployment from Fig. 6, which performs efficiently under averaged expert activation patterns. However, during real inference, as shown in (b), expert activation becomes highly skewed—Expert 1 (E1) turns into a bottleneck due to concentrated token routing. Our priority detection mechanism can anticipate such overload at runtime, identifying E1 as a high-demand expert in advance. As shown in (c), E1 is then **pre-broadcast** to all nodes before execution. Tokens such as T2 and T4 are routed to Node 2 and Node 4 respectively, both holding E1, thus avoiding additional Inter-Expert communication. This not only balances the computation load but also improves communication efficiency.

## V. EXPERIMENTAL RESULTS

# A. Experimental Setup

- 1) Models: We evaluate the performance of our proposed approach using three MoE models: Mixtral-8x7B-Instruct [2] (mixtral), DeepSeek-V2-Lite-Chat [12] (deepseek) and Qwen2-57B-A14B-Instruct [25] (qwen). All the models are large-scale MoE architectures that benefit from expert parallelism and tensor parallelism, and they are deployed on 3D NMP architectures with different mesh sizes and hardware configurations. The key parameters for both models are summarized in Table IV.
- 2) Baselines: The baseline deployment strategies include Tensor Parallelism (TP), Expert Parallelism (EP), and a hybrid TP-EP approach with compute balance. In the hybrid strategy, the 2D mesh is divided into sub-regions—8 for deepseek and qwen, and 2 for mixtral. Each sub-region applies EP, with TP used internally to parallelize expert computation. Experts are appropriately assigned to sub-regions

<span id="page-5-2"></span>![](_page_5_Picture_19.jpeg)

Fig. 7: (a) Static deployment, (b) Computation load detection, (c) Pre-broadcast the expert with the highest load and dispatch tokens to appropriate nodes without inducing additional communication overhead.

<span id="page-5-3"></span>TABLE IV: Model Parameters for mixtral, deepseek and qwen

| Parameter                   | mixtral | deepseek | qwen |
|-----------------------------|---------|----------|------|
| Number of Experts           | 8       | 64       | 64   |
| Experts per Token (Routing) | 2       | 6        | 8    |
| Number of Layers            | 32      | 27       | 28   |
| Hidden Size                 | 4096    | 2048     | 3584 |
| Intermediate Size           | 14336   | 1408     | 2560 |

to balance computation load, with each expert placed in only one sub-region. This strategy is widely used to mitigate load imbalance in large-scale systems.

3) Evaluation Metrics: We evaluate the performance of our approach and the baselines using the following metrics:

**Normalized TBT (Time-Between-Tokens):** The latency between tokens during inference divided by that latency in Tensor Parallelism.

**MoE Decomposed Latency:** The time taken to process a batch of tokens in MoE layers, including both computation and communication time.

- Computation Latency: The time spent on performing computations within each node.
- Communication Latency: The time spent on transferring data between nodes.
- 4) Dataset: We use the MT Bench dataset [28] for evaluation, which is a widely adopted benchmark for LLMs, designed to measure the performance of LLMs on various tasks.

*5) Offline Optimization:* The proposed Optimal Placement Strategy Searching process typically takes several hours for the entire procedure, which is acceptable as it only needs to be performed once.

# *B. End-to-End Performance*

In this section, we evaluate the end-to-end performance of the proposed Node-Link Balance strategy across different hardware configurations and 2D mesh sizes inferred in different batch sizes. The experiments are conducted using three hardware configurations, each with varying compute throughput and communication bandwidth:

- 2.5 TFLOPS compute throughput, 75 GB/s bandwidth
- 5 TFLOPS compute throughput, 50 GB/s bandwidth
- 10 TFLOPS compute throughput, 25 GB/s bandwidth

Additionally, we compare the performance across three different 2D mesh sizes: (4,4), (4,8), and (8,8), with 5 TFLOPS compute throughput and 50 GB/s bandwidth for consistency.

Better TBT latency through different hardware configurations: The results of these experiments, shown in Fig. [8](#page-7-0) and Fig. [9.](#page-7-1) Results in Fig. [8](#page-7-0) reveal how different methods respond to shifts in compute-to-communication ratios. When computation is limited and communication bandwidth is abundant (2.5 TFLOPS, 75 GB/s), EP suffers from severe workload imbalance, resulting in suboptimal TBT latency. In contrast, when computation is sufficient but communication becomes a bottleneck (10 TFLOPS, 25 GB/s), TP incurs heavy all-reduce communication costs, leading to degraded latency performance. Additionally, worth noting is that, for qwen, the expert routing exhibits high imbalance (Fig. [3\(](#page-2-0)a)), causing significant overhead in EP.

The Hybrid TP-EP with Compute-Balanced baseline achieves better performance by distributing expert load more evenly, but ignores communication topology, leading to degraded performance under constrained bandwidth.

In contrast, our Node-Link Balance strategy jointly considers both computation and communication during expert placement. It minimizes per-node compute load, inter-node communication volume, and per-link congestion. As a result, it consistently outperforms all baselines across different system configurations.

On average, our method achieves a speedup ranging from 1.1× to 1.8× compared to TP, 1.1× to 1.5× compared to EP, and 1.0× to 1.4× compared to Hybrid TP-EP with Compute-Balanced.

Better TBT latency through different mesh size: We evaluate the impact of mesh topology on TBT latency under a fixed configuration (5 TFLOPS, 50 GB/s). As shown in Fig. [9,](#page-7-1) our Node-Link Balance strategy consistently achieves low latency across mesh sizes, demonstrating strong adaptability.

An exception occurs in mixtral with an (8,8) mesh, where the Hybrid TP-EP with Compute-Balanced baseline achieves slightly better latency. This is likely due to mixtral's small number of experts, which must be spread across multiple nodes. In large mesh topologies, where communication regularity is more critical, the hybrid baseline benefits from its structured TP communication and moderate message volume.

Overall, our method remains effective across models and mesh sizes, particularly when the number of experts and the topology scale are well matched.

# *C. Ablation Study*

We further conduct an ablation study focused on the contribution of the Node Balance, Link Balance, and Dynamic scheduling optimization for deepseek.

*1) Node Balancing:* We first evaluate the effect of the Node Balance stage in Fig. [10.](#page-7-2) It achieves 1.0× to 3.0× speedup over TP and EP, and 1.5× over Hybrid TP-EP across various configurations, by improving compute load balance (vs. EP) and reducing communication volume (vs. TP and hybrid).

Better computation latency: Next, we specifically examine Node Balance's effect on compute imbalance in EP, by measuring its impact on computation latency within MoE layers As shown in Fig. [11,](#page-7-3) on average, Node Balance reduces EP's compute tail latency by 2.0×, confirming its effectiveness in mitigating the routing skew commonly observed in MoE models.

Better load balance: Fig. [12](#page-7-4) shows per-node compute and communication load before and after applying Node Balance. The optimized placement achieves noticeably better load balance than standard EP, which often exhibits severe hotspots.

*2) Link Balancing:* We further evaluate the contribution of the Link Balance stage by isolating its impact on communication latency in Fig. [13.](#page-7-5) Specifically, we compare against three baselines: TP, Hybrid TP-EP with Compute-Balanced, and the Node Balance without physical mapping optimization.

Better communication latency: Thanks to the Bayesian Optimization–based mapping strategy, Link Balance produces more communication-friendly mappings by assigning logical clusters to physical nodes in a topology-aware manner. This significantly reduces link congestion and results in lower communication latency than TP and hybrid baselines, which rely on regular but heavy communication.

Compared to the Node Balance–only deployment, Link Balance can also achieve an average 1.2× reduction in communication latency, highlighting the importance of mapping logical clusters to physical nodes with awareness of network structure.

Less link congestion: Fig. [14](#page-7-6) visualizes NoC link utilization using heatmaps. Compared to the Node Balance stage, the optimized placement after Link Balance leads to visibly more balanced linklevel traffic, with less link congestion and better distribution across the mesh.

*3) Dynamic Placement Strategy:* To assess the impact of the Dynamic Placement Strategy, we compare the performance of static and dynamic expert placement strategies. In these experiments, we sample multiple expert routing traces in various types of questions from the MT Bench dataset, focusing on tasks with varying expert activation patterns. We compare the latency and speedup between static (generating from reasoning questions) and dynamic strategies under two different hardware configurations and broadcasting settings:

Hardware Configuration: (5 TFLOPS, 50 GB/s bandwidth) with 512 batch size, which has enough time to pre-broadcast 2 experts per layer, the results are shown in Fig. [15\(](#page-7-7)a).

Hardware Configuration: (2.5 TFLOPS, 75 GB/s bandwidth) with 512 batch size, which has enough time to pre-broadcast 5 experts per layer, the results are shown in Fig. [15\(](#page-7-7)b).

Better performance in various scenarios: The results are shown in Fig. [15,](#page-7-7) which indicates that the Dynamic Placement Strategy provides significant speedups and maintains relatively stable inference latency across a variety of real-time inference scenarios. Notably, for tasks such as math and coding problems, which have huge differences from reasoning, the dynamic approach significantly reduces MoE layer latency compared to static deployments. Specifically, when broadcasting 2 experts per layer, the average speedup achieved by the dynamic strategy is 1.15×, and when broadcasting 5 experts per layer, the average speedup increases to 1.25×.

These results highlight the effectiveness of dynamic expert scheduling in reducing latency by adapting to inference-time workload and

<span id="page-7-0"></span>![](_page_7_Figure_0.jpeg)

Fig. 8: End-to-end Speedup for Different Hardware Configurations

<span id="page-7-1"></span>![](_page_7_Figure_2.jpeg)

Fig. 9: End-to-end Speedup for Different Mesh Shapes

<span id="page-7-2"></span>![](_page_7_Figure_4.jpeg)

Fig. 10: Node Balancing Speedup for DeepSeekMoE

<span id="page-7-3"></span>![](_page_7_Figure_6.jpeg)

Fig. 11: Node Balancing Speedup in Computation for DeepSeekMoE

<span id="page-7-4"></span>![](_page_7_Figure_8.jpeg)

Fig. 12: Visualization of Node-Level Resource Utilization With and Without Node Balance Optimization

<span id="page-7-5"></span>![](_page_7_Figure_10.jpeg)

Fig. 13: Link Balancing Speedup for DeepSeekMoE

<span id="page-7-6"></span>![](_page_7_Figure_12.jpeg)

Fig. 14: Visualization of Link-Level Resource Utilization With and Without Link Balance Optimization

<span id="page-7-7"></span>![](_page_7_Figure_14.jpeg)

Fig. 15: Latency and Speedup Comparison Between Static and Dynamic Placement Strategies Under Varying Inference Scenarios. (a) Pre-broadcast 2 experts, (b) Pre-broadcast 5 experts.

## VI. CONCLUSION

This paper presents **HD-MoE**, an offline **Automatic Hybrid Parallelism** strategy, combined with online **Dynamic Scheduling**, for efficiently deploying MoE models on 3D NMP architectures. By integrating Node Balance, Link Balance, and Dynamic Placement, our approach effectively reduces computation and communication latency in MoE layers, improving both load balancing and resource utilization. Experimental results show that our method outperforms baseline strategies, achieving a speedup ranging from  $1.1 \times$  to  $1.8 \times$  over TP and  $1.1 \times$  to  $1.5 \times$  over EP. These findings demonstrate the value of optimizing expert placement and dynamic scheduling for MoE deployment on NMP architectures.

improving both computation and communication efficiency.

# REFERENCES

- <span id="page-8-0"></span>[1] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan *et al.*, "Deepseek-v3 technical report," *arXiv preprint arXiv:2412.19437*, 2024.
- <span id="page-8-1"></span>[2] A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. d. l. Casas, E. B. Hanna, F. Bressand *et al.*, "Mixtral of experts," *arXiv preprint arXiv:2401.04088*, 2024.
- <span id="page-8-2"></span>[3] M. Horowitz, "1.1 computing's energy problem (and what we can do about it)," in *2014 IEEE international solid-state circuits conference digest of technical papers (ISSCC)*. IEEE, 2014, pp. 10–14.
- <span id="page-8-3"></span>[4] V. Iskandar, M. A. A. E. Ghany, and D. Goehringer, "Nearmemory computing on fpgas with 3d-stacked memories: Applications, architectures, and optimizations," *ACM Transactions on Reconfigurable Technology and Systems*, vol. 16, no. 1, pp. 1–32, 2022.
- <span id="page-8-4"></span>[5] M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, and B. Catanzaro, "Megatron-lm: Training multi-billion parameter language models using model parallelism," *arXiv preprint arXiv:1909.08053*, 2019.
- <span id="page-8-5"></span>[6] D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, and Z. Chen, "Gshard: Scaling giant models with conditional computation and automatic sharding," *arXiv preprint arXiv:2006.16668*, 2020.
- <span id="page-8-6"></span>[7] S. Zhong, L. Liang, Y. Wang, R. Wang, R. Huang, and M. Li, "Adapmoe: Adaptive sensitivity-based expert gating and management for efficient moe inference," in *Proceedings of the 43rd IEEE/ACM International Conference on Computer-Aided Design*, 2024, pp. 1–9.
- [8] W. Lin, X. Pan, S. Shi, X. Wang, and X. Chu, "Task scheduling for efficient inference of large language models on single moderate gpu systems," *arXiv preprint arXiv:2411.15715*, 2024.
- [9] Y. Zhang, S. Aggarwal, and T. Mitra, "Daop: Data-aware offloading and predictive pre-calculation for efficient moe inference," *arXiv preprint arXiv:2501.10375*, 2024.
- <span id="page-8-7"></span>[10] P. Tang, J. Liu, X. Hou, Y. Pu, J. Wang, P.-A. Heng, C. Li, and M. Guo, "Hobbit: A mixed precision expert offloading system for fast moe inference," *arXiv preprint arXiv:2411.01433*, 2024.
- <span id="page-8-8"></span>[11] A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei *et al.*, "Qwen2. 5 technical report," *arXiv preprint arXiv:2412.15115*, 2024.
- <span id="page-8-9"></span>[12] A. Liu, B. Feng, B. Wang, B. Wang, B. Liu, C. Zhao, C. Dengr, C. Ruan, D. Dai, D. Guo *et al.*, "Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model," *arXiv preprint arXiv:2405.04434*, 2024.
- <span id="page-8-10"></span>[13] S. Zhong, Y. Sun, L. Liang, R. Wang, R. Huang, and M. Li, "Hybrimoe: Hybrid cpu-gpu scheduling and cache management for efficient moe inference," *arXiv preprint arXiv:2504.05897*, 2025.
- <span id="page-8-11"></span>[14] Y. Qian, F. Li, X. Ji, X. Zhao, J. Tan, K. Zhang, and X. Cai, "Eps-moe: Expert pipeline scheduler for cost-efficient moe inference," 2025. [Online]. Available: <https://arxiv.org/abs/2410.12247>
- [15] S. Go and D. Mahajan, "Moetuner: Optimized mixture of expert serving with balanced expert placement and token routing," *arXiv preprint arXiv:2502.06643*, 2025.
- <span id="page-8-12"></span>[16] W. Cai, J. Jiang, L. Qin, J. Cui, S. Kim, and J. Huang, "Shortcut-connected expert parallelism for accelerating mixtureof-experts," 2024. [Online]. Available: [https://arxiv.org/abs/](https://arxiv.org/abs/2404.05019)

# [2404.05019](https://arxiv.org/abs/2404.05019)

- <span id="page-8-13"></span>[17] W. Fedus, B. Zoph, and N. Shazeer, "Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity," *Journal of Machine Learning Research*, vol. 23, no. 120, pp. 1–39, 2022.
- [18] J. He, J. Zhai, T. Antunes, H. Wang, F. Luo, S. Shi, and Q. Li, "Fastermoe: modeling and optimizing training of large-scale dynamic pre-trained models," in *Proceedings of the 27th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming*, 2022, pp. 120–134.
- <span id="page-8-14"></span>[19] X. Liu, Y. Wang, F. Fu, X. Miao, S. Zhu, X. Nie, and B. Cui, "Netmoe: Accelerating moe training through dynamic sample placement," in *The Thirteenth International Conference on Learning Representations*.
- <span id="page-8-15"></span>[20] J. H. Kim, Y. Ro, J. So, S. Lee, S.-h. Kang, Y. Cho, H. Kim, B. Kim, K. Kim, S. Park *et al.*, "Samsung pim/pnm for transfmer based ai: Energy efficiency on pim/pnm cluster," in *2023 IEEE Hot Chips 35 Symposium (HCS)*. IEEE Computer Society, 2023, pp. 1–31.
- <span id="page-8-16"></span>[21] B. Fujun, J. Xiping, W. Song, Y. Bing, T. Jie, Z. Fengguo, W. Chunjuan, W. Fan, L. Xiaodong, Y. Guoqing *et al.*, "A stacked embedded dram array for lpddr4/4x using hybrid bonding 3d integration with 34gb/s/1gb 0.88 pj/b logic-to-memory interface," in *2020 IEEE International Electron Devices Meeting (IEDM)*. IEEE, 2020, pp. 6–6.
- [22] D. Niu, S. Li, Y. Wang, W. Han, Z. Zhang, Y. Guan, T. Guan, F. Sun, F. Xue, L. Duan *et al.*, "184qps/w 64mb/mm 2 3d logic-to-dram hybrid bonding with process-near-memory engine for recommendation system," in *2022 IEEE International Solid-State Circuits Conference (ISSCC)*, vol. 65. IEEE, 2022, pp. 1–3.
- [23] Z. Yue, H. Wang, J. Fang, J. Deng, G. Lu, F. Tu, R. Guo, Y. Li, Y. Qin, Y. Wang *et al.*, "Exploiting similarity opportunities of emerging vision ai models on hybrid bonding architecture," in *2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA)*. IEEE, 2024, pp. 396–409.
- <span id="page-8-17"></span>[24] C. Li, Y. Yin, X. Wu, J. Zhu, Z. Gao, D. Niu, Q. Wu, X. Si, Y. Xie, C. Zhang *et al.*, "H2-llm: Hardware-dataflow coexploration for heterogeneous hybrid-bonding-based low-batch llm inference," in *Proceedings of the 52nd Annual International Symposium on Computer Architecture*, 2025, pp. 194–210.
- <span id="page-8-18"></span>[25] A. Yang, B. Yang, B. Hui, B. Zheng, B. Yu, C. Zhou, C. Li, C. Li, D. Liu, F. Huang *et al.*, "Qwen2 technical report," *arXiv preprint arXiv:2407.10671*, 2024, version: 2024-07. [Online]. Available:<https://arxiv.org/abs/2407.10671>
- <span id="page-8-19"></span>[26] E. Chan, R. Van De Geijn, W. Gropp, and R. Thakur, "Collective communication on architectures that support simultaneous communication over multiple links," in *Proceedings of the eleventh ACM SIGPLAN symposium on Principles and practice of parallel programming*, 2006, pp. 2–11.
- <span id="page-8-20"></span>[27] S. Rashidi, S. Sridharan, S. Srinivasan, and T. Krishna, "Astrasim: Enabling sw/hw co-design exploration for distributed dl training platforms," in *2020 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)*. IEEE, 2020, pp. 81–92.
- <span id="page-8-21"></span>[28] L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. Xing *et al.*, "Judging llm-asa-judge with mt-bench and chatbot arena," *Advances in Neural Information Processing Systems*, vol. 36, pp. 46 595–46 623, 2023.