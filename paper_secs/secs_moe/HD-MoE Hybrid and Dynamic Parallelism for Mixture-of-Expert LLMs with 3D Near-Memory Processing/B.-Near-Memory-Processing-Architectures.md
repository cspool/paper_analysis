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

