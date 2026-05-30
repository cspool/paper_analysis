# III. MOTIVATION

This study analyzed the training process of the MoE in static graph scenarios using MindSpore [\[29\].](#page-10-0) After training the MoE in an actual environment, as shown in Fig. [3,](#page-3-0) two challenges were observed.

*Load imbalance and high communication cost:* Through profiling analysis, types of data were obtained, including the distribution of tokens and the All-to-All proportion during MoE training. Two issues must be addressed: 1) Some hot experts must process a larger number of tokens, which overloads the AI accelerator, and other cold experts process only a small number of tokens, resulting in a low AI accelerator load. The load-imbalance characteristic of the expert is continuous across different layers of the MoE and persists for a long time. 2) In the expert-parallel strategy, experts are assigned to different computing nodes, and data must be exchanged, resulting in

<span id="page-3-0"></span>![](_page_3_Figure_2.jpeg)

Fig. 3. Overview of EfficientMoE.

a significant amount of All-to-All communication. For 32 AI accelerator clusters, pure communication time accounts for 75% of the total MoE training time, which significantly wastes the computing power resources of the AI accelerator and increases the burden of model training. The achievement of load balance and the reduction of communication time during training in the MoE model are challenging problems.

*Abandonment or padding of tokens:* All experts retain the same expert capacity because static graphs require input shapes to be determined before compilation, and the capacity of experts cannot be changed while training the MoE model. In contrast, the capacity requirements of the hot and cold experts differ. This limitation leads to two difficulties in the MoE model training process: 1) The hot expert is limited by expert capacity, and tokens exceeding the capacity are discarded, which affects the accuracy of model training. 2) The cold expert will maintain the expert capacity consistent with zero-padding, which burdens the memory and communication resources. Setting a suitable expert capacity for MoE models in static graph mode is also challenging.

## IV. METHOD

#### *A. Overview*

This study proposes an optimization method called Efficient-MoE based on static computational graphs for MoE model training on MindSpore to address the above challenges, as shown in Fig. 3. First, EfficientMoE designs a sampler to collect information and proposes a load prediction model based on this sampler to evaluate the load balance for experts and divide the experts into hot and cold experts. Subsequently, EfficientMoE proposes a dynamic scheduling method for experts to balance the load between AI accelerators. EfficientMoE generates replicas for hot experts and dynamically schedules these replicas to other AI accelerators to share resources with cold experts according to the characteristics of expert load sustainability. Simultaneously, replica experts process the local tokens that need to be sent to hot experts, and the remaining tokens sent to cold experts are transmitted by All-to-All communication, which reduces the frequency of All-to-All communication transmission and improves the efficiency of model training. Finally, EfficientMoE designs an expert capacity model to set the appropriate capacity values for both hot and cold experts. EfficientMoE sets a larger capacity for replicas of hot experts to mitigate token discarding and a smaller capacity for cold experts to reduce resource waste.

#### *B. Sampling and Constructing Load Prediction Model*

*1) Sampler:* To estimate and optimize the training performance of MoE models, EfficientMoE analyzes the structure of MoE models and key factors affecting their training performance, and then designs a sampler to capture important information, including token characteristics, MoE structure characteristics, and computing resource characteristics, as listed in Table [II.](#page-4-0)

*Token Features* refer to the characteristics or attributes of individual tokens (words or subwords) in a dataset. In the training of MoE models, token features play a crucial role in determining which experts are selected for processing. The distribution

<span id="page-4-0"></span>

| Dimension        | Details                              | Explanation                                                  |  |
|------------------|--------------------------------------|--------------------------------------------------------------|--|
|                  | Token Distribution per Expert        | This indicates how tokens are allocated among the experts.   |  |
| Token Feature    | Token Distribution                   | Each AI-accelerator handles the experts and their tokens.    |  |
|                  | Layer-wise Token Redistribution      | Tokens may be redistributed to different experts.            |  |
| Expert Parameter | Model Parameters(MB/expert)          | Per expert has its own set of parameters.                    |  |
|                  | Intermediate Data (MB/expert)        | Intermediate data, such as activations and gradients.        |  |
|                  | Parameter Synchronization(MB/expert) | Parameters synchronization across different AI accelerators. |  |
| Memory           | Memory per GPU(GB)                   | The memory to store the parameters and data.                 |  |
|                  | Expert Memory Requirements(GB)       | Different experts may require varying amounts of memory.     |  |
|                  | Free Memory(GB)                      | Memory left for each AI-accelerator.                         |  |
| Communication    | Intra-Node Communication(GB/s)       | Bandwidth within a single node between AI accelerators.      |  |
|                  | Inter-Node Communication(GB/s)       | Bandwidth between nodes across different machines.           |  |
|                  | Actual communication(GB/s)           | The actual communication of the cluster AI accelerators      |  |

TABLE II INFORMATION FOR SAMPLING

between the tokens and experts sampled indicates how these token features influence the gating mechanism within the MoE, which ensures that the most relevant experts are selected.

Expert Parameters include the weights and biases associated with each expert. In addition to the model parameters, intermediate data, such as activations and gradients must also be stored during forward and backward passes, which increases memory requirements. The efficient synchronization of these parameters across AI accelerators is necessary to maintain consistency and optimal model performance.

Memory must be efficiently managed to store multiple expert parameters and intermediate data. Experts may have varying memory requirements based on the size and complexity, encompassing model parameters and activations. Efficient memory management, involving both of the nodes in the cluster and the GPUs for different nodes, is essential to maximize utilization and prevent bottlenecks.

Communication refers to the rate at which data can be transmitted between nodes (inter-node communication) or within nodes (intra-node communication) in a distributed computing environment. In MoE models, efficient communication is critical for synchronizing the parameters and sharing data between AI accelerators. Inter-nodes with lower communication bandwidths affect the ability to scale across multiple machines, whereas intra-nodes with higher communication bandwidths impact the efficiency of operations across multiple AI accelerators. Optimizing the communication bandwidth is vital for reducing the latency and improving the overall training and inference speed in MoE models, including managing the actual communication volume, such as token distribution communication in different iterations and MoE layers.

2) Load Prediction Model: To achieve load balancing and communication optimization effectively, EfficientMoE proposes a load prediction model for MoE training based on the information collected by the sampler. Specifically, the Efficient-MoE defines  $T_{i,j}^k$  to represent the number of tokens distributed to the j-th expert in the k-th layer during the i-th iteration.

EfficientMoE employs a statistical approach to analyze the token processing workload of each expert across different layers and multiple iterations. In particular, EfficientMoE collects the token counts processed by expert i at  $layer_k$  over m iterations,

denoted as  $(T^k_{i,1}, T^k_{i,2}, \dots, T^k_{i,m})$ . Subsequently, EfficientMoE sorts the data in descending order of token counts and selects the top p% of the data for averaging; p will be decided by token distribution in a load prediction cycle. This step aimed to exclude extreme outliers and obtain a more robust estimation of expert workload. The resulting average value was utilized to represent the load  $L^k_i$  of each expert i at layer k. A concept of a load prediction cycle was introduced, treating m iterations as one cycle, and utilizing the average load over this cycle to predict the expert's load for future cycles, as follows:

$$L_i^k = \frac{1}{m} \sum_{i=1}^m Top\left(T_{i,j}^{k,sorted}, p\right) \tag{1}$$

where  $T_{i,j}^{k,sorted}$  denotes the j-th largest token count after sorting for expert i at layer k. The value of m is determined based on the difference in token counts across iterations, and the expert's load will be recomputed when the token count gap exceeds  $\omega$ , which is determined according to the influence of the expert's load caused by historical data in different load prediction cycles.

In addition, apart from the individual expert loads, the load of each AI accelerator must be constructed, as shown in (2):

$$D_{j} = \sum_{i \in \mathcal{E}_{j}} \left( Compute\left(T_{i}\right) + Memory\left(T_{i}\right) \right) \tag{2}$$

where  $T_i$  represents the token count processed by expert i,  $Compute(T_i)$  and  $Memory(T_i)$  are functions that calculate the required computational and storage resources respectively, and  $D_j$  denotes the load of AI-accelerator j. This ensures that the computational load balance across the AI accelerators is considered along with the expert loads when performing dynamic scheduling and data distribution in the dynamic dispatching method.

#### C. Dynamic Schedule Strategy

Based on the above work, this study has designed an expert scheduling method to address the issues of unbalanced load and All-to-All communication delays. It aims to dynamically adjust expert capacities and efficiently manage load distribution

**Algorithm 1:** Dynamic Expert Schedule Method.

in MoE models. The key idea was to assess the token loads of different experts within a load prediction cycle and categorize them based on their load factors. Experts with a load factor greater than a threshold of q were classified as hot experts, whereas others were considered cold experts, where EfficientMoE set the value of q to 60%.

Within a load-prediction cycle, EfficientMoE uses a load evaluation model to estimate the token load of each expert. The load factor was calculated, and the experts were categorized as hot or cold based on this factor. The EfficientMoE evaluates the current load conditions, including the computational and storage resources of different AI accelerators. If the current load condition of an AI accelerator meets the requirements of a hot expert, the replica expert is scheduled to that AI accelerator, sharing resources with the cold experts. This aims to localize token processing, whereby tokens originally intended for hot experts are distributed among local replica experts. Tokens were also allocated to cold experts as needed to minimize All-to-All communication. During each prediction cycle, the load of experts was dynamically assessed, and the scheduling of hot experts was adjusted accordingly. The details of this schedule strategy are described in Algorithm 1:

