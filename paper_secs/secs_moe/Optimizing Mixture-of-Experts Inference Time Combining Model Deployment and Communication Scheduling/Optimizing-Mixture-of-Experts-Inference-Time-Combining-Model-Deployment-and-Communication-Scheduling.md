# Optimizing Mixture-of-Experts Inference Time Combining Model Deployment and Communication Scheduling

JIALONG LI, SHREYANSH TRIPATHI, LAKSHAY RASTOGI, YIMING LEI, RUI PAN, YITING XIA

As machine learning models scale in size and complexity, their computational requirements become a significant barrier. Mixture-of-Experts (MoE) models alleviate this issue by selectively activating relevant experts. Despite this, MoE models are hindered by high communication overhead from all-to-all operations, low GPU utilization due to the synchronous communication constraint, and complications from heterogeneous GPU environments.

This paper presents Aurora, which optimizes both model deployment and all-to-all communication scheduling to address these challenges in MoE inference. Aurora achieves minimal communication times by strategically ordering token transmissions in all-to-all communications. It improves GPU utilization by colocating experts from different models on the same device, avoiding the limitations of synchronous all-to-all communication. We analyze Aurora's optimization strategies theoretically across four common GPU cluster settings: exclusive vs. colocated models on GPUs, and homogeneous vs. heterogeneous GPUs. Aurora provides optimal solutions for three cases, and for the remaining NP-hard scenario, it offers a polynomial-time sub-optimal solution with only a 1.07× degradation from the optimal.

Aurora is the first approach to minimize MoE inference time via optimal model deployment and communication scheduling across various scenarios. Evaluations demonstrate that Aurora significantly accelerates inference, achieving speedups of up to 2.38× in homogeneous clusters and 3.54× in heterogeneous environments. Moreover, Aurora enhances GPU utilization by up to 1.5× compared to existing methods.

## 1 INTRODUCTION

Serving deep learning and large language models has become increasingly critical as they are integrated into a wide range of online applications, such as programming assistance, search engines, and conversational bots. However, as the size and complexity of these models continue to grow, it is challenging to meet the high computational demands and stringent latency requirement.

Mixture-of-Experts (MoE) models offer an effective solution to reduce computational demands while preserving performance. They achieve this by dynamically activating only a subset of specialized components, known as experts, for input tokens. This selective activation reduces the overall computational load without sacrificing efficiency and accuracy. By engaging only the most relevant experts for specific tasks, MoE models optimize resource utilization and processing speed.

Despite the considerable benefits, inference of MoE models still faces significant challenges. The most prominent issue is high communication overhead. The all-to-all communication pattern in MoE models, identified as a major bottleneck [\[9,](#page-20-0) [11,](#page-20-1) [27\]](#page-21-0), is largely due to the dynamic selection of experts. This results in uneven data exchange among GPUs, leading to network bandwidth contention and prolonged communication times.

Moreover, MoE models suffer from low GPU utilization. This problem arises because all-to-all communication is typically implemented using synchronous operations [\[18,](#page-20-2) [19,](#page-20-3) [23,](#page-20-4) [31,](#page-21-1) [32,](#page-21-2) [38,](#page-21-3) [42\]](#page-21-4). As a result, GPUs hosting unpopular experts remain idle while waiting for communication to complete on GPUs handling popular experts.

Lastly, GPU heterogeneity, which is common due to incremental deployments, adds further complexity to MoE model deployment [\[3,](#page-20-5) [22,](#page-20-6) [37,](#page-21-5) [41\]](#page-21-6). The varied hardware configurations complicate the efficient allocation and utilization of resources across the model. To fully harness the potential of MoE models, these challenges need to be effectively addressed.

Existing solutions fail to solve the problem from all fronts. Most approaches either reduce communication overhead by balancing token loads [\[3,](#page-20-5) [4,](#page-20-7) [7,](#page-20-8) [9,](#page-20-0) [11](#page-20-1)[–13,](#page-20-9) [17,](#page-20-10) [24,](#page-21-7) [29,](#page-21-8) [30\]](#page-21-9) or by accelerating the all-to-all operation [\[8,](#page-20-11) [9,](#page-20-0) [12,](#page-20-12) [18,](#page-20-2) [27,](#page-21-0) [28,](#page-21-10) [32,](#page-21-2) [33,](#page-21-11) [42\]](#page-21-4), but still struggle with low GPU utilization.

<span id="page-1-0"></span>![](_page_1_Figure_1.jpeg)

Fig. 1. MoE model structure.

Other approaches pack multiple experts from the same model on a single GPU to reduce idle time [11, 23, 36, 38], but these experts remain blocked by synchronous all-to-all communication, preventing full interleaving of computation and communication. Besides, these methods rely on empirical approaches, lacking theoretical backing, and are designed for specific settings, failing to account for the diverse configurations of production GPU clusters, such as heterogeneous hardware.

In this paper, we propose *Aurora*, a comprehensive solution for minimizing the inference time of MoE models. Our design combines expert colocation, GPU assignment, and communication scheduling, supported by theoretical analysis across four distinct GPU cluster settings based on two key dimensions: exclusive vs. colocated experts on GPUs, and homogeneous vs. heterogeneous GPUs. Aurora achieves *optimal* inference time in most cases, except for the NP-hard scenario of colocating experts on heterogeneous GPUs, where we provide a *sub-optimal* polynomial-time solution with inference time only 1.07× the optimum, as shown in our simulations.

To the best of our knowledge, Aurora offers the first theoretical derivation of minimal MoE inference time. Our key insights can guide the development of future MoE inference systems: minimal all-to-all communication time is achieved by ordering token transmission to avoid bandwidth contention; in homogeneous clusters, minimizing inference time is equivalent to minimizing communication time; for exclusive experts on heterogeneous GPUs, assigning experts by token load to GPUs in descending capacity minimizes inference time; and the NP-hard case of colocating experts on heterogeneous GPUs is a 3-dimensional matching problem, which can be approximated by decoupling it into two dependent bipartite graphs.

Extensive simulations demonstrate the effectiveness of Aurora. Using production MoE inference traces from Google, Aurora reduces inference time by up to 2.38× in homogeneous GPU clusters and up to 3.54× in heterogeneous clusters. By colocating experts from different models, Aurora also improves GPU utilization by up to 1.5× compared to state-of-the-art solutions that colocate experts from the same model. Even with inaccurate inputs for Aurora's optimization, with up to 75% noise in model statistics, inference time is extended by only 15.8%.

#### 2 PRELIMINARIES

In this section, we first explore the structure of MoE inference to understand how the different components work together within the model (§2.1). Next, we discuss the distinctive features of MoE inference that set it apart from other architectures (§2.2). We then identify the key bottlenecks that affect MoE inference performance (§2.3). Finally, we outline the essential prerequisites required for Aurora (§2.4).

## <span id="page-2-0"></span>2.1 MoE Inference

An MoE model comprises multiple MoE layers. For MoE training, each layer involves both a forward and a backward pass, while inference requires only the forward pass. Fig. [1](#page-1-0) illustrates the process of an MoE layer, highlighting the separation of computation and communication phases. The computation phase consists of three components: the gate function, the feed-forward network (FFN), and aggregation. Two all-to-all communications occur during the communication phase. These two all-to-all communications are opposite in terms of data flows.

Gate. The gate network determines which experts should be activated for the input tokens. In general, each token will be sent to one or two experts.

FFN. An FFN is typically an expert. Each expert is responsible to process the tokens assigned by the gate network.

Aggregation. This operation reshapes the tensors and computes the weighted output. After aggregation, the process proceeds to the next MoE layer.

First all-to-all communication. The first all-to-all communication occurs after the gate network. During this process, each token is dispatched to the assigned experts.

Second all-to-all communication. The second all-to-all communication is for exchanging outputs of experts, ensuring the original sequences are organized before the start of next layer.

