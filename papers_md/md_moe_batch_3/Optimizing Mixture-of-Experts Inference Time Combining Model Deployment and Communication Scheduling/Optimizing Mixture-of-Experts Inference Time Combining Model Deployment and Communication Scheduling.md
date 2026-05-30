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

# <span id="page-2-1"></span>2.2 Characteristics of MoE Inference

Here, we outline three key characteristics of MoE inference, which shed light on the inference bottlenecks discussed in §[2.3.](#page-2-2)

Synchronous all-to-all communications. In this process, all-to-all communication is synchronous, meaning that computation (including FFN and aggregation) can only begin once every GPU has completed data transmission. This leads to the GPU computation resource idleness.

Reversed all-to-all communications. Within the same forward pass, the two all-to-all communications are reversed. For each data transfer from GPU to in the first communication, there is a corresponding data transfer from GPU to in the second. The data sizes in these transfers are identical, as the FFN architecture ensures that the input and output data sizes are the same.

Non-overlapping communication and computation. Communication and computation processes do not overlap; each step can only commence after the previous one is completed.

# <span id="page-2-2"></span>2.3 MoE Inference Bottlenecks

High communication overhead. Existing research has identified all-to-all communication as a significant bottleneck in MoE inference [\[9,](#page-20-0) [27\]](#page-21-0). A recent study [\[11\]](#page-20-1) highlights that the all-to-all communication can constitute over 60% of inference time when using four GPUs, and the overhead increases substantially with additional GPUs .

The high communication overhead arises from two main factors. First, the dynamic selection of experts results in an uneven distribution of tokens among GPUs [\[4,](#page-20-7) [9,](#page-20-0) [26\]](#page-21-13), leading to some GPUs being heavily loaded while others remain idle. Second, the all-to-all communication is typically implemented using synchronous operations [\[18\]](#page-20-2). These operations are inefficient as they cause resource wastage when either communication or computation is not fully utilized. This inefficiency is further exacerbated by the dynamic nature of expert selection.

Low GPU utilization. Uneven load distribution and synchronous all-to-all communication also contribute to low GPU utilization. GPUs supporting unpopular experts remain idle most of the time [\[13\]](#page-20-9). A study of GPU cluster data from Alibaba reveals that less than 10% of GPUs reach 80% utilization [\[40\]](#page-21-14).

<span id="page-3-1"></span>![](_page_3_Figure_1.jpeg)

Fig. 2. Aurora aims to minimize inference time across four different scenarios. It optimizes expert colocation, GPU assignment, and communication scheduling for each case. Aurora achieves optimal results in the first three scenarios and delivers suboptimal performance in the final one due to its NP-hardness.

GPU cluster heterogeneity. In production clusters, GPU heterogeneity is common due to incremental deployments and rapid advancements in GPU design [\[14,](#page-20-13) [22,](#page-20-6) [37,](#page-21-5) [41,](#page-21-6) [44\]](#page-21-15). These clusters feature varied hardware configurations, including different types of GPUs and diverse resource setups. This heterogeneity complicates the deployment of MoE models and must be considered to optimize performance.

## <span id="page-3-0"></span>2.4 Prerequisites in Aurora

Before we delve into the details of each scenario, let's outline the key prerequisites for this work. Each GPU hosts at most two models. As shown in Fig. [1,](#page-1-0) MoE inference involves alternating computation and communication phases separated by clear barriers. Colocating two models on a GPU allows them to efficiently interleave resource usage—one model performs computation while the other uses the network. Adding a third model, however, forces one to wait for resource access, leading to increased inference time.

The network is represented by a big switch model. MoE inference typically requires several to dozens of GPUs, often housed within a single rack and connected by a high-performance network. As illustrated in Fig. [4\(](#page-4-0)a), this non-blocking network fabric can be modeled as a big switch, which interconnects GPUs enabling low-latency and high-throughput communication between them.

The optimization is based on historical statistics of MoE models. Inference service providers usually collect statistics of MoE models for performance monitoring and troubleshooting, such as the token distribution across GPUs and the average computation times for the Gate network, FFN, and Aggregation operations. Aurora uses such historical data to guide optimization, and this work focuses on theoretical analysis of our optimization mechanisms based on these precise inputs. As shown in our simulations (§[8\)](#page-16-0), even with up to 75% unpredictable inference requests after the optimization plan is deployed, the inference time of Aurora is only degraded by 15.8%.

## 3 OVERVIEW

In this section, we begin by outlining Aurora's inputs and optimization goal across four scenarios of GPU cluster settings. Next, we explain how expert colocation, GPU assignment, and communication scheduling impact inference time. Finally, we provide a summary of each scenario.

Inputs. As discussed in §[2.4,](#page-3-0) we use historical model statistics to guide decisions on expert colocation, GPU assignment, and communication scheduling. At a high level, the inputs include

![](_page_4_Figure_1.jpeg)

Fig. 3. (a) Colocating experts from the same model results in wasted GPU resources and increased inference time, as follow-up computations are delayed by synchronous all-to-all communications. (b) Colocating experts from different models enables full interleaving of computation and communication, resolving this issue.

<span id="page-4-0"></span>![](_page_4_Figure_3.jpeg)

Fig. 4. (a) A big switch model representing the non-blocking inter-GPU network fabric. (b) Originally, the all-to-all communication of tokens from GPU 1 (red) and GPU 2 (yellow) to all other GPUs takes 3 units of time overall. (c) Optimizing the token order reduces transmission time to 2 units.

traffic matrices of token distribution across GPUs during each all-to-all communication, as well as computation times for the Gate network, FFN, and Aggregation operations. The detailed input parameters are listed in Table 1 and explained in §4.

**Optimization goal.** Aurora is designed to minimize the inference time of MoE models. Given the diverse settings of modern GPU clusters, we analyze two key dimensions: exclusive GPU usage per model vs. colocating models on the same GPUs, and homogeneous vs. heterogeneous GPU types. Across the four combinations of these dimensions, as shown in Fig. 2, Aurora achieves optimal performance in the first three scenarios. We prove the last scenario to be NP-hard and propose a sub-optimal solution with inference time only 1.07× the optimum, based on our evaluation in §8. When possible, we colocate MoE models on GPUs to maximize GPU utilization at best effort.

**Expert colocation.** Aurora colocates experts on the same GPU to maximize utilization, where applicable, as shown in the colocating scenarios in Fig. 2. As motivated in §2.4, Aurora colocates up to two experts per GPU, interleaving their computation and communication. Previous studies colocated experts from the same model [18], which wastes GPU resources and extends inference time. As shown in Fig. 4(a), this is because experts from the same model are bound by the synchronous all-to-all communication, delaying subsequent computation phases like FFN and Aggregation.

In contrast, Aurora colocates experts from different models. To maximize GPU utilization and minimize inference time, it identifies the optimal combination of experts that complement each other in terms of computation and communication needs. As illustrated in Fig. 4(b), the two experts take turns to use available computation and communication resources. Aurora pairs a computation-intensive expert with a communication-intensive one to efficient use of GPUs.

**GPU** assignment. Heterogeneous clusters, as shown in Fig. 2, require selecting the appropriate GPU types for experts. For instance, deploying a popular expert on a high-performance GPU with high FLOPS, memory capacity, and network bandwidth helps minimize computation and communication times. Aurora assigns experts to suitable GPU types without worrying about the

![](_page_5_Figure_1.jpeg)

 $|A_i^a|$ 

Computation time of Model a's Aggregation on GPU i

<span id="page-5-0"></span>![](_page_5_Figure_2.jpeg)

mogeneous clusters.

deployment to specific GPU IDs, as GPUs of the same type are interchangeable when connected through the "big switch" model of a non-blocking network, as discussed in §2.4.

**Communication scheduling.** All four scenarios in Fig. 2 require communication scheduling to reduce the communication time, which involves determining the order of token transmission in the all-to-all communications. Different transmission orders can lead to varying communication times. For instance, in Fig. 4(a), GPU 1 sends tokens to GPUs 2 and 3, while GPU 2 sends to GPUs 1 and 3. In Fig. 4(b), communication takes 3 units of time when GPU 1 first sends to GPU 2 and then to GPU 3, and GPU 2 sends to GPU 1 and then to GPU 3. However, as shown in Fig. 4(c), changing GPU 2's transmission order to send to GPU 3 first, then to GPU 1, reduces communication time to 2 units. In practice, reordering token transmission can be achieved with a buffer layer at the computation operations, which calls communication collective libraries, such as NCCL, in the desired order.

We summarize the main results of our theoretical analysis of each scenario as follows.

Exclusive + Homogeneous (§4). This scenario considers running models exclusively on clusters where all GPUs have identical computing power and network bandwidth. Theorem 4.1 proves that minimizing inference time is equivalent to minimizing communication time. Theorem 4.2 further shows that communication time is minimized by ordering token transmission to avoid bandwidth contention at the receiving GPUs. The minimum communication time is determined by the GPU handling the largest traffic volume, whether sending or receiving. Alg. 1 (§ 4.2) provides the algorithm for finding the optimal order that minimizes inference time.

Exclusive + Heterogeneous (§5). This scenario tackles the challenges of running models exclusively on GPUs with different computing power and network bandwidth. Theorem 5.1 demonstrates that sorting experts by token load and assigning them to GPUs in descending order of performance minimizes inference time. Theorem 5.2 proves that the transmission order for homogeneous clusters (Theorem 4.2) also minimizes communication time in a heterogeneous setting.

Colocating + Homogeneous (§6). This scenario examines improving GPU utilization by colocating two MoE models. The colocation strategy affects the aggregated communication time of the two models, and thus, their overall inference time. Theorem 6.1 shows that minimizing the aggregated communication time leads to optimal overall inference time. To this end, we solve the bottleneck matching problem to find the optimal expert colocation, thereby minimizing communication time and achieving optimal inference time.

Colocating + Heterogeneous (§7). Extending the colocation strategy to heterogeneous clusters involves communication scheduling, GPU assignment, and expert colocation, making it the most complex scenario. We model it as a 3-dimensional matching problem, which is NP-hard. By decoupling the matching into two dependent bipartite graphs, we propose a sub-optimal but effective solution, which prolongs the inference time by only 1.07× compared to the optimum.

#### <span id="page-6-0"></span>4 EXCLUSIVE MODELS ON HOMOGENEOUS CLUSTERS

In this section, we derive the minimum inference time for running models exclusively on homogeneous clusters, referred to as the Exclusive + Homogeneous scenario.

**Input parameters.** Table 1 lists the input parameters used by all four scenarios. For a specified MoE model consisting of n experts, each expert is placed on one GPU, requiring a total of n GPUs. The token distribution in each layer is known in advance is represented by a traffic matrix  $\mathbb{D}_N$  for the first all-to-all communication, and  $\mathbb{D}_C$  for the second all-to-all communication. Note that  $\mathbb{D}_N$  and  $\mathbb{D}_C$  are reversed as we state in §2.2. The matrix is an  $n \times n$  matrix with elements  $d_{ij}$ , indicating the traffic sent from GPU i to j. The symbol  $|G_i^a|$ ,  $|F_i^a|$ , and  $|A_i^a|$  represent the computation times of Model a's Gate, FFN, and Aggregation components, respectively, on GPU i.

**Solution overview.** We first prove that in the Exclusive + Homogeneous scenario, minimizing inference time is equivalent to minimizing communication time (§4.1). Next we show how to determine the transmission order to achieve minimal communication time (§4.2).

## <span id="page-6-2"></span>4.1 Minimizing inference time equals minimizing communication time

<span id="page-6-1"></span>Theorem 4.1. In the Exclusive + Homogeneous scenario, minimizing inference time is equivalent to minimizing communication time.

PROOF. This proof is straightforward because, in the Exclusive + Homogeneous scenario, the only factor influencing inference time is the communication scheduling.

We first derive the inference time expression. As shown in Fig. 5, the two all-to-all communications are synchronous across GPUs. This synchronization means that the FFN and Aggregation processes can only start after the last data flow is complete. The inference time is therefore divided into three parts: the Gate and the first all-to-all communication, the FFN and the second all-to-all communication, and the Aggregation. Due to the strict barrier between each layer, the inference time for a layer is determined by the slowest GPU. So the inference time is determined by summing the maximum values of each part, as represented by the following equation.

<span id="page-6-3"></span>Inference time 
$$t = \max(|G_i| + |N_i|) + \max(|F_j| + |C_j|) + \max(|A_k|), i, j, k \in [1, n]$$
 (1)

In Eqn. 1, the symbols  $|G_i|$ ,  $|F_i|$ , and  $|A_i|$  indicate the duration of the Gate, FFN, and Aggregation processes on GPU i. The symbols i, j, k each represent different GPUs, as the maximum processing time for each part can occur on different GPUs. For this scenario, we can make the following three observations.

- (1) The assignment of GPUs to experts in homogeneous clusters requires no special decisions, as all GPUs possess identical computational power and network bandwidth.
- (2) The computation times for Gate processes are equal across all GPUs, and the same applies to Aggregation.
- (3) The computation time for the FFN is determined by the number of tokens it processes, with more tokens resulting in longer computation times.

With observations (1) and (2), we have  $|G_i| = |G|$ ,  $|A_k| = |A|$ . Based on observation (3), we can state that  $\max(|F_j| + |C_j|) = \max(|F_j|) + \max(|C_j|)$ , since  $|F_j|$  and  $|C_j|$  increase simultaneously. Thus, Eqn. 1 can be expressed as follows.

<span id="page-6-4"></span>
$$t = |G| + \max(|N_i|) + \max(|F_i|) + \max(|C_i|) + |A|, i, j \in [1, n]$$
 (2)

As discussed in §2.2, the two all-to-all communications are reversed. The GPU receiving the highest volume of data at  $\mathbb{D}_N$  is also the one transmitting the largest amount at  $\mathbb{D}_C$ . Consequently,  $\max(|N_i|)$  and  $\max(|C_i|)$  occur on the same GPU. Therefore, Eqn. 2 can be further expressed as:

<span id="page-6-5"></span>
$$t = |G| + \max(|N_i|) + \max(|F_i|) + \max(|C_i|) + |A|, \ i \in [1, n]$$
(3)

In Eqn. 3,  $\max(|F_i|)$  represents the computation time of the FFN processing the highest number of tokens, which is constant regardless of its deployment. Therefore, to achieve optimal inference time, the remaining task is to minimize  $\max(|N_i|)$ , the first all-to-all communication time.

#### <span id="page-7-1"></span>4.2 Scheduling transmission order to minimize communication time

Fig. 4(a)-(c) show that the communication time depends on the order in which tokens are transmitted. Intuitively, the time for an all-to-all communication cannot be less than the time it takes for the GPU with the heaviest traffic to send or receive its tokens. For example, suppose GPU i receives the largest amount of traffic, denoted by d, and the bandwidth is B. This means GPU i will need at least d/B time to receive all tokens. The question is, can we design a transmission order that completes the all-to-all communication in exactly d/B time? The following theorem provides an affirmative answer.

<span id="page-7-0"></span>Theorem 4.2. The communication time is minimized by transmitting tokens in an order that avoids bandwidth contention at the receiving sides. The minimum communication time is  $b_{max} = max(\sum_{j=1}^{n} d_{ij}, \sum_{i=1}^{n} d_{ij})/B$ .

Theorem 4.2 establishes that GPU should avoid sending tokens to the same destination simultaneously. Fig. 4(c) presents an optimal order that avoids bandwidth contention at the receiving sides. This order guarantees that at any time, each GPU only receives tokens from one GPU. Theorem 4.2 also shows that the minimum communication time is determined by the maximum column or row sum in the traffic matrix  $\mathbb{D}^1$ . In other words, if the largest amount of data being sent or received on a single GPU is d, then the entire all-to-all communication can be completed in d/B time. A sketch of the proof is provided below, with the detailed proof available in Appx. A.

Sketched Proof. In homogeneous clusters, we set B to 1 for simplification. The proof involves transforming the traffic matrix  $\mathbb D$  into  $\mathbb D'$  by adding artificial traffic matrix  $\mathbb X$  with non-negative values. With the updated matrix  $\mathbb D'$ , it ensures that the sum of each column or row equals  $b_{max}$ . We then demonstrate the all traffic in  $\mathbb D'$  can be transmitted within the time  $b_{max}$ , by constructing a transmission order where GPUs do not send tokens to the same destination simultaneously. Since  $\mathbb D'$  is constructed by augmenting the original traffic matrix  $\mathbb D$  with the non-negative traffic matrix  $\mathbb X$ , the time required for transmitting traffic in  $\mathbb D$  cannot exceed  $b_{max}$ . The optimal transmission order can also be obtained by removing artificial traffic from  $\mathbb D'$ .

Moving forward, our approach unfolds in three key steps. Initially, we illustrate the conversion of the traffic matrix  $\mathbb{D}$  into  $\mathbb{D}'$  by incorporating matrix  $\mathbb{X}$ . Subsequently, we prove that the minimum communication time for  $\mathbb{D}'$  is  $b_{max}$ . Finally, we prove the existence of a non-negative  $\mathbb{X}$ .

## 1. Convert $\mathbb{D}$ to $\mathbb{D}'$ by adding non-negative $\mathbb{X}$

- Construct  $\mathbb{D}'$  by adding non-negative artificial traffic matrix  $\mathbb{X}$  to  $\mathbb{D}$ :  $\mathbb{D} + \mathbb{X} = \mathbb{D}'$ .
- Ensure for each row  $\sum_{i=1}^{n} d'_{ij} = b_{max}$ , and for each column  $\sum_{i=1}^{n} d'_{ij} = b_{max}$ ,  $d'_{ij} \in \mathbb{D}'$ .

# 2. Prove the minimum communication time for $\mathbb{D}'$ is $b_{max}$

- In each time slot, each GPU sends and receives exactly one token.
- Demonstrate that each GPU can send and receive data without interruption. As a result, all
  GPUs complete their communication within b<sub>max</sub>, making the all-to-all communication time\nequal to b<sub>max</sub>.

#### 3. Prove the existence of non-negative X

- Transform the  $n \times n$  matrix  $\mathbb{X}$  to an  $n^2 \times 1$  vector  $\mathbf{x}$ .
- Formulate the problem using the system of equations:  $Ax = \Delta b$ .

<span id="page-7-2"></span> $<sup>^1\!</sup> W\! e$  will remove the elements in the diagonal of matrix  $\mathbb D$  as the source and destination are the same.

#### **Algorithm 1:** Determine token transmission order

```
Input: All-to-all traffic matrix \mathbb{D}
Output: Token transmission order O

1 Set O \leftarrow \emptyset;

2 Find the bottleneck GPU (with the most traffic)

3 Choose a random order for tokens at the bottleneck, add to O

4 Remove the bottleneck traffic from \mathbb{D}

5 while \mathbb{D} is not empty do

6 Sort GPUs by traffic amount in descending order

7 for Each GPU i in sorted list do

8 Arrange tokens to avoid conflicts with existing order in O

Add the new order for GPU i to O

10 Remove traffic handled by GPU i from \mathbb{D}
```

• Use Farkas' Lemma [1] to show that a non-negative solution  ${\bf x}$  exists, which implies the existence of  ${\mathbb X}$ .

**Determining the token transmission order.** We show how to establish the token transmission order for each GPU, with the input of traffic matrix  $\mathbb{D}$ . According to Theorem 4.2, data transmission at the bottleneck GPU should be continuous. Therefore, we first determine the order at the bottleneck.

As shown in Alg. 1, we begin by identifying the bottleneck GPU, the one handling the most traffic. The transmission order at the bottleneck can be chosen randomly. After establishing this, we remove the traffic from  $\mathbb D$  (*Lines 1–4*). For the remaining GPUs, we sort them based on their traffic load in descending order (*Line 6*) and arrange the token transmission to avoid conflicts with the existing order (*Line 8*). The order follows the pattern illustrated in Fig. 15(b), Appx. A. We continue to remove traffic and update  $\mathbb D$  accordingly (*Line 10*). This process repeats until  $\mathbb D$  is empty, resulting in a token transmission order that meets the requirements of Theorem 4.2.

In summary, with Theorem 4.2 we can derive the optimal communication time, and further obtain the the minimum inference time with Eqn. 3.

#### Takeaway 1

- In the Exclusive + Homogeneous scenario, minimizing inference time is equivalent to minimizing communication time.
- Aurora determines the optimal token transmission order, ensuring that each GPU can receive data without bandwidth contention, thereby achieving minimum all-to-all communication time.

#### <span id="page-8-1"></span>5 EXCLUSIVE MODELS ON HETEROGENEOUS CLUSTERS

In this section, we derive the minimum inference time for running models exclusively on heterogeneous GPU clusters, referred to as the Exclusive + Heterogeneous scenario.

**Solution overview.** We first show how to assign GPUs optimally (§5.1). Next, we demonstrate that the transmission order obtained in homogeneous clusters remains optimal in a heterogeneous environment (§5.2).

#### <span id="page-9-2"></span>5.1 Finding optimal GPU assignment

Fig. 6 presents the exclusive + Heterogeneous scenario. Different from the homogeneous case, the three observations for Exclusive + Homogeneous in §4 do not hold. Most importantly, experts should be placed on the heterogeneous clusters carefully to reduce the inference time.

<span id="page-9-0"></span>THEOREM 5.1. In a heterogeneous cluster, the optimal GPU assignment is to sort the experts by the number of tokens they process in descending order and then assign them to GPUs from the highest to the lowest performance.

<span id="page-9-4"></span>![](_page_9_Figure_4.jpeg)

Fig. 6. Running MoE models exclusively on heterogeneous clusters.

PROOF. Following Theorem 5.1, we assign high-end GPUs to the most popular experts in descending order. Let's assume GPUs m and n are assigned to experts p and q, respectively. GPU m has higher performance than GPU n, and expert p is more popular than expert q. The inference times on GPUs m and n are  $t_m$  and  $t_n$ , respectively.

Now, suppose we reverse the assignment, mapping GPU m to expert q and GPU n to expert p. The new inference times on GPUs m and n are  $t'_m$  and  $t'_n$ , respectively. Because the more popular expert p is now assigned to the lower-end GPU n, we have  $t'_n > t_m^2$ . Previously, GPU n was handling the less popular expert q with inference time  $t_n$ , so  $t'_n > t_n$ . Therefore, we can conclude that  $t'_n > t_m$  and  $t'_n > t_n$ , indicating that we cannot achieve  $\max(t'_m, t'_n) < \max(t_m, t_n)$ .

Thus, altering the assignment order outlined in Theorem 5.1 will not lead to a better solution.  $\Box$ 

#### <span id="page-9-3"></span>5.2 Finding optimal transmission order

Once we have determined the GPU assignment strategy, the computation time on each GPU is known. And the next step is to determine the communication time ( $|N_i|$  and  $|C_j|$  in Eqn. 1). In §4, we propose Theorem 4.2 to calculate  $\max(|N_i|)$  and  $\max(|C_j|)$  in the homogeneous cluster. However, This theorem cannot be directly applied to the Exclusive + Heterogeneous scenario. The main difference is that network bandwidth varies across a heterogeneous cluster. Therefore, we propose an extension of Theorem 4.2 to address this issue.

<span id="page-9-1"></span>THEOREM 5.2. The transmission order obtained in homogeneous clusters remains optimal in a heterogeneous environment. The minimum communication time is  $b_{max} = max(\sum_{i=1}^{n} d_{ij}/B_i, \sum_{i=1}^{n} d_{ij}/B_i)$ .

Theorem 5.2 states that the transmission order derived for homogeneous clusters remains optimal in a heterogeneous environment. The minimum communication time is determined by the GPU that takes the longest time to complete sending or receiving.

The proof for Theorem 5.2 follows the same structure of Theorem 4.2. The detailed proof can be found in Appx. B.

<span id="page-9-5"></span><sup>&</sup>lt;sup>2</sup>In this work, a GPU with higher computational power will not have lower bandwidth compared to a lower-end GPU.

# Takeaway 2

- The GPU assignment affects the inference time in heterogeneous clusters.
- The optimal GPU assignment involves sorting experts by number of tokens processed, then assigning them to GPUs from highest to lowest performance.
- The transmission order developed for homogeneous clusters remains optimal in a heterogeneous environment.

#### <span id="page-10-0"></span>6 COLOCATING MODELS ON HOMOGENEOUS CLUSTERS

In this section, we explore the best way to place two MoE models on a homogeneous cluster. This scenario is termed Colocating + Homogeneous.

**Solution overview.** We first demonstrate that the colocation choice affects the aggregated communication time and further the inference time. Next, we prove that a colocation solution minimizing aggregated communication time will also minimize inference time (§6.1). We determine the optimal expert colocation, which minimizes communication time, by solving the bottleneck matching problem (§6.2).

## <span id="page-10-1"></span>6.1 Minimizing inference time equals minimizing communication time

Fig. 7 illustrates a scenario where two MoE models, a and b, run simultaneously on a homogeneous cluster<sup>3</sup>. Components of Model a are shown in shades of green, while those of Model b are in shades of red. The subscript numbers (1 and 2) indicate the GPU index, and the superscript letters (a and b) refer to the model index. For example,  $G_1^b$  denotes the computation time of Model b's Gate network on GPU 1.

The Colocating + Homogeneous scenario inherits characteristics of the Exclusive + Homogeneous case (§4). These characteristics include: no need to decide GPU assignment on a homogeneous cluster, equal computation time for the Gate and Aggregation, and increased computation time for an expert when processing more tokens. Additionally, running two experts on the

<span id="page-10-2"></span>![](_page_10_Figure_11.jpeg)

Fig. 7. Running colocating MoE models on homogeneous clusters.

tokens. Additionally, running two experts on the same GPU introduces new characteristics and constraints, as illustrated below.

- (1) *Computation competition.* One model's computation components cannot start if another model is still under computing processes.
- (2) Communication overlapping. The all-to-all communications from two models can overlap in the time domain. For instance, Model a's all-to-all communication,  $C_2^a$ , can begin while Model b's communication,  $N_1^b$ , on GPU 1 is still in progress.

**Aggregated communication times.** The term  $|\overline{N^a} + \overline{N^b}|$  represents the time required to complete the first all-to-all communication for two models, which we refer to as the aggregated communication time. This differs from  $|N^a| + |N^b|$ , which simply adds the communication times of each

<span id="page-10-3"></span><sup>&</sup>lt;sup>3</sup>We only colocate models with the same number of experts, even though it's not a strict requirement in theory.

| Component | Start time               | End time                         |
|-----------|--------------------------|----------------------------------|
| $G^b$     | 0                        | $ G^b $                          |
| $N^a$     | 0                        | $ \overline{N^a} $               |
| $F^a$     | $\max( G^b , N^a )$      | $\max( G^b , N^a ) +  F^a $      |
| $N^b$     | $\geq  G^b $             | $ \overline{N^a + N^b} $         |
| $F^b$     | $\max(E_{F^a}, E_{N^b})$ | $\max(E_{F^a}, E_{N^b}) +  F^b $ |
| $C^a$     | $\geq E_{Fa}$            | $ \overline{N^a + N^b + C^a} $   |
| $A^a$     | $\max(E_{F^b}, E_{C^a})$ | $\max(E_{F^b}, E_{C^a}) +  A^a $ |
| $C^{b}$   | $\geq E_{F^b}$           | $ \overline{N^a+N^b+C^a+C^b} $   |
| $A^b$     | $\max(E_{A^a},E_{C^b})$  | $\max(E_{A^a},E_{C^b}) +  A^b $  |
| $G^a$     | $E_{A^b}$                | $E_{Ab} +  G^a $                 |

<span id="page-11-1"></span>Table 2. Start and end time of each component on the Colocating + Homogeneous scenario.

model without considering potential overlap. As shown in Fig. 7, communications  $N_1^a$  from Model a and  $N_2^b$  from Model b overlap, resulting in  $|\overline{N^a+N^b}|$  being smaller than  $|N^a|+|N^b|$ .  $|\overline{N^a+N^b}|$  is impacted by the expert colocation choice. When colocating two experts, pairing one with high communication demands with another that has fewer tokens to send can reduce the aggregated communication time.

**Inference time expression.** Based on Fig. 7, we determine the finish time of each component, as shown in Table 2. For simplicity, we display only the maximum start and end times across the n GPUs for each component. For example,  $|G^b|$  is defined as  $\max(|G_i^b|)$  for  $i \in [1, n]$ , thereby omitting the GPU index subscript. Additionally, we use E to denote the end time; for instance,  $E_{A^b}$  indicates the end time of component  $A^b$ . The inference time corresponds to the end time of  $G^a$ , which is  $E_{A^b} + |G^a|$ , as shown below.

<span id="page-11-2"></span>Inference time 
$$t = E_{Ab} + |G^a|$$
 (4)

In Eqn. 4,  $|G^a|$  is known in advance, while  $E_{A^b}$ , the end time of component  $A^b$ , is given by  $\max(E_{A^a}, E_{C^b}) + |A^b|$ . Both  $E_{A^a}$  and  $E_{C^b}$  can be further defined by the start and end times of other components. Following this approach, we can derive the complete expression for the inference time t, though it is not displayed here due to its complexity.

Rather than directly targeting inference time, we approach the problem by minimizing the aggregated communication time. In §4, we minimize the communication time by determining an optimal transmission order using Theorem 4.2. This is conducted under the context where traffic matrix is fixed. In contrast, colocation scenarios present different aggregated traffic matrices depending on the colocation choice. In the Colocating + Homogeneous scenario, minimizing communication time requires finding an expert colocation choice. The resulting traffic matrix achieves the shortest communication time when applying Theorem 4.2.

<span id="page-11-0"></span>Theorem 6.1. Minimizing aggregated all-to-all communication times of two colocating models ensures minimum inference time in a homogeneous cluster.

PROOF. We use proof by contradiction. Assume we have an optimal colocating strategy that minimizes communication times, resulting in an inference time of  $t = E_{A^b} + |G^a|$ . Now, suppose there exists another colocating strategy with higher communication times but a shorter inference time:  $E'_{N^b} > E_{N^b}, E'_{C^a} > E_{C^a}, E'_{C^b} > E_{C^b}$ , and t' < t. According to Theorem 4.2, the minimum communication time is determined solely by the maximum column or row sum. Thus, different GPU

assignment solutions for Model a do not affect this value. So we have  $E'_{N^a} = |N^a|' = E_{N^a} = |N^a|$ . In a homogeneous cluster, we have  $|G^a|' = |G^a|$ ,  $|G^b|' = |G^b|$ ,  $|A^a|' = |A^a|$ , and  $|A^b|' = |A^b|$ . Since computation time is proportional to communication time in such a cluster, it follows that  $|F^a|' > |F^a|$  and  $|F^b|' > |F^b|$ . We will now proceed with the proof by contradiction to show it is impossible to achieve a lower inference time. Specifically, we need to prove that  $t' = E'_{A^b} + |G^a|' < t = E_{A^b} + |G^a|$  cannot hold.

<span id="page-12-1"></span>
$$|G^{a}|' = |G^{a}|, \ E'_{A^{b}} + |G^{a}|' < E_{A^{b}} + |G^{a}| \Rightarrow \max(E'_{A^{a}}, E'_{C^{b}}) + |A^{b}|' < \max(E_{A^{a}}, E_{C^{b}}) + |A^{b}|$$

$$|A^{b}|' = |A^{b}|, \ E'_{C^{b}} > E_{C^{b}} \Rightarrow E'_{A^{a}} < E_{A^{a}} \Rightarrow \max(E'_{F^{b}}, E'_{C^{a}}) + |A^{a}|' < \max(E_{F^{b}}, E_{C^{a}}) + |A^{a}|$$

$$|A^{a}|' = |A^{a}|, \ E'_{C^{a}} > E_{C^{a}} \Rightarrow E'_{F^{b}} < E_{F^{b}} \Rightarrow \max(E'_{F^{a}}, E'_{N^{b}}) + |F^{b}|' < \max(E_{F^{a}}, E_{N^{b}}) + |F^{b}|$$

$$|F^{b}|' > |F^{b}|, \ E'_{N^{b}} > E_{N^{b}} \Rightarrow E'_{F^{a}} < E_{F^{a}} \Rightarrow \max(|G^{b}|', |N^{a}|') + |F^{a}|' < \max(|G^{b}|, |N^{a}|) + |F^{a}|$$

$$|F^{a}|' > |F^{a}|, \ |N^{a}|' = |N^{a}| \Rightarrow |G^{b}|' < |G^{b}|$$

Eqn. 5 demonstrates that  $|G^b|' < |G^b|$ , which contradicts our earlier assertion that  $|G^b|' = |G^b|$ . This contradiction validates the correctness of Theorem 6.1.

#### <span id="page-12-0"></span>6.2 Finding optimal expert colocation

Next, we focus on finding an expert colocation which minimizes the communication time. In Table 2, for the terms related to communication times directly, the priority is to reduce  $|\overline{N^a+N^b}|$ . Since  $E_{N^a}=|N^a|$  is unaffected by the expert colocation solution, and  $|\overline{C^a+C^b}|$  equals  $|\overline{N^a+N^b}|$ , minimizing  $|\overline{N^a+N^b}|$  also minimizes  $|\overline{N^a+N^b+C^a+C^b}|$ . Because  $N^a$  and  $C^a$  do not overlap in the time domain (they are separated by  $F^a$ ),  $|\overline{N^a+N^b+C^a}|$  can be expressed as  $|\overline{N^a+N^b}|+|\overline{C^a}|$ . This highlights the importance of optimizing  $|\overline{N^a+N^b}|$ .

In the following, we discuss how to find the expert colocation to minimize  $|N^a + N^b|$ . Suppose  $\mathbb{D}_{N_1}$  and  $\mathbb{D}_{N_2}$  represent the first all-to-all communication traffic matrices of Model a and Model b, respectively. For each potential expert colocating choice, by combining the traffic matrices of  $\mathbb{D}_{N_1}$  and  $\mathbb{D}_{N_2}$ , we create a new traffic matrix  $\mathbb{D}_{new}$ .  $\mathbb{D}_{new}$  represents the aggregated traffic matrix of the two colocating models. According to Theorem 4.2, the communication time with  $\mathbb{D}_{new}$  is determined solely by the maximum sum of traffic within each column or row. In the following, we try to find the optimal expert colocating solution, which minimizes the maximum sum of each column or row among all possible  $\mathbb{D}_{new}$ .

For traffic matrix  $\mathbb{D}_{N_1}$ , the sending and receiving traffic on GPU i are denoted as  $a_i = \sum_{j=1}^n d_{ij}$  and  $a_{n+i} = \sum_{j=1}^n d_{ji}$ , respectively. Utilizing a vector  $\mathbf{a} = [(a_1, a_{n+1}), (a_2, a_{n+2}), ..., (a_n, a_{2n})]$ , the sending/receiving traffic on n GPUs is represented. Similarly, for  $\mathbb{D}_{N_2}$ , a vector  $\mathbf{b} = [(b_1, b_{n+1}), (b_2, b_{n+2}), ..., (b_n, b_{2n})]$  is generated. For an expert colocating choice, an element is selected from both  $\mathbf{a}$  and  $\mathbf{b}$  each time, creating a new vector  $\mathbf{h}$ , which is shown in the following equation<sup>4</sup>.

<span id="page-12-3"></span>
$$\mathbf{h} = \mathbf{a}_i + \mathbf{b}_j = (a_i + b_j, a_{n+i} + b_{n+j}), \ i, j \in [1, n]$$
 (6)

The vector **h** represents the column/row sum of the traffic matrix  $\mathbb{D}_{new}$ . Next, we try to minimize the maximum value in **h**.

<span id="page-12-2"></span><sup>&</sup>lt;sup>4</sup>Considering the synchronous all-to-all communication constraint in Eqn. 6, we have  $\mathbf{h} = \mathbf{a}_i + \mathbf{b}_j = (a_i + \max(b_j, G^b), a_{n+i} + \max(b_{n+j}, G^b))$ . However, this adjustment does not impact the optimal expert colocating choice. For clarity in the following proof, we will continue to use Eqn. 6.

We minimize the maximum value in h under two cases. In Case I (Fig. 8(a)), the quantity of sending traffic is equal to the receiving traffic for each GPU. In Case II (Fig. 8(b)), the sending traffic may not necessarily be equal to the receiving traffic. Case I can be considered a specific instance of Case II. The reason for categorizing this problem into two cases is the availability of a lower-complexity algorithm tailored for Case I.

<span id="page-13-0"></span>![](_page_13_Figure_2.jpeg)

Fig. 8. (a) Case I: The optimal expert colocation solution is alternating between selecting one popular and one unpopular expert from Model a and Model b. (b) Case II: Solving the bottleneck matching problem yields the optimal expert colocation solution.

## Case I: The amount of sending traffic is equal to the receiving traffic for each GPU

For this particular case, we employ the following theorem to determine the optimal expert colocation solution.

<span id="page-13-2"></span>Theorem 6.2. Vectors  $\mathbf{a}$  and  $\mathbf{b}$  are of the same sizes. Vector  $\mathbf{h}$  is formed by adding the values selected each from  $\mathbf{a}$  and  $\mathbf{b}$ . Sort  $\mathbf{a}$  in ascending and  $\mathbf{b}$  in descending order. Selecting values from  $\mathbf{a}$  and  $\mathbf{b}$  sequentially minimizes the maximum value in  $\mathbf{h}$ .

Proof.

<span id="page-13-1"></span>
$$\begin{pmatrix} a_1 & \cdots & a_{k-1} \\ b_n & \cdots & b_{n-k+2} \end{pmatrix} \begin{pmatrix} a_k \\ b_{n-k+1} \end{pmatrix} \begin{pmatrix} a_{k+1} & \cdots & a_n \\ b_{n-k} & \cdots & b_1 \end{pmatrix}$$
 (7)

Assuming **a** and **b** are sorted in ascending and descending order, respectively, denoted as  $a_1 \le a_2 \le ... \le a_n$  and  $b_n \ge b_{n-1} \ge ... \ge b_1$ , as depicted in Eqn. 7. By summing the elements from **a** and **b** in a sequential manner, we obtain  $\mathbf{h} = [a_1 + b_n, a_2 + b_{n-1}, ..., a_n + b_1]$ . Without loss of generality, let's assume the  $k_{th}$  item of **h**, which is  $a_k + b_{n-k+1}$ , serves as the maximum value. Our next objective is to demonstrate that it is impossible to rearrange **a** and **b** to create a new vector **h**, where the maximum value is smaller than  $a_k + b_{n-k+1}$ .

Let's consider the subarray  $\mathbf{a}_{[k+1,n]} = [a_{k+1}, a_{k+2}, ..., a_n] \subseteq \mathbf{a}$ . For any value  $a_i \in \mathbf{a}_{[k+1,n]}$ , it is apparent that  $a_i \geq a_k$ . To maintain the maximum value of  $\mathbf{h}$  below  $a_k + b_{n-k+1}$ ,  $a_i$  must be paired with a value smaller than  $b_{n-k+1}$ . For any value  $b_i \in \mathbf{b}_{[1,n-k]} = [b_1, b_2, ..., b_{n-k}]$ , it holds that  $b_i \leq b_{n-k+1}$ . Therefore, the values from  $\mathbf{a}_{[k+1,n]}$  must be matched with the values from  $\mathbf{b}_{[1,n-k]}$ . Similarly, the subarray  $\mathbf{a}_{[1,k-1]} = [a_1, a_2, ..., a_{k-1}] \subseteq \mathbf{a}$  must be matched with the values from  $\mathbf{b}_{[n-k+2,n]} = [b_{n-k+2}, b_{n-k+3}, ..., b_n] \subseteq \mathbf{b}$ . Now the elements from  $\mathbf{a}_{[k+1,n]}$  are paired with elements from  $\mathbf{b}_{[1,n-k]}$ , and  $\mathbf{a}_{[1,k-1]}$  is paired with  $\mathbf{b}_{[n-k+2,n]}$ . With only  $a_k$  and  $b_{n-k+1}$  remaining, they must be paired together, obtaining  $a_k + b_{n-k+1}$ . In this scenario, the maximum value of the new vector cannot be less than  $a_k + b_{n-k+1}$ .

The idea behind Theorem 6.2 is to alternate between selecting one large and one small value from  $\bf a$  and  $\bf b$ . This means we should colocate a popular expert from Model a and an unpopular expert from Model b. This strategy reduces the aggregated communication time on that GPU where these two experts are colocated.

#### Case II: The amount of sending traffic is not equal to the receiving traffic

In Case II, where  $a_i \neq a_{n+i}$  and  $b_j \neq b_{n+j}$ , Theorem 6.2 is not applicable. This is because we cannot sort **a** and **b** where one element contains two distinct values. Attempting to sort **a** and **b** according to the larger value inside an element and then applying Theorem 6.2 does not minimize the maximum value in **h**.

We reformulate the problem as a matching problem. In Fig. 8(b), we construct the graph as follows: the experts of Model a and Model b are represented as nodes on the left and right sides of the bipartite graph, respectively. Each node on the left is connected to every node on the right with an edge weighted by  $max(a_i + b_j, a_{n+i} + b_{n+j})$ , where  $i, j \in [1, n]$ . This creates a fully connected bipartite graph with  $n^2$  edges. The weight of each edge indicates the maximum amount of data transmitted (sent or received) by a GPU if the corresponding experts from each model are colocated.

There is a direct one-to-one correspondence between the mappings of sequences a and b and perfect matchings in the constructed bipartite graph. A perfect matching is one that covers all nodes. Finding an optimal sequence mapping is thus equivalent to identifying a perfect matching that minimizes the maximum edge weight. This problem is known as the bottleneck matching problem [2]. The algorithm for solving the bottleneck matching problem is straightforward. It involves a binary search [15] on the sorted array of edges to find the minimum weight  $w_{min}$  such that a perfect matching exists in the subgraph induced by all edges with weights not exceeding  $w_{min}$ . The existence of a perfect matching in a bipartite graph can be verified using the Hopcroft-Karp algorithm [10], which has a complexity of  $O(n^2\sqrt{n})$ . Combined with binary search, the overall complexity is  $O(n^2\sqrt{n}\log n)$ .

With the optimal expert colocation solution determined, we can derive the communication time and the corresponding optimal inference time as outlined in Table 2.

In summary, this section outlines the optimal expert colocation solution to minimize inference time on homogeneous clusters. We demonstrate that optimizing the communication time for colocated models is crucial. Using Theorem 6.2 and solving the bottleneck matching problem, we identify the expert colocation solution that achieves the minimum inference time.

# Takeaway 3

- Expert colocation choices impact both the aggregated communication time and, consequently, the inference time.
- Minimizing aggregated communication time ensures minimum inference time in a homogeneous cluster.
- Aurora identifies the optimal expert colocation by solving the bottleneck matching problem, thus achieving the minimum inference time.

## <span id="page-14-0"></span>7 COLOCATING MODELS ON HETEROGENEOUS CLUSTERS

In this section, we focus on colocating models on heterogeneous clusters. Achieving minimum inference time in the Colocating + Heterogeneous scenario requires expert colocation, GPU assignment, and communication scheduling.

**Solution overview.** We first identify that optimizing inference time in the Colocating + Heterogeneous scenario is an NP-hard problem (§7.1), and then we propose a sub-optimal yet effective solution (§7.2).

#### <span id="page-14-1"></span>7.1 NP-hardness proof

Fig. 9 illustrates the case of running two MoE models on a heterogeneous cluster. Similar to the Colocating + Homogeneous scenario, the inference time can be expressed using Eqn. 4, with the finish times for each component detailed in Table 2.

<span id="page-15-2"></span>![](_page_15_Figure_1.jpeg)

![](_page_15_Figure_2.jpeg)

Fig. 10. (a) Optimal expert colocation and GPU assignment solution is obtained by solving a 3-dimensional matching problem. (b) We can reduce the 3-dimensional matching problem to two 2-dimensional matching problems.

In §6, Theorem 6.1 demonstrates that minimizing aggregated communication times ensures the minimum inference time on a homogeneous cluster. However, this theorem does not apply to a heterogeneous cluster. In the homogeneous environment, computation times are identical across GPUs. Therefore, we have  $|G^a|' = |G^a|$ ,  $|G^b|' = |G^b|$ ,  $|A^a|' = |A^a|$ , and  $|A^b|' = |A^b|$  in the proof of Theorem 6.1. Additionally, we apply  $|F^a|' > |F^a|$  and  $|F^b|' > |F^b|$ , indicating that computation time is proportional to communication time. However, these equations and inequalities do not hold in a heterogeneous cluster. As demonstrated in Fig. 9,  $G_1^b \neq G_2^b$ ,  $A_1^b \neq A_2^b$ , rendering Theorem 6.1 inapplicable in such heterogeneous environments.

<span id="page-15-1"></span>![](_page_15_Figure_5.jpeg)

Fig. 9. Running colocating MoE models on heterogeneous clusters.

We can reformulate the optimization problem as a 3-dimensional matching problem, as illustrated in Fig. 10(a). Unlike the scenario depicted in Fig. 8, this formulation requires both expert colocation and GPU assignment. The 3-dimensional matching problem extends bipartite matching (also known as 2-dimensional matching). A hyperedge, connecting one GPU and one expert from Model a and one expert from Model b, represents the inference time occurring on that GPU. We must determine two perfect matchings among two bipartite graphs. Similar to the bottleneck matching problem applied in Case II (a6), we need to find a perfect matching that minimizes the maximum weight. The 3-dimensional matching problem is proven to be NP-hard [a6], meaning that we cannot solve the optimization problem in polynomial time.

## <span id="page-15-0"></span>7.2 Sub-optimal approach

We use a sub-optimal solution by decoupling the matchings in the two bipartite graphs.

We first determine the perfect matching among experts, setting aside GPU assignment initially. Following the method described in Case II (§6), we solve the bottleneck matching problem to obtain the expert colocation solution. This reduces the 3-dimensional matching problem to a 2-dimensional matching problem. In Fig. 10(b), the left side represents GPUs, and the right side represents the combination of two experts, with the edge weight indicating inference time on the connected GPU. We resolve the bottleneck matching problem to determine the minimum of the maximum weights. Combined with the expert colocation solution, this provides a complete, sub-optimal solution.

In conclusion, achieving minimum inference time in the Colocating + Heterogeneous scenario can be formulated as a 3-dimensional matching problem, which is proven to be NP-hard [6]. Based on our evaluation in §8, this solution achieves an inference time just 1.07× of the optimal.

# Takeaway 4

- In a heterogeneous cluster, minimizing aggregated all-to-all communication times of two colocating models does not ensure minimum inference time.
- Minimizing inference time in the Colocating + Heterogeneous scenario can be formulated as an NP-hard matching problem.
- We propose a sub-optimal approach by decoupling the optimization problem into two perfect matching problems.

#### <span id="page-16-0"></span>8 EVALUATION

The evaluation seeks to address the following key questions.

**Q1:** Can Aurora reduce inference time across four scenarios? Aurora achieves up to 1.38× faster inference time in the Exclusive + Homogeneous scenario and up to 1.81× faster in the Exclusive + Heterogeneous scenario. In the colocating scenario, Aurora shows an improvement of up to 2.38× in the homogeneous case and up to 3.54× in the heterogeneous case.

**Q2: Can Aurora improve GPU utilization?** In the colocation scenario, Aurora delivers a 1.28× to 1.50× improvement in GPU utilization compared to the state-of-the-art solution.

Q3: How close is Aurora to the optimum in the Colocating + Heterogeneous scenario? On average, Aurora prolongs the inference time by only  $1.07 \times$  compared to the optimum.

**Q4:** How does Aurora perform under imprecise traffic inputs? Aurora maintains inference time performance under unpredictable inference requests, with only a 15.8% degradation.

#### 8.1 Simulation setup

**GPU clusters.** The GPUs are connected through a large switch, as shown in Fig. 4(a). In homogeneous clusters, the network bandwidth is set to 100 Gbps. For heterogeneous clusters, we define four types of GPUs, with bandwidths of 100 Gbps, 80 Gbps, 50 Gbps, and 40 Gbps, ordered from highest to lowest performance. The number of GPUs for each type is the same. In the exclusive scenario, each MoE model uses the network bandwidth independently. In the colocation scenario, models only compete bandwidth when their experts are placed on the same device.

**MoE models.** We use production model statistics from Google [21] to drive our simulation. It includes data for four layers of two MoE models, B/16 and B/32, each with 8 experts. We derive Aurora's input parameters from the model information based on the COCO and ImageNet datasets. **Metrics.** We consider the following metrics in the evaluation.

- *Inference time*. We calculate the inference time for all four scenarios.
- *GPU utilization*. GPU utilization is the ratio of computation time (including the Gate, FFN, and Aggregation) to the inference time.

**Baselines.** Aurora is the first of its kind, making it difficult to find directly comparable work. For expert colocation, we compare Aurora with Lina [18], the latest approach using expert colocation. We also implement vanilla expert colocation, referred to as random expert colocation (REC), as the baseline. To ensure fairness, all solutions colocate two experts on the same device. Lina<sup>5</sup> pairs the most popular expert with the least popular one within each job, while Aurora and REC colocate experts from two different models.

For GPU assignment in heterogeneous clusters, we use the vanilla approach, random GPU assignment (RGA), as the baseline.

<span id="page-16-1"></span><sup>&</sup>lt;sup>5</sup>Lina consists of three main components: prioritizing all-to-all over all-reduce, pipelining communication and computation, and packing multiple experts on a single device. The first component is specific to MoE training and does not apply to Aurora. The second complements Aurora, while the third is closely related. We implement the third component for Lina.

<span id="page-17-0"></span>![](_page_17_Figure_1.jpeg)

Fig. 11. Inference time comparison in (a) Exclusive + Homogeneous, (b) Exclusive + Heterogeneous, (c) Colocating + Homogeneous, and (d) Colocating + Heterogeneous scenarios.

<span id="page-17-1"></span>![](_page_17_Figure_3.jpeg)

Fig. 12. GPU utilization in the (a) Colocating + Homogeneous and (b) Colocating + Heterogeneous scenarios.

For all-to-all communication scheduling, we employ the shortest job first (SJF), which is a wellknown flow scheduling policy for minimizing average flow completion time. We also include the vanilla method, random communication scheduling (RCS).

# 8.2 Results

(Q1) Aurora reduces inference time across four scenarios. We evaluate inference time across various scenarios. Fig. [11a](#page-17-0) shows a comparison of inference times for three scheduling algorithms: Aurora, SJF, and RCS. These algorithms decide the order of token transmission between GPUs during all-to-all communication, based on the traffic matrix of each layer. Aurora consistently outperforms both SJF and RCS across all model layers and datasets, achieving communication times that are up to 1.38× faster compared to SJF. This demonstrates its efficiency in minimizing communication time. In contrast, SJF shows performance similar to or even worse than RCS. This is because prioritizing tokens with less traffic offers no advantage. SJF's inability to reduce bandwidth contention results in outcomes that are nearly identical to those of RCS.

In Fig. [11b,](#page-17-0) we present a comparison of inference times between Aurora and RGA in the Exclusive + Heterogeneous scenario. With Aurora, inference times are accelerated by 1.36× to 1.81× across

<span id="page-18-1"></span>![](_page_18_Figure_1.jpeg)

Fig. 14. Inference time acceleration across different number of layers in the (a) Exclusive + Heterogeneous and (b) Colocating + Heterogeneous scenarios.

various models and layers. This is achieved by assigning popular experts to high-end GPUs, optimizing overall inference performance.

Furthermore, Fig. 11c illustrates the inference time when two experts are colocated on the same GPU. Lina colocates experts from the same model, and we show the inference time for each model separately. Aurora consistently achieves the shortest inference time compared to Lina, REC, and RGA + REC. Under the homogeneous case, Aurora is 1.25× to 2.38× faster than Lina, while in the heterogeneous scenario (Fig. 11d), it improves by 1.91× to 3.54×. Aurora places experts from two different models, allowing them to avoid the synchronous all-to-all communication constraint. In contrast, with Lina, colocated experts must wait for each other to complete communication, which can lead to longer inference times.

**(Q2) Aurora improves GPU utilization.** Fig. 12a illustrates GPU utilization in the homogeneous case. Aurora + Colocation refers to placing two experts on the same GPU, while Aurora + Exclusive represents assigning one expert per GPU. GPU utilization is notably low when running MoE models exclusively, with most models below 20%. By colocating two experts on a single device, Aurora achieves a 1.57× to 1.72× increase in GPU utilization. In general, colocating two experts is expected to nearly double GPU utilization, but the observed improvement is lower. This is because, with multiple experts sharing a GPU, inference time for Aurora + Colocation is longer compared to Aurora + Exclusive, reducing the potential GPU utilization gains. Aurora achieves a significant improvement over Lina, with an increase of 1.28× to 1.50×. A similar trend is observed in the heterogeneous case, as shown in Fig. 12b.

(Q3) Aurora realizes close performance to the optimal solution. Aurora achieves minimal inference time in most scenarios, except for the Colocating + Heterogeneous case. Fig. 13 shows the inference time gap between Aurora and the optimum, obtained through brute-force search. On average, Aurora prolongs the inference time by only 1.07×, which is a small difference given that it significantly outperforms other baseline methods.

(O4) Aurora maintains inference time

<span id="page-18-0"></span>![](_page_18_Figure_7.jpeg)

Fig. 13. Performance gap between Aurora and the optimum in the Colocating + Heterogeneous scenario.

**performance under imprecise traffic inputs.** Once Aurora's optimization plan is deployed, subsequent inference requests become unpredictable. We use the traffic matrix of the first layer for Aurora optimization and add traffic from the other three layers in the dataset as noise to simulate unpredictable requests. The level of imprecision ranges from 0% to 75% as traffic from each additional layer is incorporated.

Fig. [14](#page-18-1) shows the inference time acceleration of Aurora compared to RGA (Fig. [14a\)](#page-18-1) and RGA+REC (Fig. [14b\)](#page-18-1) across varying traffic imprecision. As expected, the inference time reduction generally decreases slightly as the traffic matrix becomes more imprecise. As expected, inference time reduction decreases slightly with increased traffic matrix imprecision. In the Exclusive+Heterogeneous scenario, acceleration drops from approximately 1.90× with precise traffic to 1.60× with 75% imprecision. Similarly, in the Colocating+Heterogeneous scenario, acceleration decreases from about 2.0× to 1.80×. The maximum performance degradation is 15.8% with high noise traffic, demonstrating that Aurora still achieves significant inference time improvements even with imprecise inputs.

# 9 RELATED WORK

Load balancing. Various gating methods have been proposed to ensure even token distribution [\[3,](#page-20-5) [7,](#page-20-8) [9,](#page-20-0) [17,](#page-20-10) [29\]](#page-21-8). Some use an auxiliary loss function to penalize imbalances [\[12,](#page-20-12) [30\]](#page-21-9), while others regulate expert capacity [\[4,](#page-20-7) [12\]](#page-20-12). Dynamic-Gating MoE [\[11\]](#page-20-1) allows experts to process a variable number of tokens, and Pre-Gated MoE [\[13\]](#page-20-9) predicts token distribution based on the previous layer's gate. However, these methods can impede model convergence and degrade overall quality.

All-to-all acceleration. All-to-all communication is a key bottleneck in MoE model training and inference [\[9,](#page-20-0) [27\]](#page-21-0). To improve efficiency, Faster-MoE [\[9\]](#page-20-0) uses a pairwise exchange algorithm, Tutel [\[12\]](#page-20-12) introduces hierarchical strategies, and DeepSpeed-MoE [\[27\]](#page-21-0) employs tensor parallelism and slicing. Fast-MoE [\[8\]](#page-20-11) also uses tensor slicing and data parallelism. However, these methods mainly focus on communication speed, neglecting GPU utilization and system heterogeneity.

Expert colocation and replication. Some solutions improve GPU utilization by colocating multiple experts from the same job. Lina [\[18\]](#page-20-2) packs experts on a single GPU to reduce all-to-all transfer sizes, while Dynamic-Gating MoE [\[11\]](#page-20-1) offloads less-used experts to CPU memory. Other methods replicate popular experts across GPUs. FlexMoE [\[23\]](#page-20-4) dynamically shifts experts based on workload, and Prophet [\[36\]](#page-21-12) maps experts to specific GPU subsets. Lazarus [\[38\]](#page-21-3) uses expert deployment and replication to enhance training during GPU failures. However, these approaches do not address copacking experts from different models or solving all-to-all communication challenges. GPU heterogeneity. GPU heterogeneity is becoming more common in production clusters [\[37\]](#page-21-5) and has drawn significant academic interest [\[14,](#page-20-13) [22,](#page-20-6) [41,](#page-21-6) [44\]](#page-21-15). While these solutions improve the management of heterogeneous GPU clusters, they do not specifically address MoE models.

Flow scheduling. Some existing research [\[5,](#page-20-20) [16,](#page-20-21) [20,](#page-20-22) [25,](#page-21-16) [34\]](#page-21-17) attempts to model various training paradigms and optimize flow scheduling to speed up the process. However, these approaches often overlook the specific advantages of all-to-all communication.

GPU sharing. Experts colocated on the same device share GPU resources, and GPU sharing has been extensively explored in previous research [\[35,](#page-21-18) [39,](#page-21-19) [40,](#page-21-14) [43\]](#page-21-20). Even though these solutions do not explore MoE models, the engineering techniques they employ can help reduce overhead when multiple experts share GPU resources alternately.

# 10 CONCLUSION

In conclusion, Aurora effectively addresses key challenges in MoE inference by optimizing model deployment and communication scheduling. While this work marks an important first step, it opens up several promising avenues for future research. One direction is extending Aurora to handle more complex environments, including those with varying network topologies and communication protocols. Another potential enhancement involves developing adaptive strategies that dynamically adjust model deployment and communication scheduling based on changing workloads, which could further improve performance. Additionally, integrating Aurora with other optimization techniques, such as job scheduling and network topology design, may provide further synergistic benefits. These efforts aim to improve the scalability and efficiency of MoE models in increasingly diverse and demanding computing environments.

#### REFERENCES

- <span id="page-20-14"></span>[1] 2024. Farkas' Lemma. https://en.wikipedia.org/wiki/Farkas%27 lemma.
- <span id="page-20-15"></span>[2] Rainer E Burkard and Ulrich Derigs. 1980. The bottleneck matching problem. In Assignment and Matching Problems: Solution Methods with FORTRAN-Programs. Springer, 60–71.
- <span id="page-20-5"></span>[3] Chang Chen, Min Li, Zhihua Wu, Dianhai Yu, and Chao Yang. 2022. Ta-moe: Topology-aware large scale mixture-of-expert training. Advances in Neural Information Processing Systems 35 (2022), 22173–22186.
- <span id="page-20-7"></span>[4] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. 2019. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509 (2019).
- <span id="page-20-20"></span>[5] Minsik Cho, Ulrich Finkler, David Kung, and Hillery Hunter. 2019. Blueconnect: Decomposing all-reduce for deep learning on heterogeneous network hierarchy. *Proceedings of Machine Learning and Systems* 1 (2019), 241–251.
- <span id="page-20-18"></span>[6] Yves Crama and Frits CR Spieksma. 1992. Approximation algorithms for three-dimensional assignment problems with triangle inequalities. *European Journal of Operational Research* 60, 3 (1992), 273–279.
- <span id="page-20-8"></span>[7] William Fedus, Barret Zoph, and Noam Shazeer. 2022. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research* 23, 120 (2022), 1–39.
- <span id="page-20-11"></span>[8] Jiaao He, Jiezhong Qiu, Aohan Zeng, Zhilin Yang, Jidong Zhai, and Jie Tang. 2021. Fastmoe: A fast mixture-of-expert training system. arXiv preprint arXiv:2103.13262 (2021).
- <span id="page-20-0"></span>[9] Jiaao He, Jidong Zhai, Tiago Antunes, Haojie Wang, Fuwen Luo, Shangfeng Shi, and Qin Li. 2022. Fastermoe: modeling and optimizing training of large-scale dynamic pre-trained models. In Proceedings of the 27th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming. 120–134.
- <span id="page-20-17"></span>[10] John E. Hopcroft and Richard M. Karp. 1973. An n<sup>5/2</sup> Algorithm for Maximum Matchings in Bipartite Graphs. SIAM J. Comput. 2, 4 (1973), 225–231.
- <span id="page-20-1"></span>[11] Haiyang Huang, Newsha Ardalani, Anna Sun, Liu Ke, Hsien-Hsin S Lee, Anjali Sridhar, Shruti Bhosale, Carole-Jean Wu, and Benjamin Lee. 2023. Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference. arXiv preprint arXiv:2303.06182 (2023).
- <span id="page-20-12"></span>[12] Changho Hwang, Wei Cui, Yifan Xiong, Ziyue Yang, Ze Liu, Han Hu, Zilong Wang, Rafael Salas, Jithin Jose, Prabhat Ram, et al. 2023. Tutel: Adaptive mixture-of-experts at scale. *Proceedings of Machine Learning and Systems* 5 (2023).
- <span id="page-20-9"></span>[13] Ranggi Hwang, Jianyu Wei, Shijie Cao, Changho Hwang, Xiaohu Tang, Ting Cao, Mao Yang, and Minsoo Rhu. 2023. Pre-gated moe: An algorithm-system co-design for fast and scalable mixture-of-expert inference. arXiv preprint arXiv:2308.12066 (2023).
- <span id="page-20-13"></span>[14] Suhas Jayaram Subramanya, Daiyaan Arfeen, Shouxu Lin, Aurick Qiao, Zhihao Jia, and Gregory R Ganger. 2023. Sia: Heterogeneity-aware, goodput-optimized ML-cluster scheduling. In Proceedings of the 29th Symposium on Operating Systems Principles. 642–657.
- <span id="page-20-16"></span>[15] D.E. Knuth. 1998. The Art of Computer Programming: Sorting and Searching, Volume 3. Pearson Education. https://books.google.de/books?id=cYULBAAAOBAJ
- <span id="page-20-21"></span>[16] ChonLam Lao, Yanfang Le, Kshiteej Mahajan, Yixi Chen, Wenfei Wu, Aditya Akella, and Michael Swift. 2021. ATP: In-network Aggregation for Multi-tenant Learning. In 18th USENIX Symposium on Networked Systems Design and Implementation (NSDI 21). 741–761.
- <span id="page-20-10"></span>[17] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. 2020. Gshard: Scaling giant models with conditional computation and automatic sharding. arXiv preprint arXiv:2006.16668 (2020).
- <span id="page-20-2"></span>[18] Jiamin Li, Yimin Jiang, Yibo Zhu, Cong Wang, and Hong Xu. 2023. Accelerating distributed {MoE} training and inference with lina. In 2023 USENIX Annual Technical Conference (USENIX ATC 23). 945–959.
- <span id="page-20-3"></span>[19] Juncai Liu, Jessie Hui Wang, and Yimin Jiang. 2023. Janus: A unified distributed training framework for sparse mixture-of-experts models. In Proceedings of the ACM SIGCOMM 2023 Conference. 486–498.
- <span id="page-20-22"></span>[20] Liang Luo, Peter West, Jacob Nelson, Arvind Krishnamurthy, and Luis Ceze. 2020. Plink: Discovering and exploiting locality for accelerated distributed training on the public cloud. Proceedings of Machine Learning and Systems 2 (2020), 82–97.
- <span id="page-20-19"></span>[21] Basil Mustafa, Carlos Riquelme, Joan Puigcerver, Rodolphe Jenatton, and Neil Houlsby. 2022. Multimodal contrastive learning with limoe: the language-image mixture of experts. Advances in Neural Information Processing Systems 35 (2022), 9564–9576.
- <span id="page-20-6"></span>[22] Deepak Narayanan, Keshav Santhanam, Fiodar Kazhamiaka, Amar Phanishayee, and Matei Zaharia. 2020. {Heterogeneity-Aware} cluster scheduling policies for deep learning workloads. In 14th USENIX Symposium on Operating Systems Design and Implementation (OSDI 20). 481–498.
- <span id="page-20-4"></span>[23] Xiaonan Nie, Xupeng Miao, Zilong Wang, Zichao Yang, Jilong Xue, Lingxiao Ma, Gang Cao, and Bin Cui. 2023. Flexmoe: Scaling large-scale sparse pre-trained model training via dynamic device placement. Proceedings of the ACM on Management of Data 1, 1 (2023), 1–19.

- <span id="page-21-7"></span>[24] Xiaonan Nie, Pinxue Zhao, Xupeng Miao, Tong Zhao, and Bin Cui. 2022. HetuMoE: An efficient trillion-scale mixture-of-expert distributed training system. arXiv preprint arXiv:2203.14685 (2022).
- <span id="page-21-16"></span>[25] Rui Pan, Yiming Lei, Jialong Li, Zhiqiang Xie, Binhang Yuan, and Yiting Xia. 2022. Efficient flow scheduling in distributed deep learning training with echelon formation. In Proceedings of the 21st ACM Workshop on Hot Topics in Networks. 93–100.
- <span id="page-21-13"></span>[26] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. 2018. Improving language understanding by generative pre-training. (2018).
- <span id="page-21-0"></span>[27] Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, and Yuxiong He. 2022. Deepspeed-moe: Advancing mixture-of-experts inference and training to power next-generation ai scale. In *International conference on machine learning*. PMLR, 18332–18346.
- <span id="page-21-10"></span>[28] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. 2020. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis. IEEE, 1–16.
- <span id="page-21-8"></span>[29] Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, André Susano Pinto, Daniel Keysers, and Neil Houlsby. 2021. Scaling vision with sparse mixture of experts. Advances in Neural Information Processing Systems 34 (2021), 8583–8595.
- <span id="page-21-9"></span>[30] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. 2017. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538 (2017).
- <span id="page-21-1"></span>[31] Liang Shen, Zhihua Wu, WeiBao Gong, Hongxiang Hao, Yangfan Bai, HuaChao Wu, Xinxuan Wu, Jiang Bian, Haoyi Xiong, Dianhai Yu, et al. 2022. Se-moe: A scalable and efficient mixture-of-experts distributed training and inference system. arXiv preprint arXiv:2205.10034 (2022).
- <span id="page-21-2"></span>[32] Shaohuai Shi, Xinglin Pan, Qiang Wang, Chengjian Liu, Xiaozhe Ren, Zhongzhe Hu, Yu Yang, Bo Li, and Xiaowen Chu. 2024. ScheMoE: An Extensible Mixture-of-Experts Distributed Training System with Tasks Scheduling. In Proceedings of the Nineteenth European Conference on Computer Systems. 236–249.
- <span id="page-21-11"></span>[33] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. 2019. Megatron-lm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053 (2019).
- <span id="page-21-17"></span>[34] Raajay Viswanathan, Arjun Balasubramanian, and Aditya Akella. 2020. Network-accelerated distributed machine learning for multi-tenant settings. In *Proceedings of the 11th ACM Symposium on Cloud Computing*. 447–461.
- <span id="page-21-18"></span>[35] Guanhua Wang, Kehan Wang, Kenan Jiang, Xiangjun Li, and Ion Stoica. 2021. Wavelet: Efficient DNN Training with Tick-Tock Scheduling. *Proceedings of Machine Learning and Systems* 3 (2021), 696–710.
- <span id="page-21-12"></span>[36] Wei Wang, Zhiquan Lai, Shengwei Li, Weijie Liu, Keshi Ge, Yujie Liu, Ao Shen, and Dongsheng Li. 2023. Prophet: Fine-grained Load Balancing for Parallel Training of Large-scale MoE Models. In 2023 IEEE International Conference on Cluster Computing (CLUSTER). IEEE, 82–94.
- <span id="page-21-5"></span>[37] Qizhen Weng, Wencong Xiao, Yinghao Yu, Wei Wang, Cheng Wang, Jian He, Yong Li, Liping Zhang, Wei Lin, and Yu Ding. 2022. {MLaaS} in the wild: Workload analysis and scheduling in {Large-Scale} heterogeneous {GPU} clusters. In 19th USENIX Symposium on Networked Systems Design and Implementation (NSDI 22). 945–960.
- <span id="page-21-3"></span>[38] Yongji Wu, Wenjie Qu, Tianyang Tao, Zhuang Wang, Wei Bai, Zhuohao Li, Yuan Tian, Jiaheng Zhang, Matthew Lentz, and Danyang Zhuo. 2024. Lazarus: Resilient and Elastic Training of Mixture-of-Experts Models with Adaptive Expert Placement. arXiv preprint arXiv:2407.04656 (2024).
- <span id="page-21-19"></span>[39] Wencong Xiao, Romil Bhardwaj, Ramachandran Ramjee, Muthian Sivathanu, Nipun Kwatra, Zhenhua Han, Pratyush Patel, Xuan Peng, Hanyu Zhao, Quanlu Zhang, et al. 2018. Gandiva: Introspective cluster scheduling for deep learning. In 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18). 595–610.
- <span id="page-21-14"></span>[40] Wencong Xiao, Shiru Ren, Yong Li, Yang Zhang, Pengyang Hou, Zhi Li, Yihui Feng, Wei Lin, and Yangqing Jia. 2020. {AntMan}: Dynamic Scaling on {GPU} Clusters for Deep Learning. In 14th USENIX Symposium on Operating Systems Design and Implementation (OSDI 20). 533-548.
- <span id="page-21-6"></span>[41] Xiaodong Yi, Shiwei Zhang, Ziyue Luo, Guoping Long, Lansong Diao, Chuan Wu, Zhen Zheng, Jun Yang, and Wei Lin. 2020. Optimizing distributed training deployment in heterogeneous GPU clusters. In Proceedings of the 16th International Conference on emerging Networking Experiments and Technologies. 93–107.
- <span id="page-21-4"></span>[42] Dianhai Yu, Liang Shen, Hongxiang Hao, Weibao Gong, Huachao Wu, Jiang Bian, Lirong Dai, and Haoyi Xiong. 2024. MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services. IEEE Transactions on Services Computing (2024).
- <span id="page-21-20"></span>[43] Peifeng Yu and Mosharaf Chowdhury. 2020. Salus: Fine-Grained GPU Sharing Primitives for Deep Learning Applications. MLSys' 20 (2020).
- <span id="page-21-15"></span>[44] Shiwei Zhang, Lansong Diao, Chuan Wu, Zongyan Cao, Siyu Wang, and Wei Lin. 2024. HAP: SPMD DNN Training on Heterogeneous GPU Clusters with Automated Program Synthesis. In Proceedings of the Nineteenth European Conference on Computer Systems. 524–541.

#### **APPENDIX**

#### <span id="page-22-0"></span>A PROOF OF THEOREM 4.2

Theorem 4.2 states that the minimum communication time with the traffic matrix  $\mathbb{D}$  is given by  $b_{max} = max(\sum_{j=1}^{n} d_{ij}, \sum_{i=1}^{n} d_{ij})/B$ . Here,  $d_{ij}$  is the element located at row i and column j in  $\mathbb{D}$ , and B denotes the bandwidth for each homogeneous GPU.

For simplicity, we set bandwidth B to 1. Our approach unfolds in three key steps. Initially, we illustrate the conversion of the traffic matrix  $\mathbb D$  into  $\mathbb D'$  by incorporating matrix  $\mathbb X$ . Subsequently, we prove that the minimum communication time for  $\mathbb D'$  is  $b_{max}$ . Finally, we prove the existence of a non-negative  $\mathbb X$ .

#### 1. Convert $\mathbb{D}$ to $\mathbb{D}'$ by adding non-negative $\mathbb{X}$

<span id="page-22-2"></span>
$$\begin{bmatrix} d_{11} & d_{12} & \dots & d_{1n} \\ d_{21} & d_{22} & \dots & d_{2n} \\ \dots & \dots & \dots & \dots \\ d_{n1} & d_{n2} & \dots & d_{nn} \end{bmatrix} b_1 & \begin{bmatrix} x_{11} & x_{12} & \dots & x_{1n} \\ x_{21} & x_{22} & \dots & x_{2n} \\ \dots & \dots & \dots & \dots \\ x_{n1} & x_{n2} & \dots & x_{nn} \end{bmatrix} \Delta b_1 & \begin{bmatrix} d_{11} & d_{12} & \dots & d_{1n} \\ d_{21} & d_{22} & \dots & d_{2n} \\ \dots & \dots & \dots & \dots \\ d_{n1} & d_{n2} & \dots & d_{nn} \end{bmatrix} b_{max} \\ \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\ d_{n1}' & d_{n2}' & \dots & d_{nn}' \end{bmatrix} b_{max} \\ \vdots & \vdots & \vdots & \vdots & \vdots \\ \vdots & \vdots & \vdots & \vdots & \vdots$$

Eqn. 8 illustrates the relationship  $\mathbb{D}+\mathbb{X}=\mathbb{D}'$ , where  $x_{ij}$  and  $d'_{ij}$  denote elements located at row i and column j in  $\mathbb{X}$  and  $\mathbb{D}'$ , respectively. Values external to the matrices, such as  $b_1$ ,  $\Delta b_1$ , and  $b_{max}$ , represent the sum of their corresponding columns or rows. For the traffic matrix  $\mathbb{D}'$ , the conditions are met such that the sum of each row  $\sum_{j=1}^n d'_{ij} = b_{max}$  and the sum of each column  $\sum_{i=1}^n d'_{ij} = b_{max}$ . That is to say, each GPU is sending and receiving precisely  $b_{max}$  traffic after adding artificial traffic matrix  $\mathbb{X}$ . As shown in Fig. 15(a), each GPU sends and receives  $b_{max}$  traffic in total. The label attached to each traffic entry indicates the target GPU to which the traffic is directed.

#### 2. Prove the minimum communication time for $\mathbb{D}'$ is $b_{max}$

Now, our attention shifts to determining the minimum communication time for  $\mathbb{D}'$ . To establish that the minimum communication time for  $\mathbb{D}'$  is indeed  $b_{max}$ , it is imperative to demonstrate that each GPU is capable of transmitting and receiving traffic without any interruptions, until all the traffic is completely finished. Any interruption would necessarily result in a communication time exceeding  $b_{max}$ , given that each GPU is expected to both send and receive a total of  $b_{max}$  traffic.

As depicted in Fig. 15(b), one *time slot* is required to transmit a token at full bandwidth. At

<span id="page-22-1"></span>![](_page_22_Figure_11.jpeg)

Fig. 15. (a) Each GPU sends/receives  $b_{max}$  traffic in total. (b) Each GPU receives only one token at a time.

any given time slot t, each GPU is configured to transmit just one token at its full bandwidth. As a result, each GPU can only receive one token at the same time. This is attributed to the parity in bandwidth between the sending and receiving sides, where the receiving side cannot simultaneously accommodate two tokens sent by two GPUs at full bandwidth.

We then proceed to establish that each GPU can transmit and receive tokens without any disruptions until all tokens are completely delivered. At time slot t, we can identify the presence

of n tokens, one originating from each GPU, with each destined for a distinct GPU among the n GPUs. These n tokens are systematically labeled from 1 to n. To verify this, we employ a proof by contradiction. We assume the hypothetical scenario where no token is directed to GPU i (i.e., a token labeled with i) at time slot t. This assumption leads to the conclusion that the receiving traffic of GPU i cannot reach the stipulated value of  $b_{max}$ . This, however, contradicts the requirement that each GPU must receive  $b_{max}$  traffic under the traffic matrix  $\mathbb{D}'$ . Given the presence of n tokens heading to n distinct GPUs, these tokens can be transmitted without any contention during time slot t. This process can be iterated until all tokens are successfully transmitted. In other words, under the traffic matrix  $\mathbb{D}'$ , all GPUs participate in a seamless exchange of traffic without any interruptions. The minimum communication time for  $\mathbb{D}'$  is firmly established as  $b_{max}$ .

<span id="page-23-0"></span>
$$\begin{cases} x_{11} + x_{12} + \dots + x_{1n} &= b_{max} - b_1 &= \Delta b_1 \\ x_{21} + x_{22} + \dots + x_{2n} &= b_{max} - b_2 &= \Delta b_2 \\ & \vdots & \vdots & \vdots \\ x_{n1} + x_{n2} + \dots + x_{nn} &= b_{max} - b_n &= \Delta b_n \\ x_{11} + x_{21} + \dots + x_{n1} &= b_{max} - b_{n+1} &= \Delta b_{n+1} \\ & \vdots & \vdots & \vdots \\ x_{1n} + x_{2n} + \dots + x_{nn} &= b_{max} - b_{2n} &= \Delta b_{2n} \end{cases}$$

$$(9)$$

<span id="page-23-1"></span>
$$\begin{vmatrix}
1 & 2 & \cdots & n & n+1 & \cdots & n^{2} \\
1 & 1 & 1 & \cdots & 1 & 0 & \cdots & 0 \\
2 & 0 & 0 & \cdots & 0 & 1 & \cdots & 0 \\
\vdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots \\
n & 0 & 0 & \cdots & 0 & 0 & \cdots & 1 \\
1 & 0 & \cdots & 0 & 1 & \cdots & 0 \\
\vdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots \\
2n & 0 & 0 & \cdots & 1 & 0 & \cdots & 1
\end{vmatrix}
\begin{vmatrix}
x_{11} \\ x_{12} \\ x_{13} \\ \vdots \\ x_{1n} \\ x_{21} \\ \vdots \\ x_{nn}
\end{vmatrix} = \begin{bmatrix}
\Delta b_{1} \\ \Delta b_{2} \\ \vdots \\ \Delta b_{n} \\ \vdots \\ \Delta b_{2n}
\end{vmatrix}$$
(10)

#### 3. Prove the existence of non-negative X

In this step, we need to prove the presence of a non-negative matrix  $\mathbb X$ . The existence of a non-negative  $\mathbb X$  is a critical factor in ensuring that the minimum communication time of  $\mathbb D$  does not exceed that of  $\mathbb D'$ .

Eqn. 9 presents the equations that elements of  $\mathbb{X}$  should satisfy. We transform the  $n \times n$  matrix  $\mathbb{X}$  to an  $n^2 \times 1$  vector  $\mathbf{x}$ . Then we express these equations in matrix format as  $\mathbb{A}\mathbf{x} = \Delta \mathbf{b}$ , which is shown in Eqn. 10. The size of  $\mathbb{A}$  is  $2n \times n^2$ . Notably, it is apparent that for every  $\Delta b_i \in \Delta \mathbf{b}$ , we have  $\Delta b_i \geq 0$  since  $b_{max} \geq b_i$ . Next, we use Farkas' Lemma [1] to prove the existence of a non-negative solution  $\mathbf{x}$ .

Farkas' Lemma [1]: Let  $A \in \mathbb{R}^{m \times n}$  and  $b \in \mathbb{R}^m$ . Then exactly one of the following two assertions is true:

- 1. There exists an  $\mathbf{x} \in \mathbb{R}^n$  such that  $A\mathbf{x} = \mathbf{b}$  and  $\mathbf{x} \ge 0$ .
- 2. There exists  $a \mathbf{y} \in \mathbb{R}^m$  such that  $\mathbb{A}^T \mathbf{y} \geq 0$  and  $\mathbf{b}^T \mathbf{y} < 0$ .

Here, the notation  $x \ge 0$  means that all components of the vector x are non-negative.

<span id="page-24-1"></span>
$$\begin{cases} y_{1} + y_{n+1} \ge 0 \\ y_{1} + y_{n+2} \ge 0 \\ \vdots \\ y_{1} + y_{2n} \ge 0 \end{cases} \begin{cases} y_{2} + y_{n+1} \ge 0 \\ y_{2} + y_{n+2} \ge 0 \\ \vdots \\ y_{2} + y_{2n} \ge 0 \end{cases} \begin{cases} y_{n} + y_{n+1} \ge 0 \\ y_{n} + y_{n+2} \ge 0 \\ \vdots \\ y_{n} + y_{2n} \ge 0 \end{cases}$$
(11)

<span id="page-24-2"></span>
$$\begin{cases}
y_{n+1} & \geq -y_1, -y_2, \dots, -y_n \\
y_{n+2} & \geq -y_1, -y_2, \dots, -y_n \\
\vdots & \vdots & \vdots \\
y_{2n} & \geq -y_1, -y_2, \dots, -y_n
\end{cases}
\Rightarrow
\begin{cases}
y_{n+1} & \geq -y_{min} \\
y_{n+2} & \geq -y_{min} \\
\vdots & \vdots \\
y_{2n} & \geq -y_{min}
\end{cases}$$
(12)

Assertion 1 aligns precisely with our objective. To affirm Assertion 1, we need to disprove Assertion 2. This can be achieved through a proof by contradiction. Assume Assertion 2 is true: there exists a y with size  $2n \times 1$  such that  $\mathbb{A}^T y \geq 0$  and  $\mathbf{b}^T y < 0$ . By applying  $\mathbb{A}^T y \geq 0$ , we derive the inequalities as shown in Eqn. 11.

Assume  $y_{min} = min(y_1, y_2, ..., y_n)$ , we have Eqn. 12. Then we calculate the value of  $\Delta \mathbf{b}^\mathsf{T} \mathbf{y}$ .

<span id="page-24-3"></span>
$$\Delta \mathbf{b}^{\mathsf{T}} \mathbf{y} = \Delta b_{1} y_{1} + \dots + \Delta b_{n} y_{n} + \Delta b_{n+1} y_{n+1} + \dots + \Delta b_{2n} y_{2n} \geq \Delta b_{1} y_{min} + \dots + \Delta b_{n} y_{min} + \Delta b_{n+1} (-y_{min}) + \dots + \Delta b_{2n} (-y_{min}) = y_{min} ((\Delta b_{1} + \dots \Delta b_{n}) - (\Delta b_{n+1} + \dots + \Delta b_{2n})) = 0$$

$$(13)$$

From Eqn. 13, we know  $\Delta \mathbf{b}^T \mathbf{y} \geq 0$ . This contradicts  $\Delta \mathbf{b}^T \mathbf{y} < 0$  in Assertion 2, proving that Assertion 2 is incorrect. As a result, we can establish the existence of a non-negative solution  $\mathbf{x}$ , and this, in turn, confirms the presence of a non-negative matrix  $\mathbb{X}$ .

#### <span id="page-24-0"></span>**B** PROOF OF THEOREM 5.2

Theorem 5.2 demonstrates that the minimum communication time with traffic matrix  $\mathbb{D}$  is  $b_{max} = max(\sum_{j=1}^{n} d_{ij}/B_i, \sum_{i=1}^{n} d_{ij}/B_i)$ , where  $d_{ij}$  is the element located at row i and column j in  $\mathbb{D}$ ,  $B_i$  is the bandwidth of GPU i.

Similar to the proof of Theorem 4.2, the approach unfolds in three steps. Initially, we convert the traffic matrix  $\mathbb D$  into  $\mathbb D'$  using matrix  $\mathbb X$ . Next, we demonstrate that the minimum communication time for  $\mathbb D'$  is  $b_{max}$ . Finally, we prove the existence of a non-negative  $\mathbb X$ . The key difference is that the network bandwidth B cannot be simplified to 1 in a heterogeneous environment. This distinction must be incorporated into the proof.

# 1. Convert $\mathbb{D}$ to $\mathbb{D}'$ by adding non-negative $\mathbb{X}$

<span id="page-24-4"></span>
$$\begin{bmatrix} \frac{d_{11}}{B_1} & \frac{d_{12}}{B_1} & \cdots & \frac{d_{1n}}{B_1} \\ \frac{d_{21}}{B_2} & \frac{d_{22}}{B_2} & \cdots & \frac{d_{2n}}{B_2} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{d_{n1}}{B_n} & \frac{d_{n2}}{B_n} & \cdots & \frac{d_{nn}}{B_n} \end{bmatrix} b_1 \\ b_2 \\ \vdots & \vdots & \ddots & \vdots \\ \frac{d_{n1}}{B_n} & \frac{d_{n2}}{B_n} & \cdots & \frac{d_{nn}}{B_n} \end{bmatrix} b_1 \\ b_2 \\ \vdots & \vdots & \ddots & \vdots \\ \frac{x_{n1}}{B_n} & \frac{x_{n2}}{B_n} & \cdots & \frac{x_{nn}}{B_n} \end{bmatrix} \Delta b_1 \\ b_2 \\ \vdots & \vdots & \ddots & \vdots \\ \frac{x_{nn}}{B_n} & \frac{x_{n2}}{B_n} & \cdots & \frac{x_{nn}}{B_n} \end{bmatrix} \Delta b_1 \\ b_2 \\ \vdots & \vdots & \ddots & \vdots \\ \frac{x_{nn}}{B_n} & \frac{x_{n2}}{B_n} & \cdots & \frac{x_{nn}}{B_n} \end{bmatrix} \Delta b_1 \\ b_2 \\ \vdots \\ b_n \\ b_{nn} \end{bmatrix} \Delta b_1 \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\ b_{nn} \\$$

To address the difference, we modify the element  $d_{ij}$  in  $\mathbb{D}$  to  $d_{ij}/\min(B_i, B_j)$ , as indicated in Eqn. 14. We apply the same adjustment to the elements in  $\mathbb{X}$  and  $\mathbb{D}'$ . For clarity and consistency, we continue to refer to these updated matrices as  $\mathbb{D}$ ,  $\mathbb{X}$ , and  $\mathbb{D}'$ .

Eqn. 14 illustrates the relationship  $\mathbb{D} + \mathbb{X} = \mathbb{D}'$ , where  $x_{ij}/B_i$  and  $d'_{ij}/B_i$  denote elements located at row i and column j in  $\mathbb{X}$  and  $\mathbb{D}'$ , respectively. Values external to the matrices, such as  $b_1$ ,  $\Delta b_1$ , and  $b_{max}$ , represent the sum of their corresponding columns or rows. For traffic matrix  $\mathbb{D}'$ , it satisfies the conditions that for each row  $\sum_{j=1}^n d'_{ij}/B_i = b_{max}$ , and for each column  $\sum_{i=1}^n d'_{ij}/B_i = b_{max}$ . That is to say, the time each GPU uses for sending and receiving is precisely  $b_{max}$  after adding artificial matrix  $\mathbb{X}$ .

#### 2. Prove the minimum communication time for $\mathbb{D}'$ is $b_{max}$

To establish that the minimum communication time for  $\mathbb{D}'$  is  $b_{max}$ , we must demonstrate that each GPU can continuously send and receive traffic until all traffic is completed. Similar to Theorem 4.2, each token requires a time slot to transmit at full bandwidth. Due to bandwidth constraints, each GPU can only send and receive one token per slot. To prove this, assume by contradiction that any GPU does not receive a token in a given slot. This failure means it won't meet the  $b_{max}$  requirement, contradicting the need for each GPU to receive  $b_{max}$  traffic. Using the same method as in Theorem 4.2, we confirm uninterrupted traffic exchange, thereby proving that  $b_{max}$  is indeed the minimum communication time for  $\mathbb{D}'$ .

#### 3. Prove the existence of non-negative X

In this step, we need to prove the presence of a non-negative matrix  $\mathbb{X}$ . This step is exactly the same as the one in Appx. A. We can still use Farkas' Lemma [1] to prove the existence of a non-negative solution  $\mathbf{x}$ .

These three steps validate the correctness of Theorem 5.2.