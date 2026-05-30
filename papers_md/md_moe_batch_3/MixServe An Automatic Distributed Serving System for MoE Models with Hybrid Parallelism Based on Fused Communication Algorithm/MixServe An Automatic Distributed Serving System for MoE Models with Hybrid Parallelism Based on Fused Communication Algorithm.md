# MixServe: An Automatic Distributed Serving System for MoE Models with Hybrid Parallelism Based on Fused Communication Algorithm

Bowen Zhou<sup>1</sup> , Jinrui Jia<sup>1</sup> , Wenhao He<sup>1</sup> , Yong Zhang<sup>2</sup> , Fang Dong1\* <sup>1</sup>School of Computer Science and Engineering, Southeast University, Nanjing, China <sup>2</sup>Department of Computer Science and Engineering, The Chinese University of Hong Kong, Hong Kong, China Emails: {bwzhou, jrjia, wenhao he}@seu.edu.cn, zhangyong@link.cuhk.edu.hk, fdong@seu.edu.cn

*Abstract***—The Mixture of Experts (MoE) models are emerging as the latest paradigm for Large Language Models (LLMs). However, due to memory constraints, MoE models with billions or even trillions of parameters can only be deployed in multi-GPU or even multi-node & multi-GPU based serving systems. Thus, communication has became a major bottleneck in distributed serving systems, especially inter-node communication. Contemporary distributed MoE models are primarily implemented using all-reduce (AR) based tensor parallelism (TP) and all-to-all (A2A) based expert parallelism (EP). However, TP generally exhibits low inter-node efficiency and is thus confined to high-speed intra-node bandwidth. In contrast, EP tends to suffer from load imbalance, especially when the parallel degree is high.**

**In this work, we introduce MixServe, a novel automatic distributed serving system for efficient deployment of MoE models by a novel TP-EP hybrid parallelism based on fused AR-A2A communication algorithm. MixServe begins by evaluating the communication overhead associated with various parallel strategies, taking into account the model hyperparameters and the configurations of network and hardware resources, and then automatically selects the most efficient parallel strategy. Then, we propose the TP-EP hybrid parallelism based on fused AR-A2A communication algorithm that overlaps intra-node AR communication and inter-node A2A communication. Extensive experiments on DeepSeek-R1 and Qwen3 models demonstrate that MixServe achieves superior inference performance, with 1.08** × ∼ **3.80** × **acceleration in time to first token (TTFT), 1.03** × ∼ **1.66** × **acceleration in inter-token latency (ITL), and 5.2%** ∼ **50.3% throughput improvement compared to existing approaches.**

*Index Terms***—Mixture of Experts (MoE); Distributed Serving Systems; Tensor Parallelism (TP); Expert Parallelism (EP); Hybrid Parallelism; Fused Communication Algorithm**

# I. Introduction

Mixture of Experts (MoE) models (*e.g.*, DeepSeek-V3 [1], DeepSeek-R1 [2], Qwen3 [3]) are emerging as the latest paradigm for Large Language Models (LLMs). Benifiting from the sparse activation mechanism, MoE models can scale up the model capacity to tens of billions, hundreds of billions, or even trillions of parameters, so that they can achieve competitive performance while maintaining inference-time computational efficiency. By leveraging a routing mechanism that activates only a subset of model parameters (experts) for each input token, MoE models achieve competitive accuracy while maintaining inference-time computational efficiency. This sparse activation strategy enables MoE models to reach parameter counts in the tens of billions, hundreds of billions, or even trillions while keeping the activated parameters manageable.

However, the increasing scale and complexity of MoE models pose significant challenges for achieving efficient inference in real-world deployment scenarios. Due to their massive parameter counts, these models can only be deployed in multi-GPU or even multi-node & multi-GPU distributed environments, where communication becomes a critical performance bottleneck, especially inter-node communication. Although intra-node highspeed interconnect protocols (*e.g.*, NVLink) offer substantial bandwidth advantages, inter-node connections typically rely on protocols like InfiniBand or RoCE<sup>1</sup> that provide bandwidth significantly lower than intra-node interconnects. The disparity in bandwidth necessitates that high-frequency or extensive internode communication must be minimized during the design of communication algorithms, as this has emerged as a significant bottleneck in distributed MoE model service systems.

Contemporary distributed MoE implementations primarily rely on two main parallelism strategies for handling the computational and communication demands: tensor parallelism (TP) using all-reduce (AR) operators, and expert parallelism (EP) using all-to-all (A2A) operators. TP distributes model parameters across multiple devices and requires frequent synchronization through AR operators, which typically performs well within a single node due to high-speed intra-node interconnects but scales poorly across nodes due to limited inter-node bandwidth. In contrast, EP distributes different experts across multiple devices and uses A2A operators to route tokens to their assigned experts, offering better inter-node scaling potential but often suffering from load imbalance and communication inefficiencies, especially when the parallel degree is high.

Building on these observations, current deployment strategies for MoE models typically adopt a hybrid approach: utilizing TP for the Attention blocks and EP for the MoE blocks, utilizing pipeline parallelism (PP) to shard the Decoders [4]. This division

<sup>\*</sup>Corresponding author.

is motivated by the fact that Attention blocks generally contain fewer parameters than MoE blocks, making them suitable for TP with manageable communication overhead. Meanwhile, data parallelism (DP) can be effectively implemented within the Attention blocks to improve overall system throughput. For instance, in the DeepSeek-V3 technical report [1], the authors advocate that MoE blocks should fully adopt EP to ensure that each expert can process sufficiently large batch sizes, thereby maximizing computational efficiency. However, existing parallel strategies suffer from the following critical limitations:

Lack of Systematic Theoretical Analysis. Existing parallel strategies are primarily based on empirical intuition and practical experience, lacking comprehensive theoretical analysis and systematic validation. These approaches fail to adequately consider the complex interplay between model hyperparameters, network topology, and hardware resource configurations.

Ineffective Exploitation of Communication Bandwidth Hierarchies. Most critically, current approaches do not effectively exploit the significant bandwidth disparities that exist between intra-node and inter-node communication channels. Current communication operators treat all communication uniformly, missing opportunities to optimize performance by leveraging the heterogeneous nature of distributed computing environments.

To address these fundamental communication and scalability challenges, we introduce MixServe, a novel automatic distributed serving system designed specifically for efficient deployment of MoE models. MixServe employs a systematic approach that begins by comprehensively evaluating the communication overhead associated with various parallel strategies, taking into account not only the hyperparameters of the MoE model but also the detailed configuration of network topology and hardware resources. Based on this analysis, MixServe automatically selects the most efficient parallel strategy for each specific deployment scenario.

The core innovation of MixServe lies in its TP-EP hybrid parallelism, implemented through a sophisticated fused AR-A2A communication algorithm. This algorithm strategically overlaps intra-node AR communication with inter-node A2A communication, effectively exploiting the bandwidth hierarchy present in modern distributed systems. By carefully orchestrating these overlapping communication patterns, MixServe minimizes overall communication latency while maintaining computational correctness. Our comprehensive evaluation on large-scale MoE models, including DeepSeek-R1 and Qwen3, demonstrates that MixServe consistently achieves superior inference performance compared to state-of-the-art approaches. Specifically, MixServe delivers  $1.08 \times \sim 3.80 \times \text{acceleration}$ in time to first token (TTFT),  $1.03 \times \sim 1.66 \times$  acceleration in inter-token latency (ITL), and  $5.2\% \sim 50.3\%$  improvement in overall throughput. By bridging the gap between system-level efficiency and model scalability, MixServe enables practical, high-performance deployment of MoE models in real-world distributed serving environments. Our contributions can be summarized as follows:

• Automatic Serving System: We present MixServe, a

![](_page_1_Figure_6.jpeg)

Fig. 1: Tensor Parallelism (TP) and All Reduce (AR) operators.

![](_page_1_Figure_8.jpeg)

Fig. 2: Expert Parallelism (EP) and All To All (A2A) operators.

novel automatic distributed serving system that systematically evaluates communication overhead and automatically selects optimal parallel strategies based on model hyperparameters and network configurations, replacing empirical intuition with rigorous analysis.

- Theoretical Communication Analysis: We conduct comprehensive theoretical analysis of communication overhead for distributed MoE serving, deriving rigorous models for TP, PP, EP, and DP strategies that enable informed strategy selection based on hardware characteristics and network bandwidth hierarchies.
- Fused AR-A2A Communication Algorithm: We propose a novel TP-EP hybrid parallelism with fused AR-A2A communication algorithm that efficiently overlaps intra-node AR communication with inter-node A2A communication, significantly reducing overall communication latency.
- Performance Evaluation on Mainstream MoEs: We demonstrate substantial performance improvements on DeepSeek-R1 and Qwen3 models: 1.08× ~ 3.80× TTFT acceleration, 1.03× ~ 1.66× ITL acceleration, and 5.2% ~ 50.3% throughput improvement compared to existing approaches.

#### II. BACKGROUND & MOTIVATION

# A. Hybrid Parallelism for MoE Models

The core component of the MoE model primarily utilizes a hybrid TP and EP parallelism strategy. TP refers to the partitioning of model parameters across multiple nodes in a distributed system, initially proposed in Megatron-LM [5]. Each node holds a replica of a subset of the model parameters, and they are updated independently using AR. AR aggregates gradients

![](_page_2_Figure_0.jpeg)

Fig. 3: Communication overhead of AR and A2A operators. The left subfigure shows the communication overhead for DeepSeek-R1 [2] and Qwen3 [3] models with different parallel degrees, while the right subfigure presents the results for intra-node and inter-node communication with different data sizes.

from all nodes and updates the parameters in a synchronous manner. This approach enables efficient training of large-scale models by distributing the workload across multiple nodes. Fig. 1a illustrates the execution process of TP, which sums up the hidden states after  $W^{\text{in}}$  and  $W^{\text{out}}$  projection. Fig. 1b shows the execution process of AR. It is usually divided into two operators: (1) Reduce Scatter (RS), (2) All Gather (AG), because it has lower communication overhead compared to direct implementations. In a Decoder-structured LLM, TP result synchronization is performed once during both the forward and backward propagation phases of the Attention and Feed-Forward Network (FFN) blocks. As a result, TP is generally unsuitable for inter-node implementation and is typically executed within a single node equipped with high-speed interconnect protocols.

EP refers to the partitioning of model computation across multiple experts in a MoE model, first explicitly proposed in Switch Transformer [6]. Each expert holds a replica of a subset of the model computation, and they are updated independently using A2A operators. A2A exchanges gradients between nodes and updates the parameters in an asynchronous manner. This approach enables efficient inference of large-scale MoE models by distributing the computation across multiple experts. In the DeepSeek-V3 technical report [1], researchers assert that the MoE block should fully adopt EP to ensure that each expert can process sufficiently large batch sizes. Fig. 2a illustrates the execution process of EP, where each token is sent to the device hosting the corresponding expert (i.e. Dispatch). Upon completing inference through the MLP layer, the results are returned along the original route (i.e. Combine). Fig. 2b shows the execution process of A2A. A2A communication can be implemented through various algorithms, among which Ring and Pairwise are commonly used. Both require N-1 rounds (N denotes the total number of participating devices) to complete the transmission and reception of all data.

# B. Challenges of Inter-node Communication

The deployment of large-scale MoE models for inference introduces significant system-level challenges. As the model parameters and experts increases, multi-GPU or even multi-node & multi-GPU deployment is required to meet the computational demands. However, the communication overhead in distributed environments quickly becomes the dominant bottleneck. Fig. 3

![](_page_2_Figure_6.jpeg)

Fig. 4: Gantt chart of the comparison between EP and TP+EP in a single MoE block, calculated from profiling data of DeepSeek-R1 in a 4-node cluster with 8 NPUs each.

illustrates the communication overhead of AR and A2A operators, which is obtained in a cluster of 4 Atlas 800T A2 servers with 8 NPUs each. Initially, we collected the profiling data of the DeepSeek-R1 [2] and Qwen3-235B-A22B [3] models under two configurations of the MoE block, specifically using TP and EP. Subsequently, we extracted the latency measurements for the two operators, AR and A2A, as shown in the left subfigure of Fig. 3. We denote d as the degree of parallelism for different parallel strategies, referring to the total number of devices participating in the communication. When  $d \leq 8$ , it signifies that communication is conducted entirely within the node. Experimental results indicate that the communication overhead within the node remains low, which can be attributed to the dedicated communication links, established between each pair of NPUs within the node. However, when d > 8, there is a significant increase in communication overhead, which is closely related to the network topology between nodes. Although, in an optimal scenario, each NPU can interconnect with the corresponding rank of NPUs in other nodes via RoCE, the bandwidth is typically several times lower than that of HCCS<sup>2</sup>. This leads to the AR-based TP generally failing to scale effectively across multiple nodes, especially TP is worse than EP when d = 32. The A2A-based EP is limited by the inter-node bandwidth; however, it can be integrated with DP to enhance system throughput. Therefore, these complex and difficult-tooptimize parallel strategies present significant challenges for serving MoE models due to issues related to network topology and bandwidth.

The right subfigure of Fig. 3 presents the results for intranode and inter-node communication with different data sizes. Intra-node communication is based on 4 NPUs within a node, while inter-node communication is based on a single NPU per node across 4 nodes. The results indicate that as the data volume increases, the communication operators within each set initially remain at a relatively low level, after which an inflection point is reached, leading to linear growth. The distinction is that, due to more intra-node bandwidth than inter-node bandwidth, the onset of this inflection point occurs relatively later. Therefore, both in terms of the parallel degree and data size, communication overhead—especially that between nodes—represents a significant bottleneck and a key challenge for the distributed MoE serving system.

<sup>&</sup>lt;sup>2</sup>Huawei Cache Coherence System

# C. Opportunities of Decoupling Intra-node Communication & Inter-node Communication

We adopt the parallel strategies outlined in the DeepSeek-V3 technical report [1], in which the MoE blocks fully employ EP while retaining the parallel strategy for the Attention blocks. This approach decouples the original EP communication group into intra-node TP groups and inter-node EP groups. We collected the profiling data of the MoE block from a layer of the Decoder in DeepSeek-R1. The results of the Gantt chart in Fig. 4 indicate that while the TP within the nodes introduced AR, it significantly assisted the EP component in sharing a substantial portion of the communication, leading to a significant reduction in the communication overhead of the EP group. Preliminary experimental results indicate that decoupling intra-node and inter-node communication allows for further optimization of communication overhead.

#### III. System Design

We introduce MixServe, a novel automatic distributed serving system that enables efficient deployment of MoE models by TP-EP hybrid parallelism based on a fused AR-A2A communication algorithm. MixServe automatically selects the optimal parallel strategy based on model parameters and network configurations.

#### A. System Overview

Fig. 5 illustrates the system overview of MixServe, which operates in two stages: (i) offline, (ii) online. During the offline stage, MixServe determines the optimal parallelism strategy based on the model's hyperparameters and the configuration of network and hardware resources. During the online stage, MixServe automatically loads and partitions the model weights according to the results of the parallelism strategy analyzed during the offline phase. Additionally, it injects collective communication operators into the model's forward method through the mixed parallel communication groups.

Offline Stage: MixServe first retrieves the model's hyperparameters and presets prompts with varying batch sizes and sequence lengths to obtain profiling data as observations. Subsequently, it uses the configuration of network and hardware resources as input, which includes computational power, as well as intra-node and inter-node network bandwidth and topology, to calculate theoretical values. Both the observations and theoretical values are then input into the analyzer to derive the optimal parallelism strategy. This will provide critical input for the weight loader and partitioner in the online phase.

Online Stage: Based on the optimal parallelism strategy derived from the offline stage, MixServe employs the weight loader to load the corresponding model weight shards through the partitioner. Subsequently, when MixServe initiates the serving service, it initializes the mixed parallel communication group and injects collective communication operators into the appropriate forward method of the MoE models. The serving service manages memory and schedules requests based on the leading vLLM [7] system currently available in the industry.

![](_page_3_Figure_8.jpeg)

Fig. 5: MixServe system overview.

# B. Automatic Analyzer

- 1) Definition of Parallel Strategies: First of all, in order to facilitate a comprehensive investigation of various parallel strategies, we define a set of context-free grammars to represent the parallel strategies employed by a single Decoder Layer, as follows:
  - 1)  $strategy \longrightarrow Decoder \mid Decoder \mid PP = degree \mid$
  - 2)  $Decoder \longrightarrow Attention, MoE$
  - 3) Attention  $\longrightarrow$  block
  - 4)  $MoE \longrightarrow block$
  - 5)  $block \longrightarrow intra-node + inter-node \mid parallel$
  - 6)  $intra-node \longrightarrow parallel$
  - 7)  $inter-node \longrightarrow parallel$
  - 8)  $parallel \longrightarrow TP \mid EP (DP) = degree$
  - 9)  $degree \longrightarrow 2^k (k \in \mathbb{N})$

The above definition indicates that for each layer of the MoE, the Attention block and the MoE can adopt different parallel strategies. Specifically, the Attention block may utilize TP and DP, while the MoE block may employ TP and EP. It is important to note that we have introduced an additional constraint on the MoE block, namely that DP is typically not considered. This is due to the fact that each expert in the MoE model functions as an independent Multi-Layer Perceptron (MLP), making EP essentially equivalent to DP among the experts. Furthermore, each block may implement different parallel strategies based on the distinct network topologies present both within and between nodes. Furthermore, PP should be applied exclusively between the layers of the Decoder. The previously defined parallelism strategies are confined to a single layer of the Decoder so that they are orthogonal and complementary. The optimal parallel strategy defined herein is the output generated by the automatic analyzer. For example, according to the details presented in the DeepSeek-V3 technical report [1], the parallelism strategy for the prefill phase is TP=4 + DP=8, EP=32.

2) Analysis of Collective Communication Operators: First of all, we conduct a fine-grained analysis of the additional communication overhead resulting from the use of various parallel strategies. Without considering PP, we focus on a single layer of the Decoder in the MoE model.

AR: According to the principles of block matrix multiplication, after each rank computes its respective results, it is

TABLE I: Overhead of collective communication operators.

| Block     | Strategy | Collective<br>Communication |                     | Communication per Round    | Algorithm | Rounds of<br>Communication | Communication<br>Domain     |
|-----------|----------|-----------------------------|---------------------|----------------------------|-----------|----------------------------|-----------------------------|
| Attention | TP       | AR                          | RS<br>AG            | $O(bs \cdot \frac{h}{d})$  | Broadcast | 1                          | Intra-node                  |
| МоЕ       | TP       | AR                          | RS<br>AG            | $O(bs \cdot \frac{h}{d})$  | Broadcast | 1                          | Intra-node                  |
|           | EP       | A2A                         | Dispatch<br>Combine | $O(\frac{bs}{d} \cdot hk)$ | Pairwise  | d - 1                      | Intra-node or<br>Inter-node |

![](_page_4_Picture_2.jpeg)

Fig. 6: Trade-off between DP and EP during A2A communication. (a)  $d_{\rm DP}=d_{\rm EP}$  (Example: DP=4, EP=4). (b)  $d_{\rm DP}>d_{\rm EP}$  (Example: DP=4, EP=2). (c)  $d_{\rm DP}< d_{\rm EP}$  (Example: DP=2, EP=4). The yellow cross marks represent the redundancy parts to drop out.

necessary to sum these results to guarantee the accuracy of the final outcome. This requires performing an AR communication among the ranks. Although the dimension of the tensor  $X \in \mathbb{R}^{b \times s \times h}$ , the AR collective communication operator can be decomposed into RS and AG to reduce the communication volume. According to Table. I, the communication volume is  $O(bs \cdot \frac{h}{d})$  per round based on Broadcast algorithm. Based on the communication capabilities of modern computational nodes, this process entails full-duplex communication between pairs of devices, which can be accomplished in a single round. Therefore, the overhead of AR communication is theoretically as follows:

$$RS(size, degree) = AG(size, degree) \propto \frac{size}{degree}$$
 (1)

$$AR(size, degree) = RS(\frac{size}{degree}, degree) + AG(\frac{size}{degree}, degree)$$
(2)

**A2A**: According to the routing mechanism of the MoE models, each token is assigned to its corresponding activated expert for inference. This requires the use of A2A communication to exchange the hidden states among the experts. According to Table I, the communication volume is  $O(\frac{bs}{d} \cdot hk)$  per round based on Pairwise algorithm, where k denotes the top-k experts

activated. The Pairwise algorithm requires d-1 rounds to complete the communication process. Therefore, the overhead of A2A communication is theoretically as follows:

A2A(size, degree) 
$$\propto \frac{\text{size}}{\text{degree}} \times (\text{degree} - 1)$$
 (3)

- 3) Trade-off between DP and EP: Based on the formal definition of parallel strategies presented in §III-B1, an important issue is the integration of DP from the Attention block and EP from the MoE block. Based on the relationships between the parallel degrees, three distinct cases can be identified as follows:
  - $d_{\rm DP} = d_{\rm EP}$ : This case represents the most balanced and easily implementable situation, in which the DP rank of the Attention block corresponds one-to-one with the EP rank of the MoE block. In this case, all devices within the communication group engage in A2A communication, as shown in Fig. 6a.
  - $d_{\rm DP} > d_{\rm EP}$ : A smaller  $d_{\rm EP}$  results in redundancy in expert weights, incurring additional memory overhead to enhance DP degree and consequently improve system throughput. In this case, a total of  $\frac{d_{\rm DP}}{d_{\rm EP}}$  communication groups perform A2A communication in parallel, with each group comprising  $d_{\rm EP}$  devices, as shown in Fig. 6b.
  - $d_{\mathrm{DP}} < d_{\mathrm{EP}}$ : A smaller  $d_{\mathrm{DP}}$  results in redundancy in hidden states and lower throughput; however, this is mitigated by an effective dropping strategy that reduces communication overhead. In this case, a total of  $\frac{d_{\mathrm{EP}}}{d_{\mathrm{DP}}}$  communication groups perform A2A communication in parallel, with each group comprising  $d_{\mathrm{DP}}$  devices, as shown in Fig. 6c.

Based on the aforementioned analysis, MixServe will automatically manage trade-offs between DP and EP, considering the specified latency and throughput requirements while adhering to memory constraints.

4) Token Generation Latency: Aside from embedding and sampling, the latency associated with generating a single token consists of three components: computational latency, communication latency, and an additional queuing latency induced by request contention at the serving system.

**Computational Latency**: MixServe analyzes and predicts computational cost under hybrid parallelism. The computational latency of each rank can be expressed as follows:

$$\tau(d_{\text{TP}}, d_{\text{EP}}, d_{\text{DP}}) \propto \frac{\Psi}{d_{\text{TP}} \cdot d_{\text{EP}}} \cdot \frac{b}{d_{\text{DP}}} \cdot sh$$
(4)

**Communication Latency**: According to the analysis in §III-B2 and §III-B3, the communication latency of each rank can be expressed as follows:

 $<sup>^{3}</sup>b$ : batch size, s: sequence length, h: hidden dimension.

![](_page_5_Figure_0.jpeg)

Fig. 7: An example of TP-EP hybrid parallelism. Assume that we have a 2-node cluster (*i.e.*  $n_{\text{node}} = d_{\text{DP}} = 2$ ) with 4 NPUs each (*i.e.*  $n_{\text{proc}} = d_{\text{TP}} = 4$ ). (a) MoE weights partition. For Attention blocks, model weights are partitioned by intra-node TP and by inter-node DP. For MoE blocks, model weights are partitioned by intra-node TP and by inter-node EP. (b) Distributed MoE inference. Activations are partitioned by inter-node DP with dimension of batch. At the output of MoE blocks, activations are communicated by RS, combine A2A and AG operators.

$$\begin{split} &\lambda(d_{\text{TP}}, d_{\text{EP}}, d_{\text{DP}}) \\ &= 2 \times \text{AR}(\frac{b}{d_{\text{DP}}} \cdot sh, d_{\text{TP}}) \\ &+ 2 \times \begin{cases} \text{A2A}(\frac{b}{d_{\text{DP}}} \cdot shk, d_{\text{EP}}) & \text{if } d_{\text{DP}} \ge d_{\text{EP}} \\ \text{A2A}(\frac{b}{d_{\text{EP}}} \cdot shk, d_{\text{DP}}) & \text{else} \end{cases} \end{split} \tag{5}$$

Notice that when  $d_{\rm DP} < d_{\rm EP}$ , the hidden states exhibits  $\frac{d_{\rm EP}}{d_{\rm DP}}$  times redundancy within DP groups, as shown in Fig. 6c illustrated in §III-B3. In Eq. (5), the batch size is specified as  $\frac{b}{d_{\rm EP}}$  because  $\frac{b}{d_{\rm EP}} = \frac{b}{d_{\rm DP}}/\frac{d_{\rm EP}}{d_{\rm DP}}$ .

Service Latency per Token: Combining computation and

**Service Latency per Token**: Combining computation and communication, the service latency for generating one token through an l-layer Decoder is

$$\Delta t_{\text{svc}} = l[\tau(d_{\text{TP}}, d_{\text{EP}}, d_{\text{DP}}) + \lambda(d_{\text{TP}}, d_{\text{EP}}, d_{\text{DP}})] + (d_{\text{PP}} - 1) \cdot P2P(\frac{b}{d_{\text{DP}}} \cdot sh)$$
(6)

Here, P2P denotes the point-to-point (P2P) communication latency induced by PP.

**Queuing Latency**: In practical online serving, token generation requests arrive stochastically and may queue before being served. We model the serving system as a queue with arrival rate  $\lambda_a$  (tokens per second) and service rate  $\mu = 1/\Delta t_{\rm syc}$ .

For analytical tractability, we adopt an M/M/1 approximation. Under the stability condition  $\rho = \lambda_a/\mu < 1$ , the expected queuing delay is given by

$$W_q = \frac{\rho}{\mu(1-\rho)} = \frac{\lambda_a}{\mu(\mu - \lambda_a)} \tag{7}$$

This term captures the contention-induced waiting time caused by concurrent requests, which becomes significant as system utilization approaches saturation.

**Constraints**: The constraints primarily stem from the limitations imposed by NPU memory, which encompasses two components: model weights and the K/V cache. Assuming that the maximum NPU memory for a single NPU is M, it is

approximated that the model weights  $\Psi$  are primarily composed of Attention blocks and MoE blocks, with a total of l layers in the Decoder.

$$\frac{\Psi_{\text{Attn}}}{d_{\text{TP}}} + \frac{\Psi_{\text{MoE}}}{d_{\text{EP}}d_{\text{TP}}} + 2bsh \cdot \frac{l}{d_{\text{PP}}} < M \tag{8}$$

5) Performance Indicators: In existing evaluations, performance indicators such as TTFT, ITL, and throughput are typically defined based on empirical measurements collected from the serving system. While these observed metrics faithfully reflect end-to-end behavior, they conflate architectural factors with workload-dependent effects. To complement empirical evaluation, we further introduce theoretically estimated performance indicators derived from the latency model in §III-B4, enabling principled analysis and optimization.

Time to First Token (TTFT): TTFT is defined as the time taken to generate the first token after receiving a request, which reflects the performance of the prefill stage. Under our queuing-aware latency model, TTFT consists of (i) the queuing delay experienced before service and (ii) the service latency required to generate the first token. Importantly, first-token generation differs from steady-state decoding in that the full prompt of length  $L_{\rm in}$  must be processed and the KV cache initialized. Therefore, the theoretically estimated TTFT is defined as:

$$TTFT = W_q + \Delta t_{\text{svc}}^{\text{prf}} = W_q + \Delta t_{\text{svc}} \Big|_{s=L_{\text{in}}}$$
 (9)

**Inter-Token Latency (ITL)**: ITL is defined as the average time interval between the generation of two consecutive tokens, which reflects the performance of the decode stage. In this phase, previously computed keys and values are reused via the KV cache, and each iteration only processes a single newly generated token. As a result, the theoretical ITL corresponds to the steady-state per-token service latency:

$$ITL = \Delta t_{\text{svc}}^{\text{dec}} = \Delta t_{\text{svc}} \Big|_{s=1}$$
 (10)

![](_page_6_Picture_0.jpeg)

Fig. 8: An example of fused AR-A2A communication algorithm. Assume that we have a 4-node cluster with 4 GPUs each. Steps 1-5 illustrates how hidden states are synchronized by intra-node TP and inter-node EP.

**Throughput:** We model throughput at the service level by jointly accounting for both the prefill stage and the steady-state decoding stage. For a request with input length  $L_{\rm in}$  and output length  $L_{\rm out}$ , the expected service time is:

$$\Theta = \frac{L_{\rm in} + L_{\rm out}}{W_q + \Delta t_{\rm svc}^{\rm prf} + L_{\rm out} \cdot \Delta t_{\rm svc}^{\rm dec}}$$
(11)

## C. Hybrid TP-EP Partitioner

1) Hybrid TP-EP Design: Fig. 7 illustrates a simple design of hybrid TP-EP. In this example, we assume that there is a 2-node cluster (i.e.  $n_{\text{node}} = d_{\text{DP}} = 2$ ) with 4 NPUs each (i.e.  $n_{\text{proc}} = d_{\text{TP}} = 4$ ). The Attention blocks are partitioned by intra-node TP and by inter-node DP, while the MoE blocks are partitioned by intra-node TP and by inter-node EP. Considering that the hybrid TP-EP involves communication between two distinct communication groups, MixServe has implemented fine-grained optimizations on the communication process. Specifically, MixServe decouples the AR of the TP group into RS and AG, and reorganizes the A2A of the EP group, ultimately forming the RS-A2A-AG communication process. Correspondingly, MixServe injects the specified parallel strategy into the model's weight loader through the partitioner.

2) Theoretical Analysis of Communication: Existing serving systems commonly utilize EP entirely within the MoE blocks, while TP is used in the Attention blocks in intra-node. Assuming a cluster of  $n_{\text{node}}$  nodes, each with  $n_{\text{proc}}$  NPUs, the parallel strategy is defined as TP =  $n_{\text{proc}}$  + DP =  $n_{\text{node}}$ , EP =  $n_{\text{node}} \cdot n_{\text{proc}}$ . Thus, communication overhead of a Docoder layer is as follows:

![](_page_6_Figure_7.jpeg)

(b) Fused AG-Dispatch Gantt Chart

Fig. 9: Gantt chart of fused AR-A2A communication algorithm. (a) Fused RS-Combine communication algorithm. (b) Fused AG-Dispatch communication algorithm. Both of them facilitate the overlapping of intra-node and inter-node communication.

$$\lambda_{\text{EP}} = \text{AR}(bsh, n_{\text{proc}}) + 2 \times \text{A2A}(bshk, n_{\text{pode}})$$
(12)

The hybrid TP-EP parallelism decouples and reorganizes AR and A2A, resulting in a reduction of the per-unit communication volume of the Combine stage, as well as a decrease in communication scale both intra-nodes and inter-nodes. Considering the impact of network topology and the interconnection methods between nodes and NPUs, pure TP or EP is often constrained by inter-node bandwidth, which prevents the optimization of the overall communication group's efficiency. When  $n_{\rm proc} = d_{\rm TP}$  and  $n_{\rm node} = d_{\rm EP}$ , the TP group and EP are precisely allocated to intra-nodes and inter-nodes. The parallel strategy of MixServe is defined as TP =  $n_{\rm proc}$  + DP =  $n_{\rm node}$ , TP =  $n_{\rm proc}$  + EP =  $n_{\rm node}$ . The communication overhead at this stage is as follows:

$$\lambda_{\text{mix}} = AR(bsh, n_{\text{proc}}) + AG(\frac{bshk}{n_{\text{proc}}}, n_{\text{proc}})$$

$$+ 2 \times A2A(\frac{bshk}{n_{\text{proc}}}, n_{\text{node}})$$
(13)

Certainly, MixServe is not restricted to merely identifying  $n_{\text{proc}} = d_{\text{TP}}$  and  $n_{\text{node}} = d_{\text{EP}}$  as the optimal parallel strategy; instead, it conducts theoretical analyses and predictions for all parallel strategies that satisfy  $n_{\text{proc}} \cdot n_{\text{node}} = d_{\text{TP}} \cdot d_{\text{EP}}$ . Following the completion of the parallel strategy decision for MoE blocks, the parallel strategy for Attention blocks will be optimized in conjunction with the theoretical analysis presented in §III-B3. Consequently, the partitioner will output the optimal parallel strategy.

# D. Fused AR-A2A Communication Algorithm

Building upon the hybrid TP-EP parallelism, we design the fused AR-A2A communication algorithm by employing the principle of mutual overlapping of intra-node and internode communication, guided by the computational dependency relationships.

Fig. 8 illustrates the overall process of the fused AR-A2A communication algorithm. Steps 1-5 demonstrate how hidden states are synchronized by intra-node TP and inter-node EP. Initially, each rank only possesses a partition of the hidden

#### **Algorithm 1** Fused RS-Combine Pairwise Communication

```
Require: An n-node cluster with m GPUs/NPUs per node; an
      input tensor X \in \mathbb{R}^{\frac{bs}{d_{\rm EP}} \times h} per node; global rank r
Ensure: An output tensor Y \in \mathbb{R}^{\frac{b}{d_{\mathrm{DP}}} \times s \times h} each node
  1: Y \leftarrow \text{empty}(\frac{b}{d_{\text{DP}}}, s, h)
2: [X_1, X_2, \cdots, X_m] \leftarrow \text{split}(X, m, -1) \rightarrow \text{Split } X \text{ into } m
      parts along the hidden dimension (the same below)
  3: [Y_1, Y_2, \cdots, Y_m] \leftarrow \text{split}(Y, m, -1)
  4: r_{\text{TP}} \leftarrow r \mod m
                                                                  ▶ Compute TP rank
  5: Initialize tensor list [S_1, S_2, \dots, S_n]
                                                           \triangleright Stage local tensor X_{r_{TP}}
  6: S_1 \leftarrow X_{r_{\text{TP}}}
  7: for i \leftarrow 1 to n - 1 do async
            r_{\text{to}} \leftarrow (r_{\text{TP}} + im) \mod mn
            isend(X_{r_{TP}}, r_{to}) \rightarrow Send X_{r_{TP}} to the same TP rank of
  9:
      the next i-step node asynchronously
            r_{\text{from}} \leftarrow (r_{\text{TP}} - im) \mod mn
            S_{i+1} \leftarrow \text{irecv}(r_{\text{from}}) \rightarrow \text{Receive } X_{r_{\text{TP}}} \text{ from the same}
      TP rank of the previous i-step node asynchronously
 12: end for
                              ▶ Inter-node A2A pairwise communication
 13: for i \leftarrow 1 to n do async
            S_i \leftarrow \mathbf{await} \text{ reduce } \mathbf{scatter}(S_i, \mathsf{TP} \mathsf{group})
            Y_i \leftarrow Y_i + \text{topk\_weights}(S_i)
 15:
```

states (the blue segment). Subsequently, intra-node AG/RS communication (the light green segment) and inter-node A2A communication (the orange segment) are performed to exchange the hidden states asynchronously step by step. Finally, each rank acquires the complete hidden states (the green segment in step 5) after the communication process concludes.

17:  $Y \leftarrow \text{all\_gather}(Y_{r_{\text{TP}}}, \text{TP group})$ 

▶ Intra-node AR communication

1) Fused RS-Combine Communication Algorithm: Fig. 9a and Alg. 1 illustrates the fused RS-Combine communication algorithm. The algorithm is designed to optimize the communication process by overlapping intra-node and inter-node communication. The key steps are as follows: (1) Intra-node RS, (2) Inter-node A2A, (3) Intra-node AG.

Initially, the hidden states at each rank within the node engage in one round of RS communication, temporarily storing the results after weighting them with the top-k weights. Concurrently, a round of communication between nodes is executed using the Pairwise algorithm, enabling each node to acquire the corresponding hidden states as input for the subsequent iteration. Upon completion of the Pairwise algorithm, all weighted results are ultimately combined through AG within the nodes. The Gantt chart in Fig. 9 illustrates the overlapping of communication processes, where the RS and A2A communication are executed concurrently, followed by the AG operation.

In summary, the algorithm necessitates  $n_{\text{node}} - 1$  rounds of communication between nodes and  $n_{\text{node}}$  rounds of communication within each node. The asynchronous mechanism facilitates overlapping communication both within and across nodes, resulting in a time complexity of  $O(n_{\text{node}})$ . Furthermore, the algorithm necessitates the allocation of additional temporary

# Algorithm 2 Fused AG-Dispatch Pairwise Communication

```
Require: An n-node cluster with m GPUs/NPUs per node; an
       input tensor X \in \mathbb{R}^{\frac{b}{d_{\mathrm{DP}}} \times s \times h} per node; global rank r
Ensure: An output tensor Y \in \mathbb{R}^{\frac{bs}{dEP} \times h} per node
  1: Y \leftarrow \text{empty}(\frac{bs}{d_{\text{EP}}}, h)
2: [X_1, X_2, \cdots, X_m] \leftarrow \text{split}(X, m, -1)
  3: [Y_{11}, Y_{12}, \dots, Y_{mn}] \leftarrow \text{split}(Y, [m, n], [0, 1]) \rightarrow \text{Split } Y
       into m \times n parts along the token and hidden dimension
  4: r_{\text{TP}} \leftarrow r \mod m
  5: [X_{r_{\text{TP}}1}, X_{r_{\text{TP}}2}, \cdots, X_{r_{\text{TP}}n}] \leftarrow \text{route}(X_{r_{\text{TP}}}) \triangleright \text{Calculate the}
       expert map on local TP rank only
  6: for i \leftarrow 1 to n - 1 do async
             r_{\text{to}} \leftarrow (r_{\text{TP}} + im) \mod mn
  8:
              isend(X_{r_{TP}i}, r_{to})
  9:
             r_{\text{from}} \leftarrow (r_{\text{TP}} - im) \mod mn
 10:
             Y_{r_{\text{TP}}r_{\text{from}}} \leftarrow \text{irecv}(r_{\text{from}})
 11: end for
 12: for i \leftarrow 1 to n do async
             Y_{:i} \leftarrow \mathbf{await} \text{ all } \mathsf{gather}(Y_{r_{\mathsf{TP}}i}, \mathsf{TP} \mathsf{group})
14: end for
```

storage space for each rank, corresponding in size to the output. Consequently, the space complexity is  $O(bsh \cdot n_{\rm proc})$  in total. The fused RS-Combine Pairwise communication algorithm we present ingeniously incorporates an overlapping mechanism for communication both within and between nodes, effectively trading off space for time. This approach significantly reduces the communication overhead associated with inference in the MoE models.

2) Fused AG-Dispatch Communication Algorithm: Similarly, it is precisely because the hidden states are replicated in the MoE TP group that they can be sharded within the MoE TP group. The sharding further minimizes the internode Dispatch communication overhead, requiring only the addition of extra intra-node AG communication, analogous to Megatron-based PP. On this basis, it is possible to allow for the overlapping of the intra-node AG communication and inter-node Dispatch communication, in a manner analogous to the Fused RS-Combine algorithm.

Fig. 9b shows the Gantt chart of the fused AG-Dispatch communication algorithm. Apart from the pairwise communication in the first round and the AG communication in the last round, the intra-node and inter-node communication during the remaining rounds can overlap with one another. Alg. 2 describes the detailed communication schedule. In contrast to Alg. 1, the total number of communication rounds both within and between nodes is  $n_{\text{node}} - 1$ , as the local shards in the TP group and EP group do not require communication. Therefore, the time complexity of the algorithm is  $O(n_{\text{node}})$  and the space complexity is O(1), respectively.

TABLE II: Configuration of parallel strategies of baselines.

| Baselines | Parallel Strategies                                  |                    |  |  |  |
|-----------|------------------------------------------------------|--------------------|--|--|--|
| Dascilles | H20                                                  | Ascend 910B        |  |  |  |
|           | TP=8 [PP=2]                                          | TP=8 [PP=4]        |  |  |  |
| vLLM      | TP=8 + DP=2, EP=16                                   | TP=8 + DP=4, EP=32 |  |  |  |
|           | TP=4 + DP=4, EP=16                                   | TP=4 + DP=8, EP=32 |  |  |  |
| Tutel     | TP=8 + DP=2, TP=8 + EP=2<br>TP=4 + DP=4, TP=4 + EP=4 | Not supported      |  |  |  |

#### IV. EVALUATION

#### A. Experimental Setup

**Hardware and Network**: We conduct our experiments on following clusters:

- A cluster of 2 servers with 8 Nvidia H20 GPUs (96 GB) each. The intra-node network is supported by NVLink 4.0 (up to 900 GB/s), while the inter-node network is connected via InfiniBand (400 Gbps).
- A cluster of 4 Atlas 800T A2 servers with 8 Ascend 910B NPUs (64 GB) each. The intra-node network is fullyconnected via HCCS (up to 480 Gbps), while the inter-node network is connected via RoCE (up to 200 Gbps).

**Implementation**: We implement MixServe based on several serving systems, including vLLM [7] (on the Ascend 910B cluster) and Tutel [8] (on the H20 cluster).

Models and Datasets: To evaluate MixServe, we adopt the following SOTA MoE models: (1) DeepSeek-R1 [2], a 671B-parameter MoE model with 256 routed experts and 1 shared expert, where 37B parameters are activated per token; and (2) Qwen3 [3], a 235B-parameter MoE model with 128 experts, with 22B parameters activated per token. We use ShareGPT-V3 [9] for benchmark evaluation, which is a large-scale dataset containing 1.2B tokens of human conversations.

**Baselines**: We compare MixServe with the following baselines: (1) vLLM [7], which utilizes hybrid TP+PP for LLM serving and hybrid DP+EP for distributed MoE model serving; and (2) Tutel [8], which employs hybrid TP+EP for distributed MoE model serving. In addition, we also set up different TP degrees (*i.e.* 4 and 8) for comparative experiments. The specific configurations of parallel strategies for baselines are summarized in Table II.

# B. Performance Evaluation

We established the range of request rates at 2, 4, and 8 requests per second (req/s), while also defining the maximum batch size as 16 and the maximum sequence length as 4096 tokens. Fig. 10 shows the performance evaluation of MixServe and baselines across different metrics.

TTFT: Fig. 10a illustrates that MixServe achieves significantly lower TTFT compared to baselines, indicating faster response times during the prefill stage. Specifically, MixServe achieves 1.08× ~ 3.80× acceleration in TTFT across different configurations and models. On the Ascend 910B cluster, MixServe demonstrates particularly impressive improvements: for DeepSeek-R1, it achieves 2.67× acceleration compared to vLLM TP+PP and 1.70× compared to vLLM DP+EP; for Qwen3-235B-A22B, it achieves 3.80× acceleration compared

![](_page_8_Figure_13.jpeg)

Fig. 10: Performance evaluation of MixServe and baselines. The results are averaged over 10 runs, with error bars representing the standard deviation.

to vLLM TP+PP and  $1.32 \times \sim 1.93 \times$  compared to vLLM DP+EP configurations. On the H20 cluster, MixServe achieves  $1.08 \times \sim 1.23 \times$  acceleration compared to various baselines. The experimental results demonstrate that: (1) the hybrid TP-EP parallelism proposed by MixServe effectively reduces TTFT across diverse hardware platforms and model architectures; (2) the overlapping communication between intra-nodes and internodes significantly reduces overall communication overhead, resulting in improved P99<sup>4</sup> performance for MixServe.

ITL: Fig. 10b shows that MixServe demonstrates lower ITL, indicating faster token generation during the decode stage. The

<sup>4</sup>P99 refers to the 99th percentile latency, which means 99% of requests are served within this time.

![](_page_9_Figure_0.jpeg)

Fig. 11: Performance comparison of MixServe with different DP and EP configurations.

hybrid TP-EP parallelism achieves  $1.03\times\sim1.66\times$  acceleration across all evaluated configurations. On the Ascend 910B cluster, MixServe reduces ITL from 227.33ms to 160.06ms (1.42× acceleration) for DeepSeek-R1 compared to vLLM TP+PP, and from 134.27ms to 81.1ms (1.66× acceleration) for Qwen3-235B-A22B. On the H20 cluster, MixServe achieves  $1.03\times\sim1.16\times$  acceleration compared to various baselines. Although the acceleration effect is less pronounced than TTFT due to the smaller communication volume in the decode stage, the consistent improvements demonstrate the effectiveness of the fused AR-A2A communication algorithm.

Throughput: Fig. 10c illustrates that MixServe achieves substantially higher throughput compared to baselines, allowing it to handle more requests simultaneously and improve overall system efficiency. The total token throughput improvements range from 5.2% to 50.3% across different configurations. On the Ascend 910B cluster, MixServe achieves 22.0% throughput improvement (from 100.61 to 122.72 tokens/s) for DeepSeek-R1 and 32.2% improvement (from 113.52 to 150.08 tokens/s) for Owen3-235B-A22B compared to vLLM TP+PP. On the H20 cluster, the improvements are even more substantial: 50.3% for DeepSeek-R1 (from 362.78 to 545.23 tokens/s) and 43.5% for Qwen3-235B-A22B (from 435.82 to 625.45 tokens/s) compared to vLLM TP+PP. When compared to other EP-based approaches, MixServe consistently achieves  $6.8\% \sim 24.5\%$  throughput improvements, demonstrating the effectiveness of the automatic parallel strategy selection and fused communication algorithm.

# C. Ablation Studies

To better understand the impact of different components in MixServe, we conduct ablation studies by systematically removing or modifying key features.

1) Trade-off between DP and EP: As §III-B3 describes, MixServe optimizes  $d_{\rm DP}$  and  $d_{\rm EP}$  by evaluating the modeled communication and computation costs across feasible TP/DP/EP tuples. We study three representative settings: (1)  $d_{\rm DP} = d_{\rm EP}$  (TP=8 + DP=4, TP=8 + EP=4), (2)  $d_{\rm DP} > d_{\rm EP}$ 

![](_page_9_Figure_7.jpeg)

Fig. 12: Impact of overlapping communication based on fused AR-A2A communication algorithm in MixServe. We evaluate that on the Ascend 910B cluster with DeepSeek-R1. (a) Gantt chart of Sync and Async communication. (b) Performance comparison of Sync and Async communication.

(TP=4 + DP=8, TP=8 + EP=4), and (3)  $d_{\rm DP} < d_{\rm EP}$  (TP=8 + DP=4, TP=4 + EP=8).

Fig. 11 summarizes the ablation results. On Ascend 910B, the balanced case attains the best latency/throughput for both DeepSeek-R1 and Qwen3 (e.g., 383.14ms TTFT and 150.08 tokens/s throughput for Qwen3 when  $d_{DP} = d_{EP}$ ), while skewing towards larger DP or larger EP degrades performance. However, on Nvidia H20, a different ordering holds:  $d_{DP} < d_{EP}$  yields the lowest TTFT (e.g., 228.99ms for Qwen3) and the highest throughput (40.00 tokens/s). These observations align with our analytical trade-off model-balancing DP and EP minimizes the dominant communication term—so the partitioner automatically selects this configuration under both high-bandwidth NVLink (H20) and RoCE/HCCS (910B) environments. When cluster bandwidth or node count changes, MixServe re-evaluates the cost model and picks the best feasible tuple, ensuring the serving system adapts its parallel strategy to the available network and compute resources.

2) Impact of Overlapping Communication: As §III-D describes, MixServe employs a fused AR-A2A communication algorithm to optimize the communication process. We evaluate the impact of this optimization on performance by whether asynchronous or synchronous communication is used.

Fig. 12a shows the Gantt chart of synchronous and asynchronous communication, where the asynchronous communication allows for overlapping of intra-node and inter-node communication. Specifically, the fused AG-Dispatch communication algorithm overlaps inter-node Dispatch communication with intra-node AG communication, while the fused RS-Combine algorithm overlaps inter-node Combine communication, intra-node RS communication, and the computation of top-*k* weights. The Gantt chart indicates that the asynchronous fused AR-A2A demonstrates a performance improvement compared to the total

latencies of the synchronous operators, which is approximately slightly greater than inter-node communication overhead.

Fig. 12b shows the performance comparison of synchronous and asynchronous communication. The results indicate that the asynchronous communication significantly reduces the overall latency, leading to improved TTFT and ITL. The throughput also increases due to the reduced communication overhead. This ablation study demonstrates the effectiveness of overlapping communication in enhancing performance.

#### V. RELATED WORK

#### A. Distributed MoE

In the early stages of research, various distributed methods facilitated parallel training of MoE models to improve throughput and efficiency. GShard [10] pioneering the use of all-to-all communication for large-scale sparsity. Subsequent frameworks like DeepSpeed-MoE [11] and Tutel [8] refined this via hybrid data/expert parallelism and fused kernels to enhance memory efficiency and multi-node scalability. To further mitigate communication bottlenecks, SmartMoE [12] introduced dynamic strategy selection, while Lina [13] optimized interleaved all-to-all operators. More recently, the field has moved toward high-dimensional parallelism; notably, MoE Parallel Folding [4] utilizes Megatron-Core [14] to integrate TP, EP, DP, PP, and context parallelism (CP) into a unified 5D scheme for heterogeneous clusters.

Our work leverages numerous methods and concepts from the training of distributed MoE models, focusing on their application in distributed MoE model serving.

## B. Distributed LLM Serving

Distributed serving systems prioritize maximizing throughput and minimizing latency for online inference. Early optimizations focused on request scheduling: Orca [15] pioneered iteration-level scheduling and selective batching, while Llumnix [16] introduced dynamic resource allocation based on workload characteristics. Regarding parallelism, Alpa [17] and AlpaServe [18] explored the synergy between intra/inter-operator parallelism and model multiplexing. To further optimize the distinct phases of inference, DistServe [19] proposed prefill/decode (P/D) disaggregation, and Sarathi-Serve [20] utilized chunked-prefills with stall-free scheduling to balance throughput-latency trade-offs. Most recently, MegaScale-Infer [21] extended disaggregation to Attention and MoE blocks, leveraging pingpong pipeline parallelism to hide communication overhead and maximize GPU utilization.

Our work focuses on parallel strategies and communication optimization, and can be effectively incorporated with various optimization methods of existing LLM serving systems, such as request scheduling, P/D disaggregation, etc.

#### VI. Conclusion

We introduce MixServe, a novel automatic distributed serving system that for efficient deployment of MoE models by hybrid TP-EP based on fused AR-A2A communication algorithm. MixServe automatically selects the optimal parallel strategy

based on model parameters and network configurations. It employs a hybrid TP-EP partitioner to optimize communication overhead and introduces a fused AR-A2A communication algorithm to enhance TTFT, ITL and throughput. MixServe's design is guided by theoretical analysis and practical considerations, ensuring efficient resource utilization and low latency. Our evaluation on mainstream MoE models such as DeepSeek-R1 and Qwen3 demonstrates that MixServe achieves significant performance improvements in MoE model serving, making it a valuable tool for deploying large-scale LLMs. We hope MixServe will contribute to the efficient deployment of MoE models in real-world applications.

#### REFERENCES

- [1] DeepSeek-AI *et al.*, "Deepseek-v3 technical report," 2025. [Online]. Available: https://arxiv.org/abs/2412.19437
- [2] —, "Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning," 2025. [Online]. Available: https://arxiv.org/abs/2501.12948
- [3] A. Yang et al., "Qwen3 technical report," 2025. [Online]. Available: https://arxiv.org/abs/2505.09388
- [4] D. Liu et al., "Moe parallel folding: Heterogeneous parallelism mappings for efficient large-scale moe model training with megatron core," 2025. [Online]. Available: https://arxiv.org/abs/2504.14960
- [5] M. Shoeybi et al., "Megatron-lm: Training multi-billion parameter language models using model parallelism," 2020. [Online]. Available: https://arxiv.org/abs/1909.08053
- [6] W. Fedus, B. Zoph, and N. Shazeer, "Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity," 2022. [Online]. Available: https://arxiv.org/abs/2101.03961
- [7] W. Kwon et al., "Efficient memory management for large language model serving with pagedattention," in Proceedings of the 29th Symposium on Operating Systems Principles, ser. SOSP '23. New York, NY, USA: Association for Computing Machinery, 2023, p. 611–626. [Online]. Available: https://doi.org/10.1145/3600006.3613165
- [8] C. Hwang et al., "Tutel: Adaptive mixture-of-experts at scale," 2023.[Online]. Available: https://arxiv.org/abs/2206.03382
- [9] OpenChat Team, "Openchat sharegpt v3," https://huggingface.co/datasets/ openchat/openchat\_sharegpt\_v3, 2023, shareGPT dataset for training OpenChat V3 series. Licensed under MIT. Accessed: 2025-08-20.
- [10] D. Lepikhin et al., "Gshard: Scaling giant models with conditional computation and automatic sharding," 2020. [Online]. Available: https://arxiv.org/abs/2006.16668
- [11] S. Rajbhandari et al., "Deepspeed-moe: Advancing mixture-of-experts inference and training to power next-generation ai scale," 2022. [Online]. Available: https://arxiv.org/abs/2201.05596
- [12] M. Zhai et al., "SmartMoE: Efficiently training Sparsely-Activated models through combining offline and online parallelization," in 2023 USENIX Annual Technical Conference (USENIX ATC 23). Boston, MA: USENIX Association, Jul. 2023, pp. 961–975. [Online]. Available: https://www.usenix.org/conference/atc23/presentation/zhai
- [13] J. Li et al., "Accelerating distributed MoE training and inference with lina," in 2023 USENIX Annual Technical Conference (USENIX ATC 23). Boston, MA: USENIX Association, Jul. 2023, pp. 945–959. [Online]. Available: https://www.usenix.org/conference/atc23/ presentation/li-jiamin
- [14] NVIDIA Corporation, "Megatron-lm: Ongoing research training transformer models at scale," https://github.com/NVIDIA/Megatron-LM, 2024, accessed: 2025-08-20.
- [15] G.-I. Yu et al., "Orca: A distributed serving system for Transformer-Based generative models," in 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22). Carlsbad, CA: USENIX Association, Jul. 2022, pp. 521–538. [Online]. Available: https://www.usenix.org/conference/osdi22/presentation/yu
- [16] B. Sun et al., "Llumnix: Dynamic scheduling for large language model serving," in 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24). Santa Clara, CA: USENIX Association, Jul. 2024, pp. 173–191. [Online]. Available: https://www.usenix.org/conference/osdi24/presentation/sun-biao

- [17] L. Zheng et al., "Alpa: Automating inter- and Intra-Operator parallelism for distributed deep learning," in 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22). Carlsbad, CA: USENIX Association, Jul. 2022, pp. 559–578. [Online]. Available: https://www.usenix.org/conference/osdi22/presentation/zheng-lianmin
- [18] Z. Li et al., "AlpaServe: Statistical multiplexing with model parallelism for deep learning serving," in 17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23). Boston, MA: USENIX Association, Jul. 2023, pp. 663–679. [Online]. Available: https://www.usenix.org/conference/osdi23/presentation/li-zhouhan
- [19] Y. Zhong et al., "DistServe: Disaggregating prefill and decoding for goodput-optimized large language model serving," in 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24). Santa Clara, CA: USENIX Association, Jul. 2024, pp. 193–210. [Online]. Available: https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin
- [20] A. Agrawal et al., "Taming Throughput-Latency tradeoff in LLM inference with Sarathi-Serve," in 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24). Santa Clara, CA: USENIX Association, Jul. 2024, pp. 117–134. [Online]. Available: https://www.usenix.org/conference/osdi24/presentation/agrawal
- [21] R. Zhu *et al.*, "Megascale-infer: Serving mixture-of-experts at scale with disaggregated expert parallelism," 2025. [Online]. Available: https://arxiv.org/abs/2504.02263