# HarMoEny: Efficient Multi-GPU Inference of MoE Models

Zachary Doucet McGill Canada Rishi Sharma EPFL Switzerland Martijn de Vos EPFL Switzerland Rafael Pires EPFL Anne-Marie Kermarrec EPFL Oana Balmau McGill

Switzerland

## Abstract

Mixture-of-Experts (MoE) models offer computational efficiency during inference by activating only a subset of specialized experts for a given input. This enables efficient model scaling on multi-GPU systems that use expert parallelism without compromising performance. However, load imbalance among experts and GPUs introduces waiting times, which can significantly increase inference latency. To address this challenge, we propose HarMoEny, a novel solution to address MoE load imbalance through two simple techniques: (i) dynamic token redistribution to underutilized GPUs and (ii) asynchronous prefetching of experts from the system to GPU memory. These techniques achieve a near-perfect load balance among experts and GPUs and mitigate delays caused by overloaded GPUs. We implement HarMoEny and compare its latency and throughput with four MoE baselines using real-world and synthetic datasets. Under heavy load imbalance, HarMoEny increases throughput by 37%–70% and reduces time-to-first-token by 34%–41%, compared to the next-best baseline. Moreover, our ablation study demonstrates that HarMoEny's scheduling policy reduces the GPU idling time by up to 84% compared to the baseline policies.

Switzerland

# CCS Concepts

• Computer systems organization <sup>→</sup> Parallel architectures; Distributed architectures.

# Keywords

ML Inference, Mixture-of-Experts Models, Load Balancing

#### ACM Reference Format:

Zachary Doucet, Rishi Sharma, Martijn de Vos, Rafael Pires, Anne-Marie Kermarrec, and Oana Balmau. 2025. HarMoEny: Efficient Multi-GPU Inference of MoE Models. In . ACM, New York, NY, USA, [13](#page-12-0) pages. [https:](https://doi.org/10.1145/nnnnnnn.nnnnnnn) [//doi.org/10.1145/nnnnnnn.nnnnnnn](https://doi.org/10.1145/nnnnnnn.nnnnnnn)

## 1 Introduction

Scaling machine learning (ML) models to billions of parameters has enabled powerful generative models [\[17,](#page-12-1) [37\]](#page-12-2). One of the main applications of these models is natural language processing, where large language models (LLMs) such as GPT-3 [\[5\]](#page-12-3) and GPT-4 [\[1\]](#page-12-4) are widely used for tasks like text generation and question answering. However, these applications come with steep costs and high energy consumption. As model sizes grow, inference becomes increasingly expensive [\[4\]](#page-12-5), and it already constitutes the majority of

ML workloads. NVIDIA and AWS estimate that up to 90% of the ML workloads are serving deep neural network models [\[2,](#page-12-6) [21\]](#page-12-7). To address this critical challenge, this paper focuses on strategies for efficient ML inference.

Canada

One promising approach is the use of Mixture-of-Experts (MoE) models [\[15\]](#page-12-8). Compared to traditional models, MoE models can provide more than 10× reduction in computation requirements for inference, without sacrificing accuracy [\[8,](#page-12-9) [22\]](#page-12-10). The Switch Transformers [\[8\]](#page-12-9), Mixtral [\[16\]](#page-12-11), Qwen [\[36\]](#page-12-12), and DeepSeek [\[24\]](#page-12-13) model families are some of the most successful MoEs. Each expert in an MoE model is trained to focus on a specific subset of tasks or data patterns. A gating mechanism, often a smaller neural network called a router, decides which experts will process a given input. This approach allows MoE models to scale, using only a fraction of their capacity per input, resulting in computational savings.

Despite the benefits in terms of computation, MoE models have a significant memory footprint. While only a fraction of experts are activated per input, all the experts need to be available in GPU memory to process a batch of requests. Therefore, serving MoE models requires a combination of multiple GPUs, each holding a subset of the experts, also known as expert parallelism (EP).

<span id="page-0-0"></span>![](_page_0_Figure_16.jpeg)

Figure 1: The ECDF of token placement across all 128 experts on the bookcorpus dataset for the Qwen and Switch128 models. The workload exhibits strong skew where a handful of experts receive over half the total number of tokens. Efficiently handling pronounced, dynamic skew is a challenge for existing MoE serving systems.

Serving MoE models using multiple GPUs has two significant bottlenecks that increase the end-to-end request latency: (i) synchronization, and (ii) load imbalance among the GPUs [\[12,](#page-12-14) [13,](#page-12-15) [23,](#page-12-16) [38\]](#page-12-17). First, MoEs models require two synchronization steps (all-to-all communication) between GPUs in every MoE block. Based on the decisions of the router in each MoE block, each GPU first scatters its inputs to the relevant experts on other GPUs and gathers them back after the computation is done. These synchronizations are essential to ensure that all GPUs have the complete results from the current MoE block before moving on to the next one. Recent MoE training and inference frameworks have alleviated this slowdown by optimizing the all-to-all communication [\[13,](#page-12-15) [23,](#page-12-16) [38\]](#page-12-17). They either resort to replication [\[13\]](#page-12-15) of popular experts or collocate popular experts across MoE layers [\[23,](#page-12-16) [38\]](#page-12-17). In these cases, the popularity of experts is determined through profiling and scheduling performed offline. However, expert popularity is dynamic and depends on the tokens in the input request. The tokens depend on the domain of the input request, i.e., a medical prompt activates different tokens when compared to a StackOverflow question [\[16\]](#page-12-11).

Indeed, depending on the workload, some experts are significantly more popular than others and are assigned more tokens than other experts. We show this in Figure [1](#page-0-0) for the Qwen and Switch128 MoE models during a run with the bookcorpus dataset. Token placements across experts are non-uniform, and tokens are disproportionately routed to a small subset of experts per input. For instance, in layer 0 of the Qwen model, only three experts receive an average of 19% of the tokens, whereas in the final layer, three experts receive as much as 60% on average. This skew becomes more pronounced in deeper layers of the model. Moreover, the expert skew is dynamic, with the token distribution varying across queries. This skew in expert popularity leads to a load imbalance among the GPUs and to significant GPU under-utilization.

In this paper, we target load imbalance caused by skewed expert popularity to improve the efficiency of multi-GPU MoE inference. Expert popularity imbalances (shown in Figure [1\)](#page-0-0) lead to significant imbalances in GPU use. Figure [2](#page-1-0) shows the load imbalance across GPUs for two popular MoE load balancing approaches and our solution, HarMoEny. DeepSpeed uses a round-robin distribution of tokens to GPUs (also used by others such as Fast-MoE [\[12\]](#page-12-14), and FasterMoE [\[13\]](#page-12-15)). Another compelling approach used by ExFlow [\[38\]](#page-12-17) is workload profiling and integer programming to determine the optimal expert placement. This technique yields better load balance at times, but is not fast enough to adapt to skew changes across batches. Both approaches yield an imbalanced distribution of tokens to experts, leading to high GPU idle times, as we will show in Section [3.](#page-3-0) We found that this idle time can take up to 86% of the time for GPUs housing unpopular experts. In contrast, HarMoEny achieves almost perfect load balance.

HarMoEny uses two simple, yet powerful techniques to achieve near-perfect load balance among GPUs: token rebalancing from overutilized to underutilized GPUs, and asynchronous prefetching of experts from system to GPU memory. HarMoEny adapts on the fly to changes in expert popularity with no drops in throughput and does not need any profiling.

<span id="page-1-0"></span>![](_page_1_Figure_6.jpeg)

Figure 2: The ECDF of token placement across GPUs in an 8-GPU NVIDIA DGX machine, over four datasets, for a Switch Transformer model [\[8\]](#page-12-9) with 128 experts. HarMoEny achieves near-perfect load balancing.

To achieve this, HarMoEny modifies the MoE logic that is replicated on all GPUs. During each batch, the GPUs exchange a summary of the token distribution to get a global vision of the tokento-GPU and token-to-expert assignment. The metadata in this step is small (4kB) and introduces minimal overheads. Then, each GPU can deterministically infer the same token-to-expert and token-to-GPU schedule in parallel, with no further synchronization needed. HarMoEny balances the token load such that some of the tokens that were destined to overutilized GPUs (i.e., hosting more popular experts) are rerouted to underutilized GPUs. To ensure that the rerouted tokens can be processed by the popular experts once they are rebalanced to the underutilized GPUs, HarMoEny uses asynchronous expert prefetching. HarMoEny can look ahead and make sure that the right experts are paged into GPU memory from the system memory, and the ones that are not needed anymore are discarded. Swapping out an expert from GPU memory only requires an overwrite.

HarMoEny uses prefetching and load rebalancing, both wellestablished techniques in datacenter scheduling [\[7,](#page-12-18) [32\]](#page-12-19). However, we are the first to adapt such scheduling techniques to MoE models and show they eliminate GPU idleness almost completely. We compare MoE inference in HarMoEny to four state-of-the-art MoE systems: ExFlow [\[38\]](#page-12-17), FastMoE [\[12\]](#page-12-14) and FasterMoE [\[13\]](#page-12-15), and DeepSpeed-Tutel [\[29\]](#page-12-20). We show that in workloads with skewed expert popularity, HarMoEny is up to 41.1% faster than the next-best baseline, in terms of time-to-first-token. Furthermore, HarMoEny maintains stable and low inference latency even as skew and expert popularity changes. Thanks to its lightweight techniques, Har-MoEny has the capacity to quickly adapt to various datasets, which is a drawback of profiling-based approaches.

Contributions. This paper makes the following contributions:

(1) We empirically study the compute utilization of a GPU cluster running MoE inference and conclude that expert popularity imbalance has a much higher impact on inference latency

than all-to-all synchronization (Section 3). With a balanced load across GPUs, the all-to-all synchronization accounts for only 2% of the total execution time. HARMOENY's design follows from this observation.

- (2) We design and implement HARMOENY (Section 4). HARMOENY uses two complementary techniques to achieve almost perfect load balancing: token rebalancing and asynchronous prefetching of experts. HARMOENY is open source<sup>1</sup> and implemented on top of PYTORCH.
- (3) We evaluate HARMOENY with real datasets and synthetic benchmarks, showing that HARMOENY maintains a low and steady inference latency in fluctuating workloads, with different skew levels, and across different datasets (Section 5). Our simple and efficient approach reduces the waiting time of GPUs in the all-to-all synchronization step by up to 84.7% compared to baseline policies.

## 2 Background

Transformer models are nowadays widely used for ML tasks [33]. A typical transformer model consists of multiple transformer blocks, each designed to process tokens through self-attention mechanisms and Feed-Forward Networks (FFNs). A *token* here refers to an intermediate value representing a single element, *e.g.*, a word [3], a sub-word [19], or a character [11]. As illustrated in Figure 3 (left), a transformer block takes some tokens as input and consists of two main components: a self-attention mechanism and an FFN. The self-attention mechanism captures relationships between input elements across the sequence, allowing the model to focus on different parts of the input simultaneously. These outputs are passed into an FFN, which applies two dense layers with an activation function in between to refine the representations. The FFN is the most time-consuming part of the transformer block [35]. The resulting output tokens are then forwarded to the subsequent transformer block.

**Mixture-of-Experts (MoEs)** is a type of sparse computation that selectively activates only parts of the network, called *experts* [15]. A transformer block using MoE is shown in Figure 3 (right). In a MoE model, some or all transformer blocks can have the MoE layer. We refer to a transformer block having the MoE layer as a *MoE block*. In contrast to a typical transformer block with a single FFN, an MoE layer contains multiple experts, implemented by smaller FFNs, with each expert having its own set of weights [30]. The number of experts in each block is typically between 8–128 [8]. MoEs assign each token to only a portion of the network, *i.e.*, a subgroup of experts rather than passing it through the entire model.

The assignment of tokens to experts is managed by the *router*, a component responsible for directing each token to a subset of the model (in green in Figure 4). The router is usually implemented as a trainable function [30], optimized through backpropagation to discover productive token-to-expert assignments. To prevent bottlenecks, a loss term during training encourages the router to distribute tokens evenly across the experts. The router assigns each expert a value between 0 and 1. Then, expert assignment is handled with a *top-k* strategy—directing the token to the k experts with the highest values, with each expert's output weighted accordingly. The parameter k is usually set to 1 or 2, as higher values quickly raise

<span id="page-2-1"></span>![](_page_2_Figure_10.jpeg)

Figure 3: A transformer block (left) and a MoE block, containing the MoE logic (right). MoE-based models replace the FFN with a router and multiple experts implemented as FFNs.

costs while offering diminishing returns [8, 16]. After processing the expert FFN, the token's output is combined, normalized, and passed to the next layer.

**Expert parallelism (EP).** In multi-GPU scenarios, transformers typically use three types of parallelism: (*i*) data parallelism, where input data is sharded across the GPUs [6], (*ii*) model parallelism, where different parts of the model are split across the GPUs at a layer- or component-granularity [18], and (*iii*) tensor parallelism, where large tensors (*i.e.*, matrix operations within a single layer) are split across GPUs [31]. Expert parallelism (EP) [12] combines aspects of data parallelism and model parallelism to support MoEs.

With EP, the self-attention and router layers are replicated across GPUs (data parallelism), but each GPU only loads a subset of experts, distributing the complete set of experts across all GPUs [8] (model parallelism). During the forward pass, each GPU receives a minibatch of the input request comprising a set of input tokens. All the GPUs independently compute self-attention on their minibatches in parallel. Here, the routers in each GPU assign the tokens

<span id="page-2-2"></span>![](_page_2_Figure_15.jpeg)

Figure 4: Token scattering (step 1) and gathering (step 2) when using expert parallelism. Experts are split across GPUs. One batch requires two all-to-all synchronization barriers.

<span id="page-2-0"></span> $<sup>^1</sup>$ See https://github.com/sacs-epfl/HarMoEny.

<span id="page-3-2"></span>![](_page_3_Figure_2.jpeg)

Figure 5: (a) Fixed expert placement causes long waiting due to load imbalance. (b) Throughput fluctuates with static placement, given the nature of requests.

from the minibatches to experts. Since the tokens and their assigned experts can potentially be on different GPUs, an all-to-all scatter communication step ensures that each GPU receives the tokens destined for the experts it hosts. This step introduces the first synchronization barrier in MoE blocks. Upon receiving the correct tokens, each GPU performs the expert computation on the received tokens using the experts it holds. After the expert computation, an all-to-all gather communication step returns the computed results for each token back to the GPUs responsible for their corresponding inputs. Finally, the resulting tokens are then used as input for the next MoE block.

Figure 4 shows an example of an EP assignment with 6 experts split across 3 GPUs demonstrating the expert computations and the all-to-all communications. Experts 1 and 2 are housed in GPU 0, experts 3 and 4 to GPU 1, and experts 5 and 6 to GPU 2. The self-attention mechanism outputs tokens  $t_1$  to  $t_6$ . When GPU 0 processes its batch, the router assigns tokens to experts—sending a tokens to expert 1, b to expert 2, c to expert 3, and so on. Since GPU 0 only houses experts 1 and 2, it must send tokens for experts 3 and 4 to GPU 1 and tokens for experts 5 and 6 to GPU 2 (step 1). Once GPU 0 completes computations for its experts, it returns the processed tokens to their originating GPUs through another all-to-all communication, posing another synchronization (step 2).

#### <span id="page-3-0"></span>3 Effect of load imbalance in MoE Inference

MoE routers are trained to balance the token load across the experts [8, 16]. However, during inference, the expert selection, and hence the computation load across GPUs is often skewed as shown in Figure 2. We now show the impact of skewed expert popularity and load imbalance on performance.

We first show that *expert popularity skew causes significant GPU under-utilization* leading to high end-to-end latency, in line with prior work. Figure 5 (a) presents the time breakdown of GPU computations when serving a Switch transformer model with 6 layers and 128 experts. We inject artificial token skew such that 90% of the tokens are assigned to the first 10 experts. We run the experiment with 8x V100 GPUs (see Section 5.1 for the full experimental setup). In this workload, GPU 0 is assigned the most popular experts. Due to the all-to-all synchronization steps, GPUs 1–7 remain idle for more than 82% of the time. In this situation, balancing the token

load across GPUs and consequentially minimizing the waiting times would lead to efficient serving and reduced end-to-end latency of inference requests.

Prior work improved the efficiency of MoE model serving in the case of static or mostly-static expert popularity [12, 13, 23, 38]. Simple replication of the popular experts onto the under-utilized GPUs as proposed by FasterMoE [13] suffices for balancing loads in workloads with static expert popularity. In scenarios where expert popularity skew slowly changes over time, profiling can be done over time and expert placement can be adjusted periodically as done in ExFlow [38].

Realistically, however, the skew and consequently the load imbalance fluctuate between batches. This variation arises because the skew depends on the token distribution within the input requests, which is in turn influenced by the domain of the requests (e.g., medical, programming, etc.) [16]. The skew fluctuation results in unstable throughput across batches for static solutions like FASTMOE. Furthermore, profiling-based solutions are inefficient for MoE serving in the presence of dynamic skew due to their cost. For instance, in our test bed, profiling and readjusting the schedule with integer programming in ExFLow takes about 8.5 minutes for the Switch transformer and as much as 45 minutes for experiments with the Owen MoE model. In contrast, even with low GPU utilization, the mean time to process a single batch of requests through a MoE block is only 289 milliseconds as shown in Figure 5 (a). Therefore, ExFLow does not have enough time to adapt to the dynamic expert popularity.

Figure 5 (b) shows the impact on the throughput of Faster-Moe [13] and ExFlow [38] in a longer run of the same setup as Figure 5 (a) with dynamically fluctuating expert-token skew across batches. FasterMoe starts with a round-robin placement of experts on GPUs and ExFlow uses 40 samples from the Bookcorpus dataset to create a schedule for expert placement on the GPUs using integer programming (see Section 5.1 for more details). We can see that ExFlow's and FasterMoe's throughput fluctuates across batches and drops by up to 37.6% within just two consecutive batches. Both systems have similar performance as neither system has time to adapt to the rapid fluctuations in the skew.

## <span id="page-3-1"></span>4 HARMoEny design

Based on our findings in Section 3, we build Harmoeny as a system to reduce inference latency for MoE models. Harmoeny is composed of two main components: (i) a scheduler that load-balances tokens across experts, and (ii) the expert pre-fetching protocol that asynchronously prefetches an expert into GPU memory. These two techniques reduce GPU idle time introduced by token load imbalance without any online profiling and allow Harmoeny to adapt to rapid workload fluctuations. We first explain the overall workflow of Harmoeny, and then explain each of these two components.

### 4.1 HARMOENY workflow

Algorithm 1 shows the operations during a forward pass through the MoE logic with HARMOENY. This FORWARD function is executed by each GPU and takes some input tokens x, which is a tensor of shape [batch size, sequence length, hidden dimension]. We assume

#### Algorithm 1: HARMOENY MoE Layer

```
Require: G: Set of GPUs.
2 Procedure FORWARD(x):
       // Step 1: token routing
       m_{expert} \leftarrow \text{ROUTER}(x)
4
       // Step 2: metadata exchange
       SENDMETADATATOGPUs(m_{expert})
       receive m_{all}[i] from each GPU i \in G
8
       // Step 3: token scheduling
10
11
       S_{initial} \leftarrow \text{INITIALASSIGN}(m_{all})
       S \leftarrow \text{rebalance}(S_{initial})
                                          ▶ See Section 4.2
12
13
       // Step 4: scatter tokens
                                          ▶ Step 1 in Figure 4
14
       SENDTOKENSTOGPUS(x, m_{expert}, S)
15
       receive x' from all other GPUs
16
17
       // Step 5: expert processing and async. loading
18
       x'' \leftarrow \text{EXPERTS}(S, x')
                                        ▶ See Section 4.3
19
       // Step 6: gather tokens
                                         ▶ Step 2 in Figure 4
21
22
       SENDTOKENSBACKTOGPUs(S, x'')
       receive y[i] from each GPU i \in G
23
       x \leftarrow \text{RECONSTRUCT}(S, y, m_{all})
24
       return x
25
```

<span id="page-4-6"></span><span id="page-4-5"></span>that input tokens x have already been passed through the self-attention layer and will now be routed to and processed by the relevant experts. This proceeds in the following six steps.

**Step 1: token routing.** Input tokens x are first assigned to specific experts based on their characteristics using a router mechanism (Line 4). This assignment creates a token-to-expert mapping,  $m_{expert}$ , which is a tensor of integers representing the target expert for each token.

**Step 2: metadata exchange.** Next, all GPUs exchange their computed token-to-expert distribution. Specifically, each GPU broadcasts its local token-to-expert assignments, allowing all GPUs to build a global, shared understanding of the token distribution. This metadata exchange step requires very little communication (a few kilobytes) and thus is efficient. This information is stored in the array  $m_{all}$  which tracks the token-to-expert assignment for all other GPUs. This step ensures that Harmoeny token scheduling operates with a complete view of the workload. We note that this metadata exchange is unique to Harmoeny, and we experimentally show in Section 5 that its overhead is negligible.

**Step 3: token scheduling.** Based on the token-to-expert assignments in  $m_{all}$ , an initial (naive) token schedule  $S_{initial}$  is generated (Line 11).  $S_{initial}$  is a three-dimensional tensor of integers where each entry represents the number of tokens assigned from a source GPU to a destination GPU for a particular expert. Since this might

result in load imbalance issues, HARMOENY then *rebalances* this schedule through an algorithm that redistributes tokens from overburdened GPUs to underutilized ones (Line 12, also see Section 4.2).

**Step 4: scatter tokens.** Based on the rebalanced schedule (S) and the local token-to-expert assignment  $m_{expert}$ , each GPU now sends its input tokens in x to the designated GPUs through an all-to-all communication step. Each GPU i then receives all tokens x' from other GPUs that should be processed by the receiving GPU.

Step 5: expert processing and asynchronous loading. The tokens in x' are then processed by the appropriate experts, resulting in x'' (Line 19). It might be that our rebalancing algorithm assigns tokens to experts that are not currently housed by a particular GPU. Harmoeny employs a novel, asynchronous expert pre-fetching protocol to ensure that the required expert weights are loaded into GPU memory without delaying token processing (see Section 4.3). Weight transfers are overlapped with computation, reducing idle GPU time.

**Step 6: gather tokens.** After expert processing is completed, each GPU sends the processed tokens x'' back to their original GPUs using a second all-to-all communication step. This ensures that each GPU receives the results for the tokens it initially routed. Each GPU i stores the received tokens in y[i]. Once all tokens are collected, the received tokens are restructured to ensure that the processed tokens are aligned correctly with their original order and source (Line 24). This completes the processing of input tokens x by the MoE logic, yielding the final output tokens.

#### <span id="page-4-1"></span>4.2 Load-aware token scheduler

One of the innovations of Harmoenv is an efficient load-aware token scheduler that assigns each token to one of the available GPUs. The scheduler is able to rebalance the load by re-assigning tokens that are destined to one GPU to a less crowded GPU. We outline the algorithm in Algorithm 2 and visualize this process in Figure 6. The procedure Rebalance takes as input an initial schedule  $S_{initial}$ , which is then assigned to the variable S.  $S_{initial}$  and S are three-dimensional arrays, containing a mapping between source GPUs, experts, and destination GPUs. Specifically,  $S[g_{from},e,g_{to}]$  denotes the number of tokens sent from GPU  $g_{from}$  for expert e to GPU  $g_{to}$ . Thus, the first dimension corresponds to the source GPUs, the second to the experts, and the third to the destination GPUs where these experts are housed.

Imbalances arise due to uneven distribution across experts (see Section 3). We show an example in Figure 6 which illustrates a system with 3 GPUs and 15 input tokens, assigned to three different experts, and using top-1 routing (e.g., in each MoE layer, each token is processed by a single expert). The color of each input token in Figure 6 (a) indicates the expert that the router has assigned to the token. Assume that experts 0, 1, and 2 are located on GPU 0, 1, and 2, respectively. Naively assigning input tokens to experts will result in GPU 2 having to process 9 tokens and GPU 0 just two tokens, resulting in load imbalance. Figure 6 (b, left) shows a timeline with operations per GPU. The disproportionate load on GPU 2 causes the computation time of expert 2 to grow, introducing waiting times for GPU 0 and 1 and prolonging the inference request duration.

HARMOENY rebalances experts across GPUs by analyzing and manipulating the expert-to-GPU assignment in *S*. If a GPU receives

<span id="page-5-1"></span>![](_page_5_Figure_2.jpeg)

![](_page_5_Figure_3.jpeg)

![](_page_5_Figure_4.jpeg)

(b) Timeline with operations per GPU, with and without rebalancing. Experts are fetched asynchronously (see Section 4.3).

Figure 6: An example of the token rebalancing process by HARMOENY. Setup: 15 input tokens, three GPUs and three experts.

```
Algorithm 2: HARMOENY Token Rebalancing
    Require: G: Set of GPUs, E: Set of experts, q: Token
                   transfer threshold.
    Output: Rebalanced schedule S
   Procedure REBALANCE(S<sub>initial</sub>):
          S \leftarrow S_{initial}
          t_{\text{avg}} \leftarrow \lfloor S.\text{sum}()/|G| \rfloor
                                                  ▶ Avg. tokens per GPU
          t_a \leftarrow S.\text{sum}(\dim = (0, 1))
                                                  ▶ Token count per GPU
          while ANY(t_q > t_{avg}) do
 6
               g_{\max} \leftarrow \text{Argmax}(t_q)
               g_{\text{from}} \leftarrow \text{Argmax}(\text{sum}(S[:,:,g_{\text{max}}], \text{dim} = 1))
 8
               e_{\max} \leftarrow ARGMAX(S[g_{\text{from}},:,g_{\max}])
10
               t_{\text{move}} \leftarrow S[g_{\text{from}}, e_{\text{max}}, g_{\text{max}}]
11
               if t_{move} < q then
12
                                         ▶ Insufficient tokens to move
                    return S
13
14
15
               g_{\min} \leftarrow \operatorname{ARGMIN}(t_q)
                                                  ▶ Find least loaded GPU
               if g_{min} = g_{max} or t_g[g_{min}] + q > t_{avg} then
16
                return S
                                               ▶ No feasible transfer
17
18
               t_s \leftarrow \min(t_{\text{move}}, t_{\text{avg}} - t_q[g_{\text{min}}])
19
               S[q_{\text{from}}, e_{\text{max}}, q_{\text{max}}] = t_s
20
               S[g_{\text{from}}, e_{\text{max}}, g_{\text{min}}] += t_s
21
22
               t_a[q_{\max}] = t_s
               t_q[g_{\min}] += t_s
23
24
          return S
25
```

<span id="page-5-7"></span><span id="page-5-6"></span>more tokens than the average allocation, it is considered overutilized. The policy then identifies the least utilized GPU and redirects as many tokens as possible to this GPU without causing it to become overutilized. We visualize this in Figure 6 (a, right) where expert 0 is replicated to GPU 1 as well, resulting in a situation where each GPU now processes an equal amount of tokens. This process repeats until either all GPUs have a balanced token load, or there are no further offloading options available.

We next provide a detailed description of our greedy scheduling algorithm in Algorithm 2. To this end, we first compute the average number of tokens  $t_{ava}$  and the total number of tokens  $t_a$ that each GPU has to process (S.SUM() returns the total number of tokens across all GPUs and experts). The rebalancing loop operates iteratively to reduce load imbalances across GPUs (Line 6). This loop runs as long as there is a GPU that receives more than the average number of tokens  $t_{avq}$ . In each batch, HARMoENY identifies the index of the most overloaded GPU, referred to as  $g_{max}$ , which has the highest total token count. The scheduler then determines the source GPU  $g_{from}$  that contributes the largest share of tokens to  $g_{max}$  (Line 8). The term  $SUM(S[:,:,g_{max}], dim=1)$  calculates the total number of tokens contributed by each source GPU to the overloaded GPU  $g_{max}$ , summing over all experts. We then identify within  $g_{from}$  the expert  $e_{max}$  that is responsible for sending the most tokens to  $g_{max}$  (Line 9).

Once the relevant source GPU and expert are determined, the algorithm calculates the number of tokens  $t_{move}$  to potentially transfer (Line 11). If  $t_{\text{move}}$  is smaller than some token threshold q, the algorithm stops the process, as moving such a small number of tokens would not sufficiently reduce the imbalance. The token threshold q is an important hyperparameter of HARMOENY that decides the lower bound number of tokens necessary to offload tokens to another GPU. The reason for introducing this threshold is to account for the time of loading an expert from memory to overlap the token processing with the expert fetching (see Figure 6 (b, right)). Moving a very small number of tokens might not sufficiently reduce the imbalance to justify the cost of communication overhead, such as reconfiguring schedules and initiating transfers between GPUs. This would result in minimal performance gains or even a net loss in efficiency. q depends on the system specifications and is independent of dynamic aspects of the workload such as expert popularity in a given batch. We discuss how to determine this hyperparameter in Section 4.3.

If there are sufficient tokens to move, the algorithm then identifies GPU  $g_{min}$  that is assigned the least number of tokens (Line 15). Tokens from  $g_{from}$  and  $e_{max}$ , destined for  $g_{max}$ , are then redirected to  $g_{min}$ , ensuring that  $g_{min}$  does not exceed the average load  $t_{avg}$ . Specifically, the exact number of tokens transferred,  $t_s$ , is the smaller of  $t_{move}$  or the remaining capacity of  $g_{min}$  (Line 19). After transferring tokens, the corresponding entries in S are updated to

reflect the new distribution of tokens. The total token counts in  $t_g$  are also adjusted accordingly. This rebalancing step is repeated until either all GPUs have token counts close to  $t_{avg}$  or no further feasible transfers can be made (e.g.,  $t_{move} < q$ ).

The ECDFs in Figure 2 illustrate the distribution of tokens assigned to each GPU across four different datasets. DeepSpeed, lacking any token load rebalancing mechanism, results in substantial load imbalances, with certain GPUs processing significantly more tokens than others. In contrast, Harmoeny employs an effective rebalancing algorithm that dynamically redistributes tokens, ensuring a near-uniform workload across all GPUs. This improvement is consistent across all evaluated datasets, showcasing Harmoeny's robustness and adaptability to different input distributions. By mitigating load skew, Harmoeny significantly enhances inference efficiency as we will show in Section 5.

## <span id="page-6-2"></span>4.3 Asynchronous expert fetching

To handle load balancing effectively, the Harmoeny scheduler may assign tokens for certain experts to GPUs that currently do not have these experts loaded in GPU memory. For example, Figure 6 (a) shows that after rebalancing, GPUs 0 and 1 get assigned some tokens designated for expert 2, which is currently not in their memory. Thus, Harmoeny needs a method to transfer experts into GPU memory as needed.

Simply loading experts after each expert completes processing introduces delays in the inference pipeline. This approach is inefficient, as the GPU must wait for the current expert to be offloaded from GPU memory before loading the next expert's weights from the system memory. Given the typically large size of expert weights (e.g., each expert in the Switch transformer and Qwen is 18 MB and 33 MB, respectively), this results in frequent stalls where the GPU is idle, waiting for data transfers to complete.

To achieve this efficiently, HARMoENY prefetches experts asynchronously, enabling transfers of experts' weights off the critical path. Specifically, once an expert completes processing its allocated tokens, it checks for any remaining experts that need to run but are not currently loaded. If one is found, HARMOENY fetches the weights for this next expert from system memory, directly overwriting the memory location of the expert that completed its processing. We show an example of this in Figure 6 (b, right), where GPU 0 and GPU 1 will asynchronously fetch expert 2 while computing with expert 0 and expert 1, respectively. This technique significantly speeds up operations compared to the traditional approach of first writing the current expert to system memory and then loading the new expert into GPU memory, as the offloading to system memory is not needed. Our measurements show that overwriting can speedup expert loading by 5.5x: reducing 11 ms to 2 ms for V100 GPUs. We note that HARMoEny benefits from asynchronous expert fetching if at least two experts fit in the GPU memory, a requirement for any system serving MoEs with many experts.

HARMOENY's prefetching protocol relies on the preceding experts to process enough tokens so that the asynchronous weight transfer can be completed before the next expert starts. If the computation of a particular expert is much quicker than the expert transfer time, the gains of this approach diminish. This is influenced by the token threshold *q*. Thus, if *q* is chosen appropriately,

the transfer time is effectively masked, minimizing idle periods and maintaining efficiency.

## 4.4 Determining the token threshold q

The token threshold q (see Section 4.2) influences the number of additional experts each GPU has to load. A small value of q causes tokens to be offloaded to GPUs without the corresponding expert and not enough tokens to amortize the cost of fetching the expert from memory. However, a high value of q might not sufficiently address the token load imbalance and not balance the processing times of GPUs.

Ideally, we fix q such that the time to execute an expert exceeds the time to load a new expert. Let |O| be the number of required floating operations to execute a particular expert,  $\phi$  the FLOPS of the GPU being used, |E| the size of an expert in bytes, and  $\beta$  the PCIe bandwidth in bytes per second.

<span id="page-6-5"></span>
$$\frac{|O|}{\phi} > \frac{|E|}{\beta} \tag{1}$$

Experts are typically two-layer MLPs, with the first one,  $W^1$ , being of size  $m \times p$ , and the second one,  $W^2$ , being  $p \times m$ . Also, let  $d_{type}$  be the size of an element in  $W^1$  or  $W^2$  and q the number of tokens being processed by the expert. The expert computation can be expressed as  $xW^1W^2$  where x are the input tokens. Thus, the size |E| of an expert is given by:

<span id="page-6-3"></span>
$$|E| = (mp + pm)d_{type} \tag{2}$$

The number of operations required to complete the expert computation with q tokens is given by:

<span id="page-6-4"></span>
$$|O| = qp(2m - 1) + qm(2p - 1) \tag{3}$$

By plugging Equations (2) and (3) into Equation (1), rearranging terms, and ignoring negligible factors, we get the following inequality:

<span id="page-6-6"></span>
$$q > \frac{\phi \cdot d_{type}}{2\beta} \tag{4}$$

Since  $\phi$ ,  $d_{type}$  and  $\beta$  can obtained with relative ease, Equation (4) guides system designers to obtain an estimate on q. A full derivation of Equation (4) is provided in Appendix B. It is important to note that q only depends on the system specification and the bit-precision of the parameters being served and does not depend on any dynamic properties. Furthermore, in our experiments, we found that Harmoeny is not extremely sensitive to q. Therefore, the lower bound of Equation (4) provides a reliable approximation.

#### <span id="page-6-0"></span>5 Evaluation

We implement Harmoenv and compare its latency and throughput with baseline systems.

#### <span id="page-6-1"></span>5.1 Experimental setup

We implement Harmoeny in 1115 lines of code in PyTorch [27]. Our implementation achieves asynchronous expert fetching through a dedicated NVIDIA CUDA stream for expert loading. The MoE layer is written as an nn. Module in PyTorch for expert parallelism. The implementation is modular and can be applied to any PyTorch

model, as we demonstrate by evaluating the performance of Har-MoEny with different models.

MoE models. We evaluate HarMoEny using two MoE models: Switch Transformer [\[8\]](#page-12-9) and Qwen [\[36\]](#page-12-12). The Switch Transformer is a language model that extends the T5 architecture [\[28\]](#page-12-31) by replacing its feed-forward layers with MoE logic. It contains 12 total transformer blocks alternating between a MoE block and a classic transformer block. Each MoE block in this model has 128 experts and we refer to this model as Switch128. Qwen is a series of Transformerbased language models developed by Alibaba Cloud, designed to handle a wide range of tasks. We use the Qwen 1.5 MoE model which features 24 transformer blocks each of which has 60 experts. We refer to this model as Qwen. We summarize the specifications of the MoE models used in our evaluation in Appendix [A.](#page-11-1)

Hardware. All experiments are performed on a DGX1 machine featuring eight NVIDIA V100 GPUs (each with 32GB GPU memory) interconnected with NVLink, and 500 GB of system memory.

Metrics. In our evaluation, we use two key metrics to quantify system performance: throughput and mean time-to-first-token (TTFT). Throughput is calculated as the total number of tokens generated across the experiment divided by the experiment length. Mean TTFT is a commonly-used metric that captures the average latency between the initiation of a request and the generation of the first token, reflecting the responsiveness of the system during inference.

5.1.1 MoE System Baselines. We compare HarMoEny against four baselines: DeepSpeed, FastMoE, FasterMoE, and ExFlow.

DeepSpeed is a framework for distributed training and inference of large ML models [\[29\]](#page-12-20). For our evaluation, we specifically compare against DeepSpeed-MoE with Tutel enabled, which is an extension of DeepSpeed and adds support for MoE training and inference. Furthermore, we use a high capacity factor to prevent DeepSpeed from dropping tokens for a fair comparison. Finally, we enable expert parallelism (EP) in DeepSpeed which uses a roundrobin placement of experts on the available GPUs.[2](#page-7-0)

FastMoE is one of the earliest systems for distributed training of MoE models [\[12\]](#page-12-14). The system enables large-scale MoE training by allowing expert modules to be placed across multiple GPUs and nodes. FastMoE provides flexibility by allowing developers to use custom gate and expert networks, with built-in support for Transformer-based models like Megatron-LM. It relies on optimized CUDA kernels for rapid data movement and expert selection resulting in high performance. FastMoE also features parallel expert computation to maximize hardware usage.

FasterMoE addresses token load imbalance in FastMoE through dynamic shadowing and fine-grained scheduling, while introducing congestion-avoiding expert selection during training [\[13\]](#page-12-15). The system introduces dynamic shadowing, which replicates parameters of heavily used experts (popular experts) across workers, reducing the communication overhead associated with imbalanced workloads. FasterMoE also introduces a topology-aware gating function that directs inputs to the experts with lower latency. This function reduces communication overhead by prioritizing local expert assignments and avoiding congested network links.

ExFlow addresses the inefficiencies of MoE model inference by exploiting inter-layer expert affinity [\[38\]](#page-12-17). The key innovation lies in leveraging the observed tendency for tokens to follow predictable routing patterns across consecutive MoE layers. ExFlow optimizes expert placement based on this affinity seen during training or a held-out dataset using an integer programming approach. Efficient placement of experts in ExFlow reduces the need for cross-GPU token routing, optimizing the duration of all-to-all communications.

<span id="page-7-1"></span>5.1.2 Datasets. Our experiments involve three real-world datasets. (1) bookcorpus is a large-scale dataset comprising up to 7185 unique books originally collected from smashwords.com [\[40\]](#page-12-32). The collection comprises multiple genres and literary styles; (2) wikitext is a collection of over 100 million tokens scraped from Good and Featured articles on Wikipedia [\[25\]](#page-12-33); (3) wmt19 includes translation pairs between two languages [\[9\]](#page-12-34). We work with the German– to-English set with 34.8 million translation pairs.

To better understand the performance of HarMoEny and baselines, we also adopt two synthetic datasets. (1) Random is a dataset that is constructed by stringing random tokens in a sequence of a desired length. A seed is set to ensure that random produces an identical dataset across separate runs. (2) Constant is a dataset where a single token is repeated to a desired length for all batches.

Expert popularity skew. In addition to real-world datasets, we also experiment with artificial expert popularity skews. To introduce artificial skew, we modify the router in each MoE block. We implement a configurable router skew mechanism according to the desired skew , where <sup>0</sup> <sup>≤</sup> <sup>≤</sup> 1, and the number of experts . The selected skewed experts are assigned a probability proportional to and the remaining experts share the remaining probability evenly, ensuring that the sum of the distribution equals 1. During routing, the router uses the multinomial distribution to sample tokens based on these probabilities, ensuring that the token distribution aligns with the desired skew. This allows us to have fine-grained control over the load imbalance and study the performance of the baselines and HarMoEny in more detail.

# 5.2 Comparison to state-of-the-art systems

Skewed datasets. HarMoEny performs especially well when workloads exhibit high expert popularity skew. To showcase this feature, we artificially create workloads with 50% and 90% skew, i.e., 50% and 90% of the tokens are routed to one popular expert, as detailed above. Figure [7](#page-8-0) (left) and Figure [8](#page-8-1) (left) show the throughput and mean TTFT for the Constant dataset with different levels of router skew. In the scenario with 90% skew, HarMoEny outperforms Fast-MoE and FasterMoE by 1.5x and 1.7x for both the Switch128 and Qwen models, respectively. HarMoEny beats DeepSpeed by 9.1x for the 90% skew and 8.8x for 50% skew scenario for Switch128. The trend remains the same for Qwen. When compared to the MoE inference solution ExFlow, HarMoEny performs 1.7x and 1.5x better on the 90% and 50% skew, respectively for Switch128. The performance boost is even higher for Qwen that has larger experts with HarMoEny outperforming ExFlow by 2.2x and 1.8x in the 90% and 50% skew scenarios. In the 50% skewed workload, HarMoEny is 1.3x and 1.4x faster than FasterMoE for Switch128 and Qwen, respectively.

<span id="page-7-0"></span><sup>2</sup>DeepSpeed-MII, separate to DeepSpeed, is not evaluated as the framework cannot execute on pre-Ampere GPUs, such as the V100s used in these experiments.

<span id="page-8-0"></span>![](_page_8_Figure_2.jpeg)

Figure 7: Throughput (<sup>↑</sup> is better) for different systems and different skews (left) and datasets (right) when using the Switch128 (top) and Qwen models (bottom).

<span id="page-8-1"></span>![](_page_8_Figure_4.jpeg)

Figure 8: Mean TTFT (<sup>↓</sup> is better) for different systems and different skews (left) and datasets (right) when using the Switch128 (top) and Qwen (bottom) models.

Similar to the baselines, there is a deterioration in the throughput of HarMoEny as the skew increases. This happens because in extremely skewed cases, all the experts except for the popular one have very few tokens to process. Therefore, HarMoEny is not able to efficiently mask the expert fetching with computation leading to slightly lower throughputs. However, it is important to note that HarMoEny outperforms the state-of-the-art baselines in the skewed scenarios.

As expected, there is little difference between most systems when all experts are equally popular (0% skew). FastMoE and Faster-MoE are 8% faster than HarMoEny in the Switch128 workload with no skew due to HarMoEny's increased overhead when scheduling experts for each batch. This overhead is visible as the number

of experts in Switch128 is high (128). The scheduling compute is not noticeable for Qwen, which uses 60 experts per MoE block. Furthermore, HarMoEny outperforms both DeepSpeed and ExFlow in terms of throughput and TTFT in the experiments with Qwen without any skew. The lower performance of DeepSpeed is due to the scheduling policy and that of ExFlow can be attributed to the inability to adapt the expert placement according to runtime skew.

Real-world datasets. Figure [7](#page-8-0) (right) and Figure [8](#page-8-1) (right) show the throughput and mean TTFT of HarMoEny, DeepSpeed, FastMoE, FasterMoE, and ExFlow running the Switch128 (top) and Qwen (bottom) models. The figures show the three real-world datasets and the random dataset described in Section [5.1.2.](#page-7-1) Notably, Har-MoEny maintains steady high throughput (201 tokens/s and 36 tokens/s for Switch128 and Qwen respectively) and low mean TTFT (5ms and 27ms for Switch128 and Qwen respectively) in all real-world datasets. Compared to ExFlow and FasterMoE, this results in a speedup of 20% and 7% on the random and wikitext datasets, respectively. DeepSpeed remains an outlier here with very low throughput and high TTFT. This is due to HarMoEny's lightweight load rebalancing mechanism and asynchronous prefetching that keep all GPUs operating at close to 100% (more details in Section [5.3.2\)](#page-9-0).

For the Switch transformer model, HarMoEny is on par with FasterMoE and FastMoE. Since the size of the experts is relatively small (18MB), FastMoE's and FasterMoE's expert shadowing mechanism is enough to ensure good load balancing because the GPU memory can comfortably host the shadowed experts. Fast-MoE and FasterMoE obtain virtually identical results, within 92% to 98% the throughput of HarMoEny across the four workloads. However, the throughput difference is accentuated for larger models. For Qwen (33MB per expert), the gap widens compared to Switch128. HarMoEny is 15% to 28% faster than FastMoE and FasterMoE because the extent of the expert shadowing is constrained by the GPU memory. ExFlow, while having consistent performance, falls behind HarMoEny for both models because it does not run the expert placement optimization often enough to keep up with expert popularity fluctuations.

Fluctuating expert popularity over time. Figure [9](#page-9-1) shows a scenario where each batch has a different randomly chosen expert popularity skew (between 0% and 50%). From the top row, it is evident that HarMoEny maintains high and steady throughput while the other baselines have drops in throughput by up to 37%. This scenario shows the effectiveness of HarMoEny's lightweight approach, which can sustain rapidly changing load imbalance without affecting throughput. Figure [9](#page-9-1) (bottom) shows the number of expert swaps in HarMoEny for every batch of input request. The dynamic number of expert swaps confirms that the expert pre-fetching mechanism running out of the critical path maintains stable high throughputs. From the zoomed-in version in Figure [9](#page-9-1) (right column), we can observe that baselines FastMoE, Faster-MoE, and ExFlow perform really well and FasterMoE surpasses HarMoEny in batches where HarMoEny swaps the least experts. These are batches where there is almost negligible load imbalance across the GPUs. In other words, the baselines work well when there is almost no skew in expert popularity, and hence, all GPUs

<span id="page-9-1"></span>![](_page_9_Figure_2.jpeg)

Figure 9: Throughput with 0%-50% skew randomly chosen every batch on SWITCH128. Right-hand zooms into a shorter interval. Swaps in HARMOENY maintain high throughput.

<span id="page-9-2"></span>![](_page_9_Figure_4.jpeg)

Figure 10: The throughput of HARMOENY and baselines across iterations (top) and skew in expert popularity (bottom), for the SWITCH128 (left) and QWEN (right) models. HARMOENY maintains consistent throughput while skew varies from 0% to 95% per batch.

perform equal computation. In such scenarios, the lower throughput of Harmoeny is due to the overhead of the token scheduler (see Section 4.2).

Figure 10 extends the analysis from Figure 9 by evaluating a broader range of expert skew values, ranging from 0% to 95%. This figure also includes results for QWEN, and the bottom row annotates the expert skew at each iteration. Harmoeny not only achieves the highest overall throughput but also demonstrates significantly lower variance across batches of the same size but differing skew levels. Specifically, Harmoeny shows a variance of  $152 \ toks^2/s^2$ 

compared to ExFlow's  $206\ toks^2/s^2$ , FasterMoE's  $447\ toks^2/s^2$ , and FastMoE's  $477\ toks^2/s^2$ . While DeepSpeed exhibits the lowest variance, this comes at the cost of throughput due to its input padding strategy, yielding only  $13\ tok/s$  versus Harmoeny's  $176\ tok/s$ . Therefore, this comparison is not entirely fair. For Qwen, Harmoeny achieves a larger variance reduction, with a variance of just  $0.59\ tok^2/s^2$ , significantly outperforming ExFlow's  $4.58\ toks^2/s^2$  ( $7.76\times$ ), FasterMoE's  $22.77\ toks^2/s^2$  ( $38.59\times$ ), and FastMoE's  $22.9\ toks^2/s^2$  ( $38.8\times$ ). In summary, when the expert popularity changes across batches, Harmoeny achieves a high throughput by dynamically adapting to the load imbalance.

## 5.3 Ablation study of HARMOENY

5.3.1 Time breakdown of Harmoeny components. We now conduct a time breakdown of the different operations when serving an MoE with Harmoeny. We adopt the Constant workload and assign 90% tokens to the first 10 experts ( $\alpha=0.9$ ). We use NVIDIA CUDA Events to obtain a fine-grained time breakdown of the operations in the first MoE layer. Figure 11 shows the duration of different operations in Harmoeny without any rebalancing (top), with rebalancing but without asynchronous expert loading (middle), and our original Harmoeny with all its components (bottom). Since the first 10 experts are loaded on GPU 0, we observe significant GPU waiting time for all other GPUs when we do not rebalance the token load using Algorithm 2, which is in line with the discussion in Section 3. Specifically, GPUs 1–7 spend on average 85.7% (QWEN) of the time waiting for GPU 0 to finish processing experts.

Our token rebalancing algorithm shrinks the waiting time on all GPUs. The mean waiting time goes from 82.6% and 85.7% of the total GPU time down to a mere 2.6% and 1% for SWITCH128 and QWEN, respectively across all GPUs, as seen in Figure 11 (middle). For the SWITCH128 model, token rebalancing reduces the mean layer latency from 289 ms to 149.5 ms, a total reduction of 48.3%. This reduction is even more pronounced with the QWEN model: 63.7% compared to when not rebalancing tokens. On average, our scheduling algorithm takes, 30.8% and 20.3% of the mean latency for the SWITCH128 and QWEN model, respectively. While scheduling and rebalancing in HARMOENY takes time, it brings a significant decrease in total latency.

Figure 11 (bottom) shows the timeline of operations of Harmoenv with all components enabled. Asynchronous expert loading further reduces the latency to 136.6 ms (-8.63% over synchronous loading) for Switch128 and to 141.8 ms (-13.8% over synchronous loading) for Qwen. Thus, we conclude that the combination of token rescheduling and asynchronous fetching in Harmoenv effectively minimizes the idling of GPUs and reduces the latency of MoE inference.

<span id="page-9-0"></span>5.3.2 Load balancing policies. We next experiment with HARMOENY and when using different token rebalancing policies and measure the throughput in tokens/s. We implement the token rebalancing policies on top of HARMOENY to account for differences in system baselines implementations.

We evaluate HARMOENY against three other policies for token routing: (1) With the Round-robin policy, tokens are sent to the GPU housing the specified expert, regardless of imbalance, with experts distributed to GPUs in a round-robin manner. This policy

<span id="page-10-0"></span>![](_page_10_Figure_2.jpeg)

Figure 11: Time breakdown of HARMOENY and baselines for the Switch128 (left) and Qwen (right) models.

<span id="page-10-1"></span>![](_page_10_Figure_4.jpeg)

Figure 12: Throughput († is better) for different token load policies and different skews (left) and datasets (right) when using the SWITCH128 (top) and QWEN models (bottom).

is employed by DEEPSPEED, FASTMOE, and FASTERMOE; (2) The EXFLOW policy utilizes an integer programming-based approach to optimize token scheduling and routing by modeling the inter-layer

<span id="page-10-2"></span>![](_page_10_Figure_7.jpeg)

Figure 13: Mean TTFT ( $\downarrow$  is better) for different token load policies and different skews (left) and datasets (right) when using the SWITCH128 (top) and QWEN models (bottom).

affinity between tokens and experts, formulating a placement optimization problem to minimize the communication cost of routing tokens between GPUs; (3) Even Split evenly distributes the tokens for each expert to each GPU and replicates all experts on all GPUs. For example, if there are a tokens for expert 0 and four GPUs then  $\frac{a}{4}$  tokens will be sent to each of the four GPUs. This achieves a perfect load balance across all experts at the cost of replication of all experts across all the GPUs.

**Skewed datasets.** Figure 12 (left) and Figure 13 (left) show the throughput and mean TTFT under different token load policies and different skew levels  $\alpha$ , for the two MoE models. In these experiments, we skew the load of a single expert. Harmoeny, when using Switch128 with  $\alpha=0$  (no skew), reaches a throughput of 213 tokens/s which is comparable to that of the Round-robin and ExFlow policies. However, as  $\alpha$  increases and consequentially, the load imbalance, Harmoeny reaches a significantly higher throughput than the other policies. For  $\alpha=0.9$ , Harmoeny reaches a throughput of 186 tokens/s compared to 106 tokens/s for ExFlow, the best-performing baseline. We observe similar trends when using the Qwen model (Figure 12, bottom left). Even though the even split policy achieves perfect load balance, its performance is relatively low. This is because each GPU has to load and execute every expert, thus increasing the time taken to process a batch of inference requests.

Figure 13 (left) shows that for  $\alpha=0$  and with the SWITCH128 model, Harmoeny has a mean TTFT of 4.68 ms, which is comparable to that of the Round-robin and ExFlow policies. However, when  $\alpha$  increases, so does the mean TTFT of other policies. With  $\alpha=0.9$  and with the SWITCH128 model, Harmoeny has a mean TTFT of 5.36 ms, compared to 9.77 ms and 9.38 ms for the Round-robin and ExFlow policies, respectively. The mean TTFT of Harmoeny is also competitive when using the Qwen model. Thus, Harmoeny exhibits excellent mean TTFT, even under heavy token imbalances.

Real-world datasets. Figure 12 (right) and Figure 13 (right) show the throughput and mean TTFT of Harmoeny and other policies for real-world datasets. For all datasets and models used, Harmoeny reaches the highest throughput. This is the most pronounced when using the random dataset and Switch128 model. Figure 13 (right) shows that Harmoeny exhibits the lowest mean TTFT, thus justifying the token distribution policy in Harmoeny.

#### 6 Related work

In this section, we discuss the related work and present the comparison with HARMOENY.

**Efficient MoE inference.** Recent works address the efficiency problem for MoE inference by optimizing communication overhead, token imbalance, and GPU kernels. DeepSpeed-MoE Inference [29] is a framework for serving MoE models providing flexible parallelization combinations, highly optimized MoE-related kernels, and an efficient communication subsystem. However, the static expert assignment in DeepSpeed-MoE struggles under skewed and dynamic workloads. Tutel improves upon DeepSpeed-MoE by introducing adaptive parallelism at runtime [14]. While Tutel handles dynamic workloads better than DeepSpeed-MoE, its static expert placement limits its ability to handle severe load imbalance. ExFlow reduces the all-to-all communication overhead by exploiting interlayer expert affinity [38]. While effective, it assumes stable expert routing patterns, which may not adapt well to fluctuating inputs. Lina [23] improves inference performance through dynamic resource scheduling to balance skewed workloads with a 2-phase scheme by profiling experts and predicting expert selection. Though effective in scenarios where inference requests are from similar domains, Lina needs to reallocate resources when the expert popularity changes. In contrast, HARMoENY directly targets load imbalance in skewed workloads through token redistribution and dynamic expert placement. In addition, frameworks like DeepSpeed-MII [26] and vLLM [20] are under active development and utilize highly optimized CUDA kernels targeted to specific GPU architecture. Our approach is orthogonal to these. HARMoEny works at the application layer and can be further optimized with specialized GPU kernels as in DeepSpeed-MII or vLLM.

Efficient MoE training. Several frameworks have been proposed to make training of MoE models efficient. FastMoE [12] introduces an MoE training system with hierarchical interfaces and optimized CUDA kernels, enabling scalability but relying on static expert placement. FasterMoE [13] builds on this by addressing load imbalance through dynamic shadowing and fine-grained scheduling, while introducing congestion-avoiding expert selection during training. Since the experts are sparsely activated, Megablocks [10] achieves hardware efficiency by combining expert computations into block-sparse operations. Similar to DeepSpeed-MoE, Smart-MoE [39] supports dynamic and hybrid parallelization strategies for MoE training. Finally, Prophet [34] utilizes a fine-grained planner and exploits similarity in token distribution across training iterations. Contrary to the aforementioned approaches, HARMOENY focuses on dynamic inference load imbalance rather than training.

#### 7 Conclusion

We presented Harmoeny, a novel system that addresses load imbalance in multi-GPU inference of MoE models. Through the combination of dynamic token rebalancing and asynchronous expert fetching, Harmoeny achieves near-perfect load balancing, significantly reducing inference latency. Our comprehensive evaluation using multiple datasets and MoE models demonstrated that Harmoeny outperforms state-of-the-art baselines. With heavy token imbalance, Harmoeny increases throughput by up to 70.1% and reduces time-to-first-token by up to 41.1%, compared to the next-best competitor, while maintaining stable throughput.

## <span id="page-11-1"></span>A Appendix: Model Statistics

<span id="page-11-2"></span>

| Model     | MoE Layers | Experts | Expert size (MB) |
|-----------|------------|---------|------------------|
| Switch128 | 12         | 128     | 18               |
| Qwen      | 24         | 60      | 33               |

Table 1: Specifications of the MoE models used in evaluation.

Table 1 shows the specifications of the two MoE models we have used in our evaluation.

## <span id="page-11-0"></span>B Appendix: Mathematical Estimate for q

Following is a step-by-step breakdown of getting a formula for estimating the minimal size for q given that the expert is a 2-layer MLP. q represents the number of tokens that are required so that its processing time is greater than the time it takes to load an expert. Let the expert have two linear layers with the first being of size  $m \times p$ , and the second being  $p \times m$ . The expert is evaluated as  $xW^1W^2$ . This can be represented as:

$$\frac{\text{Number of Floating Point Operations}}{\text{GPU FLOPS}} > \frac{\text{Expert Size}}{\text{PCIe Bandwidth}} \qquad (5)$$

$$\frac{|O|}{\phi} > \frac{|E|}{\beta} \qquad (6)$$

$$\frac{qp(2m-1) + qm(2p-1)}{\phi} > \frac{(mp+pm)d_{type}}{\beta} \qquad (7)$$

On ignoring small terms and simplifying, we get:

$$\frac{2qpm + 2qpm}{\phi} > \frac{2pm \cdot d_{type}}{\beta} \tag{9}$$

$$\frac{q \cdot 4pm}{\phi} > \frac{2pm \cdot d_{type}}{\beta} \tag{10}$$

$$q > \frac{\phi \cdot d_{type}}{2\beta} \tag{11}$$

(8)

## C Integrating HARMoENY

The following Python code demonstrates how to add Harmoenv to an existing model. The replace\_moe\_layer function injects our MoE implementation based on user-specified parameters.

```
from harmonymoe.utils import replace_moe_layer
from harmonymoe.moe_layer import MoEConfig, MoELayer
```

```
model = create_pytorch_model() # Custom model
     config = MoEConfig(
           rank,
           world_size,
           scheduling_policy,
           expert_cache_size,
10
           eq_tokens,
           d_model,
           num_experts,
14
     replace_moe_layer(
16
         model,
18
         moe_parent_type ,
19
         moe_type,
20
         path_to_experts,
         path_to_router_linear_layer,
         config.
22
23
```

#### References

- <span id="page-12-4"></span>Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 technical report. arXiv:2303.08774, 2023.
- <span id="page-12-6"></span> [2] Jeff Barr. Amazon EC2 update – inf1 instances with AWS inferentia chips for high performance cost-effective inferencing, 2019. Accessed: January 2025.
- <span id="page-12-22"></span>[3] Yoshua Bengio, Réjean Ducharme, and Pascal Vincent. A neural probabilistic language model. In NeurIPS, 2000.
- <span id="page-12-5"></span>[4] Ricardo Bianchini, Christian Belady, and Anand Sivasubramaniam. Datacenter power and energy management: past, present, and future. IEEE Micro, 2024.
- <span id="page-12-3"></span>[5] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. NeurIPS, 2020.
- <span id="page-12-27"></span>[6] Jeffrey Dean, Greg Corrado, Rajat Monga, Kai Chen, Matthieu Devin, Mark Mao, Marc' aurelio Ranzato, Andrew Senior, Paul Tucker, Ke Yang, Quoc Le, and Andrew Ng. Large scale distributed deep networks. In NeurlPS, volume 25, 2012.
- <span id="page-12-18"></span>[7] Christina Delimitrou and Christos Kozyrakis. Quasar: Resource-efficient and gos-aware cluster management. ACM Sigplan Notices, 49(4), 2014.
- <span id="page-12-9"></span>[8] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23(120), 2022.
- <span id="page-12-34"></span>[9] Wikimedia Foundation. Acl 2019 fourth conference on machine translation (wmt19), shared task: Machine translation of news.
- <span id="page-12-38"></span>[10] Trevor Gale, Deepak Narayanan, Cliff Young, and Matei Zaharia. Megablocks: Efficient sparse training with mixture-of-experts. In MLSys, 2023.
- <span id="page-12-24"></span>[11] Rohit Gupta, Laurent Besacier, Marc Dymetman, and Matthias Gallé. Character-based NMT with transformer. arXiv:1911.04997, 2019.
- <span id="page-12-14"></span>[12] Jiaao He, Jiezhong Qiu, Aohan Zeng, Zhilin Yang, Jidong Zhai, and Jie Tang. Fastmoe: A fast mixture-of-expert training system. arXiv:2103.13262, 2021.
- <span id="page-12-15"></span>[13] Jiaao He, Jidong Zhai, Tiago Antunes, Haojie Wang, Fuwen Luo, Shangfeng Shi, and Qin Li. Fastermoe: Modeling and optimizing training of large-scale dynamic pre-trained models. In PPoPP, 2022.
- <span id="page-12-35"></span>[14] Changho Hwang, Wei Cui, Yifan Xiong, Ziyue Yang, Ze Liu, Han Hu, Zilong Wang, Rafael Salas, Jithin Jose, Prabhat Ram, et al. Tutel: Adaptive mixture-of-experts at scale. MLSys, 2023.
- <span id="page-12-8"></span>[15] Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. Adaptive mixtures of local experts. Neural computation, 3(1), 1991.
- <span id="page-12-11"></span>[16] Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mixtral of experts. arXiv:2401.04088, 2024.
- <span id="page-12-1"></span>[17] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv:2001.08361, 2020.
- <span id="page-12-28"></span>[18] Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.
- <span id="page-12-23"></span>[19] Taku Kudo. Subword regularization: Improving neural network translation models with multiple subword candidates. In ACL, 2018.

- <span id="page-12-37"></span>[20] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with PagedAttention. In SOSP, 2023.
- <span id="page-12-7"></span>[21] George Leopold. AWS to offer nvidia's t4 GPUs for AI inferencing, 2019. Accessed: January 2025.
- <span id="page-12-10"></span>[22] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. In ICLR 2021
- <span id="page-12-16"></span>[23] Jiamin Li, Yimin Jiang, Yibo Zhu, Cong Wang, and Hong Xu. Accelerating distributed MoE training and inference with lina. In USENIX ATC, 2023.
- <span id="page-12-13"></span>[24] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv:2412.19437, 2024.
- <span id="page-12-33"></span>[25] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models, 2016.
- <span id="page-12-36"></span>[26] Microsoft. Deepspeed-mii: Mii makes low-latency and high-throughput inference possible, powered by deepspeed. https://github.com/microsoft/DeepSpeed-MII, 2022. Accessed: 2025-01-13.
- <span id="page-12-30"></span>[27] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. In NeurIPS, 2019
- <span id="page-12-31"></span>[28] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of machine learning research*, 21(140), 2020.
- <span id="page-12-20"></span>[29] Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, and Yuxiong He. DeepSpeed-MoE: Advancing mixture-of-experts inference and training to power next-generation ai scale. In ICML, 2022.
- <span id="page-12-26"></span>[30] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparselygated mixture-of-experts layer. In ICLR, 2017.
- <span id="page-12-29"></span>[31] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-LM: Training multi-billion parameter language models using model parallelism, 2020.
- <span id="page-12-19"></span>[32] Muhammad Tirmazi, Adam Barker, Nan Deng, Md E Haque, Zhijing Gene Qin, Steven Hand, Mor Harchol-Balter, and John Wilkes. Borg: the next generation. In EuroSys, 2020.
- <span id="page-12-21"></span>[33] A Vaswani et al. Attention is all you need. NeurIPS, 2017.
- <span id="page-12-40"></span>[34] Wei Wang, Zhiquan Lai, Shengwei Li, Weijie Liu, Keshi Ge, Yujie Liu, Ao Shen, and Dongsheng Li. Prophet: Fine-grained load balancing for parallel training of large-scale moe models. In *IEEE International Conference on Cluster Computing* (CLUSTER), 2023.
- <span id="page-12-25"></span>[35] Mengwei Xu, Dongqi Cai, Wangsong Yin, Shangguang Wang, Xin Jin, and Xuanzhe Liu. Resource-efficient algorithms and systems of foundation models: A survey. ACM Computing Surveys, 2024.
- <span id="page-12-12"></span>[36] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv:2412.15115, 2024.
- <span id="page-12-2"></span>[37] Jingfeng Yang, Hongye Jin, Ruixiang Tang, Xiaotian Han, Qizhang Feng, Haoming Jiang, Shaochen Zhong, Bing Yin, and Xia Hu. Harnessing the power of LLMs in practice: A survey on chatgpt and beyond. ACM Transactions on Knowledge Discovery from Data, 18(6), 2024.
- <span id="page-12-17"></span>[38] Jinghan Yao, Quentin Anthony, Aamir Shafi, Hari Subramoni, and Dhabaleswar K DK Panda. Exploiting inter-layer expert affinity for accelerating mixture-ofexperts model inference. In *IEEE IPDPS*, 2024.
- <span id="page-12-39"></span>[39] Mingshu Zhai, Jiaao He, Zixuan Ma, Zan Zong, Runqing Zhang, and Jidong Zhai. SmartMoE: Efficiently training Sparsely-Activated models through combining offline and online parallelization. In USENIX ATC, 2023.
- <span id="page-12-32"></span>[40] Yukun Zhu, Ryan Kiros, Richard Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In arXiv:1506.06724, 2015.