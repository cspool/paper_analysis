# *Remoe:* Towards Efficient and Low-Cost MoE Inference in Serverless Computing

Wentao Liu˚ , Yuhao Hu˚ , Ruiting Zhou˚ , Baochun Li: , Ne Wang; ˚School of Computer Science and Engineering, Southeast University, China :Department of Electrical and Computer Engineering, University of Toronto, Canada ;Department of Computing, The Hong Kong Polytechnic University, Hong Kong Email: ˚ (liuwentao, yuhaohu, ruitingzhou)@seu.edu.cn, :bli@ece.toronto.edu, ;newang@polyu.edu.hk

*Abstract*—Mixture-of-Experts (MoE) has become a dominant architecture in large language models (LLMs) due to its ability to scale model capacity via sparse expert activation. Meanwhile, serverless computing, with its elasticity and pay-per-use billing, is well-suited for deploying MoEs with bursty workloads. However, the large number of experts in MoE models incurs high inference costs due to memory-intensive parameter caching. These costs are difficult to mitigate via simple model partitioning due to inputdependent expert activation. To address these issues, we propose *Remoe*, a heterogeneous MoE inference system tailored for serverless computing. *Remoe* assigns non-expert modules to GPUs and expert modules to CPUs, and further offloads infrequently activated experts to separate serverless functions to reduce memory overhead and enable parallel execution. We incorporate three key techniques: (1) a Similar Prompts Searching (*SPS*) algorithm to predict expert activation patterns based on semantic similarity of inputs; (2) a Main Model Pre-allocation (*MMP*) algorithm to ensure service-level objectives (SLOs) via worstcase memory estimation; and (3) a joint memory and replica optimization framework leveraging Lagrangian duality and the Longest Processing Time (*LPT*) algorithm. We implement *Remoe* on Kubernetes and evaluate it across multiple LLM benchmarks. Experimental results show that *Remoe* reduces inference cost by up to 57% and cold start latency by 47% compared to state-ofthe-art baselines.

# I. INTRODUCTION

The rise of large language models (LLMs) has ushered in a new era of deep learning applications, enabling capabilities such as advanced text generation and context-aware understanding [\[1\]](#page-9-0), [\[2\]](#page-9-1). Among recent LLM architectures, the Mixture-of-Experts (MoE) model has emerged as a promising solution to scale model capacity without proportionally increasing inference computation. The foundational MoE architecture replaces a transformer's standard feed-forward network (FFN) with multiple expert FFNs and a gating network for token-to-expert routing [\[3\]](#page-9-2). This approach allows for building vastly larger and more capable models, as only a fraction of the model's total parameters (experts) are used for any given inference task. Meanwhile, serverless computing has gained traction as a cost-effective deployment paradigm for machine learning (ML) inference [\[4\]](#page-9-3), owing to its elasticity, fine-grained billing, and simplified resource management [\[5\]](#page-9-4).

Corresponding author: Ruiting Zhou (email: ruitingzhou@seu.edu.cn).

These features make it particularly attractive for LLM inference workloads that exhibit bursty traffic [\[6\]](#page-9-5).

However, the convergence of MoE models and serverless platforms is far from straightforward. Pricing for serverless computing is the product of the resources allocated to a function and its execution duration. While MoE's sparse expert activation is efficient, its vast number of experts introduces unique challenges under the serverless pricing model.

The primary challenge stems from the massive memory requirement of MoE models. Deploying the full model as a single serverless function typically requires loading all experts into memory, even if most are unused. This results in significant memory waste and high costs during inference, especially when expensive GPU memory is involved. To address the high memory occupation of MoE models, expert offloading has been widely studied [\[7\]](#page-9-6)–[\[10\]](#page-9-7), where most experts are cached on slower CPU memory, and only the predicted active experts are dynamically transferred to the GPU for inference. Existing offloading methods such as fMoE [\[7\]](#page-9-6) and HOBBIT [\[10\]](#page-9-7) implement dynamic expert swapping between the CPU and GPU through experts prefetching techniques. These approaches, however, still require a large, continuously provisioned memory pool on the CPU to hold the inactive experts. This persistent memory allocation fails to eliminate cost inefficiencies thus making existing solutions suboptimal for serverless MoE inference.

To mitigate the high memory costs, distributing experts across multiple serverless functions is a natural strategy. Unfortunately, this approach is complicated by the unbalanced and unpredictable nature of expert activation in MoE. In MoE inference, the activated experts depend heavily on the input prompt and vary across requests. Several studies [\[7\]](#page-9-6), [\[9\]](#page-9-8), [\[11\]](#page-9-9) have shown that for a single prompt, expert activation frequencies vary significantly, and this specialized pattern is difficult to predict due to the training method of the gating network [\[12\]](#page-9-10), [\[13\]](#page-9-11). It is challenging for serverless platforms to properly pre-allocate resources for these expert functions with unbalanced workloads. Current prediction methods, such as [\[9\]](#page-9-8) and [\[10\]](#page-9-7), rely on online, token-by-token predictions during inference. Such an approach is incompatible with serverless environments, because attempting to allocate resources dynamically would result in severe cold start overhead.

979-8-3315-4940-4/25/\$31.00 © 2025 IEEE Furthermore, this distributed approach introduces a funda-

mental trade-off. On one hand, deploying experts as multiple functions reduces memory usage per function, but incurs considerable latency due to the communication overhead it incurs. On the other hand, grouping experts into fewer, larger functions reduces the communication overhead, but may lead to memory inefficiency if inactive experts are loaded unnecessarily. While prior work [\[14\]](#page-9-12) simplified this by treating each expert as an independent function, it is impractical for modern MoEs. For instance, Deepseek-V3 [\[15\]](#page-9-13) contains thousands of experts (256 experts across 61 layers), and managing them as individual functions would create prohibitive deployment and management overhead. Consequently, determining an effective way to partition experts that balances cost and latency presents a major challenge in deploying MoE models in serverless environments.

To address such high costs, we present *Remoe*, a heterogeneous inference system that minimizes inference costs while satisfying service level objectives (SLOs). To our best knowledge, *Remoe* is the first work to systematically tackle cost-efficient MoE inference in a serverless setting. Highlights of our original contributions are as follows:

- ' A Heterogeneous MoE Architecture. We design a novel architecture that places non-expert modules on GPUs and expert modules on CPUs. Experts are further designated as local (co-located with the main model) or remote (deployed as separate serverless functions), significantly reducing the primary model's memory footprint and enabling parallel inference.
- ' Expert Prediction and Resource Pre-Allocation. We introduce a Similar Prompts Searching (*SPS*) algorithm to predict expert activations via a semantic clustering tree, and a Main Model Pre-allocation (*MMP*) algorithm to pre-allocate main model resources to meet performance SLOs with theoretical guarantees.
- ' Cost-Latency Optimization for Remote Experts. We formulate the configuration of remote experts as an optimization problem to balance cost and latency. We develop an efficient optimization framework based on the Lagrangian dual method and a Longest Processing Time (*LPT*) algorithm to determine memory specifications and expert replicas, supported by a formal convexity analysis.
- ' Prototype Implementation and Experiments. We implement a prototype of *Remoe* on Kubernetes. On multiple LLM datasets, our experiments show that *Remoe* reduces inference costs by up to 57.1% and significantly shortens cold start times compared to existing approaches.

# II. MOTIVATION

Partial expert activation. In a serverless context, billing is based on the amount of allocated resources and the execution time. This means that even if most of the experts are not activated, they still occupy memory and incur costs for the entire duration. An example is shown in Fig. [1.](#page-1-0) It is clear that whether an MoE model is deployed on a GPU or CPU, all of its experts incur charges for the entire runtime, even if Expert 1 and 3 are each activated just twice. Although expert offloading methods move most of the unused experts to CPUs, all experts still continuously consume memory. A lot of work [\[7\]](#page-9-6)–[\[10\]](#page-9-7) has shown that the activation frequencies of experts in MoE models differ markedly. To reduce MoE inference cost in a serverless setting, the key is to reduce the memory waste of these low-frequency experts.

<span id="page-1-0"></span>![](_page_1_Figure_9.jpeg)

Fig. 1: The runtime and charged duration of different deployment methods, and *expert offload* represents all expert offloading methods [\[7\]](#page-9-6)–[\[10\]](#page-9-7) which exchange experts to GPU during activation and offload the remaining experts to CPU.

Communication overhead between layers. One bottleneck of Serverless is the limit on the amount of data that can be communicated between functions, also known as the payload size. For example, AWS Lambda has a payload size limit of 6MB for data transmission. To transfer large amounts of data, an intermediary storage service like AWS S3 must be used, which introduces significant latency. For LLM inference, the data transferred between different layers are tokens and their data size is shown in Table [I.](#page-1-1)

<span id="page-1-1"></span>TABLE I: Token Size for different MoE models (Bfloat16)

| Model Name     | Parameters | Hidden Size | Token Size |
|----------------|------------|-------------|------------|
| Mixtral-8x7B   | 47B        | 4096        | 8 KB       |
| Mixtral-8x22B  | 141B       | 6144        | 12 KB      |
| Qwen2-57B-A14B | 57B        | 3584        | 7 KB       |
| DeepSeek-V2    | 236B       | 5120        | 10 KB      |
| DeepSeek-V3    | 671B       | 7168        | 14 KB      |
| Phi-4          | 14.7B      | 5120        | 10 KB      |

As we can see, the token size is much smaller than the payload size limit. According to previous work [\[10\]](#page-9-7), [\[16\]](#page-9-14), in lowoverhead environments (such as edge computing), requests are often single-batch. Therefore, only a few tokens are transferred between layers during the decoding, which fully meets the payload size limit. This observation makes it feasible to offload low-frequency MoE experts to separate serverless functions (model partitioning) without incurring latency overhead from intermediate storage.

Expert inference on the CPU. While deploying an entire MoE model on a CPU significantly increases inference latency, its components have varying computational demands. The attention layers are computationally intensive and benefit from GPU acceleration. In contrast, the expert modules are simpler, and since only a few are activated per token, they have lower computational needs. Numerous studies [\[17\]](#page-9-15), [\[18\]](#page-9-16) have already validated the feasibility of deploying these experts on CPUs. In a serverless environment where GPUs are much more expensive than CPUs, this enables a cost-saving heterogeneous strategy: run the computationally heavy modules on the GPU and offload the less demanding expert modules to the CPU. Therefore, combining CPU-GPU inference with model partitioning can theoretically reduce the inference cost of MoE models on serverless platforms.

#### III. SYSTEM MODEL

#### A. System Overview

In this section, we first consider a general Mixture of Experts (MoE) model. The model is composed of a preprocessing layer p, a set of intermediate layers  $\mathcal{H} = \{h_1, h_2, \ldots, h_L\}$  with length L, and a post-processing layer b. Each intermediate layer  $h_l = (\mathcal{F}_l, \mathcal{E}_l)$  consists of a non-expert module  $\mathcal{F}_l$  and an expert module  $\mathcal{E}_l$ . The non-expert module  $\mathcal{F}_l$  is typically composed of transformers and the gate. The expert module is represented as a list  $\mathcal{E}_l = \{e_{l,1}, e_{l,2}, \ldots, e_{l,K_l}\}$ , where  $e_{l,k}$  is the k-th expert in the l-th layer, and  $K_l$  is the total number of experts. For certain MoE architectures that share experts, such as DeepseekMoE [13], these shared experts are considered part of  $\mathcal{F}_l$  since they process all tokens.

For a request, the inference process of a MoE model can be divided into four stages: 1) **Pre-processing**: The raw natural language is tokenized and encoded by p and the resulting tokens are then passed to  $\mathcal{H}$ . 2) **Prefilling**: In each layer, all input tokens are processed by  $\mathcal{F}_l$  and  $\mathcal{E}_l$ . The gate routes each token to the most appropriate experts. Finally, the model outputs the most probable token, known as the *first token*. 3) **Decoding**: The *first token* is fed as input to  $\mathcal{H}$ , and the same computational process is repeated to produce the next token, continuing until all tokens are generated. 4) **Post-processing**: All generated tokens are sent to b, converted back into natural language, and then output.

To minimize the inference cost, we design a heterogeneous architecture for *Remoe*. The system overview is shown in Fig. 2. First, we pack all intermediate layers  $\mathcal{H}$  as an individual serverless function for inference (main model). The expert module  $\mathcal{E}_l$  runs on the CPU; other modules use the GPU. According to the activation frequency, we move some lowfrequency experts from the main model to extra serverless functions. For intermediate layer  $h_l$ , the low-frequency experts in  $\mathcal{E}_l$  will be allocated to the same extra function on CPU, and we call them "remote experts". The remote expert set of  $h_l$  is denoted as  $\mathcal{R}_l$ . In contrast, those high-frequency experts still remain in the main model, and we call them "local experts". This architecture significantly reduces the memory (GPU/CPU) overhead of the runtime container. Meanwhile, the local and remote experts can be computed in parallel, accelerating expert inference.

**Decision variables.** We introduce four decision variables: 1) Remote expert decision  $x_{l,k}$ .  $x_{l,k} = 1$  indicates that the expert  $e_{l,k}$  is designated as a remote expert. 2) Remote expert memory  $y_{l,v}$ . The set of all available memory specifications is denoted by  $\mathcal{M} = \{m_1, m_2, \ldots, m_V\}$ , where V is the total number of specifications.  $y_{l,v} = 1$  indicates that the memory specification v is allocated to the function holding the remote expert set  $\mathcal{R}_l$ . 3) Remote expert replicas  $z_l$ . Benefiting from the elastic scaling capabilities of serverless computing, multiple replicas can be instantiated to accelerate the expert inference

<span id="page-2-0"></span>![](_page_2_Figure_7.jpeg)

Fig. 2: System Overview

process.  $z_l$  is the replicas number of the functions for  $\mathcal{R}_l$ . 4) Main model memory  $w_v$ ,  $w_v=1$  indicates that the memory specification v is allocated to the main model. On common serverless platforms like AWS Lambda, users only need to set the memory allocation, and the platform automatically assigns corresponding vCPU resources. In this paper, we assume that 1 GB of memory corresponds to 1 vCPU.

# B. Inference Latency of Remoe

Since pre-processing and post-processing only involve fixed components and their overhead is typically negligible, we omit these stages and focus on the **Prefilling** and **Decoding** phases.

1) **Prefilling**: The total prefilling time can be presented as:

$$PT = \sum_{l=1}^{L} (PT_{l}^{f} + PT_{l}^{e})$$
 (1)

where  $PT_l^f = \tau_l^f(N^{in})$  is the prefilling time of non-expert module  $\mathcal{F}_l$ . Here,  $\tau_l^f(n)$  is the time for  $\mathcal{F}_l$  to process n tokens and  $N^{in}$  is the number of input tokens.  $PT_l^e$  is the prefilling time of the expert module  $\mathcal{E}_l$ , which can be expressed as:

$$\begin{split} PT_{l}^{e} &= \max[\sum_{k=1}^{K_{l}}(1-x_{l,k})PT_{l,k}^{loc}, \max_{j\leqslant z_{l}}\{ZT_{l,j}\}] + 2\tau^{sw}(N^{in}) \ \ (2) \\ \text{where} \sum_{k=1}^{K_{l}}(1-x_{l,k})PT_{l,k}^{loc} \ \text{and} \ \max_{j\leqslant z_{l}}\{ZT_{l,j}\} \ \text{are the end-} \end{split}$$

where  $\sum_{k=1}^{K_l} (1-x_{l,k}) PT_{l,k}^{loc}$  and  $\max_{j \leqslant z_l} \{ZT_{l,j}\}$  are the end-to-end latency of local experts and remote experts, respectively. We will describe these two parts later. Since expert modules are deployed on CPU, the data need to transfer between GPUs and CPUs twice.  $\tau^{sw}(N^{in})$  is used to denote the migration time of  $N^{in}$  tokens.

the migration time of  $N^{in}$  tokens. **Local Experts Latency**.  $PT_{l,k}^{loc} = \sum_{v=1}^{V} w_v \tau_{l,k,v}^c(N_{l,k}^{pre})$  denotes the prefilling time when  $e_{l,k}$  is local, where  $w_v$  is the memory specifications allocated to the main model's container, and  $\tau_{l,k,v}^c(N_{l,k}^{pre})$  represents the computation time for expert  $e_{l,k}$  to process  $N_{l,k}^{pre}$  tokens under memory specification v. The term  $N_{l,k}^{pre}$  is the total number of tokens routed to  $e_{l,k}$  during prefilling, calculated as the sum  $N_{l,k}^{pre} = \sum_{i=1}^{N^{in}} s_{l,k,i}$ .  $s_{l,k,i} = 1$  indicates the i-th input token is processed by  $e_{l,k}$ .

**Remote Experts Latency**. With  $x_{l,k}$ , the remote expert set can be denoted as  $\mathcal{R}_l = \{e_{l,k}|x_{l,k} = 1\}$ . Since we utilize function replicas to accelerate the remote expert inference, we split the remote expert set  $\mathcal{R}_l$  into  $\mathcal{R}_{l,1}, \mathcal{R}_{l,2}, \dots, \mathcal{R}_{l,z_l}$  and

each replica undertakes the computation of one subset. Different replicas execute simultaneously, so the end-to-end latency of the remote experts is  $\max_{j \leq z_l} \{ZT_{l,j}\}$ .  $ZT_{l,j}$  represents the

<span id="page-3-10"></span>

latency for the j-th replica. It is calculated as:  $ZT_{l,j} = \sum_{\substack{e_{l,k} \in \mathcal{R}_{l,j} \\ v \in \mathcal{N}_{l,k}}} (PT_{l,k}^{rem} + 2N_{l,k}^{pre}D/B) + t_l^{rem} \qquad (3)$  where  $PT_{l,k}^{rem} = \sum_{v=1}^{V-1} y_{l,v} \tau_{l,k,v}^c(N_{l,k}^{pre})$  is the prefilling time when  $e_{l,k}$  is remote.  $V^e$  is the total number of memory specifications for remote experts ( $V^e < V$ ). D is the size of a single token embedding and B is the network transfer rate. The term  $t_1^{rem}$  denotes the additional overhead introduced by the serverless invocation for remote experts of layer l (under warm-start conditions), which is a random variable dependent on the vCPU scheduling policy and resource contention.

2) **Decoding**: After the first token is generated, the model enters the Decoding stage. Let the total number of generated tokens be  $N^{out} + 1$  (including the first token). Decoding time can be expressed as:  $N^{in}+N$ 

where 
$$t_l^f$$
 is the single token's decoding time of  $\mathcal{F}_l$ .  $GT_{l,i}^e$  (4)

the decoding time of  $\mathcal{E}_l$  for token i; it can be calculated as:

<span id="page-3-8"></span>
$$GT_{l,i}^{e} = 2\tau^{sw}(N^{topk}) + \max\left[\sum_{k=1}^{K_{l}} (1 - x_{l,k}) s_{l,k,i} GT_{l,k}^{loc}, \sum_{k=1}^{K_{l}} x_{l,k} s_{l,k,i} (GT_{l,k}^{rem} + 2D/B + t_{l}^{rem})\right]$$
(5)

where  $N^{topk}$  is the number of experts each token is routed to.  $GT_{l,k}^{loc}$  and  $GT_{l,k}^{rem}$  are the decoding times of  $e_{l,k}$  when it is local and remote, respectively. The former is denoted as  $GT_{l,k}^{loc} = \sum_{v=1}^{V} w_v t_{l,k,v}^c$  where  $t_{l,k,v}^c$  is the time for expert  $e_{l,k}$  to process a single token under memory specification v.

Similarly,  $GT_{l,k}^{rem}$  is calculated as  $GT_{l,k}^{rem} = \sum_{v=1}^{V^e} y_{l,v} t_{l,k,v}^c$ .

3) **TTFT and TPOT**: For LLMs, SLOs are typically measured by Time-to-First-Token (TTFT) and Time-per-Output-Token (TPOT). In our model,  $T^{cold}$  is the cold start time, and TTFT can be expressed as  $T^{ttft} = PT + T^{cold}$ . Besides, TPOT is denoted as  $T^{tpot} = GT/N^{out}$ .

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

# <span id="page-4-1"></span>**Algorithm 1** Similar Prompts Searching (SPS)

```
Input: \alpha
1: Initialize clustering tree tree
   while new prompt prom arrives do
2:
3:
       \mathbb{PROM} = \Pi
       Select one leaf node leaf in tree
4.
5:
       Put samples from leaf into PROM
6:
       if len(\mathbb{PROM}) < \alpha then
          Turn to leaf's siblings and update leaf
7:
          Add samples into PROM until \alpha samples are obtained
8:
9.
       end if
       return PROM
10:
11: end while
```

SPS algorithm is outlined in Algorithm 1, where  $\mathbb{PROM}$  represents the set of top- $\alpha$  similar historical prompts searched currently. SPS initially builds the clustering tree tree (Line 1). For each new prompt, a leaf leaf is identified to retrieve similar prompts (Lines 2-5). If insufficient samples exist in leaf, its siblings are turned to (Lines 6-9). After acquiring  $\alpha$  historical prompts, the set  $\mathbb{PROM}$  is returned (Lines 10-11).

**Expert Activation Distribution Prediction.** For each historical prompt  $\zeta_j$ , we obtain its expert activation distribution matrix  $\tilde{S}$ . Matrix element  $\tilde{s}_{l,k} = \frac{frec_{l,k}}{\sum_k frec_{l,k}}$  represents the "linear scaling activation frequency" of expert  $e_{l,k}$  during prefilling of  $\zeta_j$ .  $frec_{l,k}$  is the times  $e_{l,k}$  is activated.  $\sum_k frec_{l,k}$  equals product of the number of  $\zeta_j$ 's tokens  $N_j^{in}$  and the number of experts activated by one token in each layer  $N^{topk}$ .

SCS between the new prompt and the retrieved  $\alpha$  historical prompts are converted into probability weights via soft-

max. The expert activation distribution matrices of historical prompts are then weighted-summed to predict the result.

# <span id="page-5-4"></span>C. Resource Pre-allocation for Main Model

To handle a cold start, *Remoe* pre-allocates memory for the main model as soon as a request arrives. This mechanism is separate from activation prediction (Sec. IV-B). This parallel approach is effective because the main model's pre-allocation can overlap with the pre-processing layer's cold start, which must complete before activation prediction can begin.

**Decoding Time Analysis.** We simplify Eq. (5) by removing the max function, assuming the remote expert path is always the performance bottleneck. This assumption is supported by two key observations. First, as shown in Fig. 4, the expert inference time increases nearly linearly with the ratio of remote experts. This indicates that, with the same vCPUs, remote experts dominate the inference time in Eq. (5). Second, in practical scenarios, the main model is typically allocated more vCPUs, ensuring faster computation for local experts.

<span id="page-5-0"></span>![](_page_5_Figure_4.jpeg)

Fig. 4: Expert inference time Fig. 5: Prefilling Time vs. Dewith 5 and 10 cores coding Time

<span id="page-5-1"></span>**Theorem 1.** When n tokens pass through layer l, the number of tokens processed by the k-th expert will not exceed  $\frac{\sqrt{3n}}{2} + \frac{n}{K_l}$  with a high probability (95%).

<span id="page-5-2"></span>**Corollary 1.** For n tokens and m experts, processed tokens will not exceed  $\frac{\sqrt{3n}}{2} + \frac{mn}{K_1}$  with a high probability (95%).

**Main Model Pre-allocation**. For the main model, we must pre-allocate a minimum memory specification that guarantees SLOs are met even in the worst-case scenario. Theorem 1 and Corollary 1 provide an upper bound in such a scenario. To this end, we design the Main Model Pre-allocation (*MMP*) algorithm detailed in Algorithm 2.

# <span id="page-5-3"></span>**Algorithm 2** Main Model Pre-allocation (MMP)

```
1: Initialize M^{min} = \sum_{l=1}^L \sum_{k=1}^{K_l} (1-x_{l,k}) \mu(e_{l,k}) + N^{max} D, 2: Initialize remote expert ratio b \leftarrow 1, M^{cal} \leftarrow m_{V^e}
 3: repeat
 4:
        for l=1 to L do
            Calculate the remote time based on Corollary 1 and b
 5:
 6:
        Calculate the memory of local experts M^e with \boldsymbol{b}
 7:
        Set the main model memory M \leftarrow \max(M^{min} + M^e, M^{cal})
 8.
 9:
        Calculate the TTFT and TPOT with M and b.
10:
        b \leftarrow b - \epsilon
11: until TTFT and TPOT limits are met
12: Select the minimum specification w_v that satisfies m_{w_v} \ge M
13: return w_v
```

First, MMP initializes the minimum memory  $M^{min}$  for non-expert modules caching. It also sets remote expert ratio b and  $M^{cal}$ , the minimum memory required to ensure local experts execute faster than remote ones (Lines 1-2). With

specific b, MMP first calculates the remote processing time of each layer based on Corollary 1 and b (Lines 4-6). This allows for the calculation of the worst-case remote inference latency. Then, MMP calculates the memory required to cache local experts for a given ratio b (Line 7). According to it, the main model memory is confirmed and  $M^{min} + M^e$  is the minimum memory to hold the parameters (Line 8). This process is repeated with decreasing values of b until both TTFT and TPOT are met (Lines 9-11). Finally, MMP returns the minimum specification  $w_v$ , such that  $m_{w_v} \ge M$  (Lines 12-13).

#### D. Remote Experts Selection

0.14

0.10

0.02

Given the expert activation matrix  $\tilde{S}$  and ratio b, we first calculate the expected number of tokens for each  $e_{l,k}$ . For the prefilling, this is  $E[N_{l,k}^{pre}] = N^{in} \tilde{s}_{l,k}$ , and for the decoding, it is  $E[N_{l,k}^{dec}] = N^{out} N^{topk} \tilde{s}_{l,k}$ . Our objective is to minimize latency for a given remote expert ratio, b, which is obtained in Sec. IV-C. To achieve this, we define a utility score  $u_{l,k} = E[N_{l,k}^{pre}] + E[N_{l,k}^{dec}]$  and choose the experts with the lowest utility scores to be remote. This selection is formally defined as choosing the set of remote experts,  $R_l$ , such that:  $\mathcal{R}_l = \arg\min_{\mathcal{R}_l} \sum_{e_{l,k} \in \mathcal{R}_l} u_{l,k}, |\mathcal{R}_l| = bK_l, \forall l$ . All  $x_{l,k}$  are set according to  $\mathcal{R}_l$ .

#### E. Remote Experts Memory Optimization

With  $w_v$  and  $X_l$ , the original problem transforms into an optimization problem of variables  $y_{l,v}$  and  $z_l$ .

We observe that: 1) Looser TTFT constraint: The expert replica decision variable  $z_l$  only exists in the prefilling stage. Due to the cold start time  $T^{cold}$ , the TTFT constraint is often looser than the TPOT constraint. 2) Longer decoding time: During the prefilling, each expert layer undergoes batch processing only once. Therefore, this stage is much shorter than decoding with multiple iterations [25]. Fig. 5 shows the prefilling/decoding times for different numbers of tokens.

**Problem Reformulation**. Based on these observations, the contribution of variable  $y_{l,v}$  to the optimization objective is considered to be concentrated in the decoding stage. Therefore, we can fix the prefilling time as a ratio of the decoding time to serve as an upper bound  $(PT \leq \eta GT)$ , and usually  $\eta \leq 0.1$  according to Fig. 5. After removing all constant values unrelated to y, the optimization objective for the memory allocation of remote experts can be expressed as:

$$\min_{y} P_{1} = (1+\eta) \sum_{l=1}^{L} \sum_{v=1}^{\tilde{V}^{e}} y_{l,v} (\tilde{s}_{l} T_{l,v}^{rem} + t_{l}^{rem}) (H^{w} + c^{c} m_{v})$$
 (12)

The remaining constraints are similar to those in Eq. (10) and are omitted here due to space limit. Here, the constant  $H^w = c^g M^g + c^c \sum_{v'=1}^V w_{v'} m_{v'}$  is the overhead per unit time of the main model.  $T_{l,v}^{rem} = N^{topk} t_{l,v}^c$  is the computation time for remote experts to decode all tokens (number here is  $N^{topk}$ ).  $\tilde{s}_l = \sum_{k=1}^{K_l} x_{l,k} \tilde{s}_{l,k}$  is the total probability of each token transferred to those remote experts.

**Function Construction and Fitting.** For this problem, the search space for memory size is large, and the solution complexity remains high. Therefore, we linearize the discrete

term  $\sum_{v=1}^{V^e} y_{l,v} m_v$  into a continuous variable  $\tilde{y}_l$ , where  $m_1 \leq$  $\tilde{y}_l \leqslant m_{V^e}$ . We consider that the inference time of remote experts gradually decreases as the allocated memory increases, eventually converging to a constant. To model this characteristic, we construct the formula  $\tilde{T}_l^{rem} = \theta_1 \exp(-\theta_2 \tilde{y}_l) + \theta_3$  $(\theta_1, \theta_2, \theta_3 > 0)$ . The parameters herein can be obtained by fitting data from model profiling, as illustrated by the fitted curve in Fig. 6 and the objective can be transformed into  $P_2$ :

$$\min_{y} P_{2} = (1+\eta) \sum_{l=1}^{L} \tilde{s}_{l} \left( \tilde{T}_{l}^{rem} + \frac{t_{l}^{rem}}{\tilde{s}_{l}} \right) \left( H^{w} + c^{c} \tilde{y}_{l} \right)$$
(13)

Although all integer terms have been relaxed into continuous ones, the objective function introduces non-linear terms such as  $H^w \tilde{T}_l^{rem}$  and  $c^c \tilde{y}_l \tilde{T}_l^{rem}$ , making it unsolvable by linear programming.

<span id="page-6-0"></span>![](_page_6_Figure_3.jpeg)

Fig. 6: Fitted Curves of CPU Resources vs. Inference Time

**Convexity Analysis.** To enable the subsequent optimization, we first perform the convexity analysis on the constructed functions and objective function.

<span id="page-6-1"></span>**Theorem 2.** Let  $g(\tilde{y}_l) = \left(\tilde{T}_l^{rem} + \frac{t_l^{rem}}{\tilde{s}_l}\right) (H^w + c^c \tilde{y}_l)$ . For  $\tilde{y}_l \in \left[\frac{2}{\theta_2} - \frac{H^w}{c^c}, \infty\right)$ , the function  $g(\tilde{y}_l)$  is strictly convex and continuously differentiable. And when  $\theta_2 \geqslant \frac{2c^c}{H^w}$ , the function  $g(\tilde{y}_l)$  is strictly convex on  $(0, \infty)$ .

For Theorem 2, we need to analyze whether different models satisfy this characteristic. As shown in Fig. 6, the values of  $\theta_2$  for GPT2-moe and Deepseek-v2-lite are 11.8665 and 2.4363, respectively. On commercial serverless platforms that support GPU resource allocation (e.g., Alibaba Cloud, Tencent Cloud), the overall cost standard for GPU is generally 3 times or more than that of CPU, i.e.,  $\frac{c^g}{c^c} \geqslant 3$ . Therefore, we have:  $\frac{2c^c}{H^w} = \frac{2}{c^g M^g/c^c + \sum_{v'=1}^V w_{v'} m_{v'}} \leqslant \frac{2}{3M^g + \sum_{v'=1}^V w_{v'} m_{v'}}$ . Here,  $M^g$  is the GPU memory overhead of the non-expert layers, and  $\sum_{v'=1}^V w_{v'} m_{v'}$  is the CPU memory overhead of the main model. For Deepseek-v2-lite, its non-expert layers have approximately 0.5B parameters, even if only 3GB of memory is allocated to the main model, we have  $\frac{2c^c}{H^w} \approx 0.25 \ll 2.4363$ . Under a similar analysis, when the main model retains only 12.5% of the experts as local, the value for GPT2-moe is  $\frac{2c^{\circ}}{H^{w}} \approx 2.72 \ll 11.8665$ . It can be seen that most MoE models conform to the aforementioned characteristic.

**Lagrangian Solving.** After analyzing the convexity of problem  $P_2$ , we give the dual problem of the primal problem

$$\begin{array}{ll} P_{2}, \text{ denoted as } P_{2}^{D} \colon \\ \max_{\lambda} & P_{2}^{D} = (1+\eta) \sum_{l=1}^{L} \tilde{s}_{l} g(\tilde{y}_{l}) + \sum_{j=1}^{4} \sum_{l=1}^{L} \lambda_{l,j} q_{l,j}^{c}(\tilde{y}_{l}) \\ \text{s.t.} & \lambda_{l,1}, \lambda_{l,2}, \lambda_{l,3}, \lambda_{l,4} \geqslant 0, \forall l \end{array} \tag{14a}$$

s.t. 
$$\lambda_{l,1}, \lambda_{l,2}, \lambda_{l,3}, \lambda_{l,4} \ge 0, \forall l$$
 (14b)

where  $q_{l,i}^c(\tilde{y}_l)$  represents the j-th constraint function in problem  $P_2$ , and  $\lambda_{l,j}$  is the corresponding dual variable. Thereinto,  $q_{l,1}^c(\tilde{y}_l)$  is the TPOT constraint and the rest are linear constraints on the range of  $\tilde{y}_l$ .

<span id="page-6-5"></span>**Lemma 1** (Slater's Condition). All constraints  $q_{l,j}^c(\tilde{y}_l)$  are convex, and when  $g(\tilde{y}_l)$  is strictly convex on  $(0, \infty)$ , problem  $P_2$  is a convex optimization problem and strong duality holds.

<span id="page-6-2"></span>**Theorem 3.** Let  $\tilde{y}^*$ ;  $\lambda^*$  be the solution to the dual problem  $P_2^D$  that satisfies the KKT conditions. Then  $\tilde{y}^*$  is also the optimal solution to the primal problem  $P_2$ .

According to Theorem 3, the problem  $P_2$  can be solved using the Lagrangian duality method, and the resulting remote expert memory  $y_{l,v}$  is the optimal solution for this problem.

## F. Remote Experts Multi-replicas Inference

1) Remote Expert Subsets Partitioning: In Eq. (3), we discussed partitioning the set  $\mathcal{R}_l$  into  $\mathcal{R}_{l,1}, \dots, \mathcal{R}_{l,z_l}$ . To minimize  $\max_{i} \{ZT_{l,i}\}\$ , we model the optimal partition as a Multiway Number Partitioning problem. An example is in Fig. 7.

<span id="page-6-3"></span>![](_page_6_Figure_17.jpeg)

Fig. 7: Multiway Number Partitioning problem and LPT Our objective is to assign tasks to different replicas, minimizing the completion time of all replicas. The subset  $\mathcal{R}_{l,1}$ correspond to remote expert tasks handled by replica 1. We use LPT algorithm to solve it. In simple terms, LPT sorts the tasks, and always selects the replica with the minimum load to assign tasks sequentially. The complexity of LPT is  $O(n \log n)$ , with an approximation ratio [26] of  $ZT^{max}$  $\left(\frac{4}{3} - \frac{1}{3z_l}\right)ZT^{OPT}$ . Furthermore, we can also prove an upper bound for  $\max_{j \leqslant z_l} \{ZT_{l,j}\}$ , as shown in Theorem 4.

<span id="page-6-4"></span>**Theorem 4.** Let  $T_{l,v}^{rem} = \sum_{k=1}^{K_l} \left( PT_{l,k}^{rem} + \frac{2D}{B} N_{l,k}^{pre} \right)$ , and  $N^{up}=\frac{\sqrt{3N^{in}}}{2}+\frac{N^{in}}{K_l}.$  Given  $z_l$  replicas, With a high probability (95%),  $\max_{j \leq z_l} \{ ZT_{l,j} \} \leq \frac{z_l - 1}{z_l} \left[ \sum_{v=1}^{V^e} y_{l,v} \tau_{l,v}^c(N^{up}) + \frac{2D}{B} N^{up} \right] + \frac{T_l^{rem}}{z_l} + t_l^{rem}$ 

2) Remote Expert Replicas Decision: Theorem 4 provides the worst-case prefilling time  $\max_{j \leq z_l} \{ZT_{l,j}\}$ , which enables us to optimize the replicas,  $z_l$ , to meet the TTFT constraint.

First, we initialize  $Z = (z_1, ..., z_L)$  to ensure each  $z_l$  meets the payload size. Then, for each layer, we calculate the current replica potential:

 $\overline{\omega}(l,Z) = \{C^{loc} + C^{rem}\}_{Z,z'_l = z_l} - \{C^{loc} + C^{rem}\}_{Z,z'_l = z_l + 1}$ (15)  $\{C^{loc} + C^{rem}\}_{Z,z'_l=z_l+1}$  represents the overall cost after  $z_l$ increases by 1. For the layer with the greatest replica potential,  $l^{max}$ , we let the replicas of layer add one and update Z. This process is repeated until the worst-case TPOT is satisfied. Finally, if  $\varpi(l, Z) > 0$  for some l, we continue to add replicas to reduce the overall cost until either  $\varpi(l, Z) \leq 0$  or  $z_l = z^{\max}$ for all l.

#### V. EVALUATION

#### <span id="page-7-0"></span>A. Settings

**Testbed.** We implemented a prototype of *Remoe* based on Kubernetes. It includes several key components: 1) To fit our inference framework, we modified all MoE models used in our experiments to support parallel inference with both local and remote experts. 2) We use the C++ LibTorch library and gRPC to provide efficient serverless inference services, minimizing data transfer overhead and response time. 3) Our Pod scheduler is NUMA-aware. The experimental platform is a server featuring a dual-socket configuration with two Intel Xeon Gold 6348 CPUs (totaling 56 cores, 112 threads). Furthermore, the server is equipped with two NVIDIA A100 GPUs, each providing 80 GB of VRAM.

**Dataset**. To ensure a comprehensive evaluation, our experiments are conducted on four widely-used datasets. These include: **LMSYS-Chat-1M** [22]: A dataset with 1M real-world conversations for evaluating chat and instruction-following abilities. **WikiText-2** [27]: A high-quality language modeling benchmark derived from Wikipedia articles. **C4** [28]: A massive, cleaned web-text corpus from Common Crawl, used for testing model generalization. **SlimPajama** [29]: A large-scale and high-quality dataset designed for model pre-training.

Models. We use two MoE models at different scales: 1) GPT2-moe: The original GPT2 model has 12 hidden layers and 124 million parameters. The FFN of each layer is converted into 8 experts and a gating network. Each token is routed to 2 experts per layer for inference (remote expert memory specifications: [200, 2000] MB; main model: [200, 5000] MB). 2) Deepseek-v2-lite: It has 27 hidden layers and 16 billion parameters. Each layer has 64 experts and 2 shared experts except the first dense layer. Each token is routed to 6 experts and 2 shared experts per layer (remote experts: [1000, 5000] MB; main model: [1000, 40000] MB). The step size for memory specifications is 100 MB.

# B. Prediction Accuracy

To evaluate the prediction performance of *Remoe*, we compare it with the following baselines: 1) **VarPAM.** Replace our customized k-medoids clustering with Partitioning Around Medoids algorithm [30]. 2) **VarED.** Replace our distance metric during clustering, which is semantic similarity, with Euclidean distance between expert activation distribution matrices. 3) **Distribution-Only Prediction (DOP)** [31]. Directly use the historical activation as prediction for new prompts. 4) **Fate** [32]. Predict expert activation per token using previous layer inputs. We adjust it by using the initial prompt embedding to predict activation across all layers for promptlevel prediction. 5) **Equal Frequency (EF).** Assume that the activation frequencies of all experts are equal to each other. 6) **Brute Force (BF).** Use brute-force searching to get top- $\alpha$  semantically similar historical prompts for the new one.

We randomly extract 5000 training and 500 test samples from each aforementioned dataset. Setting  $\alpha=15$  and  $\beta=150$ , Fig. 8 shows our method achieves the lowest average

JS Divergence (after VarPAM and BF) between predicted and true expert activation distributions. Partial truncation is applied to the y-axis in Fig. 8a. Crucially, VarPAM requires hours to build its clustering tree (versus ≤ 0.5 seconds for ours), and our semantically similar prompts searching method is more than 10 times faster than BF. DOP is only effective when new prompt's expert activation is similar to historical ones, and Fate uses inappropriate inputs for prediction of various layers' expert activation. While expert activation and semantic similarity generally correlate, using expert activation distribution directly for clustering (VarED) introduces noise, explaining our superior performance.

<span id="page-7-1"></span>![](_page_7_Figure_9.jpeg)

Fig. 8: JS Divergence under different datasets

# C. Overall Performance

To evaluate the overall performance of *Remoe*, we compare it with the following baselines: 1) CPU. Deploy the MoE on CPU. 2) GPU. Deploy the MoE on GPU. 3) Fetch. The ideal situation for all expert offloading methods [7]–[10]. It assumes that required experts are pre-loaded onto the GPU, with no mispredictions and no expert offloading/reloading time. 4) MIX. The expert modules are deployed on CPU, and other modules are deployed on GPU. The CPU and GPU memory are sufficient for modules caching.

We randomly sampled 50 tasks from the test set to serve as requests. For each request, we took the first 500 characters as the model input and set the number of output tokens to 200. Fig. 9 shows the cost of the two models under different baselines. For both models, *Remoe* achieves the lowest inference cost. It is observed that for the smaller MoE model (GPT2-moe), the cost difference among the methods is minor. For certain requests, Fetch even incurs lower cost than *Remoe*. However, for the larger model (Deepseek-v2-lite), the cost differences become significant, with Remoe achieving up to a 57.14% cost reduction. Among these methods, MIX shows lower costs than GPU and CPU, demonstrating that a heterogeneous model can substantially decrease inference overhead. Meanwhile, although Fetch can theoretically achieve optimal performance, it still requires caching all experts in memory and needs additional GPU memory for loading partial experts. This characteristic introduces extra costs.

# D. Cost under Different Prefilling/Decoding Ratios

In real-world scenarios, the number of tokens in the decoding phase often exceeds that in the prefilling phase. Therefore, we study the trend of inference cost under different ratios of

<span id="page-8-0"></span>![](_page_8_Figure_0.jpeg)

Fig. 9: Overall performance under 50 requests

prefilling to decoding tokens, as shown in Fig. 11. Across various ratios, *Remoe* maintains stable performance. For GPT2-moe, as the number of decoding tokens increases, CPU's cost gradually surpasses that of other methods. Although deploying the model on CPU saves memory overhead, the longer inference time clearly negates this advantage. In contrast, for Deepseek-v2-lite, GPU's cost is significantly higher than other methods in all cases. This is because larger MoE models lead to more memory waste on low-frequency experts, especially for GPUs with higher pricing.

![](_page_8_Figure_3.jpeg)

Fig. 10: Cost under different prefilling/decoding ratios

# E. Cold Start and Algorithm Overhead

Cold start is a critical issue in serverless computing. As shown in Fig. 11, we compare the cold start times across different methods. While all approaches share the same container startup time due to a common base image, *Remoe* achieves the lowest cold start time, with a reduction of up to 57.14%. This improvement stems from its strategy of partitioning numerous experts into separate serverless functions, whose cold starts (labeled as REMOTE) can overlap with the main model's startup. Furthermore, Remoe's optimization logic (CALCULATE) is highly efficient; its overhead is negligible and introduces no additional waiting time.

## VI. RELATED WORK

Serverless LLM Inference. Research on serverless LLM inference has focused on several key optimizations. To mitigate the cold start problem, techniques such as pipeline parallelism [33] and multi-tiered local storage [6] have been explored to accelerate model loading. Another key focus is resource allocation, where efforts include using elastic hardware sharing to boost GPU utilization [34] and combining adaptive configuration with real-time monitoring for stable serving [35]. For cost optimization, Liu et al. [14] proposed a specific scheduling algorithm ODS for serverless MoE inference, although limited to a pure CPU environment. Despite these advances, cost-efficient serverless MoE inference, particularly on GPU-CPU hybrid architectures, remains largely underexplored.

<span id="page-8-1"></span>![](_page_8_Figure_9.jpeg)

Fig. 11: Time for cold start, predicting, and optimization

GPU Memory-Constrained MoE Inference. Prediction-based expert caching is the dominant approach for memory-efficient Mixture-of-Experts (MoE) inference. Strategies range from using historical data [7], [8] to more fine-grained, layer-level predictions, which have been successfully applied to memory-constrained devices with enhancements like mixed-precision loading [10] and graceful degradation [36]. Another line of work employs dedicated ML predictors to achieve higher caching accuracy [9], [37]. While effective, these token-level online prediction strategies are ill-suited for serverless environments that require resource pre-allocation, as the frequent adjustments would incur severe cold start overhead during execution.

## VII. CONCLUDING REMARKS

To minimize inference cost, we propose a heterogeneous system *Remoe*. We design algorithms for expert activation prediction, resource pre-allocation, and joint memory-replica optimization. Our implementation of *Remoe* on Kubernetes shows that it reduces inference cost and cold start latency significantly. Our current approach relies on idealized assumptions about the serverless environment. Consequently, our future work will focus on designing a highly fault-tolerant system to address real-world operational complexities such as unpredictable cold start times and network latency fluctuations.

#### APPENDIX

# A. Proof of Theorem 1

*Proof.* We prove it based on Hoeffding's inequality. Due to space limit, all minor proofs (including Corollary 1, Lemma 1, Theorem 3, 4) are deferred to the technical report [38].

# B. Proof of Theorem 2

*Proof.* First, taking the derivative of  $g(\tilde{y}_l)$ , we can obtain:  $g'(\tilde{y}_l) = (c^c\theta_1 - c^c\theta_1\theta_2\tilde{y}_l - H^w\theta_1\theta_2)\exp(-\theta_2\tilde{y}_l) + c^c(\theta_3 + \frac{t_l}{\tilde{s}_l})$  The second derivative is:

$$g''(\tilde{y}_l) = c^c \theta_1 \theta_2^2 \exp(-\theta_2 \tilde{y}_l) [\tilde{y}_l - (\frac{2}{\theta_2} - \frac{H^w}{c^c})]$$

Since  $c^c \theta_1 \theta_2^2 \exp(-\theta_2 \tilde{y}_l) > 0$ ,  $g''(\tilde{y}_l)$  is a monotonically increasing function. Its zero point is  $\tilde{y}_l = \frac{2}{\theta_2} - \frac{H^w}{c^c}$ . Therefore, when  $\tilde{y}_l \geqslant \frac{2}{\theta_2} - \frac{H^w}{c^c}$ ,  $g''(\tilde{y}_l) \geqslant 0$ , and the function is convex. Meanwhile, since  $\frac{2}{\theta_2}$  is convex on  $(0,\infty)$  and  $\frac{H^w}{c^c}$  is constant, the function  $\frac{2}{\theta_2} - \frac{H^w}{c^c}$  is also convex on this interval. Furthermore, since  $\frac{d}{d\theta_2} \left( \frac{2}{\theta_2} - \frac{H^w}{c^c} \right) = -\frac{2}{\theta_2^2} < 0$ , the term  $\frac{2}{\theta_2} - \frac{H^w}{c^c}$  is monotonically decreasing in its domain. Therefore, when  $\theta_2 \geqslant \frac{2c^c}{H^w}$ , it implies that  $\frac{2}{\theta_2} - \frac{H^w}{c^c} \leqslant 0 < \tilde{y}_l$ , ensuring  $g(\tilde{y}_l)$  is strictly convex on  $(0,\infty)$ .

## REFERENCES

- <span id="page-9-0"></span> F. Barreto, L. Moharkar, M. Shirodkar, V. Sarode, S. Gonsalves, and A. Johns, "Generative artificial intelligence: Opportunities and challenges of large language models," in *Proc. of ICICN*. Springer, 2023, pp. 545–553.
- <span id="page-9-1"></span>[2] X. Ma, G. Fang, and X. Wang, "Llm-pruner: On the structural pruning of large language models," Advances in neural information processing systems, vol. 36, pp. 21702–21720, 2023.
- <span id="page-9-2"></span>[3] B. Lin, Z. Tang, Y. Ye, J. Cui, B. Zhu, P. Jin, J. Huang, J. Zhang, Y. Pang, M. Ning et al., "Moe-llava: Mixture of experts for large vision-language models," arXiv preprint arXiv:2401.15947, 2024.
- <span id="page-9-3"></span>[4] J. Duan, S. Qian, D. Yang, H. Hu, J. Cao, and G. Xue, "Mopar: A model partitioning framework for deep learning inference services on serverless platforms," arXiv preprint arXiv:2404.02445, 2024.
- <span id="page-9-4"></span>[5] Y. Li, Y. Lin, Y. Wang, K. Ye, and C. Xu, "Serverless computing: state-of-the-art, challenges and opportunities," *IEEE Transactions on Services Computing*, vol. 16, no. 2, pp. 1522–1539, 2022.
- <span id="page-9-5"></span>[6] Y. Fu, L. Xue, Y. Huang, A.-O. Brabete, D. Ustiugov, Y. Patel, and L. Mai, "{ServerlessLLM}:{Low-Latency} serverless inference for large language models," in *Proc. of OSDI*, 2024, pp. 135–153.
- <span id="page-9-6"></span>[7] H. Yu, X. Cui, H. Zhang, and H. Wang, "fmoe: Fine-grained expert offloading for large mixture-of-experts serving," arXiv preprint arXiv:2502.05370, 2025.
- <span id="page-9-34"></span>[8] L. Xue, Y. Fu, Z. Lu, C. Sun, L. Mai, and M. K. Marina, "Moe-infinity: Efficient moe inference on personal machines with sparsity-aware expert cache," 2025.
- <span id="page-9-8"></span>[9] X. Song, Z. Zhong, R. Chen, and H. Chen, "Promoe: Fast moe-based llm serving using proactive caching," arXiv preprint arXiv:2410.22134, 2024.
- <span id="page-9-7"></span>[10] P. Tang, J. Liu, X. Hou, Y. Pu, J. Wang, P.-A. Heng, C. Li, and M. Guo, "Hobbit: A mixed precision expert offloading system for fast moe inference," arXiv preprint arXiv:2411.01433, 2024.
- <span id="page-9-9"></span>[11] V. Gupta, K. Sinha, A. Gavrilovska, and A. P. Iyer, "Lynx: Enabling efficient moe inference through dynamic batch-aware expert selection," arXiv preprint arXiv:2411.08982, 2024.
- <span id="page-9-10"></span>[12] M. Abdin, J. Aneja, H. Behl, S. Bubeck, R. Eldan, S. Gunasekar, M. Harrison, R. J. Hewett, M. Javaheripi, P. Kauffmann et al., "Phi-4 technical report," arXiv preprint arXiv:2412.08905, 2024.
- <span id="page-9-11"></span>[13] D. Dai, C. Deng, C. Zhao, R. Xu, H. Gao, D. Chen, J. Li, W. Zeng, X. Yu, Y. Wu et al., "Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models," arXiv preprint arXiv:2401.06066, 2024.
- <span id="page-9-12"></span>[14] M. Liu, W. Wang, and C. Wu, "Optimizing distributed deployment of mixture-of-experts model inference in serverless computing," in *Proc.* of INFOCOM. IEEE, 2025, pp. 1–10.
- <span id="page-9-13"></span>[15] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan et al., "Deepseek-v3 technical report," arXiv preprint arXiv:2412.19437, 2024.
- <span id="page-9-14"></span>[16] R. Hwang, J. Wei, S. Cao, C. Hwang, X. Tang, T. Cao, and M. Yang, "Pre-gated moe: An algorithm-system co-design for fast and scalable mixture-of-expert inference," in *Proc. of ISCA*. IEEE, 2024, pp. 1018– 1031.
- <span id="page-9-15"></span>[17] KVCache-AI, "Ktransformers: A flexible framework for experiencing cutting-edge llm inference optimizations," https://github.com/kvcache-ai/ktransformers, 2024.
- <span id="page-9-16"></span>[18] S. Zhong, Y. Sun, L. Liang, R. Wang, R. Huang, and M. Li, "Hybrimoe: Hybrid cpu-gpu scheduling and cache management for efficient moe inference," arXiv preprint arXiv:2504.05897, 2025.

- <span id="page-9-17"></span>[19] M. Yu, Z. Jiang, H. C. Ng, W. Wang, R. Chen, and B. Li, "Gillis: Serving large neural networks in serverless functions with automatic model partitioning," in *Proc. of ICDCS*. IEEE, 2021, pp. 138–148.
- <span id="page-9-18"></span>[20] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," Advances in neural information processing systems, vol. 30, 2017.
- <span id="page-9-19"></span>neural information processing systems, vol. 30, 2017.

  [21] C. Helmberg and F. Rendl, "Solving quadratic (0, 1)-problems by semidefinite programs and cutting planes," *Mathematical programming*, vol. 82, no. 3, pp. 291–315, 1998.
- <span id="page-9-20"></span>[22] L. Zheng, W.-L. Chiang, Y. Sheng, T. Li, S. Zhuang, Z. Wu, Y. Zhuang, Z. Li, Z. Lin, E. P. Xing et al., "Lmsys-chat-1m: A large-scale real-world llm conversation dataset," arXiv preprint arXiv:2309.11998, 2023.
- <span id="page-9-21"></span>[23] L. Lee, "On the effectiveness of the skew divergence for statistical language analysis," in *International workshop on artificial intelligence* and statistics. PMLR, 2001, pp. 176–183.
- <span id="page-9-22"></span>[24] P. Sitikhu, K. Pahi, P. Thapa, and S. Shakya, "A comparison of semantic similarity methods for maximum human interpretability," in *Proc. of AITB*, vol. 1. IEEE, 2019, pp. 1–4.
- <span id="page-9-23"></span>[25] Z. Liu, J. Wang, T. Dao, T. Zhou, B. Yuan, Z. Song, A. Shrivastava, C. Zhang, Y. Tian, C. Re et al., "Deja vu: Contextual sparsity for efficient llms at inference time," in *Proc. of ICML*. PMLR, 2023, pp. 22137– 22176.
- <span id="page-9-24"></span>[26] R. L. Graham, "Bounds for certain multiprocessing anomalies," *Bell system technical journal*, vol. 45, no. 9, pp. 1563–1581, 1966.
- <span id="page-9-25"></span>[27] S. Merity, C. Xiong, J. Bradbury, and R. Socher, "Pointer sentinel mixture models," arXiv preprint arXiv:1609.07843, 2016.
- <span id="page-9-26"></span>[28] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu, "Exploring the limits of transfer learning with a unified text-to-text transformer," *Journal of machine learning* research, vol. 21, no. 140, pp. 1–67, 2020.
- <span id="page-9-27"></span>[29] Hugging Face, https://huggingface.co/datasets/cerebras/ SlimPajama-627B, June 2023.
- <span id="page-9-28"></span>[30] L. Kaufman and P. J. Rousseeuw, Finding groups in data: an introduction to cluster analysis. John Wiley & Sons, 2009.
- <span id="page-9-29"></span>[31] H. Ma, Z. Du, and Y. Chen, "Moe-gps: Guidlines for prediction strategy for dynamic expert duplication in moe load balancing," arXiv preprint arXiv:2506.07366, 2025.
- <span id="page-9-30"></span>[32] Z. Fang, Z. Hong, Y. Huang, Y. Lyu, W. Chen, Y. Yu, F. Yu, and Z. Zheng, "Accurate expert predictions in moe inference via cross-layer gate," arXiv e-prints, pp. arXiv-2502, 2025.
- <span id="page-9-31"></span>[33] C. Lou, S. Qi, C. Jin, D. Nie, H. Yang, X. Liu, and X. Jin, "To-wards swift serverless llm cold starts with paraserve," arXiv preprint arXiv:2502.15524, 2025.
- <span id="page-9-32"></span>[34] C. Xu, Z. Li, Q. Chen, H. Zhao, and M. Guo, "Llm-mesh: Enabling elastic sharing for serverless llm inference," arXiv preprint arXiv:2507.00507, 2025.
- <span id="page-9-33"></span>[35] T. Huang, P. Chen, K. Gong, J. Hawk, Z. Bright, W. Xie, K. Huang, and Z. Ji, "Enova: Autoscaling towards cost-effective and stable serverless llm serving," arXiv preprint arXiv:2407.09486, 2024.
- <span id="page-9-35"></span>[36] Y. Zhang, S. Aggarwal, and T. Mitra, "Daop: Data-aware offloading and predictive pre-calculation for efficient moe inference," in 2025 Design, Automation & Test in Europe Conference (DATE). IEEE, 2025, pp. 1–7
- <span id="page-9-36"></span>[37] X. He, S. Zhang, Y. Wang, H. Yin, Z. Zeng, S. Shi, Z. Tang, X. Chu, I. Tsang, and O. Y. Soon, "Expertflow: Optimized expert activation and token allocation for efficient mixture-of-experts inference," arXiv preprint arXiv:2410.17954, 2024.
- <span id="page-9-37"></span>[38] Anonymous, "Technical report," 2025, the technical report will be made publicly available upon acceptance of the manuscript due to the submission policy.