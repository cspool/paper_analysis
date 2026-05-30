# I. INTRODUCTION

Recently, Mixture-of-Experts (MoE) large language models (LLMs) have gained popularity in achieving state-of-theart performance while maintaining computational efficiency, making them increasingly pivotal in real-world applications, such as language modeling, machine translation, and image recognition [1]–[5]. To support their deployment, modern inference systems like vLLM and DeepSpeed-FastGen have emerged, showcasing remarkable capabilities in managing the complexity of MoE models [6]–[8] . However, these systems predominantly rely on static parallelization strategies, such as Tensor Parallelism (TP) and Expert Parallelism (EP) [9], which suffer from critical limitations in adapting to dynamic inference scenarios.

We identify three fundamental challenges associated with static parallel strategies, in aspects of computation, communication, and model architecture. 1) Unflexible tensor partition. Static strategies may result in suboptimal performance in some inference scenarios, as the fixed tensor partition fails to fully leverage the computational capabilities of the hardware for specific operators. Conversely, dynamically switching parallel strategies during inference incurs significant communication overhead, potentially offsetting any prospective performance improvements. 2) Mismatched bandwidth utilization of communication. Parallel strategies inherently impose distinct communication patterns, such as All-to-All for EP while AllReduce for TP. Static optimization strategies cannot dynamically adapt to these distinct pattern requirements, potentially exacerbating communication overhead, especially in heterogeneous hardware environments. 3) Inefficiency in addressing diverse MoE architectures. MoE models with different expert configurations, such as varying expert counts and distinct shared expert implementations, require scenariospecific computation/communication patterns. Static strategies may struggle to adapt to model-specific configurations, leading to suboptimal performance.

To address these limitations, we propose Hybrid Adaptive Parallelism (HAP), a novel method that automatically selects hybrid parallel strategies tailored to both model architectures and hardware capabilities. HAP introduces two key innovations: Module Decomposition and Hybrid Adaptive Parallel Strategies. Module Decomposition involves decomposing MoE models into Attention and Expert modules, each equipped with a dedicated inference latency simulation model to enable fine-grained inference simulation. Hybrid Adaptive Parallel Strategies constructs a hierarchical search space and leverages Integer Linear Programming (ILP) to dynamically identify the optimal combination of parallel strategies for Attention and Expert modules while adhering to hardware constraints.

Our contributions are summarized as follows:

- We develop module-specific simulation models that precisely capture computational and communication costs, enabling fine-grained performance prediction.
- We design a novel dynamic parallelism transition strategy to minimize the overhead associated with switching parallel configurations during inference.
- HAP formulates parallel strategy selection as an ILP problem, efficiently exploring a combinatorial search space to maximize device utilization and minimize end-

- to-end latency.
- Evaluations across diverse MoE architectures and GPU platforms demonstrate the generalization capability of our HAP, which achieves up to 1.77× speedup when integrated with DeepSpeed-FastGen.

### II. RELATED WORK

#### A. MoE Model Architectures

As illustrated in Fig. 1 (b), MoE models replace the standard feed-forward network (FFN) module with an Expert module comprising multiple parallel expert networks and a gating network, which employs a Top-K selection mechanism to dynamically dispatch input tokens to the most relevant experts [10], [11]. Notably, some advanced MoE models introduce shared experts [4], [12], [13], which simultaneously reduce parameter redundancy and enhance cross-layer knowledge transfer, as shown in Fig. 1(c). During inference, MoE models activate only a small subset of experts per token, thereby decoupling the growth of FLOPs from the parameter count.

![](_page_1_Figure_5.jpeg)

Fig. 1: (a) The typical dense transformer layer consists of an Attention module followed by a Feed-Forward Network (FFN) module. (b) The naive MoE transformer layer replaces the FFN module with an Expert module. (c) The MoE Transformer layer with shared experts incorporates experts that are always activated, known as shared experts.

### B. Generative Inference

Generative inference consists of two distinct computational stages: the prefill stage and the decoding stage, each exhibiting unique performance characteristics. The prefill stage processes the entire input prompt, exhibiting compute-bound characteristics due to its intensive tensor computations. Conversely, the decoding stage generates tokens sequentially, with each token generation dependent on previously generated tokens. This stage typically involves frequent loading of model weights and KV cache, leading to memory-bound behavior.

### C. Static Parallelism

a) **Data Parallelism**: (DP) replicates the entire model across devices while partitioning input data batches. In batch-oriented inference scenarios, DP eliminates communication

- overhead, enabling near-linear throughput scaling across distributed systems. However, the replication of models results in redundant memory usage, which consequently necessitates the integration of complementary parallelism strategies to alleviate the memory constraints.
- b) Tensor Parallelism: (TP) partitions model weights across multiple devices, enabling the deployment of large language models (LLMs) that exceed the memory capacity of a single device. It is widely supported by most mainstream inference frameworks, such as TensorRT-LLM, vLLM, Deepspeed-FastGen, and SGLang [6], [7], [14], [15], making it the most commonly adopted static parallelism approach in LLM inference. However, TP incurs significant communication overhead, which can degrade throughput and scalability [16].
- c) Expert Parallelism: (EP) distributes experts across multiple devices, achieving superior scalability and computational efficiency [17]. However, the frequent communication may become a performance bottleneck during inference. Furthermore, EP may lead to load imbalance, where certain devices are overutilized while others remain underutilized, reducing overall resource efficiency. Nowadays, DeepSpeed-MoE employs a static hybrid parallel strategy integrating DP, TP, and EP, demonstrating superior performance in large-scale MoE model inference [18], [19]. DeepEP is tailored for MoE inference with static EP, offering high-throughput and low-latency all-to-all GPU kernels, thereby achieving efficient inference [20].
- d) **Pipeline Parallelism**: (PP) distributes layers across multiple devices for parallel computation, thereby achieving superior scalability. However, this approach introduces pipeline bubbles that can significantly degrade inference throughput. Given the inherent performance limitation, we deliberately exclude pipeline parallelism from the architectural search space in our work.

## D. The adaptive parallelism

Alpa [21] leverages ILP to systematically identify optimal intra-node parallelism configurations, thereby providing an effective adaptive methodology. However, this approach exhibits significant limitations by exclusively focusing on communication latency while disregarding computational latency during its search process. Consequently, the derived parallelization strategies are not applicable for inference tasks, which is not selected as comparisons for our work. Moreover, Tutel [22] introduces a switchable parallelism mechanism specifically designed for training scenarios, which meticulously analyzes the communication complexity of various parallel strategies within the Expert module and implements the most optimal strategy. Nevertheless, Tutel cannot achieve zero-cost switchable parallelism during inference and fail to implement appropriate parallel strategies for both the Attention and Expert modules simultaneously.

### III. METHODOLOGY

## A. Analysis of MoE Inference

1) Runtime Breakdown of MoE Inference.: Execution time (i.e., latency) is a critical metric in model inference, as shorter

![](_page_2_Figure_0.jpeg)

Fig. 2: The per-layer latency breakdown during both prefill and decoding stages of Mixtral-8x7B inference, evaluated under TP and EP configurations across 4 NVIDIA A6000 GPUs.

latency ensures more timely service responses. In this subsection, we present a detailed performance breakdown of MoE inference with respect to latency. The evaluation is conducted on a node equipped with 4 A6000 GPUs, with a sequence length set to 2K. We compare the execution times of both the prefill stage and the decoding stage under various parallel strategies.

We decompose the inference process into three principal components: the computation of the Attention module, the computation of the Expert module, and collective communication. As illustrated in Fig. 2, the latency differences between the TP and EP strategies are primarily driven by the computational cost of the Expert module and the overhead associated with collective communication. Specifically, intranode communication is executed via PCIe [23], which is characterized by limited bandwidth, resulting in high sensitivity to data volume. Consequently, during the prefill stage, TP incurs higher communication latency compared to EP due to its larger communication volume, resulting in degraded inference performance. In the decoding stage, where the communication volume is generally minimal, the load imbalance introduced by EP leads to inefficient computation of the Expert module compared to TP.

These observations suggest that when selecting parallel strategies for the Expert module, scenarios with low intranode communication bandwidth, where communication constitutes a performance bottleneck, should favor strategies with lower communication volume, whereas environments with high intra-node communication bandwidth (e.g., A100 GPU nodes utilizing NVLink) [24] should adopt strategies that prioritize computational efficiency. Moreover, employing distinct parallel strategies for the prefill and decoding stages typically necessitates the redistribution of model weights through collective communication, which introduces substantial overhead and underscores the demand for more efficient solutions.

2) Memory Consumption.: The memory requirements are primarily attributed to three components: model weights, the KV cache, and activations. We analyze the memory consumption of the Attention module and the Expert module separately. For the Attention module, when employing TP, DP, or a combination of DP and TP, the per-device memory consumption of the KV cache and activations remains consis-

tent across these strategies. However, **DP demonstrates a**  $d \times$ increase in memory consumption of model weight relative to TP, where d denotes the DP degree. In contrast, Expert modules exhibit identical per-device model weight memory footprints regardless of parallel strategies. Current state-of-theart (SOTA) work such as DeepEP employ non-uniform Allto-All communication operations to manage expert-parallel data dispatch and combination. This approach introduces different activation memory consumption across devices due to imbalanced expert workload distribution. Consequently, the HAP should take the memory constraints incurred by DP and EP into accounts. To conservatively estimate this memory requirement for EP, we adopt an empirical upperbound activation memory approximation that doubles the baseline TP activation footprint. This bounding methodology accounts for potential workload imbalances while maintaining analytical tractability.

We design the method named HAP to automatically search the optimal hybrid parallel strategies: 1) We develop inference simulation models for the Attention module, Expert module, and collective communication operations, enabling the accurate estimation of the inference latency. 2) By leveraging these simulation models to quantitatively evaluate the end-to-end inference latency for various strategy combinations, we build a search space that encompasses all feasible parallelization strategies for both modules. 3) Aimed at minimizing total inference latency, we formulate and simplify the problem as Integer Linear Programming (ILP), allowing for the efficient identification of the optimal hybrid parallelization strategy.

## B. Inference Latency Estimation

To ensure the effectiveness of optimal strategy search, the simulation of inference latency must exhibit accuracy and stability. This subsection presents our inference latency simulation model.

We split the inference duration into several items to make an accurate simulation. Specifically, the total inference duration is represented as equation 1.

$$T_{total} = T_{prefill} + T_{decoding} \tag{1}$$

$$T_{prefill} = N_{layer} * (T_{attn} + T_{experts} + T_{comm})$$
 (2)

$$T_{decoding} = S_{output} * N_{layer} * (T_{attn} + T_{experts} + T_{comm})$$
(3)

The temporal parameters  $T_{attn}$ ,  $T_{experts}$ , and  $T_{comm}$  are obtained from inference simulation models.  $T_{total}$  denotes the overall inference time, with  $T_{prefill}$  and  $T_{decoding}$  representing the latencies of the prefill and decoding stages.  $N_{layer}$  is the number of model layers, and  $S_{output}$  is the output sequence length. Within a single layer,  $T_{attn}$ ,  $T_{experts}$ , and  $T_{comm}$  correspond to the latencies of the attention module, expert module, and communication, respectively.

The simulation models for the Attention and Expert modules are computational estimation model based on Floating-Point Operations (FLOPs). We formulate the simulation model as:  $T_{cal} = \left(\frac{F_{module}}{Max\_FLOPs/s}\right) \times \eta. \ F_{module} \ \text{denotes the number of}$ 

FLOPs for a given module and  $Max\_FLOPs/s$  represents the device's theoretical peak FLOPS. We employ the model configuration and input prompt information, i.e., b,s,h, to parameterize  $\eta$ . Additionally, these parameters are enriched through polynomial feature expansion to enhance representational capacity, which is then leveraged by an efficient random forest regression model to fit  $\eta$ . The lightweight architecture of the regression model ensures minimal computational overhead while achieving high predictive accuracy.

Similar to the computational simulation model, the communication simulation model is carried out by the following principle:  $T_{comm} = \left(\frac{V_{data}}{Bandwidth}\right) \times \rho. \ V_{data}$  is the data volume in the collective communication and Bandwidth implys the network bandwidth within the server. We also utilize a random forest regression model to estimate  $\rho$ , which exclusively takes the data volume and the device bandwidth as its input. Notably, the training datasets derive from empirically measured operator runtime latency values, acquired through systematic benchmarking protocols.

## C. Optimal Hybrid Parallel Strategies Search

We summarize all possible parallelization strategies for both the Attention and Expert modules:

- Attention module. The potential deployment strategies for the Attention module weights include DP, TP, and the hybrid parallel strategies combining both DP and TP.
- Expert module. The deployment strategies for the Expert module weights encompass EP, TP, and the hybrid parallel strategies that combine both EP and TP. Given that the Expert module weights constitute the majority of the parameters in MoE models and considering memory constraints, we exclude DP for this module from the search space.

Notably, for the aforementioned hybrid parallel strategies, the TP degree increases exponentially as a power of two. The combination of these strategies constitutes the search space. The objective of searching the optimal parallel strategy is to minimize inference latency. We formulate this cost minimization problem as an ILP problem. The formal definition of the integer programming problem is as follows:

$$\min_{S,E,R} N_{\text{layer}} \times \{\underbrace{S_k^T T_a + E_i T_e + T_{C_{ki}}}_{\text{prefill}} + S_{\text{output}} \times \underbrace{(S_k^T T_a + E_j T_e + T_{C_{kj}})}_{\text{decode}}\} + \underbrace{E_i^T \mathbf{C_{ij}} E_j}_{\text{switching cost}}$$
(4)

s.t. 
$$A_t, A_d, E_d, E_t, E_e \in \mathbb{N}^+$$

$$\frac{M_{KV} + A_d \times M_{attn} + E_d \times M_{exp}}{N} + 2M_{act} < M_{gpu}$$

$$N = A_t \times A_d = E_d \times E_t \times E_e$$

$$B = b \times A_d$$

$$Dim \mid A_t, N_{kv} \mid A_t, N_{experts} \mid E_e, Dim_{exp} \mid E_t$$
(5)

Specifically, the  $a \mid b$  represents a divides b.  $N_{\text{layer}}$  is the number of model layers, and  $S_{\text{output}}$  is the output length. Let  $A_t$  and  $A_d$  denote the TP and DP degrees for the Attention module, while  $E_t$ ,  $E_d$  and  $E_e$  represent the TP, DP and EP degrees for the Expert module, respectively.  $M_{KV}$ ,  $M_{attn}$ ,  $M_{exp}$ , and  $M_{act}$  indicate the memory usage of the KV cache, attention weights, expert weights, and activations, respectively, while  $M_{qpu}$  is GPU memory capacity. B denotes the batch size, b is the micro-batch size per DP device, and N is the total number of devices.Dim is the hidden size,  $Dim_{exp}$  is the expert's intermediate size, and  $N_{experts}$  denotes the number of experts per layer. The number of parallel strategies for the Attention module and the Expert module are are denoted by  $K_a$  and  $K_e$ , respectively. The computational cost of the Attention module is represented by the vector  $T_a$  of length  $K_a$ , where  $T_{ai}$  denotes the computational cost of the *i*-th parallel strategy for the Attention module. Similarly, the Expert module has a computational cost vector  $T_e$ . The communication cost associated with the selected parallel strategies for both modules is denoted as  $T_C$ .

Moreover, we define the one-hot decision vectors  $S_k \in$  $\{0,1\}^{K_a}$  and  $E_i \in \{0,1\}^{K_e}$  to indicate the chosen parallelization strategies, where  $S_{ki} = 1$  indicates that the *i*-th parallel strategy is selected for the Attention module. Additionally, the matrix  $\mathbf{C} \in \mathbf{R}^{K_e \times K_e}$  represents the overhead incurred when switching the Expert module's parallelization strategy from the prefill stage to the decoding stage, with Cii denoting the overhead of switching from the i-th strategy to the j-th strategy. Furthermore, we prune the search space based on memory requirements and prior empirical knowledge. Due to the KV cache, the Attention module employs a consistent parallelization strategy across both prefill and decoding stages, while the Expert module can adopt different strategies for each stage. Based on memory constraints, we preemptively eliminate some parallel strategies for the Expert module that utilize DP. Leveraging existing experience, we exclude combinations of DP, EP, and TP in the parallel strategies of Expert module for the generally suboptimal performance on a node with multiple devices.

We formulate and solve the ILP problem using Python's PuLP library. For typical limited-scale deployment scenarios (e.g., single-machine 8-GPU configurations), the optimization completes consistently within one second of search time. Crucially, our experimental methodology incorporates the ILP solver runtime into all reported end-to-end latency measurements across benchmark evaluations.

## D. The Dynamic Parallelism Transition Strategy.

Parallelism strategy transitions typically necessitate redistributing model weights via collective communication operations, such as AllGather and AllToAll, which can incur substantial communication overhead. This challenge arises due to the expert module weights, which account for approximately 90% of the total model parameters. As a result, naïve parallelism transitions may paradoxically degrade inference performance. To address this limitation, we propose a dynamic paral-

TABLE I: Performance comparison of different quantization schemes with Mixtral 8x7B.

| Quantization scheme     | MMLU           | Arc-e          | Arc-c          | NQ_OPEN        | TriQA          | HumanE         | MBPP           | GSM8K          |
|-------------------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| Origin                  | 67.7%          | 84.3%          | 56.9%          | 30.2%          | 71.4%          | 35.9%          | 47.6%          | 58.3%          |
| Per-tensor<br>Per-group | 67.7%<br>67.7% | 83.3%<br>84.0% | 56.4%<br>56.2% | 29.6%<br>30.1% | 70.2%<br>70.6% | 35.9%<br>35.4% | 47.8%<br>47.6% | 55.6%<br>58.0% |

lelism transition method with two implementation approaches: (1) Weight redistribution via collective communication; (2) As shown in Fig. 3, we maintain a 4-bit (INT4) quantized backup of expert layer weights in CPU memory. This backup is asynchronously uploaded to the corresponding devices through multi-stream pipelines and subsequently dequantized to its native precision, e.g., BF16 [25]. The optimal transition approach is selected through empirical simulations:

$$\mathbf{C_{ij}} = \min\{T_{reshard}, \max\{0, T_{upload} + T_{deguant} - (S_k^T T_a + E_i T_e + T_{C_{ki}})\}\}$$
(6)

where  $T_{reshard}$  denotes the time of weight redistribution via collective communication operations, and  $T_{upload}$  represents the time taken to upload weights from the CPU to the devices. The overhead of weight dequantization is denoted by  $T_{dequant}$ . Note that  $T_{dequant}$  scales with the number of dequantized parameters ( $V_{dequant}$ ). Accordingly, we determine  $V_{dequant}$  based on the number of GPUs, thereby constructing a dictionary of  $V_{dequant} \rightarrow T_{dequant}$ , which is queried at runtime to obtain this overhead.

![](_page_4_Figure_5.jpeg)

Fig. 3: The workflow of parallelism transition during the prefill stage.

Notably, expert weight quantization followed by dequantization maintains >99.5% cosine similarity to original weights, yet can still cause performance degradation. As demonstrated in Table I, per-tensor quantization significantly degraded GSM8K performance while remaining nearly lossless on MMLU and MBPP. Through comprehensive evaluation of per-tensor, per-channel, and per-group quantization [26], we dopted fine-grained per-group quantization, which preserved model performance with only marginal degradation.

## IV. PERFORMANCE EVALUATION

### A. Overview

As shown in Table II, we conducted systematic experiments across four orthogonal inference scenarios defined along two critical dimensions: input context scale (short versus long) and output generation complexity (constrained versus extended). Utilizing the open-source framework DeepSpeed-FastGen as

our experimental platform, we performed comparative evaluations of end-to-end latency between our proposed HAP optimized inference and TP-based inference, with the latter serving as the baseline configuration.

TABLE II: The experimental configurations used in the evaluation.

| Inference scenarios                         | Configurations                            |
|---------------------------------------------|-------------------------------------------|
| Short-context input with constrained output | 256-token context, 64-token generation    |
| Short-context input with extended output    | 256-token context, 2048-token generation  |
| Long-context input with constrained output  | 4096-token context, 64-token generation   |
| Long-context input with extended output     | 4096-token context, 2048-token generation |

We conduct extensive evaluations using HAP on A6000, A100 and V100 GPUs. The A6000 and V100 GPUs execute the intra-node communication via PCIe while A100 via NVLink. We employ the Mixtral-series and Qwen-series MoE models to demonstrate the superior performance and generalizability of HAP. The Mixtral-series features a smaller number of experts per layer, each with a larger parameter count. In contrast, the Qwen-series incorporates a larger number of experts per layer, each with fewer parameters, and additionally supports shared experts. Table III elaborates the model configurations.

TABLE III: The model configurations used for the inference performance evaluation.

| Model             | Params(B) | Layers | Q_Heads | Hidden | Experts | MoE_inter_size |
|-------------------|-----------|--------|---------|--------|---------|----------------|
| Mixtral-8x7B      | 46.7      | 32     | 32      | 4096   | 8       | 14336          |
| Qwen1.5-MoE-A2.7B | 14.3      | 24     | 16      | 2048   | 60      | 1408           |
| Qwen2-57B-A14B    | 57.4      | 28     | 28      | 3584   | 64      | 2560           |

## B. The Estimation of Simulation Models.

We constructed the test datasets using empirically measured computation and communication latency profiles obtained during the inference process. These datasets were subsequently utilized to assess the prediction accuracy of simulation models. As illustrated in Fig. 5, the communication simulation models exhibit an error margin within 5%, while the computational simulation models maintain an error below 10%. This level of accuracy is sufficient to support the construction of effective and reliable search spaces.

## C. Performance details of HAP

1) Short-context input with constrained generation.: Compared to the baseline, Fig. 4 shows that the HAP-based inference achieved maximum speedups of 1.13×, 1.12×, and 1.18× for the Mixtral-8x7B, Qwen1.5-MoE-A2.7B, and Qwen2-57B-A14B models, respectively. Moreover, HAP attained speedups of up to 1.16×, 1.37×, and 1.11× on the

![](_page_5_Figure_0.jpeg)

Fig. 4: The performance comparison conducted under a scenario featuring a 256-token context and 64-token generation between HAP and the TP baseline, with evaluations performed on Mixtral-series and Qwen-series MoE models using 4×A6000 GPUs and 4×A100 GPUs, respectively.

![](_page_5_Figure_2.jpeg)

![](_page_5_Figure_3.jpeg)

Fig. 5: Comparative evaluation of numerical accuracy in computational and communication simulation models

4×A100 GPU configuration, demonstrating consistent performance improvements across both architectures when executing Mixtral-series and Qwen-series MoE models. Even in scenarios where TP represents the optimal parallel strategy, HAP achieves comparable execution latency.

2) Short-context input with extended generation.: As demonstrated in Fig. 6, our experimental results reveal distinct acceleration patterns across the three evaluated models. HAP-based inference can achieve at most  $1.01 \times, 1.10 \times$ , and  $1.036 \times$  speedup on the A6000 GPUs, with corresponding speedups of  $1.01 \times, 1.23 \times$ , and  $1.05 \times$  observed on A100 GPUs. Notably, HAP frequently fails to surpass the acceleration performance of TP-based inference across multiple configurations. Through systematic analysis, we identify the decoding phase latency as the dominant factor in this particular inference scenario. This

observation aligns with our theoretical framework in Section III-A, which establishes that the computationally intensive decoding phase inherently favors TP. Consequently, the HAP method consistently prioritizes TP-based configurations for Expert module parallelization during the optimization process, thereby often precluding the attainment of higher acceleration in many scenarios.

3) Long-context input with constrained output.: As illustrated in Fig. 7, the HAP-based inference achieves a speedup ranging from 1.21× to 1.68× across different MoE models on A6000 GPUs. Similarly, HAP-based inference achieved a maximum speedup of 1.77× on A100 GPUs, demonstrating the adaptive superiority in specific workload. This performance discrepancy stems from the unique computational characteristics of this inference scenario, where the prefill stage consumes a significant amount of time while the decoding stage incurs minimal latency, rendering communication the primary performance bottleneck. In contrast, TP-based inference suffers from excessive communication overhead, highlighting the need for an optimal strategy that minimizes communication to substantially enhance inference efficiency. HAP searched the lowcommunication parallelization configurations, e.g., Attention module with DP and Expert module with TP or EP, to reduce the communication overhead.

We also conducted a performance evaluation using the Mixtral-8x7B model under similar inference scenarios featuring a 2048-token input context with constrained output. The experiments were carried out on two single-node platforms: one equipped with 8 A100 GPUs, which feature high intranode bandwidth via NVLINK, and another equipped with 8 NVIDIA V100 GPUs, which rely on limited intra-node bandwidth via PCIe. As evidenced in the Fig. 8 (a) and (b), the HAP-based inference yields speedups of of 1.29× and 1.57× on A100 and V100 platforms respectively, thereby demonstrating the adaptability of HAP across heterogeneous computing environments.

4) Long-context input with extended output.: This inference scenario exhibits dual-phase characteristics where decoding latency constitutes the dominant component of end-to-end

![](_page_6_Figure_0.jpeg)

![](_page_6_Figure_1.jpeg)

b) Input:256 Gen:2048 on a single node with 4 A100 GPUs

Fig. 6: The performance comparison conducted under a scenario featuring a 256-token context and 2048-token generation between HAP and the TP baseline, with evaluations performed on Mixtral-series and Qwen-series MoE models using  $4 \times A6000$  GPUs and  $4 \times A100$  GPUs, respectively.

![](_page_6_Figure_3.jpeg)

![](_page_6_Figure_4.jpeg)

Fig. 7: The performance comparison conducted under a scenario featuring a 4096-token context and 64-token generation between HAP and the TP baseline, with evaluations performed on Mixtral-series and Qwen-series MoE models using 4×A6000 GPUs and 4×A100 GPUs, respectively.

![](_page_6_Figure_6.jpeg)

Fig. 8: The performance evaluation using Mixtral-8x7B MoE model. (a) The performance comparison under the 2048-token context with 128-token output scenario on a single node with 8 A100 GPUs. (b) The performance comparison under the 2048-token context with 64-token output scenario on a single node with 8 V100 GPUs. (c) The latency comparison between TP-based, EP-based, and HAP-based inference, with evaluations conducted on Mixtral-8x7B model on 4xA6000 GPUS.

inference latency, while prefill stage still introduce substantial computational overhead. HAP establishes the necessity of phase-specific parallelization strategies: expert parallelism optimizes communication-intensive prefill stage, whereas tensor

parallelism accommodate the decoding stage. As demonstrated in Fig. 9, HAP-based inference achieves up to  $1.13\times$  speedup. Moreover, when the prefill stage accounts for a larger proportion of the overall end-to-end latency, the phase-specific parallelization strategies employed by HAP can achieve superior performance.

To substantiate our dynamic parallelism transition methodology, we analyzed the latency in both the prefill and decoding stages across TP-based, EP-based, and HAP-based inference. As depicted in Fig. 8(c), EP demonstrates significantly lower prefill latency than TP, albeit with higher decoding latency. In contrast, HAP innovatively integrates dynamic parallelism transition, achieving prefill performance comparable to EP with minimal transition overhead, while simultaneously maintaining decoding efficiency equivalent to TP. This adaptive approach effectively leverages the strengths of both parallelisms.

### V. CONCLUSION

In this paper, we propose HAP, a novel method that automatically selects hybrid parallel strategies for efficient inference. HAP contains a series of insightful strategies that are theoretically generalizable to different hardware conditions. Extensive

![](_page_7_Figure_0.jpeg)

![](_page_7_Figure_1.jpeg)

b) Input:4096 Gen:2048 on a single node with 4 A100 GPUs

Fig. 9: The performance comparison conducted under a scenario featuring a 4096-token context and 2048-token generation between HAP and the TP baseline, with evaluations using 4×A6000 GPUs and 4×A100 GPUs, respectively.

experiments have demonstrated the excellent efficiency and generalization of HAP on multiple MoE LLMs. In future work, we will apply HAP to the multi-node inference, which incorporates a more sophisticated search mechanism. We will also focus on dynamic, real-time inference serving scenarios to maximize throughput.

#### REFERENCES

- M. Artetxe, S. Bhosale, N. Goyal, T. Mihaylov, M. Ott, S. Shleifer, X. V. Lin, J. Du, S. Iyer, R. Pasunuru, et al., "Efficient large scale language modeling with mixtures of experts," arXiv preprint arXiv:2112.10684, 2021.
- [2] M. R. Costa-Jussà, J. Cross, O. Çelebi, M. Elbayad, K. Heafield, K. Heffernan, E. Kalbassi, J. Lam, D. Licht, J. Maillard, et al., "No language left behind: Scaling human-centered machine translation," arXiv preprint arXiv:2207.04672, 2022.
- [3] W. Fedus, B. Zoph, and N. Shazeer, "Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity," *Journal of Machine Learning Research*, vol. 23, no. 120, pp. 1–39, 2022.
- [4] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, et al., "Deepseek-v3 technical report," arXiv preprint arXiv:2412.19437, 2024.
- [5] C. Riquelme, J. Puigcerver, B. Mustafa, M. Neumann, R. Jenatton, A. Susano Pinto, D. Keysers, and N. Houlsby, "Scaling vision with sparse mixture of experts," *Advances in Neural Information Processing Systems*, vol. 34, pp. 8583–8595, 2021.
- [6] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica, "Efficient memory management for large language model serving with pagedattention," in *Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles*, 2023.
- [7] C. Holmes, M. Tanaka, M. Wyatt, A. A. Awan, J. Rasley, S. Rajbhandari, R. Y. Aminabadi, H. Qin, A. Bakhtiari, L. Kurilenko, et al., "Deepspeed-fastgen: High-throughput text generation for Ilms via mii and deepspeed-inference," arXiv preprint arXiv:2401.08671, 2024.
- [8] D. Yu, L. Shen, H. Hao, W. Gong, H. Wu, J. Bian, L. Dai, and H. Xiong, "Moesys: A distributed and efficient mixture-of-experts training and inference system for internet services," *IEEE Transactions on Services Computing*, vol. 17, no. 5, pp. 2626–2639, 2024.
- [9] D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, and Z. Chen, "Gshard: Scaling giant models with conditional computation and automatic sharding," arXiv preprint arXiv:2006.16668, 2020.
- [10] B. Zoph, I. Bello, S. Kumar, N. Du, Y. Huang, J. Dean, N. Shazeer, and W. Fedus, "Designing effective sparse expert models," arXiv preprint arXiv:2202.08906, vol. 2, no. 3, p. 17, 2022.
- [11] A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. d. l. Casas, E. B. Hanna, F. Bressand, et al., "Mixtral of experts," arXiv preprint arXiv:2401.04088, 2024.
- [12] Q. Team, "Qwen1.5-moe: Matching 7b model performance with 1/3 activated parameters"," February 2024.

- [13] A. Yang, B. Yang, B. Hui, B. Zheng, B. Yu, C. Zhou, C. Li, C. Li, D. Liu, F. Huang, et al., "Qwen2 technical report," 2024.
- [14] NVIDIA, "A tensorrt toolbox for optimized large language model inference." https://github.com/NVIDIA/TensorRT-LLM, 2024.
- [15] L. Zheng, L. Yin, Z. Xie, C. L. Sun, J. Huang, C. H. Yu, S. Cao, C. Kozyrakis, I. Stoica, J. E. Gonzalez, et al., "Sglang: Efficient execution of structured language model programs," Advances in Neural Information Processing Systems, vol. 37, pp. 62557–62583, 2024.
- [16] Q. Li, B. Zhang, L. Ye, Y. Zhang, W. Wu, Y. Sun, L. Ma, and Y. Xie, "Flash communication: Reducing tensor parallelization bottleneck for fast large language model inference," arXiv preprint arXiv:2412.04964, 2024.
- [17] H. Huang, N. Ardalani, A. Sun, L. Ke, H.-H. S. Lee, A. Sridhar, S. Bhosale, C.-J. Wu, and B. Lee, "Towards moe deployment: Mitigating inefficiencies in mixture-of-expert (moe) inference," arXiv preprint arXiv:2303.06182, 2023.
- [18] S. Rajbhandari, C. Li, Z. Yao, M. Zhang, R. Y. Aminabadi, A. A. Awan, J. Rasley, and Y. He, "Deepspeed-moe: Advancing mixture-of-experts inference and training to power next-generation ai scale," in *International conference on machine learning*, pp. 18332–18346, PMLR, 2022.
- [19] S. Singh, O. Ruwase, A. A. Awan, S. Rajbhandari, Y. He, and A. Bhatele, "A hybrid tensor-expert-data parallelism approach to optimize mixtureof-experts training," in *Proceedings of the 37th International Conference* on Supercomputing, pp. 203–214, 2023.
- [20] C. Zhao, S. Zhou, L. Zhang, C. Deng, Z. Xu, Y. Liu, K. Yu, J. Li, and L. Zhao, "Deepep: an efficient expert-parallel communication library." https://github.com/deepseek-ai/DeepEP, 2025.
- [21] L. Zheng, Z. Li, H. Zhang, Y. Zhuang, Z. Chen, Y. Huang, Y. Wang, Y. Xu, D. Zhuo, E. P. Xing, et al., "Alpa: Automating inter-and {Intra-Operator} parallelism for distributed deep learning," in 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22), pp. 559–578, 2022.
- [22] C. Hwang, W. Cui, Y. Xiong, Z. Yang, Z. Liu, H. Hu, Z. Wang, R. Salas, J. Jose, P. Ram, et al., "Tutel: Adaptive mixture-of-experts at scale," Proceedings of Machine Learning and Systems, vol. 5, pp. 269–287, 2023.
- [23] M. Vasa, C.-L. Liao, S. Kumar, C.-H. Chen, and B. Mutnury, "Pcie gen-5 design challenges of high-speed servers," in 2020 IEEE 29th Conference on Electrical Performance of Electronic Packaging and Systems (EPEPS), 2020.
- [24] J. Choquette and W. Gandhi, "Nvidia a100 gpu: Performance & innovation for gpu computing," in 2020 IEEE Hot Chips 32 Symposium (HCS), pp. 1–43, IEEE Computer Society, 2020.
- [25] "Bitsandbytes: a lightweight python wrapper around cuda custom functions." https://github.com/bitsandbytes-foundation/bitsandbytes, 2024.
- [26] M. Nagel, M. Fournarakis, R. A. Amjad, Y. Bondarenko, M. Van Baalen, and T. Blankevoort, "A white paper on neural network quantization," arXiv preprint arXiv:2106.08295, 2021.