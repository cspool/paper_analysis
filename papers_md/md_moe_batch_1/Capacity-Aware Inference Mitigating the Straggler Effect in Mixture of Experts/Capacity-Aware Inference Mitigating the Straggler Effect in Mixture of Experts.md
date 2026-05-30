# CAPACITY-AWARE INFERENCE: MITIGATING THE STRAGGLER EFFECT IN MIXTURE OF EXPERTS

Shwai He<sup>1</sup> Weilin Cai<sup>2</sup> Jiayi Huang<sup>2</sup> Ang Li<sup>1</sup>

wcai738@connect.hkust-gz.edu.cn, hjy@hkust-gz.edu.cn

#### ABSTRACT

The Mixture of Experts (MoE) is an effective architecture for scaling large language models by leveraging sparse expert activation to balance performance and efficiency. However, under expert parallelism, MoE suffers from inference inefficiencies due to imbalanced token-to-expert assignment, where underloaded experts complete computations early but must wait for overloaded experts, leading to global delays. We define this phenomenon as the *Straggler Effect*, as the most burdened experts dictate the overall inference latency. To address this, we first propose *Capacity*-Aware Token Drop, which enforces expert capacity limits by discarding excess tokens from overloaded experts, effectively reducing load imbalance with minimal performance impact (e.g., 30% speedup with only 0.9% degradation on OLMoE). Next, given the presence of low-load experts remaining well below the capacity threshold, we introduce Capacity-Aware Expanded Drop, which allows tokens to include additional local experts in their candidate set before enforcing strict local capacity constraints, thereby improving load balance and enhancing the utilization of underused experts. Extensive experiments on both language and multimodal MoE models demonstrate the effectiveness of our approach, yielding substantial gains in expert utilization, model performance, and inference efficiency. For example, applying Expanded Drop to Mixtral-8×7B-Instruct yields a 0.2% average performance improvement and a 1.85× inference speedup. The code is released at: https://github.com/CASE-Lab-UMD/Capacity-Aware-MoE.

## 1 Introduction

In recent years, the rapid evolution of Large Language Models (LLMs) (OpenAI, 2024; Team, 2024a; et al., 2024b) has driven a wave of innovations, continuously expanding the frontiers of AI research and applications. Among the model architectural innovations, the Mixture of Experts (MoE) framework has emerged as a pivotal technique for optimizing the cost-performance tradeoff in LLMs. Specifically, MoE (Shazeer et al., 2017b) enhances scalability by integrating multiple experts while activating only a subset per input. This selective activation substantially improves model performance without a corresponding increase in computational cost, effectively balancing efficiency and performance.

<span id="page-0-0"></span>![](_page_0_Figure_11.jpeg)

Figure 1: **Illustration of the Straggler Effect in MoE Inference.** The normalized load is computed as each expert's load divided by the mean load across all experts. Example shown with OLMoE (Muennighoff et al., 2024) on Open-BookQA (Mihaylov et al., 2018b).

Despite the success of MoE, a key efficiency challenge lies in the imbalanced token-to-expert distribution, which results in some experts being overloaded while others remain underutilized (Lepikhin et al., 2021; Zoph et al., 2022). In distributed GPU settings, experts are typically sharded across multiple devices, with each GPU responsible for a subset of the experts. Under expert parallelism, low-load experts complete their computations earlier but must wait for overloaded experts

<sup>&</sup>lt;sup>1</sup>University of Maryland, College Park

<sup>&</sup>lt;sup>2</sup>The Hong Kong University of Science and Technology (Guangzhou) {shwaihe, angliece}@umd.edu

to finish, as synchronization barriers are required before proceeding to the next stage. This expertlevel straggler effect further propagates to device-level delays, where GPUs hosting lighter expert workloads are stalled by GPUs hosting heavier workloads, leading to inefficient resource utilization and increased end-to-end latency during inference. As illustrated in Figure [1,](#page-0-0) this phenomenon is referred to as the *Straggler Effect*, where the heavily loaded experts determine the overall latency of imbalanced MoE inference.

While auxiliary balance losses have been incorporated into the training process to alleviate imbalance [\(Shazeer et al., 2017b;](#page-13-1) [Fedus et al., 2022;](#page-11-1) [et al., 2024b\)](#page-11-0), these techniques remain ineffective in mitigating imbalance during inference. Specifically, as shown in Figure [1,](#page-0-0) our findings reveal a highly uneven token distribution among experts, with the highest-load expert handling more than seven times the expected average load. Moreover, managing such imbalance during inference often incurs additional resource overhead. For example, DeepSeek-V3 mitigates this issue by duplicating high-load experts and deploying them redundantly across devices [\(et al., 2024b\)](#page-11-0). This motivates us to explore *efficient token-to-expert assignment* by addressing the key question: *How can we prevent extreme overloading of heavily utilized experts?*

We propose *Capacity-Aware Inference* to address this challenge. Specifically, for high-load experts, we introduce *Capacity-Aware Token Drop*, which imposes a maximum capacity constraint and discards excess tokens from overloaded experts. This approach alleviates severe load imbalance and significantly improves efficiency, while maintaining model performance since the dropped tokens represent only a small fraction of the total workload, e.g., OLMoE achieves a 30% speedup in MoE layers with just a 0.9% performance degradation. After removing excess tokens from overloaded experts, we observe that some low-load experts remain significantly underutilized relative to the predefined capacity constraints, yet must still wait for other experts to complete their computations. This leads us to a second key question: *How can we effectively leverage the available capacity of underutilized experts?*

For low-load experts, we extend Token Drop with *Capacity-Aware Expanded Drop*, which further utilizes the available capacity of underutilized experts to handle overflow tokens from high-load experts. Specifically, under expert parallelism across multiple GPUs, Expanded Drop allows each token to consider additional candidate experts on the same device while still enforcing strict local capacity constraints. This expanded selection improves the utilization of low-load experts and enhances the representational capacity of MoE models under capacity-constrained scenarios.

Extensive experimental results validate the effectiveness of our proposed techniques, demonstrating significant improvements in both efficiency and performance, e.g., *applying Expanded Drop to Mixtral-8*×*7B-Instruct yields a 0.2% average performance improvement and a 1.85*× *inference speedup*. Moreover, in multimodal models, we identify redundancy among image tokens and show that applying aggressive capacity constraints (e.g., setting the maximum to half of the average expert load) can still maintain performance. In short, our contributions are four-fold:

- We explicitly clarify and analyze the Straggler Effect caused by test-time token imbalance in the Mixture of Experts, highlighting the optimization potential for reducing latency.
- Toward the high-load experts, we propose Capacity-Aware Token Drop, which enforces capacity constraints by discarding excess tokens assigned to overloaded experts, thereby mitigating extreme load imbalance.
- To better utilize underloaded experts, we introduce Capacity-Aware Expanded Drop, which expands the candidate expert set to include additional local experts, further improving load balance and model performance.
- Extensive experiments on both language and multimodal models validate the effectiveness of our approach, demonstrating substantial improvements in inference efficiency with comparable performance.

## 2 RELATED WORKS

Mixture of Experts Models The Mixture of Experts (MoE) is a neural network architecture with an extended set of parameters (referred to as "experts") controlled by a router, and was first introduced in the context of conditional computation [\(Jacobs et al., 1991;](#page-11-2) [Jordan & Jacobs, 1994\)](#page-12-4). The potential of sparse activation in MoE is subsequently exploited by [Shazeer et al.](#page-12-5) [\(2017a\)](#page-12-5) for efficient training and inference on pretrained models with special designs, opening the door for MoE in various vision (Riquelme et al., 2021) and language (Lepikhin et al., 2020; Du et al., 2022; Fedus et al., 2022) scenarios. Due to its exceptional efficiency, MoE has been adopted as a foundational framework in the designs of large language models (Jiang et al., 2024; Dai et al., 2024; Xue et al., 2024a; Zhu et al., 2024; Team, 2024b), achieving superior scaling laws at low computational costs. Despite these advancements, MoE still faces efficiency challenges in both training and inference (Cai et al., 2024), and our work specifically focuses on enhancing inference-time efficiency.

Imbalance in Mixture of Experts The imbalance in token-to-expert assignments (Zhou et al., 2022; Chen et al., 2022) poses a significant challenge to the deployment of MoE. This imbalance leads to inefficiencies in computation, communication, and memory (He et al., 2023; Song et al., 2023; Xue et al., 2024b), making it a critical bottleneck for MoE scalability and deployment. To mitigate this issue, an auxiliary balance loss (Shazeer et al., 2017b) is incorporated into the training process to encourage more uniform token distribution across experts. Additionally, various training strategies have been introduced to further balance token assignments: Switch-Transformer (Fedus et al., 2022) and DeepSeek-V2 (et al., 2024a) implement Token Drop to alleviate expert overload, while DeepSeek-V3 (et al., 2024b) introduces an additional sequence-level auxiliary loss to prevent severe token imbalance.

However, these techniques primarily focus on training and fail to ensure balanced token assignments during inference. Instead, addressing token imbalance at inference often incurs additional resource costs. For example, DeepSeek-V3 (et al., 2024b) mitigates this issue by duplicating high-load experts and deploying them redundantly. In contrast, our approach effectively balances token assignments without introducing additional computational overhead.

#### 3 BACKGROUND AND MOTIVATION

## 3.1 Extremely Imbalanced Expert Utilization

A Mixture of Experts (MoE) layer consists of a collection of n experts,  $\{E_1, E_2, \dots, E_n\}$  and a router G that dynamically selects the most relevant experts for a given input x. The router computes selection scores G(x), for all experts and selects the top k experts, resulting in a sparse activation:

$$\mathcal{K} = \text{TopK}(\text{Softmax}(\mathbf{G}(\mathbf{x})), k). \tag{1}$$

The input x is processed by the selected experts, and their outputs are combined into a weighted sum based on the router's scores. This process is mathematically expressed as:

$$y = \sum_{i \in \mathcal{K}} G(x)_i \cdot E_i(x), \tag{2}$$

where  $\mathcal{K}$  denotes the indices of selected experts,  $G(x)_i$  represents the selection score for the i-th expert, and  $E_i(x)$  is the output from the i-th expert. In transformer models, the MoE layer usually replaces the feed-forward network (FFN) and only activates a subset of experts for each input.

While experts in MoE models are often deployed in parallel across distributed GPUs, imbalanced token-to-expert assignments lead to varying levels of expert utilization and potential latency. Despite the incorporation of balancing techniques during training, the load imbalance persists during inference. To further investigate this issue, we conduct preliminary experiments to analyze expert-specific utilization patterns and assess the impact of imbalance on practical latency.

To quantify expert utilization, we measure the load across different experts. Given an input batch  $\mathbf{x} \in \mathbb{R}^{b \times s \times d}$  with batch size b and sequence length s, the total number of tokens is t = bs. Since each token selects k out of n experts, the expected token count per expert is:

$$\bar{N} = \frac{tk}{n}. (3)$$

However, due to imbalanced token assignments, some experts may receive more or fewer tokens than the expected value.

<span id="page-2-0"></span>![](_page_2_Figure_15.jpeg)

Figure 2: **Test-time expert load** of OLMoE across different datasets, where each load value is normalized by the mean load  $\bar{N}$  for clarity.

Figure 2 illustrates the normalized peak token load for each expert to accommodate all tokens within a single layer of OLMoE, where some experts receive an excessively large number of tokens (e.g., more than seven times the average number of tokens), leading to severe load imbalance and, consequently, significant latency. A detailed layer-by-layer analysis is provided in Appendix F.

#### 3.2 MOTIVATION – THE STRAGGLER EFFECT

Under expert parallelism, where the number of assigned tokens dictates the processing time of each expert, high-load experts become the bottleneck for overall latency within an MoE layer. Specifically, low-load experts remain idle while waiting for high-load experts to complete, leading to synchronization delays. Therefore, the latency of an MoE layer is given by:

<span id="page-3-0"></span>
$$L \propto \max(\{N_i\}_{i=1}^n),\tag{4}$$

where  $N_i$  represents the number of tokens assigned to the i-th expert, with the total token allocation satisfying  $\sum_{i=1}^{n} N_i = tk$ . According to Eq. 4, the latency follows the **Straggler Effect: the most burdened experts dictate the overall latency of the MoE layer**. In the worst case, all tokens are assigned to the same group of experts, underutilizing the parallel processing capability of MoE. Conversely, distributing tokens evenly across experts maximizes computational efficiency and fully leverages the parallelism of multiple experts. With the bounds of the ideal and worst cases, the range of the highest load is given by:

$$\max(\{N_i\}_{i=1}^n) \in [\bar{N}, \frac{n\bar{N}}{k}]. \tag{5}$$

However, existing MoE models often adopt a dropless strategy during inference, which fails to address token imbalance and can lead to significantly increased latency.

Given that the imbalance stems from excessively high- and low-load experts, we address this issue by exploring the following questions: (1) For *high-load experts*, are there redundant tokens that can be dropped without causing significant performance degradation? (2) For *low-load experts* that must wait for high-load experts to complete forward passes, is there an opportunity to enhance their utilization and improve performance without incurring substantial additional cost?

## 4 METHODOLOGY

Token Drop Regulates the Latency of High-Load Experts To address the question about overloaded experts, we first regulate their maximum utilization. Specifically, we introduce expert capacity to control token allocation. Given a capacity factor  $\gamma$ , the maximum number of tokens assigned to each expert (i.e., expert capacity) is defined as:  $C = \gamma \bar{N}$ .

A higher  $\gamma$  allows more tokens to be retained, but experts handling excessive tokens may introduce latency. Conversely, a lower  $\gamma$  enforces stricter capacity limits, reducing latency by discarding more tokens, but at the risk of performance degradation. With the involvement of expert capacity  $\gamma$ , we constrain the upper bound of latency as follows:

$$\max(\{N_i\}_{i=1}^n) = \begin{cases} \gamma \bar{N} & \gamma < 1\\ \text{within } [\bar{N}, \gamma \bar{N}] & \gamma \ge 1 \end{cases}, \tag{6}$$

where  $\gamma$  is typically much smaller than  $\frac{n}{k}$ . This constraint ensures that no expert exceeds the specified capacity limit, effectively mitigating severe load imbalances and reducing latency. Note that tokens are distributed across devices under expert and data parallelism. To avoid additional communication overhead, we apply capacity constraints to tokens within each local device, similar to the constraints used during training (Fedus et al., 2022). This ensures that all experts respect the limits, maintaining strict control over token flow to the experts.

Specifically, when a capacity constraint is imposed on each expert, experts must evaluate the volume of assigned tokens before execution. For experts with a load below the predefined capacity, there is no difference between capacity-constrained inference and traditional inference. However, when the load exceeds the capacity, experts must discard excess tokens to adhere to the constraint. To address

this, we introduce a scoring function S to evaluate each token:

$$S(x) = \begin{bmatrix} s_{11} & s_{12} & \dots & s_{1n} \\ s_{21} & s_{22} & \dots & s_{2n} \\ \vdots & \vdots & \vdots & \vdots \\ s_{t1} & s_{t2} & \dots & s_{tn} \end{bmatrix}, \tag{7}$$

where  $s_{ij}$  denotes the importance score of the mapping from the *i*-th token to the *j*-th expert. With this score, each overflowed expert selectively discards those with lower scores. Let  $\mathcal{J}$  be the set of overflowed experts and  $\mathcal{S}_{\mathcal{J}}$  the corresponding columns of  $\mathcal{S}$ , with the Token Drop threshold set as:

$$\tau_{\mathcal{J}} = \text{KthValue}(\mathcal{S}_{\mathcal{J}}, C), \tag{8}$$

where  $\tau_{\mathcal{J}}$  represents the thresholds, i.e., C-th highest value in  $\mathcal{S}_{\mathcal{J}}$ , serving as a threshold to filter out excess tokens:

$$T_{\mathcal{J}} \leftarrow \{(t,j) \mid t \in [1,\dots,N], \ j \in \mathcal{J}, \ \mathcal{S}[t,j] \ge \tau_{\mathcal{J}}[j]\}$$

$$\tag{9}$$

$$S_{\mathcal{J}} \leftarrow S_{\mathcal{J}} \odot M_{\mathcal{J}}, \text{ where } M_{\mathcal{J}} \leftarrow \mathbb{1} \left[ S_{\mathcal{J}} \ge \tau_{\mathcal{J}} \right],$$
 (10)

where  $T_{\mathcal{J}}$  denotes the token indices retained by the experts indexed in  $\mathcal{J}$ . The scores of rejected tokens are masked to prevent them from being routed to their corresponding overflowed experts.

Regarding the specific scoring function, we explore multiple metrics and summarize them as follows:

**Order:** Discarding later tokens once earlier tokens have filled the expert capacity. This strategy was first introduced in Switch-Transformer (Fedus et al., 2022) during training, and we extend it to the inference phase.

**Reverse Order:** Instead of discarding later tokens, this approach removes earlier tokens to comply with the expert capacity constraint.

**Random:** Dropping excess tokens randomly to meet the predefined expert capacity constraints.

**Score:** Using gating scores after the softmax and top-k operations as an importance indicator and discarding tokens.

Among these metrics, "Order" and "Reverse Order" are unstable, as shuffling sequences within a batch may result in different tokens being dropped (Hayes et al., 2024). "Random" assumes all tokens have an equal probability of being dropped. In contrast, "Score" is stable, unaffected by sequence order within a batch. Notably, there is virtually no additional computational overhead associated with calculating these metrics, and the dropping operation incurs minimal cost compared to the intensive computations performed by the experts.

![](_page_4_Figure_15.jpeg)

Figure 3: **Illustration of Capacity-Aware Token Drop (a) and Expanded Drop (b).** Both methods first select experts based on gating scores. In Token Drop, tokens exceeding the local device capacity are discarded prior to All-to-All communication. Expanded Drop enhances expert utilization by allowing each token to consider additional m candidate experts on the same device while still enforcing strict local capacity constraints.

**Expanded Drop Enhances the Utilization of Low-load Experts** Token Drop exclusively targets overloaded experts by discarding overflowed tokens that exceed expert capacity but does not address the underutilization of low-load experts. Next, we introduce Expanded Drop to ensure a more balanced token-to-expert allocation.

A naive approach to rerouting under-selected tokens is to mask the mapping scores of overflowed experts and then reselect experts for these tokens. However, the reselection may still result in

overflows, necessitating multiple rounds of selection and dropping, which increases latency. Moreover, the repeated selection and dropping substantially raise the cost of token-to-expert mapping.

Expanded Drop adopts a simple yet effective strategy: for each token, it selects additional candidate experts. Given m experts deployed on a single GPU, a token not only selects the top-k experts based on gating scores, but also includes m local experts (e.g., 8 experts per device under 8-way expert parallelism across 8 GPUs for a total of 64 experts) for substitution if the initially selected experts are overflowed. As a result, each token may select up to m+k experts. The final selection is then refined as experts drop tokens as needed to satisfy capacity constraints. This makes no change in the token assignments in experts that are overflowed by the top-k experts. Meanwhile, for underutilized experts, the expanded top-k+m candidate pool increases the likelihood of receiving additional tokens. After top-k+m selection and dropping, some tokens may select more than k experts. Through empirical analysis (Appendix D), we choose not to enforce a constraint that limits each token to selecting at most k experts, thereby removing the need to explicitly retain the top-k experts at the end.

Notably, the extra cost of token routing is minimal; the only difference lies in the negligible cost in the concatenation of the gating scores from either the top-k or m experts on the local device. Moreover, processing expanded tokens within the local device eliminates inter-device communication.

#### 5 EXPERIMENTS

In this section, we conduct experiments under capacity-aware inference for MoE, with deployment details provided in Appendix A.

## 5.1 TOKEN DROP FOR HIGH-LOAD EXPERTS

<span id="page-5-0"></span>Table 1: Performance comparison across different capacity factors and selection metrics (i.e., Order, Reverse Order, Random, and Score). The baseline operates without capacity constraints, represented as  $+\infty$ . We report the average performance over multiple random seeds.

| Method        | γ   | OBQA        | PIQA        | RTE         | WinoGrande  | BoolQ       | ARC-C       | HellaSwag   | MMLU        | Avg.        |
|---------------|-----|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| Baseline      | +∞  | 45.6        | 80.1        | 53.7        | 71.2        | 74.7        | 54.5        | 79.4        | 52.5        | 64.0        |
| Order         | 2.0 | 42.0        | 71.5        | 53.1        | 71.2        | 74.2        | 49.5        | 76.6        | 48.4        | 60.8        |
| Reverse Order |     | 41.8        | 71.8        | 52.7        | 71.0        | 73.9        | 49.4        | 76.4        | 49.2        | 60.8        |
| Random        |     | 41.2        | 75.2        | 52.7        | 71.0        | 74.1        | 50.1        | 76.8        | 49.4        | 61.3        |
| Score         |     | <b>45.0</b> | <b>80.1</b> | <b>54.5</b> | <b>71.5</b> | <b>74.6</b> | <b>54.9</b> | <b>79.3</b> | <b>51.8</b> | 64.0        |
| Order         | 1.5 | 38.8        | 67.1        | 48.7        | 68.5        | 73.3        | 46.3        | 54.0        | 43.7        | 55.1        |
| Reverse Order |     | 40.2        | 67.3        | 52.7        | 70.1        | 72.7        | 45.5        | 54.4        | 45.2        | 56.0        |
| Random        |     | 39.6        | 72.1        | 53.8        | 68.3        | 73.8        | 45.8        | 74.2        | 45.2        | 59.1        |
| Score         |     | 44.8        | <b>77.5</b> | <b>55.2</b> | <b>70.8</b> | <b>74.3</b> | <b>53.4</b> | <b>78.6</b> | <b>50.0</b> | <b>63.1</b> |
| Order         | 1.0 | 36.0        | 60.2        | 52.2        | 62.6        | 69.6        | 38.7        | 58.0        | 36.9        | 51.8        |
| Reverse Order |     | 36.2        | 59.5        | 50.5        | 63.3        | 69.4        | 39.4        | 58.7        | 38.7        | 52.0        |
| Random        |     | 34.0        | 63.1        | 53.2        | 60.8        | 70.2        | 40.5        | 66.9        | 35.7        | 53.1        |
| Score         |     | <b>41.6</b> | <b>76.0</b> | <b>53.4</b> | <b>69.9</b> | <b>73.2</b> | <b>50.4</b> | <b>77.1</b> | <b>47.8</b> | <b>61.1</b> |

**Investigation on Token Drop Metrics** To assess the effectiveness of different metrics in regulating token load to the target capacity, we compare various approaches on OLMoE by discarding excess tokens and applying a range of capacity factors. As shown in Table 1, varying the dropping metrics impacts performance at different levels. With higher capacities, the model maintains comparable performance even when using naive selection methods like "Random". However, as the capacity factor decreases, performance degradation becomes more pronounced, particularly for "Order", "Reverse Order", and "Random". Notably, "Score" consistently outperforms other methods by a large margin, demonstrating the effectiveness of leveraging gating scores as an importance measure. Consequently, we adopt "Score" as the default metric.

**Efficiency Gains from Capacity-Constrained Inference** We next explore the efficiency improvements achieved by imposing expert capacity. Specifically, we employ distributed inference using eight H20 GPUs, utilizing an 8-way Data Parallelism (DP) and 8-way Expert Parallelism (EP) strategy

<span id="page-6-0"></span>Figure 4: Speedup of a single MoE layer compared to the baseline without capacity constraints, achieved through two capacity-aware inference methods: Token Drop and Expanded Drop.

<span id="page-6-1"></span>![](_page_6_Figure_3.jpeg)

![](_page_6_Figure_4.jpeg)

Figure 5: End-to-end speedup. "T.D." and "E.D." are abbreviations for Token Drop and Expanded Drop, respectively.

Figure 6: Breakdown analysis of the inference latency on OLMoE with different capacity factors (e.g., 1.0, 1.5, 2.0).

through the Megatron-LM framework (Shoeybi et al., 2019). The input batches are configured with a batch size of 8K and a sequence length of 512, simulating real-time serving scenarios with high query throughput. Notably, in Mixtral-8×7B-Instruct model, each GPU typically hosts one or two experts, whereas, in models like OLMoE-Instruct, multiple experts must be deployed on a single GPU (e.g., eight experts per GPU) due to GPU resource constraints.

As illustrated in Figure 4, imposing constraints on expert capacity through Token Drop and Expanded Drop, considerably accelerates inference across the four tested MoE models, in comparison to the baseline model without capacity limitations. The enhanced efficiency of each MoE layer (Figure 4) contributes to faster end-to-end model inference (Figure 5). Moreover, as the capacity factor  $\gamma$  decreases, capacity-aware inference methods achieve significantly greater acceleration.

Notably, the efficacy of acceleration is influenced by the numerical relationship between the total experts and the engaged GPUs in Expert Parallelism. As illustrated in Figure 5, for Mixtral-8×7B-Instruct, deploying one or two experts per GPU maximizes the effectiveness of capacity-aware inference. In this configuration, Token Drop and Expanded Drop achieve end-to-end model speedups of  $1.87\times$  and  $1.85\times$ , respectively, with  $\gamma=1.5$ . Conversely, deploying a greater number of experts on a single GPU results in more modest acceleration gains, as evidenced by the "8E/GPU" (OLMoE-Instruct and DeepSeek-V2-Lite) and "10E/GPU" (Qwen1.5-MoE-Chat) settings in Figure 4 and Figure 5. This is because the aggregated load from multiple experts diminishes the proportion of reduced load, which is achieved by limiting the straggler expert. Therefore, it is anticipated that allocating more GPUs for expert distribution, thereby reducing the number of experts per GPU, would enhance the acceleration effect of capacity-aware inference.

The breakdown analysis presented in Figure 6 demonstrates that our proposed capacity-aware inference methods substantially reduce the duration of expert computation, permutation, and communication, while preserving a comparable cost for gate processing. Notably, the durations of permutation and communication increase when tokens are expanded across a range of global experts. This is due to the increased communication workload required to transmit expanded global tokens across various GPU devices. Consequently, these results underscore the necessity of restricting the expanded tokens to be processed by local experts.

Mitigating the Straggler Effect with Minimal Token Discarding Given that expert capacity enforces MoE layers to discard overflowed tokens, we next establish the relationship between expert capacity and the corresponding number of dropped tokens. For a capacity factor  $\gamma$ , the total proportion

of dropped tokens is given by:

<span id="page-7-2"></span>
$$DT = \frac{\sum_{i=1}^{n} \text{ReLU}(N_i - \gamma \bar{N})}{\sum_{i=1}^{n} N_i},$$
(11)

where  $ReLU(N_i - \gamma \bar{N})$  represents the number of dropped tokens for the *i*-th expert.

Figure 7 visualizes the number of dropped tokens across different capacity factors for various test datasets, with a more detailed illustration provided in Appendix G. Although the most overloaded expert receives much more tokens than the expected number of tokens  $\bar{N}$ , regulating the maximum capacity has a limited impact on the overall number of accommodated tokens, thereby maintaining competitive performance even after discarding overflow tokens. Moreover, dropping a small proportion of overflowed tokens can significantly reduce the latency caused by overloaded experts (e.g., dropping 12% of overloaded tokens improves the inference speed by 85% in Mixtral-8×7B-Instruct), highlighting the efficacy of capacity-aware inference in improving both performance and efficiency.

<span id="page-7-0"></span>![](_page_7_Figure_5.jpeg)

Figure 7: Analysis of dropped tokens with respect to capacity factors.

#### 5.2 EXPANDED DROP TO LOW-LOAD EXPERTS

Some experts receive very few tokens, raising the question of whether they are redundant or can be leveraged for balanced allocation. We next examine their role and validate the effectiveness of Expanded Drop.

<span id="page-7-1"></span>Table 2: Comparison of Expert Drop, Token Drop, and Expanded Drop. The capacity factor  $\gamma$  is set to 2.0 for OLMoE and DeepSeek-V2-Lite, and 1.5 for Qwen1.5-MoE-Chat and Mixtral-8×7B-Instruct. For Expert Drop, each forward pass skips one out of eight experts for Mixtral-8×7B-Instruct, and the bottom 10% of lowest-load experts for other models.

| Model                 | Method                                     | OBQA                        | PIQA                        | RTE                                | WinoGrande                  | BoolQ                              | ARC-C                       | HellaSwag                   | MMLU                               | GSM8K                       | Avg.                 |
|-----------------------|--------------------------------------------|-----------------------------|-----------------------------|------------------------------------|-----------------------------|------------------------------------|-----------------------------|-----------------------------|------------------------------------|-----------------------------|----------------------|
|                       | Baseline                                   | 47.6                        | 80.2                        | 67.9                               | 69.9                        | 80.7                               | 57.0                        | 80.6                        | 52.8                               | 35.1                        | 63.5                 |
| OLMoE-Instruct        | Expert Drop<br>Token Drop<br>Expanded Drop | 44.6<br><b>47.8</b><br>47.2 | 76.9<br>77.9<br><b>79.4</b> | 64.0<br>64.6<br><b>66.3</b>        | 67.6<br>69.2<br><b>70.5</b> | 78.2<br>80.0<br><b>80.9</b>        | 54.4<br><b>57.2</b><br>57.1 | 77.0<br>79.7<br><b>80.3</b> | 50.6<br>51.5<br><b>52.3</b>        | 31.6<br>32.4<br><b>34.4</b> | 60.5<br>62.3<br>63.2 |
|                       | Baseline                                   | 42.4                        | 79.9                        | 72.9                               | 70.0                        | 81.3                               | 54.1                        | 80.4                        | 59.8                               | 52.0                        | <u>65.9</u>          |
| Qwen1.5-MoE-Chat      | Expert Drop<br>Token Drop<br>Expanded Drop | 41.4<br>40.4<br><b>43.4</b> | 78.7<br>78.8<br><b>79.1</b> | 71.2<br><b>72.6</b><br><b>72.6</b> | 68.6<br>69.1<br><b>69.6</b> | 80.6<br>80.9<br><b>81.1</b>        | 52.9<br>53.0<br><b>53.4</b> | 79.1<br>80.0<br><b>80.3</b> | 58.1<br><b>59.3</b><br><b>59.3</b> | 49.4<br>51.9<br><b>52.1</b> | 64.4<br>65.1<br>65.6 |
|                       | Baseline                                   | 45.4                        | 81.4                        | 72.6                               | 75.5                        | 82.9                               | 61.0                        | 81.5                        | 57.3                               | 66.4                        | 69.3                 |
| DeepSeek-V2-Lite-Chat | Expert Drop<br>Token Drop<br>Expanded Drop | 41.8<br>45.2<br><b>45.4</b> | 77.6<br>78.3<br><b>79.4</b> | 71.9<br>72.6<br><b>73.3</b>        | 72.5<br>74.0<br><b>75.4</b> | 81.6<br><b>83.2</b><br><b>83.2</b> | 57.1<br>59.3<br><b>60.4</b> | 75.5<br>80.9<br><b>81.5</b> | 53.3<br><b>57.3</b><br>57.2        | 56.0<br>62.7<br><b>64.1</b> | 65.3<br>68.2<br>68.9 |
|                       | Baseline                                   | 47.4                        | 84.8                        | 71.8                               | 82.5                        | 88.5                               | 71.7                        | 87.5                        | 70.2                               | 64.2                        | 74.3                 |
| Mixtral-8×7B-Instruct | Expert Drop<br>Token Drop<br>Expanded Drop | 46.8<br>46.4<br><b>47.8</b> | 83.2<br>83.3<br><b>85.0</b> | 70.1<br>71.7<br><b>71.8</b>        | 81.3<br>82.2<br><b>83.0</b> | 87.6<br>88.3<br><b>88.6</b>        | 67.1<br>71.2<br><b>71.5</b> | 85.6<br>87.4<br><b>87.6</b> | 66.2<br>69.1<br><b>70.2</b>        | 62.3<br>64.7<br><b>64.6</b> | 72.2<br>73.8<br>74.5 |

**The Critical Role of Low-Load Experts** To explore the impact of low-load experts, we further compare dropping tokens (i.e., Token Drop) with skipping experts (i.e., Expert Drop). For Expert Drop, we adopt a conservative strategy that dynamically skips the 10% of experts with the lowest token loads. Notably, the proportion of tokens removed in Expert Drop is significantly lower than in Token Drop (2% in Expert Drop vs. 12% in Token Drop on OLMoE-Instruct).

Despite this, as shown in Table 2, Expert Drop experiences significant performance degradation and is outperformed by Token Drop by a large margin. Moreover, due to the small proportion of tokens assigned to low-load experts, removing these experts provides only marginal improvements in

inference speed (less than a 5% speedup). These findings indicate that retaining low-load experts better preserves the performance of MoE models.

To analyze expert selection and justify Expanded Drop, we sort, for each token, all experts by their gating scores in descending order and record the ranked scores (top-1, 2, ..., top-N). Aggregating across tokens, we compute the average, maximum, and minimum score at each rank (Figure 8). The curves show that while top-ranked experts receive much higher scores, the decay across ranks is gradual rather than abrupt, yielding a relatively flat tail beyond the first few ranks. Consequently, experts just outside the top-k often have scores comparable to those near the bottom of the top-k, enabling rerouting or dropping without materially changing model behavior,

<span id="page-8-0"></span>![](_page_8_Figure_3.jpeg)

Figure 8: Gating score distribution across ranked experts.

i.e., rerouted tokens still go to reasonably relevant experts with similar gating probabilities. This validates the design of Expanded Drop, as it can exploit the flatness of the gating distribution to balance load without sacrificing accuracy.

Effectiveness of Expanded Drop We examine the effectiveness of utilizing low-load experts by Expanded Drop instead of simply discarding these tokens to meet the target capacity. Comparing Expanded Drop with Token Drop, redistributing excess tokens to low-load experts enhances performance, yielding a 0.9% improvement in the average performance of Qwen1.5-MoE-Chat. Furthermore, considering the performance degradation observed in Expert Drop, our findings highlight the crucial role of low-load experts in maintaining model effectiveness.

<span id="page-8-1"></span>![](_page_8_Figure_7.jpeg)

Figure 9: Normalized expert load after Token Drop and Expanded Drop.

Expanded Drop overselects experts for each token to expand the selection scope and encourage more

balanced token-expert assignments. As shown in Figure 9, increasing the overselection ratio m allows tokens to consider more candidate experts after being dropped from overflowed ones, thereby improving low-load expert utilization and balancing the expert load.

Advanced Variant: Device-Level Capacity-Aware Inference In scenarios where multiple experts reside on the same device, the overall straggler effect is determined by the aggregated load on each device. Expert-level capacity constraints impose a strict limit. Specifically, the number of tokens assigned to a single expert must not exceed  $\gamma \bar{N}$ . When extended to a device with  $n_l$  local experts, this implies that the total load is bounded by  $n_l \cdot \gamma \bar{N}$ . However, this strict constraint can lead to over-aggressive token dropping, as a single overloaded expert may force token removal even when sufficient spare capacity remains across other experts on the same device.

To alleviate this issue, we introduce a **device-level capacity-aware** formulation, which applies the constraint at the device granularity rather than on individual experts. For instance, we enforce  $N_1 + N_2 + \cdots + N_{n_l} \leq n_l \cdot \gamma \bar{N}$ . Unlike expert-level constraints, which may drop tokens simply because a single expert exceeds its local budget, the device-level constraint admits these tokens as long as the *total* device-level load remains within the allowable bound. This results in a smoother, less restrictive, and more utilization-efficient alternative.

<span id="page-8-2"></span>Table 3: Comparison of Device-Level and Expert-Level capacity-aware inference on Qwen3-MoE.

| Method        | Granularity      | $\gamma$   | OBQA             | PIQA                | RTE                 | WinoGrande          | BoolQ               | ARC-C               | HellaSwag           | MMLU                | GSM8K   Avg.                     |
|---------------|------------------|------------|------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|----------------------------------|
| Baseline      | _                | $+\infty$  | 45.2             | 80.6                | 82.3                | 69.5                | 88.6                | 69.5                | 77.6                | 77.9                | 89.5   <u>75.6</u>               |
| Expanded Drop | Expert<br>Device | 1.5<br>1.0 | <b>43.6</b> 42.4 | 79.4<br><b>79.7</b> | 80.3<br><b>81.2</b> | 68.4<br><b>69.3</b> | 86.5<br><b>88.0</b> | 68.8<br><b>69.4</b> | 74.4<br><b>77.1</b> | 76.8<br><b>77.1</b> | 87.0   <u>73.9</u>   <u>74.8</u> |

We evaluate this variant and observe that device-level constraints consistently outperform expert-level constraints in downstream performance, as shown in Table 3. This more flexible constraint also leads to stronger practical speedups: on Qwen3-MoE (Yang et al., 2025) with  $\gamma=1$ , the device-level formulation achieves a  $1.31\times$  end-to-end speedup and a  $1.51\times$  speedup on a single MoE layer, both surpassing the  $1.23\times$  and  $1.40\times$  speedups obtained under expert-level constraints with  $\gamma=1.5$ .

#### 5.3 EXTENSION TO MULTIMODAL MIXTURE OF EXPERTS

In addition to applying capacity-aware inference to MoE models for language tasks, we also explore its effectiveness in multimodal MoE settings. Specifically, we evaluate the OLMoE-based MolmoE (Deitke et al., 2024) across multimodal benchmarks, including MME (Fu et al., 2023), MMBench (Liu et al., 2023), and SEED-Bench (Li et al., 2024a).

Given that the input sequence contains tokens from multiple modalities, we first investigate different token dropping strategies. Specifically, we first treat all tokens equally and drop those with the lowest scores ("Uniform"). Beyond this, considering the redundancy often found in image tokens, we also experiment with a strategy that prioritizes dropping image tokens before selectively removing text tokens ("Image First"). For comparison, we also consider dropping text tokens first ("Text First"). As shown in Table 4, on the MME benchmark, this image-first strategy yields improved performance, highlighting the benefit of prioritizing image-token dropping for load balance in multimodal MoE models.

<span id="page-9-0"></span>Table 4: Capacity-aware inference for multimodal MoE models. "Percep." and "Cognit." denote Perception and Cognition, respectively.  $\gamma$  is set to 1.0.

| Method        | Strategy    | Percep. | Cognit. |
|---------------|-------------|---------|---------|
| Baseline      | -           | 1358.1  | 269.6   |
| Token Drop    | Uniform     | 1248.4  | 245.4   |
| Expanded Drop |             | 1307.6  | 273.6   |
| Token Drop    | Text First  | 1114.2  | 214.4   |
| Expanded Drop |             | 1163.6  | 241.3   |
| Token Drop    | Image First | 1346.5  | 288.9   |
| Expanded Drop |             | 1362.1  | 297.1   |

<span id="page-9-1"></span>![](_page_9_Figure_7.jpeg)

![](_page_9_Figure_8.jpeg)

Figure 10: Multimodal token assignments across different experts.

Figure 11: Comparison on MMBench across six multimodal capabilities in Appendix C.

Given the redundancy of image tokens and their large proportion in multimodal tasks, we further investigate more aggressive capacity factors for Token Drop and Expanded Drop using the "Image First" strategy. Figure 11 demonstrates the effectiveness of Capacity-Aware Inference under low capacity constraints (i.e.,  $\gamma=0.5$ ). This is largely due to the high redundancy in image tokens (Chen et al., 2024), which allows a higher dropping ratio without significantly affecting performance. The dominance of image tokens also enables the use of very low capacity factors without significantly affecting text token retention, as illustrated in Figure 10. Dropping image tokens at higher ratios leads to more balanced token assignments and substantially improved inference efficiency.

#### 6 CONCLUSION

In this paper, we identify the issue of imbalanced token-to-expert assignment in Mixture of Experts (MoE) models and introduce the Straggler Effect during inference, where high-load experts become efficiency bottlenecks and dictate overall latency. To address this problem, we propose Capacity-Aware Token Drop, which mitigates expert overload by enforcing strict capacity constraints. Additionally, to better utilize underloaded experts, we present Capacity-Aware Expanded Drop, which allows tokens to select additional experts on the same device while still respecting capacity limits, thereby improving expert utilization. Our findings and proposed methods offer valuable insights and effective strategies for improving MoE inference efficiency.

## 7 ETHICS STATEMENT

Our work focuses on improving the efficiency of large-scale Mixture-of-Experts (MoE) models at inference time. The proposed methods are lightweight modifications to model execution and do not involve collecting or annotating new data. All experiments are conducted on publicly available pretrained models and standard open-source benchmarks. We emphasize that our contributions are intended solely for research and educational purposes in efficient machine learning. We release code under a research license and include clear terms discouraging misuse in sensitive applications such as mass surveillance or privacy-intrusive deployments.

## 8 REPRODUCIBILITY STATEMENT

We are committed to ensuring full reproducibility of our results. We release the complete implementation of Capacity-Aware Inference, including inference-time strategies (Token Drop and Expanded Drop), evaluation harness integration, and scripts for running all benchmarks. Detailed instructions are provided in a reproducibility README covering environment setup, library versions, random seeds, and commands for end-to-end replication.

## REFERENCES

<span id="page-10-4"></span>Winogrande: An adversarial winograd schema challenge at scale. 2019.

<span id="page-10-7"></span>Asma Ben Abacha, Wen-wai Yim, Yadan Fan, and Thomas Lin. An empirical study of clinical note generation from doctor-patient encounters. In *Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics*, pp. 2291–2302, Dubrovnik, Croatia, May 2023. Association for Computational Linguistics. URL [https://aclanthology.org/2023.](https://aclanthology.org/2023.eacl-main.168) [eacl-main.168](https://aclanthology.org/2023.eacl-main.168).

<span id="page-10-3"></span>Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical commonsense in natural language, 2019.

<span id="page-10-0"></span>Weilin Cai, Juyong Jiang, Fan Wang, Jing Tang, Sunghun Kim, and Jiayi Huang. A survey on mixture of experts, 2024. URL <https://arxiv.org/abs/2407.06204>.

<span id="page-10-5"></span>Iñigo Casanueva, Tadas Temcinas, Daniela Gerz, Matthew Henderson, and Ivan Vulic. Efficient intent detection with dual sentence encoders. In *Proceedings of the 2nd Workshop on NLP for ConvAI - ACL 2020*, mar 2020. URL <https://arxiv.org/abs/2003.04807>. Data available at https://github.com/PolyAI-LDN/task-specific-datasets.

<span id="page-10-2"></span>Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large visionlanguage models, 2024.

<span id="page-10-6"></span>Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. 2021.

<span id="page-10-1"></span>Zixiang Chen, Yihe Deng, Yue Wu, Quanquan Gu, and Yuanzhi Li. Towards understanding the mixture-of-experts layer in deep learning. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), *Advances in Neural Information Processing Systems*, 2022. URL <https://openreview.net/forum?id=MaYzugDmQV>.

- <span id="page-11-11"></span>Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions, 2019.
- <span id="page-11-10"></span>Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge, 2018.
- <span id="page-11-13"></span>Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*, 2021.
- <span id="page-11-4"></span>Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y Wu, et al. Deepseekmoe: Towards ultimate expert specialization in mixtureof-experts language models. *arXiv preprint arXiv:2401.06066*, 2024.
- <span id="page-11-8"></span>Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, and et al. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models. *arXiv preprint arXiv:2409.17146*, 2024.
- <span id="page-11-3"></span>Nan Du, Yanping Huang, Andrew M Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, et al. Glam: Efficient scaling of language models with mixture-of-experts. In *International Conference on Machine Learning*, pp. 5547–5569. PMLR, 2022.
- <span id="page-11-6"></span>DeepSeek-AI et al. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model, 2024a. URL <https://arxiv.org/abs/2405.04434>.
- <span id="page-11-0"></span>DeepSeek-AI et al. Deepseek-v3 technical report, 2024b. URL [https://arxiv.org/abs/2412.](https://arxiv.org/abs/2412.19437) [19437](https://arxiv.org/abs/2412.19437).
- <span id="page-11-1"></span>William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23(120):1–39, 2022.
- <span id="page-11-9"></span>Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models. *ArXiv*, abs/2306.13394, 2023. URL <https://api.semanticscholar.org/CorpusID:259243928>.
- <span id="page-11-14"></span>Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac'h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 12 2023. URL <https://zenodo.org/records/10256836>.
- <span id="page-11-7"></span>Jamie Hayes, Ilia Shumailov, and Itay Yona. Buffer overflow in mixture of experts. In *Neurips Safe Generative AI Workshop 2024*, 2024. URL <https://openreview.net/forum?id=SKWidEjUgU>.
- <span id="page-11-5"></span>Shwai He, Liang Ding, Daize Dong, Boan Liu, Fuqiang Yu, and Dacheng Tao. PAD-net: An efficient framework for dynamic networks. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 14354–14366, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.803. URL [https://aclanthology.org/2023.](https://aclanthology.org/2023.acl-long.803) [acl-long.803](https://aclanthology.org/2023.acl-long.803).
- <span id="page-11-12"></span>Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding, 2021.
- <span id="page-11-2"></span>Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. Adaptive mixtures of local experts. *Neural computation*, 3(1):79–87, 1991.

- <span id="page-12-8"></span>Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. *arXiv preprint arXiv:2401.04088*, 2024.
- <span id="page-12-4"></span>Michael I Jordan and Robert A Jacobs. Hierarchical mixtures of experts and the em algorithm. *Neural computation*, 6(2):181–214, 1994.
- <span id="page-12-14"></span>Aran Komatsuzaki, Joan Puigcerver, James Lee-Thorp, Carlos Riquelme Ruiz, Basil Mustafa, Joshua Ainslie, Yi Tay, Mostafa Dehghani, and Neil Houlsby. Sparse upcycling: Training mixture-ofexperts from dense checkpoints. In *The Eleventh International Conference on Learning Representations*, 2023. URL <https://openreview.net/forum?id=T5nUQDrM4u>.
- <span id="page-12-13"></span>Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for question answering research. *Transactions of the Association for Computational Linguistics*, 7:452–466, 2019. doi: 10.1162/tacl\_a\_00276. URL <https://aclanthology.org/Q19-1026/>.
- <span id="page-12-7"></span>Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. *arXiv preprint arXiv:2006.16668*, 2020.
- <span id="page-12-3"></span>Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. {GS}hard: Scaling giant models with conditional computation and automatic sharding. In *International Conference on Learning Representations*, 2021. URL <https://openreview.net/forum?id=qrwe7XHTmYb>.
- <span id="page-12-10"></span>Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. Seed-bench: Benchmarking multimodal large language models. *2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 13299–13308, 2024a. URL [https:](https://api.semanticscholar.org/CorpusID:271963485) [//api.semanticscholar.org/CorpusID:271963485](https://api.semanticscholar.org/CorpusID:271963485).
- <span id="page-12-12"></span>Tianle Li, Ge Zhang, Quy Duc Do, Xiang Yue, and Wenhu Chen. Long-context llms struggle with long in-context learning, 2024b.
- <span id="page-12-9"></span>Yuanzhan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multimodal model an all-around player? In *European Conference on Computer Vision*, 2023. URL <https://api.semanticscholar.org/CorpusID:259837088>.
- <span id="page-12-11"></span>Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering, 2018a.
- <span id="page-12-1"></span>Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering, 2018b. URL [https://arxiv.org/](https://arxiv.org/abs/1809.02789) [abs/1809.02789](https://arxiv.org/abs/1809.02789).
- <span id="page-12-0"></span>Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Pete Walsh, Oyvind Tafjord, Nathan Lambert, Yuling Gu, Shane Arora, Akshita Bhagia, Dustin Schwenk, David Wadden, Alexander Wettig, Binyuan Hui, Tim Dettmers, Douwe Kiela, Ali Farhadi, Noah A. Smith, Pang Wei Koh, Amanpreet Singh, and Hannaneh Hajishirzi. Olmoe: Open mixture-of-experts language models, 2024. URL <https://arxiv.org/abs/2409.02060>.
- <span id="page-12-2"></span>OpenAI. Gpt-4 technical report, 2024.
- <span id="page-12-6"></span>Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, André Susano Pinto, Daniel Keysers, and Neil Houlsby. Scaling vision with sparse mixture of experts. *Advances in Neural Information Processing Systems*, 34:8583–8595, 2021.
- <span id="page-12-5"></span>Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *arXiv preprint arXiv:1701.06538*, 2017a.

- <span id="page-13-1"></span>Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer, 2017b. URL <https://arxiv.org/abs/1701.06538>.
- <span id="page-13-9"></span>Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. *arXiv preprint arXiv:1909.08053*, 2019.
- <span id="page-13-7"></span>Yixin Song, Zeyu Mi, Haotong Xie, and Haibo Chen. Powerinfer: Fast large language model serving with a consumer-grade gpu, 2023.
- <span id="page-13-0"></span>Gemini Team. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, 2024a.
- <span id="page-13-5"></span>Qwen Team. Qwen1.5-moe: Matching 7b model performance with 1/3 activated parameters", February 2024b. URL <https://qwenlm.github.io/blog/qwen-moe/>.
- <span id="page-13-12"></span>Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. 2019. In the Proceedings of ICLR.
- <span id="page-13-3"></span>Fuzhao Xue, Zian Zheng, Yao Fu, Jinjie Ni, Zangwei Zheng, Wangchunshu Zhou, and Yang You. Openmoe: An early effort on open mixture-of-experts language models. *arXiv preprint arXiv:2402.01739*, 2024a.
- <span id="page-13-8"></span>Leyang Xue, Yao Fu, Zhan Lu, Luo Mai, and Mahesh Marina. Moe-infinity: Activation-aware expert offloading for efficient moe serving, 2024b.
- <span id="page-13-10"></span>An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025. URL <https://arxiv.org/abs/2505.09388>.
- <span id="page-13-11"></span>Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence?, 2019.
- <span id="page-13-6"></span>Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Y Zhao, Andrew M. Dai, Zhifeng Chen, Quoc V Le, and James Laudon. Mixture-of-experts with expert choice routing. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), *Advances in Neural Information Processing Systems*, 2022. URL [https://openreview.net/forum?id=](https://openreview.net/forum?id=jdJo1HIVinI) [jdJo1HIVinI](https://openreview.net/forum?id=jdJo1HIVinI).
- <span id="page-13-4"></span>Tong Zhu, Xiaoye Qu, Daize Dong, Jiacheng Ruan, Jingqi Tong, Conghui He, and Yu Cheng. Llama-moe: Building mixture-of-experts from llama with continual pre-training. *arXiv preprint arXiv:2406.16554*, 2024. URL <https://arxiv.org/abs/2406.16554>.
- <span id="page-13-2"></span>Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, and William Fedus. St-moe: Designing stable and transferable sparse expert models. *arXiv preprint arXiv:2202.08906*, 2022.

## <span id="page-14-0"></span>A IMPLEMENTATION DETAILS

**Models** We mainly focus on lightweight MoE models (less than 20B parameter budget). We conduct experiments on OLMoE (Muennighoff et al., 2024), Qwen1.5-MoE (Team, 2024b), DeepSeek-V2-Lite (et al., 2024a), Mixtral (Jiang et al., 2024), MolmoE (Deitke et al., 2024), and Qwen3-MoE (Yang et al., 2025), due to their competitive performance and widespread applications.

**Datasets** To evaluate model performance, we report normalized zero-shot or few-shot accuracy on the LM-Harness benchmark. The number of shots for each task is detailed in Table 5, which includes multiple tasks: ARC-C (Clark et al., 2018), BoolQ (Clark et al., 2019), HellaSwag (Zellers et al., 2019), MMLU (Hendrycks et al., 2021), OBQA (Mihaylov et al., 2018a), PIQA (Bisk et al., 2019), RTE (Wang et al., 2019), Wino-Grande (ai2, 2019), and GSM8K (Cobbe et al., 2021). The evaluation code is based on EleutherAI's LM Harness framework (Gao et al., 2023).

<span id="page-14-1"></span>Table 5: Experimental settings for evaluated language tasks. "Norm" refers to the normalization performed with respect to the length of the input.

| Task       | Number of few-shot | Metric          |
|------------|--------------------|-----------------|
| BoolQ      | 0                  | Accuracy        |
| RTE        | 0                  | Accuracy        |
| OBQA       | 0                  | Accuracy (Norm) |
| PIQA       | 0                  | Accuracy (Norm) |
| MMLU       | 5                  | Accuracy        |
| WinoGrande | 5                  | Accuracy        |
| GSM8K      | 5                  | Exact Match     |
| HellaSwag  | 10                 | Accuracy (Norm) |
| ARC-C      | 25                 | Accuracy (Norm) |

## B ADDITIONAL RESULTS ON LANGUAGE TASKS

Long-context understanding poses substantial challenges for language models. We therefore investigate whether Token Drop and Expanded Drop can maintain competitive performance in this setting. Table 6 presents their results on a long in-context learning benchmark. Following LongICLBench (Li et al., 2024b), we evaluate on the BANKING77 dataset (Casanueva et al., 2020), with the only modification being that we sample 100 test examples. BANKING77 is an intent detection dataset in the banking domain with 77 classes. We consider 1-shot/label to 5-shot/label settings, corresponding to context lengths of 2K, 4K, 7K, 9K, and 14K tokens. Across different input sequence lengths, both Token Drop and Expanded Drop consistently match the performance of the baseline. This demonstrates the strong generalization ability and wide applicability of these techniques.

<span id="page-14-2"></span>Table 6: Performance of Token Drop and Expanded Drop on DeepSeek-V2-Lite in LongICLBench.

| Method        | 2K   | 4K   | 7K   | 9K   | 11K  | Avg. |
|---------------|------|------|------|------|------|------|
| Baseline      | 0.28 | 0.44 | 0.56 | 0.62 | 0.70 | 0.52 |
| Token Drop    | 0.27 | 0.43 | 0.57 | 0.65 | 0.71 | 0.53 |
| Expanded Drop | 0.23 | 0.53 | 0.63 | 0.62 | 0.70 | 0.54 |

To further assess the robustness of Token Drop and Expanded Drop in generation tasks, we test additional benchmarks such as HumanEval (Chen et al., 2021), NQ-Open (Kwiatkowski et al., 2019), and MTS-Dialog (Ben Abacha et al., 2023) in Table 7.

<span id="page-14-3"></span>Table 7: Performance of Token Drop and Expanded Drop across three MoE-based LLMs on HumanEval (HE), NQ-Open (NQ), and MTS-Dialog (MTS). The capacity factor is set as 1.0 here and we report pass@1 for HE, exact\_match for NQ and BERTScore for MTS, respectively.

| Method        | OL    | MoE-Ins | struct | Qwen1.5-MoE-Chat |       |       | DeepSeek-V2-Lite-Chat |       |       |  |
|---------------|-------|---------|--------|------------------|-------|-------|-----------------------|-------|-------|--|
|               | HE    | NQ      | MTS    | HE               | NQ    | MTS   | HE                    | NQ    | MTS   |  |
| Baseline      | 0.311 | 0.175   | 0.827  | 0.280            | 0.226 | 0.874 | 0.506                 | 0.265 | 0.830 |  |
| Token Drop    | 0.299 | 0.159   | 0.823  | 0.281            | 0.214 | 0.869 | 0.494                 | 0.252 | 0.831 |  |
| Expanded Drop | 0.303 | 0.170   | 0.829  | 0.291            | 0.224 | 0.872 | 0.508                 | 0.268 | 0.831 |  |

The consistent performance on code generation, open-domain question answering, and dialog creation demonstrates the robustness of Token Drop and Expanded Drop for long-range generation tasks.

## <span id="page-15-0"></span>C ADDITIONAL RESULTS ON MULTIMODAL TASKS

In the scope of this paper, multimodal tasks refer to those involving both vision and language modalities. We evaluate model performance using three representative benchmarks: MME, MMBench, and SEED-Bench, each targeting different aspects of multimodal understanding and reasoning.

MME benchmark evaluates vision-language models along two dimensions: perception, which tests visual grounding and recognition, and cognition, which assesses reasoning abilities such as counting and relational understanding. It provides a fine-grained analysis of multimodal understanding.

MMBench is a comprehensive benchmark designed to assess the general multimodal understanding ability of vision-language models. It evaluates model performance across six core capabilities: Coarse Perception (CP), Fine-grained Perception-including single-instance (FP-S) and cross-instance (FP-C), Attribute Reasoning (AR), Logical Reasoning (LR), and Relational Reasoning (RR). By covering both perception and reasoning-oriented tasks, MMBench provides detailed insights into the strengths and limitations of VLMs across diverse multimodal scenarios.

SEED-Bench is a large-scale benchmark for evaluating the generative comprehension of Multimodal Large Language Models (MLLMs) across both image and video modalities. It includes 19K human-annotated multiple-choice questions spanning 12 evaluation dimensions, enabling objective and efficient assessment without human or GPT intervention. SEED-Bench reveals model limitations and maintains a public leaderboard to support fair comparison and future research.

<span id="page-15-1"></span>

| Component   | Content                                                                                                            | Token Count |
|-------------|--------------------------------------------------------------------------------------------------------------------|-------------|
| Image       |                                                                                                                    | 576         |
| Text Prompt | Is this artwork titled virgin and child with sts catherine, cecilia, barbara, and ursula? Please answer yes or no. | 31          |
| Total       | -                                                                                                                  | 607         |

Table 8: An example multimodal query in the MME benchmark, showing the dominant proportion of image tokens compared to text tokens.

As shown in Table 8, these tasks typically introduce a large number of image tokens. When faced with imbalanced token-to-expert assignments, dropping redundant image tokens significantly improves load balancing. Moreover, due to the high redundancy among image tokens, dropping a portion of them has minimal impact on model performance.

As in MME and MMBench, the Image-First variants of Token Drop and Expanded Drop also exhibit consistent effectiveness on SEED-Bench (Li et al., 2024a), maintaining strong performance even under low capacity factors such as  $\gamma=0.5$ . In addition to the redundancy in image tokens, Figure 10 shows that text tokens constitute only a small portion of the total token assignments. This allows the regulation of  $\gamma$  to retain almost all text tokens under the Image-First strategy.

| Method        | γ   | Inst. Attr. | Inst. ID | Inst. Interact. | Inst. Loc. | Inst. Count | Scene | Spatial | Text | Reasoning | Overall |
|---------------|-----|-------------|----------|-----------------|------------|-------------|-------|---------|------|-----------|---------|
| Baseline      | ∞   | 74.2        | 71.4     | 58.8            | 62.8       | 57.0        | 73.5  | 49.6    | 72.6 | 76.4      | 68.7    |
| Token Drop    | 0.5 | 70.4        | 67.8     | 60.8            | 57.8       | 51.8        | 71.5  | 43.5    | 58.3 | 71.3      | 64.9    |
| Expanded Drop | 0.0 | 71.2        | 67.3     | 57.7            | 58.8       | 53.6        | 71.0  | 45.2    | 64.3 | 71.9      | 65.5    |
| Token Drop    | 1.0 | 73.6        | 70.2     | 58.8            | 62.6       | 56.9        | 72.7  | 48.1    | 61.9 | 73.4      | 68.0    |
| Expanded Drop | 1.0 | 73.8        | 70.7     | 59.8            | 63.5       | 56.7        | 73.1  | 49.6    | 70.2 | 74.6      | 68.4    |
| Token Drop    | 1.5 | 73.5        | 71.2     | 58.8            | 62.9       | 57.1        | 73.1  | 48.7    | 65.5 | 74.6      | 68.3    |
| Expanded Drop | 1.5 | 73.7        | 71.0     | 61.9            | 64.7       | 57.4        | 73.3  | 49.3    | 71.4 | 76.1      | 68.7    |

Table 9: Token Drop and Expanded Drop strategies for multimodal MoE models evaluated on SEED-Bench. Abbreviations: Inst. Attr. = Instance Attributes; Inst. ID = Instance Identification; Inst. Interact. = Instance Interaction; Inst. Loc. = Instance Localization; Inst. Count = Instance Counting.

## D ABLATION STUDY

**Model-Specific Imbalanced Property** We explore the imbalance property in various models, such as OLMoE, DeepSeek, Qwen, and Mixtral, which differ in both architectures (e.g., depth and width) and training strategies (e.g., training from scratch (Muennighoff et al., 2024; et al., 2024a) vs. training after upcycling (Jiang et al., 2024; Team, 2024b)).

On the one hand, our findings in Appendix F reveal that different training strategies result in significantly varying levels of imbalance. Specifically, MoE models trained from scratch exhibit a much higher degree of imbalance. For instance, OLMoE and DeepSeek-V2-Lite experience peak expertwise token allocations exceeding  $5\bar{N}$ , whereas Qwen1.5-MoE and Mixtral are upcycled from dense language models and maintain a more balanced distribution, with peak expert-wise allocations staying below  $3\bar{N}$ . This is because upcycling initializes all experts with identical parameters (Komatsuzaki et al., 2023), reducing divergence and promoting balanced training in the early stages.

On the other hand, despite the widespread use of auxiliary balance loss in MoE training, it does not guarantee balanced token assignments across experts, as token distribution still varies significantly during inference on test data. This necessitates integrating expert capacity into the inference process.

<span id="page-16-1"></span>![](_page_16_Figure_5.jpeg)

Figure 12: Performance change as capacity factors decrease from 3.0 to 0.0.

Capacity Factor Beyond the specific capacity values presented in Table 1, we further investigate a wide range of capacity factors in Figure 12, spanning from 0.0 to 3.0. We exclude values exceeding 3.0, as their performance closely aligns with capacity-agnostic scenarios. By analyzing the performance changes when decreasing the capacity factor, we find that setting  $\gamma$  to 1.5 is sufficient to maintain performance comparable to the original models. However, maintaining performance becomes challenging under excessively low capacity factors, as high-load experts are forced to drop a significant number of tokens.

**Speedup Measurement Under Different Workloads** To assess the effectiveness of Token Drop in efficiency, we measure speedup under a range of token workloads. Specifically, we vary both the sequence length and batch size, and present the speedup results on DeepSeek-V2 with a capacity factor of 2.0 in Table 10.

<span id="page-16-2"></span>

| Batch Size    | 8K    | 8K            | 8K           | 4K           | 2K           | 2K           | 1K           | 1K           | 1 K          |
|---------------|-------|---------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| Prompt Length | 0.1K  | 0.2K          | 0.4K         | 1K           | 1K           | 2K           | 1K           | 2K           | 4 K          |
| Speedup       | 1.09× | $1.18 \times$ | $1.24\times$ | $1.26\times$ | $1.27\times$ | $1.27\times$ | $1.27\times$ | $1.24\times$ | $1.23\times$ |

Table 10: Speedup results across varying batch sizes and prompt lengths.

The straggler effect becomes more pronounced under heavier workloads, where GPUs operate at higher utilization with limited spare capacity, making the speedup more noticeable. In practical server-side MoE deployments, workloads are substantially higher, and our techniques effectively mitigate the resulting straggler effect.

<span id="page-16-0"></span>**Maximum Number of Selected Experts** Expanded Drop adopts an overselection-and-dropping strategy that not only maintains load balance but also improves the utilization of underloaded experts. Although this mechanism allows some tokens to select more than k experts, Table 11 demonstrates that such flexibility benefits downstream task performance, suggesting that enforcing a strict maximum of k experts is unnecessary. Allowing additional expert selections can enhance representational capacity, whereas rigid constraints may unnecessarily limit model performance.

<span id="page-17-2"></span>Table 11: Ablation study on limiting the maximum number of k selected experts. "w/max" and "w/o max" indicate runs *with* and *without* this constraint, respectively. γ is set to 1.0.

| m  | Method  | OBQA | PIQA | RTE  | WinoGrande | BoolQ | ARC-C | HellaSwag | MMLU | Avg. |
|----|---------|------|------|------|------------|-------|-------|-----------|------|------|
| 2k | w/ max  | 42.4 | 75.8 | 53.2 | 68.6       | 72.7  | 50.3  | 77.1      | 47.6 | 61.0 |
|    | w/o max | 42.2 | 75.9 | 53.4 | 69.7       | 73.1  | 50.3  | 77.0      | 47.8 | 61.2 |
| 3k | w/ max  | 42.0 | 75.6 | 53.4 | 69.6       | 72.7  | 50.3  | 77.0      | 47.4 | 61.0 |
|    | w/o max | 42.0 | 75.6 | 53.8 | 69.8       | 72.9  | 50.3  | 77.1      | 47.6 | 61.1 |

## E IMPLEMENTATION-LEVEL CODE FOR TOKEN DROP AND EXPANDED DROP

The detailed implementation-level code for Token Drop and Expanded Drop is presented in Algorithms [1](#page-17-3) and [2,](#page-17-3) respectively.

#### <span id="page-17-3"></span>Algorithm 1 Token Drop **# x: input tokens; # k: top-k experts; # gamma: capacity factor; def token\_drop(self, x, k, gamma): logits = self.gate(x) scores = torch.softmax(logits, dim=-1) topk\_scores, topk\_idx = scores.topk(k, dim=1) topk\_mask = torch.zeros\_like(scores).scatter(1, topk\_idx, 1) masked\_scores = scores \* topk\_mask N, E = scores.size() cap = int(gamma \* (N \* k) / E) \_, keep\_idx = masked\_scores.topk(cap, dim=0) cap\_mask = torch.zeros\_like(scores).scatter(0, keep\_idx, 1) final\_map = topk\_mask \* cap\_mask final\_scores = masked\_scores \* final\_map return final\_scores, final\_map** Algorithm 2 Expanded Drop **# x: input tokens; # k: top-k experts; # gamma: capacity factor; # local\_ids: local device expert ids; def expanded\_drop(self, x, k, gamma, local\_ids): logits = self.gate(x) scores = torch.softmax(logits, dim=-1) topk\_scores, topk\_idx = scores.topk(k, dim=1) N, E = scores.size() local\_idx = torch.tensor(local\_ids, device=x. device).repeat(N, 1) exp\_idx = torch.cat([topk\_idx, local\_idx], dim =1) local\_scores = scores[:, local\_ids] exp\_scores = torch.cat([topk\_scores, local\_scores], dim=1) exp\_mask = torch.zeros\_like(scores).scatter(1, exp\_idx, 1) masked\_scores = scores \* exp\_mask cap = int(gamma \* (N \* k) / E) \_, keep\_idx = masked\_scores.topk(cap, dim=0) cap\_mask = torch.zeros\_like(scores).scatter(0, keep\_idx, 1) final\_map = exp\_mask \* cap\_mask final\_scores = masked\_scores \* final\_map**

## <span id="page-17-0"></span>F LAYER-WISE EXPERT LOAD

To analyze imbalanced token assignments, we measure the expert load for each expert by tracking the peak expert load while running MoE models on various test datasets. Figure [13,](#page-18-0) [14,](#page-19-0) [15](#page-20-0) and [16](#page-21-0) present the full results for the normalized layer-wise expert load for OLMoE, DeepSeek-V2, Qwen1.5-MoE, and Mixtral-8×7B-Instruct, respectively.

**return final\_scores, final\_map**

## <span id="page-17-1"></span>G CALCULATION OF DROPPED TOKENS

Based on Equation [11,](#page-7-2) we calculate the total number of dropped tokens across experts in each layer under different capacity factors, as illustrated in Figures [17,](#page-22-0) [19,](#page-24-0) [18,](#page-23-0) and [20.](#page-25-0)

<span id="page-18-0"></span>![](_page_18_Figure_1.jpeg)

Figure 13: Layer-wise expert load in OLMoE-Instruct.

<span id="page-19-0"></span>![](_page_19_Figure_1.jpeg)

Figure 14: Layer-wise expert load in DeepSeek-V2-Lite.

<span id="page-20-0"></span>![](_page_20_Figure_1.jpeg)

Figure 15: Layer-wise expert load in Qwen1.5-MoE-Chat.

<span id="page-21-0"></span>![](_page_21_Figure_1.jpeg)

Figure 16: Layer-wise expert load in Mixtral-8×7B-Instruct.

<span id="page-22-0"></span>![](_page_22_Figure_1.jpeg)

Figure 17: Dropped tokens with respect to capacity factors in OLMoE-Instruct.

<span id="page-23-0"></span>![](_page_23_Figure_1.jpeg)

Figure 18: Dropped tokens with respect to capacity factors in DeepSeek-V2-Chat.

<span id="page-24-0"></span>![](_page_24_Figure_1.jpeg)

Figure 19: Dropped tokens with respect to capacity factors in Qwen1.5-MoE-Chat.

<span id="page-25-0"></span>![](_page_25_Figure_1.jpeg)

Figure 20: Dropped tokens with respect to capacity factors in Mixtral-8×7B-Instruct.