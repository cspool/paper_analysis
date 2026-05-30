# Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized Mixture-of-Expert Models

Zihan Qiu\*1, Zeyu Huang\*2, Bo Zheng\*1, Kaiyue Wen³, Zekun Wang¹, Rui Men¹ Ivan Titov², Dayiheng Liu⊠¹, Jingren Zhou¹, Junyang Lin⊠¹ Qwen Team, Alibaba Group ²University of Edinburgh ³Stanford University

### **Abstract**

This paper revisits the implementation of Load-balancing Loss (LBL) when training Mixture-of-Experts (MoEs) models. Specifically, LBL for MoEs is defined as  $N_E \sum_{i=1}^{N_E} f_i p_i$ , where  $N_E$  is the total number of experts,  $f_i$  represents the frequency of expert i being selected, and  $p_i$  denotes the average gating score of the expert i. Existing MoE training frameworks usually employ the parallel training strategy so that  $f_i$  and the LBL are calculated within a micro-batch and then averaged across parallel groups. In essence, a micro-batch for training billion-scale LLMs normally contains very few sequences. So, the micro-batch LBL is almost at the sequence level, and the router is pushed to distribute the token evenly within each sequence. Under this strict constraint, even tokens from a domain-specific sequence (e.g., code) are uniformly routed to all experts, thereby inhibiting expert specialization. In this work, we propose calculating LBL using a global-batch to loose this constraint. Because a global-batch contains much more diverse sequences than a micro-batch, which will encourage load balance at the corpus level. Specifically, we introduce an extra communication step to synchronize  $f_i$  across micro-batches and then use it to calculate the LBL. Through experiments on training MoEs-based LLMs (up to 42.8B total parameters and 400B tokens), we surprisingly find that the global-batch LBL strategy yields excellent performance gains in both pre-training perplexity and downstream tasks. Our analysis reveals that the global-batch LBL also greatly improves the domain specialization of MoE experts.

#### 1 Introduction

In recent years, the Mixture-of-Experts (MoE) framework (Szymanski & Lemmon, 1993; Shazeer et al., 2017) has become a popular technique to scale the model parameters up (Jiang et al., 2024; Dai et al., 2024; Liu et al., 2024a; Yang et al., 2024). For instance, Mixtral-8x22B (Jiang et al., 2024) (141B), Deepseek-v3 (Liu et al., 2024a) (671B) and MiniMax-01 (Li et al., 2025) (456B) reach a scale of hundreds of billion parameters while maintaining affordable training and inference efficiency. Typically, standard MoE comprises a *router* network and a group of parallel *expert* modules. Given a set of inputs, the *router* distributes each input to its corresponding experts conditionally and sparsely. Then, the outputs from individual experts are aggregated based on the importance weight that the router assigned to the expert.

One critical factor for training MoE-based models is encouraging the router to assign input to experts in a balanced manner (Shazeer et al., 2017; Fedus et al., 2022; Zoph et al., 2022; Qiu et al., 2024a). The reasons are twofold: (1) *effectiveness*: if the router continually prioritizes some experts during training, these experts will get more updates than others and will soon dominate that MoE layer, finally resulting in parameter redundancy issue (Shazeer et al., 2017; Wang et al., 2024); (2) *efficiency*: training and deploying large-scale MoE-based models often requires the *Expert Parallel*, where different experts will be in different parallel groups to process their inputs. Then, their outputs will be gathered and aggregated. In this case, the imbalanced expert utilization would heavily slow the forward process. Therefore, previous works training MoE-based LLMs generally employ an auxiliary loss, called Load-balancing Loss (LBL), to encourage the balanced routing decision (Shazeer et al., 2017).

Nevertheless, in most open-source MoE training frameworks like Deepspeed-MoE (Liu et al., 2024a), Tutel (Hwang et al., 2023), Megablocks (Gale et al., 2023) and Megatron-Core (Shoeybi et al., 2019), the LBL is calculated at the *micro-batch level*, which, as we will soon empirically demonstrate, negatively affects the performance and expert specialization of MoE-based LLMs. Specifically, during large-scale MoE training, each micro-batch usually contains only up to thousands of tokens and, thus, only a handful of sequences. Therefore, the micro-batch LBL is almost calculated at the *sequence level*. Suppose

<sup>\*</sup>Equal Contribution

<sup>&</sup>lt;sup>™</sup>Corresponding authors

<span id="page-1-0"></span>![](_page_1_Figure_0.jpeg)

Figure 1: The impact of the Balance BSZ on (a) model performance and (b) expert specialization. (a) When employing micro-batch level load balance, methods based on LBL and based on auxiliary-loss-free (Wang et al., 2024) approaches perform worse than employing the global-batch balance. (b) When employing micro-batch balance, there is no significant difference in the selection frequency of different domain-specific data, and the selection frequency of different experts within the same domain is approximately the same. With global-batch balance, there is a noticeable difference in the selection frequency of experts on different domain data, and within the same domain, there are experts with high selection frequency (marked in blue).

a micro-batch contains some domain-specific sequences (*i.e.* code and math); the micro-batch LBL still pushes routers to distribute these domain-specific tokens to all experts evenly, introducing an overly strict constraint and may hurt the model performance.

Consequently, we propose calculating the LBL at the *global-batch* level by synchronizing the expert selection frequency across parallel groups and then computing the LBL. According to the Fig. 1 (a), the global-batch LBL significantly enhances model performance (approximately 0.1 in pre-training PPL and 2 in benchmark scores). Additionally, Fig. 1 (b) showcases that the domain specialization only clearly emerges when trained with the global-batch LBL. We demonstrate that the model performance effectively increases with the global batch size (Section 4.2). We conducted further ablation studies to verify that introducing more diverse training tokens instead of more training token numbers is the main contributor to performance gains (Section 5). Because the expert selection frequency is just an expert-number-dimensional vector, our method introduces less than 3% latency under appropriate configurations and achieves more performant and interpretable models.

In summary, we investigate the challenges associated with the LBL in training MoEs models. By introducing global-batch LBL, we achieve improved performance and foster expert specialization within the MoE model. We believe this advancement addresses an essential limitation in existing MoE training, offering a novel perspective for MoEs model optimization. Though mainly experimenting with language-based tasks, we hope our work could pave the way for training stronger and more specialised MoE models in various domains, *e.g.*, Computer Vision and Multi-Modality.

## 2 Preliminary

#### 2.1 Mixture-of-Experts

MoEs consist of several parallel modules (the 'experts') and a router that assigns weights to each expert for a given input. (Szymanski & Lemmon, 1993; Shazeer et al., 2017) Combined with the transformer layer (Vaswani, 2017), the most common approach is to introduce a set of parallel feed-forward networks (FFN). Suppose there are  $N_E$  experts, denoted as  $E_i, i \in [1, N_E]$ . The router g followed by a softmax function maps the input  $\mathbf{x}$  to a score distribution over the experts, softmax( $g(\mathbf{x})$ )  $\in \mathbb{R}^{N_E}$ . Typically, for each input, only topK experts with the highest scores are activated and used. Given  $\mathbf{x} \in \mathbb{R}^h$ , the output  $\mathbf{y} \in \mathbb{R}^h$  is the weighted sum of the outputs from all experts:

$$\mathbf{y} = \sum_{i \in N_E, g_i \in \text{topK}(g)} g_i(\mathbf{x}) E_i(\mathbf{x})$$
(1)

#### 2.2 Load-balancing Loss

The Load-balancing Loss (LBL) in training MoE models is a regularization technique that encourages balanced expert utilization and prevents expert collapse (Fedus et al., 2022). Without the LBL, the model tends to concentrate its updates on a limited subset of experts, leading to a severe imbalance in expert utilization. To address this issue, LBL penalizes the router if it routes excessive tokens to a few particular experts. To compute LBL for a batch of tokens, we consider the fraction of tokens  $f_i$  routed to each expert  $E_i$  and the total routing probability  $P_i$  allocated to the expert  $E_i$ . The LBL is calculated as the sum of the product of  $f_i$  and  $P_i$  across all experts  $N_E$ , normalized by the number of experts:

$$LBL = N_E \sum_{i=1}^{N_E} f_i \cdot P_i.$$
 (2)

By minimizing the load-balancing loss, the model is encouraged to distribute the considered tokens more evenly among the experts, ensuring that each expert receives a fair share of updates during training. This helps maintain a balanced utilization of experts and prevents the model from collapsing into only activating just a few experts.

However, when employing data parallelism and model parallelism strategies for LLMs pre-training, each parallel group (i.e., one GPU) only has data from very limited domains. Existing MoE frameworks (Shoeybi et al., 2019; Gale et al., 2023) only utilize the information of  $P_i$  and  $F_i$  within every single parallel group when calculating the LBL and then perform an all-gather operation to average:

<span id="page-2-0"></span>
$$LBL_{micro} = \frac{1}{N_P} \sum_{i=1}^{N_P} (N_E \sum_{i=1}^{N_E} f_i^j \cdot P_i^j).$$
 (3)

 $N_P$  is the number of parallel groups and  $f_i^j$ ,  $P_i^j$  are the fraction and probability in parallel state j. This loss requires the model to *achieve load balance within each parallel group*, thus we call it LBL<sub>micro</sub>. However, suppose one parallel group (or one micro-batch) contains the domain-specific sequences. In that case, the router is still pushed to distribute tokens uniformly to all experts, thereby hurting MoE performance and preventing specialization. This situation is even more common regarding LLMs pretraining. Because to better control the data diversity, one micro-batch is usually formed with packed and truncated sequences from one specific domain, and a global-batch consists of micro-batches sampled from different domains according to specific data recipes (Ding et al., 2024; Yang et al., 2024). So the micro-batch-based balancing approach will hinder the MoE model from allocating data from specific domains to certain experts, which also partially explains why most MoE models only observe token-level expert routing patterns rather than expert-level selections. (Jiang et al., 2024; Xue et al., 2024).

#### 3 Method

This section introduces how to turn the micro-batch LBL into global-batch LBL by allowing different parallel groups to synchronize their expert select frequencies. We then discuss the scenario in which the number of compute nodes is limited and the sum of micro-batches is smaller than the global batch size. In such cases, we propose using a buffer to store the synchronized expert select counts at each gradient accumulation (GA) step to approximate the global batch LBL.

<span id="page-2-1"></span>**Synchronizing expert selection frequency across parallel groups.** Thanks to the format of the LBL calculation in each parallel group, as shown in equation 3, we can synchronize  $f_i$  across all parallel groups to get the  $\bar{f}_i$  for the global batch. This allows the final global average LBL to be equivalent to the LBL computed by aggregating statistics from all tokens in the global-batch.

$$LBL_{global} = N_E \sum_{i=1}^{N_E} \bar{f}_i \cdot \bar{P}_i$$
 (4)

$$= N_E \sum_{i=1}^{N_E} \bar{f}_i \cdot (\frac{1}{N_P} \sum_{i=1}^{N_P} P_j)$$
 (5)

$$= \frac{1}{N_P} \sum_{i=1}^{N_P} (N_E \sum_{i=1}^{N_E} \bar{f}_i \cdot P_i^j)$$
 (6)

Communicating only  $f_i \in \mathbb{R}^{N_E}$  avoids the communication overhead of directly transmitting the token-expert selection matrix of each parallel group and the expert selection scores (with a shape of tokens numbers  $\times$  experts numbers). We develop this synchronization method independently and existing work GRIN (Liu et al., 2024b) also proposes this synchronization. We will discuss this in the related work part.

#### <span id="page-3-1"></span>Algorithm 1 Approximate Global-Batch LBL

```
1: Initialize an empty buffer for each expert, c_i = 0

2: while training continues do

3: for each gradient accumulation step do

4: Add c_i with new synchronized selection counts for expert i

5: Calculate the current f_i with c_i, i \in N_E in the buffer

6: end for

7: Optimizer step, clear gradient

8: Reset the buffer with c_i = 0

9: end while
```

<span id="page-3-2"></span>Using a buffer to approximate the Global-Batch LBL. However, when training LLMs, the global-batch size is often up to  $10^3$ . When each micro-batch size is less than  $10^1$ , due to the limited number of compute nodes, the sum of all micro-batch sizes is smaller than the global-batch size, thus gradient accumulation (GA) is often used. In this situation, we introduce a buffer to store synchronized  $c_i$ , the expert i's selection count across micro-batches in one gradient accumulation step. Then, the information in the buffer is used to calculate the current  $f_i$  at each GA step. After completing the GA, the buffer is reset. The complete algorithm is shown in the Alg. 1. Through this accumulation process,  $f_i$  approaches  $\bar{f}_i$  with gradient accumulation steps, approximating LBL<sub>global</sub> with limited compute nodes.

#### 4 Experiments

#### 4.1 Experimental Setups

**Model Architecture and Training Settings** We conduct experiments on three sizes of MoE models: **(1)** 3.4B total parameters with 0.6B activated (**3.4A0.6B**); **(2)** 15B total parameters with 2.54B activated (**15A2.54B**), and **(3)**43B total parameters with 6.6B activated (**43A6.6B**). Each model utilizes the finegrained expert (Dai et al., 2024) and shared experts (Rajbhandari et al., 2022; Dai et al., 2024) methods. Specifically, the 3.4A0.6B model employs 64 total experts with top4 activated and 4 shared experts, while the 15A2.54B and 43A6.6B models use a setting of 160 total experts with top4 activated and 4 shared experts. All models default to using softmax gating, micro-batch LBL, and z-loss. The auxiliary loss weights follow previous works (Zoph et al., 2022). To avoid the impact of token drop for different methods, we use the dropless routing strategy like dMoE (Gale et al., 2023). In the 3.4A0.6B setting, we also implement the auxiliary loss free (with sigmoid gating) method (Wang et al., 2024). We train the models on 120B and 400B high-quality tokens, encompassing multilingual, math, and general knowledge content. A sequence length of 4096 is used, with global-batch sizes of 512 and 1024 for the 120B and 400B training settings, respectively, comprising 60k and 100k optimization steps. Other hyperparameters follow the default values of the AdamW optimizer. We use the term **Balance BSZ** to indicate the number of tokens considered when calculating the expert selection frequency.

**Evaluation** We mainly test the zero-shot downstream task capabilities of the models on four popular benchmarks, including English, Hellaswag (Zellers et al., 2019), general knowledge MMLU (Hendrycks et al., 2020), math GSM8k (Cobbe et al., 2021), and Chinese proficiency C-eval (Huang et al., 2024). Given that benchmarks that are evaluated with accuracy have certain random factors, for more detailed analysis, we mainly refer to the PPL on held-out test sets, which include SFT-EN, EN-Literature, SFT-Code, SFT-Math, SFT-ZH, ZH-Law, ZH-Literature, and SFT-Other from different domains.

#### <span id="page-3-0"></span>4.2 Main Results

Global load balance boosts model performance. In this section, we compare using micro-batch and global-batch balances on three different sizes of MoE models under various training scales. The 3.4A0.6 B model can be trained using only data parallelism, with a maximum micro-batch size of 4. If  $f_i$  is synchronized among the 8 GPUs on the same node, the Balance BSZ can reach 32. When training with 16 nodes and synchronizing across data parallel groups, the Balance BSZ can reach 512. From the first part of Tab. 1, it can be seen that as the Balance BSZ increases, all metrics consistently improve. For the aux-free method, we also compare the results under micro-batch and global-batch conditions and find the latter is much more better. For the 3.4A0.6B model trained on 400B tokens, we compare the results when the Balance BSZ could only reach 128 due to the limited number of compute nodes with the results when a buffer is added to approximate the global-batch. The latter's performance is closer to the results with a Balance BSZ of 512 and significantly better than 128, proving that introducing a buffer can approximate the global-batch when nodes are limited. As training the 15A2.54B and 43A6.6B models requires using

<span id="page-4-0"></span>Table 1: Performance of different balance methods and Balance BSZ. 'LBL' refers to using LBL, and Aux Free refers to the auxiliary loss free method (Wang et al., 2024). 'LBL+sync' means synchronizing expert selection frequency across parallel groups in 3. 'LBL+sync' means further using a buffer to expand the Balance BSZ in 3.

| <b>Balance Method</b>                                    | Balance BSZ                                              | Hellaswag | MMLU  | GSM8k | C-eval | Avg PPL |
|----------------------------------------------------------|----------------------------------------------------------|-----------|-------|-------|--------|---------|
| MoE-3.4A0.6B (Train 120B Tokens, Global Batch Size 512)  |                                                          |           |       |       |        |         |
| LBL                                                      | 4                                                        | 62.81     | 41.63 | 13.57 | 41.87  | 8.167   |
| LBL+sync                                                 | 32                                                       | 63.58     | 42.08 | 15.01 | 41.58  | 8.062   |
| LBL+sync                                                 | 512                                                      | 63.75     | 43.48 | 15.31 | 44.95  | 8.038   |
| Aux Free                                                 | 4                                                        | 61.99     | 41.30 | 12.43 | 43.53  | 8.521   |
| Aux Free                                                 | 512                                                      | 63.51     | 42.74 | 14.18 | 45.03  | 8.080   |
| Mo                                                       | MoE-3.4A0.6B (Train 400B Tokens, Global Batch Size 1024) |           |       |       |        |         |
| LBL                                                      | 4                                                        | 67.21     | 48.97 | 21.30 | 49.02  | 7.347   |
| LBL+sync                                                 | 128                                                      | 68.08     | 49.02 | 28.81 | 49.12  | 7.214   |
| LBL+sync                                                 | 512                                                      | 68.32     | 49.84 | 25.40 | 51.59  | 7.198   |
| LBL+buffer                                               | 128                                                      | 68.18     | 49.59 | 24.94 | 50.37  | 7.199   |
| MoE-15A2.54B (Train 400B Tokens, Global Batch Size 1024) |                                                          |           |       |       |        |         |
| LBL                                                      | 16                                                       | 75.69     | 59.99 | 48.07 | 64.38  | 5.778   |
| LBL+sync                                                 | 512                                                      | 76.96     | 60.78 | 54.28 | 64.31  | 5.603   |
| MoE-43A6.6B (Train 120B Tokens, Global Batch Size 512)   |                                                          |           |       |       |        |         |
| LBL                                                      | 8                                                        | 75.2      | 54.98 | 42.08 | 57.06  | 5.862   |
| LBL+buffer                                               | 128                                                      | 75.94     | 57.30 | 46.32 | 57.98  | 5.779   |

model parallelism strategies, we employ expert parallelism for both models, allowing a micro-batch size of 2 and 1 per GPU, respectively. We compared the results of synchronizing  $f_i$  within the same machine and across all data parallel groups, as shown in the last two parts of Tab. 1. It is evident that increasing the Balance BSZ also significantly improves larger models.

Global load balance encourages expert specialization. We further analyse the selection frequency of each layer's experts across different domains using the held-out PPL test data. Specifically, for a given domain, we record the topK experts chosen by each token and calculate the frequency of each expert selection. In Fig. 1, we compare the expert selection distributions under SFT-Code, SFT-Math, and EN-Literature for models trained with micro-batch balance and global-batch balance. It can be observed that (1) with micro-batch balance, most of the selection frequency is the same under EN-Literature, and only a few experts have slightly higher frequencies under SFT-Code and SFT-Math, yet none exceed 0.15. This aligns with existing analysis about MoE specialization: models using default load balance hardly exhibit domain-level specialization and only show some token-level specialization (Jiang et al., 2024; Xue et al., 2024). (2) In contrast, with global-batch balance, more pronounced high-frequency experts emerge, with many experts in SFT-Math having frequencies exceeding 0.2. This confirms our previous discussion that global-batch balance is more conducive to domain specialization.

#### Model performance increasing with Balance BSZ.

To further illustrate the impact of different Balance BSZes on the model performance, we controlled the micro-batch size, synchronization scope, and number of devices in training the 3.4A0.6B model on 400B tokens, and plotted the results from a Balance BSZ of 2 (micro-batch size 2, without any synchronization for expert selection frequency) to 512, as shown in Fig. 2. As the Balance BSZ increases, the test PPL consistently decreases, with an overall decrease of 0.185 from 2 to 512. It is also noticeable that the improvement rate slows down after increasing to 128, and the result of adding the buffer is very close to that of 512. This indicates that synchronization and buffer mechanisms can bring significant improvements compared to micro-batch in MoE training across various computing node scales. Additionally, we supplemented experiments by increasing the activation from top4 experts to top6

<span id="page-4-1"></span>![](_page_4_Figure_6.jpeg)

Figure 2: The performance of MoE-3.4A0.6B trained on 400B tokens with different Balance BSZ.

experts under the micro-batch condition and found that the improvement brought by a 50% increase in activated expert FLOPs is even less than the improvement from increasing the Balance BSZ from 2 to 8. Since the additional overhead brought by synchronization and buffer is much less than that brought by increasing FLOPs, this further demonstrates the efficiency of expanding the Balance BSZ.

## <span id="page-5-0"></span>5 Analysis

Ablation Study on Token Numbers and Token Distributional Bias — As aforementioned, the crucial factor for global-batch LBL to outperform micro-batch LBL is that the latter pushes the router to achieve sequence-level balanced expert utilization, which may be overly stringent and hurt the model performance. However, another naive assumption is that the LBL<sub>global</sub> involves more tokens to estimate the expert selection frequency, thus reducing the variance and ameliorating the MoE training. To verify, we introduce another setting: *Shuffle* LBL<sub>micro</sub>. Specifically, when calculating the LBL, we first synchronize the token-expert

<span id="page-5-1"></span>Table 2: Ablation of the number of tokens and distributional bias for computing LBL on MoE-3.4A0.6B.

| LBL type                   | Hellaswag                      | MMLU                           | Avg PPL                        |
|----------------------------|--------------------------------|--------------------------------|--------------------------------|
| 120B Toke                  | ns, Global Bat                 | ch Size 512,                   | Micro Batch Size 4             |
| Micro<br>Global<br>Shuffle | 62.81<br><b>63.75</b><br>63.57 | 41.63<br><b>43.48</b><br>43.37 | 8.167<br><b>8.038</b><br>8.041 |
| 400B Toker                 | ns, Global Bato                | ch Size 1024                   | , Micro Batch Size 2           |
| Micro<br>Global<br>Shuffle | 67.22<br>68.32<br><b>68.43</b> | 48.77<br><b>49.84</b><br>49.68 | 7.383<br><b>7.198</b><br>7.214 |

score matrix G (with a shape of number of tokens  $\times$  number of experts) in all parallel groups, where  $G_{ij}=1$  if the token i selects the expert j, otherwise  $G_{ij}=0$ . Then, we randomly select a batch of tokens (without replacing) to calculate the expert selection frequency, where the batch size is equal to the micro-batch size. In this setting, the random batch has the same token numbers as the micro-batch and identical token distribution as the global-batch, enabling us to tell the difference between these two confounders. The results are shown in the Tab. 2. We experiment with two settings: train 120B tokens with a micro-batch size of 4 and global-batch size of 512, and train 400B tokens with a micro-batch size of 2 and a global-batch size of 1024. We observe that the Shuffle LBL $_{micro}$  achieves similar performance as LBL $_{global}$ , and sill significantly outperforming the LBL $_{micro}$ , verifying the motivation of our paper and the assumption about the improvement.

<span id="page-5-2"></span>![](_page_5_Figure_5.jpeg)

Figure 3: The LBL (Top) and language modeling loss (Bottom) curve for MoE-3.4A0.6B trained on 400B tokens under different Balance BSZ, with a zoom-in of the last 15k steps shown below. In addition to the settings with Balance BSZ of 512 and 2, there are settings where the Balance BSZ changes from 2 to 512 at 10k, 30k, and 50k steps and a setting where it changes from 512 to 2 at the 50k step.

LBL<sub>global</sub> is a looser constraint than LBL<sub>micro</sub>. Intuitively, global-batch balance is a looser constraint than micro-batch balance: the former requires only that tokens be evenly distributed across experts globally, while the latter demands uniform distribution within each micro-batch. In Fig. 3 (Top), we show the loss curves of the two methods using the same load balance weight for MoE-3.4A0.6B trained on 400B. Additionally, we add the results of switching from micro-batch balance to global-batch balance at 10k, 30k, and 50k training steps. It can be observed that (1) after switching to global-batch balance, the LBL rapidly decreases to a range close to that when the global-batch balance is used from scratch, and the final convergence trend is also similar. This is because transitioning from a tighter constraint (balance within a micro-batch) to a looser one (balance within a global-batch) is relatively easy. (2)Moreover, if global batch balance is switched to micro-batch balance at the 50k step, the originally converged load

balance first rises to a much higher range, then slowly decreases, and the final convergence result is still lower than that of micro-batch balance used from scratch. This indicates that transitioning from a looser constraint to a tighter one can significantly alter the convergence state.

In Fig. 3 (Bottom), we present the language modeling loss curves in the same setting, and the test PPL statistics for these settings are compiled in Tab. 3. It can be observed that (1) the loss of global-batch balance is over 0.02 lower than that of micro-batch balance, corresponding to the large performance gap between the two as shown in Tab. 3. (2) Switching from micro-batch to global-batch balance results in performance improvements, with earlier switches yielding better outcomes. However, even the switch at the 10k step is notably inferior to training with global-batch balance from scratch. This aligns with existing findings that router choices

<span id="page-6-0"></span>Table 3: The impact of changing the Balance BSZ during training on the final results. Step indicates the step at which the Balance BSZ is switched.

| Balance BSZ         | Step (/100k) | PPL   |
|---------------------|--------------|-------|
| 2                   | -            | 7.383 |
| $2\rightarrow512$   | 50k          | 7.322 |
| $2\rightarrow512$   | 30k          | 7.297 |
| $2\rightarrow512$   | 10k          | 7.283 |
| 512                 | -            | 7.199 |
| $512 \rightarrow 2$ | 50k          | 7.373 |

tend to become fixed early in training (Xue et al., 2024; Muennighoff et al., 2024b): although increasing the Balance BSZ at any training stage can bring benefits, the router trained with micro-batch balance has already saturated very early, thus the gains from switching during training are limited. (3) Switching from global-batch to micro-batch balance degrades performance, indicating that changes in expert selection during training can greatly affect model performance.

Since micro-batch balance is a tighter constraint than global-batch balance, we further test reducing the load balance weight of micro-batch balance in Tab 4. It can be observed that appropriately reducing the LBL weight can slightly improve the model's performance within a certain range, but too

Table 4: Results for different load balance weight.

<span id="page-6-1"></span>

| Balance BSZ | LBL weight | Hellaswag | MMLU  | Avg PPL |
|-------------|------------|-----------|-------|---------|
| 4           | 0.008      | 62.81     | 41.63 | 8.167   |
| 4           | 0.004      | 62.95     | 42.13 | 8.154   |
| 4           | 0.001      | 62.97     | 41.71 | 8.159   |
| 512         | 0.008      | 63.75     | 43.48 | 8.038   |

small LBL weight leads to worse results. This may be due to the overly imbalanced distribution affecting the utilization of MoE model parameters. Moreover, the performance of micro-batch balance under various LBL weights is inferior to that of global-batch balance, further highlighting the differences between the two balancing methods.

The computation cost and efficiency of global-batch balance. Because a dropless strategy is employed, the FLOPs calculation is identical across different methods. However, due to differences in local balance conditions, methods using global-batch balance may experience local computational imbalance. To address this, we recorded the speed and results of micro-batch balance and global-

<span id="page-6-2"></span>Table 5: Performance and speed (seconds per iteration) in 43A6.6B setting. '128+buffer & 8' means adding microbatch balancing loss with Balance BSZ 8.

| Balance BSZ    | Hellaswag | MMLU  | Avg PPL | Speed/s |
|----------------|-----------|-------|---------|---------|
| 8              | 75.20     | 54.98 | 5.862   | 1.55    |
| 128+buffer     | 75.94     | 57.30 | 5.779   | 1.64    |
| 128+buffer & 8 | 75.87     | 57.00 | 5.795   | 1.59    |

batch balance during the training of the 43A6.6B model in Tab. 5. (1) It can be seen that the speed using global-batch balance (1.64 s/iteration) is 5.8% slower than micro-batch balance (1.55 s/iteration). Further analysis revealed that about 1% of this slowdown is due to communication overhead within all data parallel groups, the remainder mainly due to local expert load imbalance under the dropless strategy. Drawing inspiration from sequence-level LBL, we introduced a very low weight (1% of the global-batch weight) micro-batch balancing loss into the global-batch balance at the 20k step and continued training the model. We found that (2) adding a small amount of micro-batch balancing loss increased the speed to 1.59 s/iteration (2.6% slower than the baseline) with only a minimal decrease in performance. It should be noted that since the computation of LBL is independent from other parts of the network and takes very little time, it can be overlapped to further reduce the efficiency gap to within 2%.

Global batch balance brings interpretable specialization. In this section, we further analyze the specialization of models using global-batch balance. In Fig. 4 (a), we record the scores assigned to each expert by tokens across different domains and calculate the average of the topK score sums. When all experts are assigned identity scores, the topK sum is indicated by the gray dashed line. We can observe: (1) Models using global-batch balance have a higher topK sum at each layer. Since the LBL and z-loss in MoE encourage routing scores to be uniform, while only the language modelling loss encourages an increase in routing scores, this suggests that *under the global-batch balance, routing is more aligned with the language modelling task.* (2) Models using global-batch balance have a larger topK sum in domains where expert selection is more concentrated. For example, in Fig. 4 (b), the high-frequency experts in

<span id="page-7-0"></span>![](_page_7_Figure_0.jpeg)

Figure 4: The topK score sums across layers (a), and the distribution of high-frequency experts on different domains for models using global-batch balance (b). The topK sum of global-batch balance is higher than other methods and shows a similar distribution of high-frequency experts on closer domains.

ZH-Literature are more than those of SFT-EN, especially in layers 17 to 24. Correspondingly, in Figure 2(a), the topK sum of ZH-Literature in layers 17 to 24 is higher than that of SFT-EN. (3) Models using micro-batch balance have lower topK sums, with little difference across domains, which corresponds to the existing work that current MoE routing is uncertain [\(Wu et al.,](#page-10-6) [2024\)](#page-10-6). (4) Under global-batch balance, the topK sum of using aux loss free is smaller than that of LBL, but higher than micro-batch balance. *This further illustrates that expert specialization promotes the concentration of expert scores.*

In Fig. [4](#page-7-0) (b), we compare the distribution of high-frequency experts across domains. We observed that several Chinese domains (SFT-ZH, ZH-Law, ZH-Literature) have many similar high-frequency experts (indicated by the dashed box). Moreover, although both Chinese-related domains and SFT-Code have high-frequency activated experts, these experts hardly overlap. For domains with more general content (such as SFT-EN), there are fewer instances of individual experts being highly activated.

