# **Pre-Training and Downstream Evaluation Pre-Training Tasks**

To show the acceleration of training convergence with constant model quality, we first pre-train three MoE models with different MoE structures, namely GPT-Vanilla, GPT-Fine-Grained, and GPT-BigMac, all of which use GPT3-XL as the base model. Vanilla represents the conventional MoE with large experts, Fine-Grained refers to the MoE model with small experts, while BigMac is our design. For a fair comparison, we keep the same parameter size of MoE layers across the three models. We use the Wikipedia dataset (Wikimedia 2024) containing 3.6 B tokens to train these models on Megatron (NVIDIA 2019), one of the state-of-the-art LLM training frameworks.

Figure 1 shows the curve of validation perplexity of pretraining, indicating that GPT-BigMac converges much faster than others and achieves the lowest validation perplexity within the same time. For example, to achieve the same validation perplexity of 13.69, GPT-Fine-Grained requires 38.9 hours while GPT-BigMac only needs 22.8 hours, which is  $1.71 \times$  faster. In addition, among the three model structures, GPT-Vanilla fails to converge to the same validation perplexity under the time budget, indicating that with the same parameter size, the MoE structure with small experts outperforms the conventional MoE. Further, with the evaluation on WikiText2 (Merity et al. 2016), GPT-BigMac achieves the perplexity score of 17.4, while GPT-Vanilla and GPT-Fine-Grained get 27.4 and 17.9, respectively. The hyperparameters for pre-training are shown in Table 5 and the degree of Tensor Parallelism, Expert Parallelism, and Data Parallelism is set as 4, 4, and 2, respectively.

#### **Downstream Tasks**

To demonstrate how BigMac impacts the model quality on downstream tasks, we utilized a larger dataset named Open-WebText2 dataset (EleutherAI 2020) with 14.8 B tokens. First, we compare the performance after training for the same duration (8 days) based on the hyper-parameters in Table 5. We evaluate the fine-grained and BigMac variants, which are based on GPT3-XL, on eight popular zero-shot tasks, including four long-term dependence prediction tasks (LAMBADA (Paperno et al. 2016), PTB (Marcus, Santorini, and Marcinkiewicz 1993), WikiText103 and WikiText2 (Merity et al. 2016)) and four question answering

| MoE          | PTB    | WikiText   | WikiText2 | LAMBADA | HellaSwag | WinoGrande | PIQA   | RACE-H |
|--------------|--------|------------|-----------|---------|-----------|------------|--------|--------|
| Structure    | (PPL↓) | 103 (PPL↓) | (PPL↓)    | (ACC↑)  | (ACC↑)    | (ACC↑)     | (ACC↑) | (ACC↑) |
| Fine-Grained | 51.0   | 18.2       | 16.8      | 39.9    | 31.6      | 50.7       | 65.1   | 30.5   |
| BigMac       | 34.9   | 16.8       | 15.8      | 40.8    | 33.2      | 51.1       | 65.2   | 31.3   |

Table 6: Downstream results for different MoE models (based on GPT3-XL) after training with the same time.

| MoE          | PTB    | WikiText   | WikiText2 | LAMBADA | HellaSwag | WinoGrande | PIQA   | RACE-H |
|--------------|--------|------------|-----------|---------|-----------|------------|--------|--------|
| Structure    | (PPL↓) | 103 (PPL↓) | (PPL↓)    | (ACC↑)  | (ACC↑)    | (ACC↑)     | (ACC↑) | (ACC↑) |
| Vanilla      | 57.6   | 22.3       | 20.1      | 33.8    | 28.7      | 50.3       | 61.1   | 29.5   |
| Fine-Grained | 67.7   | 19.1       | 17.9      | 38.3    | 31.0      | 49.5       | 65.0   | 29.8   |
| BigMac       | 52.3   | 19.1       | 17.7      | 37.6    | 30.8      | 51.3       | 64.2   | 30.7   |

Table 7: Downstream results for different MoE models (based on GPT3-Medium) after training with the same number of tokens.

| Depth        | 10%   | 20%  | 30%  | 40%  | 50%  | 60%  | 70%  | 80%  | 90%  |
|--------------|-------|------|------|------|------|------|------|------|------|
| Fine-Grained | 99.1  | 99.3 | 99.0 | 98.6 | 98.4 | 98.3 | 98.3 | 98.2 | 97.9 |
| BigMac       | 100.0 | 99.9 | 99.4 | 99.0 | 98.8 | 98.6 | 98.5 | 98.3 | 98.1 |

Table 8: Recall scores of NeedleInAHaystack for different MoE models after training with the same number of tokens.

tasks (PIQA (Bisk et al. 2020), HellaSwag (Zellers et al. 2019) and WinoGrande (Sakaguchi et al. 2019), and RACE-H (Lai et al. 2017)). Table 6 shows the results of the eight downstream tasks in terms of accuracy (ACC) and perplexity (PPL). It shows that after training with the same time and GPU resources, GPT-BigMac gives a better model quality.

Next, we further compare the performance after training for the same number of steps and tokens (3 epochs for all models). For efficiency, we use GPT3-Medium as the base model and use the same hyper-parameters in Table 5, except that the values of Hidden Dimension, Learning Decay Steps, and Warmup Steps are 1,024, 28,000, and 5,000, respectively. Table 7 shows the results of the eight downstream tasks. GPT-BigMac delivers comparable or better results against GPT-Fine-Grained, achieving the best performance for 5 out of 8 tasks. For example, GPT-BigMac surpasses GPT-Fine-Grained by a score of 0.9 on RACE-H. Both GPT-BigMac and GPT-Fine-Grained outperform GPT-Vanilla, which shows the superiority of finegrained MoE models. In addition, we also evaluate two tasks, including GovReport (Huang et al. 2021) for summarization and NeedleInAHaystack (Kamradt 2023) for retrieval. GPT-BigMac achieves the score of 19.5 for GovReport, which is better than 17.7 achieved by GPT-DeepSeek. For NeedleInAHaystack, GPT-BigMac delivers comparable recall scores across different depths (Table 8).

## Training and Inference Speedups

In the last section, we have shown that compared with the traditional MoE structure, MoE structures with small experts are more powerful. In this section, we further compare the communication efficiency of the fine-grained MoE structure and BigMac in more depth.

## Experimental Setup

We intensively profile the time ratios of training and inference for GPT-Fine-Grained and GPT-BigMac, based on the state-of-the-art frameworks Megatron (Shoeybi et al. 2020), Tutel (Hwang et al. 2023), and DeepSpeed-Inference (Microsoft 2024). Megatron supports various parallelism strategies including data parallelism (DP), tensor parallelism (TP), and expert parallelism (EP). Tutel is a specialized framework to optimize the All-to-All communication for MoE models. DeepSpeed-Inference supports techniques specialized for LLM inference including KV cache management to efficiently serve the models. All the experiments are conducted on a cluster of 4 machines connected with 100 Gbps InfiniBand. Each machine has the same configuration and is equipped with eight GPUs. Each GPU is connected with PCIe 4.0 x 16 and has 48 GB HBM, delivering up to 149.7 TFLOPS (FP16) with 96 cores. For all the experiments, the input sequence length is 2,048 and the global batch size is 64. We mainly compare the two structures in terms of training step latency, the corresponding All-to-All latency, and the inference throughput.

## Comparing Training Latency via Megatron

We first compare the training step time of fine-grained and BigMac models under the Megatron framework. Here, we adopt four base models including GPT3-Medium, GPT3- XL, GPT3-2.7B, and GPT3-6.7B.

Figure 3 shows that GPT-BigMac achieves the speedups of 1.53-2.41× and 2.45-3.07× than GPT-Fine-Grained for Top4 and Top8 routing settings, respectively. Note that larger top k generally indicates the heavier communication, hence GPT-BigMac enjoys greater advantages in the Top8 setting. For the MoE models with small experts, larger top k implies better performance to some extent. Due to

![](_page_6_Figure_0.jpeg)

Figure 3: Per-iteration training time comparison between the fine-grained structure and BigMac on Megatron. The models are constructed from four base models, namely GPT3- Medium, GPT3-XL, GPT3-2.7B, and GPT3-6.7B, ordered by the size of parameters.

![](_page_6_Figure_2.jpeg)

Figure 4: Training time breakdown under different parallelism settings on Megatron. The labels *(*ep*,* tp*)* represent expert parallelism degree and tensor parallelism degree, respectively. For each group, the left bar is the result of GPT-Fine-Grained, and the right bar corresponds to GPT-BigMac. The numbers displayed on the right bar indicate the speedup in end-to-end latency.

the high communication efficiency, BigMac can choose a larger top k than GPT-Fine-Grained. Surprisingly, GPT-BigMac using the Top8 routing can still outperform GPT-Fine-Grained using the Top4 routing by 27.7-55.4% in terms of the end-to-end latency.

Breakdown Analysis. To understand the above speedups in depth, we report the breakdown results for training with an emphasis on the All-to-All communication cost. In Figure 4, the (32, 1) groups refer to the setup with only the expert parallelism and its degree ep setting to 32. In this setting, Big-Mac achieves an end-to-end speedup of 2.37× and 2.95× under the Top4 and Top8 routing, respectively, compared to the fine-grained baseline, where the speedup w.r.t. the Allto-All communication is 3.48× and 3.72×, respectively. In addition to the above pure expert parallelism setting, we also consider the combinations of various parallelism modes. We adopt tensor parallelism with the following settings. Specifically, we set the tensor parallelism degree tp from 1 to 8, and then adjust expert parallelism degree ep by ep = 32/tp. In this situation, BigMac can still reduce the All-to-All communication by 2.47-3.73× and the end-to-end latency by 1.55-2.77×. In Megatron, the TP-SP communication in the MoE layer involves the operations of All-to-All, All-Gather, and Reduce-Scatter within each TP group. All these oper-

![](_page_6_Figure_6.jpeg)

Figure 5: Inference throughput comparison between GPT-Fine-Grained and GPT-BigMac on Megatron. We conduct experiments with different numbers of GPUs with expert parallelism degree ep and top k values. The numbers under x-axis represents different prompt lengths.

ations happen at the higher dimension in the original finegrained structure and the lower dimension with the design of BigMac. In this way, BigMac also reduces the TP-SP communication by 1.42-2.34× for different setups. Finally, according to the results of the four parallelism settings shown in the figure, for the sake of efficiency, expert parallelism is preferred over tensor parallelism in our setting, as tensor parallelism involves more expensive all-reduce communication.

## Inference Throughput Comparison with Megatron

For inference, we measure the throughput of the forward pass under the Megatron framework. We keep the number of the tokens per batch to be 128k, but with varying prompt lengths, ranging from 128 to 1,024. We use 16 and 32 GPUs for evaluation and we set the expert parallelism degree ep to 16 and 32, respectively. Here we do not adopt tensor parallelism since it is less efficient.

Figure 5 shows that GPT-BigMac consistently outperforms GPT-Fine-Grained and achieves 1.72-2.45× speedups across all the settings. First, BigMac can obtain higher speedups with larger top k,. Second, the amplitude of speedup decreases slightly as the prompt length increases. Note that the larger prompt length brings heavier computation overhead in the attention layer, and then the proportion of All-to-All communication decreases correspondingly, especially for BigMac, which explains its slight decline in the inference throughput.

