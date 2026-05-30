# 4 Model Evaluation

#### 4.1 Evaluation of Foundation Models

Following OpenCompass[4](#page-5-2) , in the evaluation process, we use two types of evaluation methods: discriminant analysis evaluation and generative evaluation. Discriminant analysis evaluation means combining the question with candidate answers, calculating the perplexity of all combinations, and selecting the answer with the lowest perplexity as the model's final output. Generative evaluation uses the question as the model's original input and leaves the answer area blank for the model to complete subsequently.

The performance of AquilaDense-7B, AquilaDense-16B, and AquilaMoE(8\*16B) models are presented in Table [4.](#page-6-1) The indicators ending in "ppl" represent discriminant analysis evaluation, while those ending in "gen" represent generative evaluation.

Generally, as the model size increases, the scores tend to improve. For instance, AquilaDense-7B scores 7.81 on GSM8K-gen, while AquilaDense-16B scores 28.51. A similar trend also is observed in most other tasks. The AquilaMoE models show improved performance in most tasks over AquilaDense-16B. For example, in the ARC-c-ppl task, AquilaMoE scored 43.05 compared to 38.31 for AquilaDense-16B. These findings highlight the benefits of both scaling up model parameters and implementing MoE architectures in improving model performance.

#### 4.2 Evaluation of Fine-tuned Models

Table [5](#page-7-0) presents the overall results of AquilaMoE-8\*16B after fine-tuning across various benchmark datasets. The performance is measured using generative evaluation, and the results are expressed as percentages.

<span id="page-5-2"></span><sup>4</sup> https://github.com/open-compass

|                         | 1.8B    | 8*1.8B  | <b>7B</b> (AquilaDense-7B) | <b>16B</b> (AquilaDense-16B) | <b>8*16B</b> (AquilaMoE) |
|-------------------------|---------|---------|----------------------------|------------------------------|--------------------------|
| Context Length          | 2048    | 2048    | 4096                       | 4096                         | 4096                     |
| QKV Bias                | yes     | yes     | yes                        | yes                          | yes                      |
| Layers                  | 24      | 24      | 32                         | 40                           | 40                       |
| Hidden Dim              | 2048    | 2048    | 4096                       | 5120                         | 5120                     |
| <b>Intermediate Dim</b> | 5504    | 5504    | 14336                      | 20480                        | 20480                    |
| <b>Heads Num</b>        | 32      | 32      | 32                         | 40                           | 40                       |
| KV Group                | 32      | 32      | 32                         | 8                            | 8                        |
| Trained Tokens (B)      | 3600    | 400     | 3600                       | 1200                         | 545                      |
| LR                      | 1.20e-3 | 2.20e-4 | 1.20e-3                    | 4.00e-4                      | 1.50e-4                  |
| <b>Batch Size</b>       | 12M     | 12M     | 12M                        | 12M                          | 24M                      |

<span id="page-6-0"></span>Table 3: Model configurations and training parameters for different models.

Table 4: Overall evaluation results of AquilaDense and AquilaMoE(AquilaMoE-8\*16B)

<span id="page-6-1"></span>

| Model         | AquilaDense-7B | AquilaDense-16B | AquilaMoE |
|---------------|----------------|-----------------|-----------|
| ARC-c-ppl     | 37.63          | 38.31           | 43.05     |
| ARC-e-ppl     | 56.08          | 52.2            | 65.61     |
| Hellaswag-ppl | 67.49          | 71.62           | 73.94     |
| GSM8K-gen     | 7.81           | 28.51           | 54.51     |
| HumanEval-gen | 14.02          | 29.88           | 15.85     |
| MMLU-ppl      | 46.47          | 57.11           | 61        |
| Winograd-ppl  | 50.53          | 54.04           | 55.4      |
| MATH-gen      | 1.32           | 4.24            | 10.4      |
| MBPP-gen      | 15.6           | 36.4            | 37.2      |
| DROP-gen      | 4.35           | 33.35           | 37.62     |
| AGI Eval-gen  | 14.47          | 18.57           | 13.69     |
| BBH-gen       | 34.51          | 41.45           | 46.04     |
| NQ-gen        | 8.61           | 9.94            | 10.78     |
| PIQA-ppl      | 76.71          | 79.22           | 80.3      |

#### 4.3 Comparsion of Computational Efficiency

We present the details of the training process for both scale-up + scale-out and from-scratch approaches in Table 6. The table lists the number of devices in the cluster, the GFLOPS per device, the model parameters size, the number of training tokens, the actual running tokens per day, the actual training time, and the actual training GFLOPS for each phase.

The time savings factor is calculated by comparing the total training time of the from-scratch approach to the total training time of the scale-up and scale-out approach. The formula is:

$$\text{Time Savings Factor} = \frac{\frac{\sum_{i=1}^{n} N_{\text{tokens},i}}{R_{\text{tokens/day, from scratch}}}}{\sum_{i=1}^{n} \frac{N_{\text{tokens},i}}{R_{\text{tokens/day},i}}}$$

Given the data:

Time Savings Factor = 
$$\frac{\frac{3600+1200+545}{25}}{\frac{3600}{279}+\frac{1200}{70}+\frac{545}{25}} = \frac{213.80}{51.84} \approx 4.12$$

The computational power savings factor is calculated by comparing the total GFLOPS-days of the from-scratch approach to the total GFLOPS-days of the scale-up and scale-out approach. The formula is:

$$\text{Computational Power Savings Factor} = \frac{\frac{\sum_{i=1}^{n} N_{\text{tokens,}i} \times \text{GFLOPS}_{\text{from scratch}}}{R_{\text{tokens,}i} \times \text{GFLOPS}_{\text{from scratch}}}}{\sum_{i=1}^{n} \frac{N_{\text{tokens,}i} \times \text{GFLOPS}_{i}}{R_{\text{tokens,}i \times \text{MISM}}}}$$

<span id="page-7-0"></span>

| Model      | AquilaMoE-8*16B-SFT |
|------------|---------------------|
| ARC-c      | 82.03               |
| ARC-e      | 87.3                |
| Hellaswag  | 75.08               |
| GSM8K      | 71.27               |
| NQ         | 21.39               |
| TriviaQA   | 65.33               |
| AGI Eval   | 13.61               |
| Math       | 13.26               |
| HumanEval  | 44.51               |
| PIQA       | 81.72               |
| OBQA       | 75.2                |
| DROP       | 62.32               |
| BoolQ      | 85.02               |
| GPQA       | 25.76               |
| C-Eval     | 57.99               |
| MMLU       | 61.51               |
| CMMLU      | 57.63               |
| Winogrande | 57.54               |
|            |                     |

Table 5: Overall results of AquilaMoE after fine-tuning.

<span id="page-7-1"></span>Table 6: Training details for scale-up and scale-out and from-scratch approaches, note that for preparation phase different chip is used.

| Approach/Phase    | Devices | GFLOPS/Device | Model Size (B) | Trained Tokens (B) | Training Tokens/Day (B) |
|-------------------|---------|---------------|----------------|--------------------|-------------------------|
| Preparation Phase | 480     | 989.5         | 7              | 3600               | 279                     |
| Scale-Up Phase    | 1024    | 240           | 16             | 1200               | 70                      |
| Scale-Out Phase   | 1024    | 240           | 32             | 545                | 25                      |
| From Scratch      | 1024    | 240           | 32             | 5345               | 25                      |

Given the data:

$$\begin{aligned} & \text{GFLOPS}_{\text{preparation}} = 480 \times 989.5 = 475, 360 \\ & \text{GFLOPS}_{\text{scale-up}} = 1024 \times 240 = 245, 760 \\ & \text{GFLOPS}_{\text{scale-out}} = 1024 \times 240 = 245, 760 \\ & \text{GFLOPS}_{\text{from scratch}} = 1024 \times 240 = 245, 760 \end{aligned}$$

The computational power savings factor is:

Computational Power Savings Factor = 
$$\frac{\frac{5345 \times 245,760}{25}}{\frac{3600 \times 475,360}{279} + \frac{1200 \times 245,760}{70} + \frac{545 \times 245,760}{25}} = \frac{52,592,640}{15,705,343} \approx 3.35$$

The method proposed in this paper significantly reduces both the computational power and the time required for training. By employing a scale-up and scale-out approach, we achieved a computational power savings factor of approximately 3.35 and a time savings factor of approximately 4.12.

Additionally, if we start with a pre-trained smaller model, the computational power and time required for the preparation phase can be further reduced. This approach not only accelerates the training process but also lowers the overall computational costs.

In summary, the proposed training methodology offers substantial improvements in efficiency. The combined scale-up and scale-out approach, along with the potential use of pre-trained models, represents a significant advancement in the optimization of training large-scale models.

## 5 Conclusion and Future Work

We present AquilaMoE, a bilingual 8\*16B mixture of experts (MoE) language model developed using the EfficientScale training method. EfficientScale optimizes performance while significantly reducing data requirements through a two-stage approach: Scale-Up and Scale-Out. Our contributions are as follows: 1) An effective training methodology that achieves knowledge transfer and continuous pretraining with significantly reduced data and computational needs; 2) Innovative initialization strategies, such as Functional Progressive Initialization (FPI) and Approximate Knowledge Integration (AKI), which demonstrate substantial loss retention and reduction during continual pre-training; 3) Successful training of 16B and 8\*16B AquilaMoE models using these initialization strategies, enhancing performance and training efficiency. Future work involves exploring the scalability of larger MoE models, investigating cross-linguistic knowledge transfer, developing new optimization techniques to further reduce training time and costs, fine-tuning for specific application domains, and ensuring the robustness and generalization of MoE models across diverse datasets and real-world applications.

## Authorship

Language Foundation Model & Software Team, BAAI: Bo-Wen Zhang, Liangdong Wang, Jijie Li, Shuhao Gu, Mengdi Zhao, Xinya Wu, Guang Liu (Project lead) [5](#page-8-8)

Data Research Team, BAAI: Chengwei Wu, Hanyu Zhao, Li Du, Yiming Ju, Quanyue Ma

AI Framework Research and Development Team, BAAI: Yulong Ao (Infrastructure lead), Yingli Zhao, Songhe Zhu, Zhou Cao, Dong Liang, Yonghua Lin

School of Computer Science, Peking University: Ye Yuan[6](#page-8-9) , Ming Zhang

MetaX-Tech: Shunfei Wang, Yanxin Zhou, Min Ye, Xuekai Chen, Xinyang Yu, Xiangjun Huang, Jian Yang

