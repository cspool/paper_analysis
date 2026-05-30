# <span id="page-12-1"></span>A.4 Dynamic Skipping for Domain-specific Tasks

We also perform dynamic expert skipping on domain-specific tasks (mathematical reasoning tasks). We calibrate  $\beta$  for each layer using samples from the training set of MATH and evaluate 5-shot accuracy on the GSM8K dataset. We also test and report the token generation speed of each MoE model. The results are shown in Tab. 8. In this case, dynamic expert skipping leads to more performance drops. But for the Mixtral 8x7B Instruct model, expert pruning with 2 experts and combining dynamic skipping also leads to the same inference speedup with pruning 4 experts, while achieving higher evaluation accuracy.

### A.5 Actual Memory Reduction

A more detailed statistical comparison between our expert pruning method with baseline methods on

<span id="page-12-3"></span>

| Model        | r | Pruning      | Skipping     | GSM8K (5-shot) | Speedup       |
|--------------|---|--------------|--------------|----------------|---------------|
|              | 8 |              |              | 58.61          | 1.00×         |
|              | 8 |              | <b>√</b>     | 54.28          | 1.08×         |
| Mixtral 8x7B | 6 | $\checkmark$ |              | 51.25          | $1.20 \times$ |
|              | 6 | $\checkmark$ | $\checkmark$ | 47.16          | $1.21 \times$ |
|              | 4 | $\checkmark$ |              | 37.07          | $1.29 \times$ |
|              | 4 | $\checkmark$ | ✓            | 34.80          | $1.30 \times$ |
|              | 8 |              |              | 63.46          | 1.00×         |
| Mixtral 8x7B | 8 |              | <b>√</b>     | 61.94          | 1.05×         |
| Instruct     | 6 | $\checkmark$ |              | 58.38          | $1.20 \times$ |
| 111501 400   | 6 | $\checkmark$ | $\checkmark$ | 53.98          | $1.28 \times$ |
|              | 4 | $\checkmark$ |              | 47.01          | $1.28 \times$ |
|              | 4 | $\checkmark$ | ✓            | 40.33          | $1.33 \times$ |

Table 8: Evaluation results of combining expert pruning with dynamic skipping for domain-specific tasks. Combining two expert-level sparsification methods will lead to more efficient deployment.

<span id="page-12-4"></span>the Mixtral 8x7B model is shown in Tab. 9.

| Method | Sparsity       | Memory (MB)   |
|--------|----------------|---------------|
| None   | None $(r = 8)$ | 89,926 (100%) |
| Wanda  | 2:4            | 51,214 (57%)  |
| Ours   | r = 6          | 68,383 (76%)  |
| Ours   | r=4            | 46,879 (52%)  |

Table 9: Memory reduction comparison of our expert pruning method with baselines on Mixtral 8x7B.

## A.6 Relationships with Other Network Pruning and Parameter Quantization Methods

As plug-and-play techniques, both our proposed expert pruning and dynamic skipping methods are orthogonal to other model light-weighting schemes (e.g., weight pruning (Frantar and Alistarh, 2023; Sun et al., 2023), token pruning (Kim et al., 2022; Ding et al., 2023)) and are compatible with weight quantization approaches (Frantar et al., 2022; Lin et al., 2023).

#### A.7 More Experiment Details

In this part, we give more experimental details for a better understanding of our proposed methods.

Calibration Set Construction for Expert Pruning. For task-agnostic models, we use the samples from C4 (Raffel et al., 2019) as the calibration dataset. Following the setting of Wanda, we sample from the first part of the training data<sup>6</sup>. For task-specific (mathematics) models, we use samples from the training set of MATH (Hendrycks et al., 2021). The structure of the MATH dataset is different from C4, so we reconstruct the dataset in the format of C4 and randomly sample from it.

<span id="page-12-5"></span><sup>&</sup>lt;sup>6</sup>https://huggingface.co/datasets/allenai/c4/blob/main/en/c4-train.00000-of-01024.json.gz

Calibration Set Construction for Dynamic (Expert) Skipping. To calculate  $\beta$  for dynamic expert skipping in each MoE layer, we forward the MoE model over the calibration dataset and set  $\beta$  as the median value of  $\frac{w_{e_1}}{w_{e_0}}$  separately for each layer. We choose to use the median value over the calibration dataset as in this case, the skipping will happen with around 50% possibility. Here we provide the value of  $\beta$  for the Mixtral 8x7B model with calibration data sampled from C4 and MATH respectively. As can be seen, the parameter in each layer differs significantly.

C4: 0.402,0.494,0.463,0.484,0.478,0.491,0.523, 0.521,0.544,0.570,0.574,0.489,0.503,0.618,0.568, 0.535,0.559,0.519,0.537,0.487,0.469,0.461,0.461, 0.469,0.458,0.418,0.433,0.418,0.406,0.433,0.447, 0.535

MATH: 0.503,0.586,0.505,0.531,0.509,0.422, 0.511,0.461,0.447,0.478,0.529,0.454,0.472,0.531, 0.499,0.486,0.503,0.491,0.430,0.440,0.402,0.423, 0.386,0.407,0.395,0.354,0.340,0.351,0.334,0.368, 0.365,0.346

Model Fine-tuning. In the part of task-specific expert pruning for domain-specific tasks, we fine-tune the Mixtral 8x7B and Mixtral 8x7B Instruct models with 8 experts, 7 experts, and 6 experts on the MetaMathQA (Yu et al., 2023) dataset. The training is conducted on 16 A100-80G GPUs. We train the model for 900 steps, using a learning rate of 2e-5 with the cosine learning rate scheduler.