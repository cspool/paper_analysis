# E Training Configurations

We leverage the prompt from DeepSeek-AI et al. [\[4\]](#page-10-0), which is shown in Figure [8.](#page-16-5) And the mark for thinking is "<think>...</think>".

### <span id="page-16-0"></span>E.1 Training Prompt

<span id="page-16-5"></span>We list our training prompt in Figure [8,](#page-16-5) which follows the prompt from DeepSeek-R1 [\[4\]](#page-10-0).

```
<|User|>{input}Please reason step by step, and put your final answer within \boxed{}.<|Assistant|>
Prompt:
```

Figure 8: Training prompt for our training.

#### <span id="page-16-2"></span>E.2 Training and Evaluation Details

We employ the verl [\[21\]](#page-12-9) framework for model training and Qwen-Math-Eval [\[25\]](#page-12-10) for evaluation. During training, we set the rollout batch size to 128, conduct 8 rollouts per prompt, use a temperature of 0.6, and train with a mini-batch size of 64. In our preliminary experiments, we found long-to-short RL benefits from clip-higher strategy [\[27\]](#page-12-11). So we follow DAPO [\[27\]](#page-12-11) and set ϵhigh as 0.28. For evaluation, we maintain a sampling temperature of 0.6 and permit a maximum of 32,768 tokens to be generated. The number of samplings during evaluation is contingent on the dataset size: 4 samples per question for MATH500 and OlympiadBench, and 16 samples for AIME 2024 and AMC 2023.

### <span id="page-16-1"></span>E.3 Full Hyper-Parameter List for Different Length-based Rewards

<span id="page-16-3"></span>We list the all hyper-parameters for L<sup>T</sup> , α and L<sup>A</sup> in Table [5.](#page-16-3)

| Methods             | Hyper-Parameters                              |  |  |  |  |  |  |  |  |
|---------------------|-----------------------------------------------|--|--|--|--|--|--|--|--|
| Truncation          | LT<br>= [10240, 8192, 7168, 6144, 4098, 2048] |  |  |  |  |  |  |  |  |
| Think-Prune         | LA<br>= [4096, 3072, 2048]                    |  |  |  |  |  |  |  |  |
| Group-Based Rewards | α = [0.4, 0.2, 0.1, 0.05]                     |  |  |  |  |  |  |  |  |
| L1-Max              | α = 0.01                                      |  |  |  |  |  |  |  |  |
| LASER               | LT<br>= [8192, 4096, 2048]                    |  |  |  |  |  |  |  |  |
| LASER-D             | LT<br>= [4096, 2048, 1024]                    |  |  |  |  |  |  |  |  |
| LASER-DE            | LT<br>= [4096, 2048, 1024]                    |  |  |  |  |  |  |  |  |

Table 5: The details of key hyper-parameters for different methods

