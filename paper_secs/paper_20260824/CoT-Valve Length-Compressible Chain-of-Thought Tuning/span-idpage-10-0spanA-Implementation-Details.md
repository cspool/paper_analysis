# <span id="page-10-0"></span>A Implementation Details

## A.1 Evaluation Metric.

For experiments on LLaMA, we use lm-evalharness[1](#page-10-1) to evaluate the model performance. For LLaMA-3.1-8B, we report the strict matching metric due to observed repetition in the model's responses, which causes the flexible match to extract incorrect numerical values. For LLaMA-3.2- 1B-Instruct, we report results using the flexible match metric. For QwQ-32B-Preview, DeepSeek-R1-Distill-Llama-8B and Qwen-2.5B-LIMO, we first extract the result enclosed within \boxed{}. If no such boxed answer is found, we default to using the last digit in the response as the final answer.

## A.2 Training Setting.

LLaMA-3.1-8B The model is trained using eight A5000 24GB GPUs. We set the batch size to 64 and the peak learning rate to 4e-5, following a cosine decay schedule. A weight decay of 0.01 is applied. For the progressive chain compression experiment, we train the model for two epochs with each type of solution. For all other experiments, we train for a maximum of eight epochs. For LoRA, the rank is set to 32, and the lora\_alpha for training is set to 64. During inference, the maximum number of tokens is set to 2048.

LLaMA-3.2-1B-Instruct The model is trained using 8 A5000 24GB GPUs. We set the batch size to 8 for the CoT-Valve experiment and 64 for all other experiments. The peak learning rate is 4e-5, following a cosine decay schedule, except for the SFT - GSM8K experiment, where the peak learning rate is 1e-5. A weight decay of 0.01 is applied. For the CoT-Valve and SFT-Full Finetune - GSM8k experiment, we train for a maximum of four and six epochs, respectively. For the progressive chain compression experiment, we train the model for two epochs with each type of solution. For all other experiments, training is conducted for up to 8 epochs. For LoRA, the rank is set to 32, and the lora\_alpha for training is set to 64. During inference, the maximum number of tokens is set to 2048.

QwQ-32B-Preview. The model is trained on two H100-80G GPUs. We set the batch size to 64 and trained for a maximum of five epochs. The learning rate is 1e-5, with a weight decay of 0.01 applied

during training. For LoRA, the rank is set to 2, and the lora\_alpha for training is set to 8. During inference, we set the maximum token to be 4192 for GSM8K and the maximum token as 8192 for AIME correspondingly.

DeepSeek-R1-Distill-Llama-8B. Our experiment on DeepSeek-R1-Distill-Llama-8B[2](#page-10-2) is conducted using the MixChain-zero-shot-GSM8K dataset. The batch size is set to 128, and training is performed for a maximum of five epochs. To ensure that the inference process successfully generates the final answer, we set the maximum token limit to 30K.

Qwen2.5-32B-LIMO. We fine-tuned Qwen-32B-Instruct using LIMO, training on four H100 GPUs for 10 epochs with a batch size of 4 and a maximum sequence length of 16K. The learning rate was set to 5e-6. We define Qwen-32B-Instruct as θ<sup>0</sup> and the trained model as θ1, treating the update direction between them as ∆θ. By adjusting α, we generated the MixChain-C-LIMO dataset, which includes two solutions: solution 1 (α=0.8) and solution 0 (α=0.6).

Based on this, we further trained θ<sup>2</sup> for 5 epochs with a batch size of 32, a learning rate of 5e-6, and a weight decay of 0.01, obtaining the results of MixChain-Solution 0 in Table [2.](#page-6-0) This model can be further refined through CoT-Valve (Results: CoT-Valve + MixChain - Solution 0). Unlike previous experiments, we applied full fine-tuning instead of LoRA. The maximum generated sequence length in this experiment was 15K.

## A.3 Dataset Explanation

As detailed in Section [4.2,](#page-5-2) we constructed two types of datasets: MixChain-C and MixChain-Z. The statistics for the datasets are shown in [9.](#page-11-0) For these datasets, we select α values ranging from [0.6, 0.8] for LIMO and [0.2, 0.4, 0.6, 0.8] for other datasets, ensuring all incorrect responses are excluded.

For MixChain-Z, while the training transition from θ<sup>1</sup> to θ<sup>2</sup> remains a black box, we can still identify numerous model pairs such as Qwen-32B-Instruct → QwQ-32B-Preview, and LLaMA-3.1- 8B → R1-Distill-Llama-8B, as documented in the technical report. We find that the performance of the base model significantly influences the quality of the dataset.

<span id="page-10-1"></span><sup>1</sup> https://github.com/EleutherAI/lm-evaluation-harness

<span id="page-10-2"></span>https://huggingface.co/deepseek-ai/DeepSeek-R1- Distill-Llama-8B

<span id="page-11-0"></span>

| Dataset       | Solution Index   | #Samples | #Avg Token |  |  |
|---------------|------------------|----------|------------|--|--|
|               | GSM8K            |          |            |  |  |
| Ground-Truth  | 1                | 7473     | 121.8      |  |  |
| MixChain-C    | 1                | 22419    | 294.8      |  |  |
|               | 0 (Ground-Truth) |          | 116.0      |  |  |
|               | 1                |          | 279.6      |  |  |
| MixChain-Z    | 2                | 6863     | 310.7      |  |  |
|               | 3                |          | 386.7      |  |  |
|               | 4                |          | 497.2      |  |  |
|               | PRM12K           |          |            |  |  |
| Ground-Truth  | 1                | 12000    | 223.1      |  |  |
|               | 0 (Ground-Truth) |          | 172.3      |  |  |
|               | 1                |          | 583.2      |  |  |
| MixChain-Z    | 2                | 8841     | 613.7      |  |  |
|               | 3                |          | 739.3      |  |  |
|               | 4                |          | 1003.2     |  |  |
| LIMO          |                  |          |            |  |  |
| Ground-Truth  | 1                | 817      | 6984.1     |  |  |
| MixChain-C    | 1                | 474      | 2994.7     |  |  |
| wiixCiiaiii-C | 2                | 564      | 4890.6     |  |  |

Table 9: Dataset Statistic. Here we use the tokenizer from QwQ-32B-Preview to count the number of tokens.

<span id="page-11-1"></span>

| α                    | 0     | 0.125 | 0.25  | 0.5   | 0.75  | 1.0   |
|----------------------|-------|-------|-------|-------|-------|-------|
| # Tokens<br>Accuracy | 199.8 | 219.4 | 233.4 | 257.7 | 466.3 | 772.7 |
| Accuracy             | 45.9  | 47.5  | 50.2  | 57.1  | 55.0  | 54.5  |

Table 10: Results of LLaMA-3.2-1B-Instruct trained with DoRA using different  $\alpha$  values for interpolation.

## B More Analysis

**Experiments on DoRA.** In addition to LoRA, we also train LLaMA-3.2-1B using DoRA (Liu et al., 2024a) and control the magnitude of  $\Delta\theta$  by adjusting the  $\alpha$  for DoRA. The model is trained on QwQ synthesized data for a maximum of five epochs. We set the batch size to 8 and the peak learning rate to 4e-5, following a cosine decay schedule. A weight decay of 0.01 is applied. For DoRA, the rank is set to 32, and the lora\_alpha for training is set to 64.

As shown in Table 10, the chain length increases with the  $\alpha$  value, demonstrating the effectiveness of interpolating  $\Delta\theta$  for DoRA. Furthermore, similar to our observations with LoRA, the best result is not obtained by directly training the model on long CoT data. Specifically, training on QwQ synthesized data ( $\alpha=1.0$ ) achieves an accuracy of 54.5 with 772.7 tokens, whereas the best model obtained via CoT-Valve ( $\alpha$ =0.5) achieves an accuracy of 55.72 with only 257.7 tokens.

Attention has less effect on the length of the reasoning path than MLP. We experimented

<span id="page-11-2"></span>

| Modules    | GSM8K | #Tokens | #Params | ACU↑ |
|------------|-------|---------|---------|------|
| -          | 95.1  | 741.1   | -       | 0.40 |
| K+V        | 95.0  | 687.7   | 0.005%  | 0.43 |
| Q          | 95.2  | 621.4   | 0.004%  | 0.48 |
| O          | 95.2  | 484.2   | 0.004%  | 0.61 |
| Attention  | 94.2  | 284.2   | 0.013%  | 1.04 |
| MLP        | 93.5  | 221.8   | 0.038%  | 1.32 |
| All Linear | 92.4  | 227.6   | 0.051%  | 1.27 |

Table 11: LoRA on Different Modules.

with applying LoRA fine-tuning to only specific model components in QwQ. Table 11 shows that fine-tuning only the query, key, or value projection had less effect than other linear projections on shortening the reasoning chain. This suggests that attention computation may have less impact on controlling chain length, whereas the MLP layers and the final projection in attention play a more significant role. We find this to be an intriguing phenomenon, and further investigation into its underlying causes could provide deeper insights into optimizing reasoning chain compression.

**Prompt Control.** We evaluate the length of CoT under constraint in prompts. Table 12 presents the number of tokens generated when using various prompts across two models. For these two models, the prompts used here are:

Prompt Template for QwQ-32B-Preview

**System**: You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.

**User:** Generate the solution in less than <token\_count> tokens. <Question>

Prompt Template for LLaMA-3.2-1B-Instruct

System:

Cutting Knowledge Date: December 2023

Today Date: 01 Jan 2025

User

Given the following problem, reason and give a final answer to the problem using less than <token\_count> tokens.

<Question>

From the results, we observe that while these prompts provide control over the length, there remains a significant discrepancy between the generated token count and the intended target.

<span id="page-12-1"></span>Generated Answer for the Same Question

**Question**: Amanda had 10 notebooks. This week, she ordered 6 more and then lost 2. How many notebooks does Amanda have now?

#### **Ground Truth Solution - 49 Tokens:**

This week, Amanda added 6 notebooks - 2 notebooks = <6-2=4>4 notebooks. Now, Amanda has 10 notebooks + 4 notebooks = <10+4=14>14 notebooks.

Extrapolation - 300 Tokens: So Amanda started with 10 notebooks. Then, she ordered 6 more. So, I think I need to add those 6 to the original 10. That should be 10 plus 6, which is 16. Okay, now she lost 2 notebooks. So, I need to subtract those 2 from the 16. That should be 16 minus 2, which equals 14. So, Amanda has 14 notebooks now. Wait a minute, let me double-check that. She had 10, ordered 6 more, so that's 10 + 6 = 16. Then lost 2, so 16 - 2 = 14. Yeah, that seems right. Alternatively, maybe I can think of it differently. Maybe consider the net gain of notebooks. She ordered 6 and lost 2, so net gain is 6 - 2 = 4. Then, add that net gain to the original 10: 10 + 4 = 14. Same answer. Or, maybe another way: starting with 10, losing 2 would bring her to 8, and then ordering 6 more would bring her to 14. Again, the same result. I think no matter how I look at it, the answer is 14 notebooks. \*\*Final Answer\*\* \boxed{14}

**QwQ-32B-Preview - 257 tokens**: So Amanda started with 10 notebooks. Then, she ordered 6 more. So, I need to add those 6 to the original 10. That should be 10 plus 6, which is 16. Okay, now she lost 2 notebooks. So, I need to subtract those 2 from the 16 she had after ordering more. That means 16 minus 2, which equals 14. So, Amanda has 14 notebooks now. **Wait a minute, let me double-check that**. She had 10, ordered 6 more, so that's 10 + 6 = 16. Then lost 2, so  $16 - 2 = \overline{14}$ . Yeah, that seems right. **Alternatively, maybe I can think of it differently**. Maybe combine the ordered and lost notebooks first. She ordered 6 and lost 2, so overall, she gained 6 - 2 = 4 notebooks. Then, add that to the original 10: 10 + 4 = 14. Same answer. I think that's correct. Amanda has 14 notebooks now. \*\*Final Answer\*\* \boxed{14}

**Interpolation - 207 Tokens**: Let's break this down step by step. Amanda started with 10 notebooks. Then, she ordered 6 more. So, I need to add those 6 to the original 10. That would be 10 plus 6, which equals 16 notebooks. Okay, so now she has 16 notebooks. But then, she lost 2. So, I need to subtract those 2 from the 16. That would be 16 minus 2, which equals 14 notebooks. Therefore, Amanda now has 14 notebooks. **Wait a minute, let me double-check that.** She started with 10, ordered 6 more, so that's 10 + 6 = 16. Then lost 2, so 16 - 2 = 14. Yeah, that seems right. I think that's the answer: Amanda has 14 notebooks now.

**Interpolation - 133 Tokens**: Let's solve this step-by-step: 1. \*\*Initial number of notebooks\*\*: Amanda started with 10 notebooks. 2. \*\*Notebooks ordered\*\*: This week, she ordered 6 more notebooks. - So, we add these 6 notebooks to her initial 10 notebooks. - 10 + 6 = 16 notebooks. 3. \*\*Notebooks lost\*\*: Then, Amanda lost 2 notebooks. - We need to subtract these 2 lost notebooks from the total she had after ordering more. - 16 - 2 = 14 notebooks. Therefore, Amanda now has 14 notebooks.

**Extrapolation - 103 Tokens**: To find out how many notebooks Amanda has now, we can break the problem into steps: 1. Amanda initially had 10 notebooks. 2. This week, she ordered 6 more notebooks, so we add 6 to the initial amount: 10 + 6 = 16 notebooks. 3. Amanda then lost 2 notebooks, so we subtract 2 from the total: 16 - 2 = 14 notebooks. So, Amanda has 14 notebooks now.

Figure 4: An example of the generated solution for the same question.

## **C** Examples

Here we show in Fig.4 an example of the generated CoT from short to long, and we also show two extrapolation cases to show the generalization ability of our method. Our method notably generates a longer reasoning process compared to the original QwQ model, incorporating an extra reflection phase. During the chain shortening process, it reduces multiple rounds of reasoning and streamlines the language, ultimately enabling us to produce an answer with only 103 tokens through extrapolation.

<span id="page-12-0"></span>

| QwQ-32          | B-Preview        | Llama-3.2-1B Instruct |                  |  |
|-----------------|------------------|-----------------------|------------------|--|
| Token in Prompt | #Token Generated | Token in Prompt       | #Token Generated |  |
| 20              | 355              | 50                    | 118              |  |
| 50              | 422              | 100                   | 132              |  |
| 100             | 511              | 200                   | 141              |  |
| 200             | 569              | 300                   | 160              |  |
| 300             | 623              | 400                   | 183              |  |
| 400             | 666              | 500                   | 186              |  |

Table 12: Significant discrepancies exist between the conditions specified in the prompt and the number of generated tokens on GSM8k.