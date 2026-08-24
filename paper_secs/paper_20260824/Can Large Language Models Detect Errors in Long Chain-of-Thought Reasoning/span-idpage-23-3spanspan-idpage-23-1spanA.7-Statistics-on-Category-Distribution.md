# <span id="page-23-3"></span><span id="page-23-1"></span>A.7 Statistics on Category Distribution

| Domain<br>Subcategory |                                       | Number |
|-----------------------|---------------------------------------|--------|
|                       | Discrete Mathematics                  | 144    |
|                       | Number Theory                         | 104    |
|                       | Geometry                              | 101    |
| Math                  | Others                                | 74     |
|                       | Calculus and Analysis                 | 58     |
|                       | Statistics and Other Decision Science | 45     |
|                       | Algebra                               | 36     |
|                       | Basic Programming                     | 133    |
|                       | Mathematics                           | 86     |
|                       | Advanced Programming                  | 48     |
| Programming           | Data Analysis                         | 41     |
|                       | Desktop and Web Development           | 27     |
|                       | Others                                | 24     |
|                       | Software Engineering                  | 14     |
|                       | Chemistry                             | 64     |
| PCB                   | Physics                               | 63     |
|                       | Biology                               | 27     |
|                       | Logical Reasoning                     | 56     |
|                       | Symbolic Reasoning                    | 28     |
|                       | Quantitative Reasoning                | 24     |
| General Reasoning     | Strategic Reasoning                   | 12     |
|                       | Common Sense Reasoning                | 9      |
|                       | Spatio-temporal Reasoning             | 9      |
|                       | Others                                | 9      |
|                       | Total                                 | 1236   |

Table 6: Detailed categories of DeltaBench and corresponding data volume statistics.

Table [6](#page-23-3) shows the subcategories and corresponding data volumes of DeltaBench across various domains. In obtaining queries and annotations, we strive to ensure balance across categories while also balancing annotation difficulty and accuracy.

### <span id="page-24-3"></span><span id="page-24-0"></span>A.8 Analysis of Other Evaluation Metrics

| Model                | F1-Score | First Error Acc. | Any Error Acc. |
|----------------------|----------|------------------|----------------|
| GPT-4-turbo-128k     | 40.76    | 57.04            | 69.17          |
| GPT-4o               | 30.85    | 36.89            | 50.89          |
| DeepSeek-V3          | 27.33    | 31.72            | 42.39          |
| Qwen2.5-32B-Instruct | 26.73    | 30.58            | 42.23          |
| DeepSeek-R1          | 28.43    | 29.94            | 40.78          |
| Qwen2.5-7B-Instruct  | 18.63    | 22.25            | 30.74          |
| GPT-3.5              | 7.98     | 6.15             | 11.65          |

Table 7: The table compares different accuracy metrics for each model. 'First Error Acc.' is the accuracy in identifying the first error, and 'Any Error Acc.' is the accuracy in detecting any error.

In Table [7,](#page-24-3) we present the performance of several models across different accuracy metrics: F1-Score, First Error Accuracy, and Any Error Accuracy. These metrics evaluate the models' ability to identify the first error and detect any error within a given sequence. A key observation is that the relative rankings of the models across the First Error Accuracy and Any Error Accuracy metrics closely align with their F1-Score. This consistency across different evaluation measures highlights the robustness of the F1-Score as a comprehensive indicator of model performance and suggests a strong correlation between the ability to detect the first error and the ability to identify any error in the sequence. Additionally, GPT-4-turbo consistently outperforms the other models, regardless of the evaluation metric used. Its Any Error Accuracy reaches 69%, significantly higher than the other models in the comparison. This finding underscores the model's superior performance in error recognition, yet it also points to the limitations that remain in current LLMs.

<span id="page-24-4"></span>

| Model                                     | Quantile | Threshold | prec  | recall | F1    |
|-------------------------------------------|----------|-----------|-------|--------|-------|
| Qwen/Qwen2.5-Math-PRM-7B                  | 5%       | 0.2168    | 39.81 | 73.86  | 46.48 |
| Qwen/Qwen2.5-Math-PRM-72B                 | 5%       | 0.2119    | 33.51 | 65.11  | 40.44 |
| RLHFlow/Llama3.1-8B-PRM-Deepseek-Data     | 5%       | 0.2021    | 24.19 | 56.1   | 30.88 |
| RLHFlow/Llama3.1-8B-PRM-Mistral-Data      | 5%       | 0.2949    | 23.18 | 51.46  | 29.68 |
| Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B | 5%       | 0.0303    | 19.48 | 46.76  | 24.45 |
| Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B   | 5%       | 0.0278    | 18.68 | 46.14  | 23.46 |

Table 8: Performance of PRMs using the overall reward quantile as the threshold.

For PRMs, aside from outlier detection, we also experimented with evaluating using a fixed threshold based on quantiles. Specifically, we used the ascending 5% quantile of all rewards on DeltaBench as the threshold, considering sections below this value as incorrect. The evaluation results are shown in table [8.](#page-24-4) However, compared to outlier detection, we found that this approach overestimates the performance of PRMs. This is because using a quantile as a threshold effectively forces PRMs to consider a fixed proportion of sections as incorrect.

