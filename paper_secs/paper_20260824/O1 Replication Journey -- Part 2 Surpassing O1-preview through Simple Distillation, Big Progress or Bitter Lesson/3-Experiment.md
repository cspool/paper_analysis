# 3 Experiment

## 3.1 Benchmark Usage

We select several widely recognized and commonly used benchmarks in the field of mathematical reasoning, chosen for their challenging nature. These include MATH [\(Hendrycks et al.,](#page-14-10) [2021\)](#page-14-10) and AIME. Specifically, we use the streamlined MATH500 subset to facilitate more extensive inference-time scaling experiments. For AIME, we utilize the newly released problems in 2024 to minimize the risk of data leakage (we refer to it as AIME2024). Additionally, we curate a set of 30 problems from the 2024 China National High School Mathematics Competition, serving as an additional benchmark (MATH2024) to diversify and enrich our evaluation. This combination of benchmarks ensures a comprehensive assessment of our model's mathematical reasoning capabilities.

#### 3.2 Evaluation Metric for Inference-Time Scaling

Unlike conventional evaluation strategies that rely solely on metrics such as Pass@k [\(Chen et al.,](#page-14-11) [2021b\)](#page-14-11), Maj@k [\(Wang et al.,](#page-15-13) [2022\)](#page-15-13), or RM@k [\(Lightman et al.,](#page-15-14) [2024\)](#page-15-14), we introduce a novel metric designed to evaluate model performance across varying computational cost scenarios. This new approach reflects the realities of inference-time scaling [\(Snell et al.,](#page-15-15) [2024\)](#page-15-15), where test-time compute plays a crucial role in determining the effectiveness and efficiency of modern large-scale models. In the era of inference-time scaling, models like OpenAI's O1-series have demonstrated that performance is not solely dependent on training-time compute but also significantly influenced by the time spent "thinking" during inference. This shift necessitates a more nuanced evaluation framework that accounts for the trade-off between computational cost and performance. Our proposed

metric directly addresses this by measuring the model's reasoning ability under constrained test-token budgets, ensuring that evaluations reflect real-world constraints and deployment scenarios.

Specifically, we measure the computational cost of a model on a given benchmark test set using the average token count for its outputs. This metric reflects the test-time computational expense, where longer average token outputs correspond to more extensive reasoning steps. Models capable of generating longer, more detailed outputs are often able to capture complex reasoning patterns more effectively, demonstrating their scalability under inference-time compute. Furthermore, this average token metric is inherently extensible. In scenarios where the evaluation requires a higher average token count than what is typically generated in a single response, we leverage the Maj@k metric to approximate the model's performance without using any extra reward model. This approach reflects the model's reasoning ability at extended computational costs, even when a single output does not naturally reach the desired token length.

By employing this method, we ensure a scalable and fair evaluation framework that captures model performance across different inference-time compute settings. This approach avoids artificial constraints and allows for meaningful comparisons without relying on external reward signals, focusing solely on the model's intrinsic reasoning capabilities.

### 3.3 Performance Analysis

<span id="page-4-0"></span>Comparison with O1's performance As is shown in Table [1,](#page-4-0) under similar "reasoning computational costs" (i.e., with comparable average output tokens on the corresponding benchmark), the distilled model demonstrates outstanding performance, surpassing the results of O1-preview on AIME2024.

| Model               |          | AIME(2024)      | MATH500  |                 |  |  |  |
|---------------------|----------|-----------------|----------|-----------------|--|--|--|
|                     | Accuracy | # Average Token | Accuracy | # Average Token |  |  |  |
| Proprietary         |          |                 |          |                 |  |  |  |
| o1-preview          | 12/30    | 9083            | 85.5     | 1501            |  |  |  |
| o1-mini             | 21/30    | 9903            | 90.0     | 944             |  |  |  |
| Parameter Size: 72B |          |                 |          |                 |  |  |  |
| Ours-72B            | 13/30    | 8016            | 87.2     | 2235            |  |  |  |

Table 1: Comparison of the performance between the distilled O1-mini model and O1-series models on the AIME2024 and MATH500 benchmarks under specific inference cost constraints.

Analysis of model behavior and limitations While the model achieves impressive results, there remains a noticeable gap compared to O1-mini in terms of mathematical reasoning performance. Additionally, the generated long thought solutions still exhibit imperfections. Addressing these limitations is critical for closing the performance gap and ensuring the generated long thought solutions meet the highest standards of clarity and correctness.

