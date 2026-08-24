# <span id="page-15-2"></span>**H. Generalization of pricing configurations**

In Section [6.3,](#page-7-3) we examined Orchestrator-8B's ability to generalize to unseen tools. Here, we investigate its generalization across heterogeneous pricing regimes, where the same tools are assigned different costs. We evaluate whether the model adapts by adjusting its tool-calling strategy to optimize outcomes, efficiency, and alignment with user preferences—reflecting realistic settings in which tool costs vary across users. We test Orchestrator-8B under a pricing configuration not encountered during training. Specifically, we use the pricing configuration from deepinfra[5](#page-15-3) . As shown in Table [4,](#page-16-2) it consistently delivers superior performance, cost

<span id="page-15-3"></span><sup>5</sup>https://deepinfra.com

<span id="page-16-2"></span>Table 4 | Generalization performance under different a pricing configuration. Orchestrator-8B consistently performs the best in terms of performance, cost and latency, which illustates the robustness of the Orchestrator

|                 | HLE (↑) | Frames (↑) | 2<br>-Bench (↑)<br>𝜏 | Cost (↓) | Latency (↓) |
|-----------------|---------|------------|----------------------|----------|-------------|
| Qwen3-8B        | 29.7    | 68.2       | 71.9                 | 27.4     | 17.9        |
| Nemotron-49B    | 25.6    | 57.8       | 66.3                 | 24.3     | 17.2        |
| Llama-3.3-70B   | 19.6    | 52.2       | 55.4                 | 17.9     | 12.0        |
| Qwen3-235B-A22B | 32.4    | 74.1       | 75.3                 | 27.9     | 20.8        |
| Claude Opus 4.1 | 34.5    | 72.3       | 76.4                 | 52.3     | 25.1        |
| GPT-5           | 20.8    | 57.3       | 61.9                 | 17.5     | 13.2        |
| Orchestrator-8B | 36.9    | 76.6       | 80.4                 | 7.5      | 7.8         |

efficiency, and accuracy. These results demonstrate that training with diverse pricing configurations produces a model that is not constrained to a particular tool setup and can robustly generalize across diverse user scenarios.

