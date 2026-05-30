# <span id="page-31-0"></span>**F. Additional Evaluations on Math and Code**

The evaluation employs the SC-Math6 corpus, which consists of thousands of Chinese math problems. DeepSeek-V2 Chat (RL) outperforms all Chinese LLMs, including both open-source and close-source models.

We further share more results in Figure [5](#page-33-0) on HumanEval and LiveCodeBench, where the

<span id="page-32-1"></span>

| Agreement          | Ground-Truth Label | Annotator 1 | Annotator 2 | Annotator 3 |
|--------------------|--------------------|-------------|-------------|-------------|
| Ground-Truth Label | 100.0%             | 66.7%       | 59.8%       | 42.1%       |
| Annotator 1        | 66.7%              | 100.0%      | 57.9%       | 69.0%       |
| Annotator 2        | 59.8%              | 57.9%       | 100.0%      | 65.5%       |
| Annotator 3        | 42.1%              | 69.0%       | 65.5%       | 100.0%      |

Table 10 | Three well-educated human annotators conduct independent annotations on 420 moral scenarios from the MMLU Humanity-Moral subset, on which DeepSeek-V2 and its competitive models demonstrate performance inconsistency. Three annotators and the ground-truth label exhibit a low agreement with each other. This indicates that the answers to the Humanity-Moral subset can be contentious according to specific regional cultures.

| Model Name            | R Level | Comp. Score | Reas. Steps Score | OvrAcc Score |
|-----------------------|---------|-------------|-------------------|--------------|
| GPT-4-1106-Preview    | 5       | 90.71       | 91.65             | 89.77        |
| GPT-4                 | 5       | 88.40       | 89.10             | 87.71        |
| DeepSeek-V2 Chat (RL) | 5       | 83.35       | 85.73             | 84.54        |
| Ernie-bot 4.0         | 5       | 85.60       | 86.82             | 84.38        |
| Qwen-110B-Chat        | 5       | 83.25       | 84.93             | 84.09        |
| GLM-4                 | 5       | 84.24       | 85.72             | 82.77        |
| Xinghuo 3.5           | 5       | 83.73       | 85.37             | 82.09        |
| Qwen-72B-Chat         | 4       | 78.42       | 80.07             | 79.25        |
| ChatGLM-Turbo         | 4       | 57.70       | 60.32             | 55.09        |
| GPT-3.5-Turbo         | 4       | 57.05       | 59.61             | 54.50        |
| Qwen-14B-Chat         | 4       | 53.12       | 55.99             | 50.26        |
| ChatGLM3-6B           | 3       | 40.90       | 44.20             | 37.60        |
| Xinghuo 3.0           | 3       | 40.08       | 45.27             | 34.89        |
| Baichuan2-13B-Chat    | 3       | 39.40       | 42.63             | 36.18        |
| Ernie-3.5-turbo       | 2       | 25.19       | 27.70             | 22.67        |
| Chinese-Alpaca2-13B   | 2       | 20.55       | 22.52             | 18.58        |

Table 11 | SC-Math6 Model Reasoning Level. "R Level" stands for Reasoning Level, "Comp. Score" stands for Comprehensive Score, "Reas. Steps Score" stands for Reasoning Steps Score, and "OvrAcc Score" stands for Overall Accuracy Score.

questions of LiveCodeBench are selected from the period between September 1st, 2023, and April 1st, 2024. As shown in the figure, DeepSeek-V2 Chat (RL) demonstrates considerable proficiency in LiveCodeBench, achieving a Pass@1 score that even surpasses some giant models. This performance highlights the strong capability of DeepSeek-V2 Chat (RL) in tackling live coding tasks.

