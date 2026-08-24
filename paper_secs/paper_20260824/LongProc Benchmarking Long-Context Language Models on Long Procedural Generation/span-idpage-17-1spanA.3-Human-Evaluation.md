# <span id="page-17-1"></span>**A.3 Human Evaluation**

|            | Countdown | Travel |
|------------|-----------|--------|
| GPT-4o     | 68.6      | 14.2   |
| Gemini-pro | 32.3      | 40.0   |
| Human      | 100.0     | 91.4   |

Table 6: Human evaluation results on Countdown and Travel Planning that suggest substantial gaps between best LCLM performance and human performance.

To validate model performance against human capabilities, we conducted a small-scale human evaluation study. Our authors, who **did not** participate in the corresponding generation process of Countdown and Travel Planning, manually attempted 35 8K-level problems from each of these two tasks. We leverage the same prompt and evaluation setup as in our model evaluation. As shown in Table [6,](#page-17-1) our human evaluators achieves an accuracy of 100% on Countdown and 91% on Travel Planning, while frontier LCLMs only solved around 68% and 40% of the same problems, respectively. This comparison demonstrates a substantial performance gap between current LCLMs and human capability.

