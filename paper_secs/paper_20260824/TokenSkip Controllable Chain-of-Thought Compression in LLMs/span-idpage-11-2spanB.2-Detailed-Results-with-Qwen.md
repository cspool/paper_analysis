# <span id="page-11-2"></span>B.2 Detailed Results with Qwen

We provide detailed experimental results of the Qwen2.5-Instruct series evaluated on GSM8K in Table [2.](#page-11-6) As the model scale increases, there is less performance degradation at higher compression ratios, indicating that larger LLMs are better at identifying shortcuts between critical reasoning tokens, enabling more efficient CoT generation.

<span id="page-11-6"></span>

| Scale | Methods   | Ratio | Accuracy   | Tokens | ActRatio |
|-------|-----------|-------|------------|--------|----------|
| 3B    | Original  | -     | 83.7(0.0↓) | 314.87 | -        |
|       |           | 1.0   | 83.4(0.3↓) | 318.79 | 1.00     |
|       | TokenSkip | 0.9   | 83.2(0.5↓) | 262.99 | 0.83     |
|       |           | 0.8   | 81.6(2.1↓) | 250.71 | 0.79     |
|       |           | 0.7   | 80.1(3.6↓) | 233.03 | 0.73     |
|       |           | 0.6   | 77.3(6.4↓) | 199.55 | 0.63     |
|       |           | 0.5   | 74.4(9.3↓) | 170.55 | 0.54     |
|       | Original  | -     | 91.4(0.0↓) | 297.83 | -        |
|       | TokenSkip | 1.0   | 91.7(0.3↑) | 295.78 | 1.00     |
| 7B    |           | 0.9   | 91.1(0.3↓) | 254.77 | 0.86     |
|       |           | 0.8   | 90.1(1.3↓) | 237.27 | 0.80     |
|       |           | 0.7   | 89.9(1.5↓) | 216.73 | 0.73     |
|       |           | 0.6   | 87.9(3.5↓) | 178.07 | 0.60     |
|       |           | 0.5   | 86.0(5.4↓) | 151.44 | 0.51     |
| 14B   | Original  | -     | 93.1(0.0↓) | 313.11 | -        |
|       | TokenSkip | 1.0   | 93.0(0.1↓) | 314.55 | 1.00     |
|       |           | 0.9   | 93.3(0.2↑) | 269.22 | 0.86     |
|       |           | 0.8   | 93.2(0.1↑) | 247.24 | 0.79     |
|       |           | 0.7   | 93.4(0.3↑) | 218.62 | 0.70     |
|       |           | 0.6   | 92.7(0.4↓) | 180.68 | 0.57     |
|       |           | 0.5   | 91.4(1.7↓) | 156.85 | 0.50     |

Table 2: Experimental results on the Qwen2.5-Instruct series. We report accuracy, average CoT token count, and actual compression ratio (*Act*Ratio) for comparison.

<span id="page-11-5"></span><sup>4</sup> Since many samples reach the maximum length when testing TokenSkip on MATH-500, we adjust its length budget to max\_len×γ, with no adjustment for GSM8K.

