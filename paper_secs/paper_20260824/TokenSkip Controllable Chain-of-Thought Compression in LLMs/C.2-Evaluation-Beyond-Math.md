# C.2 Evaluation Beyond Math

To demonstrate the generalizability of TokenSkip beyond mathematical reasoning, we present results on CommonsenseQA [\(Talmor et al.,](#page-10-16) [2019\)](#page-10-16), a widely used multiple-choice question answering dataset that requires diverse commonsense knowledge to predict correct answers. For this experiment, we used 9,700 samples from the training set and evaluated TokenSkip on the validation set.

Experimental results on Qwen2.5-Instruct models are shown in Table [5,](#page-12-4) which demonstrate that TokenSkip effectively reduces CoT length by 50% without any performance degradation. These findings further highlight the generalizability of TokenSkip beyond the mathematical reasoning.

<span id="page-12-3"></span>

| Methods                   | Ratio | Accuracy   | Tokens | ActRatio |  |  |  |
|---------------------------|-------|------------|--------|----------|--|--|--|
| MATH-500 (in-domain)      |       |            |        |          |  |  |  |
| Original                  | -     | 48.6(0.0↓) | 502.60 | -        |  |  |  |
|                           | 1.0   | 48.2(0.4↓) | 504.79 | 1.00     |  |  |  |
|                           | 0.9   | 47.8(0.8↓) | 448.31 | 0.89     |  |  |  |
|                           | 0.8   | 47.3(1.3↓) | 398.94 | 0.79     |  |  |  |
| TokenSkip                 | 0.7   | 46.7(1.9↓) | 349.13 | 0.69     |  |  |  |
|                           | 0.6   | 42.0(6.6↓) | 318.36 | 0.63     |  |  |  |
|                           | 0.5   | 40.2(8.4↓) | 292.17 | 0.58     |  |  |  |
| GSM8K (out-of-domain)     |       |            |        |          |  |  |  |
| Original                  | -     | 86.2(0.0↓) | 213.17 | -        |  |  |  |
|                           | 1.0   | 86.0(0.2↓) | 214.49 | 1.00     |  |  |  |
|                           | 0.9   | 84.9(1.3↓) | 201.84 | 0.95     |  |  |  |
|                           | 0.8   | 83.7(2.5↓) | 175.24 | 0.82     |  |  |  |
| TokenSkip                 | 0.7   | 82.6(3.6↓) | 152.32 | 0.71     |  |  |  |
|                           | 0.6   | 79.8(6.4↓) | 136.95 | 0.64     |  |  |  |
|                           | 0.5   | 76.6(9.6↓) | 122.55 | 0.58     |  |  |  |
| MMLU-STEM (out-of-domain) |       |            |        |          |  |  |  |
| Original                  | -     | 58.5(0.0↓) | 356.31 | -        |  |  |  |
|                           | 1.0   | 58.4(0.1↓) | 354.25 | 1.00     |  |  |  |
|                           | 0.9   | 59.4(0.9↑) | 327.18 | 0.92     |  |  |  |
|                           | 0.8   | 59.3(0.8↑) | 286.15 | 0.80     |  |  |  |
| TokenSkip                 | 0.7   | 58.9(0.4↑) | 257.26 | 0.72     |  |  |  |
|                           | 0.6   | 59.2(0.7↑) | 225.33 | 0.63     |  |  |  |
|                           | 0.5   | 58.1(0.4↓) | 188.87 | 0.53     |  |  |  |

Table 4: Out-of-domain results on LLaMA-3.1-8B-Instruct. We report accuracy, average CoT token count, and actual compression ratio (*Act*Ratio) for comparison.

<span id="page-12-4"></span>

| Methods              | Ratio | Accuracy   | Tokens | ActRatio |  |  |  |
|----------------------|-------|------------|--------|----------|--|--|--|
| Qwen2.5-7B-Instruct  |       |            |        |          |  |  |  |
| Original             | -     | 80.3(0.0↓) | 272.13 | -        |  |  |  |
|                      | 1.0   | 80.4(0.1↑) | 273.64 | 1.00     |  |  |  |
|                      | 0.9   | 80.9(0.6↑) | 245.70 | 0.90     |  |  |  |
|                      | 0.8   | 81.1(0.8↑) | 218.73 | 0.80     |  |  |  |
| TokenSkip            | 0.7   | 82.0(1.7↑) | 188.78 | 0.69     |  |  |  |
|                      | 0.6   | 81.5(1.2↑) | 153.17 | 0.56     |  |  |  |
|                      | 0.5   | 80.6(0.3↑) | 128.43 | 0.47     |  |  |  |
| Qwen2.5-14B-Instruct |       |            |        |          |  |  |  |
| Original             | -     | 82.1(0.0↓) | 247.81 | -        |  |  |  |
|                      | 1.0   | 83.8(1.7↑) | 247.34 | 1.00     |  |  |  |
|                      | 0.9   | 82.9(0.8↑) | 221.75 | 0.95     |  |  |  |
|                      | 0.8   | 82.3(0.2↑) | 199.07 | 0.82     |  |  |  |
| TokenSkip            | 0.7   | 82.1(0.0↓) | 172.44 | 0.71     |  |  |  |
|                      | 0.6   | 82.0(0.1↓) | 146.68 | 0.59     |  |  |  |
|                      | 0.5   | 82.1(0.0↓) | 121.03 | 0.49     |  |  |  |

Table 5: Experimental results on CommonsenseQA with Qwen2.5-Instruct models. We report accuracy, average CoT token count, and actual compression ratio (*Act*Ratio) for comparison.