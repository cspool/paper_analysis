# <span id="page-15-0"></span>E Further details on the Prompting Method

In this section, we will further discuss three types of prompt methods. As shown in Table [9,](#page-15-2) we found that as the number of image tokens decreases, the effectiveness of the image first prompt and sandwich prompt also monotonically decreases, which is intuitively expected as fewer image tokens contain less pictorial information. Nonetheless, the image last prompt does not exhibit a strictly decreasing trend; it initially increases and subsequently decreases, achieving optimal performance at the point of 145 image tokens. The effect is especially evident in scenarios of train-test mismatch. We term this the information barrier formed by image tokens, which hinders the model's information transfer.

<span id="page-15-2"></span>An additional observation indicates that the sandwich prompt is capable of mitigating information loss, sustaining good performance even with a limited number of image tokens. In contrast, the other two types of prompt methods fail to achieve this.

| Method                                                | Size | Prompt   | Image Tokens | ScienceQA | TextVQA | GQA    |
|-------------------------------------------------------|------|----------|--------------|-----------|---------|--------|
| VisualRWKV-Base<br>VisualRWKV-Base<br>VisualRWKV-Base |      |          | 577          | 65.59%    | 47.13%  | 48.52% |
|                                                       |      |          | 145          | 64.14%    | 42.91%  | 45.99% |
|                                                       |      |          | 65           | 64.01%    | 40.67%  | 44.08% |
|                                                       |      |          | 37           | 62.87%    | 39.90%  | 43.44% |
|                                                       | 7B   | First    | 17           | 61.23%    | 39.96%  | 43.31% |
|                                                       |      |          | 10           | 60.29%    | 39.65%  | 43.23% |
|                                                       |      |          | 5            | 59.35%    | 39.80%  | 43.16% |
|                                                       |      |          | 1            | 57.11%    | 39.34%  | 43.53% |
|                                                       |      |          | 577          | 57.66%    | 48.52%  | 44.19% |
|                                                       |      | Last     | 145          | 58.75%    | 45.29%  | 42.93% |
|                                                       |      |          | 65           | 56.07%    | 43.89%  | 42.38% |
|                                                       |      |          | 37           | 53.35%    | 43.03%  | 42.07% |
|                                                       | 7B   |          | 17           | 50.37%    | 42.50%  | 42.03% |
|                                                       |      |          | 10           | 50.72%    | 42.18%  | 42.10% |
|                                                       |      |          | 5            | 49.23%    | 41.20%  | 41.80% |
|                                                       |      |          | 1            | 50.67%    | 41.19%  | 41.93% |
|                                                       |      |          | 577          | 65.20%    | 50.25%  | 50.50% |
|                                                       |      |          | 145          | 64.90%    | 46.38%  | 47.47% |
|                                                       |      |          | 65           | 64.40%    | 44.58%  | 45.09% |
|                                                       |      |          | 37           | 64.11%    | 44.01%  | 44.78% |
|                                                       | 7B   | Sandwich | 17           | 63.86%    | 43.61%  | 44.57% |
|                                                       |      |          | 10           | 63.26%    | 43.27%  | 44.37% |
|                                                       |      |          | 5            | 62.87%    | 43.03%  | 44.08% |
|                                                       |      |          | 1            | 60.34%    | 41.72%  | 36.09% |

Table 9: Full Results for three prompting method.

