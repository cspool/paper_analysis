# <span id="page-15-0"></span>D SPEED COMPARISON OF REMOE AND MOE

<span id="page-15-2"></span>We measure the end-to-end training time for MoE and ReMoE with models of N =469M training over 120B tokens. The time consumption across stages is summarized in Table [6.](#page-15-2) We find Stage I and Stage II account for ∼1.02% of the total training time and incur ∼0.58% overhead.

| Model | Stage I | Stage II | Stage III | Total  |
|-------|---------|----------|-----------|--------|
| MoE   | 0.12    | 0.41     | 119.12    | 119.65 |
| ReMoE | 0.32    | 0.91     | 119.25    | 120.48 |

Table 6: End-to-end training time comparison across stages (in hours). The time is measured on N = 469M, E = 8, k = 1 models training over 120B tokens.

<span id="page-15-3"></span>

| # Parameters | TP | Model        | Train TFLOPS     | Train Diff. | Infer TFLOPS     | Infer Diff. |
|--------------|----|--------------|------------------|-------------|------------------|-------------|
| 182M         | 1  | MoE<br>ReMoE | 103.49<br>105.38 | ↑1.82%      | 78.47<br>80.19   | ↑2.19%      |
| 469M         | 1  | MoE<br>ReMoE | 138.58<br>136.69 | ↓1.37%      | 107.52<br>111.71 | ↑3.89%      |
| 978M         | 1  | MoE<br>ReMoE | 160.46<br>157.61 | ↓1.77%      | 153.11<br>152.76 | ↓0.23%      |
| 978M         | 2  | MoE<br>ReMoE | 133.40<br>132.49 | ↓0.68%      | 118.55<br>117.27 | ↓1.08%      |
| 978M         | 4  | MoE<br>ReMoE | 103.61<br>101.23 | ↓2.29%      | 85.96<br>87.96   | ↑2.33%      |

Table 7: Throughput comparison between TopK-routed MoE and ReLU-routed ReMoE models. TP indicates the tensor parallel size. Train Diff. and Infer Diff. indicate the relative TFLOPS difference of ReMoE compared to MoE, where ↑ denotes ReMoE is faster, and ↓ denotes it is slower.

We further measure the throughput of ReMoE against TopK-routed MoE across different model sizes and tensor parallel sizes during Stage III. The results, presented in Table [7,](#page-15-3) indicate that ReMoE achieves comparable training and inference speeds with MoE, with a minor deviation ranging from −2.29% to +3.89%. This speed consistency is desirable, as ReMoE introduces only a minimal modification to the standard MoE architecture by adjusting the routing function, thereby avoiding additional computational overhead.

