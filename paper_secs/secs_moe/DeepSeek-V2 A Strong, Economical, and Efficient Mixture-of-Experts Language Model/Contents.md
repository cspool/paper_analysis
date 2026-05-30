# **Contents**

| 1<br>Introduction |     |              |                                                                 |    |  |  |  |
|-------------------|-----|--------------|-----------------------------------------------------------------|----|--|--|--|
| 2                 |     | Architecture |                                                                 | 6  |  |  |  |
|                   | 2.1 |              | Multi-Head Latent Attention: Boosting Inference Efficiency      | 6  |  |  |  |
|                   |     | 2.1.1        | Preliminaries: Standard Multi-Head Attention<br>                | 6  |  |  |  |
|                   |     | 2.1.2        | Low-Rank Key-Value Joint Compression<br>                        | 7  |  |  |  |
|                   |     | 2.1.3        | Decoupled Rotary Position Embedding<br>                         | 8  |  |  |  |
|                   |     | 2.1.4        | Comparison of Key-Value Cache<br>                               | 8  |  |  |  |
|                   | 2.2 |              | DeepSeekMoE: Training Strong Models at Economical Costs<br>     | 9  |  |  |  |
|                   |     | 2.2.1        | Basic Architecture<br>                                          | 9  |  |  |  |
|                   |     | 2.2.2        | Device-Limited Routing<br>                                      | 9  |  |  |  |
|                   |     | 2.2.3        | Auxiliary Loss for Load Balance<br>                             | 10 |  |  |  |
|                   |     | 2.2.4        | Token-Dropping Strategy<br>                                     | 11 |  |  |  |
| 3                 |     | Pre-Training |                                                                 | 11 |  |  |  |
|                   | 3.1 |              | Experimental Setups<br>                                         | 11 |  |  |  |
|                   |     | 3.1.1        | Data Construction<br>                                           | 11 |  |  |  |
|                   |     | 3.1.2        | Hyper-Parameters<br>                                            | 12 |  |  |  |
|                   |     | 3.1.3        | Infrastructures<br>                                             | 12 |  |  |  |
|                   |     | 3.1.4        | Long Context Extension<br>                                      | 13 |  |  |  |
|                   | 3.2 |              | Evaluations<br>                                                 | 13 |  |  |  |
|                   |     | 3.2.1        | Evaluation Benchmarks<br>                                       | 13 |  |  |  |
|                   |     | 3.2.2        | Evaluation Results<br>                                          | 14 |  |  |  |
|                   |     | 3.2.3        | Training and Inference Efficiency                               | 16 |  |  |  |
| 4                 |     | Alignment    |                                                                 | 16 |  |  |  |
|                   | 4.1 |              | Supervised Fine-Tuning<br>                                      | 16 |  |  |  |
|                   | 4.2 |              | Reinforcement Learning<br>                                      | 17 |  |  |  |
|                   | 4.3 |              | Evaluation Results<br>                                          | 18 |  |  |  |
|                   | 4.4 | Discussion   |                                                                 | 20 |  |  |  |
| 5                 |     |              | Conclusion, Limitation, and Future Work                         | 21 |  |  |  |
| A                 |     |              | Contributions and Acknowledgments                               | 27 |  |  |  |
| B                 |     |              | DeepSeek-V2-Lite: A 16B Model Equipped with MLA and DeepSeekMoE | 29 |  |  |  |

|   | B.1<br>Model Description<br>                 | 29 |
|---|----------------------------------------------|----|
|   | B.2<br>Performance Evaluation<br>            | 30 |
| C | Full Formulas of MLA                         | 31 |
| D | Ablation of Attention Mechanisms             | 31 |
|   | D.1<br>Ablation of MHA, GQA, and MQA         | 31 |
|   | D.2<br>Comparison Between MLA and MHA<br>    | 31 |
| E | Discussion About Pre-Training Data Debiasing | 32 |
| F | Additional Evaluations on Math and Code      | 32 |
| G | Evaluation Formats                           | 33 |

