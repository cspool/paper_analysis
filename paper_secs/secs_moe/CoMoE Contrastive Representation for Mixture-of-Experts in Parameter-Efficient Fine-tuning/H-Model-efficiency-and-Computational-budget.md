# H Model efficiency and Computational budget

To evaluate the model efficiency and computational budget of CoMoE, we compare it with the baselines(LORA, DoRA, MixLoRA, OMOE) in three aspects: inference latency, memory cost, and training time. We base our evaluation on the following three metrics: (a) the inference time required for generating responses (ms), (b) the GPU memory cost (MiB), and (c) the training time in multi-task settings (h). In the multi-task setting, ARC-e, ARC-c, BoolQ, and OBQA are trained

| Number of Expertsn | ARC-e | ARC-c | BoolQ | OBQA | Avg. |
|--------------------|-------|-------|-------|------|------|
| n=4                | 80.0  | 66.6  | 71.2  | 71.6 | 73.9 |
| n=5                | 79.8  | 64.4  | 70.9  | 714  | 73.1 |
| n=6                | 80.6  | 64.5  | 73.1  | 76.0 | 73.6 |
| n=7                | 79.7  | 65.4  | 72.5  | 78.0 | 73.9 |
| n=8                | 79.1  | 64.7  | 72.0  | 80.8 | 74.2 |

Table 7: Accuracy results across different expert configurations (from 4 to 8 experts) on multi-task evaluation.

| Method       | ARC-e | ARC-c | BoolQ | OBQA | Avg. |
|--------------|-------|-------|-------|------|------|
| MixLoRA      | 22.4  | 24.0  | 62.2  | 27.6 | 34.1 |
| CoMoE-LoRA — | 25.7  | 23.7  | 62.2  | 25.2 | 34.2 |

Table 8: Comparison of MixLoRA and CoMoE in multi-task learning. The backbone model is Gemma 2B.

| Method      | Latency (ms) |        | Memory (MiB) _ Training time (h) |
|-------------|--------------|--------|----------------------------------|
| LoRA        | 2,096        | +1,630 | 1.8h                             |
| DoRA        | 1,748        | +2,184 | 1.7h                             |
| MixLoRA     | 4,217        | +1,776 | 2.2h                             |
| OMoE(Top-2) | 4,863        | +1,776 | 2.3h                             |
| CoMoE       | 3,789        | +1,311 | 3.5h                             |

Table 9: The inference latency, memory cost and training time of the LLaMA-2 7B for generating a batch of responses using CoMoE and baselines.

simultaneously. The results are provided in Table 9. From Table 9, we observe that compared to the well-performing MixLoRA, CoMoE achieves a 10% improvement in inference efficiency while reducing GPU memory usage by 465 MiB. In terms of training time, CoMoE requires 3.5 hours of training on an A6000 GPU under the multi-task setting. Although CoMoE increases the training burden, it does not compromise inference efficiency and simultaneously enhances model performance. Furthermore, we conducted experiments to investigate the relationship between training time and the number of experts, as shown in Table 10. In our proposed method, negative samples are drawn from the set of inactive experts. As a result, the training cost increases linearly with the number of experts, with a time complexity of O(n). To mitigate this computational overhead, we introduce a fixed-size sampling strategy, wherein a constant number of negative samples are randomly selected from the inactive experts at each training step. This reduces the complexity from O(n) to constant time O(1), rendering it independent of the total number of experts. The results of this strategy are summarized in Table 11. Theoretically, this approach introduces a looser lower bound (see Theorem 1); however, empirical results show no degradation in performance, as evidenced by the comparable average

accuracy between Table 10 and Table 11.

