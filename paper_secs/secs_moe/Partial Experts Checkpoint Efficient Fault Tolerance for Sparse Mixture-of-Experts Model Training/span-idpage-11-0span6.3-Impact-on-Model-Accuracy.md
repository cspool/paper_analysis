# <span id="page-11-0"></span>6.3 Impact on Model Accuracy

In Section 3.1.2 and Figure 5, we initially demonstrate the efficacy of our proposed PEC in reducing checkpoint size

<span id="page-12-0"></span>**Table 3.** Accuracy results (%) of the models on downstream tasks, pre-trained as shown in Figure 14. The downstream tasks includes: HellaSwag [80], PIQA [4], WinoGrande [57], BoolQ [10], ARC-Easy [11], OBQA [41], RACE [29], MathQA [2]. "Ckpt" indicates the relative total checkpoint size compared to the baseline, which saves the full model states. "Deviation" shows the deviation of the minimum and maximum accuracy of our methods from the baseline.

| Method    | Ckpt | HellaSwag    | PIQA          | WinoGrande    | BoolQ        | ARC-E        | OBQA          | RACE         | MathQA        | Avg. (↑)     |
|-----------|------|--------------|---------------|---------------|--------------|--------------|---------------|--------------|---------------|--------------|
| Baseline  | 1    | 26.85        | 58.22         | 49.09         | 54.77        | 36.83        | 13.00         | 24.21        | 20.54         | 35.44        |
| W         | 0.88 | 26.92        | 58.16         | 49.72         | 57.52        | 37.84        | 12.80         | 24.69        | 20.84         | 36.06        |
| O         | 0.54 | 26.93        | 58.00         | 48.54         | 61.28        | 37.21        | 13.40         | 25.26        | 19.97         | 36.32        |
| WO        | 0.42 | 26.91        | 58.38         | 49.33         | 61.31        | 37.33        | 13.20         | 24.50        | 20.20         | 36.40        |
| WO-2L     | 0.42 | 26.96        | 58.49         | 50.12         | 61.74        | 37.12        | 13.20         | 24.40        | 20.13         | 36.52        |
| Deviation | -    | (0.06, 0.11) | (-0.22, 0.27) | (-0.55, 1.03) | (2.75, 6.97) | (0.29, 1.01) | (-0.20, 0.40) | (0.19, 1.05) | (-0.57, 0.30) | (0.62, 1.08) |

without compromising model accuracy. We then conduct an in-depth evaluation of its impact on model accuracy.

As shown in Figure 14(a), applying PEC to save model weights ("W"), optimizer states ("O"), or both ("WO" and "WO-2L") results in a validation loss curve comparable to the baseline, which saves the full states during the pre-training of the GPT-350M-16E model.

Given the similar training curves across different checkpointing methods, we further evaluate downstream tasks for each pre-trained model. Compared to the baseline method, which retains all states, our lossy methods ("W", "O", "WO" and "WO-2L") achieve higher average accuracy, ranging from 0.62% to 1.08%, as shown in Table 3. Notably, our methods show the most significant accuracy improvement on the BoolQ task, ranging from 2.75% to 6.97%. We hypothesize that this level of improvement may result from state loss caused by our PEC, acting as a variant of dropout [64], which helps prevent overfitting in certain domains.

**6.3.1 Two-level PEC Saving and Recovery.** We evaluate the effectiveness of our two-level PEC saving and recovery scheme in minimizing PLT and maintaining model accuracy. Given the faster speed of the snapshot process compared to the persist process, we configure  $K_{persist} = 1$  and experiment with varying  $K_{snapshot}$  values, as depicted in Figure 15(a). Compared with the baseline ( $K_{snapshot} = 1$ ,  $K_{persist} = 1$ ) setup, increasing  $K_{snapshot}$  markedly reduces PLT, owing to the retrieval of partial experts from the in-memory snapshots on the non-fault node. Moreover, the two-level recovery with the ( $K_{snapshot} = 4$ ,  $K_{persist} = 1$ ) setup ("WO-2L" in Table3) achieves the highest average accuracy on downstream tasks, exceeding the baseline by 1.08%.

6.3.2 Sequential versus Load-aware Selection. We conduct experiments on the SwinV2-MoE model pre-training to evaluate the impact of different partial expert selection methods on model accuracy. As shown in Figure 14(b), the three methods—baseline, PEC with sequential selection, and PEC with load-aware selection—exhibit minimal differences, with less than a 0.0012% variance in test accuracy after 80 training epochs. Considering that load-aware selection incurs

<span id="page-12-1"></span>![](_page_12_Figure_9.jpeg)

<span id="page-12-2"></span>**Figure 15.** (a) shows the correlation between PLT and various combinations of  $K_{snapshot}$  and  $K_{persist}$ , using two-level recovery. The error bar represents the fluctuation in measured values. (b) demonstrates the efficacy of our Dynamic-K strategy in reducing PLT, with the red line tracking the dynamic adjustments of  $K_{pec}$ . These experiments are conducted during the pre-training of the GPT-350M-16E model in Case2.

additional control and synchronization costs while maintaining comparable accuracy, sequential selection appears to be the more practical choice for real-world applications. Additionally, these experiments confirm that our PEC method is applicable to both language and vision models.

**6.3.3 Dynamic-K.** We evaluate the efficacy of our proposed dynamic-K strategy in ensuring that the PLT does not exceed the pre-set threshold of 3.75% as the number of faults increases. As shown in Figure 15(b), the value of  $K_{pec}$  dynamically adjusts from 1 to 4, in response to escalating fault occurrences. With this strategy, the cumulative PLT remains at a low level, whereas a constant setting of  $K_{pec} = 1$  results in a linear increase.

6.3.4 Fault Tolerance during Fine-Tuning. In addition to the model's pre-training phase, fine-tuning is another crucial stage that requires extended training periods and fault tolerance. To evaluate the impact of our proposed PEC during the fine-tuning phase, we conduct experiments using the Alpaca dataset [65] to fine-tune the open-source, pre-trained OLMoE model [43]. We set a fault interruption occurring

<span id="page-13-14"></span>Table 4. Accuracy results from fine-tuning the OLMoE [\[43\]](#page-14-29) model using various methods. "Base" refers to the pre-trained model without fine-tuning, "FT-w.o.E" indicates the finetuned model without fine-tuning all expert parameters, "FT-Full" represents the fine-tuned model with full state saving at each checkpointing, and "FT-PEC" denotes the fine-tuned model utilizing PEC that saves 1/8 of the experts at each checkpoint. The tasks includes: HellaSwag [\[80\]](#page-16-9), PIQA [\[4\]](#page-13-10), WG [\[57\]](#page-15-26), BoolQ [\[10\]](#page-13-11), ARC-C [\[11\]](#page-13-12), OBQA [\[41\]](#page-14-27), RTE [\[69\]](#page-15-29).

| Method   | HS    | PIQA  | WG    | BQ    | ARC   | OBQA  | RTE   | Avg.  |
|----------|-------|-------|-------|-------|-------|-------|-------|-------|
| Base     | 57.99 | 80.52 | 68.59 | 74.46 | 47.27 | 44.80 | 54.51 | 61.16 |
| FT-w.o.E | 58.58 | 81.88 | 68.51 | 76.82 | 48.72 | 45.20 | 63.54 | 63.32 |
| FT-Full  | 58.34 | 81.34 | 70.40 | 79.11 | 48.38 | 45.00 | 66.06 | 64.09 |
| FT-PEC   | 58.78 | 81.45 | 70.24 | 79.17 | 48.23 | 45.00 | 65.58 | 64.06 |

halfway through the process. As shown in Table [4,](#page-13-14) PEC maintains accuracy comparable to the full-saving method. Additionally, we conduct experiments on fine-tuning with freezing all the expert parameters. This approach still achieves an increase in average accuracy, from 61.16% to 63.32%, with only a slight degradation of 0.77% compared to full-parameter fine-tuning. These results further substantiate that the expert parameters are less sensitive to a limited number of updates.

