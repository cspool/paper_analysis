# C Selection of top M experts in different tasks

We also conduct experiments on Qwen unlearned by GA with RWKU dataset to investigate the optimal selection of M. The results in Tab. 7 indicate that SEUF achieves the best performance when only one expert is unlearned M=1, which is consistent with the Insight 4.

<span id="page-13-1"></span>Table 7: Model utility (UT $\uparrow$ ) comparison at the same level of forget efficacy (FE $\approx$  0.25), when the top M experts from either the same layer or different layers in Qwen are unlearned using GA on RWKU benchmark, also when 4 shared experts are included

| Selected experts | Top-1  | Top-3  | Top-6  | Top-1+4-shared |
|------------------|--------|--------|--------|----------------|
| Same layer       | 0.5709 | 0.3695 | 0.2572 | 0.2445         |
| Different layers | 0.5709 | 0.4224 | 0.3872 | -              |

<span id="page-13-3"></span>Table 8: Expert selection overlap between different sampling splits

| Dataset  | Full | Subset 1 | Subset 2 |
|----------|------|----------|----------|
| Full     | 1.00 | 0.94     | 0.85     |
| Subset 1 | 0.94 | 1.00     | 0.87     |
| Subset 2 | 0.85 | 0.87     | 1.00     |

<span id="page-13-4"></span>Table 9: Performance of Mixtral 8x7B unlearned by GA on WMDP and RWKU datasets.

| Method     | WN                      | 1DP                   | RW                      | 'KU         |
|------------|-------------------------|-----------------------|-------------------------|-------------|
|            | $\mathbf{FE}\downarrow$ | $\mathbf{UT}\uparrow$ | $\mathbf{FE}\downarrow$ | <b>UT</b> ↑ |
| Pretrained | 0.5229                  | 0.6885                | 0.5820                  | 0.6885      |
| LoRA       | 0.2658                  | 0.2597                | 0.0000                  | 0.2295      |
| ESFT       | 0.2574                  | 0.6386                | 0.0542                  | 0.6743      |
| SEUF       | 0.2608                  | 0.6364                | 0.0455                  | 0.6713      |

### **D** Robustness of Expert Selection

To evaluate the robustness of expert selection under token sampling, we conducted an additional experiment on a consistency analysis on the DeepSeek-V2-Lite model using the WMDP forget set. Specifically, we computed the overlap ratio of selected experts across different token subsets, where overlap is defined as the proportion of shared top-6 experts at each MoE layer.

As shown in Table 8, a subset of 100,000 tokens yields a high overlap (0.94) with the expert selections derived from the full dataset. Furthermore, two independently sampled subsets also show strong agreement with each other (0.87 overlap), indicating that the attribution process is stable across different sampling runs.

#### E Experiments on Larger MoE Models

To explore if SEUF can be applied to larger MoE models, we evaluated SEUF on mistralai/Mixtral-8x7B-Instruct-v0.1 (Mixtral 8x7B), one of the most widely used large-scale open-source MoE models, and compared its performance to other parameter-efficient unlearning baselines.

As shown in Table 9, SEUF achieves comparable or even better utility (UT) while maintaining strong forget efficacy (FE). On the WMDP dataset, SEUF achieves a UT of 0.6364, close to ESFT's 0.6386 and far better than LoRA's 0.2597. On RWKU, SEUF reaches 0.6713, again comparable to ESFT (0.6743) and significantly ahead of LoRA (0.2295). Importantly, SEUF does so while updating only

<span id="page-14-0"></span>Table 10: Tunable parameter ratio of different methods

| Method | Tunable Parameter Ratio ↓ |
|--------|---------------------------|
| LoRA   | 0.26%                     |
| ESFT   | 14%                       |
| SEUF   | 0.41%                     |

<span id="page-14-1"></span>Table 11: Table A: The Spearman's rank correlation between gi,t and gi,tE(xi) for All experts and Top 6 experts across all layers in DeepSeek.

| Range       | All experts | Top 6 |
|-------------|-------------|-------|
| Correlation | 1.0         | 1.0   |

0.41% of parameters, as shown in Table [10,](#page-14-0) substantially fewer than ESFT's 14%.

