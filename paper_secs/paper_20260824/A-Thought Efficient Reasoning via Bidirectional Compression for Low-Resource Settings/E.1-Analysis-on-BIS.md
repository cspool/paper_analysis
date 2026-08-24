# E.1 Analysis on BIS

The design of BIS significantly influences the quality of assessing the importance of individual thought steps. ATTN (attention level importance) and NLL (model level importance) represent distinct measures of reasoning step significance. As demonstrated in Figure 8, manipulating their individual and combined effects on BIS reveals that their joint application is superior for improving the model performance across various budgets.

> **[图片提取文字 (无描述)]:**
> 20 Accuracy (%) Random NLL Attn NLL&Attn
![](_page_16_Figure_4.jpeg)

<span id="page-16-0"></span>Figure 8: Effect of ATTN and NLL on BIS under the 512-token budget.

The parameter  $\alpha$  modulates the balance between question and solution information within BIS. We conducted an ablation analysis, varying  $\alpha$  across the discrete set of values [0, 0.25, 0.5, 0.75, 1], to determine its effect on reasoning data quality and, consequently, on model performance. Specifically, with  $\alpha=0$ , the score is determined only by the information related to the question. In contrast, setting  $\alpha=1$  results in a score based entirely on information pertaining to the solution. As Figure 9 illustrates, optimal model performance is achieved when the BIS effectively integrates both question and solution perspectives, guided by an appropriate setting of  $\alpha$  that ensures a judicious allocation of their respective contributions.

> **[图片提取文字 (无描述)]:**
> 25 37 24 Accuracy (%) 22 20 20 10 Accuracy (%) 35 19 18 34 0.25 0.5 0.75 1.0 0.25 0.5 0.75 1.0 0 0 (a) 512 Tokens (b) 1024 Tokens 52 61 Accuracy (%) 85 85 7 85 85 65 85 85 85 85 85 85 85 85 85 85 85 85 85 51 Accuracy (%) 20 48 48 48 48 48 48 48 48 48 48 48 48 48 48<sup>J</sup> 47 54 46 53 0.25 0.75 1.0 0.25 Ó 0.5 0.5 0.75 1.0 0 (c) 2048 Tokens (d) 4096 Tokens
![](_page_16_Figure_7.jpeg)

<span id="page-16-1"></span>Figure 9: Effect of the hyperparameter  $\alpha$  on model performance.

#### E.2 Analysis on A\* Search

Appropriately setting the maximum exploration steps  $k_{\rm max}$  are keys to optimizing the trade-off between performance and efficiency.

Figure 10 highlights distinct trends: moderate exploration steps are more effective for low-budget scenarios (512-2048 tokens). In contrast, for a 4096-token budget, performance benefits from a greater number of exploration steps. This is likely because more extensive exploration (i.e., deep search) can lead to more concise overall reasoning paths or solutions. Based on these observations, we set  $k_{\rm max}=20$  in our main experiments by default.

> **[图片提取文字 (无描述)]:**
> 60  $k_{\text{max}} = 10$  $k_{\text{max}} = 15$ 50  $k_{\text{max}} = 20$ Accuracy (%) 00 00 00 00 00 00 00 00 00 00 00 00 00 10 512 2048 1024 4096 Tokens
![](_page_17_Figure_3.jpeg)

<span id="page-17-0"></span>Figure 10: Relationship between the exploration step limit  $k_{\rm max}$  and model performance.

The parameter  $\beta$  is used to adjust the weight of the current cost function  $g(\cdot)$  in the overall cost function  $f(\cdot)$ . In the following supplementary experiments, we discussed its discrete values in [0.1, 0.5, 0.9] on the ARC and LiveCodeBench, the experiment results are shown in Table 9.

<span id="page-17-1"></span>Table 9: Effect of the hyperparameter  $\beta$  on model performance.

| Methods       | ARC          |                | LiveCoo    | leBench        | Ave     | ACU            |      |
|---------------|--------------|----------------|------------|----------------|---------|----------------|------|
| 1110111001    | Acc.(\u00e7) | Len.(\( \psi\) | Acc.(†)    | Len.(\( \psi\) | Acc.(†) | Len.(\( \psi\) | 1200 |
|               |              | Bı             | udget: 512 | 2 Tokens       |         |                |      |
| $\beta = 0.1$ | 63.5         | 381.02         | 4.5        | 509.53         | 34.00   | 445.28         | 7.64 |
| $\beta = 0.5$ | 52.5         | 438.03         | 4.0        | 510.73         | 28.25   | 474.38         | 5.96 |
| $\beta = 0.9$ | 48.2         | 469.22         | 5.8        | 506.15         | 27.00   | 487.69         | 5.54 |

