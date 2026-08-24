# C.11 Classification Performance under Ablation Settings

#### C.11.1 Analysis of Training Data Composition

Figure [9](#page-21-0) presents row-normalized confusion matrices comparing classification performance across three training data configurations: Ours (Pos+H-Neg+Neg), Pos+H-Neg, and Pos+Neg. Under the Ours setup, the classifier displays a balanced ability to identify both "Yes" (relevant) and "No" (irrelevant) sentences, achieving an F1-score of 0.92 for both classes. By contrast, excluding one type of negative sample (Pos+H-Neg or Pos+Neg) reduces overall robustness, evidenced by declines in both accuracy and class-wise F1 scores. For instance, the Pos+Neg configuration struggles to maintain balance, accurately identifying "Yes" instances but misclassifying a substantial number of "No" cases. These results confirm that incorporating a comprehensive mix of positive, hard-negative, and randomnegative samples leads to more reliable and contextually aware sentence selection, thereby improving the classifier's performance in practical retrievalaugmented QA scenarios.

<span id="page-21-0"></span>> **[图片提取文字 (无描述)]:**
> Ours (Pos+H-Neg+Neg) Pos+H-Neg Pos+Neg Yes -0.93 0.07 Yes -0.93 0.07 Yes -0.97 0.03 Actual 0.09 0.91 No -0.22 0.78 No -0.40 No -Yes No Yes No Yes No Predicted Predicted Predicted
![](_page_21_Figure_0.jpeg)

Figure 9: Row-normalized confusion matrices for classification performance under different training data conditions: Ours (Pos+H-Neg+Neg), Pos+H-Neg, and Pos+Neg. Each matrix compares the predicted ("Yes"/ "No") labels against the actual labels.

<span id="page-21-1"></span>> **[图片提取文字 (无描述)]:**
> **HQA** (with Context) HQA (w/o Context) 0.93 0.07 0.89 0.11 Actual Actual 0.09 0.91 0.19 0.81 Yes Yes No No Predicted Predicted
![](_page_21_Figure_2.jpeg)

Figure 10: Row-normalized confusion matrices comparing classification performance with (left) and without (right) contextual information on HQA. The availability of context improves the model's ability to accurately distinguish relevant ("Yes") from non-relevant ("No") sentences.

