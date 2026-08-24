# C.10 Classification Performance

To better understand the effectiveness of our context-aware relevance classifier, we report rownormalized confusion matrices for both in-domain (HQA) and out-of-domain (2WIKI) datasets, as shown in Figure [8.](#page-20-3) On HQA, the classifier displays a perfectly balanced ability to recognize both relevant ("Yes") and irrelevant ("No") sentences, achieving over 90% precision and recall in each category. While, on the 2WIKI dataset, the classifier exhibits a slight drop in recall for "Yes" sentences, it still shows a strong classification ability with over 70% recall and 90% precision.

These results confirm that the classifier performs robustly in its training domain and generalizes reasonably well to unseen queries, though we leave

<span id="page-20-2"></span>Table 15: Classification performance for Yes/No labels.

| Overall       |                                       |      |      |  |  |  |  |  |  |  |
|---------------|---------------------------------------|------|------|--|--|--|--|--|--|--|
|               | Class Precision ↑ Recall ↑ F1-Score ↑ |      |      |  |  |  |  |  |  |  |
| Yes           | 0.91                                  | 0.93 | 0.92 |  |  |  |  |  |  |  |
| No            | 0.93                                  | 0.91 | 0.92 |  |  |  |  |  |  |  |
| Hard Negative |                                       |      |      |  |  |  |  |  |  |  |
| Yes           | 0.86                                  | 0.93 | 0.89 |  |  |  |  |  |  |  |
| No            | 0.93                                  | 0.84 | 0.88 |  |  |  |  |  |  |  |
| Negative      |                                       |      |      |  |  |  |  |  |  |  |
| Yes           | 0.96                                  | 0.93 | 0.95 |  |  |  |  |  |  |  |
| No            | 0.93                                  | 0.96 | 0.95 |  |  |  |  |  |  |  |

<span id="page-20-3"></span>> **[图片提取文字 (无描述)]:**
> HQA (In-Domain) 2WIKI (Out-of-Domain) Yes 0.93 0.07 0.76 0.24 Actual Actual 0.09 0.91 0.06 0.94 Yes Yes No No Predicted Predicted
![](_page_20_Figure_11.jpeg)

Figure 8: Confusion matrices (row-normalized) for context-aware relevance classification on HQA (indomain) and 2WIKI (out-of-domain).

narrowing this discrepancy as a valuable future research direction.

