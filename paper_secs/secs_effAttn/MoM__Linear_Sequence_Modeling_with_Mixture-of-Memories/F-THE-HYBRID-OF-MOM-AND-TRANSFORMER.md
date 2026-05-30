# F THE HYBRID OF MOM AND TRANSFORMER

We delve deeper into the hybridization of MoM and Transformer layers by integrating 1 Transformer layer after every 7 MoM layers, resulting in only 3 Transformer layers across a total of 24 layers. The performance on commonsense reasoning and recall-intensive tasks is presented in the table [8.](#page-15-2) MoM-Hybrid demonstrates significantly improved results compared to Transformer models, despite using only 3 layers of global attention.

<span id="page-15-2"></span>Table 8: Hybrid Model Performance. The hybrid model integrates 1 Transformer layer after every 7 MoM layers, resulting in only 3 Transformer layers across a total of 24 layers.

| Model                | FDA            | SWDE           | SQUAD          | NQ             | TriviaQA       | Drop           | Avg.           |
|----------------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| Transformer++<br>MoM | 46.14<br>22.98 | 25.87<br>29.90 | 33.22<br>29.69 | 18.94<br>16.60 | 45.97<br>48.82 | 20.03<br>20.99 | 31.70<br>28.16 |
| MoM Hybrid           | 58.13          | 44.05          | 35.71          | 20.18          | 48.10          | 20.60          | 37.80          |
|                      |                |                |                |                |                |                |                |

| Model                | ARC-e<br>acc↑  | ARC-c<br>accn↑ | Hella.<br>accn↑ | Lamb.<br>acc↑  | PIQA<br>acc↑   | Wino.<br>acc↑  | Avg.           |
|----------------------|----------------|----------------|-----------------|----------------|----------------|----------------|----------------|
| Transformer++<br>MoM | 44.91<br>44.65 | 25.94<br>24.74 | 34.95<br>36.54  | 26.90<br>27.93 | 64.31<br>66.16 | 51.07<br>51.78 | 41.35<br>41.97 |
| MoM Hybrid           | 46.55          | 24.49          | 36.45           | 28.86          | 65.51          | 52.41          | 42.38          |

