# <span id="page-10-0"></span>6. More Baseline Comparisons

We further compare with TFMQ [16] below:

| Bedroom  | W8A8         | W4A8        |
|----------|--------------|-------------|
| TFMQ-DM  | 3.14         | 3.68        |
| QuEST    | <b>3.03</b>  | <b>3.26</b> |
| ImageNet | W8A8         | W4A8        |
| TFMQ-DM  | 10.79        | 10.29       |
| QuEST    | <b>10.43</b> | <b>8.48</b> |

Table 9. Comparing TFMQ.

We also supplement the metrics for Table 3:

| W8A8        | sFID ↓ | IS↑  |
|-------------|--------|------|
| QDiffusion  | 8.19   | 2.25 |
| PTQD        | 9.89   | 2.25 |
| EfficientDM | N/A    | N/A  |
| Ours        | 6.86   | 2.27 |
| W4A4        | sFID ↓ | IS↑  |
| QDiffusion  | N/A    | N/A  |
| DTOD        | NT/A   | N/A  |
| PTQD        | N/A    | IN/A |
| EfficientDM | 15.15  | 2.27 |

Table 10. Additional metrics on LSUN-Bedrooms. "N/A" represents generation failure.

