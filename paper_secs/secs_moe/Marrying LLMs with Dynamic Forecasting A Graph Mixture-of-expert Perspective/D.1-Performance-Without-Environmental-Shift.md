# D.1 Performance Without Environmental Shift

In Table [7,](#page-13-2) we can see that there is a significant MSE reduction when we combine the base model with our method.

<span id="page-13-2"></span>

|              | No environmental shift |       |       |       |  |  |  |  |  |  |
|--------------|------------------------|-------|-------|-------|--|--|--|--|--|--|
| Model        | qx                     | qy    | qz    | q     |  |  |  |  |  |  |
| Dynamic      | 10.380                 | 6.398 | 9.496 | 8.333 |  |  |  |  |  |  |
| Linear       | 8.072                  | 5.674 | 7.980 | 6.781 |  |  |  |  |  |  |
| GNN          | 1.591                  | 1.775 | 1.758 | 1.708 |  |  |  |  |  |  |
| Radial Field | 1.133                  | 1.164 | 1.249 | 1.182 |  |  |  |  |  |  |
| EGNN         | 0.671                  | 0.662 | 0.681 | 0.617 |  |  |  |  |  |  |
| EGNN+LEGO    | 0.526                  | 0.649 | 0.574 | 0.583 |  |  |  |  |  |  |
| EGNO         | 0.515                  | 0.544 | 0.498 | 0.519 |  |  |  |  |  |  |
| EGNO+LEGO    | 0.425                  | 0.478 | 0.432 | 0.445 |  |  |  |  |  |  |

Table 7: The MSE (×10−<sup>2</sup> ) of various models on *Charged* without environmental shift.

