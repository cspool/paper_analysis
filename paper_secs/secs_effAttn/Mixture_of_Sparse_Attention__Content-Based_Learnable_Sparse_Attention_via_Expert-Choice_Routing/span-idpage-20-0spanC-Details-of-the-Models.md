# <span id="page-20-0"></span>C Details of the Models

In the Table [4](#page-21-0) we list hyperparameters all of dense baselines.

<span id="page-21-0"></span>

|                         | Tiny  | Small  | Medium | Large    |
|-------------------------|-------|--------|--------|----------|
| FLOPs per pass (G)      | 54.76 | 219.85 | 430.70 | 1,130.65 |
| Layers                  | 6     | 9      | 18     | 27       |
| Hidden size             | 512   | 1,024  | 1,024  | 1,280    |
| Feedforward hidden size | 2,048 | 4,096  | 4,096  | 5,120    |
| Head hidden size        | 64    | 64     | 64     | 64       |
| Number of heads         | 9     | 9      | 9      | 16       |

Table 4: Hyperparameters of the different model variants and the corresponding FLOP cost of the forward pass for a sequence length of T = 1024.

<span id="page-22-0"></span>

|       |           | Sparsity                                |       |       |       |                                         |       |       |       |       |
|-------|-----------|-----------------------------------------|-------|-------|-------|-----------------------------------------|-------|-------|-------|-------|
|       |           | 1                                       | 2     | 4     | 8     | 16                                      | 32    | 64    | 128   | 256   |
|       |           | Perplexity (↓) for given sparsity       |       |       |       |                                         |       |       |       |       |
| Tiny  | MoSA      | 22.46                                   | 21.76 | 20.45 | 19.24 | 18.00                                   | 16.90 | 16.39 | 17.27 | 18.06 |
|       | Pure MoSA | 22.46                                   | 22.96 | 23.30 | 24.78 | 29.76                                   | -     | -     | -     | -     |
| Small | MoSA      | 16.01                                   | 15.74 | 15.10 | 14.48 | 13.65                                   | 12.97 | 12.85 | -     | -     |
|       | Pure MoSA | 16.01                                   | 16.35 | 17.16 | 19.61 | 25.41                                   | -     | -     | -     | -     |
|       | MoSA      | 13.95                                   | 13.52 | 12.81 | 12.16 | 11.47                                   | 11.06 | -     | -     | -     |
| Med.  | Pure MoSA | 13.95                                   | 14.03 | 14.40 | 15.87 | 20.63                                   | -     | -     | -     | -     |
|       | MoSA      | 12.20                                   | 11.33 | 10.58 | -     | -                                       | -     | -     | -     | -     |
| Large | Pure MoSA | 12.20                                   | 11.83 | 11.97 | -     | -                                       | -     | -     | -     | -     |
|       |           |                                         |       |       |       |                                         |       |       |       |       |
|       |           | Number of parameters for given sparsity |       |       |       |                                         |       |       |       |       |
| Tiny  | MoSA      | 28M                                     | 34M   | 48M   | 78M   | 136M                                    | 242M  | 423M  | 693M  | 1B    |
|       | Pure MoSA | 28M                                     | 39M   | 65M   | 119M  | 222M                                    | -     | -     | -     | -     |
| Small | MoSA      | 113M                                    | 127M  | 163M  | 229M  | 360M                                    | 599M  | 1B    | -     | -     |
|       | Pure MoSA | 113M                                    | 142M  | 203M  | 324M  | 559M                                    | -     | -     | -     | -     |
| Med.  | MoSA      | 210M                                    | 239M  | 310M  | 442M  | 703M                                    | 1.2B  | -     | -     | -     |
|       | Pure MoSA | 210M                                    | 267M  | 390M  | 632M  | 1.1B                                    | -     | -     | -     | -     |
| Large | MoSA      | 516M                                    | 650M  | 943M  | -     | -                                       | -     | -     | -     | -     |
|       | Pure MoSA | 516M                                    | 703M  | 1B    | -     | -                                       | -     | -     | -     | -     |
|       |           |                                         |       |       |       |                                         |       |       |       |       |
|       |           |                                         |       |       |       | Number of MoSA heads for given sparsity |       |       |       |       |
| Tiny  | MoSA      | 0                                       | 13    | 31    | 69    | 142                                     | 276   | 505   | 848   | 1277  |
|       | Pure MoSA | 0                                       | 23    | 56    | 124   | 255                                     | -     | -     | -     | -     |
| Small | MoSA      | 0                                       | 11    | 26    | 54    | 109                                     | 210   | 381   | -     | -     |
|       | Pure MoSA | 0                                       | 21    | 47    | 98    | 197                                     | -     | -     | -     | -     |
| Med.  | MoSA      | 0                                       | 11    | 26    | 54    | 109                                     | 210   | -     | -     | -     |
|       | Pure MoSA | 0                                       | 21    | 47    | 98    | 197                                     | -     | -     | -     | -     |
| Large | MoSA      | 0                                       | 27    | 60    | -     | -                                       | -     | -     | -     | -     |
|       | Pure MoSA | 0                                       | 37    | 80    | -     | -                                       | -     | -     | -     | -     |

Table 5: Detailed statistics of the main IsoFLOP experiments from Sec. [3.2.](#page-7-0) Models Tiny, Small, Medium, and Large are as described in App[.C.](#page-20-0) Sparsity 1 corresponds to dense baselines. Pure MoSA models for sparsities ≥ 1 have only MoSA heads, calculated as the biggest number of heads that will not increase the FLOP budget of the dense baseline (other hyperparameters stay the same as in the baseline). MoSA models have 4 dense heads and the rest of the heads are sparse, calculated such that the flop cost of both dense and sparse heads is lower than the baseline. Therefore, the total number of heads in hybrid models (with sparsity ≥ 1) is the number shown in the bottom table + 4. For perplexity, the best result for each row is bold.