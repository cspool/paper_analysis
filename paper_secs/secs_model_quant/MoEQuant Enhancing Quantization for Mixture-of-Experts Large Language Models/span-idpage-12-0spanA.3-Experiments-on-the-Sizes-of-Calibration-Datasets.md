# <span id="page-12-0"></span>A.3 Experiments on the Sizes of Calibration Datasets

We prune the Mixtral 8x7b model with different sizes of calibration datasets. To be specific, we randomly sample 1, 2, 4, 16, 64, and 256 sequences (each composed of 2048 tokens) from the C4 dataset to form calibration datasets. Then the model is pruned to r=6 and evaluated on various benchmarks. The average LM-eval results are reported in Tab. 7.

<span id="page-12-2"></span>

| Number of Sequence | LM-eval |
|--------------------|---------|
| 1                  | 62.63   |
| 2                  | 63.93   |
| 4                  | 63.53   |
| 16                 | 63.59   |
| 64                 | 64.32   |
| 128                | 64.22   |
| 256                | 63.94   |

Table 7: Performances of expert pruning with different sizes of calibration datasets.

As can be seen, using 64 and 128 sequences can result in the highest overall results ( $\geq$  64). Using a small set of sequences will possibly lead to performance degradation (especially for using just 1 sequence), but our method is somewhat robust to the size of datasets, as seen from the table.

