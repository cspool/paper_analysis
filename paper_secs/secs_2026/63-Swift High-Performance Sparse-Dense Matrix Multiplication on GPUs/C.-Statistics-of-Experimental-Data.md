# *C. Statistics of Experimental Data*

We analyze the distribution of the performance rankings of the five considered methods across different configurations. Table III summarizes the distribution of the performance rankings of these methods for all the 2757 considered matrices. The performance of the five methods on each matrix is ranked, with the best performer being labeled best, and so on, and the worst performer labeled worst. For example, in the 3rd column, the value for each method represents the percentage of matrices where that method achieves the best performance. Our evaluation reveals that Swift achieves the best performance on most matrices across the first 3 configurations. In particular, for FP64 and *N* = 32, Swift achieves the best performance in 91.22% of the matrices. However, under the configuration of FP32 and N=128, the performance of Swift does not perform as well as ASpT, Swift only achieves the best performance in 36.67% of the matrices but ASpT achieves 55.14%. Still, Swift obtains a 1.02→ with respect to ASpT, as Table I reports. In each row, the percentages indicate the distribution of the performance ranking of a method. Swift can achieve the best and 2nd best performance in all configurations. Even for the FP32 and *N* = 128 configuration, Swift can achieve the best or second-best performance for more than 80% of the matrices.

TABLE II: Performance Metrics for Swift. Sputnik data appear inside parentheses.

| Matrix         | NNZ      | Memory<br>Bandwidth<br>Utilization(%) | Memory<br>Coalescing(%) | L2<br>Hit Rate(%) | SM Occupancy(%) |
|----------------|----------|---------------------------------------|-------------------------|-------------------|-----------------|
| gridgena       | 512084   | 90.65 (48.31)                         | 70 (62)                 | 94.27 (85.68)     | 93.52 (16.03)   |
| 2D 54019 highK | 996414   | 91.95 (31.38)                         | 77 (77)                 | 97 (86.69)        | 90.97 (16.14)   |
| bcsstk35       | 1450163  | 93.68 (28.68)                         | 85 (66)                 | 93.68 (93.75)     | 92.50 (16.09)   |
| matrix 9       | 2121550  | 91.42 (33.70)                         | 82 (82)                 | 96.05 (85.82)     | 85.93 (16.24)   |
| matrix-new 3   | 2678750  | 88.73 (28.27)                         | 76 (76)                 | 94.87 (84.34)     | 85.88 (15.93)   |
| srb1           | 2962152  | 83.58 (27.54)                         | 98 (79)                 | 98.19 (86.91)     | 81.31 (16.45)   |
| pkustk03       | 3130416  | 87.73 (27.64)                         | 97 (70)                 | 98.16 (89.06)     | 86.77 (16.46)   |
| oilpan         | 3597188  | 90.74 (25.97)                         | 94 (52)                 | 97.94 (90.38)     | 84.40 (15.77)   |
| s3dkt3m2       | 3753461  | 89.48 (27)                            | 93 (78)                 | 97.86 (87.27)     | 85.04 (16.33)   |
| engine         | 4706073  | 87.17 (46.31)                         | 70 (63)                 | 94.62 (82.09)     | 88.53 (16.33)   |
| s3dkq4m2       | 4820891  | 88.29 (27.18)                         | 93 (77)                 | 98.13 (89.63)     | 82.20 (16.36)   |
| pkustk11       | 5217912  | 88.11 (29.93)                         | 94 (79)                 | 98.21 (87.82)     | 84.40 (16.44)   |
| shipsec8       | 6653399  | 91.53 (30.44)                         | 89 (79)                 | 97.76 (87.82)     | 86.97 (16.34)   |
| boneS01        | 6715152  | 93.34 (42.03)                         | 70 (63)                 | 97.14 (85.97)     | 87.17 (16.39)   |
| bmw7st 1       | 7339667  | 87.41 (30.95)                         | 84 (65)                 | 94.36 (86.88)     | 91.43 (16.32)   |
| shipsec1       | 7813404  | 89.57 (32.43)                         | 91 (71)                 | 95.84 (87.94)     | 81.83 (16.53)   |
| ship 003       | 8086034  | 91.74 (34.13)                         | 89 (71)                 | 97.57 (88.03)     | 87.12 (16.51)   |
| m t1           | 9753570  | 86.23 (29.69)                         | 88 (65)                 | 98.20 (91.95)     | 82.47 (16.45)   |
| shipsec5       | 10113096 | 70.20 (30.11)                         | 90 (69)                 | 87.71 (87.2)      | 86.58 (16.43)   |
| x104           | 10167624 | 88.82 (29.79)                         | 91 (65)                 | 97.77 (89.85)     | 80.65 (16.47)   |
| hood           | 10768436 | 84.44 (32.63)                         | 90 (53)                 | 95.36 (82.05)     | 83.07 (16.06)   |
| fcondp2        | 11294316 | 85.85 (27.24)                         | 95 (68)                 | 95.28 (87.46)     | 79.97 (16.47)   |
| fullb          | 11708077 | 78.42 (29.43)                         | 91 (70)                 | 91.01 (85.95)     | 85.95 (16.52)   |
| troll          | 11985111 | 81.02 (29.83)                         | 83 (64)                 | 91.34 (86.70)     | 83.85 (16.34)   |
| BenElechi1     | 13150496 | 81.87 (27.52)                         | 95 (69)                 | 91.38 (88.93)     | 80.39 (16.54)   |
| af 0 k101      | 17550675 | 87.97 (26.59)                         | 92 (52)                 | 92.97 (87.75)     | 81.18 (16.55)   |
| af shell1      | 17588875 | 88.25 (26.94)                         | 91 (52)                 | 94.22 (87.44)     | 80.32 (16.55)   |
| CoupCons3D     | 22322336 | 80.09 (51.65)                         | 83 (82)                 | 91.08 (78.56)     | 84.02 (16.56)   |
| ML Laplace     | 27689972 | 88.65 (33.51)                         | 75 (75)                 | 91.79 (92.61)     | 84.04 (16.59)   |
| inline 1       | 36816342 | 84.44 (67.96)                         | 90 (54)                 | 95.36 (72.64)     | 83.07 (16.51)   |

![](_page_8_Figure_12.jpeg)

Fig. 13: Relationship between the average speedup of Swift over ASpT, Sputnik, RoDe, and cuSPARSE and the distribution of nonzero elements (N=128).

## *D. Analysis of the Best and Worst Performing Matrices*

We further investigate the impact of the matrix sparse patterns on the performance of Swift. In general, Swift performs its very best when the non-zero elements are evenly distributed. Figure 12a shows the matrix *Journals* where

![](_page_9_Figure_0.jpeg)

Fig. 14: Speedup of Swift over the four SOTA methods on RTX 4080s (FP64).

TABLE III: Ranking distribution of performance (%).

| Configuration | Methods  | Best  | 2nd best | 3rd best | 4th best | Worst |
|---------------|----------|-------|----------|----------|----------|-------|
|               | Swift    | 91.22 | 2.94     | 3.42     | 2.13     | 0.29  |
|               | ASpT     | 5.84  | 88.57    | 3.45     | 2.13     | 0.01  |
| FP64, N=32    | cuSPARSE | 0.00  | 0.00     | 0.66     | 3.27     | 96.07 |
|               | RoDe     | 2.90  | 7.09     | 84.9     | 5.07     | 0.04  |
|               | Sputnik  | 0.04  | 1.40     | 7.57     | 87.40    | 3.59  |
|               | Swift    | 65.72 | 20.92    | 3.84     | 4.61     | 4.91  |
|               | ASpT     | 27.71 | 63.21    | 5.79     | 3.14     | 0.15  |
| FP64, N=128   | cuSPARSE | 0.00  | 0.44     | 3.10     | 8.38     | 88.08 |
|               | RoDe     | 5.35  | 9.30     | 74.72    | 9.30     | 1.33  |
|               | Sputnik  | 1.22  | 6.13     | 12.55    | 74.57    | 5.53  |
|               | Swift    | 62.92 | 15.11    | 8.25     | 12.80    | 0.92  |
|               | ASpT     | 22.15 | 69.2     | 6.64     | 1.94     | 0.07  |
| FP32, N=32    | cuSPARSE | 0.00  | 0.00     | 1.32     | 2.16     | 96.52 |
|               | RoDe     | 12.69 | 11.44    | 75.61    | 0.26     | 0.00  |
|               | Sputnik  | 2.24  | 4.25     | 8.18     | 82.84    | 2.49  |
|               | Swift    | 36.67 | 45.20    | 9.09     | 7.24     | 1.80  |
|               | ASpT     | 55.14 | 38.96    | 3.90     | 2.00     | 0.00  |
| FP32, N=128   | cuSPARSE | 0.00  | 0.00     | 0.90     | 2.20     | 96.90 |
|               | RoDe     | 7.59  | 12.84    | 78.97    | 0.60     | 0.00  |
|               | Sputnik  | 0.60  | 3.01     | 7.14     | 87.96    | 1.30  |

Swift achieves the best performance, characterized by an even distribution of non-zero elements. This pattern's advantage is that after Swift's sorting and blocking, the elements handled by a warp at a time are distributed across different rows, reducing the impact of atomic operations. In contrast, Figure 12b displays the matrix *cegb2919* with non-zero elements concentrated along the diagonal, and this concentration increases the impact of atomic operations, thereby weakening the performance improvement of Swift.

*1) Analysis Considering the Distribution of Nonzero Elements:* To describe the relationship between the distribution of nonzero elements in sparse matrices and the performance of Swift, all input sparse matrices are partitioned into 32!32 blocks. The ratio of the all-zero-blocks to the total number of blocks is as used as the x-axis of Figure 13, and the average speedup of Swift relative to all other baselines in the y-axis. The higher ratio, the more blocks are empty, implying that the nonzero elements are concentrated within fewer blocks. Conversely, a low ratio implies fewer empty blocks, meaning that the nonzero elements are more evenly distributed across the matrix. Figure 13, indicates that as the nonzero elements of the sparse matrix become are less evenly distributed, the average speedup of Swift over ASpT, Sputnik, RoDe, and cuSPARSE decreases. Therefore, Swift achieves better performance improvements when the nonzero elements are evenly distributed, while its performance slightly declines when the nonzero elements are irregularly distributed.

