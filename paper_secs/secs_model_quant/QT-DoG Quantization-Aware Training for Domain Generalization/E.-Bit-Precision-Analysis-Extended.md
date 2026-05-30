# E. Bit Precision Analysis Extended

<span id="page-14-2"></span>In contrast to main manuscript, Table [9](#page-14-2) provides all the results in a tabular form. We show how quantization outperforms the vanilla ERM approach. This shows the superior performance of quantization over ERM despite being more than 6 times smaller in the case of 5 bit-precision.

|                |             | PACS       |            | TerraInc   |            |  |
|----------------|-------------|------------|------------|------------|------------|--|
| Algorithm      | Compression | In-domain  | Out-domain | In-domain  | Out-domain |  |
| ERM (our runs) | -           | 96.9 ± 0.1 | 84.7 ± 0.5 | 91.7 ± 0.2 | 47.2 ± 0.4 |  |
| QT-DoG(8)      | 4x          | 97.0 ± 0.1 | 85.0 ± 0.1 | 90.9 ± 0.2 | 49.1 ± 0.1 |  |
| QT-DoG(7)      | 4.6x        | 97.3 ± 0.2 | 87.8 ± 0.3 | 92.3 ± 0.2 | 50.8± 0.2  |  |
| QT-DoG(6)      | 5.3x        | 97.1 ± 0.1 | 86.5 ± 0.1 | 91.1 ± 0.0 | 49.0 ± 0.3 |  |
| QT-DoG(5)      | 6.4x        | 97.0 ± 0.1 | 85.3 ± 0.4 | 91.0 ± 0.1 | 48.4 ± 0.2 |  |

Table 9. Model quantization with different bit-precisions vs vanilla ERM. We report the average target domain accuracy as well as the average source domain accuracy across all domains for the PACS [\(Li et al.,](#page-10-27) [2017\)](#page-10-27) and TerraIncognita [\(Beery et al.,](#page-9-22) [2018\)](#page-9-22) datasets. Quantization not only enhances the generalization ability but also retains the source domain performance. QT-DoG(x) indicates a model quantized with x bit-precision.

However, as shown in Table [10,](#page-15-0) decreasing bit-precision through quantization does not always improve performance above the baseline; after a point, there is a tradeoff between compression and generalization. Specifically, our experiments with 4-bit precision and lower did not yield satisfactory results. Finding the sweet spot for balancing speed and performance can <span id="page-15-0"></span>be an interesting research direction. Our results evidence that there exist configurations that can improve both speed and performance.

| Algorithm | Bit-Precision | PACS       |
|-----------|---------------|------------|
| ERM       | 32            | 84.7 ± 0.5 |
|           | 7             | 87.8 ± 0.3 |
|           | 6             | 86.5 ± 0.1 |
|           | 5             | 85.3 ± 0.4 |
| QT-DoG    | 4             | 84.3 ± 0.3 |
|           | 3             | 83.3 ± 0.4 |
|           | 2             | 82.8 ± 0.2 |

Table 10. Effect of aggressive quantization. Performance comparison between ERM and QT-DoG with varying bit-precision on PACS.

