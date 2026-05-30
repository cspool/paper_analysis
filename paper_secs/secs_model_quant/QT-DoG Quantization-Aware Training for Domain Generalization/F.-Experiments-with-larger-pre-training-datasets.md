# F. Experiments with larger pre-training datasets

We also show experimental results with ResNeXt-50-32x4 in Table [11.](#page-15-1) Note that both ResNet-50 and ResNeXt-50-32x4d have 25M parameters. However, ResNeXt-50-32x4d is pre-trained on a larger dataset i.e Instagram 1B images[\(Yalniz et al.,](#page-12-21) [2019\)](#page-12-21). It is evident from Table [11](#page-15-1) that incorporating quantization into training consistenlty improve accuracy even when a network is pre-trained on a larger dataset. Furthermore, EoQ again showed superior performance in comparison to other methods across five DomainBed datasets.

<span id="page-15-1"></span>

| Algorithm      | M | S    | PACS                                                     | VLCS | Office                                                 | TerraInc | DomainNet | Avg. |
|----------------|---|------|----------------------------------------------------------|------|--------------------------------------------------------|----------|-----------|------|
|                |   |      | ResNeXt-50 32x4d (25M Parameters, Pre-trained 1B Images) |      |                                                        |          |           |      |
| ERM            | 1 | 1x   |                                                          |      | 88.7 ± 0.3 79.0 ± 0.1 70.9 ± 0.5 51.4 ± 1.2 48.1 ± 0.2 |          |           | 67.7 |
| SMA            | 1 | 1x   |                                                          |      | 92.7 ± 0.3 79.7 ± 0.3 78.6 ± 0.1 53.3 ± 0.1 53.5 ± 0.1 |          |           | 71.6 |
| QT-DoG (ours)  | 1 | 1x   |                                                          |      | 92.9 ± 0.3 79.2 ± 0.4 78.9 ± 0.3 54.1 ± 0.2 53.9 ± 0.2 |          |           | 71.8 |
| ERM Ens.†      | 6 | 6x   | 91.2                                                     | 80.3 | 77.8                                                   | 53.5     | 52.8      | 71.1 |
| EoA†           | 6 | 6x   | 93.2                                                     | 80.4 | 80.2                                                   | 55.2     | 54.6      | 72.7 |
| EoQ†<br>(ours) | 5 | 1.1x | 93.5                                                     | 80.3 | 80.3                                                   | 55.6     | 54.8      | 72.9 |

Table 11. Comparison with other methods for ResNeXt-50. Performance benchmarking on 5 datasets of the DomainBed benchmark. Highest accuracy is shown in bold, while second best is underlined. Ensembles† do not have confidence interval because an ensemble uses all the models to make a prediction. Our proposed method is colored in Gray. Average accuracies and standard errors are reported from three trials. For all the reported results, we use the same training-domain validation protocol as [\(Gulrajani & Lopez-Paz,](#page-10-10) [2021\)](#page-10-10). M corresponds to the number of models trained during training and S corresponds to the relative network size.

