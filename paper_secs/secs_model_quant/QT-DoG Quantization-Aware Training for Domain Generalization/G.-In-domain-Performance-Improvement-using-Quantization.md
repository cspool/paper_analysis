# G. In-domain Performance Improvement using Quantization

We further study the in-domain test accuracy of our quantization approach without ensembling on PACS and TerraIncognita datasets. As [\(Cha et al.,](#page-9-5) [2021\)](#page-9-5), we split the in-domain datasets into training (60%), validation (20%), and test (20%) sets. We choose the best model based on the validation set and report the results on the test set in Table [12.](#page-16-0) The results with quantization correspond to 7 bit-precision.

QT-DoG also enhances the in-domain performance. The regularization effect introduced by quantization prevents the model from overfitting to edge cases and pushes it to learn more meaningful and generalizable features, which we also demonstrate in Section [4.2.](#page-8-2) As the training data consists of various domains and the quantization limits the range of weight values, it discourages the model from becoming overly complex and overfitting to the noise in the training data. Therefore, the model is more robust to minor input fluctuations.

| Method | PACS       | TerraInc   | Compression |
|--------|------------|------------|-------------|
| ERM    | 96.6 ± 0.2 | 90.1 ± 0.2 | -           |
| SAM    | 97.3 ± 0.1 | 90.8 ± 0.1 | -           |
| SWA    | 97.1 ± 0.1 | 90.7 ± 0.1 | -           |
| SMA    | 96.8 ± 0.2 | 90.7 ± 0.4 | -           |
| SWAD   | 97.7 ± 0.2 | 90.8 ± 0.3 | -           |
| QT-DoG | 97.3 ± 0.2 | 91.1 ± 0.2 | ∼4.6x       |

<span id="page-16-0"></span>Table 12. Comparison between generalization methods on PACS and TerraInc for IID settings. We report the accuracy averaged across all domains. Our proposed approach is shaded in Gray. Highest accuracy is shown in bold, while second best is underlined.

