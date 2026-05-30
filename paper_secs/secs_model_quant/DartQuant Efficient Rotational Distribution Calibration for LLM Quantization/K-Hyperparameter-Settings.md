# K Hyperparameter Settings

The specific hyperparameter settings for DartQuant are shown in Table [23.](#page-27-0) It is important to note that the latent parameter Z<sup>0</sup> is initialized using a random Hadamard matrix.

| Rotation | Model | LR       | Epoch | Optimizer | BS |
|----------|-------|----------|-------|-----------|----|
|          | 2-7b  | 2.00E-03 | 10    | SGD       | 64 |
|          | 2-13b | 1.00E-02 | 10    | SGD       | 64 |
| R1       | 2-70b | 1.00E-03 | 10    | SGD       | 64 |
|          | 3-8b  | 8.00E-03 | 10    | SGD       | 64 |
|          | 3-70b | 3.00E-03 | 10    | SGD       | 64 |

R2 All 1.00E-03 10 SGD 64

<span id="page-27-0"></span>Table 23: Comparison of zero-shot accuracy and perplexity across different loss functions.

