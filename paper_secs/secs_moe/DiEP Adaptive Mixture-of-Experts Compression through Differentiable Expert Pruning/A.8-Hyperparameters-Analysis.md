# A.8 Hyperparameters Analysis

There are two key hyperparameters, including the number of epochs during differentiable search and the value of the weight λ between reconstruction regularization term and cross-entropy loss

<span id="page-15-2"></span>![](_page_15_Figure_0.jpeg)

(a) Hyperparameters analysis in terms of value of (b) Hyperparameters analysis in terms of the number weight coefficient  $\lambda$  in Eq. 7. of epochs.

Figure 8: Hyperparameters analysis in terms of the number of clients and weight coefficient  $\lambda$  on Mixtral8×7B under 25% and 50% expert sparsity.

of the overall objective function in Eq. 7. We first analyze the impact of  $\lambda$  by varying its value in  $\{5, 10, 15, 20, 30\}$ . Figure 8a demonstrates that optimal performance is achieved with  $\lambda = 0.01$  for both 25% and 50% expert sparsity. Additionally, we investigate how the number of epochs affects model performance. As shown in Figure 8b, DiEP achieves optimal results when trained for 10 epochs under both 25% and 50% expert sparsity settings.

#### <span id="page-15-1"></span>A.9 Results on More Datasets.

We provide more experimental results on more datasets including ARC-c, ARC-e[7], HellaSwag [45] and WinoGrand [36] on Mixtral 8×7B, and our DiEP is much better than NAEE across all tasks. As shown in Figure 10, these results further demonstrate the effectiveness of our proposed method.

<span id="page-15-3"></span>Table 10: Zero-shot evaluation result on more datasets, including ARC-c, ARC-e, HellaSwag, WinoGrande.

| Model        | Method     | ARC-c              | ARC-e              | HellaSwag          | WinoGrande         |
|--------------|------------|--------------------|--------------------|--------------------|--------------------|
| Mixtral 8×7B | NAEE       | 51.62 /48.89       | 81.94/78.16        | 61.60/57.66        | 75.37/72.85        |
|              | DiEP(Ours) | <b>52.54/49.26</b> | <b>83.31/82.52</b> | <b>63.22/58.96</b> | <b>76.03/73.55</b> |

