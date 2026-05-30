# A Sparseness Adjustment

Although our method does not explicitly specify the activation quantity (sparsity degree) of the MoE model, the sparsity of DSMoE can be adjusted by modulating the hyperparameter τ . The specific regulatory effects are shown in the table [4.](#page-10-15)

<span id="page-10-15"></span>Table 4: DSMoE LLaMA Models: Threshold (τ ) vs. Performance and Parameter Activation

| τ   |      | LLaMA-7B         | LLaMA-1B |                  |  |  |
|-----|------|------------------|----------|------------------|--|--|
|     | PPL  | activated params | PPL      | activated params |  |  |
| 0.2 | 3.82 | 65.45%           | 7.22     | 64.19%           |  |  |
| 0.3 | 3.83 | 62.70%           | 7.24     | 62.32%           |  |  |
| 0.4 | 3.85 | 60.43%           | 7.29     | 60.79%           |  |  |
| 0.5 | 3.91 | 58.46%           | 7.41     | 59.35%           |  |  |
| 0.6 | 4.02 | 56.54%           | 7.61     | 57.87%           |  |  |
| 0.7 | 4.28 | 54.77%           | 8.01     | 56.34%           |  |  |
| 0.8 | 5.09 | 52.54%           | 8.85     | 54.37%           |  |  |

The results demonstrate that as τ increases from 0.2 to 0.8, perplexity gradually increases while the percentage of activated parameters decreases, which aligns with intuitive expectations. Performance degradation is relatively modest in the range of τ=0.2 to τ=0.5, but becomes more pronounced beyond τ=0.5.

We selected τ=0.5 as the default value for our main experiments because it offers an optimal balance between model performance and computational efficiency. In practical applications, τ can function as an adjustable parameter that users can tune according to their specific computational resource constraints and performance requirements.

