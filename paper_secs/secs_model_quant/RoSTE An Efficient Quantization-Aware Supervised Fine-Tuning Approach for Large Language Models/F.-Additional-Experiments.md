# F. Additional Experiments

We show additional experiment results for the resultant accuracies of fine-tuning the LLMs with different configurations of quantization parameters. Particularly, the results for Exp.1 on the Pythia models can be found in Table [7,](#page-17-0) and the results for Exp.2 on the Pythia models can be found in Table [9.](#page-19-1) We observe consistent improvements with the RoSTE algorithm.

<span id="page-16-0"></span>![](_page_16_Figure_1.jpeg)

Figure 7. Effects of incoherence processing using rotation matrices on different layers of Pythia and Llama models using the pre-trained weights. (Left) Relative reduction rates of quantization error, calculated as  $\frac{\text{Error w/o rotation} - \text{Error w/ rotation}}{\text{Error w/o rotation}} \times 100\%$ . Note that the reduction rate can be negative if the rotation is not beneficial. (Right) Reduction rate of dynamic ranges of the activations after rotation.

Additional comparisons to QLoRA (Dettmers et al., 2023), LLM-QAT (Liu et al., 2023) and DuQuant (Lin et al., 2024) are provided in some setups, for instance, we observed a significant performance degradation on DuQuant when the KV cache is quantized below 8 bits in Table 8.

Table 7. Additional experiments for Exp.1 with different bit-width configurations and different model sizes.

<span id="page-17-0"></span>

| Bit-width | Method         | ROUGE-1 | ROUGE-2 | ROUGE-L     | ROUGE-LSum | ROUGE (Avg.) |
|-----------|----------------|---------|---------|-------------|------------|--------------|
|           |                |         |         | Pythia-1B   |            |              |
|           | Base           | 22.40   | 5.73    | 17.35       | 17.59      | 15.77        |
| FP16      | SFT            | 32.80   | 11.84   | 25.49       | 25.50      | 23.91        |
|           | RTN            | 6.05    | 0.06    | 5.21        | 5.67       | 4.25         |
|           | GPTQ           | 10.16   | 0.30    | 8.41        | 8.84       | 6.93         |
|           | LLM-QAT        | 19.71   | 4.03    | 15.82       | 15.83      | 13.85        |
|           | QuaRot         | 16.57   | 1.66    | 13.61       | 13.70      | 11.39        |
| W4A4KV4   | SpinQuant      | 13.52   | 0.40    | 11.21       | 11.10      | 9.06         |
|           | QLoRA (r = 64) | 22.58   | 5.87    | 17.48       | 17.71      | 15.91        |
|           | STE            | 31.03   | 10.44   | 24.01       | 24.01      | 22.37        |
|           | RoSTE (ours)   | 31.80   | 11.03   | 24.71       | 24.71      | 23.07        |
|           | RTN            | 24.19   | 6.94    | 19.29       | 19.13      | 17.39        |
|           | GPTQ           | 29.77   | 9.81    | 23.38       | 23.50      | 21.52        |
|           | LLM-QAT        | 29.54   | 9.60    | 23.08       | 23.08      | 21.33        |
| W4A8KV4   | QuaRot         | 30.14   | 9.24    | 22.97       | 23.03      | 21.35        |
|           | SpinQuant      | 30.37   | 9.73    | 23.15       | 23.43      | 21.67        |
|           | STE            | 32.44   | 11.48   | 25.24       | 25.24      | 23.60        |
|           | RoSTE (ours)   | 32.67   | 11.61   | 25.37       | 25.37      | 23.76        |
|           |                |         |         | Pythia-6.9B |            |              |
|           | Base           | 28.81   | 9.45    | 22.29       | 22.91      | 20.87        |
| FP16      | SFT            | 33.69   | 12.60   | 26.27       | 26.31      | 24.72        |
|           | RTN            | 7.42    | 0.06    | 6.53        | 6.56       | 5.14         |
|           | GPTQ           | 8.16    | 0.08    | 7.06        | 7.60       | 5.73         |
|           | LLM-QAT        | 18.73   | 3.71    | 15.31       | 15.01      | 13.19        |
| W4A4KV4   | QuaRot         | 11.70   | 0.23    | 8.52        | 9.39       | 7.46         |
|           | SpinQuant      | 8.61    | 0.10    | 8.10        | 8.07       | 6.22         |
|           | QLoRA (r = 64) | 27.92   | 8.91    | 21.97       | 22.00      | 20.20        |
|           | STE            | 28.91   | 9.07    | 22.30       | 22.33      | 20.65        |
|           | RoSTE (ours)   | 32.60   | 11.54   | 25.25       | 25.25      | 23.66        |
|           | RTN            | 21.77   | 5.31    | 17.31       | 17.22      | 15.40        |
|           | GPTQ           | 32.42   | 10.71   | 24.56       | 24.59      | 23.07        |
|           | LLM-QAT        | 29.24   | 9.16    | 22.64       | 22.64      | 20.92        |
| W4A8KV4   | QuaRot         | 26.08   | 8.17    | 20.97       | 20.98      | 19.05        |
|           | SpinQuant      | 31.69   | 10.70   | 24.69       | 24.68      | 22.94        |
|           | STE            | 33.05   | 11.94   | 25.58       | 25.61      | 24.05        |
|           | RoSTE (ours)   | 33.18   | 12.05   | 25.86       | 25.88      | 24.24        |

Table 8. Additional experiments for Exp.1 with different bit-width configurations and different model sizes.

<span id="page-18-0"></span>

| Bit-width | Method         | ROUGE-1 | ROUGE-2 | ROUGE-L      | ROUGE-LSum | ROUGE (Avg.) |
|-----------|----------------|---------|---------|--------------|------------|--------------|
|           |                |         |         | Qwen2.5-0.5B |            |              |
|           | Base           | 23.79   | 6.63    | 18.46        | 18.56      | 16.86        |
| BF16      | SFT            | 32.58   | 11.93   | 25.53        | 25.55      | 23.90        |
|           | RTN            | 10.04   | 0.37    | 8.15         | 8.34       | 6.73         |
|           | GPTQ           | 12.53   | 0.92    | 10.08        | 10.50      | 8.51         |
|           | QuaRot         | 9.94    | 0.57    | 8.18         | 8.38       | 6.67         |
| W4A4KV4   | SpinQuant      | 12.16   | 1.22    | 10.69        | 10.72      | 8.70         |
|           | DuQuant        | 4.05    | 0.09    | 3.53         | 3.58       | 2.81         |
|           | QLoRA (r = 64) | 24.88   | 7.18    | 19.28        | 19.43      | 17.69        |
|           | STE            | 29.97   | 9.92    | 23.39        | 23.39      | 21.67        |
|           | RoSTE (ours)   | 30.75   | 10.44   | 23.96        | 23.96      | 22.28        |
|           | RTN            | 9.51    | 1.06    | 9.02         | 8.90       | 7.12         |
|           | GPTQ           | 9.53    | 1.04    | 8.80         | 8.73       | 7.03         |
|           | QuaRot         | 8.24    | 1.25    | 7.51         | 7.23       | 6.06         |
| W4A8KV4   | SpinQuant      | 9.10    | 1.11    | 8.31         | 8.12       | 6.66         |
|           | DuQuant        | 3.91    | 0.06    | 3.56         | 3.53       | 2.77         |
|           | STE            | 32.14   | 11.50   | 25.18        | 25.18      | 23.50        |
|           | RoSTE (ours)   | 32.31   | 11.79   | 25.37        | 25.38      | 23.71        |
| W4A4KV8   | QuaRot         | 29.34   | 9.08    | 22.21        | 22.15      | 20.70        |
|           | DuQuant        | 30.22   | 10.25   | 23.17        | 23.20      | 21.71        |
|           |                |         |         | Qwen2.5-7B   |            |              |
|           | Base           | 32.72   | 11.82   | 25.18        | 25.42      | 23.79        |
| BF16      | SFT            | 34.75   | 13.59   | 27.56        | 27.58      | 25.87        |
|           | RTN            | 1.07    | 0.00    | 1.01         | 1.01       | 0.77         |
|           | GPTQ           | 0.72    | 0.00    | 0.69         | 0.69       | 0.53         |
|           | QuaRot         | 7.21    | 0.10    | 5.93         | 5.93       | 4.79         |
| W4A4KV4   | SpinQuant      | 6.87    | 0.29    | 5.97         | 6.12       | 4.81         |
|           | DuQuant        | 0.00    | 0.00    | 0.00         | 0.00       | 0.00         |
|           | QLoRA (r = 64) | 32.22   | 11.41   | 24.75        | 24.89      | 23.32        |
|           | STE            | 30.86   | 10.16   | 23.73        | 23.73      | 22.12        |
|           | RoSTE (ours)   | 34.01   | 12.89   | 26.74        | 26.74      | 25.10        |
|           | RTN            | 5.73    | 0.23    | 4.72         | 4.74       | 3.86         |
|           | GPTQ           | 7.48    | 0.27    | 6.22         | 6.36       | 5.08         |
|           | QuaRot         | 5.62    | 0.15    | 5.08         | 5.14       | 3.99         |
| W4A8KV4   | SpinQuant      | 0.64    | 0.30    | 5.64         | 5.81       | 4.54         |
|           | DuQuant        | 0.24    | 0.00    | 0.24         | 0.24       | 0.18         |
|           | STE            | 34.44   | 13.29   | 27.16        | 27.17      | 25.52        |
|           | RoSTE (ours)   | 34.58   | 13.46   | 27.34        | 27.35      | 25.68        |
|           | QuaRot         | 31.96   | 10.98   | 24.73        | 24.88      | 23.13        |
| W4A4KV8   | DuQuant        | 33.47   | 12.13   | 25.28        | 25.30      | 24.05        |

Table 9. Additional experiments for Exp.2 on different bit-width configurations.

<span id="page-19-1"></span>

| Bit-width | Method       | TruthfulQA | MMLU-Pro | BigBenchHard | AGIEval | GSM8K | Math  | Avg.  |
|-----------|--------------|------------|----------|--------------|---------|-------|-------|-------|
|           | Base         | 28.51      | 19.57    | 62.26        | 30.16   | 56.86 | 18.20 | 35.93 |
| FP16      | SFT          | 31.82      | 33.07    | 65.67        | 34.86   | 64.89 | 22.66 | 42.16 |
|           | RTN          | 23.01      | 0        | 0            | 17.03   | 1.03  | 0     | 6.85  |
|           | GPTQ         | 25.34      | 0.02     | 2.55         | 16.48   | 2.05  | 0     | 7.74  |
|           | QuaRot       | 27.66      | 21.53    | 47.69        | 29.05   | 37.91 | 6.90  | 28.46 |
| W4A4KV4   | SpinQuant    | 26.19      | 21.58    | 49.56        | 28.50   | 38.36 | 10.56 | 29.13 |
|           | STE          | 26.68      | 9.13     | 24.58        | 17.63   | 22.82 | 1.90  | 17.14 |
|           | RoSTE (ours) | 26.44      | 25.12    | 52.00        | 30.11   | 44.50 | 11.94 | 31.69 |
|           | RTN          | 28.76      | 19.29    | 42.96        | 27.75   | 28.66 | 7.84  | 25.88 |
|           | GPTQ         | 28.52      | 25.54    | 46.38        | 29.26   | 48.60 | 0.02  | 29.72 |
|           | QuaRot       | 27.42      | 26.78    | 53.79        | 32.01   | 49.20 | 12.72 | 33.65 |
| W4A8KV4   | SpinQuant    | 28.15      | 26.66    | 55.74        | 32.01   | 52.16 | 15.38 | 35.02 |
|           | STE          | 29.62      | 24.09    | 54.62        | 29.44   | 52.62 | 4.08  | 32.41 |
|           | RoSTE (ours) | 30.84      | 28.23    | 59.25        | 34.03   | 56.94 | 16.88 | 37.70 |

