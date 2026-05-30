# <span id="page-15-1"></span>A WEIGHT-ONLY QUANTIZATION RESULTS

The results for weight-only quantization are provided in Table [2.](#page-15-0) One can observe that similary to the weight and activation quantization case INT4 and NVFP4 perform similarly, while MXFP suffers much significant accuracy drop. Even for weight-only case there is 2% accuracy drop on average relative to the original model.

<span id="page-15-0"></span>

| Format | Quantization | MMLU  | GSM8k | HellaSwag | WinoGrande | Avg.  | Recovery% |
|--------|--------------|-------|-------|-----------|------------|-------|-----------|
| FP16   | -            | 72.80 | 85.10 | 80.00     | 78.90      |       | –         |
|        | RTN          | 69.38 | 81.80 | 79.41     | 77.90      | 77.12 | 97.71     |
| INT4   | RTN+Had      | 70.27 | 82.56 | 79.18     | 76.64      | 77.16 | 97.76     |
|        | GPTQ         | 70.25 | 80.52 | 79.01     | 76.64      | 76.60 | 97.05     |
|        | RTN          | 70.64 | 82.26 | 79.24     | 77.35      | 77.37 | 98.02     |
|        | RTN+Had      | 69.26 | 80.82 | 78.52     | 77.03      | 76.41 | 96.80     |
| NVFP4  | GPTQ         | 70.52 | 82.49 | 79.35     | 76.95      | 77.33 | 97.96     |
|        | AWQ          | 70.57 | 82.71 | 79.30     | 77.03      | 77.40 | 98.06     |
|        | RTN          | 68.23 | 80.36 | 77.26     | 75.93      | 75.44 | 95.58     |
|        | RTN+Had      | 66.24 | 77.56 | 77.34     | 74.11      | 73.81 | 93.51     |
| MXFP   | GPTQ         | 68.79 | 81.43 | 78.40     | 76.88      | 76.37 | 96.76     |
|        | AWQ          | 68.16 | 78.70 | 78.56     | 75.30      | 75.18 | 95.25     |

Table 2: Performance of Llama-3.1-8B-Instruct under different weight-only quantization settings.

