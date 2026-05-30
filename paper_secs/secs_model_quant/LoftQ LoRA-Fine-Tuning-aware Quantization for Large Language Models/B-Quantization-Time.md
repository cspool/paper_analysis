# B Quantization Time

We report the execution time of LoftQ applying to a single weight matrix in Table [9.](#page-18-1) The time is tested on Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz.

Table 9: Execution time of LoftQ applying to different weight matrices.

<span id="page-18-1"></span>

| Model          | Size              | Step<br>T | Quantization method | Time |
|----------------|-------------------|-----------|---------------------|------|
| DeBERTaV3-base | ×<br>768<br>768   | 5         | Uniform             | 1s   |
| BART-large     | ×<br>1024<br>1024 | 5         | NF4                 | 1s   |
| LLAMA-2-7b     | ×<br>4096<br>4096 | 5         | NF4                 | 21s  |
| LLAMA-2-13b    | ×<br>5120<br>5120 | 5         | NF4                 | 43s  |

