# <span id="page-17-0"></span>A Model Compression Ratio and Memory Footprint

We report the compression ratio after applying LoftQ in Table [7.](#page-17-1) It is defined as

$$compression \ ration = \frac{backbone \ size + LoRA \ adapter \ size}{pre\text{-trained size}}.$$

We also measure the GPU memory cost during training. Given that GPU memory varies by models, tasks, sequence lengths, batch sizes, etc. We report LLAMA-2 on GSM8K as an example in Table [8.](#page-17-2)

Table 7: Compression ratios of backbones.

<span id="page-17-1"></span>

| Model          | Compression<br>ratio (%) | Trainable<br>ratio (%) | Rank | Bits | Quantization<br>method |
|----------------|--------------------------|------------------------|------|------|------------------------|
| DeBERTaV3-base | 15.6                     | 3.1                    | 16   | 2    | Uniform                |
| DeBERTaV3-base | 18.8                     | 6.3                    | 32   | 2    | Uniform                |
| DeBERTaV3-base | 17.2                     | 3.1                    | 16   | 2    | NF2                    |
| DeBERTaV3-base | 20.4                     | 6.3                    | 32   | 2    | NF2                    |
| BART-large     | 15.3                     | 1.2                    | 8    | 4    | NF2                    |
| BART-large     | 16.7                     | 2.5                    | 16   | 4    | NF2                    |
| BART-large     | 27.8                     | 1.2                    | 8    | 4    | NF4                    |
| BART-large     | 29.0                     | 2.5                    | 16   | 4    | NF4                    |
| BART-large     | 26.2                     | 1.2                    | 8    | 4    | Uniform                |
| BART-large     | 27.5                     | 2.5                    | 16   | 4    | Uniform                |
| LLAMA-2-7b     | 16.6                     | 2.4                    | 64   | 2    | Nf2                    |
| LLAMA-2-7b     | 29.0                     | 2.4                    | 64   | 4    | Nf4                    |
| LLAMA-2-13b    | 16.0                     | 1.9                    | 64   | 2    | Nf2                    |
| LLAMA-2-13b    | 28.5                     | 1.9                    | 64   | 4    | Nf4                    |

Table 8: GPU memory footprint

<span id="page-17-2"></span>

| Model       | Dataset | Seq length | Batch size | GPU Mem |
|-------------|---------|------------|------------|---------|
| LLAMA-2-7b  | GSM8K   | 384        | 1          | 15GB    |
| LLAMA-2-13b | GSM8K   | 384        | 1          | 24GB    |

