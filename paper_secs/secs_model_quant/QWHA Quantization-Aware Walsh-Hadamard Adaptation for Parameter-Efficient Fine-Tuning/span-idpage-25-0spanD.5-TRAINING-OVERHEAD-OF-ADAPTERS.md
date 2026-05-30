# <span id="page-25-0"></span>D.5 TRAINING OVERHEAD OF ADAPTERS

Train time We compare the training time of adapters using single-transform and two-transform designs on both WHT and conventional transform kernels, such as the DCT used in LoCA and the DHT used in SSH, in Table [15.](#page-25-1) While employing WHA reduces training time, applying it with a single transform further decreases computation. The impact of a single transform is especially evident in DCT and DHT, where training time is substantially reduced since their computational overhead due to the transform is larger than that of WHT. Note that DCT and DHT have identical training times, as their computational cost is the same and differs only in the element values within the transform kernel. Our proposed WHA employs a 1D WHT in the context of quantization, whereas conventional FT-based PEFT methods such as LoCA and SSH use 2D DCT and 2D DHT, respectively.

<span id="page-25-1"></span>Table 15: Training time (hours) of FT-based adapters with different transform kernels on LLaMA-3.1-8B with the Alpaca dataset.

| Batch Size | WHT  |      | DCT / DHT |      |
|------------|------|------|-----------|------|
|            | 1D   | 2D   | 1D        | 2D   |
| 1          | 18.2 | 25.3 | 46.2      | 63.3 |
| 2          | 9.7  | 13.1 | 32.1      | 45.8 |
| 4          | 6.0  | 8.0  | 17.4      | 26.1 |
| 8          | 4.6  | 5.5  | 9.0       | 13.3 |
| 16         | 3.9  | 4.3  | 6.7       | 8.3  |

Memory Usage We report the memory usage of each method under the same experimental setting as in Section [4,](#page-6-2) using NVIDIA A100 80GB GPU. As shown in Table [16,](#page-25-2) QWHA shows memory usage comparable to LoRA. SSH also exhibits similar memory usage as QWHA, since the only difference between them is the computation with a pre-defined transform kernel matrix. Since this matrix is shared across layers, the memory overhead is negligible. In contrast, LoCA incurs additional memory consumption due to the training of location parameters, resulting in a few gigabytes of overhead depending on the batch size.

<span id="page-25-2"></span>Table 16: GPU memory usage (GB) during fine-tuning on LLaMA-3.1-8B with 4-bit quantization using the Alpaca dataset. All adapters use the same number of trainable parameters with P(r = 64).

| Batch Size | CLoQ | QWHA | LoCA |
|------------|------|------|------|
| 1          | 22.1 | 22.1 | 23.3 |
| 2          | 26.6 | 27.2 | 28.4 |
| 4          | 32.6 | 33.2 | 34.4 |
| 8          | 44.7 | 45.3 | 46.4 |
| 16         | 68.8 | 69.4 | 70.6 |

