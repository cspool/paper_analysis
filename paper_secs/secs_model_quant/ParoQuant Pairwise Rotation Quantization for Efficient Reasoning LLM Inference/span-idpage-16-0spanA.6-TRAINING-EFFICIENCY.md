# <span id="page-16-0"></span>A.6 TRAINING EFFICIENCY

Table [A5](#page-16-4) shows the calibration size and GPU time for quantizing LLaMA-3-8B on an NVIDIA H200 GPU. Although ParoQuant is slower than EfficientQAT due to an extra tuning stage and the additional computation graph nodes from independent rotations, it is significantly faster than QTIP, which requires significantly more calibration data and is slowed down by two extra steps in addition to layer-wise fine-tuning: generating Hessian matrices and end-to-end fine-tuning.

<span id="page-16-4"></span>Table A5: Calibration data (# samples × sequence length) and GPU time for quantizing LLaMA-3-8B on an NVIDIA H200 GPU.

|                  | AWQ       | E-QAT       | QTIP        | PAROQ       |
|------------------|-----------|-------------|-------------|-------------|
| Calibration Data | 128 × 512 | 4096 × 2048 | 4096 × 8192 | 2048 × 2048 |
| GPU Time         | minutes   | ≈ 3 hours   | ≈ 20 hours  | ≈ 9 hours   |

