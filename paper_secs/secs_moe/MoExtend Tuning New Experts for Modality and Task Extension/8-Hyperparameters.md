# 8 Hyperparameters

<span id="page-8-0"></span>Table 6: Training hyperparameters of MoExtend.

| Hyperparameter       | Pretrain            | Fine-tune           |  |  |
|----------------------|---------------------|---------------------|--|--|
| batch size           | 256                 | 128                 |  |  |
| learning rate        | 1E-03               | 2E-05               |  |  |
| schedule             | cosine decay        | cosine decay        |  |  |
| warmup ratio         | 0.03                | 0.03                |  |  |
| weight decay         | 0                   | 0                   |  |  |
| optimizer            | AdamW               | AdamW               |  |  |
| epoch                | 1                   | 1                   |  |  |
| aux loss coefficient | 0.001               | 0.001               |  |  |
| precision            | BF16                | BF16                |  |  |
| GPU                  | $8 \times A800-80G$ | $8 \times A800-80G$ |  |  |
| text max length      | 1024                | 2048                |  |  |
| deepspeed stage      | 2                   | 3                   |  |  |

## 9 Acknowledgments

This work was supported by National Natural Science Foundation of China (No.61876045, 623B2099, U1711264). Pan Zhou acknowledges support from the Singapore Ministry of Education (MOE) Academic Research Fund (AcRF) Tier 1 grant.

