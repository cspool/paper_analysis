# <span id="page-10-0"></span>A Data and Hyperparameters

Training Data RWKV-X training is divided into two phases. The first phase, the Alignment Phase, uses the minipile dataset with 1.5 billion tokens. The second phase, the Long Context Phase, draws randomly sampled data from the ProLong-64K dataset with a total of 40 billion tokens.

Hyperparameters The following hyperparameters were used to train a range of RWKV-X models, from 0.22B to 3.6B parameters, as shown in Table [7.](#page-10-1)

<span id="page-10-1"></span>

| Hyperparameter          |           | 0.22B Model  | 3.6B Model |              |  |
|-------------------------|-----------|--------------|------------|--------------|--|
|                         | Alignment | Long Context | Alignment  | Long Context |  |
| Batch size (tokens)     | -         | 8.192M       | 4.096M     | 1.024M       |  |
| Context length (tokens) | -         | 64,000       | 4,096      | 64,000       |  |
| Tokens trained (B)      | -         | 20           | 1.5        | 1            |  |
| Initial learning rate   | -         | 1e-5         | 1e-5       | 1e-5         |  |
| Final learning rate     | -         | 1e-5         | 1e-5       | 1e-5         |  |
| Learning rate schedule  | -         | Constant     | Constant   | Constant     |  |
| Warmup ratio            | -         | 0            | 0          | 0            |  |
| Weight decay            | -         | 0            | 0          | 0            |  |
| Optimizer               | -         | AdamW        | AdamW      | AdamW        |  |
| DeepSpeed stage         | -         | 1            | 1          | 1            |  |
| GPU Configuration       | -         | 8×H20        | 4×H20      | 8×H200       |  |
| Total GPU Hours (h)     | -         | 576          | 6          | 80           |  |

Table 7: Training hyperparameters and compute configurations for RWKV-X models.

