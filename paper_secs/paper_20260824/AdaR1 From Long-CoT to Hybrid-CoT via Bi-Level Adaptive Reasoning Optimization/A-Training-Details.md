# A Training Details

For both models, we selected 2,500 problems from the mixed Mathematics as training data. For each problem, we sample 12 times. From each set of solutions, we randomly selected 2 solutions for training. After computing the rewards, we normalized the reward values. Both models are trained with 8 \* A800-80G GPUs. The other hyperparameters used in the training process are presented in the table below.

Table 4: Hyperparameters for the Deepseek-Distill-1.5B and Deepseek-Distill-7B.

| Hyperparameter    | Deepseek-Distill-1.5B | Deepseek-Distill-7B. |
|-------------------|-----------------------|----------------------|
| cutoff_len        | 4096                  | 4096                 |
| batch_size        | 32                    | 32                   |
| learning_rate     | 5.0e-7                | 5.0e-7               |
| num_train_epochs  | 2.0                   | 2.0                  |
| lr_scheduler_type | constant              | constant             |
| M1                | 4                     | 4                    |
| M2                | 2                     | 2                    |
| beta              | 0.05                  | 0.1                  |

