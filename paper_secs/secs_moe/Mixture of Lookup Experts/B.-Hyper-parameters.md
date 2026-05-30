# B. Hyper-parameters

| Configuration Key       | Value                     |
|-------------------------|---------------------------|
| attention-dropout       | 0                         |
| dtype                   | bf16                      |
| global-batch-size       | 1024                      |
| gradient-clipping       | 1.0                       |
| hidden-dropout          | 0                         |
| lr-decay-style          | cosine                    |
| max-position-embeddings | 2048                      |
| min-lr                  | 0.1 ∗ optimizer.params.lr |
| no-weight-tying         | True                      |
| norm                    | RMSNorm                   |
| optimizer.params.betas  | [0.9, 0.95]               |
| optimizer.params.eps    | 1e-08                     |
| optimizer.type          | Adam                      |
| pos-emb                 | rotary                    |
| rotary-pct              | 0.25                      |
| seq-length              | 2048                      |
| train-iters             | 50000                     |
| warmup                  | 0.01                      |
| weight-decay            | 0.01                      |