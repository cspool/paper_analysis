# H Study on Weight Decay

Having established the best learning rate, we conducted additional investigations into weight decay. Weight decay was imposed solely on the model's linear layers. The Table [13](#page-17-0) illustrates that, currently, the

<span id="page-17-0"></span>

| Model           | Weight Decay | Learning Rate  | VQA   | SQA    | TQA   | GQA   | VizWiz | MME     |
|-----------------|--------------|----------------|-------|--------|-------|-------|--------|---------|
| VisualRWKV 1.6B | 0            | 6e-5 to 1.5e-5 | 69.42 | 59.05  | 43.57 | 55.23 | 29.84  | 1204.90 |
| VisualRWKV 1.6B | 0.1          | 6e-5 to 1.5e-5 | 68.48 | 58.85% | 41.58 | 54.34 | 28.05  | 1173.03 |
| VisualRWKV 1.6B | 0.01         | 6e-5 to 1.5e-5 | 68.53 | 59.40% | 42.24 | 54.24 | 27.86  | 1154.52 |

Table 13: Impact of Weight Decay on the Performance of the VisualRWKV on 6 benchmarks.

best outcomes are achieved without weight decay. The role of weight decay is complex and may require further exploration in the future.

