# <span id="page-37-1"></span>**H Hyperparameters**

All Eagle and Finch models were trained under bfloat16 format for most parameters, except that float32 was used to compute *WKV* for numerical stability. The Adam optimizer was configured with *β*<sup>1</sup> = 0.9, *β*<sup>2</sup> = 0.99 and 0.001 weight decay applied only to linear layers and embedding weights. The context length for pretraining was 4096 tokens. Learning rate for all models followed a linear 10 step warmup schedule from 20% to 100% of the maximum learning rate, followed by cosine decay to the minimum learning rate.

The time\_decay *w* parameters are placed into a special 2x learning rate multiplier grouping.

