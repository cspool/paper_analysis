# <span id="page-20-0"></span>E Summarization

### E.1 Training Details

We choose Adam as the optimizer and try learning rate from{1×10−<sup>5</sup> *,*5×10−<sup>5</sup> *,*7×10−<sup>5</sup> *,*2×10−<sup>4</sup> *,*3× 10−<sup>4</sup> *,*4 × 10−<sup>4</sup> }. We show the optimal learning rate for different settings in Table [14.](#page-20-2) We use LoftQ of 1 iteration for all BART-large experiments. Table [14](#page-20-2) and Table [15](#page-20-3) summarize the learning rate and other hyper-parameters for CNN/DailyMail and XSum.

<span id="page-20-2"></span>Table 14: Hyper-parameter setup of LoftQ BART-large on CNN/DailyMail

|                | NF4   |        | 4-bit Uniform |        | NF2   |        |
|----------------|-------|--------|---------------|--------|-------|--------|
| Hyperparameter | rank8 | rank16 | rank8         | rank16 | rank8 | rank16 |
| Learning rate  | 2e-4  | 2e-4   | 2e-4          | 3e-4   | 2e-4  | 2e-4   |
| Epoch          | 15    | 15     | 15            | 15     | 15    | 15     |
| Batch size     | 64    | 64     | 64            | 64     | 64    | 64     |

Table 15: Hyper-parameter setup of LoftQ BART-large on XSum

<span id="page-20-3"></span>

|                | NF4   |        |       | 4-bit Uniform | NF2   |        |
|----------------|-------|--------|-------|---------------|-------|--------|
| Hyperparameter | rank8 | rank16 | rank8 | rank16        | rank8 | rank16 |
| Learning rate  | 2e-4  | 2e-4   | 2e-4  | 2e-4          | 2e-4  | 2e-4   |
| Epoch          | 25    | 25     | 25    | 25            | 25    | 25     |
| Batch size     | 32    | 32     | 32    | 32            | 32    | 32     |

