# A Evaluation Details

#### <span id="page-11-1"></span><span id="page-11-0"></span>A.1 Models

| Model Name | nlayers | dmodel | dhead | nheads |
|------------|---------|--------|-------|--------|
| MoE-S      | 12      | 768    | 3072  | 12     |
| MoE-M      | 24      | 1024   | 4096  | 16     |
| MoE-L      | 24      | 1536   | 6144  | 16     |
| MoE-XL     | 24      | 1536   | 8192  | 24     |

Table 4: Model configurations for GPT-S, M, L, and XL

To create an MoE model based on GPT models, we follow the approach used by Switch [\(Fedus et al.,](#page-8-2) [2022\)](#page-8-2), which converts the Feed-Forward Network (FFN) in the Transformer architecture into experts and creates multiple copies as the number of experts increases.

To build MoE models of various sizes, we adopt hyperparameter configurations from GPT-3 models. [Table 4](#page-11-1) shows the details, where nlayers, dmodel, dhead, nheads respectively indicate the number of decoder layers, embedding dimension, feedforward layer embedding dimension, and the number of attention heads.

### <span id="page-11-2"></span>A.2 Additional Evaluations

| Metrics                 | Fairseq   | ES-MoE    |  |
|-------------------------|-----------|-----------|--|
| Processed Tokens        | 1B tokens | 1B tokens |  |
| Training Loss           | 3.353     | 3.344     |  |
| Valid Loss              | 5.149     | 5.144     |  |
| Training Duration (hrs) | 9.47      | 6.90      |  |

| # Exp. | Params. | Zero-OffloadE | ES-MoE | # GPUs |
|--------|---------|---------------|--------|--------|
| 16     | 13.3B   | 41%           | 59%    | 8      |
| 24     | 19.8B   | 40%           | 57%    | 16     |
| 32     | 23.1B   | 32%           | 39%    | 16     |
| 40     | 32.7B   | 28%           | 33%    | 16     |
| 48     | 39.1B   | 18%           | 31%    | 32     |

Table 5: ES-MoE accelerates the training of the MoE-L model, achieving the same loss 37% more quickly.

Table 6: GPU utilization while training a MoE-L model with varying numbers of experts.

Case Study: Pretraining. In [Table 5,](#page-11-2) we show a comparative analysis of end-to-end training time and training loss for MoE-L with 8 experts and a batch size of 128. Although both implementations are mathematically equivalent and demonstrate almost identical training and validation losses, a notable difference is observed in training efficiency. When compared to Fairseq [\(Lepikhin et al.,](#page-9-2) [2020\)](#page-9-2), ES-MoE completes the training process 37% more quickly, highlighting its enhanced efficiency in model training.

Communication and computation overhead. In [Table 6,](#page-11-2) we present an analysis of ES-MoE's offloading overhead by examining the effective GPU utilization percentage. Across various configurations, ES-MoE outperforms Zero-Offload<sup>E</sup> in effective GPU utilization, coming from pipelined expert scheduling and dynamic expert placement. The effective GPU utilization decreases with the growth in the number of experts for both Zero-Offload<sup>E</sup> and ES-MoE. This trend results from the fixed batch size, causing a reduction in assigned tokens per expert with an increased expert count. Consequently, the overlapping of computation and communication is diminished, and the required CPU computation escalates. Despite this limitation, ES-MoE demonstrates a significant capability to train large MoE models that conventionally require up to 32 GPUs, using even a single GPU. This flexibility in resource utilization may have broader implications, overshadowing the effects of offloading overhead.