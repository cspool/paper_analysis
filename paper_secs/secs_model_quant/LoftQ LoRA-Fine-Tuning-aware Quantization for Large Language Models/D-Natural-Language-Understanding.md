# D Natural Language Understanding

### <span id="page-19-1"></span>D.1 GLUE with 4-bit

We show the 4-bits results in the Table [11.](#page-19-2) Both methods can achieve performance close to full-finetuning.

<span id="page-19-2"></span>Table 11: Results with 4-bit LoftQ of DeBERTaV3-base models on GLUE development set using NF4 quantization. We report the median over four seeds. Results with N.A. indicate the model does not converge. The best results on each dataset are shown in bold

| Method  | Rank | MNLI      | SST-2 | QNLI | ANLI |
|---------|------|-----------|-------|------|------|
|         |      | m / mm    | Acc   | Acc  | Acc  |
| Full FT | -    | 90.5/90.6 | 95.3  | 94.0 | 59.8 |
| QLoRA   | 32   | 89.9/89.9 | 95.3  | 94.2 | 59.4 |
| LoftQ   | 32   | 89.9/90.0 | 95.3  | 94.1 | 59.9 |

### <span id="page-19-0"></span>D.2 Training Details

Implementation Details. The implementation of LoftQ is based on publicly available Huggingface [\(Paszke et al.,](#page-15-4) [2019\)](#page-15-4) code-base [\\*\\*](#page-19-3) .

Hyper-parameter Details. We select the learning rate of {1 × 10−<sup>5</sup> *,*5 × 10−<sup>5</sup> *,*1 × 10−<sup>4</sup> *,*5 × 10−<sup>4</sup> }, and use the selected learning rate for both uniform quantization experiments and nf2 quantization experiments. We use batch size of 32 for all GLUE tasks and ANLI. We use batch size of 16 for SQuADv1.1. We use LoftQ of 5 iterations for all GLUE tasks.

Table [12](#page-19-4) summarizes the detailed hyperparameters for each task used in training DeBERTaV3 base using uniform quantization. Table [13](#page-20-1) summarizes the detailed hyperparameters for each task used in training DeBERTaV3-base using nf2 quantization.

<span id="page-19-4"></span>Table 12: Hyper-parameter setup of LoftQ for GLUE benchmark for training DeBERTaV3-base using Uniform quantization.

| Hyper-parameter | MNLI     | RTE      | QNLI     | MRPC     | QQP      | SST-2    | CoLA     | STS-B    | SQuADv1.1 | ANLI     |
|-----------------|----------|----------|----------|----------|----------|----------|----------|----------|-----------|----------|
| # epochs        | 5        | 20       | 10       | 60       | 10       | 10       | 60       | 60       | 10        | 12       |
| Learning rate   | 1 × 10−4 | 5 × 10−4 | 5 × 10−5 | 1 × 10−4 | 5 × 10−5 | 5 × 10−5 | 5 × 10−5 | 5 × 10−5 | 5 × 10−5  | 5 × 10−5 |

<span id="page-19-3"></span><sup>\*\*</sup><https://github.com/huggingface/transformers/tree/main/examples/pytorch>

<span id="page-20-1"></span>Table 13: Hyper-parameter setup of LoftQ for GLUE benchmark for training DeBERTaV3-base using NF2 quantization.

| Hyper-parameter | MNLI     | RTE      | QNLI     | MRPC     | QQP      | SST-2    | CoLA     | STS-B    | SQuADv1.1 | ANLI     |
|-----------------|----------|----------|----------|----------|----------|----------|----------|----------|-----------|----------|
| # epochs        | 5        | 20       | 10       | 60       | 10       | 10       | 60       | 60       | 10        | 12       |
| Learning rate   | 1 × 10−4 | 5 × 10−5 | 5 × 10−5 | 1 × 10−4 | 5 × 10−5 | 5 × 10−5 | 5 × 10−5 | 1 × 10−4 | 5 × 10−5  | 5 × 10−5 |

