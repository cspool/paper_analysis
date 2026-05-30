# <span id="page-33-2"></span>I Hyperparameter details for LM training

We use the OLMoE codebase and its downstream tasks in the default configuration<sup>38</sup> except for MMLU: WinoGrande ("wino") (Sakaguchi et al. 2020), Social IQA ("SIQA") (Sap et al. 2019), SciQ (Johannes Welbl 2017), PIQA (Bisk et al.

<span id="page-33-4"></span> $<sup>^{38} \</sup>texttt{https://github.com/allenai/OLMoE/blob/357454f4f647385839c0ff6b99a688dc7cd9c13f/configs/OLMoE-1B-7B-0924.yml}$ 

<span id="page-34-0"></span>Table 9: Benchmark configurations used by Figure [10,](#page-10-0) [11a,](#page-12-0) and [11b,](#page-12-0) and all other kernel-level ablation studies on H100 and B300 GPUs.

(a) Benchmark configurations used by Figure [10](#page-10-0) and [11a,](#page-12-0) and all other ablation studies on H100 GPUs.

| Model Size | T                   | d   | n    | E      | K |
|------------|---------------------|-----|------|--------|---|
|            | 40960               | 768 | 256  | 128    | 8 |
| 1.4B       | 40960               | 768 | 512  | 64     | 4 |
|            | 40960               | 768 | 1024 | 32     | 2 |
|            | 24576 1536          |     | 256  | 128    | 8 |
| 7B         | 24576 1536          |     | 512  | 64     | 4 |
|            | 24576 1536 1024     |     |      | 32     | 2 |
|            | 32768 4096          |     | 256  | 256 16 |   |
| 30B        | 32768 4096          |     | 512  | 128    | 8 |
|            | 32768 4096 1024     |     |      | 64     | 4 |
|            | 32768 4096          |     | 512  | 256 16 |   |
| 120B       | 32768 4096 1024 128 |     |      |        | 8 |
|            | 32768 4096 2048     |     |      | 64     | 4 |

(b) Benchmark configurations used by Figure [11b,](#page-12-0) and all other ablation studies on B300 GPUs.

| Model Size | T      | d    | n             | E      | K |
|------------|--------|------|---------------|--------|---|
|            | 131072 | 768  | 256           | 128    | 8 |
| 1.4B       | 131072 | 768  | 512           | 64     | 4 |
|            | 131072 | 768  | 1024          | 32     | 2 |
|            | 81920  | 1536 | 256           | 128    | 8 |
| 7B         | 81920  | 1536 | 512           | 64     | 4 |
|            | 81920  |      | 1536 1024     | 32     | 2 |
|            | 32768  | 4096 | 256           | 256 16 |   |
| 30B        | 32768  | 4096 | 512           | 128    | 8 |
|            | 32768  |      | 4096 1024     | 64     | 4 |
|            | 32768  | 4096 | 512           | 256 16 |   |
| 120B       | 32768  |      | 4096 1024 128 |        | 8 |
|            | 32768  |      | 4096 2048     | 64     | 4 |

[2020\)](#page-16-16), OpenBookQA ("OBQA") (Mihaylov et al. [2018\)](#page-17-23), HellaSwag ("HS") (Zellers et al. [2019\)](#page-18-17), COPA (Roemmele, Bejan, and Gordon [2011\)](#page-18-18), CommonsenseQA ("CSQA") (Talmor et al. [2019\)](#page-18-19), BoolQ (Clark et al. [2019\)](#page-16-17), Arc-Easy and Arc-Challenge ("ArcE" and "ArcC") (Clark et al. [2018\)](#page-16-18) datasets. We use a deduplicated version of FineWeb-Edu (Ben Allal et al. [2024\)](#page-16-6)[39](#page-34-1) as the pretraining corpus, and train all models with a context length of 4096 tokens.

We always use MoE with SwiGLU for the MoE layers and we use an auxiliary load balancing loss (Shazeer et al. [2017\)](#page-18-0) with coefficient 0.01 but we do not use the router Z loss (Zoph et al. [2022\)](#page-19-7). Our attention block architecture is identical to OLMoE's attention block. We always tie the weight of the LM head with the weight of the token embedding matrix.

Table 10: Common configurations for MoE pretraining experiment

| Config name in Tables 2 and 6             |    | # layers # attn heads | d | n           | E  | K | # tokens in a minibatch | LR | WD | LR scheduler                            |
|-------------------------------------------|----|-----------------------|---|-------------|----|---|-------------------------|----|----|-----------------------------------------|
| 0.5B params, 20B tokens, 8/64 activated   | 12 | 12                    |   | 768 256     | 64 | 8 | 0.5M                    |    |    | 6e-4 0.01 cosine w/. warmup (10% steps) |
| 0.5B params, 40B tokens, 2/64 activated   | 12 | 12                    |   | 768 256     | 64 | 2 | 1M                      |    |    | 6e-4 0.01 cosine w/. warmup (10% steps) |
| 1.8B params, 40B tokens, 8/256 activated  | 12 | 12                    |   | 768 256 256 |    | 8 | 1M                      |    |    | 6e-4 0.01 cosine w/. warmup (10% steps) |
| 1.4B params, 50B tokens, 8/128 activated  | 18 | 12                    |   | 768 256 128 |    | 8 | 1M                      |    |    | 4e-4 0.01 cosine w/. warmup (10% steps) |
| 1.4B params, 100B tokens, 2/128 activated | 18 | 12                    |   | 768 256 128 |    | 2 | 2M                      |    |    | 4e-4 0.01 cosine w/. warmup (10% steps) |

For all EC with finetuned TC router experiments in Table [2,](#page-14-0) we use an additional 4B tokens and we only finetune the router weights with TC top-K routing (all other parameters are frozen). We always use a learning rate of 2e-4, weight decay of 0.01 and cosine learning rate scheduler with 10% warmup steps. The number of tokens per minibatch during finetuning is 1M. We disable auxiliary load balancing loss during TC finetuning.

For all EC with auxiliary router experiments, we use a 2-layer MLP (each linear layer has size E × E with SiLU activation) which takes as input the raw router logits and makes E independent binary predictions for all experts. We compute the averaged binary cross entropy loss over E labels using the multi-label prediction loss, and scale the loss by 0.01. During the evaluation, we will let the EC router compute the raw logits and raw scores and let the auxiliary router mask the token-expert pair with its own confidence score.

We implement "TC (token drop)" by discarding tokens selected from the TC top-K sorting.

<span id="page-34-1"></span><sup>39</sup><https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus>