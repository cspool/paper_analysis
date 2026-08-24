# <span id="page-17-0"></span>D Implementation Details

For the language models we use, Mixtral-8x7b is approximately 6.5 times larger in scale compared to Mistral-7b and features a divergent architectural approach—specifically, a dense versus mixture-ofexperts design. For our assessments, we employed the instruction-tuned variants of these models.

Owing to efficiency constraints, we opted not to perform on-the-fly retrieval. Instead, we preconstructed a retrieval index using the efficient and robust multi-vector retriever, ColBERT-v2 [\[68\]](#page-13-15), from which we retrieved the *top-1* ranked document for inclusion in our instruction-tuning dataset and for the evaluation of downstream tasks. Subsequently, we re-encoded these documents using the embedding model of interest (e.g., SFR). This strategy allows us to iterate data-centric experiments quickly. All experiments are conducted on the a setup of 8xNvidia A100 GPUs.

<span id="page-17-1"></span>In Table [9](#page-17-1) and Table [10,](#page-17-2) we list the hyperparameters for Paraphrase Pretraining and Context-aware Instruction Tuning.

| Hyperparameter              | Assignment |
|-----------------------------|------------|
| optimizer                   | AdamW      |
| learning rate               | 6e-3       |
| lr scheduler type           | linear     |
| warmup ratio                | 0.03       |
| weight dacay                | 0.0        |
| epochs                      | 1          |
| flash attention             | True       |
| batch size                  | 12         |
| gradient accumulation steps | 4          |
| num GPUs                    | 8          |
| max sequence length         | 336        |
| max train samples           | 2,000,000  |

<span id="page-17-2"></span>Table 9: Hyperparameters for Paraphrase Pretraining.

| Hyperparameter              | Assignment |
|-----------------------------|------------|
| optimizer                   | AdamW      |
| learning rate               | 2e-5       |
| lr scheduler type           | linear     |
| warmup ratio                | 0.03       |
| weight dacay                | 0.0        |
| epochs                      | 1          |
| KL α                        | 2.0        |
| KL temperature              | 1.0        |
| flash attention             | True       |
| batch size                  | 4          |
| gradient accumulation steps | 2          |
| num GPUs                    | 8          |
| max sequence length         | 1024       |
| max train samples           | 955,338    |

Table 10: Hyperparameters for Context-aware Instruction Tuning.

### <span id="page-18-0"></span>**E** About different Embedding Models

In our primary experiments, we use the SFR model as our default sentence embedding model. This section delves into the effects of different embedding models. We examine four universal text embedding models: E5-Mistral and E5-Large [73] alongside BGE-Large and BGE-Base [78]. Additionally, we assess two retrieval-specific models: Dragon [48] and DPR [34]. The configurations of the different retrievers and their MTEB scores<sup>6</sup>—a metric indicating their general sentence representation capability—are listed in Table 11. To isolate the impact of potentially different retrieved documents, we ensure that all models utilize the same *top-1* document. The performance is averaged over four question answering datasets. A general pattern is that embedding models with stronger sentence representation capabilities tend to further enhance the downstream performance. Remarkably, the Dragon model, despite being a BERT-base-sized retrieval-specific model, outperforms general text embedding models that are twice its size (BGE-Large).

<span id="page-18-2"></span>

|            | Model         |      | Model Embedding Universal MTE |           | MTEB  | A           |            |       |
|------------|---------------|------|-------------------------------|-----------|-------|-------------|------------|-------|
| Model      |               | Size | Dim                           | Embedding | Score | Performance | Resilience | Boost |
| Mistral-7b |               |      |                               |           |       |             |            |       |
|            | w/o retrieval |      |                               |           |       | 37.2        | -          | -     |
|            | w retrieval   |      |                               |           |       | 46.3        | 74.1%      | 29.5% |
|            | w SFR         | 7B   | 4096                          |           | 67.56 | 44.5        | 82.3%      | 22.2% |
|            | w E5-Mistral  | 7B   | 4096                          | ✓         | 66.63 | 44.0        | 84.0%      | 20.6% |
|            | w E5-Large    | 335M | 1024                          | ✓         | 62.25 | 42.1        | 80.2%      | 19.6% |
| xRAG       | w BGE-Large   | 335M | 1024                          | ✓         | 64.23 | 41.6        | 78.2%      | 19.8% |
|            | w BGE-Base    | 109M | 768                           | ✓         | 63.55 | 41.2        | 78.7%      | 18.9% |
|            | w Dragon      | 109M | 768                           | X         | -     | 42.1        | 84.2%      | 16.9% |
|            | w DPR         | 109M | 768                           | X         | -     | 40.5        | 77.4%      | 18.2% |

Table 11: Ablation on different sentence embedding models.

<span id="page-18-1"></span> $<sup>^6 {\</sup>tt https://huggingface.co/spaces/mteb/leaderboard}$ 

