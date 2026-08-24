# <span id="page-11-1"></span>C Fine-tuning details

#### **C.1 GPT-2**

| Hyperparameter | Value    |
|----------------|----------|
| Start LR       | 5e-5     |
| LR Schedule    | inv_sqrt |
| fp16           | True     |
| batch size     | 2        |
| n_steps        | 100K     |

Table 5: Hyperparameters used for GPT-2 fine-tuning

#### C.2 ViT

| Hyperparameter | Value              |  |  |
|----------------|--------------------|--|--|
| Start LR       | 5e-5               |  |  |
| LR Schedule    | lin_decay with min |  |  |
| decay_steps    | 20K                |  |  |
| Min LR         | 1e-6               |  |  |
| fp16           | True               |  |  |
| batch size     | 128                |  |  |
| n_steps        | 50K                |  |  |

Table 6: Hyperparameters used for ViT fine-tuning

#### **C.3** Machine Translation

We select our best model using validation BLEU, computed on a 2000 instance subset of the full Tatoeba validation set.

| Hyperparameter | Value    |
|----------------|----------|
| Start LR       | 5e-5     |
| LR Schedule    | inv_sqrt |
| fp16           | True     |
| batch size     | 64       |
| n_steps        | 100K     |

Table 7: Hyperparameters used for OPUS-MT fine-tuning

#### C.4 OLMo-7B QLoRA

We select our best model using validation loss.

| Hyperparameter | Value      |
|----------------|------------|
| Start LR       | 5e-4       |
| LR Schedule    | inv_sqrt   |
| warmup ratio   | 0.01       |
| fp16           | True       |
| batch size     | 8          |
| n_steps        | 3000       |
| weight decay   | 0.01       |
| LoRA rank      | 8          |
| LoRA α         | 16         |
| LoRA modules   | all linear |
| LoRA dropout   | 0.2        |
|                |            |

Table 8: Hyperparameters used for OLMo-7B QLoRA fine-tuning

## <span id="page-12-0"></span>D Dataset details

We report the dataset statistics for our evaluations and training data used in this work in Table [9.](#page-12-3) For fine-tuning data, we note that updates reported in Appendix [C](#page-11-1) give a representation of data usage rather than the training counts provided here.

<span id="page-12-3"></span>

| Dataset      | Train      | Validation | Test   |
|--------------|------------|------------|--------|
| ImageNet-1k  | 12,281,167 | 50,000     | -      |
| Wikitext-103 | 1,801,350  | 3,760      | 4,358  |
| OPUS/Tatoeba | 41,649,946 | 43,074     | 10,389 |
| SamSum       | 14,732     | 818        | 819    |

Table 9: The number of instances used in each finetuning and evaluation datasets. Instances are image for ImageNet, lines of text for Wikitext-103, bitext pairs for OPUS/Tatoeba, and dialogue/summary pairs for Sam-Sum.

