# 6 Experiments & Evaluation

## 6.1 Baseline Models

We mainly select open-source models of approximately 3B parameters for comparing model performance. Additionally, we include some state-of-the-art (SOTA) models with around 7B parameters. The details of the selected models are shown in table [4.](#page-9-0)

| Model                            | #Param. | #Tokens | Open-Sourced Data | Language      |
|----------------------------------|---------|---------|-------------------|---------------|
| LLaMA2 Touvron et al. (2023)     | 7B      | 2T      | NO                | EN*           |
| Baichuan2 Yang et al. (2023a)    | 7B      | 2.6T    | NO                | ZH,EN         |
| ChatGLM3 Du et al. (2022)        | 6B      | -       | NO                | ZH,EN         |
| Chinese-LLaMA2 Cui et al. (2023) | 7B      | -       | NO                | ZH,EN         |
| TinyLlama Zhang et al. (2024)    | 1.1B    | 3T      | YES               | EN            |
| OpenLLaMA-v2 Geng & Liu (2023)   | 3B      | 1T      | YES               | EN            |
| BLOOM Le Scao et al. (2022)      | 3B      | 350B    | YES               | Multi-Lingual |
| MindLLM Yang et al. (2023b)      | 1.3B    | 323B    | YES               | ZH,EN         |
| GPT-NEO Black et al. (2021)      | 2.7B    | 420B    | YES               | EN            |
| INCITE-Base Computer (2023)      | 3B      | 800B    | YES               | EN            |
| Sheard-LLaMA Xia et al. (2024)   | 2.7B    | 50B     | YES               | EN            |
| Phi2 Li et al. (2023c)           | 2.7B    | 1.4T    | NO                | ZH,EN         |
| Qwen Bai et al. (2023)           | 1.8B    | 2.2T    | NO                | ZH,EN         |
| Mini-CPM mic (2024)              | 2.7B    | 1.1T    | NO                | ZH,EN         |

<span id="page-9-0"></span>Table 4: Details of the baseline models. We denote the training corpus of LLaMA2 with \* because English significantly predominates in terms of proportion among all languages.

## 6.2 Settings

We select a diverse range of tasks for evaluation. For world common knowledge, we evaluate the models on MMLU [\(Hendrycks et al., 2020\)](#page-15-12), CMMLU [\(Li et al., 2023a\)](#page-16-10), C-EVAL [\(Huang et al.,](#page-16-11) [2024\)](#page-16-11) and BBH [\(Suzgun et al., 2022\)](#page-17-16). For commonsense reasoning and reading comprehension, we select SciQ [Johannes Welbl](#page-16-12) [\(2017\)](#page-16-12), PIQA [Bisk et al.](#page-14-12) [\(2020\)](#page-14-12), ARC [Clark et al.](#page-14-13) [\(2018\)](#page-14-13), LogiQA [Liu](#page-16-13) [et al.](#page-16-13) [\(2020\)](#page-16-13), and BoolQ [Clark et al.](#page-14-14) [\(2019\)](#page-14-14). We annotate the num-shots used during evaluation in the parentheses on the right of the dataset name. For the performance of the baseline models, if results for the corresponding dataset are available in the original paper, we directly report those results; otherwise, we report the results we reproduced. We use the LM-Evaluation-Harness repository [Gao](#page-15-13) [et al.](#page-15-13) [\(2023\)](#page-15-13) to reproduce the results of the baseline models. Since LM-Evaluation-Harness is not well adapted for encoder-decoder architecture, we have modified the original repository and redeveloped the evaluation code for OpenBA-V2.

