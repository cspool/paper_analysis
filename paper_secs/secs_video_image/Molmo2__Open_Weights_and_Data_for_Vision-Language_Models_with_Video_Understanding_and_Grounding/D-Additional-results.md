# **D Additional results**

In this section, we present several additional evaluations.

## <span id="page-37-2"></span>**D.1 Additional model ablations**

| Pretrain      | Video QA | Molmo2 Video Cap. | Image QA | Image Pointing |
|---------------|----------|-------------------|----------|----------------|
| With pointing | 66.8     | 31.8              | 80.9     | 73.0           |
| No pointing   | 65.9     | 31.3              | 80.1     | 71.8           |

**Table 18 Pre-training ablations**. Columns show the average of our 12 video benchmarks, using validation sets for EgoSchema, PerceptionText, and MLVU, video captioning F1, the average of the 11 image benchmarks using validation sets for InfoQA, DocQA, ChartQA, VQA v2, and AI2D, and the average score in Point-Bench.

**Pre-traing ablation**. We also present an ablation without image-pointing pre-training in Table [18.](#page-37-2) This model is only trained on image captioning and NLP data. For the SFT stage, it uses 2x the sampling rate for the image pointing datasets and 28k steps of training instead of 25k to compensate for the fact that the image pointing data is not seen during pre-training. We observe a small decrease in the benchmarks in this setting, even for those not related to image pointing. We hypothesize that pointing pre-training simplifies the SFT stage for the model since it no longer needs to learn the basic pointing format and task, allowing for more focus on the non-pointing tasks.

## <span id="page-37-3"></span>**D.2 NLP Benchmarks**

| Model                   | MMLU [50] | GSM8K [24] | ARC-C [23] | MBPP+ [8] |
|-------------------------|-----------|------------|------------|-----------|
| Qwen3-4B [169]          | 72.2      | 87.8       | 83.3       | 59.5      |
| Qwen3-8B [169]          | 76.8      | 89.8       | 88.3       | 62.2      |
| OLMo3-7B-Instruct [112] | 69.1      | 90.1       | 72.2       | 60.2      |
| Molmo2-4B               | 72.2      | 86.6       | 89.3       | 56.2      |
| Molmo2-8B               | 76.6      | 89.7       | 89.6       | 57.5      |
| Molmo2-O-7B             | 64.1      | 89.0       | 79.9       | 55.7      |

**Table 19 Results on selective NLP benchmarks**, including MMLU for general knowledge QA, GSM8K for math, ARC-C for reasoning, and MBPP+ for coding tasks.

We evaluate Molmo2 on selective NLP benchmarks covering general knowledge QA, math, reasoning, and coding tasks and report their results compared to the base language models Qwen3 in Table [19.](#page-37-3) We run evaluations for all models following OLMo 3's evaluation protocol, except for OLMo3-7B-Instruct's MMLU and MBPP+ numbers, which we take directly from OLMo3's model card. We find that Molmo2 achieves comparable numbers on the general knowledge QA and math benchmarks, MMLU and GSM8K, but suffers from some drops in coding on the MBPP+ coding benchmark [\[8\]](#page-17-18). Interestingly, both Molmo2-4B and Molmo2- 8B perform slightly better than their respective base language models in the ARC Challenge multiple-choice evaluation.

