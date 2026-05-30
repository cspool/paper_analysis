# <span id="page-13-0"></span>*3.2.2. Evaluation Results*

In Table [2,](#page-14-0) we compare DeepSeek-V2 with several representative open-source models, including DeepSeek 67B [\(DeepSeek-AI, 2024\)](#page-22-1) (our previous release), Qwen1.5 72B [\(Bai et al., 2023\)](#page-21-9), LLaMA3 70B [\(AI@Meta, 2024\)](#page-20-2), and Mixtral 8x22B [\(Mistral, 2024\)](#page-23-10). We evaluate all these models with our internal evaluation framework, and ensure that they share the same evaluation setting. Overall, with only 21B activated parameters, DeepSeek-V2 significantly outperforms DeepSeek 67B on almost all benchmarks, and achieves top-tier performance among open-source models.

Further, we elaborately compare DeepSeek-V2 with its open-source counterparts one by one. (1) Compared with Qwen1.5 72B, another model that supports both Chinese and English, DeepSeek-V2 demonstrates overwhelming advantages on the majority of English, code, and math benchmarks. As for Chinese benchmarks, Qwen1.5 72B shows better performance on

<span id="page-14-0"></span>

|         | Benchmark (Metric)    | # Shots | DeepSeek<br>67B | Qwen1.5<br>72B | Mixtral<br>8x22B | LLaMA 3<br>70B | DeepSeek-V2 |
|---------|-----------------------|---------|-----------------|----------------|------------------|----------------|-------------|
|         | Architecture          | -       | Dense           | Dense          | MoE              | Dense          | MoE         |
|         | # Activated Params    | -       | 67B             | 72B            | 39B              | 70B            | 21B         |
|         | # Total Params        | -       | 67B             | 72B            | 141B             | 70B            | 236B        |
|         | Pile-test (BPB)       | -       | 0.642           | 0.637          | 0.623            | 0.602          | 0.606       |
|         | BBH (EM)              | 3-shot  | 68.7            | 59.9           | 78.9             | 81.0           | 78.9        |
|         | MMLU (Acc.)           | 5-shot  | 71.3            | 77.2           | 77.6             | 78.9           | 78.5        |
|         | DROP (F1)             | 3-shot  | 69.7            | 71.5           | 80.4             | 82.5           | 80.1        |
|         | ARC-Easy (Acc.)       | 25-shot | 95.3            | 97.1           | 97.3             | 97.9           | 97.6        |
|         | ARC-Challenge (Acc.)  | 25-shot | 86.4            | 92.8           | 91.2             | 93.3           | 92.4        |
|         | HellaSwag (Acc.)      | 10-shot | 86.3            | 85.8           | 86.6             | 87.9           | 84.2        |
|         | PIQA (Acc.)           | 0-shot  | 83.6            | 83.3           | 83.6             | 85.0           | 83.7        |
| English | WinoGrande (Acc.)     | 5-shot  | 84.9            | 82.4           | 83.7             | 85.7           | 84.9        |
|         | RACE-Middle (Acc.)    | 5-shot  | 69.9            | 63.4           | 73.3             | 73.3           | 73.1        |
|         | RACE-High (Acc.)      | 5-shot  | 50.7            | 47.0           | 56.7             | 57.9           | 52.7        |
|         | TriviaQA (EM)         | 5-shot  | 78.9            | 73.1           | 82.1             | 81.6           | 79.9        |
|         | NaturalQuestions (EM) | 5-shot  | 36.6            | 35.6           | 39.6             | 40.2           | 38.7        |
|         | AGIEval (Acc.)        | 0-shot  | 41.3            | 64.4           | 43.4             | 49.8           | 51.2        |
|         | HumanEval (Pass@1)    | 0-shot  | 45.1            | 43.9           | 53.1             | 48.2           | 48.8        |
|         | MBPP (Pass@1)         | 3-shot  | 57.4            | 53.6           | 64.2             | 68.6           | 66.6        |
| Code    | CRUXEval-I (Acc.)     | 2-shot  | 42.5            | 44.3           | 52.4             | 49.4           | 52.8        |
|         | CRUXEval-O (Acc.)     | 2-shot  | 41.0            | 42.3           | 52.8             | 54.3           | 49.8        |
|         | GSM8K (EM)            | 8-shot  | 63.4            | 77.9           | 80.3             | 83.0           | 79.2        |
| Math    | MATH (EM)             | 4-shot  | 18.7            | 41.4           | 42.5             | 42.2           | 43.6        |
|         | CMath (EM)            | 3-shot  | 63.0            | 77.8           | 72.3             | 73.9           | 78.7        |
|         | CLUEWSC (EM)          | 5-shot  | 81.0            | 80.5           | 77.5             | 78.3           | 82.2        |
|         | C-Eval (Acc.)         | 5-shot  | 66.1            | 83.7           | 59.6             | 67.5           | 81.7        |
|         | CMMLU (Acc.)          | 5-shot  | 70.8            | 84.3           | 60.0             | 69.3           | 84.0        |
| Chinese | CMRC (EM)             | 1-shot  | 73.4            | 66.6           | 73.1             | 73.3           | 77.5        |
|         | C3 (Acc.)             | 0-shot  | 75.3            | 78.2           | 71.4             | 74.0           | 77.4        |
|         | CHID (Acc.)           | 0-shot  | 92.1            | -              | 57.0             | 83.2           | 92.7        |
|         | CCPM (Acc.)           | 0-shot  | 88.5            | 88.1           | 61.0             | 68.1           | 93.1        |

Table 2 | Comparison among DeepSeek-V2 and other representative open-source models. All models are evaluated in our internal framework and share the same evaluation setting. **Bold** denotes the best and underline denotes the second-best. Scores with a gap smaller than 0.3 are regarded as at the same level. With only 21B activated parameters, DeepSeek-V2 achieves top-tier performance among open-source models.

multi-subject multiple-choice tasks while DeepSeek-V2 is comparable or better on others. Note that for the CHID benchmark, the tokenizer of Qwen1.5 72B will encounter errors in our evaluation framework, so we leave the CHID score blank for Qwen1.5 72B. (2) Compared with Mixtral 8x22B, DeepSeek-V2 achieves comparable or better English performance, except for TriviaQA, NaturalQuestions, and HellaSwag, which are closely related to English commonsense knowledge. Notably, DeepSeek-V2 outperforms Mixtral 8x22B on MMLU. On code and math benchmarks, DeepSeek-V2 demonstrates comparable performance with Mixtral 8x22B. Since Mixtral 8x22B is not specifically trained on Chinese data, its Chinese capability lags far behind DeepSeek-V2. (3) Compared with LLaMA3 70B, DeepSeek-V2 is trained on fewer than a quarter of English tokens. Therefore, we acknowledge that DeepSeek-V2 still has a slight gap in basic English capabilities with LLaMA3 70B. However, even with much fewer training tokens and activated parameters, DeepSeek-V2 still demonstrates comparable code and math capability with LLaMA3 70B. Also, as a bilingual language model, DeepSeek-V2 outperforms LLaMA3 70B overwhelmingly on Chinese benchmarks.

Finally, it is worth mentioning that certain prior studies [\(Hu et al., 2024\)](#page-22-11) incorporate SFT data during the pre-training stage, whereas DeepSeek-V2 has never been exposed to SFT data during pre-training.

## <span id="page-15-0"></span>*3.2.3. Training and Inference Efficiency*

**Training Costs.** Since DeepSeek-V2 activates fewer parameters for each token and requires fewer FLOPs than DeepSeek 67B, training DeepSeek-V2 will be more economical than training DeepSeek 67B theoretically. Although training an MoE model will introduce additional communication overheads, through our operator and communication optimizations, the training for DeepSeek-V2 can attain a relatively high Model FLOPs Utilization (MFU). During our practical training on the H800 cluster, for training on each trillion tokens, DeepSeek 67B requires 300.6K GPU hours, while DeepSeek-V2 needs only 172.8K GPU hours, i.e., sparse DeepSeek-V2 can save 42.5% training costs compared with dense DeepSeek 67B.

**Inference Efficiency.** In order to efficiently deploy DeepSeek-V2 for service, we first convert its parameters into the precision of FP8. In addition, we also perform KV cache quantization [\(Hooper et al., 2024;](#page-22-12) [Zhao et al., 2023\)](#page-25-4) for DeepSeek-V2 to further compress each element in its KV cache into 6 bits on average. Benefiting from MLA and these optimizations, actually deployed DeepSeek-V2 requires significantly less KV cache than DeepSeek 67B, and thus can serve a much larger batch size. We evaluate the generation throughput of DeepSeek-V2 based on the prompt and generation length distribution from the actually deployed DeepSeek 67B service. On a single node with 8 H800 GPUs, DeepSeek-V2 achieves a generation throughput exceeding 50K tokens per second, which is 5.76 times the maximum generation throughput of DeepSeek 67B. In addition, the prompt input throughput of DeepSeek-V2 exceeds 100K tokens per second.

