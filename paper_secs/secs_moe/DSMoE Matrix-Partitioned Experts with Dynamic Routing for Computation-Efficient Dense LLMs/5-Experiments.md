# 5 Experiments

## 5.1 Dataset

We gathered datasets from various domains to continually pre-train the base model. For the general domain, we used the Fineweb-edu dataset, which consists of high-quality educational web pages filtered from the Fineweb dataset [\(Penedo et al.,](#page-10-9) [2024\)](#page-10-9). In the math and coding domains, we selected the OpenWebMath [\(Paster et al.,](#page-9-14) [2024\)](#page-9-14) and StarCoder [\(Li et al.,](#page-9-15) [2023\)](#page-9-15) datasets respectively. The OpenWebMath dataset contains high-quality mathematical text data extracted from web pages, while the StarCoder dataset offers a diverse range of code data and has been demonstrated to effectively pre-train well-behaved code models. Furthermore, it has been demonstrated that incorporating synthetic data enhances model pre-training performance [\(Abdin et al.,](#page-8-6) [2024\)](#page-8-6). Therefore, we introduced the Cosmopedia dataset to leverage this advantage[\(Ben Allal et al.,](#page-8-7) [2024\)](#page-8-7).

Furthermore, we mixed datasets from different domains. Due to computational resource limitations, we set the total amount of training data to 10 billion tokens. Finally, we used the tokenizers from LLaMA to segment the data, limiting the maximum sample length to 1024 tokens for each. We randomly sampled 5,000 non-overlapping instances from each dataset as the validation set, ensuring no intersection with the training set.

#### 5.2 Experimental Setup

We evaluate DSMoE on two pre-trained models of different scales: Llama-7B[1](#page-5-0) and Llama-1B[2](#page-5-1) . For

[2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)

our method's hyperparameters, we simply set the activation threshold τ = 0.5, learning rate to 2e-5, batch size to 32, and sequence length to 1024. To ensure fair evaluation, all baseline methods underwent continued training on identical data quantities (10B tokens) with the same training configurations.

We compare our approach with several baselines: the channel-wise and block-wise methods from LLM-Pruner (a structured pruning approach), and SparseGPT (an unstructured pruning method). To ensure fair comparison, all baseline methods (LLM-Pruner channel/block-wise, SparseGPT, and LLaMA-MoE) were trained on identical data quantities (10B tokens) and configured to match DSMoE's activated parameter count.

Additionally, we compare against LLaMA-MoE, which applies a similar FFN partitioning scheme but follows the traditional MoE paradigm with fixed top-k expert selection and standard MoE training objectives, to investigate whether conventional MoE frameworks can effectively leverage pre-trained weights through warm-starting.

#### 5.3 Main Results

We first present the model's perplexity on the validation set. Following previous work[\(Touvron et al.,](#page-10-0) [2023;](#page-10-0) [Brown et al.,](#page-8-8) [2020;](#page-8-8) [Su et al.,](#page-10-10) [2024;](#page-10-10) [Xiong](#page-10-11) [et al.,](#page-10-11) [2024;](#page-10-11) [Dai et al.,](#page-8-4) [2024\)](#page-8-4), we then evaluate the model's performance on downstream benchmarks, which includes zero-shot accuracy testing on HellaSwag[\(Zellers et al.,](#page-10-12) [2019\)](#page-10-12), LAMBADA[\(Paperno](#page-9-16) [et al.,](#page-9-16) [2016\)](#page-9-16), SIQA[\(Sap et al.,](#page-10-13) [2019\)](#page-10-13), PIQA[\(Bisk](#page-8-9) [et al.,](#page-8-9) [2020\)](#page-8-9), StoryCloze[\(Mostafazadeh et al.,](#page-9-17) [2016\)](#page-9-17), and Winogrande[\(Sakaguchi et al.,](#page-10-14) [2021\)](#page-10-14). Additionally, we conduct 5-shot evaluation measuring exact match performance on TriviaQA[\(Joshi](#page-9-18) [et al.,](#page-9-18) [2017\)](#page-9-18), WebQuestions (WebQs)[\(Berant et al.,](#page-8-10) [2013\)](#page-8-10), GSM8K[\(Cobbe et al.,](#page-8-11) [2021\)](#page-8-11), and Natural Questions (NaturalQs)[\(Kwiatkowski et al.,](#page-9-19) [2019\)](#page-9-19).

#### 5.3.1 Perplexity Results

Table [1](#page-4-0) presents the perplexity results of the baseline dense model and its pruned, sparsified variants. The results demonstrate that DSMoE consistently outperforms baseline models under equivalent activation constraints. Our experimental results indicate that DSMoE achieves superior efficiency compared to static parameter pruning. Furthermore, DSMoE exhibits better performance than fixed-activation methods like MoE, which can be attributed to the fact that knowledge from all experts contributes to the model's learning process,

<span id="page-5-0"></span><sup>1</sup> <https://huggingface.co/meta-llama/Llama-2-7b>

<span id="page-5-1"></span>[https://huggingface.co/meta-llama/Llama-3.](https://huggingface.co/meta-llama/Llama-3.2-1B)

<span id="page-6-0"></span>

| Model              | Hellaswag | LAMBAD | A PIQA | SIQA  | StoryCloze | Wino  | GSM8K | NaturalQs | TriviaQA | WebQs |
|--------------------|-----------|--------|--------|-------|------------|-------|-------|-----------|----------|-------|
| LLaMA-1B           | 64.09     | 61.05  | 75.51  | 42.47 | 72.58      | 60.85 | 4.85  | 12.52     | 36.08    | 22.49 |
| LLaMA-7B           | 76.39     | 72.34  | 79.05  | 44.67 | 79.15      | 70.87 | 14.70 | 26.28     | 61.89    | 32.82 |
| LLaMA-1B           |           |        |        |       |            |       |       |           |          |       |
| LLM-Pruner-channel | 53.44     | 45.04  | 71.43  | 40.94 | 68.67      | 58.45 | 1.44  | 6.98      | 17.46    | 14.56 |
| LLM-Pruner-block   | 51.05     | 46.28  | 71.71  | 41.04 | 68.62      | 56.27 | 1.36  | 7.28      | 18.46    | 14.56 |
| SparseGPT          | 54.01     | 56.49  | 71.10  | 40.68 | 68.05      | 57.30 | 1.51  | 5.29      | 14.44    | 11.61 |
| LLaMA-MoE          | 49.06     | 44.84  | 70.02  | 41.05 | 65.47      | 55.64 | 1.62  | 5.76      | 13.49    | 11.27 |
| DSMoE(ours)        | 50.92     | 48.12  | 72.36  | 41.14 | 68.78      | 56.35 | 1.67  | 8.17      | 25.52    | 18.21 |
| LLaMA-7B           |           |        |        |       |            |       |       |           |          |       |
| LLM-Pruner-channel | 66.41     | 61.63  | 74.97  | 43.19 | 75.30      | 66.85 | 4.85  | 12.63     | 36.02    | 20.57 |
| LLM-Pruner-block   | 67.93     | 62.02  | 76.22  | 44.26 | 75.46      | 63.53 | 1.81  | 12.96     | 38.77    | 21.65 |
| SparseGPT          | 73.60     | 67.43  | 77.36  | 44.21 | 76.37      | 70.48 | 8.33  | 17.61     | 47.83    | 24.90 |
| LLaMA-MoE          | 63.89     | 60.49  | 74.10  | 43.29 | 72.90      | 61.17 | 3.26  | 11.58     | 31.25    | 19.09 |
| DSMoE(ours)        | 70.22     | 67.61  | 78.12  | 44.31 | 76.37      | 66.77 | 6.41  | 22.04     | 57.94    | 29.92 |

Table 2: Performances of language models on downstream tasks. The best score is marked in **bold**.

enabling it to develop the ability to flexibly select activations based on input. Additionally, DSMoE exhibits distinctive feature processing capabilities, learning layer-specific activation patterns that naturally emerge from the input complexity. We will examine these emergent patterns in detail in the analysis section.

In conclusion, DSMoE demonstrates consistent superiority across models of two different scales, highlighting its robust advantages.

#### 5.3.2 Benchmark Results

Table 2 presents the benchmark performance of various pruning methods, traditional MoE approaches, and DSMoE. DSMoE achieved the best performance in 7 out of 10 benchmarks for both LLaMA-1B and LLaMA-7B model architectures, demonstrating superior effectiveness over existing sparsification methods across most evaluation metrics.

Specifically, DSMoE exhibited excellent performance on inference tasks (i.e., the first 6 benchmarks), achieving the best results on PIQA, SIQA, and StoryCloze test sets. While not achieving top performance on Hellaswag, LAMBADA, and Wino test sets, DSMoE still ranked among the leading models. For generation tasks (i.e., the last 4 benchmarks), DSMoE demonstrated remarkable effectiveness. Apart from slightly lower performance on GSM8K with LLaMA-7B compared to SparseGPT, it significantly outperformed other sparse methods on all other test sets, with performance only a few points below the dense model. These results highlight DSMoE's potential, particularly in generation tasks.

Furthermore, we observed that the performance gap between DSMoE and other sparse approaches

<span id="page-6-1"></span>

| Model      | DSMoE | w/o S(x) |
|------------|-------|----------|
| Hellaswag  | 50.92 | 32.29    |
| LAMBADA    | 48.12 | 27.79    |
| PIQA       | 72.36 | 62.73    |
| SIQA       | 41.14 | 39.30    |
| StoryCloze | 68.67 | 57.14    |
| Wino       | 56.35 | 50.83    |
| GSM8K      | 1.67  | 0.38     |
| NaturalQs  | 8.17  | 2.47     |
| TriviaQA   | 25.52 | 2.95     |
| WebQs      | 18.21 | 1.00     |
| PPL        | 7.41  | 12.75    |

Table 3: Ablation study of DMoE against the model without direct estimation function S(x), where G(x) is employed in place of S(x).

was more pronounced in LLaMA-7B compared to LLaMA-1B. This may be attributed to greater model redundancy at larger parameter scales, enabling DSMoE to more effectively prune unnecessary information. This observation suggests the potential scalability of DSMoE to models with larger parameter counts.

## 6 Analyses

