# <span id="page-2-1"></span>3 METHODS

Figure [2](#page-2-0) illustrates the workflows for three categories of prompt compression methods, from which we select six methods: (1) *RL-based:* KiS, SCRL, (2) *LLM scoring-based:* Selective Context, and (3) *LLM annotation-based:* LLMLingua, LongLLMLingua, LLMLingua-2. Among them, KiS does not typically trim words but uses an autoregressive approach to regenerate a shorter context, which can be time-intensive. However, we include it for comparison.

KiS. [Laban et al.](#page-11-6) [\(2021\)](#page-11-6) tackles the challenge of text simplification in an unsupervised manner, balancing fluency, salience, and simplicity. The model leverages reinforcement learning to enhance its performance by generating multiple candidate simplifications and optimizing for a composite reward. Utilizing a k-SCST algorithm, KiS generates k candidate outputs for each input, computing a reward for each, and promotes candidates surpassing the mean reward.

SCRL. [Ghalandari et al.](#page-10-4) [\(2022\)](#page-10-4) also represents unsupervised sentence compression via reinforcement learning, focusing on sequence labeling. It fine-tunes a pre-trained transformer model using a simple policy gradient approach. Each token in a sentence is labeled as essential or non-essential, optimizing the reward functions to maximize the compression quality while maintaining fluency and faithfulness.

Selective Context. [Li et al.](#page-11-1) [\(2023\)](#page-11-1) involves assessing the informativeness of lexical units by computing their self-information using a base causal language model. By pruning the redundant parts, a more concise context is obtained.

**LLMLingua.** Jiang et al. (2023) introduces a coarse-to-fine prompt compression method to handle lengthy prompts. LLMLingua includes a budget controller to ensure semantic integrity during high compression ratios, a token-level iterative compression algorithm to model interdependencies, and instruction tuning to align distributions between a small model and LLMs.

**LongLLMLingua.** Building on LLMLingua, LongLLMLingua (Jiang et al., 2024) is tailored for long context scenarios. It employs a question-aware coarse-to-fine compression technique and reorders documents to mitigate position bias (Liu et al., 2023a). It supports dynamic compression ratios and includes a post-compression strategy to ensure the preservation of content integrity.

**LLMLingua-2.** Developed as an advancement over LLMLingua, LLMLingua-2 (Pan et al., 2024) focuses on task-agnostic prompt compression for enhanced generalizability and efficiency. It introduces a data distillation procedure from GPT-4, creating an extractive text compression dataset to align with compression objectives effectively. LLMLingua-2 frames prompt compression as a token classification task using a Transformer encoder to leverage full bidirectional context, addressing the reliance on unidirectional context in prior approaches.

#### 4 EXPERIMENT SETUP

### 4.1 TASKS AND DATASETS

For our study on prompt compression for LLMs, we designated three tasks: summarization, reconstruction, and question answering (QA). The summarization task involves generating summaries from both the original and compressed contexts and measuring the similarity between these summaries. We use datasets including Gigaword (Rush et al., 2015), DUC2004 (Over et al., 2007), BNC (Consortium, 2007), Google (Filippova & Altun, 2013), and Broadcast (Clarke & Lapata, 2008a). The reconstruction task involves prompting the LLM to reconstruct the original prompt from the compressed prompt and includes datasets like GSM8K (Cobbe et al., 2021), BBC News, Arxiv articles, and ShareGPT (Li et al., 2023). The QA task¹ leverages datasets including LongBench (Bai et al., 2024), BBH (Suzgun et al., 2023), and GSM8K².

For MLLMs, our primary focus is on their performance in the VQA task, utilizing datasets including IconQA (Lu et al., 2021) and OK-VQA (Marino et al., 2019). Further details about these datasets can be found in Appendix A.1.

#### 4.2 METRICS

For the summarization and reconstruction tasks, we utilize BLEU<sup>3</sup> Papineni et al. (2002b), ROUGE<sup>4</sup> (Lin, 2004b), and BERTScore<sup>5</sup> (Zhang\* et al., 2020) to measure the similarity between the generated and reference outputs. For QA and VQA tasks, we differentiate the evaluation metrics based on the nature of the answers. For tasks with clear, precise answers, accuracy is used as the evaluation metric. For open-ended questions, we assess the similarity between the generated responses and reference answers using F1 (Bai et al., 2024). For hallucination detection, following (Li et al., 2024), we use micro hallucination rate (MiHR) and macro hallucination rate (MaHR) to evaluate the degree of hallucination. Further details about the computation of these metrics can be found in Appendix A.2.

#### <span id="page-3-5"></span>4.3 IMPLEMENTATIONS

In our experiments, we selected the six prompt compression methods mentioned in Section 3: KiS, SCRL, Selective Context, LLMLingua, LongLLMLingua, and LLMLingua-2. For KiS and SCRL, the compression ratio is self-adapted, while for Selective Context, LLMLingua, LongLLMLingua, and LLMLingua-2, the compression ratio is adjustable. We set it to 0.5 unless otherwise specified.

<span id="page-3-1"></span><span id="page-3-0"></span><sup>&</sup>lt;sup>1</sup>We also categorize mathematical problems and multiple-choice questions under the scope of QA.

<sup>&</sup>lt;sup>2</sup>We utilized GSM8K in both reconstruction and QA tasks. For the former, we only evaluate the performance of reconstruction without providing answers.

<span id="page-3-2"></span><sup>3</sup>https://github.com/nltk/nltk

<span id="page-3-3"></span><sup>4</sup>https://github.com/pltrdy/rouge

<span id="page-3-4"></span><sup>&</sup>lt;sup>5</sup>https://github.com/Tiiiger/bert\_score

<span id="page-4-0"></span>Table 1: **Performance for reconstruction and summarization tasks.** We grouped LLMLingua and LongLLMLingua together, as the performance differences between these two compressors were minimal for these two tasks. For each setting, we averaged the scores across three models: GPT-3.5-turbo, GPT-4o-mini, and Claude-3-Haiku.

| Method            | Metric        | Reconstruction |          |          |       | Summarization |         |      |           |        |
|-------------------|---------------|----------------|----------|----------|-------|---------------|---------|------|-----------|--------|
| Method            | Metric        | GSM8K          | BBC News | ShareGPT | Arxiv | Gigaword      | DUC2004 | BNC  | Broadcast | Google |
| Random Selection  |               | 0.40           | 0.23     | 0.19     | 0.07  | 0.25          | 0.21    | 0.21 | 0.10      | 0.23   |
| KiS               |               | 0.58           | 0.16     | 0.11     | 0.05  | 0.20          | 0.23    | 0.55 | 0.49      | 0.36   |
| SCRL              | BLEU (†)      | 0.38           | 0.14     | 0.30     | 0.10  | 0.28          | 0.28    | 0.57 | 0.47      | 0.46   |
| Selective Context | BLEU ( )      | 0.58           | 0.33     | 0.35     | 0.31  | 0.26          | 0.25    | 0.56 | 0.50      | 0.47   |
| (Long)LLMLingua   |               | 0.76           | 0.19     | 0.26     | 0.15  | 0.22          | 0.23    | 0.66 | 0.73      | 0.42   |
| LLMLingua-2       |               | 0.55           | 0.28     | 0.26     | 0.45  | 0.29          | 0.34    | 0.46 | 0.61      | 0.48   |
| Random Selection  |               | 0.60           | 0.43     | 0.54     | 0.22  | 0.21          | 0.15    | 0.32 | 0.44      | 0.37   |
| KiS               | ROUGE L (†)   | 0.75           | 0.32     | 0.38     | 0.13  | 0.20          | 0.18    | 0.63 | 0.61      | 0.49   |
| SCRL              |               | 0.55           | 0.30     | 0.61     | 0.28  | 0.22          | 0.15    | 0.44 | 0.43      | 0.40   |
| Selective Context | ROUGE L ( )   | 0.82           | 0.67     | 0.66     | 0.56  | 0.21          | 0.15    | 0.59 | 0.59      | 0.54   |
| (Long)LLMLingua   |               | 0.89           | 0.52     | 0.58     | 0.45  | 0.23          | 0.19    | 0.80 | 0.88      | 0.56   |
| LLMLingua-2       |               | 0.86           | 0.47     | 0.48     | 0.35  | 0.26          | 0.21    | 0.64 | 0.57      | 0.53   |
| Random Selection  |               | 0.93           | 0.85     | 0.87     | 0.81  | 0.83          | 0.84    | 0.84 | 0.82      | 0.85   |
| KiS               |               | 0.94           | 0.88     | 0.86     | 0.82  | 0.85          | 0.87    | 0.92 | 0.91      | 0.92   |
| SCRL              | BERTScore (†) | 0.69           | 0.84     | 0.89     | 0.85  | 0.84          | 0.84    | 0.85 | 0.82      | 0.88   |
| Selective Context |               | 0.96           | 0.90     | 0.91     | 0.92  | 0.85          | 0.85    | 0.89 | 0.89      | 0.89   |
| (Long)LLMLingua   |               | 0.98           | 0.87     | 0.90     | 0.89  | 0.85          | 0.87    | 0.95 | 0.96      | 0.93   |
| LLMLingua-2       |               | 0.94           | 0.85     | 0.86     | 0.84  | 0.86          | 0.85    | 0.90 | 0.91      | 0.90   |

<span id="page-4-1"></span>Table 2: **Performance for QA tasks.** For each setting, we averaged the scores across three models: GPT-3.5-turbo, GPT-40-mini, and Claude-3-Haiku.

| Method            | BBH Boolean<br>Expression | BBH Causal<br>Judgement | BBH<br>Web of Lies | GSM8K<br>Math | LongBench<br>SingleDoc | LongBench<br>MultiDoc | LongBench<br>FewShot | LongBench<br>Synth. |
|-------------------|---------------------------|-------------------------|--------------------|---------------|------------------------|-----------------------|----------------------|---------------------|
|                   | Acc. (†)                  | Acc. (†)                | Acc. (†)           | Acc. (†)      | F1 (†)                 | F1 (†)                | Acc. (†)             | Acc. (†)            |
| Original Prompt   | 0.516                     | 0.648                   | 0.556              | 0.337         | 0.149                  | 0.095                 | 0.334                | 0.174               |
| Random Selection  | 0.468                     | 0.556                   | 0.532              | 0.030         | 0.146                  | 0.108                 | 0.356                | 0.192               |
| KiS               | 0.576                     | 0.480                   | 0.512              | 0.149         | 0.118                  | 0.092                 | 0.312                | 0.166               |
| SCRL              | 0.464                     | 0.472                   | 0.556              | 0.218         | 0.214                  | 0.302                 | 0.378                | 0.176               |
| Selective Context | 0.480                     | 0.616                   | 0.552              | 0.179         | 0.185                  | 0.101                 | 0.412                | 0.288               |
| LLMLingua         | 0.528                     | 0.504                   | 0.536              | 0.297         | 0.286                  | 0.319                 | 0.620                | 0.128               |
| LongLLMLingua     | 0.524                     | 0.536                   | 0.492              | 0.218         | 0.301                  | 0.334                 | 0.640                | 0.582               |
| LLMLingua-2       | 0.484                     | 0.584                   | 0.520              | 0.220         | 0.223                  | 0.312                 | 0.632                | 0.210               |

Additionally, we included a random selection strategy, which involves randomly picking words from the original prompt based on the compression ratio, to serve as a baseline comparison. We evaluated these methods' performance across three (M)LLMs: GPT-3.5-turbo, GPT-40-mini, and Claude-3-Haiku.

Moreover, we have compiled our implementation into a comprehensive toolkit, which we have open-sourced to facilitate reproducibility and further research. More details about the toolkit are provided in Appendix C.

### 5 EXPERIMENTAL RESULTS

### 5.1 MAIN RESULTS

**Question 1:** Which prompt compression method performs best across different tasks?

Table 1 presents a detailed comparison of different prompt compression methods, assessing their performance in various scenarios. Summarization tasks focus on retaining critical information, while reconstruction tasks emphasize detail preservation. Table 2 and Figure 4 are utilized to elucidate the performance of these methods in QA tasks with varying context lengths, from shorter contexts (BBH, GSM8K) to longer ones (LongBench). Furthermore, we assess the computational overhead of these methods, as provided in Table 3, to determine their practicality concerning time cost and memory consumption.

<span id="page-4-2"></span>Table 3: Computational overhead for different prompt compression methods. "Time per token" refers to the time taken divided by the number of tokens removed. All metrics are evaluated on a single A6000 GPU with 48 GB memory.

| Method            | Time per<br>Prompt (ms) | Time per<br>Token (ms) | Memory (MB) |  |
|-------------------|-------------------------|------------------------|-------------|--|
| KiS               | 2410                    | 5.03                   | 1378        |  |
| SCRL              | 67                      | 0.15                   | 315         |  |
| Selective Context | 319                     | 0.69                   | 487         |  |
| LLMLingua         | 180                     | 0.39                   | 5309        |  |
| LongLLMLingua     | 184                     | 0.40                   | 5309        |  |
| LLMLingua-2       | 115                     | 0.25                   | 2137        |  |

<span id="page-5-1"></span>> **[图片提取文字 (无描述)]:**
> 1.0 1.0 1.0 1.0 0.8 0.8 0.8 0.8 0.6 0.6 0.6 0.4 0.4 0.4 0.4 - BLEU - BLEU - BLEU - BLEU 0.2 ROUGE - ROUGE - ROUGE ROUGE - BERT - BERT - BERT - BERT 0.0 0.0 0.0 0.0 0.9 0.5 0.7 0.5 0.5 0.9 0.7 0.9 Compression Ratio Compression Ratio Compression Ratio Compression Ratio (a) Random Selection (b) Selective Context (d) LLMLingua-2 (c) (Long)LLMLingua
![](_page_5_Figure_1.jpeg)

Figure 3: **Performance on compression tasks under different compression ratios.** We measured the performance of four compression methods by changing the compression ratio while keeping all other settings in accordance with Table 1. For each dataset, we randomly sampled 100 instances for evaluation and averaged their metrics. As mentioned in Section 4.3, KiS and SCRL cannot adjust the compression ratio and are thus not considered.

Our main findings are the following:

- (Long)LLMLingua and LLMLingua-2 excel in summarization tasks, while Selective Context leads in reconstruction tasks. (Long)LLMLingua is best for math contexts (GSM8K), LLMLingua-2 for news articles (Gigaword, DUC2004), and Selective Context for human-centric datasets (BBC News, ShareGPT). We observed that (Long)LLMLingua and LLMLingua-2 retain tokens that are concentrated around semantically rich sections of the text, which helps in creating summaries that capture the essential points effectively. On the other hand, Selective Context retains tokens more evenly distributed across the text, which aids in reconstruction tasks.
- LongLLMLingua excels in QA tasks with longer contexts. This demonstrates its capacity to handle extensive information more effectively. For shorter contexts, performance varies across methods and datasets. Compared to short contexts, long contexts have the problem of diluting relevant information with irrelevant details. Unlike other methods, LongLLM-Lingua is question-aware, meaning it compresses prompts by considering the user's question in the prompt. We think that in long contexts, this approach helps to ensure that the most critical information related to the question is retained. This aligns with the ablation results from the LongLLMLingua paper regarding the question-aware mechanism.
- *SCRL offers the best computational efficiency*. As indicated in Table 3, SCRL achieves the lowest time cost and minimal memory consumption. This makes it a practical choice for real-world applications where computational resources are limited.

