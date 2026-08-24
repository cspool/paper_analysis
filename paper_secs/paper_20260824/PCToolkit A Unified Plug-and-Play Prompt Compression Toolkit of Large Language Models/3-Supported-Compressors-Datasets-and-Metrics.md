# 3 Supported Compressors, Datasets and Metrics

Table [1](#page-2-0) presents an overview of the supported tasks, compressors, and datasets within PCToolkit. Each

<span id="page-1-1"></span><sup>1</sup> <https://www.promptotype.io/>

<span id="page-2-0"></span>

| Tasks                  | Supported Compressors                   | Supported Datasets                                          |  |  |
|------------------------|-----------------------------------------|-------------------------------------------------------------|--|--|
| Reconstruction         | SC, LLMLingua, LongLLMLingua, SCRL, KiS | BBC, ShareGPT, Arxiv, GSM8K                                 |  |  |
| Mathematical promblems | SC, LLMLingua, LongLLMLingua, SCRL, KiS | GSM8K, BBH                                                  |  |  |
| Boolean expressions    | SC, LLMLingua, LongLLMLingua, SCRL, KiS | BBH                                                         |  |  |
| Multiple choice        | SC, LLMLingua, LongLLMLingua, SCRL, KiS | BBH                                                         |  |  |
| Lies recognition       | SC, LLMLingua, LongLLMLingua, SCRL, KiS | BBH                                                         |  |  |
|                        | SC, LLMLingua, LongLLMLingua, SCRL, KiS | BBC, Arxiv.<br>Gigaword, DUC2004,<br>BNC, Broadcast, Google |  |  |
| Summarization          | LLMLingua, LongLLMLingua                | LongBench                                                   |  |  |
|                        | SC, LLMLingua, LongLLMLingua, SCRL, KiS | BBH                                                         |  |  |
| Question and Answer    | LLMLingua, LongLLMLingua                | LongBench                                                   |  |  |
| Few-shot learning      | LLMLingua, LongLLMLingua                | LongBench                                                   |  |  |
| Synthetic tasks        | LLMLingua, LongLLMLingua                | LongBench                                                   |  |  |
| Code completion        | LLMLingua, LongLLMLingua                | LongBench                                                   |  |  |

Table 1: An overview of PCToolkit, including different evaluation tasks, compressors and datasets.

component are described in detail in Section 4 Toolkit Design. Evaluation of all compression methods across various datasets for different tasks is depicted in Table [6,](#page-9-0) with results to be discussed in Section 5 Evaluation.

### 3.1 Compressors

PCToolkit integrates 5 state-of-the-art prompt compression methods in total: Selective Context [\(Li](#page-7-7) [et al.,](#page-7-7) [2023\)](#page-7-7), LLMLingua [\(Jiang et al.,](#page-7-8) [2023a\)](#page-7-8), LongLLMLingua [\(Jiang et al.,](#page-7-9) [2023b\)](#page-7-9), SCRL [\(Gha](#page-7-10)[landari et al.,](#page-7-10) [2022\)](#page-7-10) and KiS [\(Laban et al.,](#page-7-11) [2021\)](#page-7-11). These compressors are plug-and-play implemented, therefore can be invoked directly.

Selective Context. Selective Context improves the context efficiency of LLMs in inference by removing redundant content measured by selfinformation [\(Shannon,](#page-7-14) [1948\)](#page-7-14).

LLMLingua. LLMLingua involves a budget controller to maintain semantic integrity under high compression ratios. LLMLingua compresses information within prompts by capitalizing on the compression-like characteristics of LLMs [\(Jiang](#page-7-8) [et al.,](#page-7-8) [2023a\)](#page-7-8).

LongLLMLingua. LongLLMLingua came into stage with an enhancement on dealing with the inherent challenge of the *lost in the middle* issue [\(Liu et al.,](#page-7-15) [2023\)](#page-7-15), which is a phenomenon that performance of LLM can degrade significantly when models must access relevant information in the middle of long contexts [\(Jiang et al.,](#page-7-9) [2023b\)](#page-7-9).

SCRL. SCRL is a reinforcement learning-based approach designed to remove or retain tokens according to the probabilities [\(Ghalandari et al.,](#page-7-10) [2022\)](#page-7-10).

KiS. KiS is an approach of unsupervised text simplification, which learns to balance a reward across three properties: fluency, salience and simplicity [\(Laban et al.,](#page-7-11) [2021\)](#page-7-11).

### 3.2 Datasets

Table [2](#page-3-0) shows all datasets supported in PCToolkit. GSM8K. GSM8K [\(Cobbe et al.,](#page-6-5) [2021\)](#page-6-5) contains 8.5K high-quality linguistically diverse word problems in elementary school mathematics. Each item contains a problem and its solution.

BBC News, Arxiv articles and ShareGPT. [Li](#page-7-7) [et al.](#page-7-7) [\(2023\)](#page-7-7) provided the three datasets. BBC News provides news articles from BBC, which is a typical context of human daily lives. Arxiv articles provides scientific articles that represents a formal context. ShareGPT contains contexts that is collected from human-AI conversations, which is a normal communication context.

Big Bench Hard (BBH). BBH [\(Suzgun et al.,](#page-7-16) [2022\)](#page-7-16) is a diverse evaluation suite that focuses on a suite of 23 challenging tasks from BIG-Bench that were found to be beyond the capabilities of current language models.

LongBench. LongBench [\(Bai et al.,](#page-6-6) [2023\)](#page-6-6) is the first benchmark for bilingual, multitask and comprehensive assessment of long context understanding capabilities of large language models. LongBench has six different task scenarios including single-document question & answer, multidocument question & answer, summarization, fewshot learning, synthetic tasks and code completion.

Gigaword, BNC, DUC2004, Broadcast and Google. [Ghalandari et al.](#page-7-10) [\(2022\)](#page-7-10) provided the five datasets. While Gigaword [\(Rush et al.,](#page-7-17) [2015\)](#page-7-17) and DUC2004 [\(Over and Yen.,](#page-7-18) [2004\)](#page-7-18) contain abstractive ground truth summaries, the remaining three datasets [\(Filippova and Altun,](#page-6-7) [2013;](#page-6-7) [Clarke and](#page-6-8) [Lapata,](#page-6-8) [2008\)](#page-6-8) have token-level extractive ground truth summaries.

<span id="page-3-0"></span>

| Datasets       | <b>Supporting Compressors</b> | <b>Supporting Metrics</b>                       |
|----------------|-------------------------------|-------------------------------------------------|
| BBH            | All                           | Accuracy                                        |
| Gigaword       | All                           | ROUGE, Token-F1                                 |
| BNC            | All                           | ROUGE, Token-F1                                 |
| DUC2004        | All                           | ROUGE, Token-F1                                 |
| Broadcast      | All                           | ROUGE, Token-F1                                 |
| Google         | All                           | ROUGE, Token-F1                                 |
| GSM8K          | All                           | Accuracy, BLEU, ROUGE, BERTScore                |
| BBC News       | All                           | BLEU, ROUGE, BERTScore                          |
| Arxiv articles | All                           | BLEU, ROUGE, BERTScore                          |
| ShareGPT       | All                           | BLEU, ROUGE, BERTScore                          |
| LongBench      | LLMLingua, LongLLMLingua      | Accuracy, BLEU, ROUGE, BERTScore, Edit-distance |

Table 2: Datasets and corresponding compressors and metrics supported in PCToolkit.

#### 3.3 Metrics

PCToolkit provides different metrics, including BLEU, ROUGE, BERTScore, Edit distance and Accuracy. The first four metrics are used to compare the difference between two strings, while Accuracy judges the results provided by LLM with the ground truth answer.

**BLEU.** Proposed by Papineni et al. (2002), Bilingual Evaluation Understudy (BLEU) is a metric used to evaluate machine-translated text by comparing it to reference translations (Papineni et al., 2002; Li et al., 2023).

**ROUGE.** Proposed by Lin (2004), Recall-Oriented Understudy for Gisting Evaluation (ROUGE) is a set of metrics used for evaluating the quality of summaries produced by automatic summarization systems (Lin, 2004; Li et al., 2023; Bai et al., 2023).

**BERTScore.** Proposed by Zhang\* et al. (2020), BERTScore evaluates text similarity using contextual embeddings from BERT (Devlin et al., 2019). It measures the similarity between reference and candidate sentences, providing a score between 0 and 1, where 1 indicates perfect semantic similarity (Zhang\* et al., 2020; Li et al., 2023).

Edit distance. Edit distance (Levenshtein distance) is popularly used in code generation evaluation (Svyatkovskiy et al., 2020; Yujian and Bo, 2007). Edit Distance is a metric used to quantify the difference between two sequences of strings (Bai et al., 2023).

#### 4 Toolkit Design

#### 4.1 Modular Design

As shown in Figure 1, PCToolkit is designed with a modular architecture, consisting of Compressor, Dataset, Metrics and Runner.

**Compressors.** pctoolkit.compressors module in PCToolkit encompasses five state-of-the-art compression methods tailored for prompt optimization. All compressors can be invoked through a unified interface shown in **Section 4.2**.

**Datasets.** pctoolkit.datasets module boasts a diverse collection of over ten datasets, each meticulously curated to cover a wide array of natural language tasks. As shown in Table 3, from tasks like reconstruction, summarization, question answering, to more specialized domains such as code completion and lies recognition, the datasets in PCToolkit offer a comprehensive testing ground for assessing the efficacy of prompt compression techniques.

**Metrics.** pctoolkit.metrics module plays a crucial role in quantifying the performance of the compression methods across different tasks. All metrics needed can be easily contained inside a list that tells the Runner which metrics are required measuring.

Runners. pctoolkit.runners module serves as the engine that drives the evaluation process, orchestrating the interaction between the compression methods, datasets, and evaluation metrics. Researchers and practitioners can seamlessly execute experiments, compare results, and analyze the performance of different compression techniques using the Runner component. This streamlined workflow ensures efficient experimentation and evaluation of prompt compression strategies within the toolkit.

By integrating these components, PCToolkit offers a comprehensive and user-friendly platform for prompt compression and evaluation, empowering researchers and practitioners to optimize prompts for enhanced model performance in natural language processing tasks.

<span id="page-4-0"></span>

| Algorithms Metrics |              | Datasets |          |                                |          |                  |                  |                  |                  |                  |
|--------------------|--------------|----------|----------|--------------------------------|----------|------------------|------------------|------------------|------------------|------------------|
| Algoriums Wettes   | GSM8K        | BBC News | ShareGPT | Arxiv                          | Gigaword | DUC2004          | BNC              | Broadcast        | Google           |                  |
| Selective Context  |              | 0.56     | 0.32     | <b>0.37</b> <sub>(+0.12)</sub> | 0.29     | 0.24             | 0.24             | 0.54             | 0.45             | 0.45             |
| (Long)LLMLingua    | BLEU         | 0.78     | 0.17     | $0.25_{(-0.02)}$               | 0.12     | 0.20             | 0.21             | 0.69             | 0.81             | 0.41             |
| SCRL               | BLEU         | 0.34     | 0.02     | 0.27                           | 0.05     | 0.26             | 0.25             | 0.55             | 0.45             | 0.45             |
| KiS                |              | 0.52     | 0.02     | 0.07                           | 0.00     | 0.17             | 0.20             | 0.50             | 0.45             | 0.33             |
| Selective Context  |              | 0.82     | 0.69     | <b>0.70</b> <sub>(+0.33)</sub> | 0.57     | 0.19             | 0.14             | 0.58             | 0.57             | 0.51             |
| (Long)LLMLingua    | ROUGE L      | 0.90     | 0.50     | $0.56_{(+0.17)}$               | 0.42     | 0.21             | 0.17             | 0.82             | 0.90             | 0.57             |
| SCRL               | KOUGE L      | 0.53     | 0.27     | 0.58                           | 0.24     | $0.21_{(-0.02)}$ | $0.13_{(-0.09)}$ | $0.41_{(-0.38)}$ | $0.41_{(-0.41)}$ | $0.36_{(-0.34)}$ |
| KiS                |              | 0.73     | 0.31     | 0.32                           | 0.08     | 0.17             | 0.16             | 0.60             | 0.58             | 0.45             |
| Selective Context  | Bertscore P  | 0.96     | 0.89     | 0.90                           | 0.93     | 0.86             | 0.86             | 0.89             | 0.88             | 0.92             |
| (Long)LLMLingua    |              | 0.98     | 0.86     | 0.89                           | 0.88     | 0.88             | 0.89             | 0.97             | 0.98             | 0.96             |
| SCRL               |              | 0.68     | 0.81     | 0.87                           | 0.85     | 0.86             | 0.86             | 0.86             | 0.82             | 0.91             |
| KiS                |              | 0.95     | 0.83     | 0.81                           | 0.80     | 0.87             | 0.89             | 0.93             | 0.93             | 0.95             |
| Selective Context  |              | 0.97     | 0.91     | 0.92                           | 0.92     | 0.83             | 0.84             | 0.90             | 0.90             | 0.87             |
| (Long)LLMLingua    | D D          | 0.98     | 0.89     | 0.91                           | 0.90     | 0.83             | 0.85             | 0.94             | 0.96             | 0.89             |
| SCRL               | Bertscore R  | 0.70     | 0.88     | 0.92                           | 0.86     | 0.82             | 0.82             | 0.85             | 0.83             | 0.85             |
| KiS                |              | 0.95     | 0.93     | 0.90                           | 0.84     | 0.84             | 0.86             | 0.91             | 0.90             | 0.89             |
| Selective Context  | Bertscore F1 | 0.97     | 0.90     | <b>0.91</b> <sub>(+0.02)</sub> | 0.92     | 0.85             | 0.85             | 0.89             | 0.89             | 0.89             |
| (Long)LLMLingua    |              | 0.98     | 0.88     | $0.90_{(+0.05)}$               | 0.89     | 0.85             | 0.87             | 0.96             | 0.97             | 0.93             |
| SCRL               |              | 0.69     | 0.84     | 0.89                           | 0.85     | 0.84             | 0.84             | $0.85_{(+0.09)}$ | $0.82_{(+0.03)}$ | $0.88_{(+0.11)}$ |
| KiS                |              | 0.95     | 0.88     | 0.85                           | 0.82     | 0.85             | 0.87             | 0.92             | 0.91             | 0.92             |

Table 3: Performance measured for reconstruction and summarization tasks in PCToolkit. (Long)LLMLingua means we considered LLMLingua and LongLLMLingua together, since for small scale datas, these two compressors showed very slight differences. Numbers in parenthesis are the difference between the original results provided by former experiments and our results.

#### 4.2 Unified Interface

In PCToolkit, a unified interface for invoking prompt compression methods is provided. In the following example, we show how to simply invoke the compressing methods within few lines.

```
from pctoolkit.compressors import
    PromptCompressor

compressor = PromptCompressor(
type='SCCompressor', device='cuda')

test_prompt = "test prompt"
ratio = 0.3
result = compressor.compressgo(
test_prompt, ratio)
print(result)
```

Different parameters for compressors can be included inside compressgo.

For simple compression task, one compressor is selected. Following the example given above, an original prompt is input to the compressor, and the compressor outputs the target compressed prompt. For datasets evaluation, one datasets and multiple metrics are selected, along with the compressor chosen, these three parts are deployed in Runner. The Runner will provide the evaluation results according to the metrics list, which includes all metrics expected. The following example shows how to modularistically use PCToolkit.

```
from pctoolkit.runners import run
from pctoolkit.datasets import
    load_dataset
from pctoolkit.metrics import
    load_metrics
```

```
compressor = PromptCompressor(
type='SCCompressor', device='cuda')
dataset_name = 'arxiv'
dataset = load_dataset(dataset_name)

run(compressor=compressor,
dataset=dataset,
metrics=load_metrics, ratio=0.1)
```

Currently, the supporting datasets calls are implemented inside run. Users can also following the format in run to adapt their own datasets or metrics.

#### 5 Evaluation

Compression ratio. Following Li et al. (2023), we define the compression ratio to be the ratio of reduced context length comparing with the original context length. That is, ratio  $\rho=1-\frac{L_c}{L_O}$  where  $L_c$  represents the length of compressed context and  $L_o$  represents the length of original context. Compression ratio is an essential parameter that measures how much deletion is needed for a prompt.

#### **5.1** Short Context Tasks

We conducted evaluations using different datasets as outlined in Table 1 and assessed them across various metrics. The results, presented in Table 3, utilized metrics like BLEU, ROUGE, and BERTScore for testing tasks that do not have a definitive answer, such as reconstruction and summarization. Following the methodologies of Li et al. (2023) and Jiang et al. (2023a), GPT-3.5-Turbo was employed

<span id="page-5-1"></span>

| Compressors   | LongBench <sup>3</sup> |                  |                  |                  |                  |                  |                  |       |
|---------------|------------------------|------------------|------------------|------------------|------------------|------------------|------------------|-------|
| Compressors   | SingleDoc              | MultiDoc         | Summ.            | FewShot          | Synth.           | Code             | AVG              | Ratio |
| LLMLingua     | $0.30_{(-0.01)}$       | $0.34_{(-0.04)}$ | $0.22_{(-0.04)}$ | $0.63_{(-0.04)}$ | $0.11_{(+0.03)}$ | $0.37_{(-0.16)}$ | $0.33_{(-0.04)}$ | 0.66  |
| LongLLMLingua | $0.41_{(+0.01)}$       | $0.39_{(-0.07)}$ | $0.22_{(-0.05)}$ | $0.63_{(-0.08)}$ | $0.75_{(+0.22)}$ | $0.42_{(-0.13)}$ | $0.47_{(-0.02)}$ | 0.66  |
| LLMLingua     | $0.26_{(+0.04)}$       | $0.36_{(+0.04)}$ | $0.22_{(-0.03)}$ | $0.60_{(-0.01)}$ | $0.10_{(0.00)}$  | $0.36_{(-0.21)}$ | $0.32_{(-0.03)}$ | 0.80  |
| LongLLMLingua | $0.38_{(-0.01)}$       | $0.39_{(-0.03)}$ | $0.22_{(-0.05)}$ | $0.62_{(-0.07)}$ | $0.60_{(+0.04)}$ | $0.37_{(-0.2)}$  | $0.43_{(-0.05)}$ | 0.80  |

Table 4: Performance measured in LongBench datasets. Numbers in parenthesis are the difference between the original results provided by former experiments and our results. We evaluated each task by the metric provided by LongBench.

as a frozen LLM for reconstruction tasks. It received compressed prompts from the compressors and generated reconstructed prompts, which were then compared with the compressed ones. For summarization tasks, the frozen LLM provided a pair of summaries, one from the original context and the other from the compressed context. These pairs of summaries were evaluated using the specified metrics. In our experiment, datasets like GSM8K, BBC News, and ShareGPT were designated for reconstruction tasks, while the rest were assigned to summarization tasks.

For tasks with precise answers, such as mathematical problems, metrics like accuracy and edit distance are commonly used. As shown in Table 5, we tested all compression methods across various task types. The GSM8K dataset includes mathematical problems, while BBH encompasses a diverse range of tasks. For instance, the Boolean Expression task requires the LLM to provide answers to specific logical expressions; the Movie Recommendation task tasks the LLM with selecting the most suitable movie from a list based on a given description; and the Web of Lies task involves the LLM determining if a particular character is lying.

#### **5.2** Long Context Tasks

For datasets that contains longer contexts, we evaluate LLMLingua and LongLLMLingua on them. With specified questions, LongLLMLingua performed much better than LLMLingua. Results are shown in Table 4. The evaluation settings are different from original ones, as the authors of LLM-Lingua mentioned on GitHub<sup>2</sup>, they used the completion mode of GPT-3.5-turbo, which is recently disabled by OpenAI. Thus, we used the chat mode instead, which caused a little deviation from the original results.

<span id="page-5-0"></span>

| Compressors       | Dataset         | Accuracy                       |  |
|-------------------|-----------------|--------------------------------|--|
| Baseline          |                 | 0.51                           |  |
| Selective Context | BBH Boolean     | <b>0.54</b> <sub>(+0.03)</sub> |  |
| (Long)LLMLingua   | Expression      | $0.54_{(+0.03)}$               |  |
| SCRL              | Expression      | $0.54_{(+0.03)}$               |  |
| KiS               |                 | <b>0.54</b> <sub>(+0.03)</sub> |  |
| Baseline          |                 | 0.33                           |  |
| Selective Context | BBH Movie       | $0.63_{(+0.30)}$               |  |
| (Long)LLMLingua   | Recommendation  | $0.67_{(+0.34)}$               |  |
| SCRL              | Recommendation  | $0.59_{(+0.26)}$               |  |
| KiS               |                 | 0.48(+0.15)                    |  |
| Baseline          |                 | 0.89                           |  |
| Selective Context |                 | $0.39_{(-0.50)}$               |  |
| (Long)LLMLingua   | BBH Web of Lies | $0.62_{(-0.27)}$               |  |
| SCRL              |                 | $0.31_{(-0.58)}$               |  |
| KiS               |                 | $0.41_{(-0.48)}$               |  |
| Baseline          |                 | 0.29                           |  |
| Selective Context |                 | $0.09_{(-0.20)}$               |  |
| (Long)LLMLingua   | GSM8K           | $0.25_{(-0.04)}$               |  |
| SCRL              |                 | $0.05_{(-0.24)}$               |  |
| KiS               |                 | $0.13_{(-0.16)}$               |  |
|                   |                 |                                |  |

Table 5: Performance measured in BBH & GSM8K datasets. Our baseline is the performance without using any compression methods. Numbers in parenthesis are the difference between the baseline results and each results with different compressors.

## 6 Conclusion and Future work

In conclusion, we introduced PCToolkit, an opensource project designed for prompt compression and evaluation. This toolkit offers researchers and practitioners a user-friendly and comprehensive resource, featuring five cutting-edge compression methods and over ten diverse datasets encompassing a wide range of natural language tasks. Through rigorous evaluations across various tasks such as reconstruction, summarization, mathematical problem-solving, question answering, few-shot learning, and more, we demonstrated the effectiveness and versatility of the compression techniques integrated into PCToolkit.

Our future endeavors focus on expanding PC-Toolkit with more compression methods, datasets, and evaluation metrics to further enhance its capabilities for prompt compression and model optimization in natural language processing.

<span id="page-5-2"></span><sup>2</sup>https://github.com/microsoft/LLMLingua/blob/ main/Transparency\_FAQ.md

<sup>&</sup>lt;sup>3</sup>https://github.com/THUDM/LongBench

## 7 Broader Impact

The findings and methodologies presented in this study have broader implications for the field of natural language processing (NLP) and the development of language models. By exploring and evaluating various compression techniques within the PCToolkit, we contribute to the ongoing efforts to enhance the efficiency and performance of large-scale language models. The insights gained from this research can potentially inform the design of more streamlined and effective compression methods, paving the way for advancements in NLP applications across diverse domains.

Furthermore, the development of optimized compression methods could lead to more sustainable and eco-friendly practices in AI research and deployment. By reducing the computational resources required for training and inference, we may contribute to a more energy-efficient and costeffective utilization of AI technologies.

## 8 Limitations

Despite the advancements made in this study, there are inherent limitations that should be acknowledged. One notable limitation is that the PCToolkit, while effective in compressing prompts and enhancing model performance, may still face challenges in handling toxic or harmful content present in NLP datasets. The toolkit's current capabilities may not extend to effectively filtering out such content, highlighting the ongoing need for robust ethical guidelines and content moderation strategies in NLP research.

Additionally, the generalizability of the compression techniques evaluated in this study may be limited to specific task domains or dataset characteristics. Further research is needed to explore the scalability and adaptability of these methods across a wider range of tasks and datasets to fully assess their utility and effectiveness in diverse applications.

Overall, while the PCToolkit offers valuable tools for prompt compression and model optimization, researchers and practitioners are encouraged to remain vigilant about the broader impacts and limitations associated with the use of such technologies in NLP research and development.

