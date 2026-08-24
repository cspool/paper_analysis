# 4 Experimental Setup

### <span id="page-4-0"></span>4.1 Evaluation Dataset

We evaluated the performance of xRAG primarily on knowledge-intensive tasks which encompass a range of challenges: (1) three Open Domain Question Answering datasets that address questions on a wide array of topics: Natural Questions [\[40\]](#page-11-8), TriviaQA [\[32\]](#page-11-9), and Web Questions [\[7\]](#page-9-6). (2) one Multihop Question Answering dataset, HotpotQA [\[81\]](#page-14-3), which necessitates multi-step reasoning to generate answers. (3) one Long-form Question Answering dataset, TruthfulQA [\[49\]](#page-12-14), that requires the generation of long-form and truthful responses. (4) one fact-checking dataset, FactKG [\[38\]](#page-11-10), which challenges the model to use complex reasoning to determine the factual accuracy of given claims.

In line with the KILT [63] and GenRead [84], we assessed three ODQA datasets and HotpotQA using the Exact Match (EM) metric, FactKG with Accuracy, and for the long-form QA, we used both the F1 score and Rouge-L (R-L) score. These tasks demand a broad spectrum of world knowledge and have been extensively explored in the retrieval-augmentation literature [41, 21, 34, 9, 24, 70].

#### 4.2 Implementation Details

To demonstrate the versatility of our framework, we choose two backbones, differing in scale and architecture: Mistral-7b [26] and Mixtral-8x7b [27]. For the retrieval corpus, we utilized the Wikipedia dump from December 2021, which was pre-processed into passages following the methodology described in [25]. This resulted in approximately 37 million passages, each averaging 180 tokens in length. Our default retrieval model is the SFR [60], which, at the time of writing this paper, holds the leading position on the MTEB leaderboard [59]. We use *top-1* ranked document for inclusion in our instruction-tuning dataset and for the evaluation of downstream tasks. More details are provided in Appendix D. Code is available at: https://github.com/Hannibal046/xRAG.

#### 4.3 Baselines

In determining appropriate baselines for comparison, we adhered to a fundamental principle: the selected compression methods must support the general plug-and-play capability of retrieval augmentation. This entails that they should function effectively without the need for dataset-specific tuning [79, 76], or any alteration to the parameters of LLMs [76, 58]. Furthermore, given the extensive volume of the retrieval corpus, it is essential that these compression methods demonstrate memory efficiency, specifically by not requiring the storage of LLM activations for each individual token [58, 19, 14]. With these criteria in mind, our evaluation compares xRAG to the following baselines: (I) Primarily, we consider two variants of LLMs: one that operates without retrieval augmentation and another that includes it. These serve as the lower and upper performance bounds for our study of compression techniques, respectively. (II) Additionally, our comparisons extend to LLMLingua [28], a plug-and-play approach for context compression. (III) Taking inspiration from [58], we incorporate a method of discrete compression using TF-IDF. This approach yields compression rates comparable to those achieved by xRAG and serves as the lower bound for discrete compression.

<span id="page-5-0"></span>Table 1: Experimental results on six downstream tasks. The best results are in **bold** and the second best are with <u>underscore</u>. Percentage in the brackets denotes the relative improvement over non-retrieval setting. LLMs are frozen during the experiments and retrieved documents are set the same for different compression methods. ‡ and † denotes different compression ratio.

| Task Type              | NQ (  | <b>TriviaQA</b><br>Open-Domain<br>(EM) | <b>WebQA</b><br>QA | HotpotQA<br>Multihop QA<br>(EM) | Long-f | <b>fulQA</b><br>orm QA<br>R-L) | FactKG<br>Fact Checking<br>(Acc) | Average              | # Doc<br>Length |
|------------------------|-------|----------------------------------------|--------------------|---------------------------------|--------|--------------------------------|----------------------------------|----------------------|-----------------|
| Mistral-7b             |       |                                        |                    |                                 |        |                                |                                  |                      |                 |
| w/o retrieval          | 30.25 | 57.08                                  | 34.89              | 27.02                           | 26.23  | 25.51                          | 54.78                            | 36.54 (0.0%)         | 0               |
| w retrieval            | 42.71 | 65.88                                  | 37.84              | 38.79                           | 26.50  | 25.92                          | 67.76                            | <b>43.63</b> (19.4%) | 175.1           |
| *with Compre           | ssion |                                        |                    |                                 |        |                                |                                  |                      |                 |
| LLMLingua †            | 30.64 | 57.94                                  | 32.63              | 29.91                           | 25.70  | 25.10                          | 64.17                            | 38.01 (4.0%)         | 98.6            |
| LLMLingua ‡            | 28.81 | 57.09                                  | 32.33              | 29.13                           | 26.10  | 25.39                          | 63.57                            | 37.48 (2.5%)         | 61.1            |
| TF-IDF                 | 30.25 | 58.49                                  | 35.43              | 26.62                           | 26.33  | 25.83                          | 59.56                            | 37.49 (2.6%)         | 1               |
| xRAG                   | 39.10 | <u>65.77</u>                           | 39.40              | 34.05                           | 28.10  | 27.71                          | <u>63.08</u>                     | <u>42.46</u> (16.2%) | 1               |
| Mixtral-8x7b           |       |                                        |                    |                                 |        |                                |                                  |                      |                 |
| w/o retrieval          | 41.99 | 71.10                                  | 40.31              | 32.87                           | 25.60  | 24.90                          | 62.64                            | 42.76 (0.0%)         | 0               |
| w retrieval            | 45.15 | 70.34                                  | 41.26              | 43.46                           | 27.10  | 25.80                          | 70.42                            | 46.22 (8.0%)         | 175.1           |
| *with Compre           | ssion |                                        |                    |                                 |        |                                |                                  |                      |                 |
| LLMLingua†             | 37.65 | 67.70                                  | 36.02              | 35.66                           | 25.99  | 25.39                          | 67.98                            | 42.32 (-1.0%)        | 96.6            |
| LLMLingua <sup>‡</sup> | 37.81 | 67.81                                  | 35.78              | 35.27                           | 25.68  | 25.00                          | 68.03                            | 44.17 (-1.3%)        | 61.1            |
| TF-IDF                 | 41.19 | 69.94                                  | 41.63              | 32.05                           | 26.80  | 26.00                          | 66.17                            | 43.41 (1.4%)         | 1               |
| xRAG                   | 47.28 | 74.14                                  | 44.50              | 39.66                           | 27.80  | 26.64                          | 68.20                            | <b>46.91</b> (9.7%)  | 1               |

### 5 Experimental Results

### <span id="page-5-1"></span>5.1 Knowledge Intensive Tasks

In Table 1, we present our main results. Across both Mistral-7b and Mixtral-8x7b configurations, we observe a consistent and significant improvement when retrieval augmentation is applied (p-value

< 0.05), although the gains are more modest for the larger model configurations. This trend aligns with observations reported by [\[70,](#page-13-0) [50\]](#page-12-6). Further analysis on the efficacy of various compression techniques reveals that xRAG outperforms other approaches by a large margin. Remarkably, xRAG not only reduces the token count drastically—from 175.1 to a single token—but also maintains robust performance levels. In some instances, xRAG's performance is comparable to, or even exceeds, that of the uncompressed models. Specifically, in the Mistral-7b configuration, xRAG achieves nearly the same performance improvement as the uncompressed model (16.6% compared to 19.4%), and in the Mixtral-8x7b configuration, it surpasses the uncompressed model (9.7% compared to 8.0%). One possible reason lies in the vulnerability of current RAG system when the irrelevant or misleading documents are presented, a topic detailed discussed in [§6.1.](#page-6-0) We also observe that xRAG performs well in tasks that require document understanding, such as TriviaQA. However, in tasks that demand reasoning over document, like HotpotQA and FactKG, xRAG lags behind by a considerable gap.

<span id="page-6-2"></span>

| Table 2: Comparison of RAG and xRAG performance in CUDA Time and GFLOPS. |  |  |
|--------------------------------------------------------------------------|--|--|
|                                                                          |  |  |

|          |       | CUDA Time (ms) |             | GFLOPs |        |             |  |
|----------|-------|----------------|-------------|--------|--------|-------------|--|
|          | RAG   | xRAG           | Improvement | RAG    | xRAG   | Improvement |  |
| FactKG   | 431.5 | 215.6          | x2.01       | 4683.8 | 1289.5 | x3.63       |  |
| NQ       | 918.7 | 611.3          | x1.51       | 1338.6 | 384.0  | x3.48       |  |
| TriviaQA | 807.1 | 512.1          | x1.57       | 1667.2 | 492.3  | x3.38       |  |
| WebQA    | 872.6 | 577.3          | x1.51       | 1405.1 | 386.8  | x3.63       |  |
| Average  |       |                | x1.64       |        |        | x3.53       |  |

### <span id="page-6-3"></span>5.2 Computational Efficiency

In this section, we conduct a thorough assessment of our framework's computational efficiency and memory management. To rigorously evaluate our model, we employed Torch Profiler[3](#page-6-1) to measure the CUDA Time (milliseconds) and Giga FLOPs of both the RAG and xRAG models across four real-world datasets. In these evaluations, the Mistral-7b, operating in bfloat16 inference mode, served as the base LLM. CUDA Time and GFLOPs were calculated on an average per batch basis with a fixed batch size, and GFLOPs were normalized by the number of generated tokens. These experiments were performed on the same computational hardware, specifically an Nvidia A100 and an AMD EPYC 7V12 64-Core Processor. As depicted in Table [2,](#page-6-2) despite variations in prompt and generation lengths across the datasets, xRAG significantly outpaced the RAG model, achieving a x1.64 increase in CUDA Time efficiency and a x3.53 reduction in GFLOPs.

