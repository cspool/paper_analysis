# <span id="page-16-0"></span>C Additional Experimental Results and Analyses

Table 21 shows detailed results for each dataset and model configuration under zero-shot QA prompts, using both Top-5 and Top-20 retrieval. For the few-shot QA prompts, Table 22 summarizes the results, where we randomly selected five training examples per dataset as demonstrations. Table 23 provides

<span id="page-16-3"></span>Table 7: A prompt template used by the reader model for the QA task.

| QA Prompt Template                                                                                                                |
|-----------------------------------------------------------------------------------------------------------------------------------|
| Context information is below.                                                                                                     |
| {context}                                                                                                                         |
| Given the context information and not prior knowledge, answer the query. Do not provide any explanation.  Query: {query}  Answer: |

<span id="page-16-4"></span>Table 8: Performance comparison between in-domain (HQA) and out-of-domain (2WIKI) datasets using BM25 as the retriever model. Best results are highlighted in **bold**, and second best results are underlined.

| Compressor   Type |                 |      | ]     | HQA            | 2WIKI       |             |                |  |  |
|-------------------|-----------------|------|-------|----------------|-------------|-------------|----------------|--|--|
|                   | '               | ЕМ↑  | F1↑   | #token (%) ↓   | ЕМ↑         | Fl↑         | #token (%) ↓   |  |  |
|                   | Top-5 Documents |      |       |                |             |             |                |  |  |
| Original Docs     | -               | 28.2 | 39.1  | 755.8 (100.0)  | 19.6        | 25.9        | 789.7 (100.0)  |  |  |
| RECOMP-Abst       | Abs.            | 27.8 | 39.2  | 63.0 (8.3)     | 25.0        | 30.6        | 55.7 (7.1)     |  |  |
| CompAct           | Abs.            | 30.6 | 41.2  | 76.9 (10.2)    | 19.6        | 29.1        | 71.0 (9.0)     |  |  |
| Refiner           | Abs.            | 30.8 | 42.3  | 84.0 (11.1)    | 19.6        | 27.8        | 62.9 (8.0)     |  |  |
| RECOMP-Extr       | Ext.            | 28.2 | 37.5  | 93.1 (12.3)    | 11.8        | 19.1        | 99.1 (12.5)    |  |  |
| LongLLMLingua     | Ext.            | 28.6 | 39.7  | 230.3 (30.5)   | 22.2        | 27.4        | 236.6 (30.0)   |  |  |
| EXIT (Ours)       | Ext.            | 33.4 | 44.5  | 177.8 (23.5)   | <u>24.4</u> | <u>29.1</u> | 138.2 (17.5)   |  |  |
|                   |                 |      | Top-2 | 0 Documents    |             |             |                |  |  |
| Original Docs     | -               | 31.2 | 42.5  | 3009.5 (100.0) | 23.0        | 30.6        | 3132.5 (100.0) |  |  |
| RECOMP-Abst       | Abs.            | 29.0 | 39.7  | 69.7 (2.3)     | 22.8        | 28.2        | 52.3 (1.7)     |  |  |
| CompAct           | Abs.            | 31.6 | 43.0  | 109.5 (3.6)    | 20.4        | 28.7        | 113.2 (3.6)    |  |  |
| Refiner           | Abs.            | 28.4 | 38.8  | 136.1 (4.5)    | 18.8        | 26.9        | 108.4 (3.5)    |  |  |
| RECOMP-Extr       | Ext.            | 28.4 | 37.0  | 93.5 (3.1)     | 11.2        | 19.3        | 96.7 (3.1)     |  |  |
| LongLLMLingua     | Ext.            | 31.0 | 40.7  | 558.0 (18.5)   | 23.6        | 28.3        | 593.7 (19.0)   |  |  |
| EXIT (Ours)       | Ext.            | 35.2 | 46.9  | 411.3 (13.7)   | 27.2        | 32.4        | 312.7 (10.0)   |  |  |

comprehensive token count statistics, and Table 24 breaks down end-to-end latency.

#### **C.1** Performance with Sparse Retrieval

To assess EXIT's robustness with different retrieval architectures, we evaluate it with BM25, a sparse retrieval method. Table 8 compares EXIT's performance on HQA (in-domain) and 2WIKI (out-of-domain) under Top-5 and Top-20 retrieval settings.

With Top-5 retrieval, EXIT shows notable gains on HQA, improving EM (33.4 vs. 28.2) and F1 (44.5 vs. 39.1) over the uncompressed baseline. Although RECOMP-Abst performs best on 2WIKI (25.0 EM, 30.6 F1), EXIT remains competitive (24.4 EM, 29.1 F1).

EXIT's advantages grow with Top-20 retrieval. On HQA, EXIT outperforms all baselines, improving EM by 4.0 points (35.2 vs. 31.2) and F1 by

<span id="page-17-0"></span>Table 9: Performance comparison between in-domain (HQA) and out-of-domain (2WIKI) datasets using GPT-40 as the reader model. Best results are highlighted in **bold**, and second best results are <u>underlined</u>.

| Compressor      | Туре |             | ]           | HQA                     | 2WIKI       |             |                         |  |
|-----------------|------|-------------|-------------|-------------------------|-------------|-------------|-------------------------|--|
|                 |      | ЕМ↑         | F1↑         | #token (%) $\downarrow$ | ЕМ↑         | Fl ↑        | #token (%) $\downarrow$ |  |
| Top-5 Documents |      |             |             |                         |             |             |                         |  |
| Original Docs   | -    | 37.2        | 48.6        | 735.3 (100.0)           | 31.2        | 35.3        | 764.5 (100.0)           |  |
| RECOMP-Abst     | Abs. | 29.4        | 40.2        | 62.8 (8.5)              | 23.8        | 27.8        | 53.5 (7.0)              |  |
| CompAct         | Abs. | <u>37.4</u> | 48.0        | 74.3 (10.1)             | 30.0        | 33.6        | 67.6 (8.8)              |  |
| Refiner         | Abs. | 35.8        | 47.5        | 71.4 (9.7)              | 27.8        | 32.6        | 54.2 (7.1)              |  |
| RECOMP-Extr     | Ext. | 32.4        | 41.7        | 87.1 (11.8)             | 25.6        | 28.6        | 93.6 (12.2)             |  |
| LongLLMLingua   | Ext. | 34.2        | 45.2        | 223.1 (30.3)            | 30.6        | 34.5        | 230.5 (30.2)            |  |
| EXIT (Ours)     | Ext. | 38.2        | 50.4        | 191.2 (26.0)            | 31.8        | 35.8        | 145.6 (19.0)            |  |
|                 |      |             | Top-2       | 0 Documents             |             |             |                         |  |
| Original Docs   | -    | 39.6        | 51.8        | 2940.5 (100.0)          | 40.0        | 43.8        | 3066.2 (100.0)          |  |
| RECOMP-Abst     | Abs. | 33.6        | 44.2        | 62.7 (2.1)              | 26.6        | 32.1        | 48.9 (1.6)              |  |
| CompAct         | Abs. | 33.0        | 43.7        | 106.0 (3.6)             | 23.0        | 27.3        | 105.1 (3.4)             |  |
| Refiner         | Abs. | 31.8        | 41.5        | 130.6 (4.4)             | 31.0        | 35.5        | 100.3 (3.3)             |  |
| RECOMP-Extr     | Ext. | 31.2        | 39.6        | 86.2 (2.9)              | 23.2        | 27.2        | 91.0 (3.0)              |  |
| LongLLMLingua   | Ext. | 38.8        | 49.4        | 549.5 (18.7)            | 35.4        | 39.6        | 581.5 (19.0)            |  |
| EXIT (Ours)     | Ext. | <u>39.4</u> | <u>50.1</u> | 453.6 (15.4)            | <u>35.6</u> | <u>40.3</u> | 346.7 (11.3)            |  |

4.4 points (46.9 vs. 42.5) compared to using uncompressed documents. On 2WIKI, EXIT achieves the highest scores (27.2 EM, 32.4 F1), confirming its generalizability across domains and retrieval strategies.

#### **C.2** Performance with Proprietary Model

We further examine EXIT's effectiveness using GPT-40 as the reader. Table 9 compares performance on HQA (in-domain) and 2WIKI (out-of-domain), along with compression rates.

For Top-5 retrieval, EXIT attains the best accuracy on HQA (38.2 EM, 50.4 F1) while retaining only 26.0% of tokens. This surpasses the uncompressed baseline (37.2 EM, 48.6 F1) with a 74% token reduction. On 2WIKI, EXIT maintains leading accuracy (31.8 EM, 35.8 F1) while using just 19.0% of the original tokens.

Under Top-20 retrieval, where uncompressed documents benefit from greater coverage, EXIT still achieves competitive accuracy with substantially fewer tokens. On HQA, EXIT closely matches the uncompressed EM score (39.4 vs. 39.6) while using only 15.4% of tokens. Although RECOMP variants compress more aggressively, they suffer marked performance drops. LongLLM-Lingua performs similarly to EXIT but retains more tokens (18.7% vs. 15.4%).

These findings illustrate EXIT's ability to balance performance and efficiency, making it valuable for API-based proprietary models where token costs and accuracy both matter.

<span id="page-17-1"></span>Table 10: Evaluation of EXIT and baselines in specialized domains. BioASQ evaluates biomedical QA using PubMed documents, and COVID-QA focuses on COVID-specific QA over the TREC-COVID corpus.

| BioASC        | BioASQ (Biomedical QA) |         |              |  |  |  |  |  |
|---------------|------------------------|---------|--------------|--|--|--|--|--|
| Method        | EM                     | F1      | Avg. #Tokens |  |  |  |  |  |
| Original Docs | 57.2                   | 68.6    | 1791.49      |  |  |  |  |  |
| RECOMP-Abst   | 54.6                   | 63.8    | 37.72        |  |  |  |  |  |
| CompAct       | 58.0                   | 68.2    | 94.81        |  |  |  |  |  |
| Refiner       | 58.8                   | 68.5    | 229.22       |  |  |  |  |  |
| RECOMP-Extr   | 55.8                   | 65.0    | 109.72       |  |  |  |  |  |
| LongLLMLingua | 58.2                   | 66.4    | 364.06       |  |  |  |  |  |
| EXIT (Ours)   | 58.6                   | 67.1    | 132.28       |  |  |  |  |  |
| COVID-        | QA (CC                 | OVID-19 | 9 QA)        |  |  |  |  |  |
| Method        | EM                     | F1      | Avg. #Tokens |  |  |  |  |  |
| Original Docs | 3.8                    | 16.0    | 1775.66      |  |  |  |  |  |
| RECOMP-Abst   | 2.2                    | 11.5    | 17.94        |  |  |  |  |  |
| CompAct       | 3.4                    | 14.6    | 95.65        |  |  |  |  |  |
| Refiner       | 3.8                    | 14.6    | 201.38       |  |  |  |  |  |
| RECOMP-Extr   | 1.4                    | 11.4    | 103.11       |  |  |  |  |  |
| T TTMT'       | 2.2                    | 13.5    | 353.03       |  |  |  |  |  |
| LongLLMLingua | 2.2                    | 15.5    | 333.03       |  |  |  |  |  |

#### **C.3** Evaluation in Specialized Domains

To assess the generalizability of EXIT beyond general-domain datasets, we conducted additional evaluations on two specialized QA benchmarks:

1) **BioASQ** (Tsatsaronis et al., 2015), a biomedical QA dataset using the PubMed corpus (14.9M documents), 2) **COVID-QA** (Möller et al., 2020), a COVID-specific QA benchmark based on the TREC-COVID collection (Roberts et al., 2021) (171K documents).

Although our classifier was trained solely on HQA, a general-domain dataset, EXIT performs competitively or better than existing baselines in both specialized domains. Table 10 shows that EXIT achieves high EM and F1 scores while substantially reducing token counts. On COVID-QA, EXIT notably outperforms all baselines in EM and F1 despite the domain shift.

These results suggest that EXIT's context-aware extraction strategy generalizes well, preserving key evidence without requiring domain-specific fine-tuning. This highlights EXIT's robustness across diverse retrieval scenarios.

