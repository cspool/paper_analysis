# <span id="page-2-0"></span>**4 The BenchPress Evaluation Suite**

Our evaluation suite is designed around components that are not specific to any single compression paradigm: its datasets, metrics, and protocol can be applied to soft compression,

<span id="page-2-2"></span><sup>2</sup>We were unable to include GMSA in our evaluation as their code and models were not publicly available at the time of writing.

<span id="page-3-0"></span>

| Dataset                                  | Avg. Context Tokens | #Samples | #Contexts |
|------------------------------------------|---------------------|----------|-----------|
| AdversarialQA (Bartolo et al., 2020)     | 154                 | 1,000    | 341       |
| HotpotQA (Yang et al., 2018)             | 254                 | 7,394    | 7,352     |
| NarrativeQA (Koˇciský et al., 2018)      | 639                 | 3,002    | 100       |
| ParaphraseRC (Saha et al., 2018)         | 685                 | 4,835    | 560       |
| SQuAD (Rajpurkar et al., 2016)           | 169                 | 5,928    | 1,204     |
| TriviaQA (Verified) (Joshi et al., 2017) | 539                 | 185      | 185       |
| Total                                    | 375                 | 22,344   | 9,742     |

Table 1: Short-context evaluation datasets. The overall average context length is weighted by the number of samples.

<span id="page-3-2"></span>

| Dataset                                                                             | Avg. Context Tokens | #Samples   | #Contexts  |
|-------------------------------------------------------------------------------------|---------------------|------------|------------|
| Single-Doc QA<br>QASPER (Dasigi et al., 2021)<br>MultiFieldQA-en (Bai et al., 2024) | 4,901<br>4,725      | 192<br>93  | 133<br>66  |
| Multi-Doc QA<br>HotpotQA (Yang et al., 2018)<br>2WikiMultihopQA (Ho et al., 2020)   | 4,997<br>5,426      | 128<br>165 | 128<br>165 |
| Total                                                                               | 5,044               | 578        | 492        |

Table 2: Mid-range context length LongBench-E evaluation datasets. The overall average context length is weighted by the number of samples. We only include samples for which the context length is under 8K tokens.

KV cache methods, and hard prompt compression alike. We demonstrate this breadth by evaluating both soft compression baselines and a hard-prompt method (LLMLingua2) within the same framework. A central design principle is to isolate compression quality from retrieval noise: we focus on reading comprehension, where contexts are guaranteed to contain the necessary evidence, enabling controlled comparison across datasets that stress both single-hop and multi-hop reasoning.

