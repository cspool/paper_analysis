# C.9 Analysis of Classification Performance Across Negative Sample Types

Table [15](#page-20-2) presents EXIT's sentence-level classification performance, broken down by negative sample type. EXIT achieves 0.92 F1 for both positive ("Yes") and negative ("No") classes overall, indicating a balanced ability to identify essential and non-essential sentences.

The model excels at filtering random negatives (0.95 F1), effectively discarding irrelevant content. With hard negatives (topically related but not essential), EXIT still performs well (0.89 F1 for positive, 0.88 for negative), handling nuanced relevance distinctions.

These results highlight EXIT's adaptability and confirm its suitability for real-world RAG scenarios

<span id="page-20-1"></span>Table 13: Comparison of EXIT using a zero-shot (frozen Gemma-2B) vs. fine-tuned classifier on HotpotQA (HQA) and 2WikiMultiHopQA (2WIKI).

| Dataset | Method           | EM   | F1   |
|---------|------------------|------|------|
| 2WIKI   | Original Docs    | 18.0 | 25.7 |
|         | EXIT (Zero-shot) | 22.4 | 27.9 |
|         | EXIT (Ours)      | 24.8 | 30.1 |
| HQA     | Original Docs    | 29.2 | 40.2 |
|         | EXIT (Zero-shot) | 31.4 | 43.1 |
|         | EXIT (Ours)      | 31.6 | 42.6 |

| Dataset | Method                                           | Avg. #Tokens               |
|---------|--------------------------------------------------|----------------------------|
| 2WIKI   | Original Docs<br>EXIT (Zero-shot)<br>EXIT (Ours) | 764.47<br>731.35<br>145.62 |
| HQA     | Original Docs<br>EXIT (Zero-shot)<br>EXIT (Ours) | 735.24<br>727.87<br>191.11 |

<span id="page-20-0"></span>Table 14: Comparison of EXIT with baselines on QA tasks on HQA. EXIT (GPT-4o) refers to a variant using GPT-4o as a zero-shot relevance classifier, while EXIT (Ours) represents our proposed method trained with supervised data.

| Method                       | EM           | F1           | #token        |
|------------------------------|--------------|--------------|---------------|
| Original Docs                | 29.2         | 40.2         | 735.2         |
| RECOMP-Extr                  | 25.2         | 34.8         | 89.7          |
| RECOMP-Abst                  | 19.8         | 24.7         | 55.5          |
| LongLLMLingua                | 27.4         | 40.2         | 227.1         |
| CompAct                      | 29.4         | 40.9         | 76.3          |
| Refiner                      | 28.8         | 40.7         | 73.4          |
| EXIT (GPT-4o)<br>EXIT (Ours) | 30.4<br>31.6 | 41.9<br>42.6 | 84.2<br>191.0 |

where both overtly irrelevant and subtly extraneous content must be managed.

