# 3 Experimental Setup

In our experiments, we employ document question answering (DocQA) as our primary evaluation task for long-context reasoning capabilities, as it inherently requires both contextual grounding and multi-step reasoning. This section details our experimental setup for training and evaluation.

## 3.1 Training Datasets

RL Dataset To construct a challenging RL dataset for verifiable long-context reasoning, we develop DOCQA-RL-1.6K, which comprises 1.6K DocQA problems across three reasoning domains: (1)

<span id="page-6-1"></span>Table 3: Detailed statistics of our train and test datasets. Length is calculated by the Qwen tokenizer.

| Statistics  | Train Dataset |        | Test Dataset |         |        |        |        |        |        |  |
|-------------|---------------|--------|--------------|---------|--------|--------|--------|--------|--------|--|
|             | SFT           | RL     | DocMath      | Frames  | 2Wiki  | HQA    | Musi   | NarQA  | Qasp   |  |
| # Examples  | 5,305         | 1,591  | 200          | 824     | 200    | 200    | 200    | 200    | 200    |  |
| Avg. Length | 13,064        | 11,437 | 17,645       | 15,756  | 7,530  | 13,431 | 16,327 | 29,887 | 5,074  |  |
| Max. Length | 20,003        | 59,559 | 176,285      | 117,131 | 17,035 | 17,640 | 17,883 | 65,357 | 21,927 |  |

**Mathematical Reasoning**: We use 600 problems from the DocMath [57] dataset, requiring numerical reasoning across long and specialized documents such as financial reports<sup>2</sup>; (2) **Logical Reasoning**: We employ DeepSeek-R1 [11] to synthesize 600 multi-choice questions requiring logic analysis of real-world documents spanning legal, financial, insurance, and production domains from our curated collection; (3) **Multi-Hop Reasoning**: We sample 200 examples from MultiHopRAG [36] and 200 examples from Musique [44], emphasizing cross-document reasoning.

**SFT Dataset** To establish a robust starting point for RL optimization, we distill 5.3K high-quality question-document-answer triplets through DeepSeek-R1 [11]. Aligned with recent data curation methods for LRMs [25, 53], we clean and filter questions based on quality, complexity, and diversity. Additionally, we control the quality and length of the documents to ensure precise contextual information. In Table 3, we provide the statistics of our RL and SFT datasets.

