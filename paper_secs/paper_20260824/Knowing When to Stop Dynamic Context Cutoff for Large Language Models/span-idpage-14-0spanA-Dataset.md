# <span id="page-14-0"></span>A Dataset

#### <span id="page-14-1"></span>A.1 Statistics

We define gold location as the gold information span's end position divided by the total number of tokens in the input, which is uniformly distributed. We provide statistics for the Long datasets as shown below in Table [6](#page-14-3) and Short-form dataset in Table [7.](#page-14-4) Each dataset contains 600 data points, and the train-validation-test split is 80%, 10%, and 10%, respectively.

<span id="page-14-3"></span>Table 6: The datasets are grouped based on reasoning type. Code Understanding (Code) is distinct as it involves synthetic code understanding, whereas SQuAD and Natural Questions focus on retrieving answers from a single passage. Multi-hop Key-Value Retrieval (KV) is separate as a synthetic multihop key-value retrieval task, while HotpotQA and MUSIQUE involve natural multi-hop reasoning across multiple passages. Statistics include Token Count and Gold Location.

| Statistic     |           | Single-hop                |           | Multi-hop          |
|---------------|-----------|---------------------------|-----------|--------------------|
|               | Code      | SQuAD & Natural Questions | KV        | HotpotQA & MUSIQUE |
| Token Count   |           |                           |           |                    |
| Mean          | 22,758.96 | 22,653.68                 | 22,494.19 | 24,970.78          |
| Median        | 22,739.50 | 22,584.00                 | 22,619.00 | 25,168.50          |
| Std Dev       | 7,299.95  | 6,758.97                  | 7,290.87  | 7,159.93           |
| Max           | 35,465    | 36,888                    | 35,426    | 40,622             |
| Min           | 10,107    | 9,955                     | 10,023    | 10,430             |
| Gold Location |           |                           |           |                    |
| Mean          | 0.50      | 0.50                      | 0.51      | 0.49               |
| Median        | 0.50      | 0.50                      | 0.49      | 0.49               |
| Std Dev       | 0.28      | 0.28                      | 0.25      | 0.26               |
| Max           | 0.99      | 0.99                      | 0.96      | 0.96               |
| Min           | 0.01      | 0.01                      | 0.04      | 0.02               |

<span id="page-14-4"></span>Table 7: Categorized by reasoning type, with single-hop tasks (Code Understanding, SQuAD, and Natural Questions) involving direct retrieval from a passage, and multi-hop tasks (Multi-hop Key-Value Retrieval, HotpotQA, and MUSIQUE) requiring inference across multiple segments. Statistics include both token count and gold location.

| Statistic     |         | Single-hop                |         | Multi-hop          |
|---------------|---------|---------------------------|---------|--------------------|
|               | Code    | SQuAD & Natural Questions | KV      | HotpotQA & MUSIQUE |
| Token Count   |         |                           |         |                    |
| Mean          | 2951.23 | 1678.98                   | 2723.65 | 1676.15            |
| Median        | 2910.50 | 694.50                    | 2704.00 | 1735.00            |
| Std Dev       | 562.14  | 1269.42                   | 450.02  | 646.41             |
| Max           | 4990    | 4993                      | 4616    | 2815               |
| Min           | 1489    | 600                       | 1572    | 578                |
| Gold Location |         |                           |         |                    |
| Mean          | 0.51    | 0.48                      | 0.56    | 0.53               |
| Median        | 0.52    | 0.49                      | 0.55    | 0.50               |
| Std Dev       | 0.23    | 0.28                      | 0.20    | 0.28               |
| Max           | 0.98    | 0.99                      | 0.91    | 0.99               |
| Min           | 0.02    | 0.02                      | 0.21    | 0.02               |

### <span id="page-14-2"></span>A.2 Dataset Balance

Our information sufficiency evaluation dataset is carefully balanced by design. As described in Section [A,](#page-14-0) the gold location in our context is sampled from a uniform distribution, placing the required information approximately at the middle of the context. This creates a balanced evaluation where approximately 50% of context chunks are classified as "insufficient" (before the gold location) and 50% as "sufficient" (after and including the gold location). This balanced distribution ensures that our evaluation is not biased towards either early or late stopping decisions, providing a fair assessment of the method's ability to detect context sufficiency.

#### <span id="page-15-0"></span>A.3 Sufficiency Label Collection Process

To train our sufficiency detection classifiers, we create datapoints by labeling context chunks as either sufficient or insufficient for answering the given question. Here we describe our methodology for generating these sufficiency labels.

Label Generation Process. We generate sufficiency labels by first splitting the context into nonoverlapping chunks according to our chunking strategy (e.g., 10% of total tokens per chunk). Using the ground truth answer location(s), we identify the answer-containing chunk(s) in the document. We then label all chunks that appear before the answer-containing chunk as insufficient (0), while marking the answer-containing chunk itself and all subsequent chunks as sufficient (1). This labeling approach is based on the intuition that a question becomes answerable *if and only if* all necessary information chunks are present in the context. Note that the labeling process varies slightly for different question types:

- Single-hop Questions: These typically require information from a single passage or section within the document. Depending on the chunking strategy, there is usually only one answercontaining chunk. All chunks before this are labeled as insufficient, while this chunk and all subsequent chunks are labeled as sufficient.
- Multi-hop Questions: These questions require integrating information from multiple parts of the document. There may be multiple answer-containing chunks (e.g., different pieces of information needed from different sections). In these cases, only the last answer-containing chunk and all subsequent chunks are labeled as sufficient, as all required information is only available after that point.

#### <span id="page-15-1"></span>A.4 Synthetic Sufficiency Labels

Most existing QA datasets (including all six datasets used in our paper) are constructed with known answer locations, making it straightforward to generate sufficiency labels as described above. However, this approach may not be directly applicable to scenarios where answer locations are not explicitly provided. To address this limitation, we investigated whether large language models could generate synthetic sufficiency labels that perform comparably to those derived from human-annotated ground truth locations. We conducted experiments comparing classifiers trained with two types of labels:

- Original Labels: Generated using the ground truth answer locations as described above.
- Synthetic Labels: Generated using GPT-4o to predict answer locations within the documents.

For the synthetic label generation, we prompt GPT-4o to identify the minimal set of context chunks required to answer each question completely. We then use these predictions to label chunks as sufficient or insufficient following the same methodology used for original labels. Table [8](#page-16-4) shows the performance comparison between classifiers trained with synthetic versus original labels. The evaluation pipeline for both remained identical, relying on the same ground-truth labels for testing.

The results show that while there is a modest performance gap, classifiers trained with synthetic labels still achieve strong performance that is competitive with those trained on original labels. This indicates that our approach can be effectively extended to scenarios where explicit answer locations are not available, by leveraging LLMs to generate reasonably accurate sufficiency labels.

<span id="page-16-4"></span>Table 8: Performance comparison between classifiers trained with synthetic (GPT-4o generated) versus original (human-annotated) sufficiency labels across different model sizes.

| Task Type  |      | Synthetic | Original |      |  |
|------------|------|-----------|----------|------|--|
|            | 1B   | 8B        | 1B       | 8B   |  |
| Single-hop | 82.1 | 84.4      | 85.7     | 89.3 |  |
| Multi-hop  | 87.1 | 89.6      | 90.9     | 90.3 |  |
| Overall F1 | 84.6 | 87.0      | 88.3     | 89.8 |  |
| P90        | 79.3 | 82.7      | 85.9     | 90.1 |  |

