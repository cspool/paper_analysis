# <span id="page-16-0"></span>C Details for Instruction Tuning Dataset

After collecting the raw data from different categories, we use templates[4](#page-16-1) from FLAN [\[77\]](#page-13-9) to construct instruction tuning dataset. In Table [7](#page-16-2) we list an overview and in Table [8](#page-16-3) we list the detailed information for each subtask of our dataset. For QA tasks that lack an explicit context, we perform a retrieval operation within the corpus D to identify the most relevant document to serve as context. This approach is akin to the retrieval-augmented instruction tuning depicted in [\[54,](#page-12-12) [50\]](#page-12-6).

<span id="page-16-2"></span>

| Task Type             | # Involved datasets | # Train | # Prompt | # Label |
|-----------------------|---------------------|---------|----------|---------|
| Reading Comprehension | 7                   | 488,344 | 447.62   | 30.34   |
| Summarization         | 3                   | 81,821  | 483.49   | 53.29   |
| Open Domain QA        | 7                   | 385,173 | 203.55   | 20.09   |

Table 7: Overall statistics of Instruction Tuning dataset.

<span id="page-16-3"></span>

| Task Type     | Dataset            | # Train | # Prompt Len | # Label Len |
|---------------|--------------------|---------|--------------|-------------|
|               | CoQA [66]          | 7101    | 617.98       | 77.75       |
|               | DROP [16]          | 76098   | 356.06       | 3.86        |
|               | NarrativeQA [39]   | 32747   | 702.39       | 7.86        |
| Reading       | PubMedQA [31]      | 1000    | 397.91       | 65.4        |
| Comprehension | QuAIL [67]         | 10246   | 512.9        | 2.0         |
|               | SQuAD v2 [65]      | 130319  | 214.54       | 6.87        |
|               | PwC [19]           | 241564  | 571.35       | 53.07       |
|               | NQ [40]            | 87925   | 203.62       | 5.976       |
|               | TriviaQA [32]      | 78785   | 216.1        | 6.49        |
|               | CommonsenseQA [72] | 9741    | 223.64       | 2.0         |
| Open Domain   | WikiQA [80]        | 1040    | 192.89       | 40.79       |
| QA            | YahooQA5           | 87358   | 196.56       | 56.7        |
|               | FreebaseQA [30]    | 20353   | 218.49       | 4.87        |
|               | MSMarco [5]        | 99994   | 194.82       | 15.91       |
|               | CNN/DM [69]        | 100000  | 616.99       | 63.37       |
| Summarization | SamSum [20]        | 14731   | 187.87       | 29.12       |
|               | DialogSum [10]     | 12460   | 247          | 37.61       |

Table 8: Detailed data statistics for our Context-aware Instruction Tuning Dataset.

<span id="page-16-1"></span><sup>4</sup> <https://github.com/google-research/FLAN/blob/main/flan/templates.py>

