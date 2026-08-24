# <span id="page-10-0"></span>C Retrieval Performance

We assessed how retrieving more results would affect the amount of relevant information placed in the context of the generation model. Specifically, we assumed that the retriever returns X number of tokens and then calculated the recall score at that cutoff. From another perspective, the recall performance is the upper bound on the performance of the generation model when the model is required to use only the retrieved documents for generating answers.

Below are the recall@k results for the OpenAI text-embedding-3-large embedding model on 3 datasets and different context lengths (Table [S5\)](#page-10-2). We use chunk size 512 tokens and leave a 1.5k buffer for the prompt and generation. Recall@k here is different for each run based on the total number of retrieved chunks; for example, when 1 chunk is retrieved, we report recall@1, and when 61 chunks are retrieved we report recall@61. We note the relationship between the number of retrieved chunks and the maximum context length in Table [S5.](#page-10-2)

<span id="page-10-2"></span>

| Num.<br>Retrieved<br>chunks | 1              | 5              | 13           | 29           | 61           | 125          | 189          | 253          | 317          | 381          |
|-----------------------------|----------------|----------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| Context Length              | 2k             | 4k             | 8k           | 16k          | 32k          | 64k          | 96k          | 128k         | 160k         | 192k         |
| Databricks<br>Doc<br>sQA    | 0.547          | 0.856          | 0.906        | 0.957        | 0.978        | 0.986        | 0.993        | 0.993        | 0.993        | 0.993        |
| FinanceBench<br>NQ          | 0.097<br>0.845 | 0.287<br>0.992 | 0.493<br>1.0 | 0.603<br>1.0 | 0.764<br>1.0 | 0.856<br>1.0 | 0.916<br>1.0 | 0.916<br>1.0 | 0.916<br>1.0 | 0.916<br>1.0 |

Table S5: Retrieval performance (recall@k) for OpenAI text-embedding-3-large, which was used as the retriever in all of our experiments.

Saturation point: as can be observed in the table, each dataset's retrieval recall score saturates at a different context length. For the NQ dataset, it saturates early at 8k context length, whereas DocsQA and FinanceBench datasets saturate at 96k and 128k context length, respectively. These results demonstrate that with a simple retrieval approach, there is additional relevant information available to the generation model all the way up to 96k or 128k tokens. Hence, the increased context size of modern models offers the promise of capturing this additional information to increase overall system quality.

Similar to Fig. 2 in [Jin et al.,](#page-7-1) we find that retrieval accuracy monotonically increases. However, as shown in our main text, this does not necessarily mean that RAG accuracy monotonically increases.

