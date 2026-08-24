# <span id="page-11-3"></span>D Baseline Methods

Full or Chunked Text Content For texts with fewer tokens than the LLM's input window, we can input the text directly into the LLM to obtain an answer. We refer to this method as Full Text Read, with the specific prompt provided in Figure [15.](#page-22-2) However, this approach is not applicable to texts exceeding the token limit of the LLM's input window. In such cases, [Lee et al.](#page-9-6) truncated the text to fit it into the LLM, but this method obviously results in information loss. We propose a method that does not lose information, offering a better comparison. This method involves dividing the entire text into chunks (using the same chunking method as GraphReader) and then having the LLM read these chunks sequentially according to the text order, thus enabling the handling of overly long texts with a limited input window. During the reading process, there are two main strategies: Chunk Read and Chunk Read with Notes. In the Chunk Read approach, the LLM only sees the current chunk during each reading, which is suitable for single-hop QA tasks. In the Chunk Read with Notes approach, the LLM can summarize useful information from the current chunk and provide it to the subsequent reading process, which is suitable for multi-hop QA tasks.

In the experiment, we divide the chunks in the same way as GraphReader, and the maximum length of the chunk is set to 2k. The specific prompts are in Figure [16](#page-22-3) and [17](#page-23-0) respectively.

Retrieval-Augmented Generation (RAG) RAG is a commonly used approach for addressing longtext problems. In this work, we compare the traditional RAG method, including retrieval methods based on Okapi BM25 (Robertson and Zaragoza, 2009) and the OpenAI API embedding model (text-embedding-ada-002). Specifically, we first split the text into chunks in the same method as GraphReader, then use the aforementioned methods to calculate the relevance scores between the question and these chunks, and finally input the topn chunks with the highest relevance scores together with the question for the LLM to answer. To ensure a fair comparison, we control the input window to 4k in the experiments. Specifically, in order to fill the input window as much as possible, we set the maximum length of the chunk to 38k when selecting the top-1 chunk for answering; when opting for the top-3 chunks, we set the maximum length of each chunk to 1k. The specific prompt can be found in Figure 18.

In addition to traditional RAG methods, we also compared GraphRAG (Edge et al., 2024) and LongRAG (Jiang et al., 2024). GraphRAG utilizes LLMs to construct a graph-based text index in two distinct stages. The first stage involves extracting an entity knowledge graph from the source documents, while the second stage focuses on generating summaries for groups of entities. When a question is posed, each summary provides partial information, which is then combined into the final answer for the user. LongRAG introduces a "long retriever" and a "long reader", allowing the entire corpus to be processed into larger-sized units, which reduces the number of units needed during retrieval and alleviates the burden on the retriever.

**Agent Style Methods** We also compared our method with similar approaches for handling long texts with small input windows, such as ReadAgent (Lee et al., 2024). ReadAgent is a method that segments long texts and generates gist memories, which are then looked up to search for information in order to answer questions. In the experiments, for datasets from LongBench, we adopted the default hyperparameters declared in the ReadAgent paper, specifically a max words of 600 and min\_words of 280 when splitting pages. For HotpotWikiQA-mixup from LV-Eval, we scaled these two hyperparameters using the same approach as in the ReadAgent paper. Specifically, for datasets with lengths of 256k and 128k, we used max\_words=10000 and min\_words=2000; for those with lengths of 64k, 32k and 16k, we used

max\_words=5000 and min\_words=1000. At the same time, we employed the ReadAgent-S method, which ReadAgent claims to be the most effective, reading the pages in sequence. Additionally, we allowed reading up to 5 pages (Look up 1-5 pages).

We also compared PEARL (Sun et al., 2024), a prompting framework to enhance reasoning capabilities for long documents. PEARL is structured into three stages: action mining, plan formulation, and plan execution. It decomposes complex questions into actionable steps and utilizes LLMs for zero-shot or few-shot prompting execution.

## <span id="page-12-0"></span>**E** Additional Experimental Results

<span id="page-12-2"></span>

| Method      | Input  |      | QuAL | ITY |       | Natural Question |      |      |       |  |
|-------------|--------|------|------|-----|-------|------------------|------|------|-------|--|
|             | Window | LR-1 | LR-2 | EM  | $F_1$ | LR-1             | LR-2 | EM   | $F_1$ |  |
| GPT-4-128k  | 128k   | 45.7 | 60.3 | 2.7 | 9.9   | 75.0             | 81.0 | 41.0 | 57.2  |  |
| Pearl       | 128k   | 52.3 | 72.7 | 3.0 | 9.6   | 74.0             | 79.0 | 38.0 | 56.8  |  |
| LongRAG     | 128k   | 52.3 | 67.0 | 3.7 | 11.6  | 77.7             | 83.0 | 47.3 | 60.0  |  |
| GraphRAG    | 128k   | 46.0 | 68.7 | 2.0 | 6.1   | 67.0             | 77.0 | 47.0 | 53.2  |  |
| GraphReader | 4k     | 57.3 | 82.3 | 4.3 | 14.3  | 79.0             | 84.7 | 48.3 | 62.1  |  |

Table 7: Performance (%) comparison of different baselines on two additional datasets. The best performance and the second-best performance are denoted in bold and underlined fonts, respectively.

Table 7 presents additional experimental results for two datasets, QuALITY and Natural Questions, both of which are highly relevant to real-world question-answering scenarios. The results indicate that our method significantly outperforms other baseline models in real-world scenarios.

## <span id="page-12-1"></span>**F** Evaluation Recall for Supporting Facts

We evaluate the recall rate of supporting facts for different methods using GPT-4-128k, with the temperature set to 0.1. Figure 19 shows the specific evaluation prompt.

For GraphReader, we evaluate the memory recorded in the final notebook. For ReadAgent, the evaluation focused on the final text segments reviewed. In the case of Chunk Read with Notes, we evaluate both the memory and the chunk read at the time of the final answer; for the RAG methods, we assess the retrieved chunks.

