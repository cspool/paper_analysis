# 3 Dataset DeFine

## 3.1 Data Source

The DeFine leverages wikipedia[2](#page-2-0) as the primary data source due to its extensive article length, rich structured information (including hierarchical headings and paragraphs), and detailed citation references, which collectively facilitate the construction of the Long-Form Article Generation (LFAG) dataset.

## 3.2 Data Annotation

Existing datasets suffer from a lack of hierarchical structure and task decomposition, and most only provide evaluation benchmarks or focus on personalized or question-answering tasks, lacking the fine-grained, multi-step annotations required for long-form text generation. To address the above issues, we built a multi-agent pipeline, which is broken down into the following manageable agents: Data Miner, Cite Retriever, Q&A Annotator and Data Cleaner.

## 3.2.1 Data Miner: Outline Data Extraction

This step leverages human-built web crawling techniques implemented by the Data Miner agent to meticulously extract structural elements—such as titles, subtitles, and hierarchical sections—from high-quality Wikipedia articles. By constructing a well-organized and detailed outline, Data Miner ensures that the extracted data maintains a clear and logical structure.

