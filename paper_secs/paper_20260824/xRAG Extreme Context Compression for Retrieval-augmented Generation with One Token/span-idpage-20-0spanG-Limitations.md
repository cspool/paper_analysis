# <span id="page-20-0"></span>G Limitations

We discuss the limitations of our framework as follows:

- In this work, we only consider the most commonly used retrieval system—single dense vector retrieval, while sparse retrieval methods such as BM25 or multi-vector retrieval methods like ColBERT are not included. We believe that combining these methods would be a promising direction for xRAG, as sparse vectors could complement dense vectors, and multi-vector retrieval would provide xRAG with more flexibility by not condensing all information into one token.
- Currently, xRAG delivers decent performance when a relevant document is fetched; however, it lags behind RAG by a considerable margin in tasks that require reasoning (such as HotpotQA and FactKG). One possible reason is that during the training phase of xRAG, reasoning-relevant data is not provided. How to make xRAG a better reasoner remains our future work.
- We only consider the *Top-1* retrieval setting, while ensembling multiple relevant documents has been shown to be effective for RAG systems due to the complementary information contained in *Top-K* documents. We believe there is potential advantage for xRAG to scale to multi-document settings, as the input length of xRAG for multi-documents scales by a factor of 1, while for RAG, it scales by the document length factor.

