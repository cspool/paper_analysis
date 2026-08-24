# 3 Method

In this section, we first present our problem formulation, the RAG pipeline with a compression stage, and our novel compression framework, EXIT, which is designed to extract key evidence for answering in a parallel manner.

#### 3.1 Problem Formulation

RAG Pipeline with Compression. Given a query q and a document corpus C, a RAG pipeline first retrieves Top-k relevant document set D:

$$D = \{d_1, \dots, d_k\} = \mathsf{Retriever}(q, \mathcal{C}), \tag{1}$$

The retrieved documents within the document set D are then processed by a compression module that preserves query-relevant information while significantly reducing input length:

$$D' = \mathsf{Compressor}(q, D) \text{ s.t. } l(D') \ll l(D),$$
 (2)

where l(·) represents the function calculating the number of tokens in the document set. After compression, the number of tokens included in D′ is substantially decreased compared to D. Finally, an LLM generates the answer a using the compressed set D′ and the given query q:

$$a = \mathsf{LLM}(q, D'). \tag{3}$$

Objectives of Compression. Effective compression in the RAG pipeline must satisfy three key criteria: (1) Token Reduction–D′ should have fewer tokens to speed up answer generation; (2) Retention of Key Evidence–essential information must be preserved to maintain answer accuracy; and (3) Efficient Processing–compression should be fast enough to avoid introducing significant latency.

While token count influences the reader's reading time, it alone does not capture the full latency of the pipeline. In practice, end-to-end latency also includes the time spent on the compression step itself. Therefore, an efficient RAG system must minimize both the size of the input and the total time taken from retrieval to answer generation.

#### 3.2 Extractive Context Compression (EXIT)

To achieve three objectives of the compression step, EXIT consists of three main components: sentencelevel decomposition, context-aware relevance classification, and document reassembly. Importantly, EXIT functions as a plug-and-play module that integrates seamlessly into existing RAG pipelines, independent of the specific retriever or reader model used.

Sentence-Level Decomposition. In Step 1 of Figure [2,](#page-2-0) EXIT divides each retrieved document into individual sentences using a rule-based sentence tokenizer. For each document d<sup>i</sup> ∈ D, it produces a sentence set S<sup>i</sup> = {si1, si2, . . . , sin}, where sij is the j-th sentence in document i. Operating at the sentence level avoids the fragmentation of key phrases and preserves entity relationships that token-level compression techniques [\(Jiang et al.,](#page-10-2) [2023\)](#page-10-2) often disrupt. As a result, the compressed context preserves both syntactic coherence and semantic integrity, ensuring that key information is retained.

Context-Aware Relevance Classification. To efficiently filter sentences in D that contain key evidence for answering a query, we introduce a relevance evaluation method based on context awareness and single-token prediction. First, incorporating the entire document d<sup>i</sup> as context is essential, as understanding a sentence often requires the broader document context rather than an isolated sentence. This ensures that no relevant information is overlooked and enables more effective compression. Second, to maintain efficiency, we adopt lightweight single-token prediction instead of multi-token generation [\(Yoon et al.,](#page-14-3) [2024;](#page-14-3) [Li et al.,](#page-11-6) [2024\)](#page-11-6), which can introduce computational overhead. Inspired by [Zhong et al.](#page-14-6) [\(2022\)](#page-14-6), we leverage probabilistic classification for sentence relevance assessment. Given query q, document d<sup>i</sup> , and sentence sij , the evaluation model predicts relevance using a binary classification with "Yes" and "No" labels:

$$r_{ij} = \frac{P(\text{"Yes"}|q, d_i, s_{ij})}{P(\text{"Yes"}|q, d_i, s_{ij}) + P(\text{"No"}|q, d_i, s_{ij})}$$
(4)

where P(·|·) represents the likelihood from the evaluation model. This computation is parallelized across sentences for efficiency.

EXIT then selects sentences with a relevance score above a predefined threshold τ , ensuring that only key information is retained. Unlike fixed-size selection, our framework produces an adaptive number of sentences in the compressed set D′ , aligning with prior work [\(Jeong et al.,](#page-10-0) [2024\)](#page-10-0) that acknowledges query-dependent information needs. This approach optimizes compression while preserving essential evidence.

Document Reassembly. As shown in Step 3 of Figure [2,](#page-2-0) EXIT reconstructs the compressed document D′ by concatenating selected sentences in their original order, preserving coherence and logical flow [\(Hwang et al.,](#page-10-8) [2024\)](#page-10-8) for accurate downstream reasoning.

#### 3.3 Classifier Model Training

Training Strategy. Our goal is to train a relevance classifier capable of accurately identifying which sentences provide the evidence required to answer a query. To approximate real-world complexity, we utilize a question-answering dataset that requires multi-sentence reasoning and offers explicit sentence-level annotations of essential information. Leveraging these annotations, we model three typical retrieval outcomes: (1) sentences containing necessary evidence, (2) seemingly relevant sentences missing key details, and (3) entirely irrelevant sentences.

Data Sampling. To train a robust relevance classifier, we construct a diverse dataset reflecting various post-retrieval conditions. Positive samples include sentences essential for correct answers, while hard negatives consist of remaining sentences from the same documents. Additionally, random negatives pair queries with unrelated sentences, helping the model distinguish relevance from noise. This balanced sampling enables effective filtering of non-essential information without relying on explicit supervision.

Training Procedure. Each training instance is represented as (q, s, d, l), where q is the query, s is a candidate sentence, d is the document containing s, and l ∈ {"Yes", "No"} indicates whether s provides the required evidence. We employ a binary cross-entropy loss function to train the classifier:

$$\mathcal{L} = -\mathbb{1}_{l=\text{``Yes''}} \log P(\text{``Yes''}) - (1-\mathbb{1}_{l=\text{``Yes''}}) \log P(\text{``No''}), \tag{5}$$

By exposing the classifier to a balanced and diverse set of retrieval scenarios, we improve its ability to generalize and reliably identify sentences that contain the critical evidence for answering queries.

