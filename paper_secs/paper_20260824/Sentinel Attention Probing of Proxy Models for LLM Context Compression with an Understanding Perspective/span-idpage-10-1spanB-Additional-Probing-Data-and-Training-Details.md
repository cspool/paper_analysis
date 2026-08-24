# <span id="page-10-1"></span>B Additional Probing Data and Training Details

Probing Data Construction The probing classifier is trained on 3,000 QA examples sampled from NewsQA (50%), SQuAD (20%), and HotpotQA (30%). Each QA example yields one positive sentence containing the gold answer span and one negative sentence sampled from the same retrieved context, resulting in 6,000 training instances in total.

Retrieved contexts are segmented into sentences using spaCy's sentencizer, and sentence boundaries are used consistently for both supervision construction and attention aggregation.

For completeness, we report the context length distribution under the Qwen-2.5 tokenizer. In NewsQA, 30.1% of examples contain 0–500 tokens and 69.9% contain 500–1,000 tokens. In SQuAD, 99.3% of examples fall within the 0–500 token

range. For HotpotQA, all examples are restricted to 0–500 tokens by limiting unrelated retrieved content.

Prompt Template Sentence-level attention features are extracted using a fixed QA-style prompt applied to each query–context pair. The prompt format is shown below:

Given the following information: {context} Answer the following question based on the given information with one or a few words: {question} Answer:

For each prompted input, we collect decoder attention weights from the final decoding token across all layers and attention heads. Attention weights directed to tokens belonging to each sentence are aggregated and normalized to form fixedlength sentence-level feature vectors, which are then used as input to the probing classifier.

Context-Reliant Sample Selection To improve supervision quality, we retain only context-reliant QA examples where access to the retrieved context substantially improves answer correctness.

For NewsQA and SQuAD, we retain examples with memory-based EM = 0 and context-based EM = 1. For HotpotQA, we retain examples with memory-based F1 ≤ 0.2 and context-based F1 ≥ 0.5.

Probing Classifier Training We train a logistic regression (LR) classifier on attention-derived features using 5-fold cross-validation. We perform grid search over regularization strengths C ∈ {0.01, 0.1, 1.0, 10.0, 100.0} and select the best model based on validation AUC. Training uses the liblinear solver with ℓ<sup>2</sup> regularization, classbalanced weighting, and a maximum of 2,000 iterations.

## <span id="page-10-2"></span>C Baseline Descriptions

We compare Sentinel against the following baseline methods, grouped by their design paradigms:

- LLMLingua-1/2 [\(Jiang et al.,](#page-8-2) [2023;](#page-8-2) [Pan et al.,](#page-9-7) [2024\)](#page-9-7): Token-level compression methods based on saliency estimation via perplexity and LLM distillation. These methods are taskagnostic and do not condition on the query.
- Selective-Context [\(Li et al.,](#page-9-5) [2023\)](#page-9-5): A sentence-level, task-agnostic method that

<span id="page-11-2"></span>

| Method                                          | LongBench-Zh (GPT-3.5-Turbo) | Compression |       |      |        |       |  |  |
|-------------------------------------------------|------------------------------|-------------|-------|------|--------|-------|--|--|
|                                                 | SingleDoc                    | MultiDoc    | Summ. | AVG  | Tokens | Ratio |  |  |
| Metric-Based Compression (3K Constraint)        |                              |             |       |      |        |       |  |  |
| LLMLingua                                       | 35.2                         | 20.4        | 11.8  | 22.5 | 3,060  | 5×    |  |  |
| LLMLingua-2                                     | 46.7                         | 23.0        | 15.3  | 28.3 | 3,023  | 5×    |  |  |
| Contextual Utilization Decoding (2K Constraint) |                              |             |       |      |        |       |  |  |
| Sentinel (Qwen2.5-0.5B-Instruct)                | 64.8                         | 25.1        | 14.3  | 34.7 | 1,932  | 5×    |  |  |
| Sentinel (Qwen2.5-1.5B-Instruct)                | 63.3                         | 24.9        | 14.8  | 34.3 | 1,929  | 5×    |  |  |
| Original Prompt                                 | 61.2                         | 28.7        | 16.0  | 35.3 | 14,940 | –     |  |  |

Table 4: Performance comparison on filtered LongBench-Zh tasks using GPT-3.5-Turbo. LLMLingua baselines are evaluated under a 3K-token budget, while Sentinel is evaluated under a stricter 2K-token constraint.

scores context segments based on general informativeness, independent of the question.

- LongLLMLingua [\(Jiang et al.,](#page-8-9) [2024b\)](#page-8-9): A query-aware, multi-stage compression system using query-conditioned perplexity scoring, document reordering, and adaptive compression ratios.
- CPC [\(Liskavets et al.,](#page-9-13) [2024\)](#page-9-13): A contrastively trained sentence-ranking model that selects sentences based on semantic similarity to the query in embedding space. It is query-aware and trained on synthetic QA data.
- Raw Attention [\(Wang et al.,](#page-9-6) [2024;](#page-9-6) [Fang](#page-8-4) [et al.,](#page-8-4) [2025\)](#page-8-4): An attention-based heuristic baseline that ranks sentences using normalized decoder attention weights derived from concatenated query–context inputs, following prior attention-based compression methods such as QUITO and AttentionRAG.
- Random Selection: Sentences are sampled uniformly at random until the token budget is met, serving as a lower-bound reference.
- Empty Context: The model receives only the question without any retrieved context, serving as a zero-context baseline.

All baselines are evaluated under the same token budget and LLM generation setting for fair comparison.

