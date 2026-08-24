# C.12 Sentence Reordering Strategy Analysis

One potential concern with extractive compression is that removing many sentences may lead to unnatural or ambiguous contexts when the remaining content is reassembled. In EXIT, we preserve the original sentence order during reconstruction to maintain document coherence, following prior work [\(Hwang et al.,](#page-10-8) [2024\)](#page-10-8). However, we acknowledge that this decision could lead to suboptimal flow when large gaps exist between selected sentences.

To assess this design choice, we conducted an ablation study comparing four sentence reordering strategies: 1) Random: randomly shuffles selected

<span id="page-21-2"></span>Table 16: Ablation results comparing sentence reordering strategies in EXIT on HotpotQA. "Random" shuffles selected sentences, while "Ascending" and "Descending" sort by relevance scores. "Original order" maintains the sentence sequence from the original document.

| Reordering Strategy      | EM    | F1    |
|--------------------------|-------|-------|
| Random                   | 30.20 | 41.30 |
| Ascending (score-based)  | 30.43 | 41.44 |
| Descending (score-based) | 30.33 | 41.16 |
| Original order (EXIT)    | 30.61 | 41.49 |

<span id="page-21-3"></span>Table 17: Training data ablation comparison. Best results per metric are highlighted in bold.

| Training Data | EM ↑ | F1 ↑ | # token ↓ |
|---------------|------|------|-----------|
| HQA           | 31.6 | 42.6 | 195.1     |
| 2WIKI         | 29.2 | 40.3 | 135.3     |
| 2WIKI+HQA     | 30.6 | 42.0 | 232.2     |

sentences, 2) Ascending: sorts by relevance score (low to high), 3) Descending: sorts by relevance score (high to low), 4) Original order (EXIT): preserves sentence sequence from the source document

Table [16](#page-21-2) shows that maintaining the original order consistently yields the best EM and F1 scores on HQA. This confirms that preserving the source structure helps retain contextual flow, even when intermediate content is omitted. We include this result to support our design and to acknowledge the known limitations of extractive methods in handling discourse coherence.

#### C.13 Training Data Ablation Analysis

Table [17](#page-21-3) compares models trained on HQA, 2WIKI, or both. Training solely on HQA yields the highest EM and F1 (31.6 EM, 42.6 F1) with moderate token usage. By contrast, 2WIKI training improves compression but lowers accuracy (29.2 EM, 40.3 F1). Combining datasets does not surpass HQA alone.

This finding suggests that data quality and structure matter more than quantity. HQA's annotations appear particularly effective for learning robust compression strategies that generalize well, validating our choice to use it as the primary training dataset.

