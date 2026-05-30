# C Additional Module Ablation

Table [13](#page-13-0) reports module-wise ablations on additional benchmarks, including NExT-QA, IntentQA, MSVD, MSRVTT, TGIF, and ANet. For multichoice VideoQA, all modules yield incremental gains. For open-ended VideoQA, where questions are generally simpler (Section [4.4\)](#page-6-3), question decomposition can mislead the model.

Table [14](#page-13-1) shows an MSVD case where decomposition, though semantically valid, overcomplicates a simple spatial query and lowers accuracy. This indicates that decomposition is most effective for complex or multi-step reasoning.

Table 13: Module Ablation on Other Benchmarks.

<span id="page-13-0"></span>

| Module                              | Multi-choice VideoQA (Acc., ↑) |          | Open-ended VideoQA (Acc./Score, ↑) |          |          |          |
|-------------------------------------|--------------------------------|----------|------------------------------------|----------|----------|----------|
|                                     | NExT-QA                        | IntentQA | MSVD                               | MSRVTT   | TGIF     | ANet     |
| Baseline                            | 65.4                           | 61.3     | 77.8/4.0                           | 62.8/3.5 | 76.9/4.0 | 54.2/3.3 |
| + Dynamic Spatial Token Compression | 66.7                           | 62.2     | 79.4/4.0                           | 63.6/3.5 | 78.9/4.1 | 55.4/3.3 |
| + Dynamic Temporal Frame Selection  | 67.0                           | 62.9     | 80.0/4.1                           | 64.2/3.5 | 79.1/4.1 | 56.4/3.4 |
| + Question Decomposition            | 68.3                           | 64.2     | 72.4/3.8                           | 62.2/3.5 | 75.7/4.0 | 53.8/3.3 |

<span id="page-13-1"></span>Table 14: Question Decomposition Example on MSVD

| "What is a man sitting on?" |  |
|-----------------------------|--|
| Original Question:          |  |

- 1. Does the man change location during the video, or does he remain in one place?
- 2. At what point does he begin sitting down, and what happens before that?
- 3. What object does he touch when sitting down?
- 4. Does that object stay consistent across the video?
- 5. Do we observe interactions confirming it's a seat?
- <span id="page-13-2"></span>6. Do any video perspectives help reveal more detail?

Table 15: Error Analysis on MSRVTT-QA

| Method        | Full Set | Top-100 Scene Change Samples |
|---------------|----------|------------------------------|
| SF-LLaVA      | 65.8/3.6 | 64.0/3.5                     |
| D-CoDe (Ours) | 64.2/3.5 | 56.0/3.3                     |

### D Error Analysis

As noted in Section [4.4,](#page-6-3) D-CoDe performs worse on videos with frequent scene transitions. Table [15](#page-13-2) shows results on MSRVTT-QA using the full set and 100 samples with the most transitions. SF-LLaVA remains stable (65.8 vs. 64.0), whereas D-CoDe drops sharply (64.2 vs. 56.0), confirming its sensitivity to rapid scene changes.

