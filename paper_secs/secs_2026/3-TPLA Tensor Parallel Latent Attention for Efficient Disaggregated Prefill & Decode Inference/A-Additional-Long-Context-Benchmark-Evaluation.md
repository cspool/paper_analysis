# A Additional Long-Context Benchmark Evaluation

In addition to the commonsense reasoning tasks and Long-Bench reported in the main text (Table [1](#page-8-0) and Table [2\)](#page-8-1), we further evaluate on two additional long-context benchmarks: LOFT [\[27\]](#page-12-28) and RULER [\[23\]](#page-12-29).

LOFT evaluates in-context retrieval and reasoning across multiple tasks with sequence lengths ranging from 32K to 1M tokens. Since DeepSeek-V2-Lite is not optimized for extremely long contexts, its performance on LOFT retrieval, RAG, and SQL tasks is very low (approximately 0–0.15). We therefore focus on the 32K ICL tasks, where the model behaves reliably. In this setting, MLA achieves an accuracy of 0.345, and TPLA (pd sep.) matches it at 0.345, further confirming that TPLA preserves accuracy.

RULER provides synthetic long-context tasks with configurable sequence length and task complexity, measuring capabilities beyond simple in-context recall. We evaluate DeepSeek-V2-Lite and its TPLA variants on all 13 RULER

<span id="page-14-0"></span>tasks across four context lengths (4,096 / 8,192 / 16,384 / 32,684 tokens). Table [3](#page-14-1) reports the average score across the 13 tasks at each context length.

Overall, the results suggest that TPLA with PD-separation better preserves performance in shorter-context decoding scenarios (e.g., LOFT ICL and the commonsense tasks in Table [1,](#page-8-0) both evaluated in a multiple-choice setting). In contrast, for long-decoding settings (e.g., long-CoT style generation), the TPLA conversion can introduce non-negligible approximation errors. As the generation length increases, such errors may accumulate, leading to a larger accuracy drop. We further observe that the aligned TPLA variant narrows the gap at shorter contexts (Table [1](#page-8-0) and [2\)](#page-8-1), but still exhibits a notable performance drop at longer contexts in RULER (Table [3\)](#page-14-1). This is expected, as our lightweight alignment uses

training sequences up to 4K tokens, and thus does not directly optimize the model for substantially longer-context behaviors (e.g., 16K–32K). Developing more effective and efficient long-context alignment strategies for TPLA therefore remains an important direction for future work.

<span id="page-14-1"></span>Table 3. RULER accuracy under different context lengths.

| Model            | 4K    | 8K    | 16K   | 32K   |
|------------------|-------|-------|-------|-------|
| DeepSeek-V2-Lite | 52.50 | 42.88 | 46.97 | 37.92 |
| - TPLA (pd sep.) | 21.82 | 15.38 | 16.58 | 17.41 |
| - TPLA (align)   | 47.58 | 42.53 | 35.47 | 26.87 |