# <span id="page-16-0"></span>**A Benchmark Details**

This section details the benchmarks used in our evaluation. A statistical overview of each dataset is provided in Table [3.](#page-16-1)

**MLVU.** MLVU [\[54\]](#page-14-0) is a multi-task benchmark for long video understanding, comprising 3,102 questions across 9 categories. The dataset is partitioned into a dev set (2,593 questions) and a test set (509 questions). Tasks are categorized into three primary types: 1) holistic analysis, 2) single-detail identification, and 3) multi-detail reasoning. For our evaluation, we utilize only multiple-choice questions from the dev set and exclude open-ended questions.

**LongVideoBench.** LongVideoBench [\[55\]](#page-14-1) is a question-answering benchmark featuring 3,763 web-collected videos and 6,678 human-annotated, multiple-choice questions spanning 17 fine-grained categories. The benchmark is designed to test referring reasoning by requiring models to retrieve and reason over detailed information. In our study, we utilize only the validation set of this benchmark.

**VideoMME.** VideoMME [\[56\]](#page-14-2) is a multi-modal benchmark covering 30 subdomains across 6 primary visual domains. It contains 900 videos, totaling approximately 254 hours, and 2,700 question-answer pairs. The dataset includes multiple modalities (e.g., video, subtitles, audio) and splits videos by duration (short, medium, long). To focus our evaluation on long-form video understanding, we use only the medium and long duration splits. Furthermore, we leverage only the video data and corresponding questions, excluding all other modalities like subtitles.

<span id="page-16-1"></span>**Table 3:** *Dataset Statistics. Overview of the data statistics across LongVideoBench [\[55\]](#page-14-1), MLVU [\[54\]](#page-14-0) and VideoMME [\[56\]](#page-14-2).*

| Dataset                 | Avg. Duration (s) | #QA Pairs |
|-------------------------|-------------------|-----------|
| MLVU [54]               | 636.2             | 2174      |
| LongVideoBench-val [55] | 732.2             | 1337      |
| VideoMME-short [56]     | 80.7              | 900       |
| VideoMME-medium [56]    | 516.8             | 900       |
| VideoMME-long [56]      | 2466.3            | 900       |

