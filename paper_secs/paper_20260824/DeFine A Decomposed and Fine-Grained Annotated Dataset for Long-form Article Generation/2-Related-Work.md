# 2 Related Work

## 2.1 LFAG Datasets

High-quality datasets are essential for advancing research on Long-Form Article Generation (LFAG). Within the question-answering (QA) domain, several existing datasets focus on long-form answers, aiming to generate detailed and well-grounded responses [\(Fan et al.,](#page-8-6) [2019;](#page-8-6) [Stelmakh et al.,](#page-9-5) [2023;](#page-9-5) [Cohen et al.,](#page-8-7) [2021;](#page-8-7) [Jin et al.,](#page-9-6) [2019;](#page-9-6) [Kocisk](#page-9-7) ˇ y et al. ` , [2018\)](#page-9-7). While these datasets encourage lengthier outputs, they lack structural decomposition and hierarchical content organization, making them insufficient for complex article generation. In addition, LongLaMP [\(Kumar et al.,](#page-9-3) [2024\)](#page-9-3) provides a benchmark for personalized long-text generation, capturing user-specific writing styles but without a structured decomposition process. LongWriter [\(Bai et al.,](#page-8-5) [2024a\)](#page-8-5) extends LLM output length but does not provide task-level decompositions. LongCite [\(Zhang et al.,](#page-9-8) [2024a\)](#page-9-8) introduces sentencelevel citation annotations, but focusing primarily on retrieval-based QA rather than structured article synthesis.

To address these limitations, DeFine employs a clear hierarchical decomposition, enabling finegrained control over article generation, with features like comprehensive outline creation, detailed reference use, and accurate citation metadata, ensuring coherent, factually consistent long-form content across various domains.

