# Limitation

This work focuses on RAG pipelines, and its effectiveness in general long-context scenarios like LongBench [\(Bai et al.,](#page-9-4) [2024\)](#page-9-4) remains future work. Our approach relies on explicit sentence-level annotations for training, which were manually obtained in our experiments. While these could be automated using GPT-4 supervision or reader-derived signals (as discussed in Section [C.8\)](#page-18-0), we have not yet explored fully automated annotation strategies—an important direction for future work.

EXIT performs sentence-level extraction, which may introduce limitations when sentences are overly long, noisy, or contain irrelevant details. While our context-aware classifier helps mitigate this, some ambiguity may persist, especially in cases involving complex or entity-heavy sentences. Additionally, although preserving the original sentence order empirically yields the best performance, we observed that removing intermediate sentences can occasionally lead to unnatural or incoherent reconstructions. Addressing this may require future improvements such as sentence decomposition, coherence-aware selection, or lightweight sentence refinement mechanisms.

We also focus on a single-step RAG setting, excluding iterative or recursive retrieval [\(Shao et al.,](#page-13-7) [2023;](#page-13-7) [Trivedi et al.,](#page-13-8) [2023;](#page-13-8) [Khattab et al.,](#page-11-11) [2022\)](#page-11-11). However, EXIT is orthogonal to these approaches and can be integrated by compressing retrieved content at each step.

Finally, while EXIT avoids the latency bottlenecks of autoregressive generation, further efficiency gains could be achieved through architectural optimizations such as prefix caching or Radix-Attention [\(Zheng et al.,](#page-14-7) [2024\)](#page-14-7), which remain open areas for future exploration.

