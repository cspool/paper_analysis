# 8 Conclusion

This paper presents VDHA, a GPU-based SpMSpV algorithm targeting the costly write-back problem. VDHA combines long-column decomposition with reordering, sharedmemory hash aggregation, and a fetch–compute–writeback pipeline to improve locality, reduce conflicts, and reduce hash costs. Experiments on over 300 SuiteSparse and web-scale matrices with more than 5 million nonzeros show consistent gains over state-of-the-art baselines, with up to 3.42× speedup (1.41× on average) on web graphs and up to 2.55× (1.13× on average) on scientific matrices. We further propose a lightweight analysis method to predict when VDHA is beneficial, achieving 91.3% accuracy.

