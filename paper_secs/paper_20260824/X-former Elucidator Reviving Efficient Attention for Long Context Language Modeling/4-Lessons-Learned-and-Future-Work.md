# 4 Lessons Learned and Future Work

Section [2](#page-1-2) and Section [3](#page-4-3) provide a new taxonomy of existing Transformer methods and show a comprehensive analysis of their performance. In this section, we summarize the lessons learned from these analysis in two aspects. From the algorithm design perspective:

- Although positional selection methods are following simple and static token importance assumptions, they surprisingly outperform the other emerging methods in preserving models' performance (even better than full attention in some cases).
- The contextual compression approach seems more reasonable to preserve information but existing methods cannot achieve consistent performance improvements on different tasks.
- Kernelization methods show advantages on long sequence tasks, but meanwhile, they suffer from severe performance degradation in short sequence tasks.
- Factorization may not be suitable for LLMs because of the auto-regressive decoding algorithm. Existing LLMs usually have a growing KV cache, making it diffcult to dynamically factorize these keys and queries.
- The combination of different methods is a promising avenue for future research as it leverages the advantages from different sides. However, how to effectively combine these methods remains an open research question.

From the system implementation perspective:

• Theoretical time complexity does not always match execution performance on GPUs; this is because the implementation of a GPU kernel can signifcantly affect its runtime performance. Enabling effcient kernel-level optimizations (e.g., block-sparse kernels [\[Gray](#page-7-26) *et al.*, [2017\]](#page-7-26) and sparse tensor compilers [Ye *et al.*[, 2023\]](#page-8-27)) is becoming increasingly important for supporting effcient Transformer computation.

• The memory consumption also depends highly on the implementation. For resource-starved environments (e.g., consumer-level GPUs, mobile and edge devices), it is necessary to re-consider the choice of effcient attention mechanisms within the limited memory budget.

Besides, algorithm-hardware co-design is also becoming a promising direction to improve the overall effciency from both sides. In summary, there is no one-size-fts-all solution of effcient attention mechanisms for long context LLMs. Considering the unique task characteristics (e.g., sequence length, token dependencies, attention score distribution) is essential when designing new methods.

