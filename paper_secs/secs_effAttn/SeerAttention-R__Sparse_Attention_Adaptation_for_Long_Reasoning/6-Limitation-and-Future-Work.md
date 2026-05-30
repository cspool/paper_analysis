# 6 Limitation and Future Work

#### 6.1 End-to-end Speedup

The current focus of this work is on the accuracy of sparse decoding and its kernel-level speedup, while leaving end-to-end system support and optimization for future work. Achieving significant endto-end speedup will require integration with state-of-the-art inference frameworks such as vllm [\[32\]](#page-14-3), sglang [\[78\]](#page-17-1) and Lserve [\[67\]](#page-16-3), along with additional support for sparse kernels with PagedAttention. In addition, SeerAttention-R can possibly be combined with KV cache offloading technique similar to previous works [\[62,](#page-16-4) [42,](#page-14-4) [11,](#page-13-6) [24\]](#page-13-7) to save GPU memory and only keep the K Compression Cache with AttnGate to dynamically control computation/communication.

#### 6.2 Adaptive Sparsity Ratio

Determining the optimal sparsity ratio for attention is a non-trivial challenge. It involves a fundamental trade-off between accuracy and efficiency, and depends on multiple dynamic factors including the input length, task difficulty, and reasoning length. In general, easier tasks often allow for greater sparsity. However, for reasoning models, more difficult tasks tend to require longer reasoning steps, where attention computation only becomes a bottleneck for longer sequences. This apparent contradiction highlights the need for (1) a sufficiently precise sparsity selection algorithm and (2) automatic adaptation of sparsity ratios based on task complexity. One promising solution is to use Top-p (Nucleus sampling) in sparsity selection, which has previously been explored in Twilight [\[38\]](#page-14-5) and MagicPIG [\[11\]](#page-13-6). Specifically, Twilight adopts a binary search algorithm to find the optimal threshold that satisfies Top-p.

