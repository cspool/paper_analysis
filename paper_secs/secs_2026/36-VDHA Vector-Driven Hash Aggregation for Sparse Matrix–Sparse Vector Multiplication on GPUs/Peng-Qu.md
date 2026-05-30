# Peng Qu

Department of Computer Science and Technology, Beijing National Research Center for Information Science and Technology, Tsinghua University Beijing, China qp2018@mail.tsinghua.edu.cn

### **Abstract**

Sparse matrix-sparse vector multiplication (SpMSpV) is a core primitive in graph analytics and scientific computing, also arising in spiking neural networks for event-driven spike propagation. On GPUs, the performance of the prevalent and efficient SpMSpV paradigm is often bottlenecked by the write-back phase of accumulating non-zero multiply-accumulate results; its many-to-one index scatter pattern causes severe conflicts and poor bandwidth utilization on GPUs. We present VDHA, a GPU-based weighted SpM-SpV kernel that leverages block-private hash tables for local aggregation, substantially reducing write conflicts and improving memory coalescing. To further amplify this benefit, we incorporate column splitting with lightweight reordering to expose more locality, and employ a fetch-computewriteback pipeline to overlap hash computation with memory accesses. Extensive evaluation on over 300 matrices with more than 5 million nonzeros, including web-scale graphs (Konect/LAW) and scientific workloads (SuiteSparse), shows that VDHA consistently outperforms state-of-the-art baselines. On web graphs, it achieves a 1.41× geometric-mean speedup (up to 3.42×), while on SuiteSparse it delivers 1.13× (up to 2.55×). We also provide a lightweight predictive model that identifies matrices favorable to VDHA with 91.3% accuracy.

![](_page_0_Picture_9.jpeg)

This work is licensed under a Creative Commons Attribution 4.0 International License.

PPoPP '26, Sydney, NSW, Australia
© 2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2310-0/2026/01
https://doi.org/10.1145/3774934.3786447

