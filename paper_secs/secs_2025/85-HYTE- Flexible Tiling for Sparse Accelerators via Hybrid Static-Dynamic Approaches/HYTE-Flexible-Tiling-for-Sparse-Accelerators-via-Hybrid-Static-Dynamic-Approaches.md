# HYTE: Flexible Tiling for Sparse Accelerators via Hybrid Static-Dynamic Approaches

Xintong Li

Tsinghua University Beijing, China lixt21@mails.tsinghua.edu.cn Zhiyao Li

Tsinghua University Beijing, China lizhiyao19@mails.tsinghua.edu.cn Mingyu Gao

Tsinghua University
Beijing, China
Shanghai Artificial Intelligence Lab
Shanghai, China
Shanghai Qi Zhi Institute
Shanghai, China
gaomy@tsinghua.edu.cn

'25), June 21–25, 2025, Tokyo, Japan. ACM, New York, NY, USA, 14 pages.
https://doi.org/10.1145/3695053.3731044

## Abstract

Specialized hardware accelerators are widely used for sparse tensor computations. For very large tensors that do not fit in on-chip buffers, tiling is a promising solution to improve data reuse on these sparse accelerators. Nevertheless, existing tiling strategies on sparse accelerators are either purely dynamic and suffering from high design complexity, or purely static and using simple heuristics with insufficient adaptivity. In addition, they have not extensively explored the full design space of tiling to identify the optimal schemes, nor have they supported efficient management of the non-negligible metadata needed for tiling. We propose HYTE, a hybrid static-dynamic framework to enable flexible and efficient tiling on sparse accelerators. HYTE relies on a static offline scheduler to first identify a near-optimal initial tiling scheme through effective and lightweight sampling. The tile size and shape, the dimension iteration order across different tiles, and the buffer allocation policies can all be flexibly configured to adapt to the specific data sparsity patterns. Then at runtime, HYTE supports efficient management of the tiling metadata in both the off-chip memory and the on-chip buffer, as well as a technique of dynamic tuning on the tile shape to ensure high buffer utilization in the presence of highly varying local data patterns. Our evaluation shows that HYTE outperforms state-of-the-art sparse tiling strategies by  $3.3 \times$ to 6.2× on average for diverse sparse matrices.

