# G. Performances on Various Data Distributions

To study noise separation in TRIDENTANN varying with data distributions, we conduct experiments on the uniform SIFT and skewed datasets (i.e., SPACEV, GloVe, and NY-Times) in Figure 18, with same recall qualifications (i.e., recall@10 > 90%). The left figure shows the throughput speedup ratio (with 10% noise separation vs. without noise separation), showing robust improvements on all datasets. The right figure further splits queries into two groups by average local intrinsic dimensionality (LID) [28, 43], showing that higher-LID queries tend to benefit more from our hybrid structure. Since skew degrades the clustering quality of local spaces, which can be mitigated by our hybrid structure.

| System | SPANN   | DiskANN & PipeANN | CPU-Only  | GPU-Acc    |
|--------|---------|-------------------|-----------|------------|
| Time   | 4-5 day | 2.5-3 day         | 2-2.5 day | 20-22 hour |

TABLE IV: *Building time on SIFT1B and SPACEV1B.*

# G. Performances on Various Data Distributions

To study noise separation in TRIDENTANN varying with data distributions, we conduct experiments on the uniform SIFT and skewed datasets (i.e., SPACEV, GloVe, and NY-Times) in Figure 18, with same recall qualifications (i.e., recall@10 > 90%). The left figure shows the throughput speedup ratio (with 10% noise separation vs. without noise separation), showing robust improvements on all datasets. The right figure further splits queries into two groups by average local intrinsic dimensionality (LID) [28, 43], showing that higher-LID queries tend to benefit more from our hybrid structure. Since skew degrades the clustering quality of local spaces, which can be mitigated by our hybrid structure.

| System | SPANN   | DiskANN & PipeANN | CPU-Only  | GPU-Acc    |
|--------|---------|-------------------|-----------|------------|
| Time   | 4-5 day | 2.5-3 day         | 2-2.5 day | 20-22 hour |

TABLE IV: *Building time on SIFT1B and SPACEV1B.*

