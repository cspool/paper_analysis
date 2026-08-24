# 7 Related Work

Parallel Training Optimization. During hybrid parallel training, the optimization of parallel strategies is crucial. Many advanced parallelism optimization techniques [\[21,](#page-13-16) [30,](#page-14-17) [44,](#page-14-20) [46,](#page-14-18) [51,](#page-14-21) [53\]](#page-14-19) are developed to automate the tuning of parallel strategies. However, these works are designed mainly for homogeneous training corpora, while FlexSP focuses on flexible strategies for data with heterogeneous lengths.

Long Context Training. Efforts to optimize long context training have led to various elaborate parallel strategies, such as ring attention for LLMs [\[6,](#page-12-3) [24,](#page-13-4) [27,](#page-13-5) [29\]](#page-13-6), though they often suffer from communication overhead and inefficiencies with severe communication cost. These methods are orthogonal, and can be integrated into FlexSP seamlessly, detailed in Appendix [E.](#page-16-6)Other works aim to support long context training by extending attention mechanism [\[4,](#page-12-2) [8\]](#page-12-4) and optimizing position embedding [\[14,](#page-13-17) [55\]](#page-15-1), which are also orthogonal.

Heterogeneous Cluster Training. Training efficiency on heterogeneous GPU clusters is the main topic of these works [\[20,](#page-13-18) [25,](#page-13-19) [34,](#page-14-22) [41\]](#page-14-23), focusing on the heterogeneity of hardware. In contrast, FlexSP emphasizes flexible parallelism to address the workload heterogeneity of varied-length data.

