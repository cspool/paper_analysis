# E.5 How to select (K, L)

Finding the optimal (K, L) for high accuracy as well as efficiency is a long-standing problem in LSH. Similar to the traditional hyper-parameter tuning process in machine learning, K, and L are configured offline based on data subsets. In LSH, K is a more sensitive hyper-parameter than L. A slight change of K can drastically influence the number of retrieved items (i.e., budget/cost) and quality. In MagicPIG, K=8-10 is manually determined by ablations on small-scale tasks and found to be effective across various models and tasks; then, we adjust L to obtain the wanted computation cost/budget.

Here, we present two ablations to demonstrate the selection of K in Tables [12](#page-23-1) and [13.](#page-23-2)

<span id="page-23-1"></span>Table 12 Fixing the budget/cost to 4%, we ablation the performance of different (K, L) on RULER [\(Hsieh et al.,](#page-13-11) [2024\)](#page-13-11) 16K.

| Models/Config         | Full | (10, 240) | (9, 120) | (8, 65) | (7, 35) |
|-----------------------|------|-----------|----------|---------|---------|
| Llama-3.1-8B-Instruct | 94.2 | 94.2      | 92.8     | 92.3    | 88.5    |

<span id="page-23-2"></span>Table 13 Fixing L as 120, we ablation the performance of different K on RULER [\(Hsieh et al.,](#page-13-11) [2024\)](#page-13-11) 16K for Llama-3.1-8B-Instruct.

| (K, L)   | Full | (10, 120) | (9, 120) | (8, 120) | (7, 120) |
|----------|------|-----------|----------|----------|----------|
| Cost     | 1.0  | 0.012     | 0.04     | 0.11     | 0.27     |
| Accuracy | 94.2 | 92.8      | 92.8     | 94.1     | 94.3     |

If we want the computation cost to be below 5% and L below 200 (to reduce memory overhead in the CPU), then K=8-10 is a reasonable choice. Unlike K, L is not that sensitive. We select L based on the following principle after determining K: we can allow the computation cost to be smaller for larger K since the sampling is more precise. This is why we choose to use (8, 75), (9, 120), and (10, 150).

It's worth pointing out that tuning (K, L) is a challenging and long-standing problem in LSH, and we only give an example of practice in MagicPIG. More advanced hashing algorithms (such as Cross-polytope [\(Andoni](#page-12-16) [et al.,](#page-12-16) [2015\)](#page-12-16) or data-dependent ones [\(Andoni and Razenshteyn,](#page-12-17) [2015\)](#page-12-17)) can improve the trade-off between memory overhead and accuracy. We leave it as a future direction.

