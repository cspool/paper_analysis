# <span id="page-7-0"></span>**5.3.2** Visualization Analysis for $\alpha$ and $\beta$

To further validate the effectiveness of our proposed method, we visualized the variation of the updated intra-layer scores  $\alpha$  and inter-layer scores  $\beta$  after the pruning stage. As shown in Figure 4a, the distribution of intra-layer importance scores  $\alpha$  reveals that experts in layers 1–15 tend to have higher average scores compared to those in layers 16–32. This suggests that shallower layers generally play a more significant role in the overall model. Figure 4b illustrates the inter-layer importance scores, which corroborate the intra-layer observations. The overall trend indicates that the alternating update strategy effectively captures both intra- and inter-layer dependencies, ensuring that the MoE

<span id="page-8-0"></span>![](_page_8_Figure_0.jpeg)

Figure 4: Visualization of values distribution for intra-layer scores  $\alpha$  and inter-layer scores  $\beta$  on Mixtral 8×7B when r = 50%..

model retains critical information from shallower layers. Furthermore, a closer examination of layer 2 reveals that two experts. Specifically, the fourth and eighth experts exhibit markedly high  $\alpha$  values relative to their peers. This shows that these experts are consistently considered highly important by the model. Conversely, the remaining experts in layer 2 generally have low importance scores, indicating that the pruning strategy in this layer is not governed by a single importance criterion. Instead, it shows a clear preference for retaining these two experts through a global selection. Overall, these empirical findings further confirm the efficacy of our proposed differentiable expert pruning approach and underscore the synergistic relationship between  $\alpha$  and  $\beta$ .

#### 5.3.3 Computation Cost Analysis

We further analyze the efficiency of our DiEP during the pruning and inference stages. For pruning, as shown in Table 3, our baseline (NAEE), using an exhaustive heuristic search, becomes computationally prohibitive for models with large expert pools like Deepseek-MoE-16B and Owen2-57B-14A. In contrast, our DiEP, with only a 0.01% parameter overhead, maintains consistent pruning time and achieves superior performance regardless of model architecture or expert count. Furthermore, Table 4 shows DiEP's inference cost reductions on Mixtral 8×7B in terms of latency and GPU

<span id="page-8-1"></span>Table 3: Pruning time comparison of our DiEP and NAEE on different models under 25% expert sparsity.

| Method     | Mixtral 8×7B | Mixtral 8×22B | Deepseek-MoE-16B | Qwen2-57B-14A |
|------------|--------------|---------------|------------------|---------------|
| NAEE       | 1.31h        | 1.57h         | ≈ 94000d         | ≈ 113000d     |
| DiEP(Ours) | 0.23h        | 0.31h         | 0.28h            | 0.34h         |

Table 4: Inference cost analysis on Mixtral  $8 \times 7B$  after expert pruning.

| r   | Pruning  | Skipping | Avg. Acc | Speedup ↑ | GPU ↓         |
|-----|----------|----------|----------|-----------|---------------|
| 0%  |          |          | 65.1     | 1.00×     | 1.00 ×        |
| 0%  |          | ✓        | 64.1     | 1.07 ×    | $1.00 \times$ |
| 25% | <b>√</b> |          | 63.8     | 1.18×     | 0.76 ×        |
| 25% | ✓        | ✓        | 63.3     | 1.21×     | $0.76 \times$ |
| 50% | <b>√</b> |          | 59.9     | 1.26×     | 0.52 ×        |
| 50% | ✓        | ✓        | 59.6     | 1.28 ×    | 0.52 ×        |

memory. Our DiEP enhances inference efficiency via an online expert skipping, which adjusts router weights according to expert similarity with negligible loss in performance. Using half the experts, DiEP retains nearly 92% performance on Mixtral  $8\times7B$ , achieving  $1.28\times$  token generation speedup and 48% memory savings. We provide more experimental analysis for ablation study in Appendix A.

