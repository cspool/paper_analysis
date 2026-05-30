# <span id="page-10-2"></span>Table 7 Ablation study of the proposed MHLA.

(a) Ablation of init strategy on DeiT-T. LB-init denotes Locality-biased Initialization.

| LB-init | Learnable | $\operatorname{Top1-acc}(\%)$ |
|---------|-----------|-------------------------------|
|         | ✓         | 75.4                          |
| ✓       |           | 75.1                          |
| /       | /         | 75.8                          |

(**b**) Token-level head number ablation on DiT-S/2, 512px.

| Head number | $\mathrm{FID}{\downarrow}$ | $Throughput {\uparrow}$ |
|-------------|----------------------------|-------------------------|
| 4           | 79.56                      | 435                     |
| 16          | 78.63                      | 435                     |
| 64          | 79.50                      | 408                     |

throughput, implying that MHLA can reach best performance with a relatively small M and thus leading to almost no overhead.

#### 6 Conclusion

In this paper, we introduce a novel linear attention mechanism, termed Multi-Head Linear Attention (MHLA). By partitioning tokens into multiple groups, MHLA effectively preserves token-wise diversity. Without relying on additional modules such as depthwise convolutions or hybrid self-attention layers, MHLA achieves performance comparable to or even surpassing that of self-attention-based models. We envision this work as establishing a fundamental attention mechanism that can benefit a wide range of downstream applications, such as high-quality image generation, long-horizon video synthesis, and large-scale language modeling.

