# <span id="page-21-3"></span>E.5 COMPUTATIONAL OVERHEAD OF EACH COMPONENT

Table 14: Computational overhead (FLOPs) of each component in LLaVA-Mini.

| Methods    | Res. | FLOPs (T)      |            |             |            |        |       |  |
|------------|------|----------------|------------|-------------|------------|--------|-------|--|
|            |      | Vision Encoder | Projection | Compression | Pre-fusion | LLM    | Total |  |
| LLaVA-v1.5 | 336  | 0.349          | 0.024      | -           | -          | 8.177  | 8.55  |  |
| LLaVA-Mini | 336  | 0.349          | 0.024      | 0.001       | 0.125      | 1.460  | 1.96  |  |
| LLaVA-v1.5 | 672  | 1.745          | 0.121      | -           | -          | 38.623 | 40.49 |  |
| LLaVA-Mini | 672  | 1.745          | 0.121      | 0.009       | 1.183      | 4.131  | 7.19  |  |

LLaVA-Mini significantly reduces the computational load of LMMs by decreasing the number of vision tokens. To further study the proportion of computational load contributed by each component in LLaVA-Mini, we compute the FLOPs of each module, as shown in Table [14.](#page-21-3) The proposed compression module and pre-fusion module incur minimal computational cost, while the computation required by the LLM backbone is significantly reduced.

