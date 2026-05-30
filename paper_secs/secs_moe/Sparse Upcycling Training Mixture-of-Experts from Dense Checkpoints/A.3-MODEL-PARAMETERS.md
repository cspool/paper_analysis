# A.3 MODEL PARAMETERS

Table 1 gives the number of parameters for models used in the main text.

<span id="page-15-3"></span><sup>&</sup>lt;sup>5</sup>Note that this is slightly different to ViT (Dosovitskiy et al., 2021), which changing the learning rate slightly based on the model variant.

<span id="page-16-2"></span>Table 1: Model sizes. The number of parameters for sparsely upcycled and MoE-from-scratch models are the same (both are of type "Sparse"). The number of parameters is also unchanged between different routing mechanisms.

| Modality | Model | Type   | Fraction of MoE Layers | # Experts | # Parameters |
|----------|-------|--------|------------------------|-----------|--------------|
| Vision   | B/32  | Dense  | –                      | –         | 101M         |
| Vision   | B/16  | Dense  | –                      | –         | 100M         |
| Vision   | L/32  | Dense  | –                      | –         | 324M         |
| Vision   | L/16  | Dense  | –                      | –         | 322M         |
| Vision   | B/32  | Sparse | 6 / 12                 | 32        | 980M         |
| Vision   | B/16  | Sparse | 6 / 12                 | 32        | 978M         |
| Vision   | L/32  | Sparse | 12 / 24                | 32        | 3.44B        |
| Vision   | L/16  | Sparse | 12 / 24                | 32        | 3.44B        |
| Language | Base  | Dense  | –                      | –         | 248M         |
| Language | Large | Dense  | –                      | –         | 783M         |
| Language | XL    | Dense  | –                      | –         | 2.85B        |
| Language | Base  | Sparse | 6 / 12                 | 32        | 2.00B        |
| Language | Large | Sparse | 12 / 24                | 32        | 7.22B        |
| Language | XL    | Sparse | 12 / 24                | 32        | 26.26B       |

