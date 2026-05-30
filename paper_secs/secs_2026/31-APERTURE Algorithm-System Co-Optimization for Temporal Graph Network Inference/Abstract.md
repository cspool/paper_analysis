# Abstract

Temporal Graph Networks (TGNs) are widely used to model evolving relationships in dynamic graphs. However, existing inference systems enforce a step-wise paradigm: processing each temporal graph sequentially with a memory update followed by aggregation. We break this dependency by decoupling memory updates from aggregation while preserving prediction accuracy, thereby enabling a global view for fine-grained parallelism control. This design unlocks new optimization opportunities but introduces three system-level challenges: managing intermediate multi-state representations, curbing memory-bound update overheads, and selecting a safe yet efficient aggregation granularity. We present APERTURE, a TGN inference framework that bridges algorithmic semantics and system design. To address the above challenges, APERTURE (1) jointly aggregates temporal states via computation graph transformation, (2) minimizes redundant memory traffic through dependency-aware update reconstruction; (3) selects the optimal granularity by analytically modeling. The experimental results show that APER-TURE achieves up to 59.3× speedup over state-of-the-art baselines without compromising accuracy.

<sup>∗</sup>Corresponding author.

![](_page_0_Picture_21.jpeg)

[This work is licensed under a Creative Commons Attribution 4.0 Interna](https://creativecommons.org/licenses/by/4.0)[tional License.](https://creativecommons.org/licenses/by/4.0)

PPoPP '26, Sydney, NSW, Australia © 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2310-0/2026/01 <https://doi.org/10.1145/3774934.3786450>

