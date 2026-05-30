# 5 Conclusion and Future Work

This work mitigates the challenge of improving the generalization of Graph Neural Networks (GNNs) to real-world data splits and dynamic graph distributions. To tackle these shifts, we introduce GraphMETRO, a mixture-of-aligned-experts architecture, which models graph distribution shifts as mixtures of shift components, each controlling shifts in unique directions with varying complexity.

GraphMETRO distinguishes itself from traditional invariant learning methods, which often rely on environment variables to partition data. Instead, our method treats distribution shifts as mixtures, represented by the gating function's score vector, allowing for infinite environments due to the continuous nature of the score. When restricted to binary outputs, GraphMETRO can simulate finite environments, making it flexible and versatile. Furthermore, the introduction of referential invariant representation via a reference model is a key innovation of our approach.

Experimental results demonstrate that GraphMETRO consistently outperforms baseline methods on real-world datasets, achieving significant improvements. Additional synthetic studies and case analyses further validate the method's effectiveness and adaptability across diverse scenarios.

In future work, we aim to explore the broader applicability of GraphMETRO , including potential extensions to address label distributional shifts. Detailed discussions on these directions are provided in Appendix [F.](#page-20-0)

