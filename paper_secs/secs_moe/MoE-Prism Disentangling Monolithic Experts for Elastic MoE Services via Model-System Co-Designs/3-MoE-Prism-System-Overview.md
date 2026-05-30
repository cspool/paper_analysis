# 3 MoE-Prism System Overview

MoE-Prism is a holistic model-system co-design that transforms rigid, monolithic MoE models into elastic assets that can be dynamically controlled at serving time. Our system's architecture is founded on a clear separation of concerns, dividing the complex problem into two distinct parts as illustrated in Figure 4.

This offline-online design is a deliberate choice. By paying a one-time, upfront computational cost during the offline phase, we unlock permanent runtime flexibility. This avoids imposing the overhead of model analysis onto the critical path of online inference, enabling the serving system to be both intelligent and highly performant.

<span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

**Figure 4.** The MoE-Prism System Architecture. The co-design consists of two phases. The *Offline Refactoring Engine* performs a one-time transformation of a standard MoE model, deconstructing monolithic experts into fine-grained sub-experts. The *Online Scheduling Engine* then leverages this elasticity to power sophisticated, QoS-aware serving strategies, such as offloading sub-experts to overcome GPU memory limits or co-optimizing batching and quality selection to maximize server throughput.

Offline Phase. The goal of this phase is to introduce fine-grained control points into a static, pre-trained MoE model without the prohibitive cost of retraining from scratch. This is handled by our **Offline Refactoring Engine**. It takes a standard MoE model as input and systematically re-architects its expert layers. It first employs a novel optimization solver to decompose each large expert into a group of smaller, functionally coherent "sub-experts". It then constructs a new, lightweight gating mechanism capable of efficiently routing requests to these sub-experts. The final output is a "refactored" model that is architecturally elastic and ready for dynamic deployment. This one-time process is detailed in Section 4.

Online Phase. The online phase is managed by the **Online Scheduling Engine**, a QoS-aware serving system designed to exploit the refactored model's elasticity. The engine acts as the brain of the serving stack, making dynamic, real-time decisions about how many sub-experts to activate. We demonstrate its power by designing specialized scheduling policies for two high-impact systems problems: (1) maximum system throughput in cloud environments, (2) minimum end-to-end latency on resource-constrained devices. The online runtime is detailed in Section 5.

