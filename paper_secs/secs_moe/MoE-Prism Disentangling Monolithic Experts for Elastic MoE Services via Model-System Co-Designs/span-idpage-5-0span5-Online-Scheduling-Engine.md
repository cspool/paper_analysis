# <span id="page-5-0"></span>5 Online Scheduling Engine

The MoE-Prism's scheduling engine is the online serving system that translates the architectural elasticity unlocked by our offline engine into concrete, service-level advantages. It acts as the intelligent control plane of the serving stack, making real-time, QoS-aware decisions.

The foundation of its intelligence is a lightweight performance model, created via a one-time, pre-deployment benchmark, which maps the number of active sub-experts  $(k_{active})$  to performance metrics like latency and memory usage. This process yields a lightweight lookup table or analytical model,  $C(k_{active})$ , which provides an accurate cost prediction for any given configuration. Armed with this model, the runtime can employ specialized scheduling policies. We demonstrate its effectiveness by developing policies that target two distinct and critical operating points within this space: one optimized for maximum system throughput in cloud environments, and another for minimum end-to-end latency on resource-constrained devices.

