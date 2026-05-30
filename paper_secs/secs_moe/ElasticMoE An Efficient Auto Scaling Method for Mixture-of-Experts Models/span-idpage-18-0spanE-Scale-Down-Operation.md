# <span id="page-18-0"></span>E Scale Down Operation

A scale-down operation in ElasticMoE reduces the number of NPUs allocated to an active inference instance, typically in response to declining load. For example, the system may transition from a configuration of DP2-TP2-EP4 across NPUs 0–3, to DP1-TP2-EP2 on NPUs 0–1.

Similar to scale-up, the Coordinator initiates the transition by signaling the HMM and IMM, which then execute their respective sub-tasks in parallel.

The HMM computes a weight reconfiguration plan that prioritizes zero-copy reuse and minimizes unnecessary P2P transfers. Since the tensor parallelism degree is fixed, both the attention weights and KV cache on NPUs 0–1 remain valid and reusable. No copying or reallocation is required for these components.

The only reconfiguration needed involves the expert layers. Expert weights residing on NPUs 2–3 must be transferred to NPUs 0–1. To support this, new physical memory pages are allocated for incoming experts, followed by P2P transfers of the corresponding weights. Once copied, Elastic-MoE remaps the virtual memory backing the expert blocks to the new physical pages using virtual memory primitives provided by ACL/CANN, maintaining the contiguous layout expected by inference kernels.

On the IMM side, the appropriate inference instance is retrieved or spawned, attaches to the newly mapped weights and cache via zero-copy, and signals readiness to the Coordinator. The Coordinator then transitions traffic to the scaled-down instance once all in-flight requests on the old instance have completed.

Overall, scale-down is a symmetric but simpler operation than scale-up, typically requiring fewer weight movements and incurring even lower peak memory overhead.