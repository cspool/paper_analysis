# Making MoE-based LLM Inference Resilient with TARRAGON

Songyu Zhang *UC Riverside*

Aaron Tam *UC Riverside*

Myungjin Lee *Cisco Research*

Shixiong Qi *University of Kentucky*

K. K. Ramakrishnan *UC Riverside*

## Abstract

Mixture-of-Experts (MoE) models are increasingly used to serve LLMs at scale, but failures become common as deployment scale grows. Existing systems exhibit poor failure resilience: even a single worker failure triggers a coarse-grained, service-wide restart, discarding accumulated progress and halting the entire inference pipeline during recovery—an approach clearly ill-suited for latency-sensitive, LLM services.

We present TARRAGON, a resilient MoE inference framework that confines the failure's impact to individual workers while allowing the rest of the pipeline to continue making forward progress. TARRAGON exploits the natural separation between the attention and expert computation in MoE-based transformers, treating attention workers (AWs) and expert workers (EWs) as distinct failure domains. TARRAGON introduces a reconfigurable datapath to mask failures by rerouting requests to healthy workers. On top of this datapath, TARRAGON implements a self-healing mechanism that relaxes the tightly synchronized execution of existing MoE frameworks. For stateful AWs, TARRAGON performs asynchronous, incremental KV cache checkpointing with perrequest restoration, and for stateless EWs, it leverages residual GPU memory to deploy shadow experts. These together keep recovery cost and recomputation overhead extremely low. Our evaluation shows that, compared to state-of-the-art MegaScale-Infer, TARRAGON reduces failure-induced stalls by 160–213× (from ∼64 s down to 0.3–0.4 s) while preserving performance when no failures occur.[1](#page-0-0)

## 1 Introduction

Large Language Models (LLMs) have become a foundation of modern AI-powered applications (*e.g.,* conversation agents and code assistants). However, as model sizes grow into hundreds of billions of parameters, serving them at low latency and at reasonable cost is increasingly challenging. The Mixture-of-Experts (MoE) architecture tackles this challenge

by partitioning each dense feedforward network (FFN) layer into many smaller "experts" and using a "gating network" to activate only a subset of experts for each input token. By avoiding full-model activation on every token, MoE enables LLMs to scale total parameter capacity and throughput without a proportional growth in per-token compute cost [\[14,](#page-12-0) [21,](#page-13-0) [34\]](#page-14-0).

As MoE serving scales to hundreds of GPUs, however, failures become increasingly common. Measurements from large-scale GPU deployments show non-trivial failure rates: for example, [\[9\]](#page-12-1) reports an average per-node uptime of 99.5%, corresponding to around seven minutes of downtime per GPU node per day. So, in a 40-node (320 GPU) deployment like DeepSeek's [\[26\]](#page-13-1), this translates to roughly an 18.1% probability (*i.e.,* 1−0.99540) of at least one node outage at any given time. Even worse, [\[35\]](#page-14-1) finds that GPU-related faults alone account for 27.05% of total system errors, representing the single largest failure source.

Unfortunately, today's MoE serving systems are extremely brittle under such failures. In many systems, a single worker's failure triggers a *coarse-grained* recovery: the entire inference job is torn down and restarted [\[16,](#page-12-2) [37\]](#page-14-2), discarding all "in-flight" state (*e.g.,* partially constructed KV caches). Here, a "worker" refers to a process running on a GPU in a node participating in distributed inference, each typically executing a full transformer stack in monolithic serving engines (*e.g.,* vLLM [\[23\]](#page-13-2)). This strategy scales poorly, as restarting hundreds of workers incurs long stalls, violates latency SLOs, and wastes computation already completed. The root cause of this behavior lies in the tightly synchronized, monolithic execution model adopted by existing inference frameworks that rely on synchronous collective communications. Thus, a single worker failure can trigger full restart—an approach ill-suited to interactive LLM services, where even sub-second disruptions degrade user experience.

