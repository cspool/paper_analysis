# 2 Background

## <span id="page-2-0"></span>2.1 Distributed Inference for MoE Models

Mixture-of-Experts (MoE) models extend transformers with sparsely activated expert layers. Each layer contains a shared attention module followed by many experts, of which only a few are selected per token via a gating function (e.g., 8 of 256 in DeepSeek V3 [\[4\]](#page-12-2)). This conditional computation enables massive capacity at lower per-token compute cost.

Due to large parameter counts, KV caches, and activations, MoE inference typically spans multiple accelerators. Efficient deployment combines three forms of parallelism. Data parallelism (DP) replicates attention and feed-forward modules to process independent batches concurrently. Within each replica, Tensor parallelism (TP) shards large matrix operations (typically attention operations) across devices to reduce memory load. Expert parallelism (EP) distributes experts across devices, with tokens dynamically routed to their assigned experts. A common configuration sets EP = TP × DP, ensuring one expert per device, while other strategies replicate or unevenly distribute experts to balance load and mitigate stragglers [\[4,](#page-12-2) [22,](#page-13-3) [23\]](#page-13-4). These choices directly impact scalability, throughput, and cost efficiency.

## 2.2 Autoscaling in Cloud Environments

Autoscaling enables LLM serving systems to adapt resources to fluctuating demand while balancing cost and latency. Realworld traffic is highly bursty, with request rates surging by more than 10× within minutes in production deployments [\[26\]](#page-14-1). Without rapid scale-up, requests quickly overwhelm serving capacity and violate service-level objectives (SLOs). Conversely, when demand falls, systems must scale down promptly to avoid wasted resources.

The dominant approach is horizontal scaling, which launches or removes full serving instances on independent nodes. For example, a minimal configuration of DP2-TP2-EP4 across 4 accelerators must be scaled out by adding an entirely new DP2–TP2–EP4 replica, doubling resource use regardless of actual demand. While this strategy is easy to integrate with orchestrators such as Kubernetes, Ray Serve [\[15\]](#page-12-6), and AWS SageMaker, it is coarse-grained and slow: each instance requires container startup and weight loading. For MoE models, it is further inefficient (Fig. [4b](#page-3-0) and Fig. [1a\)](#page-0-0) since experts remain confined within isolated instances.

An alternative is vertical scaling, which enlarges the resource footprint of a single instance, for example, reconfiguring from DP2–TP2–EP4 on 4 accelerators to DP3–TP2–EP6 on 6 accelerators. This offers finer granularity and greater flexibility in expert distribution, but typically requires tearing down and reinitializing the instance, incurring cold-start latency and significant downtime that can be detrimental for SLOs, defeating the very purpose of scaling.

Both strategies therefore trade off latency, granularity, flexibility, and overhead. Overcoming these limitations is essential for efficient autoscaling of MoE inference, as we discuss in the next section.

