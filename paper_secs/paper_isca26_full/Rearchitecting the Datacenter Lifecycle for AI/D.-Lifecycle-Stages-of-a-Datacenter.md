# *D. Lifecycle Stages of a Datacenter*

Overview. Based on these AI workloads, hardware, and infrastructure trends, we examine the datacenter lifecycle to identify

<span id="page-2-0"></span>

| Stage        | Description                              | Timeline        |
|--------------|------------------------------------------|-----------------|
| Build        | Site selection and facility construction | 15–30 years     |
| IT provision | IT hardware deployment and upgrades      | 4–6 years [111] |
| Operate      | Workload sched. and resource manag.      | Per inference   |

TABLE I: Lifecycle stages for a datacenter.

<span id="page-2-1"></span>

| Stage        | Traditional Approach Characteristics                                                                                                                |
|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| Build        | Hierarchical power; Air cooling; Ethernet.                                                                                                          |
| IT provision | Fixed per-server lifecycle; New server generations<br>released every 2–3 years; Gradual replacement.                                                |
| Operate      | Services tied to fixed hardware configurations; In<br>stances migrated to new hardware when released;<br>Legacy applications remain on old servers. |

TABLE II: Overview of the lifecycle management for traditional datacenters handling general-purpose CPU workloads.

opportunities for reducing TCO. [Table I](#page-2-0) breaks the lifecycle into: *build*, *IT provisioning*, and *operate*. These stages help us explore how traditional lifecycle policies must be revisited to address the scale, density, and performance demands of AI. On this foundation, we develop a TCO model to evaluate costs across design choices over the datacenter's lifetime.

*Build.* This stage designs and constructs the datacenter facility, setting long-term constraints on utility capacity, substation feeds, power distribution (flat *vs.* hierarchical), cooling (air *vs.* liquid), floor space, and network fabric. These choices determine maximum rack density, define fault domains, and affect future upgrades such as liquid cooling or higher-voltage buses. Networking decisions (Ethernet *vs.* InfiniBand, optical reach, oversubscription) influence job scaling efficiency and east–west traffic costs, critical for large AI models.

*IT provisioning.* This stage determines when and how to deploy new accelerators and retire or repurpose older ones, balancing performance-per-watt improvements, hardware cost, software maturity, depreciation schedules, and risk of underutilized power or cooling. IT provisioning may involve mixedgeneration GPU pools or reassigning older GPUs to lowerperformance workloads (*e.g.*, fine-tuning or batch analytics). *Operate.* Decisions here focus on model placement, query scheduling, and efficient execution. Placement aligns models with accelerator generations for optimal performance-to-cost, while scheduling considers service-level objectives (SLOs), query complexity, and flexible workload timing or location. Execution uses AI-specific optimizations (*e.g.*, batching, quantization, speculative decoding, distillation, and disaggregation) to minimize cost per query while meeting SLOs.

Traditional Approach. [Table II](#page-2-1) summarizes the lifecycle for general-purpose datacenters.

*Build.* They rely on a conservative, uniform infrastructure. Power distribution usually follows a hierarchical topology: from the colo-level to rows, and then to individual racks, with each level having its own power caps [\[120\]](#page-15-10). Cooling is airbased and networking uses standard Ethernet [\[8\]](#page-13-24), [\[34\]](#page-13-25).

*IT provisioning.* Servers follow a fixed 4–6 year lifecycle, with new hardware released every 2–3 years and older servers phased out accordingly [\[111\]](#page-15-22).

*Operate.* Services run on fixed hardware generations. New services migrate to the latest servers, while legacy applications remain on older ones. This ensures stability and predictability but limits flexibility to leverage hardware heterogeneity or optimize performance for specific workloads.

Rearchitecting for AI. AI workloads challenge traditional datacenter design. Modern accelerators' high power and thermal demands make high-density racks and liquid cooling more valuable, while space and density constraints favor scale-up architectures such as NVLink designs [\[90\]](#page-15-11).

*Memory capacity and bandwidth* have increased to support larger model contexts, enabling more complex workloads but raising costs, making efficient provisioning essential. Likewise, *interconnect* performance is crucial for parallel efficiency, with low bandwidth or high latency limiting scaling.

Trade-offs between cost and performance must be evaluated both within each stage (build, IT provisioning, and operate) and across them. For example, during *operation*, separating prefill and decode across servers [\[94\]](#page-15-2) enables using heterogeneous hardware and influences *build* and *refresh* strategies.

