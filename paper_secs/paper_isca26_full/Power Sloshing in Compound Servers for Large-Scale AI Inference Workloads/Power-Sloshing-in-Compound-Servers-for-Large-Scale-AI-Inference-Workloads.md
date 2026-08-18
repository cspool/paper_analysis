# Power Sloshing in Compound Servers for Large-Scale AI Inference Workloads

Albert Cho<sup>1\*†</sup> Jovan Stojkovic<sup>\*§</sup> Leonardo Piga\* Abhishek Dhanotia\* Sultan Mahmud Sajal\* Gefei Zuo\* Krishna Malladi\* Devon Akers\* Kalyan Subramanian\* Shobhit Kanaujia\* Alexandros Daglis<sup>†‡</sup> \*Meta <sup>†</sup>Georgia Institute of Technology <sup>‡</sup>University of Edinburgh <sup>§</sup>The University of Texas at Austin

Abstract—AI workloads are rapidly emerging as an important component of datacenter operations, with inference services in particular consuming an ever-increasing share of computational cycles. To keep pace with the growing demand for AI-driven applications, datacenters have begun deploying advanced compound servers that tightly integrate accelerators such as GPUs with traditional CPUs. While these platforms enable high performance, they also drive up the power requirements of datacenters, creating new challenges for scaling the infrastructure.

To address these challenges, we conduct a comprehensive characterization of power usage patterns in datacenters hosting AI services. We find that power consumption fluctuates widely across time, models, services, and server components. These observations reveal the inefficiency of applying fixed, uniform power limits across servers, which can result in either underprovisioning and degraded performance, or overprovisioning and wasted planned power. Motivated by these insights, we investigate dynamic power control to optimize both power consumption and performance. Through experiments on production workloads, we demonstrate that controlled power sloshing can deliver benefits, yielding up to 30% power savings. We further develop a scalable, automated algorithm for server-level power management, which could be deployed in fleet and reduce power by up to 11% without degrading the quality of service. Finally, based on our production experience, we offer practical guidelines for hardware and software co-design in next-generation AI platforms.

#### I. INTRODUCTION

The rapid proliferation of AI workloads has transformed the landscape of modern datacenter operations. Inference services have emerged as a dominant consumer of computational resources, powering a wide array of applications from personalized recommendations to real-time content moderation [9] and, more recently, to large language models (LLMs) [6], [32], [46]. As user demand for AI-driven services continues to surge [22], datacenter operators need to deliver high performance while maintaining efficiency and scalability.

To meet these demands, the industry has embraced advanced compound servers that tightly couple accelerators, such as GPUs, with CPUs optimized for AI workloads [31]. These platforms offer high computational throughput, enabling the deployment of larger and more sophisticated AI models while serving more users. However, this leap in capability comes at the cost of a rise in datacenter power requirements, straining both infrastructure and budgets.

Power consumption is a first-order concern for datacenter operators [12], [23], [40]–[42], [49]. In particular, provisioning sufficient power and cooling capacity represents a

<sup>1</sup>Work done while at Meta

challenge [43], with electricity adding to the overall problem. Moreover, as datacenters approach the limits of their power envelopes, scaling the AI services is increasingly constrained by power availability rather than compute resources [11]. This new reality has intensified focus on power management as an enabler of sustainable AI growth.

Prior studies that have analyzed power patterns in datacenters have primarily focused on traditional CPU workloads [12], [23], [40], [49] or narrow classes of AI tasks [41], [42], and typically address power management at the datacenter level. Consequently, they overlook the fine-grained, per-server-component variability and the full spectrum of heterogeneity in modern AI inference-serving environments.

To better understand these challenges, we conduct a comprehensive characterization of power usage patterns in datacenters hosting AI inference services. First, we observe that compound AI Inference servers, comprising both CPU and GPU components, are becoming an increasingly significant portion of the fleet. Second, the fleet supports a highly diverse set of AI inference services and models at the same time. This diversity encompasses a wide range of model architectures, sizes, and usage patterns. Finally, we observe that such heterogeneity leads to pronounced fluctuations in power distribution across server components. Power usage varies not only between CPUs and GPUs, but also among multiple GPUs within the same server, as different workloads stress hardware components in different ways.

Despite this variability in power usage, conventional AI inference servers assign static, independent power limits to different server components, regardless of the workload running on each server. However, different AI services exhibit widely varying demands across server components. Some may be GPU-intensive while others rely more on CPUs. Fixed power allocations cannot adapt to these differences—thus, some components end up power-constrained, throttling their performance, while others end up underutilized. Our findings highlight an opportunity: by dynamically adjusting power limits across server components (e.g., CPU and GPU) in response to each workload's real-time needs, AI inference servers can better match resource allocation to demand, enabling power savings or improved performance.

To unveil power optimization opportunities, we conduct an in-depth analysis of a few important production AI workloads running on optimized inference platforms. Our investigation reveals two findings. First, careful tuning of CPU and GPU

power limits for each AI model can yield power savings, up to 31%, without sacrificing performance. Second, the optimal frequency across CPU and GPU modules, and thus performance-per-watt setting of a given service, varies across different models. Hence, an effective power management scheme needs to take into account workload-specific properties.

While these initial results from fine-tuning are promising, such an approach has a limitation: manual fine-tuning of hardware settings for individual models is not scalable in datacenter environments. With many models being deployed and frequently updated, it is impractical to rely on per-model, hand-crafted configurations. Instead, the power management system needs to be generalized and automated. Furthermore, any automated solution must operate with minimal overhead at scale. Finally, the system should not require modifications to application code or direct access to application-level performance metrics. Instead, it should leverage readily available application-agnostic signals.

Motivated by these requirements, we design a simple yet effective *module-level power sloshing* algorithm that is scalable and easy to deploy. Rather than enforcing static, percomponent limits, our controller continuously monitors if there is an opportunity to *harvest* slack power from underutilized components and *reallocate* it to the components that are currently power-limited, maximizing Performance/Watt under a fixed aggregate server power envelope. We realize these dynamic budget splits using existing production actuators—GPU frequency scaling and CPU power capping—and a lightweight, threshold-based policy that adapts online as the workload mix changes, making the mechanism's deployment practical. Despite its simplicity, our system reduces the power consumption of production-grade AI inference workloads by 11% over the state-of-the-practice baseline.

Finally, our experience prototyping these techniques in production uncovers several practical considerations for hardware-software co-design. For example, we identify opportunities to improve the granularity and responsiveness of power management interfaces, as well as the need for better integration between workload schedulers and hardware controllers. These lessons inform our recommendations for the design of next-generation AI platforms.

In summary, this paper makes the following contributions:

- Comprehensive analysis of power usage patterns across a diverse set of AI inference workloads.
- Quantification of potential power savings through manual, workload-specific tuning for production models.
- Scalable, automated approach for dynamic power management that achieves power savings on production models, without per-workload profiling or customization.
- Guidelines for hardware and software optimizations to inform the design of next-generation AI platforms.

Paper outline: § II provides background on modern GPU inference and highlights GPU power allocation and power draw patterns in production that motivate dynamic power harvesting and reallocation mechanisms for efficiency improvements. § III characterizes the power-performance tradeoff to

<span id="page-1-1"></span>![](_page_1_Figure_10.jpeg)

Fig. 1: Normalized power usage of AI inference clusters in GPU-equipped servers and CPU-only servers over five years.

reveal improved power management opportunities on production AI systems and § IV analyzes the theoretically attainable power reduction for our models via frequency modulation, while abiding by their service-level objectives. Leveraging insights from these analyses, § V presents the design of our power management scheme, which is then evaluated with case studies on production AI inference workloads and platforms in § VI. Finally, § VII and § VIII discuss future directions and related work, respectively, and § IX concludes.

#### II. BACKGROUND AND MOTIVATION

#### <span id="page-1-0"></span>A. The Rise of GPU Inference

As AI models have grown in scale and complexity, GPU servers have become the backbone of modern datacenter inference workloads. Unlike earlier generations of CPU-centric infrastructure, today's AI services, ranging from recommendation systems to LLMs, demand the high throughput and parallelism that only accelerators such as GPUs can provide.

This shift comes at a cost: GPU servers are power-hungry, and their share of total datacenter power consumption is rapidly increasing. Fig. 1 shows the power consumption of AI inference clusters. The power is further broken down into two components: power spent on CPU-only servers and power spent on GPU-equipped servers. We observe two trends. First, over this 5-year period, the total power demand of AI clusters has tripled. Second, the fraction of total datacenter power attributable to GPU servers increased from 8.5% to 40.9%. Note that this analysis includes only AI inference servers. It does not include servers used for AI training or general-purpose compute.

As AI Inference servers become the dominant consumers of power, optimizing their efficiency is becoming an imperative for both cost control and sustainability [41], [42]. Even small improvements in GPU inference efficiency can translate into operational savings and reduced environmental impact at scale. In this paper, we focus on power management at the AI Inference server level that can maximize power and performance efficiency of AI inference workloads across our fleet.

### B. Service and Model Heterogeneity

AI fleet is heterogeneous, supporting a diverse array of services and models such as LLMs, recommendation engines, and specialized models for search, translation, and content generation. Fig. 2 breaks down power usage of fleet across dominant use cases. For CPU inference capacity, approximately 10% goes into generative AI, represented by the two

<span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

Fig. 2: Breakdown of power capacity for AI Inference for compound servers (featuring CPUs and GPUs).

large language models. The rest of CPU inference capacity is spent on various products, with recommendation system 1 (RS1) consuming about 60% of the CPU capacity. About 40% is spent on low complexity workloads, the rest on high complexity products. Our primary metric to classify model complexity is size, i.e., number of parameters and layers. In GPU inference, about 20% of the capacity is spent on generative AI, 70% on RS1, and 10% on other services. All workloads running on GPUs are high complexity. Each of these use cases comprises many different downstream services and AI models. The deployed models differ greatly in size and architecture. Some are compact and can be efficiently served on a single CPU or GPU, while others, such as emerging LLMs with many parameters, require multi-GPU configurations for a single inference request.

This architectural diversity is further compounded by the wide range of service use cases and load patterns. For example, real-time recommendation systems often experience high and unpredictable query rates, demanding low-latency responses and sustained resource utilization. In contrast, batch analytics or background workloads tend to have more predictable, lower-intensity demands, and may leave resources underutilized for extended periods.

Service-level objectives (SLOs) such as latency and throughput further influence how resources are provisioned and utilized. Services with stringent performance requirements may be allocated more generous power budgets to ensure responsiveness, while others can operate efficiently with less.

Taken together, factors such as model size, service load, and performance requirements result in highly non-uniform power consumption patterns across workloads in the fleet. Fig. 3 shows the cumulative distribution function (CDF) of power utilization across AI inference services. We observe that the majority of services under-utilize their assigned power budgets. For instance, 60% of services use less than 80% of their maximum power limit. On the other hand, some services need their full power budgets and could further improve their performance if allocated more power.

This high degree of heterogeneity in large-scale AI fleets, though underexplored in previous research, underscores the need for power management schemes that can adapt to diverse and evolving workload demands.

<span id="page-2-1"></span>![](_page_2_Figure_7.jpeg)

Fig. 3: CDF of power utilization across services.

#### C. Variations in Power Usage Across Server Components

A typical compound AI Inference server is composed of multiple GPUs and a CPU host [31]. Power limits (TDPs) are statically assigned to each component. For example, a platform with 8 GPUs and a CPU host assigns 1 kW per GPU and 300 W for the CPU. Unfortunately, static partitioning is often inefficient given the diversity of workloads and their dynamic behavior. Throughout the paper, "TDP" refers to the device's (configurable) software-set power cap rather than the manufacturer's peak power rating.

The usage of CPU and GPU components within a server varies across services. For instance, some AI services perform extensive data pre-processing, such as feature extraction and input normalization, as well as post-processing, like result aggregation and formatting on the CPU. In these cases, the CPU may reach its 300 W power limit, throttling the service, while the GPUs remain underutilized and consume less than their allocated power budget. In contrast, some other services, such as long-context LLMs, use host CPUs only for request scheduling; thus, they have low CPU utilization, while the GPU usage is very high. Hence, setting fixed power limits for CPUs and GPUs independently is often inefficient, as one component may require more power to sustain performance while the other remains underutilized.

Moreover, it is common to co-locate multiple services on the same AI Inference server for better resource utilization. For example, two services may each be assigned 4 GPUs on an 8-GPU AI Inference server. If one service is GPU-bound and fully utilizes its GPUs, its performance will be restricted by the available assigned power limit. On the other hand, if the other is lightly loaded or idle, its GPUs will consume far less than their power cap, resulting in unused power headroom.

Fig. 4 shows the power draw of 8 GPUs within an AI Inference server running multiple collocated production AI services over a day. Power usage across GPUs fluctuates over time, and, most of the time, GPUs differ in their respective power draw. This example also shows that even within a single service, workload intensity fluctuates over time due to diurnal patterns, traffic spikes, or model updates, leading to periods where some server components are underutilized.

Overall, these mismatches between static power allocation and diverse and dynamic workload demands result in power misallocation and missed performance opportunities.

