# DROPS: Managing Serverless Resource Pools in Microsoft Azure Functions

Ahmed Alquraan\* University of Waterloo Abdelrahman Baba\* University of Waterloo Rafael Mendes da Silva Microsoft Research

Sameh Elnikety Microsoft Research Paul Batum Microsoft Yan Chen Microsoft

Hamid Henry Safi

Seth Fine

Samer Al-Kiswany University of Waterloo

#### Abstract

Azure Functions maintains pools of pre-warmed containers to avoid the high container-allocation latency. The size of a pool is important: a pool that is too small leads to high allocation latency, whereas a pool that is too large wastes resources and increases cost. Service providers typically oversize pools to meet service-level objectives (SLOs). Our findings indicate that the cost of maintaining pre-warmed container pools dominates the overall platform cost, motivating the need for effective pool management strategies.

We characterize container-allocation traces from Azure Functions, revealing key findings: the demand for container allocation is highly bursty, while the supply of new containers is virtually unlimited but may have long delays. Traditional resource management approaches, which rely on prediction or reactive techniques, fail to reduce costs while meeting the target SLO due to the bursty nature of the workload. These insights lead us to develop a new statistical, data-driven pool optimization method that uses historical traces to compute the size of each pool. Our evaluation shows that the proposed method meets the target SLO while reducing the operational cost by 41% compared to the static approach employed earlier in the production platform.

CCS Concepts: • Computer systems organization → Cloud computing; • Social and professional topics → Pricing and resource allocation.

*Keywords:* serverless computing, cloud platforms, resource management, performance optimization

#### **ACM Reference Format:**

Ahmed Alquraan, Abdelrahman Baba, Rafael Mendes da Silva, Sameh Elnikety, Paul Batum, Yan Chen, Hamid Henry Safi, Seth Fine, and Samer

\*Both authors contributed equally to the paper

![](_page_0_Picture_18.jpeg)

This work is licensed under a Creative Commons Attribution 4.0 International License

EUROSYS '26, Edinburgh, Scotland Uk
© 2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2212-7/26/04
https://doi.org/10.1145/3767295.3769350

Al-Kiswany. 2026. DROPS: Managing Serverless Resource Pools in Microsoft Azure Functions. In 21st European Conference on Computer Systems (EUROSYS '26), April 27–30, 2026, Edinburgh, Scotland Uk. ACM, New York, NY, USA, 17 pages. https://doi.org/10.1145/3767295. 3769350

#### 1 Introduction

Serverless computing offers an attractive computing model for developers, allowing them to express their application as event-driven functions. The provider is responsible for deploying, maintaining, and managing resources. In addition, it offers automatic dynamic scalability and a fine-grained billing model, in which users are charged only for the actual execution time. Leading serverless platforms, such as Azure Functions [1], AWS Lambda [2], and Google Cloud Run [3], support a broad range of runtime environments (e.g., Python, .NET, and Node.js) and diverse resource configurations (e.g., 1-core and 2-core containers).

The Azure Functions platform is composed of a set of virtual machines (VMs). Each VM hosts a set of smaller nested VMs (NVMs), and each nested VM runs a single container with a specific runtime. The platform uses nested virtualization (i.e., each container runs in a nested VM) to isolate tenants. The platform does not employ resource oversubscription: each nested VM reserves exclusive cores and memory from the corresponding VM. For the remainder of this paper, we use VM to refer to the VM that hosts nested VMs, and container to refer to both the container and its associated nested VM.

A user registers a serverless function and specifies its runtime version, container size, and concurrency settings. When the user invokes the function for the first time, a containerallocation request is generated. The serverless platform fulfills the allocation request by providing a container with the requested runtime and size from a pool of pre-warmed containers. A *pre-warmed container* is a container that is initialized with a runtime but not with the user code. After processing the user invocation and as soon as the container becomes idle, the container is deleted and its resources are recycled. Notice that the concurrency setting allows a single container to process multiple concurrent invocations, which reduces the cost

![](_page_1_Figure_0.jpeg)

Figure 1. The architecture of the serverless platform.

per invocation for the user and, as a side e"ect, extends the lifetime of the container.

The platform aims to satisfy every allocation request immediately using a pre-warmed container. If the containerallocation request is not satis!ed immediately, we call this an *allocation failure*. A failed allocation request is handled by creating a new container, which requires allocating resources from a VM. The ful!llment latency of the container-allocation request depends on the availability of required resources. If an existing VM has su#cient available cores and memory, a new container can be immediately created, resulting in a ful!llment latency ranging from hundreds of milliseconds to a few seconds.We refer to this scenario as a *container-allocation failure*. In contrast, if no existing VM has su#cient available cores and memory, the platform must provision a new VM, leading to a much higher ful!llment latency, typically on the order of minutes. We refer to this scenario as a *core-allocation failure*.In thiswork,we use the target success rate of containerallocation requests as the service level objective (SLO).

Figure 1 illustrates the architecture of the platform. To eliminate container-allocation failures, the platform maintains a dedicated pool of pre-warmed containers for each supported runtime and con!guration size. Additionally, it maintains a pool of ready VMs to prevent core-allocation failures and enable rapid replenishment of the container pools. The VM pool and each container pool have static sizes. The platform actively maintains these pools: whenever a resource (i.e., a VM or container) is consumed from a pool, it triggers a replenishment to create a new resource and restore the pool to its predetermined size.

Balancing performance and operational cost is a signi! cant challenge. The size of pre-warmed resource pools signi!cantly impacts the performance and the costs. Oversized pools result in substantial underutilization and unnecessary expenses, whereas undersized pools fail to meet performance expectations, leading to increased latency and potential SLO violations. Typically, these pools are oversized in production environments. For instance, the pre-warmed resource pools in our production platform incur 76% of the overall cost of the platform (Section 7).

This research investigates the optimization of pre-warmed resource pools (i.e., VM and container pools) to reduce the operational costs, while maintaining platform performance and meeting the target SLO. The contribution of this work is threefold.

First, we present a comprehensive analysis of containerallocation traces from the production environment of Azure Functions. The container-allocation trace di"ers from the function invocation trace, as not every user function invocation results in a container allocation. We highlight that this work is the !rst to study this kind of trace at a large scale. We study various features of the container-allocation workload, including runtime popularity, container lifecycle, and request arrival patterns, burstiness, and periodicity. We !nd that runtimes vary signi!cantly in their popularity and arrival rates, with a few runtimes dominating the workload. In addition, our analysis shows that the workload is highly bursty, lacks periodicity, and exhibits long-tailed creation latencies, making e#cient pool size management more challenging.

Second, we assess the viability of using traditional forecasting and reactive approaches to manage resource pools. Our !ndings (Section 3) indicate that workload forecasting using state-of-the-art statistical and machine learning models [4–9] is an ine"ective method to manage resource pools because it fails to accurately predict future workload bursts. Also, our evaluation (Section 7) shows that reactive methods [10] cannot handle bursty workload due to their inherent lagging behavior.

Motivated by our !ndings, we introduce DROPS, a novel data-driven resource optimization method to manage serverless resource pools. The core of DROPS is a statistical analysis of the demand and supply of the workload. DROPS captures workload burstiness relative to resource creation latencies through an e#cient sliding window analysis over the container-allocation trace and the container-lifecycle trace. Based on this analysis, DROPS builds a mapping that accurately determines the minimum pool size for a target SLO. Our evaluation demonstrates that DROPS generates pool sizes that consistently meet the target SLO, while reducing the operational cost by 41% compared to the static approach that was previously deployed in the production environment. DROPS is currently deployed in production in all Azure regions.

The rest of the paper is organized as follows. Section 2 presents the workload characterization. Section 3 assesses the viability of using workload forecasting tomanage resource pools. Section 4 discusses common resource pool management methods. Section 5 introduces DROPS, the proposed resource optimization method. Section 7 presents the evaluation results. Section 8 discusses related work. Finally, the paper concludes in Section 9.

