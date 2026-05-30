# Xianwei Zhang\*

Sun Yat-sen University Guangzhou, China zhangxw79@mail.sysu.edu.cn

## **Abstract**

Modern large language model (LLM) serving systems confront inefficient GPU utilization due to the fundamental mismatch between compute-intensive prefill phase and memorybound decode phase. While current practices attempt to address this by organizing these phases into hybrid batches, such solutions create an inefficient tradeoff that sacrifices either throughput or latency, leaving substantial GPU resources underutilized. For this, we identify two key root causes: 1) the prefill phase suffers from suboptimal compute utilization due to wave quantization and attention bottlenecks, and 2) hybrid batching disproportionately prioritizes latency over throughput, wasting both compute resources and memory bandwidth. To mitigate the issues, we present Bullet, a novel spatial-temporal orchestration system that eliminates these inefficiencies through fine-grained phase coordination. Bullet enables concurrent execution of prefill and decode requests, while dynamically provisioning GPU resources based on real-time performance modeling. By integrating SLO-aware scheduling and adaptive resource allocation, Bullet maximizes GPU utilization without compromising latency targets. Experimental evaluations on real-world workloads demonstrate that Bullet delivers 1.26× average throughput gains (up to 1.55×) over state-of-the-arts, while consistently meeting latency constraints.

CCS Concepts: • Computing methodologies  $\rightarrow$  Natural language processing; • Computer systems organization  $\rightarrow$  Parallel architectures.

 $^* Corresponding \ author.$ 

![](_page_0_Picture_18.jpeg)

This work is licensed under a Creative Commons Attribution 4.0 International License.

ASPLOS '26, Pittsburgh, PA, USA
© 2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2359-9/2026/03
https://doi.org/10.1145/3779212.3790135

Keywords: Large language models; GPU; Inference serving

#### **ACM Reference Format:**

Zejia Lin, Hongxin Xu, Guanyi Chen, Zhiguang Chen, Yutong Lu, and Xianwei Zhang. 2026. Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration. In Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '26), March 22–26, 2026, Pittsburgh, PA, USA. ACM, New York, NY, USA, 17 pages. https://doi.org/10.1145/3779212.3790135

## 1 Introduction

GPUs have become the predominant computing platform for large language model (LLM) services [27, 32, 50], powering a wide range of applications with varying computational and latency demands [1, 38]. As these applications continue to grow in scale and complexity, maximizing GPU utilization has become crucial for elevating service quality [46, 80]. In response, a plethora of systems have been developed to optimize different aspects of LLM serving, such as kernellevel performance [26, 43, 54], scheduling strategies [3, 53, 79], and parallelization techniques [8, 12, 36, 65].

However, the divergent computational characteristics of LLM inference make high GPU resource utilization particularly challenging. In detail, the workflow consists of a computationally intensive *prefill* phase that processes all inputs in parallel, succeeded by a memory-bound *decode* phase that generates tokens sequentially. These two contrasting phases lead to a structural imbalance in the utilization of

<span id="page-0-0"></span>![](_page_0_Figure_27.jpeg)

**Figure 1.** Request-level computation of throughput-oriented systems (top), chunked prefill (middle) and ideal (bottom).

computational resources and memory bandwidth. The dynamic nature of workload patterns further exacerbates this imbalance, forcing GPUs to alternate between these complementary resource utilization states. Moreover, the distinct characteristics of the two phases inherently introduce a **throughput-latency tradeoff** [3]. For example, throughput-oriented systems [70] (Figure 1a) prioritize the prefill, which thus increases the size of subsequent decode batches. While this improves throughput, it also penalizes latency, since prefill operations monopolize GPU resources. Therefore, an optimized system necessitates fine-grained control to balance throughput and latency while maintaining high GPU utilization, and is non-trivial to achieve.

Prior attempts have sought to address phase imbalance either by disaggregating prefill and decode across different GPUs [27, 53, 79], or by co-locating them on the same device [19, 43, 60, 78]. Prefill-decode disaggregation, however, require careful tuning of GPU allocations for each phase tailored to specific workload patterns [27], and struggle to reconfigure quickly under fluctuating request loads [79]. Furthermore, they impose heavy demands on high-bandwidth interconnects to support frequent state migration [55], significantly restricting the deployment scenarios of limited inter-node bandwidth or reinforcement learning rollout [57]. In practice, chunked prefill [3] (Figure 1b) has been pervasively adopted in production systems [19, 43, 60, 78]. This method leverages a fixed token budget to combine prefill and decode requests into hybrid batches, with longer sequences being partitioned into chunks to fit within capacity. While tunable chunk sizes provide a degree of latency control, the approach inevitably compromises throughput [3, 54]. Smaller chunk sizes often underutilize GPU capacity, while larger ones negate latency improvements, reflecting the inherent trade-off in balancing throughput and latency.

This paper targets inefficiencies in prefill-decode co-location schemes and reveals the critical hardware underutilization in existing systems through measurements and theory. First, wave quantization [18] and attention inefficiencies remain significant for widely-used chunk sizes (§2.2) despite optimized implementations [26, 48, 69]. Second, the throughput-latency tradeoff in chunked prefill exhibits sub-linear scaling with chunk sizes that prolongs execution time for successive chunks, ultimately incurring cascaded request queuing (§2.3). While spatial sharing has the potential to exploit the natural complementarity between compute-intensive prefill and memory-bound decode phases through concurrent execution (Figure 1c), effective orchestration is required to guarantee service level objectives (SLOs).

To address the suboptimal balance between latency and hardware efficiency, we present Bullet, an LLM serving system that saturates GPU resources through **spatial-temporal orchestration with fine-grained resource provisioning**. Bullet disaggregates prefill and decode in a single GPU, proactively monitors request progress and dynamically

<span id="page-1-0"></span>**Table 1.** Comparison of GPU-sharing LLM serving systems (SBP: SM-budget partitions, RF: re-partition frequency, CAM: contention-aware modeling, SG: scheduling granularity).

| System        | SBP         | RF | CAM          | SG      |
|---------------|-------------|----|--------------|---------|
| MuxServe [29] | Static      | /  | ×            | Request |
| NanoFlow [80] | Static      | /  | ×            | Chunk   |
| Semi-PD [37]  | Dynamic     | s  | $\checkmark$ | Request |
| Drift [25]    | Pre-defined | ms | ×            | Layer   |
| Bullet        | Dynamic     | ms | ✓            | Layer   |

adjusts resource provision to sustain high utilization while satisfying latency requirements. Despite GPU sharing for *independent* workloads with varying latency constraints [13, 35, 59, 63, 75, 76] have been extensively studied, Bullet tackles the unique challenges of co-locating *dependent* prefill and decode with a balance of time-to-first-token (TTFT) and time-per-output-token (TPOT). Table 1 summarizes the key features of GPU-sharing LLM serving systems.

First, an accurate performance model accounting for interphase contention under varying streaming multiprocessor (SM) budgets is critical to achieve optimal resource provisioning. Due to the dynamic batch sizes in LLM serving, the performance model exhibits exponential parameter combinations and non-linear characteristics. Existing general-purpose estimators [13, 59, 76] necessitate extensive profiling to accurately model contention. Meanwhile, LLM-specific ones are simplified as empirical profiling in NanoFlow [80], or queuing model [33] in Semi-PD [37]. In contrast, we quantify resource behavior, including compute, memory and network bandwidth, versus SM budgets. Based on the insights, we propose SM-scaling roofline model (§3.2), achieving high accuracy with few samples and profiling overhead.

Second, designing an efficient scheduler to determine the optimal SM-budget to balance TTFT and TPOT is challenging, particularly when navigating the complex, non-linear relationship between performance and SM budgets. The scheduler must be reactive to system state and request rate fluctuations to rapidly provision SMs. Bullet's SLO-aware scheduler (§3.3) achieves this by monitoring these operational states and dynamically determining the optimal SM partition in a layer-wise fashion.

Third, this fine-grained orchestration and high-frequency SM re-partitioning requires low runtime overhead. The interdependent prefill and decode instances further create unique challenges in both the control and data plane. Bullet addresses this issue by proposing concurrent execution engine (§3.4) that enables asynchronous CPU control flow and GPU execution. The engine incorporates non-intrusive model instrumentation to monitor execution progress and real-time scheduling with microsecond-level overhead. We open source Bullet at https://github.com/zejia-lin/Bullet.

In summary, the paper makes the following contributions:

- We identify the inefficiencies that hinder GPU utilization in existing LLM serving systems while navigating throughput-latency tradeoffs, and systematically analyze opportunities for boosting utilization.
- We establish accurate modeling for spatial-temporal shared phases, and introduce fine-grained control mechanisms for latency-aware resource provisioning.
- We design and implement an LLM serving system to effectively integrate the proposed techniques into readily available frameworks.
- Experimental evaluations on real-world workloads show that Bullet outperforms state-of-the-arts in both latency and throughput while achieving significantly higher GPU utilization.

