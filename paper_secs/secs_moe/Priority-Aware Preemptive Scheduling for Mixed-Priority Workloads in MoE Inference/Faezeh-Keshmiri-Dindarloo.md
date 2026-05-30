# Faezeh Keshmiri Dindarloo\*

Unaffiliated Researcher Stockholm, Sweden

### Marco Chiesa

KTH Royal Institute of Technology Stockholm, Sweden

### 1 Introduction

Large Language Models (LLMs) have significantly advanced the domain of Natural Language Processing (NLP), enabling tasks such as machine translation, summarization [16, 20, 23], code synthesis and completion [4], and conversational AI [1, 22]. The Mixture of Experts (MoE) architecture [11], a transformer variant that selectively activates subsets of specialized feedforward layers per token, has emerged as the preferred paradigm for large-scale models that can attain superior performance while ensuring rapid inference. Although the model in its entirety can be extensive (e.g., 671 B parameters for DeepSeek-R1 [6]), the selective activation of so-called experts can substantially reduce inferencing costs and subsequently enhance adoption [13].

Data centers serving LLMs typically handle two types of requests: high priority, which are *latency-sensitive (LS)*, and low priority, which are *best-effort (BE)* [26, 28]. High priority requests might originate from users with paid subscriptions that include a *Service Level Objective* (SLO) agreement or from interactive applications like ChatBots. In contrast, BE requests could come from users on free tiers or throughput-oriented jobs such as document summarization [28]. Consequently, inference systems must identify and differentiate between LS and BE requests, ensuring low *time-to-first-token (TTFT)* and quick turnaround time for LS jobs while maintaining high throughput for BE jobs.

Current large-scale inference systems for LLMs, such as Orca [10], vLLM [27], and Hugging Face (HF) TGI [7], predominantly employ iteration-level scheduling, where new jobs are incorporated, and completed jobs are removed only at the end of each iteration. While this approach enables efficient batching, it adheres to a first-comefirst-served (FCFS) strategy, treating all inference jobs equally and failing to prioritize LS jobs over BE workloads. As a result, LS jobs frequently experience head-of-line (HOL) blocking [12], where large BE jobs with long input and output sequences monopolize resources, delaying LS execution. These delays are particularly pronounced in LLM inference workloads, where request sizes vary significantly, exacerbating scheduling inefficiencies. Addressing this issue requires an inference system capable of distinguishing between LS and BE jobs, reacting to LS job arrivals with minimal delay, and enabling fine-grained preemption of BE computations to improve overall system responsiveness.

The dynamic top-*k* token routing inherent to MoE architectures necessitates granular state management: preempted sequences must retain not only their KV cache, but also expert assignments, routing metadata, and partial computations of the *k* selected experts

to ensure deterministic resumption [\[25\]](#page-6-16). Fine-grained preemptive scheduling is becoming increasingly feasible with modern hardware advancements, such as NVLink's low-latency interconnects [\[19\]](#page-6-17) and unified memory architectures [\[18\]](#page-6-18), which are becoming increasingly prevalent in data centers. However, these advancements also require specialized scheduling mechanisms tailored to contemporary MoE models.

This paper introduces QLLM, an inference system that reduces LS job latency in MoE models through fine-grained preemption and priority-aware scheduling at the expert level. QLLM features: (1) a redesigned MoE layer with per-expert queues for dynamic buffering and low-overhead state management, and (2) a priorityaware scheduler that mitigates HOL blocking by distinguishing LS and BE jobs. Unlike existing inference systems using iteration-level execution, QLLM allows independent expert processing, allowing LS jobs to preempt BE jobs without discarding intermediate computations. An efficient state management mechanism preserves execution progress, allowing seamless BE resumption. The scheduler optimizes LS latency while maintaining high throughput.

Our evaluation on an Nvidia A100 80 GB GPU shows that QLLM reduces TTFT by up to 101.6× (avg. 65.2×), enabling SLO compliance for up to 7 jobs per second. QLLM maintains comparable or higher throughput than existing systems and reduces LS turnaround time by up to 12.8×.

Our work makes the following contributions:

Novel MoE Layer Design: We introduce per-expert queues to enable token buffering and deferred execution, eliminating rigid layer-wise synchronization. This design allows independent expert execution, enhancing scheduling flexibility.

Priority-Aware Scheduler: QLLM incorporates a scheduler that differentiates LS and BE jobs, ensuring low-latency scheduling and efficient GPU resource allocation.

Fine-Grained Expert-Level Preemption: QLLM enables BE job preemption at the expert level, reducing LS job queuing delays. This is achieved via a lightweight state management mechanism, a unified KV cache abstraction for batch updates, and per-expert queuing.

Real-World Evaluation: We evaluate QLLM on real hardware with Mixtral 8×7B, demonstrating improved LS job latency while maintaining high throughput.

Modular and Extensible Framework: QLLM integrates seamlessly with Hugging Face MoE models with minimal modifications (e.g., class inheritance), facilitating deployment, extensibility, and further research in MoE inference.

This paper discusses our initial findings, a preliminary evaluation, and limitations. Our ultimate plan is to release QLLM as an open-source project in future versions.

