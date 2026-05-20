## **TetriServe: Efficiently Serving Mixed DiT Workloads** 

Shiqi He[∗] 

## Wenxuan Tan 

Runyu Lu[∗] runyulu@umich.edu University of Michigan Ann Arbor, Michigan, USA 

shiqihe@umich.edu University of Michigan Ann Arbor, Michigan, USA 

wtan45@wisc.edu University of Wisconsin-Madison Madison, Wisconsin, USA 

## Shenggui Li 

## Ruofan Wu 

## Jeff J. Ma 

jeffjma@umich.edu University of Michigan Ann Arbor, Michigan, USA 

shenggui001@e.ntu.edu.sg Nanyang Technological University Singapore, Singapore 

ruofanw@umich.edu University of Michigan Ann Arbor, Michigan, USA 

Ang Chen 

## Mosharaf Chowdhury 

chenang@umich.edu University of Michigan Ann Arbor, Michigan, USA 

mosharaf@umich.edu University of Michigan Ann Arbor, Michigan, USA 

## **Abstract** 

_**Keywords:**_ diffusion transformer serving, gpu resource scheduling, sequence parallelism 

Diffusion Transformer (DiT) models excel at generating highquality images through iterative denoising steps, but serving them under strict Service Level Objectives (SLOs) is challenging due to their high computational cost, particularly at larger resolutions. Existing serving systems use fixed-degree sequence parallelism, which is inefficient for heterogeneous workloads with mixed resolutions and deadlines, leading to poor GPU utilization and low SLO attainment. 

## **ACM Reference Format:** 

Runyu Lu, Shiqi He, Wenxuan Tan, Shenggui Li, Ruofan Wu, Jeff J. Ma, Ang Chen, and Mosharaf Chowdhury. 2026. TetriServe: Efficiently Serving Mixed DiT Workloads. In _Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS ’26), March 22–26, 2026, Pittsburgh, PA, USA._ ACM, New York, NY, USA, 16 pages. https://doi.org/10.1145/3779212.3790233 

In this paper, we propose step-level sequence parallelism to dynamically adjust the degree of parallelism of individual requests according to their deadlines. We present TetriServe[1] , a DiT serving system that implements this strategy for highly efficient image generation. Specifically, TetriServe introduces a novel round-based scheduling mechanism that improves SLO attainment by (1) discretizing time into fixed rounds to make deadline-aware scheduling tractable, (2) adapting parallelism at the step level and minimizing GPU hour consumption, and (3) jointly packing requests to minimize late completions. Extensive evaluation on stateof-the-art DiT models shows that TetriServe achieves up to 32% higher SLO attainment compared to existing solutions without degrading image quality. 

## **1 Introduction** 

Diffusion models [3, 4, 16, 21, 34, 37, 38] have significantly advanced text-to-image and text-to-video generation, enabling photorealistic content from natural language descriptions. They now power a wide range of commercial and creative services such as OpenAI Sora [7] and Adobe Firefly [2]. At the core of these breakthroughs are _Diffusion Transformers (DiTs)_ [34], which have become the backbone of leading models including Stable Diffusion 3 (SD3) [3] and FLUX.1-dev [21]. By replacing conventional UNet architectures [16, 36], DiTs achieve higher fidelity by iteratively refining a full-image latent representation over a sequence of discrete denoising steps, setting a new standard for generation quality. 

## _**CCS Concepts:**_ • **Computer systems organization** → **Cloud computing** . 

As DiT models move into production, _online DiT serving_ becomes a key systems challenge. Deployments such as Flux AI [13] must satisfy strict service level objectives (SLOs) in the form of a _deadline_ for each request while sharing a fixed GPU pool across many users to minimize cost. Serving is particularly challenging because requests arrive with heterogeneous output resolutions and tight deadlines. 

∗Both authors contributed equally to this research. 1TetriServe is available at https://github.com/DiT-Serving/TetriServe. 

Despite advances in LLM serving [10, 20, 27–29, 33, 43, 47], these solutions are insufficient: DiTs have fundamentally different serving characteristics. Specifically, DiT inference differs from LLMs in three ways: (i) it is stateless, requiring no KV cache; (ii) it is compute-bound, as multiple denoising steps operate on the full set of latent image tokens; and (iii) 

This work is licensed under a Creative Commons Attribution 4.0 International License. 

_ASPLOS ’26, Pittsburgh, PA, USA_ 

© 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2359-9/2026/03 https://doi.org/10.1145/3779212.3790233 

1982 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

Runyu Lu et al. 

**==> picture [176 x 206] intentionally omitted <==**

**----- Start of picture text -----**<br>
Small<br>Medium<br>Large<br>GPU 0 1 2 3 4 5<br>GPU 1 1 2 3 4 5<br>GPU 2 1 2 3 4 5<br>GPU 3<br>(a) xDiT (SP=1)<br>GPU 0<br>GPU 1<br>1 2 3 4 5 1 2 3 4 5 1 2 3 4 5<br>GPU 2<br>GPU 3<br>(b) xDiT (SP=4)<br>GPU 0 1 2 3 4 5<br>3 4<br>GPU 1<br>1 2 5<br>GPU 2<br>1 4 5<br>GPU 3 2 3 Time<br>**----- End of picture text -----**<br>


**(c) TetriServe** 

**Figure 1.** Three DiT serving requests—each with 5 denoising steps—arrive over time with different SLOs and output resolutions. DiT serving solutions using static parallelism cannot adapt and fail to meet multiple SLOs. TetriServe meets more SLOs via SLO-aware scheduling and packing. 

model sizes are small enough to fit on a single GPU. Consequently, generating a high-resolution 2048 × 2048 image on a single H100 GPU can take up to a minute, while a 4096×4096 image may exceed ten minutes. To meet the stringent latency demands of online serving, parallelism is essential. 

The most common approach for parallelizing DiTs is _sequence parallelism (SP)_ [18, 25], which partitions the sequence of image tokens across GPUs. However, simply applying a fixed degree of SP to all requests is inefficient and leads to poor SLO attainment. This is because the optimal degree of parallelism is highly sensitive to the input image resolution; a configuration that is ideal for one resolution can be detrimental to another. As shown in the toy example in Figure 1, the fixed-degree SP approach creates a fundamental tradeoff: low degrees of parallelism (e.g., SP=1 or 2) are efficient for small inputs but underutilize the GPU cluster for large ones by leaving some GPUs idle and prolonging request runtime, while high degrees of parallelism (e.g., SP=4 or 8) accelerate large inputs but introduce excessive communication overhead for small ones, leading to head-of-line blocking. Compounding this issue, existing DiT inference engines [12] are non-preemptive: once a request begins execution with a fixed degree of parallelism, it holds its allocated GPU(s) until completion, preventing more optimal scheduling of other requests in the queue. 

We observe that _step-level scheduling_ , in which the degree of parallelism is adjusted across steps within each request 

based on its resolution and deadline, can significantly improve the serving efficiency of mixed DiT workloads. Highresolution or urgent requests can be accelerated with more GPUs, while smaller or less urgent ones conserve resources. Unfortunately, we prove that finding a globally optimal steplevel schedule that maximizes deadline satisfaction under a fixed GPU budget is NP-hard (§4.1). In addition, the online arrival of requests and the need for millisecond-level scheduling decisions make exhaustive optimization infeasible. 

We present _TetriServe_ , a step-level DiT serving system designed to maximize SLO attainment under deadline constraints. At its core, TetriServe introduces a _deadline-aware round-based scheduler_ that transforms the continuous time in the serving problem into a sequence of tractable, fixedduration rounds. In each round, the scheduler decides which requests to serve and at what GPU parallelism degree. To make these decisions, TetriServe leverages a cost model that profiles per-step latency as a function of GPU count and identifies the _minimal feasible GPU allocation_ for each request that can still meet its deadline. This allows TetriServe to construct a set of candidate allocations and perform request packing with the explicit goal of minimizing the number of requests that would otherwise become late in the next round. 

TetriServe further enhances GPU efficiency while preserving request deadlines. It uses _selective continuous batching_ to merge steps across small-resolution requests, reducing kernel launch overhead and boosting throughput. Meanwhile, _GPU placement preservation_ and _work-conserving elastic scale-up_ ensure idle GPUs are utilized without remapping distributed jobs. Together with the round-based scheduler, these techniques allow TetriServe to handle diverse DiT workloads—from small to large resolutions—while substantially improving deadline satisfaction over fixed-degree baselines. 

We evaluate TetriServe on popular open-source DiT models (FLUX.1-dev and SD3) and different hardware platforms (8×H100 and 4×A40 nodes). We show that TetriServe consistently outperforms xDiT [12]—a DiT-serving engine that allows different fixed SP configurations—across diverse experimental settings by up to 32% in terms of SLO attainment ratio. TetriServe is also robust to bursty request arrival patterns, diverse workload mixes, and different model–hardware combinations. 

We summarize the contributions as follows: 

- We cast DiT serving as a step-level GPU scheduling problem and prove its NP-hardness. 

- We present TetriServe, a deadline-aware round-based scheduler that minimizes late completions via dynamic programming. 

- We show that TetriServe achieves substantial gains in SLO attainment over fixed-degree baselines on state-ofthe-art DiT models while maintaining image quality. 

1983 

TetriServe : Efficiently Serving Mixed DiT Workloads 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

## **2 Motivation** 

Serving DiT models has become a popular workload for modern image generation systems [2, 12]. DiT inference is both compute-intensive and latency-sensitive. To better understand the challenges of serving such workloads, in this section, we discuss DiT background, workload characteristics, and the resulting opportunities and challenges. 

## **2.1 DiT Background** 

Diffusion models [7, 16, 34, 37, 38] have significantly advanced text-to-image and text-to-video generation, enabling photorealistic content from natural language descriptions. Each step operates on the full latent representation, removing noise based on a learned denoising function. Although early diffusion models used _UNet_ architectures [16, 36], modern high-quality image generators use _Diffusion Transformers (DiTs)_ [9, 34] as their backbone. DiTs use attention [41] to capture global context and long-range dependencies. 

_**DiT vs. LLM Parallelism.**_ Although both DiTs and LLMs are built upon the Transformer architecture, their inference characteristics diverge significantly, requiring different parallelism strategies. Traditional model-sharding strategies for LLMs, such as tensor and pipeline parallelism, are inefficient for DiTs. This is because DiT models are typically small enough to fit on a single GPU. For example, the largest opensource text-to-image DiT has only 12B parameters [21] and fits comfortably on a single 80GB H100 GPU. Consequently, applying model sharding introduces unnecessary communication overhead without the benefit of accommodating a larger model, resulting in poor hardware utilization. 

DiTs adopt _sequence parallelism (SP)_ [18, 23, 25], a more efficient parallel approach tailored to their compute-bound nature. In SP, token sequences (image tokens) are distributed across GPUs, enabling collaborative computation within each transformer layer. Two representative implementations are _Ulysses attention_ [18], which uses all-to-all collectives to transpose tokens and heads across GPUs before local attention, and _Ring attention_ [25], which arranges GPUs in a ring and passes partial Q, K, V slices peer-to-peer, overlapping communication with computation. In practice, Ulysses attention is often preferred on systems with high-bandwidth interconnects like NVLink, as its use of collective primitives can be more efficient [12]. 

## **2.2 Characteristics of DiT Workloads** 

DiT serving exhibits distinctive workload characteristics that affect the design of scheduling and resource management. 

_**Heterogeneous Inputs.**_ Unlike LLM workloads, where input text can vary widely in length, DiT serving workloads are characterized by a small, discrete set of possible input image resolutions [13, 39]. In this work, we focus on four 

**Table 1.** Characteristics of representative input sizes for the FLUX.1-dev model [21], including latent tokens and computational cost (TFLOPS). Execution stability (CV) is measured over 20 steps on 8xH100 GPU for different sequence parallelism (SP) degrees. 

**==> picture [236 x 165] intentionally omitted <==**

**----- Start of picture text -----**<br>
Image Size Tokens TFLOPs SP=1 SP=2 SP=4 SP=8<br>256 × 256 256 556.48 0.13% 0.31% 0.67% 0.62%<br>512 × 512 1024 1388.24 0.06% 0.15% 0.14% 0.53%<br>1024 × 1024 4096 5045.92 0.07% 0.12% 0.04% 0.09%<br>2048 × 2048 16384 24964.72 0.05% 0.11% 0.14% 0.28%<br>40 Degree of Parallelism 2<br>Degree of Parallelism 4<br>30 Degree of Parallelism 8<br>20<br>10<br>0<br>256x256 512x512 1024x1024 2048x2048<br>Image Size<br>Comm Percentage (%)<br>**----- End of picture text -----**<br>


**Figure 2.** Percentage of time spent in communication for FLUX.1-dev for four resolutions on an 8×H100 server (Batch Size = 4). Larger resolutions benefit more from increased parallelism because of relatively less communication overhead. 

representative resolutions common in production environments; their characteristics for the FLUX.1-dev model [21] are detailed in Table 1. Despite the small number of distinct input sizes, the substantial differences in their computational demands still lead to highly heterogeneous resource requirements across requests. 

_**Predictable Execution.**_ Despite input diversity, DiT inference remains compute-bound and therefore exhibits stable per-step runtimes across a wide range of input resolutions. As shown in Table 1, execution time is highly stable: profiling over 100 runs with varying sequence-parallel degrees yields a coefficient of variation (CV) below 0.7% in all cases. This low variability indicates that DiT model inference is predictable across resolutions and degrees of parallelism, enabling accurate performance modeling and effective deadline-aware scheduling. 

**Insight 1:** _DiT workloads consist of heterogeneous input requests with different output resolutions, but per-step runtime for each resolution is highly predictable._ 

_**Scaling Efficiency of Sequence Parallelism.**_ Sequence parallelism distributes tokens across GPUs, but its scaling efficiency is sublinear to the degree of parallelism. Two factors drive this: (i) communication overhead from collectives (all-to-all or ring exchanges) that scales with the degree of parallelism and sequence length; and (ii) reduced per-GPU 

1984 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

Runyu Lu et al. 

**==> picture [241 x 229] intentionally omitted <==**

**----- Start of picture text -----**<br>
BS=1 BS=2 BS=4<br>3<br>2 4<br>1 2<br>0 1 2 4 8 0 1 2 4 8<br>Ulysses Degree Ulysses Degree<br>(a)  Image Size: 256×256 (b)  Image Size: 512×512<br>100<br>15<br>75<br>10 50<br>5 25<br>0 1 2 4 8 0 1 2 4 8<br>Ulysses Degree Ulysses Degree<br>(c)  Image Size: 1024×1024 (d)  Image Size: 2048×2048<br>End-to-End Time (s) End-to-End Time (s)<br>End-to-End Time (s) End-to-End Time (s)<br>**----- End of picture text -----**<br>


**Figure 3.** End-to-end scaling efficiency of FLUX.1-dev for four resolutions on an 8×H100 server for different batch size (BS). Efficiency scales sublinearly. Larger resolutions benefit more from increased parallelism, while smaller resolutions exhibit limited scalability. Note different Y-axes scales. 

kernel efficiency when workloads are split, lowering occupancy and cache locality. Figure 2 quantifies this by showing the communication percentage across image sizes and degrees of parallelism. For small inputs (e.g., 256 × 256 and 512 × 512), increasing the degree of parallelism rapidly increases the communication percentage, exceeding 30% at higher degrees. In this case, communication dominates execution time, leading to poor scaling and decreasing the benefits from additional GPUs. Figure 3 shows that small inputs (e.g., 256 × 256, 512 × 512) underutilize GPUs and scale poorly, while larger inputs (e.g., 1024 × 1024, 2048 × 2048) improve efficiency though computation remains the bottleneck. This explains why in Figure 1, latency does not scale linearly with the number of GPUs. 

**Insight 2:** _Sequence parallelism in DiT workloads scales sublinearly with the degree of parallelism and differently for each input resolution._ 

## **2.3 Challenges and Opportunities** 

_**Limitations of Current Solutions.**_ Conventional serving strategies using a fixed degree of parallelism are ill-suited for the heterogeneous nature of DiT workloads, a limitation illustrated in the toy example in Figure 1. With data parallelism (xDiT, SP=1), the small request meets its deadline, but the larger requests fail due to insufficient processing speed. Conversely, a high fixed degree of parallelism (xDiT, SP=4) 

**==> picture [242 x 116] intentionally omitted <==**

**----- Start of picture text -----**<br>
xDiT (SP=1) xDiT (SP=2) xDiT (SP=4) xDiT (SP=8)<br>256x256<br>0.6<br>1.0<br>0.8<br>0.5 0.6<br>0.4<br>0.2<br>0.4<br>0.3<br>0.2<br>6 9 12 18<br>Arrival Rate (req/min) 1024x1024<br>512x512<br>2048x2048<br>SLO Attainment Ratio<br>SLO Attainment Ratio (SAR)<br>**----- End of picture text -----**<br>


**(a)** Fixed Degree Performance **(b)** Breakdown by Resolution 

**Figure 4.** Performance of fixed degree xDiT variants under the Uniform workload. (a) The overall SLO Attainment Ratio (SAR) is low for all fixed strategies. (b) The spider plot, shown for a representative arrival rate of 12 req/min, reveals the underlying reason: low SP degrees fail on large resolutions, while high SP degrees perform poorly on small ones. 

handles the large request well, but it still misses the deadline (along with the medium one) due to head-of-line blocking and inefficient resource use of the small request. 

Experimental results confirm this trade-off. As shown in Figure 4a, under a Uniform workload with a tight SLO Scale of 1 _._ 0×, no fixed-parallelism strategy achieves an SLO Attainment Ratio (SAR) above 0.6. The spider plot in Figure 4b reveals why: each fixed strategy only works well for specific resolutions. _SP=1_ and _SP=2_ achieve near-perfect SAR for 256 × 256 images but fail completely for 2048 × 2048, while _SP=4_ and _SP=8_ handle 2048 × 2048 effectively but perform poorly on smaller resolutions due to scaling inefficiency and head-of-line blocking. No single parallelism degree works across the board. 

_**Optimization Opportunities.**_ The limitations of fixed parallelism highlight a key opportunity: moving to dynamic, _step-level sequence parallelism_ . As shown in Figure 1(c), our approach, TetriServe, meets all three deadlines by adapting the degree of parallelism for each request at the step level. It assigns fewer GPUs to the initial steps of the medium request, freeing up resources, and then scales up to meet the deadline, thus avoiding the rigid trade-offs of fixed strategies. 

This flexibility to adjust the sequence parallelism degree _per step_ allows a scheduler to allocate more GPUs when deadlines are tight and fewer when they are not, freeing capacity for other requests. By exploiting DiTs’ predictable step execution times and heterogeneous scaling behavior, this approach enables finer-grained resource shaping and better SLO attainment than conventional fixed-SP policies. 

**Insight 3:** _Step-level parallelism adapts GPU allocation to request deadlines, avoiding the resource waste of fixed parallelism and improving SLO attainment._ 

1985 

TetriServe : Efficiently Serving Mixed DiT Workloads 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

**==> picture [241 x 110] intentionally omitted <==**

**----- Start of picture text -----**<br>
Scheduler Execution Engine<br>Deadline-Aware Round Request GPU  Latent<br>GPU Allocation Packing Submit Worker-0 Manager<br>request<br>steps GPU  R1<br>Worker-1 Latent<br>GPU<br>Request Tracker Worker-2 R2<br>Req R1 1 2 … 20 … 50 … Latent<br>GPU  …<br>Resp R2 1 … 15 … 49 50 Requeststatus Worker-7<br>…<br>**----- End of picture text -----**<br>


**Figure 5.** TetriServe architecture and request lifecycle. 

## **3 TetriServe Overview** 

TetriServe allows more DiT serving requests with heterogeneous output resolutions to meet their SLOs by judiciously scheduling and packing them on shared GPU resources. In this section, we provide an overview of how TetriServe fits in the DiT serving lifecycle to help the reader follow the subsequent sections. 

_**System Components.**_ TetriServe is designed around a scheduler that makes deadline-aware GPU allocation decisions in a round-based manner. Its key components are: 

- **Request Tracker:** Maintains metadata on active requests, including resolutions, deadlines, and execution states (e.g., remaining steps). 

- **Scheduler:** The core component consists of _deadlineaware GPU allocation_ and _round-based request packing_ . At every round, it minimizes individual requests’ GPU consumption while maximizing SLO attainment. 

- **Execution Engine:** A distributed pool of GPU workers that execute assigned diffusion steps in parallel. 

- **Latent Manager:** Handles intermediate latent representations across steps, reducing redundant computation and memory overhead. 

Together, these components enable TetriServe to adapt resource allocation at millisecond scale, sustaining high throughput and SLO attainment for heterogeneous DiT workloads. 

_**Request Lifecycle.**_ When a request arrives, the _Request Tracker_ records its resolution, state, and deadline. The _Scheduler_ then places it into the next scheduling round _○_ 1 , where a deadline-aware policy determines GPU allocations in terms of step numbers for each request for one round. For example, in Figure 5, it selects Request 1 to run 20 steps on 2 GPUs (orange) and Request 2 to run 15 steps on 1 GPU (blue) for the scheduling round. Different requests are dispatched to GPU workers in the _Execution Engine ○_ 2 , which compute diffusion steps and produce intermediate latents managed by the _Latent Manager ○_ 3 . Upon completion, workers notify the request tracker to update dependent steps _○_ 4 . After all steps finish, the final output is returned to the user. 

## **4 Deadline-Aware Round-Based Scheduler** 

TetriServe introduces a deadline-aware scheduler designed to optimize SLO attainment for DiT serving. We begin with a formal definition of the GPU scheduling problem in the offline scenario and prove that it is NP-hard. We then propose a _round-based scheduling mechanism_ , which maximizes goodput via minimizing GPU-hour consumption for each request. Later we proposes enhancements so that TetriServe balances utilization, latency, and scalability in DiT serving. 

## **4.1 Problem Statement** 

Given a collection of GPUs and requests, the DiT serving objective for each invocation of the scheduler is the following: _Find a step-level schedule that maximizes the number of requests meeting their deadlines given a fixed number of GPUs._ 

_**Problem Formulation.**_ Consider an _𝑁_ -GPU cluster and _𝑅_ outstanding requests. Each request _𝑟𝑒𝑞𝑖_ consists of a sequence of _𝑆𝑖_ dependent diffusion steps { _𝑠𝑖_ 1 _,𝑠𝑖_ 2 _, . . . ,𝑠𝑖𝑆𝑖_ }. Each step _𝑠𝑖𝑗_ can be executed using _𝑘_ ∈{1 _,_ 2 _,_ 4 _, . . . , 𝑁_ } GPUs, where _𝑘_ is a power of two. The execution time of a step, denoted _𝑇𝑖𝑗_ ( _𝑘_ ), is a function of _𝑘_ . The completion time of a request is defined as: 

**==> picture [103 x 31] intentionally omitted <==**

where _𝑄𝑖𝑗_ is the queuing delay before step _𝑠𝑖𝑗_ begins and _𝐴𝑖𝑗_ is the number of GPUs allocated. Then we can formulate the DiT serving objective as: 

**==> picture [195 x 31] intentionally omitted <==**

This formulation is subject to the following conditions: 

1. **Step Dependency** : A step _𝑠𝑖𝑗_ can start only after the previous step completes: 

**==> picture [182 x 12] intentionally omitted <==**

Therefore, at most one step of a request can be executed at any time. 

2. **GPU Capacity** : At any time, the total number of GPUs allocated across all steps cannot exceed _𝑁_ : 

**==> picture [97 x 31] intentionally omitted <==**

where _𝐴𝑖𝑗_ ( _𝑡_ ) denotes the GPUs allocated to step _𝑠𝑖𝑗_ if it is running at time _𝑡_ , and zero otherwise. 

The goal is to find a set of GPU assignments { _𝐴𝑖𝑗_ } that maximizes the number of requests meeting their deadlines. 

_**NP-hardness.**_ To highlight the computational complexity, we consider the special case where each request has a single non-preemptive step ( _𝑆𝑖_ = 1). Time is discretized into slots T = {0 _,_ 1 _, . . . ,𝑇_ max − 1}. Let K = {1 _,_ 2 _,_ 4 _, . . . , 𝑁_ } denote the 

1986 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

Runyu Lu et al. 

**Table 2.** Notations used in the GPU Scheduling Problem. 

|**Symbol**|**Description**|
|---|---|
|_𝑁_|Total number of GPUs.|
|_𝑅_|Number of requests.|
|_𝑆𝑖_|Number of steps in request_𝑟𝑒𝑞𝑖_.|
|_𝐷𝑖_|Deadline of request_𝑟𝑒𝑞𝑖_.|
|_𝑇𝑖𝑗_(_𝑘_)|Execution time of step_𝑠𝑖𝑗_with_𝑘_GPUs.|
|_𝑄𝑖𝑗_|Queueing delay before step_𝑠𝑖𝑗_starts.|
|_𝐴𝑖𝑗_|GPU allocation for step_𝑠𝑖𝑗_.|
|_𝐶𝑖_|Completion time of request_𝑟𝑒𝑞𝑖_.|



allowed GPU allocations. For each request _𝑖_ , start time _𝑡_ ∈T , and GPU count _𝑘_ ∈K, introduce a binary decision variable: 

**==> picture [214 x 30] intentionally omitted <==**

**Objective.** Maximize the number of requests completing by their deadlines: 

**==> picture [88 x 23] intentionally omitted <==**

## **Constraints.** 

**==> picture [162 x 23] intentionally omitted <==**

**==> picture [174 x 10] intentionally omitted <==**

**==> picture [146 x 10] intentionally omitted <==**

**==> picture [229 x 25] intentionally omitted <==**

**==> picture [129 x 11] intentionally omitted <==**

Constraint (1) ensures each request starts at most once. Constraints (2) and (3) enforce arrival times and deadline feasibility. Constraint (4) enforces that at any time slot _𝑢_ , the sum of GPUs assigned to running requests does not exceed system capacity _𝑁_ . Constraint (5) enforces integrality. 

This Zero-one Integer Linear Program (ZILP) exactly captures the offline DiT serving problem in the single-step case, where _𝐼𝑖_ =[�] _𝑡_ ∈T � _𝑘_ ∈K _[𝑥] 𝑖,𝑡,𝑘_[. We show in Appendix][ A][ that] solving such formulations is NP-hard [5, 14, 19, 32, 45]. Therefore, **multi-step DiT serving is NP-hard** as well. 

## **4.2 Round-Based Scheduling** 

Step-level scheduling for DiT serving is NP-hard, making global optimization expensive. To enable practical scheduling, TetriServe adopts a round-based heuristic: instead of scheduling steps arbitrarily in a continuous global timeline, _we discretize execution into rounds_ , where each round corresponds to a fixed-length GPU execution window. This allows us to _(i) limit the scheduling search space_ and _(ii) enable efficient preemption between rounds_ . Within each round, TetriServe determines the minimal required GPU allocation for 

requests and dynamically packs these requests to maximize SLO attainment ratio. 

**4.2.1 Deadline-Aware GPU Allocation.** Exhaustively enumerating GPU allocations for each step is infeasible, and over-allocation wastes resources due to scaling inefficiencies in DiT models (e.g., kernel launch and communication overheads). While more GPUs reduce latency, they increase total GPU hours. To balance these trade-offs, TetriServe identifies the minimal GPU allocation needed for each request to meet its deadline at the beginning of each round. Since required allocation depends mainly on resolution and deadline, this approach avoids exploring the full allocation space. 

For a step _𝑠𝑖𝑗_ , the execution time _𝑇𝑖𝑗_ ( _𝑘_ ) is a function of the number of GPUs _𝑘_ . The GPU hour for executing step _𝑠𝑖𝑗_ with _𝑘_ GPUs is _𝑘_ × _𝑇𝑖𝑗_ ( _𝑘_ ). The goal is to minimize the total GPU hour for each request: 

**==> picture [233 x 31] intentionally omitted <==**

where _𝐴𝑖𝑗_ is the GPU allocation for step _𝑠𝑖𝑗_ . 

_**Offline Profiling for Cost Model.**_ To make the optimization tractable, TetriServe profiles execution times offline. For every step type _𝑠𝑖𝑗_ and GPU count _𝑘_ ∈{1 _,_ 2 _,_ 4 _, . . . , 𝑁_ }, we measure the actual execution time _𝑇𝑖𝑗_ ( _𝑘_ ). From this, we derive the GPU hour _𝑘_ × _𝑇𝑖𝑗_ ( _𝑘_ ) and store it in a lookup table. At runtime, TetriServe simply enumerates candidate GPU assignments using these pre-profiled values. 

The above process aims to assign each request the minimum number of GPUs required to meet its deadline while minimizing the total GPU hours. Figure 6 illustrates this process with a concrete example: three requests (R1–R3), each with five steps, arrive over time. R1 has a small resolution (e.g., 256) and is fixed at SP=1 since higher parallelism would reduce efficiency (see Figure 3). For R2 and R3, TetriServe identifies GPU allocations with two parallelism degrees that just meet their deadlines while minimizing overall GPU usage. The GPU allocations produced by this selection serve as the input to the subsequent request packing stage, where TetriServe schedules requests across GPUs to maximize goodput. 

**4.2.2 Request Packing.** The objective of scheduling is to maximize the number of requests that complete before their deadlines. To make the problem tractable, we approximate it by minimizing the number of requests that become _definitely late_ —those that cannot meet their deadlines even under maximal parallelism if not advanced in the current round. Deadline-aware GPU allocation determines the minimal GPU allocations needed for each request to meet its deadline while minimizing GPU hours. This makes it possible to pack more requests into each round and thereby reduce the number that would otherwise be definitely late. 

1987 

TetriServe : Efficiently Serving Mixed DiT Workloads 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

**==> picture [505 x 385] intentionally omitted <==**

**----- Start of picture text -----**<br>
R1 Algorithm 1:  DP Round Scheduler<br>Request  R2<br>Tracker R3 Input : Pending requests  𝑅 with {( 𝑠𝑖 [𝑚][,𝐴][𝑚] 𝑖 [)}] [𝑚] [∈M] 𝑖<br>Time and  𝑇𝑖 (·); capacity  𝑁 ; round length  𝜏 ;<br>Scheduler current time  𝑡𝑟<br>Deadline-aware 1 2 3 4 5 Output: Selected plan<br>GPU Allocation 1 2 3 4 5 1 𝑡𝑟 +1 ← 𝑡𝑟 +  𝜏<br>1 2 3 4 5 2 foreach  𝑖 ∈ 𝑅 do<br>Round-based Round 1 Round 2 Round 3 Round 4 3 foreach  𝑚 ∈M 𝑖 do<br>Packing<br>GPU 0 1 2 3 4 5 4 𝑞 [𝑚] 𝑖 [←] [min][{] [𝑠] 𝑖 [𝑚][,] [ ⌊] [𝜏] [/] [𝑇][𝑖] [(] [𝐴][𝑚] 𝑖 [)⌋}]<br>GPU 1GPU 2GPU 3 1 2 3 14 2 35 4 5 5 𝑇𝑖 [min] ← min 𝑘 ∈ 𝐾 𝑇𝑖 ( 𝑘 )<br>GPU Placement Elastic Scale Up 6 O 𝑖 ←{none} ∪{ 𝑚 ∈M 𝑖 |  𝑞 [𝑚] 𝑖 [>] [ 0][ ∧] [𝐴][𝑚] 𝑖 [≤] [𝑁] [}]<br>GPU 0 1 2 3 4 5 5 7 foreach  𝑜 ∈O 𝑖 do<br>GPU 1GPU 2GPU 3 1 2 3 14 25 3 4 5 89 foreach 𝑠 ˜ [𝑚]  𝑚 [←] ∈M [𝑠][𝑚] 𝑖 do<br>Time 𝑖 [(] [𝑜] [)] 𝑖 [−] [I][[] [𝑜] [=] [ 𝑚] [] ·] [ 𝑞][𝑚] 𝑖<br>10 LB 𝑖 ( 𝑜 ) ← [��] 𝑚 ∈M 𝑖 [𝑠] [˜] 𝑖 [𝑚] [(] [𝑜] [)][�] [𝑇] 𝑖 [min]<br>Figure 6. Illustration of TetriServe’s scheduling process. 11 sv 𝑖 ( 𝑜 ) ← I[  𝑡𝑟 +1 + LB 𝑖 ( 𝑜 ) ≤ 𝐷𝑖 ]<br>The progression is shown from top to bottom: each row 12 𝑤𝑖 ( 𝑜 ) ← 0 if  𝑜 = none else  𝐴 [𝑜] 𝑖<br>represents an intermediate scheduling step, while the final<br>row shows the actual GPU allocation decision. Time is fixed 13 Initialize dp[0 ..𝑁 ] ←−∞, dp[0] ← 0<br>across rows. 14 foreach  𝑖 ∈ 𝑅 do<br>15 next[0 ..𝑁 ] ← dp<br>After deadline-aware GPU allocation, each request  𝑟𝑒𝑞𝑖 is 16 for  𝑐 = 0  to  𝑁 do<br>described by a set of allocations ( 𝑠𝑖 [𝑚][,𝐴][𝑚] 𝑖 [)][, where] [ 𝑠] 𝑖 [𝑚] [is the]<br>number of steps executed with allocation  𝐴 [𝑚] 𝑖 [, and per-step] 1718 foreachif  𝑤𝑖 𝑜 ( 𝑜 )∈O≤ 𝑖𝑐 dothen<br>times  𝑇𝑖 ( 𝐴 [𝑚] 𝑖 [)] [are obtained from the cost model. To schedule] 19 next[ 𝑐 ] ←<br>requests acrossof fixed duration  𝑁 𝜏 GPUs, TetriServe divides time into, which serves as the scheduling granular-  rounds max{next[ 𝑐 ] , dp[ 𝑐 − 𝑤𝑖 ( 𝑜 )] + sv 𝑖 ( 𝑜 )}<br>ity. The choice of  𝜏 balances overhead and responsiveness:<br>20 dp ← next<br>shorter rounds allow finer-grained preemption and more<br>adaptive scheduling, while longer rounds reduce overhead 21 𝑐 [★] ← arg max 𝑐 dp[ 𝑐 ]<br>but make scheduling coarser. 22 return  plan reconstructed from back-pointers at  𝑐 [★]<br>**----- End of picture text -----**<br>


**Figure 6.** Illustration of TetriServe’s scheduling process. The progression is shown from top to bottom: each row represents an intermediate scheduling step, while the final row shows the actual GPU allocation decision. Time is fixed across rows. 

At the beginning of each round _𝑟_ (time _𝑡𝑟_ ), the scheduler considers all pending requests and their allocations, and decides which to place within the _𝑁_ GPUs. Within a round of duration _𝜏_ , if GPU allocation _𝑚_ of request _𝑖_ is chosen, the number of steps that can complete is 

where _𝑠_ ˜ _[𝑚]_[is the updated step count. A request survives] _𝑖_[(] _[𝑜]_[)] only if 

**==> picture [78 x 10] intentionally omitted <==**

Each option _𝑜_ consumes _𝑤𝑖_ ( _𝑜_ ) GPUs: _𝑤𝑖_ (none) = 0, _𝑤𝑖_ ( _𝑚_ ) = _𝐴[𝑚] 𝑖_[. The per-round scheduling problem is therefore to select] at most one option per request, with total GPU ≤ _𝑁_ , maximizing the number of requests that survive to the next round. For requests that have already missed their deadlines, we assign at most one GPU in a best-effort manner without impacting other requests, and scale them up later if idle GPUs become available. By anchoring scheduling decisions on the round duration _𝜏_ , TetriServe balances preemption overhead and responsiveness, while ensuring urgent requests receive priority. 

**==> picture [105 x 19] intentionally omitted <==**

Options with _𝑞[𝑚] 𝑖_ = 0 are discarded to avoid wasting resources. Choosing option _𝑜_ ∈{none _,_ 1 _,_ 2 _, . . ._ } updates the remaining steps as 

**==> picture [107 x 11] intentionally omitted <==**

clipped at zero, where I[ _𝑜_ = _𝑚_ ] equals 1 if _𝑜_ = _𝑚_ and 0 otherwise. The next round begins at _𝑡𝑟_ +1 = _𝑡𝑟_ + _𝜏_ . 

To decide which requests must be scheduled _now_ , we identify those that would become _definitely late_ at _𝑡𝑟_ +1 if not advanced in this round. Using the fastest possible step time _𝑇𝑖_[min] = min _𝑘_ ∈{1 _,_ 2 _,_ 4 _,...,𝑁_ } _𝑇𝑖_ ( _𝑘_ ), we define the _residual completion time lower bound_ under option _𝑜_ as 

_**Dynamic Programming.**_ Naively enumerating all perrequest options O _𝑖_ for feasible packings within a round is exponential in the number of requests and quickly becomes intractable. We observe that the per-round decision has the _group-knapsack_ structure: for each request _𝑖_ (a group), we 

**==> picture [110 x 24] intentionally omitted <==**

1988 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

Runyu Lu et al. 

must choose at most one option _𝑜_ (run one of its GPU allocation this round or none), each option consumes width (GPUs) and yields a binary “survival” value indicating whether the request is _not definitely late_ at the next round start. This lets us replace exhaustive search with a dynamic program (DP) that maximizes the number of surviving requests under the round capacity _𝑁_ . 

Concretely, the DP state dp[ _𝑐_ ] stores, after processing the first _𝑖_ requests, the maximum number of surviving requests achievable with exactly capacity _𝑐_ ∈{0 _, . . . , 𝑁_ } consumed in the current round. For request _𝑖_ , we build its option set O _𝑖_ once (group constraint): none (consume zero GPUs, no progress) and one option per allocation _𝑚_ that can make progress in this round, i.e., _𝑞[𝑚] 𝑖_[=] � _𝜏_ / _𝑇𝑖_ ( _𝐴[𝑚] 𝑖_[)] � _>_ 0 and _𝐴[𝑚] 𝑖_[≤] _𝑁_ . For each option _𝑜_ ∈O _𝑖_ , we compute: 

1. **Line 9:** the updated remaining steps _𝑠_ ˜ _𝑖[𝑚]_[(] _[𝑜]_[)][.] 

2. **Line 10:** a conservative lower bound LB _𝑖_ ( _𝑜_ ) on the residual processing time from _𝑡𝑟_ +1 = _𝑡𝑟_ + _𝜏_ . 

3. **Line 12:** its width _𝑤𝑖_ ( _𝑜_ ) (0 for none, _𝐴[𝑚] 𝑖_[for allocation] _𝑚_ ). 

We then set the survival indicator sv _𝑖_ ( _𝑜_ ) = I[ _𝑡𝑟_ +1+LB _𝑖_ ( _𝑜_ ) ≤ _𝐷𝑖_ ]. The DP transition iterates options once per request (respecting the group constraint) and, for each capacity _𝑐_ , admits only options with _𝑤𝑖_ ( _𝑜_ ) ≤ _𝑐_ (respecting the capacity constraint): 

**==> picture [206 x 13] intentionally omitted <==**

Using a rolling array yields _𝑂_ ( _𝑁_ ) space. Since each request contributes at most |O _𝑖_ | options, DP runs in _𝑂_ ( _𝑅𝑁_ ) time and _𝑂_ ( _𝑁_ ) space per round (rolling array), which is tractable even at millisecond-scale rounds for moderate _𝑁_ . This is orders of magnitude cheaper than enumerating all feasible packing combinations. 

_**Round Duration.**_ Algorithm 1 schedules in fixed-length rounds of duration _𝜏_ . The choice of _𝜏_ balances two factors: short rounds reduce admission delay for new requests but increase scheduling frequency, while long rounds amortize scheduling cost but risk larger queueing delay and deadline misses. For a given GPU configuration (e.g., NVIDIA H100), TetriServe adapts _𝜏_ to the step execution times of requests across different resolutions, so that requests with heterogeneous step lengths can finish around the same round boundary. This minimizes idle bubbles while keeping _𝜏_ short enough to avoid excessive queueing delay. In practice, we determine _𝜏_ by the _step granularity_ , which means each round executes multiple diffusion steps. We will further discuss the impact of round duration in the evaluation section (§6.4). 

**4.2.3 Efficient GPU Placement and Allocation.** In the round-based framework (Algorithm 1), TetriServe improves efficiency via two complementary steps: _placement preservation_ and _work-conserving elastic scale-up_ , illustrated in Figure 6. First, to avoid idle bubbles between rounds, TetriServe 

adopts a placement-aware policy: requests continue on the same GPUs across consecutive rounds whenever possible. This eliminates state-transfer delays and ensures immediate progress at round boundaries. 

Second, any GPUs left idle after placement are reclaimed through a work-conserving elastic scale-up policy. Requests with sufficient remaining steps are granted additional GPUs if _𝑇𝑖_ ( _𝑘𝑖_[′][)] _[<][ 𝑇][𝑖]_[(] _[𝑘][𝑖]_[)][, prioritizing those that benefit most from] parallelism. This ensures no GPU remains unused within a round, reducing future load and improving deadline satisfaction. Together, placement preservation minimizes interround stalls, while elastic scale-up guarantees work-conserving allocation within each round. 

## **5 Implementation** 

TetriServe is implemented in 5,033 lines of Python and C++ code. We reuse components from existing solutions, including the sequence parallelism engine from xDiT [12], async logic from vLLM [20], and process launcher from MuxServe [11] and SGLang [47]. 

_**Scheduler.**_ The scheduler’s core decision loop is implemented in C++ and exposed via lightweight bindings, achieving millisecond-level control-plane latency. 

_**VAE Decoder Sequential Execution.**_ The VAE decoder imposes a large activation-memory footprint at high resolutions and batch sizes, whereas its wall-clock cost is very small relative to diffusion steps. Accordingly, we adopt sequential per-request decoding to bound peak memory by avoiding concurrent decoder activations across a batch. Because the decoder is largely off the critical path, this design does not increase end-to-end latency. The reduced peak usage also increases headroom for model state and communication buffers, lowering the risk of out-of-memory failures under mixed workloads. 

_**Communication Process Groups Warmup.**_ We precreate process groups for all relevant combinations of devices (e.g.,[�] _𝑘_[8] � groups for degrees _𝑘_ ∈{1 _, . . . ,_ 8}). Creating the group itself is lightweight and does not materially consume GPU memory. However, the _first_ invocation on a group initializes NCCL [31] channels and allocates persistent device buffers for subsequent collectives. Proactively warming _every_ group therefore inflates memory usage and can exceed available HBM. To balance startup latency and memory footprint, we warm only a compact set of commonly used, overlapping groups (e.g., [0,1,2,3], [0,2,3,4]) and defer others to on-demand warmup. Empirically, this strategy preserves performance while maintaining low peak memory. 

_**Latent Transfer.**_ Because TetriServe executes at step granularity, intermediate latents and lightweight metadata 

1989 

TetriServe : Efficiently Serving Mixed DiT Workloads 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

must be handed off across GPU groups. We provide a Futurelike abstraction for latents that enables asynchronous, nonblocking transfer between steps. Latent tensors are compact (in the compressed latent space), so transfer overhead is negligible; consequently, the scheduler excludes latent-transfer time from deadline accounting. We quantify this overhead in Section 6.4 and show it remains below 0.05% of per-step latency across all configurations. 

_**Selective Continuous Batching.**_ Batching in diffusion inference is only effective for identical, small-resolution requests that would otherwise underutilize GPUs. This creates a throughput-latency trade-off. Our scheduler employs a selective, step-level batching strategy that only groups requests if their SLOs are not compromised, thus improving resource utilization without harming latency. 

## **6 Evaluation** 

We evaluate TetriServe against state-of-the-art baselines across diverse workloads. Key findings: 

- TetriServe outperforms baselines by up to 32% across all resolutions (§6.2). 

- TetriServe is robust to bursty arrivals and adapts to changing resolution mixes (§6.3). 

- Sensitivity analysis confirms TetriServe’s advantage holds across varying arrival rates, step granularities, and homogeneous workloads (§6.4). 

- Ablation studies show that GPU placement preservation and elastic scale-up are crucial to TetriServe’s performance (§6.5). 

## **6.1 Methodology** 

_**Testbed.**_ We conduct experiments on two GPU clusters. The first comprises nodes with 8 NVIDIA H100-80GB HBM3 GPUs interconnected via NVLink 4.0 (900 GB/s inter-GPU bandwidth). The second features nodes with 4 NVIDIA A4048GB GPUs connected in pairs via NVLink and interfaced to the host via PCIe 4.0. Our software environment is based on NVIDIA’s NGC container with CUDA 12.5, NCCL 2.22.3 [31], PyTorch 2.4.0 [46], and xDiT [12] (git-hash 8f4b9d30). 

_**Models and Metrics.**_ We select _FLUX.1-dev_ [21] and _Stable Diffusion 3 Medium_ (SD3) [3] as representative models, evaluating them on H100 and A40 clusters, respectively. We report SLO Attainment Ratio (SAR; fraction of requests finishing within SLO) as our primary metric and plot end-to-end latency CDFs to show the latency distribution. 

_**Baselines.**_ We compare TetriServe against: 

- **xDiT (SP=1/2/4/8).** Fixed sequence parallelism degree; each request uses a constant number of GPUs. 

- **Resolution-Specific SP (RSSP).** Selects the best SP degree per resolution via offline profiling: SP=1 for 256 × 256 

and 512×512, SP=2 for 1024×1024, and SP=8 for 2048×2048. Represents an oracle static configuration. 

_**SLO Settings.**_ We adopt resolution-specific latency targets grounded in user-perceived responsiveness. Prior research [1] reports that 63% of users prefer a maximum response delay of 5 seconds in interactive settings. Accordingly, we cap the target at 1.5 seconds for small images and set an upper bound of 5.0 seconds for the largest resolution: (256 _,_ 256) = 1.5 s, (512 _,_ 512) = 2.0 s, (1024 _,_ 1024) = 3.0 s, and (2048 _,_ 2048) = 5.0 s. We sweep SLO Scale from 1 _._ 0× to 1 _._ 5× relative to each resolution’s baseline. 

_**Workload and Dataset.**_ We sample 300 prompts from DiffusionDB [42] to generate requests. By default, requests arrive as a Poisson process at 12 requests/minute. 

We consider two resolution mixes: 

- _Uniform_ : equal number of requests across resolutions {256, 512, 1024, 2048}. 

- _Skewed_ : resolutions sampled with exponential weight over latent length, _𝑝𝑖_ ∝ exp( _𝛼_ · _𝐿𝑖_ / _𝐿_ max), with _𝛼_ = 1 _._ 0 and _𝐿𝑖_ = ( _𝐻𝑖_ · _𝑊𝑖_ )/16[2] , biasing toward larger resolutions. 

## **6.2 End-to-End Performance** 

_**TetriServe Improves SAR..**_ Figures 7 and 8 show the endto-end SLO Attainment Ratio (SAR) of TetriServe compared to fixed-parallelism baselines for FLUX on H100s for both the Uniform and Skewed workload mixes at an arrival rate of 12 requests per minute. As shown in Figures 7a and 8a, TetriServe consistently achieves the highest SAR across all SLO scales and both workload distributions. This demonstrates the effectiveness of its step-level parallelism control and request packing, which allow it to dynamically adapt to the workload and outperform the rigid strategies of the baselines. 

On average, TetriServe outperforms the best fixed parallelism strategy by 10% for the Uniform mix and 15% for the Skewed mix. The performance gap is particularly pronounced at tighter SLOs. For instance, with an SLO scale of 1.1× in the Uniform mix, TetriServe outperforms the best baseline by 28%. Similarly, in the Skewed mix with a 1.2× SLO scale, TetriServe’s SAR is 32% higher than the bestperforming fixed strategy. 

Notably, this advantage holds even when compared against RSSP, a strong per-resolution baseline that selects the best fixed parallelism degree for each input resolution. Despite this, RSSP remains fundamentally limited by its lack of deadline awareness and runtime adaptation, whereas TetriServe dynamically adjusts parallelism at the step level to meet per-request SLOs. This highlights TetriServe’s superior performance under challenging, tightly constrained Workloads. 

_**TetriServe Benefits All Resolutions.**_ TetriServe’s strength lies in its ability to deliver high SAR across all request resolutions, unlike fixed strategies that only excel at specific 

1990 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

Runyu Lu et al. 

**==> picture [241 x 244] intentionally omitted <==**

**----- Start of picture text -----**<br>
TetriServe (ours) xDiT (SP=2) xDiT (SP=8)<br>xDiT (SP=1) xDiT (SP=4) RSSP<br>1.0<br>0.8<br>0.6<br>0.4<br>0.2<br>0.0<br>1.0x 1.1x 1.2x 1.3x 1.4x 1.5x<br>SLO Scale<br>(a)  SLO Attainment Ratio (SAR) of Uniform Workload<br>256x256 256x256<br>1.0 1.0<br>0.8 0.8<br>0.6 0.6<br>0.4 0.4<br>0.2 0.2<br>1024x1024 1024x1024<br>(b)  Uniform, SLO Scale=1.0× (c)  Uniform, SLO Scale=1.5×<br>SLO Attainment Ratio<br>512x512 512x512<br>2048x2048 2048x2048<br>SLO Attainment Ratio (SAR) SLO Attainment Ratio (SAR)<br>**----- End of picture text -----**<br>


**Figure 7.** End-to-end performance on the Uniform workload at 12 req/min. **(Top)** TetriServe achieves the highest SLO Attainment Ratio (SAR) across all SLO scales. **(Bottom)** The spider plots show that xDiT variants only perform well for specific resolutions, TetriServe delivers high SAR across all resolutions no matter tight or loose SLO Setting. 

ones. The spider plots in the bottom row of Figures 7 and 8 break down SAR by resolution. With a relaxed SLO of 1.5× (Figures 7c and 8c), TetriServe achieves near-perfect SAR across all resolutions for both workload mixes, consistently outperforming all xDiT baselines. Under the tightest SLO of 1.0× (Figures 7b and 8b), TetriServe provides the best overall performance. While some fixed-parallelism strategies may marginally outperform TetriServe on a single resolution (e.g., xDiT SP=1 on 256px), they perform poorly on others. In contrast, TetriServe dynamically adapts its parallelism, providing high SAR across the entire spectrum of resolutions. 

Conceptually, RSSP is a restricted variant of TetriServe in which the scheduler cannot adjust parallelism beyond a fixed configuration. Since RSSP explores only a subset of TetriServe’s decision space, it cannot exploit additional parallelism for deadline-critical requests, resulting in uniformly lower SAR across resolutions. In contrast, TetriServe avoids over parallelization for less urgent requests and prioritizes more GPU resources for more urgent requests, thus performing well on all resolutions. 

_**Tail Latency.**_ Figure 9 plots the CDF of end-to-end request latency under the tightest SLO setting (SLO scale = 1.0×) for both the Uniform and Skewed mixes. We compute the CDF over completed requests only, i.e., requests 

**==> picture [242 x 244] intentionally omitted <==**

**----- Start of picture text -----**<br>
TetriServe (ours) xDiT (SP=2) xDiT (SP=8)<br>xDiT (SP=1) xDiT (SP=4) RSSP<br>1.0<br>0.8<br>0.6<br>0.4<br>0.2<br>0.0<br>1.0x 1.1x 1.2x 1.3x 1.4x 1.5x<br>SLO Scale<br>(a)  SAR of Skewed Workload<br>256x256 256x256<br>1.0 1.0<br>0.8 0.8<br>0.6 0.6<br>0.4 0.4<br>0.2 0.2<br>1024x1024 1024x1024<br>(b)  Skewed, SLO Scale=1.0× (c)  Skewed, SLO Scale=1.5×<br>SLO Attainment Ratio<br>512x512 512x512<br>2048x2048 2048x2048<br>SLO Attainment Ratio (SAR) SLO Attainment Ratio (SAR)<br>**----- End of picture text -----**<br>


**Figure 8.** End-to-end performance on the Skewed workload at 12 req/min. **(Top)** TetriServe again achieves the highest SLO Attainment Ratio (SAR) across all SLO scales. **(Bottom)** The spider plots confirm that TetriServe’s adaptive parallelism provides robust performance across all resolutions, even in a workload dominated by large images 

that finish execution at least once (those that miss the deadline and are dropped/timeout are excluded from the latency distribution). Across both workload mixes, TetriServe produces a consistently more favorable tail distribution than fixed-parallelism baselines and RSSP. Compared to fixed SP baselines, TetriServe shifts the latency distribution left and reaches high completion probability at lower latency, indicating that most served requests finish quickly even under strict deadlines. Compared to RSSP, which restricts scheduling to a smaller decision space, TetriServe further reduces tail latency by dynamically reallocating GPUs toward more urgent requests and avoiding over-parallelization on less critical ones. Overall, these results show that TetriServe improves not only SAR but also keep the steady long tail latency under tight SLO scale. 

_**Compatibility with Cache-Based Diffusion Acceleration.**_ TetriServe is orthogonal and compatible with cachebased diffusion acceleration techniques. To demonstrate this, we integrate Nirvana [2] into our system. Nirvana accelerates diffusion inference by reusing intermediate denoising latents from prior requests. Each incoming prompt is embedded using CLIP [35] and matched against a cache of previously served prompts. Based on prompt similarity, the system determines how many initial diffusion steps can be skipped, yielding an effective diffusion length of _𝑁_ − _𝑘_ steps, where 

1991 

TetriServe : Efficiently Serving Mixed DiT Workloads 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

**Table 3. SAR with Nirvana Integration.** SLO Attainment Ratio (SAR) under uniform and skewed workload mixes (12 req/min, SLO Scale = 1 _._ 0×). TetriServe combined with Nirvana [2] achieves the highest SAR by jointly exploiting cache-based step reduction and adaptive GPU parallelism. 

**==> picture [241 x 206] intentionally omitted <==**

**----- Start of picture text -----**<br>
Workload RSSP TetriServe RSSP TetriServe<br>+ Nirvana + Nirvana<br>Uniform 0.32 0.42 0.77 0.88<br>Skewed 0.04 0.19 0.53 0.75<br>TetriServe xDiT (SP=2) xDiT (SP=8)<br>xDiT (SP=1) xDiT (SP=4) RSSP<br>1.0 1.0<br>0.8 0.8<br>0.6 0.6<br>0.4 0.4<br>0.2 0.2<br>0.0 0.0<br>0 5 10 15 0 5 10 15<br>Latency (s) Latency (s)<br>(a)  Uniform mix (b)  Skewed mix<br>CDF CDF<br>**----- End of picture text -----**<br>


**Figure 9. End-to-end latency CDF under strict SLOs (FLUX on H100, SLO scale = 1.0** × **).** TetriServe shows more consistent and better tail latency distribution than other baselines under strict SLO settings. The x-axis is truncated at 17s for readability; the SP=1 baseline has a much heavier tail beyond this range. 

_𝑘_ ∈{5 _,_ 10 _,_ 15 _,_ 20 _,_ 25} and _𝑁_ = 50 by default. We warm up the cache using the first 10K requests and then maintain a fixed-size cache with LRU eviction for online requests. 

Table 3 compares four configurations: RSSP, TetriServe, RSSP combined with Nirvana, and TetriServe combined with Nirvana, under both Uniform and Skewed mix workloads under the SLO Scale of 1.0×. While Nirvana alone substantially improves SLO attainment by reducing per-request computation, it does not address resource fragmentation caused by heterogeneous request resolutions. By contrast, TetriServe further improves SLO attainment by dynamically adjusting GPU parallelism to match the reduced and variable step counts introduced by caching. As a result, the combined system achieves the highest SLO attainment across both mixes, confirming that cache-based step reduction and TetriServe’s scheduling operate on complementary and orthogonal dimensions. 

## **6.3 Performance Stability under Bursty Traffic** 

TetriServe maintains a high and stable SAR even under bursty arrival patterns, whereas fixed-parallelism approaches exhibit significant performance oscillations. For instance, Figure 10 plots the SAR over time for the Uniform mix (12 

**==> picture [236 x 143] intentionally omitted <==**

**----- Start of picture text -----**<br>
TetriServe (ours) xDiT (SP=2) xDiT (SP=8)<br>xDiT (SP=1) xDiT (SP=4)<br>1.0<br>0.8<br>0.6<br>0.4<br>0.2<br>0.0<br>0 5 10 15 20 25<br>Time (minutes)<br>SLO Attainment Ratio<br>**----- End of picture text -----**<br>


**Figure 10.** Performance stability under the Uniform workload at 12 req/min with a 1.5x SLO Scale. TetriServe maintains a high and stable SLO Attainment Ratio (SAR) over time, which handles burstiness well. 

**==> picture [229 x 138] intentionally omitted <==**

**----- Start of picture text -----**<br>
TetriServe (ours)<br>8<br>4<br>2<br>1<br>0 5 10 15 20 25<br>Time (minutes)<br>Average Parallel Degree<br>**----- End of picture text -----**<br>


**Figure 11.** Average parallel degree of TetriServe during serving under the Uniform workload (1.5× SLO Scale). TetriServe dynamically adjusts sequence parallelism (SP) per request, assigning more GPUs to intensive requests (longer bars) to meet deadlines. 

req/min, SLO Scale=1.5×). TetriServe’s SAR remains consistently high with low variance. In contrast, the fixed xDiT variants suffer from periodic drops in SAR, a result of utilization bubbles and subsequent queueing delays when bursty arrivals create contention. 

The key to TetriServe’s stability is its ability to adapt the degree of sequence parallelism (SP) at the step level. As shown in Figure 11, when bursty arrivals create contention, TetriServe dynamically raises the SP degree for computationally intensive, urgent requests to shorten their critical path and reduce SLO violation risk. Conversely, it scales down the degree for less urgent requests steps while maintain SLO Attainment Ratio. This fine-grained, adaptive parallelism is how TetriServe handles burstiness and achieves superior efficiency and responsiveness compared to rigid, fixed-degree systems. 

1992 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

Runyu Lu et al. 

**==> picture [241 x 232] intentionally omitted <==**

**----- Start of picture text -----**<br>
TetriServe (ours) xDiT (SP=2)<br>xDiT (SP=1) xDiT (SP=4)<br>1.0<br>0.8<br>0.6<br>0.4<br>0.2<br>0.0<br>1.0x 1.1x 1.3x 1.5x<br>SLO Scale<br>(a)  SAR vs. SLO Scale (SD3, Uniform mix)<br>TetriServe (ours) xDiT (SP=2)<br>xDiT (SP=1) xDiT (SP=4)<br>1.0<br>0.8<br>0.6<br>0.4<br>0.2<br>0.0<br>1.0x 1.1x 1.3x 1.5x<br>SLO Scale<br>(b)  SAR vs. SLO Scale (SD3, Skewed mix)<br>SLO Attainment Ratio<br>SLO Attainment Ratio<br>**----- End of picture text -----**<br>


**Figure 12.** TetriServe’s performance on the Stable Diffusion 3 (SD3) model. The plots show the SLO Attainment Ratio (SAR) as a function of SLO Scale for the Uniform mix (left) and Skewed mix (right) on 4×A40 GPUs. In both workloads, TetriServe consistently outperforms all xDiT variants 

**==> picture [236 x 113] intentionally omitted <==**

**----- Start of picture text -----**<br>
TetriServe (ours) xDiT (SP=2) xDiT (SP=8)<br>xDiT (SP=1) xDiT (SP=4)<br>0.9<br>0.8<br>0.6<br>0.4<br>0.2<br>6 8 10 12 14 16 18<br>Arrival Rate (req/min)<br>Ratio<br>SLO Attainment<br>**----- End of picture text -----**<br>


**Figure 13.** SLO Attainment Ratio vs. arrival rate under the Uniform mix (SLO Scale=1.0x). TetriServe gracefully handles increasing load, maintaining a high SAR. 

## **6.4 Sensitivity Analysis** 

_**Different GPU Settings and Models.**_ On SD3, trends align with FLUX. In both the Uniform mix (Figure 12a) and Skewed mix (Figure 12b), TetriServe achieves the highest SAR across all SLO scales, with the largest margins at tight SLOs (1.0×). As SLOs loosen, fixed SP2 and SP4 improve but remain below TetriServe, while fixed SP1 underutilize and plateau. This indicates the benefits generalize to a different DiT architecture. On the A40 cluster, NVLink links GPUs only in pairs; at SP=4, collectives traverse PCIe, and even at SP=2 poor placement can cross PCIe. For SD3 this communication path becomes the bottleneck, so SP2 and SP4 perform notably worse than on H100. 

**==> picture [229 x 98] intentionally omitted <==**

**----- Start of picture text -----**<br>
TetriServe (ours) xDiT (SP=2) xDiT (SP=8)<br>xDiT (SP=1) xDiT (SP=4)<br>1.0<br>0.8<br>0.6<br>0.4<br>0.2<br>0.0<br>256x256 512x512 1024x1024 2048x2048<br>Shape Distribution<br>SLO Attainment Ratio<br>**----- End of picture text -----**<br>


**Figure 14.** SLO Attainment Ratio for homogeneous workloads at 12 req/min with a 1.5x SLO Scale. Each group of bars represents a workload with only one resolution type. TetriServe consistently achieves the highest SAR across all resolutions. 

**==> picture [229 x 102] intentionally omitted <==**

**----- Start of picture text -----**<br>
6 req/min 12 req/min 18 req/min<br>0.9<br>0.8<br>0.7<br>0.6<br>0.5<br>0.4<br>1 2 5 10 15<br>Step Granularity<br>Ratio<br>SLO Attainment<br>**----- End of picture text -----**<br>


**Figure 15.** Sensitivity of SLO Attainment Ratio to step granularity and arrival rate under the Uniform mix (SLO Scale=1.0x). A moderate granularity (5/10 steps) provides the most robust performance as system load increases, balancing scheduling flexibility and overhead. 

_**Arrival Rate.**_ Figure 13 shows the SAR of different scheduling strategies under the Uniform mix with a tight SLO of 1.0× as the arrival rate increases from 6 to 18 req/min. TetriServe demonstrates superior performance across the full range of arrival rates. At low-to-medium rates, TetriServe maintains a consistently high SAR, while fixed-parallelism strategies already show signs of degradation. At high arrival rates, where the system is under heavy load, TetriServe’s SAR remains relatively high, showcasing graceful degradation. 

_**Homogeneous Resolutions.**_ To isolate the effect of input resolution on parallelism strategies, we evaluate homogeneous workloads containing only a single resolution. Figure 14 shows the SLO Attainment Ratio (SAR) for workloads consisting of only one resolution type at an arrival rate of 12 req/min and an SLO Scale of 1.5x. Even in these simplified scenarios, TetriServe still achieves the highest SAR across all resolution types. This demonstrates that TetriServe’s adaptive scheduling is effective not only for mixed workloads but also for homogeneous ones, as it can still optimize resource allocation to better meet deadlines. 

_**Step Granularity.**_ We examine the impact of step granularity, which defines how frequently TetriServe can reschedule and change the degree of parallelism for an in-flight 

1993 

TetriServe : Efficiently Serving Mixed DiT Workloads 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

**Table 4.** Latent transfer overhead as a percentage of inference step latency. Across all configurations, the overhead is negligible ( _<_ 0 _._ 05%). 

|**Batch Size**|**256**×**256**<br>**512**×**512**<br>**1024**×**1024**<br>**2048**×**2048**|
|---|---|
|||
|BS = 1<br>BS = 2<br>BS = 4|0.03%<br>0.03%<br>0.04%<br>0.01%<br>0.04%<br>0.03%<br>0.05%<br>0.02%<br>0.04%<br>0.05%<br>0.03%<br>0.01%|



request. This presents a fundamental trade-off: fine-grained control (e.g., every 1-2 steps) offers maximum flexibility at the cost of high scheduling overhead, while coarse-grained control (e.g., every 10 steps) minimizes overhead but creates longer, non-preemptible execution blocks that reduce adaptability. Figure 15 illustrates this trade-off under the Uniform mix (SLO Scale=1.0x) across different arrival rates. At low rates, performance is less sensitive to granularity. However, as load increases, a moderate granularity of 5 steps proves most robust, balancing adaptability and overhead. Very finegrained control (1 step) suffers from excessive overhead, while coarse-grained control (10 steps) is too inflexible to handle preemption, leading to lower SLO attainment. 

_**Parallel Reconfiguration Overhead.**_ TetriServe performs step-level scheduling, which requires transferring intermediate latent representations and metadata across GPU groups when parallelism changes between steps. Table 4 quantifies this parallel reconfiguration overhead as a percentage of per step inference latency across varying resolutions and batch sizes. We observe that the overhead is consistently negligible, accounting for at most 0.05% of step latency in all configurations. As a result, TetriServe’s scheduler can safely ignore latent transfer time in deadline accounting without affecting SLO accuracy. 

## **6.5 Ablation Study** 

TetriServe includes two practical mechanisms on top of the round-based DP scheduler: (i) _GPU Placement Preservation_ , which keeps a request on the same GPU set across rounds whenever possible to avoid remapping stalls; and (ii) _Elastic Scale-up_ , which makes use of idle GPUs after placement and temporarily grants extra GPUs to requests that benefit from higher parallelism. To quantify their impact, we ablate these components under two SLO scales (1.0× and 1.5×) on two workload mixes: Uniform and Skewed. Table 5 reports the SLO Attainment Ratio and mean latency. 

Overall, both mechanisms are important for improving serving efficiency. GPU Placement Preservation improves SAR and/or mean latency in most settings by avoiding remapping overhead and enabling immediate progress at round boundaries, while Elastic Scale-up consistently increases SAR (up to +0.11 absolute on Skewed mix at 1.5×) and typically further reduces mean latency by utilizing idle GPUs. Consequently, enabling both GPU placement preservation 

**Table 5. Ablation of scheduling mechanisms.** GPU Placement Preservation reduces inter-round stalls by keeping requests on the same GPU set; Elastic Scale-up opportunistically reallocates idle GPUs to requests that benefit from extra parallelism. 

||**(a) Uniform Mix**.||
|---|---|---|
|Variant|SLO = 1.0×|SLO = 1.5×|
||SAR↑/ Mean Lat.↓|SAR↑/ Mean Lat.↓|
|TetriServe schedule|0.54 / 4.45|0.74 /**4.81**|
|+ Placement|0.56 / 3.96|0.69 / 5.14|
|+ Elastic Scale-Up|**0.63**/**3.89**|**0.78**/ 4.83|
||**(b) Skewed Mix**.||
|Variant|SLO = 1.0×|SLO = 1.5×|
||SAR↑/ Mean Lat.↓|SAR↑/ Mean Lat.↓|
|TetriServe schedule|0.27 / 8.43|0.38 / 9.92|
|+ Placement|0.31 /**7.64**|0.45 / 8.16|
|+ Elastic Scale-Up|**0.36**/ 7.68|**0.55**/**7.71**|



and Elastic Scale-up achieves the best SLO Attainment Ratio across all tested scenarios, while also improving latency compared to disabling these optimizations. 

## **7 Related Work** 

_**LLM Serving Frameworks.**_ LLM serving systems [20, 47] are not directly applicable to DiT workloads. LoongServe [43] optimizes prefill-decode stages for long-context LLMs, while PrefillOnly [10] targets memory efficiency for short, prefillintensive requests. Neither suits the multi-step, stateless inference pattern of DiTs. 

_**DiT Inference and Serving.**_ DiT-specific serving systems are still emerging. xDiT [12] uses fixed sequence parallelism, which is inefficient for heterogeneous workloads. DDiT [17] targets video generation and maximizes throughput rather than meeting SLOs. TetriServe uniquely prioritizes SLO attainment for heterogeneous requeststhrough cost-modeldriven scheduling. 

_**Text-to-Image Caching.**_ Several systems accelerate textto-image diffusion via caching. AsyncDiff [8] parallelizes diffusion through asynchronous denoising cross requests. Caching-based approaches exploit reuse across prompts or adapters, including approximate latent caching in Nirvana [2], layer-level caching [26], final image caching [44], workflowaware reuse [24], and patch-level reuse [40]. These techniques reduce redundant computation; TetriServe addresses an orthogonal dimension by scheduling GPU parallelism across concurrent requests and could integrate these methods for further gains. 

_**Resource Scheduling.**_ In VM allocation frameworks [6], machine count is fixed at admission. GPU schedulers like 

1994 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

Runyu Lu et al. 

Gavel [30], Tiresias [15], and AlloX [22] focus on job placement and fairness but require users to specify parallelism. In contrast, TetriServe treats parallelism as a scheduling decision, dynamically adjusting GPU degree at step granularity based on deadlines and scaling efficiency. 

## **8 Conclusion** 

We presented TetriServe, a deadline-aware round-based DiT serving system that addresses the challenge of meeting SLOs under heterogeneous workloads. TetriServe dynamically adapts parallelism at the _step level_ , guided by a profilingdriven cost model and a deadline-aware scheduling algorithm. Extensive evaluation shows that TetriServe consistently outperforms fixed-parallelism baselines, achieving up to 32% higher SLO attainment and robust performance across varying resolutions, workload distributions, and arrival rates. 

## **Acknowledgements** 

We thank the ASPLOS reviewers, as well as members of SymbioticLab and UseSysLab, for their helpful feedback. This work was supported in part by NSF grants CCF-2450085, CNS-2106184, CNS-2214272 and CNS-2106751, and by grants from Ford and Cisco. 

## **References** 

- [1] Tahir Abbas, Ujwal Gadiraju, Vassilis-Javed Khan, and Panos Markopoulos. 2022. Understanding User Perceptions of Response Delays in Crowd-Powered Conversational Systems. _Proceedings of the ACM on Human-Computer Interaction_ (2022). 

- [2] Shubham Agarwal, Subrata Mitra, Sarthak Chakraborty, Srikrishna Karanam, Koyel Mukherjee, and Shiv Kumar Saini. 2024. Approximate Caching for Efficiently Serving Text-to-Image Diffusion Models. In _NSDI_ . 

- [3] Stability AI. 2024. Stable Diffusion 3 Medium. https://huggingface.co/ stabilityai/stable-diffusion-3-medium. 

- [4] Stability AI. 2024. Stable Diffusion 3.5 Large. https://huggingface.co/ stabilityai/stable-diffusion-3.5-large. 

- [5] Amotz Bar-Noy, Sudipto Guha, Joseph Naor, and Baruch Schieber. 1999. Approximating the Throughput of Multiple Machines under Real-Time Scheduling. In _STOC_ . 

- [6] Hugo Barbalho, Patricia Kovaleski, Beibin Li, Luke Marshall, Marco Molinaro, Abhisek Pan, Eli Cortez, Matheus Leao, Harsh Patwari, Zuzu Tang, et al. 2023. Virtual Machine Allocation with Lifetime Predictions. In _MLSys_ . 

- [7] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. 2024. Video Generation Models as World Simulators. _OpenAI Blog_ (2024). 

- [8] Zigeng Chen, Xinyin Ma, Gongfan Fang, Zhenxiong Tan, and Xinchao Wang. 2024. AsyncDiff: Parallelizing Diffusion Models by Asynchronous Denoising. _NeurIPS_ . 

- [9] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 2021. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In _ICLR_ . 

- [10] Kuntai Du, Bowen Wang, Chen Zhang, Yiming Cheng, Qing Lan, Hejian Sang, Yihua Cheng, Jiayi Yao, Xiaoxuan Liu, Yifan Qiao, Ion Stoica, and Junchen Jiang. 2025. PrefillOnly: An Inference Engine 

   - for Prefill-only Workloads in Large Language Model Applications. In _SOSP_ . 

- [11] Jiangfei Duan, Runyu Lu, Haojie Duanmu, Xiuhong Li, Xingcheng Zhang, Dahua Lin, Ion Stoica, and Hao Zhang. 2024. MuxServe: Flexible Spatial-Temporal Multiplexing for Multiple LLM Serving. In _ICML_ . 

- [12] Jiarui Fang, Jinzhe Pan, Xibo Sun, Aoyu Li, and Jiannan Wang. 2024. xDiT: an Inference Engine for Diffusion Transformers (DiTs) with Massive Parallelism. _arXiv preprint arXiv:2411.01738_ (2024). 

- [13] Flux.1 AI. 2025. _Flux.1 AI Image Generator_ . https://flux1.ai/create 

- [14] Michael R Garey and David S. Johnson. 1977. Two-Processor Scheduling with Start-Times and Deadlines. _SIAM journal on Computing_ (1977). 

- [15] Juncheng Gu, Mosharaf Chowdhury, Kang G. Shin, Yibo Zhu, Myeongjae Jeon, Junjie Qian, Hongqiang Harry Liu, and Chuanxiong Guo. 2019. Tiresias: A GPU Cluster Manager for Distributed Deep Learning. In _NSDI_ . 

- [16] Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. In _NeurIPS_ . 

- [17] Heyang Huang, Cunchen Hu, Jiaqi Zhu, Ziyuan Gao, Liangliang Xu, Yizhou Shan, Yungang Bao, Sun Ninghui, Tianwei Zhang, and Sa Wang. 2025. DDiT: Dynamic Resource Allocation for Diffusion Transformer Model Serving. _arXiv preprint arXiv:2506.13497_ (2025). 

- [18] Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He. 2023. DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models. _arXiv preprint arXiv:2309.14509_ (2023). 

- [19] Alind Khare, Dhruv Garg, Sukrit Kalra, Snigdha Grandhi, Ion Stoica, and Alexey Tumanov. 2025. SuperServe: Fine-Grained Inference Serving for Unpredictable Workloads. In _NSDI_ . 

- [20] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient Memory Management for Large Language Model Serving with PagedAttention. In _SOSP_ . 

- [21] Black Forest Labs. 2024. _FLUX.1-dev: Text-to-Image Generation Model_ . 

- [22] Tan N. Le, Xiao Sun, Mosharaf Chowdhury, and Zhenhua Liu. 2020. AlloX: Compute Allocation in Hybrid Clusters. In _EuroSys_ . 

- [23] Shenggui Li, Fuzhao Xue, Chaitanya Baranwal, Yongbin Li, and Yang You. 2023. Sequence Parallelism: Long Sequence Training from System Perspective. In _ACL_ . 

- [24] Suyi Li, Lingyun Yang, Xiaoxiao Jiang, Hanfeng Lu, Dakai An, Zhipeng Di, Weiyi Lu, Jiawei Chen, Kan Liu, Yinghao Yu, Tao Lan, Guodong Yang, Lin Qu, Liping Zhang, and Wei Wang. 2025. Katz: Efficient Workflow Serving for Diffusion Models with Many Adapters. In _ATC_ . 

- [25] Hao Liu, Matei Zaharia, and Pieter Abbeel. 2023. Ring Attention with Blockwise Transformers for Near-Infinite Context. _arXiv preprint arXiv:2310.01889_ (2023). 

- [26] Xinyin Ma, Gongfan Fang, Michael Bi Mi, and Xinchao Wang. 2024. Learning-to-Cache: Accelerating Diffusion Transformer via Layer Caching. _NeurIPS_ . 

- [27] Yixuan Mei, Yonghao Zhuang, Xupeng Miao, Juncheng Yang, Zhihao Jia, and Rashmi Vinayak. 2025. Helix: Serving Large Language Models over Heterogeneous GPUs and Network via Max-Flow. In _ASPLOS_ . 

- [28] Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, Chunan Shi, Zhuoming Chen, Daiyaan Arfeen, Reyna Abhyankar, and Zhihao Jia. 2024. SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification. In _ASPLOS_ . 

- [29] Xupeng Miao, Chunan Shi, Jiangfei Duan, Xiaoli Xi, Dahua Lin, Bin Cui, and Zhihao Jia. 2024. SpotServe: Serving Generative Large Language Models on Preemptible Instances. In _ASPLOS_ . 

- [30] Deepak Narayanan, Keshav Santhanam, Fiodar Kazhamiaka, Amar Phanishayee, and Matei Zaharia. 2020. Heterogeneity-Aware Cluster 

1995 

TetriServe : Efficiently Serving Mixed DiT Workloads 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

Scheduling Policies for Deep Learning Workloads. In _OSDI_ . 

- [31] NVIDIA. 2022. NVIDIA Collective Communication Library (NCCL) Documentation. https://docs.nvidia.com/deeplearning/nccl/userguide/docs/index.html. 

- [32] Christos H Papadimitriou and Kenneth Steiglitz. 1998. _Combinatorial Optimization: Algorithms and Complexity_ . Courier Corporation. 

- [33] Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, and Ricardo Bianchini. 2024. Splitwise: Efficient Generative LLM Inference Using Phase Splitting. In _ISCA_ . 

- [34] William Peebles and Saining Xie. 2023. Scalable Diffusion Models with Transformers. In _ICCV_ . 

- [35] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning Transferable Visual Models From Natural Language Supervision. In _ICML_ . 

- [36] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. 2015. U-Net: Convolutional networks for biomedical image segmentation. In _MICCAI_ . 

- [37] Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. 2015. Deep unsupervised learning using nonequilibrium thermodynamics. _arXiv preprint arXiv:1503.03585_ (2015). 

- [38] Yang Song and Stefano Ermon. 2021. Score-Based Generative Modeling through Stochastic Differential Equations. In _ICLR_ . 

   - [41] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is All You Need. In _NeurIPS_ . 

   - [42] Zijie J Wang, Evan Montoya, David Munechika, Haoyang Yang, Benjamin Hoover, and Duen Horng Chau. 2023. DiffusionDB: A LargeScale Prompt Gallery Dataset for Text-to-Image Generative Models. In _ACL_ . 

   - [43] Bingyang Wu, Shengyu Liu, Yinmin Zhong, Peng Sun, Xuanzhe Liu, and Xin Jin. 2024. LoongServe: Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism. In _SOSP_ . 

   - [44] Yuchen Xia, Divyam Sharma, Yichao Yuan, Souvik Kundu, and Nishil Talati. 2026. MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models. In _ASPLOS_ . 

   - [45] Hong Zhang, Yupeng Tang, Anurag Khandelwal, and Ion Stoica. 2023. SHEPHERD: Serving DNNs in the Wild. In _NSDI_ . 

   - [46] Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al. 2023. PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel. In _VLDB_ . 

   - [47] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Livia Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. 2024. SGLang: Efficient Execution of Structured Language Model Programs. In _NeurIPS_ . 

- [39] Stability AI. 2024. Stability AI Platform API Reference. https:// platform.stability.ai/docs/api-reference Accessed: 2024-11-26. 

- [40] Desen Sun, Zepeng Zhao, and Yuke Wang. 2026. MixFusion: A PatchLevel Parallel Serving System for Mixed-Resolution Diffusion Models. In _PPoPP_ . 

1996 

ASPLOS ’26, March 22–26, 2026, Pittsburgh, PA, USA 

Runyu Lu et al. 

## **A NP-Hardness of DiT Serving** 

We prove NP-hardness for the DiT serving problem defined in TetriServe, which maximizes the number of requests that complete by deadlines under GPU capacity constraints. 

Let us first define the decision problem DiT-ServingDecision: given an instance, and an integer target _𝐵_ , decide whether there exists a schedule in which at least _𝐵_ requests meet their deadlines. This is the natural decision version of TetriServe’s objective max[�] _𝑖[𝐼] 𝑖_[.] 

Bar-Noy et al. [5, 14] state that the following real-time (RT) scheduling feasibility decision problem (RT-Feasibility) is NP-hard in the strong sense: on a _single_ machine, given jobs with release times _𝑟𝑖_ , deadlines _𝑑𝑖_ , and processing times _𝑙𝑖_ , decide whether _all_ jobs can be scheduled within their time windows. Since RT-Feasibility is strongly NP-hard, it remains NP-hard even when all numeric parameters are bounded by a polynomial in the input size. Therefore, _𝑇_ max = max _𝑖 𝑑𝑖_ is polynomially bounded, and our time-indexed reduction is polynomial-time. 

_**Reduction to DiT serving with**_ K = {1} _**.**_ Given a RTFeasibility instance [5] with jobs _𝑖_ = 1 _, . . . ,𝑛_ and parameters ( _𝑟𝑖,𝑑𝑖,𝑙𝑖_ ), let us construct a single-step DiT instance as follows: _𝑁_ := 1 _, 𝑅_ := _𝑛,𝑆𝑖_ := 1 _, 𝐾_ := {1} _,_ arrival_time( _𝑖_ ) := _𝑟𝑖, 𝐷𝑖_ := _𝑑𝑖,𝑇𝑖_ (1) := _𝑙𝑖 ._ Set the throughput target _𝐵_ := _𝑛_ . 

Equivalently, in TetriServe’s single-step time-indexed formulation with variables _𝑥𝑖,𝑡,𝑘_ and constraints (1)–(5), we restrict to _𝑘_ = 1 and _𝑁_ = 1, and disallow infeasible start times by setting _𝑥𝑖,𝑡,_ 1 = 0 whenever _𝑡 < 𝑟𝑖_ or _𝑡_ + _𝑙𝑖 > 𝑑𝑖_ . 

_**Correctness.**_ (⇒) If the RT-Feasibility instance is feasible, let _𝑠𝑖_ be the start time of job _𝑖_ in a feasible single-machine schedule. Schedule each corresponding DiT request _𝑖_ to start at time _𝑠𝑖_ using one GPU. All requests meet deadlines, so � _𝑖[𝐼] 𝑖_[=] _[ 𝑛]_[≥] _[𝐵]_[.] (⇐) If the constructed DiT instance has a schedule with � _𝑖[𝐼] 𝑖_[≥] _[𝑛]_[, then all] _[ 𝑛]_[requests meet deadlines. Since] _[ 𝑁]_[=][ 1 and] each request uses one GPU, the capacity constraint implies no two requests overlap. Thus the chosen start times form a feasible non-preemptive single-machine schedule for all jobs in the original RT-Feasibility instance. 

|**# Reqs**<br>**Time (s)**<br>1<br>_<0.01_<br>2<br>_0.27_<br>3<br>_52.56_<br>4<br>_>60.00_<br>**(a)**4 GPUs|**# Reqs**<br>**Time (s)**|
|---|---|
||1<br>_0.02_<br>2<br>_11.12_<br>3<br>_>60.00_<br>4<br>_>60.00_|
||**(b)**8 GPUs|



**Table 6. Scheduling overhead of exhaustive search.** Control plane scheduling time under different GPU budgets and queue sizes. TetriServe remains lightweight: it takes _**<0.01 s**_ compared to exhaustive search following the same settings, enabling online scheduling in practice. 

**Experimental Setup.** We implement an exact baseline solver that enumerates the complete decision space to maximize SLO attainment. The solver explores two dimensions of complexity for each request: (1) all feasible sequenceparallel degrees per diffusion step (e.g., _𝑘_ ∈{1 _,_ 2 _,_ 4 _,_ 8}), and (2) all valid permutations of physical GPU mapping for those degrees. The objective is to identify the schedule with the highest SLO attainment, using minimum total GPU hours as a tie-breaker. We measure the wall clock latency required to generate a single scheduling plan using an AMD EPYC 7513 32-Core CPU, varying the queue depth ( _𝑅_ ) under fixed GPU budgets of _𝑁_ ∈{4 _,_ 8}. 

**Results.** Table 6 presents the scheduling overhead. The baseline exhibits immediate combinatorial explosion: with a budget of 8 GPUs, optimally scheduling merely three requests exceeds a 60-second timeout. This intractability stems from the factorial growth of permutation possibilities as the number of available GPUs increases. In contrast, TetriServe maintains a decision latency of _<10 ms_ . These results confirm that exhaustive optimization is prohibitive for online serving, necessitating the efficient round-based planning strategy employed by TetriServe. 

Therefore, we can convert any RT-Feasibility instance into a DiT-Serving-Decision instance in polynomial time such that a feasible schedule exists in the former iff one exists in the latter. DiT-Serving-Decision is NP-hard even for the restricted case _𝑆𝑖_ = 1 and K = {1}; consequently, the general multi-step DiT serving problem is NP-hard. 

## **B Scheduling Overhead Analysis** 

To validate the necessity of TetriServe’s heuristic approach, we quantify the computational cost of finding a globally optimal schedule via exhaustive search. As established in Appendix A, the underlying step-level scheduling problem is NP-hard. 

1997 

