# *C. Challenges of serving MoE with future GPUs*

Unlike current multi-GPU systems, wafer-scale GPUs can fit entire MoE models on a single chip and support batch sizes over 10,000. However, current GPU architectures introduce two key limitations for such large-scale chips.

Simplistic Task Allocation. Current GPUs integrate a CPU in their SoC to serve as a command processor and allocate tasks to all SMs. However, the traditional command processors treat all SMs equally, ignoring their physical locations and data placement [\[49\]](#page-14-36), [\[50\]](#page-14-37). This oblivious task-to-SM assignment generates excessive D2D traffic and ignores MoE expert selection skewness, leading to poor utilization when most dies remain idle while others become overloaded.

Inadequate Local HBM Management. Current GPUs treat all HBM dies as uniform memory space, but wafer-scale GPUs connect each compute die directly to local HBM, where access is significantly faster than a remote HBM. Frequently accessed experts in remote HBM could be cached locally to minimize D2D traffic, but current GPUs do not distinguish between local and remote HBM and therefore generate unnecessary traffic.

![](_page_7_Figure_0.jpeg)

<span id="page-7-0"></span>Figure 10. (a) Wafer-scale multi-chiplet GPU architecture with additional units highlighted in orange. (b) SoW (System-on-Wafer) technology structure. (c) Data format in the Global Command Processor for our proposed task distribution strategy.

#### D. Motivation and Insights

To address these challenges, we propose two strategies with architectural support. First, based on <a href="Insight 3">Insight 3</a> that identifies the need for expert-placement-aware task distribution, we propose an intelligent task allocation algorithm with a multi-level, data-placement-aware command processor architecture. This approach considers expert placement and selection skewness across dies, enabling dynamic task allocation that minimizes D2D traffic while balancing workload.

Second, leveraging Insight 1 and Insight 2 that reveal the predictability behind expert selection across different timescales, we introduce a data-driven predictor with hardware-managed HBM architecture. Local HBM caches frequently accessed experts from remote dies, while a lightweight predictor analyzes selection patterns to estimate future needs, caching predicted experts locally to reduce D2D traffic.

To implement these two strategies under a single-GPU-like programming model, we a few architectural extensions to the GPU architecture. If future programming models evolve toward multi-GPU-like abstractions with finer-grained control over each die, these strategies could alternatively be realized at the system level without any architectural modification.

