# VII. EVALUATION

## *A. Experiment Setup*

Model: To evaluate Tetris's performance at different scales, we use LLaMA3-8B and LLaMA3-70B [15] models. We employ their context-extended variants with RoPE scaling [38] to support the context window in our workloads.

Testbed: We conduct experiments on A100 GPU clusters. Each node contains eight NVIDIA-A100-SXM4-80GB GPUs connected with NVLINK, 128 CPU cores, 2TB host memory, and eight 200 Gbps InfiniBand NICs. We deploy LLaMA3-8B on four nodes and LLaMA3-70B on eight nodes.

Workload: We collect three real-world request traces with different length distributions from an online long-context LLM service provided by Bytedance Doubao Service [6]. Specifically, the Short trace's sequence length ranges from 4k to 95k, with an average length of 23.6k. The Medium trace's sequence

length ranges from 8k to 142k, with an average length of 32.8k. The Long trace's sequence length ranges from 16k to 190k, with an average length of 50.1k.

Metric: As discussed in Sec. II-B, we adopt TTFT and TBT, the key metrics for online LLM serving, to measure each system's performance. We report both P50 and P99 values to characterize the overall latency distribution.

Baseline: We compare Tetris with the following baselines:

- (1) LoongServe [43]: It is the first and the only SPenabled long-context LLM serving framework. Moreover, it reports state-of-the-art long-context LLM serving performance compared with existing best-performing non-SP serving systems [1], [19], [23], [47]. We set TP=1 for LLaMA3-8B and TP=4 for LLaMA3-70B to maximize its flexibility (i.e., ESP size) while ensuring sufficient cache slots on each instance. To avoid TTFT interference as discussed in Sec. II-D (*Limitation (2)*), we adopt single-request scheduling to minimize its TTFT. (2) LoongServe Disaggregated: This is a prefill-decoding decoupled cluster similar to Tetris's architecture, while the prefill scheduler adopts LoongServe's single-request scheduling. We set the P/D ratio to 1:1 after carefully balancing TTFT and TBT. For LLaMA3-8B, the TP sizes of prefill and decoding instances are 1 (identical to LoongServe) and 8. For LLaMA3- 70B, since decoding latency reports marginal improvement beyond TP=4, we set TP size to 4 (identical to LoongServe) for all instances and focus on TTFT evaluation.
- (3) Fixed-SP Scheduling: It also adopts the prefill-decoding disaggregation architecture, where prefill instances are organized into multiple independent SP groups. We evaluate fixed SP sizes of 8 and 16, co-locating each group's instances on the same node where possible. Requests are scheduled to the group with the lowest queuing delay, which is estimated using Eq. (1). The P/D ratio and TP size allocation are identical to LoongServe Disaggregated.

For Tetris, we also adopt the same P/D ratio and TP size allocation as LoongServe Disaggregated for fair comparison. The SP size candidates are set to powers of two to reduce resource fragmentation. We adopt the simulator to collect optimal improvement rates (ranging from 0.05 to 0.75) for request rates incremented by 0.5 req/s. During serving, the improvement rate is updated every 30 seconds. The scheduler selects the recorded request rate closest to the observed value and applies the corresponding optimal improvement rate.

