# F. Scheduler Analysis

**Simulator Accuracy:** To evaluate the performance model's accuracy, we collect prefill latency measurements across all combinations of historical token numbers (0-256k, in 8k steps) and current token numbers (8k-256k, in 8k steps). After

![](_page_12_Figure_0.jpeg)

Fig. 17. Cache Transfer Overhead Analysis.

TABLE II
SCHEDULER OVERHEAD UNDER DIFFERENT SP SIZES.

| Max SP Size           | 8         | 16        | 32        | 64        | 128       |
|-----------------------|-----------|-----------|-----------|-----------|-----------|
| Avg./Max Latency (us) | 22.8/52.5 | 25.8/86.8 | 22.9/53.4 | 24.9/45.1 | 30.6/73.7 |

skipping out-of-memory points, a subset is sampled at 16k intervals for model fitting, and accuracy is assessed on the full dataset. For all SP size candidates, the model yields up to 7.64%/6.35% error on LLaMA3-8B/70B, respectively.

We also assess the fidelity of the performance-model-based simulator by simulating all test cases of Tetris in Fig. 11. The simulator yields 0.9%–13.3%/0.4%–14.3% error on LLaMA3-8B/70B, with average errors of 6.9%/2.5%, respectively. The performance model is reliable enough to guide CDSP scheduling and improvement rate selection.

CDSP Scheduling Overhead: To evaluate the efficiency of CDSP prefill scheduling, we measure its execution latency under different SP sizes by randomly sampling request length and instance queuing latency. Each SP size is tested 1000 times. As listed in Table II, even when SP=128, the scheduling latency remains ≤86.8us, proving Algorithm 1's efficiency in meeting the real-time requirements of online serving.

To quantify the end-to-end scheduling overhead of CDSP in a serving system, we measure the scheduler latency for requests with varying prompt lengths during LLaMA3-8B/-70B deployment. The cluster configuration follows Sec. VII-A. To capture diverse queuing conditions, we randomly sampled instance queuing delays and request arrival timestamps from serving logs in Sec. VII-B. Each prompt length is evaluated over 1,000 trials. As shown in Table III-IV, the scheduling overhead is bounded by 93.79µs/32.90µs for LLaMA3-8B/-70B, respectively. Given that prefill latency typically spans hundreds of milliseconds or more (Table I), these results demonstrate that CDSP scheduling incurs negligible end-to-end overhead in online serving systems.

## VIII. DISCUSSION

