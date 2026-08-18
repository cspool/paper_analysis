# C. Implementation Details for Scheduling

**Model features.** CHIME describes a sub-batch i with following arguments:  $c_{p_i}$ , the list of chunk sizes of each prefilling request;  $f_{p_i}$ , the list of finished token numbers of each prefilling request;  $f_{d_i}$ , the list of finished token numbers (on each rank) of each decoding request. The latencies of both sub-batches can be modeled as follows:

<span id="page-8-2"></span>
$$T_{\text{GPU}_0} = t_p(c_{p_0}, f_{p_0}) + t_{\text{batch}}(c_{p_0}, f_{d_1})$$
 (2)

<span id="page-8-4"></span>
$$T_{\text{PIM}_0} = t_d(f_{d_0}) + t_{\text{comm}}(f_{d_0}, c_{p_1})$$
(3)

<span id="page-8-5"></span><span id="page-8-3"></span>
$$T_{\text{GPU}_1} = t_p(c_{p_1}, f_{p_1}) + t_{\text{batch}}(c_{p_1}, f_{d_0})$$
 (4)

$$T_{\text{PIM}_1} = t_d(f_{d_1}) + t_{\text{comm}}(f_{d_1}, c_{p_0})$$
 (5)

Equation 2,4 denotes the latency on the GPU side, which contains the latency of prefilling attention and batched FC operations. Equation 3,5 denotes the latency on the CHIME-PIM in the granularity of rank, which contains the latency of decoding attention, and overlapped data transfer overheads.

**Scheduling policy.** With the following steps, the scheduling policy selects requests to form sub-batches with aligned parallel cross-device execution:

- 1) Add one prefilling request into each sub-batch. When there is no prefilling request, skip the step.
- 2) For each prefilling request added (which implies an increase in  $T_{GPU}$ ), CHIME adds N decoding requests to each sub-batch and assign them to ranks in a load balancing manner. After that, it predicts  $T_{PIM}$  and  $T_{GPU}$  for each sub-batch. If  $T_{PIM} < T_{GPU}$ , it adds another N decoding requests until  $T_{PIM} > T_{GPU}$ .
- If there are remaining prefill requests, it repeats step
   until the PIM memory is exhausted. When the

TABLE I SIMULATOR DETAILS.

<span id="page-9-0"></span>

| GPU Configuration          |                                               | ACC Configuration |                              |  |  |  |  |
|----------------------------|-----------------------------------------------|-------------------|------------------------------|--|--|--|--|
| Processor                  | 8 A100                                        | CPU               | 2TB + 406GB/s                |  |  |  |  |
| Capacity                   | 640GB                                         | НВМ-РІМ           | 640GB + 260.8TB/s (GPU-side) |  |  |  |  |
|                            |                                               |                   | 320GB + 130.4TB/s (Extended) |  |  |  |  |
| Bandwidth                  | 16.3TB/s                                      | DIMM-PIM          | 2TB + 1.6TB/s (R-PIM)        |  |  |  |  |
|                            |                                               |                   | 2TB + 13.0TB/s (CHIME)       |  |  |  |  |
| DIMM / DIMM-PIM: DDR4-3200 |                                               |                   |                              |  |  |  |  |
| Hierarchy                  | 2 Ranks (8 Chips) × 4 Bank Groups × 4 Banks   |                   |                              |  |  |  |  |
| DRAM Timing                | BL=4:CCD=4:RRD=4/8:RCD=22:RAS=52:RP=22:RC=74: |                   |                              |  |  |  |  |
|                            | CL=22:WL=16:CDLR=4/12:WR=24:CCDL=8:RTP=12     |                   |                              |  |  |  |  |

<sup>\*</sup> Bank PU's compute-memory ratio  $N_{cmr}$ =n for GQA-n.

memory is saturated and the bubble is on the PIM side, the last prefilling request of each sub-batch is chunked, dynamically adjusting  $T_{\rm GPU}$  to make it as close as possible to the  $T_{\rm PIM}$  of the other sub-batch.

The larger the value of N, the coarser the granularity of batch execution time adjustment, but the more beneficial it is for achieving load balance among ranks for requests. In our implementation, since MHA has more heads and these heads can be evenly distributed across chips, the N for MHA is set to 1, while the N for GQA is set to 16.

Model selection. Now we describe how CHIME models the execution latencies of various inference operations. First, to model  $T_{\rm GPU}$ , we leverage Random Forest Regression (RFR) for its three advantages: capability of incremental learning, low latency, and high accuracy. The efficiency of RFR for execution latency prediction has been proven in many prior works [53], [80], and it is open to use other models that feature similarly. Second, to model  $T_{\rm PIM}$ , given that CHIME-PIM execution is featured predictable (i.e., execution time is linearly related to the number of computed/transferred tokens), we can use a simple and fast linear model for  $t_{\rm comm}$  and each rank's  $t_d$ . The overall  $t_d$  is determined by the rank with the longest execution time. The models can be described with the following formula (take sub-batch 0 as an example):

$$\begin{split} T_{\text{PIM}_0} &= \text{Linear}(\sum f_{d_0}, \text{len}(f_{d_0}), \sum c_{p_1}) \\ T_{\text{GPU}_0} &= \text{RFR}_p(c_{p_0}, f_{p_0}) + \text{RFR}_{\text{batch}}(\sum c_{p_0}, \text{len}(f_{d_0})) \end{split}$$

Runtime profiling. CHIME applies runtime profiling, which collects the latency information and corresponding batch information during inference. The dataset maintained by CHIME dynamically evolves with the collected data for incrementally updating the model, thereby adapting to changing execution environment. Moreover, we evaluate the prediction models of the CHIME scheduler by splitting the collected 1000 data points into training and test sets in an 8:2 ratio. The experimental results show that our model exhibits great performance predictability, achieving prediction relative errors of less than  $\sim$ 1% (median value < 0.5%).

<span id="page-9-1"></span>

| Model    | Layers N <sub>1</sub> | Heads N <sub>h</sub> | Embedding D <sub>e</sub> | Type  | TP | DP |
|----------|-----------------------|----------------------|--------------------------|-------|----|----|
| OPT-66B  | 64                    | 72                   | 9216                     | MHA   | 2  | 4  |
| QWEN-72B | 80                    | 64                   | 8192                     | GQA-8 | 4  | 2  |
| GPT-175B | 96                    | 96                   | 12288                    | MHA   | 8  | 1  |

<sup>\*</sup> Precision: FP16; Head Embedding:  $E_h = D_e/N_h = 128$ .

TABLE III REAL-WORLD TRACES.

<span id="page-9-2"></span>

| Trace                      | Avg. Lin | Std. L <sub>in</sub> | Avg. Lout | Std. Lout |
|----------------------------|----------|----------------------|-----------|-----------|
| OpenR1-Math-220k [5]       | 96.0     | 75.1                 | 12684.1   | 8464.6    |
| Dolphin-r1 [2]             | 201.9    | 563.0                | 3926.2    | 4216.0    |
| OpenThoughts-114k-math [6] | 89.4     | 66.7                 | 6366.7    | 4662.9    |

#### VII. EVALUATION

#### A. Methodology

Experimental setup. Following prior works [12], [28], [69], our goal is achieving high inference throughput in the long-context and decoding-dominant scenario. The evaluation is built upon DGX-A100 [57]. On the GPU side, 8 NVIDIA A100 GPUs, each with 5 HBM2e of 80 GB capacity, are integrated, providing a total of 156 TFLOPs on FP16. The GPUs are connected via NVLink [74]. On the CPU side, there are 16 channels with 2 DIMMs of total 2TB capacity, equipped with the proposed PIM. We develop a simulator integrating (1) AttAcc [61], the roofline-based simulation for GPU; and (2) CHIME-PIM-sim based on modified DRAMSim3 [51], the trace-driven simulation for CHIME-PIM. We add PIM commands, related timing constraints, and FIFO-based scheduling methods for cycle-level simulation.

**Baseline systems.** We compare CHIME with one GPU baseline and four AFD baseline systems: (1) GPU-only, which executes LLM inference exclusively on GPUs. (2) GPU with HBM-PIM, which equips all existing GPU HBMs with banklevel PUs. (3) GPU with HBM-PIM-EXT, which equips with extended disaggregated HBM-PIMs. Compared to the "HBM-PIM" baseline, the HBM-PIMs in "HBM-PIM-EXT" do not store model parameters. (4) GPU with rank-level DIMM-PIM (R-PIM), which equips all DIMMs with rank-level PUs. (5) GPU with CPU offloading, which leverages CPU as the accelerator. Our simulations of baseline systems uniformly assume maximal accelerator-side bandwidth utilization during attention computation. Table I lists the hardware specifications. For GPU, CPU, and DIMM(-PIM) configurations, we follow the configurations in DGX-A100 systems. For HBM-PIM-EXT, considering that HBM2e is more than  $6\times$  the price of DDR4 [3], [4], [7], we configure 20 HBM-PIM modules (320 GB) to provide  $\sim 1/6$  of the capacity available in CHIME.

*LLM models.* We evaluate CHIME on three LLM models with varying model sizes: OPT-66B, QWEN-72B, and GPT-175B, with FP16 as the data precision. Considering both the GPU

![](_page_10_Figure_0.jpeg)

<span id="page-10-1"></span>Fig. 11. End-to-end inference throughput on real-world traces. The throughput of GPU-only baseline is normalized to 1. For the OPT-66B model, GPU out-of-memory (OOM) errors occurred in two traces, so the throughput of CHIME is normalized to 1. In case (d), length of requests is reduced by 10×.

memory capacity and inter-GPU communication overhead, we apply various parallelism configurations, listed in Table II.

Workloads. We use three real-world LLM inference datasets: OpenR1-Math-220K [5], OpenThoughts-114k-math [6], Dolphin-r1 [2]. All of these datasets are used for evaluating the performance of LLM inference on long context. The details about the traces are shown in Table III. Further, to evaluate the short context scenario, we add an additional trace, Dolphin-short, which shortens each text entry in Dolphin-r1 to 1/10 of its original length.

#### <span id="page-10-0"></span>B. End-to-end Performance

We evaluate the end-to-end throughput for each LLM model and workload using various baselines. For each evaluation, we randomly select and execute 1,000 requests from the traces. We measure the overall throughput by dividing the total number of output tokens by the total execution time.

Throughput with real-world traces. Fig. 11-a,b,c presents the normalized throughput of CHIME compared with five baselines. The results show that CHIME achieves the highest throughput over all baselines with various settings, up to  $5.15 \times$  higher throughput than the HBM-PIM baseline,  $3.45 \times$ than the HBM-PIM-EXT baseline, 3.94× than the GPUonly baseline, and 7.21× higher than R-PIM baseline. These improvements are attributed to the following reasons. First, CHIME achieves much larger batch sizes compared to the HBM-based baselines. For example, for GPT-175B, CHIME has 2 TB of memory on the host memory for KV cache storage, while GPU and HBM-PIM baselines have only about 310 GB of memory in total after deploying the model, and HBM-PIM-EXT has only 320GB of memory under the same cost budget. Second, the CPU and R-PIM baselines suffer from the limited bandwidth. For example, CHIME offers about  $8\times$ the bandwidth of R-PIM with bank-level PUs.

We observe that HBM-PIM's sub-batch method demonstrates promising efficiency in achieving high throughput with enhanced parallelism, though our analysis reveals certain limitations when processing Dolphin trace on OPT-66B. Dividing a batch into sub-batches results in smaller batch sizes, leading to severe GPU underutilization and performance below that of the conventional GPU-only implementation. On the contrary, CHIME benefits from much larger batch sizes and thus achieves higher throughput.

![](_page_10_Figure_8.jpeg)

<span id="page-10-2"></span>Fig. 12. Breakdown of performance improvement with CHIME. The results are evaluated using the OpenR1 trace. "C-N%" denotes CHIME with N% available memory capacity. "BS" denotes "batch size". "Latency" is "latency per batch". The results of GPU baseline are normalized to 1

**Performance improvement breakdown.** We analyze the sources of performance improvement compared with the GPU and HBM-PIM baselines. Fig.12 shows the throughput, average batch size (the sum of two sub-batches), and average latency per batch from our end-to-end evaluation. Compared with GPU and HBM-PIM, CHIME increases the batch size by  $6.6\times$ , while the latency per batch increases by only  $2.2\times$ . This indicates that the throughput improvement of CHIME primarily stems from the increased batch size and the corresponding improvement in GPU utilization.

In addition, Fig. 12 illustrates the trend of how batch size growth affects throughput. We manually limit the memory capacity available to CHIME and measure the resulting performance. As the available memory gradually increases from 10% to 100%, the batch size scales linearly. However, since the latency per batch also increases, the rate of throughput improvement gradually diminishes. This observation is consistent with the analysis in Fig. 2, which shows that the marginal benefit of increasing batch size on throughput decreases.

Performance in short context scenarios. As shown in Fig. 11-d, in the short context scenario, CHIME still achieves higher throughput than both the GPU and HBM-PIM baselines by achieving higher batch sizes. However, compared to Fig. 11-c, the advantage of CHIME on each model is diminished. This is because in the short context scenario, the marginal benefit of increasing the batch size exhibits diminishing returns. This observation is also consistent with our analysis in §III-D.

**Scalability analysis.** We further evaluate CHIME performance with various CHIME-PIM configurations, showing that performance scalability requires simultaneous scaling

![](_page_11_Figure_0.jpeg)

<span id="page-11-1"></span>Fig. 13. Scalability analysis. The results are evaluated using the OpenR1 trace. The performance of the base memory configuration is normalized to 1.

![](_page_11_Figure_2.jpeg)

<span id="page-11-2"></span>Fig. 14. Hardware ablation study. The latency of CHIME's naive implementation is normalized to 1.

in memory capacity and bandwidth. We first establish the base memory configuration as "1 rankset, 512 GB". Based on this, we scale up the memory in different ways for different baselines: "Bw-only" indicates that we only scale up the memory bandwidth with more ranksets but keep the capacity at 512GB, while "Cap-only" means we only scale up the capacity from 512GB to 4TB. "Bw-Cap" implies that we scale up both the bandwidth and capacity. We execute 1,000 requests from the trace and calculate the average throughput.

Fig. 13 shows the results of GPT-175B and QWEN-72B with OpenR1 trace. Results for other models and traces are similar. It shows that expanding either bandwidth or capacity alone does not effectively improve throughput. For example, when exclusively scaling memory capacity or bandwidth by  $8\times$  on GPT-175B, the throughput only increases by  $2.28\times$  and  $1.01\times$ , respectively. In contrast, when both bandwidth and capacity are enlarged, the throughput increases by  $8.23\times$ . This demonstrates that CHIME effectively leverages the value of both scalability aspects.

# C. Implementation Details for Scheduling

**Model features.** CHIME describes a sub-batch i with following arguments:  $c_{p_i}$ , the list of chunk sizes of each prefilling request;  $f_{p_i}$ , the list of finished token numbers of each prefilling request;  $f_{d_i}$ , the list of finished token numbers (on each rank) of each decoding request. The latencies of both sub-batches can be modeled as follows:

<span id="page-8-2"></span>
$$T_{\text{GPU}_0} = t_p(c_{p_0}, f_{p_0}) + t_{\text{batch}}(c_{p_0}, f_{d_1})$$
 (2)

<span id="page-8-4"></span>
$$T_{\text{PIM}_0} = t_d(f_{d_0}) + t_{\text{comm}}(f_{d_0}, c_{p_1})$$
(3)

<span id="page-8-5"></span><span id="page-8-3"></span>
$$T_{\text{GPU}_1} = t_p(c_{p_1}, f_{p_1}) + t_{\text{batch}}(c_{p_1}, f_{d_0})$$
 (4)

$$T_{\text{PIM}_1} = t_d(f_{d_1}) + t_{\text{comm}}(f_{d_1}, c_{p_0})$$
 (5)

Equation 2,4 denotes the latency on the GPU side, which contains the latency of prefilling attention and batched FC operations. Equation 3,5 denotes the latency on the CHIME-PIM in the granularity of rank, which contains the latency of decoding attention, and overlapped data transfer overheads.

**Scheduling policy.** With the following steps, the scheduling policy selects requests to form sub-batches with aligned parallel cross-device execution:

- 1) Add one prefilling request into each sub-batch. When there is no prefilling request, skip the step.
- 2) For each prefilling request added (which implies an increase in  $T_{GPU}$ ), CHIME adds N decoding requests to each sub-batch and assign them to ranks in a load balancing manner. After that, it predicts  $T_{PIM}$  and  $T_{GPU}$  for each sub-batch. If  $T_{PIM} < T_{GPU}$ , it adds another N decoding requests until  $T_{PIM} > T_{GPU}$ .
- If there are remaining prefill requests, it repeats step
   until the PIM memory is exhausted. When the

TABLE I SIMULATOR DETAILS.

<span id="page-9-0"></span>

| GPU Configuration          |                                               | ACC Configuration |                              |  |  |  |  |
|----------------------------|-----------------------------------------------|-------------------|------------------------------|--|--|--|--|
| Processor                  | 8 A100                                        | CPU               | 2TB + 406GB/s                |  |  |  |  |
| Capacity                   | 640GB                                         | НВМ-РІМ           | 640GB + 260.8TB/s (GPU-side) |  |  |  |  |
|                            |                                               |                   | 320GB + 130.4TB/s (Extended) |  |  |  |  |
| Bandwidth                  | 16.3TB/s                                      | DIMM-PIM          | 2TB + 1.6TB/s (R-PIM)        |  |  |  |  |
|                            |                                               |                   | 2TB + 13.0TB/s (CHIME)       |  |  |  |  |
| DIMM / DIMM-PIM: DDR4-3200 |                                               |                   |                              |  |  |  |  |
| Hierarchy                  | 2 Ranks (8 Chips) × 4 Bank Groups × 4 Banks   |                   |                              |  |  |  |  |
| DRAM Timing                | BL=4:CCD=4:RRD=4/8:RCD=22:RAS=52:RP=22:RC=74: |                   |                              |  |  |  |  |
|                            | CL=22:WL=16:CDLR=4/12:WR=24:CCDL=8:RTP=12     |                   |                              |  |  |  |  |

<sup>\*</sup> Bank PU's compute-memory ratio  $N_{cmr}$ =n for GQA-n.

memory is saturated and the bubble is on the PIM side, the last prefilling request of each sub-batch is chunked, dynamically adjusting  $T_{\rm GPU}$  to make it as close as possible to the  $T_{\rm PIM}$  of the other sub-batch.

The larger the value of N, the coarser the granularity of batch execution time adjustment, but the more beneficial it is for achieving load balance among ranks for requests. In our implementation, since MHA has more heads and these heads can be evenly distributed across chips, the N for MHA is set to 1, while the N for GQA is set to 16.

Model selection. Now we describe how CHIME models the execution latencies of various inference operations. First, to model  $T_{\rm GPU}$ , we leverage Random Forest Regression (RFR) for its three advantages: capability of incremental learning, low latency, and high accuracy. The efficiency of RFR for execution latency prediction has been proven in many prior works [53], [80], and it is open to use other models that feature similarly. Second, to model  $T_{\rm PIM}$ , given that CHIME-PIM execution is featured predictable (i.e., execution time is linearly related to the number of computed/transferred tokens), we can use a simple and fast linear model for  $t_{\rm comm}$  and each rank's  $t_d$ . The overall  $t_d$  is determined by the rank with the longest execution time. The models can be described with the following formula (take sub-batch 0 as an example):

$$\begin{split} T_{\text{PIM}_0} &= \text{Linear}(\sum f_{d_0}, \text{len}(f_{d_0}), \sum c_{p_1}) \\ T_{\text{GPU}_0} &= \text{RFR}_p(c_{p_0}, f_{p_0}) + \text{RFR}_{\text{batch}}(\sum c_{p_0}, \text{len}(f_{d_0})) \end{split}$$

Runtime profiling. CHIME applies runtime profiling, which collects the latency information and corresponding batch information during inference. The dataset maintained by CHIME dynamically evolves with the collected data for incrementally updating the model, thereby adapting to changing execution environment. Moreover, we evaluate the prediction models of the CHIME scheduler by splitting the collected 1000 data points into training and test sets in an 8:2 ratio. The experimental results show that our model exhibits great performance predictability, achieving prediction relative errors of less than  $\sim$ 1% (median value < 0.5%).

<span id="page-9-1"></span>

| Model    | Layers N <sub>1</sub> | Heads N <sub>h</sub> | Embedding D <sub>e</sub> | Type  | TP | DP |
|----------|-----------------------|----------------------|--------------------------|-------|----|----|
| OPT-66B  | 64                    | 72                   | 9216                     | MHA   | 2  | 4  |
| QWEN-72B | 80                    | 64                   | 8192                     | GQA-8 | 4  | 2  |
| GPT-175B | 96                    | 96                   | 12288                    | MHA   | 8  | 1  |

<sup>\*</sup> Precision: FP16; Head Embedding:  $E_h = D_e/N_h = 128$ .

TABLE III REAL-WORLD TRACES.

<span id="page-9-2"></span>

| Trace                      | Avg. Lin | Std. L <sub>in</sub> | Avg. Lout | Std. Lout |
|----------------------------|----------|----------------------|-----------|-----------|
| OpenR1-Math-220k [5]       | 96.0     | 75.1                 | 12684.1   | 8464.6    |
| Dolphin-r1 [2]             | 201.9    | 563.0                | 3926.2    | 4216.0    |
| OpenThoughts-114k-math [6] | 89.4     | 66.7                 | 6366.7    | 4662.9    |

#### VII. EVALUATION

#### A. Methodology

Experimental setup. Following prior works [12], [28], [69], our goal is achieving high inference throughput in the long-context and decoding-dominant scenario. The evaluation is built upon DGX-A100 [57]. On the GPU side, 8 NVIDIA A100 GPUs, each with 5 HBM2e of 80 GB capacity, are integrated, providing a total of 156 TFLOPs on FP16. The GPUs are connected via NVLink [74]. On the CPU side, there are 16 channels with 2 DIMMs of total 2TB capacity, equipped with the proposed PIM. We develop a simulator integrating (1) AttAcc [61], the roofline-based simulation for GPU; and (2) CHIME-PIM-sim based on modified DRAMSim3 [51], the trace-driven simulation for CHIME-PIM. We add PIM commands, related timing constraints, and FIFO-based scheduling methods for cycle-level simulation.

**Baseline systems.** We compare CHIME with one GPU baseline and four AFD baseline systems: (1) GPU-only, which executes LLM inference exclusively on GPUs. (2) GPU with HBM-PIM, which equips all existing GPU HBMs with banklevel PUs. (3) GPU with HBM-PIM-EXT, which equips with extended disaggregated HBM-PIMs. Compared to the "HBM-PIM" baseline, the HBM-PIMs in "HBM-PIM-EXT" do not store model parameters. (4) GPU with rank-level DIMM-PIM (R-PIM), which equips all DIMMs with rank-level PUs. (5) GPU with CPU offloading, which leverages CPU as the accelerator. Our simulations of baseline systems uniformly assume maximal accelerator-side bandwidth utilization during attention computation. Table I lists the hardware specifications. For GPU, CPU, and DIMM(-PIM) configurations, we follow the configurations in DGX-A100 systems. For HBM-PIM-EXT, considering that HBM2e is more than  $6\times$  the price of DDR4 [3], [4], [7], we configure 20 HBM-PIM modules (320 GB) to provide  $\sim 1/6$  of the capacity available in CHIME.

*LLM models.* We evaluate CHIME on three LLM models with varying model sizes: OPT-66B, QWEN-72B, and GPT-175B, with FP16 as the data precision. Considering both the GPU

![](_page_10_Figure_0.jpeg)

<span id="page-10-1"></span>Fig. 11. End-to-end inference throughput on real-world traces. The throughput of GPU-only baseline is normalized to 1. For the OPT-66B model, GPU out-of-memory (OOM) errors occurred in two traces, so the throughput of CHIME is normalized to 1. In case (d), length of requests is reduced by 10×.

memory capacity and inter-GPU communication overhead, we apply various parallelism configurations, listed in Table II.

Workloads. We use three real-world LLM inference datasets: OpenR1-Math-220K [5], OpenThoughts-114k-math [6], Dolphin-r1 [2]. All of these datasets are used for evaluating the performance of LLM inference on long context. The details about the traces are shown in Table III. Further, to evaluate the short context scenario, we add an additional trace, Dolphin-short, which shortens each text entry in Dolphin-r1 to 1/10 of its original length.

#### <span id="page-10-0"></span>B. End-to-end Performance

We evaluate the end-to-end throughput for each LLM model and workload using various baselines. For each evaluation, we randomly select and execute 1,000 requests from the traces. We measure the overall throughput by dividing the total number of output tokens by the total execution time.

Throughput with real-world traces. Fig. 11-a,b,c presents the normalized throughput of CHIME compared with five baselines. The results show that CHIME achieves the highest throughput over all baselines with various settings, up to  $5.15 \times$  higher throughput than the HBM-PIM baseline,  $3.45 \times$ than the HBM-PIM-EXT baseline, 3.94× than the GPUonly baseline, and 7.21× higher than R-PIM baseline. These improvements are attributed to the following reasons. First, CHIME achieves much larger batch sizes compared to the HBM-based baselines. For example, for GPT-175B, CHIME has 2 TB of memory on the host memory for KV cache storage, while GPU and HBM-PIM baselines have only about 310 GB of memory in total after deploying the model, and HBM-PIM-EXT has only 320GB of memory under the same cost budget. Second, the CPU and R-PIM baselines suffer from the limited bandwidth. For example, CHIME offers about  $8\times$ the bandwidth of R-PIM with bank-level PUs.

We observe that HBM-PIM's sub-batch method demonstrates promising efficiency in achieving high throughput with enhanced parallelism, though our analysis reveals certain limitations when processing Dolphin trace on OPT-66B. Dividing a batch into sub-batches results in smaller batch sizes, leading to severe GPU underutilization and performance below that of the conventional GPU-only implementation. On the contrary, CHIME benefits from much larger batch sizes and thus achieves higher throughput.

![](_page_10_Figure_8.jpeg)

<span id="page-10-2"></span>Fig. 12. Breakdown of performance improvement with CHIME. The results are evaluated using the OpenR1 trace. "C-N%" denotes CHIME with N% available memory capacity. "BS" denotes "batch size". "Latency" is "latency per batch". The results of GPU baseline are normalized to 1

**Performance improvement breakdown.** We analyze the sources of performance improvement compared with the GPU and HBM-PIM baselines. Fig.12 shows the throughput, average batch size (the sum of two sub-batches), and average latency per batch from our end-to-end evaluation. Compared with GPU and HBM-PIM, CHIME increases the batch size by  $6.6\times$ , while the latency per batch increases by only  $2.2\times$ . This indicates that the throughput improvement of CHIME primarily stems from the increased batch size and the corresponding improvement in GPU utilization.

In addition, Fig. 12 illustrates the trend of how batch size growth affects throughput. We manually limit the memory capacity available to CHIME and measure the resulting performance. As the available memory gradually increases from 10% to 100%, the batch size scales linearly. However, since the latency per batch also increases, the rate of throughput improvement gradually diminishes. This observation is consistent with the analysis in Fig. 2, which shows that the marginal benefit of increasing batch size on throughput decreases.

Performance in short context scenarios. As shown in Fig. 11-d, in the short context scenario, CHIME still achieves higher throughput than both the GPU and HBM-PIM baselines by achieving higher batch sizes. However, compared to Fig. 11-c, the advantage of CHIME on each model is diminished. This is because in the short context scenario, the marginal benefit of increasing the batch size exhibits diminishing returns. This observation is also consistent with our analysis in §III-D.

**Scalability analysis.** We further evaluate CHIME performance with various CHIME-PIM configurations, showing that performance scalability requires simultaneous scaling

![](_page_11_Figure_0.jpeg)

<span id="page-11-1"></span>Fig. 13. Scalability analysis. The results are evaluated using the OpenR1 trace. The performance of the base memory configuration is normalized to 1.

![](_page_11_Figure_2.jpeg)

<span id="page-11-2"></span>Fig. 14. Hardware ablation study. The latency of CHIME's naive implementation is normalized to 1.

in memory capacity and bandwidth. We first establish the base memory configuration as "1 rankset, 512 GB". Based on this, we scale up the memory in different ways for different baselines: "Bw-only" indicates that we only scale up the memory bandwidth with more ranksets but keep the capacity at 512GB, while "Cap-only" means we only scale up the capacity from 512GB to 4TB. "Bw-Cap" implies that we scale up both the bandwidth and capacity. We execute 1,000 requests from the trace and calculate the average throughput.

Fig. 13 shows the results of GPT-175B and QWEN-72B with OpenR1 trace. Results for other models and traces are similar. It shows that expanding either bandwidth or capacity alone does not effectively improve throughput. For example, when exclusively scaling memory capacity or bandwidth by  $8\times$  on GPT-175B, the throughput only increases by  $2.28\times$  and  $1.01\times$ , respectively. In contrast, when both bandwidth and capacity are enlarged, the throughput increases by  $8.23\times$ . This demonstrates that CHIME effectively leverages the value of both scalability aspects.

