# VII. EXPERIMENTAL SETUP & METHODOLOGY

Infrastructure. Evaluation is performed on an NVIDIA DGX B200 running the Ubuntu 24.04.2-based DGX Server Version 7.0.2 software stack with NVIDIA driver 580.82.07. The machine has 8 B200 GPUs, each with 192 GB of memory and 148 SMs. We use the vLLM 0.10.2 inference server [26] with PyTorch 2.8 for CUDA 12.8 on Python 3.12. For disaggregated prefill, we use LMCache 0.3.6 [9] and NIXL 0.6.0 [27]. We build the agentic pipeline with AutoGen [57].

Models and Loads. We use a total of six LLMs from the Llama 3 herd [15] (3.2-1B and 3.1-8B) and the Qwen3 family [59] (4B, 8B, 14B, and 32B-FP8) as well as the LLaVA vision model [29]–[31].

Depending on the experiment, loads follow either a Poisson process or the Azure LLM conversation inference trace [34]. Inputs come from the ShareGPT Vicuna [19] and scientific papers [11] datasets. For SLOs, we use those from MLPerf Inference 5.1 benchmark [36]: the interactive scenario with 0.5 s TTFT and 30 ms TPOT for the shorter ShareGPT inputs and the server scenario with 2 s TTFT and 100 ms TPOT for the longer inputs from scientific papers. For the Azure trace, we use SLOs from DynamoLLM [51], shown in Table I. Frequency Domains. Current GPUs do not expose finegrained frequency domains to software, so we emulate spatial DVFS by running workloads on multiple GPUs and allocating TPCs such that the total allocation equals one full GPU. To account for shared-resource effects absent from this setup, we compare: (i) isolated execution, (ii) same-GPU compute partitioning, and (iii) MIG-based partitioning. The observed contention increases TTFT and TPOT by around 3% on average and less than 7% in the worst case, so we conservatively scale SLO targets by this amount in all emulated spatial-DVFS experiments, making our multi-GPU emulation as realistic as possible without compromising measurement validity.

TABLE I: Azure trace request class SLOs.

| Request length | TTFT (ms) | TPOT (ms) |
|----------------|-----------|-----------|
| < 256          | 250       | 100       |
| < 1024         | 400       | 100       |
| ≥ 8192         | 2000      | 100       |

Baseline. We compare PowerWeave to LithOS [12], the stateof-the-art DVFS scheme that relies on a single frequency domain for the entire GPU. LithOS performs spatial multitenancy and sets a device-wide frequency based on requirements of all collocated models.

Modeling Spatial DVFS. We measure total energy as the sum across all GPUs used in an experiment, subtracting idle energy in proportion to each workload's unallocated TPC share. In this way, each measurement accounts only for the idle power attributable to its resource allocation. This provides realistic energy measurements under our real-hardware setup. On B200 GPUs, idle power measures ≈140 W. Because the baseline runs on one GPU, no idle power needs to be deducted.

We retrieve energy and power measurements using NVIDIA Data Center GPU Manager (DCGM) 4.2.2 [40]. Because prior work shows that NVIDIA's monitoring tools may produce inconsistent power readings [60], we validate DCGM energy against the product of DCGM-reported power and experiment duration. We retain measurements when both values closely agree; otherwise, we rerun the experiment. Since experiments run sufficiently long, energy metrics are stable and consistent. Area Estimation. To estimate the area overhead of each additional DVFS domain, we synthesize/place-route the voltagedomain boundary synchronization logic and voltage regulator control logic in mflowgen [8].

For voltage regulator control logic, we use the open-source OpenFASoC DLDO generator [3], which provides synthesizable RTL for parameterized regulators. We generate the DLDO controller macro, extract the control logic, run the full RTL-to-GDSII flow using a 7 nm PDK, and extract the post-layout area [10]. This value is then scaled to match the 5nm technology node [22]. For the voltage-domain boundary synchronization logic, the FIFO RTL is based on [45]. The logic is synthesized using a 130 nm PDK, and the post-layout area is scaled to match the 5 nm technology node [13], [22].

