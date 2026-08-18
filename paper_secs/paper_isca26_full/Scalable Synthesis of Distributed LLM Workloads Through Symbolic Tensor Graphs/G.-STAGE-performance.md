# G. STAGE performance

We evaluate STAGE in terms of runtime and memory footprint across scales. Results show that STAGE significantly reduces the time required to collect graph workloads for simulation. Experiments are conducted on a Linux server with four Intel Xeon E7-8880 v4 processors (2.2 GHz) and 354 GiB of DDR3-1333 memory.

<span id="page-11-3"></span><sup>8</sup>Note that due to differences in modeling scope, target systems, and execution environments across backends, the reported runtimes by each simulator do not encompass all workload components, and so comparisons across the different simulators is not the focus of this experiment.

<span id="page-11-2"></span>TABLE IX
STAGE WITH DIFFERENT SIMULATION/EMULATION BACKENDS
LLAMA3.1-70B, TRAINING, DP=2, TP=4, 32 MICRO-BATCHES

| Simulator  | Target System | Runtime [ms] | LoC for<br>Adaption |
|------------|---------------|--------------|---------------------|
| SimAI      | 8xH100        | 3,909.5      | 73                  |
| SililAi    | 8xH200        | 3,791.5      | 1 /3                |
| ScaleSim   | 8xTPUv5e      | 1,843.8      | 34                  |
| Scalesiiii | 8xTPUv4       | 1,452.9      | 34                  |
| Genie      | 8x100Gbps IB  | 33,128.5     | 46                  |
| Genie      | 8x400Gbps IB  | 11,441.7     | 40                  |

![](_page_11_Figure_11.jpeg)

<span id="page-11-4"></span>Fig. 16. STAGE Runtime Scaling with Number of GPUs

We also evaluated STAGE across a wide range of GPU scales to assess how generation time grows with model and system size. As shown in Fig. 16, runtime increases nonlinearly due to the expanding parallel configuration space, yet STAGE remains highly efficient. At 32K GPUs, it generates graphs for a 540B dense LLM in just 28 minutes. For more complex models like Mixtral-8x7B, with added expert parallelism, generation remains practical at around 50 minutes. For a larger scale of 128K devices, for which to the best of our knowledge, no publicly accessible real-world system currently exists, STAGE still generates the workload within hours while keeping memory usage below 400 MB in all cases. In contrast, real-system trace generation is expensive and slow. For instance, collecting an execution trace for training LLaMA-3.1-70B with 128 micro-batches on 32 H100 GPUs takes approximately 47 GPU-minutes, whereas STAGE synthesizes the corresponding workload in only **37 CPU-seconds**. Moreover, real-system traces require re-collection when the target system changes, as traces are inherently system-specific. In contrast, STAGE provides a more generalized solution. Furthermore, access to the physical system required for trace collection may not always be feasible.

