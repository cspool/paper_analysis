# 3 Overview of ExeGPT

ExeGPT is a system for running LLM inference with optimal execution schedule. Under a given latency constraint, it finds and runs with the execution schedule that maximizes inference throughput. ExeGPT offers efficient methods for throughput and latency trade-off ensuring that the execution maintains adequate decoding batches without introducing substantial pipeline bubbles.

The system comprises four primary components: XProfiler, XSimulator, XScheduler, and XRunner, which are illustrated in Figure 2. XProfiler measures the execution times of a single encoding and decoding layer across all possible configurations. With the execution times of a single layer, XSimulator constructs the timeline when requested by XScheduler. The scheduler identifies optimal configuration parameter values using the simulator to estimate throughput and latency at selected configuration points. XRunner then executes with the configuration parameters determined by the scheduler. Below we provide further details of each of these four system components.

**XProfiler.** For a single encoding and decoding layer, the profiler separately measures the execution times of the attention kernel and the rest of the encoding/decoding layer, considering all feasible tensor-parallel degrees. For the former (the attention kernel), we conduct sweeps across batch sizes and, for each batch size, perform sweeps over possible sequence lengths. For the latter, we measure the execution time with sweeping input sizes (i.e., batch sizes×input lengths).

We also measure the synchronization overhead of tensorand pipeline-parallel execution. These synchronization operations do not interfere with each other as they either occur at different time points (between tensor- and pipeline-parallel synchronization) or utilize point-to-point communication channels (between different tensor-parallel groups).

**XSimulator.** Using the profiling results, the simulator constructs the execution timeline based on the configuration parameters, such as batch size and tensor-parallel degree, given by the scheduler. For the simulation, we calculate the expected batch sizes of each encoding/decoding iteration, utilizing the probability distribution of the input/output sequence lengths. Our scheduling algorithm allows queries from different encoding iterations to be decoded in a same batch. The simulator takes this into account and calculates the corresponding distribution for the timeline simulation (details explained in Section 6).

**XScheduler.** The scheduler searches over the solution space to find the optimal values of the configuration parameters that maximize throughput under a given latency bound. The details are explained in Section 5, but we have developed a branch-and-bound search algorithm specifically designed for the efficient exploration. This algorithm exploits the monotonic property of configuration parameters with respect to throughput and latency, enabling to examine a small subset of the configuration points to pinpoint the optimal solution. **XRunner.** This serves as the execution engine for running LLM inference on multiple distributed GPUs. It takes as input the execution schedule provided by XScheduler and ensures that the schedule is enforced during the execution.

To implement XRunner, we extended NVIDIA's Faster-Transformer. We first implemented early-termination of completed queries in a batch, along with the compaction of the key/value cache entries for those terminated queries. For WAA Scheduling (explained in Section 4), we implemented the transfer of the key/value cache entries from encoding GPUs to decoding GPUs. To minimize interference, we copy the entries from the source GPU memory to CPU memory and then transfer them to the target GPU memory. We also implemented several optimizations, such as overlapping of computation and communication for pipeline-parallel execution.

### 4 Scheduling Strategies and Trade-offs

For efficient LLM inference, we must strike a balance between encoding and decoding operations. This balance must ensure maintaining sufficiently large decoding batches while also minimizing pipeline bubbles. However, in current inference systems, these two operations are tightly coupled, hindering efficient execution. For instance, in all existing systems, there is an implicit assumption that the decoding of a query takes place immediately after its encoding, and it is further assumed that all decoding iterations of a query are executed consecutively. From a resource allocation perspective, GPUs that execute the *i*'th encoding layer are expected to handle the *i*'th decoding layer in all existing systems.

In this section, we challenge these assumptions and propose two alternative scheduling strategies that decouple the

![](_page_4_Figure_2.jpeg)

Figure 3. Comparing RRA and WAA Scheduling strategies

execution of encoding and decoding differently, allowing for flexible resource allocation and efficient execution. Furthermore, we introduce four control mechanisms that offer a flexible trade-off between latency and throughput. These methods are employed by XScheduler's optimal scheduling algorithm, which is described in detail in Section 5.

#### 4.1 Decoupled Scheduling of Encoders and Decoders

We first present our layer allocation policies and then the scheduling strategies for each allocation. Our scheduling is based on two allocation policies: Round-Robin Allocation (RRA) and Workload-Aware Allocation (WAA). RRA simply assigns GPUs to the encoders and decoders of the model in a round-robin fashion. That is, when allocating for N GPUs, RRA assigns  $\frac{E}{N}$  consecutive encoders and  $\frac{D}{N}$  consecutive decoders to each GPU, where E and D are the number of encoders and decoders respectively. Figure 3 illustrates this allocation at the top for 4 GPUs and 8 encoders/decoders.

The WAA policy estimates the workload sizes of the encoding/decoding layers and assigns GPUs proportionally to these sizes, with some GPUs dedicated to encoding and others to decoding in a pipelined manner. It has two sub-types, WAA-C and WAA-M, which assign the layers based on their computation time and memory consumption, respectively. When allocating for N GPUs and N pipeline stages, WAA-C assigns  $[N_{\overline{(C_E+C_D)}}]$  GPUs to encoders and  $[N_{\overline{(C_E+C_D)}}]$  GPUs to decoders, where  $C_E$  and  $C_D$  are estimated encoding/decoding times. This balances the amount of computation between encoding and decoding GPUs. Notably, we decouple the encoding and decoding stages and run them asynchronously to prevent delays caused by varying input lengths during encoding. Figure 3 (bottom) shows an example of WAA policy.

WAA-M assigns GPUs to ensure that all GPUs consume an equal amount of memory rather than equal computation. In WAA-C the decoding GPUs consume more memory due to larger key/value caches for storing output tokens. When this becomes the performance bottleneck, WAA-M can be used to improve the performance by, for example, increasing the batch size. However, using WAA-M may result in much longer encoding stages, as fewer GPUs are assigned to encoding. This increases the worst-case latency, which we consider in our scheduling to meet latency constraints.

![](_page_4_Figure_9.jpeg)

**Figure 4.** Execution timelines of RRA/WAA Scheduling with GPU<sub>1</sub>–GPU<sub>4</sub>. The blue/red boxes are encoding/decoding iterations, respectively. The numbers are mini-batch IDs.

For a decoder-only model, where a decoding layer runs both input encoding and output decoding, WAA stores two copies of the model as it dedicates each GPU for either encoding or decoding. This memory overhead is reasonably modest compared to the key/value cache sizes, and we provide a detailed evaluation in Section 7.3. We now describe the scheduling strategies.

**RRA Scheduling.** This strategy executes input encoding once and then decoding for  $N_D$  iterations, repeatedly executing the encoding and decoding phases as in Figure 4(a). By running the decoding only for a fixed number of iterations and then feeding in more input data, we keep the decoding batch size sufficiently large for high resource utilization. It is worth noting that running encoding more frequently (decreasing  $N_D$ ) increases throughput but comes at the cost of longer latency.

To configure RRA Scheduling, we need to set the encoding batch size  $B_E$ , decoding batch size  $B_D$ , and decoding iteration  $N_D$ . We set  $B_E$  to match the number of completed queries in the previous  $N_D$  decoding iterations to maintain consistent batch sizes across repeated encoding/decoding phases.

**WAA Scheduling.** This strategy executes encoders and decoders in their assigned GPUs, decoupling the two phases, as shown in Figure 4(b). The decoupling enables encoding and decoding to operate with different batch sizes. When an

input batch is encoded and passed for decoding, it is merged and decoded with previously decoded batch data.

To configure WAA Scheduling, we need to determine the encoding and decoding batch sizes. To maintain consistent batch sizes, we use the output sequence length to set the batch sizes. That is, if the average length of output sequences is  $S_D$ , we set the decoding batch size  $B_D$  equal to the encoding batch size  $B_E$  multiplied by  $S_D$  (i.e.,  $B_D = B_E \cdot S_D$ ). These batch sizes impact the encoding and decoding times of a pipeline stage, which is used for the allocation of layers by the WAA policy.

Comparison of the Strategies. RRA and WAA Scheduling both address the performance challenges in existing systems by simultaneously tackling the problems of diminishing decoding batches (in FT and DSI) and pipeline bubbles (in ORCA). RRA Scheduling places more emphasis on maintaining large decoding batches than minimizing pipeline bubbles, allowing for a small amount of bubbles between encoding and decoding phases as shown in Figure 4(a). In contrast, WAA Scheduling results in smaller encoding/decoding batches than those of RRA for decoder-only models due to the memory overhead. However, it excels in minimizing pipeline bubbles by separately optimizing the encoding and decoding stages.

Due to these differences, WAA performs better when the output sequence length is relatively short. In such cases, RRA requires more frequent execution of encoding phases, exacerbating the pipeline bubble problem. Conversely, for large decoder-only models, RRA outperforms WAA because it does not incur the memory overhead associated with WAA. We discuss these differences further in our evaluation.

#### 4.2 Controlling Latency and Throughput Trade-off

RRA and WAA provide four control mechanisms to tailor execution schedules: batch size, decoder micro-batch, partial tensor parallelism, and encoding frequency. These variables enable to increase throughput at the cost of longer latency. Notably, three of the mechanisms, except batch size, are novel proposals in this paper. We explain these four mechanisms. **Batch size.** Increasing the batch size improves the parallelism of the inference kernels and thus improves the throughput but it also increases the latency. Hence the batch size is generally used for the trade off of inference throughput and latency.

**Decoder micro-batch.** In WAA, we make it possible to split a decoding batch into multiple micro-batches, which can largely reduce the latency. Figure 4(b) and (c) illustrate the impact of applying decoder micro-batches. In Figure 4(b), with  $GPU_1$  encoding and  $GPU_2$ – $GPU_4$  decoding,  $GPU_2$  has to wait for  $GPU_4$  to complete the decoding iteration before it can start the next iteration, thus requiring seven pipeline stages to generate two tokens. In contrast, applying micro-batches, as shown in Figure 4(c), enables overlapping of decoding between micro-batches, requiring only  $3\frac{2}{3}$  pipeline

stages for generating two tokens. Decoder micro-batches can be used for the trade off of throughput and latency as increasing the number of micro-batches reduces the latency but decreases the throughput.

Partial tensor parallelism. By splitting a layer and executing it on multiple GPUs in a tensor parallel way, we can reduce the depth of the execution pipeline and decrease latency at the cost of lower throughput due to synchronization overhead. Our approach allows tensor parallelism to be applied to a subset of GPUs, as shown in Figure,4(d), enabling a trade-off between latency and throughput by increasing or decreasing the subset. When the degree of tensor parallelism is fixed, increasing the number of GPUs in tensor-parallel execution reduces latency by decreasing the pipeline depth but increases throughput.

**Encoding frequency.** In RRA Scheduling, adjusting the encoding frequency can increase throughput or decrease latency. That is, running the encoding more frequently maintains larger decoding batches, increasing throughput at the cost of latency.

XScheduler optimizes throughput under a given latency bound by adjusting these four variables. Its scheduling algorithm takes advantage of their characteristics of throughput and latency trade-off, which we explain in the next section.

