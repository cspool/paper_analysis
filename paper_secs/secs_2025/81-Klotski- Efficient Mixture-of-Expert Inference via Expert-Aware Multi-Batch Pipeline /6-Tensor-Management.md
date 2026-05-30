# 6 Tensor Management

### <span id="page-6-2"></span>6.1 Adaptive Tensor Placement

KLOTSKI constructs a multi-level heterogeneous memory space consisting of VRAM, DRAM, and disk to meet the storage demands of MoE models in resource-constrained environments. Then, we propose an adaptive tensor placement, which intelligently allocates tensors based on the available memory resources in the current environment, thereby enhancing the utilization of existing resources.

Firstly, the GPU memory is primarily used to store necessary tensors required for current computations and prefetched tensors. When there is ample free GPU memory available, it can be further utilized to reduce some I/O operations. Specifically, we can choose storage locations for different types of tensors such as expert, gate, attention, KV cache, and activation. Furthermore, support is provided for layer granularity distribution. For example, placing the experts of the first three layers in VRAM, the experts of the next twenty layers in DRAM, and the remaining in disk.

Secondly, inactive tensors can be offloaded to either CPU memory or disk. We prioritize allocating CPU memory to experts. This is because the MoE layer faces the challenge that the experts requested by the gating function cannot be accurately predicted in advance. Therefore, when handling tasks with large batch sizes, it is highly likely that immediate transfers of experts will be needed, necessitating the rapid transfer of the required expert to GPU memory. Considering

the faster transfer bandwidth of CPU memory, which provides quicker response times, we prioritize placing expert parts in CPU memory.

Additionally, when sufficient CPU memory is available, we use  $pin\_memory$  to achieve faster CPU-GPU communication. When CPU memory is insufficient and disk usage is necessary, to reduce the GPU getting tensors from disk, which is slow, we dynamically maintain tensors in the CPU memory. Specifically, we dynamically manage tensors for a fixed number of layers L within the limited CPU memory. As the computation proceeds to layer i, the GPU prefetches tensors for layer i+1 from CPU memory, while the CPU prefetches tensors for layer i+L from the disk and removes tensors for layer i. This strategy effectively utilizes the idle CPU-disk bandwidth, thereby reducing the interaction between GPU and disk.

### 6.2 Correlation-aware Expert Prefetcher

For dense models, the offloading strategies can directly prefetch the next layer. However, it is different for MoE models. Only after completing the computation of the gate can the activated experts be determined, making it challenging to design a unified prefetching strategy.

To address this, Klotski design a correlation-aware expert prefetcher. In § 5, the prefetched experts need to engage in most computations across multiple batches to reduce intralayer bubbles effectively. As illustrated in Figure 5, there are hot experts in the inference of MoE, where a few experts cover the majority of computations. Therefore, the prefetching targets for the MoE layer are the gate and hot experts. Since MoE is data-sensitive and hot experts may vary with different inputs, we establish a data-aware expert correlation table to identify the hot experts that tokens in the current multi-batch tend to select. Specifically, we record the correlations (i.e., frequency relationships) between experts activated by tokens at different layers through pre-run, resulting in a table. During inference, we use this table to determine each token's expert tendency in the current layer based on its selections in the previous *l* layers. The larger the value of *l*, the more accurate the prefetching. This process is illustrated in Figure 8, where each layer has four experts, the gate selects the top-1 expert, and l = 1. For the expert activation path of each token in the multi-batch, we look up the table to determine their expert tendencies in the current layer. We then aggregate the tendencies of all tokens across multiple batches and select the top-K experts for prefetching. K is by default equal to k in top-k because, based on the observation in § 3.2, K experts will generally cover the majority of the token computations.

In addition, the expert correlation table is updated during the inference so that expert prefetching can become more and more accurate, as the table is continuously updated to understand the tasks at hand. To prevent the prefetching

<span id="page-7-0"></span>

| Layer i            |                     | Laye                          | er i+1     |                                         |        |                      |
|--------------------|---------------------|-------------------------------|------------|-----------------------------------------|--------|----------------------|
| Expert             | Expert              | Historical Selected Frequency |            |                                         |        |                      |
|                    | 0                   |                               | 38         |                                         |        |                      |
| 0                  | 1                   |                               | 27         |                                         |        |                      |
| U                  | 2                   |                               | 97         |                                         |        |                      |
|                    | 3                   |                               | 15         |                                         |        |                      |
|                    | 0                   |                               | 66         |                                         |        |                      |
| 1                  | 1                   |                               | 35         |                                         |        |                      |
|                    | 2                   | 41                            |            |                                         |        |                      |
|                    | 3                   |                               | 117        |                                         |        |                      |
|                    |                     |                               |            |                                         |        |                      |
| ********           |                     |                               |            | ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, |        |                      |
| Batch<br>Act. path | _                   |                               | Batch      |                                         | Expert | Aggregate<br>Results |
| Act. pati          |                     |                               | ► tendency | aggregate                               | 9 _ 1  | 3                    |
| Act. patt          |                     |                               |            |                                         | 2      | 18 🗸                 |
| Act poth           | $\exists \parallel$ |                               | tendency   |                                         | 3      | 9                    |
| Act. path          |                     |                               | teridericy |                                         | 4      | 2                    |

**Figure 8.** An example of the expert correlation table. Each expert layer has four experts. The gate selects the top-1 expert. The correlation path length l is 1.

tendencies of other tasks from influencing current tasks, we refrain from saving the updates to the file.

On the other hand, for non-expert tensors, we adopt a prefetching strategy similar to that used for dense models, where we prefetch the tensor during the computations of the previous layer. This is because non-expert tensors are involved in computation only once during a forward pass and remain inactive at other times.

