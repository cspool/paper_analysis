# <span id="page-7-1"></span>5 Communication Compression

We further reduce communication overhead by applying communication compression. To maintain convergence stability, mixed-precision training frameworks typically transfer tensors awaiting reduction in higher precision, such as FP32, to ensure more accurate accumulation. A common example of this is gradient reduce-scatter in data parallelism.

DP communication compression. As MoE model parameters increase, so does the communication overhead for parameter and gradient synchronization in data parallelism. Prior work has explored gradient compression to mitigate this cost. In our BF16 mixed-precision training, we carefully apply FP32-to-BF16 precision reduction for gradient synchronization, balancing efficiency and convergence stability.

Specifically, as shown in Figure [11,](#page-7-0) we retain the main gradients in FP32 during local gradient accumulation in pipeline parallelism. After each model stage completes accumulation, instead of relying solely on reduce-scatter for gradient synchronization, we cast gradients to BF16 and perform all-to-all communication within the data parallel group to gather the required gradient shards, which are then locally aggregated in FP32. Our results show that this approach introduces negligible precision loss compared to directly performing reduce-scatter with FP32, while reducing gradient communication overhead by 50%.

This approach minimizes risk for two key reasons. First, it performs a one-time conversion of accumulated gradients to BF16 during communication, while the local gradient accumulation is maintained in FP32 precision. Second, instead of using ring-style reduce for BF16 gradient communication, it employs all-to-all communication, with the final reduction computed using FP32 summation. This design prevents precision loss that could arise from repeated accumulation of BF16 values in ring-based reductions.

We observe that casting large gradients and performing all-to-all communication increases peak memory consumption, potentially causing out-of-memory errors. To mitigate this, we develop a memory-efficient operator that in-places BF16 gradients into half of the FP32 input buffer while using

<span id="page-8-2"></span>

| Ī | Name          | #layers | h    | #heads | m  | $h_{ffn}$ | #experts | top-k |
|---|---------------|---------|------|--------|----|-----------|----------|-------|
|   | Internal-352B | 60      | 4096 | 32     | 4  | 14336     | 32       | 3     |
|   | Mixtral-8×7B  | 32      | 4096 | 32     | 4  | 14336     | 8        | 2     |
|   | Mixtral-8×22B | 56      | 6144 | 48     | 6  | 16384     | 8        | 2     |
| I | Hunyuan-Large | 64      | 6400 | 80     | 10 | 18304     | 16       | 1     |
|   | Phi-3.5-MoE   | 32      | 4096 | 32     | 4  | 6400      | 16       | 2     |
|   | DeepSeekMoE   | 28      | 2048 | 16     | 1  | 1408      | 64       | 6     |
|   |               |         |      |        |    |           |          |       |

Table 2. Model configurations in evaluation.

the remaining half as the output buffer for BF16 all-to-all communication, preventing peak memory growth.

Communication compression for FP8 training. In lowprecision FP8 training, the proportion of communication time increases due to reduced computation time. To mitigate communication overhead, we explore compressing communication volume using FP8 precision with appropriate quantization techniques. Currently, we apply communication compression in FP8 MoE training with tensor parallelism, focusing on reduction scenarios prone to overflow or underflow. For example, we adopt the E4M3 format (4-bit exponent and 3-bit mantissa) for all tensors. Similar to DP reduce-scatter compression, we replace BF16 TP reduce-scatter with FP8 all-to-all in forward propagation and perform reduction in FP32 precision. In the corresponding backward propagation, we apply FP8 all-gather for gradients. Notably, simply reducing precision leads to loss misalignment with BF16 training. To mitigate this, we apply per-token activation quantization for forward communication and per-channel quantization for backward communication. In backward propagation, we further group quantization along the token dimension using a small group size (e.g., 128).

#### <span id="page-8-0"></span>6 Evaluation

In this section, we present a comprehensive evaluation of MegaScale-MoE, covering overall training performance (§6.1), ablation studies of MegaScale-MoE's key optimizations (§6.2), and the effectiveness of the precision-communication co-design (§6.3). Table 2 lists the configurations of the MoE models used in our evaluation, detailing hidden size (h), FFN intermediate size ( $h_{ffn}$ ), number of experts, and top-k values. The evaluation is conducted on NVIDIA H800 GPUs unless otherwise specified, with the specifications provided in Table 4.

