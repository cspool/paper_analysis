## Efficient and Adaptable Overlapping for Computation and Communication via Signaling and Reordering

> **近似层次匹配说明**：本文核心创新在 kernel 级的 signaling + reordering 重叠设计，但端到端评估通过修改 vLLM/Megatron-LM/xDiT 框架将 FlashOverlap 集成到多 GPU Serving 系统中，以 throughput 为指标评测端到端性能提升，属于 Serving 调度层的集成与优化。

- 属于Serving调度的实现是什么？实验比较什么？
  FlashOverlap 通过替换原始 linear layer 和后续通信原语（AllReduce、ReduceScatter、All-to-All）为带 overlap 的 FlashOverlap 实现，集成到 vLLM（LLM 推理）、Megatron-LM（LLM 训练）、xDiT（text-to-video 生成）三个主流 Serving/训练框架中。实验比较集成 FlashOverlap 前后的端到端 throughput，以及与 decomposition-based（Async-TP）和 fusion-based（FLUX）方法的 throughput 差异。

- 硬件平台是什么，配置是什么。
  NVIDIA A800 GPU（NVLink pairwise 连接，1935GB/s HBM 带宽，312 TFLOPS FP16），用于所有端到端评估。软件环境：CUDA 12.1、NCCL 2.19.3、PyTorch 2.5.1、CUTLASS 3.6.0。

- 开源Serving框架是什么。修改了什么。
  - **vLLM**（LLM 推理）：替换 Llama3-70B TP=8 配置中 attention 和 FFN 后的 GEMM+AllReduce 对为 FlashOverlap 实现。chunk_size=16384。不修改请求调度逻辑（continuous batching 保留）。
  - **Megatron-LM**（LLM 训练）：替换 Llama3-70B TP=8 和 Mixtral-8x7B EP=4,TP=2 配置中的 GEMM+ReduceScatter 和 GEMM+All-to-All 对为 FlashOverlap 实现。input_token 分别为 16384 和 32768。层数设 8 和 4 以适配单节点内存。
  - **xDiT**（text-to-video 生成）：替换 Step-Video-T2V TP=4 配置中的 GEMM+AllReduce 为 FlashOverlap 实现。input_token=33792。
  所有框架的修改均为：定位到 linear layer 输出后的通信原语调用 → 替换为 FlashOverlap 的 GEMM-with-signaling + NCCL 通信双 stream 实现。框架的其余调度逻辑（batching、pipeline、memory management）保持不变。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  已开源：https://github.com/infinigence/FlashOverlap（ae 分支），Zenodo DOI: 10.5281/zenodo.17201530。

  **端到端 Serving 集成全过程**（以 Llama3-70B TP=8 LLM 推理 on vLLM + 8×A800 GPUs 为例）：
  1. **输入**：用户请求到达 vLLM API server。vLLM continuous batching 调度器将请求按可用 KV cache slot 批量组成 batch → tokenize → 转换为 input tensor。
  2. **模型前向传播**：vLLM 按 TP=8 分区执行 Llama3-70B 各 transformer layer。每个 layer 的 attention projection 和 FFN 输出均为 GEMM 在 8 个 GPU 上的部分结果（每 GPU 持有 M/8 行）。
  3. **FlashOverlap 执行的 GEMM+AllReduce**（替代原 sequential 路径）：
     a. **GEMM kernel (Stream A)**：CUTLASS GEMM 执行 M×N×K 矩阵乘法，main loop 完整不变。tile 按 wave pattern 顺序完成，epilogue 中 pre-communication reordering 将完成 tile 的数据散射到连续地址通信 buffer，同时 atomicAdd 更新 counting table。
     b. **Signaling kernel (Stream B)**：周期性查询 counting table，当某 wave group 的所有 tile 完成时，调用 `ncclAllReduce(sendbuf, recvbuf, data_size, ncclFloat16, ncclSum, comm, stream_B)` 对该 group 的数据执行 AllReduce。
     c. **Overlap 并发**：Stream A 中后续 wave group 的 GEMM 计算与 Stream B 中前一个 wave group 的 NVLink AllReduce 通信并发执行。
  4. **Post-communication reordering**：fused 到后续 RMSNorm kernel 中，根据 mapping table 将通信后的数据恢复原始顺序。
  5. **其余层正常执行**：FlashOverlap 不修改 attention、embedding、softmax 等其他算子。
  6. **输出**：vLLM 完成所有层后 decode 出 token → 逐 token 生成直到 <eos> 或 max_length → 返回响应。

  **作用**：在保持现有 Serving 框架调度逻辑不变的前提下，通过替换 GEMM+通信对的实现降低通信瓶颈。端到端 speedup：LLM 推理（Llama3-70B/vLLM）1.05×，LLM 训练（Llama3-70B/Megatron-LM）1.08-1.13×，LLM 训练（Mixtral-8x7B/Megatron-LM）1.05-1.05×，T2V 生成（Step-Video-T2V/xDiT）1.11-1.12×。T2V 因大 input token 数（33792）通信占比较高、加速最大。FlashOverlap 通过 interference-free computation 保证不退化：即使 overlap 效果有限的 case 也不会比 non-overlap baseline 更差。
