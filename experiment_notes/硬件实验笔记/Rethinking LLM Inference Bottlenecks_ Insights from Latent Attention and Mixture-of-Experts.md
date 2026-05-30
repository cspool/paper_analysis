## Rethinking LLM Inference Bottlenecks: Insights from Latent Attention and Mixture-of-Experts

- 属于硬件架构的实现是什么？实验比较什么？
  - 实现：论文评估了 **PIM（Processing-in-Memory）架构**在 MLA + MoE 场景下的有效性，具体建模了 Duplex [62]——一种基于 HBM 的 PIM 架构，专为加速 MoE 层设计。Duplex 使用 RP_acc = 8 的 PIM 设备（利用 4 倍于 GPU 的 HBM 内存带宽）。论文分析 PIM 在不同 batch size 下的 MoE 执行吞吐量，并与 GPU baseline 对比。
  - 实验比较：(Figure 14) **PIM vs GPU normalized throughput**：在不同 batch size（1 到 128）和 sequence length（2048, 8192）下，Duplex PIM 执行 MoE 层相对于 GPU baseline 的归一化吞吐量。

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  - 论文使用自研 in-house simulator（基于 LLMSimulator: https://github.com/scale-snu/LLMSimulator）对 Duplex PIM 进行建模。LLMSimulator 是论文团队 SNU-SCALE 开发的 LLM 推理模拟器。Duplex 原型论文发表于 MICRO 2024。

- 模拟器模拟什么的性能，修改了什么。
  - 模拟器模拟 Duplex PIM 设备执行 MoE expert 计算的性能（吞吐量、延迟），与 GPU baseline 对比。PIM 设备模型参数：RP_acc = 8（低 ridge point，因高内存带宽），内存带宽为 GPU HBM 的 4 倍。模拟器评估 MoE expert FC 层在不同 batch size 下在 PIM 上的执行时间，计算归一化吞吐量。

- 开源情况。基于开源文档和论文，使用例子解释模拟器如何使用？作用是什么？至少具体到模拟器模拟性能的原理和模拟器输入到性能输出的全过程。
  - 开源情况：LLMSimulator 已开源（https://github.com/scale-snu/LLMSimulator），论文的 in-house simulator 在其基础上构建，增加了 PIM 设备模型和 MoE 通信模型。论文未开源其修改版。
  - 模拟器工作原理：
    ```
    # === LLMSimulator-based PIM 评估流程 ===
    # 输入参数:
    #   - 模型配置: DeepSeek-R1 (d_emb, n_hd, n_e, n_k, etc.)
    #   - 硬件配置: PIM RP_acc=8, BW=4×GPU HBM
    #   - 工作负载: batch_size, sequence_length, skewness
    #
    # 模拟流程:
    # 1. Layer-wise 计算分析:
    #    对每个 decoder block 的每层:
    #      FLOPs = f(B, L, d_emb, d_hd, ...)
    #      Memory_access = g(B, L, d_emb, d_hd, ...)
    #      ArI = FLOPs / Memory_access
    # 2. 性能模型:
    #    if ArI < RP_device:
    #      time = Memory_access / BW_device     # memory-bound
    #    else:
    #      time = FLOPs / Throughput_device     # compute-bound
    # 3. PIM vs GPU 对比:
    #    - PIM device: 高 BW (4× HBM), 低 RP (8 Op/B)
    #      适合低 batch size (ArI 低, memory-bound)
    #    - GPU device: 高 Throughput (2250 TFLOPS), 高 RP (281 Op/B)
    #      适合高 batch size (ArI 高, compute-bound)
    # 4. 输出: 归一化吞吐量 = Throughput_PIM / Throughput_GPU
    ```
    - 关键结论：当 batch size < 32 时，PIM 因其高内存带宽可有效降低延迟和提高吞吐量；当 batch size 增大时，PIM 因低 ridge point 而 compute-bound，GPU 更优。因此 MLA+MoE 场景下 PIM 更适用于低 batch/低序列长度的推理场景。
