## ScheMoE- An Extensible Mixture-of-Experts Distributed Training System with Tasks Scheduling

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - Pipe-A2A：一种新型 pipelined all-to-all 通信算法，将 intra-node Send/Recv 操作与 inter-node Send/Recv 操作通过两个异步 CUDA stream（Intra-Stream 和 Inter-Stream）并行执行，充分利用异构带宽资源。
  - OptSche 最优调度算法：在 MoE layer 的 forward/backward 中，将 compute tasks（compression、decompression、expert computation）与 communication tasks（A2A dispatch/combine）按最优执行顺序重叠，最大化隐藏通信开销。
  - 数据压缩集成：支持 ZFP、FP16、INT8 等压缩算法在 MoE A2A 通信前后的 compress/decompress 任务纳入调度。
  - 实验比较：ScheMoE vs Tutel vs Faster-MoE 的 step time；Pipe-A2A vs NCCL-A2A、1DH-A2A、2DH-A2A 的 A2A 通信时间；消融实验（Naive / +ZFP / +ZFP+Pipe-A2A / +全部）。

- 后端平台是什么，配置是什么。
  - 32-GPU 集群：8 nodes × 4 Nvidia RTX2080Ti（@1.35GHz, 11GB Memory）
  - CPU：Dual Intel Xeon Gold 6230 CPU@2.10GHz，Memory: 512GB DDR4
  - Intra-node：PCIe 3.0 ×16
  - Inter-node：Mellanox MT27800 (ConnectX-5) 100Gb/s InfiniBand
  - 软件栈：PyTorch-1.10, Ubuntu-18.04, CUDA-10.2, cuDNN-7.6, OpenMPI-4.1.4, NCCL-2.13

- 评估性能的软件/脚本是什么。修改了什么。
  - ScheMoE 系统本身（开源于 https://github.com/Fragile-azalea/ScheMoE），基于 PyTorch 的 C/C++ 和 CUDA 扩展实现（~1200行 C/C++）。
  - 修改 PyTorch MoE layer：将原有 MoE layer 替换为 ScheMoE 的抽象模块（AbsCompressor、AbsAlltoAll、AbsExpert），支持任务队列化、Profiler 性能建模、Scheduler 调度。
  - 修改 A2A 实现：新增 Pipe-A2A 算法，使用两个异步 stream 分别处理 intra-node 和 inter-node SR（Send/Recv）操作。
  - 开源情况：代码开源在 GitHub（https://github.com/Fragile-azalea/ScheMoE），使用 ZFP、NCCL、Hetu、Tutel 等第三方库，采用 MIT 或类似许可证。
  - 评估原理与全过程（以 CT-MoE 模型为例）：
    ```
    输入：MoE layer config (B, L, M, H, E=32, k)
    ↓
    Step 1 - Profiler 预热：对 AbsCompressor (compress/decompress)、AbsAlltoAll (A2A)、AbsExpert (fflayer)
              分别 profile 时间，构建 t(C1), t(C2), t(A1), t(A2), t(D1), t(D2), t(E) 性能模型
    ↓
    Step 2 - 输入分区：将 gating 输出 tensor I ∈ R^{(E, C, M)} 按容量 C 均匀划分为 r=2 份
              I_1, I_2 → 各自进入独立的任务管道
    ↓
    Step 3 - OptSche 调度（r=2 时最优顺序）：
              CompTask顺序: (C_1^1 → C_1^2) → (D_1^1 → E^1 → C_2^1) → (D_1^2 → E^2 → C_2^2) → (D_2^1 → D_2^2)
              CommTask(A1,A2): 在前置 CompTask 完成后立即启动，由 CUDA stream 异步执行
    ↓
    Step 4 - Pipe-A2A（以 8-GPU, 2-node 为例）：
              Intra-Stream: SR(i, intra-node-j) 依次执行
              Inter-Stream: SR(i, inter-node-k) 依次执行
              → 两 stream 并行，intra 通信被 inter 通信隐藏
    ↓
    Step 5 - 数据压缩（可选）：
              Compress: I → ZFP_compress(I) 或 FP32→FP16 量化（减少 4× volume）
              → A2A dispatch (压缩后数据)
              → Decompress: 恢复原精度 → expert computation
    ↓
    Step 6 - 计时：cudaEvent 记录 MoE layer 起始到结束的 wall-clock time
    输出：step_time (ms) → speedup = t_baseline / t_ScheMoE
    ```
