## Optimizing Dynamic Neural Networks with Brainstorm

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是 Brainstorm 的高效 GPU kernel 集合（~3,000 LOC C++/CUDA），包含：(1) **Cell 重排列 kernel**——单 GPU 内根据 Router 的路由决策（Routes tensor）将 tensor 中的 Cell 按目标 branch 重新排列，生成每个 branch 对应的连续输出 tensor，借鉴 Tutel 的 MoE token rearrangement 思路但泛化到任意动态网络；(2) **Sparse All-to-All 通信原语**——多 GPU 场景下用点对点通信集合（point-to-point communication）替代 dense all-to-all，仅传输实际需要跨 GPU 的 Cell，避免因 routing 不均衡导致的 padding 冗余通信；(3) **动态水平融合 kernel 调度**——基于 Router profile 中的 Cell 负载分布，为每个 branch 编译多个不同 shape 的 tuned kernel（通过 TVM auto-tuning），运行时根据实际 Cell 数选择 nearest tuned kernel 并 padding 输入，一个 fused kernel 通过一次 GPU launch 并发执行所有激活的 branch。实验通过 micro-benchmark 比较：(a) 稀疏通信 vs PyTorch all-to-all (NCCL) 的延迟，变化 branch 数（2~8）、GPU 数（2~8）、Cell 大小（32~2048 float32）；(b) 动态水平融合 vs 串行执行（PyTorch）vs 仅 vertical fusion 的延迟，变化 branch 数（2~10）；(c) Profile-guided placement 通信延迟对比；(d) Speculative routing 和 preloading 的 hit/miss 延迟对比。

- 后端平台是什么，配置是什么。
  单 GPU：NVIDIA A100 80GB，CUDA 11.3 + cuDNN 8.6。多 GPU：8× NVIDIA V100 32GB（NVLink），CUDA 11.3 + cuDNN 8.2，NCCL 通信后端。CPU：单 GPU 实验用 AMD EPYC 7V13，多 GPU 实验用 Intel Xeon E5-2690 v4。

- 评估性能的软件/脚本是什么。修改了什么。
  Brainstorm 自定义 CUDA kernel 代码。micro-benchmark 和 end-to-end benchmark 脚本在 GitHub artifact（osdi2023ae 分支）中，提供 shell 脚本自动复现 Figures 12-23。修改内容：
  1. Cell 重排列 kernel：在单个 GPU 上接收 (input_tensor, routes_tensor, num_branches) → 扫描 routes 统计每个 branch 的 Cell 数量 → 分配输出 buffer → 根据 routes 将每个 Cell 从源 tensor scatter copy 到对应 branch 的输出 tensor。实现为 CUDA kernel，支持并行处理多个 branch 的 scatter 操作。
  2. 稀疏通信：将多 GPU 的 Cell routing 实现为 N×N 个点对点 send/recv（而不是一次 all-to-all collective），每对 GPU 间只传输实际需要的 Cell 数量（无 padding）。例如 1024 Cells 分发到 4 个 GPU 上 4 个 branch 各一个，PyTorch all-to-all 需将每个 GPU→每个 GPU 通道 padding 为等量（max 量），而稀疏通信仅传输实际数据。
  3. 动态水平融合的 kernel dispatch：fused kernel 编译时包含多个 tuned 子 kernel（如 Conv2D for 4/32/64 cells），运行时通过 Router 得知每个 branch 的实际 Cell 数后，选择 nearest tuned kernel 并 padding 到对应 shape，计算 global input pointer offsets 后单次 GPU launch 执行所有激活 branch。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  全开源：https://github.com/Raphael-Hao/brainstorm，OSDI 2023 artifact。Docker 镜像预配置，artifact 中每个 Figure 有对应的 shell 脚本自动复现。

  **评估原理与 kernel 全流程（以 sparse all-to-all 和动态水平融合为例）**：

  **Sparse All-to-All 通信原理（Fig 12/14）**：
  1. **输入**：GPU-0 上有 1024 个 Cell（每个 512 float32），Router 决策将 Cell 不均匀分发到 4 个 GPU 上的各 4 个 branch（共 16 个目标）。例如 GPU-0 的 Branch-0 收到 200 个 Cell，Branch-1 收到 10 个，Branch-2 收到 50 个，Branch-3 收到 100 个；GPU-1 同理。
  2. **Baseline (PyTorch all-to-all)**：将 1024 Cells reshape 为 (4, 256)（4 GPU 均匀分片），调用 NCCL all-to-all。由于实际分布不均，每个 GPU 接收 400 Cells 的 buffer 中仅部分有效（其余为 padding），且总的传输量为 1024×4=4096 Cells（含 padding）。
  3. **Brainstorm 稀疏通信**：Router 输出 routes，统计每个 (src_gpu, dst_gpu) pair 的实际 Cell 数量。生成 N×N 个 point-to-point ncclSend/ncclRecv，每对仅发送实际需要的 Cell 数。例如 GPU-0→GPU-1 仅需传 60 个 Cell（含 padding 的 256 相比），总传输量 = sum(实际 Cell 数) 而非 N×总 Cell 数。
  4. **性能输出**：1024 Cells × 512 float32，4 branch/GPU，2 GPU 加速 2.13×，8 GPU 加速 2.66×。加速来源为避免 padding 的额外带宽消耗。

  **动态水平融合 kernel 原理（Fig 13）**：
  1. **Profile 阶段**：Router 收集 branch-0/1/2/3 的历史 Cell 负载分布。发现 P50=4 patches, P90=9 patches, P100=27 patches。
  2. **Tuning 阶段**：Brainstorm 对每个 branch 的 Conv2D 算子用 TVM 为 shape (4, C, H, W) 和 (27, C, H, W) 各 auto-tune 一个 kernel。
  3. **Fused kernel 构建**：生成一个 fused kernel，内含两个子 kernel（4-patch kernel 和 27-patch kernel），通过 global input pointer offsets 指向不同 branch 的输入 buffer。
  4. **运行时**：Router 得知当前 branch-0 收到 4 patches → 使用 4-patch kernel（无 padding）。branch-1 收到 8 patches → 选择 27-patch kernel（padding 19）。branch-2 收到 0 patches → 跳过。一次 GPU launch 并发执行 branch-0 和 branch-1，显著提高 CU utilization。
  5. **性能输出**：vs PyTorch 串行执行加速最高 41.8×（其中 kernel tuning 贡献 13.1×，并发执行贡献 3.18×）。Branch 数较少（≤3）时 overhead 略大于收益，BRT+HF 比 BRT+VF 慢仅 12.3μs。
