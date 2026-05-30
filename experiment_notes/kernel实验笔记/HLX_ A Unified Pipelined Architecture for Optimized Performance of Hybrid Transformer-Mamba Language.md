## HLX: A Unified Pipelined Architecture for Optimized Performance of Hybrid Transformer-Mamba Language Models

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  HLX 在 kernel 调度/数据流层面的核心实现包括两个新的细粒度流水线数据流：(i) **PipeFlash**——在 FA-2 的块级计算基础上进一步细粒度划分，每次处理 Q block 中的两行，将 QK^T、local softmax、PV、update O 四个步骤以流水线方式执行，其中 softmax 和 update O 与 MatMul（QK^T 和 PV）并发执行，实现非 MatMul 延迟隐藏。中间数据量从 FA-2 的 128KB（score/probability 矩阵）降至 1KB，减少 4.8×；(ii) **PipeSSD**——首次提出 SSD 的融合流水线执行，将 fused SSD 分为三个流水线阶段：第1阶段为 dA 相关预处理，第2阶段为 CB^T、CB^TLdt、Y_Diag，第3阶段为 Y_Off 与 states_N 并发计算 + Y_Final 与 update states。PipeSSD 减少 DRAM 访问 6.8×，中间数据从 642KB 降至 58.5KB（11×）。两个数据流通过控制每引擎处理的行数实现流水线阶段平衡：PipeFlash 中 QK^T 与 PV 按 `⌈block_size/d_head⌉` 调整行数；PipeSSD 中 Y_Off/states_N 阶段通过平衡总计算周期而非严格匹配行数实现灵活平衡。

  实验比较：(1) FA-2 和 SSD 的 compute utilization 对比 GPU (A100, H100) 和 TPUv3 baseline（序列长度 1K-128K）；(2) FA-2/SSD kernel 延迟加速比 vs GPU 和 TPU；(3) FA-3 on H100 vs PipeFlash on HLX^60（序列长度和 batch size 扫描）；(4) 端到端 Hybrid-2.7B 模型延迟加速比；(5) batch size 1-128 下的 compute utilization 和加速比变化（固定 seqlen=1K）；(6) PipeFlash vs FA-3 on H100 的单独比较。结果：FA-2 compute utilization 97.5%@128K（A100 native 61%），SSD compute utilization 78.4%（A100 26.9%，H100 <40%），平均加速 FA-2 1.75×/2.78×（vs A100/H100），SSD 2.91×/4.95×（vs A100/H100），端到端 1.56×/2.08×（vs A100/H100）。

- 后端平台是什么，配置是什么。
  GPU baseline: NVIDIA A100 80GB (312 TFLOPS, 1935 GB/s BW, 7nm, 826mm², 300W), NVIDIA H100 80GB (756 TFLOPS, 2000 GB/s BW, 4nm, 814mm², 350W)。TPU baseline: TPUv3 (61.5 TFLOPS, 450 GB/s BW, 16nm, 324mm², 225W, 16MB on-chip SRAM, 16GB DRAM)。HLX 配置三档：HLX^60 (614.4 TFLOPS, 30.4MB SRAM, 14nm, 475mm², 358W, 对标 H100)；HLX^30 (307.2 TFLOPS, 15.2MB SRAM, 14nm, 235.8mm², 174.64W, 对标 A100)；HLX^6 (61.44 TFLOPS, 3.04MB SRAM, 14nm, 47.16mm², 35.06W, 对标 TPUv3)。模型：Hybrid-2.7B (Mamba2attn-2.7B)，注意力层 30 head × d_head=128，SSD 80 head × d_head=64，d_state=128，block_size=256。数据类型 FP16。

- 评估性能的软件/脚本是什么。修改了什么。
  GPU baseline 使用 NVIDIA Nsight Systems 和 Nsight Compute 测量执行时间和 compute utilization。GPU 端运行来自 Mamba-2 GitHub 仓库 (https://github.com/state-spaces/mamba) [11] 的 CUDA 优化 FA-2/FA-3/SSD kernel。HLX 使用自研 cycle-level simulator，实现了 PipeFlash 和 PipeSSD 数据流的逐 cycle 模拟，以及完整 Hybrid 模型的端到端评估（含 FFN、conv1D、RMSNorm）。

  修改/新增内容：
  - PipeFlash 数据流：FA-2 的 4 步计算（QK^T → softmax → PV → update O）从块级改为更细粒度的行级（每次 2 行 Q）流水线，使 softmax/update O 与 MatMul 并发，score/probability 矩阵从 128KB 降为 1KB
  - PipeSSD 数据流：首次将 SSD 的 5 个分离 kernel（chunk cumsum, chunk state, state passing, BMM chunk, chunk scan）融合为单 kernel 的三阶段流水线（预处理 → Y_Diag → Y_Off∥states_N + Y_Final∥update states），中间数据 642KB→58.5KB
  - 流水线阶段平衡策略：根据 bottleneck（DPE MatMul 计算周期）控制每引擎处理行数，当 block_size=d_head=d_state 时可达近 100% compute utilization
  - 自定义 cycle-level simulator：模拟 URSC 内 DPE/RVPE/UpE 的流水线执行、NoC 数据转发、GS 暂存、DRAM 访问

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未提供 HLX simulator 的开源链接（2026年5月检索未找到公开代码仓库）。GPU baseline 使用开源 Mamba-2 仓库 [11] (https://github.com/state-spaces/mamba) 的 CUDA kernel。HLX 自研 cycle-level simulator 的评估原理：以 Hybrid-2.7B 模型的计算图作为输入，将 FA-2 部分映射为 PipeFlash 数据流（Q block 内逐行流水线 QK^T→softmax→PV→update O），SSD 部分映射为 PipeSSD 数据流（三阶段流水线预处理→Y_Diag→Y_Off/states_N/Y_Final/update states），在三个 HLX 配置（HLX^6/30/60）下模拟 URSC 流水线执行周期，计算 DRAM 访问延迟（根据配置的 HBM2E/HBM2 带宽），输出每层/每 kernel 的 compute utilization 和延迟。GPU 端同理，使用 Nsight Compute/Systems 测量实际 CUDA kernel 执行时间，读取 SM 利用率。compute utilization 定义为实际达到的 TFLOPS / 理论峰值 TFLOPS。
