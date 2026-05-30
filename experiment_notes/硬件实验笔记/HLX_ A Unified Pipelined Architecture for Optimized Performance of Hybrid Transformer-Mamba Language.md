## HLX: A Unified Pipelined Architecture for Optimized Performance of Hybrid Transformer-Mamba Language Models

- 属于硬件架构的实现是什么？实验比较什么？
  HLX 是首个统一支持 Hybrid Transformer-Mamba 模型的硬件加速器架构，核心创新为 URSC（Unified Reconfigurable Streamlined Core），由以下模块组成：(i) **DPE（Dot-Product Engine）×2**：每个 DPE 含 32 个 DPU lane，每 lane 含 8 个 DPU，每个 DPU 含 16 个 FP16 乘法器 + adder tree + accumulator，支持 MatMul 和 conv1D（通过 demux 控制 accumulation 旁路）。DPU lane 内 8 个 DPU 共享 16 个 broadcast activations，各自接收不同 weight 以支持行级流水线；(ii) **RVPE（Reconfigurable Vector Processing Engine）**：含 2 个 RVPU + VMEM（存预处理中间数据）。每个 RVPU 含 256 元素 add/sub 单元、rowsum/cumsum 单元、2 个乘法单元、SFU（reciprocal/exp/max/log/sqrt/SiLU），通过可重构 local NoC 支持 4 种操作模式（PipeFlash local softmax、PipeSSD pre-processing、Y_Diag element-wise mul、Y_Off/states_N element-wise mul）；(iii) **UpE（Update Engine）**：含 2 个 UpU（Update Unit），执行 update O/update states（旧值×exp(decay) + 新值）、Y_Final 计算（Y_Diag + Y_Off element-wise add）、最终 O_i 归一化。全局架构含 top controller（管理计算模式和 DRAM 访问）、transpose unit、URSC、global scratchpad (GS)，通过 NoC 互联。

  实验比较：(1) compute utilization vs A100/H100 GPU 和 TPUv3（序列长度 1K-128K）；(2) FA-2/SSD 延迟加速比；(3) 端到端 Hybrid-2.7B 速度 vs GPU 和 TPU；(4) batch size 变化（1-128，固定 seqlen=1K）；(5) 面积/功耗 vs A100/H100/TPUv3（14nm 实测 → 7nm 缩放）；(6) HLX vs FA-3 on H100；(7) HLX vs SOTA 加速器（VGA/MARCA/SOFA）；(8) 统一设计 overhead（Transformer-only vs Mamba-2-only vs HLX unified）。结果：HLX^30 端到端加速 1.56× vs A100，面积仅 10.2% (83.9mm² vs 826mm²)，功耗 36.2% (108.47W vs 300W)；HLX^60 端到端加速 2.08× vs H100，面积 20.8% (169mm² vs 814mm²)，功耗 57.5% (201.8W vs 350W)；统一设计 overhead 3.0% 面积 + 2.9% 功耗 vs Transformer-only，4.4% 面积 + 3.5% 功耗 vs Mamba-2-only。

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  自研 custom cycle-level simulator。论文未提供开源链接（2026年5月检索未找到公开仓库）。GPU baseline 使用 NVIDIA Nsight Systems 和 Nsight Compute 测量实际硬件性能。HLX 硬件设计使用 SystemVerilog RTL 实现，Synopsys Design Compiler 在 14nm 工艺、625MHz、0.8V 下综合。SRAM 使用 memory compiler 生成。

- 模拟器模拟什么的性能，修改了什么。
  Cycle-level simulator 模拟 HLX 的 URSC 流水线执行性能：(i) DPE MatMul 操作周期（公式 ⌈d_reduction/DPU_size⌉ × ⌈(d_in×d_out)/DPE_size⌉）；(ii) RVPE 向量/SFU 操作延迟；(iii) UpE 更新操作延迟；(iv) DPE→RVPE→DPE→UpE 之间的数据转发延迟（通过 NoC）；(v) GS 读写延迟；(vi) DRAM 访问延迟（基于配置的 HBM2E/HBM2 带宽参数）。Simulator 建模了 PipeFlash（QK^T in DPE#0 → local softmax in RVPE → PV in DPE#1 → update O in UpE）和 PipeSSD（pre-processing in RVPE → CB^T/CB^TLdt/Y_Diag in DPE#0→RVPE→DPE#1 → Y_Off/states_N/Y_Final/update states in RVPE→DPE#0/DPE#1→UpE）的精确流水线时序。

- 开源情况。基于开源文档和论文，使用例子解释模拟器如何使用？作用是什么？至少具体到模拟器模拟性能的原理和模拟器输入到性能输出的全过程。
  论文未开源 HLX simulator 或 RTL。模拟器使用方式（根据论文描述重构）：输入为 Hybrid 模型的计算图（层类型、维度 [batch, head, seq_len, d_head/d_state]）、HLX 配置（URSC 数量、SRAM 容量、DRAM 带宽、频率），模拟器将每层映射到 URSC 的数据流模板（FA-2→PipeFlash，SSD→PipeSSD，FFN/conv1D/RMSNorm→对应 DPE/RVPE 执行），逐 cycle 追踪每个引擎的 busy/idle 状态、数据在 NoC 和 GS 中的驻留周期、DRAM 读写排队延迟。最终输出每 kernel 的 compute utilization（实际 MatMul FLOPS / 理论峰值 FLOPS）、端到端延迟（kernel execution time only，不含 CPU-GPU 通信和 kernel launch overhead）、面积/功耗估算（RTL 综合结果 × URSC 数量 + DRAM 功耗模型）。

  面积/功耗评估原理：单 core RTL 综合（14nm, 625MHz, 0.8V）→ 按 URSC 数量线性扩展 → 按工艺缩放因子缩至 7nm（参考 VGA [25] 方法）→ 加 DRAM 功耗（基于 HBM2E/HBM2 vendor 数据）。GPU 功耗通过 NVIDIA-SMI 实测。
