## Xilinx Alveo U280（数据中心 FPGA 加速卡，HBM2）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Xilinx Alveo U280 是赛灵思（AMD）的数据中心 FPGA 加速卡，搭载 XCU280 UltraScale+ 芯片（108 万 LUT、9MB 片上 BRAM、30MB UltraRAM、9,024 DSP slice）与两个 4GB HBM2 stack（总带宽 460GB/s），通过 PCIe 接入主机。是 TAGT 加速器（ISCA 2026）的实现平台：RTL 经 Xilinx Vivado 2019.1 综合，保守工作频率 280MHz，在其上评估图 Transformer 推理（5 数据集 × 4 GT 模型）。
- TAGT 资源占用（Table IV，按模型）：DSP 73.6%–80.2%、LUT 40.1%–49.5%、FF 30.4%–35.2%、BRAM 59.3%–69.7%、UltraRAM 80.3%–89.7%。片内缓冲共约 3.3MB（Feature 1MB + Weight 1MB + Partial 512KB + TDS-CSR 512KB + Task FIFO 128KB + Output 128KB）。本库已有 Alveo U55C/U200（HBM/DDR 图加速平台）与 Alveo U50（HBM2+PCIe）条目，U280 是更大规模 HBM2 版本。

从硬件架构角度拆解术语，比如术语在硬件架构中发挥作用的流程例子。通过联网搜索让回答具体和精准。
- TAGT 在 U280 上的运转：HBM2（460GB/s）存 CSR 图数据/顶点特征/权重/结构编码 → TDL 的 MAPE 6 级流水取数 → TCU（FUU+MOU）构造 TDS → Task Dispatcher 合并派发 → GTPU（16 FAU）流式注意力 + SCU 块级异步 softmax → 结果写回 HBM 或缓存片上供下一层。U280 的 460GB/s HBM2 与 FlowGNN/MEGA/BingoGCN baseline 的 off-chip 配置一致（Table V），保证对比公平；TAGT 通过 TDS 稀疏化+紧凑 TDS-CSR+去重/合并把 off-chip 流量降 42.1%–81.6%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 通用：Alveo 卡用 Xilinx Vitis/Vivado 开发，RTL 或 HLS 设计经综合实现生成 bitstream 加载，主机经 PCIe/OpenCL 或 XDMA 驱动数据；U280 双 HBM2 stack 共 8GB、460GB/s（每 stack 2 个 pseudo-channel 组）。
- TAGT 用法：Vivado 2019.1 综合 RTL @280MHz，直接测量端到端执行时间/带宽利用/功耗（对比 CPU RAPL、GPU nvidia-smi 能耗），输出相对 DGL-CPU/TorchGT/FlowGNN/MEGA/BingoGCN 的加速比与节能。

涉及论文标题：
- TAGT: An Efficient Graph Transformer Accelerator with Topology-aware Sparsification and Merging
