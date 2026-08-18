## GTPU / FAU（Graph Transformer Processing Unit / Fast Attention Unit，含 UPE / VPE 数据通路）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- GTPU（Graph Transformer Processing Unit）是 TAGT 加速器（ISCA 2026，Xilinx Alveo U280 FPGA）内专用于图 Transformer 注意力的处理单元，由一组 Fast Attention Unit（FAU）组成（TAGT 配置 16 个 FAU）。每个 FAU 集成两类处理元件：UPE（Update Processing Element，用于 MM/矩阵-矩阵运算：FFN 与原始注意力分数 QK^T 计算）与 VPE（Vector Processing Element，用于 VM/向量-矩阵运算：softmax 归一化后的 value 加权聚合）。注意：此 VPE 与 NASZIP 的 VPE（Vector Process Engine，DIMM 内近存向量引擎）同名不同义。
- 执行模型：UPE 计算目标顶点与其 TDS 关联顶点的原始注意力分数，流式送 SCU（Specific Computing Unit）做块级 softmax 归一化，SCU 产出归一化部分贡献，VPE 消费并聚合得到目标顶点最终嵌入——分数不物化为中间注意力矩阵。UPE 内 Private Registers (PRs) 做本地部分和累加；可配置 Switch 在 FFN 阶段把 VPE 的 MAC 资源并入 UPE 形成更大 MM 引擎（Reconfigured MM 模式），提高 attention/FFN 双阶段硬件利用率。Task Dispatcher 把共享 Associated ID 的多个目标任务打包成单一请求，使一个共享关联特征一次 dispatch 服务多次注意力计算。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- FAU 内注意力流水：① UPE datapath——从 TDS-CSR 关联列表取目标 v 与其关联顶点特征，算 h_v W_Q 与 H^v W_K 的 MM 得原始分数 S，写入 PR 并流式送往 SCU；② SCU——块级异步 softmax（BPM 分段 → ECU 并行算局部分子/分母 → 64 CRU 按 TID 异步累加 → Divider Pool 归一化）返回归一化部分贡献；③ VPE datapath——把归一化贡献与 V 向量做 VM 聚合得 h̄_v；④ FFN 阶段 Switch 切换为 Reconfigured MM 模式（UPE+VPE MAC 合并）跑 FFN；⑤ 结果写回 HBM 或缓存到 Structure and Feature Buffer 供下一层。
- 并行性来源：16 个 FAU 并行处理多个目标顶点子任务；跨 partition 的注意力分数结果存 Partial Buffer、目标顶点特征留 Feature Buffer 防重复取数；ping-pong buffering 隐藏访存延迟。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：RTL 综合到 Alveo U280（Vivado 2019.1，280MHz）。TAGT 资源规模（Table V）：16 FAU（每 FAU 256 UPE + 128 VPE）、8 FUU、8 MOU、1 SCU、共 4,096 MACs；片上 Feature 1MB + Weight 1MB + Partial 512KB + TDS-CSR 512KB + Task FIFO 128KB + Output 128KB。资源利用率（Table IV）：DSP 73.6%–80.2%、LUT 40.1%–49.5%、BRAM 59.3%–69.7%、UltraRAM 80.3%–89.7%。
- 效果：相对 DGL-CPU 175.4×、TorchGT 18.6×、GNN 加速器 FlowGNN/MEGA/BingoGCN 8.2×/6.9×/4.7× 加速；off-chip 流量降 42.1%–81.6%；w/o TBFA 消融 2.48×（贡献 28.62% 总收益）。

涉及论文标题：
- TAGT: An Efficient Graph Transformer Accelerator with Topology-aware Sparsification and Merging
