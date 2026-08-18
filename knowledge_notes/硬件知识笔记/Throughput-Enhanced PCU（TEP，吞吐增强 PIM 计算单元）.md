## Throughput-Enhanced PCU（TEP，吞吐增强 PIM 计算单元）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Throughput-Enhanced PCU（TEP）是 P3-LLM（ISCA 2026）在低精度 PCU 之上提出的时间维吞吐增强技术。关键观察：低精度 PCU（6-bit 定点乘法器替代 FP16 乘法器、去掉指数对齐与高精度累加）面积大幅缩小，从而允许更高的工作频率——P3-LLM 的 PCU 可稳定跑到 HBM-PIM FP32 累加版本 2× 的频率。HBM-PIM 以 tCCD_L（4 个内存总线周期）为 PIM 命令节奏，而 P3-LLM 可以 tCCD_S（各代 HBM 均约为 tCCD_L 的一半）运行。由此同一 256-bit 权重切片（行缓冲中）可以在一个 tCCD_L 窗口内被两个不同输入各消费一次（temporal input reuse），等效把 GEMV 吞吐再翻倍而无需增加任何乘法器——这是"空间扩展（加乘法器受 DRAM 面积限制）不可行"时的替代路线。论文对比过空间方案：把激活 tile 从 1×4 扩到 n×4 会增加 n× 乘法器面积；减小权重 tile 到 16/n 又因 256-bit 列访问粒度导致 PIM 带宽欠利用；TEP 则完全避免面积代价。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
TEP 的时序工作流（Fig.7 命令时序对比）：
```
HBM-PIM（tCCD_L 节奏）：           P3-LLM/TEP（tCCD_S 节奏）：
NPU 发输入到 PCU 寄存器             NPU 发输入1、输入2 到 PCU 寄存器
每 tCCD_L 发起一次 PIM 命令          每 tCCD_S 发起一次 PIM 命令
└ 读 256-bit 权重 → 1×1×16 GEMV     └ 读 256-bit 权重 → 1×4×16 GEMV
                                    └ 同一权重切片被输入1、输入2 先后复用
                                    （两个 tCCD_S 窗口 = 一个 tCCD_L）
```
效果：batch=2 时单个 tCCD_L 窗口内处理两个输入向量，吞吐翻倍；每 PCU 64 乘法器（4× HBM-PIM）× 频率 2×（TEP）= 相对 HBM-PIM 总 roofline 8×。能耗：TEP 以 tCCD_S 运行使 PIM 功耗 +28%（主要来自 DRAM cell 访问与列译码切换，这些不随权重复用而变化），但每次内存访问被复用两次、减少重复 DRAM row activation，整体能量效率 1.56× 更好。batch=2 是 P3-LLM 相对各 baseline 速度提升最大的点（7.8× vs NPU）；GQA 的共享 KV（Llama-3.1-8B group=4、Llama-3.2-3B group=3）在 batch 2-64 下仍受益于 TEP 的 temporal 复用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：作为 PCU 微架构的一部分在 SystemVerilog RTL 中实现，Synopsys Design Compiler + TSMC 28nm 综合验证频率提升，cycle-level simulator（PIM 部分按 Newton 方法论扩展）评估吞吐与能耗；PCU 运行频率设为 500 MHz（=1GHz/2，对应 tCCD_S=2 个 DRAM 时钟），NPU 1 GHz。使用场景：任何低精度 PIM/GEMV 加速器（HBM-PIM、GDDR、LPDDR-PIM——论文指出 LPDDR 的 tCCD_S 同样为 tCCD_L 的一半），用于提升低 batch 解码与 GQA 场景下 PIM 的计算吞吐，无需增加乘法器面积。开源状态：RTL 与模拟器未开源（开源仓库 https://github.com/yc2367/P3-LLM 仅含量化算法代码）。

涉及论文标题：
- P3-LLM An Integrated NPU-PIM Accelerator for Edge LLM Inference Using Hybrid Numerical Formats
