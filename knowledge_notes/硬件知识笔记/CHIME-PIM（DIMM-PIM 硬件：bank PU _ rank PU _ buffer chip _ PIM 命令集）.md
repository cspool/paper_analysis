## CHIME-PIM（DIMM-PIM 硬件：bank PU / rank PU / buffer chip / PIM 命令集）

术语解释
CHIME（ISCA'26）提出的 DIMM-PIM 硬件：在 DDR4 DIMM 的 DRAM chips 内集成 bank 级 PU、在 buffer chip 上集成 rank 级 PU，两者异步协作执行 attention，通过标准 DDR 接口通信且不改主机内存控制器。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CHIME-PIM 由两级处理单元组成：(1) bank PU——以 DRAM 工艺集成在 bank 旁，从 bank 读取 KV cache 并与 shared buffer 广播的输入向量做 MAC（score 与 context 计算），结果写各 bank 的 result buffer；全部 DRAM chips 的 bank PU 并发执行。(2) rank PU——位于 buffer chip 上、以逻辑工艺实现，含 softmax 单元、adder 单元与 re-layout 单元，与 bank PU 异步运行（bank PU 算 MAC 的同时 rank PU 做 softmax/re-layout）。控制路径：CPU 以正常写把 PIM 请求发给 rank PU 译码为 PIM 命令，专用 PIM 控制器经标准 DDR 接口向 DRAM chips 发命令——无需修改主机内存控制器。PIM 命令集：PIM_WR_R（写寄存器/配置：加法树或累加器范式、buffer 自增索引等）、PIM_LD_SB（bank→shared buffer）、PIM_WR_SB（rank PU→shared buffer）、PIM_MAC（全 bank MAC，同时读 DRAM cell 与 shared buffer）、PIM_RD_RB（result buffer→rank PU）。对比：rank PU ≈ UPMEM DPU 式 rank 级近存（约 4× 主机带宽，CHIME 的 R-PIM 基线 2TB+1.6TB/s），bank PU ≈ Newton/AiM/HBM-PIM 式 bank 级存内（>30×，CHIME 达 2TB+13.0TB/s）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
decode 一个 head 的硬件运转流：CPU 写 PIM 请求 → rank PU 译码 → PIM_WR_R 配置寄存器 → PIM_WR_SB 把 Q 写入 shared buffer（Q 经 shared buffer 广播到所有 bank PU）→ PIM_MAC：各 bank 的 bank PU 从 DRAM cell 读 K 与 shared buffer 的 Q 做 MAC，score 按 chunk 写 result buffer → PIM_RD_RB：rank PU 经外部总线取回 O^s → adder 累加 → softmax 单元 per-chunk softmax → 跨 chunk 归一化出 S → S 写回 DRAM → bank PU 执行 S×V（context）。关键设计：内部总线（bank PU↔bank）与外部总线（rank PU↔bank PU）解耦使传输与 MAC 并行（bubble-free pipelining 的硬件基础）；re-layout 单元在 offload/onload 传输中做布局变换（hybrid-grained re-layout）。面积（TSMC 28nm 综合，复用命令译码与数据通路）：单 bank PU 0.0032 mm²、shared buffer 0.051 mm²，换算 1z-nm DRAM 节点约翻倍；能量：CHIME 总能耗较 GPU 基线 -40%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：bank PU 集成进 DRAM die（DRAM 工艺，代价是逻辑密度与面积惩罚，可被 DIMM 大容量摊薄、3D 集成缓解）；buffer chip 承载 rank PU（类似 RCD/LRDIMM 缓冲芯片的扩展）；PIM 命令扩展标准 DDR 指令序列。评估用 CHIME-PIM-sim（修改版 DRAMSim3 加 PIM 命令/时序/FIFO 调度）+ AttAcc（GPU 侧），配置 DDR4-3200、2 Ranks(8 chips)×4 BG×4 Banks、2TB、13.0TB/s。使用方式：作为 AFD 系统的 attention 加速器（存 KV cache + 算 decoding attention）；bank PU 的计算访存比 N_cmr 需按 GQA-n 配置（N_cmr=n）以满带宽利用；支持 MHA 与 GQA（head 映射 N_hc=8/1）。

涉及论文标题：
- CHIME: A Case for Efficient Long-Context Attention-FC Disaggregated Inference with DIMM-PIM
