## DRAMSim3（cycle-accurate DRAM 时序模拟器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DRAMSim3 是马里兰大学（umd-memsys）的开源逐周期 DRAM 模拟器（C++），模拟内存控制器、channel、rank、bank 层次的状态机与全部 DRAM 时序约束（tRCD/tRP/tCL/tRAS/tCCDL 等），支持 DDR4/DDR5/HBM/GDDR 等多种协议配置（含热建模），输入为配置文件（ini）+ 访存 trace 或 CPU 前端耦合。开源：https://github.com/umd-memsys/DRAMSim3。CHIME 以其为基础构建 CHIME-PIM-sim：新增 PIM 命令（PIM_WR_R/PIM_LD_SB/PIM_WR_SB/PIM_MAC/PIM_RD_RB）、对应时序约束与 FIFO 调度方式，做 trace-driven 逐周期模拟。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
CHIME-PIM-sim 的推演流：PIM 命令流按 DDR 时序注入 → 逐 cycle 推进各 bank 状态机（ACT/PRE/RD/WR 与 bank PU MAC 的重叠）→ 统计 bank PU 忙闲、行切换、跨 chip 传输与 rank PU 处理时间 → 输出 attention 延迟与流水气泡（Fig.17 时间线显示 GPU/PIM 两侧流水、气泡极小）。DDR4-3200 配置：2 Ranks(8 chips)×4 BG×4 Banks，BL=4:CCD=4:RRD=4/8:RCD=22:RAS=52:RP=22:RC=74:CL=22:WL=16:CDLR=4/12:WR=24:CCDL=8:RTP=12。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：经典的内存控制器模型（刷新、bank 冲突、读写转换）+ 可插协议；被大量 PIM 工作复用（UniNDP、FlexQ-NDP 等都以 DRAMSim3 类时序建模 PIM）。使用方式：评估新 DRAM 架构/命令扩展前先在 DRAMSim3 上建模时序与状态机；CHIME 表明扩展点 = 命令集 + 时序约束 + 调度器（FIFO），而 GPU 侧用 roofline 近似（AttAcc）即可，无需全细节 GPU 仿真。

FlashTFHE 补充视角（ISCA'26，HBM2E 两栈作为 TFHE 加速器片外内存）：FlashTFHE 用 DRAMSim3 模拟两栈 HBM2E（合计 819GB/s）承载 BSK/KSK 的流式读取与 GLWE/LWE 密文流；由于 BSK chunk 在片内 0.8MB GGSW buffer 中跨 batch 复用，DRAM 只承担"每 chunk 一次"的带宽，两栈 HBM2E 可支撑 8 core（Figure 15a：核数 2→8 时 GLWE/LWE 带宽线性增长、BSK/KSK 因核间共享而恒定）；顺序访问模式用 16KB read/store FIFO 队列隐藏 DRAM 延迟；accumulator buffer 缩到 9216KB 以下时数据换出到 DRAM 会停顿 BRU 流水线（Figure 16），故 9.2MB 是临界容量。

NS-FPS 补充视角（ISCA'26，点云 FPS 加速器片外内存建模）：NS-FPS 用 DRAMsim3 + Micron 8GB DDR4-2400 参数建模其 ASIC 的片外访存行为——Page Memory（Morton 点云页存储）与 Point Buffer 位于 DRAM，邻居搜索每轮只读搜索球相关 cube（16 点/页、匹配 DDR4 64B burst），经 32 项片上 Morton Cube Buffer 命中复用；DRAMsim3 按 DDR4-2400 时序对访存请求排队给出周期级行为。结果：120k 点帧内存访问较 GPU 实现降 1700×、较 QuickFPS 降 8.4–13.4×，且 DRAM 功耗 2066.69mW 占总功耗 82.7%——论文以该占比论证"FPS 是内存受限、必须停止多余 DRAM 请求而非加 ALU"的硬件设计取向。

涉及论文标题：
- CHIME: A Case for Efficient Long-Context Attention-FC Disaggregated Inference with DIMM-PIM
- FlashTFHE: A Scalable Architecture for Efficient Multi-bit Fully Homomorphic Encryption
- NS-FPS: Accelerating Farthest Point Sampling via Neighbor Search in Large-Scale Point Clouds
