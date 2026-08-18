## SCALE-Sim（systolic array cycle-level 模拟器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SCALE-Sim（Systolic CNN AcceLErator Simulator，ARM Research，github.com/ARM-software/SCALE-Sim）是开源的 cycle-accurate systolic array 加速器模拟器（TPU 风格）：输入 accelerator 配置（阵列高宽、IFM/Filter/OFM SRAM 大小、dataflow：weight/input/output stationary）与 layer 配置（GEMM/卷积 shape），逐周期推演 PE 计算、SRAM 读写与 DRAM 传输，输出运行周期、SRAM 利用率、DRAM 流量与带宽需求；可与 CACTI/DRAMPower/Accelergy 联用估算面积与功耗。新版 SCALE-Sim v3（github.com/scalesim-project/SCALE-Sim）增加多核、稀疏格式（CSR/CSC/blocked ELLPACK/N:M）、Ramulator DRAM 集成与自定义存储布局。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
config/accelerator.yaml: ArrayHeight/Width, IFM/Filter/OFM SRAM, dataflow
config/layer.yaml:       M/N/K（或 conv 参数）
→ 逐 cycle 推演：PE 阵列计算 → SRAM 层次读写 → DRAM 突发传输
→ 输出：cycles、utilization、DRAM bw/traffic
```
本文用法：Cassandra 扩展 SCALE-Sim 与 LPU simulator 构建自研 NPU cycle-level 模拟器（64 TFLOPS MAC、273 GB/s、128GB LPDDR5X、9MB scratchpad），在其中实现 DMA 内 encoder/decoder 与 superblock 数据通路，评估 NPU 上 Cassandra-1/2 的吞吐增益（图 12 中 Systolic Array NPU 组，1.78–2.41× 范围内）；模拟器输出每 token 周期数与实测接受率结合换算总吞吐。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
pip install scalesim，Python API 或 CLI 运行；用途：加速器 DSE（RTL 之前探索阵列/存储/数据流配置）。局限：内存时序抽象较粗——本文因此另扩展 LPU simulator 补足 DMA/scratchpad 行为；SCALE-Sim 本身不含 LLM decode 的完整软件栈（需自行构造 GEMM/attention 层序列）。与 Accel-Sim 分工：后者模拟 GPU 侧集成，SCALE-Sim 覆盖 NPU 侧。

涉及论文标题：
- Cassandra: Enabling Reasoning LLMs at Edge via Self-Speculative Decoding
