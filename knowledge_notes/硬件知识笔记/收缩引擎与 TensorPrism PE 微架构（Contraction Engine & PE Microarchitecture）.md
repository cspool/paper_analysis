## 收缩引擎与 TensorPrism PE 微架构（Contraction Engine & PE Microarchitecture）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TensorPrism 由 16 个 PE（4×4 mesh 互连）组成，每 PE 实现稀疏张量收缩流水线，含 8 组取数单元（fetch unit）、收缩引擎（contraction engine）、中间结果缓冲（IR buffer）与提交单元（commit unit）。Fetch unit 带 fetch sequencer 把分区索引转成显式地址流；稠密张量按取数单元分片（每单元拥有不相交切片缓存），跨单元经 ring 转发请求（缩小网络直径）。收缩引擎由 feed unit、寄存器堆（单端口 SRAM + 小 cache）、算子控制器（operation controller）、MAC 单元组成：每 MAC unit 8 个 MAC、每引擎 64 个、每 PE 512 个 FP32 MAC（16 PE 共 8K MAC @650MHz）；8 个 MAC 共享 feed unit 的稀疏输入（广播复用）、寄存器堆供不同 32 FP32 稠密输入；feed unit 连续周期重发数据增强时间复用、多累加器保存不同部分和；Operation Controller 经 DMUX 把局部 buffer 条目重分配给欠利用 MAC（运行时负载均衡）。Commit unit + 多维地址生成器（MAG）把张量索引映射物理地址写回。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
执行流程（一个分区 push 数据流）：fetch unit 从 GLB 取共现图条目+稠密行 B[K,:]（分片缓存、ring 跨单元转发）→ feed unit 广播稀疏输入 A[I,J,K] 给 8 个 MAC → 寄存器堆供稠密输入向量 → MAC 标量-向量乘+向量累加（多累加器存不同部分和 C[I,J,:]）→ commit unit + MAG 把输出坐标映射物理地址写回 GLB。微架构设计动机：稀疏张量收缩是非零驱动的标量-向量乘而非稠密 GEMM；广播+重发实现稠密行最高 128× 复用/取数（消除 SPADE 重复取数）；多累加器支持输出不冲突并行累加（消除 GSpTC 归约串行化）。面积：PE 内收缩引擎占 86.2%（本地 buffer/寄存器 77.4%、MAC 15.1%），印证"存储支撑复用"设计。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Verilog RTL（TSMC 28nm + Synopsys Design Compiler 综合、PrimeTime PX 功耗、CACTI 7.0 SRAM 建模），4×4 mesh 可扩展更大 PE 阵列；每 PE 48KB 局部存储。使用场景：8 个 FROSTT 数据集 + LLaMA 注意力张量收缩；8 组引擎并行处理多分区、DMUX 运行时均衡补足 CoGTP 静态均衡。性能：2.22×/2.40×/1.71×/1.76×/1.49×（vs SPADE/HotTiles/GSpTC/TCP/HyperSB）。

涉及论文标题：
- TensorPrism: Rethinking Sparse High-order Tensor Acceleration via Co-occurrence Graph
