## 分层 ISA 与自动指令翻译（Row-Level / Packet-Level ISA、Path Generation）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
分层 ISA 把一条用户级指令（Row-Level：SIMD 语义、DRAM bank 粒度）静态降级（lowering）为一组硬件执行级指令（Packet-Level：MIMD 语义、NoC router 粒度），解决异构计算单元执行模型不一致的编程问题：DRAM-PIM 是 SIMD 集中控制（所有 bank 共享指令上下文），SRAM-PIM/NoC router 是 MIMD 分布式执行（每 bank 私有上下文）。CompAir 的 Row-Level ISA 共 7 条：NoC_Scalar/Access/BCast/Reduce/Exchange（NoC 标量计算与集合通信）+ SRAM_Write/Comp（权重写入与矩阵乘）；Packet-Level ISA 每条为 Type 4b / Data 16b BF16 / IterNum 4b / Path[0..3]×12b（Path 内含 X/Y 坐标、WrReg、IterTag、Opcode）。设计取舍：行级是用户编程接口（保守地"DRAM row→Curry ALU→DRAM row"数据流），包级才是 NoC 指令缓冲实际存的内容。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
编译期翻译流程（host 侧编译器、离线执行）：① 用户写 Row-Level ISA 程序，如 NoC_Reduce('+', Src, Dst, Mask, DstBank)；② 静态 row→packet 降级：NoC_Reduce 按固定树模板 + bank id 实例化为每个 bank 的归约 packet 序列；③ path generation（借鉴算子融合 [DNNfusion 等]）：把连续 NoC_Scalar 按生产者-消费者链（DST→SRC）融合成单 packet，一次携带全部计算与路由，规避行级 ISA 每步写回 DRAM 的低效（33–50% 延迟优化）。执行期数据变换：router 按数据粒度与位宽自动把 DRAM 行串行化为多 flit packet、算完自动反串行化写回（软件透明、跨 packet 自动流水）。效果：Qwen 8K input-split 下 +27% 本地指令稀释为系统级 +2%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：host 运行时把编译产物填每个 bank 的指令缓冲（离线降级，非 JIT）；path generation = 指令级 DAG 融合（类比 Triton/TVM 算子融合，但对象是 NoC 路径而非 GPU kernel）。使用方式：编程者只见 SIMD 行级语义（Mask/Offset/Group 等参数），MIMD 细节由翻译器生成；避免给每 bank 私有控制器（prior work [82] 的 MIMD 扩展方案面积 +17–20% 逻辑 die）。开源：https://github.com/Man0xbfc00380/comp-air.git（translate/ 模块）。

涉及论文标题：
- Bridging Efficiency and Scalability in LLM System via 3D Hybrid PIM with 2D In-Transit Computation
