## Unified Buffer（UB）与 AI Core 多级片上缓冲

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Ascend AI Core 的片上存储层次：全局内存 GM（HBM，GB 级、高延迟）→ L2（跨核共享）→ 每核本地缓冲：Unified Buffer（UB，向量/标量单元的输入输出缓冲区，910B2 约 192KB）、L1（Cube 数据复用的大缓冲）、L0A/L0B（Cube 指令输入，fractal 格式）、L0C（Cube 输出）、BT（bias 表）/FP（FixPipe，量化/ReLU 参数）。AIV 运算的源/目标数据必须驻留 UB（32 字节对齐）；数据在层次间移动由 MTE 完成。ENEC 的块大小选择直接受 UB 约束：压缩块取 16384 元素（16K），因为 32768 元素的内存足迹超过 UB 的 192KB 限制；每线程还用 32KB 打包 buffer（8192 个 32-bit lane 对应 8192 元素）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
ENEC 压缩在 UB 中的运转：
```
# 每 AIV 线程
CopyIn: HBM → UB（16384 个 BF16 元素，受 192KB UB 上限约束 → 块大小上限）
Compute: 在 UB 内做拆分/变换/打包；32KB 打包 buffer 累积 lane
if lane 满 16 位: 输出低 16 位 + bit mask 到流, 右移继续    # UB 内就地循环
CopyOut: UB → HBM 压缩流
```
Annotations：UB 容量决定单线程可同时处理的数据量（tiling 粒度）；ENEC 用 32KB 专用 buffer 做位累积，避免频繁搬移；32 字节对齐要求是 IDD-Scan 与 L≥16 组长的直接原因（half 每行 16 个=32B）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：片上 SRAM 缓冲，AscendC 的 queue/pipe 管理其分配与双缓冲重叠（CopyIn 下一块与 Compute 当前块重叠）；UB 大小因芯片而异（910B2 约 192KB，其他代/型号 512KB-1MB 量级）。使用：所有 AIV 算子的临时数据必须进 UB，因此算子设计（含压缩 kernel）都以"切块塞进 UB + 流水重叠"为核心；ENEC 论文在消融中验证 16K 块吞吐最优、32K 因超 UB 退化。

涉及论文标题：
- ENEC: A Lossless AI Model Compression Method Enabling Fast Inference on Ascend NPUs
