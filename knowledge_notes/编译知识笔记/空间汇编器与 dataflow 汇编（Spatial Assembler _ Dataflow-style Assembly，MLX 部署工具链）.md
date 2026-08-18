## 空间汇编器与 dataflow 汇编（Spatial Assembler / Dataflow-style Assembly，MLX 部署工具链）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 空间汇编器（spatial assembler）是 MLX 软件部署工具链的汇编环节：把"指定每个 PE 操作的 dataflow 风格汇编文本"（dataflow-style assembly，程序员手写）或 LLVM-based C 编译器（引 DACO，参考文献[35]）生成的中间表示，编译为二进制，并导出为 header 文件供 RISC-V host 配置 MLX 空间阵列。整体软件栈（论文 A 节 Software/Hardware Implementation）：RISC-V CPU 作 host controller → 开发者写 dataflow 式汇编（或 C）→ 轻量 spatial assembler 汇编成二进制 bitstream → 作为 header 嵌入 C 程序 → host 发紧凑命令给空间阵列（仅需最小 ISA 扩展加载 MLX 配置与协调内存移动）。
- 与通用 GPU 编译栈（如 ptxas/SASS）的区别：MLX 的"二进制"是 PE 级 tagged-block 配置（每 PE 的 LD-COMP-XFER 指令序列 + loop trip count + 路由类参数），而非 SIMT warp 指令流；编译器（LLVM-based + assembler）负责把算子数据流图划分为 CDC、生成层内静态指令序列，硬件只做 tag 级跨层协调（hybridized scheduling）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
MLX 部署工具链运转流程（一次 BSMM kernel 的编译-部署）：
```
# 1) 编程：写 dataflow 式汇编，指定每 PE 操作
#    或：C 代码 → LLVM-based C compiler（DACO 式，挖掘 dataflow 机会）
# 2) 空间汇编器：汇编文本 → 二进制（含 CDC/tagged-block 编码、CDC-to-PE 静态放置、
#    (Δx,Δy) 路由类、残差 hop 数、loop trip count）
# 3) 导出为 header 文件 → 嵌入 C 程序
# 4) RISC-V host controller 加载配置（最小 ISA 扩展）、发紧凑命令、协调内存移动
# 5) 空间阵列按 tagged block 执行（层内静态、跨层 tag 级弹性）
```
作用：把"指定每个 PE 操作的文本格式"变成可配置位流，使空间加速器 bitstream 可嵌入常规 C 程序（软件部署），host 侧控制面保持与现有 RISC-V 软件兼容。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现与使用：论文未开源该工具链（spatial assembler、LLVM-based 编译器均未提供仓库）；实际使用方式是"程序员写 dataflow 式汇编 + assembler 编译 + header 导出"，或 C 经 LLVM-based 编译器生成。这是 MLX 架构的软件部署路径——与 M100（orchestrated dataflow）的 NPU 固件 JIT、与 CODO 的 HLS dataflow 编译等空间架构工具链同属"把数据流映射到空间阵列"的编译框架类别，但 MLX 的编译产物是 tagged-block 配置（层粒度）而非任务级 FIFO 流水或 tensor 指令流。局限：论文未给出 assembler 的指令集细节与编译器优化pass（论文未明确说明）。
- 涉及论文标题：MLX: Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures
