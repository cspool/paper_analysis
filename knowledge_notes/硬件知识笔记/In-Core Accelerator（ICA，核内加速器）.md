## In-Core Accelerator（ICA，核内加速器）

术语解释
集成在 CPU 核流水线内的加速器抽象：从核架构寄存器读输入、把输出写回架构寄存器，调用无状态（执行后不留内部状态）。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ICA 是"加速器作为高级功能单元"的形式化描述。代表：Intel TMUL（矩阵乘法，由 AMX 指令控制）、各厂商 AVX/ARM Neon/SVE 向量单元，以及研究性方案（如 SPADE 的 tile ISA 计算阵列）。因为是寄存器↔寄存器接口且无状态，ICA 指令可以和普通指令一样乱序发出、推测执行：错预测产生的错误数据仍在未提交寄存器里，用现成寄存器重命名/回滚机制即可撤销，不会在加速器内留下错误状态（这正是 Intel AVX/AMX 能推测执行的原因）。代价：ICA 取数完全依赖核的通用访存硬件（load-store queue、预取器），内存级并行（MLP）受限——论文实测：SDDMM 上"完美 ICA"（计算零延迟）仅把总周期降低 25%，因为 62%+ 的周期仍是 memory-bound，核无法以加速器所需速率供数。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
以 AMX TMUL 为例（论文 baseline 之一）：核执行 `LDTILECFG` 配置 tile 寄存器 → 用普通 load 把 A/B 矩阵子块经 LSQ/L1 装入 `tmm0..tmm7` 1KB tile 寄存器 → `TDPBF16PS/TDPBSSD` 类指令在核内矩阵单元完成 C+=A×B → 结果留在 tile 寄存器供后续指令消费或 store 回内存。瓶颈链路：每个待算 tile 都必须由核的 load 指令逐条拉入，在访存队列、MSHR、ROB 容量与预取器精度约束下，MLP 上限远低于专用加速器的需求（论文把 SPR 的 L2 MSHR 放大到 128 来专门消除这一限制）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
落地形态：向量执行单元（SIMD，如 AVX-512 的 512-bit 通路）、矩阵执行单元（AMX TMUL，8 个 1KB tile 寄存器 + 16×16×32 tile 乘加，Sapphire Rapids 起出货）、或更专用的 PIM 风格功能单元。编程接口是 ISA 扩展 + 编译器 intrinsic（`_tile_loadd`、`_tile_dpbssd` 等），状态经 XSAVE（XTILECFG/XTILEDATA，共约 8KB 架构状态）支持上下文切换。适用场景：输入大部分在寄存器/L1 中、与核指令流细粒度交错的短任务；不适合内存驻留的大输入量加速（MLP 受限）。

涉及论文标题：
- ATX: Accelerator Task Extensions
