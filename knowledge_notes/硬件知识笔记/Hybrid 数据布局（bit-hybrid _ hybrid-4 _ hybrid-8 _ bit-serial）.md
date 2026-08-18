## Hybrid 数据布局（bit-hybrid / hybrid-4 / hybrid-8 / bit-serial）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hybrid 数据布局（bit-hybrid）是 in-SRAM 计算阵列中决定"一个数据元素如何在 wordline/bitline 上排布"的存储组织方式，直接影响计算 SRAM 的利用率与吞吐。三种代表性布局：(1) **bit-serial**（Neural Cache [6]、Duality Cache [11]）：一个数据元素的多 bit 沿多条 wordline 按位展开，位并行吞吐高但执行复杂算术延迟大；(2) **bit-parallel**（Compute Cache [1]）：整元素沿 wordline 展开，延迟低但吞吐受限；(3) **bit-hybrid**（EVE [2]）：把元素拆成 n-bit 段，段内 bit-parallel、段间 bit-serial，平衡吞吐与延迟。PipeIMC 采用 EVE 风格 hybrid 布局的两种变体：**hybrid-4**（SIMT-EVE 与 PipeIMC-scoreboard 配置用，每线程 32 GPR，4 条 bitline 存储一个线程的通用寄存器、每条 GPR 沿 4 条 wordline）与 **hybrid-8**（PipeIMC 配置用，每线程 64 GPR 以支持寄存器重命名，8 条 bitline 存一个线程的 GPR、每条 GPR 沿 4 条 wordline；论文 Fig.4(c) 展示线程与 warp 到计算 SRAM 阵列的映射）。hybrid-8 的 8-bit 计算外围电路由 EVE 的 1-bit 外围电路 8 组跨 8 bitline 组成，每 bitline 一套 8-bit 计算外围电路执行 in-SRAM 操作。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
布局决定微码周期：基于 8-bit-hybrid 布局，计算 SRAM 的加法/逻辑操作 8 cycles（奇数周期多行访问、偶数周期写），乘法 32 次移位+加法迭代（105–634 cycles），除法 145–1174 cycles。bit-serial（Duality Cache）在 256×256 阵列上的周期不同（论文未给出其微码周期，重实现时保留其原始布局）。数据到达阵列前还需 data transpose unit 把低层内存层次的数据重排为 hybrid-8 布局（每 transpose unit 256B，每个 memory phase 拆成多个 8-thread 段）。布局与线程/寄存器数的耦合：hybrid-4 支持 32 线程×32 GPR/线程的满阵列利用，hybrid-8 因每线程寄存器翻倍（64）而减少并发线程数（PipeIMC 每执行单元 32 线程 vs SIMT-EVE 64 线程），但换得重命名能力。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：bitline 组织决定外围电路宽度——hybrid-8 用 8-bit 外围电路（8 组 1-bit logic/add/shift/writeback 四层外围电路），hybrid-4 用 4-bit 外围电路；数据写回时由 writeback 层选择结果写回对应 wordline。使用：EVE 用 hybrid-4 发现计算 SRAM 阵列利用不足（先前架构数据布局问题），MagiCache 进一步细粒度 cacheline 级管理；PipeIMC 为支持重命名而把 GPR 数翻倍（hybrid-8）。评估中 SIMT-EVE 用 hybrid-4、Duality Cache 保留 bit-serial、PipeIMC 用 hybrid-8，是各架构横向对比的数据布局前提。Vault 无专门笔记证据。

涉及论文标题：
- PipeIMC a Pipelined In-SRAM Computing Architecture
