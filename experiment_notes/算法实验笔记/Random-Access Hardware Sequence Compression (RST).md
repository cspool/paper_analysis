## Random-Access Hardware Sequence Compression (RST)

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现是 RST（Randomly-decompressible Sequence Compression with Top-utility Selection）——一种面向硬件内存压缩的新序列压缩算法，让压缩页支持"随机解压"（random access）单个 64B 块，同时保持与 state-of-the-art 页级序列压缩（LZ 家族）相当的压缩率。核心是 top-utility 序列选择：对页内所有可能的 2~5 符号序列（4096B 页最多 8×10^6 种唯一序列，见 Fig.7：4095 条 2-symbol、4094 条 3-symbol…）计算 utility（utility = 该序列若入选字典可带来的总空间节省 / 该序列消耗的字典空间），迭代式地每轮选出 utility 最高的序列填入显式序列字典（每轮选中后先对所有出现做替换 substitution，再对与选中序列重叠的序列重算 utility 的 utility update，直到字典满 128B 或无正 utility 序列，Algorithm 1）。字典总开销压到每页 128B，由三部分组成：A) 显式序列字典（本身再做序列压缩，Fig.5 布局：序列数+长度数组 L1..Ln+序列背靠背存储）；B) 位置元数据 48B→8B（每 DRAM 块内嵌 6-bit 偏移定位块内首个逻辑块 + 页级 64-bit 逐块 0/1 向量）；C) 符号字典→0B（静态符号字典：零最短码、reuse codes 编码"重复 i 位置前的符号"，Table 1 码长，最坏 12.5% 字面量开销）。输出 (D, page') = 显式序列字典 + 静态符号编码后的压缩页。
  - 实验比较：压缩率上对比 ASIC Deflate（TMCC [24] 的页级 Deflate = LZ77+Huffman）在 4KB 页粒度、同一 88 个 benchmark 与同一方法论（忽略内存 dump 中全零页）下：RST 几何平均 3.4× vs ASIC Deflate 3.3×（软件 zlib Deflate 3.84× 作参考；~10–12% 硬件/软件差距与先验观察一致，Fig.18）。Fig.19 消融显示 top-utility 选择贡献最大；128B 字典为性价比点（256B 仅小幅提升，但每次访问需多取字典，per-access 流量 192B→320B +67%）。解压延迟：18ns vs TMCC 140ns（半页），还对比 OCP Zipline ~2µs、IBM ~1µs、CDPU ~1µs（Table 3）。

- 硬件平台是什么，配置是什么。
  - 无真实芯片流片。RTL 用 Synopsys Design Compiler 在 ASAP 7nm PDK（[8]）下综合到 2.5GHz。综合的压缩器配置：32 SRAM banks（4 个子表 = 2/3/4/5 符号长度各一）、16 sets/bank、4-way 组相联；序列长度上限 5 符号（L=2 位长度字段）。面积/功耗/吞吐：压缩器 0.0923mm²、349mW 峰值、4.13GB/s；解压器 0.03mm²、91mW、13.3GB/s（trace-driven 活动向量测功耗）。系统级用 gem5+Ramulator 全系统仿真（Table 4：4 核 2.8GHz 4-wide OoO、ROB 224、1024 TLB 项、L1D/L1I 32KB、L2 256KB、L3 2MB/核共 8MB、1 通道 25.6GB/s、8 ranks、FR-FCFS、tCL/tRCD/tRP=13.75ns；atomic warmup 5s + detailed warmup 1ms + 详细仿真 4ms）。

- 模型是什么。数据集和bench分别是什么。
  - 非神经网络模型：RST 是压缩算法+专用硬件，无模型推理成分。数据集：从 88 个 benchmark 采样的内存 dump（忽略全零页），覆盖 7 类：数据库（Redis OSS v7.2、TPC、SPECjbb 2015）、GraphBig、PARSEC-3.0、SPEC CPU 2017、Spark Bench、DaCapo、Renaissance（§A.3.4，内存 dump 打包在 Zenodo artifact 中）。压缩率在 88 benchmark 上测（每类算几何平均再对 7 类取几何平均）；系统级仿真用 DyLeCT 的 12 个 application workload + 新增 1KB-strided 微基准 readStride-1K（低局部性压力测试，5.1× 加速但不计入平均）。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：GitHub https://github.com/HEAP-Lab-VT/rst（BSD 3-Clause Clear；含 README、visualizations/、LICENSE，完整 artifact 指向 Zenodo）；Zenodo artifact https://doi.org/10.5281/zenodo.19449274（rst-isca2026-artifact.tar.gz 4.3GB，含 C++ 参考实现、SystemVerilog RTL、QEMU VM 镜像、内存 dump、预生成综合报告与 synthesis/results/ 和 evidence/signoff_summary.md）。复现：E1 压缩率 `bash regenerate_figures.sh`（~50 分钟，生成 Fig.18/19 对应 PDF；--quick 冒烟测试 ~2 分钟）；E2 硬件验证 `python3 tools/run_rst_verify.py`（见硬件架构层）。
  - 算法 pipeline 伪代码（Algorithm 1，每页）：
    ```
    Input: page（4096B）
    U = COUNTSEQS2TO5(page)            // 统计所有 2~5 符号唯一序列出现次数进 utility 表
    D = {}                              // 空序列字典
    page' = page
    while HASSPACE(D) and HASPOSITIVEUTILITYSEQUENCE(U):
        s* = FINDTOPUTILITYSEQUENCE(U)          // 每长度子表取最高 count 算 utility，跨长度取最大（Fig.8b）
        dict_idx = AddToDictionary(D, s*)        // 显式存入字典
        substitution_sites[] = SUBSTITUTION(page', s*, dict_idx)   // CAM 匹配+替换+compaction（overlap filter 去重叠）
        UTILITYUPDATE(U, page', substitution_sites)   // splice-and-cancel 防双计数，更新受影响序列 count
    return (D, page')                   // 字典 + 压缩数据
    ```
    例子（页内 4 次 "XY"、多次 "YZ" 等，Fig.1/8/9/10）：COUNTSEQS2TO5 把 2-symbol（4095 条）、3-symbol（4094 条）…唯一序列按值分组记 count；第一轮 "XY"（count=4）的 utility = S/D = [4·2·9−(4·9+D)]/D，其中 D=2·9+L（2 个 9-bit 符号 + L=2 位长度字段），S=4 次出现共省 4·2·9 bits 减去 4 个 9-bit 索引与 D；选最高者入字典，页内全部 "XY" 替换为 9-bit 字典索引 1*；选中后与 "XY" 出现位置重叠的序列（如 "YZ"）count 需重算，重复直至字典满（最多 64 个序列）或无正 utility。序列选择上限 5 符号使跨迭代仍能捕获长重复：25 字节 run 两轮即可压成单个索引（"VWXYZ"×5 → 1*×5 → 2*）。每页 >3×10^5 次操作（经局部更新+5 符号上限优化降 ~1000×），靠三模块并行硬件单周期吞吐。解压反过程：8B 位置元数据前缀和(~1ns)定位 DRAM 块 → 静态符号字典解码（数据 ~8 cycle、序列字典 ~16 cycle）→ 展开表（≤2880 bits 寄存器）→ 多 LIFO 并行展开嵌套索引（如 [2*,1*]→2*=Z1*Y→展开为 ZXYYXY），最坏 <128 cycle，平均 18ns。
