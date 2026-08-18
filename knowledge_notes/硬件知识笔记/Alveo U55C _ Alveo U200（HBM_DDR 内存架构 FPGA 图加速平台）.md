## Alveo U55C / Alveo U200（HBM/DDR 内存架构 FPGA 图加速平台）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 两块 AMD/Xilinx Alveo FPGA 加速卡，代表两种片外内存架构，Graph.hls 用它们验证框架跨内存架构泛化。Alveo U55C（HBM 型）：1,304K LUTs、960 URAMs、3 SLRs、460 GB/s、32 HBM 通道、32 端口、115W TDP（Web 佐证：U55C 基于 Virtex UltraScale+ XCU55C，HBM2 直连 SLR0）。Alveo U200（DDR 型）：1,182K LUTs、960 URAMs、3 SLRs、77 GB/s、4 DDR 通道、215W TDP。宿主服务器 CPU：AMD EPYC 7C13；工具链 Xilinx Vitis 2024.1（含 XRT）。
- 在 Graph.hls 中的角色：U55C 上以 ReGraph 为 baseline（ReGraph 只支持 HBM、在 HBM 上强于 ThunderGP），U200 上以 ThunderGP 为 baseline（ReGraph 不支持 DRAM）。对比策略：L2/L3 参数固定匹配 baseline 配置（32-bit 属性、相同固定 pipeline 结构），仅 L1 参数由 GH-Architect 探索——U55C vs ReGraph 平均 2.6×、U200 vs ThunderGP 平均 1.2×（且 ThunderGP 在 5 个大图上 OOM 而 Graph.hls 可运行）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 硬件架构视角：U55C 的 32 HBM 通道×460 GB/s 是带宽主导（memory-bound）平台——pipeline 设计以饱和 HBM 带宽为目标，URAM 容量（960）是 max partition size 的硬约束（前述 URAM 账本）；U200 的 4 DDR 通道×77 GB/s 带宽低 6×，pipeline 形态与缓存策略需适配 DRAM 时延/带宽画像。GH-Architect 的内置 FPGA 模板（如 AMD Alveo）为两块板提供资源约束（SLR 数、URAM、HBM/DDR 通道、总线宽度）参与 L1/L2 传播。
- 运转流程例：U55C 上 L3 启发式用"3 SLR→14 pipeline slot"作容量前提分组 11 little+3 big；system.cfg 把 kernel 绑定到 SLR 与 32 HBM 通道；综合 4–6h 出 bitstream；FPGA 运行 1–10 min/图。平台差异被封装为模板约束，用户代码无需针对平台修改（平台无关设计）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：Vitis 2024.1 + XRT 综合执行；Graph.hls artifact 脚本（ae_emit/ae_build/ae_run/ae_plot）驱动两平台实验（Fig 6/7/8 需真实板卡，GH-Scope 模拟与代码生成仅需 Linux+Rust）。数据集 14 个（graph500/R-MAT/SNAP/Network Repository）。
- 跨论文复用：U55C（HBM 大带宽）与 U200（DDR 低带宽）是 FPGA 图/稀疏加速研究的标准对比平台对——论文用"参数匹配的公平对比"方法（固定 L2/L3、只探 L1）隔离框架表达力与参数调优的贡献。

涉及论文标题：
- Graph.hls: A Compiler Framework for Composable Graph Accelerator Design
