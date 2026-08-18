## Bank-PIM 与 Rank-PIM（PIM 计算单元放置粒度）

术语解释
PIM 按计算单元在内存层级中的放置位置分为两类：bank-PIM 把计算单元放在每个 bank 旁（bank 本地访问，内部带宽最高），rank-PIM 把计算单元放在 rank 级（继承 rank 级 ECC 保护，但牺牲内部带宽）。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PIM（Processing-in-Memory）把计算集成进内存芯片附近以消除数据搬运。放置粒度决定内部带宽、峰值性能与可用的可靠性机制：rank-PIM 的计算单元位于 rank 级（如 Samsung AxDIMM/CXL-PNM、rank 级与 CXL 基 PIM 设计），与 host 读取同一 rank 粒度，天然继承 rank 级 ECC（chipkill 级）保护——能容忍整个 x4 器件失效，但计算单元无法独立利用每 chip 内部带宽，性能受限（论文给出 DDR5 bank-PIM 比 rank-PIM 快 8×）；bank-PIM 的计算单元位于每 bank 旁（UPMEM、SK Hynix AiM/GDDR6-AIM、Samsung all-bank PIM），每 chip 内部带宽可被独立利用（DDR5 配置下 8× 外部带宽、每通道最高 32× 理论加速比），但冗余通常局限在 bank 内，仅靠 on-die ECC 无法检测多比特错误 → SDC 风险。论文的核心论点即"bank-PIM 不能仅靠 bank 级保护获得数据中心级可靠性"，并据此提出两层 ECC 方案（见本库"Two-tier ECC"条目）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在硬件架构中的差异直接体现为数据路径：rank-PIM 执行 GEMV 时，权重从 rank 内所有 chip 读出、经 rank 级计算单元聚合——访问粒度与 host 相同，可享受 rank 级 ECC 的全码字保护，但一次只能服务一个 bank 粒度内的数据（内部带宽受 rank 级共享限制）；bank-PIM 执行同一 GEMV 时，host 发 all-bank 命令，每个 bank 旁 PIM 单元（论文配置：每 2 bank 一个、4 个 BF16/FP16 MUL/ADD FPU、64-bit 数据通路、1KB scratchpad、每 tCCDL=16 窗口一条 PIM 指令）以 204.8GB/s/rank 片内带宽并行算部分和，结果经 PIM SRAM 缓冲后 host 读回做 replication/reduction。可靠性上，rank-PIM 的每次访问都走 rank 级 ECC 编解码（可靠但慢）；bank-PIM 的 PIM 访问若也要强保护，要么在 bank 内做（冗余不足、检测弱），要么回 rank 级（性能惩罚）——这就是论文要解决的根本权衡。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：rank-PIM 以 rank 级计算单元 + rank 级 ECC 为特征（Samsung rank-PIM/CXL-PNM），bank-PIM 以 bank 旁数字逻辑 + PIM 单元为特征（UPMEM DPU 见本库"UPMEM DPU 与银行级 PIM 架构"；SK Hynix AiM、Samsung all-bank PIM 见"All-bank PIM"条目）。使用：性能敏感读为主负载（LLM GEMV、PIMBench 数据密集 kernel）选 bank-PIM（论文实测 1.5–4× vs rank-PIM，Linear Regression 达 7.9×），可靠性敏感场景须配论文的两层 ECC 或退而求其次用 rank-PIM；写密集负载（vector add/AXPY）bank-PIM 优势消失甚至落后（论文 0.7×），因为写需要 rank 级 ECC 更新。
- HE² 补充视角（ISCA'26，HBM 内 bank-level PE 集成的 FHE 应用）：HE² 的 xMU（近存模块）把 PE 部署在 HBM 所有 bank 的 column decoder 内（bank 级粒度，遵循 AiM/Newton 等先例），只执行轻量 MemOps（CtAdd、PtMul、IP、Autom），避免在 DRAM 内集成复杂 ComOps 逻辑（DRAM 逻辑密度比 CMOS 低 10×、速度慢 3×，面积预算受限）——xMU PE 仅占 HBM 模块面积 11.1%、峰值功耗在 all-bank-interleave 预算与 85°C 热包络内（12nm PDK 综合）。每 PE 从 global row buffer 取 256-bit 到 local buffer 隐藏 bank 访问延迟，row-major 数据布局把每个多项式摊到所有 bank 让每 PE 本地取操作数；MemOp fusion 消除顺序 MemOps 的 row-switch 写回；in-DRAM automorphism 复用原生 DRAM 数据通路（见"xPU-xMU 异构架构"条目）。选型理由与 bank-PIM 一致：用 bank 级内部带宽换带宽密集的向量运算吞吐，但通过"只放轻量算子 + 面积/功耗约束"规避 bank-PIM 在 DRAM 工艺下的逻辑集成难题。

涉及论文标题：
- ECC Enabled Reliable and Performant Processing-in-Memory
- HE^2: A Communication-Light Heterogeneous Architecture for Efficient Fully Homomorphic Encryption
