## DIMM 组织与跨芯片元素交织（channel/rank/chip/bank 层次与 burst striping）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DIMM 的内存组织层次：channel（独立数据总线，DGX-A100 有 16 通道）→ rank（一个 DIMM 上协同工作的 DRAM chips 组，2 Rank/DIMM）→ chip（×8 芯片输出 8-bit DQ，8 个 ×8 chips 拼成 64-bit 数据总线）→ bank group → bank（2 Rank×8 chips×4 BG×4 Banks=16 banks/rank）→ row/column。数据总线上的连续突发（burst）按 beat 把位交织（stripe）到各 chip：一个 64B 的 burst 的数据分布在 8 个 chips 上，单个多字节元素（如 FP16=16 bit）因此跨越 2 个 ×8 chips——这是 DIMM-PIM 的"布局同步"问题根源：PIM 计算要求每个元素完整位于单一 chip（bank PU 只能访问本 chip 的 bank），存储布局与计算布局不匹配，必须先做 re-layout。CHIME 还定义 rankset = 每通道取一个 rank 的集合（最小同时用满所有通道的单位），作为通信/计算独立调度的粒度。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
CHIME 的芯片级运转例子（Fig.8，8 chips）：offload QKV 时，数据先缓存在 buffer chip（rank PU）的 SRAM，由 re-layout 单元做两级变换后写入 DRAM chips——fine-grained：元素 E0 的 16 bit 被排进两个连续 burst beat、完整存入 Chip 0（不再跨 chip）；coarse-grained：按 head 映射 N_hc 组织 burst beat 内容（N_hc=8 时每 beat 只含单 head 元素、一个 head 分布到 8 个 chips；N_hc=1 时 8 个 head 的元素混排一个 beat、每 head 落单 chip）。DRAM 时序（Table I：BL=4、CCD=4、RCD=22、RAS=52、RP=22、CL=22、CDLR=4/12、CCDL=8 等）决定 bank PU 读 K 的行切换开销与流水可隐藏窗口。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：标准 DDR4-3200 DIMM 即 64-bit 总线 + 8×8 chips + BL4/8 突发；地址映射（row/bank/chip 交织策略）决定元素分布——商用系统固定交织，PIM 系统需灵活映射（Facil）或传输中重排（CHIME）。使用方式：设计多 chip 存内计算时必须显式处理"元素跨 chip"问题（UPMEM/Chameleon 用专用引擎、PIM-MMU 用 MMU 级变换、CHIME 用 in-flight re-layout）；bank 级 PIM 的带宽收益（>30× 主机）依赖元素与 head 到 chips 的布局与 bank 级并行的匹配。

Taking Analytic Databases to the Bank 补充数据库视角（BLIMP OLAP）：论文的 DDR 组织描述与 CHIME 一致——层级为 channel → slot（DIMM 槽位）→ DIMM → rank → chip（DRAM chip）→ bank（16-128MB）→ subarray → mat，每 bank 有 1D 行结构 row buffer（通常 1-4KB）。地址映射采用常见 "8×8 chips" 方案：一个 rank 内 8 个 ×8 chip 拼 64-bit 字，一个 64-bit 字按字节 striping 到 8 个 chip（X 地址的字节落在 chip 0 bank 0，X+1 落 chip 1 bank 0，……，X+8 再落回 chip 0 的 bank 内下一位置），论文称该布局为 "host format"。关键结论：在标准 DDR 地址映射下，单个字的字节跨 chip 分布，BLIMP 核（只能访问本地 bank）无法直接读取整字；修改地址映射不可行（需改变内存标准且 memory mode 下损失并行/带宽），因此由 host 软件 relayout 把字的字节 shuffle 后写入，使现有硬件地址映射"撤销"软件 shuffle、整字落到目标 bank——写入一个 64-bit 字需 8 次内存写、读取需 8 次读，relayout 在 host 侧平均吞吐仅 29GBps（对比 90GBps 峰值带宽），是 PIM offload 的主要运行时开销，端到端评估中隔离计划平均 22% 查询时间花在 relayout。

涉及论文标题：
- CHIME: A Case for Efficient Long-Context Attention-FC Disaggregated Inference with DIMM-PIM
- Taking Analytic Databases to the Bank
