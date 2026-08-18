## 隔离行计算行布局（Isolation-Row Compute Row Layout，PuDGhost 缓解）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 为缓解 PuDGhost 的"相邻行数据在执行期间不可控"缺陷，PuDGhost 论文提出的一种 DRAM 阵列物理布局：在每个 compute row group 之间放置专用 **isolation rows**——初始化一次后写入固定数据模式并正常刷新，保证 compute rows 的相邻行在 PuD 执行期间始终存储固定数据。与 CS-2 列筛选组合：screening 与执行使用同一固定相邻行数据模式，使筛选结论在执行期保持有效。面积/延迟开销低：按 Ambit 的 6 条 compute rows 需 7 条 isolation rows，1024-row subarray 仅 +0.68% 行数开销，且无额外执行延迟（仅一次性初始化）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 对比两种布局：contiguous 布局（所有 compute rows 紧邻，Ambit 传统）中，一次 MAJX 的 compute rows 相邻于可参与其他 MAJX 的 compute rows，这些行会被新操作数覆盖，故无法维持 CS-2 要求的固定相邻行数据 → 与 CS-2 不兼容。isolation-row 布局（论文 Fig.19b）把每个 compute row group 放在固定数据行之间，初始化（一次）→ 正常刷新 → 执行期相邻行数据恒定 → 兼容 CS-2。芯片级数据路径：GEMV 请求 → 系统初始化 isolation rows（全 1）→ RowCopy 把操作数搬入 compute rows → APA 序列执行 MAJ3（8 行：3 操作数×2 冗余 + 常量 0/1 行）→ SA 采样输出，相邻行恒为固定模式，PuDGhost 偏置被消除。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：DRAM 阵列设计期确定 compute rows 与 isolation rows 的物理排布；内存控制器/系统软件在初始化时写 isolation rows 固定模式，随后按正常刷新周期维护。论文实测（CS-2-1 = 相邻行固定全 1）：相对 PuDGhost-unaware Base，CPR（列通过率，吞吐）1.06×、CBR 125× 降、BER 91× 降；GEMV NMSE 相对 Base-worst 413× 降、TRNG 保留 93% 熵。开销权衡：CS-2 需运行时维持相邻行固定（isolation rows），换取比 CS-1（多变模式筛选，无运行时支持但可用列少）更高 CPR（CS-2-1 比 CS-1-01 CPR 高 1.14×）。

涉及论文标题：
- PuDGhost: Experimental Analysis of Computation Result Corruption in Processing-using-DRAM Operations on Real DRAM Chips and Implications for Future Systems
