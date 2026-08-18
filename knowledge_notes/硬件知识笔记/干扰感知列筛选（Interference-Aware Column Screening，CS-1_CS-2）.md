## 干扰感知列筛选（Interference-Aware Column Screening，CS-1/CS-2）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 列筛选（column screening）是 PuD 系统的可靠列识别方法：用随机输入跑 MAJX（或 SiMRA）并过滤掉产生错误的"不可靠列"，只保留可靠列参与 PuD 执行（post-manufacturing 或启动时进行）。PuDGhost 论文指出：若筛选不感知 PuDGhost（screening 时相邻行数据固定随机、而执行时相邻行数据动态变化），会把不可靠列误标为可靠（实测 Base 配置 CBR 2.1%、BER 9.2×10⁻⁵）。为此提出两种 PuDGhost-aware 筛选：
  - CS-1（screening 时变化相邻行数据）：在多个相邻行数据模式（全 0/全 1 两极端即够，CS-1-sweep 五段扫 p_c 与 CS-1-01 两极端无显著差异）下筛选，只保留所有模式下都可靠的列；无需运行时支持，但可用列数少。
  - CS-2（screening 与执行固定同一相邻行数据）：screening 与 PuD 执行使用同一固定相邻行模式（CS-2-0 全 0 / CS-2-1 全 1 / CS-2-check 棋盘格），需运行时支持维持相邻行固定（配合隔离行布局，见知识库_芯片设计"隔离行计算行布局"条目）；保留更多可用列。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 两阶段评估协议（论文 §10.1，DRAM Bender 在真实 DDR4 上）：Stage 1（screening）每配置 8192 组随机 MAJ3 输入，某列零错误才标 reliable → 得到 CPR（column passing rate，通过列占比，决定系统吞吐）；Stage 2（execution）只在 passed 列上跑 128 组额外随机样本 → 统计 CBR（column break rate，Stage 1 通过但在 Stage 2 出错的列占比）与 BER（bit error rate）。每样本协议：(i) 写相邻行指定模式 → (ii) 写 SiMRA 行随机输入 → (iii) 执行 SiMRA → (iv) 读出。MAJ3 用 8 行 SiMRA（3 操作数×2 冗余 + 常量 0/1 行）。
- 结果：Base（PuDGhost-unaware）CBR 2.1%、BER 9.2×10⁻⁵；CS-2-1（相邻行固定全 1）相对 Base CPR 1.06×、CBR 125× 降、BER 91× 降（估计对应 8.3×10³× 更低 ECC 校正失败率，按 Count2Multiply ECC 配置）；CS-2-1 比 CS-1-01 CPR 高 1.14×，代价是隔离行的容量开销（+0.68% 行数），即"容量 vs 吞吐"权衡。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：在 PuD 系统的内存控制器/系统软件中，于制造后或首次启动时对每个 subarray 的列执行两阶段筛选；CS-2 需要配套的 compute row layout（隔离行）保证执行期相邻行数据不变，并由运行时维护该固定模式（正常刷新）。应用侧：GEMV 用 32768 通过筛选的列（4096×8 bits）执行 8-bit 位串行 MAJ3 乘法，CS-1/CS-2 使 NMSE 全维度 <10⁻³（相对 Base-worst 413× 降 @N=32）；TRNG 用 Fixed 条件（相邻行与非源列数据固定）保留 93% 熵。论文强调筛选无法单独消除 PuDGhost 的全部影响，残余错误由 PuD ECC（如 Count2Multiply）或 error-tolerant 应用（EDEN 重训练的 DNN）兜底。

涉及论文标题：
- PuDGhost: Experimental Analysis of Computation Result Corruption in Processing-using-DRAM Operations on Real DRAM Chips and Implications for Future Systems
