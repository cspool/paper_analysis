## Extended Directory（ED）/ Snoop Filter 与 Traditional Directory（TD）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
目录分片的两个组成部分：TD（Traditional Directory）条目与 LLC 条目一一对应（目录项伴随 LLC 行存在）；ED（Extended Directory，也称 snoop filter）为非包容 cache 层级服务，记录"在私有缓存（如 L2）但已不在 LLC 中"的行的目录条目——非包容层级下 LLC 不保证含有上层私有缓存的行，没有 ED 就无法从 LLC/目录发现这些行。ED 与 TD 的条目数可以不同。同源概念：Intel 非包容 L3 每 slice 的 CHA 含 snoop filter（https://stackoverflow.com/questions/65316397），DRAM 中扩展目录位（memory directory bits）在本地 load 时不更新、写回失效后清除；专利 US5909697 与 ARM GB2539382A 描述 snoop filter 在非包容系统中的实现与陈旧条目处理。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Dorado 中每个簇的每个目录分片 = TD + ED；行的目录项（Global home 或 Temporary home）按目录分配算法落 TD 或 ED——TD 建条目必连带把数据行插进对应 LLC 项，ED 条目则无 LLC 数据。例子（Fig.6）：LLptr 条目的两种形态（TD 条目 + LLC 数据行 / ED 条目无数据行）；RLptr（Temporary home）条目同样两种形态。因 ED 的存在，"簇内有目录项"不代表"行在簇内 LLC"，供数时需先查目录再决定从本地 LLC、本地 L2 还是远端取数（Table III 优先序）。Dynamic Apportioning 的前提正是 TD/ED 条目与三类指针共用单一结构。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：TD 实现为与 LLC tag 并行的每路目录状态；ED 为独立组相联结构（条目数按"私缓存可能驻留但 LLC 不驻留"的行数定）。使用要点：非包容层级省 tag/状态但必须配 ED/snoop filter；ED 溢出时需回退策略（Flask 用 Bloom filter 容忍信息不完整、按需广播重建条目，https://www.scilit.com/publications/cab3880fb78809e5b3ac7dde8efbd1ff）。论文未明确说明 Dorado 的 TD/ED 各自条目数与替换策略细节。

涉及论文标题：
- Dorado: Clustered Hardware Cache Coherence for 1,000+ Cores
