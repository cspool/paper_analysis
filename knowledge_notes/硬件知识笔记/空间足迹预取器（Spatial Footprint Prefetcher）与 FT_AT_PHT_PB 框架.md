## 空间足迹预取器（Spatial Footprint Prefetcher）与 FT/AT/PHT/PB 框架

术语解释
CPU 缓存预取器家族：把内存区域（如 4 KB 页）的访问模式记录为位向量足迹（每 cache line 1 bit），用历史足迹预测同一区域或共享相似事件键的其他区域的未来访问；由 SMS 引入 Filter Table（FT）/ Accumulation Table（AT）/ Pattern History Table（PHT）三表框架，后续 Bingo、DSPatch、PMP、Gaze、Planaria 均在此框架上演进。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 空间足迹（spatial footprint）预取的核心思想：若程序在区域 A 表现出某种足迹（bit-vector，一页内被访问的 cache line 集合），则再次访问 A 或共享相似事件键（PC、offset 或更丰富组合）的其他区域时，很可能出现相似足迹。逻辑链：记录（学习）→ 事件键匹配 → 预测下发。通用框架（SMS 引入）：FT 过滤掉活跃度可忽略的 4 KB 页；剩余页在 AT 中按位向量累积足迹；页失活（如被 AT 逐出）时把完成足迹连同事件键提交到 PHT；PHT 通常组相联，键匹配时把历史足迹交给预取决策。事件键沿频谱分布：高频简单键（页 offset，键空间小、匹配多但信息少）vs 低频特定键（全地址/页号，键空间大、存储多但更精确）。STEP（ISCA'26）在此框架上把"单触发点"改为"多点时序触发"，并将元数据合并进单一 PHT（TOE 蕴含 SOE/FOE，按事件截断 tag），总存储 10.5 KB。本地证据：`paper_secs/paper_isca26_full/STEP Spatial Footprint Prefetcher with Multi-Point Temporal Triggers/*.md`（omnisearch score 111–167，Pattern History Table 命中）。
- Web 佐证：SMS 原文（Somogyi et al., ISCA 2006）定义 FT/AT/PHT/PB 结构；Bingo（Bakhshalipour et al., HPCA 2019）以"specific→fallback→aggregate"强化单点触发。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- STEP 的硬件运转（学习流 + 预取流双流程，Fig.4）：demand 访问先探 AT（命中则置对应 offset 的 footprint 位）；AT 未命中则访问转 FT 筛选——FT miss 分配新条目存第一 offset，FT hit 且第二 offset 无效则补记第二 offset，FT hit 且第二 offset 有效（第三次访问）则页通过筛选，把前三 offset 送入 AT 分配条目开始累积完整足迹；AT 新条目分配逐出旧条目，被逐出的完成足迹+前三 offset 写入 PHT（学习完成）。预取流：FT 按事件（FOE/SOE/TOE）向 PHT 发查表请求，PHT 按第一 offset 索引、以第 2/3 offset 作 tag（SOE 查 tag 上 6 位、TOE 查全 tag），命中后交 Prefetch-Confidence Evaluator 判断收敛性，收敛则把匹配足迹交集下发到 PB/L2 预取队列。
- 存储配置（Table I）：FT 256-entry 8-way（65 bits/entry：36b tag + 3b LRU + 12b hashed PC + 13b offsets + 1b issued）、AT 128-entry 8-way（133 bits/entry：36b tag + 3b LRU + 12b hashed PC + 18b offsets + 64b footprint）、PHT 512-entry 8-way（92 bits/entry：12b tag + 3b LRU + 64b footprint + 12b hashed PC + 1b maturity）、PB 32-entry（103 bits/entry）、DPCT 8-entry，合计 10.50 KB。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：以表驱动硬件为主（FT/AT/PHT 均为组相联 SRAM 表），PHT 每个历史足迹条目附带事件键与 maturity 标志；在 ChampSim 的 prefetcher/ 目录实现为 L2C 预取器（l2c_prefetcher_operate/update 回调），评估平台为 ChampSim + SPEC CPU2006/2017 + CloudSuite 的 130 条 trace（MPKI≥1）。使用场景：任何需要隐藏内存延迟的 CPU 缓存层次（L1/L2），本论文主要作为 L2 旁 add-on 硬件（监听 L1-L2 总线插入预取请求），并验证 L1 级与 L1+L2 多级组合仍有效。

涉及论文标题：
- STEP: Spatial Footprint Prefetcher with Multi-Point Temporal Triggers
