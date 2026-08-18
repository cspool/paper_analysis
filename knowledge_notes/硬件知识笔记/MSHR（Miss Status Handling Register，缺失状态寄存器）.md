## MSHR（Miss Status Handling Register，缺失状态寄存器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- MSHR 是缓存中跟踪未完成（in-flight）cache miss 的硬件结构：每个 MSHR 项记录一个未返回的缺失行地址、请求它的指令/PC、以及该行合并的多个 target（同一缓存行被多个访问命中合并）。逻辑链：(1) cache miss 发生时需在数百周期内存访问期间记住"谁在等这个数据"；(2) MSHR 即此状态存储——miss 请求进入 MSHR，返回的 fill 数据按 MSHR 记录的目标分发回等待者；(3) MSHR 数量（如 L1 16、L2 32、L3 36）决定可同时飞行的 miss 数（MLP 上限之一）。ICP 论文利用 MSHR 作为预取触发信息的载体：把 MSHR 的每个 target 项扩展一个压缩 PC 字段（10-bit），当 fill 数据返回时随行把"触发该请求的 PC"送回预取器，从而让 ICP 知道这次 cache line 响应对应哪个 PC_pre、可否触发投机执行。压缩方法：同一依赖链内指令的 PC 高位常相同，故只保留低 3–4 位 + 高位哈希，共 10 bit；开销 = N_MSHR × T × PC_bits = 16 × 8 × 10 = 1280 bit ≈ 160B。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- MSHR 在 ICP 中的运转流程（Section IV-E）：demand 请求进入 L1 → miss 时在 MSHR 分配一项（记录行地址 + 各 target 的压缩 PC）→ 请求下行 L2/L3/DRAM → fill 数据返回时 MSHR 目标项（含压缩 PC）被读出 → 数据经数据总线 snoop 给 ICP 的 Data Extractor（获取 PC_pre 访问的数据），压缩 PC 用于查 Correlation Table → 若命中 PC_pre 且是 friendly（或 demand 触发），ICP 启动投机执行链 → PC_suc 预取请求发出。既有间接预取器（DMP/Tyche/IMP）也通过 snoop 数据总线获取返回数据，ICP 的创新是扩展 MSHR target 记录 PC，把"哪个指令触发了这次填充"随数据一起传递（共需 ~160B，远小于表结构本身）。局限性：MSHR 深度（16/32/36）同时限制 ICP 的并发触发能力与核的 MLP。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：MSHR 是缓存控制器标准部件，通常为 CAM/表结构，每个 miss 分配一项，hit 合并到已有项（合并多个等待者到同一行），fill 后广播分发给等待者并释放项；ICP 在其 target 项中追加压缩 PC 字段（10-bit）+ 相应读写逻辑。使用场景：作为预取器/缓存设计获取"数据 + 触发 PC"的标准接口（论文指出 snoop 数据总线获数据、扩展 MSHR 获 PC 是间接预取器 [19][57][58] 与相关微架构 [52][49] 的常见需求）；MSHR 容量是 MLP 与预取器覆盖的共同约束。

R-Max 补充视角（ISCA'26，MSHR 作为预取现实约束）：R-Max 明确把 MSHR 数量当作"现实约束"的核心之一——每级缓存有限 MSHR（L1I 8、L1D 16、L2 32、LLC 64）且每周期 tag 检查次数受限（带宽约束）。运转：miss 时在该级分配 MSHR 项并转发下一级，全部 miss 则到 DRAM，数据返回后各级按序释放 MSHR；R-Max 的预取与 demand 走同一 MSHR 路径，MSHR 可用时才从 Pending Prefetch Queue 发出预取（直接开 MSHR）。MSHR 相关 miss 来源：预取 in-flight 时 demand 到达触发 MSHR merge（若计数器归零则 Do Not Fill 跳过填充）；×100 MSHR 消融（Fig.8）显示仅增加 MSHR 数收益不大（SPP 5.96%→6.68%），说明 R-Max 的瓶颈主要是容量/带宽/预测而非 MSHR 深度。ChampSim log 中预取的 ISSUED 字段可能显示 0（直接开 MSHR 所致），accuracy 用 useful/(useful+useless) 另算。
TTP 补充视角（ISCA'26，预取请求的缓存响应分析）：预取请求到达缓存有四种结果——(1) Hit；(2) MSHR hit（与在途请求合并，若有 merge entry 可用）；(3) MSHR miss 且有空 MSHR entry（新建条目）；(4) MSHR miss 且无空条目（丢弃）。TTP 把 prefetcher efficiency 定义为「预取请求 miss 于 cache 且 miss 于 MSHR、且有可用 MSHR entry」的比例（即理想响应类型），平均 L1 58.56%、L2 64.85%；命中/被合并的预取冗余但不污染缓存。Vulkan-sim 配置 L1 256 MSHR、L2 768 MSHR。
涉及论文标题：
- ICP: Exploiting Instruction Correlation for Prefetching Irregular Memory Accesses
- R-Max: Extending Bélády's MIN with Prefetching to Bound Realistic Cache Performance
- TTP A Hardware-Efficient Design for Precise Prefetching in Ray Tracing
