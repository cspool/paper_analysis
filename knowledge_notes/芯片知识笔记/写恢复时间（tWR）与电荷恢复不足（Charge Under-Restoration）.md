## 写恢复时间（tWR）与电荷恢复不足（Charge Under-Restoration）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- tWR（write recovery time）是 JEDEC 规定的写时序约束：从最后一个写数据到达 DRAM 到可向同一 bank 发 PRE 命令之间的最短时间，给 BLSA 足够时间把位线驱动到全摆幅、使单元电容电荷完全恢复/翻转；tWR 是模拟量（如 DDR 约 15ns）而非时钟周期数。不满足 tWR 则数据未完全写入（电荷恢复不足，charge under-restoration），表现为数据易丢或后续可靠性下降。整行写（128 个 cache line）是串行过程：先写的 cache line 有效恢复时间比后写的多（每相邻 WR 差 tCCD_L），行尾 cache line 恢复时间最短。Web 来源：Samsung 应用笔记（samsung.com/global/business/semiconductor/products/dram/downloads/applicationnote/tWR.pdf）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 DejaVu 中，电荷恢复不足是解释"OverWrite 降低 ACmin"的第一假说：覆写反码（尤其写 0xFF，NMOS 存取管源端电压升高 → V_GS 下降 → 导通能力变弱）在标准 tWR 内无法把电容充分充/放，残余欠恢复使受害行更易被读干扰翻转。但敏感性实验否定其充分性：给 OverWrite 第二次写追加写恢复时间（0→6100ns 扫描），ACmin 随恢复时间上升，但仍需 >400× 标准 tWR（>6100ns）才接近 SameWrite，而 SameWrite 的 ACmin 对附加恢复几乎不变——若纯恢复不足主导，SameWrite 也应受益于更多恢复时间。结论：恢复不足是贡献因子之一，不足以单独解释 DejaVu（需结合电荷陷阱态假说，见电荷陷阱态条目）。此外首翻 cache line 索引分布（多数模块均匀、仅 M1–M4 偏行尾）也不支持"行尾恢复时间失衡主导"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：DRAM 控制器按 JEDEC 时序表在 WR 与 PRE 之间插入 tWR 等待；测试平台（DRAM Bender）可编程任意附加等待做敏感性扫描。研究用法：用 tWR 扫描的响应区分"写入不完整（欠恢复）"与"写入改变陷阱态"两类机制；工程用法：真实系统的写操作普遍是 OverWrite（数据更新），其欠恢复风险是 DejaVu 影响真实系统安全裕量的途径之一。

涉及论文标题：
- DejaVu: Why You Should Write to Your DRAM Rows Twice, Carefully
