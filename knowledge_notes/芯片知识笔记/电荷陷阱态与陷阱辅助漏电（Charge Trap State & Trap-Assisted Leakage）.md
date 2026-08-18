## 电荷陷阱态与陷阱辅助漏电（Charge Trap State & Trap-Assisted Leakage）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 电荷陷阱（charge trap）是 DRAM 存取管硅衬底/氧化物界面处可捕获并延迟释放载流子的缺陷态；陷阱辅助电子迁移/注入（trap-assisted electron migration/injection）是器件级研究确认的 RowHammer/RowPress 主要物理机制之一：反复激活 wordline 使电子经陷阱逐步迁移进相邻受害行区域，改变其存取管阈值电压、增大亚阈值漏电，最终使存储电荷流失导致 bitflip。陷阱占据态不是固定的：行打开时捕获电子、关闭时释放。Web 来源：Double-sided Row Hammer Effect in Sub-20 nm DRAM（IRPS 2023，3D TCAD 证实陷阱辅助电子迁移主导双面 RowHammer 的 1→0 失败）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- DejaVu 的第二假说（论文认为比恢复不足假说更成立）：写不同数据值会改变 active region 陷阱占据态。芯片级两条证据链：(i) 现代 6F² 高密度布局中相邻两行共享同一物理 active region，写 victim 行可能扰动 aggressor 行附近的陷阱占据，改变后续 hammering 期间陷阱辅助漏电强度——对应 OverWrite 降 ACmin、SameWrite 升 ACmin；(ii) 陷阱占据态分布改变存取管导通能力分布，影响 PUD 多行电荷共享的可靠性——写两次使分布更均匀 → bitline failure rate 降 32.7%。DejaVu 本身无法直接观测陷阱（真实芯片黑盒），用 PUD 灵敏度实验作为代理证据。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：器件级研究用 3D TCAD 工艺仿真定量（如 IRPS 2023 对双面 RowHammer 的机制判别）；芯片级研究用间接实验——DejaVu 的 tWR 扫描排除恢复不足假说、PUD 可靠性做代理测量。工程含义：DRAM 可靠性建模需把陷阱占据态当作有状态的物理量（写入历史是输入），而非只按电压/温度/时序静态建模。

涉及论文标题：
- DejaVu: Why You Should Write to Your DRAM Rows Twice, Carefully
