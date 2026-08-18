## Open-Bitline DRAM Architecture（开放位线架构）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- DRAM 单元阵列的两种互补 bitline 组织方式之一：**开放位线（open bitline / crosspoint）**把列灵敏放大器（SA）放在两个相邻 subarray 之间，SA 两侧各接一条互补 bitline（一侧 BL、另一侧 /BL）；优点是单元密度最高（可实现 6F² 单元）、SA 可共享，缺点是两条互补 bitline 来自不同阵列、差分噪声免疫差。**折叠位线（folded bitline）**让互补 bitline 并行走在同一侧，共模噪声抵消好、可靠性高，但占用更多面积与走线资源。现代高密度 DRAM 为密度采用开放位线 + 共享 SA。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 开放位线直接决定 ColumnDisturb 的跨 subarray 破坏范围，也是 ColumnKeeper-D 计数的物理依据。运转例子（本论文 Fig.3 与 §3.2）：SA 行位于 subarray 之间，激活 subarray k 的一行时，该行所有列的 SA 被驱动，同时 (i) k 自身全部列 bitline 被扰动，(ii) 与 k-1 共享的偶列 SA 使 k-1 偶列 bitline 被扰动，(iii) 与 k+1 共享的奇列 SA 使 k+1 奇列 bitline 被扰动。由此：攻击者在 subarray A 锤 x 次、C 锤 y 次时，中间 B 的偶列真实被锤 x 次、奇列被锤 y 次，B 的实际 hammer 计数是 max(x,y)，而朴素按"激活总次数"计数会记成 x+y——即 ColumnKeeper 所称"double-counting"，x=y 时多记一倍。CK-D 据此为每 subarray 设偶/奇两张计数器表，ACT 时对 k 两表都加、对 k-1 只加偶表、对 k+1 只加奇表。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 现代 DRAM 普遍采用开放位线 + 共享 SA + 隔离晶体管（SHR 信号选择连接哪一侧阵列）；系统层防御列级干扰必须在控制器中建模这一共享关系（ColumnKeeper），或由 DRAM 内部结构（in-DRAM 版 CK-D）自处理。Web 补充（专利，说明 open/crosspoint 与 folded 结构对比及共享 SA 隔离机制）：https://patents.google.com/patent/US20110310676A1/en 。

PuDGhost 视角（ISCA'26）：开放位线的 even/odd 列共享 SA 结构直接决定并发计算列干扰的空间特性——Opposite-Parity 配置（目标列全为奇/偶列、controlled 为异奇偶列）中干扰单调（随逻辑-1 比例单调偏置，Obsv. 13），Same-Parity 配置中经共享 SA 电路相互作用呈非单调（8+ 行激活时高逻辑-1 比例下偏置方向反转，Obsv. 14）；两者叠加解释总体 48% 级列间干扰（Obsv. 10）。PuDGhost 论文用 RowCopy 与跨 subarray 电荷共享逆向识别 even/odd 列分配（部分模块无法可靠确定则从奇偶实验排除）。
涉及论文标题：
- ColumnKeeper: Efficient Solutions to the ColumnDisturb Vulnerability in DRAM-based Systems
