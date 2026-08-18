## RowHammer 与 Row-based Read Disturbance（行级读干扰）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- RowHammer（ISCA 2014）是反复"激活-预充电"攻击行（hammering）使相邻受害行单元发生位翻转的读干扰现象；RowPress（ISCA 2023）是长时间保持攻击行开启（长 tRAS）即可诱发翻转的变体，单次激活 + 约 30ms 保持即可翻转。读干扰阈值 N_RH（诱发翻转所需激活次数）随工艺缩进骤降：2010-2013 年芯片约 69.2K → 2019-2020 年约 4.8K（降 14.4×），RowPress 可再降 1-2 个数量级。攻击后果：提权、信息泄露、VM escape、数据损坏、DoS。Web 来源：RowHammer: A Retrospective（arXiv:1904.09724）、RowPress（ISCA 2023，github.com/CMU-SAFARI/RowPress）。DejaVu（ISCA 2026）补充：同一芯片的读干扰脆弱性还取决于 victim 行**写入历史**——先写反码再覆写目标数据（OverWrite）使双面 RowHammer 的 ACmin 比"只写一次"基线平均再降 2.8%–3.5%（最大 28.1%），相同数据写两次（SameWrite）使 ACmin 平均升 3.0%–3.6%（最大 46.1%），且 RowPress 在 DejaVu 初始化下可诱导基线诱导不出的 bitflip——说明既有"victim 行只写一次"的表征方法对阈值的估计偏乐观（见 DejaVu 与 ACmin 条目）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 芯片级根源是同一 wordline 驱动器反复开启与相邻行间电荷泄漏/耦合。防御分层：芯片内（厂商 TRR；JEDEC DDR5 的 PRAC 逐行计数，见 PRAC 条目）或内存控制器内（PARA 概率刷新、Graphene 映射感知、Hydra 低成本计数、ABACuS 全 bank 计数）。本论文的定位：这些防御全部以"行"为粒度，且只保护攻击行的少数邻行——对列粒度、跨三个连续 subarray 的 ColumnDisturb 无效；但 ColumnKeeper 与 RowHammer 防御**正交共存**：§4.4 让双方把彼此的预防性刷新计入各自计数，§6.4 评估了与 Graphene/PRAC（DDR5）/Hydra 组合（N_RH=128）时 CK-D 仅额外降 IPC 1.70%~2.15%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现上：TRR 厂商内置但可被绕过（Half-Double、Blacksmith 等模式）；PRAC 把逐行计数器放入 DRAM 并经 ALERT/RFM 闭环；控制器级用计数器表（Graphene/Hydra/ABACuS）或概率刷新（PARA）。研究评估普遍用 Ramulator 2.0 集成这些机制，并对照"无防御"基线测量 IPC/能耗开销与安全阈值。

Raptor 补充视角（ISCA'26，刷新间隔驱动的固有缓解）：3D-DRAM 的深 bank 化几何使 RowHammer 被"刷新窗口 < 锤击窗口"固有缓解——RowHammer 阈值 200K（旧工艺节点）下，t_RC=44ns 的背靠背激活需 8.8ms 才达到锤击阈值；Raptor 在 105°C 结温用 4ms 刷新间隔（比 HBM 标称 32ms 密 8×），4ms < 8.8ms → 每行在邻居积累足够激活前就被刷新，无需 TRR/PRAC 类额外防护。代价量化：4ms 刷新仅损失 1.37% 带宽（103.57 TB/s @700MHz）；每 bank 仅 1364 行（比常规 DRAM bank 少 16-32×）使单 bank 刷新延迟远低于商品 DRAM。与 DejaVu 的"写历史影响阈值"、ColumnKeeper 的"列级干扰"视角互补：Raptor 是从刷新频率与行数的物理参数空间消除锤击窗口，而非在控制器/芯片内加计数防护。

Sigries 生产视角（ISCA'26，Azure Cobalt 200 SoC 的混合防御）：DRAM 内置防御既"多孔"又依赖"安全通过隐蔽"（vendors 不愿披露 TRR 实现），JEDEC 两份白皮书（JEP300-1/JEP301-1）也承认现有防御无法消除全部攻击；随工艺缩进，诱发翻转所需 hammers 从十年前数十万降到当代数千（RowPress 等新模式让云厂商措手不及）。生产侧对策：云厂商自研 SoC（AWS Graviton、Google Axion、Azure Cobalt 200）在内存控制器内实现控制器级防御；Rowhammer 阈值由 DRAM qualification 流程跨多厂商 DIMM 样片实测得出（业界预期阈值不会低于数千，与学术界"低数百"的假设不同）。Sigries 实测：Azure Cobalt 200 上所有 commodity workload 在 32ms refresh window 内无行激活超几百次（Intel 因在 DRAM 存一致性目录位才触发高激活率，Cobalt 不共享该设计），故无需任何缓解动作、零 DRAM 带宽开销。

PrISM 视角（ISCA'26，Loaded Dice 论文）：把双面 RowHammer 阈值（TRH-D，两相邻 aggressor 交替锤击 victim）作为主要指标，威胁模型为无特权攻击者——知道部署的 in-DRAM 防御并可构造定制访问模式绕过；所有概率缓解（MINT/PrISM）都用 fractal mitigation [70] 防御 transitive attacks（Half-Double 等）；RowPress/ColumnDisturb 在主威胁模型外（附录 C 讨论扩展：RowPress 用 ImPress 的 EACT 等效激活计数扩展、ColumnDisturb 用 PrISM 作 aggressor 过滤器 + PRVR 缓解）。评估配置在 32ms tREFW 内以 MTTF 10,000 年/bank 为安全目标（见 MTTF 条目），并指出 TRH-D 随工艺持续下降（图 1）使概率缓解的固定速率升级路径在低阈值下不可持续——这是 PrISM 的动机。

PVAC 视角（ISCA'26）：PVAC 显式区分 NRH（aggressor 侧最大锤击次数）与 NRHv（victim 侧最大被锤计数）两个阈值——二者是同一物理扰动的两个视角，BR 内多个 aggressor 的扰动在 victim 处累加，故 aggressor 计数必须把 NBO 压到 NRHv/BR 量级，BR 越大越保守；PVAC 直接约束 victim 的 hammered count（HC），NBO 无需除以 BR（HC=128 时 PVAC 85/102/108 vs PRAC 3/15/19、Chronus 31）。防御实现上把计数器迁出数据行放入独立 Counter Subarray（CSA）并按 victim 语义更新（见 CSA 与 victim-based counting 条目）；对 DDR5 的 ABO/RFM 协议只复用 RFM 时序窗口、不用 PRAC 的扩展时序。与 DejaVu（写历史影响阈值）、Raptor（刷新窗口压缩锤击窗口）、Sigries（控制器级生产防御）、PrISM（概率缓解）不同，PVAC 属于"改变计数语义"的精确计数路线，可在默认 DDR5 时序下落地。

涉及论文标题：
- ColumnKeeper: Efficient Solutions to the ColumnDisturb Vulnerability in DRAM-based Systems
- DejaVu: Why You Should Write to Your DRAM Rows Twice, Carefully
- Early Silicon of Raptor: The First 3D-DRAM Accelerator for Generative Inference
- From Lab to Fleet: Building and Deploying a Practical Rowhammer Defense in Cloud SoCs
- Loaded Dice: Solving the Non-Selection Problem for Scalable Probabilistic RowHammer Defense
- PVAC: A RowHammer Mitigation Architecture Exploiting Per-victim-row Counting
