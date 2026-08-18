## Misra-Gries（流式 heavy-hitter 算法：欠配置计数器表 + spillover）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Misra-Gries（MG，Misra & Gries, Science of Computer Programming 1982）是求解"heavy hitters（频繁项）"问题的确定性流式算法：维护至多 k 个 (元素, 计数) 对的计数器表，处理流中每个元素 e：(1) e 已在表中 → 计数+1；(2) e 不在表中且表未满 → 插入 e 计数 1；(3) e 不在表且表已满 → 全表计数各减 1（移除归零项，即"canceling"步）。空间 O(k log m) 位；对每个元素 i 有估计 f̂ᵢ 满足 fᵢ − m/k ≤ f̂ᵢ ≤ fᵢ（至多低估 m/k）；频率 fᵢ > m/k 的项保证出现在输出中；canceling 步至多发生 m/(k+1) 次。Boyer-Moore majority voting 是 k=2 的特例。近下界（arXiv:2406.12149）证明其在流式模型下基本最优。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- Sigries（ISCA'26）把 MG 硬件化在 Azure Cobalt 200 SoC 内存控制器中，作为 light mode 的 aggressor 行跟踪器，并用一个**spillover counter** 实现 MG 的"全表-1"步的紧凑形态：spillover counter 是所有未列入表元素激活次数的上界，MG 保证表中每个计数器值 ≥ spillover；每当 spillover 会超过某表项计数时，就把该项计数+1 并用当前激活行地址覆写该表项地址（维持不变量）。运转流程（每次 row activation）：查 sub-bank 计数器表（SRAM set-associative，每 set 一 sub-bank）→ 命中则计数+1 → 未命中则与 spillover 比较替换 → 计数达 Rowhammer 阈值时发 DRFM、计数复位、置 lock bit（禁止再被替换）→ spillover 达阈值-1 时该 sub-bank 切 heavy mode（采样）。关键设计：**欠配置（under-provisioned）**——每 sub-bank 仅几十项，远小于完整 MG 需要的 k；只要 spillover 低于阈值，欠配置 MG 仍阻止所有攻击（Insight #1）。完整 MG 硬件（Graphene 式）需要数百项/大 CAM，把内存控制器面积放大 ~40× 且无法在 DDR5 频率下构建运行，故不可实现。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 通用实现：软件流式框架（计数数组 + 哈希）或硬件计数器表；Rowhammer 域的代表实现是 Graphene（MICRO 2020）——映射感知的 MG 计数（每 bank 一个表、找最低计数值做替换），以及 ProTRR 的 Proactive MG（in-DRAM）。Sigries 用法：按 sub-bank 分表 + 懒清零（clear bit 在下次激活时清空整个 sub-bank 表，省掉空闲时的读-写清零功耗）+ Dafny 形式化验证（证明行访问计数 < 阈值等不变量，捕获 3 个 RTL 验证抓不到的算法 bug：clear bit 未清除、lock bit 置 0 而非 1、空条目查找未检查 lock 位）。Web 来源：MG 讲义（https://www.mit.edu/~vakilian/courses/lecs/lec_7_6104.pdf）、Graphene（MICRO 2020）。

涉及论文标题：
- From Lab to Fleet: Building and Deploying a Practical Rowhammer Defense in Cloud SoCs
