## SALP / MASA（Subarray-Level Parallelism，子阵列级并行）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- ISCA 2012（Kim et al., CMU）提出的 DRAM 微架构改造：让同一 bank 内的多个 subarray 并行工作，缓解"同 bank 请求必须串行"的冲突。三级方案：SALP-1 让一个 subarray 的 precharge 与另一个的 activation 重叠（仅对 tRP 约束重解释，零结构改动）；SALP-2 在当前 subarray 预充电前就可对另一 subarray 发 ACTIVATE（把全局地址锁存推入每个 subarray，面积开销 <0.15%，两个 subarray 可同时保持激活）；MASA（Multiple-Activated Subarrays）允许多个 subarray 同时激活、列命令时由控制器指定唯一驱动全局 bitline 的 subarray（避免短路），等效每 bank 缓存多行、提高行缓冲命中率。Web 来源：ISCA 2012 论文、arXiv:1805.01966（DRAM 微架构与 MLP）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 对 ColumnKeeper 的价值（§6.7）：ColumnKeeper 的预防性刷新是注入到正常命令流的额外 ACT+PRE，传统 bank 结构下与正常请求串行；MASA 使预防性刷新可与同 bank 其他 subarray 的请求重叠执行。运转例子：bank b 的 subarray k 触发预防性刷新（ACT+PRE 约 t_RC）期间，MASA 允许该 bank 的 subarray k' 继续服务正常读写，刷新延迟被隐藏。结果：16K 阈值下 SALP-CK-D（SALP-CK-P）平均开销从非 SALP 的 20%+ 降到 7.87%（10.62%），最大慢化从 >61.79%（79.16%）降到 56.42%（65.34%）；1M 阈值时仅 0.11%（0.16%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：全局行译码/全局位线之外给每个 subarray 加本地地址锁存与多行缓冲控制（SALP-2/MASA），在模拟器（Ramulator 系）中由 tRP 重解释与多 subarray 状态机实现；被后续 RowHammer 缓解、PIM/in-situ 加速器（PENDRAM、RACAM、专利 US11776594B2）广泛复用为低成本并行基板。

涉及论文标题：
- ColumnKeeper: Efficient Solutions to the ColumnDisturb Vulnerability in DRAM-based Systems
